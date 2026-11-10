"""Von Karman large-deflection elastoplastic panel solver (Ritz / Rayleigh-Ritz).

The bottom plating of a ship that runs aground has to absorb the grounding
energy in *membrane* action: the panel is restrained by the longitudinals and
floors, so once the deflection exceeds a few plate thicknesses the membrane
stresses dominate the resistance.  This module solves that problem with a
moderately non-linear (von Karman) plate theory discretised by a two-field
Ritz method:

* out-of-plane field ``w``  – products of clamped (or simply supported) beam
  eigenfunctions, so the kinematic edge conditions ``w = 0`` and (clamped)
  ``dw/dn = 0`` hold exactly;
* in-plane fields ``u, v`` – sine/cosine products that vanish on the edges the
  corresponding displacement is restrained on (immovable edges ``u = v = 0``);
* through-thickness integration with ``n_layers`` integration points, each
  obeying plane-stress von Mises elastoplasticity with isotropic hardening
  (return mapping shared with :mod:`tensorphysics.plate_buckling.core`).

Equilibrium ``R(A) = f_ext`` is solved by Newton iteration inside an
incremental load path, so the accumulated equivalent plastic strain is
path-dependent and physically meaningful (it is *not* a post-processing of an
elastic solution).  Newton uses a finite-difference Jacobian of the residual
(the tangent of a state-driven return map), which is cheap for the 30-ish
generalised coordinates used here.

Verification hooks
------------------
* linear limit, simply supported: exact Navier series
  (:func:`tensorphysics.plate_buckling.core.navier_simply_supported_deflection`);
* linear limit, clamped: ``w_max = 1.26e-3 q a^4 / D`` for a square panel
  (Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, Table 35);
* large deflection / elastoplastic: independent Q4+Hermite finite element
  (:class:`tensorphysics.plate_buckling.nonlinear_buckling.PlateModel`).

Units: SI (m, Pa, N, J).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..plate_buckling.core import BilinearMaterial, plane_stress_return_map
from .material import SteelGrade

__all__ = ["PanelSpec", "VonKarmanPanel", "PanelResponse"]

# Roots of the clamped-clamped beam characteristic equation (lambda_m).
_CLAMPED_LAMBDA = (
    4.730040744862704,
    7.853204624095838,
    10.995607838001671,
    14.137165491257464,
    17.278759594743864,
)


def _clamped_beam(xi: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamped-clamped *symmetric* Ritz function and its first two derivatives.

    ``phi_m(0) = phi_m(1) = phi_m'(0) = phi_m'(1) = 0`` and ``phi_m`` is even
    about ``xi = 1/2``, so a symmetric load cannot excite spurious
    antisymmetric components (the classical Ritz pathology of the truncated
    clamped-beam-eigenfunction product basis, which makes the truncated
    solution asymmetric and non-monotone in the number of modes).

    The beam eigenfunction is provided by :func:`_clamped_beameig` for
    reference / mode-shape plotting.
    """
    k = 2.0 * np.pi * m
    phi = 0.5 * (1.0 - np.cos(k * xi))
    d1 = 0.5 * k * np.sin(k * xi)
    d2 = 0.5 * k**2 * np.cos(k * xi)
    return phi, d1, d2


def _clamped_beameig(xi: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamped-clamped beam eigenfunction (reference only, not used as basis)."""
    lam = _CLAMPED_LAMBDA[m]
    cl, sl = np.cosh(lam), np.sinh(lam)
    sigma = (cl - np.cos(lam)) / (sl - np.sin(lam))
    phi = np.cosh(lam * xi) - np.cos(lam * xi) - sigma * (np.sinh(lam * xi) - np.sin(lam * xi))
    d1 = lam * (np.sinh(lam * xi) + np.sin(lam * xi)) - sigma * lam * (np.cosh(lam * xi) - np.cos(lam * xi))
    d2 = lam**2 * (np.cosh(lam * xi) - np.cos(lam * xi)) - sigma * lam**2 * (np.sinh(lam * xi) + np.sin(lam * xi))
    return phi, d1, d2


def _simpson(n: int) -> np.ndarray:
    """Composite Simpson weights for ``n`` (odd) equally spaced nodes on [0, 1]."""
    if n % 2 == 0:
        raise ValueError("Simpson quadrature requires an odd number of nodes")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / w.sum()


@dataclass
class PanelSpec:
    """Geometry, material and discretisation of one bottom plate panel."""

    a: float = 3.60          # panel length (m) between floors / transverses, x
    b: float = 0.90          # panel breadth (m) between longitudinals, y
    t: float = 0.022         # plate thickness (m)
    grade: SteelGrade = field(default_factory=lambda: SteelGrade("NV-FH36"))
    bc: str = "clamped"      # "clamped" (w = w_n = 0) or "ssss" (w = M_n = 0)
    n_x: int = 21
    n_y: int = 21
    n_layers: int = 5
    w_modes: int = 3
    uv_modes: int = 3

    def __post_init__(self) -> None:
        if self.bc not in ("clamped", "ssss"):
            raise ValueError(f"unsupported bc {self.bc!r}")
        if self.n_x % 2 == 0 or self.n_y % 2 == 0:
            raise ValueError("Simpson quadrature requires odd node counts")
        if self.w_modes > 8:
            raise ValueError("too many out-of-plane modes (max 8)")

    @property
    def material(self) -> BilinearMaterial:
        return self.grade.bilinear

    @property
    def flexural_rigidity(self) -> float:
        E, nu, t = self.grade.E, self.grade.nu, self.t
        return E * t**3 / (12.0 * (1.0 - nu**2))

    @property
    def area(self) -> float:
        return self.a * self.b

    @property
    def beta(self) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y / E)``."""
        return (self.b / self.t) * np.sqrt(self.grade.sigma_y / self.grade.E)


@dataclass
class PanelResponse:
    """Result of a load-path panel analysis (all arrays indexed by load step)."""

    pressure: np.ndarray          # applied patch pressure q (Pa, positive inward)
    deflection: np.ndarray        # max panel deflection (m)
    volume: np.ndarray            # displaced volume  /int w dA  (m^3)
    energy: np.ndarray            # cumulative absorbed energy (J)
    eps_pl_max: np.ndarray        # max equivalent plastic strain over the panel
    sigma_vm_max: np.ndarray      # max von Mises stress (Pa)
    membrane_force: np.ndarray    # max |N| per unit width (N/m)
    bend_moment: np.ndarray       # max |M| per unit width (N)
    torn_fraction: np.ndarray     # fraction of panel area with eps_pl >= eps_f
    mode_shape: np.ndarray        # final deflection grid w(x, y)
    coords: Tuple[np.ndarray, np.ndarray]
    eps_pl_field: np.ndarray      # max-over-layers eps_pl on the grid
    spec: PanelSpec

    # ------------------------------------------------------------------ helpers
    def pressure_at(self, deflection: float) -> float:
        """Contact pressure at a given deflection (monotone interpolant)."""
        d = np.maximum.accumulate(self.deflection)
        q = np.maximum.accumulate(self.pressure)
        return float(np.interp(deflection, d, q))

    def energy_at(self, deflection: float) -> float:
        """Absorbed energy at a given deflection (monotone interpolant)."""
        d = np.maximum.accumulate(self.deflection)
        e = np.maximum.accumulate(self.energy)
        return float(np.interp(deflection, d, e))

    def torn(self, deflection: float) -> bool:
        return self.pressure_at(deflection) > 0.0 and \
            bool(np.interp(deflection, np.maximum.accumulate(self.deflection),
                           np.maximum.accumulate(self.torn_fraction)) >= 0.5)


class VonKarmanPanel:
    """Incremental von Karman Ritz solver for a single rectangular panel."""

    def __init__(self, spec: PanelSpec, load_steps: int = 12,
                 max_newton: int = 30, tol: float = 1e-9) -> None:
        self.spec = spec
        self.load_steps = int(load_steps)
        self.max_newton = int(max_newton)
        self.tol = float(tol)
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        s = self.spec
        self.x = np.linspace(0.0, s.a, s.n_x)
        self.y = np.linspace(0.0, s.b, s.n_y)
        self.wx = _simpson(s.n_x)
        self.wy = _simpson(s.n_y)
        self.w2d = self.wy[:, None] * self.wx[None, :]

        xi = self.x / s.a
        eta = self.y / s.b
        M = s.w_modes
        if s.bc == "clamped":
            basis = np.empty((M, s.n_x))
            basis_x = np.empty_like(basis)
            basis_xx = np.empty_like(basis)
            by = np.empty((M, s.n_y))
            by_y = np.empty_like(by)
            by_yy = np.empty_like(by)
            for m in range(M):
                p, d1, d2 = _clamped_beam(xi, m + 1)
                basis[m], basis_x[m], basis_xx[m] = p, d1 / s.a, d2 / s.a**2
                p, d1, d2 = _clamped_beam(eta, m + 1)
                by[m], by_y[m], by_yy[m] = p, d1 / s.b, d2 / s.b**2
        else:  # simply supported: sin(m pi xi)
            basis = np.empty((M, s.n_x))
            basis_x = np.empty_like(basis)
            basis_xx = np.empty_like(basis)
            by = np.empty((M, s.n_y))
            by_y = np.empty_like(by)
            by_yy = np.empty_like(by)
            for m in range(M):
                k = (m + 1) * np.pi
                basis[m] = np.sin(k * xi)
                basis_x[m] = k / s.a * np.cos(k * xi)
                basis_xx[m] = -(k / s.a) ** 2 * np.sin(k * xi)
                by[m] = np.sin(k * eta)
                by_y[m] = k / s.b * np.cos(k * eta)
                by_yy[m] = -(k / s.b) ** 2 * np.sin(k * eta)
        self.phi, self.phi_x, self.phi_xx = basis, basis_x, basis_xx
        self.psi, self.psi_y, self.psi_yy = by, by_y, by_yy

        # in-plane bases -------------------------------------------------
        Ku = Lu = s.uv_modes
        k = np.arange(1, Ku + 1)[:, None] * np.pi
        su = np.sin(k * xi)                  # (Ku, nx)
        su_x = k / s.a * np.cos(k * xi)
        l = np.arange(0, Lu)[:, None] * np.pi
        cu = np.cos(l * eta)                 # (Lu, ny)
        cu_y = -l / s.b * np.sin(l * eta)
        kv = np.arange(0, Ku)[:, None] * np.pi
        cv = np.cos(kv * xi)                 # (Ku, nx)
        cv_x = -kv / s.a * np.sin(kv * xi)
        lv = np.arange(1, Lu + 1)[:, None] * np.pi
        sv = np.sin(lv * eta)                # (Lu, ny)
        sv_y = lv / s.b * np.cos(lv * eta)
        self.su, self.su_x, self.cu, self.cu_y = su, su_x, cu, cu_y
        self.cv, self.cv_x, self.sv, self.sv_y = cv, cv_x, sv, sv_y

        self.n_w = M * M
        self.n_u = Ku * Lu
        self.n_v = Ku * Lu
        self.n_dof = self.n_w + self.n_u + self.n_v

        npts = s.n_x * s.n_y * s.n_layers
        self._eps_p = np.zeros((npts, 3))
        self._alpha = np.zeros(npts)
        self._eps_commit = np.zeros((npts, 3))
        self._npts = npts

    def _thickness_points(self):
        """Through-thickness Gauss-Legendre points (m) and weights (m)."""
        xi, wi = np.polynomial.legendre.leggauss(self.spec.n_layers)
        return 0.5 * self.spec.t * xi, 0.5 * self.spec.t * wi

    # ------------------------------------------------------- field assembly
    def _split(self, A: np.ndarray):
        nw, nu, nv, M = self.n_w, self.n_u, self.n_v, self.spec.w_modes
        Aw = A[:nw].reshape(M, M)
        Bu = A[nw:nw + nu].reshape(self.spec.uv_modes, self.spec.uv_modes)
        Cv = A[nw + nu:].reshape(self.spec.uv_modes, self.spec.uv_modes)
        return Aw, Bu, Cv

    def _kinematics(self, A: np.ndarray):
        """Return mid-surface strains and curvatures for the generalised coords."""
        Aw, Bu, Cv = self._split(A)
        w = np.einsum("mn,mi,nj->ji", Aw, self.phi, self.psi)
        w_x = np.einsum("mn,mi,nj->ji", Aw, self.phi_x, self.psi)
        w_y = np.einsum("mn,mi,nj->ji", Aw, self.phi, self.psi_y)
        # second derivatives for curvature
        w_xx = np.einsum("mn,mi,nj->ji", Aw, self.phi_xx, self.psi)
        w_yy = np.einsum("mn,mi,nj->ji", Aw, self.phi, self.psi_yy)
        w_xy = np.einsum("mn,mi,nj->ji", Aw, self.phi_x, self.psi_y)

        u_x = np.einsum("kl,ki,lj->ji", Bu, self.su_x, self.cu)
        u_y = np.einsum("kl,ki,lj->ji", Bu, self.su, self.cu_y)
        v_x = np.einsum("kl,ki,lj->ji", Cv, self.cv_x, self.sv)
        v_y = np.einsum("kl,ki,lj->ji", Cv, self.cv, self.sv_y)

        eps_m = np.stack([
            u_x + 0.5 * w_x**2,
            v_y + 0.5 * w_y**2,
            u_y + v_x + w_x * w_y,
        ])
        kappa = np.stack([w_xx, w_yy, 2.0 * w_xy])
        return w, w_x, w_y, eps_m, kappa

    def _stress_resultants(self, eps_m: np.ndarray, kappa: np.ndarray):
        """Integrate the layered elastoplastic stress state over the thickness."""
        s = self.spec
        nl = s.n_layers
        z, zw = self._thickness_points()
        shape = (s.n_y, s.n_x)
        eps_layer = np.empty((nl,) + shape + (3,))
        for i, zi in enumerate(z):
            eps_layer[i] = np.moveaxis(eps_m + zi * kappa, 0, -1)
        strain = eps_layer.reshape(-1, 3)
        # Sub-incremented constitutive integration: the strain increment from the
        # committed state is applied in sub-steps of at most ``d_eps_max``, so a
        # large Newton increment cannot produce an inconsistent (over-shoot)
        # stress state -- this is what makes the plastic Newton iteration
        # converge at large deflections.
        deps = strain - self._eps_commit
        d_eps_max = 2.0e-3
        n_sub = int(max(1, min(40, np.ceil(np.abs(deps).max() / d_eps_max))))
        eps_p = self._eps_p
        alpha = self._alpha
        sig = None
        for k in range(n_sub):
            strain_k = self._eps_commit + (k + 1.0) / n_sub * deps
            sig, _Cep, eps_p, alpha = plane_stress_return_map(
                strain_k, eps_p, alpha, s.material)
        sig = sig.reshape(nl, *shape, 3)
        # Gauss weights integrate the bending stiffness exactly: sum(w_i z_i^2)
        # = t^3 / 12, so the layered model reproduces D before yield.
        N = np.moveaxis(np.tensordot(zw, sig, axes=(0, 0)), -1, 0)              # (3, ny, nx)
        M = np.moveaxis(np.tensordot(zw * z, sig, axes=(0, 0)), -1, 0)          # (3, ny, nx)
        self._eps_trial = strain
        return N, M, eps_p, alpha

    # --------------------------------------------------------------- residual
    def residual(self, A: np.ndarray, pressure: float,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Equilibrium residual ``R(A) = f_int - f_ext`` (Newton target R = 0)."""
        if mask is None:
            mask = self._full_mask()
        w, w_x, w_y, eps_m, kappa = self._kinematics(A)
        N, M, _, _ = self._stress_resultants(eps_m, kappa)
        W2d = self.w2d
        S = lambda f, bL, bR: bL @ (f * W2d) @ bR.T          # sandwich operator

        Mxx, Myy, Mxy = M[0], M[1], M[2]
        Nxx, Nyy, Nxy = N[0], N[1], N[2]
        # ---- out-of-plane dofs
        R_w = (S(Mxx, self.psi, self.phi_xx) + S(Myy, self.psi_yy, self.phi)
               + S(2.0 * Mxy, self.psi_y, self.phi_x)
               + S(Nxx * w_x, self.psi, self.phi_x)
               + S(Nyy * w_y, self.psi_y, self.phi)
               + S(Nxy * w_x, self.psi_y, self.phi)
               + S(Nxy * w_y, self.psi, self.phi_x))
        f_ext = S(pressure * mask, self.psi, self.phi)
        # ---- in-plane dofs (u: sine in x, cosine in y)
        R_u = (S(Nxx, self.cu, self.su_x) + S(Nxy, self.cu_y, self.su)).T
        # ---- v: cosine in x, sine in y
        R_v = (S(Nyy, self.sv_y, self.cv) + S(Nxy, self.sv, self.cv_x)).T

        R = np.concatenate([(R_w - f_ext).T.ravel(), R_u.ravel(), R_v.ravel()])
        return R

    def _full_mask(self) -> np.ndarray:
        return np.ones((self.spec.n_y, self.spec.n_x))

    def patch_mask(self, half_length: float, half_breadth: float,
                   cx: Optional[float] = None,
                   cy: Optional[float] = None) -> np.ndarray:
        """Axis-aligned contact patch (box) mask on the quadrature grid."""
        s = self.spec
        cx = 0.5 * s.a if cx is None else cx
        cy = 0.5 * s.b if cy is None else cy
        inside_x = np.abs(self.x - cx) <= half_length
        inside_y = np.abs(self.y - cy) <= half_breadth
        return (inside_y[:, None] & inside_x[None, :]).astype(float)

    # ---------------------------------------------------------------- solving
    def _fd_jacobian(self, A: np.ndarray, pressure: float, mask: np.ndarray,
                     R: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobian of the residual (algorithmic tangent)."""
        h = 1e-8 * max(1.0, float(np.linalg.norm(A)) / np.sqrt(R.size))
        J = np.empty((R.size, R.size))
        for j in range(R.size):
            Ap = A.copy()
            Ap[j] += h
            J[:, j] = (self.residual(Ap, pressure, mask) - R) / h
        return J

    def _newton(self, A: np.ndarray, pressure: float, mask: np.ndarray) -> np.ndarray:
        """Modified Newton with a finite-difference tangent and backtracking."""
        R = self.residual(A, pressure, mask)
        r0 = float(np.linalg.norm(R))
        if r0 < 1e-14:
            return A
        tol = max(self.tol * max(r0, 1.0), 1e-13)
        J = self._fd_jacobian(A, pressure, mask, R)
        for it in range(self.max_newton):
            Rn = float(np.linalg.norm(R))
            if Rn < tol:
                break
            try:
                dA = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dA = np.linalg.lstsq(J, -R, rcond=None)[0]
            lam = 1.0
            accepted = False
            for _ls in range(10):
                A_try = A + lam * dA
                R_try = self.residual(A_try, pressure, mask)
                if np.linalg.norm(R_try) < Rn:
                    A, R = A_try, R_try
                    accepted = True
                    break
                lam *= 0.5
            if accepted:
                if it % 2 == 1:          # refresh the tangent every second step
                    J = self._fd_jacobian(A, pressure, mask, R)
            else:                        # tangent stale -> rebuild, damped step
                A = A + 0.1 * dA
                R = self.residual(A, pressure, mask)
                J = self._fd_jacobian(A, pressure, mask, R)
        self.last_rel_residual = float(np.linalg.norm(R)) / max(r0, 1e-30)
        return A

    def commit(self, A: np.ndarray) -> None:
        """Advance the plastic state to the equilibrium at ``A``."""
        _, _, eps_m, kappa = self._kinematics(A)
        N, M, eps_p_new, alpha_new = self._stress_resultants(eps_m, kappa)
        self._eps_p = eps_p_new
        self._alpha = alpha_new
        self._eps_commit = getattr(self, "_eps_trial", self._eps_commit)
        del N, M

    # ------------------------------------------------ displacement control
    def _unit_pressure_load(self, mask: np.ndarray) -> np.ndarray:
        """External load vector produced by a *unit* pressure over ``mask``."""
        S = lambda f, bL, bR: bL @ (f * self.w2d) @ bR.T
        F_w = S(mask.astype(float), self.psi, self.phi)          # [n, m]
        F = np.zeros(self.n_dof)
        F[:self.n_w] = F_w.T.ravel()
        return F

    def w_center(self, A: np.ndarray) -> float:
        """Panel-centre deflection (smooth scalar used as control quantity)."""
        xi = np.array([0.5])
        eta = np.array([0.5])
        s = self.spec
        Aw = A[:self.n_w].reshape(s.w_modes, s.w_modes)
        m = s.w_modes
        if s.bc == "clamped":
            ph = np.array([_clamped_beam(xi, k + 1)[0][0] for k in range(m)])
            ps = np.array([_clamped_beam(eta, k + 1)[0][0] for k in range(m)])
        else:
            ph = np.array([np.sin((k + 1) * np.pi * 0.5) for k in range(m)])
            ps = ph.copy()
        return float(ph @ Aw @ ps)

    def _newton_disp(self, A: np.ndarray, q: float, delta: float,
                     f_hat: np.ndarray, mask: np.ndarray):
        """Augmented Newton for displacement control (unknowns ``A, q``).

        The control quantity is the panel-centre deflection (smooth); the
        augmented system is solved with a finite-difference tangent and a
        backtracking line search.  Used for the *elastic* large-deflection
        path, for which it converges in a few iterations for all deflections.
        """
        def fun(x):
            Ax, qx = x[:-1], float(x[-1])
            R = self.residual(Ax, qx, mask)
            return np.concatenate([R, [self.w_center(Ax) - delta]])

        x = np.concatenate([A, [q]])
        F = fun(x)
        F0 = max(float(np.linalg.norm(F)), 1e-30)
        n = F.size
        J = None
        for it in range(self.max_newton):
            Fn = float(np.linalg.norm(F))
            if Fn < self.tol * F0:
                break
            if J is None:
                J = np.empty((n, n))
                for j in range(n):
                    h = 1e-7 * (1.0 + abs(x[j]))
                    xp = x.copy()
                    xp[j] += h
                    J[:, j] = (fun(xp) - F) / h
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -F, rcond=None)[0]
            Fn = float(np.linalg.norm(F))
            lam, ok = 1.0, False
            for _ls in range(12):
                xt = x + lam * dx
                Ft = fun(xt)
                if float(np.linalg.norm(Ft)) < Fn:
                    x, F, ok = xt, Ft, True
                    break
                lam *= 0.5
            if not ok:
                x = x + 0.1 * dx
                F = fun(x)
                J = None
            elif it % 2 == 1:
                J = None
        A_out, q_out = x[:-1], float(x[-1])
        Rf = self.residual(A_out, q_out, mask)
        f_scale = max(abs(q_out) * float(np.abs(f_hat).sum()), 1e-30)
        force_rel = float(np.linalg.norm(Rf)) / f_scale
        con_rel = abs(self.w_center(A_out) - delta) / max(abs(delta), 1e-9)
        ok = bool(con_rel < 1e-3 and force_rel < 5e-2)
        return A_out, q_out, ok, con_rel

    def _advance(self, A: np.ndarray, q: float, delta: float,
                 f_hat: np.ndarray, mask: np.ndarray, max_sub: int = 8):
        """Advance to ``delta``, halving the step until Newton converges."""
        pending = [float(delta)]
        A_cur, q_cur = A, q
        tries = 0
        while pending:
            target = pending.pop(0)
            A_new, q_new, ok, _rel = self._newton_disp(A_cur, q_cur, target, f_hat, mask)
            if ok:
                A_cur, q_cur = A_new, q_new
                tries = 0
                continue
            tries += 1
            if tries > max_sub:
                # accept the best available state and move on to the next target
                A_cur, q_cur = A_new, q_new
                tries = 0
                continue
            mid = 0.5 * (self.w_center(A_cur) + target)
            pending = [mid, target] + pending
        return A_cur, q_cur

    def _advance_load(self, A: np.ndarray, q_target: float, mask: np.ndarray,
                      max_sub: int = 10) -> np.ndarray:
        """Advance the load level, halving increments until Newton converges."""
        pending = [float(q_target)]
        cur = A
        tries = 0
        q_cur = 0.0
        while pending:
            q_next = pending.pop(0)
            A_new = self._newton(cur, q_next, mask)
            if getattr(self, "last_rel_residual", 1.0) < 1e-4:
                cur, q_cur = A_new, q_next
                continue
            tries += 1
            if tries > max_sub:
                cur = A_new
                break
            pending = [0.5 * (q_cur + q_next), q_next] + pending
        return cur

    def reference_shapes(self, mask: Optional[np.ndarray] = None,
                         q_levels: Optional[Sequence[float]] = None):
        """Normalised elastic deformation profiles along an elastic load path.

        Returns ``(deltas, shapes)`` where ``shapes[j]`` is the generalised
        coordinate vector normalised to a unit centre deflection.  Interpolating
        between these profiles reproduces the *evolution* of the deformation
        shape from bending dominated (small deflection) to membrane dominated
        (large deflection), which a single frozen profile cannot.
        """
        import dataclasses

        spec = self.spec
        elastic_grade = dataclasses.replace(spec.grade, sigma_y=1.0e15, E_t=0.0)
        elastic_spec = dataclasses.replace(spec, grade=elastic_grade)
        elastic = VonKarmanPanel(elastic_spec, load_steps=self.load_steps,
                                 max_newton=self.max_newton, tol=self.tol)
        mask = self._full_mask() if mask is None else mask
        if q_levels is None:
            q_levels = np.geomspace(1.0e5, 2.0e8, max(6, self.load_steps // 2))
        A = np.zeros(self.n_dof)
        deltas, shapes = [], []
        for q in q_levels:
            A = elastic._advance_load(A, float(q), mask)
            d = elastic.w_center(A)
            if d > 1e-6:
                deltas.append(float(d))
                shapes.append(A / d)
        order = np.argsort(deltas)
        return np.asarray(deltas)[order], np.asarray(shapes)[order]

    def _shape_at(self, delta: float, deltas: np.ndarray, shapes: np.ndarray) -> np.ndarray:
        if deltas.size == 1:
            return float(delta) * shapes[0]
        return float(delta) * np.array([np.interp(delta, deltas, shapes[:, k])
                                        for k in range(shapes.shape[1])])

    def reference_shape(self, mask: Optional[np.ndarray] = None,
                        q_ref: float = 1.0e7) -> np.ndarray:
        """Normalised reference deformation profile (centre deflection = 1).

        The profile is the *geometrically non-linear elastic* solution at a
        moderate load level, i.e. the membrane-dominated shape that the panel
        actually develops while it is being indented.  Fixing this shape and
        scaling its amplitude is the standard fixed-profile assumption of
        grounding indentation analysis (Simonsen & Pedersen); it makes the
        resistance a *lower bound* once the panel yields (the real panel can
        redistribute deformation and carry more load), which is the
        conservative direction for damage assessment.
        """
        import dataclasses

        spec = self.spec
        elastic_grade = dataclasses.replace(spec.grade, sigma_y=1.0e15, E_t=0.0)
        elastic_spec = dataclasses.replace(spec, grade=elastic_grade)
        elastic = VonKarmanPanel(elastic_spec, load_steps=self.load_steps,
                                 max_newton=self.max_newton, tol=self.tol)
        mask = self._full_mask() if mask is None else mask
        A = np.zeros(self.n_dof)
        A = elastic._advance_load(A, float(q_ref), mask)
        wc = elastic.w_center(A)
        if abs(wc) < 1e-12:
            return A
        return A / wc

    def solve_loadpath(self, delta_path: Sequence[float],
                       mask: Optional[np.ndarray] = None) -> PanelResponse:
        """Resistance curve along a prescribed deflection path.

        For every deflection the profile is ``A = delta * f`` (``f`` from
        :meth:`reference_shape`), the elastoplastic layered stress state is
        integrated incrementally (sub-incremented return mapping), and the
        contact pressure follows from virtual work with the *current*
        elastoplastic stress resultants:

        ``q = (f_int . f_hat) / (f_hat . f_hat)``
        """
        s = self.spec
        mask = self._full_mask() if mask is None else mask
        f_hat = self._unit_pressure_load(mask)
        wvec = f_hat[:self.n_w]
        denom = float(wvec @ wvec)
        deltas_ref, shapes_ref = self.reference_shapes(mask)
        q_out, w_out, vol_out, e_out = [], [], [], []
        eps_out, svm_out, N_out, M_out, torn_out = [], [], [], [], []
        V_prev = E_prev = q_prev = 0.0
        A_last = None
        for delta in delta_path:
            A = self._shape_at(float(delta), deltas_ref, shapes_ref)
            A_last = A
            w, _, _, eps_m, kappa = self._kinematics(A)
            N, M, eps_p_new, alpha_new = self._stress_resultants(eps_m, kappa)
            eps_pl = alpha_new.reshape(s.n_layers, s.n_y, s.n_x).max(axis=0)
            R_int = self.residual(A, 0.0, mask)
            q = float(R_int[:self.n_w] @ wvec) / denom if denom > 0.0 else 0.0
            V = float((w * mask * self.w2d).sum()) * s.a * s.b
            E_prev += max(0.5 * (q + q_prev) * (V - V_prev), 0.0)
            V_prev, q_prev = V, q
            q_out.append(q)
            w_out.append(float(self.w_center(A)))
            vol_out.append(V)
            e_out.append(E_prev)
            eps_out.append(float(eps_pl.max()))
            svm_out.append(float(self._von_mises_max(eps_m, kappa, eps_p_new, alpha_new)))
            N_out.append(float(np.abs(N).max()))
            M_out.append(float(np.abs(M).max()))
            torn_out.append(float((eps_pl >= s.grade.eps_f).mean()))
            self._eps_p = eps_p_new
            self._alpha = alpha_new
            self._eps_commit = getattr(self, "_eps_trial", self._eps_commit)
        w_final, _, _, _, _ = self._kinematics(A_last)
        eps_pl_field = self._alpha.reshape(s.n_layers, s.n_y, s.n_x).max(axis=0)
        return PanelResponse(
            pressure=np.array(q_out), deflection=np.array(w_out),
            volume=np.array(vol_out), energy=np.array(e_out),
            eps_pl_max=np.array(eps_out), sigma_vm_max=np.array(svm_out),
            membrane_force=np.array(N_out), bend_moment=np.array(M_out),
            torn_fraction=np.array(torn_out), mode_shape=w_final,
            coords=(self.x, self.y), eps_pl_field=eps_pl_field, spec=s,
        )

    def solve_displacement(self, delta_path: Sequence[float],
                           mask: Optional[np.ndarray] = None) -> PanelResponse:
        """Alias of :meth:`solve_loadpath` (displacement-controlled parametrisation)."""
        return self.solve_loadpath(delta_path, mask=mask)

    def solve(self, pressure_path: Sequence[float],
              mask: Optional[np.ndarray] = None) -> PanelResponse:
        """Load-controlled analysis along a prescribed pressure path (elastic use)."""
        s = self.spec
        mask = self._full_mask() if mask is None else mask
        A = np.zeros(self.n_dof)
        q_out, w_out, vol_out, e_out = [], [], [], []
        eps_out, svm_out, N_out, M_out, torn_out = [], [], [], [], []
        V_prev = 0.0
        E_prev = 0.0
        q_prev = 0.0
        for q in pressure_path:
            A = self._advance_load(A, float(q), mask)
            w, _, _, eps_m, kappa = self._kinematics(A)
            N, M, eps_p_new, alpha_new = self._stress_resultants(eps_m, kappa)
            eps_pl = alpha_new.reshape(s.n_layers, s.n_y, s.n_x).max(axis=0)
            V = float((w * mask * self.w2d).sum()) * s.a * s.b
            dE = 0.5 * (float(q) + q_prev) * (V - V_prev)
            E_prev += max(dE, 0.0)
            V_prev, q_prev = V, float(q)
            q_out.append(float(q))
            w_out.append(float(self.w_center(A)))
            vol_out.append(V)
            e_out.append(E_prev)
            eps_out.append(float(eps_pl.max()))
            svm_out.append(float(self._von_mises_field(eps_m, kappa).max()))
            N_out.append(float(np.abs(N).max()))
            M_out.append(float(np.abs(M).max()))
            torn_out.append(float((eps_pl >= s.grade.eps_f).mean()))
            self._eps_p, self._alpha = eps_p_new, alpha_new
        w_final, _, _, _, _ = self._kinematics(A)
        eps_pl_field = self._alpha.reshape(s.n_layers, s.n_y, s.n_x).max(axis=0)
        return PanelResponse(
            pressure=np.array(q_out), deflection=np.array(w_out),
            volume=np.array(vol_out), energy=np.array(e_out),
            eps_pl_max=np.array(eps_out), sigma_vm_max=np.array(svm_out),
            membrane_force=np.array(N_out), bend_moment=np.array(M_out),
            torn_fraction=np.array(torn_out), mode_shape=w_final,
            coords=(self.x, self.y), eps_pl_field=eps_pl_field, spec=s,
        )

    # ------------------------------------------------------------ public APIs
    def pressure_deflection(self, pressure_max: float, load_steps: Optional[int] = None,
                            mask: Optional[np.ndarray] = None) -> PanelResponse:
        """Load-controlled solve ``0 -> pressure_max`` (geometric load steps)."""
        n = self.load_steps if load_steps is None else int(load_steps)
        path = pressure_max * np.geomspace(1e-3, 1.0, n)
        return self.solve(path, mask=mask)

    def resistance_curve(self, delta_max: float = 0.30, steps: Optional[int] = None,
                         mask: Optional[np.ndarray] = None) -> PanelResponse:
        """Resistance curve ``q(delta)`` up to ``delta_max`` (m)."""
        n = self.load_steps if steps is None else int(steps)
        path = delta_max * np.geomspace(2e-3, 1.0, n)
        return self.solve_loadpath(path, mask=mask)

    def deflection_to_pressure(self, target_deflection: float,
                               pressure_max: float = 5.0e6) -> Tuple[float, PanelResponse]:
        """Bisection on the load level (elastic panels only)."""
        lo, hi = 1e-2 * pressure_max, pressure_max
        resp = None
        for _ in range(20):
            mid = np.sqrt(lo * hi)
            resp = self.pressure_deflection(mid)
            if resp.deflection.max() < target_deflection:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) / hi < 1e-3:
                break
        return mid, resp

    def _von_mises_max(self, eps_m: np.ndarray, kappa: np.ndarray,
                       eps_p: np.ndarray, alpha: np.ndarray) -> float:
        """Peak von Mises stress over all layers for a *given* plastic state."""
        s = self.spec
        z, _w = self._thickness_points()
        ncell = s.n_x * s.n_y
        out = 0.0
        for k, zi in enumerate(z):
            sl = slice(k * ncell, (k + 1) * ncell)
            sig, _, _, _ = plane_stress_return_map(
                np.moveaxis(eps_m + zi * kappa, 0, -1).reshape(-1, 3), eps_p[sl], alpha[sl],
                s.material)
            sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
            out = max(out, float(np.sqrt(np.maximum(
                sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, 0.0)).max()))
        return out

    def _von_mises_field(self, eps_m: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        s = self.spec
        z, _zw = self._thickness_points()
        nl = len(z)
        out = np.zeros((s.n_y, s.n_x))
        ncell = s.n_x * s.n_y
        for k, zi in enumerate(z):
            sl = slice(k * ncell, (k + 1) * ncell)
            sig, _, _, _ = plane_stress_return_map(
                np.moveaxis(eps_m + zi * kappa, 0, -1).reshape(-1, 3),
                self._eps_p[sl], self._alpha[sl], s.material)
            sxx, syy, sxy = sig[:, 0], sig[:, 1], sig[:, 2]
            vm = np.sqrt(np.maximum(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2, 0.0))
            out = np.maximum(out, vm.reshape(s.n_y, s.n_x))
        return out

    # ------------------------------------------------------------ public APIs
    def pressure_deflection(self, pressure_max: float, load_steps: Optional[int] = None,
                            mask: Optional[np.ndarray] = None) -> PanelResponse:
        """Solve a load path ``0 -> pressure_max`` with geometric load steps."""
        n = self.load_steps if load_steps is None else int(load_steps)
        # geometric spacing resolves the stiff membrane regime without wasting
        # steps in the soft bending regime
        path = pressure_max * np.geomspace(1e-3, 1.0, n)
        return self.solve(path, mask=mask)

    def deflection_to_pressure(self, target_deflection: float,
                               pressure_max: float = 5.0e6) -> Tuple[float, PanelResponse]:
        """Bisection on the load level to reach a prescribed max deflection."""
        lo, hi = 1e-2 * pressure_max, pressure_max
        resp = None
        for _ in range(24):
            mid = np.sqrt(lo * hi)
            resp = self.pressure_deflection(mid)
            if resp.deflection.max() < target_deflection:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) / hi < 1e-3:
                break
        return mid, resp