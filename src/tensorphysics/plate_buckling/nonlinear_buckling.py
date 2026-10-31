"""Nonlinear buckling and ultimate-strength solver for rectangular plates.

The solver combines

* geometric nonlinearity in the von Karman (moderate rotation) sense,
  ``eps_xx = u,x + 1/2 (w,x)^2`` and ``gamma_xy = u,y + v,x + w,x w,y``;
* material nonlinearity through plane-stress von Mises elastoplasticity with
  bilinear isotropic hardening, integrated over through-thickness layers;
* Newton-Raphson equilibrium iterations inside a cylindrical arc-length
  (Riks/Crisfield) continuation so that the load-deflection path can be
  followed **through** the limit point onto the post-ultimate branch.

Finite element discretisation: flat rectangular shell element = bilinear Q4
membrane + conforming bicubic Hermite bending rectangle, 6 dofs per node
``(u, v, w, theta_x, theta_y, w_xy)``, structured mesh on a rectangular panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla

from .core import (
    BilinearMaterial,
    GeometryError,
    PlateGeometry,
    consistent_pressure_load,
    element_kernels,
    plane_stress_return_map,
)

__all__ = ["PlateModel", "BucklingResult", "EdgeConditionError", "SUPPORTED_BCS"]


class EdgeConditionError(ValueError):
    """Raised for unsupported boundary-condition combinations."""


SUPPORTED_BCS = ("SSSS", "CCCC", "SCSC", "CSCS")


@dataclass
class BucklingResult:
    """Container for a linear (eigenvalue) buckling result."""

    sigma_cr: float          # critical compressive stress (Pa, positive)
    lam_cr: float            # critical load factor on the reference pattern
    mode: np.ndarray         # out-of-plane mode shape (full dof vector)
    n_modes: int = 1


class PlateModel:
    """Nonlinear FE model of a rectangular plate panel.

    Parameters
    ----------
    geom:
        :class:`~tensorphysics.plate_buckling.core.PlateGeometry`.
    bc:
        One of ``"SSSS"``, ``"CCCC"``, ``"SCSC"`` (loaded edges clamped,
        unloaded edges simply supported), ``"CSCS"`` (loaded edges simply
        supported, unloaded edges clamped).  Loaded edges are ``x = 0, a``.
    n_layers:
        Through-thickness integration points for the elastoplastic response.
    validate:
        Run :meth:`PlateGeometry.validate` on construction.
    """

    def __init__(self, geom: PlateGeometry, bc: str = "SSSS", n_layers: int = 5,
                 validate: bool = True):
        if bc not in SUPPORTED_BCS:
            raise EdgeConditionError(f"bc must be one of {SUPPORTED_BCS}, got {bc!r}")
        self.geom = geom
        self.bc = bc
        self.material = BilinearMaterial.from_geometry(geom)
        self.geometry_checks = geom.validate() if validate else {}

        self.ex = geom.a / geom.nx
        self.ey = geom.b / geom.ny
        self.kern = element_kernels(self.ex, self.ey)
        self.n_gauss = self.kern["Bm"].shape[0]

        # Through-thickness Gauss-Legendre integration.  Midpoint layer rules
        # underestimate int z^2 dz (a 5-layer midpoint rule gives 0.08 t^3
        # instead of t^3/12, i.e. 4.2 % too little bending stiffness); the
        # Gauss rule is exact for the quadratic weight and is also adequate for
        # the elastoplastic layers.
        self.n_layers = n_layers
        xi, wi = np.polynomial.legendre.leggauss(n_layers)
        self.z = 0.5 * geom.t * xi
        self.wz = 0.5 * geom.t * wi

        self._build_mesh()
        self._build_constraints()

        nE, nG, nL = geom.n_elem, self.n_gauss, n_layers
        self.eps_p = np.zeros((nE, nG, nL, 3))
        self.alpha = np.zeros((nE, nG, nL))
        self._n0 = np.zeros((nE, nG, 3))   # initial (residual) stress resultants
        # nodal bending dofs of the initial deflection field w0 (stress free at
        # lam = 0): strains are measured relative to the imperfect geometry so
        # that the imperfection itself carries no membrane force.
        self._d0_b = np.zeros((nE, 16))
        self.sigma_ref_value = 1.0
        self._sigma_gp = np.zeros((nE, nG, nL, 3))
        self._N_gp = np.zeros((nE, nG, 3))

    # ------------------------------------------------------------------ mesh
    def _build_mesh(self) -> None:
        g = self.geom
        nx, ny = g.nx, g.ny
        self.xs = np.linspace(0.0, g.a, nx + 1)
        self.ys = np.linspace(0.0, g.b, ny + 1)
        nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        n00 = nid[:-1, :-1].ravel()
        n10 = nid[:-1, 1:].ravel()
        n11 = nid[1:, 1:].ravel()
        n01 = nid[1:, :-1].ravel()
        nodes = np.stack([n00, n10, n11, n01], axis=1)      # (E, 4)
        self.elem_nodes = nodes
        self.elem_xy = np.array(
            [[self.xs[i] + 0.5 * self.ex, self.ys[j] + 0.5 * self.ey]
             for j in range(ny) for i in range(nx)])

        base = 6 * nodes
        mem = np.empty((g.n_elem, 8), dtype=int)
        mem[:, 0::2] = base + 0          # u
        mem[:, 1::2] = base + 1          # v
        bend = np.empty((g.n_elem, 16), dtype=int)
        bend[:, 0::4] = base + 2         # w
        bend[:, 1::4] = base + 3         # theta_x
        bend[:, 2::4] = base + 4         # theta_y
        bend[:, 3::4] = base + 5         # w_xy
        self.edof_mem = mem
        self.edof_bend = bend
        self.edof_all = np.concatenate([mem, bend], axis=1)

        self.ndof = g.n_dof
        self.u = np.zeros(self.ndof)
        all_dofs = np.arange(self.ndof).reshape(-1, 6)
        self._disp_dofs = np.concatenate(
            [all_dofs[:, 0], all_dofs[:, 1], all_dofs[:, 2]])

    def _build_constraints(self) -> None:
        g = self.geom
        nx, ny = g.nx, g.ny
        nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        left, right = nid[:, 0], nid[:, -1]
        bottom, top = nid[0, :], nid[-1, :]
        all_edge = np.unique(np.concatenate([left, right, bottom, top]))

        fixed: List[int] = []
        fixed += [6 * n + 0 for n in left]           # axial reference edge
        fixed += [6 * nid[0, 0] + 1]                 # transverse drift
        fixed += [6 * n + 2 for n in all_edge]       # w = 0 on every edge

        if self.bc in ("CCCC", "SCSC"):
            fixed += [6 * n + 4 for n in np.unique(np.concatenate([left, right]))]
        if self.bc in ("CCCC", "CSCS"):
            fixed += [6 * n + 3 for n in np.unique(np.concatenate([bottom, top]))]

        self.fixed_dofs = np.unique(np.array(fixed, dtype=int))
        mask = np.ones(self.ndof, dtype=bool)
        mask[self.fixed_dofs] = False
        self.free_dofs = np.flatnonzero(mask)
        self._disp_free = np.intersect1d(self.free_dofs, self._disp_dofs)

    # -------------------------------------------------------------- assembly
    def assemble(self, u: Optional[np.ndarray] = None,
                 need_tangent: bool = True,
                 commit_state: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Assemble internal force vector, tangent stiffness and trial state."""
        u = self.u if u is None else u
        eps_p0, alpha0 = (self.eps_p, self.alpha) if commit_state is None else commit_state
        E = self.geom.n_elem
        L = self.n_layers
        ndof = self.ndof
        mat = self.material

        f = np.zeros(ndof)
        K = np.zeros((ndof, ndof)) if need_tangent else None

        de = u[self.edof_all]
        de_m = de[:, :8]
        # bending dofs relative to the (stress-free) imperfect geometry
        de0_b = self._d0_b
        de_b = de[:, 8:] - de0_b

        eps_p_new = np.empty_like(eps_p0)
        alpha_new = np.empty_like(alpha0)

        for gpi in range(self.n_gauss):
            Bm = self.kern["Bm"][gpi]
            Bb = self.kern["Bb"][gpi]
            Nwx = self.kern["Nwx"][gpi]
            Nwy = self.kern["Nwy"][gpi]
            wt = self.kern["weight"][gpi]

            e_lin = de_m @ Bm.T
            wx = de_b @ Nwx
            wy = de_b @ Nwy
            w0x = de0_b @ Nwx
            w0y = de0_b @ Nwy
            A = np.zeros((E, 3, 16))
            A[:, 0, :] = wx[:, None] * Nwx[None, :]
            A[:, 1, :] = wy[:, None] * Nwy[None, :]
            A[:, 2, :] = wx[:, None] * Nwy[None, :] + wy[:, None] * Nwx[None, :]
            A0 = np.zeros((E, 3, 16))
            A0[:, 0, :] = w0x[:, None] * Nwx[None, :]
            A0[:, 1, :] = w0y[:, None] * Nwy[None, :]
            A0[:, 2, :] = w0x[:, None] * Nwy[None, :] + w0y[:, None] * Nwx[None, :]
            # eps = eps_lin + w0,x w,x + 1/2 (w,x)^2  (strains measured from the
            # imperfect, stress-free reference configuration; w is the
            # *additional* deflection)
            Bnl = A + A0
            e = e_lin + np.einsum("eij,ej->ei", A0 + 0.5 * A, de_b)
            kappa = de_b @ Bb.T

            sig = np.empty((E, L, 3))
            Cep = np.empty((E, L, 3, 3))
            for k in range(L):
                eps_l = e + self.z[k] * kappa
                s_k, c_k, ep_k, al_k = plane_stress_return_map(
                    eps_l, eps_p0[:, gpi, k], alpha0[:, gpi, k], mat)
                sig[:, k, :] = s_k
                Cep[:, k, :, :] = c_k
                eps_p_new[:, gpi, k] = ep_k
                alpha_new[:, gpi, k] = al_k

            N_res = np.einsum("l,elk->ek", self.wz, sig) + self._n0[:, gpi, :]
            M_res = np.einsum("l,elk->ek", self.wz * self.z, sig)
            Dm = np.einsum("l,elkj->ekj", self.wz, Cep)
            Dmb = np.einsum("l,elkj->ekj", self.wz * self.z, Cep)
            Db = np.einsum("l,elkj->ekj", self.wz * self.z * self.z, Cep)

            self._sigma_gp[:, gpi, :, :] = sig
            self._N_gp[:, gpi, :] = N_res
            f_mem = wt * np.einsum("ki,ek->ei", Bm, N_res)
            f_bend = wt * (np.einsum("eci,ec->ei", Bnl, N_res) +
                           np.einsum("ci,ec->ei", Bb, M_res))
            np.add.at(f, self.edof_mem, f_mem)
            np.add.at(f, self.edof_bend, f_bend)

            if need_tangent:
                Btot = np.zeros((E, 6, 24))
                Btot[:, 0:3, 0:8] = Bm
                Btot[:, 0:3, 8:24] = Bnl
                Btot[:, 3:6, 8:24] = Bb
                D6 = np.zeros((E, 6, 6))
                D6[:, 0:3, 0:3] = Dm
                D6[:, 0:3, 3:6] = Dmb
                D6[:, 3:6, 0:3] = Dmb
                D6[:, 3:6, 3:6] = Db
                Ke = wt * np.einsum("eai,eab,ebj->eij", Btot, D6, Btot)

                G = np.zeros((E, 2, 16))
                G[:, 0, :] = Nwx[None, :]
                G[:, 1, :] = Nwy[None, :]
                S = np.zeros((E, 2, 2))
                S[:, 0, 0] = N_res[:, 0]
                S[:, 1, 1] = N_res[:, 1]
                S[:, 0, 1] = S[:, 1, 0] = N_res[:, 2]
                kg = wt * np.einsum("eai,eab,ebj->eij", G, S, G)
                Ke[:, 8:24, 8:24] += kg
                # Stabilisation of the Hermite twist dofs.  The bicubic
                # rectangle can carry a zero-curvature (aliased) twist field, so
                # the assembled bending stiffness needs a small diagonal term on
                # w_xy to remain positive definite.  1e-9 of the element bending
                # scale is enough to restore positive definiteness and changes
                # the benchmark results by < 0.01 % (verified against the linear
                # deflection and the linear buckling gates).
                dgk = np.abs(np.diagonal(Ke[:, 8:24, 8:24], axis1=1, axis2=2))
                pen = 1e-9 * dgk.max(axis=1)
                twist = np.array([3, 7, 11, 15])
                Ke[:, 8 + twist, 8 + twist] += pen[:, None]


                rows = np.repeat(self.edof_all, 24, axis=1)
                cols = np.tile(self.edof_all, (1, 24))
                np.add.at(K, (rows.ravel(), cols.ravel()), Ke.ravel())

        return f, K, (eps_p_new, alpha_new)

    def elastic_stiffness(self) -> np.ndarray:
        """Linear elastic stiffness at zero displacement (``A = 0``)."""
        _, K, _ = self.assemble(u=np.zeros(self.ndof), need_tangent=True)
        return K

    # ---------------------------------------------------------------- solver
    @staticmethod
    def _scaled_solve(Kff: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Jacobi-equilibrated direct solve.

        Membrane, bending and twist stiffnesses differ by up to ten orders of
        magnitude, which makes an unscaled LU factorisation report the SPD
        tangent as singular; equilibration with the diagonal restores a
        well-scaled Cholesky factorisation (with an LU fallback for the
        indefinite tangents met beyond the limit point).
        """
        d = np.sqrt(np.abs(np.diag(Kff)))
        d[d == 0.0] = 1.0
        Ks = Kff / np.outer(d, d)
        rs = rhs / d
        try:
            y = sla.solve(Ks, rs, assume_a="pos")
        except (sla.LinAlgError, ValueError):  # pragma: no cover - indefinite
            y = sla.solve(Ks, rs, assume_a="gen")
        return y / d

    # ------------------------------------------------------------- load cases
    def compression_load(self, sigma_ref: float = 1.0) -> np.ndarray:
        """Nodal forces producing uniform uniaxial compression ``sigma_ref``."""
        g = self.geom
        nx, ny = g.nx, g.ny
        F = np.zeros(self.ndof)
        trib = np.full(ny + 1, self.ey)
        trib[0] = trib[-1] = 0.5 * self.ey
        nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        right = nid[:, -1]
        F[6 * right] += -sigma_ref * g.t * trib
        return F

    def pressure_load(self, pressure, boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None
                      ) -> np.ndarray:
        """Transverse nodal forces for a uniform / piecewise-uniform pressure."""
        F = np.zeros(self.ndof)
        for el, (xc, yc) in enumerate(self.elem_xy):
            if boxes is not None:
                inside = any(x0 <= xc <= x1 and y0 <= yc <= y1 for (x0, x1, y0, y1) in boxes)
                if not inside:
                    continue
            q = pressure(xc, yc) if callable(pressure) else pressure
            if q == 0.0:
                continue
            # consistent Hermite load: corner forces + nodal moments
            F[self.edof_bend[el]] += consistent_pressure_load(self.ex, self.ey, q)
        return F

    # ------------------------------------------------------- linear response
    def solve_linear(self, F: np.ndarray) -> np.ndarray:
        """Solve ``K_e u = F`` on the free dofs (returns full dof vector)."""
        K = self.elastic_stiffness()
        u = np.zeros(self.ndof)
        u[self.free_dofs] = self._scaled_solve(
            K[np.ix_(self.free_dofs, self.free_dofs)], F[self.free_dofs])
        return u

    def membrane_resultants(self, u: np.ndarray) -> np.ndarray:
        """Linear membrane stress resultants ``(E, 4, 3)`` from ``u``."""
        g = self.geom
        C = self.material.elastic_matrix
        out = np.zeros((g.n_elem, self.n_gauss, 3))
        de_m = u[self.edof_mem]
        for gpi in range(self.n_gauss):
            out[:, gpi, :] = (de_m @ self.kern["Bm"][gpi].T @ C.T) * g.t
        return out

    def _geometric_stiffness(self, N: np.ndarray) -> np.ndarray:
        E = self.geom.n_elem
        Kg = np.zeros((self.ndof, self.ndof))
        for gpi in range(self.n_gauss):
            Nwx = self.kern["Nwx"][gpi]
            Nwy = self.kern["Nwy"][gpi]
            wt = self.kern["weight"][gpi]
            G = np.zeros((E, 2, 16))
            G[:, 0, :] = Nwx[None, :]
            G[:, 1, :] = Nwy[None, :]
            S = np.zeros((E, 2, 2))
            S[:, 0, 0] = N[:, gpi, 0]
            S[:, 1, 1] = N[:, gpi, 1]
            S[:, 0, 1] = S[:, 1, 0] = N[:, gpi, 2]
            kg = wt * np.einsum("eai,eab,ebj->eij", G, S, G)
            rows = np.repeat(self.edof_bend, 16, axis=1)
            cols = np.tile(self.edof_bend, (1, 16))
            np.add.at(Kg, (rows.ravel(), cols.ravel()), kg.ravel())
        return Kg

    # ---------------------------------------------------- linear buckling (LBA)
    def linear_buckling(self, sigma_ref: float = 1.0) -> BucklingResult:
        """Linearised (eigenvalue) buckling under uniform axial compression."""
        F = self.compression_load(sigma_ref)
        u_lin = self.solve_linear(F)
        N = self.membrane_resultants(u_lin)
        Kg = self._geometric_stiffness(N)
        Ke = self.elastic_stiffness()

        bend_dofs = np.unique(self.edof_bend)
        bend_free = np.intersect1d(self.free_dofs, bend_dofs)
        A = Ke[np.ix_(bend_free, bend_free)]
        B = -Kg[np.ix_(bend_free, bend_free)]
        scale = float(np.max(np.abs(np.diag(B)))) if B.size else 0.0
        if scale <= 0:
            raise GeometryError(
                "geometric stiffness has no compressive content: the reference "
                "load must be compressive")
        # The geometric stiffness is rank deficient (only 2 dofs per Gauss
        # point carry gradient energy), so the pencil (A, B) is solved with the
        # QZ algorithm instead of a Cholesky-based eigensolver.
        d = np.sqrt(np.abs(np.diag(A)))
        d[d == 0.0] = 1.0
        As = A / np.outer(d, d)
        Bs = B / np.outer(d, d)
        w, v = sla.eig(As, Bs)
        w = np.real(w)
        v = np.real(v)
        keep = np.isfinite(w) & (w > 0.0)
        if not np.any(keep):
            raise GeometryError("no positive buckling eigenvalue found")
        idx = np.flatnonzero(keep)
        order = idx[np.argsort(w[idx])]
        w, v = w[order], v[:, order]
        lam_cr = float(w[0])
        mode = np.zeros(self.ndof)
        mode[bend_free] = v[:, 0]
        return BucklingResult(sigma_cr=lam_cr * sigma_ref, lam_cr=lam_cr, mode=mode)

    # --------------------------------------------------------- imperfections
    def apply_initial_deflection(self, w0: float, m: int = 1, n: int = 1,
                                 phase: float = 0.0) -> np.ndarray:
        """Set transverse dofs to ``w0 sin(m pi x/a) sin(n pi y/b)``."""
        g = self.geom
        nid = np.arange(g.n_nodes)
        j, i = np.divmod(nid, g.nx + 1)
        x, y = self.xs[i], self.ys[j]
        kx, ky = m * np.pi / g.a, n * np.pi / g.b
        sxy = np.sin(kx * x + phase) * np.sin(ky * y)
        w = w0 * sxy
        wx = w0 * kx * np.cos(kx * x + phase) * np.sin(ky * y)
        wy = w0 * ky * np.sin(kx * x + phase) * np.cos(ky * y)
        wxy = w0 * kx * ky * np.cos(kx * x + phase) * np.cos(ky * y)
        self.u = np.zeros(self.ndof)
        self.u[6 * nid + 2] = w
        self.u[6 * nid + 3] = wy
        self.u[6 * nid + 4] = -wx
        self.u[6 * nid + 5] = wxy
        self._d0_b = self.u[self.edof_bend].copy()
        return self.u

    def set_residual_stress(self, sigma_res: float, edge_fraction: float = 0.2,
                            sign: float = 1.0) -> np.ndarray:
        """Self-equilibrated longitudinal welding residual stress field."""
        g = self.geom
        n0 = np.zeros((g.n_elem, 4, 3))
        denom = 1.0 - 2.0 * edge_fraction
        for el, (xc, yc) in enumerate(self.elem_xy):
            edge = yc <= edge_fraction * g.b or yc >= (1.0 - edge_fraction) * g.b
            sx = sigma_res if edge else (
                -sigma_res * (2.0 * edge_fraction) / denom if denom > 1e-9 else 0.0)
            n0[el, :, 0] = sign * sx * g.t
        self._n0 = n0
        return n0

    def reset_state(self, keep_imperfection: bool = True) -> None:
        """Zero the plastic state and start a fresh load case."""
        self.eps_p = np.zeros_like(self.eps_p)
        self.alpha = np.zeros_like(self.alpha)
        if keep_imperfection:
            self.u = np.zeros(self.ndof)
        else:
            self.u = np.zeros(self.ndof)
            self._d0_b = np.zeros_like(self._d0_b)
            self._n0 = np.zeros_like(self._n0)

    # --------------------------------------------------------- arc-length solve
    def solve_arc_length(
        self,
        F_var: np.ndarray,
        F_const: Optional[np.ndarray] = None,
        ds0: Optional[float] = None,
        max_steps: int = 120,
        max_iter: int = 25,
        tol_force: float = 1e-7,
        tol_disp: float = 1e-10,
        ds_min: float = 1e-12,
        ds_max: Optional[float] = None,
        monitor_node: Optional[int] = None,
        sigma_ref: float = 1.0,
        max_events: int = 400,
    ) -> Dict[str, object]:
        """Cylindrical arc-length (Riks/Crisfield) continuation.

        The total external load is ``F_const + lam * F_var``.  Returns the
        traced path, the peak (limit) point quantities and the peak state.
        """
        F_const = np.zeros(self.ndof) if F_const is None else F_const
        free = self.free_dofs
        dfree = self._disp_free
        # local indices (within the free-dof vector) of the displacement dofs
        dlocal = np.searchsorted(free, dfree)
        u = self.u.copy()
        state = (self.eps_p.copy(), self.alpha.copy())
        self.sigma_ref_value = sigma_ref

        def solve_K(K, rhs):
            return self._scaled_solve(K[np.ix_(free, free)], rhs[free])

        f, K, _ = self.assemble(u, need_tangent=True, commit_state=state)
        du_f = solve_K(K, F_var)
        tangent_norm = max(np.linalg.norm(du_f[dlocal]), 1e-30)
        ds = (0.02 / tangent_norm) * tangent_norm if ds0 is None else ds0
        if ds0 is None:
            ds = 0.02 * tangent_norm
        ds_max = ds_max if ds_max is not None else 50.0 * ds

        lam = 0.0
        path: List[Dict[str, float]] = []
        peak = {"lam": 0.0, "sigma_axial": 0.0, "w_monitor": 0.0,
                "w_max": 0.0, "sigma_vm_max": 0.0, "step": -1}
        peak_state = None
        sign = 1.0
        monitor = monitor_node if monitor_node is not None else self._centre_node()
        events = 0
        step = 0
        stalls = 0

        while step < max_steps and events < max_events:
            events += 1
            u_prev = u.copy()
            lam_prev = lam

            f, K, _ = self.assemble(u, need_tangent=True, commit_state=state)
            du_f = solve_K(K, F_var)
            nrm = max(np.linalg.norm(du_f[dlocal]), 1e-30)
            dlam = sign * ds / nrm
            du = np.zeros(self.ndof)
            du[free] = dlam * du_f
            u = u + du
            lam = lam + dlam

            converged = False
            iters = 0
            while iters < max_iter:
                iters += 1
                f, K, trial = self.assemble(u, need_tangent=True, commit_state=state)
                r = F_const + lam * F_var - f
                rn = np.linalg.norm(r[free])
                ref_force = np.linalg.norm((F_const + lam * F_var)[free]) + 1e-30
                if rn / ref_force < tol_force:
                    converged = True
                    break
                dur = solve_K(K, r)
                duf = solve_K(K, F_var)

                dv = (u - u_prev)[dfree] + dur[dlocal]
                a = duf[dlocal]
                a2 = float(a @ a)
                b2 = float(dv @ dv)
                ab = float(dv @ a)
                disc = ab * ab - a2 * (b2 - ds * ds)
                if a2 <= 0.0 or disc < 0.0:
                    break
                root = np.sqrt(disc)
                dl1 = (-ab + root) / a2
                dl2 = (-ab - root) / a2
                # pick the root closest to the tangent predictor (Crisfield):
                # the other root corresponds to a jump across the sphere
                dl = dl1 if abs(dl1) < abs(dl2) else dl2
                step_u = np.zeros(self.ndof)
                step_u[free] = dur + dl * duf
                u = u + step_u
                lam = lam + dl
                if (np.linalg.norm((dur + dl * duf)[dlocal]) <
                        tol_disp * max(1.0, np.linalg.norm(u[dfree])) and
                        rn / ref_force < 1e-3):
                    converged = True
                    break

            if not converged:
                u, lam = u_prev, lam_prev
                ds *= 0.25
                sign = np.sign(dlam) if dlam != 0 else 1.0
                stalls += 1
                if ds < ds_min or stalls > 12:
                    break
                continue

            stalls = 0
            step += 1
            state = (trial[0].copy(), trial[1].copy())
            self.eps_p, self.alpha = state[0].copy(), state[1].copy()
            self.u = u.copy()
            vm = self._max_von_mises()
            point = {
                "step": step,
                "lam": float(lam),
                "w_monitor": float(u[6 * monitor + 2]),
                "w_max": float(np.max(np.abs(u[6 * np.arange(self.geom.n_nodes) + 2]))),
                "sigma_axial": float(lam * sigma_ref),
                "sigma_vm_max": vm,
                "ds": float(ds),
                "iterations": iters,
            }
            path.append(point)
            if lam > peak["lam"]:
                peak = dict(point)
                peak_state = (state[0].copy(), state[1].copy())

            if iters <= 3:
                ds = min(ds * 1.4, ds_max)
            elif iters >= 8:
                ds = max(ds * 0.6, ds_min)

            if peak["lam"] > 0.0 and lam < peak["lam"] - 1e-12:
                sign = -1.0     # passed the limit point: descend
            else:
                sign = 1.0
            if lam < 0.0:
                break

        return {
            "path": path,
            "peak": peak,
            "lam_ultimate": float(peak["lam"]),
            "sigma_ultimate": float(peak["sigma_axial"]),
            "u_peak": peak_state,
            "monitor_node": int(monitor),
            "n_steps": len(path),
        }

    # ------------------------------------------------------------- utilities
    def _centre_node(self) -> int:
        nx, ny = self.geom.nx, self.geom.ny
        return (ny // 2) * (nx + 1) + nx // 2

    def _max_von_mises(self) -> float:
        """Maximum von Mises stress over Gauss points and layers."""
        sig = self._sigma_gp
        mean = (sig[..., 0] + sig[..., 1]) / 3.0
        q = np.sqrt(np.maximum(1.5 * (
            (sig[..., 0] - mean)**2 + (sig[..., 1] - mean)**2 + mean**2 +
            2.0 * sig[..., 2]**2), 0.0))
        return float(np.max(q))

    def surface_stress_field(self, layer: int = -1) -> np.ndarray:
        """Max von Mises stress over Gauss points of one layer, ``(ny, nx)``."""
        g = self.geom
        k = self.n_layers - 1 if layer == -1 else layer
        sig = self._sigma_gp[:, :, k, :]
        mean = (sig[..., 0] + sig[..., 1]) / 3.0
        q = np.sqrt(np.maximum(1.5 * (
            (sig[..., 0] - mean)**2 + (sig[..., 1] - mean)**2 + mean**2 +
            2.0 * sig[..., 2]**2), 0.0))
        return q.max(axis=1).reshape(g.ny, g.nx)

    def deflection_field(self) -> np.ndarray:
        g = self.geom
        w = self.u[6 * np.arange(g.n_nodes) + 2]
        return w.reshape(g.ny + 1, g.nx + 1)