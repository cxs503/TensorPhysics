"""Axisymmetric cylindrical shell under internal pressure: FE + closed form.

Physics target: an open-ended thin cylindrical shell (radius ``R``, wall ``t``,
length ``L``) under uniform internal pressure ``p``, clamped against radial
displacement and rotation at ``x = 0`` and held at the membrane expansion at
``x = L``.  The response is the textbook membrane-plus-edge-bending problem:

* far from the ends the shell is in a membrane state,
  ``sigma_theta = p R / t``, ``w = p R^2 / (E t)``, ``M_x -> 0``;
* the clamped edge forces a bending boundary layer whose amplitude decays like
  ``exp(-beta x)`` with

  .. math:: \\beta = \\left(\\frac{E t}{4 D R^2}\\right)^{1/4}
            = \\frac{(3(1-\\nu^2))^{1/4}}{\\sqrt{R t}},

  i.e. an attenuation length ``1/beta = 0.778 sqrt(R t)`` for ``nu = 0.3``
  (Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
  ch. 15-16; Calladine, *Theory of Shell Structures*, ch. 5).  This is the
  classic ``sqrt(R t)`` geometric-mean length scale of shell edge effects.

Model reduction used for the closed form (Donnell, axisymmetric, ``v = 0``,
no axial load):

.. math::
    D w'''' + \\frac{E t}{R^2} w = p,
    \\qquad \\varepsilon_x = u', \\; \\varepsilon_\\theta = w/R, \\;
    \\kappa_x = -w''

with total potential (per radian of circumference)

.. math::
    \\Pi = \\tfrac12 \\int_0^L \\big[ K(\\varepsilon_x^2
           + 2\\nu\\varepsilon_x\\varepsilon_\\theta + \\varepsilon_\\theta^2)
           + D \\kappa_x^2 - 2 p w \\big] dx,
    \\quad K = \\frac{E t}{1-\\nu^2}.

The **finite element carries the full membrane + bending coupling**, so the
closed form is an independent solution of the same variational problem, not an
input to the element.

Element: two nodes with ``(u, w, theta = dw/dx)``; ``u`` interpolated linearly
(``C^0``) and ``w`` by cubic Hermite (``C^1``), 4-point Gauss integration.  The
quadrature is exact for the Hermite bending term (degree 2 integrand), only
approximate for the coupled term ``2 nu eps_x eps_theta`` (``eps_x`` constant,
``eps_theta`` cubic -> degree 3, integrated exactly by 4-point Gauss); all
integrands here are polynomials of degree <= 6, well inside the exactness of a
4-point rule?  A 4-point Gauss rule is exact to degree 7, so yes: the element
integral is evaluated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "CylinderPressureFE",
    "cylinder_edge_solution",
    "beta_edge",
    "membrane_solution",
    "shell_convergence_study",
    "fit_edge_decay",
    "edge_decay_fit",
]

_GX, _GW = np.polynomial.legendre.leggauss(4)


def beta_edge(R: float, t: float, E: float, nu: float) -> float:
    """Edge-effect decay exponent ``beta = (E t / (4 D R^2))^{1/4}`` [1/m]."""
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    return (E * t / (4.0 * D * R ** 2)) ** 0.25


def membrane_solution(R: float, t: float, E: float, p: float) -> Dict[str, float]:
    """Membrane (unrestrained) state: ``sigma_theta = pR/t``, ``w = pR^2/(Et)``."""
    return {"sigma_theta": p * R / t, "w": p * R ** 2 / (E * t),
            "N_theta": p * R}


def _hermite(xi: float):
    """Cubic Hermite basis and its first two derivatives at ``xi in [0, 1]``."""
    h = np.array([1 - 3 * xi ** 2 + 2 * xi ** 3,
                  xi - 2 * xi ** 2 + xi ** 3,
                  3 * xi ** 2 - 2 * xi ** 3,
                  -xi ** 2 + xi ** 3])
    dh = np.array([-6 * xi + 6 * xi ** 2,
                   1 - 4 * xi + 3 * xi ** 2,
                   6 * xi - 6 * xi ** 2,
                   -2 * xi + 3 * xi ** 2])
    ddh = np.array([-6 + 12 * xi, -4 + 6 * xi, 6 - 12 * xi, -2 + 6 * xi])
    return h, dh, ddh


def cylinder_edge_solution(R: float, L: float, t: float, E: float, nu: float,
                           p: float, x) -> Dict[str, np.ndarray]:
    """Closed-form solution of ``D w'''' + (E t / R^2) w = p`` (Donnell).

    Boundary conditions: ``w(0) = 0``, ``w'(0) = 0`` (clamped edge, free to
    slide axially so that ``N_x = 0``) and ``w(L) = p R^2/(E t)``,
    ``w'(L) = 0`` (far end held at the membrane expansion).  The homogeneous
    part keeps the full four-term ``exp(+/- beta x)(cos, sin)`` combination, so
    the finite length is treated exactly (no semi-infinite approximation).

    Returns ``w``, ``M_x = -D w''``, ``N_theta = E t w / R``,
    ``sigma_theta = E w / R``, ``sigma_x_bend = 6 M_x / t^2`` (outer fibre) and
    the axial displacement from ``N_x = 0`` (``u' = -nu w / R``, ``u(0) = 0``).
    """
    x = np.asarray(x, dtype=float)
    Dplate = E * t ** 3 / (12.0 * (1.0 - nu ** 2))   # flexural rigidity
    beta = beta_edge(R, t, E, nu)
    w_p = p * R ** 2 / (E * t)

    bl = beta * L
    cL, sL = np.cos(bl), np.sin(bl)
    eL, eM = np.exp(bl), np.exp(-bl)
    # unknowns (A, B, C, D) of
    #  w = w_p + e^{bx}(A cos bx + B sin bx) + e^{-bx}(C cos bx + D sin bx)
    A_mat = np.array([
        [1.0, 0.0, 1.0, 0.0],                       # w(0) = 0
        [1.0, 1.0, -1.0, 1.0],                      # w'(0) = 0
        [eL * cL, eL * sL, eM * cL, eM * sL],        # w(L) = w_p
        [eL * (cL - sL), eL * (cL + sL),
         -eM * (cL + sL), eM * (cL - sL)],           # w'(L) = 0
    ])
    rhs = np.array([-w_p, 0.0, 0.0, 0.0])
    A, B, C, Dc = np.linalg.solve(A_mat, rhs)   # Dc = 4th constant (not D!)

    bx = beta * x
    ex, emx = np.exp(bx), np.exp(-bx)
    cs, sn = np.cos(bx), np.sin(bx)
    w = w_p + ex * (A * cs + B * sn) + emx * (C * cs + Dc * sn)
    # d2/dx2 of the homogeneous part (2nd derivative of e^{+-bx}(cos, sin))
    w_xx = beta ** 2 * (ex * (2.0 * B * cs - 2.0 * A * sn)
                        + emx * (-2.0 * Dc * cs + 2.0 * C * sn))
    Mx = -Dplate * w_xx
    Nt = E * t * w / R
    sigma_theta = E * w / R

    # axial displacement from N_x = 0 (trapezoidal integration on the given grid)
    flat = x.ravel()
    order = np.argsort(flat)
    xs = flat[order]
    w_sorted = w.ravel()[order]
    incr = 0.5 * (-nu / R) * (w_sorted[1:] + w_sorted[:-1]) * np.diff(xs)
    u_sorted = np.concatenate([[0.0], np.cumsum(incr)])
    u = np.empty_like(u_sorted)
    u[order] = u_sorted

    return {
        "x": x, "w": w, "w_membrane": np.full_like(w, w_p),
        "Mx": Mx, "N_theta": Nt, "sigma_theta": sigma_theta,
        "sigma_x_bend": 6.0 * Mx / t ** 2, "u": u.reshape(x.shape),
        "beta": np.array(beta), "w_p": np.array(w_p),
        "D": np.array(Dplate),
        "coeffs": np.array([A, B, C, Dc]),
    }


@dataclass
class CylinderPressureFE:
    """Axisymmetric cylindrical shell FE under internal pressure.

    Parameters
    ----------
    R, L, t : float
        Radius, length, thickness [m].
    E, nu : float
        Elastic constants.
    n_elem : int
        Number of axial elements.
    """

    R: float
    L: float
    t: float
    E: float
    nu: float
    n_elem: int = 32
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def D(self) -> float:
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    @property
    def Kps(self) -> float:
        return self.E * self.t / (1.0 - self.nu ** 2)

    @property
    def le(self) -> float:
        return self.L / self.n_elem

    @property
    def ndof(self) -> int:
        return 3 * (self.n_elem + 1)

    def _dofs(self, e: int) -> np.ndarray:
        return np.array([3 * e, 3 * e + 1, 3 * e + 2,
                         3 * e + 3, 3 * e + 4, 3 * e + 5])

    def element_matrices(self):
        """6x6 stiffness and unit-pressure load vector of a single element.

        Node dof triple ``(u, w, theta = dw/dx)``; ``w`` uses Hermite cubics so
        the stored slope dof is ``w' * le``.
        """
        if "Ke" in self._cache:
            return self._cache["Ke"], self._cache["fe"]
        le = self.le
        C = self.Kps * np.array([[1.0, self.nu], [self.nu, 1.0]])
        Ke = np.zeros((6, 6))
        fe = np.zeros(6)
        for xi, wgt in zip(_GX, _GW):
            t = 0.5 * (1.0 + xi)
            h, _, ddh = _hermite(t)
            Bm = np.zeros((2, 6))
            Bm[0, 0], Bm[0, 3] = -1.0 / le, 1.0 / le          # eps_x = u'
            Bm[1, 1], Bm[1, 2] = h[0] / self.R, le * h[1] / self.R
            Bm[1, 4], Bm[1, 5] = h[2] / self.R, le * h[3] / self.R
            Bk = np.zeros((1, 6))                             # kappa_x = -w''
            Bk[0, 1], Bk[0, 2] = -ddh[0] / le ** 2, -le * ddh[1] / le ** 2
            Bk[0, 4], Bk[0, 5] = -ddh[2] / le ** 2, -le * ddh[3] / le ** 2
            Nw = np.array([0.0, h[0], le * h[1], 0.0, h[2], le * h[3]])
            fac = 0.5 * le * wgt                              # int_0^1 ... le dt
            Ke += fac * (Bm.T @ C @ Bm + self.D * (Bk.T @ Bk))
            fe += fac * Nw
        self._cache["Ke"], self._cache["fe"] = Ke, fe
        return Ke, fe

    def assemble(self):
        Ke, fe = self.element_matrices()
        rows, cols, vals = [], [], []
        f = np.zeros(self.ndof)
        for e in range(self.n_elem):
            d = self._dofs(e)
            rows.append(np.repeat(d, 6))
            cols.append(np.tile(d, 6))
            vals.append(Ke.ravel())
            f[d] += fe
        K = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.ndof, self.ndof)).tocsr()
        return K, f

    def boundary_conditions(self, p: float) -> Dict[int, float]:
        w_p = p * self.R ** 2 / (self.E * self.t)
        last = 3 * self.n_elem
        return {0: 0.0, 1: 0.0, 2: 0.0, last + 1: w_p, last + 2: 0.0}

    def solve(self, p: float = 1.0e6) -> np.ndarray:
        K, f_unit = self.assemble()
        cons = self.boundary_conditions(p)
        fixed = np.array(sorted(cons), dtype=int)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        u = np.zeros(self.ndof)
        u[fixed] = np.array([cons[k] for k in fixed])
        Kff = K[free][:, free].tocsc()
        u[free] = spla.spsolve(Kff, f_unit[free] * p - K[free][:, fixed] @ u[fixed])
        self._cache["u"] = u
        self._cache["p"] = p
        self._cache["K"] = K
        self._cache["f_unit"] = f_unit
        return u

    @property
    def u(self) -> np.ndarray:
        if "u" not in self._cache:
            raise RuntimeError("call solve(p) before reading the solution")
        return self._cache["u"]

    def x_nodes(self) -> np.ndarray:
        return np.linspace(0.0, self.L, self.n_elem + 1)

    def reaction_check(self) -> Dict[str, float]:
        """Global radial force residual: ``sum(K u) - f`` on the free dofs."""
        K = self._cache["K"]
        f = self._cache["f_unit"] * self._cache["p"]
        r = K @ self.u - f
        cons = np.array(sorted(self.boundary_conditions(self._cache["p"])))
        free = np.setdiff1d(np.arange(self.ndof), cons)
        return {"max_free_residual": float(np.max(np.abs(r[free]))),
                "radial_reaction_sum": float(np.sum(r[cons[cons % 3 == 1]]))}

    def fields_at(self, x) -> Dict[str, np.ndarray]:
        """Evaluate ``w, u, M_x, N_theta, sigma_theta, sigma_x_bend`` at ``x``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        R, t, nu, le = self.R, self.t, self.nu, self.le
        keys = ("w", "u", "Mx", "N_theta", "sigma_theta", "sigma_x_bend",
                "eps_x", "sigma_theta_eq")
        out = {k: np.zeros_like(x) for k in keys}
        for e in range(self.n_elem):
            x0 = e * le
            mask = (x >= x0 - 1e-12) & (x <= x0 + le + 1e-12)
            if not np.any(mask):
                continue
            idx = np.flatnonzero(mask)
            ue = self.u[self._dofs(e)]
            for k in idx:
                t_ = float(np.clip((x[k] - x0) / le, 0.0, 1.0))
                h, _, ddh = _hermite(t_)
                w = (h[0] * ue[1] + le * h[1] * ue[2]
                     + h[2] * ue[4] + le * h[3] * ue[5])
                d2w = (ddh[0] * ue[1] + le * ddh[1] * ue[2]
                       + ddh[2] * ue[4] + le * ddh[3] * ue[5]) / le ** 2
                du = (ue[3] - ue[0]) / le
                out["w"][k] = w
                out["u"][k] = (1.0 - t_) * ue[0] + t_ * ue[3]
                out["eps_x"][k] = du
                out["Mx"][k] = -self.D * d2w
                out["N_theta"][k] = self.Kps * (w / R + nu * du)
                out["sigma_theta"][k] = out["N_theta"][k] / t
                # equilibrium-consistent recovery: the FE carries no axial load
                # (N_x = 0 at the free end, uniform by axial equilibrium), hence
                # N_theta = E t eps_theta (1 - nu^2) / (1 - nu^2) = E t w / R
                out["sigma_theta_eq"][k] = self.E * w / R
                out["sigma_x_bend"][k] = 6.0 * out["Mx"][k] / t ** 2
        return out

    def energy(self) -> Dict[str, float]:
        """Energy balance of the mixed-boundary-value problem.

        Linear elasticity gives the exact identity ``U = (f^T u + u_c^T r_c)/2``
        with ``r_c = (K u - f)`` on the constrained dofs; the residual of that
        identity is a global equilibrium check that also covers the
        *prescribed* far-end displacement (which does work).
        """
        K = self._cache["K"]
        f = self._cache["f_unit"] * self._cache["p"]
        u = self.u
        cons = np.array(sorted(self.boundary_conditions(self._cache["p"])))
        r = K @ u - f
        strain = 0.5 * float(u @ (K @ u))
        w_ext = float(u @ f)
        w_reac = float(u[cons] @ r[cons])
        expect = 0.5 * (w_ext + w_reac)
        return {"strain_energy": strain, "external_work_pressure": w_ext,
                "reaction_work": w_reac,
                "energy_balance_rel_error": abs(strain - expect) / strain}


def fit_edge_decay(x, dw, beta_guess: float) -> Dict[str, float]:
    """Fit ``dw = R0 exp(-beta x) cos(beta x - phi)`` to an edge disturbance.

    Returns the fitted ``beta`` and its relative deviation from
    ``beta_guess`` (the Donnell prediction).  Used to *measure* the boundary
    layer attenuation length ``1/beta`` instead of assuming ``sqrt(R t)``.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    dw = np.asarray(dw, dtype=float)

    def model(xx, R0, beta, phi):
        return R0 * np.exp(-beta * xx) * np.cos(beta * xx - phi)

    amp = np.max(np.abs(dw))
    p0 = [amp, beta_guess, 0.0]
    popt, _ = curve_fit(model, x, dw, p0=p0, maxfev=20000)
    R0, beta_fit, phi = (float(v) for v in popt)
    resid = model(x, *popt) - dw
    return {
        "R0": R0, "beta_fit": beta_fit, "phi": phi,
        "beta_theory": float(beta_guess),
        "beta_rel_error": abs(beta_fit - beta_guess) / beta_guess,
        "decay_length": 1.0 / beta_fit,
        "rel_residual_rms": float(np.sqrt(np.mean(resid ** 2)) / amp),
    }


def edge_decay_fit(fe: CylinderPressureFE, p: float,
                   x_lo: float = 0.0, x_hi: float | None = None,
                   n: int = 400) -> Dict[str, float]:
    """Measure the boundary-layer exponent from the FE deflection field.

    Fits ``w(x) - w_membrane = R0 exp(-beta x) cos(beta x - phi)`` to the FE
    solution and compares ``beta`` with the Donnell prediction
    ``(E t / (4 D R^2))^{1/4}`` (= ``(3(1-nu^2))^{1/4}/sqrt(R t)``), i.e. it
    *measures* the ``sqrt(R t)`` attenuation length instead of assuming it.
    """
    beta = beta_edge(fe.R, fe.t, fe.E, fe.nu)
    if x_hi is None:
        x_hi = min(fe.L, 3.0 / beta)
    x = np.linspace(x_lo, x_hi, n)
    f = fe.fields_at(x)
    w_p = p * fe.R ** 2 / (fe.E * fe.t)
    out = fit_edge_decay(x, f["w"] - w_p, beta)
    out["sqrt_Rt"] = float(np.sqrt(fe.R * fe.t))
    out["decay_length_over_sqrt_Rt"] = float(out["decay_length"]
                                             / np.sqrt(fe.R * fe.t))
    out["beta_sqrt_Rt"] = float(beta * np.sqrt(fe.R * fe.t))
    return out


def shell_convergence_study(R: float, L: float, t: float, E: float, nu: float,
                            p: float, n_elems: Sequence[int] = (8, 16, 32, 64),
                            membrane_factor: float = 6.0) -> List[Dict[str, float]]:
    """Convergence of the shell FE against the closed form.

    Reported per mesh:

    * ``err_Mx_edge``  -- clamp-edge bending moment vs the closed form;
    * ``err_sigma_theta_membrane`` -- hoop stress in the membrane region
      ``x >= membrane_factor/beta`` against the membrane value ``pR/t``
      (physics asymptote: the analytic field itself is still ~0.2 % away at
      ``6/beta``, so this is a membrane-state check, not a discretisation
      measure);
    * ``err_sigma_theta_pointwise`` -- max pointwise deviation from the closed
      form over the whole length (true discretisation error, constitutive
      recovery with the element-constant ``eps_x``);
    * ``err_sigma_theta_eq_pointwise`` -- same, using the equilibrium-consistent
      recovery ``sigma_theta = E w / R``;
    * ``err_w_field`` -- relative field error of the radial deflection.
    """
    beta = beta_edge(R, t, E, nu)
    x_probe = np.linspace(1e-9, L, 801)
    ref = cylinder_edge_solution(R, L, t, E, nu, p, x_probe)
    mem = x_probe >= membrane_factor / beta
    stress_mem = p * R / t
    sigma_scale = float(np.max(np.abs(ref["sigma_theta"])))

    rows: List[Dict[str, float]] = []
    for n in n_elems:
        fe = CylinderPressureFE(R=R, L=L, t=t, E=E, nu=nu, n_elem=n)
        fe.solve(p)
        f = fe.fields_at(x_probe)
        w_ref = ref["w"]
        Mx_edge_fe = float(np.max(np.abs(f["Mx"])))
        Mx_edge_ref = float(np.max(np.abs(ref["Mx"])))
        rows.append({
            "n_elem": float(n),
            "h": float(L / n),
            "beta_h": float(beta * L / n),
            "ndof": float(fe.ndof),
            "Mx_edge": Mx_edge_fe,
            "Mx_edge_ref": Mx_edge_ref,
            "err_Mx_edge": abs(Mx_edge_fe - Mx_edge_ref) / Mx_edge_ref,
            "sigma_theta_membrane_mean": float(np.mean(
                f["sigma_theta"][mem])),
            "sigma_theta_ref": stress_mem,
            "err_sigma_theta_membrane": float(
                np.max(np.abs(f["sigma_theta"][mem] - stress_mem))
                / stress_mem),
            "err_sigma_theta_pointwise": float(
                np.max(np.abs(f["sigma_theta"] - ref["sigma_theta"]))
                / sigma_scale),
            "err_sigma_theta_eq_pointwise": float(
                np.max(np.abs(f["sigma_theta_eq"] - ref["sigma_theta"]))
                / sigma_scale),
            "err_w_field": float(np.max(np.abs(f["w"] - w_ref))
                                 / float(np.max(np.abs(w_ref)))),
            "sigma_theta_edge": float(f["sigma_theta"][0]),
            "sigma_theta_eq_edge": float(f["sigma_theta_eq"][0]),
            "sigma_x_bend_edge": float(f["sigma_x_bend"][0]),
            "sigma_x_bend_edge_ref": float(ref["sigma_x_bend"][0]),
        })
    return rows
