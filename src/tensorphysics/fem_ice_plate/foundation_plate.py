"""Axisymmetric Kirchhoff plate on a Winkler foundation with a cohesive ring crack.

Physics
-------
An infinite ice sheet idealised as a Kirchhoff (thin) plate of flexural
rigidity ``D = E h^3 / 12(1-nu^2)`` resting on a Winkler foundation of modulus
``k`` (for a floating ice sheet ``k = rho_w g``) is loaded by a uniform
pressure over a circular patch.  A **circumferential (ring) crack** at radius
``a_c`` is modelled as a cohesive hinge: the transverse displacement ``w`` is
continuous while the radial slope ``theta = dw/dr`` jumps by ``phi``, and the
transmitted radial moment follows a moment-rotation cohesive law
(:class:`tensorphysics.fem_ice_plate.cohesive_zone.CohesiveHinge`).

Variational form (axisymmetric Kirchhoff plate)::

    U = (D/2) int [ (w'')^2 + (w'/r)^2 + 2 nu w'' w'/r ] r dr
        + (k/2) int w^2 r dr
        + 2 pi a_c int M(phi) dphi  -  int q w r dr

Discretisation
--------------
Hermite cubic elements (2 DOF per node: ``w`` and ``theta = dw/dr``).  The
crack node carries **two** rotational DOFs (left/right), which is exactly the
fictitious-crack kinematics.  With no crack -- or with ``phi`` tied to zero --
the solver reduces to the intact plate, which is verified against the
closed-form Hetenyi/Kelvin patch-load solution of
:mod:`tensorphysics.elastic_foundation.plates` (< 3 %).

The solver is displacement controlled (the centre deflection is prescribed,
the load factor is an extra unknown), so that the softening branch after the
first-crack load can be traced without arc-length machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cohesive_zone import CohesiveHinge

__all__ = ["AxisymPlateResult", "solve_axisym_plate_foundation", "plate_flexural_rigidity"]

_GAUSS = np.polynomial.legendre.leggauss(8)


def plate_flexural_rigidity(E: float, h: float, nu: float) -> float:
    """``D = E h^3 / [12 (1 - nu^2)]`` [N m]."""
    return E * h**3 / (12.0 * (1.0 - nu**2))


def _hermite(xi: np.ndarray, le: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermite shape functions and their 1st/2nd r-derivatives."""
    N = np.array(
        [
            1.0 - 3.0 * xi**2 + 2.0 * xi**3,
            le * (xi - 2.0 * xi**2 + xi**3),
            3.0 * xi**2 - 2.0 * xi**3,
            le * (-xi**2 + xi**3),
        ]
    )
    dN = np.array(
        [
            (-6.0 * xi + 6.0 * xi**2) / le,
            1.0 - 4.0 * xi + 3.0 * xi**2,
            (6.0 * xi - 6.0 * xi**2) / le,
            -2.0 * xi + 3.0 * xi**2,
        ]
    )
    ddN = np.array(
        [
            (-6.0 + 12.0 * xi) / le**2,
            (-4.0 + 6.0 * xi) / le,
            (6.0 - 12.0 * xi) / le**2,
            (-2.0 + 6.0 * xi) / le,
        ]
    )
    return N, dN, ddN


@dataclass
class AxisymPlateResult:
    """Result container of :func:`solve_axisym_plate_foundation`."""

    r: np.ndarray
    w: np.ndarray
    theta_left: np.ndarray
    theta_right: np.ndarray
    M_r: np.ndarray
    M_theta: np.ndarray
    r_elem: np.ndarray
    M_r_elem: np.ndarray
    M_theta_elem: np.ndarray
    load_factor: np.ndarray  # per load step (patch pressure scale)
    total_load: np.ndarray  # per load step [N]
    w_center: np.ndarray
    phi_history: np.ndarray
    fracture_energy: float  # 2 pi a_c int M dphi [J]
    W_ext: float
    U_bending: float
    U_foundation: float
    converged: bool
    crack_radius: Optional[float]
    hinge: Optional[CohesiveHinge]
    D: float
    nu: float
    k: float

    @property
    def peak_load(self) -> float:
        return float(np.max(self.total_load))

    def energy_balance(self) -> Dict[str, float]:
        stored = self.U_bending + self.U_foundation
        total = stored + self.fracture_energy
        return {
            "W_ext": self.W_ext,
            "U_bending": self.U_bending,
            "U_foundation": self.U_foundation,
            "fracture_energy": self.fracture_energy,
            "stored": stored,
            "total_consumed": total,
            "closure_ratio": total / self.W_ext if self.W_ext != 0.0 else np.nan,
            "residual_rel": abs(self.W_ext - total) / self.W_ext if self.W_ext else np.nan,
        }


def solve_axisym_plate_foundation(
    D: float,
    nu: float,
    k: float,
    patch_radius: float,
    total_load: float = 1.0,
    R_out: Optional[float] = None,
    n_el: int = 400,
    crack_radius: Optional[float] = None,
    hinge: Optional[CohesiveHinge] = None,
    w_center_steps: Optional[np.ndarray] = None,
    rigid_hinge: bool = False,
    tol: float = 1e-10,
    max_iter: int = 60,
    grading: float = 2.0,
) -> AxisymPlateResult:
    """Solve the (possibly cracked) axisymmetric plate on a Winkler foundation.

    Parameters
    ----------
    D, nu : float
        Flexural rigidity and Poisson ratio.
    k : float
        Foundation modulus [N/m^3].
    patch_radius : float
        Radius of the uniformly loaded patch [m].
    total_load : float
        Total load at the end of the loading path [N].
    R_out : float, optional
        Outer radius of the truncated domain; default ``20 (D/k)^(1/4)``.
    n_el : int
        Number of radial elements (geometrically graded towards the centre).
    crack_radius : float, optional
        Radius of the cohesive ring crack.  ``None`` -> intact plate.
    hinge : CohesiveHinge, optional
        Cohesive moment-rotation law (required when ``crack_radius`` is given,
        unless ``rigid_hinge=True``).
    rigid_hinge : bool
        If True the crack node rotations are tied (``phi = 0``): the rigid
        limit used to validate against the intact Hetenyi solution.
    w_center_steps : ndarray, optional
        Prescribed centre deflections (displacement control).  Defaults to a
        ramp over 40 steps up to a value inferred from ``total_load``.
    """
    if D <= 0.0 or k <= 0.0 or patch_radius <= 0.0:
        raise ValueError("D, k, patch_radius must be positive")
    l0 = (D / k) ** 0.25
    if R_out is None:
        R_out = 20.0 * l0

    # --- mesh: geometric grading towards the centre -----------------------
    t = np.linspace(0.0, 1.0, n_el + 1)
    r_nodes = R_out * t**grading
    le = np.diff(r_nodes)

    n_nodes = n_el + 1
    w_dof = np.arange(n_nodes)
    nxt = n_nodes
    thl = np.zeros(n_nodes, dtype=int)
    thr = np.zeros(n_nodes, dtype=int)
    crack_node = -1
    if crack_radius is not None and not rigid_hinge:
        crack_node = int(np.argmin(np.abs(r_nodes - crack_radius)))
    for n in range(n_nodes):
        if n == crack_node:
            thl[n] = nxt
            thr[n] = nxt + 1
            nxt += 2
        else:
            thl[n] = nxt
            thr[n] = nxt
            nxt += 1
    if rigid_hinge and crack_radius is not None:
        crack_node = int(np.argmin(np.abs(r_nodes - crack_radius)))
    n_dof = nxt

    # --- element matrices -------------------------------------------------
    K = np.zeros((n_dof, n_dof))
    f_unit = np.zeros(n_dof)
    patch_area = np.pi * patch_radius**2
    q_unit = 1.0 / patch_area  # unit total load spread over the patch

    for e in range(n_el):
        r1 = r_nodes[e]
        r2 = r_nodes[e + 1]
        Le = le[e]
        n1, n2 = e, e + 1
        dofs = np.array([w_dof[n1], thr[n1], w_dof[n2], thl[n2]])
        Ke = np.zeros((4, 4))
        fe = np.zeros(4)

        for xi, wq in zip(*_GAUSS):
            r = r1 + 0.5 * Le * (xi + 1.0)
            jac = 0.5 * Le
            N, dN, ddN = _hermite(np.array([(xi + 1.0) / 2.0]), Le)
            N, dN, ddN = N[:, 0], dN[:, 0], ddN[:, 0]
            Ke += (
                wq
                * jac
                * D
                * (
                    np.outer(ddN, ddN)
                    + np.outer(dN, dN) / r**2
                    + nu / r * (np.outer(ddN, dN) + np.outer(dN, ddN))
                )
                * r
            )
            Ke += wq * jac * k * np.outer(N, N) * r
        K[np.ix_(dofs, dofs)] += Ke
        # patch load: integrate exactly up to r = patch_radius (the boundary is
        # generally inside an element, so the element is split there)
        if r1 < patch_radius:
            r_hi = min(r2, patch_radius)
            for xi, wq in zip(*_GAUSS):
                r = 0.5 * (r1 + r_hi) + 0.5 * (r_hi - r1) * xi
                jac = 0.5 * (r_hi - r1)
                xin = (r - r1) / Le
                N = _hermite(np.array([xin]), Le)[0][:, 0]
                fe += wq * jac * q_unit * N * r
        f_unit[dofs] += fe

    # --- boundary conditions ---------------------------------------------
    # symmetry at the centre: theta(0) = 0
    fixed = [int(thl[0])]
    free = np.setdiff1d(np.arange(n_dof), np.array(fixed, dtype=int))

    hinge_dofs: Optional[Tuple[int, int]] = None
    if crack_node >= 0 and crack_radius is not None and not rigid_hinge:
        hinge_dofs = (int(thr[crack_node]), int(thl[crack_node]))
    elif rigid_hinge and crack_radius is not None:
        # tie theta_left = theta_right via a penalty-free substitution:
        # merge the two rotational DOFs of the crack node
        pass

    rc = float(r_nodes[crack_node]) if crack_node >= 0 else None

    def internal_force(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """K u plus the cohesive hinge contribution; returns (F, K_tan, phi, D_hinge)."""
        F = K @ u
        Kt = K.copy()
        phi = 0.0
        diss = 0.0
        if hinge_dofs is not None and hinge is not None:
            i_r, i_l = hinge_dofs
            phi = u[i_r] - u[i_l]
            M = hinge.moment(phi)
            kT = hinge.tangent(phi)
            # NB: the plate assembly works *per radian* (the 2 pi of the
            # axisymmetric measure is factored out of both K and f), so the
            # hinge contribution carries no 2 pi either.
            Fh = rc * M
            F[i_r] += Fh
            F[i_l] -= Fh
            for a, b, s in ((i_r, i_r, 1.0), (i_r, i_l, -1.0), (i_l, i_r, -1.0), (i_l, i_l, 1.0)):
                Kt[a, b] += rc * kT * s
            diss = rc * hinge.dissipate(phi)
        return F, Kt, phi, diss

    max_disp = 0.02 * l0  # fallback if displacement steps are not given
    if w_center_steps is None:
        w_center_steps = np.linspace(0.0, max_disp, 41)[1:]

    u = np.zeros(n_dof)
    lam = 0.0
    hist: List[Tuple[float, np.ndarray, float, float]] = []
    converged = True
    W_ext = 0.0
    prev_S: Optional[float] = None
    prev_lam: Optional[float] = None

    for step, w_target in enumerate(w_center_steps):
        lam_k = lam
        for _ in range(max_iter):
            F, Kt, phi, diss = internal_force(u)
            R = F - lam_k * f_unit
            res_vec = np.concatenate([R[free], [u[0] - w_target]])
            scale = max(np.max(np.abs(F)), 1.0)
            if np.max(np.abs(res_vec)) <= tol * scale:
                break
            # bordered Newton system (displacement control)
            n = len(free)
            A = np.zeros((n + 1, n + 1))
            A[:n, :n] = Kt[np.ix_(free, free)]
            A[:n, n] = -f_unit[free]
            A[n, :n] = np.eye(1, n, list(free).index(0))[0]
            A[n, n] = 0.0
            sol = np.linalg.solve(A, -res_vec)
            u[free] += sol[:n]
            lam_k += sol[n]
        else:
            converged = False

        u[0] = w_target
        F, Kt, phi, diss = internal_force(u)
        S = float(f_unit @ u)
        if prev_S is not None and prev_lam is not None:
            W_ext += 0.5 * (lam_k * 1.0 + prev_lam) * (S - prev_S)
        prev_S, prev_lam = S, lam_k
        hist.append((lam_k, u.copy(), phi, diss))
        lam = lam_k

    # --- post-processing --------------------------------------------------
    F, Kt, phi, diss = internal_force(u)
    q = lam * q_unit
    M_r = np.zeros(n_nodes)
    M_t = np.zeros(n_nodes)
    w = u[w_dof]
    # slope at nodes: element-wise Hermite derivative averaged (left/right
    # values of the crack node are reported separately)
    th_l = np.zeros(n_nodes)
    th_r = np.zeros(n_nodes)
    for n in range(n_nodes):
        th_l[n] = u[thl[n]]
        th_r[n] = u[thr[n]]
    # curvature: w'' and w' at nodes from the element containing the node
    d2w = np.zeros(n_nodes)
    for e in range(n_el):
        for node, side in ((e, "right"), (e + 1, "left")):
            xi = 0.0 if side == "right" else 1.0
            N, dN, ddN = _hermite(np.array([xi]), le[e])
            n1, n2 = e, e + 1
            ue = np.array([u[w_dof[n1]], u[thr[n1]], u[w_dof[n2]], u[thl[n2]]])
            d2w[node] = float(ddN[:, 0] @ ue)
            if side == "right":
                th_r[node] = float(dN[:, 0] @ ue)
            else:
                th_l[node] = float(dN[:, 0] @ ue)
    r_safe = np.where(r_nodes > 0.0, r_nodes, 1e-30 * l0)
    wp = th_r  # element-right slope is the natural one away from the crack
    M_r = -D * (d2w + nu * wp / r_safe)
    M_t = -D * (nu * d2w + wp / r_safe)

    # element-centre (superconvergent) bending moments
    r_elem = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    M_r_elem = np.zeros(n_el)
    M_t_elem = np.zeros(n_el)
    for e in range(n_el):
        n1, n2 = e, e + 1
        ue = np.array([u[w_dof[n1]], u[thr[n1]], u[w_dof[n2]], u[thl[n2]]])
        N, dN, ddN = _hermite(np.array([0.5]), le[e])
        d2w_e = float(ddN[:, 0] @ ue)
        dw_e = float(dN[:, 0] @ ue)
        M_r_elem[e] = -D * (d2w_e + nu * dw_e / r_elem[e])
        M_t_elem[e] = -D * (nu * d2w_e + dw_e / r_elem[e])

    U_bend = 0.5 * float(u @ (K @ u))
    # split the stored energy: recompute the foundation part
    U_found = 0.0
    for e in range(n_el):
        r1, Le = r_nodes[e], le[e]
        n1, n2 = e, e + 1
        dofs = np.array([w_dof[n1], thr[n1], w_dof[n2], thl[n2]])
        ue = u[dofs]
        Ke = np.zeros((4, 4))
        for xi, wq in zip(*_GAUSS):
            r = r1 + 0.5 * Le * (xi + 1.0)
            jac = 0.5 * Le
            N = _hermite(np.array([(xi + 1.0) / 2.0]), Le)[0][:, 0]
            Ke += wq * jac * k * np.outer(N, N) * r
        U_found += 0.5 * float(ue @ Exc(Ke) @ ue)
    U_bend_plate = U_bend - U_found

    load_hist = np.array([h[0] for h in hist])
    wc_hist = np.array([h[1][0] for h in hist])
    total_load_hist = load_hist * 1.0  # unit load -> load factor equals load

    return AxisymPlateResult(
        r=r_nodes,
        w=w,
        theta_left=th_l,
        theta_right=th_r,
        M_r=M_r,
        M_theta=M_t,
        r_elem=r_elem,
        M_r_elem=M_r_elem,
        M_theta_elem=M_t_elem,
        load_factor=load_hist,
        total_load=total_load_hist,
        w_center=wc_hist,
        phi_history=np.array([h[2] for h in hist]),
        fracture_energy=float(2.0 * np.pi * diss),
        W_ext=float(2.0 * np.pi * W_ext),
        U_bending=float(2.0 * np.pi * U_bend_plate),
        U_foundation=float(2.0 * np.pi * U_found),
        converged=converged,
        crack_radius=rc,
        hinge=hinge,
        D=D,
        nu=nu,
        k=k,
    )


def Exc(A: np.ndarray) -> np.ndarray:
    """Identity helper kept for readability of the energy split."""
    return A