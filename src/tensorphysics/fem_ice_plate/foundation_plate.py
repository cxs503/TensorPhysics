"""Axisymmetric Kirchhoff plate on a Winkler / Pasternak foundation with a cohesive ring crack.

Physics
-------
An infinite ice sheet idealised as a Kirchhoff (thin) plate of flexural
rigidity ``D = E h^3 / 12(1-nu^2)`` resting on a **Pasternak two-parameter**
foundation is loaded by a uniform pressure over a circular patch.  The
foundation reaction is

``p(r) = k w(r) - G_p lap(w)(r)``,

where ``k`` [Pa/m] is the subgrade modulus (for a floating ice sheet
``k = rho_w g``) and ``G_p`` [N/m] is the shear-layer modulus.
``G_p = 0`` recovers the Winkler (independent-spring) model exactly.

A **circumferential (ring) crack** at radius ``a_c`` is modelled as a cohesive
hinge: the transverse displacement ``w`` is continuous while the radial slope
``theta = dw/dr`` jumps by ``phi``, and the transmitted radial moment follows
a moment-rotation cohesive law
(:class:`tensorphysics.fem_ice_plate.cohesive_zone.CohesiveHinge`).

Variational form (axisymmetric Kirchhoff plate on Pasternak foundation)::

    U = (D/2) int [ (w'')^2 + (w'/r)^2 + 2 nu w'' w'/r ] r dr
        + (k/2) int w^2 r dr
        + (G_p/2) int (w')^2 r dr
        + 2 pi a_c int M(phi) dphi  -  int q w r dr

The Pasternak shear term ``G_p/2 int (w')^2 r dr`` adds the gradient stiffness
``K_G = G_p int (dN/dr)^T (dN/dr) r dr`` to each element matrix.

Discretisation
--------------
Hermite cubic elements (2 DOF per node: ``w`` and ``theta = dw/dr``).  The
crack node carries **two** rotational DOFs (left/right), which is exactly the
fictitious-crack kinematics.  With no crack -- or with ``phi`` tied to zero --
the solver reduces to the intact plate, which is verified against the
closed-form Hetenyi/Kelvin (Winkler) and Pasternak Green functions
(< 3 %).

The solver is displacement controlled (the centre deflection is prescribed,
the load factor is an extra unknown), so that the softening branch after the
first-crack load can be traced without arc-length machinery.

Characteristic length
---------------------
The Winkler characteristic length is ``l_c = (D/k)^{1/4}``.  For Pasternak
foundation the governing equation ``D lap^4 w - G_p lap^2 w + k w = q`` has
two length scales: the bending length ``l_c`` and the shear spreading length
``l_s = sqrt(G_p/k)``.  The dimensionless coupling ``gamma = G_p / sqrt(D k)``
determines the regime: ``gamma < 2`` gives damped-oscillatory (complex) roots,
``gamma > 2`` gives two real decay lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cohesive_zone import CohesiveHinge

__all__ = [
    "AxisymPlateResult",
    "solve_axisym_plate_foundation",
    "plate_flexural_rigidity",
    "pasternak_char_length",
    "pasternak_coupling",
    "hetenyi_axisym_point_load",
]

_GAUSS = np.polynomial.legendre.leggauss(8)


def plate_flexural_rigidity(E: float, h: float, nu: float) -> float:
    """``D = E h^3 / [12 (1 - nu^2)]`` [N m]."""
    return E * h**3 / (12.0 * (1.0 - nu**2))


def pasternak_char_length(D: float, k: float) -> float:
    """Winkler characteristic length ``l_c = (D/k)^{1/4}`` [m]."""
    return (float(D) / float(k)) ** 0.25


def pasternak_shear_length(G_p: float, k: float) -> float:
    """Pasternak shear spreading length ``l_s = sqrt(G_p/k)`` [m]."""
    if G_p <= 0.0:
        return 0.0
    return (float(G_p) / float(k)) ** 0.5


def pasternak_coupling(D: float, k: float, G_p: float) -> float:
    """Dimensionless Pasternak coupling ``gamma = G_p / sqrt(D k)`` [-].

    ``gamma = 0`` -> Winkler; ``gamma < 2`` -> complex (oscillatory) roots;
    ``gamma > 2`` -> two real decay lengths.
    """
    return float(G_p) / np.sqrt(float(D) * float(k))


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


def hetenyi_axisym_point_load(r, P: float, D: float, k: float,
                              G_p: float = 0.0, nu: float = 0.3):
    """Hetényi / Pasternak axisymmetric point-load Green function.

    For ``G_p = 0`` (Winkler): ``w(r) = -P kei(r/l) / (2 pi k l^2)`` with
    ``l = (D/k)^{1/4}`` and ``w(0) = P / (8 sqrt(k D))``.

    For ``G_p > 0`` (Pasternak): uses the closed-form from
    :mod:`tensorphysics.elastic_foundation.pasternak_plate`.

    Parameters
    ----------
    r : array_like
        Radius [m].
    P : float
        Point load [N], positive downwards.
    D, k, G_p : float
        Flexural rigidity [N m], subgrade modulus [Pa/m], shear-layer modulus
        [N/m].
    nu : float
        Poisson ratio (for moment computation).

    Returns
    -------
    dict
        ``{"w": w, "M_r": M_r, "M_t": M_t}`` with deflection [m] and moments
        [N m/m].
    """
    from tensorphysics.elastic_foundation.pasternak_plate import (
        pasternak_plate_point_load,
        pasternak_plate_moment_r,
    )

    rr = np.atleast_1d(np.asarray(r, dtype=float))
    w = pasternak_plate_point_load(rr, P, D, k, G_p, deriv=0)
    M_r = pasternak_plate_moment_r(rr, P, D, k, nu, G_p)
    # M_t = -D (w'/r + nu w'')
    d1 = pasternak_plate_point_load(rr, P, D, k, G_p, deriv=1)
    d2 = pasternak_plate_point_load(rr, P, D, k, G_p, deriv=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        M_t = -D * (d1 / rr + nu * d2)
    M_t = np.where(rr == 0.0, -D * (1.0 + nu) * d2, M_t)
    return {"w": w, "M_r": M_r, "M_t": M_t}


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
    U_shear: float  # Pasternak shear-layer strain energy [J]
    converged: bool
    crack_radius: Optional[float]
    hinge: Optional[CohesiveHinge]
    D: float
    nu: float
    k: float
    G_p: float  # Pasternak shear-layer modulus [N/m]

    @property
    def peak_load(self) -> float:
        return float(np.max(self.total_load))

    def energy_balance(self) -> Dict[str, float]:
        stored = self.U_bending + self.U_foundation + self.U_shear
        total = stored + self.fracture_energy
        return {
            "W_ext": self.W_ext,
            "U_bending": self.U_bending,
            "U_foundation": self.U_foundation,
            "U_shear": self.U_shear,
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
    G_p: float = 0.0,
) -> AxisymPlateResult:
    """Solve the (possibly cracked) axisymmetric plate on a Winkler / Pasternak foundation.

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
    G_p : float
        Pasternak shear-layer modulus [N/m].  ``G_p = 0`` recovers Winkler
        exactly.  Must be non-negative.
    """
    if D <= 0.0 or k <= 0.0 or patch_radius <= 0.0:
        raise ValueError("D, k, patch_radius must be positive")
    if G_p < 0.0:
        raise ValueError("G_p must be non-negative")
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
            # Bending stiffness
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
            # Winkler foundation: k w^2
            Ke += wq * jac * k * np.outer(N, N) * r
            # Pasternak shear layer: G_p (w')^2
            if G_p > 0.0:
                Ke += wq * jac * G_p * np.outer(dN, dN) * r
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
    # values of the crack node are kept separate)
    theta_l = np.zeros(n_nodes)
    theta_r = np.zeros(n_nodes)
    for e in range(n_el):
        r1, r2 = r_nodes[e], r_nodes[e + 1]
        Le = le[e]
        dofs = np.array([w_dof[e], thr[e], w_dof[e + 1], thl[e + 1]])
        d = u[dofs]
        # at left node (xi=0): dN/dx = [-6/Le, 1, 6/Le, -2/Le] * d
        theta_l[e] = (-6.0 * d[0] / Le + d[1] + 6.0 * d[2] / Le - 2.0 * d[3] / Le)
        # at right node (xi=1): dN/dx = [0, -1, 0, 1] * d  (but we use the
        # element-left slope of the next element for the right node)
        if e == 0:
            theta_r[e] = theta_l[e]  # centre: theta = 0 by symmetry
    # use the left-slope of each element as the nodal slope (consistent with
    # the DOF ordering)
    theta_l[0] = 0.0  # symmetry
    for e in range(n_el):
        r1, r2 = r_nodes[e], r_nodes[e + 1]
        Le = le[e]
        dofs = np.array([w_dof[e], thr[e], w_dof[e + 1], thl[e + 1]])
        d = u[dofs]
        theta_l[e + 1] = (-6.0 * d[0] / Le + d[1] + 6.0 * d[2] / Le - 2.0 * d[3] / Le)

    # moments at nodes: M_r = -D (w'' + nu w'/r), M_t = -D (w'/r + nu w'')
    for e in range(n_el):
        r1, r2 = r_nodes[e], r_nodes[e + 1]
        Le = le[e]
        dofs = np.array([w_dof[e], thr[e], w_dof[e + 1], thl[e + 1]])
        d = u[dofs]
        for xi_node in (0.0, 1.0):
            n_idx = e if xi_node == 0.0 else e + 1
            if n_idx == 0:
                continue  # centre handled separately
            r = r_nodes[n_idx]
            _, dN, ddN = _hermite(np.array([xi_node]), Le)
            dN, ddN = dN[:, 0], ddN[:, 0]
            w_prime = dN @ d
            w_dd = ddN @ d
            M_r[n_idx] = -D * (w_dd + nu * w_prime / r)
            M_t[n_idx] = -D * (w_prime / r + nu * w_dd)

    # element-centre moments (for plotting)
    r_elem = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    M_r_elem = np.zeros(n_el)
    M_t_elem = np.zeros(n_el)
    for e in range(n_el):
        r1, r2 = r_nodes[e], r_nodes[e + 1]
        Le = le[e]
        dofs = np.array([w_dof[e], thr[e], w_dof[e + 1], thl[e + 1]])
        d = u[dofs]
        _, dN, ddN = _hermite(np.array([0.5]), Le)
        dN, ddN = dN[:, 0], ddN[:, 0]
        w_prime = dN @ d
        w_dd = ddN @ d
        r = r_elem[e]
        M_r_elem[e] = -D * (w_dd + nu * w_prime / r)
        M_t_elem[e] = -D * (w_prime / r + nu * w_dd)

    # energy: U_bending = 1/2 u^T K_b u, U_foundation = 1/2 u^T K_k u,
    # U_shear = 1/2 u^T K_G u
    # We recompute by re-assembling the separate contributions
    U_bending = 0.0
    U_foundation = 0.0
    U_shear = 0.0
    for e in range(n_el):
        r1, r2 = r_nodes[e], r_nodes[e + 1]
        Le = le[e]
        dofs = np.array([w_dof[e], thr[e], w_dof[e + 1], thl[e + 1]])
        d = u[dofs]
        for xi, wq in zip(*_GAUSS):
            r = r1 + 0.5 * Le * (xi + 1.0)
            jac = 0.5 * Le
            N, dN, ddN = _hermite(np.array([(xi + 1.0) / 2.0]), Le)
            N, dN, ddN = N[:, 0], dN[:, 0], ddN[:, 0]
            w_val = N @ d
            w_prime = dN @ d
            w_dd = ddN @ d
            U_bending += wq * jac * D * (w_dd**2 + (w_prime / r)**2 + 2.0 * nu * w_dd * w_prime / r) * r
            U_foundation += wq * jac * k * w_val**2 * r
            if G_p > 0.0:
                U_shear += wq * jac * G_p * w_prime**2 * r

    # load factor and total load history
    load_factor = np.array([h[0] for h in hist])
    total_load_arr = load_factor * total_load
    w_center_arr = np.array([h[1][0] for h in hist])
    phi_hist = np.array([h[2] for h in hist])

    return AxisymPlateResult(
        r=r_nodes,
        w=w,
        theta_left=theta_l,
        theta_right=theta_r,
        M_r=M_r,
        M_theta=M_t,
        r_elem=r_elem,
        M_r_elem=M_r_elem,
        M_theta_elem=M_t_elem,
        load_factor=load_factor,
        total_load=total_load_arr,
        w_center=w_center_arr,
        phi_history=phi_hist,
        fracture_energy=diss,
        W_ext=W_ext,
        U_bending=U_bending,
        U_foundation=U_foundation,
        U_shear=U_shear,
        converged=converged,
        crack_radius=crack_radius,
        hinge=hinge,
        D=D,
        nu=nu,
        k=k,
        G_p=G_p,
    )
