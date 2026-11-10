"""Thermo-elastic finite elements: thermal eigenstrain as an initial strain.

Constitutive law used throughout::

    sigma = C : (eps - eps_th),     eps_th = alpha * dT * {1, 1, 0}

Three genuine FE discretisations are provided:

* ``q4_plane_stress`` / ``q8_plane_stress`` -- 2-D continuum (plane stress,
  bilinear Q4 and 8-node serendipity Q8) driven by a nodal temperature field;
  the Q8 element is used for the 2-D bimetal cross-check because bilinear Q4
  locks in in-plane bending.
* ``LayeredBeamSection`` + ``beam_solve`` -- composite Euler-Bernoulli frame
  element (2-node Hermite bending + linear axial) with layered section
  integration of the thermal thrust/moment.
* ``axisym_ring_thermoelastic`` -- axisymmetric plane-stress ring elements for
  the steady radial temperature gradient around a circular hole.

Sign conventions (used consistently by every solver and the analytics):
``y`` points up, layers are stacked bottom -> top, and the *strain-gradient*
curvature is positive when the total axial strain increases with ``y``::

    eps_xx(y) = eps0 + kappa (y - y_bar),    kappa = M_T / EI

with the thermal moment ``M_T = int E alpha dT (y - y_bar) dA`` about the
elastic centroid.  A bimetal strip whose high-expansion layer is on top
therefore has ``kappa > 0`` and deflects towards the low-expansion side.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plane_stress_C",
    "q4_plane_stress",
    "q8_plane_stress",
    "LayeredBeamSection",
    "beam_solve",
    "hermite_beam_stiffness",
    "axisym_ring_thermoelastic",
    "apply_dirichlet",
]


# --------------------------------------------------------------------------
# continuum: plane stress with thermal eigenstrain
# --------------------------------------------------------------------------
def plane_stress_C(E: float, nu: float) -> np.ndarray:
    """Plane-stress elasticity matrix for ``[eps_xx, eps_yy, gamma_xy]``."""
    f = E / (1.0 - nu * nu)
    return f * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])


_G2 = 1.0 / np.sqrt(3.0)
_Q4_GP = [(-_G2, -_G2), (_G2, -_G2), (_G2, _G2), (-_G2, _G2)]
_G3 = np.sqrt(3.0 / 5.0)
_Q8_GP = [(a, b) for b in (-_G3, 0.0, _G3) for a in (-_G3, 0.0, _G3)]


def _q4_shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


def _q8_shape(xi, eta):
    """8-node serendipity shape functions (corners 1-4, mid-sides 5-8)."""
    N = np.array([
        -(1 - xi) * (1 - eta) * (1 + xi + eta) / 4.0,
        -(1 + xi) * (1 - eta) * (1 - xi + eta) / 4.0,
        -(1 + xi) * (1 + eta) * (1 - xi - eta) / 4.0,
        -(1 - xi) * (1 + eta) * (1 + xi - eta) / 4.0,
        (1 - xi ** 2) * (1 - eta) / 2.0,
        (1 + xi) * (1 - eta ** 2) / 2.0,
        (1 - xi ** 2) * (1 + eta) / 2.0,
        (1 - xi) * (1 - eta ** 2) / 2.0,
    ])
    dN_dxi = np.array([
        (1 - eta) * (2 * xi + eta) / 4.0,
        (1 - eta) * (2 * xi - eta) / 4.0,
        (1 + eta) * (2 * xi + eta) / 4.0,
        (1 + eta) * (2 * xi - eta) / 4.0,
        -xi * (1 - eta),
        (1 - eta ** 2) / 2.0,
        -xi * (1 + eta),
        -(1 - eta ** 2) / 2.0,
    ])
    dN_deta = np.array([
        (1 - xi) * (xi + 2 * eta) / 4.0,
        (1 + xi) * (-xi + 2 * eta) / 4.0,
        (1 + xi) * (xi + 2 * eta) / 4.0,
        (1 - xi) * (-xi + 2 * eta) / 4.0,
        -(1 - xi ** 2) / 2.0,
        -(1 + xi) * eta,
        (1 - xi ** 2) / 2.0,
        -(1 - xi) * eta,
    ])
    return N, dN_dxi, dN_deta


def _B_and_detJ(xy, xi, eta, shape):
    N, dN_dxi, dN_deta = shape(xi, eta)
    J = np.array([[dN_dxi @ xy[:, 0], dN_dxi @ xy[:, 1]],
                  [dN_deta @ xy[:, 0], dN_deta @ xy[:, 1]]])
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError("element with non-positive Jacobian")
    dN = np.linalg.solve(J, np.vstack([dN_dxi, dN_deta]))  # rows: d/dx, d/dy
    nn = xy.shape[0]
    B = np.zeros((3, 2 * nn))
    B[0, 0::2] = dN[0]
    B[1, 1::2] = dN[1]
    B[2, 0::2] = dN[1]
    B[2, 1::2] = dN[0]
    return B, detJ, N


def _continuum_plane_stress(nodes, elems, E, nu, alpha, dT_nodes, thickness,
                            fixed, f_ext, shape, gps):
    nodes = np.asarray(nodes, dtype=float)
    elems = np.asarray(elems, dtype=int)
    n = nodes.shape[0]
    m, nn = elems.shape
    E_arr = np.broadcast_to(np.asarray(E, dtype=float), (m,)).astype(float)
    alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), (m,)).astype(float)
    dT = np.asarray(dT_nodes, dtype=float)
    ndof = 2 * n
    f = np.zeros(ndof) if f_ext is None else np.array(f_ext, dtype=float)
    rows, cols, vals = [], [], []
    for e, el in enumerate(elems):
        xy = nodes[el]
        C = plane_stress_C(float(E_arr[e]), nu)
        dofs = np.empty(2 * nn, dtype=int)
        dofs[0::2] = 2 * el
        dofs[1::2] = 2 * el + 1
        for xi, eta in gps:
            B, detJ, N = _B_and_detJ(xy, xi, eta, shape)
            w = thickness * detJ
            ke = w * (B.T @ C @ B)
            rows.extend(np.repeat(dofs, dofs.size))
            cols.extend(np.tile(dofs, dofs.size))
            vals.extend(ke.ravel())
            eps_th = float(alpha_arr[e]) * float(N @ dT[el]) * np.array([1.0, 1.0, 0.0])
            f[dofs] += w * (B.T @ (C @ eps_th))
    if len(rows):
        from scipy.sparse import coo_matrix
        K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    else:                                                    # pragma: no cover
        K = np.zeros((ndof, ndof))

    fixed_dofs = {}
    for key, val in dict(fixed or {}).items():
        if isinstance(key, tuple) and np.isscalar(val):
            node, comp = int(key[0]), int(key[1])
            fixed_dofs[2 * node + comp] = float(val)
        elif isinstance(key, tuple):
            fixed_dofs[2 * int(key[0])] = float(val[0])
            fixed_dofs[2 * int(key[0]) + 1] = float(val[1])
        else:
            fixed_dofs[int(key)] = float(val)
    u = apply_dirichlet(K, f, fixed_dofs)

    sigma = np.zeros((m, 3))
    eps_tot = np.zeros((m, 3))
    for e, el in enumerate(elems):
        xy = nodes[el]
        C = plane_stress_C(float(E_arr[e]), nu)
        B, _, N = _B_and_detJ(xy, 0.0, 0.0, shape)
        dofs = np.empty(2 * nn, dtype=int)
        dofs[0::2] = 2 * el
        dofs[1::2] = 2 * el + 1
        eps = B @ u[dofs]
        eps_th = float(alpha_arr[e]) * float(N @ dT[el]) * np.array([1.0, 1.0, 0.0])
        eps_tot[e] = eps
        sigma[e] = C @ (eps - eps_th)
    return {"u": u, "sigma": sigma, "eps": eps_tot, "K": K, "f": f,
            "alpha": alpha_arr, "E": E_arr}


def q4_plane_stress(nodes, elems, E, nu, alpha, dT_nodes, thickness=1.0,
                    fixed=None, f_ext=None):
    """2-D plane-stress FE with thermal eigenstrain, bilinear Q4 elements."""
    return _continuum_plane_stress(nodes, elems, E, nu, alpha, dT_nodes, thickness,
                                   fixed, f_ext, _q4_shape, _Q4_GP)


def q8_plane_stress(nodes, elems, E, nu, alpha, dT_nodes, thickness=1.0,
                    fixed=None, f_ext=None):
    """2-D plane-stress FE with thermal eigenstrain, 8-node serendipity Q8."""
    return _continuum_plane_stress(nodes, elems, E, nu, alpha, dT_nodes, thickness,
                                   fixed, f_ext, _q8_shape, _Q8_GP)


def apply_dirichlet(K, f, fixed, sparse_threshold=200):
    """Solve ``K u = f`` with prescribed dofs ``{dof: value}``."""
    n = f.size
    f = np.asarray(f, dtype=float)
    fixed = {int(k): float(v) for k, v in fixed.items()}
    fdofs = np.array(sorted(fixed), dtype=int)
    free = np.setdiff1d(np.arange(n), fdofs)
    u = np.zeros(n)
    for d, v in fixed.items():
        u[d] = v
    if free.size == 0:
        return u
    is_sparse = hasattr(K, "tocsr")
    if is_sparse or n > sparse_threshold:
        from scipy.sparse import csc_matrix, csr_matrix
        from scipy.sparse.linalg import spsolve
        Kc = csc_matrix(K)
        rhs = f[free] - Kc[free][:, fdofs] @ u[fdofs] if fdofs.size else f[free]
        u[free] = spsolve(Kc[free][:, free], rhs)
        return u
    Kd = np.asarray(K, dtype=float)
    rhs = f[free] - (Kd[np.ix_(free, fdofs)] @ u[fdofs] if fdofs.size else 0.0)
    u[free] = np.linalg.solve(Kd[np.ix_(free, free)], rhs)
    return u


# --------------------------------------------------------------------------
# layered composite beam: section + frame element with thermal loads
# --------------------------------------------------------------------------
class LayeredBeamSection:
    """Composite Euler-Bernoulli section built from horizontal layers.

    ``layers`` is a list of dicts ``{"thickness": h_i, "E": E_i,
    "alpha": alpha_i}`` ordered bottom -> top (optional ``width``, default 1).

    Reference axis = elastic centroid of the modular-ratio transformed section,
    hence the axial/bending response decouples:
    ``N = EA*eps0 - N_T``, ``M = EI*kappa - M_T``.
    """

    def __init__(self, layers, dT: float = 0.0):
        self.layers = [dict(l) for l in layers]
        self.dT = float(dT)
        y = 0.0
        for l in self.layers:
            l.setdefault("width", 1.0)
            l["y_bottom"] = y
            y += l["thickness"]
        self.total_thickness = y
        E_ref = self.layers[0]["E"]
        A_t = sum(l["E"] / E_ref * l["thickness"] * l["width"] for l in self.layers)
        S_t = sum(l["E"] / E_ref * l["thickness"] * l["width"]
                  * (l["y_bottom"] + l["thickness"] / 2) for l in self.layers)
        self.y_bar = S_t / A_t
        self.EA = E_ref * A_t
        self.EI = 0.0
        for l in self.layers:
            n_i = l["E"] / E_ref
            h, b = l["thickness"], l["width"]
            yc = l["y_bottom"] + h / 2 - self.y_bar
            self.EI += E_ref * n_i * (b * h ** 3 / 12.0 + b * h * yc ** 2)
        self.ES = 0.0  # about the centroid, by construction

    def thermal_resultants(self, dT=None):
        """Thermal thrust ``N_T`` and thermal moment ``M_T`` about the centroid."""
        dT = self.dT if dT is None else float(dT)
        N_T = 0.0
        M_T = 0.0
        for l in self.layers:
            A = l["thickness"] * l["width"]
            yc = l["y_bottom"] + l["thickness"] / 2 - self.y_bar
            N_T += l["E"] * l["alpha"] * dT * A
            M_T += l["E"] * l["alpha"] * dT * A * yc
        return N_T, M_T

    def free_strain(self, dT=None):
        """Stress-free-compatible axial strain ``eps0`` and curvature ``kappa``."""
        N_T, M_T = self.thermal_resultants(dT)
        return N_T / self.EA, M_T / self.EI


def hermite_beam_stiffness(EI, L):
    """Euler-Bernoulli element stiffness on (w1, th1, w2, th2)."""
    return EI / L ** 3 * np.array([
        [12.0, 6.0 * L, -12.0, 6.0 * L],
        [6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
        [-12.0, -6.0 * L, 12.0, -6.0 * L],
        [6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
    ])


def beam_solve(section, length, n_elems=8, dT=None, tip_load=0.0, tip_moment=0.0):
    """Cantilever composite beam FE (3 dofs/node: u, w, theta = dw/dx).

    Uniform thermal eigenstrain enters through the equivalent nodal loads of
    ``N_T``/``M_T``; optional tip force ``tip_load`` [N] and tip moment
    ``tip_moment`` [N m] are applied at the free end.

    The reported ``kappa`` is the strain-gradient curvature
    ``kappa = -d(theta)/dx`` (see module docstring), matching
    ``M_T / EI`` for the free bimetal and negative for a sagging bending.
    """
    L_el = length / n_elems
    ndof = 3 * (n_elems + 1)
    K = np.zeros((ndof, ndof))
    f = np.zeros(ndof)
    N_T, M_T = section.thermal_resultants(dT)
    EA, EI = section.EA, section.EI
    kb = hermite_beam_stiffness(EI, L_el)
    for e in range(n_elems):
        base = 3 * e
        d = [base + i for i in range(6)]
        ka = EA / L_el
        K[d[0], d[0]] += ka
        K[d[0], d[3]] -= ka
        K[d[3], d[0]] -= ka
        K[d[3], d[3]] += ka
        ich = [1, 2, 4, 5]                      # (w1, th1, w2, th2)
        for a, ia in enumerate(ich):
            for b, ib in enumerate(ich):
                K[d[ia], d[ib]] += kb[a, b]
        # eigenstrain equivalent nodal loads (uniform N_T, M_T)
        f[d[0]] += -N_T
        f[d[3]] += +N_T
        f[d[2]] += +M_T
        f[d[5]] += -M_T
    fixed = {0: 0.0, 1: 0.0, 2: 0.0}            # clamped root
    last = 3 * n_elems
    f[last + 1] += tip_load
    f[last + 2] += tip_moment
    u = apply_dirichlet(K, f, fixed)
    theta = u[3 * np.arange(n_elems + 1) + 2]
    return {
        "u": u.reshape(-1, 3),
        "kappa": -(np.diff(theta) / L_el),
        "tip_deflection": float(u[last + 1]),
        "tip_rotation": float(u[last + 2]),
        "N_T": N_T,
        "M_T": M_T,
        "K": K,
        "f": f,
    }


# --------------------------------------------------------------------------
# axisymmetric plane-stress ring elements (radial temperature gradient)
# --------------------------------------------------------------------------
def axisym_ring_thermoelastic(r_nodes, T_nodes, E, nu, alpha, thickness=1.0):
    """Axisymmetric plane-stress ring FE for an annular plate, free-free.

    ``eps_r = du/dr``, ``eps_theta = u/r``; thermal eigenstrain
    ``eps_th = alpha T`` on both components (because ``C {1,1} = E/(1-nu) {1,1}``
    reproduces the plane-stress thermal stress ``-E alpha T/(1-nu)``).  Both
    edges are traction free, so the system is loaded only thermally.
    """
    r = np.asarray(r_nodes, dtype=float)
    T = np.asarray(T_nodes, dtype=float)
    n = r.size
    ne = n - 1
    # 2x2 plane-stress operator on (eps_r, eps_theta)
    C = E / (1.0 - nu * nu) * np.array([[1.0, nu], [nu, 1.0]])
    K = np.zeros((n, n))
    f = np.zeros(n)
    gp = 1.0 / np.sqrt(3.0)
    for e in range(ne):
        r1, r2 = r[e], r[e + 1]
        for xi in (-gp, gp):
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            r_gp = N1 * r1 + N2 * r2
            dNdr = np.array([-1.0 / (r2 - r1), 1.0 / (r2 - r1)])
            B = np.array([[dNdr[0], dNdr[1]], [N1 / r_gp, N2 / r_gp]])
            w = 2.0 * np.pi * r_gp * thickness * (r2 - r1) / 2.0
            K[np.ix_([e, e + 1], [e, e + 1])] += w * (B.T @ C @ B)
            T_gp = N1 * T[e] + N2 * T[e + 1]
            eps_th = alpha * T_gp * np.array([1.0, 1.0])
            f[[e, e + 1]] += w * (B.T @ (C @ eps_th))
    u = apply_dirichlet(K, f, {})
    sig_r = np.zeros(ne)
    sig_t = np.zeros(ne)
    for e in range(ne):
        r1, r2 = r[e], r[e + 1]
        rc = 0.5 * (r1 + r2)
        dNdr = np.array([-1.0 / (r2 - r1), 1.0 / (r2 - r1)])
        B = np.array([[dNdr[0], dNdr[1]], [0.5 / rc, 0.5 / rc]])
        ue = np.array([u[e], u[e + 1]])
        eps = B @ ue
        T_gp = 0.5 * (T[e] + T[e + 1])
        eps_th = alpha * T_gp * np.array([1.0, 1.0])
        s = C @ (eps - eps_th)
        sig_r[e], sig_t[e] = s[0], s[1]
    return {"r": r, "u": u, "sigma_r": sig_r, "sigma_theta": sig_t, "K": K, "f": f}