"""Fourier heat conduction finite elements (thermal side of thermo-mechanical coupling).

Provides the temperature field that drives the thermal eigenstrain:

* ``heat1d_steady``    -- 1-D steady conduction (plane wall / rod) with Dirichlet,
                          Neumann (prescribed flux) and Robin (convection) ends.
* ``heat1d_transient`` -- 1-D transient conduction, theta time integration
                          (theta=1 backward Euler, theta=0.5 Crank-Nicolson),
                          consistent capacitance matrix.
* ``q4_steady_conduction`` -- 2-D bilinear (Q4) steady conduction/Poisson, used
                          for the manufactured-solution convergence test.

All solvers are pure numpy; element integrals use full Gauss quadrature.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "heat1d_steady",
    "heat1d_transient",
    "graded_mesh",
    "q4_steady_conduction",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def graded_mesh(length: float, first: float, growth: float) -> np.ndarray:
    """1-D node coordinates graded towards ``x = 0`` (boundary-layer friendly)."""
    if first <= 0 or growth <= 1.0 or length <= 0:
        raise ValueError("graded_mesh needs first>0, growth>1, length>0")
    xs = [0.0]
    dx = first
    while xs[-1] < length:
        xs.append(min(xs[-1] + dx, length))
        dx *= growth
        if len(xs) > 20000:
            raise RuntimeError("graded_mesh did not terminate")
    return np.asarray(xs, dtype=float)


def _check_bc(bc):
    if bc is None:
        return {"type": "flux", "value": 0.0}
    kind = bc.get("type", "flux")
    if kind not in ("dirichlet", "flux", "convection"):
        raise ValueError(f"unknown boundary type {kind!r}")
    if kind == "convection" and ("h" not in bc or "T_inf" not in bc):
        raise ValueError("convection BC needs 'h' and 'T_inf'")
    return bc


def _bc_value(bc, t):
    val = bc.get("value", 0.0)
    return val(t) if callable(val) else val


def _apply_robin(K, f, node, bc):
    """Convection at a 1-D end: q = h (T_inf - T) entering the body."""
    h = float(bc["h"])
    T_inf = float(bc["T_inf"])
    K[node, node] += h
    f[node] += h * T_inf


# --------------------------------------------------------------------------
# 1-D steady conduction
# --------------------------------------------------------------------------
def heat1d_steady(x, k, area=1.0, bc_left=None, bc_right=None, source=None):
    """Steady 1-D conduction on the node array ``x``.

    Parameters
    ----------
    x : (n,) node coordinates [m] (must be strictly increasing)
    k : scalar or (n_elems,) conductivity [W/(m K)]
    area : cross-section (per unit depth use 1.0)
    bc_left, bc_right : dict with ``type`` in {"dirichlet","flux","convection"}
        * dirichlet  -> ``value`` (K) imposed temperature
        * flux       -> ``value`` (W/m^2) heat flux *entering* the domain
        * convection -> ``h`` (W/(m^2 K)) and ``T_inf`` (K)
    source : scalar or (n_elems,) volumetric source [W/m^3]

    Returns
    -------
    dict with ``T`` (nodal temperatures), ``flux`` (element heat flux
    ``q = -k dT/dx``, positive in +x) and the assembled system.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    ne = n - 1
    if n < 2:
        raise ValueError("need at least 2 nodes")
    if np.any(np.diff(x) <= 0):
        raise ValueError("node coordinates must be strictly increasing")
    k = np.broadcast_to(np.asarray(k, dtype=float), (ne,)) if np.ndim(k) else np.full(ne, float(k))
    src = np.zeros(ne) if source is None else np.broadcast_to(np.asarray(source, dtype=float), (ne,))

    K = np.zeros((n, n))
    f = np.zeros(n)
    for e in range(ne):
        L = x[e + 1] - x[e]
        ke = k[e] * area / L
        K[e, e] += ke
        K[e, e + 1] -= ke
        K[e + 1, e] -= ke
        K[e + 1, e + 1] += ke
        fe = src[e] * area * L / 2.0
        f[e] += fe
        f[e + 1] += fe

    bc_left = _check_bc(bc_left)
    bc_right = _check_bc(bc_right)
    fixed = {}
    for node, bc in ((0, bc_left), (n - 1, bc_right)):
        kind = bc["type"]
        if kind == "dirichlet":
            fixed[node] = _bc_value(bc, 0.0)
        elif kind == "flux":
            f[node] += float(bc.get("value", 0.0)) * area
        else:
            _apply_robin(K, f, node, bc)

    T = _solve_with_dirichlet(K, f, fixed)
    dTdx = np.diff(T) / np.diff(x)
    return {"T": T, "flux": -k * dTdx, "K": K, "f": f, "x": x}


def _solve_with_dirichlet(K, f, fixed):
    n = f.size
    free = np.setdiff1d(np.arange(n), np.asarray(list(fixed.keys()), dtype=int))
    T = np.zeros(n)
    for node, val in fixed.items():
        T[node] = val
    if free.size:
        T_free = np.linalg.solve(K[np.ix_(free, free)], f[free] - K[np.ix_(free, list(fixed.keys()))] @ np.asarray(list(fixed.values()), dtype=float))
        T[free] = T_free
    return T


# --------------------------------------------------------------------------
# 1-D transient conduction
# --------------------------------------------------------------------------
def heat1d_transient(x, k, rho, c, times, T0=0.0, bc_left=None, bc_right=None,
                     theta=1.0, source=None):
    """Transient 1-D conduction with theta time integration.

    ``times`` must be the *absolute* solution times (increasing); consecutive
    differences are used as time steps.  Dirichlet values may be callables of
    time (used for the thermal-shock temperature step).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    ne = n - 1
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be an increasing 1-D array with >=2 entries")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta in (0, 1]")
    k = np.full(ne, float(k)) if np.ndim(k) == 0 else np.asarray(k, dtype=float)
    src = np.zeros(ne) if source is None else np.broadcast_to(np.asarray(source, dtype=float), (ne,))

    K = np.zeros((n, n))
    C = np.zeros((n, n))
    for e in range(ne):
        L = x[e + 1] - x[e]
        ke = k[e] / L
        K[e, e] += ke
        K[e, e + 1] -= ke
        K[e + 1, e] -= ke
        K[e + 1, e + 1] += ke
        ce = rho * c * L / 6.0
        C[e, e] += 2 * ce
        C[e, e + 1] += ce
        C[e + 1, e] += ce
        C[e + 1, e + 1] += 2 * ce

    bc_left = _check_bc(bc_left)
    bc_right = _check_bc(bc_right)

    T = np.full(n, float(T0))
    history = [T.copy()]
    for step in range(times.size - 1):
        t0, t1 = times[step], times[step + 1]
        dt = t1 - t0
        Keff = C + theta * dt * K
        rhs = (C - (1.0 - theta) * dt * K) @ T
        for e in range(ne):  # constant source, consistent lumping over the step
            fe = src[e] * dt * (x[e + 1] - x[e]) / 2.0
            rhs[e] += fe
            rhs[e + 1] += fe
        fixed = {}
        for node, bc in ((0, bc_left), (n - 1, bc_right)):
            kind = bc["type"]
            if kind == "dirichlet":
                fixed[node] = _bc_value(bc, t1)
            elif kind == "flux":
                rhs[node] += theta * dt * float(bc.get("value", 0.0))
            else:
                h, T_inf = float(bc["h"]), float(bc["T_inf"])
                Keff[node, node] += theta * dt * h
                rhs[node] += theta * dt * h * T_inf + (1 - theta) * dt * h * (T_inf - T[node])
                Keff[node, node] += (1 - theta) * dt * h
        T = _solve_with_dirichlet(Keff, rhs, fixed)
        history.append(T.copy())
    return {"x": x, "times": times, "T": np.asarray(history)}


# --------------------------------------------------------------------------
# 2-D Q4 steady conduction (Poisson) -- manufactured solution verification
# --------------------------------------------------------------------------
def _q4_shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


_Q4_GP = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
          (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]


def q4_steady_conduction(nodes, elems, k=1.0, dirichlet=None, source=0.0, thickness=1.0):
    """Steady 2-D conduction ``-div(k grad T) = f`` on a Q4 mesh.

    ``dirichlet`` maps node index -> temperature; all other nodes are natural
    (insulated) boundaries.  ``source`` is a constant volumetric source.
    """
    nodes = np.asarray(nodes, dtype=float)
    elems = np.asarray(elems, dtype=int)
    n = nodes.shape[0]
    K = np.zeros((n, n))
    f = np.zeros(n)
    for el in elems:
        xy = nodes[el]
        for xi, eta in _Q4_GP:
            N, dN_dxi, dN_deta = _q4_shape(xi, eta)
            J = np.array([[dN_dxi @ xy[:, 0], dN_dxi @ xy[:, 1]],
                          [dN_deta @ xy[:, 0], dN_deta @ xy[:, 1]]])
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("element with non-positive Jacobian")
            dN = np.linalg.solve(J, np.vstack([dN_dxi, dN_deta]))
            K[np.ix_(el, el)] += k * thickness * detJ * (dN.T @ dN)
            f[el] += source * thickness * detJ * N
    fixed = dict(dirichlet or {})
    return {"T": _solve_with_dirichlet(K, f, fixed), "K": K, "f": f}