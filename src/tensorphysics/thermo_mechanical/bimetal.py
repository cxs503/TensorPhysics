"""Bimetallic strip thermo-mechanical verification anchors.

References
----------
* S. Timoshenko, "Analysis of bi-metal thermostats", J. Opt. Soc. Am. 11 (1925)
  233-255 (curvature of a two-layer strip under uniform temperature rise).
* Boley & Weiner, *Theory of Thermal Stresses* (composite beam, plane sections).

Analytic curvature (equal widths, layer 1 on top of layer 2, ``m = h1/h2``,
``n = E1/E2``, total thickness ``h = h1 + h2``)::

    kappa = 6 (alpha1 - alpha2) dT (1+m)^2
            / [ h ( 3 (1+m)^2 + (1 + m n)(m^2 + 1/(m n)) ) ]

It is compared against an *independently assembled* layered-section solution
(``N_T/EI`` from modular-ratio integration) and against three finite element
discretisations: composite frame element, 2-D Q4 (bilinear, locks in bending)
and 2-D Q8 (serendipity, quadratic convergence).

Sign convention: ``y`` up, layer 1 (top) has the larger ``alpha`` in the
acceptance cases, hence ``kappa > 0`` (strain increasing upwards) and the free
strip deflects towards the low-expansion side (negative ``w``).
"""

from __future__ import annotations

import numpy as np

from .thermoelastic import (LayeredBeamSection, beam_solve, q4_plane_stress,
                            q8_plane_stress)

__all__ = [
    "timoshenko_curvature",
    "timoshenko_end_deflection",
    "arc_end_deflection",
    "exact_composite_solution",
    "bimetal_section",
    "bimetal_beam_fe",
    "bimetal_q4_mesh",
    "bimetal_q4_fe",
    "bimetal_q8_mesh",
    "bimetal_q8_fe",
    "curvature_from_strain_profile",
    "DEFAULT_CASE",
]

#: reference bimetal used by the acceptance criteria (unit width)
DEFAULT_CASE = {
    "h1": 1.0e-3,       # top layer thickness [m]
    "h2": 1.0e-3,       # bottom layer thickness [m]
    "E1": 200.0e9,      # [Pa]
    "E2": 200.0e9,      # [Pa]
    "alpha2": 6.0e-6,   # [1/K]
    "dT": 100.0,        # uniform temperature rise [K]
    "length": 0.10,     # cantilever length [m]
    "nu": 0.3,
}


def timoshenko_curvature(h1, h2, E1, E2, alpha1, alpha2, dT):
    """Timoshenko (1925) free curvature of a two-layer strip [1/m]."""
    m = h1 / h2
    n = E1 / E2
    h = h1 + h2
    num = 6.0 * (alpha1 - alpha2) * dT * (1.0 + m) ** 2
    den = h * (3.0 * (1.0 + m) ** 2 + (1.0 + m * n) * (m ** 2 + 1.0 / (m * n)))
    return num / den


def arc_end_deflection(kappa, length):
    """Exact inextensible-elastica tip deflection of a pure-bending cantilever."""
    k = abs(kappa)
    if k < 1e-15:
        return 0.5 * k * length ** 2
    return (1.0 - np.cos(k * length)) / k


def timoshenko_end_deflection(kappa, length):
    """Small-rotation tip deflection of a uniformly curved cantilever, |kappa|L^2/2."""
    return 0.5 * abs(kappa) * length ** 2


def bimetal_section(h1, h2, E1, E2, alpha1, alpha2, width=1.0, dT=0.0):
    """Layered section with layer 1 stacked on top of layer 2."""
    layers = [
        {"thickness": h2, "E": E2, "alpha": alpha2, "width": width},
        {"thickness": h1, "E": E1, "alpha": alpha1, "width": width},
    ]
    return LayeredBeamSection(layers, dT=dT)


def exact_composite_solution(h1, h2, E1, E2, alpha1, alpha2, dT,
                             width=1.0, length=0.10):
    """Independent (non-Timoshenko) composite-beam solution.

    ``kappa = M_T / EI``, ``eps0 = N_T / EA`` from modular-ratio first and
    second moments; used to cross-check the closed-form 1925 formula.
    """
    sec = bimetal_section(h1, h2, E1, E2, alpha1, alpha2, width, dT)
    eps0, kappa = sec.free_strain(dT)
    N_T, M_T = sec.thermal_resultants(dT)
    return {
        "eps0": eps0,
        "kappa": kappa,
        "N_T": N_T,
        "M_T": M_T,
        "y_bar": sec.y_bar,
        "EA": sec.EA,
        "EI": sec.EI,
        "tip_deflection": timoshenko_end_deflection(kappa, length),
    }


def bimetal_beam_fe(h1, h2, E1, E2, alpha1, alpha2, dT, length=0.10,
                    width=1.0, n_elems=8, tip_load=0.0, tip_moment=0.0):
    """Composite frame-element FE solution of the bimetal cantilever."""
    sec = bimetal_section(h1, h2, E1, E2, alpha1, alpha2, width, dT)
    res = beam_solve(sec, length, n_elems=n_elems, dT=dT,
                     tip_load=tip_load, tip_moment=tip_moment)
    res["section"] = sec
    res["kappa_mean"] = float(np.mean(res["kappa"]))
    return res


# --------------------------------------------------------------------------
# 2-D continuum models of the same strip
# --------------------------------------------------------------------------
def curvature_from_strain_profile(y, eps_xx):
    """Least-squares ``eps = eps0 + kappa (y - mean(y))``; returns kappa, eps0."""
    y = np.asarray(y, dtype=float)
    e = np.asarray(eps_xx, dtype=float)
    A = np.vstack([np.ones_like(y), y - y.mean()]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    return float(coef[1]), float(coef[0])


def _strip_rows(h1, h2, ny1, ny2):
    return np.concatenate([np.linspace(0.0, h2, ny2 + 1)[:-1],
                           np.linspace(h2, h1 + h2, ny1 + 1)])


def bimetal_q4_mesh(length, h1, h2, nx, ny1, ny2):
    """Structured Q4 mesh of a two-layer strip, interface row shared."""
    ys = _strip_rows(h1, h2, ny1, ny2)
    x = np.linspace(0.0, length, nx + 1)
    nodes = np.array([[xi, yi] for yi in ys for xi in x])
    ncol = nx + 1
    elems = []
    layer = []
    for j in range(ys.size - 1):
        for i in range(nx):
            n0 = j * ncol + i
            elems.append([n0, n0 + 1, n0 + 1 + ncol, n0 + ncol])
            layer.append(0 if ys[j + 1] <= h2 + 1e-15 else 1)
    return nodes, np.asarray(elems, dtype=int), np.asarray(layer, dtype=int)


def bimetal_q8_mesh(length, h1, h2, nx, ny1, ny2):
    """Structured 8-node serendipity mesh of a two-layer strip.

    A ``(2 nx + 1) x (2 ny + 1)`` lattice is generated; unused face-centre
    nodes are dropped and the connectivity remapped.
    """
    ys = _strip_rows(h1, h2, ny1, ny2)
    ny = ys.size - 1
    xs = np.linspace(0.0, length, 2 * nx + 1)
    yfine = []
    for j in range(ny):
        yfine.append(ys[j])
        yfine.append(0.5 * (ys[j] + ys[j + 1]))
    yfine.append(ys[-1])
    nfx, nfy = xs.size, len(yfine)
    lat = np.array([[x, y] for y in yfine for x in xs])
    elems = []
    layer = []
    for j in range(ny):
        for i in range(nx):
            def idx(a, b):
                return (2 * j + b) * nfx + (2 * i + a)
            elems.append([idx(0, 0), idx(2, 0), idx(2, 2), idx(0, 2),
                          idx(1, 0), idx(2, 1), idx(1, 2), idx(0, 1)])
            layer.append(0 if ys[j + 1] <= h2 + 1e-15 else 1)
    elems = np.asarray(elems, dtype=int)
    used = np.unique(elems)
    remap = -np.ones(lat.shape[0], dtype=int)
    remap[used] = np.arange(used.size)
    return lat[used], remap[elems], np.asarray(layer, dtype=int)


def _strip_fixed(nodes):
    fixed = {}
    for i in np.where(np.isclose(nodes[:, 0], 0.0))[0]:
        fixed[(int(i), 0)] = 0.0
        fixed[(int(i), 1)] = 0.0
    return fixed


def _strip_postprocess(nodes, elems, layer, res, E1, E2, alpha1, alpha2, dT,
                       length, nx):
    E_elem = np.where(layer == 1, E1, E2)
    cents = nodes[elems].mean(axis=1)
    eps_xx = res["eps"][:, 0]
    band = np.abs(cents[:, 0] - 0.5 * length) <= 1.5 * length / nx
    if band.sum() < 4:
        band = np.argsort(np.abs(cents[:, 0] - 0.5 * length))[:8]
    kappa, eps0 = curvature_from_strain_profile(cents[band, 1], eps_xx[band])
    tip = np.isclose(nodes[:, 0], length)
    u = res["u"].reshape(-1, 2)
    return {
        "nodes": nodes,
        "elems": elems,
        "layer": layer,
        "u": u,
        "sigma": res["sigma"],
        "eps_xx": eps_xx,
        "eps0": eps0,
        "kappa": kappa,
        "tip_deflection": float(np.mean(u[tip, 1])),
        "n_dof": 2 * nodes.shape[0],
        "n_elem": elems.shape[0],
        "E_elem": E_elem,
    }


def bimetal_q4_fe(h1, h2, E1, E2, alpha1, alpha2, dT, length=0.10, nu=0.3,
                  nx=100, ny1=4, ny2=4, thickness=1.0):
    """2-D plane-stress Q4 continuum solution of the bimetal cantilever."""
    nodes, elems, layer = bimetal_q4_mesh(length, h1, h2, nx, ny1, ny2)
    E_elem = np.where(layer == 1, E1, E2)
    res = q4_plane_stress(nodes, elems, E=E_elem, nu=nu,
                          alpha=np.where(layer == 1, alpha1, alpha2),
                          dT_nodes=np.full(nodes.shape[0], float(dT)),
                          thickness=thickness, fixed=_strip_fixed(nodes))
    out = _strip_postprocess(nodes, elems, layer, res, E1, E2, alpha1, alpha2,
                             dT, length, nx)
    out["element"] = "Q4"
    return out


def bimetal_q8_fe(h1, h2, E1, E2, alpha1, alpha2, dT, length=0.10, nu=0.3,
                  nx=50, ny1=2, ny2=2, thickness=1.0):
    """2-D plane-stress Q8 continuum solution of the bimetal cantilever."""
    nodes, elems, layer = bimetal_q8_mesh(length, h1, h2, nx, ny1, ny2)
    E_elem = np.where(layer == 1, E1, E2)
    res = q8_plane_stress(nodes, elems, E=E_elem, nu=nu,
                          alpha=np.where(layer == 1, alpha1, alpha2),
                          dT_nodes=np.full(nodes.shape[0], float(dT)),
                          thickness=thickness, fixed=_strip_fixed(nodes))
    out = _strip_postprocess(nodes, elems, layer, res, E1, E2, alpha1, alpha2,
                             dT, length, nx)
    out["element"] = "Q8"
    return out