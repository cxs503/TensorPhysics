"""Thermal stress concentration at a circular hole under a radial gradient.

Geometry: annular plate, inner radius ``a`` (the hole), outer radius ``b``
(``b/a = 20`` approximates a hole in an infinite plate).  Steady radial
conduction between the hole edge (``T_a``) and the outer edge (``T_b``) gives
the logarithmic field ``T(r) = A ln r + B``.

Exact plane-stress thermo-elastic solution for traction-free inner and outer
edges (Timoshenko & Goodier, *Theory of Elasticity*, art. 13 / Boley & Weiner,
*Theory of Thermal Stresses*, ch. 4)::

    I(r)  = int_a^r T(rho) rho drho,   I_b = I(b)
    sigma_r = E alpha / r^2 * [ (r^2-a^2)/(b^2-a^2) I_b - I(r) ]
    sigma_t = E alpha / r^2 * [ (r^2+a^2)/(b^2-a^2) I_b + I(r) - T(r) r^2 ]

Both traction conditions ``sigma_r(a) = sigma_r(b) = 0`` hold identically.

The thermal stress concentration factor is defined against the thermal stress
scale of the gradient, ``K_t = sigma_theta(a) / (E alpha dT)`` with
``dT = T_a - T_b`` (no far-field mechanical stress exists for a free plate
under an in-plane gradient, so ``E alpha dT`` is the natural normaliser).
"""

from __future__ import annotations

import numpy as np

from .thermoelastic import axisym_ring_thermoelastic

__all__ = [
    "annulus_temperature",
    "annulus_integral_Tr",
    "annulus_thermal_stress",
    "thermal_stress_concentration",
    "hole_fe_sweep",
]


def _log_coeffs(a, b, Ta, Tb):
    A = (Tb - Ta) / np.log(b / a)
    B = Ta - A * np.log(a)
    return A, B


def annulus_temperature(r, a, b, Ta, Tb):
    """Steady radial conduction temperature field ``T(r) = A ln r + B``."""
    A, B = _log_coeffs(a, b, Ta, Tb)
    return A * np.log(np.asarray(r, dtype=float)) + B


def annulus_integral_Tr(r, a, b, Ta, Tb):
    """Closed-form ``int_a^r T(rho) rho drho`` with ``T = A ln rho + B``."""
    A, B = _log_coeffs(a, b, Ta, Tb)

    def F(x):
        x = np.asarray(x, dtype=float)
        return A * (0.5 * x ** 2 * np.log(x) - 0.25 * x ** 2) + 0.5 * B * x ** 2

    return F(r) - F(a)


def annulus_thermal_stress(r, a, b, Ta, Tb, E, alpha):
    """Exact plane-stress radial/hoop thermal stress in the annulus [Pa]."""
    r = np.asarray(r, dtype=float)
    I_b = annulus_integral_Tr(b, a, b, Ta, Tb)
    I_r = annulus_integral_Tr(r, a, b, Ta, Tb)
    T = annulus_temperature(r, a, b, Ta, Tb)
    fac = E * alpha / r ** 2
    s_r = fac * ((r ** 2 - a ** 2) / (b ** 2 - a ** 2) * I_b - I_r)
    s_t = fac * ((r ** 2 + a ** 2) / (b ** 2 - a ** 2) * I_b + I_r - T * r ** 2)
    return s_r, s_t


def thermal_stress_concentration(a, b, Ta, Tb, E, alpha, r=None):
    """Hoop stress at/near the hole edge and the dimensionless ``K_t``.

    ``K_t = sigma_theta(r) / (E alpha (Ta - Tb))`` evaluated at ``r`` (default
    ``r = a``, i.e. the hole edge where the hoop stress is extremal).
    """
    rr = a if r is None else r
    _, s_t = annulus_thermal_stress(rr, a, b, Ta, Tb, E, alpha)
    dT = Ta - Tb
    return {"sigma_theta": float(s_t), "K_t": float(s_t / (E * alpha * dT)),
            "dT": float(dT), "r": float(rr)}


def hole_fe_sweep(a, b, Ta, Tb, E, alpha, nu=0.3, n_elems=64, growth=1.0):
    """Ring-element FE solution and per-node comparison with the exact solution.

    ``growth > 1`` grades the mesh towards the hole (geometric ring spacing).
    Returns the FE hoop stress at the hole edge, its error and the nodal
    comparison arrays.
    """
    if growth == 1.0:
        r = np.linspace(a, b, n_elems + 1)
    else:
        r = [a]
        dr = (b - a) * (growth - 1.0) / (growth ** n_elems - 1.0)
        for _ in range(n_elems):
            r.append(min(r[-1] + dr, b))
            dr *= growth
        r = np.asarray(r)
    T = annulus_temperature(r, a, b, Ta, Tb)
    fe = axisym_ring_thermoelastic(r, T, E, nu, alpha)
    r_mid = 0.5 * (r[:-1] + r[1:])                      # element mid-points
    s_r_ex, s_t_ex = annulus_thermal_stress(r_mid, a, b, Ta, Tb, E, alpha)
    dT_ref = (Ta - Tb) if (Ta - Tb) != 0.0 else 1.0   # K_t = 0 for an isothermal ring
    s_t_fe = fe["sigma_theta"]
    # (i) like-for-like comparison: FE element value vs exact at the same radius
    s_t_fe_mid = float(s_t_fe[0])
    s_t_ex_mid = float(s_t_ex[0])
    # (ii) hole-edge value: linear extrapolation of the two innermost elements
    if s_t_fe.size > 1:
        slope = (s_t_fe[0] - s_t_fe[1]) / (r_mid[0] - r_mid[1])
        s_t_fe_edge = float(s_t_fe[0] + slope * (a - r_mid[0]))
    else:                                             # pragma: no cover
        s_t_fe_edge = s_t_fe_mid
    s_t_ex_edge = float(annulus_thermal_stress(a, a, b, Ta, Tb, E, alpha)[1])
    s_t_ex_nodal = annulus_thermal_stress(r, a, b, Ta, Tb, E, alpha)[1]
    return {
        "r": r,
        "sigma_theta_fe": fe["sigma_theta"],
        "sigma_r_fe": fe["sigma_r"],
        "sigma_theta_exact": s_t_ex,
        "sigma_r_exact": s_r_ex,
        "r_mid": r_mid,
        "sigma_theta_exact_nodes": s_t_ex_nodal,
        "K_t_fe": s_t_fe_mid / (E * alpha * dT_ref),
        "K_t_exact": s_t_ex_mid / (E * alpha * dT_ref),
        "K_t_fe_edge": s_t_fe_edge / (E * alpha * dT_ref),
        "K_t_exact_edge": s_t_ex_edge / (E * alpha * dT_ref),
        "dT": float(Ta - Tb),
        "rel_error_Kt": abs(s_t_fe_mid - s_t_ex_mid) / abs(s_t_ex_mid),
        "rel_error_Kt_edge": abs(s_t_fe_edge - s_t_ex_edge) / abs(s_t_ex_edge),
        "sigma_theta_fe_mid": s_t_fe_mid,
        "sigma_theta_exact_mid": s_t_ex_mid,
        "max_rel_error_profile": float(
            np.max(np.abs(s_t_fe - s_t_ex)) / np.max(np.abs(s_t_ex))),
        "n_elems": n_elems,
    }