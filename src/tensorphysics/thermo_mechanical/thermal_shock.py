"""Transient thermal shock of a semi-infinite solid (Carslaw & Jaeger anchor).

Surface of a half-space at initial temperature ``T0`` is suddenly brought to
``Ts`` at ``t = 0``.  The classic similarity solution (Carslaw & Jaeger,
*Conduction of Heat in Solids*, 2nd ed., §2.4)::

    T(y,t) = T0 + (Ts - T0) erfc( y / (2 sqrt(kappa t)) ),   kappa = k/(rho c)

For a semi-infinite body whose in-plane expansion is prevented (all points at
the same depth share the same temperature), the elastic state is uniaxial
stress in ``y`` with ``sigma_yy = 0`` and the in-plane thermal stress::

    sigma_xx(y,t) = sigma_zz(y,t) = -E alpha (T(y,t) - T0) / (1 - nu)

so the surface value jumps immediately to ``-E alpha dT / (1 - nu)`` (the
classical thermal-shock stress, independent of time) and the depth profile
mirrors the erfc temperature profile.

The FE chain couples two genuine discretisations: transient 1-D conduction
(theta-method, graded mesh) feeding a 1-D plane-strain-blocked elasticity FE
with thermal eigenstrain ``eps_th = (3 lambda + 2 mu) alpha T / (lambda + 2 mu)``
and modulus ``lambda + 2 mu``; ``sigma_yy`` must stay identically zero.
"""

from __future__ import annotations

import numpy as np

from .heat import graded_mesh, heat1d_transient
from .thermoelastic import apply_dirichlet

__all__ = [
    "lamé",
    "kappa_thermal",
    "surface_step_temperature",
    "surface_step_stress",
    "half_space_stress_from_temperature",
    "thermal_shock_fe",
    "DEFAULT_MATERIAL",
]

DEFAULT_MATERIAL = {
    "k": 50.0,          # W/(m K)
    "rho": 7800.0,      # kg/m^3
    "c": 500.0,         # J/(kg K)
    "E": 210.0e9,       # Pa
    "nu": 0.3,
    "alpha": 12.0e-6,   # 1/K
    "T0": 20.0,         # degC
    "Ts": 0.0,          # surface step temperature [degC]
}


def lamé(E, nu):
    """Lamé parameters ``(lambda, mu)`` from ``E, nu``."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def kappa_thermal(k, rho, c):
    """Thermal diffusivity [m^2/s]."""
    return k / (rho * c)


def surface_step_temperature(y, t, Ts, T0, kappa):
    """``T(y,t)`` for a surface temperature step (erfc similarity solution)."""
    from math import erfc

    y = np.asarray(y, dtype=float)
    eta = y / (2.0 * np.sqrt(kappa * t))
    vals = np.array([erfc(float(e)) for e in np.atleast_1d(eta)])
    return T0 + (Ts - T0) * vals.reshape(y.shape)


def surface_step_stress(T, T0, E, alpha, nu):
    """In-plane thermal stress ``-E alpha (T-T0)/(1-nu)`` [Pa]."""
    return -E * alpha * (np.asarray(T, dtype=float) - T0) / (1.0 - nu)


def half_space_stress_from_temperature(y, T, T0, E, alpha, nu):
    """Convenience wrapper: stresses of the blocked half-space at depth ``y``."""
    sig = surface_step_stress(T, T0, E, alpha, nu)
    return {"y": np.asarray(y, dtype=float), "sigma_xx": sig,
            "sigma_yy": np.zeros_like(sig)}


def thermal_shock_fe(times, material=None, y_max=0.15, first_elem=5.0e-5,
                     growth=1.15, theta=1.0, n_extra_steps=None):
    """Coupled transient conduction + elasticity FE for the surface step.

    Parameters
    ----------
    times : increasing array of output times [s]
    material : dict overriding ``DEFAULT_MATERIAL``
    y_max : depth of the FE model [m] (must exceed several diffusion lengths)
    first_elem, growth : graded mesh parameters (boundary layer at ``y = 0``)
    theta : time integration parameter (1 = backward Euler)
    n_extra_steps : optional list of sub-step counts per output interval
        controlling ``dt`` (default: 40 sub-steps per interval, and at least
        200 uniform sub-steps on the first interval to capture the step).

    Returns
    -------
    dict with the mesh, temperature/stress histories at each output time,
    the discrete elastic solve residual (``sigma_yy`` should vanish) and the
    analytic comparison arrays.
    """
    mat = dict(DEFAULT_MATERIAL)
    if material:
        mat.update(material)
    k, rho, c = mat["k"], mat["rho"], mat["c"]
    E, nu, alpha = mat["E"], mat["nu"], mat["alpha"]
    T0, Ts = mat["T0"], mat["Ts"]
    kap = kappa_thermal(k, rho, c)

    y = graded_mesh(y_max, first_elem, growth)
    times = np.asarray(times, dtype=float)
    # build a fine time grid (sub-stepping), denser on the first interval
    t_grid = [0.0]
    for i in range(times.size):
        t_start = 0.0 if i == 0 else times[i - 1]
        n_sub = 400 if i == 0 else (n_extra_steps[i] if n_extra_steps else 40)
        seg = np.linspace(t_start, times[i], n_sub + 1)[1:]
        t_grid.extend(seg.tolist())
    t_grid = np.asarray(t_grid, dtype=float)
    bc_top = {"type": "dirichlet", "value": lambda t: Ts if t > 0 else T0}
    bc_bot = {"type": "dirichlet", "value": T0}
    heat = heat1d_transient(y, k, rho, c, t_grid, T0=T0,
                            bc_left=bc_top, bc_right=bc_bot, theta=theta)
    T_hist = heat["T"]

    lam, mu = lamé(E, nu)
    E_eff = lam + 2.0 * mu
    th_scale = (3.0 * lam + 2.0 * mu) * alpha / (lam + 2.0 * mu)

    out = {"y": y, "kappa": kap, "material": mat,
           "times": times, "T": [], "sigma_xx": [], "sigma_yy": [],
           "T_exact": [], "sigma_xx_exact": [], "max_rel_err_profile": []}
    idx_of = {t: i for i, t in enumerate(t_grid)}
    for t in times:
        T = T_hist[idx_of[t]].copy()
        # elastic FE: 1-D bar with eigenstrain, plane strain blocked in-plane
        n = y.size
        K = np.zeros((n, n))
        f = np.zeros(n)
        for e in range(n - 1):
            L = y[e + 1] - y[e]
            ke = E_eff / L
            K[e, e] += ke
            K[e, e + 1] -= ke
            K[e + 1, e] -= ke
            K[e + 1, e + 1] += ke
            eps_th = th_scale * 0.5 * (T[e] + T[e + 1])   # relative to T0
            f[e] += -E_eff * eps_th
            f[e + 1] += E_eff * eps_th
        # far end fixed only: the surface stays traction free (sigma_yy = 0)
        u = apply_dirichlet(K, f, {n - 1: 0.0})
        eps_yy = np.diff(u) / np.diff(y)
        sig_yy = np.zeros(n - 1)
        for e in range(n - 1):
            eps_th = th_scale * 0.5 * (T[e] + T[e + 1])
            sig_yy[e] = E_eff * (eps_yy[e] - eps_th)
        sig_xx = surface_step_stress(T, T0, E, alpha, nu)
        T_ex = surface_step_temperature(y, t, Ts, T0, kap)
        out["T"].append(T)
        out["sigma_xx"].append(sig_xx)
        out["sigma_yy"].append(sig_yy)
        out["T_exact"].append(T_ex)
        out["sigma_xx_exact"].append(surface_step_stress(T_ex, T0, E, alpha, nu))
        ref = max(abs(surface_step_stress(np.array([Ts]), T0, E, alpha, nu)[0]), 1e-30)
        out["max_rel_err_profile"].append(
            float(np.max(np.abs(sig_xx - surface_step_stress(T_ex, T0, E, alpha, nu))) / ref))
    out["times"] = times
    out["surface_stress_fe"] = [float(s[0]) for s in out["sigma_xx"]]
    out["surface_stress_exact"] = [-E * alpha * (Ts - T0) / (1.0 - nu)] * times.size
    out["max_sigma_yy"] = float(max(np.max(np.abs(s)) for s in out["sigma_yy"]))
    return out