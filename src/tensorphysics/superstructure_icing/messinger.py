"""Messinger (1953) icing heat balance and 1-D along-surface ice growth.

The classical control-volume ("extended Messinger") energy balance is solved for
the freezing fraction ``n = mdot_freeze / mdot_imp`` and the surface temperature
``T_s``:

.. math::

    n\\,\\dot m_{im} L_f
      + \\dot m_{im}\\frac{U^2}{2}
      + \\dot m_{im}\\,c_w (T_a - T_s) + \\dot m_{in}\\,c_w (T_{rb} - T_s)
      + \\dot q_h
      =
      h_c (T_s - T_a)
      + \\dot m_{ev} L_v
      + \\varepsilon\\sigma (T_s^4 - T_a^4)

Two regimes follow:

* **rime (dry growth)** -- ``n = 1`` and ``T_s < 0`` is the root of the balance;
  all impinging water freezes.
* **glaze (wet growth)** -- ``T_s = 0`` is imposed and ``n < 1`` follows, the
  remaining water running back down the surface.

Ice thickness grows as ``dh/dt = n \\dot m_{im} / \\rho_{ice}(regime)`` and the
runback is transported cell by cell from the stagnation point outward, which is
the 1-D form of the Messinger water balance used by icing codes (LEWICE /
TURBICE lineage).

Benchmarks used by the tests (see ``tests/test_superstructure_icing.py``):

* rime closed form ``h = \\dot m_{im} t / \\rho_{ice}`` (``n = 1``);
* independent ``solve_ivp`` reference of the stagnation-cell ODE;
* global water-mass conservation (ice + runback = impinged water).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import air
from .droplets import ImpingementResult


# ---------------------------------------------------------------------------
# external heat and mass transfer
# ---------------------------------------------------------------------------
def nusselt_cylinder_churchill_bernstein(re: float, pr: float) -> float:
    """Churchill-Bernstein correlation for cross-flow over a cylinder."""
    return 0.3 + 0.62 * np.sqrt(re) / (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25 * (
        1.0 + (re / 282000.0) ** (5.0 / 8.0)
    ) ** 0.8


def nusselt_cylinder_hilpert(re: float, pr: float) -> float:
    """Hilpert correlation (used as an independent cross-check)."""
    if re < 4.0:
        c, m = 0.989, 0.330
    elif re < 40.0:
        c, m = 0.911, 0.385
    elif re < 4000.0:
        c, m = 0.683, 0.466
    elif re < 40000.0:
        c, m = 0.193, 0.618
    else:
        c, m = 0.027, 0.805
    return c * re**m * pr ** (1.0 / 3.0)


def nusselt_flat_plate(re: float, pr: float) -> float:
    """Laminar flat-plate average Nusselt number (Blasius)."""
    return 0.664 * np.sqrt(re) * pr ** (1.0 / 3.0)


def heat_transfer_coefficient(
    U: float, L: float, T_air: float, kind: str = "cylinder"
) -> float:
    """Convective heat-transfer coefficient ``h_c`` [W/(m2 K)]."""
    nu = air.nu_air(T_air)
    pr = air.pr_air(T_air)
    re = max(U * L / nu, 1.0)
    if kind == "cylinder":
        nu_d = nusselt_cylinder_churchill_bernstein(re, pr)
    elif kind == "plate":
        nu_d = nusselt_flat_plate(re, pr)
    else:
        raise ValueError(f"unknown heat-transfer surface: {kind!r}")
    return float(nu_d * air.k_air(T_air) / L)


def mass_transfer_coefficient(h_c: float, T_air: float) -> float:
    """Heat/mass transfer analogy (Chilton-Colburn) -> ``h_m`` [m/s]."""
    cp = air.cp_air(T_air)
    rho = air.rho_air(T_air)
    le = air.le_air(T_air)
    return h_c / (rho * cp * le ** (2.0 / 3.0))


def evaporation_flux(h_c: float, T_s, T_air: float) -> np.ndarray:
    """Evaporative (or sublimative) mass flux [kg/(m2 s)]."""
    T_s = np.asarray(T_s, dtype=float)
    h_m = mass_transfer_coefficient(h_c, T_air)
    drho = air.rho_v_sat(T_s) - air.rho_v_sat(np.full_like(T_s, T_air))
    return np.clip(h_m * drho, 0.0, None)


def radiation_flux(T_s, T_air: float, eps: float = air.EPS_ICE) -> np.ndarray:
    """Long-wave radiative loss to the environment [W/m2]."""
    T_s = np.asarray(T_s, dtype=float)
    return eps * air.SIGMA_SB * (air.tk(T_s) ** 4 - air.tk(T_air) ** 4)


# ---------------------------------------------------------------------------
# Messinger control volume
# ---------------------------------------------------------------------------
@dataclass
class MessingerState:
    """Freezing fraction / surface temperature / regimes on a surface strip."""

    freezing_fraction: np.ndarray
    T_surface: np.ndarray
    regime: np.ndarray  # 1 = glaze (wet growth), 0 = rime (dry growth)
    evaporative_flux: np.ndarray
    residual: np.ndarray = None  # Newton residual of the dry-growth branch


def _rho_v_sat_and_slope(T_c):
    """Saturation vapour density and its temperature slope [kg/m3, kg/(m3 K)].

    Clausius-Clapeyron + ideal gas, over ice below 0 C (sublimation).
    """
    T_c = np.asarray(T_c, dtype=float)
    T = air.tk(T_c)
    rho = air.rho_v_sat(T_c)
    latent = air.l_vap(T_c)
    drho = rho * (latent / (air.R_V * T * T) - 1.0 / T)
    return rho, drho


def messinger_balance(
    mdot_im,
    mdot_in,
    T_air: float,
    h_c: float,
    U: float,
    q_heat=None,
    r_kin: float = 1.0,
    include_evaporation: bool = True,
    include_radiation: bool = True,
    T_melt: float = 0.0,
    T_runback: float | None = None,
    n_newton: int = 15,
) -> MessingerState:
    """Solve the Messinger balance on a vector of surface cells.

    Parameters
    ----------
    mdot_im, mdot_in
        Impinging and runback-in water fluxes [kg/(m2 s)].
    T_air, h_c, U
        Ambient temperature [C], convective coefficient [W/(m2 K)], airspeed.
    q_heat
        Electro-thermal (or other) heat input [W/m2].

    The dry-growth branch is solved by a damped Newton iteration on ``T_s``
    (the balance is monotone in ``T_s``); if the wall temperature would exceed
    the melting point the wet-growth branch is used instead with ``T_s = 0``.
    """
    mdot_im = np.atleast_1d(np.asarray(mdot_im, dtype=float))
    mdot_in = np.broadcast_to(np.atleast_1d(np.asarray(mdot_in, dtype=float)), mdot_im.shape).copy()
    q = (
        np.zeros_like(mdot_im)
        if q_heat is None
        else np.broadcast_to(np.asarray(q_heat, dtype=float), mdot_im.shape).copy()
    )

    T_runback = T_melt if T_runback is None else T_runback
    # hoisted constants (keeps the Newton loop free of python-level calls)
    h_m = mass_transfer_coefficient(h_c, T_air) if include_evaporation else 0.0
    rho_v_air = float(air.rho_v_sat(np.array([T_air]))[0])
    eps_sigma = air.EPS_ICE * air.SIGMA_SB if include_radiation else 0.0
    ta_k = air.tk(T_air)
    # Runback water leaves an upstream cell at the melting temperature (glaze
    # regime), so it enters the next cell at T_runback -- *not* at the ambient
    # temperature.  Warming it from T_air again would double count the sensible
    # heat and make downstream protection artificially expensive.
    m_sum = mdot_im + mdot_in
    base = (
        mdot_im * r_kin * U**2 / 2.0
        + mdot_im * air.C_W * T_air
        + mdot_in * air.C_W * T_runback
        + q
        + h_c * T_air
    )
    denom = h_c + m_sum * air.C_W  # d/dT_s of the linear (non-radiative) part

    def _resid(Ts, n, rho_v, d_rho_v):
        ev = h_m * np.maximum(rho_v - rho_v_air, 0.0)
        rad = eps_sigma * (air.tk(Ts) ** 4 - ta_k**4)
        return (
            n * mdot_im * air.L_F
            + base
            - denom * Ts
            - ev * air.l_vap(Ts)
            - rad
        )

    # --- regime test: can the balance hold T_s at the melting point? -----
    rho_v0, _ = _rho_v_sat_and_slope(np.full_like(mdot_im, T_melt))
    f0 = _resid(np.full_like(mdot_im, T_melt), 1.0, rho_v0, rho_v0)
    glaze = f0 > 0.0
    has_water = mdot_im > 1e-14

    # --- dry growth (n = 1), damped Newton on T_s for the rime cells only -
    Ts = np.where(has_water, T_melt, T_air)
    f_end = np.zeros_like(mdot_im)
    idx = np.nonzero(~glaze & has_water)[0]
    if idx.size:
        Ts_sub = np.full(idx.size, min(T_air, T_melt))
        base_i, den_i = base[idx], denom[idx]
        mdot_i = mdot_im[idx]
        for _ in range(n_newton):
            rho_v, d_rho_v = _rho_v_sat_and_slope(Ts_sub)
            ev = h_m * np.maximum(rho_v - rho_v_air, 0.0)
            rad = eps_sigma * (air.tk(Ts_sub) ** 4 - ta_k**4)
            f = mdot_i * air.L_F + base_i - den_i * Ts_sub - ev * air.l_vap(Ts_sub) - rad
            df = -den_i - h_m * air.l_vap(Ts_sub) * d_rho_v - eps_sigma * 4.0 * air.tk(Ts_sub) ** 3
            step = f / np.where(np.abs(df) > 1e-12, df, -1.0)
            step = np.clip(step, -20.0, 20.0)
            Ts_sub = np.clip(Ts_sub - step, -78.0, T_melt)
            if np.max(np.abs(step)) < 1e-8:
                break
        Ts[idx] = Ts_sub
        f_end[idx] = f

    # --- wet growth (T_s = T_melt): freezing fraction ---------------------
    rho_v0, _ = _rho_v_sat_and_slope(np.full_like(mdot_im, T_melt))
    f0 = _resid(np.full_like(mdot_im, T_melt), 1.0, rho_v0, rho_v0)
    glaze = f0 > 0.0
    has_water = mdot_im > 1e-14
    ev0 = h_m * np.maximum(rho_v0 - rho_v_air, 0.0)
    rad0 = eps_sigma * (air.tk(T_melt) ** 4 - ta_k**4)
    n_glaze = np.where(
        has_water,
        (
            h_c * (T_melt - T_air)
            + ev0 * air.l_vap(np.full_like(mdot_im, T_melt))
            + rad0
            - mdot_im * r_kin * U**2 / 2.0
            - mdot_im * air.C_W * (T_air - T_melt)
            - mdot_in * air.C_W * (T_runback - T_melt)
            - q
        )
        / np.where(has_water, mdot_im * air.L_F, 1.0),
        0.0,
    )
    n_glaze = np.clip(n_glaze, 0.0, 1.0)

    n = np.where(glaze, n_glaze, np.where(has_water, 1.0, 0.0))
    T_s = np.where(glaze, T_melt, np.where(has_water, Ts, T_air))
    rho_v_s, _ = _rho_v_sat_and_slope(T_s)
    evap = h_m * np.maximum(rho_v_s - rho_v_air, 0.0)
    return MessingerState(
        freezing_fraction=n,
        T_surface=T_s,
        regime=glaze.astype(int),
        evaporative_flux=evap,
        residual=np.where(has_water, np.where(glaze, 0.0, f_end), 0.0),
    )


def rime_thickness(t, lwc: float, U: float, beta, rho_i: float = air.RHO_ICE_RIME):
    """Closed-form dry-growth (rime) ice thickness [m].

    Valid whenever the Messinger balance keeps every impinging droplet frozen
    (``n = 1``): ``h(t) = LWC U beta t / rho_i``.
    """
    return lwc * U * np.asarray(beta, dtype=float) * t / rho_i


def stefan_rime_thickness(t, dT: float, rho_i: float = air.RHO_ICE_RIME, k_i: float = air.K_ICE):
    """Conduction-limited (Stefan) rime thickness ``h = sqrt(2 k dT t /(rho L))``.

    Used only as an *upper-bound* comparator: the quasi-steady Messinger model
    neglects conduction through the ice layer and therefore equals the Stefan
    solution only at short times.
    """
    return np.sqrt(2.0 * k_i * dT * np.asarray(t, dtype=float) / (rho_i * air.L_F))


# ---------------------------------------------------------------------------
# 1-D along-surface growth
# ---------------------------------------------------------------------------
@dataclass
class IceGrowthResult:
    """Output of :func:`ice_growth_1d`."""

    s: np.ndarray
    t: np.ndarray
    h_ice: np.ndarray  # (n_t, n_s) ice thickness [m]
    freezing_fraction: np.ndarray  # final
    T_surface: np.ndarray
    regime: np.ndarray
    runback_out: float  # water leaving the strip per unit span [kg/m]
    impinged: float  # water impinged per unit span [kg/m]
    ice_mass: float  # ice mass per unit span [kg/m]
    q_heat: np.ndarray

    @property
    def h_final(self) -> np.ndarray:
        return self.h_ice[-1]

    @property
    def closure_error(self) -> float:
        """Relative water-mass closure error (should be ~1e-12)."""
        tot = self.impinged
        return abs(self.ice_mass + self.runback_out + self.evaporated - tot) / max(tot, 1e-30)

    evaporated: float = 0.0


def _rho_v_sat_scalar(t_c: float) -> tuple[float, float]:
    """Scalar saturation vapour density and slope (keeps the march fast)."""
    import math

    T = t_c + 273.15
    if t_c < 0.0:
        e = 611.15 * math.exp((23.036 - t_c / 333.7) * (t_c / (279.82 + t_c)))
        latent = air.L_S
    else:
        e = 611.21 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
        latent = air.L_V
    rho = e / (air.R_V * T)
    return rho, rho * (latent / (air.R_V * T * T) - 1.0 / T)


def messinger_balance_scalar(
    mdot_im: float,
    mdot_in: float,
    T_air: float,
    h_c: float,
    U: float,
    q: float = 0.0,
    r_kin: float = 1.0,
    include_evaporation: bool = True,
    include_radiation: bool = True,
    T_melt: float = 0.0,
    T_runback: float | None = None,
) -> tuple[float, float, int, float, float]:
    """Scalar Messinger balance (same model as :func:`messinger_balance`).

    Returns ``(freezing_fraction, T_surface, regime, evaporative_flux,
    residual)``.  Implemented with plain floats: the downstream march calls it
    once per cell per time step, where the numpy dispatch overhead of the
    vectorised version dominates the arithmetic.
    """
    import math

    T_runback = T_melt if T_runback is None else T_runback
    h_m = mass_transfer_coefficient(h_c, T_air) if include_evaporation else 0.0
    rho_v_air, _ = _rho_v_sat_scalar(T_air)
    eps_sigma = air.EPS_ICE * air.SIGMA_SB if include_radiation else 0.0
    ta_k = T_air + 273.15
    m_sum = mdot_im + mdot_in
    base = (
        mdot_im * r_kin * U * U / 2.0
        + mdot_im * air.C_W * T_air
        + mdot_in * air.C_W * T_runback
        + q
        + h_c * T_air
    )
    denom = h_c + m_sum * air.C_W

    def resid(Ts, n):
        rho_v, _d = _rho_v_sat_scalar(Ts)
        ev = h_m * max(rho_v - rho_v_air, 0.0)
        latent = air.L_S if Ts < 0.0 else air.L_V
        rad = eps_sigma * ((Ts + 273.15) ** 4 - ta_k**4)
        return n * mdot_im * air.L_F + base - denom * Ts - ev * latent - rad

    f0 = resid(T_melt, 1.0)
    if mdot_im <= 1e-14:
        n, Ts, regime, res = 0.0, T_air, 0, 0.0
    elif f0 > 0.0:
        rho_v0, _d = _rho_v_sat_scalar(T_melt)
        ev0 = h_m * max(rho_v0 - rho_v_air, 0.0)
        rad0 = eps_sigma * ((T_melt + 273.15) ** 4 - ta_k**4)
        latent = air.L_S if T_melt < 0.0 else air.L_V
        n = (h_c * (T_melt - T_air) + ev0 * latent + rad0
             - mdot_im * r_kin * U * U / 2.0
             - mdot_im * air.C_W * (T_air - T_melt)
             - mdot_in * air.C_W * (T_runback - T_melt)
             - q) / (mdot_im * air.L_F)
        n = min(max(n, 0.0), 1.0)
        Ts, regime, res = T_melt, 1, 0.0
    else:
        Ts = min(T_air, T_melt)
        for _ in range(25):
            rho_v, d_rho_v = _rho_v_sat_scalar(Ts)
            ev = h_m * max(rho_v - rho_v_air, 0.0)
            latent = air.L_S if Ts < 0.0 else air.L_V
            rad = eps_sigma * ((Ts + 273.15) ** 4 - ta_k**4)
            f = mdot_im * air.L_F + base - denom * Ts - ev * latent - rad
            df = -denom - h_m * latent * d_rho_v - eps_sigma * 4.0 * (Ts + 273.15) ** 3
            step = f / df if abs(df) > 1e-12 else 0.0
            step = min(max(step, -20.0), 20.0)
            Ts = min(max(Ts - step, -78.0), T_melt)
            if abs(step) < 1e-9:
                break
        n, regime, res = 1.0, 0, f
    rho_v_s, _d = _rho_v_sat_scalar(Ts)
    ev = h_m * max(rho_v_s - rho_v_air, 0.0)
    return n, Ts, regime, ev, res


def _cell_widths(s):
    """Cell widths of an arc-length grid (last cell copied from the previous)."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    ds = np.empty(s.size)
    if s.size == 1:
        ds[0] = max(s[0], 1e-3)
    else:
        ds[:-1] = np.diff(s)
        ds[-1] = ds[-2]
    return ds


def march_runback(s, ds, mdot_im, T_air, h_c, U, q):
    """Exact downstream march of the 1-D Messinger water film.

    Cells are visited from the stagnation point outwards; the runback entering a
    cell is *exactly* the runback leaving the previous one, so no iteration is
    needed (this replaces a Gauss-Seidel sweep, which needs one sweep per cell to
    propagate information downstream).  Evaporation is limited by the water
    actually available in the film, which keeps the water budget closed.
    """
    n_cell = s.size
    n = np.zeros(n_cell)
    ev = np.zeros(n_cell)
    Ts = np.zeros(n_cell)
    reg = np.zeros(n_cell, dtype=int)
    frozen = np.zeros(n_cell)
    w = 0.0
    ev_total = 0.0
    for i in range(n_cell):
        n_i, T_i, r_i, e_i, _res = messinger_balance_scalar(
            float(mdot_im[i]), float(w / ds[i]), T_air, h_c, U, q=float(q[i])
        )
        n[i] = n_i
        ev[i] = e_i
        Ts[i] = T_i
        reg[i] = r_i
        frozen[i] = n[i] * mdot_im[i]
        avail = w + (mdot_im[i] - frozen[i]) * ds[i]
        ev_cell = min(ev[i] * ds[i], max(avail, 0.0))
        ev_total += ev_cell
        w = max(avail - ev_cell, 0.0)
    return {
        "freezing_fraction": n,
        "evaporative_flux": ev,
        "T_surface": Ts,
        "regime": reg,
        "frozen_flux": frozen,
        "runback_out": w,
        "evaporated_cell": ev_total,
    }


def ice_growth_1d(
    s,
    beta,
    U: float,
    T_air: float,
    LWC: float,
    t_end: float,
    dt: float = 2.0,
    h_c: float | None = None,
    L_char: float | None = None,
    q_heat=None,
    rho_ice: float | None = None,
    kind: str = "cylinder",
    record_every: int = 1,
) -> IceGrowthResult:
    """Transient 1-D Messinger ice growth on a surface strip.

    The strip is parameterised by arc length ``s`` from the stagnation point;
    ``beta(s)`` is the local collection efficiency.  Water enters from the
    stagnation point, freezes partially, and the remainder runs back to the next
    cell (quasi-steady film, no film inertia).
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n_cell = s.size
    ds = np.empty(n_cell)
    if n_cell == 1:
        ds[0] = max(s[0], 1e-3)
    else:
        ds[:-1] = np.diff(s)
        ds[-1] = ds[-2]

    L_c = float(L_char if L_char is not None else max(2.0 * s.max(), 0.05))
    if h_c is None:
        h_c = heat_transfer_coefficient(U, L_c, T_air, kind=kind)
    q = np.zeros(n_cell) if q_heat is None else np.broadcast_to(np.asarray(q_heat, dtype=float), (n_cell,)).copy()

    rho = float(rho_ice) if rho_ice is not None else air.RHO_ICE_GLAZE
    n_steps = max(int(np.ceil(t_end / dt)), 1)
    dt_eff = t_end / n_steps

    h = np.zeros(n_cell)
    H = np.zeros((n_steps + 1, n_cell))
    times = np.linspace(0.0, t_end, n_steps + 1)
    runback_out = 0.0
    n_last = np.zeros(n_cell)
    Ts_last = np.full(n_cell, T_air)
    reg_last = np.zeros(n_cell, dtype=int)

    mdot_im = LWC * U * beta
    evap_total = 0.0
    for k in range(n_steps):
        m = march_runback(s, ds, mdot_im, T_air, h_c, U, q)
        frozen = m["frozen_flux"]
        n_last = m["freezing_fraction"]
        Ts_last = m["T_surface"]
        reg_last = m["regime"]
        h = h + frozen * dt_eff / rho
        evap_total += m["evaporated_cell"] * dt_eff
        runback_out += m["runback_out"] * dt_eff
        H[k + 1] = h
    impinged = float(LWC * U * np.sum(beta * ds) * t_end)
    ice_mass = float(rho * np.sum(h * ds))
    return IceGrowthResult(
        s=s,
        t=times,
        h_ice=H,
        freezing_fraction=n_last,
        T_surface=Ts_last,
        regime=reg_last,
        runback_out=runback_out,
        impinged=impinged,
        ice_mass=ice_mass,
        q_heat=q,
        evaporated=evap_total,
    )


def ice_growth_steady(
    s,
    beta,
    U: float,
    T_air: float,
    LWC: float,
    t_end: float,
    h_c: float,
    L_char: float | None = None,
    q_heat=None,
    rho_ice: float = air.RHO_ICE_GLAZE,
    n_sweep: int = 80,
    tol: float = 1e-12,
) -> dict:
    """Quasi-steady Messinger ice thickness (no time stepping).

    Under constant ambient conditions the Messinger state (freezing fraction,
    runback flux) is time independent, so the ice thickness is exactly
    ``h_i = n_i \\dot m_{im,i} t_end / \\rho_ice``.  This is algebraically the
    same model as :func:`ice_growth_1d` but without the marching loop, which makes
    it usable as the inner routine of a heater-power optimiser.  Agreement with
    the transient solver is asserted in the test-suite.
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    ds = _cell_widths(s)
    n_cell = s.size
    q = np.zeros(n_cell) if q_heat is None else np.broadcast_to(np.asarray(q_heat, dtype=float), (n_cell,))
    mdot_im = LWC * U * beta
    m = march_runback(s, ds, mdot_im, T_air, h_c, U, q)
    h = m["frozen_flux"] * t_end / rho_ice
    return {
        "h": h,
        "s": s,
        "freezing_fraction": m["freezing_fraction"],
        "T_surface": m["T_surface"],
        "regime": m["regime"],
        "runback_out": m["runback_out"] * t_end,
        "impinged": float(np.sum(mdot_im * ds) * t_end),
        "ice_mass": float(np.sum(rho_ice * h * ds)),
        "sweeps": 1,
    }


def messinger_stagnation_reference(
    beta0: float,
    U: float,
    T_air: float,
    LWC: float,
    h_c: float,
    t_end: float,
    rho_i: float = air.RHO_ICE_GLAZE,
    n_eval: int = 601,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference solution of the stagnation-cell Messinger ODE.

    Written as a *separate* code path from :func:`messinger_balance` (closed-form
    glaze expression, adaptive ``solve_ivp`` integration) so that the two
    implementations can be cross-checked without sharing a bug.
    """
    from scipy.integrate import solve_ivp

    mdot = LWC * U * beta0

    def freeze_flux(Ts_guess=0.0):
        # glaze assumption at T_s = 0 (valid for the validation conditions)
        h_m = h_c / (air.rho_air(T_air) * air.cp_air(T_air) * air.le_air(T_air) ** (2.0 / 3.0))
        ev = max(
            h_m * (air.rho_v_sat(0.0) - air.rho_v_sat(T_air)),
            0.0,
        )
        q_conv = h_c * (0.0 - T_air)
        q_evap = ev * air.L_V  # glaze: liquid film at 0 C
        q_rad = air.EPS_ICE * air.SIGMA_SB * (air.tk(0.0) ** 4 - air.tk(T_air) ** 4)
        q_sens = mdot * air.C_W * (T_air - 0.0)
        q_kin = mdot * U**2 / 2.0
        return (q_conv + q_evap + q_rad - q_sens - q_kin) / (mdot * air.L_F)

    n = float(np.clip(freeze_flux(), 0.0, 1.0))
    # rime check: if the balance cannot hold T_s at 0, everything freezes
    if (h_c * (0.0 - T_air)) < (mdot * air.C_W * (T_air - 0.0) + mdot * U**2 / 2.0):
        n = 1.0

    sol = solve_ivp(
        lambda t, y: np.array([n * mdot / rho_i]),
        (0.0, t_end),
        np.array([0.0]),
        rtol=1e-10,
        atol=1e-14,
        dense_output=True,
    )
    t = np.linspace(0.0, t_end, n_eval)
    return t, sol.sol(t)[0]