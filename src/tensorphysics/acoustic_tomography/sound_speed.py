"""Seawater / sea-ice sound-speed relations and their inversion.

Two seawater models are implemented so that the campaign fixed relation and an
independent literature relation can be cross-checked against each other:

``engineering_sound_speed``
    The closed form fixed by the campaign specification (Mackenzie-type series
    in temperature, linear salinity and linear pressure terms; ``pressure_dbar``
    is numerically ~ metres of depth):

        c = 1449.2 + 4.6 T - 5.5e-2 T^2 + 2.9e-4 T^3
            + 1.34 (S - 35) + 1.6e-2 P

``mackenzie_sound_speed``
    The full nine-term Mackenzie (1981) expression with explicit depth
    dependence, used as the independent cross-check.

Validity range of both (Mackenzie 1981):  -2 <= T <= 30 degC, 25 <= S <= 40 psu,
0 <= D <= 8000 m.  Over that box ``dc/dT > 0`` for both models, so the
temperature inversion is unique and a bracketed iteration converges.

For fresh-water ice Ih an engineering linear relation is used
(``ice_sound_speed``); it is only needed for the through-ice echo mode and is
deliberately kept out of the draft-based ice-thickness inversion.

References
----------
Mackenzie, K.V. (1981), "Nine-term equation for sound speed in the oceans",
    J. Acoust. Soc. Am. 70(3), 807-812.
Fofonoff & Millard (1983), UNESCO technical papers in marine science 44
    (EOS-80 density of sea water, used for the floe draft/thickness ratio).
Timco & Weeks (2010), "A review of the engineering properties of sea ice",
    Cold Regions Science and Technology 60, 107-129 (ice density / moduli).
"""

from __future__ import annotations

import numpy as np

# Validity box of the Mackenzie (1981) relation.
T_MIN: float = -2.0
T_MAX: float = 30.0
S_MIN: float = 25.0
S_MAX: float = 40.0
D_MIN: float = 0.0
D_MAX: float = 8000.0

#: Physical lower bound used by the campaign negative control (d):
#: at 0 degC fresh water the sound speed must stay above 1400 m/s.
SOUND_SPEED_LOWER_BOUND: float = 1400.0

#: Typical densities for the hydrostatic draft <-> thickness conversion.
RHO_ICE: float = 917.0      # kg/m^3, sea-ice density (Timco & Weeks 2010)
RHO_WATER: float = 1027.0   # kg/m^3, Arctic surface sea-water density


def engineering_sound_speed(temperature, salinity=35.0, pressure_dbar=0.0):
    """Campaign-fixed sound speed [m/s]; ``pressure_dbar`` ~ depth in metres."""
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    p = np.asarray(pressure_dbar, dtype=float)
    return (
        1449.2
        + 4.6 * t
        - 5.5e-2 * t**2
        + 2.9e-4 * t**3
        + 1.34 * (s - 35.0)
        + 1.6e-2 * p
    )


def mackenzie_sound_speed(temperature, salinity=35.0, depth=0.0):
    """Nine-term Mackenzie (1981) sound speed [m/s] (depth in metres)."""
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    d = np.asarray(depth, dtype=float)
    return (
        1448.96
        + 4.591 * t
        - 5.304e-2 * t**2
        + 2.374e-4 * t**3
        + 1.340 * (s - 35.0)
        + 1.630e-2 * d
        + 1.675e-7 * d**2
        - 1.025e-2 * t * (s - 35.0)
        - 7.139e-13 * t * d**3
    )


def engineering_dc_dt(temperature):
    """Analytic sensitivity dc/dT [m/s/degC] of :func:`engineering_sound_speed`."""
    t = np.asarray(temperature, dtype=float)
    return 4.6 - 1.1e-1 * t + 8.7e-4 * t**2


def invert_temperature_from_sound_speed(
    sound_speed, salinity=35.0, pressure_dbar=0.0, t_lo=T_MIN, t_hi=T_MAX, n_iter=100
):
    """Invert the campaign relation for temperature [degC].

    The cubic in ``T`` is strictly increasing on ``[t_lo, t_hi]`` (``dc/dT > 0``
    there), so the root is unique.  A bracketed Newton/bisection iteration is
    used, vectorised with numpy broadcasting.  Targets outside the reachable
    interval ``[c(t_lo), c(t_hi)]`` return ``nan`` -- no silent clamping, so
    downstream statistics expose such points instead of hiding them.
    """
    c = np.asarray(sound_speed, dtype=float)
    s = np.asarray(salinity, dtype=float)
    p = np.asarray(pressure_dbar, dtype=float)

    c_lo = engineering_sound_speed(t_lo, s, p)
    c_hi = engineering_sound_speed(t_hi, s, p)
    reachable = (c >= c_lo) & (c <= c_hi)

    lo = np.full(np.shape(c), float(t_lo))
    hi = np.full(np.shape(c), float(t_hi))
    t = 0.5 * (lo + hi)
    for _ in range(n_iter):
        f = engineering_sound_speed(t, s, p) - c
        # bracket update (f is increasing in t)
        lo = np.where(f < 0.0, t, lo)
        hi = np.where(f > 0.0, t, hi)
        slope = engineering_dc_dt(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(np.abs(slope) > 1e-9, f / slope, 0.0)
        newton = t - step
        # accept Newton where it stays inside the current bracket
        t_next = np.where((newton > lo) & (newton < hi), newton, 0.5 * (lo + hi))
        if np.all(np.abs(t_next - t) <= 1e-12):
            t = t_next
            break
        t = t_next
    residual = np.abs(engineering_sound_speed(t, s, p) - c)
    return np.where(reachable & (residual < 1e-6), t, np.nan)


def temperature_uncertainty(sound_speed_uncertainty, temperature):
    """Propagated uncertainty sigma_T = sigma_c / (dc/dT) [degC]."""
    sigma_c = np.asarray(sound_speed_uncertainty, dtype=float)
    slope = engineering_dc_dt(np.asarray(temperature, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(slope) > 1e-9, sigma_c / np.abs(slope), np.nan)


def ice_sound_speed(temperature_celsius):
    """Longitudinal sound speed in fresh-water ice Ih [m/s] (engineering fit).

    Linear fit anchored at 3800 m/s at 0 degC with +2.5 m/s per K of cooling,
    consistent with the 3.8-3.9 km/s range reported for ice Ih between -2 and
    -20 degC (Timco & Weeks 2010 review of elastic properties).  Stated
    accuracy ~1 %; used only for the through-ice echo mode, never for the
    draft-based thickness inversion.
    """
    t = np.asarray(temperature_celsius, dtype=float)
    return 3800.0 - 2.5 * t


def sound_speed_valid_mask(temperature, salinity, depth):
    """Boolean mask of points inside the Mackenzie (1981) validity box."""
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    d = np.asarray(depth, dtype=float)
    return (
        (t >= T_MIN) & (t <= T_MAX) & (s >= S_MIN) & (s <= S_MAX) & (d >= D_MIN) & (d <= D_MAX)
    )