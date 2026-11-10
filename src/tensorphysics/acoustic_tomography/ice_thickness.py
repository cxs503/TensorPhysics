"""Ice-thickness inversion from under-ice acoustic reflection (echo) travel time.

Two modes are supported, and they are kept strictly separate because they
measure different quantities:

**Draft mode (primary).**  An under-ice transducer at known depth radiates a
short pulse upwards; the echo from the ice underside returns after

    t_echo = 2 d / c_w          ->      d = c_w t_echo / 2

where ``d`` is the *draft* (keel depth below the ice/water interface) and
``c_w`` the local sea-water sound speed.  For a floe in hydrostatic
equilibrium the thickness follows from Archimedes' principle,

    rho_i h = rho_w d      ->      h = (rho_w / rho_i) d = 1.120 d

with ``rho_i = 917``, ``rho_w = 1027 kg/m^3``.  This is the relation used for
the thickness statistics: the acoustic measurement constrains the draft, and
the density ratio converts to thickness (a systematic, documented limitation).

**Through-ice mode (secondary).**  For a transducer frozen into the ice the
two-way time through the ice plate alone gives ``h = c_ice t / 2``.

Negative control (campaign criterion d2): as ``h -> 0`` the echo time goes to
zero identically, because both modes are linear in the geometric length and
carry no offset.
"""

from __future__ import annotations

import numpy as np

from .sound_speed import RHO_ICE, RHO_WATER

#: draft -> thickness factor rho_w / rho_i
DRAFT_TO_THICKNESS: float = RHO_WATER / RHO_ICE


def echo_time(draft, sound_speed):
    """Two-way echo travel time [s] for a keel at ``draft`` [m] (water column)."""
    d = np.asarray(draft, dtype=float)
    c = np.asarray(sound_speed, dtype=float)
    return 2.0 * d / c


def draft_from_echo_time(two_way_time, sound_speed):
    """Draft [m] from the two-way echo time [s] and the local sound speed."""
    t = np.asarray(two_way_time, dtype=float)
    c = np.asarray(sound_speed, dtype=float)
    return 0.5 * c * t


def thickness_from_draft(draft, rho_ice: float = RHO_ICE, rho_water: float = RHO_WATER):
    """Level-ice thickness [m] in hydrostatic equilibrium from the keel draft."""
    return np.asarray(draft, dtype=float) * (rho_water / rho_ice)


def draft_from_thickness(thickness, rho_ice: float = RHO_ICE, rho_water: float = RHO_WATER):
    """Keel draft [m] of level ice of the given thickness."""
    return np.asarray(thickness, dtype=float) * (rho_ice / rho_water)


def thickness_from_echo_time(
    two_way_time, sound_speed, rho_ice: float = RHO_ICE, rho_water: float = RHO_WATER
):
    """Ice thickness [m] from the water-column echo time (draft mode)."""
    return thickness_from_draft(
        draft_from_echo_time(two_way_time, sound_speed), rho_ice=rho_ice, rho_water=rho_water
    )


def thickness_from_ice_echo_time(two_way_time, ice_sound_speed):
    """Ice thickness [m] from the through-ice echo time (secondary mode)."""
    t = np.asarray(two_way_time, dtype=float)
    c = np.asarray(ice_sound_speed, dtype=float)
    return 0.5 * c * t


def thickness_uncertainty(
    two_way_time,
    sigma_time,
    sound_speed,
    sigma_sound_speed=0.0,
    rho_ice: float = RHO_ICE,
    rho_water: float = RHO_WATER,
    sigma_density_ratio: float = 0.0,
):
    """First-order uncertainty of the draft-mode thickness [m].

    sigma_h^2 = (dh/dt)^2 sigma_t^2 + (dh/dc)^2 sigma_c^2
                + (dh/d(ratio))^2 sigma_ratio^2

    with ``dh/dt = c ratio / 2``, ``dh/dc = t ratio / 2`` and
    ``dh/d(density ratio) = d``.
    """
    t = np.asarray(two_way_time, dtype=float)
    c = np.asarray(sound_speed, dtype=float)
    ratio = rho_water / rho_ice
    d = draft_from_echo_time(t, c)
    var = (0.5 * c * ratio) ** 2 * np.asarray(sigma_time, dtype=float) ** 2
    var = var + (0.5 * t * ratio) ** 2 * np.asarray(sigma_sound_speed, dtype=float) ** 2
    var = var + (d * ratio) ** 2 * np.asarray(sigma_density_ratio, dtype=float) ** 2
    return np.sqrt(var)


def measure_draft_echo(
    draft_true,
    sound_speed,
    sigma_time: float = 5.0e-5,
    seed: int = 0,
    bias_time: float = 0.0,
):
    """Synthetic echo-time measurement (Gaussian timing noise) for a draft field."""
    rng = np.random.default_rng(seed)
    t_true = echo_time(draft_true, sound_speed)
    return t_true + bias_time + rng.normal(0.0, sigma_time, size=np.shape(t_true))