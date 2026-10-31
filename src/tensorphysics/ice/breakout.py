"""Breakout criterion for a vessel beset (frozen in) in an ice cover.

The breakout ("escape") force is the force the propulsion plant must supply
to shear the adfreeze joint and overcome hull/ice friction:

    F_breakout = tau_residual * A_contact + mu * p_n * A_contact

with

``A_contact``
    Effective ice/hull contact area.  Computed by integrating the hull's
    waterline geometry: for every station along the waterline the ice sheet
    of thickness ``h_i`` bears on a vertical band whose height is limited by
    the local hull slope (flare) - a strongly flared or rounded bow cannot
    hold a tall frozen band, so the band height is
    ``h_band = min(h_i, gap_limit/tan(flare))``.  The band is taken on both
    sides of the vessel.
``p_n``
    Mean normal pressure of the ice on the hull, which drives the Coulomb
    friction term [Pa].
``mu``
    Ice/steel friction coefficient (wet, near 0 degC: 0.05-0.2).

The available thrust is computed with propeller momentum theory in the
bollard condition (``J = 0``)::

    P_shaft = 2*pi*n*Q,  Q = C_Q*rho*n**2*D**5  ->  n
    T       = C_T*rho*n**2*D**4

with ice-class propeller coefficients ``C_T``, ``C_Q`` (nozzle, bollard).

The engineering output is the **critical contact-area fraction**: the share
of the frozen contact that must be debonded before thrust suffices,

    A_crit/A_contact = T_available / F_breakout(full contact)

which is the physically meaningful escape criterion for a full-length frozen
joint (a pure pull-out with the whole joint intact is normally impossible;
escape requires rocking/heeling to peel the joint progressively).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .adfreeze import GRAVITY, RHO_ICE, adfreeze_strength

RHO_SEA_WATER = 1025.0

#: Ice/steel friction coefficient, wet contact near the freezing point.
DEFAULT_FRICTION_COEFF = 0.12
#: Mean normal pressure of the ice cover on the hull in the beset condition
#: [Pa] - engineering estimate (10-50 kPa for a beset vessel; design ice
#: pressures used for structural checks are far higher and are not relevant
#: for beset friction).
DEFAULT_NORMAL_PRESSURE_PA = 3.0e4
#: Joint clearance tolerance [m]: how far the ice edge can be pulled away from
#: the hull before contact is lost.  Physically this is the thickness of the
#: crushed/damaged ice layer that forms at a ice-steel waterline contact
#: (ISO 19906 high-pressure zone, 50-200 mm), not a steel-surface roughness.
#: It controls how tall a frozen band a flared hull can hold.
DEFAULT_CLEARANCE_M = 1.0e-1


@dataclass(frozen=True)
class ShipDefinition:
    """Minimal hull/propulsion description used by the breakout model."""

    name: str
    length_pp_m: float
    beam_m: float
    draft_m: float
    ice_thickness_m: float
    beset_length_fraction: float
    bow_entrance_half_angle_deg: float
    midbody_flare_deg: float
    shaft_power_kw: float
    propeller_diameter_m: float
    propeller_count: int = 2
    c_t_bollard: float = 0.45
    c_q_bollard: float = 0.060
    provenance: str = "engineering_estimate"


@dataclass(frozen=True)
class BreakoutResult:
    """Breakout force and thrust comparison for one vessel/ice state."""

    ship: str
    contact_area_m2: float
    tau_adfreeze_mpa: float
    adhesion_force_mn: float
    friction_force_mn: float
    breakout_force_mn: float
    thrust_mn: float
    thrust_minus_friction_mn: float
    critical_area_fraction: float
    margin: float

    def as_dict(self) -> dict:
        return {
            "ship": self.ship,
            "contact_area_m2": self.contact_area_m2,
            "tau_adfreeze_MPa": self.tau_adfreeze_mpa,
            "adhesion_force_MN": self.adhesion_force_mn,
            "friction_force_MN": self.friction_force_mn,
            "breakout_force_MN": self.breakout_force_mn,
            "thrust_MN": self.thrust_mn,
            "thrust_minus_friction_MN": self.thrust_minus_friction_mn,
            "critical_area_fraction": self.critical_area_fraction,
            "margin_thrust_over_breakout": self.margin,
        }


# ---------------------------------------------------------------------------
# Hull / ice contact area
# ---------------------------------------------------------------------------
def waterline_half_breadth(ship: ShipDefinition, x_m: float) -> float:
    """Waterline half-breadth [m] at station ``x_m`` from midship.

    Parallel midbody over the central 50 % of ``L_pp``, cosine entrance in the
    bow and a shorter run in the stern.  This is a schematic hull form used
    for the geometric integration, not a measured lines plan.
    """
    half_l = 0.5 * ship.length_pp_m
    x = float(x_m)
    b2 = 0.5 * ship.beam_m
    if abs(x) <= 0.25 * ship.length_pp_m:
        return b2
    if x > 0.0:  # bow: taper to the stem over 25 % L
        s = (x - 0.25 * ship.length_pp_m) / (0.75 * half_l)
        return b2 * float(np.cos(0.5 * np.pi * min(s, 1.0)))
    # stern: taper over 15 % L
    s = (-x - 0.25 * ship.length_pp_m) / (0.25 * half_l)
    return b2 * (1.0 - 0.55 * min(s, 1.0) ** 2)


def contact_band_height(
    ship: ShipDefinition,
    ice_thickness_m: float,
    flare_deg: float,
    clearance_m: float = DEFAULT_CLEARANCE_M,
) -> float:
    """Frozen band height [m] on the hull at a station with a given flare.

    The ice edge cannot follow a flared hull: the horizontal gap opens by
    ``z*tan(flare)`` away from the waterline, i.e. by half of that on each side
    of the ice sheet.  Once the gap exceeds the joint clearance the ice no
    longer bears on the hull, hence
    ``h_band = min(h_i, 2*clearance/tan(flare))`` (vertical sides keep
    ``h_i``).
    """
    if ice_thickness_m <= 0.0:
        raise ValueError("ice thickness must be positive")
    if flare_deg < 0.0:
        raise ValueError("flare must be non-negative")
    tan_f = np.tan(np.deg2rad(flare_deg))
    if tan_f <= 1e-9:
        return float(ice_thickness_m)
    return float(min(ice_thickness_m, 2.0 * clearance_m / tan_f))


def hull_ice_contact_area(
    ship: ShipDefinition,
    ice_thickness_m: float | None = None,
    contact_conformance: float = 1.0,
    n_stations: int = 401,
    clearance_m: float = DEFAULT_CLEARANCE_M,
) -> float:
    """Effective ice/hull contact area [m^2] (both sides), by integration.

    ``contact_conformance`` scales the *geometric* area to the fraction of
    the hull surface that is actually in tight ice contact (asperity contact,
    air pockets); it is an engineering closure and is reported explicitly.
    """
    h_i = ship.ice_thickness_m if ice_thickness_m is None else float(ice_thickness_m)
    if not 0.0 < contact_conformance <= 1.0:
        raise ValueError("contact_conformance must lie in (0, 1]")
    half_window = 0.5 * ship.length_pp_m * ship.beset_length_fraction
    xs = np.linspace(-half_window, half_window, n_stations)
    ds = xs[1] - xs[0]
    mid_x = 0.25 * ship.length_pp_m
    total = 0.0
    for x in xs:
        # flare: 0 at the parallel midbody, opening up towards the bow
        if abs(x) <= mid_x:
            flare = ship.midbody_flare_deg
        else:
            bow = x > 0.0
            span = (half_window - mid_x) if half_window > mid_x else 1.0
            frac = min((abs(x) - mid_x) / max(span, 1e-9), 1.0)
            base = ship.bow_entrance_half_angle_deg if bow else 0.5 * ship.bow_entrance_half_angle_deg
            flare = ship.midbody_flare_deg + frac * base
        total += contact_band_height(ship, h_i, flare, clearance_m) * ds
    return float(2.0 * total * contact_conformance)


def empirical_contact_area(ship: ShipDefinition, ice_thickness_m: float | None = None) -> float:
    """Closed-form (rectangular) contact-area estimate [m^2].

    Engineering practice: the beset contact is a rectangle of length
    ``beset_length_fraction*L`` and height ``h_i`` on both sides.  This
    closed form is *independent of the hull-form integration* in
    :func:`hull_ice_contact_area` and is used as the cross-check reference
    in the validation suite.
    """
    h_i = ship.ice_thickness_m if ice_thickness_m is None else float(ice_thickness_m)
    return float(
        2.0
        * ship.beset_length_fraction
        * ship.length_pp_m
        * h_i
    )


# ---------------------------------------------------------------------------
# Friction, thrust, breakout force
# ---------------------------------------------------------------------------
def friction_force_mn(
    contact_area_m2: float,
    mu: float = DEFAULT_FRICTION_COEFF,
    normal_pressure_pa: float = DEFAULT_NORMAL_PRESSURE_PA,
) -> float:
    """Coulomb friction force [MN] on the ice/hull contact."""
    if contact_area_m2 < 0.0:
        raise ValueError("contact area must be non-negative")
    if mu < 0.0:
        raise ValueError("friction coefficient must be non-negative")
    return float(mu * normal_pressure_pa * contact_area_m2 / 1.0e6)


def bollard_thrust_mn(
    shaft_power_kw: float,
    propeller_diameter_m: float,
    propeller_count: int = 1,
    c_t: float = 0.45,
    c_q: float = 0.060,
    rho_kg_m3: float = RHO_SEA_WATER,
) -> float:
    """Bollard-pull thrust [MN] from propeller momentum theory (``J = 0``).

    ``P/N = 2*pi*n*Q`` with ``Q = C_Q*rho*n**2*D**5`` gives
    ``n = (P/(N*2*pi*C_Q*rho*D**5))**(1/3)`` and ``T = N*C_T*rho*n**2*D**4``.
    """
    if shaft_power_kw <= 0.0:
        raise ValueError("shaft power must be positive")
    if propeller_diameter_m <= 0.0:
        raise ValueError("propeller diameter must be positive")
    if propeller_count < 1:
        raise ValueError("propeller count must be at least one")
    power_per_prop = shaft_power_kw * 1.0e3 / propeller_count
    n = (power_per_prop / (2.0 * np.pi * c_q * rho_kg_m3 * propeller_diameter_m**5)) ** (1.0 / 3.0)
    t_prop = c_t * rho_kg_m3 * n**2 * propeller_diameter_m**4
    return float(propeller_count * t_prop / 1.0e6)


def breakout_force_mn(
    contact_area_m2: float,
    tau_adfreeze_mpa: float,
    mu: float = DEFAULT_FRICTION_COEFF,
    normal_pressure_pa: float = DEFAULT_NORMAL_PRESSURE_PA,
) -> tuple[float, float, float]:
    """``(breakout, adhesion, friction)`` forces in MN."""
    if tau_adfreeze_mpa < 0.0:
        raise ValueError("adfreeze strength must be non-negative")
    adhesion = tau_adfreeze_mpa * contact_area_m2  # MPa*m^2 = MN
    friction = friction_force_mn(contact_area_m2, mu, normal_pressure_pa)
    return float(adhesion + friction), float(adhesion), float(friction)


def evaluate_breakout(
    ship: ShipDefinition,
    ice_temperature_c: float,
    salinity_gkg: float = 0.0,
    contact_time_s: float = 1.0e4,
    roughness_um: float = 2.5,
    ice_thickness_m: float | None = None,
    contact_conformance: float = 1.0,
    mu: float = DEFAULT_FRICTION_COEFF,
    normal_pressure_pa: float = DEFAULT_NORMAL_PRESSURE_PA,
    clearance_m: float = DEFAULT_CLEARANCE_M,
) -> BreakoutResult:
    """Full breakout evaluation for one vessel in one ice state."""
    h_i = ship.ice_thickness_m if ice_thickness_m is None else float(ice_thickness_m)
    area = hull_ice_contact_area(ship, h_i, contact_conformance, clearance_m=clearance_m)
    tau = adfreeze_strength(ice_temperature_c, contact_time_s, salinity_gkg, roughness_um)
    f_bo, f_adh, f_fric = breakout_force_mn(area, tau, mu, normal_pressure_pa)
    thrust = bollard_thrust_mn(
        ship.shaft_power_kw,
        ship.propeller_diameter_m,
        ship.propeller_count,
        ship.c_t_bollard,
        ship.c_q_bollard,
    )
    net_thrust = thrust - f_fric
    if f_adh > 0.0:
        crit = float(max(0.0, min(1.0, net_thrust / f_adh)))
    else:
        crit = 1.0
    margin = float(thrust / f_bo) if f_bo > 0.0 else float("inf")
    return BreakoutResult(
        ship=ship.name,
        contact_area_m2=area,
        tau_adfreeze_mpa=tau,
        adhesion_force_mn=f_adh,
        friction_force_mn=f_fric,
        breakout_force_mn=f_bo,
        thrust_mn=thrust,
        thrust_minus_friction_mn=net_thrust,
        critical_area_fraction=crit,
        margin=margin,
    )


def required_debonded_area(
    net_thrust_mn: float,
    tau_adfreeze_mpa: float,
    contact_area_m2: float,
) -> float:
    """Adhesion area [m^2] that must be peeled so that thrust suffices.

    ``A_peel = A_contact - net_thrust/tau`` (clipped at 0).  This is the
    actionable form of the escape criterion: it tells the operator how much
    of the frozen joint must be broken before the vessel can move.
    """
    if tau_adfreeze_mpa <= 0.0:
        return 0.0
    remaining = net_thrust_mn / tau_adfreeze_mpa
    return float(max(0.0, contact_area_m2 - remaining))