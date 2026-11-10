"""Shear-stress detachment threshold for underwater attached ice.

Physical model
==============
Ice accreted on a submerged surface is subject to hydrodynamic shear stress
from the boundary layer flow. Detachment occurs when the wall shear stress
exceeds the adhesive strength of the ice-surface bond.

Wall shear stress
-----------------
For a turbulent boundary layer, the wall shear stress is::

    tau_wall = rho_water * u_star^2

where ``u_star`` is the friction velocity.

Adhesive strength
-----------------
The adhesive strength of the ice-surface bond depends on:

1. **Bond strength** (from Frederking 1984 adfreeze model): the intrinsic
   shear strength of the ice-steel interface, which depends on temperature
   and surface conditions.
2. **Ice thickness**: thicker ice has more area for stress concentration
   and may have internal weaknesses. We model this as a thickness-dependent
   reduction factor.

The adhesive strength is modelled as::

    tau_adhesive = tau_bond * f_thickness(h_ice)

where ``tau_bond`` is the bond strength from the adfreeze model and
``f_thickness`` is a reduction factor that decreases with ice thickness::

    f_thickness(h) = 1 / (1 + h / h_ref)

with ``h_ref`` a reference thickness (~10 mm) representing the scale at
which internal ice weaknesses become significant.

Detachment threshold velocity
-----------------------------
The critical friction velocity for detachment is::

    u_star_crit = sqrt(tau_adhesive / rho_water)

When the actual friction velocity exceeds this threshold, the ice layer
is stripped from the surface.

Monotonicity property (C4)
--------------------------
For a fixed set of conditions, increasing the flow velocity (and hence
u_star) should monotonically decrease the steady-state ice thickness,
because higher shear leads to earlier detachment. This is verified by
property-based testing with 200 sample points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import RHO_WATER, RHO_ICE, K_ICE, L_FUSION, T_FREEZE_FW

# Reference thickness for adhesive strength reduction [m]
H_REF_M = 0.010  # 10 mm

# Default bond strength from Frederking 1984 (freshwater, -5 degC) [Pa]
DEFAULT_BOND_STRENGTH_PA = 500e3  # 0.5 MPa


@dataclass(frozen=True)
class DetachmentState:
    """State of the ice detachment analysis."""

    #: Wall shear stress [Pa]
    tau_wall: float
    #: Adhesive strength [Pa]
    tau_adhesive: float
    #: Ice thickness [m]
    h_ice: float
    #: Bond strength [Pa]
    tau_bond: float
    #: Thickness reduction factor (dimensionless)
    f_thickness: float
    #: Whether detachment has occurred
    is_detached: bool
    #: Safety factor (tau_adhesive / tau_wall)
    safety_factor: float

    def as_dict(self) -> dict:
        return {
            "tau_wall_Pa": self.tau_wall,
            "tau_adhesive_Pa": self.tau_adhesive,
            "h_ice_m": self.h_ice,
            "tau_bond_Pa": self.tau_bond,
            "f_thickness": self.f_thickness,
            "is_detached": self.is_detached,
            "safety_factor": self.safety_factor,
        }


def wall_shear_stress(
    u_star: float,
    rho_water: float = RHO_WATER,
) -> float:
    """Wall shear stress from friction velocity [Pa].

    Parameters
    ----------
    u_star : float
        Friction velocity [m/s].
    rho_water : float
        Water density [kg/m^3].

    Returns
    -------
    float
        Wall shear stress [Pa].
    """
    u_s = float(u_star)
    if u_s < 0.0:
        raise ValueError("friction velocity must be non-negative")
    return float(rho_water * u_s**2)


def adhesive_strength(
    h_ice: float,
    tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    h_ref: float = H_REF_M,
) -> float:
    """Adhesive strength of the ice-surface bond [Pa].

    Parameters
    ----------
    h_ice : float
        Ice thickness [m].
    tau_bond : float
        Intrinsic bond strength [Pa] (from adfreeze model).
    h_ref : float
        Reference thickness for reduction [m].

    Returns
    -------
    float
        Effective adhesive strength [Pa].
    """
    h = float(h_ice)
    if h < 0.0:
        raise ValueError("ice thickness must be non-negative")
    if tau_bond < 0.0:
        raise ValueError("bond strength must be non-negative")

    f_thick = 1.0 / (1.0 + h / h_ref)
    return float(tau_bond * f_thick)


def detachment_threshold_velocity(
    tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    h_ice: float = 0.0,
    rho_water: float = RHO_WATER,
    h_ref: float = H_REF_M,
) -> float:
    """Critical friction velocity for detachment [m/s].

    Parameters
    ----------
    tau_bond : float
        Bond strength [Pa].
    h_ice : float
        Ice thickness [m].
    rho_water : float
        Water density [kg/m^3].
    h_ref : float
        Reference thickness [m].

    Returns
    -------
    float
        Critical friction velocity [m/s]. Returns inf if no detachment
        possible (zero bond strength).
    """
    tau_adh = adhesive_strength(h_ice, tau_bond, h_ref)
    if tau_adh <= 0.0:
        return float("inf")
    return float(np.sqrt(tau_adh / rho_water))


def detachment_state(
    u_star: float,
    h_ice: float,
    tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    rho_water: float = RHO_WATER,
    h_ref: float = H_REF_M,
) -> DetachmentState:
    """Compute the full detachment state.

    Parameters
    ----------
    u_star : float
        Friction velocity [m/s].
    h_ice : float
        Ice thickness [m].
    tau_bond : float
        Bond strength [Pa].
    rho_water : float
        Water density [kg/m^3].
    h_ref : float
        Reference thickness [m].

    Returns
    -------
    DetachmentState
    """
    tau_w = wall_shear_stress(u_star, rho_water)
    tau_a = adhesive_strength(h_ice, tau_bond, h_ref)
    f_thick = 1.0 / (1.0 + float(h_ice) / h_ref)
    is_detached = tau_w > tau_a
    sf = tau_a / tau_w if tau_w > 0.0 else float("inf")

    return DetachmentState(
        tau_wall=tau_w,
        tau_adhesive=tau_a,
        h_ice=float(h_ice),
        tau_bond=float(tau_bond),
        f_thickness=f_thick,
        is_detached=is_detached,
        safety_factor=sf,
    )


def detachment_limited_thickness(
    u_star: float,
    tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    h_ref: float = H_REF_M,
    rho_water: float = RHO_WATER,
) -> float:
    """Largest thickness whose adhesive strength still carries the wall shear.

    Inversion of ``tau_bond / (1 + h/h_ref) = rho_water u_star**2``::

        h_detach = h_ref * (tau_bond / (rho_water u_star**2) - 1)

    Returns ``0.0`` when the shear already exceeds the zero-thickness bond
    strength (nothing can adhere) and ``inf`` for a stress-free wall.
    """
    u_s = float(u_star)
    if u_s < 0.0:
        raise ValueError("friction velocity must be non-negative")
    tau_wall = wall_shear_stress(u_s, rho_water)
    if tau_wall <= 0.0:
        return float("inf")
    ratio = float(tau_bond) / tau_wall
    return float(h_ref * (ratio - 1.0)) if ratio > 1.0 else 0.0


def detachment_regime(
    u_star: float,
    tau_bond: float = DEFAULT_BOND_STRENGTH_PA,
    h_ice: float = 0.0,
    h_ref: float = H_REF_M,
    rho_water: float = RHO_WATER,
    cf: float = 0.0015,
) -> dict:
    """Reachability diagnostic for shear-driven detachment.

    Shear stripping only matters when the achievable wall shear is comparable to
    the ice/steel bond strength.  With Frederking-type bond strengths
    (order 1e5 Pa) and hull-boundary-layer shear (order 1e1 Pa at
    ``u_star = 0.1 m/s``) the two differ by four orders of magnitude, so the
    stripping branch of the ODE is inert under ship-scale forcing and the
    physically active removal mechanism is melting — this helper makes that
    statement auditable instead of implicit.

    The mean-flow speed corresponding to ``u_star`` uses the standard
    friction-velocity relation ``u_star = U * sqrt(cf/2)``.

    Returns
    -------
    dict
        ``tau_wall_Pa``, ``tau_bond_Pa``, ``margin`` (= tau_bond/tau_wall),
        ``shear_detachment_active`` (margin <= 1), ``u_star_crit_m_s`` (from
        :func:`detachment_threshold_velocity`), ``speed_crit_m_s`` and
        ``speed_at_state_m_s``.
    """
    tau_wall = wall_shear_stress(u_star, rho_water)
    tau_adh = adhesive_strength(h_ice, tau_bond, h_ref)
    margin = tau_bond / tau_wall if tau_wall > 0.0 else float("inf")
    u_crit = detachment_threshold_velocity(tau_bond, h_ice, rho_water, h_ref)
    c = float(cf)
    if c <= 0.0:
        raise ValueError("friction coefficient must be positive")
    return {
        "u_star_m_s": float(u_star),
        "tau_wall_Pa": tau_wall,
        "tau_bond_Pa": float(tau_bond),
        "tau_adhesive_Pa": tau_adh,
        "margin": margin,
        "shear_detachment_active": bool(tau_wall > tau_adh),
        "u_star_crit_m_s": u_crit,
        "speed_crit_m_s": u_crit / np.sqrt(c / 2.0),
        "speed_at_state_m_s": float(u_star) / np.sqrt(c / 2.0),
    }


def friction_velocity_from_speed(speed_m_s: float, cf: float = 0.0015) -> float:
    """``u_star = U * sqrt(cf/2)`` — mean-flow speed to friction velocity."""
    U = float(speed_m_s)
    c = float(cf)
    if U < 0.0:
        raise ValueError("speed must be non-negative")
    if c <= 0.0:
        raise ValueError("friction coefficient must be positive")
    return float(U * np.sqrt(c / 2.0))


def speed_from_friction_velocity(u_star: float, cf: float = 0.0015) -> float:
    """``U = u_star / sqrt(cf/2)`` — inverse of :func:`friction_velocity_from_speed`."""
    u_s = float(u_star)
    c = float(cf)
    if u_s < 0.0:
        raise ValueError("friction velocity must be non-negative")
    if c <= 0.0:
        raise ValueError("friction coefficient must be positive")
    return float(u_s / np.sqrt(c / 2.0))


def bond_strength_from_adfreeze_pa(
    ice_temperature_c: float,
    contact_time_s: float = 1.0e4,
    salinity_gkg: float = 0.0,
    roughness_um: Optional[float] = None,
    gap_scale_m: Optional[float] = None,
) -> float:
    """Bond strength [Pa] from the Frederking (1984) adfreeze model [MPa].

    Single nerve point between the two ice modules: ``tensorphysics.ice.adfreeze``
    returns MPa, the attached-ice detachment model works in Pa.  Doing the
    conversion here (and only here) means the two modules cannot drift apart in
    units — criterion C6.

    Parameters
    ----------
    ice_temperature_c : float
        Joint / ice interface temperature [degC].
    contact_time_s : float
        Time the joint has been frozen [s].
    salinity_gkg : float
        Ice bulk salinity [g/kg], 0 for freshwater.
    roughness_um : float, optional
        Surface arithmetic mean roughness Ra [um]; ``None`` keeps the adfreeze
        module default (single source of truth — no duplicated constants here).
    gap_scale_m : float, optional
        Water-filled gap width [m]; ``None`` keeps the adfreeze default.

    Returns
    -------
    float
        Adfreeze shear strength [Pa].
    """
    from ..adfreeze import adfreeze_strength  # local import: avoids an import cycle

    from ..adfreeze import DEFAULT_GAP_SCALE_M, DEFAULT_ROUGHNESS_UM

    tau_mpa = adfreeze_strength(
        ice_temperature_c=ice_temperature_c,
        contact_time_s=contact_time_s,
        salinity_gkg=salinity_gkg,
        roughness_um=DEFAULT_ROUGHNESS_UM if roughness_um is None else roughness_um,
        gap_scale_m=DEFAULT_GAP_SCALE_M if gap_scale_m is None else gap_scale_m,
    )
    if tau_mpa < 0.0:
        raise ValueError("adfreeze model returned a negative strength")
    return float(tau_mpa * 1.0e6)
