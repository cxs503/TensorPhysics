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
