"""Underwater attached-ice module for TensorPhysics.

Models the growth and detachment of ice accreted on submerged hull surfaces
from frazil-laden flow, and maps the resulting ice layer to hydrodynamic
roughness and resistance penalties.

Sub-modules
-----------
flux      : frazil transport & impact flux to the wall
growth    : interface energy-balance ODE for ice thickness h_ice(t)
detach    : shear-stress detachment threshold
roughness : equivalent sand-grain roughness k_s(t) and ITTC-1978 penalty

Physical constants
------------------
All constants are exported here for cross-module consistency.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared physical constants (fresh water / ice / seawater)
# ---------------------------------------------------------------------------
RHO_ICE = 917.0          # kg/m^3, pure ice
RHO_WATER = 1025.0       # kg/m^3, seawater
RHO_WATER_FW = 1000.0    # kg/m^3, freshwater
K_ICE = 2.2              # W/(m*K), thermal conductivity of ice
K_WATER = 0.56           # W/(m*K), thermal conductivity of water (~5 degC)
L_FUSION = 3.34e5        # J/kg, latent heat of fusion of ice
C_P_WATER = 4186.0       # J/(kg*K), specific heat of water
MU_WATER = 1.519e-3      # Pa*s, dynamic viscosity of seawater (~0 degC)
T_FREEZE_FW = 273.15     # K, freezing point of freshwater
T_FREEZE_SW = 271.35     # K, freezing point of seawater (~ -1.8 degC)
GRAVITY = 9.81           # m/s^2

# ITTC-1978 roughness allowance coefficient
ITTC_DELTA_CF_ALPHA = 0.004  # dimensionless, per ITTC-1978 recommendation

# Default frazil particle diameter (Stokes regime)
DEFAULT_FRAZIL_DIAMETER_M = 1.0e-4  # 0.1 mm

# Default sticking efficiency (fraction of impacting frazil that adheres)
DEFAULT_STICKING_EFFICIENCY = 1.0

# Default surface roughness of clean hull (painted steel)
DEFAULT_HULL_ROUGHNESS_M = 1.5e-5  # 15 um

# Reference length for ITTC-1978 (ship length)
DEFAULT_SHIP_LENGTH_M = 150.0

# Default bond strength from Frederking 1984 (freshwater, -5 degC) [Pa]
DEFAULT_BOND_STRENGTH_PA = 500e3  # 0.5 MPa

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from .flux import (
    frazil_impact_flux,
    deposition_velocity,
    settling_velocity_stokes,
    FrazilFluxState,
)
from .growth import (
    IceGrowthODE,
    solve_growth,
    stefan_analytic,
    energy_residual,
    GrowthResult,
)
from .detach import (
    wall_shear_stress,
    adhesive_strength,
    detachment_threshold_velocity,
    detachment_state,
    DetachmentState,
)
from .roughness import (
    equivalent_sand_roughness,
    ittc_1978_roughness_penalty,
    ittc_1978_thrust_loss,
    roughness_state,
    p50_p90_envelope,
    RoughnessState,
)

__all__ = [
    # constants
    "RHO_ICE", "RHO_WATER", "RHO_WATER_FW",
    "K_ICE", "K_WATER", "L_FUSION", "C_P_WATER", "MU_WATER",
    "T_FREEZE_FW", "T_FREEZE_SW", "GRAVITY",
    "ITTC_DELTA_CF_ALPHA", "DEFAULT_FRAZIL_DIAMETER_M",
    "DEFAULT_STICKING_EFFICIENCY", "DEFAULT_HULL_ROUGHNESS_M",
    "DEFAULT_SHIP_LENGTH_M", "DEFAULT_BOND_STRENGTH_PA",
    # flux
    "frazil_impact_flux", "deposition_velocity", "settling_velocity_stokes",
    "FrazilFluxState",
    # growth
    "IceGrowthODE", "solve_growth", "stefan_analytic", "energy_residual",
    "GrowthResult",
    # detach
    "wall_shear_stress", "adhesive_strength", "detachment_threshold_velocity",
    "detachment_state", "DetachmentState",
    # roughness
    "equivalent_sand_roughness", "ittc_1978_roughness_penalty",
    "ittc_1978_thrust_loss", "roughness_state", "p50_p90_envelope",
    "RoughnessState",
]
