"""Ice Sheet Thermal Stress and Thermal Cracking Module.

Implements the physics of thermal stress development and crack initiation
in ice sheets due to temperature gradients, providing the physical basis
for the floe size distribution (f_FSD).

Physical Model:
  - 1D transient heat conduction in ice sheet
  - Thermal stress from constrained thermal contraction
  - Crack initiation when thermal stress exceeds tensile strength
  - Crack spacing scaling law based on energy balance

Key Physics:
  When an ice sheet cools from the surface, a temperature gradient develops.
  The colder surface layer wants to contract but is constrained by the warmer
  interior, generating tensile thermal stress. When this stress exceeds the
  tensile strength, thermal cracks form. The spacing between cracks follows
  a scaling law related to ice thickness and material properties.

References:
  - Schulson, E.M. (2001). "Brittle failure of ice." Eng. Fract. Mech.
  - Mellor, M. (1986). "Mechanical behavior of ice." CRREL Monograph.
  - van der Veen, C.J. (1998). "Fracture surface topography of ice."
  - Weiss, J. & Schulson, E.M. (1995). "Ice cracking and spalling."

Author: TensorPhysics Team
Date: 2026-11-11
"""

from .thermal_conduction import (
    IceThermalConduction,
    transient_temperature_profile,
    cooling_rate,
)
from .thermal_stress import (
    ThermalStressSolver,
    thermal_stress_profile,
    max_thermal_stress,
    stress_intensity_factor,
)
from .crack_initiation import (
    CrackInitiationSolver,
    crack_initiation_time,
    critical_temperature_difference,
)
from .crack_spacing import (
    CrackSpacingModel,
    crack_spacing_scaling,
    crack_spacing_distribution,
)
from .fsd_interface import (
    FSDInterface,
    initial_crack_spacing_to_fsd,
    thermal_fsd_parameters,
)

__all__ = [
    # Thermal conduction
    "IceThermalConduction",
    "transient_temperature_profile",
    "cooling_rate",
    # Thermal stress
    "ThermalStressSolver",
    "thermal_stress_profile",
    "max_thermal_stress",
    "stress_intensity_factor",
    # Crack initiation
    "CrackInitiationSolver",
    "crack_initiation_time",
    "critical_temperature_difference",
    # Crack spacing
    "CrackSpacingModel",
    "crack_spacing_scaling",
    "crack_spacing_distribution",
    # FSD interface
    "FSDInterface",
    "initial_crack_spacing_to_fsd",
    "thermal_fsd_parameters",
]
