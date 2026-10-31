"""Ice sheet mechanics module for TensorPhysics.

Provides Winkler elastic foundation plate models for floating ice sheets,
including analytical (Navier/Hetényi-type) and finite-difference solvers,
plus water added-mass integration for fluid–structure interaction.

Physical domain: ice_mechanics / fluid_structure_interaction
Campaign: SIR-PC2 | F01 → F02
"""

from .winkler_plate import (
    WinklerPlateParams,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
    winkler_kw_parametric,
)
from .added_mass import (
    water_added_mass_per_area,
    IceSheetFSIParams,
    fem_mass_matrix_with_added_mass,
    natural_frequency_plate_on_foundation,
    natural_frequency_fluid_solid_reference,
)

__all__ = [
    "WinklerPlateParams",
    "navier_center_deflection",
    "navier_deflection_field",
    "solve_winkler_plate_fd",
    "winkler_kw_parametric",
    "water_added_mass_per_area",
    "IceSheetFSIParams",
    "fem_mass_matrix_with_added_mass",
    "natural_frequency_plate_on_foundation",
    "natural_frequency_fluid_solid_reference",
]
