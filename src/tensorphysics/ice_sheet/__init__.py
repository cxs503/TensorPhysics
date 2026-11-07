"""Ice sheet mechanics module for TensorPhysics.

Provides Winkler elastic foundation plate models for floating ice sheets:
analytical (Navier/Hetenyi-type) and finite-difference solvers, plus a
conforming Kirchhoff thin-plate finite-element solver (Bogner-Fox-Schmit)
used for level-ice bending-failure analysis under ship-bow indentation loads.

Physical domain: ice_mechanics
Campaigns: SIR-PC2 | F01 | Winkler elastic foundation plate
           IN11-H3 | level-ice bending failure, Kirchhoff plate FEM
"""

from .kirchhoff_plate_fem import (
    BFSPlateSolver,
    KirchhoffWinklerPlate,
    PlateField,
    bending_stress,
    critical_point_load,
    fan_sector_area,
    fan_sector_mask,
    graded_axis,
    graded_axis_centered,
    kelvin_point_load,
    kelvin_point_load_moments,
    limiting_moment,
)
from .winkler_plate import (
    WinklerPlateParams,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
    winkler_kw_parametric,
)

__all__ = [
    "WinklerPlateParams",
    "navier_center_deflection",
    "navier_deflection_field",
    "solve_winkler_plate_fd",
    "winkler_kw_parametric",
    "BFSPlateSolver",
    "KirchhoffWinklerPlate",
    "PlateField",
    "bending_stress",
    "critical_point_load",
    "fan_sector_area",
    "fan_sector_mask",
    "graded_axis",
    "graded_axis_centered",
    "kelvin_point_load",
    "kelvin_point_load_moments",
    "limiting_moment",
]
