"""Ice sheet mechanics module for TensorPhysics.

Provides Winkler elastic foundation plate models for floating ice sheets,
including analytical (Navier/Hetényi-type) and finite-difference solvers.

Physical domain: ice_mechanics
Campaign: SIR-PC2 | F01 | Winkler elastic foundation plate
"""

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
]
