"""Ice sheet mechanics module for TensorPhysics.

Provides analytical and numerical solutions for ice plates on elastic foundations,
validating FEM implementations against classical series solutions.
"""

from .hetenyi_plate import (
    HetenyiPlateParams,
    hetenyi_center_deflection,
    hetenyi_deflection_field,
    hetenyi_convergence_study,
)

__all__ = [
    "HetenyiPlateParams",
    "hetenyi_center_deflection",
    "hetenyi_deflection_field",
    "hetenyi_convergence_study",
]
