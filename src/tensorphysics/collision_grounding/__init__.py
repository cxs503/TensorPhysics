"""
Collision and Grounding Elastic-Plastic FEM Module

This module provides elastic-plastic analysis capabilities for ship collision
and grounding scenarios, including:
- Material models: von Mises yield with isotropic hardening
- Strain rate effects: Cowper-Symonds model
- Contact algorithms: master-slave contact with penalty method
- Geometric nonlinearity: Green-Lagrange strain, 2nd Piola-Kirchhoff stress
- Arc-length method for post-buckling analysis
- Verification cases: double-side structure crushing
- ISO18072 damage length criterion
"""

from .material_models import (
    ElasticPlasticMaterial,
    CowperSymondsModel,
    von_mises_yield,
    isotropic_hardening,
)
from .contact import (
    MasterSlaveContact,
    PenaltyContactMethod,
    hertz_contact_pressure,
)
from .geometric_nonlinear import (
    GreenLagrangeStrain,
    SecondPiolaKirchhoffStress,
    LargeDeformationSolver,
)
from .solvers import (
    ArcLengthMethod,
    NewtonRaphsonSolver,
)
from .verification import (
    SimplySupportedPlateUltimate,
    CylindricalShellCrushing,
    DoubleSideStructureCrushing,
)
from .iso18072 import (
    ISO18072DamageLength,
    calculate_damage_length,
    safety_margin,
)

__all__ = [
    # Material models
    "ElasticPlasticMaterial",
    "CowperSymondsModel",
    "von_mises_yield",
    "isotropic_hardening",
    # Contact
    "MasterSlaveContact",
    "PenaltyContactMethod",
    "hertz_contact_pressure",
    # Geometric nonlinearity
    "GreenLagrangeStrain",
    "SecondPiolaKirchhoffStress",
    "LargeDeformationSolver",
    # Solvers
    "ArcLengthMethod",
    "NewtonRaphsonSolver",
    # Verification
    "SimplySupportedPlateUltimate",
    "CylindricalShellCrushing",
    "DoubleSideStructureCrushing",
    # ISO18072
    "ISO18072DamageLength",
    "calculate_damage_length",
    "safety_margin",
]
