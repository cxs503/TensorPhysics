"""Fluid-Structure Interaction (FSI) module for cross-solver verification.

This module implements FSI coupling between TensorLBM (fluid) and TensorFEM (structure)
for 2D airfoil flutter verification benchmarks.

Key components:
- airfoil_flutter: 2D airfoil flutter physics (Theodorsen-type unsteady aerodynamics)
- coupling_schemes: Loose and tight coupling algorithms
- conservation_gates: Momentum and energy conservation checks
- lbm_adapter: TensorLBM surface force extraction
- fem_adapter: TensorFEM structural response computation
"""

from .airfoil_flutter import (
    AirfoilFlutter,
    FlutterConfig,
    FlutterResult,
    theodorsen_function,
    compute_flutter_speed,
)
from .coupling_schemes import (
    LooseCoupling,
    TightCoupling,
    CouplingResult,
)
from .conservation_gates import (
    MomentumGate,
    EnergyGate,
    ConservationCheck,
)
from .lbm_adapter import LBMForceAdapter
from .fem_adapter import FEMStructuralAdapter

__all__ = [
    "AirfoilFlutter",
    "FlutterConfig",
    "FlutterResult",
    "theodorsen_function",
    "compute_flutter_speed",
    "LooseCoupling",
    "TightCoupling",
    "CouplingResult",
    "MomentumGate",
    "EnergyGate",
    "ConservationCheck",
    "LBMForceAdapter",
    "FEMStructuralAdapter",
]
