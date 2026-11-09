"""
V&V (Verification & Validation) Machine Gate for TensorPhysics.

Provides Richardson extrapolation with GCI, ASME V&V 20 validation uncertainty,
and cross-solver comparison adapters.

This module provides two API layers:
1. High-level classes (RichardsonExtrapolation, ASMEVV20, CrossSolverAdapter) for ease of use
2. Low-level functions for integration with existing test infrastructure
"""

# High-level API (new implementation)
from .richardson import RichardsonExtrapolation, GCIResult, ConvergenceStatus
from .asme_vv20 import ASMEVV20, UncertaintyBudget
from .cross_solver import (
    CrossSolverAdapter,
    ComparisonResult,
    ComparisonStatus,
    CanonicalCase,
    create_boundary_layer_case,
    create_cylinder_drag_case,
    create_lid_driven_cavity_case,
)

# Low-level API (for compatibility with existing tests)
from .validation import (
    validate,
    validation_uncertainty,
    evaluate_validation_table,
)

__all__ = [
    # High-level API
    'RichardsonExtrapolation',
    'GCIResult',
    'ConvergenceStatus',
    'ASMEVV20',
    'UncertaintyBudget',
    'CrossSolverAdapter',
    'ComparisonResult',
    'ComparisonStatus',
    'CanonicalCase',
    'create_boundary_layer_case',
    'create_cylinder_drag_case',
    'create_lid_driven_cavity_case',
    # Low-level API
    'validate',
    'validation_uncertainty',
    'evaluate_validation_table',
]
