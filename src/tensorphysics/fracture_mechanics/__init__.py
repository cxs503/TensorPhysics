"""Fracture mechanics module for cohesive element validation.

This module implements:
- Benzeggagh-Kenane (BK) mixed-mode fracture criterion
- Cohesive zone model for mixed-mode loading
- Energy release rate computation
- Validation utilities for non-proportional loading paths
"""

from .bk_law import BKLaw, compute_mixed_mode_ratio, compute_effective_fracture_toughness
from .cohesive import CohesiveElement, MixedModeCohesiveModel
from .validation import MixedModeValidator, ValidationReport

__all__ = [
    "BKLaw",
    "compute_mixed_mode_ratio",
    "compute_effective_fracture_toughness",
    "CohesiveElement",
    "MixedModeCohesiveModel",
    "MixedModeValidator",
    "ValidationReport",
]
