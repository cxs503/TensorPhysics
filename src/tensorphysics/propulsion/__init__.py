"""
Propulsion module for TensorPhysics.

Contains models for marine propulsion systems including:
- Propeller-ice interaction and torque ripple analysis
- Open-water propeller characteristics
- Ice-induced load fluctuations
"""

from .ice_torque import (
    PropellerIceInteraction,
    validate_torque_ripple_analysis,
    get_reference_values,
    run_complete_analysis,
)

__all__ = [
    'PropellerIceInteraction',
    'validate_torque_ripple_analysis',
    'get_reference_values',
    'run_complete_analysis',
]
