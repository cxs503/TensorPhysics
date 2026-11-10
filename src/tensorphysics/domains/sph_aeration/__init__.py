"""TensorSPH Aeration Domain Module (N04)

SPH-based simulation of air entrainment in free-surface flows.
Computes void fraction fields for coupling with resistance correction modules.

Physics:
- Two-phase SPH with air-water density ratio
- Void fraction transport: dα/dt + ∇·(αu) = S_entrainment
- Entrainment source model based on turbulent kinetic energy threshold
- Output: spatially-resolved void fraction α(x,t)

Interface contract:
- Outputs mean void fraction α_mean at hull surface patches
- Units: dimensionless [0,1]
- Temporal resolution: steady-state or time-averaged
- Spatial resolution: patch-averaged over hull panels

Evidence level: REAL (validated against empirical entrainment correlations)

References:
- Monaghan (1994). SPH tutorial review.
- Marrone et al. (2011). Free-surface flows with SPH.
- Liu & Liu (2010). Smoothed Particle Hydrodynamics.
"""

from .aeration_solver import (
    SPHAerationSolver,
    AerationResult,
    VoidFractionField,
    compute_void_fraction_profile,
    estimate_entrainment_rate,
    aeration_resistance_correction,
)
from .interface import (
    SPHToResistanceInterface,
    AerationInterfaceContract,
    validate_interface_consistency,
)

__all__ = [
    "SPHAerationSolver",
    "AerationResult",
    "VoidFractionField",
    "compute_void_fraction_profile",
    "estimate_entrainment_rate",
    "aeration_resistance_correction",
    "SPHToResistanceInterface",
    "AerationInterfaceContract",
    "validate_interface_consistency",
]

__evidence_level__ = "REAL"
__evidence_trace__ = {
    "validation": "empirical_entrainment_correlation_breaking_2018",
    "convergence": "sph_resolution_study_void_fraction",
    "interface_contract": "v1.0_sph_to_resistance",
}
