"""TensorFEM Phase-Field Fracture Domain Module (N05)

Phase-field modeling of brittle fracture for structural integrity assessment.
Computes crack paths and crack length evolution for coupling with structural
integrity evaluation modules.

Physics:
- Variational fracture mechanics (Bourdin et al. 2000)
- Phase-field variable φ ∈ [0,1]: φ=0 intact, φ=1 fully broken
- Energy functional: E = ∫ [g(φ)ψ⁺(ε) + ψ⁻(ε)] dV + G_c ∫ [φ²/(2l) + l|∇φ|²/2] dV
- Degradation function: g(φ) = (1-φ)² + k_residual
- Crack length tracking: a(t) = ∫_{crack_path} φ ds

Interface contract:
- Outputs crack length a(t) at specified time points
- Units: meters
- Temporal resolution: quasi-static load steps
- Spatial resolution: crack path in 2D/3D

Evidence level: REAL (validated against classical fracture benchmarks)

References:
- Bourdin, Francfort, Marigo (2000). Numerical experiments in revisited brittle fracture.
- Miehe, Hofacker, Welschinger (2010). A phase field model for rate-independent crack propagation.
- Ambati et al. (2015). A review on phase-field models of brittle fracture.
"""

from .fracture_solver import (
    PhaseFieldFractureSolver,
    FractureResult,
    CrackPath,
    compute_crack_length,
    energy_release_rate,
    stress_intensity_factor,
)
from .interface import (
    FEMToStructuralInterface,
    FractureInterfaceContract,
    validate_crack_interface_consistency,
)

__all__ = [
    "PhaseFieldFractureSolver",
    "FractureResult",
    "CrackPath",
    "compute_crack_length",
    "energy_release_rate",
    "stress_intensity_factor",
    "FEMToStructuralInterface",
    "FractureInterfaceContract",
    "validate_crack_interface_consistency",
]

__evidence_level__ = "REAL"
__evidence_trace__ = {
    "validation": "mode_I_benchmark_griffith_crack",
    "convergence": "mesh_refinement_crack_length",
    "interface_contract": "v1.0_fem_to_structural",
}
