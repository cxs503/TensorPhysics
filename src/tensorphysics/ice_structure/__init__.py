"""Ice-structure partitioned coupling stability analysis.

This module provides:
- 1-DOF and 2-DOF toy models for ice-structure interaction
- Partitioned coupling schemes (loose/tight with relaxation)
- Aitken dynamic relaxation for stability enhancement
- Stability domain scanner (mu vs omega)
- Analytical spectral radius computation
- Systematic stability scan and error envelope quantification (IN66-N07)

References
----------
- Causin, P., Gerbeau, J.-F., Nobile, F. (2005). Added-mass effect in the
  design of partitioned algorithms for fluid-structure problems.
- Küttler, U., Wall, W.A. (2008). Vector and matrix relaxation techniques
  for partitioned FSI.
"""

from .models import (
    OneDOFModel,
    TwoDOFModel,
    IceStructureSystem,
)
from .coupling import (
    PartitionedCoupling,
    CouplingResult,
    CouplingScheme,
    run_partitioned_coupling,
)
from .stability import (
    StabilityAnalyzer,
    StabilityResult,
    scan_stability_domain,
    compute_analytical_spectral_radius,
    compute_critical_mu,
)
from .aitken import (
    AitkenRelaxation,
    DynamicRelaxation,
)
from .stability_scan import (
    run_stability_scan,
    StabilityScanResult,
    ScanPoint,
    ErrorEnvelopePoint,
    generate_figures,
    generate_report_json,
    run_full_campaign,
)

__all__ = [
    # Models
    "OneDOFModel",
    "TwoDOFModel",
    "IceStructureSystem",
    # Coupling
    "PartitionedCoupling",
    "CouplingResult",
    "CouplingScheme",
    "run_partitioned_coupling",
    # Stability
    "StabilityAnalyzer",
    "StabilityResult",
    "scan_stability_domain",
    "compute_analytical_spectral_radius",
    "compute_critical_mu",
    # Relaxation
    "AitkenRelaxation",
    "DynamicRelaxation",
    # Stability Scan (IN66-N07)
    "run_stability_scan",
    "StabilityScanResult",
    "ScanPoint",
    "ErrorEnvelopePoint",
    "generate_figures",
    "generate_report_json",
    "run_full_campaign",
]
