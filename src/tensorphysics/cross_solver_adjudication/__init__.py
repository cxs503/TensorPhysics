"""Cross-solver adjudication framework for TensorPhysics.

This module implements systematic code-to-code comparison across multiple
discretization paradigms (LBM, FVM, FEM, DG, SPH) with:

1. **Lid-driven cavity benchmark**: LBM/FVM/FEM comparison against Ghia et al.
   reference data at Re=100, 400.
2. **MMS convergence studies**: DG vs FVM on Kovasznay flow with manufactured
   solutions, observing expected convergence orders.
3. **SPH↔LBM verification**: Poiseuille flow (parabolic profile) and shallow
   water wave speed √(gh) anchors.
4. **Error attribution**: Decomposition of inter-solver discrepancies into
   truncation error (T), boundary treatment (B), and limiter/subcycling (L)
   contributions via factorial design experiments.

The framework produces machine-readable error budgets (JSON) and supports
reproducible one-click re-computation.

References
----------
- Ghia, Ghia & Shin (1982). High-Re solutions for incompressible flow using
  the Navier-Stokes equations and a multigrid method. JCP, 48, 382-411.
- Kovasznay (1948). Laminar flow behind a two-dimensional grid. Proc. Camb.
  Phil. Soc., 44, 58-62.
- Poiseuille (1840). Movement of liquids in narrow tubes.
- Ritter (1892). Die Fortpflanzung der Wasserwellen. Zeitschrift des Vereines
  deutscher Ingenieure, 36, 947-954.
"""

from .lid_cavity import (
    LidCavityLBM,
    LidCavityFVM,
    LidCavityFEM,
    GhiaReference,
    run_lid_cavity_comparison,
)
from .mms_convergence import (
    KovasznayFlow,
    DGMSolver,
    FVMSolver,
    run_mms_convergence_study,
)
from .sph_lbm_anchors import (
    PoiseuilleSPH,
    PoiseuilleLBM,
    ShallowWaterSPH,
    ShallowWaterLBM,
    run_sph_lbm_anchors,
)
from .error_attribution import (
    ErrorAttribution,
    FactorialDesign,
    attribute_residual,
    run_24_9_attribution,
)
from .adjudication import (
    CrossSolverAdjudicator,
    AdjudicationResult,
    run_full_adjudication,
)

__all__ = [
    # Lid-driven cavity
    "LidCavityLBM",
    "LidCavityFVM",
    "LidCavityFEM",
    "GhiaReference",
    "run_lid_cavity_comparison",
    # MMS convergence
    "KovasznayFlow",
    "DGMSolver",
    "FVMSolver",
    "run_mms_convergence_study",
    # SPH↔LBM anchors
    "PoiseuilleSPH",
    "PoiseuilleLBM",
    "ShallowWaterSPH",
    "ShallowWaterLBM",
    "run_sph_lbm_anchors",
    # Error attribution
    "ErrorAttribution",
    "FactorialDesign",
    "attribute_residual",
    "run_24_9_attribution",
    # Adjudication
    "CrossSolverAdjudicator",
    "AdjudicationResult",
    "run_full_adjudication",
]
