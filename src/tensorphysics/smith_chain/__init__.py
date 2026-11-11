"""Cross-scale verification chain for the incremental Smith method.

Public surface
--------------
``idealisation``  element vocabulary (``PlatePanel``, ``StiffenerGroup``,
                  ``GirderSection``) plus geometric section classification.
``laws``          plate (Faulkner/von Karman) and strut (JOH column/tripping)
                  element laws, scalar and vectorised.
``solver``        ``SmithSolver`` - incremental curvature sweep with neutral-axis
                  equilibrium (bending) or uniform shortening (panel).
``convergence``   Richardson extrapolation / grid-convergence study.
``modes``         automatic failure-mode attribution from the element states.
``benchmarks``    panel -> box girder -> double-bottom benchmark definitions and
                  their independent closed-form references.
``interface``     ``analyse_section`` / ``analyse_with_convergence`` - the
                  unified TensorPhysics entry points used by campaign nodes.

Validation chain (see ``docs/smith_chain/report.md``):
stiffened panel (element laws) -> unstiffened box girder (sectional integration,
analytic plastic limit load) -> double-bottom hull girder (scale-up, grid
convergence, failure-mode identification).
"""

from __future__ import annotations

from .benchmarks import (
    AH32,
    MILD_STEEL,
    box_girder,
    csr_min_of_modes_panel_strength,
    effective_width_column_ratio,
    effective_width_ultimate_moment,
    exact_box_plastic_modulus,
    issc2009_double_bottom,
    paik_panel,
    paik_panel_series,
    paik_thayamballi_panel_strength,
)
from .convergence import (
    RichardsonResult,
    grid_convergence_study,
    richardson_extrapolation,
)
from .idealisation import (
    STEEL,
    GirderSection,
    PlatePanel,
    StiffenerGroup,
    classify_section,
)
from .interface import SmithAnalysis, analyse_section, analyse_with_convergence
from .laws import (
    PlateLaw,
    StrutLaw,
    column_strength,
    effective_plate_buckling_stress,
    plate_buckling_stress,
    plate_slenderness,
    plate_strength,
    stiffener_strength,
)
from .modes import identify_failure_mode
from .solver import ElementModel, SmithResult, SmithSolver, smith_analysis

__all__ = [
    "STEEL",
    "AH32",
    "MILD_STEEL",
    "PlatePanel",
    "StiffenerGroup",
    "GirderSection",
    "classify_section",
    "PlateLaw",
    "StrutLaw",
    "plate_slenderness",
    "plate_buckling_stress",
    "plate_strength",
    "effective_plate_buckling_stress",
    "column_strength",
    "stiffener_strength",
    "ElementModel",
    "SmithResult",
    "SmithSolver",
    "smith_analysis",
    "RichardsonResult",
    "richardson_extrapolation",
    "grid_convergence_study",
    "identify_failure_mode",
    "paik_panel",
    "paik_panel_series",
    "box_girder",
    "issc2009_double_bottom",
    "paik_thayamballi_panel_strength",
    "csr_min_of_modes_panel_strength",
    "effective_width_ultimate_moment",
    "effective_width_column_ratio",
    "exact_box_plastic_modulus",
    "SmithAnalysis",
    "analyse_section",
    "analyse_with_convergence",
]