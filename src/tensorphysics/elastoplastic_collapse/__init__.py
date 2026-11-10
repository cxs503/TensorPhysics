"""Elastoplastic progressive collapse and grillage ultimate strength.

This package provides a unified verification surface for:

1. **von Mises elastoplastic constitutive law** (isotropic hardening)
   σ_y = σ_y0 + H ε_p, with consistent tangent stiffness.

2. **Stiffened plate collapse** (Smith method elastoplastic extension)
   Single-span single-stiffener plate grid under axial compression,
   tracking load-displacement to ultimate strength σ_u.

3. **Grillage size-convergence error bound** δ_iso
   3×3 → 5×5 panel convergence criterion (genuine midpoint-quadrature
   convergence of the imperfection-field average) and isolated-element
   deviation.

4. **Tripping mode switch**
   Stiffener torsional buckling (Vlasov/Timoshenko) with the DNV-RP-C201
   elastoplastic reduction and an explicit mode switch against plate local
   buckling, column buckling and yielding.

5. **Matured Smith panel solver** (``panel_ultimate``)
   Iso-strain assembly of intercolumn units with four competing limits,
   transverse-frame spacing effects and the negative-control hooks used by the
   p167w2-n04 acceptance suite.

6. **Thin-walled section properties** (``panel_section``)
   ``I_x``, ``I_y``, ``J`` (thin-wall and exact-rectangle), ``C_w``, polar
   moments and shear centre of a stiffener with attached plating.

7. **IACS ice-belt pressure-area law**
   P = P0 (A/A0)^(-0.3) verification for ice belt design.

References
----------
- Smith, C.S. (1977). "Influence of local imperfections on intercolumn
  behaviour of stiffened plates." Trans. SNAME, 85.
- Faulkner, D. (1975). "A review of effective plating for use in the
  analysis of stiffened plating in bending." Trans. SNAME, 83.
- IACS UR I (2021). Unified Requirements for Polar Class Ships.
- Paik, J.K. & Thayamballi, A.K. (2003). "Ultimate Limit State Design
  of Steel-Plated Structures." Wiley.
"""

from .von_mises_plasticity import (  # noqa: F401
    VonMisesMaterial,
    elastoplastic_tangent,
    return_mapping,
    yield_function,
)
from .stiffened_plate_collapse import (  # noqa: F401
    StiffenedPlateGeometry,
    SmithCollapseSolver,
    CollapseResult,
    faulkner_effective_width,
)
from .grillage_error_bound import (  # noqa: F401
    GrillageModel,
    grillage_convergence_study,
    isolated_element_deviation,
)
from .tripping_mode import (  # noqa: F401
    TrippingAnalysis,
    TrippingResult,
    tripping_capacity,
    ltb_critical_moment,
    ltb_numerical,
)
from .panel_section import (  # noqa: F401
    StiffenerSection,
    stiffener_section,
    rectangle_torsion_constant,
)
from .panel_ultimate import (  # noqa: F401
    PanelScantlings,
    SmithPanelSolver,
    PanelUltimateResult,
    faulkner_phi,
    dnv_column_curve,
)
from .iacs_icebelt_pressure import (  # noqa: F401
    IACSIceBeltPressure,
    pressure_area_law_icebelt,
    verify_pressure_area_exponent,
)

__all__ = [
    "VonMisesMaterial",
    "elastoplastic_tangent",
    "return_mapping",
    "yield_function",
    "StiffenedPlateGeometry",
    "SmithCollapseSolver",
    "CollapseResult",
    "faulkner_effective_width",
    "GrillageModel",
    "grillage_convergence_study",
    "isolated_element_deviation",
    "TrippingAnalysis",
    "TrippingResult",
    "tripping_capacity",
    "ltb_critical_moment",
    "ltb_numerical",
    "StiffenerSection",
    "stiffener_section",
    "rectangle_torsion_constant",
    "PanelScantlings",
    "SmithPanelSolver",
    "PanelUltimateResult",
    "faulkner_phi",
    "dnv_column_curve",
    "IACSIceBeltPressure",
    "pressure_area_law_icebelt",
    "verify_pressure_area_exponent",
]
