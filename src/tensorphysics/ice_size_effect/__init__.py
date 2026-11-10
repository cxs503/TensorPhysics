"""Ice breaking force size effect scaling with uncertainty quantification.

This module implements the model-to-full-scale extrapolation chain for ice
breaking forces, addressing the classical size effect problem in ice mechanics.

Physical background
-------------------
Ice is a quasi-brittle material. When a ship breaks ice, the failure process
involves both gravity-driven bending (Froude scaling, F ∝ L³) and fracture
mechanics (LEFM scaling, F ∝ L²·⁵). The actual scaling exponent n lies
between 2.5 and 3.0, depending on:

* Ice thickness relative to the fracture process zone size
* The ratio of gravitational to fracture energy dissipation
* The brittleness number of the ice-structure system

The size effect law implemented here follows the Bažant-type scaling for
quasi-brittle materials, adapted to ice-ship interaction:

    F(L) = F₀ · (L/L₀)^n · [1 + (L/L_ch)]^(-δ)

where L_ch is a characteristic length related to the fracture process zone,
and δ controls the transition between fracture-dominated and gravity-dominated
regimes.

Module structure
----------------
* **size_effect_law** — core scaling law with exponent n ∈ [2.5, 3.0]
* **ittc_comparison** — comparison with ITTC (2017) recommended procedure
* **uncertainty** — Monte Carlo uncertainty propagation (P50/P95, CV)

References
----------
* Bažant, Z.P. (2002). "Scaling of Structural Strength." Hermes Science.
* Bažant, Z.P. & Planas, J. (1998). "Fracture and Size Effect in Concrete
  and Other Quasibrittle Materials." CRC Press.
* ITTC (2017). "Ice Committee — Final Report and Recommendations."
  28th ITTC, Wuxi, China.
* Timco, G.W. & Frederking, R. (2010). "Compressive strength of
  freshwater ice." Cold Regions Sci. Tech. 61, 115–123.
* Dempster, J. et al. (2019). "Size effects in model ice testing."
  Proc. 25th Int. Conf. Port and Ocean Engineering under Arctic Conditions.
* ISO 19906:2019. "Petroleum and natural gas industries — Arctic structures."
"""

from .size_effect_law import (
    SizeEffectLaw,
    DEFAULT_SIZE_EFFECT_PARAMS,
    compute_size_effect_force,
    compute_size_effect_exponent,
    bazant_type_size_effect,
    froude_vs_fracture_transition,
)
from .ittc_comparison import (
    ITTCRecommendedMethod,
    compute_ittc_full_scale,
    compare_scaling_methods,
    ittc_correction_factor,
)
from .uncertainty import (
    MonteCarloUQ,
    run_monte_carlo,
    compute_uq_statistics,
    convergence_check,
)

__all__ = [
    "SizeEffectLaw",
    "DEFAULT_SIZE_EFFECT_PARAMS",
    "compute_size_effect_force",
    "compute_size_effect_exponent",
    "bazant_type_size_effect",
    "froude_vs_fracture_transition",
    "ITTCRecommendedMethod",
    "compute_ittc_full_scale",
    "compare_scaling_methods",
    "ittc_correction_factor",
    "MonteCarloUQ",
    "run_monte_carlo",
    "compute_uq_statistics",
    "convergence_check",
]
