"""Ice model-test scaling: dimensionless groups, scaling laws, error budget.

This module implements the dimensional-analysis framework for extrapolating
model-scale ice tests to full-scale ship performance predictions.

Physical background
-------------------
When a ship model is tested in an ice tank, the model-ice properties cannot
simultaneously satisfy all the dimensionless groups that govern full-scale
ice-structure interaction.  The primary scaling law is Froude similitude
(length ∝ λ, velocity ∝ √λ, time ∝ √λ, force ∝ λ³), but ice strength,
elastic modulus, fracture toughness, friction, and porosity each have their
own scaling requirements that generally conflict.

This module provides:

* **dimensionless** — the complete set of governing Π-groups and their
  Froude-law scaling exponents;
* **scaling_laws** — Froude (primary), alternative strength laws
  (λ⁻¹, λ⁻⁰·⁵, Cauchy/noscaling), and round-trip consistency checks;
* **budget** — additive/multiplicative error ledger that closes to < 1 %;
* **conflict_analysis** — identification of which Π-groups cannot be
  simultaneously satisfied and the cost of each concession.

References
----------
* Lindqvist, G. (1989). POAC'89, Luleå, Vol. 2, pp. 722–735.
* ITTC (2017). Ice Committee Final Report, 28th ITTC, Wuxi.
* ISO 19906:2019. Petroleum and natural gas industries — Arctic structures.
* Timco, G.W. & Frederking, R. (2010). Cold Regions Sci. Tech. 61, 115–123.
* Kerr, A.D. (1996). "Bearing capacity of ice sheets." Cold Regions Sci.
  Tech. 24, 125–138.
"""

from .dimensionless import (
    DimensionlessGroup,
    ALL_GROUPS,
    compute_all_groups,
    froude_scale_group,
    group_value,
)
from .scaling_laws import (
    HALF_LAW,
    ScalingLaw,
    FROUDE_LAW,
    CAUCHY_LAW,
    STRENGTH_LAWS,
    scale_property,
    froude_round_trip,
    froude_internal_consistency,
)
from .budget import (
    ErrorBudget,
    BudgetItem,
    compute_error_budget,
    budget_closure_check,
)
from .conflict_analysis import (
    ConflictDimension,
    analyse_conflicts,
    feasible_parameter_window,
    lambda_sweep,
)

__all__ = [
    "DimensionlessGroup",
    "ALL_GROUPS",
    "compute_all_groups",
    "froude_scale_group",
    "group_value",
    "ScalingLaw",
    "FROUDE_LAW",
    "HALF_LAW",
    "CAUCHY_LAW",
    "STRENGTH_LAWS",
    "scale_property",
    "froude_round_trip",
    "froude_internal_consistency",
    "ErrorBudget",
    "BudgetItem",
    "compute_error_budget",
    "budget_closure_check",
    "ConflictDimension",
    "analyse_conflicts",
    "feasible_parameter_window",
    "lambda_sweep",
]
