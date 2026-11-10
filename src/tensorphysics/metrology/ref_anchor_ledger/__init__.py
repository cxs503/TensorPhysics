"""
Reference Anchor Uncertainty Ledger
====================================

Provides a structured uncertainty accounting system for reference anchors
used in ship performance calculations. Each anchor has:
- A nominal value with units
- An expanded uncertainty U_ref at coverage factor k=2
- A traceable source (spec clause, paper, or dataset)
- Provenance metadata (file, sha256, locator)

The ledger enables three-state judgment of validation criteria:
- ROBUST: |Δ| ≤ k·√(U_num² + U_ref²)
- CRITICAL: k·√(...) < |Δ| ≤ 2k·√(...)
- INDETERMINATE: U_ref missing or too large → fail-closed

Uncertainty propagates from anchors through calculations to final verdicts.
"""

from .schema import RefAnchor, AnchorSource, AnchorProvenance
from .ledger import RefAnchorLedger, load_ledger
from .judgment import JudgmentResult, ThreeStateJudgment, judge_criterion
from .propagation import propagate_uncertainty_analytical, monte_carlo_propagation

__all__ = [
    "RefAnchor",
    "AnchorSource",
    "AnchorProvenance",
    "RefAnchorLedger",
    "load_ledger",
    "JudgmentResult",
    "ThreeStateJudgment",
    "judge_criterion",
    "propagate_uncertainty_analytical",
    "monte_carlo_propagation",
]

__version__ = "1.0.0"
