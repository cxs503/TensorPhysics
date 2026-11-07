"""Scientific Contribution Measurement Unit Ontology.

Six contribution categories (C1-C6) with evidence thresholds,
a rule-based classifier with LLM fallback, golden set evaluation,
and CLI tools for classification and reporting.
"""

from .ontology import (
    ContributionCategory,
    CONTRIBUTION_ONTOLOGY,
    EVIDENCE_THRESHOLDS,
    COUNTER_EXAMPLES,
    get_category,
    list_categories,
)
from .classifier import ContributionClassifier, classify_item
from .golden_set import GOLDEN_SET, load_golden_set
from .metrics import compute_f1, compute_productivity_rate

__all__ = [
    "ContributionCategory",
    "CONTRIBUTION_ONTOLOGY",
    "EVIDENCE_THRESHOLDS",
    "COUNTER_EXAMPLES",
    "get_category",
    "list_categories",
    "ContributionClassifier",
    "classify_item",
    "GOLDEN_SET",
    "load_golden_set",
    "compute_f1",
    "compute_productivity_rate",
]
