"""First-class acceptance criteria: registry + cross-campaign conflict arbitration.

Pipeline::

    text --extract--> Criterion(quantity, unit, statistic, threshold, direction,
                                 provenance)
         --normalise--> canonical quantity / unit / statistic / threshold
         --detect-----> Conflict (same quantity+statistic+direction, |Δ| > band)
         --arbitrate--> ArbitrationRecord (closes conflict, replayable chain)

Domain vocabulary and tolerance bands are injected through ``ConflictConfig``
(JSON); this package contains no domain semantics.
"""
from .config import (
    CONFIG_SCHEMA,
    ConfigError,
    ConflictConfig,
    MetricSpec,
    StatisticSpec,
    leak_terms,
    load_config,
)
from .model import Criterion, SourceRef, canonical_json, make_criterion_id
from .normalize import (
    normalize_comparator,
    normalize_statistic,
    normalize_threshold,
    normalize_unit,
)
from .extract import ExtractionResult, extract_criteria, extract_from_entries
from .conflict import Conflict, conflicts_digest, detect_conflicts
from .registry import CorpusEntry, CriterionRegistry
from .arbitration import (
    ALL_DECISIONS,
    CLOSING_DECISIONS,
    GENESIS,
    ArbitrationError,
    ArbitrationLedger,
    ArbitrationRecord,
)

__all__ = [
    "CONFIG_SCHEMA", "ConfigError", "ConflictConfig", "MetricSpec", "StatisticSpec",
    "leak_terms", "load_config",
    "Criterion", "SourceRef", "canonical_json", "make_criterion_id",
    "normalize_comparator", "normalize_statistic", "normalize_threshold", "normalize_unit",
    "ExtractionResult", "extract_criteria", "extract_from_entries",
    "Conflict", "conflicts_digest", "detect_conflicts",
    "CorpusEntry", "CriterionRegistry",
    "ALL_DECISIONS", "CLOSING_DECISIONS", "GENESIS", "ArbitrationError",
    "ArbitrationLedger", "ArbitrationRecord",
]
