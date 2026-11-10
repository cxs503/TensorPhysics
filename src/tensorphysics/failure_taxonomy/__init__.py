"""Failure-mode taxonomy + reroute decision table for blocked campaign nodes.

Public surface
--------------
``taxonomy``   six method-level failure classes with definitions and the
               evidence fields that discriminate them;
``evidence``   structural signal extraction from a FAIL record;
``decide``     the class -> reroute-move decision table with its schema gate;
``view``       ``{failure_class, evidence, reroute_suggestion}`` per blocked node,
               plus the ``falsified_hypothesis`` ledger rendering;
``agreement``  Cohen's kappa / accuracy / confusion matrix / negative controls;
``dataset``    loader for the pinned frozen FAIL corpus and the external vocabulary;
``purity``     domain-vocabulary scanner (zero-domain-semantics gate).

The package is deliberately free of physics and application-domain vocabulary: wording
lives in ``data/*.json`` and is checked by :func:`tensorphysics.failure_taxonomy.purity.assert_pure`.
"""
from __future__ import annotations

from .agreement import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    observed_agreement,
    permute_labels,
    replace_bodies,
)
from .dataset import (
    FailRecord,
    FrozenSetError,
    load_cue_lexicon,
    load_decoys,
    load_domain_lexicon,
    load_frozen_set,
    load_manifest,
    load_rule_weights,
    load_seeds,
    stratum_split,
)
from .decide import (
    BASELINE_SWAP,
    DEFAULT_TABLE,
    ESCALATE,
    METHOD_SWAP,
    REROUTE_ACTIONS,
    SCOPE_REDUCTION,
    DecisionTableError,
    RerouteDecision,
    RerouteDirective,
    decide,
    table_coverage,
    validate_decision_table,
)
from .evidence import SIGNAL_KEYS, Evidence, extract
from .labelers import SeedPrototypeLabeler, SignalRuleLabeler, SupervisedCentroidLabeler
from .purity import DomainPurityError, assert_pure, scan_paths, scan_text
from .taxonomy import CLASS_CODES, CLASSES, RANDOM_BASELINE, SPECS, FailureClass, spec
from .view import (
    UNCLASSIFIED,
    FailRecordView,
    blocked_node_view,
    summarise_view,
    to_ledger_entry,
)

__all__ = [
    "BASELINE_SWAP",
    "CLASSES",
    "CLASS_CODES",
    "DEFAULT_TABLE",
    "DecisionTableError",
    "DomainPurityError",
    "ESCALATE",
    "Evidence",
    "FailRecord",
    "FailRecordView",
    "FailureClass",
    "FrozenSetError",
    "METHOD_SWAP",
    "RANDOM_BASELINE",
    "REROUTE_ACTIONS",
    "RerouteDecision",
    "RerouteDirective",
    "SCOPE_REDUCTION",
    "SIGNAL_KEYS",
    "SPECS",
    "SeedPrototypeLabeler",
    "SignalRuleLabeler",
    "SupervisedCentroidLabeler",
    "UNCLASSIFIED",
    "accuracy",
    "assert_pure",
    "blocked_node_view",
    "cohen_kappa",
    "confusion_matrix",
    "decide",
    "extract",
    "load_cue_lexicon",
    "load_decoys",
    "load_domain_lexicon",
    "load_frozen_set",
    "load_manifest",
    "load_rule_weights",
    "load_seeds",
    "macro_f1",
    "observed_agreement",
    "permute_labels",
    "replace_bodies",
    "scan_paths",
    "scan_text",
    "spec",
    "stratum_split",
    "summarise_view",
    "table_coverage",
    "to_ledger_entry",
    "validate_decision_table",
]