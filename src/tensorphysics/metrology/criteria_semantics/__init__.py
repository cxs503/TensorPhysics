"""Criterion metrology: metric semantics, degeneracy taxonomy, adaptive criterion family.

Three capabilities, each independently testable:

``parser``
    lift ``(y, y_ref, tau, metric + unit + condition)`` out of a report's error-verdict
    table or out of a structured record;
``degeneracy``
    classify T1 (reference at/below the noise floor), T2 (sign crossing), T3 (plural
    tolerances / reference interval with no priority);
``adaptive``
    a metric family that keeps every number bounded and returns PASS / FAIL / WITHHELD
    (never "部分通过").

Plus ``replay`` (re-judge a historical corpus, emit the criterion-degeneracy ledger) and
``corpus`` (harvest/freeze/load that corpus with byte-exact provenance).
"""
from __future__ import annotations

from .adaptive import (
    evaluate,
    evaluate_all,
    evaluate_baseline,
    log_ratio_error,
    magnitude_normalised_error,
    merge_multi_metric,
    signed_error,
)
from .corpus import (
    ReportDoc,
    corpus_from_docs,
    harvest_report_docs,
    load_corpus,
    md_sha256,
    write_corpus,
)
from .degeneracy import DegeneracyReport, classify, near_zero_threshold
from .model import (
    Criterion,
    Degeneracy,
    Evaluation,
    Measurement,
    Metric,
    Verdict,
    relative_error,
)
from .parser import (
    CriterionRow,
    criterion_from_dict,
    parse_error_cell,
    parse_markdown_tables,
    parse_reference_cell,
    parse_report_criteria,
    parse_row,
    parse_verdict_cell,
)
from .replay import (
    FLIP_FAIL_TO_PASS,
    FLIP_JUDGED_TO_UNDECIDED,
    FLIP_NONE,
    FLIP_PASS_TO_FAIL,
    FLIP_UNDECIDED_TO_JUDGED,
    ReplayEntry,
    replay_corpus,
    replay_row,
    summarize_ledger,
)

__all__ = [
    "Criterion", "CriterionRow", "Degeneracy", "DegeneracyReport", "Evaluation",
    "Measurement", "Metric", "ReportDoc", "ReplayEntry", "Verdict",
    "classify", "corpus_from_docs", "criterion_from_dict", "evaluate", "evaluate_all",
    "evaluate_baseline", "harvest_report_docs", "load_corpus", "log_ratio_error",
    "magnitude_normalised_error", "merge_multi_metric", "md_sha256", "near_zero_threshold",
    "parse_error_cell", "parse_markdown_tables", "parse_reference_cell",
    "parse_report_criteria", "parse_row", "parse_verdict_cell", "relative_error",
    "replay_corpus", "replay_row", "signed_error", "summarize_ledger", "write_corpus",
    "FLIP_NONE", "FLIP_PASS_TO_FAIL", "FLIP_FAIL_TO_PASS", "FLIP_UNDECIDED_TO_JUDGED",
    "FLIP_JUDGED_TO_UNDECIDED",
]