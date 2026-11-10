"""Criterion metrology: error criteria with metric semantics (P266W2-N09).

The three defects this package removes:

* an ``error`` is judged by a bare string (``"相对误差 < 2%"``) with no record of what the
  reference magnitude was, so identical strings can mean "excellent" and "meaningless";
* a relative error is *unbounded* when the reference tends to zero -- the comparison then
  silently becomes a coin flip;
* an artifact may state several metrics for one measurand with no precedence rule, which
  is what "部分通过" really means.

Public API
----------
parsing      :func:`parse_criterion`, :func:`extract_quadruple`, :func:`parse_goal_criteria`
classification :func:`classify`, :func:`analyze`, :class:`Degeneracy`, ``T1``/``T2``/``T3``
adaptive family:func:`judge`, :func:`judge_family`, :func:`merge_verdicts`,
                :func:`scaled_delta`, :func:`log_ratio`, :func:`signed_abs`,
                :func:`bounded_relative_error`
replay       :func:`replay`, :func:`replay_repository`, :func:`summarize`, :func:`case_studies`
controls     :func:`evaluate_controls`, :func:`evaluate_labeled`

Example
-------
>>> from tensorphysics.metrology.criteria import extract_quadruple, judge, classify
>>> rec = extract_quadruple("相对误差 < 2%", 1.05, 1e-9, scale=1.0)
>>> classify(rec).classes
('T1_zero_reference',)
>>> judge(rec).status
'PASS'
"""
from __future__ import annotations

from .model import (
    ABSOLUTE,
    DEFAULT_NOISE_REL,
    FAIL,
    LOG_RATIO,
    METRICS,
    NOT_APPLICABLE,
    PASS,
    RELATIVE,
    SIGNED,
    UNPARSED,
    VERDICTS,
    WITHHELD,
    CriterionRecord,
    MetricEvaluation,
    ParsedCriterion,
    Verdict,
    finite,
    sign,
)
from .parser import (
    GoalCriterion,
    extract_quadruple,
    parse_criterion,
    parse_goal_criteria,
    parse_markdown_criteria_table,
)
from .degeneracy import (
    DEGENERACY_TYPES,
    T1,
    T2,
    T3,
    Degeneracy,
    analyze,
    classify,
    multi_metric_families,
    resolve_scale,
)
from .adaptive import (
    LOW_SCALE_FRACTION,
    bounded_relative_error,
    deviation_fraction,
    direction,
    judge,
    judge_family,
    log_ratio,
    merge_verdicts,
    scaled_delta,
    signed_abs,
)
from .harvest import HarvestStats, build_corpus, harvest_json, harvest_markdown
from .replay import (
    CORPUS_MIN_RECORDS,
    DEGRADED_MIN_FRACTION,
    ReplayResult,
    ReplayRow,
    baseline_verdict,
    case_studies,
    json_safe,
    replay,
    replay_repository,
    summarize,
)
from .stats import agreement, cohen_kappa, confusion_matrix
from .controls import evaluate_controls, evaluate_labeled, labeled_set, load_controls

__all__ = [
    # model
    "PASS", "FAIL", "WITHHELD", "NOT_APPLICABLE", "VERDICTS",
    "RELATIVE", "ABSOLUTE", "LOG_RATIO", "SIGNED", "METRICS", "DEFAULT_NOISE_REL",
    "CriterionRecord", "ParsedCriterion", "MetricEvaluation", "Verdict", "UNPARSED",
    "finite", "sign",
    # parser
    "parse_criterion", "extract_quadruple", "parse_goal_criteria", "GoalCriterion",
    "parse_markdown_criteria_table",
    # degeneracy
    "classify", "analyze", "Degeneracy", "DEGENERACY_TYPES", "T1", "T2", "T3",
    "multi_metric_families", "resolve_scale",
    # adaptive
    "judge", "judge_family", "merge_verdicts", "scaled_delta", "log_ratio", "signed_abs",
    "bounded_relative_error", "deviation_fraction", "direction", "LOW_SCALE_FRACTION",
    # replay
    "replay", "replay_repository", "summarize", "case_studies", "baseline_verdict", "json_safe",
    "ReplayResult", "ReplayRow", "HarvestStats", "build_corpus", "harvest_json",
    "harvest_markdown", "CORPUS_MIN_RECORDS", "DEGRADED_MIN_FRACTION",
    # stats / controls
    "agreement", "cohen_kappa", "confusion_matrix", "evaluate_controls", "evaluate_labeled",
    "labeled_set", "load_controls",
]

__version__ = "1.0.0"