"""Judgeability census of campaign acceptance criteria (P106W3A2-N08).

The platform's campaign nodes declare their acceptance criteria as free text. This
package answers one question with a number instead of a compliment:

    *how much of a node's PASS actually rests on a criterion that can fail?*

Four classes (:class:`Judgeability`) are defined and one support ratio is measured
(:func:`~tensorphysics.acceptance_judgeability.support.judgeability_support`):

=========================  =====================================================
JUDGEABLE                  quantity + reference/baseline + tolerance
THRESHOLD_ONLY             quantity + bound, but no reference to be judged against
RECORD_ONLY                only records/monotonicity/non-crash/declaration
WITHHELD_OK                statement explicitly declares that data is missing
=========================  =====================================================

Mechanism (the part that changes behaviour): :mod:`...gate` refuses a PASS whose
support ratio is below the policy threshold, and refuses a PASS supported only by
``RECORD_ONLY`` clauses. ``RECORD_ONLY`` can never add to the numerator, so
"the report contains a figure" can no longer be the whole reason a node passed.

Design constraints
------------------
* **Zero domain semantics.** The vocabulary used to recognise a tolerance, a
  reference, a withheld declaration or a record-only requirement lives in
  ``data/lexicon_default.json`` and is *metrology* vocabulary only; the logic
  never names a physics quantity, a solver or a discipline. A caller may inject
  its own lexicon (:func:`classify.get_lexicon`) without touching code.
* **No silent pass.** An unclassifiable clause is reported as
  ``RECORD_ONLY/UNCLASSIFIED`` with ``needs_review`` set, and it is excluded from
  the numerator by construction -- ambiguity never becomes support.
* **Reproducible.** :mod:`...census` hashes the support table, so two runs over
  the same frozen corpus produce the same digest.
"""

from .classify import (
    DEFAULT_LEXICON,
    Lexicon,
    lexicon_from_dict,
    ClauseFeatures,
    ClassifiedClause,
    Judgeability,
    Subtype,
    classify,
    classify_all,
    get_lexicon,
    set_lexicon,
)
from .gate import (
    blocking_clauses,
    clauses_from_node_metadata,
    enforce,
    judgeability_flag,
    judgeability_flag_for_node,
)
from .support import (
    DEFAULT_KS_MIN,
    IMPLICIT_POLICY,
    DEFAULT_POLICY,
    STRICT_POLICY,
    TOLERANCE_WEIGHTS,
    UNIFORM_WEIGHTS,
    Adjudication,
    Clause,
    Evidence,
    KsRow,
    PassPolicy,
    WeightPolicy,
    adjudicate,
    gate_flag,
    judgeability_support,
    node_ks,
    upgrade_paths,
)
from .agreement import cohen_kappa, confusion_matrix, stratified_sample
from .census import (
    CensusResult,
    corpus_digest,
    ks_table_digest,
    load_corpus,
    run_census,
)

__all__ = [
    "DEFAULT_KS_MIN",
    "IMPLICIT_POLICY",
    "DEFAULT_POLICY",
    "STRICT_POLICY",
    "TOLERANCE_WEIGHTS",
    "UNIFORM_WEIGHTS",
    "DEFAULT_LEXICON",
    "Lexicon",
    "lexicon_from_dict",
    "Adjudication",
    "CensusResult",
    "Clause",
    "ClauseFeatures",
    "ClassifiedClause",
    "Evidence",
    "Judgeability",
    "KsRow",
    "PassPolicy",
    "Subtype",
    "WeightPolicy",
    "adjudicate",
    "blocking_clauses",
    "clauses_from_node_metadata",
    "enforce",
    "judgeability_flag",
    "judgeability_flag_for_node",
    "classify",
    "classify_all",
    "cohen_kappa",
    "confusion_matrix",
    "corpus_digest",
    "gate_flag",
    "get_lexicon",
    "judgeability_support",
    "ks_table_digest",
    "load_corpus",
    "node_ks",
    "run_census",
    "set_lexicon",
    "stratified_sample",
    "upgrade_paths",
]