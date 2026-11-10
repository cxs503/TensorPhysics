"""Reroute decision table: failure class -> what the scheduler should do next.

Three reroute moves are available to an autonomous campaign scheduler:

``SCOPE_REDUCTION``
    keep the method, shrink the claim to the sub-range / sub-case where the
    evidence actually supports it (narrower regime, smaller domain, weaker
    statement, or a documented bound instead of a point value).
``METHOD_SWAP``
    keep the question, replace the machinery (different discretisation,
    regularisation, estimator, coupling scheme, or a fixed component).
``BASELINE_SWAP``
    keep both, replace the yardstick (another analytic limit, a published
    benchmark, a cross-repository reference implementation, or a self-consistency
    identity).

Each class gets exactly one primary move, one secondary move, and a non-empty
list of **trigger evidence fields**: the signal keys from
:mod:`tensorphysics.failure_taxonomy.evidence` that justify the move.  The
schema gate :func:`validate_decision_table` rejects a table where any class
lacks a recommendation or where a recommendation cites no trigger, which makes
criterion C4 machine-checkable rather than a prose promise.

If none of a class's trigger fields fired, the record does not contain enough
evidence to route automatically: :func:`decide` then returns
``action="ESCALATE"`` with ``evidence_gap=True`` instead of guessing.  Escalation
is *not* a reroute move and is therefore never a table entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .evidence import SIGNAL_KEYS
from .taxonomy import CLASS_CODES, FailureClass

SCOPE_REDUCTION = "SCOPE_REDUCTION"
METHOD_SWAP = "METHOD_SWAP"
BASELINE_SWAP = "BASELINE_SWAP"
ESCALATE = "ESCALATE"

REROUTE_ACTIONS: tuple[str, ...] = (SCOPE_REDUCTION, METHOD_SWAP, BASELINE_SWAP)


class DecisionTableError(ValueError):
    """The decision table is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class RerouteDirective:
    """A table row: one class -> primary/secondary move + trigger evidence."""

    failure_class: str
    action: str
    secondary_action: str
    trigger_fields: tuple[str, ...]
    rationale_code: str
    expected_effect: str


@dataclass(frozen=True)
class RerouteDecision:
    """Outcome of routing one record."""

    failure_class: str
    action: str
    secondary_action: str
    matched_triggers: tuple[str, ...] = ()
    evidence_gap: bool = False
    rationale_code: str = ""
    expected_effect: str = ""

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "secondary_action": self.secondary_action,
            "matched_triggers": list(self.matched_triggers),
            "evidence_gap": self.evidence_gap,
            "rationale_code": self.rationale_code,
            "expected_effect": self.expected_effect,
        }


DEFAULT_TABLE: tuple[RerouteDirective, ...] = (
    RerouteDirective(
        failure_class=FailureClass.BASELINE_ABSENT.code,
        action=BASELINE_SWAP,
        secondary_action=SCOPE_REDUCTION,
        trigger_fields=("reference_lexeme", "absence_lexeme"),
        rationale_code="no_yardstick_but_result_exists",
        expected_effect=(
            "an independent reference (analytic limit, published benchmark, "
            "cross-implementation value) turns an ungradeable node into a graded "
            "one without touching the method"
        ),
    ),
    RerouteDirective(
        failure_class=FailureClass.TOLERANCE_EXCEEDED.code,
        action=SCOPE_REDUCTION,
        secondary_action=METHOD_SWAP,
        trigger_fields=("quantified_gap", "threshold_lexeme"),
        rationale_code="quantified_miss_of_declared_band",
        expected_effect=(
            "re-stating the claim on the sub-range where the band is met keeps the "
            "result honest; the residual miss is then chased by a method change"
        ),
    ),
    RerouteDirective(
        failure_class=FailureClass.CONVERGENCE_STALL.code,
        action=METHOD_SWAP,
        secondary_action=SCOPE_REDUCTION,
        trigger_fields=("convergence_lexeme", "residual_magnitude"),
        rationale_code="iterate_never_settled",
        expected_effect=(
            "a changed solver/preconditioner/estimator budget restores a converging "
            "sequence; only if the stall is structural is the scope narrowed"
        ),
    ),
    RerouteDirective(
        failure_class=FailureClass.MODEL_FIDELITY_LIMIT.code,
        action=METHOD_SWAP,
        secondary_action=SCOPE_REDUCTION,
        trigger_fields=("assumption_lexeme", "envelope_lexeme"),
        rationale_code="intrinsic_formulation_limit",
        expected_effect=(
            "higher-fidelity formulation (or explicit validity envelope) replaces "
            "an assumption whose error cannot be reduced by tuning"
        ),
    ),
    RerouteDirective(
        failure_class=FailureClass.DATA_INSUFFICIENCY.code,
        action=SCOPE_REDUCTION,
        secondary_action=BASELINE_SWAP,
        trigger_fields=("absence_lexeme", "artifact_lexeme", "withheld_lexeme"),
        rationale_code="missing_inputs_or_deliverables",
        expected_effect=(
            "shrink the claim to what the available inputs determine; the withheld "
            "branch is re-opened when the artefact or calibration data lands"
        ),
    ),
    RerouteDirective(
        failure_class=FailureClass.IMPLEMENTATION_DEFECT.code,
        action=METHOD_SWAP,
        secondary_action=SCOPE_REDUCTION,
        trigger_fields=("defect_lexeme",),
        rationale_code="localised_code_error",
        expected_effect=(
            "patch the faulty component first (a method swap of the affected leg); "
            "only re-narrow scope if the gap survives the fix"
        ),
    ),
)


def _as_table(table: Iterable[RerouteDirective] | Mapping[str, RerouteDirective]):
    if isinstance(table, Mapping):
        return list(table.values())
    return list(table)


def validate_decision_table(
    table: Iterable[RerouteDirective] | Mapping[str, RerouteDirective],
    classes: Iterable[str] = CLASS_CODES,
) -> None:
    """Schema gate for criterion C4.

    Raises :class:`DecisionTableError` when

    * a class is missing (no recommendation),
    * a row's ``action``/``secondary_action`` is not one of the three moves,
    * a row carries no trigger field (a recommendation with no evidence hook),
    * a trigger field is not a known signal key,
    * a class is listed more than once.
    """
    rows = _as_table(table)
    seen: dict[str, RerouteDirective] = {}
    for row in rows:
        if not isinstance(row, RerouteDirective):
            raise DecisionTableError(f"not a RerouteDirective: {row!r}")
        if row.failure_class in seen:
            raise DecisionTableError(f"duplicate class: {row.failure_class}")
        seen[row.failure_class] = row

    for code in classes:
        if code not in seen:
            raise DecisionTableError(f"class without recommendation: {code}")
        row = seen[code]
        for label, action in (("action", row.action), ("secondary_action", row.secondary_action)):
            if action not in REROUTE_ACTIONS:
                raise DecisionTableError(f"{code}: {label}={action!r} is not a reroute move")
        if not row.trigger_fields:
            raise DecisionTableError(f"{code}: recommendation without trigger evidence field")
        for trig in row.trigger_fields:
            if trig not in SIGNAL_KEYS:
                raise DecisionTableError(f"{code}: unknown trigger evidence field {trig!r}")
        if row.action == row.secondary_action:
            raise DecisionTableError(f"{code}: secondary move equals primary move")
        if not row.rationale_code:
            raise DecisionTableError(f"{code}: missing rationale code")


def table_lookup(table, failure_class: str) -> RerouteDirective:
    for row in _as_table(table):
        if row.failure_class == failure_class:
            return row
    raise DecisionTableError(f"class without recommendation: {failure_class}")


def decide(
    failure_class: str,
    present_fields: Iterable[str],
    table=DEFAULT_TABLE,
) -> RerouteDecision:
    """Route one classified record to a reroute move (or escalate)."""
    directive = table_lookup(table, failure_class)
    present = tuple(present_fields)
    matched = tuple(t for t in directive.trigger_fields if t in present)
    if not matched:
        return RerouteDecision(
            failure_class=failure_class,
            action=ESCALATE,
            secondary_action="",
            matched_triggers=(),
            evidence_gap=True,
            rationale_code="no_trigger_evidence_present",
            expected_effect="manual triage: the record carries none of the class triggers",
        )
    return RerouteDecision(
        failure_class=failure_class,
        action=directive.action,
        secondary_action=directive.secondary_action,
        matched_triggers=matched,
        evidence_gap=False,
        rationale_code=directive.rationale_code,
        expected_effect=directive.expected_effect,
    )


def table_coverage(table=DEFAULT_TABLE) -> dict:
    """Machine-readable coverage summary used by the report."""
    validate_decision_table(table)
    return {
        "classes": len(_as_table(table)),
        "actions": {row.failure_class: row.action for row in _as_table(table)},
        "trigger_counts": {
            row.failure_class: len(row.trigger_fields) for row in _as_table(table)
        },
        "min_triggers_per_row": min(len(r.trigger_fields) for r in _as_table(table)),
    }


__all__ = [
    "BASELINE_SWAP",
    "DEFAULT_TABLE",
    "ESCALATE",
    "METHOD_SWAP",
    "REROUTE_ACTIONS",
    "RerouteDecision",
    "RerouteDirective",
    "SCOPE_REDUCTION",
    "DecisionTableError",
    "decide",
    "table_coverage",
    "table_lookup",
    "validate_decision_table",
]