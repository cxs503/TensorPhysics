"""Drift audit: compare a frozen criterion manifest with a receipt-time goal.

The audit answers one question per criterion: *did the acceptance requirement
move, and in which direction?*  Classes (``DriftClass``):

``unchanged``      same metric/op/value/scope/count -- no movement.
``tightened``      strictly harder requirement (higher ``>=`` value, lower
                   ``<=`` value, ``>=`` replaced by ``>``, or a criterion that
                   the receipt *added*).  Allowed: the fail-closed direction.
``loosened``       strictly easier requirement (lower ``>=`` value, higher
                   ``<=`` value, or a criterion the receipt *dropped*).
``scope_changed``  same requirement but a different evidence scope (sample size
                   ``count``, unit, or scope descriptor) -- not comparable.
``undefined``      cannot be decided: incomparable operator change, exact-value
                   (``==``) value change, mapping-target change, unparseable or
                   missing receipt text.

Verdict policy (fail-closed): ``PASS`` only if the manifest verifies as
``registered`` **and** every criterion is ``unchanged`` or ``tightened``.
Anything else -- including ``unregistered`` (empty receipt) and ``tampered``
(digest mismatch) -- yields ``FAIL``.  Each report also emits event records for
the event stream and a KPI view (:mod:`...criterion_prereg.kpi`).

Zero domain semantics: class names, ops and thresholds are data; no
engineering-domain vocabulary is present in this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
from decimal import Decimal
from enum import Enum

from .criterion_manifest import (
    REGISTERED,
    TAMPERED,
    UNREGISTERED,
    Criterion,
    criteria_of,
    extract_criteria,
    verify_manifest,
)

EVENT_SCHEMA = "criterion-prereg-event/1"

#: Severity ranking: a criterion showing several movements reports the worst.
_SEVERITY = ("loosened", "scope_changed", "undefined", "tightened", "unchanged")


class DriftClass(str, Enum):
    UNCHANGED = "unchanged"
    TIGHTENED = "tightened"
    LOOSENED = "loosened"
    SCOPE_CHANGED = "scope_changed"
    UNDEFINED = "undefined"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


#: Classes that do *not* fail the gate.
_ALLOWED = {DriftClass.UNCHANGED, DriftClass.TIGHTENED}


@dataclass
class CriterionDrift:
    identity: str
    metric: str
    klass: DriftClass
    reason: str
    frozen: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    diffs: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "metric": self.metric,
            "class": self.klass.value,
            "reason": self.reason,
            "frozen": self.frozen,
            "receipt": self.receipt,
            "diffs": {k: list(v) for k, v in self.diffs.items()},
        }


@dataclass
class DriftReport:
    state: str
    verdict: Verdict
    items: list[CriterionDrift] = field(default_factory=list)
    mission_id: str | int | None = None
    node_id: str | None = None
    frozen_sha256: str = ""
    receipt_sha256: str = ""
    notes: list[str] = field(default_factory=list)
    audited_at: str = ""

    # -- derived views ----------------------------------------------------- #
    @property
    def counts(self) -> dict[str, int]:
        out = {c.value: 0 for c in DriftClass}
        for item in self.items:
            out[item.klass.value] += 1
        return out

    @property
    def drifting(self) -> list[CriterionDrift]:
        return [i for i in self.items if i.klass not in _ALLOWED]

    @property
    def n_drift(self) -> int:
        return len(self.drifting)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def dominant_class(self) -> str:
        """Worst movement in the report (``unchanged`` when nothing moved)."""
        if not self.items:
            return DriftClass.UNCHANGED.value
        worst = min(self.items, key=lambda i: _SEVERITY.index(i.klass.value))
        return worst.klass.value

    @property
    def drift_rate(self) -> float:
        return (self.n_drift / self.n_items) if self.n_items else 0.0

    def failed(self) -> bool:
        return self.verdict is Verdict.FAIL

    # -- event stream ------------------------------------------------------ #
    def events(self) -> list[dict[str, Any]]:
        stamp = self.audited_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        out: list[dict[str, Any]] = []
        if self.state != REGISTERED:
            out.append(
                {
                    "type": "criterion.prereg.anomaly",
                    "schema": EVENT_SCHEMA,
                    "mission_id": self.mission_id,
                    "node_id": self.node_id,
                    "state": self.state,
                    "severity": "error",
                    "reason": "receipt_manifest_" + self.state,
                    "frozen_sha256": self.frozen_sha256,
                    "receipt_sha256": self.receipt_sha256,
                    "occurred_at": stamp,
                }
            )
        for item in self.drifting:
            out.append(
                {
                    "type": "criterion.drift",
                    "schema": EVENT_SCHEMA,
                    "mission_id": self.mission_id,
                    "node_id": self.node_id,
                    "state": self.state,
                    "severity": "error",
                    "class": item.klass.value,
                    "metric": item.metric,
                    "reason": item.reason,
                    "diffs": {k: list(v) for k, v in item.diffs.items()},
                    "frozen_sha256": self.frozen_sha256,
                    "receipt_sha256": self.receipt_sha256,
                    "occurred_at": stamp,
                }
            )
        if not out:
            out.append(
                {
                    "type": "criterion.drift.clean",
                    "schema": EVENT_SCHEMA,
                    "mission_id": self.mission_id,
                    "node_id": self.node_id,
                    "state": self.state,
                    "severity": "info",
                    "class": DriftClass.UNCHANGED.value,
                    "reason": "no_drift",
                    "frozen_sha256": self.frozen_sha256,
                    "receipt_sha256": self.receipt_sha256,
                    "occurred_at": stamp,
                }
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "criterion-drift-report/1",
            "state": self.state,
            "verdict": self.verdict.value,
            "mission_id": self.mission_id,
            "node_id": self.node_id,
            "frozen_sha256": self.frozen_sha256,
            "receipt_sha256": self.receipt_sha256,
            "n_items": self.n_items,
            "n_drift": self.n_drift,
            "dominant_class": self.dominant_class,
            "drift_rate": self.drift_rate,
            "counts": self.counts,
            "notes": list(self.notes),
            "audited_at": self.audited_at,
            "items": [i.to_dict() for i in self.items],
        }


# --------------------------------------------------------------------------- #
# comparison primitives
# --------------------------------------------------------------------------- #
#: Which numeric direction is *stricter* for a given operator.
_UPPER_BOUND_OPS = {">=": True, ">": True, "<=": False, "<": False}


def _value_class(frozen: Criterion, receipt: Criterion) -> tuple[DriftClass, str]:
    """Classify a same-op, threshold-vs-threshold value movement."""
    if frozen.op != receipt.op:
        return DriftClass.UNDEFINED, "op_changed"
    if frozen.op == "==":
        if frozen.value == receipt.value:
            return DriftClass.UNCHANGED, "value_equal"
        return DriftClass.UNDEFINED, "exact_value_changed"
    if frozen.unit != receipt.unit:
        return DriftClass.SCOPE_CHANGED, "unit_changed"
    try:
        f = Decimal(frozen.value)
        r = Decimal(receipt.value)
    except Exception:  # pragma: no cover - values are validated at parse
        return DriftClass.UNDEFINED, "value_unparseable"
    if f == r:
        return DriftClass.UNCHANGED, "value_equal"
    is_upper = _UPPER_BOUND_OPS[frozen.op]
    harder = (r > f) if is_upper else (r < f)
    return (DriftClass.TIGHTENED if harder else DriftClass.LOOSENED), (
        "value_raised" if r > f else "value_lowered"
    )


def _op_class(frozen: Criterion, receipt: Criterion) -> tuple[DriftClass, str] | None:
    """Classify an operator change; ``None`` when the comparison is comparable."""
    fo, ro = frozen.op, receipt.op
    if fo == ro:
        return None
    if fo in _UPPER_BOUND_OPS and ro in _UPPER_BOUND_OPS:
        if _UPPER_BOUND_OPS[fo] != _UPPER_BOUND_OPS[ro]:
            return DriftClass.UNDEFINED, "op_family_changed"
        # same family: ">" is strictly stricter than ">=" (and "<" vs "<=").
        stricter = (ro == ">") if _UPPER_BOUND_OPS[fo] else (ro == "<")
        return (DriftClass.TIGHTENED if stricter else DriftClass.LOOSENED), "op_strengthened" if stricter else "op_weakened"
    if fo in _UPPER_BOUND_OPS or ro in _UPPER_BOUND_OPS or "==" in (fo, ro):
        return DriftClass.UNDEFINED, "op_family_changed"
    if fo == "->" or ro == "->":
        return DriftClass.UNDEFINED, "kind_changed"
    return DriftClass.UNDEFINED, "op_changed"


def compare_criterion(frozen: Criterion, receipt: Criterion) -> CriterionDrift:
    """Field-by-field comparison of one paired criterion pair."""
    diffs: dict[str, tuple[Any, Any]] = {}
    classes: list[tuple[DriftClass, str]] = []

    for name in ("label", "kind", "metric", "op", "value", "unit", "count", "scope"):
        a, b = getattr(frozen, name), getattr(receipt, name)
        if a != b:
            diffs[name] = (a, b)

    if frozen.kind != receipt.kind:
        classes.append((DriftClass.UNDEFINED, "kind_changed"))
    elif frozen.kind == "mapping":
        if frozen.value != receipt.value:
            classes.append((DriftClass.UNDEFINED, "mapping_target_changed"))
        elif frozen.scope != receipt.scope or frozen.count != receipt.count:
            classes.append((DriftClass.SCOPE_CHANGED, "scope_changed"))
        else:
            classes.append((DriftClass.UNCHANGED, "value_equal"))
    else:
        op_cls = _op_class(frozen, receipt)
        if op_cls is not None:
            classes.append(op_cls)
        else:
            classes.append(_value_class(frozen, receipt))
            if frozen.op not in ("==",):
                if frozen.unit != receipt.unit:
                    classes.append((DriftClass.SCOPE_CHANGED, "unit_changed"))
                if frozen.count != receipt.count:
                    classes.append((DriftClass.SCOPE_CHANGED, "count_changed"))
                if frozen.scope != receipt.scope:
                    classes.append((DriftClass.SCOPE_CHANGED, "scope_changed"))
            elif frozen.count != receipt.count or frozen.scope != receipt.scope:
                classes.append((DriftClass.SCOPE_CHANGED, "scope_changed"))

    worst = min(classes, key=lambda c: _SEVERITY.index(c[0].value)) if classes else (DriftClass.UNCHANGED, "identical")
    all_reasons = sorted({r for _, r in classes})
    reason = worst[1] if len(all_reasons) == 1 else "+".join(all_reasons)
    return CriterionDrift(
        identity=frozen.identity,
        metric=frozen.metric,
        klass=worst[0],
        reason=reason,
        frozen=frozen.to_dict(),
        receipt=receipt.to_dict(),
        diffs=diffs,
    )


def _pair(frozen: list[Criterion], receipt: list[Criterion]) -> tuple[list[tuple[Criterion, Criterion]], list[Criterion], list[Criterion]]:
    """Greedy identity pairing; unmatched frozen/receipt leftovers are returned."""
    remaining = list(receipt)
    pairs: list[tuple[Criterion, Criterion]] = []
    unmatched_frozen: list[Criterion] = []
    for f in frozen:
        hit = next((r for r in remaining if r.identity == f.identity), None)
        if hit is not None:
            pairs.append((f, hit))
            remaining.remove(hit)
        else:
            unmatched_frozen.append(f)
    return pairs, unmatched_frozen, remaining


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def audit(
    frozen_manifest: Mapping[str, Any],
    *,
    receipt_goal: str | None = None,
    receipt_manifest: Mapping[str, Any] | None = None,
    mission_id: str | int | None = None,
    node_id: str | None = None,
    audited_at: str | None = None,
) -> DriftReport:
    """Audit a receipt against its frozen manifest.

    Either the receipt *goal text* (``receipt_goal``) or a receipt *manifest*
    (``receipt_manifest``) may be supplied; the goal text is the substance, the
    manifest is checked for digest integrity first.
    """
    stamp = audited_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    mission = mission_id if mission_id is not None else frozen_manifest.get("mission_id")
    node = node_id if node_id is not None else frozen_manifest.get("node_id")

    frozen_state = verify_manifest(frozen_manifest)
    frozen_sha = str(frozen_manifest.get("manifest_sha256") or "")

    if frozen_state != REGISTERED:
        return DriftReport(
            state=frozen_state,
            verdict=Verdict.FAIL,
            mission_id=mission,
            node_id=node,
            frozen_sha256=frozen_sha,
            notes=[f"frozen_manifest_{frozen_state}"],
            audited_at=stamp,
        )

    notes: list[str] = []
    receipt_sha = ""
    if receipt_manifest is not None:
        receipt_sha = str(receipt_manifest.get("manifest_sha256") or "")
        state = verify_manifest(receipt_manifest)
        if state == TAMPERED:
            return DriftReport(
                state=TAMPERED,
                verdict=Verdict.FAIL,
                mission_id=mission,
                node_id=node,
                frozen_sha256=frozen_sha,
                receipt_sha256=receipt_sha,
                notes=["receipt_manifest_tampered"],
                audited_at=stamp,
            )
        receipt_criteria = criteria_of(receipt_manifest)
        if receipt_goal is None:
            notes.append("receipt_from_manifest_only")
        else:
            extracted = extract_criteria(receipt_goal)
            if extracted.criteria and [c.to_dict() for c in extracted.criteria] != [c.to_dict() for c in receipt_criteria]:
                notes.append("receipt_manifest_does_not_match_receipt_goal")
    elif receipt_goal is not None:
        extraction = extract_criteria(receipt_goal)
        receipt_criteria = extraction.criteria
        if extraction.warnings:
            notes.extend(extraction.warnings)
    else:
        return DriftReport(
            state=UNREGISTERED,
            verdict=Verdict.FAIL,
            mission_id=mission,
            node_id=node,
            frozen_sha256=frozen_sha,
            notes=["no_receipt_supplied"],
            audited_at=stamp,
        )

    if not receipt_criteria:
        return DriftReport(
            state=UNREGISTERED,
            verdict=Verdict.FAIL,
            mission_id=mission,
            node_id=node,
            frozen_sha256=frozen_sha,
            receipt_sha256=receipt_sha,
            notes=notes + ["receipt_unregistered"],
            audited_at=stamp,
        )

    frozen_criteria = criteria_of(frozen_manifest)
    pairs, dropped, added = _pair(frozen_criteria, receipt_criteria)

    items: list[CriterionDrift] = [compare_criterion(f, r) for f, r in pairs]
    for f in dropped:
        items.append(
            CriterionDrift(
                identity=f.identity,
                metric=f.metric,
                klass=DriftClass.LOOSENED,
                reason="criterion_dropped",
                frozen=f.to_dict(),
                receipt=None,
            )
        )
    for r in added:
        items.append(
            CriterionDrift(
                identity=r.identity,
                metric=r.metric,
                klass=DriftClass.TIGHTENED,
                reason="criterion_added",
                frozen=None,
                receipt=r.to_dict(),
            )
        )

    clean = all(i.klass in _ALLOWED for i in items) and not notes_hard_fail(notes)
    verdict = Verdict.PASS if clean else Verdict.FAIL
    return DriftReport(
        state=REGISTERED,
        verdict=verdict,
        items=items,
        mission_id=mission,
        node_id=node,
        frozen_sha256=frozen_sha,
        receipt_sha256=receipt_sha,
        notes=notes,
        audited_at=stamp,
    )


def notes_hard_fail(notes: list[str]) -> bool:
    """Receipt-integrity notes that must fail the gate regardless of classes."""
    hard = ("receipt_manifest_does_not_match_receipt_goal", "receipt_manifest_tampered")
    return any(n in hard for n in notes)


def audit_goals(
    frozen_manifest: Mapping[str, Any],
    frozen_goal: str,
    receipt_goal: str,
    **kwargs: Any,
) -> DriftReport:
    """Convenience wrapper: verify the frozen goal text, then audit the receipt."""
    state = verify_manifest(frozen_manifest, goal=frozen_goal)
    if state != REGISTERED:
        return DriftReport(
            state=state,
            verdict=Verdict.FAIL,
            mission_id=frozen_manifest.get("mission_id"),
            node_id=frozen_manifest.get("node_id"),
            frozen_sha256=str(frozen_manifest.get("manifest_sha256") or ""),
            notes=[f"frozen_goal_{state}"],
            audited_at=kwargs.get("audited_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    return audit(frozen_manifest, receipt_goal=receipt_goal, **kwargs)


# --- KPI scorecard over drift audits (aggregation only, no domain semantics) ---




def _as_dict(report: DriftReport | Mapping[str, Any]) -> dict[str, Any]:
    return report.to_dict() if isinstance(report, DriftReport) else dict(report)


def kpi_view(reports: Iterable[DriftReport | Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate audits into a KPI scorecard."""
    rows = [_as_dict(r) for r in reports]
    n_audits = len(rows)
    states = Counter(r.get("state", "") for r in rows)
    verdicts = Counter(r.get("verdict", "") for r in rows)
    classes = Counter()
    reasons = Counter()
    n_items = 0
    n_drift = 0
    nodes: dict[str, list[str]] = {}
    for r in rows:
        n_items += int(r.get("n_items", 0))
        n_drift += int(r.get("n_drift", 0))
        for item in r.get("items", []):
            classes[item.get("class", "")] += 1
            reasons[item.get("reason", "")] += 1
        node = str(r.get("node_id") or "")
        if node:
            nodes.setdefault(node, []).append(str(r.get("verdict", "")))

    for cls in DriftClass:
        classes.setdefault(cls.value, 0)
    n_pass = verdicts.get(Verdict.PASS.value, 0)
    n_registered = states.get(REGISTERED, 0)
    fail_closed = sum(v for k, v in states.items() if k and k != REGISTERED)
    return {
        "schema": "criterion-prereg-kpi/1",
        "n_audits": n_audits,
        "n_registered": n_registered,
        "prereg_coverage": (n_registered / n_audits) if n_audits else 0.0,
        "n_pass": n_pass,
        "audit_pass_rate": (n_pass / n_audits) if n_audits else 0.0,
        "n_items": n_items,
        "n_drift": n_drift,
        "drift_rate": (n_drift / n_items) if n_items else 0.0,
        "class_histogram": dict(sorted(classes.items())),
        "reason_histogram": dict(sorted(reasons.items())),
        "state_histogram": dict(sorted(states.items())),
        "verdict_histogram": dict(sorted(verdicts.items())),
        "fail_closed_incidents": fail_closed,
        "fail_closed_rate": (fail_closed / n_audits) if n_audits else 0.0,
        "nodes": {k: {"n": len(v), "n_fail": sum(1 for x in v if x == Verdict.FAIL.value)} for k, v in sorted(nodes.items())},
    }


__all__ = [
    "EVENT_SCHEMA",
    "DriftClass",
    "Verdict",
    "CriterionDrift",
    "DriftReport",
    "compare_criterion",
    "audit",
    "audit_goals",
    "notes_hard_fail",
    "kpi_view",
]