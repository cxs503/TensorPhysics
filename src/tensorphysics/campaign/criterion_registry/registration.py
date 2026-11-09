"""Fail-closed registration and event-stream auditing of criterion manifests.

Two entry points make the freeze→audit cycle operational for campaign nodes:

``register_goal(goal_text, frozen_by)``
    Extract the structural criteria of a node goal and freeze each one
    (canonical JSON -> sha256 -> metadata).  The registration is *fail-closed*:
    a goal with no numeric level, an empty string, ``None`` or a non-string
    payload registers **nothing** and is reported as ``unregistered`` -- it can
    never be mistaken for "criterion satisfied".

``audit_delivery(registrations, report_text, mission_id, ledger)``
    Re-extract the criteria from the delivered report, classify drift against
    the frozen manifests field by field, append one event per criterion to the
    drift ledger (the event stream) and return the KPI view
    (``registered_rate`` / ``loosened_rate`` / ``unregistered_rate``).

Both helpers are domain-agnostic: no domain vocabulary is embedded here, all
domain content arrives through the goal / report text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .drift import DriftResult, classify_drift
from .kpi import KPIView, compute_kpi
from .ledger import CriterionLedger, record_drift_event
from .manifest import (
    CriterionManifest,
    extract_criteria_from_goal,
    freeze_manifest,
)


UNREGISTERED = "unregistered"


@dataclass
class Registration:
    """One pre-registered criterion (or an explicit registration failure)."""

    criterion_id: str
    status: str  # 'registered' | 'unregistered'
    manifest: Optional[CriterionManifest] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status == UNREGISTERED and not self.reason:
            self.reason = "no numeric level could be extracted (fail-closed)"

    @property
    def is_registered(self) -> bool:
        return self.status == "registered" and self.manifest is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "status": self.status,
            "reason": self.reason,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


def register_goal(
    goal_text: Any,
    frozen_by: str,
    frozen_at: Optional[Any] = None,
) -> list[Registration]:
    """Extract and freeze all criteria of a goal (fail-closed).

    An empty / ``None`` / non-string goal, or a goal whose blocks carry no
    numeric level, yields a single ``unregistered`` registration and *no*
    frozen manifest.
    """
    criteria = extract_criteria_from_goal(goal_text)
    if not criteria:
        return [Registration(criterion_id="criterion_000", status=UNREGISTERED)]

    registrations: list[Registration] = []
    for crit in criteria:
        frozen = freeze_manifest(crit, frozen_by=frozen_by, frozen_at=frozen_at)
        registrations.append(Registration(
            criterion_id=frozen.criterion_id,
            status="registered",
            manifest=frozen,
        ))
    return registrations


@dataclass
class DeliveryAudit:
    """Result of auditing one delivered report against frozen criteria."""

    mission_id: str
    results: list[DriftResult] = field(default_factory=list)
    kpi: Optional[KPIView] = None

    @property
    def failed_closed(self) -> bool:
        """True when at least one criterion is unregistered/tampered/undefined."""
        return any(r.classification in (UNREGISTERED, "tampered", "undefined")
                   for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "failed_closed": self.failed_closed,
            "results": [r.to_dict() for r in self.results],
            "kpi": self.kpi.to_dict() if self.kpi else None,
        }


def audit_delivery(
    registrations: Sequence[Registration],
    report_text: Any,
    mission_id: str,
    ledger: Optional[CriterionLedger] = None,
) -> DeliveryAudit:
    """Audit a delivered report and append drift events to the ledger.

    Fail-closed rules:
    * an unregistered registration is recorded as an ``unregistered`` event;
    * a frozen criterion missing from the report is recorded as
      ``unregistered`` (never ``unchanged``);
    * an unparseable report makes every frozen criterion ``unregistered``.
    """
    current = extract_criteria_from_goal(report_text)
    by_id = {c.criterion_id: c for c in current}
    ledger = ledger if ledger is not None else CriterionLedger()

    audit = DeliveryAudit(mission_id=mission_id)
    for registration in registrations:
        if not registration.is_registered:
            result = DriftResult(
                criterion_id=registration.criterion_id,
                classification=UNREGISTERED,
                reason=registration.reason,
            )
        else:
            frozen = registration.manifest
            assert frozen is not None
            candidate = by_id.get(frozen.criterion_id)
            if candidate is None:
                if current:
                    # Positional fallback preserves the original intent, but
                    # only when both sides are parseable and same cardinality.
                    positional = _positional_match(registrations, current, frozen.criterion_id)
                    candidate = positional
                if candidate is None:
                    result = DriftResult(
                        criterion_id=frozen.criterion_id,
                        classification=UNREGISTERED,
                        reason="criterion missing from report (fail-closed)",
                    )
                else:
                    result = classify_drift(frozen, candidate)
            else:
                result = classify_drift(frozen, candidate)

        audit.results.append(result)
        record_drift_event(
            ledger,
            mission_id=mission_id,
            criterion_id=result.criterion_id,
            classification=result.classification,
            reason=result.reason,
            metadata={"field_diffs": result.field_diffs},
        )

    audit.kpi = compute_kpi(ledger, mission_id=mission_id)
    return audit


def _positional_match(
    registrations: Sequence[Registration],
    current: Sequence[CriterionManifest],
    criterion_id: str,
) -> Optional[CriterionManifest]:
    order = [r.criterion_id for r in registrations if r.is_registered]
    if len(current) != len(order):
        return None
    try:
        return current[order.index(criterion_id)]
    except ValueError:
        return None