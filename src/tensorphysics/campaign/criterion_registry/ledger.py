"""Criterion drift event ledger.

Records drift events (loosened, scope_changed, etc.) with human-readable
reasons. Supports SQL-style queries for KPI aggregation.

Fail-closed: events with missing required fields are rejected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterator


@dataclass
class CriterionEvent:
    """A single drift event in the ledger.

    Attributes:
        event_id: unique event identifier
        mission_id: the mission/campaign this event belongs to
        criterion_id: the criterion that drifted
        classification: drift type (unchanged, tightened, loosened, scope_changed, undefined, unregistered, tampered)
        reason: human-readable explanation (required for loosened/scope_changed)
        timestamp: ISO timestamp of the event
        metadata: additional structured data
    """
    event_id: str
    mission_id: str
    criterion_id: str
    classification: str
    reason: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail-closed: loosened/scope_changed must have a reason
        if self.classification in ("loosened", "scope_changed") and not self.reason:
            raise ValueError(
                f"classification '{self.classification}' requires a non-empty reason"
            )
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionEvent":
        return cls(**data)


class CriterionLedger:
    """Append-only ledger of criterion drift events.

    Supports:
    - Appending events
    - Querying by mission_id, classification, time range
    - SQL-style aggregation for KPI computation
    - Persistence to JSON-lines file
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._events: list[CriterionEvent] = []
        self._storage_path = storage_path
        if storage_path and storage_path.exists():
            self._load()

    def _load(self) -> None:
        """Load events from JSON-lines file."""
        if not self._storage_path:
            return
        with open(self._storage_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self._events.append(CriterionEvent.from_dict(data))

    def _persist(self, event: CriterionEvent) -> None:
        """Append event to storage file."""
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def append(self, event: CriterionEvent) -> None:
        """Append an event to the ledger."""
        self._events.append(event)
        self._persist(event)

    def query(
        self,
        mission_id: Optional[str] = None,
        classification: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> list[CriterionEvent]:
        """Query events with optional filters.

        Args:
            mission_id: filter by mission
            classification: filter by drift classification
            since: ISO timestamp lower bound (inclusive)
            until: ISO timestamp upper bound (inclusive)

        Returns:
            List of matching events.
        """
        results = []
        for event in self._events:
            if mission_id is not None and event.mission_id != mission_id:
                continue
            if classification is not None and event.classification != classification:
                continue
            if since is not None and event.timestamp < since:
                continue
            if until is not None and event.timestamp > until:
                continue
            results.append(event)
        return results

    def __iter__(self) -> Iterator[CriterionEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def to_list(self) -> list[dict[str, Any]]:
        """Export all events as list of dicts."""
        return [e.to_dict() for e in self._events]


def record_drift_event(
    ledger: CriterionLedger,
    mission_id: str,
    criterion_id: str,
    classification: str,
    reason: Optional[str] = None,
    event_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> CriterionEvent:
    """Convenience function: create and record a drift event.

    Args:
        ledger: The ledger to append to
        mission_id: mission identifier
        criterion_id: criterion identifier
        classification: drift classification
        reason: human-readable reason (required for loosened/scope_changed)
        event_id: optional explicit event id (auto-generated if None)
        metadata: optional additional data

    Returns:
        The created CriterionEvent.
    """
    import uuid
    if event_id is None:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"

    event = CriterionEvent(
        event_id=event_id,
        mission_id=mission_id,
        criterion_id=criterion_id,
        classification=classification,
        reason=reason,
        metadata=metadata or {},
    )
    ledger.append(event)
    return event


def query_kpi(
    ledger: CriterionLedger,
    mission_id: Optional[str] = None,
) -> dict[str, Any]:
    """Query KPI summary from ledger.

    Args:
        ledger: The ledger to query
        mission_id: optional mission filter

    Returns:
        Dict with counts by classification.
    """
    events = ledger.query(mission_id=mission_id)
    counts: dict[str, int] = {}
    for event in events:
        cls = event.classification
        counts[cls] = counts.get(cls, 0) + 1

    total = len(events)
    return {
        "total_events": total,
        "counts_by_classification": counts,
        "mission_id": mission_id,
    }
