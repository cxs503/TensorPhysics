"""KPI view for criterion pre-registration mechanism.

Provides:
- registered_rate: fraction of criteria that were frozen before execution
- loosened_rate: fraction of frozen criteria that were loosened after execution
- unregistered_rate: fraction of criteria that could not be parsed/registered

These KPIs are for the arbitrator to assess node quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .ledger import CriterionLedger


@dataclass
class KPIView:
    """KPI summary for criterion pre-registration.

    Attributes:
        total_criteria: total number of criteria encountered
        registered: number of criteria that were frozen (registered)
        unregistered: number of criteria that could not be parsed
        unchanged: number of frozen criteria that remained unchanged
        tightened: number of frozen criteria that were tightened
        loosened: number of frozen criteria that were loosened
        scope_changed: number of frozen criteria with scope changes
        tampered: number of frozen criteria with hash mismatch
        undefined: number of criteria with undefined drift
    """
    total_criteria: int = 0
    registered: int = 0
    unregistered: int = 0
    unchanged: int = 0
    tightened: int = 0
    loosened: int = 0
    scope_changed: int = 0
    tampered: int = 0
    undefined: int = 0

    @property
    def registered_rate(self) -> float:
        """Fraction of criteria that were registered."""
        if self.total_criteria == 0:
            return 0.0
        return self.registered / self.total_criteria

    @property
    def unregistered_rate(self) -> float:
        """Fraction of criteria that were unregistered."""
        if self.total_criteria == 0:
            return 0.0
        return self.unregistered / self.total_criteria

    @property
    def loosened_rate(self) -> float:
        """Fraction of registered criteria that were loosened."""
        if self.registered == 0:
            return 0.0
        return self.loosened / self.registered

    @property
    def tightened_rate(self) -> float:
        """Fraction of registered criteria that were tightened."""
        if self.registered == 0:
            return 0.0
        return self.tightened / self.registered

    @property
    def scope_changed_rate(self) -> float:
        """Fraction of registered criteria with scope changes."""
        if self.registered == 0:
            return 0.0
        return self.scope_changed / self.registered

    @property
    def tampered_rate(self) -> float:
        """Fraction of registered criteria that were tampered."""
        if self.registered == 0:
            return 0.0
        return self.tampered / self.registered

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_criteria": self.total_criteria,
            "registered": self.registered,
            "unregistered": self.unregistered,
            "unchanged": self.unchanged,
            "tightened": self.tightened,
            "loosened": self.loosened,
            "scope_changed": self.scope_changed,
            "tampered": self.tampered,
            "undefined": self.undefined,
            "registered_rate": self.registered_rate,
            "unregistered_rate": self.unregistered_rate,
            "loosened_rate": self.loosened_rate,
            "tightened_rate": self.tightened_rate,
            "scope_changed_rate": self.scope_changed_rate,
            "tampered_rate": self.tampered_rate,
        }


def compute_kpi(
    ledger: CriterionLedger,
    mission_id: Optional[str] = None,
) -> KPIView:
    """Compute KPI view from ledger events.

    Args:
        ledger: The criterion ledger
        mission_id: optional mission filter

    Returns:
        KPIView with computed rates.
    """
    events = ledger.query(mission_id=mission_id)

    kpi = KPIView()
    kpi.total_criteria = len(events)

    for event in events:
        cls = event.classification
        if cls == "unregistered":
            kpi.unregistered += 1
        elif cls == "unchanged":
            kpi.registered += 1
            kpi.unchanged += 1
        elif cls == "tightened":
            kpi.registered += 1
            kpi.tightened += 1
        elif cls == "loosened":
            kpi.registered += 1
            kpi.loosened += 1
        elif cls == "scope_changed":
            kpi.registered += 1
            kpi.scope_changed += 1
        elif cls == "tampered":
            kpi.registered += 1
            kpi.tampered += 1
        elif cls == "undefined":
            kpi.registered += 1
            kpi.undefined += 1

    return kpi
