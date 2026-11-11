"""Capability-aware router with evidence-ordered fallback and decision journal.

Routing semantics (the contract the tests pin down):

1. The requested (``preferred``) solver is *never* trusted on its name alone --
   it goes through the same evidence gate as any other candidate.
2. Under the default ``evidence_first`` preference, candidates are examined in
   descending evidence rank; the requested solver only breaks ties inside one
   grade.  A refused preferred solver therefore degrades to the best-graded
   admissible alternative, never to an arbitrary or placeholder one.
3. A request that names a ``NOT_SUPPORTED`` implementation is refused outright
   (``EVIDENCE_NOT_SUPPORTED``): outage of an admissible solver degrades along
   the evidence order, absence of a capability does not.
4. If no candidate survives the gate the router raises
   :class:`~.gate.RoutingRejected` (code ``NO_ADMISSIBLE_SOLVER``) -- the caller
   gets an explicit error, not a fabricated result.
5. Every decision is journalled with the grade of every candidate examined, the
   policy tier, and the measured gate latency, so an audit can answer "which
   evidence level produced this run?" without re-running anything.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .evidence import AdmissibilityPolicy, EvidenceLevel
from .gate import EvidenceGate, RoutingRejected
from .ledger import CapabilityLedger

__all__ = [
    "GATE_LATENCY_BUDGET_MS",
    "RoutingDecision",
    "DecisionJournal",
    "CapabilityAwareRouter",
    "latency_stats",
    "describe_latency",
]

#: Acceptance threshold: the evidence gate may add at most this per route.
GATE_LATENCY_BUDGET_MS = 50.0


@dataclass(frozen=True)
class RoutingDecision:
    """One recorded routing decision (selected solver + full evidence trail)."""

    sequence: int
    capability_id: str
    preferred: "str | None"
    selected: "str | None"
    level: "EvidenceLevel | None"
    admitted: bool
    code: str
    examined: tuple[tuple[str, "str | None", str], ...]
    unavailable: tuple[str, ...]
    tier: str
    preference: str
    gate_latency_ms: float
    total_latency_ms: float
    ledger_fingerprint: str
    decided_at: str
    mock_consumed: bool = False
    provisional: bool = False

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "capability_id": self.capability_id,
            "preferred": self.preferred,
            "selected": self.selected,
            "level": self.level.value if self.level else None,
            "admitted": self.admitted,
            "code": self.code,
            "examined": [{"solver_id": s, "level": lv, "code": c} for s, lv, c in self.examined],
            "unavailable": list(self.unavailable),
            "tier": self.tier,
            "preference": self.preference,
            "gate_latency_ms": self.gate_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "ledger_fingerprint": self.ledger_fingerprint,
            "decided_at": self.decided_at,
            "mock_consumed": self.mock_consumed,
            "provisional": self.provisional,
        }


class DecisionJournal:
    """Append-only in-memory log of routing decisions with optional JSONL sink."""

    def __init__(self, path: "str | Path | None" = None) -> None:
        self.path = Path(path) if path is not None else None
        self._decisions: list[RoutingDecision] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def record(self, decision: RoutingDecision) -> RoutingDecision:
        self._decisions.append(decision)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision.as_dict(), ensure_ascii=False) + "\n")
        return decision

    @property
    def decisions(self) -> tuple[RoutingDecision, ...]:
        return tuple(self._decisions)

    def __len__(self) -> int:
        return len(self._decisions)

    def as_dicts(self) -> list[dict]:
        return [d.as_dict() for d in self._decisions]


class CapabilityAwareRouter:
    """Routes solver requests through the evidence gate, ranking by grade."""

    def __init__(self, ledger: CapabilityLedger,
                 policy: "AdmissibilityPolicy | None" = None,
                 journal: "DecisionJournal | None" = None,
                 clock: Callable[[], float] = time.perf_counter,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 gate: "EvidenceGate | None" = None) -> None:
        self.ledger = ledger
        self.policy = policy or AdmissibilityPolicy()
        self.journal = journal if journal is not None else DecisionJournal()
        self.clock = clock
        self.now = now
        self.gate = gate or EvidenceGate(ledger, self.policy)
        if self.gate.policy != self.policy:
            raise ValueError("gate policy must match the router policy")
        self._sequence = 0

    # ------------------------------------------------------------------ order
    def candidate_order(self, capability_id: str, preferred: "str | None" = None,
                        unavailable: Iterable[str] = ()) -> tuple[str, ...]:
        """Candidate solver ids in examination order (evidence rank first)."""
        ranked = self.ledger.ranked(capability_id, unavailable)
        order = [r.solver_id for r in ranked]
        if preferred is not None and preferred in order:
            order.remove(preferred)
            if self.policy.preference == "preferred_first":
                order.insert(0, preferred)
            else:
                # evidence_first: only promote inside the best grade present
                best_rank = ranked[0].level.rank
                promoted = next(r for r in ranked if r.solver_id == preferred)
                if promoted.level.rank == best_rank:
                    order.insert(0, preferred)
                else:
                    order.append(preferred)
        return tuple(order)

    # ------------------------------------------------------------------ route
    def route(self, capability_id: str, preferred: "str | None" = None,
              unavailable: Iterable[str] = (), context: "Mapping | None" = None
              ) -> RoutingDecision:
        started = self.clock()
        excluded = tuple(sorted({unavailable} if isinstance(unavailable, str)
                                else set(unavailable)))
        # Hard refusal: asking for a NOT_SUPPORTED implementation never degrades
        # to a lower-graded sibling -- that would be the silent-placeholder bug
        # this layer exists to remove.  (Fallback applies to *outages* of an
        # admissible solver, not to capabilities that are not implemented.)
        requested = self.ledger.record(capability_id, preferred) if preferred else None
        if requested is not None and requested.level is EvidenceLevel.NOT_SUPPORTED:
            total_ms = (self.clock() - started) * 1e3
            self._sequence += 1
            examined = ((requested.solver_id, requested.level.value,
                         "EVIDENCE_NOT_SUPPORTED"),)
            self.journal.record(RoutingDecision(
                sequence=self._sequence,
                capability_id=capability_id,
                preferred=preferred,
                selected=None,
                level=requested.level,
                admitted=False,
                code="EVIDENCE_NOT_SUPPORTED",
                examined=examined,
                unavailable=excluded,
                tier=self.policy.tier,
                preference=self.policy.preference,
                gate_latency_ms=0.0,
                total_latency_ms=total_ms,
                ledger_fingerprint=self.ledger.fingerprint(),
                decided_at=self.now().isoformat(timespec="seconds"),
            ))
            raise RoutingRejected(
                "EVIDENCE_NOT_SUPPORTED",
                f"capability {capability_id!r} requested solver {preferred!r} is graded "
                "NOT_SUPPORTED; refusing instead of degrading to a placeholder result",
                capability_id=capability_id,
                solver_id=preferred,
                level=requested.level,
                examined=examined,
            )
        order = self.candidate_order(capability_id, preferred, excluded)
        gate_ms = 0.0
        examined: list[tuple[str, "str | None", str]] = []
        chosen = None
        for solver_id in order:
            t0 = self.clock()
            admission = self.gate.admit(capability_id, solver_id, excluded)
            gate_ms += (self.clock() - t0) * 1e3
            examined.append((solver_id, admission.level.value if admission.level else None,
                             admission.code))
            if admission.admitted:
                chosen = admission
                break
        total_ms = (self.clock() - started) * 1e3
        self._sequence += 1
        if chosen is None:
            code = ("CAPABILITY_UNKNOWN" if not self.ledger.candidates(capability_id)
                    else "NO_ADMISSIBLE_SOLVER")
            decision = RoutingDecision(
                sequence=self._sequence,
                capability_id=capability_id,
                preferred=preferred,
                selected=None,
                level=None,
                admitted=False,
                code=code,
                examined=tuple(examined),
                unavailable=excluded,
                tier=self.policy.tier,
                preference=self.policy.preference,
                gate_latency_ms=gate_ms,
                total_latency_ms=total_ms,
                ledger_fingerprint=self.ledger.fingerprint(),
                decided_at=self.now().isoformat(timespec="seconds"),
            )
            self.journal.record(decision)
            raise RoutingRejected(
                code,
                f"no admissible implementation for capability {capability_id!r} under "
                f"tier={self.policy.tier} (examined {[e[0] for e in examined] or 'none'}); "
                "refusing instead of returning a placeholder result",
                capability_id=capability_id,
                solver_id=preferred,
                examined=examined,
            )
        decision = RoutingDecision(
            sequence=self._sequence,
            capability_id=capability_id,
            preferred=preferred,
            selected=chosen.solver_id,
            level=chosen.level,
            admitted=True,
            code=chosen.code,
            examined=tuple(examined),
            unavailable=excluded,
            tier=self.policy.tier,
            preference=self.policy.preference,
            gate_latency_ms=gate_ms,
            total_latency_ms=total_ms,
            ledger_fingerprint=self.ledger.fingerprint(),
            decided_at=self.now().isoformat(timespec="seconds"),
            mock_consumed=chosen.mock_consumed,
            provisional=chosen.provisional,
        )
        return self.journal.record(decision)

    # ------------------------------------------------------------- reporting
    def latency_stats(self) -> dict:
        return latency_stats(self.journal.decisions)

    def summary(self) -> dict:
        decisions = self.journal.decisions
        by_code: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for decision in decisions:
            by_code[decision.code] = by_code.get(decision.code, 0) + 1
            key = decision.level.value if decision.level else "none"
            by_level[key] = by_level.get(key, 0) + 1
        return {
            "policy": self.policy.as_dict(),
            "decisions": len(decisions),
            "selected_total": sum(1 for d in decisions if d.admitted),
            "refused_total": sum(1 for d in decisions if not d.admitted),
            "by_code": dict(sorted(by_code.items())),
            "by_level": dict(sorted(by_level.items())),
            "latency": self.latency_stats(),
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    frac = position - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def latency_stats(decisions: Iterable[RoutingDecision],
                  budget_ms: float = GATE_LATENCY_BUDGET_MS) -> dict:
    """Latency distribution of the gate (and of the full route) in milliseconds."""
    gate = sorted(d.gate_latency_ms for d in decisions)
    total = sorted(d.total_latency_ms for d in decisions)
    n = len(gate)
    return {
        "n": n,
        "budget_ms": budget_ms,
        "gate": describe_latency(gate),
        "total": describe_latency(total),
        "gate_max_ms": gate[-1] if gate else 0.0,
        "gate_over_budget": sum(1 for value in gate if value > budget_ms),
        "gate_within_budget": all(value <= budget_ms for value in gate),
    }


def describe_latency(values: list[float]) -> dict:
    return {
        "mean_ms": (sum(values) / len(values)) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": values[-1] if values else 0.0,
    }