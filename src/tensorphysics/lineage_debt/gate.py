"""Fail-closed seeding gate: an unpaid debt cannot be silently inherited.

The gate answers one question before a new wave of child nodes is admitted:
*for every open debt of the lineage, is there either a descendant criterion
that re-measures it, or a written reason why it is deferred?*  If not, seeding
is **blocked** and the verdict enumerates the missing items one by one, so the
arbiter receives a work list instead of a silent inheritance.

Three machine surfaces come out of the same verdict:

``verdict``
    ``"pass"`` / ``"blocked"`` plus counts -- the machine-read gate.
``missing``
    the enumeration of unpaid debts (debt id, key, domain, quantity, evidence
    pointer, and the reason that is required).
``autonomy_patch``
    ``{"autonomy": {"debt_unpaid": [...]}}`` -- mergeable into a plan's
    metadata when the arbiter explicitly wants to carry the debt forward
    instead of blocking (the escape hatch stays *auditable*, not silent).

Settled debts (state ``debt_closed``) are exempt: closure is earned by a later
in-threshold observation, never by a claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import VocabConfig
from .model import DebtItem

__all__ = ["SeedingBlocked", "SeedingVerdict", "seeding_check", "enforce", "apply_autonomy_patch"]

GATE_SCHEMA = "lineage_debt_seed_gate.v1"


class SeedingBlocked(RuntimeError):
    """Raised by :func:`enforce` when the seeding gate refuses to admit a wave."""

    def __init__(self, verdict: "SeedingVerdict"):
        self.verdict = verdict
        super().__init__(f"seeding blocked for plan {verdict.plan_id}: "
                         f"{len(verdict.missing)} unpaid debt(s) without criterion or reason")


@dataclass(frozen=True)
class SeedingVerdict:
    plan_id: int | None
    verdict: str
    missing: tuple[dict, ...]
    settled: tuple[dict, ...]
    weak: tuple[dict, ...]
    counts: Mapping[str, Any]
    autonomy_patch: Mapping[str, Any]

    @property
    def blocked(self) -> bool:
        return self.verdict == "blocked"

    def as_dict(self) -> dict:
        return {
            "schema": GATE_SCHEMA,
            "plan_id": self.plan_id,
            "verdict": self.verdict,
            "missing": [dict(m) for m in self.missing],
            "weak": [dict(w) for w in self.weak],
            "settled": [dict(s) for s in self.settled],
            "counts": dict(self.counts),
            "autonomy_patch": dict(self.autonomy_patch),
        }


def _missing_entry(debt: DebtItem) -> dict:
    evidence = debt.evidence[0] if debt.evidence else {}
    return {
        "debt_id": debt.id,
        "debt_key": debt.key,
        "source_kind": debt.source_kind,
        "domain": debt.domain,
        "quantity": debt.quantity,
        "condition": debt.condition,
        "state": debt.state,
        "report_id": debt.report_id,
        "node_id": debt.node_id,
        "mission_id": debt.mission_id,
        "err_pct": None if debt.err_pct is None else round(float(debt.err_pct), 6),
        "evidence_ref": {
            "report_id": evidence.get("report_id"),
            "line_no": evidence.get("line_no"),
            "report_sha256": evidence.get("report_sha256"),
        },
        "required": "a descendant criterion re-measuring this quantity, an explicit evidence "
                    "reference, or a written deferral reason",
    }


def seeding_check(plan_id: int | None, debts: list[DebtItem], matrix: Mapping[str, Any],
                  cfg: VocabConfig) -> SeedingVerdict:
    """Fail-closed verdict over the coverage matrix."""
    dropped_name = cfg.coverage.get("dropped", "dropped")
    open_state = cfg.states.get("open", "debt_open")
    closed_state = cfg.states.get("closed", "debt_closed")

    missing: list[dict] = []
    settled: list[dict] = []
    for debt in debts:
        if debt.state == closed_state:
            settled.append({"debt_id": debt.id, "debt_key": debt.key,
                            "closed_by": debt.closed_by, "state": debt.state})
            continue
        if debt.state != open_state:
            continue
        reason = (debt.deferred_reason or "").strip()
        if debt.coverage == dropped_name and not reason:
            missing.append(_missing_entry(debt))
    missing.sort(key=lambda m: (str(m["debt_key"]), str(m["debt_id"])))
    settled.sort(key=lambda s: (str(s["debt_key"]), str(s["debt_id"])))

    weak = [{"debt_id": d.id, "debt_key": d.key} for d in debts
            if d.state == open_state and d.coverage == cfg.coverage.get("covered_by_domain", "covered_by_domain")]
    weak.sort(key=lambda w: (str(w["debt_key"]), str(w["debt_id"])))
    counts = {
        "debts_total": len(debts),
        "open": sum(1 for d in debts if d.state == open_state),
        "settled": len(settled),
        "missing": len(missing),
        "weak": len(weak),
        "coverage": dict(sorted((matrix.get("counts") or {}).items())),
        "rules": dict(sorted((matrix.get("rules") or {}).items())),
        "criteria_total": matrix.get("criteria_total", 0),
    }
    verdict = "blocked" if missing else "pass"
    patch: dict[str, Any] = {"autonomy": {"debt_unpaid": [m["debt_key"] for m in missing]} if missing else {"debt_unpaid": []}}
    patch["autonomy"]["debt_unpaid_gate"] = {
        "verdict": verdict,
        "plan_id": plan_id,
        "count": len(missing),
        "schema": GATE_SCHEMA,
    }
    return SeedingVerdict(plan_id=plan_id, verdict=verdict, missing=tuple(missing),
                          settled=tuple(settled), weak=tuple(weak), counts=counts,
                          autonomy_patch=patch)


def enforce(plan_id: int | None, debts: list[DebtItem], matrix: Mapping[str, Any],
            cfg: VocabConfig) -> SeedingVerdict:
    """Raise :class:`SeedingBlocked` instead of returning a blocked verdict."""
    verdict = seeding_check(plan_id, debts, matrix, cfg)
    if verdict.blocked:
        raise SeedingBlocked(verdict)
    return verdict


def apply_autonomy_patch(plan_metadata: Mapping[str, Any] | None, verdict: SeedingVerdict) -> dict:
    """Merge the verdict into a copy of plan metadata (never mutates the input)."""
    meta = dict(plan_metadata or {})
    autonomy = dict(meta.get("autonomy") or {})
    patch = dict(verdict.autonomy_patch.get("autonomy") or {})
    autonomy.update(patch)
    meta["autonomy"] = autonomy
    return meta