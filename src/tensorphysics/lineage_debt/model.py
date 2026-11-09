"""Data model of the lineage-inheritance debt ledger (deterministic by construction).

Two object families live here and nothing else does:

``ErrorObservation``
    one numeric error cell harvested from a report table (whether or not it
    breaks its threshold).  Keeping the *passing* rows is what makes debt
    closure machine-checkable later on: a debt can only be closed by a *later*
    observation of the same key that sits inside the threshold.

``DebtItem``
    one payable debt of the campaign lineage: a quantity (or a terminal node)
    whose honest FAIL / blocked end-state was not carried into a machine
    checkable acceptance criterion of the descendants.

Every serialisation is canonical (sorted keys, fixed float rounding, explicit
lineage steps) so that two replays of the same input are byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "LEDGER_SCHEMA",
    "ErrorObservation",
    "DebtItem",
    "canonical_json",
    "sha256_text",
    "stable_id",
]

LEDGER_SCHEMA = "lineage_debt_ledger.v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Byte-stable JSON rendering (sorted keys, no ASCII escaping, trailing \\n)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("\u241f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _r(value: float | None, nd: int = 6) -> float | None:
    return None if value is None else round(float(value), nd)


@dataclass(frozen=True)
class ErrorObservation:
    """One error cell from one report table row."""

    report_id: int | None
    plan_id: int | None
    node_id: int | None
    mission_id: int | None
    raw_quantity: str
    quantity: str
    domain: str
    condition: str
    err_pct: float
    threshold_pct: float
    exceeded: bool
    table_index: int
    line_no: int
    header: str
    raw_value: str
    report_sha256: str = ""
    verdict_markers: tuple[str, ...] = ()
    report_created_at: str = ""
    domain_source: str = "vocabulary"
    condition_source: str = "column"

    @property
    def key(self) -> str:
        return f"{self.domain}|{self.quantity}|{self.condition}"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "report_id": self.report_id,
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "mission_id": self.mission_id,
            "raw_quantity": self.raw_quantity,
            "quantity": self.quantity,
            "domain": self.domain,
            "condition": self.condition,
            "err_pct": _r(self.err_pct),
            "threshold_pct": _r(self.threshold_pct),
            "exceeded": bool(self.exceeded),
            "table_index": self.table_index,
            "line_no": self.line_no,
            "header": self.header,
            "raw_value": self.raw_value,
            "report_sha256": self.report_sha256,
            "verdict_markers": list(self.verdict_markers),
            "report_created_at": self.report_created_at,
            "domain_source": self.domain_source,
            "condition_source": self.condition_source,
        }


@dataclass
class DebtItem:
    """One payable debt in the lineage (canonical, serialisable, auditable)."""

    key: str
    domain: str
    quantity: str | None
    condition: str | None
    source_kind: str  # "error_row" | "node_terminal"
    plan_id: int | None = None
    node_id: int | None = None
    mission_id: int | None = None
    report_id: int | None = None
    err_pct: float | None = None
    threshold_pct: float | None = None
    state: str = "debt_open"
    closed_by: int | None = None
    coverage: str | None = None
    covered_by: list[dict] = field(default_factory=list)
    deferred_reason: str | None = None
    lineage: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    node_status: str | None = None

    @property
    def abs_err_pct(self) -> float | None:
        return None if self.err_pct is None else abs(self.err_pct)

    @property
    def refs(self) -> dict[str, set[int]]:
        """Identifiers a descendant criterion may cite to acknowledge the debt."""
        return {
            "report": {self.report_id} if self.report_id is not None else set(),
            "mission": {self.mission_id} if self.mission_id is not None else set(),
            "node": {self.node_id} if self.node_id is not None else set(),
            "plan": {self.plan_id} if self.plan_id is not None else set(),
        }

    @property
    def id(self) -> str:
        return stable_id(self.source_kind, self.key, self.plan_id, self.node_id, self.report_id)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "domain": self.domain,
            "quantity": self.quantity,
            "condition": self.condition,
            "source_kind": self.source_kind,
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "mission_id": self.mission_id,
            "report_id": self.report_id,
            "err_pct": _r(self.err_pct),
            "abs_err_pct": _r(self.abs_err_pct),
            "threshold_pct": _r(self.threshold_pct),
            "state": self.state,
            "closed_by": self.closed_by,
            "coverage": self.coverage,
            "covered_by": sorted(self.covered_by, key=lambda c: (str(c.get("criterion_id")),)),
            "deferred_reason": self.deferred_reason,
            "lineage": list(self.lineage),
            "evidence": sorted(self.evidence, key=lambda e: (int(e.get("report_id") or 0), int(e.get("line_no") or 0))),
            "node_status": self.node_status,
        }


def dedupe_debts(items: list[DebtItem]) -> list[DebtItem]:
    """Merge debts with the same key; worst |err| wins, evidence accumulates.

    Deterministic: output sorted by ``key`` then ``id``.
    """
    merged: dict[str, DebtItem] = {}
    for item in items:
        cur = merged.get(item.key)
        if cur is None:
            merged[item.key] = DebtItem(**{**item.__dict__,
                                           "covered_by": list(item.covered_by),
                                           "lineage": list(item.lineage),
                                           "evidence": list(item.evidence)})
            continue
        if item.abs_err_pct is not None and (cur.abs_err_pct is None or item.abs_err_pct > cur.abs_err_pct):
            cur.err_pct = item.err_pct
            cur.threshold_pct = item.threshold_pct
            cur.report_id = item.report_id if cur.report_id is None else cur.report_id
        for step in item.lineage:
            if step not in cur.lineage:
                cur.lineage.append(step)
        for ev in item.evidence:
            if ev not in cur.evidence:
                cur.evidence.append(ev)
        for ref in item.covered_by:
            if ref not in cur.covered_by:
                cur.covered_by.append(ref)
        if cur.state == "debt_closed" and item.state == "debt_open":
            cur.state = item.state
    out = list(merged.values())
    out.sort(key=lambda d: (d.key, d.id))
    for d in out:
        d.lineage.sort(key=lambda s: (str(s.get("kind")), int(s.get("plan_id") or 0),
                                      int(s.get("node_id") or 0), int(s.get("report_id") or 0)))
        d.covered_by.sort(key=lambda c: str(c.get("criterion_id")))
        d.evidence.sort(key=lambda e: (int(e.get("report_id") or 0), int(e.get("line_no") or 0)))
    return out


def summarize(debts: list[DebtItem]) -> dict:
    """Counts used by the gate and the report (deterministic ordering)."""
    by_state: dict[str, int] = {}
    by_coverage: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for d in debts:
        by_state[d.state] = by_state.get(d.state, 0) + 1
        cov = d.coverage or "unjudged"
        by_coverage[cov] = by_coverage.get(cov, 0) + 1
        by_source[d.source_kind] = by_source.get(d.source_kind, 0) + 1
        by_domain[d.domain] = by_domain.get(d.domain, 0) + 1
    return {
        "total": len(debts),
        "by_state": dict(sorted(by_state.items())),
        "by_coverage": dict(sorted(by_coverage.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_domain": dict(sorted(by_domain.items())),
    }


def lineage_steps(plan_chain: list[int], node: Mapping[str, Any] | None,
                  report_id: int | None) -> list[dict]:
    """Deterministic ancestry chain (root plan first, evidence last)."""
    steps: list[dict] = [{"kind": "plan", "plan_id": int(pid), "role": "ancestor" if i < len(plan_chain) - 1 else "owner"}
                         for i, pid in enumerate(plan_chain)]
    if node is not None:
        steps.append({
            "kind": "node",
            "plan_id": int(node.get("plan_id") or 0) or None,
            "node_id": int(node["id"]) if node.get("id") is not None else None,
            "mission_id": int(node["mission_id"]) if node.get("mission_id") is not None else None,
            "status": node.get("status"),
        })
    if report_id is not None:
        steps.append({"kind": "report", "report_id": int(report_id),
                      "node_id": int(node["id"]) if node is not None and node.get("id") is not None else None})
    return steps