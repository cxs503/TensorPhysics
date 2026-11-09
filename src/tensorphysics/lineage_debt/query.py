"""Query surface over the per-wave debt ledgers.

The campaign needs to *ask* the ledger questions without re-running the replay:
"which debts of wave 6 are still open?", "what did plan 173 drop?", "which
domain carries the most unpaid error?".  All filters are conjunctive and every
result set is sorted, so a query is as reproducible as the ledger itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["load_ledger", "query", "coverage_table", "gate_report"]


def load_ledger(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def query(ledger: Mapping[str, Any], *, plan_id: int | None = None, state: str | None = None,
          coverage: str | None = None, domain: str | None = None, quantity: str | None = None,
          source_kind: str | None = None, min_abs_err_pct: float | None = None,
          node_id: int | None = None, mission_id: int | None = None,
          report_id: int | None = None) -> list[dict]:
    """Conjunctive filter over a ledger's debts (sorted by key, then id)."""
    out: list[dict] = []
    for debt in ledger.get("debts") or []:
        if state is not None and debt.get("state") != state:
            continue
        if coverage is not None and debt.get("coverage") != coverage:
            continue
        if domain is not None and debt.get("domain") != domain:
            continue
        if quantity is not None and debt.get("quantity") != quantity:
            continue
        if source_kind is not None and debt.get("source_kind") != source_kind:
            continue
        if node_id is not None and debt.get("node_id") != node_id:
            continue
        if mission_id is not None and debt.get("mission_id") != mission_id:
            continue
        if report_id is not None and debt.get("report_id") != report_id:
            continue
        if min_abs_err_pct is not None:
            val = debt.get("abs_err_pct")
            if val is None or float(val) < float(min_abs_err_pct):
                continue
        if plan_id is not None:
            steps = debt.get("lineage") or []
            if not any(step.get("kind") == "plan" and step.get("plan_id") == plan_id for step in steps):
                continue
        out.append(debt)
    out.sort(key=lambda d: (str(d.get("key")), str(d.get("id"))))
    return out


def coverage_table(ledger: Mapping[str, Any]) -> list[dict]:
    """One flat row per debt x coverage outcome (the matrix as a table)."""
    rows = []
    for row in (ledger.get("coverage_matrix") or {}).get("matrix") or []:
        rows.append({"debt_key": row.get("debt_key"), "debt_id": row.get("debt_id"),
                     "coverage": row.get("coverage"),
                     "matched_criteria": list(row.get("matched_criteria") or [])})
    rows.sort(key=lambda r: (str(r["debt_key"]), str(r["debt_id"])))
    return rows


def gate_report(ledgers: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Gate verdict per plan (for the campaign dashboard)."""
    out = []
    for led in ledgers:
        gate = led.get("seed_gate") or {}
        out.append({
            "plan_id": led.get("plan_id"),
            "verdict": gate.get("verdict"),
            "missing": len(gate.get("missing") or []),
            "counts": dict(gate.get("counts") or {}),
        })
    out.sort(key=lambda r: (str(r.get("plan_id"))))
    return out