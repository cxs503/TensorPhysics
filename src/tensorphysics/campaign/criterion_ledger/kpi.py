"""Ledger KPIs: per-gate kill rate, weakest mutants, monotonic curve,
falsifiability marking.

All functions are pure: they take audit result dicts (the JSON shape written
by :mod:`...criterion_ledger.runner`) and return numbers.  No I/O, no
domain semantics.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def kill_rate(res: dict) -> float:
    """Killed / total of one gate audit result (0.0 for an empty audit)."""
    total = res.get("total", 0)
    return round(res.get("killed", 0) / total, 4) if total else 0.0


def weakest_mutants(res: dict, n: int = 20) -> List[dict]:
    """Surviving mutants, ordered by (class, line) -- the gate's blind spots.

    "Weakest" = the mutants the suite failed to kill; the list is the
    follow-up work queue (write killer tests against exactly these).
    """
    survivors = res.get("survivors")
    if survivors is None:
        survivors = [m for m in res.get("mutants", []) if not m.get("killed")]
    ordered = sorted(survivors, key=lambda m: (m.get("op", ""), m.get("line", 0)))
    return ordered[:n]


def monotonic_kill_curve(killed_by_eps: Dict[float, int]) -> dict:
    """Check the M1 kill curve: kill count must be non-decreasing in eps.

    ``killed_by_eps`` maps perturbation level (0.01 / 0.03 / 0.15) to the
    number of mutants killed at that level.  A larger perturbation must
    never kill *fewer* mutants than a smaller one: a threshold-based test
    that catches a small deviation catches the larger one too.
    """
    eps_sorted = sorted(killed_by_eps)
    counts = [killed_by_eps[e] for e in eps_sorted]
    non_decreasing = all(b >= a for a, b in zip(counts, counts[1:]))
    return {
        "eps": eps_sorted,
        "killed": counts,
        "non_decreasing": non_decreasing,
        "monotonic": non_decreasing,
    }


def falsifiability(res: dict) -> dict:
    """Falsifiability mark for one gate audit.

    A gate whose kill rate is exactly 0 (no mutant killed) is *unfalsifiable*
    as tested: its test suite cannot distinguish the gate from a tautology.
    """
    rate = kill_rate(res)
    return {
        "kill_rate": rate,
        "falsifiable": rate > 0.0,
        "mark": "FALSIFIABLE" if rate > 0.0 else "UNFALSIFIABLE",
    }


def ledger_kpi(per_gate: Dict[str, dict]) -> dict:
    """Full-ledger KPI: per-gate kill rate + weakest-mutant lists + overall."""
    gates: Dict[str, dict] = {}
    for gid, res in per_gate.items():
        gates[gid] = {
            "total": res.get("total", 0),
            "killed": res.get("killed", 0),
            "survived": res.get("survived", 0),
            "kill_rate": kill_rate(res),
            "falsifiability": falsifiability(res),
            "weakest_mutants": weakest_mutants(res, n=10),
        }
    total = sum(g["total"] for g in gates.values())
    killed = sum(g["killed"] for g in gates.values())
    return {
        "gates": len(gates),
        "total_mutants": total,
        "killed": killed,
        "overall_kill_rate": round(killed / total, 4) if total else 0.0,
        "per_gate": gates,
    }
