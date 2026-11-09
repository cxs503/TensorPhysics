"""Fail-closed seeding gate: dropped debts without a reason must block seeding.

Acceptance criterion (c) lives here: when a debt is dropped and no reason is
recorded, the gate refuses to admit the wave and returns the enumeration of the
missing items; the plan metadata then carries ``autonomy.debt_unpaid`` so the
debt stays visible even if an arbiter overrides the block.
"""
from __future__ import annotations

import pytest
from lineage_debt_synth import SYNTH_VOCAB

from tensorphysics.lineage_debt import load_config
from tensorphysics.lineage_debt.coverage import build_matrix
from tensorphysics.lineage_debt.criteria import parse_node_criteria
from tensorphysics.lineage_debt.gate import (SeedingBlocked, apply_autonomy_patch, enforce,
                                             seeding_check)
from tensorphysics.lineage_debt.model import DebtItem


def _cfg():
    return load_config(SYNTH_VOCAB)


def _debt(key: str, domain: str, quantity: str | None, *, state: str = "debt_open",
          report_id: int = 9001) -> DebtItem:
    return DebtItem(key=key, domain=domain, quantity=quantity, condition="ALL",
                    source_kind="error_row", plan_id=1, node_id=11, mission_id=101,
                    report_id=report_id, err_pct=25.0, threshold_pct=15.0, state=state,
                    evidence=[{"report_id": report_id, "line_no": 3, "report_sha256": "deadbeef"}])


def test_c_fail_closed_enumerates_missing_items():
    """(c) a dropped debt without a reason blocks the seeding and is enumerated."""
    cfg = _cfg()
    criteria = [c for n in [{"id": 31, "plan_id": 2, "mission_id": 301, "name": "n", "status": "planned",
                             "metadata": {"acceptance_criteria": ["q1 error<10%"]}}]
                for c in parse_node_criteria(n, cfg)]
    debts = [_debt("alpha|q1|ALL", "alpha", "q1"),      # covered
             _debt("beta|q9|ALL", "beta", "q9")]        # nobody covers it
    matrix = build_matrix(debts, criteria, cfg)
    verdict = seeding_check(2, debts, matrix, cfg)
    assert verdict.verdict == "blocked" and verdict.blocked
    assert [m["debt_key"] for m in verdict.missing] == ["beta|q9|ALL"]
    missing = verdict.missing[0]
    assert missing["domain"] == "beta" and missing["quantity"] == "q9"
    assert missing["evidence_ref"]["report_id"] == 9001
    assert "reason" in missing["required"]
    assert verdict.counts["missing"] == 1 and verdict.counts["debts_total"] == 2

    with pytest.raises(SeedingBlocked) as excinfo:
        enforce(2, debts, matrix, cfg)
    assert excinfo.value.verdict.missing[0]["debt_key"] == "beta|q9|ALL"

    patched = apply_autonomy_patch({"autonomy": {"wave": 7}}, verdict)
    assert patched["autonomy"]["debt_unpaid"] == ["beta|q9|ALL"]
    assert patched["autonomy"]["wave"] == 7                       # existing keys survive
    assert patched["autonomy"]["debt_unpaid_gate"]["verdict"] == "blocked"
    assert "autonomy" not in verdict.as_dict()["counts"]           # counts stay numeric


def test_pass_when_covered_or_deferred():
    cfg = _cfg()
    criteria = [c for n in [{"id": 31, "plan_id": 2, "mission_id": 301, "name": "n", "status": "planned",
                             "metadata": {"acceptance_criteria": ["q1 error<10%", "beta surface work"]}}]
                for c in parse_node_criteria(n, cfg)]
    debts = [_debt("alpha|q1|ALL", "alpha", "q1"),
             _debt("alpha|q2|ALL", "alpha", "q2")]              # covered_by_domain only
    matrix = build_matrix(debts, criteria, cfg, deferrals={"alpha|q2|ALL": "deferred to wave 9"})
    verdict = seeding_check(2, debts, matrix, cfg)
    assert verdict.verdict == "pass" and not verdict.blocked
    assert verdict.missing == ()
    assert verdict.counts["weak"] == 1
    assert verdict.autonomy_patch["autonomy"]["debt_unpaid"] == []


def test_settled_debt_is_exempt_but_visible():
    """A debt closed by later evidence needs no criterion anymore."""
    cfg = _cfg()
    debts = [_debt("beta|q9|ALL", "beta", "q9", state="debt_closed")]
    matrix = build_matrix(debts, [], cfg)
    verdict = seeding_check(2, debts, matrix, cfg)
    assert verdict.verdict == "pass"
    assert verdict.settled[0]["debt_key"] == "beta|q9|ALL"
    assert verdict.counts["settled"] == 1 and verdict.counts["missing"] == 0