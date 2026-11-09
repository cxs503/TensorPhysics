"""Debt extraction contract: error tables and terminal nodes -> debt items.

Acceptance criterion (a) of the campaign node lives here: six synthetic reports
carrying exactly three ``|err| > threshold`` rows must be extracted with
precision = recall = 1.0.  The remaining tests pin the structural rules that
make the extraction trustworthy (threshold injection, percent/fraction units,
header-group conditions, numeric row labels, and the verdict-word precision
guard).
"""
from __future__ import annotations

from lineage_debt_synth import SYNTH_VOCAB, table, report

from tensorphysics.lineage_debt import load_config
from tensorphysics.lineage_debt.extract import (extract_node_terminal_debts,
                                                extract_report_observations,
                                                observations_to_debts, parse_numeric_pct)


def _cfg(**overrides):
    spec = {**SYNTH_VOCAB}
    spec.update(overrides)
    return load_config(spec)


def _debts(reports, cfg):
    obs = []
    for rep in reports:
        obs.extend(extract_report_observations(rep, cfg))
    return observations_to_debts(obs, cfg)


def test_a_extraction_precision_recall_contract():
    """(a) 6 synthetic reports / 3 honest-FAIL rows -> precision = recall = 1.0."""
    from lineage_debt_synth import CONTRACT_GOLD, CONTRACT_REPORTS

    debts = _debts(CONTRACT_REPORTS, _cfg())
    got = {d.key for d in debts}
    true_positive = len(got & CONTRACT_GOLD)
    precision = true_positive / len(got) if got else 0.0
    recall = true_positive / len(CONTRACT_GOLD)
    assert precision == 1.0, f"precision={precision} got={sorted(got)}"
    assert recall == 1.0, f"recall={recall} gold={sorted(CONTRACT_GOLD)}"
    assert len(debts) == 3
    # each debt points back at the report and line it came from
    for d in debts:
        assert d.report_id in (9001, 9002, 9003)
        assert d.evidence and d.evidence[0]["report_sha256"]


def test_threshold_is_config_injected():
    """A per-quantity threshold moves the debt boundary (no logic change)."""
    from lineage_debt_synth import CONTRACT_REPORTS

    base = _debts(CONTRACT_REPORTS, _cfg())
    tighter = _cfg(thresholds={"default_pct": 15.0, "by_quantity": {"q2": 5.0}, "by_domain": {}})
    debts = _debts(CONTRACT_REPORTS, tighter)
    assert {d.key for d in debts} > {d.key for d in base}
    assert "alpha|q2|CASE_C" in {d.key for d in debts}  # 8.0% now breaks its 5% gate


def test_percent_and_fraction_units():
    assert parse_numeric_pct("+22.5%") == 22.5
    assert parse_numeric_pct("-1.8 %") == -1.8
    assert parse_numeric_pct("0.019") == 1.9          # bare fraction -> percent
    assert parse_numeric_pct("20.0") == 20.0          # bare, > 1 -> already percent
    assert parse_numeric_pct("—") is None
    assert parse_numeric_pct("3.55 (err 18.4%)") is None   # free text is not a number


def test_group_prefix_gives_conditions_for_repeated_error_columns():
    """Two error columns in one table: each group's prefix labels its condition."""
    md = table(["量", "plat_A 预测", "plat_A 试验", "误差", "plat_B 预测", "plat_B 试验", "误差"], [
        ["q1", "1.00", "0.95", "+30.0%", "1.00", "0.99", "+2.0%"],
        ["q2", "1.00", "0.90", "-11.1%", "1.00", "0.70", "-42.8%"],
    ])
    cfg = _cfg()
    obs = extract_report_observations(report(9100, md), cfg)
    keys = {o.key for o in obs if o.exceeded}
    assert keys == {"alpha|q1|PLAT_A", "alpha|q2|PLAT_B"}


def test_numeric_row_label_becomes_condition():
    # the row label is a parameter value; the measured quantity is named by the header
    md = table(["case", "q1 error"], [["7", "33.0%"], ["9", "2.0%"]])
    obs = extract_report_observations(report(9101, md), _cfg())
    exceeded = [o for o in obs if o.exceeded]
    assert len(exceeded) == 1
    assert exceeded[0].condition == "7"


def test_verdict_word_alone_is_not_a_debt():
    """Precision guard: a FAIL verdict without a numeric error creates no debt."""
    md = table(["metric", "case", "error", "verdict"], [
        ["q1", "case_A", "—", "FAIL"],
        ["q2", "case_A", "3.0%", "PASS"],
    ])
    cfg = _cfg()
    obs = extract_report_observations(report(9102, md), cfg)
    assert observations_to_debts(obs, cfg) == []


def test_terminal_nodes_become_debts():
    nodes = [
        {"id": 11, "plan_id": 1, "mission_id": 101, "status": "blocked",
         "name": "alpha surface", "metadata": {"goal": "alpha work"}, "block_reason": "ran out of budget"},
        {"id": 12, "plan_id": 1, "mission_id": 102, "status": "done",
         "name": "beta surface", "metadata": {}, "block_reason": None},
    ]
    debts = extract_node_terminal_debts(nodes, _cfg(), plan_chain=[1])
    assert len(debts) == 1
    debt = debts[0]
    assert debt.source_kind == "node_terminal" and debt.node_id == 11
    assert debt.state == "debt_open" and debt.domain == "alpha"
    assert debt.evidence[0]["block_reason"] == "ran out of budget"