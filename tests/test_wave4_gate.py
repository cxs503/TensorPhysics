"""Tests for the wave-4 (IN40-G40) plan-closure gate.

The gate's whole value is that it cannot be talked into a rosier verdict, so the
tests are organised around that claim: the 80% boundary, the critical-failure
veto, the "withheld is not a pass" rule, the integrity check, and a regression
test that re-derives the committed closing report from its own evidence snapshot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign import wave4_gate as w4  # noqa: E402
from tensorphysics.campaign.wave3_gate import GateError, GateIntegrityError, Lineage, Verdict  # noqa: E402

REAL_RAW = ROOT / "artifacts" / "in40_g40_wave4_gate" / "raw" / "nodes_plan84.json"
REAL_REPORT = ROOT / "artifacts" / "in40_g40_wave4_gate" / "wave4_gate_report.json"


# --------------------------------------------------------------------------- helpers
def ledger_entry(node_id: int, ext_id: str, **kw) -> dict:
    n_criteria = kw.pop("n_criteria", 3)
    metadata = {
        "ext_id": ext_id,
        "repo": kw.pop("repo", "TensorFVM"),
        "goal": kw.pop("goal", "【物理目标】测试节点。"),
        "acceptance_criteria": [f"判据 {i}" for i in range(1, n_criteria + 1)],
    }
    metadata.update(kw.pop("extra_meta", {}))
    entry = {
        "node_id": node_id,
        "name": ext_id,
        "status": kw.pop("status", "in-flight"),
        "mission_id": kw.pop("mission_id", 5000 + node_id),
        "mission_status": kw.pop("mission_status", "running"),
        "reproduction_verdict": kw.pop("reproduction_verdict", None),
        "consecutive_fails": kw.pop("consecutive_fails", 0),
        "report_count": kw.pop("report_count", 0),
        "metadata": metadata,
    }
    entry.update(kw)
    return entry


def accepting_submission(node_id: int, n_criteria: int = 3, measured: float = 100.0, reference: float = 100.0) -> dict:
    return {
        "node_id": node_id,
        "evidence_grade": "REAL",
        "rationale": "measured against reference",
        "criteria": {
            f"C{i}": {
                "verdict": "accept",
                "measured": measured,
                "reference": reference,
                "unit": "-",
                "evidence": "pytest run + reference table",
            }
            for i in range(1, n_criteria + 1)
        },
    }


def passing_node(node_id: int, n_criteria: int = 3) -> w4.Wave4Node:
    entry = ledger_entry(node_id, f"IN40-N{node_id:02d}", status="done",
                         mission_status="succeeded", reproduction_verdict="pass",
                         n_criteria=n_criteria)
    return w4.node_from_ledger(entry, submission=accepting_submission(node_id, n_criteria))


def running_node(node_id: int, **kw) -> w4.Wave4Node:
    return w4.node_from_ledger(ledger_entry(node_id, f"IN40-N{node_id:02d}", **kw))


def ten_nodes(pass_count: int, **kw) -> list[w4.Wave4Node]:
    nodes = [passing_node(i) for i in range(1, pass_count + 1)]
    nodes += [running_node(i, **kw) for i in range(pass_count + 1, 11)]
    return nodes


def build(nodes, *, limitations=None, guide_gaps=(), next_wave=()) -> w4.Wave4Report:
    if limitations is None:
        limitations = [
            f"{n.ext_id} 未过门：裁决 {n.verdict.value}" for n in nodes if n.verdict is not Verdict.ACCEPT
        ] + ["报告级局限：证据截止时刻节点仍在跑。"]
    return w4.build_report(
        plan_id=w4.PLAN_ID,
        plan_name="test-plan",
        gate_ext_id=w4.GATE_EXT_ID,
        gate_mission_id=5546,
        cutoff="2026-11-07T00:00:00Z",
        nodes=nodes,
        guide_gaps=guide_gaps or w4.guide_gaps_from_rows(
            [{"metric_key": "guide.success_rate", "metric_value": 0.8, "target_value": 0.9, "notes": "x"}]
        ),
        limitations=limitations,
        next_wave=next_wave or ["seed"],
    )


# --------------------------------------------------------------------------- criteria parsing
def test_criteria_taken_from_metadata_list_first():
    md = {"acceptance_criteria": ["网格收敛≤2%", "偏差≤5%"], "goal": "【机器可算验收判据】C9 不该被采用"}
    assert w4.parse_acceptance_criteria(md) == [("C1", "网格收敛≤2%"), ("C2", "偏差≤5%")]


def test_criteria_fallback_to_parenthesised_numbering():
    goal = "【物理目标】x。【机器可算验收判据】(1) 网格收敛≤2%；(2) 形式因子偏差≤5%【pytest 覆盖(1)-(2)】"
    assert w4.parse_acceptance_criteria({"goal": goal}) == [
        ("C1", "网格收敛≤2%"),
        ("C2", "形式因子偏差≤5%"),
    ]


def test_criteria_missing_is_unadjudicable():
    md = {"goal": "没有判据段的节点"}
    assert w4.parse_acceptance_criteria(md) == []
    node = w4.node_from_ledger(ledger_entry(1, "IN40-N01", n_criteria=0) | {"metadata": md})
    with pytest.raises(GateError):
        w4.build_report(plan_id=84, plan_name="p", gate_ext_id="g", gate_mission_id=1,
                        cutoff="t", nodes=[node] * 10)


# --------------------------------------------------------------------------- plan rule
def test_all_accept_completes_plan_and_triggers_next_wave():
    nodes = ten_nodes(10)
    report = build(nodes)
    acc = report.acceptance
    assert acc.pass_count == 10 and acc.pass_rate == pytest.approx(1.0)
    assert acc.decision is w4.PlanDecision.COMPLETED
    assert acc.next_wave_triggered is True
    assert report.prerequisites["all_satisfied"] is True
    assert report.final_verdict == "ACCEPTED_PLAN_COMPLETED"
    assert set(report.lineage_table[Lineage.MATURE.value]) == {n.ext_id for n in nodes}
    assert report.proposal_trigger()["triggered"] is True
    assert all(c.passed for c in report.checks), [c.as_dict() for c in report.checks]


def test_boundary_eighty_percent_completes():
    report = build(ten_nodes(8))
    acc = report.acceptance
    assert acc.pass_rate == pytest.approx(0.8)
    assert acc.decision is w4.PlanDecision.COMPLETED
    assert acc.next_wave_triggered is True
    # the plan may close, but the gate cannot claim every node was done+passing
    assert report.prerequisites["all_satisfied"] is False
    assert report.final_verdict == "NOT_ACCEPTED_NODES_OPEN"


def test_seventy_percent_keeps_plan_active():
    report = build(ten_nodes(7))
    acc = report.acceptance
    assert acc.pass_rate == pytest.approx(0.7)
    assert acc.decision is w4.PlanDecision.RETAINED_ACTIVE
    assert acc.next_wave_triggered is False
    assert report.proposal_trigger()["triggered"] is False
    assert report.final_verdict == "NOT_ACCEPTED_NODES_OPEN"


def test_critical_failure_vetoes_completion():
    nodes = ten_nodes(9)
    # one node measured and missed completely: verdict FAIL -> critical
    failing = w4.node_from_ledger(
        ledger_entry(10, "IN40-N10", status="done", mission_status="succeeded", reproduction_verdict="pass"),
        submission={
            "node_id": 10,
            "evidence_grade": "REAL",
            "criteria": {"C1": {"verdict": "fail", "measured": 150.0, "reference": 100.0, "unit": "-"}},
        },
    )
    nodes[-1] = failing
    acc = w4.decide_plan(nodes)
    assert acc.pass_rate == pytest.approx(0.9)
    assert acc.critical == ("IN40-N10",)
    assert acc.decision is w4.PlanDecision.RETAINED_ACTIVE
    assert "critical failure" in acc.reason
    report = build(nodes)
    assert report.final_verdict == "NOT_ACCEPTED_NODES_OPEN"


def test_repeated_attempt_collapse_is_critical():
    nodes = ten_nodes(9)
    nodes[-1] = running_node(10, consecutive_fails=w4.CRITICAL_CONSECUTIVE_FAILS)
    acc = w4.decide_plan(nodes)
    assert acc.critical == ("IN40-N10",)
    assert acc.decision is w4.PlanDecision.RETAINED_ACTIVE


# --------------------------------------------------------------------------- withheld is not a pass
def test_absence_of_evidence_is_never_a_pass():
    nodes = [running_node(i) for i in range(1, 11)]
    report = build(nodes)
    assert {n.verdict for n in nodes} == {Verdict.WITHHELD}
    assert report.acceptance.pass_count == 0
    assert report.acceptance.decision is w4.PlanDecision.RETAINED_ACTIVE
    assert set(report.lineage_table[Lineage.ITERATE.value]) == {n.ext_id for n in nodes}
    assert all(n.root_cause is not None for n in nodes)
    assert report.final_verdict == "NOT_ACCEPTED_NODES_OPEN"
    # matrix still covers every node x criterion, each cell an explicit verdict
    assert report.matrix_shape == (10, 30)
    assert all(row["verdict"] == "withheld" for m in report.matrix for row in m["rows"])


def test_missing_limitations_blocks_the_gate_document():
    nodes = ten_nodes(8)
    report = build(nodes, limitations=[])
    checks = {c.cid: c for c in report.checks}
    assert checks["G4"].passed is False
    assert report.final_verdict == "BLOCKED_CRITERIA"


def test_unattributed_open_node_blocks_the_gate_document():
    nodes = ten_nodes(8)
    report = build(nodes, limitations=["只有一条与节点无关的局限说明"])
    checks = {c.cid: c for c in report.checks}
    assert checks["G4"].passed is False
    assert report.final_verdict == "BLOCKED_CRITERIA"


# --------------------------------------------------------------------------- guide tracker
def test_guide_gaps_rank_and_flag():
    gaps = w4.guide_gaps_from_rows(
        [
            {"metric_key": "guide.b", "metric_value": 0.0, "target_value": 1000},
            {"metric_key": "guide.a", "metric_value": 1.0, "target_value": 1.0},
            {"metric_key": "guide.c", "metric_value": 0.8, "target_value": 0.9},
        ]
    )
    assert [g.metric_key for g in gaps] == ["guide.b", "guide.c", "guide.a"]
    assert gaps[0].progress == 0.0 and gaps[0].gap == 1000.0 and gaps[0].unmet
    assert gaps[1].progress == pytest.approx(0.888888, rel=1e-4)
    assert gaps[2].unmet is False
    assert gaps[2].as_dict()["progress"] == 1.0


def test_report_requires_guide_gaps():
    nodes = ten_nodes(10)
    report = w4.build_report(
        plan_id=84, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
        nodes=nodes, guide_gaps=(), limitations=["x"], next_wave=[],
    )
    checks = {c.cid: c for c in report.checks}
    assert checks["G5"].passed is False
    assert report.final_verdict == "BLOCKED_CRITERIA"


# --------------------------------------------------------------------------- integrity
def test_integrity_rejects_rosier_plan_decision():
    report = build(ten_nodes(0))
    payload = report.to_json()
    assert payload["plan_acceptance"]["plan_decision"] == "retained_active"
    w4.verify_report_integrity(report, payload)  # honest payload passes
    rosy = json.loads(json.dumps(payload))
    rosy["plan_acceptance"]["plan_decision"] = "completed"
    with pytest.raises(GateIntegrityError):
        w4.verify_report_integrity(report, rosy)


def test_integrity_rejects_rosier_final_verdict_and_pass_rate():
    report = build(ten_nodes(5))
    payload = report.to_json()
    rosy = json.loads(json.dumps(payload))
    rosy["final_verdict"] = "ACCEPTED_PLAN_COMPLETED"
    with pytest.raises(GateIntegrityError):
        w4.verify_report_integrity(report, rosy)
    inflated = json.loads(json.dumps(payload))
    inflated["plan_acceptance"]["pass_rate"] = 1.0
    with pytest.raises(GateIntegrityError):
        w4.verify_report_integrity(report, inflated)


def test_plan_update_sql_is_noop_below_threshold():
    report = build(ten_nodes(7))
    assert "status = status" in w4.plan_update_sql(report)
    done = build(ten_nodes(10))
    assert "status = 'completed'" in w4.plan_update_sql(done)


# --------------------------------------------------------------------------- report shape
def test_digest_is_deterministic_and_markdown_self_consistent():
    a = build(ten_nodes(8))
    b = build(ten_nodes(8))
    assert a.body_digest == b.body_digest
    md = a.render_markdown()
    for section in ("案例介绍", "怎么做的", "计算结果", "结果图", "误差判定", "局限节", "guide_tracker", "谱系继承"):
        assert section in md
    assert a.body_digest in md
    assert "IN40-N09" in md  # every open node named in the limitations section


def test_matrix_carries_key_error_numbers():
    nodes = ten_nodes(1)
    nodes[0] = w4.node_from_ledger(
        ledger_entry(1, "IN40-N01", status="done", mission_status="succeeded", reproduction_verdict="pass"),
        submission=accepting_submission(1, 3, measured=104.0, reference=100.0),
    )
    report = build(nodes)
    row = report.error_table[0]
    assert row["rel_error_pct"] == pytest.approx(4.0, abs=1e-6)
    assert row["basis"] == "node_criterion"
    assert row["attained"] == pytest.approx(104.0)
    assert report.matrix[0]["criteria_accepted"] == 3
    # the closing report must answer "is it < 3%?" explicitly, for every node
    error_section = report.render_markdown().split("## 5.")[1].split("## 6.")[0]
    line = next(l for l in error_section.splitlines() if l.startswith("| IN40-N01"))
    assert "| 否 |" in line


# --------------------------------------------------------------------------- real snapshot
@pytest.mark.skipif(not REAL_RAW.exists() or not REAL_REPORT.exists(), reason="closing report not generated")
def test_committed_closing_report_matches_its_evidence_snapshot():
    """Re-derive the published numbers from raw/ and prove nothing was edited."""
    raw = json.loads(REAL_RAW.read_text(encoding="utf-8"))
    payload = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    entries = [
        e for e in raw
        if str((e.get("metadata") or {}).get("ext_id")) != w4.GATE_EXT_ID
    ]
    nodes = [w4.node_from_ledger(e) for e in entries]
    nodes.sort(key=lambda n: n.ext_id)
    assert len(nodes) == w4.GATE_NODES
    acc = w4.decide_plan(nodes)
    assert payload["plan_acceptance"]["pass_count"] == acc.pass_count
    assert payload["plan_acceptance"]["pass_rate"] == pytest.approx(acc.pass_rate)
    assert payload["plan_acceptance"]["plan_decision"] == acc.decision.value
    assert payload["matrix_shape"]["nodes"] == w4.GATE_NODES
    assert payload["matrix_shape"]["criteria"] == sum(len(n.criteria) for n in nodes)
    assert payload["report_digest"]