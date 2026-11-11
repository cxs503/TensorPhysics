"""Tests for the acceptance-criteria judgeability census (P106W3A2-N08).

The tests are the mechanism's regression guard: if a later refactor lets prose
support a PASS, or lets an unmeasured criterion count, one of them goes red.
Fixtures use synthetic, discipline-neutral text only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorphysics.acceptance_judgeability import (
    DEFAULT_LEXICON,
    DEFAULT_POLICY,
    IMPLICIT_POLICY,
    STRICT_POLICY,
    UNIFORM_WEIGHTS,
    Adjudication,
    Evidence,
    Judgeability,
    Lexicon,
    adjudicate,
    classify,
    classify_all,
    cohen_kappa,
    enforce,
    gate_flag,
    get_lexicon,
    judgeability_support,
    lexicon_from_dict,
    node_ks,
    run_census,
    set_lexicon,
    stratified_sample,
    upgrade_paths,
)
from tensorphysics.acceptance_judgeability.support import TOLERANCE_WEIGHTS
from tensorphysics.campaign.wave3_gate import Criterion, Grade, NodeRecord, Verdict
from tensorphysics.campaign.wave4_gate import parse_acceptance_criteria

ART = Path(__file__).resolve().parents[1] / "artifacts" / "acceptance_judgeability"
MEASURED = Evidence(True, "NODE_DONE_ACCEPTED", "status=done fails=0")
UNMEASURED = Evidence(False, "NODE_NOT_DONE", "status=pending fails=0")


def clause(cid: str, text: str) -> dict[str, str]:
    return {"cid": cid, "text": text}


# --------------------------------------------------------------------------- taxonomy
@pytest.mark.parametrize(
    "text,expected",
    [
        ("(1+k) 与 Hughes(1954) 实测相对偏差 ≤5%", Judgeability.JUDGEABLE),
        ("残差 ≤0.1%", Judgeability.JUDGEABLE),
        ("value in [0.15, 0.25]", Judgeability.THRESHOLD_ONLY),
        ("pytest ≥12 cases", Judgeability.THRESHOLD_ONLY),
        ("the trend is monotone in the input", Judgeability.RECORD_ONLY),
        ("attach a screenshot of the endpoint", Judgeability.RECORD_ONLY),
        ("无基准：该工况缺实测数据，显式声明缺数据", Judgeability.WITHHELD_OK),
    ],
)
def test_taxonomy_four_classes(text, expected):
    assert classify(text).judgeability is expected


def test_self_reference_is_judgeable_but_flagged():
    out = classify("能量守恒：应变能=外功，误差<2%")
    assert out.judgeability is Judgeability.JUDGEABLE
    assert out.subtype.value == "SELF_REFERENCE"


def test_strict_policy_drops_self_reference_baseline():
    ks_default, _ = judgeability_support([clause("C1", "残差 ≤0.1%")], MEASURED)
    ks_strict, _ = judgeability_support(
        [clause("C1", "残差 ≤0.1%")], MEASURED, policy=STRICT_POLICY
    )
    assert ks_default == pytest.approx(1.0)
    assert ks_strict == pytest.approx(0.0)


def test_criterion_id_prefix_is_not_a_quantity():
    """``AC6 pytest`` must not read the 6 as a tolerance-bearing number."""
    out = classify("AC6 pytest全绿")
    assert out.judgeability is Judgeability.RECORD_ONLY


# --------------------------------------------------------------------------- support ratio
def test_record_only_never_enters_the_numerator():
    ks, comp = judgeability_support(
        [clause("C1", "记录峰值曲线"), clause("C2", "在报告中给出时程图")], MEASURED
    )
    assert ks == pytest.approx(0.0)
    assert comp["counts"]["RECORD_ONLY"] == 2


def test_ks_is_a_weight_share():
    ks, comp = judgeability_support(
        [
            clause("C1", "(1+k) 与 Hughes(1954) 实测相对偏差 ≤5%"),
            clause("C2", "pytest ≥12 cases"),
            clause("C3", "记录峰值曲线"),
            clause("C4", "the trend is monotone"),
        ],
        MEASURED,
    )
    assert comp["weight_total"] == pytest.approx(4.0)
    assert ks == pytest.approx(0.25)


def test_ks_is_undefined_not_zero_without_clauses():
    ks, comp = judgeability_support([], MEASURED)
    assert ks is None and comp["n_clauses"] == 0


def test_unmeasured_evidence_cannot_support():
    ks_measured, _ = judgeability_support([clause("C1", "残差 ≤0.1%")], MEASURED)
    ks_unmeasured, _ = judgeability_support([clause("C1", "残差 ≤0.1%")], UNMEASURED)
    assert ks_measured > 0.0
    assert ks_unmeasured == pytest.approx(0.0)


def test_adding_prose_never_raises_support():
    base, _ = judgeability_support([clause("C1", "残差 ≤0.1%")], MEASURED)
    with_prose, _ = judgeability_support(
        [clause("C1", "残差 ≤0.1%"), clause("C2", "记录峰值曲线")], MEASURED
    )
    assert with_prose < base


# --------------------------------------------------------------------------- adjudication
def test_record_only_alone_is_hollow():
    adj = adjudicate([clause("C1", "记录峰值曲线")], MEASURED)
    assert adj.outcome == "HOLLOW_PASS" and adj.supports_pass is False
    assert "RECORD_ONLY_ONLY" in adj.blocking


def test_no_criteria_is_not_a_pass():
    adj = adjudicate([], MEASURED)
    assert adj.outcome == "NO_CRITERIA" and adj.supports_pass is False


def test_judgeable_but_unmeasured_is_refused():
    adj = adjudicate([clause("C1", "残差 ≤0.1%")], UNMEASURED)
    assert adj.outcome == "UNMEASURED" and adj.supports_pass is False


def test_negative_control_twenty_prose_clauses_all_hollow():
    prose = [
        "记录峰值压力曲线", "报告中给出时程图", "运行不崩溃且无异常日志", "补充适用域声明",
        "列出全部假设与局限", "输出结果文件并归档", "在报告中说明参数取值", "给出与上一版结果的对比说明",
        "结果趋势单调", "给出网格无关性讨论", "存档求解器日志与版本号", "递交截图证明端点可调用",
        "说明各符号的物理含义", "列出未纳入本验证的效应", "运行端到端流程并保留日志",
        "在报告附录给出原始数据清单", "曲线形状符合预期", "补充术语表", "记录算例清单与耗时",
        "说明与上游模块的接口",
    ]
    outcomes = [adjudicate([clause("C1", t)], MEASURED) for t in prose]
    assert len(outcomes) == 20
    assert all(o.outcome == "HOLLOW_PASS" and not o.supports_pass for o in outcomes)


def test_enforce_downgrades_pass_but_never_softens_fail():
    hollow = adjudicate([clause("C1", "记录峰值曲线")], MEASURED)
    supported = adjudicate([clause("C1", "残差 ≤0.1%")], MEASURED)
    assert enforce("ACCEPT", hollow) == "WITHHELD"
    assert enforce("ACCEPT", supported) == "ACCEPT"
    assert enforce("FAIL", supported) == "FAIL"
    assert gate_flag(hollow)["gate"] == "acceptance_judgeability"


def test_node_record_gate_downgrades_hollow_accept():
    """The shared gate record refuses an ACCEPT whose criteria cannot carry it."""
    crit = (Criterion("C1", "记录峰值曲线", verdict=Verdict.ACCEPT),)
    clean = NodeRecord(1, "X-1", "n", "repo", 1, "succeeded", False, criteria=crit, evidence_grade=Grade.REAL)
    assert clean.verdict is Verdict.ACCEPT
    flag = gate_flag(adjudicate([clause("C1", "记录峰值曲线")], MEASURED))
    guarded = NodeRecord(
        1, "X-1", "n", "repo", 1, "succeeded", False,
        criteria=crit, evidence_grade=Grade.REAL, judgeability=flag,
    )
    assert guarded.verdict is Verdict.WITHHELD
    assert guarded.as_dict()["judgeability"]["supports_pass"] is False


def test_node_record_fail_is_never_overridden_by_the_guard():
    crit = (Criterion("C1", "记录峰值曲线", verdict=Verdict.FAIL),)
    flag = gate_flag(adjudicate([clause("C1", "记录峰值曲线")], MEASURED))
    record = NodeRecord(
        1, "X-1", "n", "repo", 1, "succeeded", False,
        criteria=crit, evidence_grade=Grade.REAL, judgeability=flag,
    )
    assert record.verdict is Verdict.FAIL


# --------------------------------------------------------------------------- lexicon
def test_lexicon_is_injectable_and_restored():
    raw = dict(DEFAULT_LEXICON.raw)
    raw["reference_words"] = raw["reference_words"] + ["benchmarkcase"]
    set_lexicon(lexicon_from_dict(raw))
    try:
        assert classify("偏差 ≤1% benchmarkcase").judgeability is Judgeability.JUDGEABLE
    finally:
        set_lexicon(None)
    assert get_lexicon() is DEFAULT_LEXICON
    assert classify("偏差 ≤1% benchmarkcase").judgeability is Judgeability.THRESHOLD_ONLY


def test_implicit_pair_policy_is_off_by_default_and_changes_the_verdict():
    clauses = [clause("C1", "second_moment_error < 0.10")]
    assert adjudicate(clauses, MEASURED).outcome == "HOLLOW_PASS"
    assert adjudicate(clauses, MEASURED, policy=IMPLICIT_POLICY).outcome == "SUPPORTED"


# --------------------------------------------------------------------------- extraction
def test_json_encoded_criteria_are_decoded_not_iterated_per_character():
    payload = json.dumps(["alpha > 0.05", "beta < 0.10"])
    parsed = parse_acceptance_criteria({"acceptance_criteria": payload})
    assert [text for _, text in parsed] == ["alpha > 0.05", "beta < 0.10"]


def test_structured_criteria_render_deterministically():
    parsed = parse_acceptance_criteria(
        {"acceptance_criteria": [{"metric": "m", "threshold": 0.03, "unit": "relative"}]}
    )
    assert parsed == [("C1", "metric=m; threshold=0.03; unit=relative")]
    assert parsed == parse_acceptance_criteria(
        {"acceptance_criteria": [{"metric": "m", "threshold": 0.03, "unit": "relative"}]}
    )


# --------------------------------------------------------------------------- census/agreement
def _node(node_id: int, criteria, status="done", fails=0):
    return {
        "node_id": node_id,
        "plan_id": 1,
        "ext_id": f"T-{node_id}",
        "repo": "synthetic",
        "status": status,
        "consecutive_fails": fails,
        "criteria": criteria,
    }


def test_census_is_recomputable_and_detects_hollow_passes():
    nodes = [
        _node(1, [clause("C1", "残差 ≤0.1%"), clause("C2", "记录峰值曲线")]),
        _node(2, [clause("C1", "记录峰值曲线"), clause("C2", "补充术语表")]),
        _node(3, [clause("C1", "pytest ≥12 cases")], status="in-flight"),
    ]
    a = run_census(nodes)
    b = run_census(nodes)
    assert a.ks_table_sha256 == b.ks_table_sha256
    assert a.n_nodes == 3 and a.n_criteria == 5
    assert [r.node_id for r in a.hollow] == [2]          # node 2 (done, all prose)
    assert a.hollow_ratio == pytest.approx(0.5)          # 1 of 2 done nodes
    assert a.hollow[0].severity == "TOTAL"
    # a single changed KS must change the digest
    tampered = [dict(nodes[0]), dict(nodes[1]), dict(nodes[2])]
    tampered[0] = _node(1, [clause("C1", "残差 ≤0.1%")])
    assert run_census(tampered).ks_table_sha256 != a.ks_table_sha256


def test_node_ks_row_reports_counts_and_evidence():
    row = node_ks(_node(7, [clause("C1", "残差 ≤0.1%"), clause("C2", "记录峰值曲线")]))
    assert row.counts["JUDGEABLE"] == 1 and row.counts["RECORD_ONLY"] == 1
    assert row.ks == pytest.approx(0.5)
    assert row.evidence.measured is True
    assert row.as_dict()["severity"] == "SUPPORTED"


def test_weight_policy_changes_support():
    clauses = [clause("C1", "残差 ≤0.1%"), clause("C2", "记录峰值曲线")]
    uniform, _ = judgeability_support(clauses, MEASURED, weights=UNIFORM_WEIGHTS)
    weighted, _ = judgeability_support(clauses, MEASURED, weights=TOLERANCE_WEIGHTS)
    # the weight policy decides emphasis: doubling the hard clause raises support
    assert weighted > uniform
    assert uniform == pytest.approx(0.5) and weighted == pytest.approx(2.0 / 3.0)


def test_upgrade_paths_name_the_missing_piece():
    paths = {p["subtype"]: p for p in upgrade_paths([clause("C1", "pytest ≥12 cases"), clause("C2", "the trend is monotone")])}
    assert "reference" in paths["NO_BASELINE"]["missing"]
    assert "tolerance" in paths["MONOTONIC"]["missing"]
    assert all(p["action"] for p in paths.values())


def test_kappa_statistics():
    truth = ["JUDGEABLE", "RECORD_ONLY", "THRESHOLD_ONLY", "THRESHOLD_ONLY"]
    assert cohen_kappa(truth, truth) == pytest.approx(1.0)
    # chance-level agreement on a balanced two-class sample scores 0
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "A", "B", "B"]) == pytest.approx(0.0)


def test_stratified_sample_is_deterministic_and_sized():
    items = [{"id": i, "s": "a" if i % 3 else "b"} for i in range(60)]
    one = stratified_sample(items, stratum=lambda it: it["s"], ident=lambda it: str(it["id"]), n=10, seed="t")
    two = stratified_sample(items, stratum=lambda it: it["s"], ident=lambda it: str(it["id"]), n=10, seed="t")
    assert [i["id"] for i in one] == [i["id"] for i in two]
    assert len(one) == 10
    assert {it["s"] for it in one} == {"a", "b"}


# --------------------------------------------------------------------------- artefacts
@pytest.mark.skipif(not (ART / "gold_labels_100.json").exists(), reason="gold labels not generated")
def test_gold_kappa_meets_the_acceptance_threshold():
    gold = json.loads((ART / "gold_labels_100.json").read_text(encoding="utf-8"))
    human = [i["human"] for i in gold["items"]]
    machine = [classify(i["text"], cid=i["cid"]).judgeability.value for i in gold["items"]]
    assert cohen_kappa(human, machine) >= 0.80


@pytest.mark.skipif(not (ART / "census.json").exists(), reason="census artefact not generated")
def test_committed_census_headline_numbers():
    payload = json.loads((ART / "census.json").read_text(encoding="utf-8"))["census"]
    assert payload["corpus"]["n_nodes"] > 3000
    assert 0.0 < payload["ks"]["judgeable_clause_share"] < 0.6
    assert payload["pass_dependency"]["hollow_ratio"] < 1.0
    assert payload["ks"]["table_sha256"]
    assert math.isfinite(payload["ks"]["mean"])