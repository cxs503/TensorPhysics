"""Tests for the scientific-contribution meter (ontology, evidence extraction, classifier CLI).

Run:
    python3 -m pytest tests/test_contribution_meter.py -q
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOOLS = os.path.join(ROOT, "tools", "contribution_meter")
sys.path.insert(0, TOOLS)

import classifier as cm  # noqa: E402
import evidence  # noqa: E402

GOLDEN = os.path.join(ROOT, "data", "golden_set_contributions.json")
CORPUS = os.path.join(ROOT, "data", "corpus_reports.jsonl")
if not os.path.exists(CORPUS) and os.path.exists(CORPUS + ".gz"):
    CORPUS = CORPUS + ".gz"
ONTOLOGY = os.path.join(TOOLS, "ontology.json")


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ontology() -> dict:
    with open(ONTOLOGY, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def corpus() -> dict:
    rows = {}
    opener = (gzip.open(CORPUS, "rt", encoding="utf-8") if CORPUS.endswith(".gz")
              else open(CORPUS, encoding="utf-8"))
    with opener as fh:
        for line in fh:
            row = json.loads(line)
            rows.setdefault(str(row["report_id"]), row)
    return rows


# --------------------------------------------------------------------------------------
# ontology
# --------------------------------------------------------------------------------------
def test_ontology_defines_six_classes_plus_none(ontology):
    assert set(ontology["classes"]) == {"C1", "C2", "C3", "C4", "C5", "C6", "NONE"}
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        cls = ontology["classes"][cid]
        assert len(cls["minimum_evidence_gate"]) >= 2, cid
        assert len(cls["counterexamples"]) >= 2, cid
        assert cls["one_line"]


def test_ontology_thresholds_match_classifier(ontology):
    thr = ontology["thresholds"]
    assert thr["accuracy_bar_pct"] == cm.ERR_ACCEPT_PCT
    assert thr["min_dimensioned_quantities"] == cm.MIN_UNITS_PHYSICS
    assert thr["gold_macro_f1"] == 0.85
    assert thr["gold_per_class_f1"] == 0.70
    assert ontology["weights"]["C1"] == 1.0
    assert set(ontology["weights"]) >= {"C1", "C2", "C3", "C4", "C5", "C6"}


def test_ontology_precedence_covers_all_classes(ontology):
    for key in ("correction_pattern", "doc_or_method_shaped", "default"):
        order = ontology["precedence"][key]
        assert sorted(order) == ["C1", "C2", "C3", "C4", "C5", "C6"], key


# --------------------------------------------------------------------------------------
# golden set integrity
# --------------------------------------------------------------------------------------
def test_golden_set_has_60_items_8_per_class_10_adversarial(golden, corpus):
    items = golden["items"]
    assert len(items) == 60
    counts = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert counts.get(cid, 0) >= 8, (cid, counts)
    assert sum(1 for it in items if it["adversarial_looks_like_C1"]) >= 10


def test_golden_items_carry_mission_id_and_rationale(golden, corpus):
    for it in golden["items"]:
        assert isinstance(it["mission_id"], int)
        assert it["rationale"] and len(it["rationale"]) > 40
        assert "evidence_basis" in it
        assert it["report_id"] in corpus, it["report_id"]


# --------------------------------------------------------------------------------------
# evidence extraction
# --------------------------------------------------------------------------------------
def test_extraction_is_deterministic(corpus):
    rep = next(iter(corpus.values()))
    ev1 = evidence.extract(rep)
    ev2 = evidence.extract(rep)
    assert ev1 == ev2


def test_markdown_table_error_rows_are_parsed():
    md = (
        "## 判据\n\n"
        "| 量 | 解析解 | 实测 | 相对误差 |\n"
        "|---|---|---|---|\n"
        "| Ct | 0.0246 | 0.0247 | 0.30% |\n"
        "| S | 5.18 | 5.19 | 0.19% |\n"
    )
    ev = evidence.extract({"mission_id": 1, "report_id": "t", "title": "解析解对拍 验证",
                           "summary": "", "markdown": md})
    assert ev["n_error_rows"] >= 2
    assert ev["min_err_pct"] is not None and ev["min_err_pct"] <= 15.0
    assert "analytic" in ev["reference_kinds"]


def test_zero_error_cells_are_ignored():
    md = "| 项 | 误差 |\n|---|---|\n| a | 0.00% |\n| b | 0.50% |\n"
    rows = evidence.extract_error_rows(md)
    assert all(r["pct"] > 0 for r in rows)


# --------------------------------------------------------------------------------------
# gate logic (synthetic evidence: no dependence on the corpus)
# --------------------------------------------------------------------------------------
def _ev(**kw):
    base = {
        "mission_id": 1, "report_id": "x", "title": "", "summary": "", "scenario_key": "",
        "markdown_len": 5000, "n_error_rows": 4, "error_rows": [], "min_err_pct": 0.5,
        "max_err_pct": 1.2, "reference_kinds": ["analytic"], "n_external_ref_kinds": 1,
        "n_external_error_rows": 4, "min_external_err_pct": 0.5,
        "gate": {"n_acceptance_lines": 5, "n_pass": 4, "n_fail": 0},
        "level_up": [], "n_level_up": 0, "mentions_mock": False, "mentions_stub": False,
        "mentions_real": True, "defect_markers": [], "n_defect_markers": 0,
        "correction_pattern": [], "n_correction_pattern": 0, "has_gate_lock": False,
        "failure_verdicts": [], "n_failure_verdicts": 0, "negative_markers": [],
        "n_negative_markers": 0, "doc_shape_tokens": [], "n_doc_shape_tokens": 0,
        "title_shape": [], "n_title_shape": 0, "capability_language": False,
        "has_method_language": False, "n_formulas": 0, "n_sections": 6, "sections": [],
        "infra_markers": [], "n_infra_markers": 1, "n_pytest_mentions": 1, "n_units": 40,
        "physics_terms": ["阻力"], "n_physics_terms": 6, "n_files": 3, "files": ["a.py"],
        "verification_names": ["verified"], "n_verification_names": 2,
    }
    base.update(kw)
    return base


def test_c1_positive_and_missing_external_reference():
    assert cm.classify_evidence(_ev())["label"] == "C1"
    internal_only = _ev(reference_kinds=["internal_baseline"], n_external_error_rows=0,
                        min_external_err_pct=None, title="台账对拍")
    assert cm.classify_evidence(internal_only)["label"] != "C1"


def test_looks_like_c1_is_c6_when_only_self_consistency():
    """The adversarial definition: error tables without an external reference stay C6."""
    ev = _ev(reference_kinds=["internal_baseline"], n_external_error_rows=0,
             min_external_err_pct=None, n_physics_terms=2,
             infra_markers=["账本", "对拍", "dashboard"], n_infra_markers=12,
             title="四中心最小视图：44/44 SQL 对拍")
    res = cm.classify_evidence(ev)
    assert res["label"] == "C6"


def test_c3_requires_stated_correction():
    ev = _ev(correction_pattern=["高估 +62% → ≤3.61%"], n_correction_pattern=1,
             n_defect_markers=4, has_gate_lock=True, max_err_pct=140.0)
    assert cm.classify_evidence(ev)["label"] == "C3"
    no_correction = _ev(n_defect_markers=4, has_gate_lock=True)
    assert cm.classify_evidence(no_correction)["label"] != "C3"


def test_c5_requires_headline_verdict_and_quantitative_load():
    ev = _ev(title="DTMB 5415 阻力对标 honest-FAIL", summary="5/5 点超 10% 门，最大偏差 165%",
             n_failure_verdicts=3, max_err_pct=165.0, gate={"n_acceptance_lines": 6, "n_pass": 0,
                                                           "n_fail": 5},
             min_external_err_pct=165.0, n_external_error_rows=5, reference_kinds=["experiment"])
    assert cm.classify_evidence(ev)["label"] == "C5"
    weak = _ev(title="验证面", summary="一项 honest FAIL 已如实记录", n_failure_verdicts=1)
    assert cm.classify_evidence(weak)["label"] != "C5"


def test_weights_and_none_are_defined():
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "NONE"):
        assert cid in cm.WEIGHTS
    assert cm.WEIGHTS["C1"] > cm.WEIGHTS["C6"]


# --------------------------------------------------------------------------------------
# CLI + gold-set metrics + window aggregation
# --------------------------------------------------------------------------------------
def test_cli_classify_runs():
    out = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "classifier.py"), "classify",
         "--corpus", CORPUS, "--mission-id", "4886"],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr[-800:]
    payload = json.loads(out.stdout)
    assert payload["label"] in cm.CLASSES + ["NONE"]
    assert payload["trace"] and payload["reasons_against"] is not None


def test_cli_report_is_recomputable(tmp_path):
    out_path = tmp_path / "rep.json"
    out = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "classifier.py"), "report",
         "--corpus", CORPUS, "--missions", os.path.join(ROOT, "data", "missions_window.jsonl"),
         "--window-hours", "48", "--out", str(out_path)],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr[-800:]
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    conv = payload["conversion"]
    # enumeration closes: every mission lands in exactly one bucket
    assert sum(conv["units_by_class"].values()) == conv["n_missions"]
    assert len(conv["zero_contribution_missions"]) == round(
        conv["zero_contribution_ratio"] * conv["n_missions"])
    assert conv["n_scientific_units"] == sum(conv["units_by_class"][c] for c in cm.SCIENTIFIC)
    assert conv["n_missions_without_report"] == conv["n_missions"] - conv["n_missions_with_report"]


def test_gold_metrics_recompute(golden, corpus):
    gold, preds = {}, {}
    for it in golden["items"]:
        rep = corpus[it["report_id"]]
        gold[str(it["report_id"])] = it["label"]
        preds[str(it["report_id"])] = cm.classify_report(rep)["label"]
    metrics = cm.evaluate(preds, gold)
    assert metrics["n"] == 60
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert metrics["per_class"][cid]["support"] >= 8
    # the report's declared numbers must match a fresh recomputation
    with open(os.path.join(ROOT, "data", "golden_eval.json"), encoding="utf-8") as fh:
        declared = json.load(fh)
    assert abs(declared["metrics"]["macro_f1"] - metrics["macro_f1"]) < 1e-9
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert abs(declared["metrics"]["per_class"][cid]["f1"]
                   - metrics["per_class"][cid]["f1"]) < 1e-9


def test_class_below_threshold_is_declared(golden, corpus):
    """Every class under the 0.70 bar must be named in the report artifact (honest FAIL)."""
    with open(os.path.join(ROOT, "data", "golden_eval.json"), encoding="utf-8") as fh:
        declared = json.load(fh)["metrics"]
    below = {c for c, v in declared["per_class"].items() if v["f1"] < 0.70}
    assert set(declared["classes_below_threshold"]) == below
    report = open(os.path.join(ROOT, "docs", "contribution_meter_report.md"), encoding="utf-8").read()
    for cid in below:
        assert cid in report

# --------------------------------------------------------------------------------------
# library facade (landed in TensorPhysics as tensorphysics.metrology)
# --------------------------------------------------------------------------------------
def test_library_facade_exposes_ontology_and_classifier():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tensorphysics.metrology import ContributionMeter

    meter = ContributionMeter()
    assert meter.ontology["classes"]["C1"]["name"] == "verified-case"
    assert meter.WEIGHTS["C1"] == 1.0
    res = meter.classify_report({"mission_id": 1, "report_id": "r", "title": "台账对拍 44/44",
                                 "summary": "", "markdown": "| 项 | 误差 |\n|---|---|\n| a | 1.0% |\n"})
    assert res["label"] in meter.CLASSES + ("NONE",)
    assert res["trace"]


def test_library_facade_window_report_recomputes():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tensorphysics.metrology import ContributionMeter

    meter = ContributionMeter()
    import evidence as _ev

    rows = [json.loads(l) for l in _ev.open_text(CORPUS).readlines()[:40]]
    out = meter.window_report(rows, [], window_hours=48.0)
    conv = out["conversion"]
    assert sum(conv["units_by_class"].values()) == conv["n_missions"]
    assert conv["n_scientific_units"] <= conv["n_missions"]
