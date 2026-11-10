"""Tests for the read-only harvester, the baseline verdict and the replay payload."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.metrology.criteria.harvest import (  # noqa: E402
    HarvestStats,
    build_corpus,
    harvest_json,
    harvest_markdown,
    harvest_markdown_bullets,
    harvest_markdown_inline,
    iter_json_documents,
)
from tensorphysics.metrology.criteria.replay import (  # noqa: E402
    ReplayResult,
    baseline_verdict,
    replay,
    summarize,
)
from tensorphysics.metrology.criteria.model import CriterionRecord  # noqa: E402
from tensorphysics.metrology.criteria.parser import extract_quadruple  # noqa: E402


DOC = """{
  "anchors": [
    {"name": "fidelity", "criterion": "相对误差 < 2%", "value": 0.017, "passed": true},
    {"name": "conservation", "criterion": "max|A-B| <= 1e-12", "value": 1e-13, "passed": true},
    {"name": "ratio check", "requirement": "|w_fem - w_ref| / w_ref < 0.05",
     "computed": 1.2, "reference": 1.0, "status": "FAIL"}
  ]
}
"""

TABLE = """
| 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| 位移 L2 误差 | <1% | 0.0034 | PASS |
| 能量误差 | <2% | — | WITHHELD |
"""


def test_harvest_json_records(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(DOC, encoding="utf-8")
    stats = HarvestStats()
    recs = harvest_json(str(path), json.loads(DOC), stats)
    assert len(recs) == 3
    by_name = {r.measurand: r for r in recs}
    fidelity = by_name["fidelity"]
    assert fidelity.y == pytest.approx(0.017)
    assert fidelity.y_ref is None
    ratio = by_name["ratio check"]
    assert ratio.has_pair()
    assert ratio.y == pytest.approx(1.2) and ratio.y_ref == pytest.approx(1.0)
    assert stats.with_numeric_pair >= 1
    for rec in recs:
        assert rec.source == str(path)


def test_harvest_json_array_pairs(tmp_path):
    doc = {"case": {"profiles": {"reference": [1.0, 2.0], "computed": [1.01, 2.02],
                                 "tolerance_note": "n/a"}},
           "tolerance": 0.05}
    recs = harvest_json("x.json", doc)
    elements = [r for r in recs if r.evidence.get("element") is not None]
    assert len(elements) == 2
    assert all(r.has_pair() for r in elements)


def test_harvest_markdown_table_and_bullets():
    recs = harvest_markdown("doc.md", TABLE)
    assert len(recs) == 2
    with_value = [r for r in recs if r.reported_error is not None]
    assert with_value and with_value[0].reported_error == pytest.approx(0.0034)
    no_value = [r for r in recs if r.reported_error is None]
    assert no_value, "a criterion archived without an outcome must still be harvested"
    bullets = harvest_markdown_bullets("doc.md", "- 判据 (a) 相对误差 < 3% 严格\n- plain text\n")
    assert bullets and bullets[0].parsed.tolerance == pytest.approx(0.03)
    assert bullets[0].reported_error is None


def test_harvest_markdown_inline_pairs_error_and_threshold():
    text = "计算完成。位移相对误差 = 1.2 %（判据 < 3% → PASS）。\n\n另一段无关文字。"
    recs = harvest_markdown_inline("doc.md", text)
    assert len(recs) == 1
    assert recs[0].reported_error == pytest.approx(0.012)
    assert recs[0].parsed.tolerance == pytest.approx(0.03)


def test_harvest_ignores_serialised_blobs_and_unit_mismatch():
    table = """
| 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| 相对误差 | <2% | {"energy": 1e-14} | PASS |
"""
    assert harvest_markdown("doc.md", table) == []


def test_iter_json_documents_skips_broken(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    good = tmp_path / "ok.json"
    good.write_text('{"criterion": "相对误差 < 2%", "value": 0.01}', encoding="utf-8")
    docs = list(iter_json_documents(str(tmp_path), subdirs=(".",)))
    assert len(docs) == 1


def test_build_corpus_deduplicates(tmp_path):
    (tmp_path / "a.md").write_text(TABLE + "\n| 判据 | 阈值 | 实测 | 结论 |\n|---|---|---|---|\n| 位移 L2 误差 | <1% | 0.0034 | PASS |\n",
                                   encoding="utf-8")
    recs, stats = build_corpus(str(tmp_path), subdirs=(), docs_subdirs=(".",))
    keys = [(r.source, r.criterion, r.reported_error) for r in recs]
    assert len(keys) == len(set(keys))
    assert stats.unique_records == len(recs)


# --------------------------------------------------------------------------------------
# baseline verdict semantics
# --------------------------------------------------------------------------------------
def test_baseline_relative_pass_fail():
    assert baseline_verdict(extract_quadruple("相对误差 < 2%", 1.01, 1.0)).status == "PASS"
    assert baseline_verdict(extract_quadruple("相对误差 < 2%", 1.05, 1.0)).status == "FAIL"


def test_baseline_infinite_relative_error_is_fail():
    verdict = baseline_verdict(extract_quadruple("相对误差 < 2%", 3.0, 0.0, scale=3.0))
    assert verdict.status == "FAIL"
    assert verdict.value == math.inf
    assert any("unbounded" in r for r in verdict.reasons)


def test_baseline_without_evidence_is_withheld():
    assert baseline_verdict(extract_quadruple("相对误差 < 2%", None, None)).status == "WITHHELD"
    assert baseline_verdict(extract_quadruple("no tolerance", 1.0, 1.0)).status == "WITHHELD"


def test_baseline_uses_recorded_error():
    rec = extract_quadruple("相对误差 < 2%", None, None, reported_error=0.01)
    verdict = baseline_verdict(rec)
    assert verdict.status == "PASS"
    assert verdict.value == pytest.approx(0.01)


# --------------------------------------------------------------------------------------
# replay payload
# --------------------------------------------------------------------------------------
def test_replay_small_corpus_and_payload(tmp_path):
    records = [
        extract_quadruple("相对误差 < 2%", 1.05, 1.0, rid="a", source="s"),
        extract_quadruple("相对误差 < 2%", 1.0, 1e-12, scale=1.0, rid="b", source="s"),
        extract_quadruple("相对误差 < 2%", None, None, reported_error=0.001, rid="c", source="s"),
    ]
    result = replay(records)
    payload = result.as_dict()
    assert payload["kpi"]["n_records"] == 3
    assert len(payload["rows"]) == 3
    json.dumps(payload, allow_nan=False)  # must be strict-JSON serialisable
    assert any(row["alternative"]["status"] for row in payload["rows"])


def test_summarize_writes_artifact(tmp_path):
    out = tmp_path / "nested" / "replay.json"
    payload = summarize(str(ROOT), out_path=str(out))
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["kpi"]["n_records"] == payload["kpi"]["n_records"]
    assert loaded["controls"]["n_degenerate_hit"] == 12
    assert loaded["case_studies"]["n_cases"] == 4
    assert len(loaded["degeneracy_list"]) == payload["kpi"]["n_degraded"]

def test_json_safe_scrubs_non_finite():
    from tensorphysics.metrology.criteria import json_safe

    payload = {"a": math.inf, "b": float("nan"), "c": [1.0, -math.inf], "d": {"e": float("nan")}}
    cleaned = json_safe(payload)
    assert cleaned == {"a": None, "b": None, "c": [1.0, None], "d": {"e": None}}
    json.dumps(cleaned, allow_nan=False)
