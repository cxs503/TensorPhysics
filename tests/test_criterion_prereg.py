"""Acceptance tests for the criterion pre-registration gate (p92w2b-n07).

Machine-checkable acceptance criteria of this node:

(a) synthetic detection set (24 cases): drift-detection recall >= 0.92 and
    precision >= 0.95 -- plus per-class accuracy against generator truth;
(b) extractor field-level F1 >= 0.95 (development seed *and* a held-out seed);
(c) fail-closed negative controls: empty goal -> ``unregistered``, tampered
    digest -> ``tampered``, and no digest is published in those cases;
(d) zero-domain-semantics scan of the package hits == 0.

The synthetic corpora carry *generator-side* ground truth (see
``criterion_prereg.synth``), so the numbers are measured against what the
generator wrote, not against the extractor's own output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.criterion_prereg import (  # noqa: E402
    REGISTERED,
    TAMPERED,
    UNREGISTERED,
    ManifestError,
    attach_to_metadata,
    audit,
    canonical_json,
    extract_criteria,
    freeze,
    kpi_view,
    manifest_from_metadata,
    sha256_obj,
    verify_manifest,
)
from tensorphysics.campaign.criterion_prereg.scan import scan_paths  # noqa: E402
from tensorphysics.campaign.criterion_prereg.synth import (  # noqa: E402
    baseline_text_hash,
    detection_metrics,
    field_scores,
    gen_drift_cases,
    gen_extraction_corpus,
)

PKG = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_prereg"
STAMP = "2026-11-09T00:00:00+00:00"

GOAL = "机器验收判据：(a) 合成检测集24例recall≥0.92、precision≥0.95；(b) 抽取器字段级F1≥0.95；(c) 空goal→unregistered；(d) 扫描命中=0。"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _audit_case(case):
    fz = freeze(case.frozen_goal, frozen_at=STAMP)
    assert fz.registered, case
    report = audit(fz.manifest, receipt_goal=case.receipt_goal)
    return report


# --------------------------------------------------------------------------- #
# (b) extractor fidelity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [20261109, 42])
def test_extractor_field_f1(seed):
    scores = field_scores(gen_extraction_corpus(20, seed=seed))
    assert scores["n_cases"] == 20
    assert scores["precision"] >= 0.95, scores
    assert scores["recall"] >= 0.95, scores
    assert scores["f1"] >= 0.95, scores


def test_extractor_surface_invariance():
    """Equivalent surface forms must extract to the same criterion."""
    variants = [
        "机器验收判据：(a) 合成检测集24例recall>=0.92。",
        "机器验收判据：[A] 合成检测集24casesrecall≥0.920；",
        "机器验收判据：A. 合成检测集 24条 recall ＞= 0.92",
    ]
    got = [{k: v for k, v in c.to_dict().items() if k not in ("cid", "text")} for c in
           (extract_criteria(v).criteria[0] for v in variants)]
    for rec in got[1:]:
        assert rec == got[0]


def test_extractor_behavioural_mapping_criterion():
    crit = extract_criteria("机器验收判据：(c) fail-closed负控：空goal→unregistered。").criteria
    assert crit, "mapping clause not extracted"
    assert crit[0].kind == "mapping"
    assert crit[0].op == "->"
    assert crit[0].value == "unregistered"


# --------------------------------------------------------------------------- #
# (a) drift detection on the 24-case synthetic set
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [20261109, 7])
def test_drift_detection_recall_precision(seed):
    cases = gen_drift_cases(seed=seed)
    assert len(cases) == 24
    predictions = []
    for case in cases:
        report = _audit_case(case)
        predictions.append((report.failed(), report.dominant_class))
    metrics = detection_metrics(predictions, cases)
    assert metrics["recall"] >= 0.92, metrics
    assert metrics["precision"] >= 0.95, metrics
    # per-class labels must also be right (fail-closed classification, not just a flag)
    assert metrics["class_accuracy"] >= 0.90, metrics["confusion"]
    # the naive text-digest baseline is beaten on precision (reformatting/tightening
    # are not drift) while matching recall -- this is what the structured audit buys.
    base = detection_metrics(
        [(baseline_text_hash(c.frozen_goal, c.receipt_goal), "x") for c in cases], cases
    )
    assert base["recall"] == metrics["recall"]
    assert base["precision"] < metrics["precision"]


def test_tightened_is_recorded_but_passes():
    fz = freeze(GOAL, frozen_at=STAMP)
    receipt = GOAL.replace("recall≥0.92", "recall≥0.97")
    report = audit(fz.manifest, receipt_goal=receipt)
    assert report.verdict.value == "pass"
    assert "tightened" in report.counts and report.counts["tightened"] >= 1
    assert any(e["type"] == "criterion.drift.clean" for e in report.events())


def test_loosened_fails_closed():
    fz = freeze(GOAL, frozen_at=STAMP)
    receipt = GOAL.replace("recall≥0.92", "recall≥0.80")
    report = audit(fz.manifest, receipt_goal=receipt)
    assert report.failed()
    assert report.dominant_class == "loosened"
    events = report.events()
    assert any(e["type"] == "criterion.drift" and e["class"] == "loosened" and e["severity"] == "error" for e in events)


def test_scope_change_and_dropped_criterion():
    fz = freeze(GOAL, frozen_at=STAMP)
    narrowed = audit(fz.manifest, receipt_goal=GOAL.replace("24例", "12例"))
    assert narrowed.dominant_class == "scope_changed" and narrowed.failed()

    dropped = audit(fz.manifest, receipt_goal=GOAL.replace("；(d) 扫描命中=0", ""))
    assert dropped.dominant_class == "loosened" and dropped.failed()
    assert any(i.reason == "criterion_dropped" for i in dropped.items)


def test_added_criterion_counts_as_tightened():
    fz = freeze(GOAL, frozen_at=STAMP)
    report = audit(fz.manifest, receipt_goal=GOAL.replace("；(d)", "；(e) 回归用例集12例boundary≥0.9；(d)"))
    assert report.verdict.value == "pass"
    assert any(i.reason == "criterion_added" for i in report.items)


def test_mapping_target_change_is_undefined():
    fz = freeze(GOAL, frozen_at=STAMP)
    report = audit(fz.manifest, receipt_goal=GOAL.replace("空goal→unregistered", "空goal→accepted"))
    assert report.dominant_class == "undefined"
    assert report.failed()


# --------------------------------------------------------------------------- #
# (c) fail-closed negative controls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("goal", ["", "   \n\n ", "机器验收判据："])
def test_empty_goal_is_unregistered(goal):
    fz = freeze(goal, mission_id=7803)
    assert fz.state == UNREGISTERED
    assert fz.sha256 == ""
    assert fz.manifest["n_criteria"] == 0
    with pytest.raises(ManifestError):
        attach_to_metadata({"mission_id": 7803}, fz)
    assert verify_manifest(fz.manifest) == UNREGISTERED
    report = audit(fz.manifest, receipt_goal=GOAL)
    assert report.state == UNREGISTERED and report.failed()
    assert report.events()[0]["type"] == "criterion.prereg.anomaly"
    assert report.events()[0]["severity"] == "error"


def test_empty_receipt_is_unregistered():
    fz = freeze(GOAL, frozen_at=STAMP)
    report = audit(fz.manifest, receipt_goal="")
    assert report.state == UNREGISTERED and report.failed()
    assert "receipt_unregistered" in report.notes


def test_tampered_digest_detected():
    fz = freeze(GOAL, frozen_at=STAMP)
    manifest = json.loads(canonical_json(fz.manifest))
    assert verify_manifest(manifest) == REGISTERED
    manifest["criteria"][0]["value"] = "0.80"  # someone loosened the frozen record
    assert verify_manifest(manifest) == TAMPERED
    report = audit(manifest, receipt_goal=GOAL)
    assert report.state == TAMPERED and report.failed()
    assert report.events()[0]["type"] == "criterion.prereg.anomaly"
    assert report.events()[0]["reason"] == "receipt_manifest_tampered" or report.notes == ["frozen_manifest_tampered"]


def test_goal_text_tamper_detected():
    fz = freeze(GOAL, frozen_at=STAMP)
    assert verify_manifest(fz.manifest, goal=GOAL) == REGISTERED
    assert verify_manifest(fz.manifest, goal=GOAL.replace("0.92", "0.80")) == TAMPERED


# --------------------------------------------------------------------------- #
# canonical form / metadata round-trip
# --------------------------------------------------------------------------- #
def test_canonical_json_is_order_free():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert sha256_obj({"b": 1, "a": [1, 2]}) == sha256_obj({"a": [1, 2], "b": 1})


def test_metadata_roundtrip_is_reproducible():
    first = freeze(GOAL, mission_id=7803, node_id="p92w2b-n07", frozen_at=STAMP)
    second = freeze(GOAL, mission_id=7803, node_id="p92w2b-n07", frozen_at=STAMP)
    assert first.sha256 == second.sha256, "freeze is not reproducible"
    meta = attach_to_metadata({"mission_id": 7803, "owner": "autonomy"}, first)
    payload = manifest_from_metadata(meta)
    assert payload["manifest_sha256"] == first.sha256
    # (a) expands into two criteria (recall, precision) -> 5 structured rows
    assert payload["n_criteria"] == first.manifest["n_criteria"] == 5


def test_reformatted_goal_keeps_criteria_digest():
    fz = freeze(GOAL, frozen_at=STAMP)
    reformatted = GOAL.replace("≥", ">=").replace("、", "；")
    other = freeze(reformatted, frozen_at=STAMP)
    from tensorphysics.campaign.criterion_prereg import semantic_digest
    assert semantic_digest(fz.manifest) == semantic_digest(other.manifest)
    assert fz.manifest["goal_sha256"] != other.manifest["goal_sha256"]


def test_cli_exit_codes_are_fail_closed(tmp_path):
    from tensorphysics.campaign.criterion_prereg.__main__ import main

    goal = tmp_path / "goal.txt"
    goal.write_text(GOAL, encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    assert main(["freeze", "--goal-file", str(goal), "--mission-id", "7803", "--out", str(frozen)]) == 0
    assert main(["verify", "--frozen", str(frozen)]) == 0

    loosened = tmp_path / "loosened.txt"
    loosened.write_text(GOAL.replace("recall≥0.92", "recall≥0.70"), encoding="utf-8")
    assert main(["audit", "--frozen", str(frozen), "--receipt-file", str(loosened)]) == 1

    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    assert main(["freeze", "--goal-file", str(empty)]) == 2
    assert main(["audit", "--frozen", str(frozen), "--receipt-file", str(empty)]) == 2


# --------------------------------------------------------------------------- #
# KPI view + events
# --------------------------------------------------------------------------- #
def test_kpi_view():
    fz = freeze(GOAL, frozen_at=STAMP)
    reports = [
        audit(fz.manifest, receipt_goal=GOAL),  # pass
        audit(fz.manifest, receipt_goal=GOAL.replace("recall≥0.92", "recall≥0.80")),  # fail (loosened)
        audit(fz.manifest, receipt_goal=""),  # fail-closed (unregistered receipt)
    ]
    kpi = kpi_view(reports)
    assert kpi["n_audits"] == 3
    assert kpi["n_pass"] == 1
    assert kpi["prereg_coverage"] == pytest.approx(2 / 3)
    assert kpi["fail_closed_incidents"] == 1
    assert kpi["class_histogram"]["loosened"] == 1
    assert kpi["drift_rate"] > 0


def test_event_records_are_json_serialisable():
    fz = freeze(GOAL, frozen_at=STAMP)
    report = audit(fz.manifest, receipt_goal=GOAL.replace("F1≥0.95", "F1≥0.60"))
    blob = json.dumps(report.events(), ensure_ascii=False)
    restored = json.loads(blob)
    assert restored and restored[0]["schema"] == "criterion-prereg-event/1"
    assert {"type", "mission_id", "node_id", "severity", "reason", "frozen_sha256", "occurred_at"} <= set(restored[0])


# --------------------------------------------------------------------------- #
# (d) zero domain semantics
# --------------------------------------------------------------------------- #
def test_zero_domain_semantics():
    result = scan_paths([PKG])
    assert result.vocabulary_loaded and result.n_terms > 0
    assert result.clean, [h.to_dict() for h in result.hits]


def test_scan_detects_a_planted_term():
    """The scanner must not be vacuously green."""
    from tensorphysics.campaign.criterion_prereg.scan import scan_text
    hits = scan_text("# placeholder probe\nvalue = 'propeller'\n", ["propeller"], path="probe.py")
    assert len(hits) == 1 and hits[0].term == "propeller"