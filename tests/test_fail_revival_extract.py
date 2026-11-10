"""Extraction tests: completeness, digest consistency, precision on controls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.fail_revival import extract as ex
from tensorphysics.fail_revival import synth
from tensorphysics.fail_revival.model import sha256_text

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tensor_reports" / "campaign_nodes"
LAYOUT = ex.load_layout()


def test_layout_is_externalised_data():
    lay = json.loads(ex.LAYOUT_FILE.read_text(encoding="utf-8"))
    assert lay["verdict_classes"]["fail_unambiguous"]
    assert "criterion_headings" in lay
    assert ex.LAYOUT_FILE.exists()


def test_criterion_block_is_sliced_and_hashed():
    goal = synth.criterion_goal(("alpha error <= 3%", "beta error <= 1%"))
    block, digest, items = ex.criterion_block(goal, LAYOUT)
    assert block.startswith("机器判据")
    assert digest == sha256_text(block)
    assert len(items) == 2
    assert items[0].startswith("1)")
    # the stop marker trims the implementation-path section
    assert "nothing" not in block


def test_synthetic_corpus_extracts_exactly_the_ground_truth():
    art = synth.corpus()[0]
    result = ex.extract_artifact(art, layout=LAYOUT)
    truth = synth.expected_fail_rows()
    assert result.status == "ok"
    assert result.fail_marks == len(truth) == len(result.records)
    rec = result.records[0]
    assert rec.mission_id == truth[0]["mission_id"]
    assert rec.line_no == truth[0]["line_no"]
    assert rec.metric == truth[0]["metric"]
    assert rec.error == truth[0]["error"]
    assert rec.allowed == truth[0]["allowed"]
    assert rec.allowed_source == "header_threshold"


def test_rejection_column_with_no_is_a_passing_control():
    art = synth.corpus()[0]
    result = ex.extract_artifact(art, layout=LAYOUT)
    metrics = {r.metric for r in result.records}
    assert "control_a" not in metrics
    assert "sigma_1" in metrics
    rows = [r for r in result.records if r.mission_id == 9101]
    assert len(rows) == 1


def test_marker_must_start_the_cell():
    md = "\n".join([
        "| quantity | note |",
        "|---|---|",
        "| notchfree | all nodes declared honest FAIL |",
        "| q2 | noise |",
    ])
    art = {
        "schema": ex.ARTIFACT_SCHEMA,
        "mission_id": 9201,
        "plan_id": 1,
        "goal": synth.criterion_goal(("q error <= 1%",)),
        "report": {"id": 1, "markdown": md, "sha256": sha256_text(md)},
    }
    result = ex.extract_artifact(art, layout=LAYOUT)
    assert result.records == ()
    assert result.fail_marks == 0


def test_withheld_rows_are_out_of_scope_not_fail():
    md = "\n".join([
        "| criterion | verdict | evidence |",
        "|---|---|---|",
        "| c1 | WITHHELD | no sourced tabulation available |",
    ])
    art = {
        "schema": ex.ARTIFACT_SCHEMA,
        "mission_id": 9202,
        "plan_id": 1,
        "goal": synth.criterion_goal(("c1 error <= 1%",)),
        "report": {"id": 2, "markdown": md, "sha256": sha256_text(md)},
    }
    result = ex.extract_artifact(art, layout=LAYOUT)
    assert result.records == ()
    assert result.withheld_marks == 1


def test_tampered_artifact_is_fail_closed():
    art = synth.corpus()[0]
    art = json.loads(json.dumps(art))
    art["report"]["markdown"] += "\n| injected | 0 | 9 | 900% | 否 |"
    result = ex.extract_artifact(art, layout=LAYOUT)
    assert result.status == "tampered"
    assert result.records == ()


@pytest.mark.skipif(not CORPUS.exists(), reason="real campaign corpus not synced")
def test_real_corpus_completeness_and_digest_consistency():
    extraction = ex.extract_directory(CORPUS)
    marks = sum(e.fail_marks for e in extraction.extractions)
    records = extraction.records
    assert marks == len(records)
    assert len(records) >= 20
    assert not extraction.tampered
    check = ex.cross_check(extraction, CORPUS, sample=20)
    assert check["sample_checked"] == 20
    assert check["criterion_match_rate"] == 1.0
    assert check["report_digest_match_rate"] == 1.0
    # every record's criterion digest is a digest of its verbatim row text
    for rec in records:
        assert rec.criterion_sha256 == sha256_text(rec.criterion_text)
        assert rec.verdict == "FAIL"
        assert rec.report_sha256


@pytest.mark.skipif(not CORPUS.exists(), reason="real campaign corpus not synced")
def test_real_corpus_negative_controls_not_extracted():
    extraction = ex.extract_directory(CORPUS)
    metrics = {r.metric for r in extraction.records}
    # a control row that was *not* rejected must never appear as a FAIL record
    assert not any("perturb" in m and "1e-13" in m for m in metrics)