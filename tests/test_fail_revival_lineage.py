"""Adjudication, lineage and anti-whitewash tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from tensorphysics.fail_revival.adjudicate import (RevivalOutcome, adjudicate as adjudicate_run,
                                                   parse_number, parse_tolerance)
from tensorphysics.fail_revival import audit as audit_mod
from tensorphysics.fail_revival import lineage as lin
from tensorphysics.fail_revival import synth
from tensorphysics.fail_revival.extract import extract_artifact, load_layout
from tensorphysics.fail_revival.model import Adjudication, RevivalTask, sha256_text

LAYOUT = load_layout()


def _task(rec, digest: str = None, blocker: str = "capability:alpha") -> RevivalTask:
    from tensorphysics.fail_revival.scheduler import baseline_input_digest

    d = digest or baseline_input_digest(rec)
    return RevivalTask(task_key=sha256_text("t|" + rec.criterion_sha256 + d),
                       fail_key=rec.fail_id, criterion_sha256=rec.criterion_sha256,
                       input_sha256=d, blocker_id=blocker, blocker_kind="capability_slug",
                       mission_id=rec.mission_id, cycle=1)


@pytest.fixture()
def record():
    return extract_artifact(synth.corpus()[0], layout=LAYOUT).records[0]


def test_tolerance_and_number_parsing():
    assert parse_tolerance("<3%") == ("<", 3.0)
    assert parse_tolerance("≤10 %") == ("≤", 10.0)
    assert parse_tolerance(None) is None
    assert parse_tolerance("unbounded") is None
    assert parse_number("**-61.2%**") == -61.2
    assert parse_number("+219.8%") == 219.8
    assert parse_number(None) is None


def test_pass_requires_comparable_and_within_tolerance(record):
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:alpha",
                                 code_version="v2", measured=1.2, reproduced_baseline=True)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "PASS"
    assert verdict.family_member is False


def test_still_fail_outside_tolerance(record):
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:alpha",
                                 code_version="v2", measured=50.0, reproduced_baseline=True)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "STILL_FAIL"
    assert verdict.family_member is True


def test_input_change_forces_not_comparable_with_change_list(record):
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256="f" * 64, caliber_id="caliber:alpha",
                                 code_version="v2", measured=0.1, reproduced_baseline=True)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "NOT_COMPARABLE"
    assert verdict.family_member is True
    assert [c["field"] for c in verdict.changes] == ["inputs"]


def test_caliber_change_forces_not_comparable(record):
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:revised",
                                 code_version="v2", measured=0.1, reproduced_baseline=True)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "NOT_COMPARABLE"
    assert verdict.changes[0]["field"] == "caliber"


def test_code_change_without_reproduction_is_not_comparable(record):
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:alpha",
                                 code_version="v3", measured=0.1, reproduced_baseline=False)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "NOT_COMPARABLE"
    assert verdict.changes[0]["field"] == "code_version"


def test_code_change_scoped_by_reproduction_stays_comparable(record):
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:alpha",
                                 code_version="v3", measured=0.5, reproduced_baseline=True)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "PASS"
    assert "code_version_changed_but_baseline_reproduced" in verdict.reasons


def test_not_comparable_without_change_list_is_rejected():
    with pytest.raises(ValueError):
        Adjudication(task_key="k", fail_key="f", criterion_sha256="h",
                     verdict="NOT_COMPARABLE", changes=())


def test_absent_tolerance_cannot_earn_pass(record):
    import dataclasses

    bare = dataclasses.replace(record, allowed=None, allowed_source="absent")
    task = _task(bare)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:alpha",
                                 code_version="v2", measured=0.001, reproduced_baseline=True)
    verdict = adjudicate_run(task, bare, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    assert verdict.verdict == "STILL_FAIL"
    assert "tolerance_absent_in_registry" in verdict.reasons


def test_lineage_edge_is_written_and_queryable(tmp_path, record):
    conn = lin.connect(tmp_path / "lineage.sqlite3")
    task = _task(record)
    outcome = RevivalOutcome(inputs_sha256=task.input_sha256, caliber_id="caliber:alpha",
                                 code_version="v2", measured=0.5, reproduced_baseline=True)
    verdict = adjudicate_run(task, record, outcome, baseline_inputs_sha256=task.input_sha256,
                             baseline_caliber="caliber:alpha", baseline_code_version="v2")
    audit_mod.audited_adjudicate(conn, verdict, record, task, at="2026-11-10T00:00:00+00:00")
    rows = lin.lineage_of(conn, record.fail_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["old_verdict"] == "FAIL" and row["new_verdict"] == "PASS"
    assert row["criterion_sha256"] == record.criterion_sha256
    assert row["blocker_id"] == "capability:alpha"
    assert row["adjudicated_at"] == "2026-11-10T00:00:00+00:00"
    assert lin.edge_of_task(conn, task.task_key)[0]["edge_id"] == row["edge_id"]
    assert lin.lineage_of_criterion(conn, record.criterion_sha256)[0]["edge_id"] == row["edge_id"]


def test_whitewash_attempt_is_refused_and_audited(tmp_path, record):
    conn = lin.connect(tmp_path / "lineage.sqlite3")
    task = _task(record)
    forged = Adjudication(task_key=task.task_key, fail_key=record.fail_id,
                          criterion_sha256="0" * 64, verdict="PASS", family_member=False)
    with pytest.raises(audit_mod.WhitewashRefused):
        audit_mod.audited_adjudicate(conn, forged, record, task, at="2026-11-10T00:00:00+00:00")
    assert lin.edge_count(conn) == 0
    rows = lin.ledger(conn, audit_mod.OVERCLAIM)
    assert len(rows) == 1
    assert rows[0]["reason"] == "criterion_digest_mismatch"
    assert rows[0]["registry_sha256"] == record.criterion_sha256
    assert rows[0]["criterion_sha256"] == "0" * 64


def test_unknown_record_is_an_overclaim(tmp_path, record):
    conn = lin.connect(tmp_path / "lineage.sqlite3")
    task = _task(record)
    forged = Adjudication(task_key=task.task_key, fail_key="FR-ghost",
                          criterion_sha256=record.criterion_sha256, verdict="PASS",
                          family_member=False)
    with pytest.raises(audit_mod.WhitewashRefused):
        audit_mod.audited_adjudicate(conn, forged, None, task)
    assert lin.ledger(conn, audit_mod.OVERCLAIM)[0]["reason"] == "criterion_not_in_registry"