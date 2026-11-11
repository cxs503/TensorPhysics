"""Tests for the mission acceptance-record census (``scripts/audit_acceptance_records.py``).

The census exists because a mission record can lie cheaply: ``docs/m8598/outputs/
acceptance_criteria.json`` shipped for days as a byte-for-byte copy of another
mission's shaft-torsion record, and only a full re-derivation noticed.  A record
sealed with ``payload_sha256`` cannot lie quietly, and the census is what makes
that property visible tree-wide -- including for the records that predate it.

Fixtures are synthetic (a tiny fake payload, ``tmp_path`` trees), so these tests
do not depend on any mission's artefacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_acceptance_records as aud  # noqa: E402
import run_criterion_prereg_eval as ev  # noqa: E402

#: A payload shaped like the evaluation record, with no mission-specific values.
SYNTH_PAYLOAD = {
    "node": "synthetic-node",
    "mission_id": 424242,
    "all_acceptance_pass": True,
    "acceptance": {
        "a": {"value": 0.97, "threshold": 0.95, "pass": True},
        "b": {"value": 1.0, "pass": True},
    },
}


def _sealed(payload: dict = SYNTH_PAYLOAD) -> dict:
    return ev.deliverable_record(payload, "artifacts/synthetic/eval.json")


def _write(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _tree(tmp_path: Path, records: dict[str, dict]) -> Path:
    for rel, record in records.items():
        _write(tmp_path / rel, record)
    return tmp_path


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def test_sealed_record_verifies(tmp_path):
    root = _tree(tmp_path, {"docs/m1/outputs/acceptance_criteria.json": _sealed()})
    [row] = aud.scan(root)
    assert row["sealed"] is True and row["reason"] == "ok"
    assert row["mission_id"] == 424242
    assert row["n_acceptance_blocks"] == 2
    assert row["all_acceptance_pass"] is True


def test_string_typed_measurement_is_caught_by_the_seal(tmp_path):
    """A re-typed threshold flips the digest: half a million to "half a million"."""
    record = _sealed()
    record["acceptance"]["a"]["value"] = "0.97"  # same number, different type
    root = _tree(tmp_path, {"docs/m1/outputs/acceptance_criteria.json": record})
    [row] = aud.scan(root)
    assert row["sealed"] is False
    assert "mismatch" in row["reason"]


def test_unsealed_record_is_reported_not_crashed(tmp_path):
    legacy = {"convergence_order": 1.99, "transient_energy_drift": 7.3e-12,
              "pytest_test_shaft_torsion": True}
    root = _tree(tmp_path, {"docs/m2/outputs/acceptance_criteria.json": legacy,
                            "docs/m3/acceptance_criteria.json": legacy})
    rows = aud.scan(root)
    assert len(rows) == 2
    assert all(r["sealed"] is False for r in rows)
    assert all("missing payload_sha256" in r["reason"] for r in rows)
    assert all(r["mission_id"] is None for r in rows)


def test_unreadable_and_non_object_files_are_rows_not_exceptions(tmp_path):
    (tmp_path / "docs" / "m4" / "outputs").mkdir(parents=True)
    (tmp_path / "docs" / "m4" / "outputs" / "acceptance_criteria.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path / "docs" / "m5" / "acceptance_criteria.json", {"mission_id": 5})  # unsealed but valid
    (tmp_path / "docs" / "m6").mkdir(parents=True)
    (tmp_path / "docs" / "m6" / "acceptance_criteria.json").write_text("[1, 2]", encoding="utf-8")
    rows = {r["path"]: r for r in aud.scan(tmp_path)}
    assert rows["docs/m4/outputs/acceptance_criteria.json"]["sealed"] is False
    assert "unreadable" in rows["docs/m4/outputs/acceptance_criteria.json"]["reason"]
    assert "not a JSON object" == rows["docs/m6/acceptance_criteria.json"]["reason"]


# --------------------------------------------------------------------------- #
# the census: shared bodies are the "copy" signal
# --------------------------------------------------------------------------- #
def test_identical_bodies_are_grouped_as_copies(tmp_path):
    legacy = {"convergence_order": 1.99, "pytest_test_shaft_torsion": True}
    root = _tree(tmp_path, {f"docs/m{i}/outputs/acceptance_criteria.json": legacy for i in (7, 8, 9)})
    report = aud.summarise(root)
    assert report["n_records"] == 3
    assert report["n_sealed"] == 0
    assert report["n_duplicated_bodies"] == 3
    [group] = report["copies"]
    assert len(group["paths"]) == 3
    assert group["body_sha256"] == aud.body_digest(legacy)


def test_distinct_sealed_records_are_not_flagged_as_copies(tmp_path):
    other = json.loads(json.dumps(SYNTH_PAYLOAD))
    other["acceptance"]["b"]["pass"] = False
    root = _tree(tmp_path, {"docs/m1/outputs/acceptance_criteria.json": _sealed(),
                            "docs/m2/outputs/acceptance_criteria.json": _sealed(other)})
    report = aud.summarise(root)
    assert report["copies"] == []
    assert report["n_sealed"] == 2 and report["all_sealed"] is True
    assert report["verdict"] == "ok"


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_gate_exit_codes(tmp_path, capsys):
    all_sealed = _tree(tmp_path / "sealed",
                       {"docs/m1/outputs/acceptance_criteria.json": _sealed()})
    assert aud.main(["--root", str(all_sealed), "--fail-on-unsealed", "--quiet"]) == 0

    mixed = _tree(tmp_path / "mixed",
                  {"docs/m1/outputs/acceptance_criteria.json": _sealed(),
                   "docs/m2/outputs/acceptance_criteria.json": {"convergence_order": 1.9}})
    assert aud.main(["--root", str(mixed), "--fail-on-unsealed", "--quiet"]) == 1
    # the default is a census, not a gate: older records predate the seal
    assert aud.main(["--root", str(mixed), "--quiet"]) == 0
    assert "FAIL" in capsys.readouterr().out


def test_empty_tree_is_not_green(tmp_path):
    report = aud.summarise(_tree(tmp_path, {}))
    assert report["n_records"] == 0
    assert report["verdict"] == "no records"


# --------------------------------------------------------------------------- #
# this checkout: the record this mission was reopened for must stay sealed
# --------------------------------------------------------------------------- #
def test_m8598_record_is_present_and_sealed_in_this_checkout():
    """Regression guard for the reopen: no cross-mission copy may come back."""
    record = ROOT / "docs" / "m8598" / "outputs" / "acceptance_criteria.json"
    assert record.exists(), "the mission's acceptance record is missing"
    shipped = json.loads(record.read_text(encoding="utf-8"))
    ok, reason = ev.verify_deliverable(shipped)
    assert ok, reason
    assert shipped["mission_id"] == 8598
    assert shipped["all_acceptance_pass"] is True
    # the copied-from mission's fingerprint must be absent
    assert "pytest_test_shaft_torsion" not in shipped
    assert "convergence_order" not in shipped


def test_this_checkout_census_reports_the_sealed_mission():
    report = aud.summarise(ROOT)
    row = next((r for r in report["records"]
                if r["path"] == "docs/m8598/outputs/acceptance_criteria.json"), None)
    assert row is not None, "the census must see this mission's record"
    assert row["sealed"] is True and row["reason"] == "ok"
    assert report["n_sealed"] >= 1