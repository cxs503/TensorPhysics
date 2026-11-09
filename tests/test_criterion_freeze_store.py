"""Tests for the durable criterion freeze store (cross-process pre-registration).

Covers:
* freeze → new process → load → audit round trip (no in-memory state reused)
* per-criterion hash tampering is detected (status ``tampered``, never registered)
* bundle-level digest tampering is detected (``StoreCorruptError``)
* fail-closed behaviour: goal without numeric level writes nothing and audits
  as ``unregistered``; never-frozen mission audits as ``unregistered``
* idempotent re-freeze vs. refusing a re-freeze with different content
* path-safe mission ids, atomic write leaves no temp files behind
* the audit classifies a genuine loosening as ``loosened`` with a reason

Zero domain semantics: criterion texts below are synthetic placeholders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorphysics.campaign.criterion_registry import (  # noqa: E402
    CriterionLedger,
    PreregistrationConflict,
    StoreCorruptError,
    audit_from_store,
    bundle_path,
    freeze_to_store,
    load_bundle,
    load_registrations,
)
from tensorphysics.campaign.criterion_registry.store import STORE_SCHEMA  # noqa: E402

MISSION = "m9001"
GOAL = """
- recall >= 0.92 on the held-out split
- latency <= 25 ms of the p95 baseline
"""


def _fresh_audit(store, mission=MISSION, report=None, ledger=None):
    """Simulate a *separate process*: nothing but the store directory survives."""
    return audit_from_store(store, mission, report if report is not None else GOAL,
                            ledger=ledger)


# ---------------------------------------------------------------------------
# round trip / persistence
# ---------------------------------------------------------------------------

def test_freeze_writes_one_bundle_per_mission(tmp_path):
    bundle = freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695",
                             frozen_at="2026-11-09T00:00:00+00:00")
    assert bundle.path == bundle_path(tmp_path, MISSION)
    assert bundle.path.exists()
    assert bundle.is_registered
    assert len(bundle.registered_ids) == 2

    data = json.loads(bundle.path.read_text(encoding="utf-8"))
    assert data["schema"] == STORE_SCHEMA
    assert data["mission_id"] == MISSION
    assert data["frozen_by"] == "agent/m8695"
    assert data["frozen_at"] == "2026-11-09T00:00:00+00:00"
    assert data["bundle_sha256"] == bundle.bundle_sha256
    assert [c["status"] for c in data["criteria"]] == ["registered", "registered"]


def test_reload_in_fresh_process_state_reproduces_hashes(tmp_path):
    written = freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    loaded = load_bundle(tmp_path, MISSION)
    assert loaded is not None
    assert loaded.criteria == written.criteria
    assert loaded.bundle_sha256 == written.bundle_sha256
    assert loaded.is_registered
    # every per-criterion hash still verifies after the JSON round trip
    assert all(r.is_registered for r in loaded.registrations)
    assert all(r.manifest.sha256 for r in loaded.registrations)


def test_floats_survive_canonical_json_round_trip(tmp_path):
    goal = "- throughput >= 0.123456789 units per second\n- drift <= 1e-06 pct\n"
    freeze_to_store(tmp_path, "m9002", goal, frozen_by="agent/x")
    loaded = load_bundle(tmp_path, "m9002")
    thresholds = [r.manifest.fields["threshold"].value
                  for r in loaded.registrations]
    assert thresholds == [0.123456789, 1e-06]


def test_audit_from_store_unchanged_when_report_repeats_goal(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    ledger = CriterionLedger()
    audit = _fresh_audit(tmp_path, ledger=ledger)
    assert [r.classification for r in audit.results] == ["unchanged", "unchanged"]
    assert audit.failed_closed is False
    assert audit.kpi.registered_rate == pytest.approx(1.0)
    assert audit.kpi.loosened_rate == pytest.approx(0.0)
    assert len(ledger) == 2


def test_audit_from_store_detects_loosening(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    loosened_report = """
    - recall >= 0.90 on the held-out split
    - latency <= 25 ms of the p95 baseline
    """
    audit = _fresh_audit(tmp_path, report=loosened_report)
    first = audit.results[0]
    assert first.classification == "loosened"
    assert "threshold" in first.field_diffs
    assert first.reason
    # the tightening direction is unchanged for the other criterion
    assert audit.results[1].classification == "unchanged"
    assert audit.kpi.loosened_rate > 0.0
    assert audit.failed_closed is False


def test_store_file_is_byte_stable_for_same_goal(tmp_path):
    a = freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    first = a.path.read_bytes()
    import os
    mtime = os.stat(a.path).st_mtime_ns
    b = freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    assert b.bundle_sha256 == a.bundle_sha256
    assert b.path.read_bytes() == first
    assert os.stat(a.path).st_mtime_ns == mtime  # no rewrite


# ---------------------------------------------------------------------------
# tamper detection
# ---------------------------------------------------------------------------

def test_edited_threshold_loads_as_tampered_not_registered(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    path = bundle_path(tmp_path, MISSION)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["criteria"][0]["fields"]["threshold"]["value"] = 0.5  # hand-edit a level
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8")

    with pytest.raises(StoreCorruptError):
        load_bundle(tmp_path, MISSION)


def test_manifest_hash_mismatch_marks_criterion_tampered(tmp_path):
    """A console-consistent edit (bundle digest recomputed) still trips the
    per-criterion hash and never yields ``unchanged``."""
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    path = bundle_path(tmp_path, MISSION)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["criteria"][0]["fields"]["threshold"]["value"] = 0.5
    # attacker recomputes the *bundle* digest to paper over the edit
    from tensorphysics.campaign.criterion_registry.store import FrozenBundle
    forged = FrozenBundle(
        mission_id=raw["mission_id"], frozen_by=raw["frozen_by"],
        frozen_at=raw["frozen_at"], criteria=raw["criteria"],
    )
    raw["bundle_sha256"] = forged.compute_sha256()
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8")

    loaded = load_bundle(tmp_path, MISSION)
    statuses = [r.status for r in loaded.registrations]
    assert statuses == ["tampered", "registered"]
    assert loaded.is_registered is False

    audit = audit_from_store(tmp_path, MISSION, GOAL)
    assert audit.results[0].classification == "tampered"
    assert audit.results[0].reason
    assert audit.failed_closed is True
    assert audit.kpi.tampered == 1


def test_corrupt_json_raises_store_corrupt_error(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    bundle_path(tmp_path, MISSION).write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreCorruptError):
        load_bundle(tmp_path, MISSION)


def test_unknown_schema_raises_store_corrupt_error(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    path = bundle_path(tmp_path, MISSION)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema"] = "criterion_freeze_store/v0"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StoreCorruptError):
        load_bundle(tmp_path, MISSION)


# ---------------------------------------------------------------------------
# fail-closed behaviour
# ---------------------------------------------------------------------------

def test_goal_without_numeric_level_writes_nothing(tmp_path):
    bundle = freeze_to_store(tmp_path, "m9003", "- ship a good report\n",
                             frozen_by="agent/x")
    assert bundle.criteria == []
    assert bundle.is_registered is False
    assert list(tmp_path.glob("*.json")) == []

    audit = audit_from_store(tmp_path, "m9003", "- ship a good report\n")
    assert [r.classification for r in audit.results] == ["unregistered"]
    assert audit.failed_closed is True
    assert audit.kpi.registered_rate == pytest.approx(0.0)


@pytest.mark.parametrize("goal", [None, "", 42, {"goal": "x"}])
def test_non_text_goal_is_fail_closed(tmp_path, goal):
    bundle = freeze_to_store(tmp_path, "m9004", goal, frozen_by="agent/x")
    assert bundle.is_registered is False
    assert list(tmp_path.glob("*.json")) == []


def test_never_frozen_mission_audits_unregistered(tmp_path):
    regs = load_registrations(tmp_path, "m9005")
    assert [r.status for r in regs] == ["unregistered"]
    assert "no frozen bundle" in regs[0].reason
    audit = audit_from_store(tmp_path, "m9005", GOAL)
    assert [r.classification for r in audit.results] == ["unregistered"]
    assert audit.failed_closed is True


def test_missing_report_keeps_registered_criteria_fail_closed(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    audit = audit_from_store(tmp_path, MISSION, "")
    assert {r.classification for r in audit.results} == {"unregistered"}
    assert audit.failed_closed is True


# ---------------------------------------------------------------------------
# idempotency / conflict / path safety
# ---------------------------------------------------------------------------

def test_refreeze_with_different_content_conflicts(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    with pytest.raises(PreregistrationConflict):
        freeze_to_store(tmp_path, MISSION, "- recall >= 0.50 on the held-out split\n",
                        frozen_by="agent/m8695")
    # original pre-registration is untouched
    loaded = load_bundle(tmp_path, MISSION)
    assert loaded.criteria[0]["fields"]["threshold"]["value"] == 0.92


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "x" * 200, None, 7])
def test_invalid_mission_id_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        freeze_to_store(tmp_path, bad, GOAL, frozen_by="agent/x")


def test_no_temp_files_left_behind(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    assert [p.name for p in tmp_path.iterdir()] == [f"{MISSION}.json"]


def test_ledger_persistence_carries_drift_events_across_processes(tmp_path):
    freeze_to_store(tmp_path, MISSION, GOAL, frozen_by="agent/m8695")
    ledger_path = tmp_path / "ledger.jsonl"
    audit = audit_from_store(tmp_path, MISSION,
                             "- recall >= 0.99 on the held-out split\n- latency <= 25 ms of the p95 baseline\n",
                             ledger=CriterionLedger(storage_path=ledger_path))
    assert audit.results[0].classification == "tightened"
    lines = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["classification"] == "tightened"
    assert lines[0]["mission_id"] == MISSION


# ---------------------------------------------------------------------------
# structural guard: the store stays domain-free
# ---------------------------------------------------------------------------

DOMAIN_WORDS = (
    "ice", "hull", "resist", "propuls", "seakeep", "manoeuv", "lindqvist",
    "lbm", "fem", "hydrostat", "ship",
)


def test_store_has_zero_domain_semantics():
    source = (ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry"
              / "store.py").read_text(encoding="utf-8").lower()
    hits = [w for w in DOMAIN_WORDS if w in source]
    assert hits == [], f"domain vocabulary leaked into store.py: {hits}"