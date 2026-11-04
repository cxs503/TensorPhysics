"""G-A/B/C consumer-side contract tests (TensorPhysics mirror).

Locks: the embedded contract copies (byte-identical to TensorPlatform
backend/contracts/), the platform_registry vocabularies derived from them,
and the TaskStatus -> TP mission-status mapping. TensorPlatform's
backend/tests/test_status_contracts.py locks the producer side -- a one-sided
edit on either repo fails exactly one of the two suites
(docs/architecture/migration_disposition_2026-11.md SS3, upstream).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics import status_map
from tensorphysics.contracts import (
    mission_status_contract,
    mission_task_status_map_contract,
    research_run_contract,
)
from tensorphysics.platform_registry import (
    _MAX_MANIFEST_BYTES,
    _RUN_STATUSES,
    _VALIDATION_STATUSES,
)
from tensorphysics.research_runtime import TaskStatus

SRC = Path(__file__).resolve().parents[1] / "src" / "tensorphysics"
CONTRACTS = SRC / "contracts"


class TestEmbeddedContracts:
    def test_copies_wellformed(self):
        for name, expected in (
            ("mission_status_vocab.v1.json", "mission_status_vocab"),
            ("research_run_status_vocab.v1.json", "research_run_status_vocab"),
            ("mission_task_status_map.v1.json", "mission_task_status_map"),
        ):
            doc = json.loads((CONTRACTS / name).read_text())
            assert doc["contract"] == expected
            assert isinstance(doc["version"], int) and doc["version"] >= 1

    def test_mission_vocab_shape(self):
        c = mission_status_contract()
        assert len(c["statuses"]) == 11
        assert set(c["terminal"]) <= set(c["statuses"])
        assert set(c["platform_only"]) <= set(c["statuses"])

    def test_mission_vocab_v2_pr_merge_gate(self):
        """v2 (owner 2026-11-05): awaiting_pr_merge 非终态、platform_only。"""
        c = mission_status_contract()
        assert c["version"] >= 2
        assert "awaiting_pr_merge" in c["statuses"]
        assert "awaiting_pr_merge" in c["platform_only"]
        assert "awaiting_pr_merge" not in c["terminal"]

    def test_cross_repo_diff_opt_in(self):
        """Byte-compare against TensorPlatform main (opt-in; honest skip)."""
        import os
        import urllib.request

        base = os.environ.get("TPH_CONTRACT_REMOTE", "")
        if not base:
            pytest.skip("set TPH_CONTRACT_REMOTE=<raw base url> to enable the diff=0 check")
        token = os.environ.get("GITHUB_TOKEN", "")
        for f in sorted(CONTRACTS.glob("*.v1.json")):
            req = urllib.request.Request(f"{base.rstrip('/')}/{f.name}")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            remote = urllib.request.urlopen(req, timeout=15).read()
            assert remote == f.read_bytes(), f"{f.name} drifted from upstream"


class TestRegistryVocabDerived:
    def test_frozensets_equal_contract(self):
        c = research_run_contract()
        assert _RUN_STATUSES == frozenset(c["run_statuses"])
        assert _VALIDATION_STATUSES == frozenset(c["validation_statuses"])
        assert _MAX_MANIFEST_BYTES == c["max_manifest_bytes"] == 1_000_000

    def test_hand_kept_vocab_is_gone(self):
        src = (SRC / "platform_registry.py").read_text()
        assert "Keep this in sync" not in src
        assert '"planned", "validating"' not in src


class TestStatusMap:
    def test_import_time_coverage_holds(self):
        # importing status_map already validates 100% TaskStatus coverage with
        # no orphans; reaching here is the assertion
        assert status_map.to_platform(TaskStatus.VERIFYING) == "running"

    def test_maps_each_status_per_contract(self):
        assert status_map.to_platform(TaskStatus.QUEUED) == "queued"
        assert status_map.to_platform(TaskStatus.CLAIMED) == "claimed"
        for s in (TaskStatus.RUNNING, TaskStatus.VERIFYING,
                  TaskStatus.WAITING_INPUT, TaskStatus.RETRY_WAIT):
            assert status_map.to_platform(s) == "running"
            assert not status_map.is_terminal_on_platform(s)
        assert status_map.to_platform(TaskStatus.SUCCEEDED) == "succeeded"
        assert status_map.to_platform(TaskStatus.FAILED) == "failed"
        assert status_map.to_platform(TaskStatus.CANCELLED) == "stopped"
        assert status_map.is_terminal_on_platform(TaskStatus.SUCCEEDED)
        assert status_map.is_terminal_on_platform(TaskStatus.FAILED)

    def test_platform_only_never_a_mapping_target(self):
        platform_only = set(mission_status_contract()["platform_only"])
        images = {r["tp"] for r in mission_task_status_map_contract()["mapping"]}
        assert platform_only & images == set()

    def test_unknown_status_rejected(self):
        with pytest.raises(ValueError):
            status_map.to_platform("paused")


class TestGB_SingleWriter:
    def test_run_registry_writes_no_events(self):
        src = (SRC / "platform_registry.py").read_text()
        assert "append_event" not in src

    def test_runner_registers_only_terminal(self):
        src = (SRC / "research_runner.py").read_text()
        assert "status_map.to_platform" in src
        assert 'status="succeeded"' not in src
        assert 'status="failed"' not in src
