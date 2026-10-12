"""Unit tests for fail-closed campaign control without invoking the solver."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import research_campaign_dem as campaign_module  # noqa: E402


def _result(status: str, run_id: str) -> dict:
    return {
        "validation_report": {
            "status": status,
            "checks": [{"name": "injected_gate", "passed": status == "passed"}],
        },
        "manifest_artifact": {
            "uri": f"file:///artifacts/{run_id}/run_manifest.json",
            "checksum_sha256": "a" * 64,
        },
        "research_memory": {
            "next_action": "run_bounded_parameter_sweep" if status == "passed"
            else "stop_and_repair_before_retry",
        },
    }


def test_campaign_completes_after_two_passed_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_run_demo(work_dir: Path, run_id: str, nx: int, ny: int, steps: int) -> dict:
        calls.append((run_id, nx))
        return _result("passed", run_id)

    monkeypatch.setattr(campaign_module, "run_demo", fake_run_demo)
    report = campaign_module.run_campaign(tmp_path, "test-campaign", 7, 3, 20)

    assert calls == [("test-campaign-r01", 7), ("test-campaign-r02", 8)]
    assert report["decision"] == "bounded_campaign_complete_integrity_gates_passed"
    assert len(report["runs"]) == 2
    assert json.loads((tmp_path / "campaign_report.json").read_text())["scientific_status"] == "not_a_physics_benchmark"


def test_campaign_stops_immediately_after_first_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_run_demo(work_dir: Path, run_id: str, nx: int, ny: int, steps: int) -> dict:
        calls.append(run_id)
        return _result("failed", run_id)

    monkeypatch.setattr(campaign_module, "run_demo", fake_run_demo)
    report = campaign_module.run_campaign(tmp_path, "failure-injection", 7, 3, 20)

    assert calls == ["failure-injection-r01"]
    assert report["decision"] == "stop_campaign_and_repair_failed_gate"
    assert len(report["runs"]) == 1
    assert report["runs"][0]["validation_status"] == "failed"


def test_campaign_stops_after_second_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_run_demo(work_dir: Path, run_id: str, nx: int, ny: int, steps: int) -> dict:
        calls.append(run_id)
        return _result("passed" if len(calls) == 1 else "failed", run_id)

    monkeypatch.setattr(campaign_module, "run_demo", fake_run_demo)
    report = campaign_module.run_campaign(tmp_path, "second-round-failure", 7, 3, 20)

    assert calls == ["second-round-failure-r01", "second-round-failure-r02"]
    assert report["decision"] == "stop_campaign_and_repair_failed_gate"
    assert [run["validation_status"] for run in report["runs"]] == ["passed", "failed"]
