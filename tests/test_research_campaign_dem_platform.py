"""Unit tests for platform-backed campaign gating without network or solver calls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import research_campaign_dem_platform as campaign  # noqa: E402


def _registered(status: str, run_id: str, parents: list[str]) -> dict:
    return {
        "run_id": run_id,
        "validation_status": status,
        "platform_retrieval_verified": True,
        "lineage_verified": True,
        "uploaded_artifacts": [{"role": "validation-report", "uri": f"s3://bucket/{run_id}/report.json"}],
        "work_dir": f"/tmp/{run_id}",
        "parent_run_ids": parents,
    }


def test_campaign_links_child_to_passing_parent_and_writes_report(tmp_path: Path) -> None:
    calls = []

    def fake_register(work_dir, run_id, nx, ny, steps, **kwargs):
        calls.append((run_id, nx, kwargs["parent_run_ids"]))
        return _registered("passed", run_id, kwargs["parent_run_ids"])

    report = campaign.run_campaign(
        tmp_path, "campaign-a", 7, 3, 20, bucket="bucket",
        register_run=fake_register,
    )

    assert calls == [
        ("campaign-a-r01", 7, []),
        ("campaign-a-r02", 8, ["campaign-a-r01"]),
    ]
    assert report["decision"] == "campaign_complete_all_gates_passed"
    assert report["runs"][1]["parent_run_ids"] == ["campaign-a-r01"]
    saved = json.loads((tmp_path / "campaign-a" / "campaign_report.json").read_text())
    assert saved["scientific_status"] == "not_a_physics_benchmark"


def test_campaign_stops_after_failed_validation(tmp_path: Path) -> None:
    calls = []

    def fake_register(work_dir, run_id, nx, ny, steps, **kwargs):
        calls.append(run_id)
        return _registered("failed", run_id, kwargs["parent_run_ids"])

    report = campaign.run_campaign(
        tmp_path, "campaign-fail", 7, 3, 20, bucket="bucket",
        register_run=fake_register,
    )
    assert calls == ["campaign-fail-r01"]
    assert report["decision"] == "stop_campaign_and_repair_or_verify_failed_gate"
    assert len(report["runs"]) == 1


def test_campaign_stops_if_platform_retrieval_was_not_verified(tmp_path: Path) -> None:
    calls = []

    def fake_register(work_dir, run_id, nx, ny, steps, **kwargs):
        calls.append(run_id)
        result = _registered("passed", run_id, kwargs["parent_run_ids"])
        result["platform_retrieval_verified"] = False
        return result

    report = campaign.run_campaign(
        tmp_path, "campaign-unverified", 7, 3, 20, bucket="bucket",
        register_run=fake_register,
    )
    assert calls == ["campaign-unverified-r01"]
    assert report["decision"] == "stop_campaign_and_repair_or_verify_failed_gate"




def test_campaign_stops_if_registered_lineage_does_not_match(tmp_path: Path) -> None:
    calls = []

    def fake_register(work_dir, run_id, nx, ny, steps, **kwargs):
        calls.append(run_id)
        result = _registered("passed", run_id, kwargs["parent_run_ids"])
        result["lineage_verified"] = False
        return result

    report = campaign.run_campaign(
        tmp_path, "campaign-lineage-mismatch", 7, 3, 20, bucket="bucket",
        register_run=fake_register,
    )
    assert calls == ["campaign-lineage-mismatch-r01"]
    assert report["decision"] == "stop_campaign_and_repair_or_verify_failed_gate"
    assert report["runs"][0]["lineage_verified"] is False

@pytest.mark.parametrize("campaign_id", ["../escape", "/tmp/x", "", "has space"])
def test_campaign_rejects_unsafe_campaign_ids(tmp_path: Path, campaign_id: str) -> None:
    with pytest.raises(ValueError, match="safe filename"):
        campaign.run_campaign(
            tmp_path, campaign_id, 7, 3, 20, bucket="bucket",
            register_run=lambda *args, **kwargs: {},
        )
