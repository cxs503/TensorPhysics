from __future__ import annotations

import pytest

from tensorphysics.platform_manifest import build_platform_manifest
from tensorphysics.provenance import ArtifactRef


def _ref(uri: str, body: bytes, media_type: str = "application/json") -> ArtifactRef:
    import hashlib
    return ArtifactRef(uri=uri, media_type=media_type,
                       checksum_sha256=hashlib.sha256(body).hexdigest(),
                       size_bytes=len(body))


def _result(status: str = "succeeded", validation_status: str = "passed"):
    return {
        "goal": {"objective": "bounded DEM test"},
        "plan": {"parameters": {"nx": 7, "ny": 3, "steps": 20}},
        "solver_returncode": 0 if status == "succeeded" else 1,
        "validation_report": {
            "status": validation_status, "summary": "deterministic checks",
            "checks": [{"name": "finite", "passed": validation_status == "passed"}],
        },
        "run_manifest": {
            "schema_version": 1, "run_id": "dem-run-01", "status": status,
            "validation_status": validation_status,
            "created_at": "2026-10-14T00:00:00+00:00", "config_sha256": "a" * 64,
            "environment": {"python": "3.12"}, "resources": {"execution": "local"},
            "inputs": [], "parent_run_ids": [],
        },
    }


def test_builds_versioned_manifest_with_checksum_backed_outputs():
    history = _ref("s3://bucket/runs/dem-run-01/history.csv", b"time,force\n0,1\n", "text/csv")
    report = _ref("s3://bucket/runs/dem-run-01/report.json", b'{"status":"passed"}')
    manifest = build_platform_manifest(_result(), [("solver-history", history)], report,
                                       producer_revision="abc123")
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "dem-run-01"
    assert manifest["status"] == "succeeded"
    assert manifest["validation_status"] == "passed"
    assert manifest["producer"]["revision"] == "abc123"
    assert {item["role"] for item in manifest["outputs"]} == {"solver-history", "validation-report"}
    assert all(item["checksum_sha256"] and item["size_bytes"] >= 0 for item in manifest["outputs"])
    assert manifest["validation"]["report_uri"] == report.uri


def test_failed_run_is_not_promoted_by_manifest_adapter():
    report = _ref("s3://bucket/runs/dem-run-01/report.json", b'{"status":"failed"}')
    manifest = build_platform_manifest(_result("failed", "failed"), [], report)
    assert manifest["status"] == "failed"
    assert manifest["validation_status"] == "failed"
    assert manifest["failure"]["reason"] == "deterministic checks"


def test_platform_manifest_rejects_local_file_artifact_references():
    local = _ref("file:///tmp/history.csv", b"time\n0\n", "text/csv")
    report = _ref("s3://bucket/runs/dem-run-01/report.json", b'{}')
    with pytest.raises(ValueError, match="shared remote URI"):
        build_platform_manifest(_result(), [("solver-history", local)], report)


def test_platform_manifest_rejects_artifacts_without_checksums():
    local_ref = ArtifactRef(uri="s3://bucket/runs/x/history.csv", media_type="text/csv")
    report = _ref("s3://bucket/runs/x/report.json", b'{}')
    with pytest.raises(ValueError, match="checksum and size"):
        build_platform_manifest(_result(), [("solver-history", local_ref)], report)
