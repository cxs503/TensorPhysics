from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from tensorphysics.provenance import ArtifactRef
import examples.end_to_end_s3_artifact_store as workflow


class FakeS3ArtifactStore:
    """No-network store fake for exercising workflow/manifest integration."""

    def __init__(self, bucket: str, *, prefix: str = "", endpoint_url=None, region_name=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.objects: dict[str, tuple[str, int, str]] = {}

    def put_file(self, source, key: str, *, media_type: str) -> ArtifactRef:
        path = Path(source)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        object_key = f"{self.prefix}/{key}" if self.prefix else key
        uri = f"s3://{self.bucket}/{object_key}"
        self.objects[uri] = (digest, len(payload), media_type)
        return ArtifactRef(
            uri=uri, media_type=media_type, checksum_sha256=digest,
            size_bytes=len(payload), metadata={"storage": "s3", "verified": True},
        )

    def verify(self, artifact: ArtifactRef) -> bool:
        record = self.objects.get(artifact.uri)
        return record == (artifact.checksum_sha256, artifact.size_bytes, artifact.media_type)


def test_s3_workflow_writes_manifest_and_references_verified_artifacts(tmp_path, monkeypatch):
    store = FakeS3ArtifactStore("test-bucket", prefix="ci-runs")
    monkeypatch.setattr(workflow, "S3ArtifactStore", lambda *args, **kwargs: store)

    result = workflow.run_workflow(
        tmp_path / "output", "ci-run-001", bucket="test-bucket", prefix="ci-runs"
    )

    assert result["verified"] is True
    assert result["validation_status"] == "not_run"
    assert result["manifest_uri"] == "s3://test-bucket/ci-runs/ci-run-001/run_manifest.json"
    assert result["artifacts"][0]["uri"] == "s3://test-bucket/ci-runs/ci-run-001/history.csv"
    persisted = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    assert persisted["run_id"] == "ci-run-001"
    assert persisted["status"] == "succeeded"
    assert persisted["validation_status"] == "not_run"
    assert persisted["outputs"][0]["uri"] == result["artifacts"][0]["uri"]
    assert persisted["outputs"][0]["checksum_sha256"] == hashlib.sha256(
        (tmp_path / "output" / "history.csv").read_bytes()
    ).hexdigest()
    assert len(store.objects) == 2
