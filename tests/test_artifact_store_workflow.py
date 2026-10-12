import json
from pathlib import Path

from tensorphysics.artifact_store import LocalArtifactStore
from examples.end_to_end_artifact_store import run_workflow


def test_local_artifact_workflow_stores_and_verifies_manifest_and_outputs(tmp_path: Path):
    result = run_workflow(tmp_path / "output dir", tmp_path / "artifact store", "run-smoke-42")

    assert result["verified"] is True
    assert result["run_id"] == "run-smoke-42"
    assert result["validation_status"] == "not_run"
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["size_bytes"] > 0
    assert len(artifact["checksum_sha256"]) == 64

    store = LocalArtifactStore(tmp_path / "artifact store")
    from tensorphysics.provenance import ArtifactRef

    history_ref = ArtifactRef(**artifact)
    assert store.verify(history_ref) is True
    manifest_path = tmp_path / "artifact store" / "run-smoke-42" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-smoke-42"
    assert manifest["outputs"][0]["checksum_sha256"] == artifact["checksum_sha256"]
