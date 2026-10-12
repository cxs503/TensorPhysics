import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_end_to_end_example_writes_checksummed_artifact_and_manifest(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "end_to_end_artifact_run.py"),
            "--output",
            str(tmp_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    artifact = tmp_path / "toy_history.csv"
    manifest_path = tmp_path / "run_manifest.json"
    assert artifact.is_file()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["validation_status"] == "not_run"
    assert manifest["config_sha256"]
    assert len(manifest["outputs"]) == 1
    reference = manifest["outputs"][0]
    assert reference["uri"] == artifact.resolve().as_uri()
    assert reference["checksum_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert reference["size_bytes"] == artifact.stat().st_size
