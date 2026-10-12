"""Run a local artifact-storage round trip with provenance and SHA-256 checks.

This executable smoke example demonstrates the local closed loop:
produce bytes -> store atomically -> verify stored bytes -> serialize a manifest.
It is a software integration check, not a physical solver validation.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from tensorphysics import LocalArtifactStore, RunManifest, RunStatus, ValidationStatus


def run_workflow(output_dir: str | Path, store_root: str | Path, run_id: str) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(store_root)

    history = output / "history.csv"
    history.write_text(
        "time_seconds,force_newtons\n0.0,0.0\n0.1,12.5\n0.2,18.75\n",
        encoding="utf-8",
    )
    history_ref = store.put_file(history, f"{run_id}/history.csv", media_type="text/csv")
    if not store.verify(history_ref):
        raise RuntimeError("stored history artifact failed checksum verification")

    config = {"workflow": "local-artifact-store-smoke", "sample_count": 3}
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCEEDED,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_sha256=RunManifest.digest_config(config),
        code_revisions={"cxs503/TensorPhysics": os.getenv("TENSORPHYSICS_GIT_REVISION", "unknown")},
        environment={"python": platform.python_version(), "platform": platform.platform()},
        resources={"execution": "local", "store": "LocalArtifactStore"},
        outputs=[history_ref],
        validation_status=ValidationStatus.NOT_RUN,
    )
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    manifest_ref = store.put_file(
        manifest_path, f"{run_id}/run_manifest.json", media_type="application/json"
    )
    if not store.verify(manifest_ref):
        raise RuntimeError("stored manifest failed checksum verification")

    # Verify the persisted JSON and its referenced artifact, not only the in-memory objects.
    manifest_uri = urlsplit(manifest_ref.uri)
    persisted_path = Path(unquote(manifest_uri.path))
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    if persisted["run_id"] != run_id or persisted["outputs"][0]["checksum_sha256"] != history_ref.checksum_sha256:
        raise RuntimeError("persisted manifest does not match stored artifact metadata")

    return {
        "run_id": run_id,
        "manifest_uri": manifest_ref.uri,
        "manifest_sha256": manifest_ref.checksum_sha256,
        "artifacts": [history_ref.to_dict()],
        "verified": store.verify(history_ref) and store.verify(manifest_ref),
        "validation_status": manifest.validation_status.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/artifact_store_smoke"))
    parser.add_argument("--store", type=Path, default=Path("outputs/artifact_store"))
    parser.add_argument("--run-id", default="local-artifact-smoke-001")
    args = parser.parse_args()
    result = run_workflow(args.output, args.store, args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verified"]:
        raise SystemExit("artifact store smoke check failed")


if __name__ == "__main__":
    main()
