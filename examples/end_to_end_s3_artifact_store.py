"""End-to-end S3 artifact workflow, optionally registering metadata in TensorPlatform.

Requires: pip install -e '.[s3]'
Configure AWS credentials with boto3's normal provider chain. For S3-compatible
services, set TENSORPHYSICS_S3_ENDPOINT_URL. No cloud calls occur unless this
example is run explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorphysics import RunManifest, RunStatus, S3ArtifactStore, ValidationStatus
from tensorphysics.platform_registry import TensorPlatformRunRegistry


def run_workflow(output_dir: str | Path, run_id: str, *, bucket: str,
                 prefix: str = "tensorphysics-runs", endpoint_url: str | None = None,
                 register_platform: bool = False) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = S3ArtifactStore(
        bucket, prefix=prefix, endpoint_url=endpoint_url,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
    )

    # Replace this small CSV with a solver-produced artifact in an application adapter.
    history = output / "history.csv"
    history.write_text(
        "time_seconds,force_newtons\n0.0,0.0\n0.1,12.5\n0.2,18.75\n",
        encoding="utf-8",
    )
    history_ref = store.put_file(history, f"{run_id}/history.csv", media_type="text/csv")
    if not store.verify(history_ref):
        raise RuntimeError("remote history artifact failed HEAD metadata verification")

    config = {"workflow": "s3-artifact-store-smoke", "sample_count": 3}
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCEEDED,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_sha256=RunManifest.digest_config(config),
        code_revisions={"cxs503/TensorPhysics": os.getenv("TENSORPHYSICS_GIT_REVISION", "unknown")},
        environment={"python": platform.python_version(), "platform": platform.platform()},
        resources={"execution": "local", "store": "S3ArtifactStore", "bucket": bucket},
        outputs=[history_ref],
        validation_status=ValidationStatus.NOT_RUN,
    )
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    manifest_ref = store.put_file(
        manifest_path, f"{run_id}/run_manifest.json", media_type="application/json"
    )
    if not store.verify(manifest_ref):
        raise RuntimeError("remote manifest failed HEAD metadata verification")

    result: dict[str, Any] = {
        "run_id": run_id,
        "manifest_uri": manifest_ref.uri,
        "manifest_sha256": manifest_ref.checksum_sha256,
        "artifacts": [history_ref.to_dict()],
        "verified": store.verify(history_ref) and store.verify(manifest_ref),
        "validation_status": manifest.validation_status.value,
        "platform_registration": None,
    }
    if register_platform:
        registry = TensorPlatformRunRegistry.from_env()
        result["platform_registration"] = registry.register(manifest.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.getenv("TENSORPHYSICS_S3_BUCKET"),
                        help="S3 bucket (or set TENSORPHYSICS_S3_BUCKET)")
    parser.add_argument("--prefix", default=os.getenv("TENSORPHYSICS_S3_PREFIX", "tensorphysics-runs"))
    parser.add_argument("--endpoint-url", default=os.getenv("TENSORPHYSICS_S3_ENDPOINT_URL"))
    parser.add_argument("--output", type=Path, default=Path("outputs/s3_artifact_smoke"))
    parser.add_argument("--run-id", default="s3-artifact-smoke-001")
    parser.add_argument("--register-platform", action="store_true",
                        help="also register metadata using TensorPlatform environment settings")
    args = parser.parse_args()
    if not args.bucket:
        parser.error("--bucket or TENSORPHYSICS_S3_BUCKET is required")
    try:
        result = run_workflow(
            args.output, args.run_id, bucket=args.bucket, prefix=args.prefix,
            endpoint_url=args.endpoint_url, register_platform=args.register_platform,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["verified"]:
        raise SystemExit("S3 artifact workflow verification failed")


if __name__ == "__main__":
    main()
