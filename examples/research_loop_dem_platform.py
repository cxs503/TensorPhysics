"""Run TensorDEM, upload real artifacts to S3, and register/retrieve the run.

Requires TensorDEM, TensorPhysics, boto3, S3 credentials, and an authorized
TensorPlatform API token. No network calls occur unless explicitly run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tensorphysics import S3ArtifactStore, TensorPlatformRunRegistry
from tensorphysics.platform_manifest import build_platform_manifest


def run_and_register(
    work_dir: Path,
    run_id: str,
    nx: int,
    ny: int,
    steps: int,
    *,
    bucket: str,
    prefix: str,
    endpoint_url: str | None = None,
    parent_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    root = work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    demo_script = Path(__file__).with_name("research_loop_dem.py")
    command = [
        sys.executable, str(demo_script), "--work-dir", str(root),
        "--run-id", run_id, "--nx", str(nx), "--ny", str(ny), "--steps", str(steps),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result_path = root / "demo_result.json"
    if not result_path.is_file():
        raise RuntimeError(
            "TensorDEM research loop produced no demo_result.json; "
            f"returncode={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    store = S3ArtifactStore(
        bucket, prefix=prefix, endpoint_url=endpoint_url,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
    )
    solver_dir = root / "solver-output"
    candidates = [
        ("solver-history", solver_dir / "history.csv", "text/csv"),
        ("solver-trajectory", solver_dir / "trajectory.pt", "application/octet-stream"),
    ]
    uploaded_refs = []
    for role, path, media_type in candidates:
        if not path.is_file():
            continue
        ref = store.put_file(path, f"{run_id}/{path.name}", media_type=media_type)
        if not store.verify(ref):
            raise RuntimeError(f"S3 verification failed for {path.name}")
        uploaded_refs.append((role, ref))

    report_path = root / "validation_report.json"
    if not report_path.is_file():
        raise RuntimeError("research loop did not write validation_report.json")
    report_ref = store.put_file(
        report_path, f"{run_id}/validation_report.json", media_type="application/json"
    )
    if not store.verify(report_ref):
        raise RuntimeError("S3 verification failed for validation report")

    manifest = build_platform_manifest(
        result, uploaded_refs, report_ref,
        producer_revision=os.getenv("TENSORPHYSICS_GIT_REVISION", "unknown"),
    )
    registry = TensorPlatformRunRegistry.from_env()
    registered = registry.register(manifest)
    retrieved = registry.get(run_id)
    retrieved_manifest = retrieved["manifest"]
    if retrieved_manifest != manifest:
        raise RuntimeError("retrieved platform manifest differs from the submitted manifest")

    summary = {
        "run_id": run_id,
        "solver_process_returncode": completed.returncode,
        "solver_returncode": result["solver_returncode"],
        "validation_status": manifest["validation_status"],
        "uploaded_artifacts": [
            {"role": role, "uri": ref.uri, "sha256": ref.checksum_sha256, "size_bytes": ref.size_bytes}
            for role, ref in uploaded_refs
        ] + [{
            "role": "validation-report", "uri": report_ref.uri,
            "sha256": report_ref.checksum_sha256, "size_bytes": report_ref.size_bytes,
        }],
        "platform_registration": registered,
        "platform_retrieval_verified": True,
        "work_dir": str(root),
    }
    (root / "platform_registration_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/research-loop-dem-platform"))
    parser.add_argument("--run-id", default="research-loop-dem-platform-001")
    parser.add_argument("--nx", type=int, default=7)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bucket", default=os.getenv("TENSORPHYSICS_S3_BUCKET"))
    parser.add_argument("--prefix", default=os.getenv("TENSORPHYSICS_S3_PREFIX", "tensorphysics-runs"))
    parser.add_argument("--endpoint-url", default=os.getenv("TENSORPHYSICS_S3_ENDPOINT_URL"))
    parser.add_argument("--parent-run-id", action="append", default=[],
                        help="parent run ID; may be supplied more than once")
    args = parser.parse_args()
    if min(args.nx, args.ny, args.steps) < 1:
        parser.error("--nx, --ny and --steps must be positive")
    if not args.bucket:
        parser.error("--bucket or TENSORPHYSICS_S3_BUCKET is required")
    try:
        result = run_and_register(
            args.work_dir, args.run_id, args.nx, args.ny, args.steps,
            bucket=args.bucket, prefix=args.prefix, endpoint_url=args.endpoint_url,
            parent_run_ids=args.parent_run_id,
        )
    except Exception as exc:
        print(f"research-loop registration failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["validation_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
