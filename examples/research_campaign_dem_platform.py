"""Run a bounded TensorDEM campaign and register each run with lineage.

The two-point nx sweep is an integration test, not a physical parameter study.
Each run is uploaded and registered before the next candidate is allowed to run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from research_loop_dem_platform import run_and_register


RegisterRun = Callable[..., dict[str, Any]]


def run_campaign(
    work_dir: Path,
    campaign_id: str,
    nx: int,
    ny: int,
    steps: int,
    *,
    bucket: str,
    prefix: str,
    endpoint_url: str | None = None,
    register_run: RegisterRun = run_and_register,
) -> dict[str, Any]:
    """Run baseline then one bounded variant, failing closed on any failed gate."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", campaign_id):
        raise ValueError("campaign_id must be 1-100 safe filename characters")
    if min(nx, ny, steps) < 1:
        raise ValueError("nx, ny and steps must be positive")
    root = work_dir.expanduser().resolve() / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    designs = (
        {"label": "baseline", "nx": nx},
        {"label": "variant", "nx": nx + 1},
    )
    runs: list[dict[str, Any]] = []
    parent_run_ids: list[str] = []
    for index, design in enumerate(designs, start=1):
        run_id = f"{campaign_id}-r{index:02d}"
        result = register_run(
            root / f"round-{index:02d}",
            run_id,
            design["nx"],
            ny,
            steps,
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint_url,
            parent_run_ids=list(parent_run_ids),
        )
        status = result.get("validation_status", "incomplete")
        runs.append({
            "round": index,
            "label": design["label"],
            "run_id": run_id,
            "parent_run_ids": list(parent_run_ids),
            "parameters": {"nx": design["nx"], "ny": ny, "steps": steps, "device": "cpu"},
            "validation_status": status,
            "platform_retrieval_verified": bool(result.get("platform_retrieval_verified")),
            "uploaded_artifacts": result.get("uploaded_artifacts", []),
            "work_dir": result.get("work_dir"),
        })
        # Only a passing prior run may authorize spending the next run's budget.
        if status != "passed" or not result.get("platform_retrieval_verified"):
            break
        parent_run_ids = [run_id]

    completed = len(runs) == len(designs) and all(
        run["validation_status"] == "passed" and run["platform_retrieval_verified"]
        for run in runs
    )
    report = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "design": "two-point bounded software integration sweep; nx varies by +1",
        "budget": {"max_runs": len(designs), "max_steps_per_run": steps},
        "runs": runs,
        "decision": "campaign_complete_all_gates_passed" if completed
                    else "stop_campaign_and_repair_or_verify_failed_gate",
        "scientific_status": "not_a_physics_benchmark",
    }
    report_path = root / "campaign_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/research-campaign-dem-platform"))
    parser.add_argument("--campaign-id", default="research-campaign-platform-001")
    parser.add_argument("--nx", type=int, default=7)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bucket", default=os.getenv("TENSORPHYSICS_S3_BUCKET"))
    parser.add_argument("--prefix", default=os.getenv("TENSORPHYSICS_S3_PREFIX", "tensorphysics-runs"))
    parser.add_argument("--endpoint-url", default=os.getenv("TENSORPHYSICS_S3_ENDPOINT_URL"))
    args = parser.parse_args()
    if min(args.nx, args.ny, args.steps) < 1:
        parser.error("--nx, --ny and --steps must be positive")
    if not args.bucket:
        parser.error("--bucket or TENSORPHYSICS_S3_BUCKET is required")
    try:
        report = run_campaign(
            args.work_dir, args.campaign_id, args.nx, args.ny, args.steps,
            bucket=args.bucket, prefix=args.prefix, endpoint_url=args.endpoint_url,
        )
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps({
        "campaign_id": report["campaign_id"],
        "run_count": len(report["runs"]),
        "decision": report["decision"],
        "report": str(args.work_dir.expanduser().resolve() / args.campaign_id / "campaign_report.json"),
    }, indent=2, sort_keys=True))
    if report["decision"] != "campaign_complete_all_gates_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
