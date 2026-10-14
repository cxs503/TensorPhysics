"""Run one bounded TensorDEM experiment through validation and artifact lineage.

This is a deterministic software/finite-output gate, not a physical ice-fracture
benchmark. TensorDEM must be installed in the active Python environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import torch

from tensorphysics import (
    ArtifactRef, LocalArtifactStore, RunManifest, RunStatus, ValidationStatus,
)
from tensorphysics.research_contracts import (
    ExperimentPlan, ResearchGoal, ResearchMemoryRecord, ValidationReport,
)


def _check_history(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"name": "history_nonempty", "passed": False, "detail": "no data rows"}
    try:
        values = [[float(value) for value in row.values()] for row in rows]
    except (TypeError, ValueError) as exc:
        return {"name": "history_numeric", "passed": False, "detail": str(exc)}
    finite = all(math.isfinite(value) for row in values for value in row)
    times = [row.get("time") for row in rows]
    monotonic = True
    if times and all(value not in (None, "") for value in times):
        try:
            t = [float(value) for value in times]
            monotonic = all(b >= a for a, b in zip(t, t[1:]))
        except ValueError:
            monotonic = False
    return {"name": "history_finite_and_monotonic", "passed": finite and monotonic,
            "detail": f"{len(rows)} rows; finite={finite}; monotonic_time={monotonic}"}


def _check_trajectory(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        positions = payload["positions"]
        if not isinstance(positions, torch.Tensor) or positions.numel() == 0:
            raise ValueError("positions tensor missing or empty")
        finite = bool(torch.isfinite(positions).all())
        times = payload.get("times")
        monotonic = True
        if isinstance(times, torch.Tensor):
            monotonic = bool(torch.isfinite(times).all()) and bool((times[1:] >= times[:-1]).all())
        passed = finite and monotonic
        return {"name": "trajectory_finite_and_time_ordered", "passed": passed,
                "detail": f"positions_shape={list(positions.shape)}; finite={finite}; monotonic_time={monotonic}"}
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        return {"name": "trajectory_readable_and_finite", "passed": False, "detail": str(exc)}


def run_demo(
    work_dir: Path, run_id: str, nx: int, ny: int, steps: int,
    *, parent_run_ids: list[str] | None = None,
) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    output = work_dir / "solver-output"
    output.mkdir(parents=True, exist_ok=True)
    goal = ResearchGoal(
        goal_id=f"{run_id}-goal",
        objective="Execute one bounded TensorDEM run and independently gate output integrity.",
        acceptance_criteria=("solver process exits successfully", "history is finite and time ordered",
                              "trajectory is readable, finite, and time ordered"),
        constraints={"device": "cpu"},
        budget={"max_runs": 1, "max_steps": steps},
    )
    plan = ExperimentPlan(
        plan_id=f"{run_id}-plan", goal_id=goal.goal_id, solver="TensorDEM",
        parameters={"nx": nx, "ny": ny, "steps": steps, "device": "cpu"},
        validation_checks=("history_finite_and_monotonic", "trajectory_finite_and_time_ordered"),
        max_runs=1,
    )
    command = [sys.executable, "-m", "tensordem", "--nx", str(nx), "--ny", str(ny),
               "--steps", str(steps), "--save-every", str(max(1, steps // 5)),
               "--device", "cpu", "--output", str(output)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    checks: list[dict[str, Any]] = [{
        "name": "solver_exit_zero", "passed": completed.returncode == 0,
        "detail": f"returncode={completed.returncode}; stdout={completed.stdout[-1000:]}; stderr={completed.stderr[-1000:]}",
    }]
    if completed.returncode == 0:
        checks.extend((_check_history(output / "history.csv"),
                       _check_trajectory(output / "trajectory.pt")))
    else:
        checks.append({"name": "solver_outputs_available", "passed": False,
                       "detail": "solver failed; output validation skipped"})
    passed = all(item["passed"] for item in checks)
    report = ValidationReport(
        run_id=run_id, status="passed" if passed else "failed", checks=tuple(checks),
        summary="All deterministic output-integrity checks passed." if passed
        else "One or more deterministic output-integrity checks failed; do not promote this run as validated.",
    )
    report_path = work_dir / "validation_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")

    store = LocalArtifactStore(work_dir / "artifact-store")
    output_refs: list[ArtifactRef] = []
    if completed.returncode == 0:
        for path, media_type in ((output / "history.csv", "text/csv"),
                                 (output / "trajectory.pt", "application/octet-stream")):
            ref = store.put_file(path, f"{run_id}/{path.name}", media_type=media_type)
            if not store.verify(ref):
                raise RuntimeError(f"artifact store verification failed: {path.name}")
            output_refs.append(ref)
    report_ref = store.put_file(report_path, f"{run_id}/validation_report.json", media_type="application/json")
    if not store.verify(report_ref):
        raise RuntimeError("validation report artifact verification failed")

    config = plan.to_dict()
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCEEDED if completed.returncode == 0 else RunStatus.FAILED,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_sha256=RunManifest.digest_config(config),
        code_revisions={"cxs503/TensorPhysics": os.getenv("TENSORPHYSICS_GIT_REVISION", "unknown"),
                        "cxs503/TensorDEM": os.getenv("TENSORDEM_GIT_REVISION", "unknown")},
        environment={"python": sys.version.split()[0], "platform": sys.platform},
        resources={"execution": "local", "device": "cpu"},
        outputs=output_refs,
        validation_status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
        validation_report=report_ref,
        parent_run_ids=list(parent_run_ids or []),
    )
    manifest_path = work_dir / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    manifest_ref = store.put_file(manifest_path, f"{run_id}/run_manifest.json", media_type="application/json")
    if not store.verify(manifest_ref):
        raise RuntimeError("run manifest artifact verification failed")

    next_action = "run_bounded_parameter_sweep" if passed else "stop_and_repair_before_retry"
    memory = ResearchMemoryRecord(
        goal_id=goal.goal_id, run_id=run_id,
        finding=report.summary,
        evidence_uris=tuple(ref.uri for ref in [*output_refs, report_ref, manifest_ref]),
        next_action=next_action,
    )
    memory_path = work_dir / "research_memory.json"
    memory_path.write_text(json.dumps(memory.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    result = {
        "goal": goal.to_dict(), "plan": plan.to_dict(), "solver_command": command,
        "solver_returncode": completed.returncode, "validation_report": report.to_dict(),
        "run_manifest": manifest.to_dict(), "manifest_artifact": manifest_ref.to_dict(),
        "research_memory": memory.to_dict(), "work_dir": str(work_dir),
    }
    (work_dir / "demo_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/research-loop-dem"))
    parser.add_argument("--run-id", default="research-loop-dem-001")
    parser.add_argument("--nx", type=int, default=7)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--parent-run-id", action="append", default=[],
                        help="parent run ID; may be supplied more than once")
    args = parser.parse_args()
    if min(args.nx, args.ny, args.steps) < 1:
        parser.error("--nx, --ny and --steps must be positive")
    result = run_demo(
        args.work_dir, args.run_id, args.nx, args.ny, args.steps,
        parent_run_ids=args.parent_run_id,
    )
    print(json.dumps({
        "run_id": result["run_manifest"]["run_id"],
        "validation_status": result["run_manifest"]["validation_status"],
        "next_action": result["research_memory"]["next_action"],
        "work_dir": result["work_dir"],
    }, indent=2, sort_keys=True))
    if result["validation_report"]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
