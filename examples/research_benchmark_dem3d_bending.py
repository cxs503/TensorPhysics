"""Run TensorDEM's DEM3D three-point-bending fixture with independent evidence gates.

This is a numerical load-transfer/fracture-tracking integration benchmark, not
calibrated flexural-strength validation. TensorDEM must be checked out locally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from tensorphysics import ArtifactRef, LocalArtifactStore, RunManifest, RunStatus, ValidationStatus
from tensorphysics.research_contracts import ResearchGoal, ResearchMemoryRecord, ValidationReport


def _check_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["rows"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("report rows must be a non-empty list")
        numeric_keys = (
            "time_s", "applied_load_N", "support_reaction_z_N",
            "vertical_force_balance_residual_N", "midspan_deflection_m",
            "kinetic_energy_J", "bond_elastic_energy_J", "mechanical_energy_J",
        )
        for index, row in enumerate(rows):
            for key in numeric_keys:
                value = float(row[key])
                if not math.isfinite(value):
                    raise ValueError(f"row {index} has non-finite {key}")
        broken = [int(row["broken_bonds"]) for row in rows]
        if any(b < a for a, b in zip(broken, broken[1:])):
            raise ValueError("broken_bonds is not monotonic")
        times = [float(row["time_s"]) for row in rows]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("time_s is not monotonic")
        return report, rows
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid three-point-bending report: {exc}") from exc


def run_benchmark(work_dir: Path, tensordem_root: Path, run_id: str, steps: int) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    tensordem_root = tensordem_root.resolve()
    script = tensordem_root / "scripts" / "benchmark_dem3d_three_point_bending.py"
    if not script.is_file():
        raise FileNotFoundError(f"TensorDEM benchmark script not found: {script}")
    output = work_dir / "solver-output"
    output.mkdir(parents=True, exist_ok=True)

    goal = ResearchGoal(
        goal_id=f"{run_id}-goal",
        objective="Run the TensorDEM 3D bending fixture and independently validate numerical evidence.",
        acceptance_criteria=(
            "benchmark process exits successfully",
            "report and CSV history are readable and finite",
            "load transfer, deflection, and fracture-history gates pass",
        ),
        constraints={"device": "cpu", "benchmark": "three_point_bending_load_transfer_smoke"},
        budget={"max_runs": 1, "max_steps": steps},
    )
    command = [
        sys.executable, str(script), "--output", str(output), "--steps", str(steps),
    ]
    completed = subprocess.run(
        command, cwd=tensordem_root, capture_output=True, text=True, check=False,
    )
    checks: list[dict[str, Any]] = [{
        "name": "benchmark_exit_zero",
        "passed": completed.returncode == 0,
        "detail": f"returncode={completed.returncode}; stdout={completed.stdout[-1500:]}; stderr={completed.stderr[-1500:]}",
    }]

    report_path = output / "three_point_bending_report.json"
    history_path = output / "three_point_bending_history.csv"
    benchmark_report: dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            benchmark_report, rows = _check_report(report_path)
            with history_path.open(newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            checks.extend([
                {"name": "report_and_csv_readable", "passed": bool(csv_rows) and len(csv_rows) == len(rows),
                 "detail": f"json_rows={len(rows)}; csv_rows={len(csv_rows)}"},
                {"name": "benchmark_verdict_pass", "passed": benchmark_report.get("verdict") == "PASS",
                 "detail": f"verdict={benchmark_report.get('verdict')}; failed_checks={benchmark_report.get('failed_checks', [])}"},
                {"name": "load_transfer_and_deflection",
                 "passed": float(benchmark_report["peak_support_reaction_abs_N"]) > 0
                           and float(benchmark_report["peak_midspan_deflection_abs_m"]) > 0,
                 "detail": "peak support reaction and midspan deflection must both be positive"},
                {"name": "finite_ordered_history",
                 "passed": all(math.isfinite(float(row["mechanical_energy_J"])) for row in rows)
                           and all(float(b["time_s"]) >= float(a["time_s"]) for a, b in zip(rows, rows[1:]))
                           and all(int(b["broken_bonds"]) >= int(a["broken_bonds"]) for a, b in zip(rows, rows[1:])),
                 "detail": "mechanical energy finite, time nondecreasing, broken-bond count nondecreasing"},
            ])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks.append({"name": "report_independently_validated", "passed": False, "detail": str(exc)})
    else:
        checks.append({"name": "benchmark_outputs_available", "passed": False,
                       "detail": "benchmark failed; report validation skipped"})

    passed = all(item["passed"] for item in checks)
    validation = ValidationReport(
        run_id=run_id, status="passed" if passed else "failed", checks=tuple(checks),
        summary=("Numerical bending-fixture evidence gates passed; this is not physical calibration."
                 if passed else "Evidence gate failed; do not promote this run."),
    )
    validation_path = work_dir / "validation_report.json"
    validation_path.write_text(json.dumps(validation.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
                               encoding="utf-8")

    store = LocalArtifactStore(work_dir / "artifact-store")
    refs: list[ArtifactRef] = []
    for path, media_type in (
        (history_path, "text/csv"), (report_path, "application/json"),
        (validation_path, "application/json"),
    ):
        if path.is_file():
            ref = store.put_file(path, f"{run_id}/{path.name}", media_type=media_type)
            if not store.verify(ref):
                raise RuntimeError(f"artifact verification failed: {path.name}")
            refs.append(ref)

    config = {"benchmark": "tensordem-dem3d-three-point-bending-v1", "steps": steps, "device": "cpu"}
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCEEDED if completed.returncode == 0 else RunStatus.FAILED,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config, config_sha256=RunManifest.digest_config(config),
        code_revisions={
            "cxs503/TensorPhysics": os.getenv("TENSORPHYSICS_GIT_REVISION", "unknown"),
            "cxs503/TensorDEM": os.getenv("TENSORDEM_GIT_REVISION", "unknown"),
        },
        environment={"python": sys.version.split()[0], "platform": sys.platform},
        resources={"execution": "local", "device": "cpu"},
        outputs=refs,
        validation_status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
        validation_report=next((ref for ref in refs if ref.uri.endswith("validation_report.json")), None),
    )
    manifest_path = work_dir / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    manifest_ref = store.put_file(manifest_path, f"{run_id}/run_manifest.json", media_type="application/json")
    if not store.verify(manifest_ref):
        raise RuntimeError("run manifest artifact verification failed")

    memory = ResearchMemoryRecord(
        goal_id=goal.goal_id, run_id=run_id,
        finding=validation.summary,
        evidence_uris=tuple(ref.uri for ref in [*refs, manifest_ref]),
        next_action="prepare_calibrated_material_validation" if passed else "stop_and_repair_failed_gate",
    )
    memory_path = work_dir / "research_memory.json"
    memory_path.write_text(json.dumps(memory.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    result = {
        "goal": goal.to_dict(), "solver_command": command, "solver_returncode": completed.returncode,
        "benchmark_report": {k: v for k, v in benchmark_report.items() if k != "rows"},
        "validation_report": validation.to_dict(), "run_manifest": manifest.to_dict(),
        "manifest_artifact": manifest_ref.to_dict(), "research_memory": memory.to_dict(),
        "work_dir": str(work_dir),
        "scientific_status": "numerical_fixture_only_not_physical_validation",
    }
    (work_dir / "demo_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                               encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/dem3d-bending-research"))
    parser.add_argument("--tensordem-root", type=Path, default=Path("../TensorDEM"))
    parser.add_argument("--run-id", default="research-dem3d-bending-001")
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    result = run_benchmark(args.work_dir, args.tensordem_root, args.run_id, args.steps)
    print(json.dumps({
        "run_id": result["run_manifest"]["run_id"],
        "validation_status": result["run_manifest"]["validation_status"],
        "scientific_status": result["scientific_status"],
        "next_action": result["research_memory"]["next_action"],
        "work_dir": result["work_dir"],
    }, indent=2, sort_keys=True))
    if result["validation_report"]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
