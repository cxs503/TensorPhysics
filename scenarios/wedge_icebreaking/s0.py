"""S0 joint-debug smoke for the wedge-icebreaking staged integration ladder.

Runs four independently verified solver entrypoints (TensorLBM, TensorFVM,
TensorFEM, TensorDEM) under one fail-closed controller with independent
output-integrity gates and full RunManifest / ArtifactRef lineage.

This scenario exercises the integration pipeline only: process invocation,
artifact checksums, manifests, and fail-closed gating. It is NOT a physics
benchmark and must not be promoted as one (see scenarios/wedge_icebreaking/README.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from tensorphysics import (
    LocalArtifactStore, RunManifest, RunStatus, ValidationStatus,
)
from tensorphysics.research_contracts import (
    ExperimentPlan, ResearchGoal, ResearchMemoryRecord, ValidationReport,
)

DEFAULT_ECO_ROOT = Path("/nfs/wangxi/tensor_eco")
DEFAULT_VENV_ECO = Path("/nfs/wangxi/venvs/tensor_eco/bin/python")
DEFAULT_VENV_LBM = Path("/nfs/wangxi/venvs/tensorlbm/bin/python")


def _git_revision(repo_dir: Path) -> str:
    """Live revision of a checkout; 'unknown' keeps manifests constructible offline."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _load_json(path: Path | None) -> tuple[bool, Any, str]:
    if path is None:
        return False, None, "file not found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, None, f"{path.name}: {exc}"
    return True, payload, f"loaded {path}"


def _numbers_finite(payload: Any) -> bool:
    """Every numeric leaf (lists, dicts, scalars) is finite; non-numeric data is ignored."""
    if isinstance(payload, bool):
        return True
    if isinstance(payload, (int, float)):
        return math.isfinite(payload)
    if isinstance(payload, dict):
        return all(_numbers_finite(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(_numbers_finite(value) for value in payload)
    return True


def _first_list_length(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for value in payload.values():
            length = _first_list_length(value)
            if length is not None:
                return length
    return None


def _check_numeric_csv(path: Path | None, name: str) -> dict[str, Any]:
    """CSV has at least one row with numeric data and every numeric cell is finite."""
    if path is None:
        return _check(name, False, "file not found")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        return _check(name, False, str(exc))
    if not rows:
        return _check(name, False, "no data rows")
    numeric_rows = 0
    finite = True
    for row in rows:
        values = []
        for cell in row.values():
            if cell in (None, ""):
                continue
            try:
                values.append(float(cell))
            except (TypeError, ValueError):
                continue
        if values:
            numeric_rows += 1
        if not all(math.isfinite(value) for value in values):
            finite = False
    return _check(name, numeric_rows > 0 and finite,
                  f"{path.name}: {len(rows)} rows; numeric_rows={numeric_rows}; finite={finite}")


def _check_dem_history(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return _check("dem_history_nonempty", False, "no data rows")
    try:
        values = [[float(value) for value in row.values()] for row in rows]
    except (TypeError, ValueError) as exc:
        return _check("dem_history_numeric", False, str(exc))
    finite = all(math.isfinite(value) for row in values for value in row)
    times = [row.get("time") for row in rows]
    monotonic = True
    if times and all(value not in (None, "") for value in times):
        try:
            t = [float(value) for value in times]
            monotonic = all(b >= a for a, b in zip(t, t[1:]))
        except ValueError:
            monotonic = False
    return _check("dem_history_finite_and_monotonic", finite and monotonic,
                  f"{len(rows)} rows; finite={finite}; monotonic_time={monotonic}")


def _check_dem_trajectory(path: Path) -> dict[str, Any]:
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
        return _check("dem_trajectory_finite_and_time_ordered", finite and monotonic,
                      f"positions_shape={list(positions.shape)}; finite={finite}; monotonic_time={monotonic}")
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        return _check("dem_trajectory_readable_and_finite", False, str(exc))


def _rglob_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _run_leg(
    *,
    name: str,
    work_dir: Path,
    run_id: str,
    parent_run_id: str,
    goal_id: str,
    command: list[str],
    cwd: Path,
    extra_env: dict[str, str] | None,
    repo_dir: Path,
    repo_key: str,
    physics_dir: Path,
    allowed_codes: tuple[int, ...],
    collect_outputs: Callable[[], tuple[list[dict[str, Any]], list[tuple[Path, str]]]],
    plan_parameters: dict[str, Any],
    validation_check_names: tuple[str, ...],
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    plan = ExperimentPlan(
        plan_id=f"{run_id}-plan", goal_id=goal_id, solver=repo_key,
        parameters=plan_parameters, validation_checks=validation_check_names, max_runs=1,
    )
    env = {**os.environ, **(extra_env or {})}
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, cwd=str(cwd), env=env,
    )
    exit_ok = completed.returncode in allowed_codes
    checks: list[dict[str, Any]] = [
        _check(f"{name}_exit_ok", exit_ok,
               f"returncode={completed.returncode}; allowed={list(allowed_codes)}; "
               f"stdout_tail={completed.stdout[-800:]}; stderr_tail={completed.stderr[-800:]}")
    ]
    artifacts: list[tuple[Path, str]] = []
    if exit_ok:
        leg_checks, artifacts = collect_outputs()
        checks.extend(leg_checks)
    else:
        checks.append(_check("solver_outputs_available", False,
                             "solver failed; output validation skipped"))
    passed = all(item["passed"] for item in checks)
    report = ValidationReport(
        run_id=run_id, status="passed" if passed else "failed", checks=tuple(checks),
        summary=("All deterministic output-integrity checks passed." if passed
                 else "One or more deterministic output-integrity checks failed; "
                      "do not promote this run as validated."),
    )
    report_path = work_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    store = LocalArtifactStore(work_dir / "artifact-store")
    output_refs = []
    for path, media_type in artifacts:
        ref = store.put_file(path, f"{run_id}/{path.name}", media_type=media_type)
        if not store.verify(ref):
            raise RuntimeError(f"artifact store verification failed: {path.name}")
        output_refs.append(ref)
    report_ref = store.put_file(report_path, f"{run_id}/validation_report.json",
                                media_type="application/json")
    if not store.verify(report_ref):
        raise RuntimeError("validation report artifact verification failed")

    config = plan.to_dict()
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.SUCCEEDED if exit_ok else RunStatus.FAILED,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_sha256=RunManifest.digest_config(config),
        code_revisions={"cxs503/TensorPhysics": _git_revision(physics_dir),
                        repo_key: _git_revision(repo_dir)},
        environment={"python": sys.version.split()[0], "platform": sys.platform},
        resources={"execution": "local", "device": "cpu"},
        outputs=output_refs,
        validation_status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
        validation_report=report_ref,
        parent_run_ids=[parent_run_id],
    )
    manifest_path = work_dir / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    manifest_ref = store.put_file(manifest_path, f"{run_id}/run_manifest.json",
                                  media_type="application/json")
    if not store.verify(manifest_ref):
        raise RuntimeError("run manifest artifact verification failed")

    memory = ResearchMemoryRecord(
        goal_id=goal_id, run_id=run_id, finding=report.summary,
        evidence_uris=tuple(ref.uri for ref in [*output_refs, report_ref, manifest_ref]),
        next_action="continue_to_next_leg" if passed else "stop_and_repair_before_retry",
    )
    (work_dir / "research_memory.json").write_text(
        json.dumps(memory.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "leg": name, "solver_command": command, "solver_returncode": completed.returncode,
        "validation_report": report.to_dict(), "run_manifest": manifest.to_dict(),
        "manifest_artifact": manifest_ref.to_dict(), "research_memory": memory.to_dict(),
        "work_dir": str(work_dir),
    }
    (work_dir / "leg_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _leg_lbm(scenario_id: str, eco_root: Path, venv_lbm: Path, work_dir: Path) -> dict[str, Any]:
    repo = eco_root / "TensorLBM"
    out_dir = work_dir / "solver-output"

    def collect() -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
        checks: list[dict[str, Any]] = []
        artifacts: list[tuple[Path, str]] = []
        metadata = _rglob_one(out_dir, "run_metadata.json")
        ok, payload, detail = _load_json(metadata)
        checks.append(_check("lbm_metadata_valid_json", ok, detail))
        if metadata is not None and ok:
            artifacts.append((metadata, "application/json"))
        forces = _rglob_one(out_dir, "forces.csv")
        if forces is not None:
            checks.append(_check_numeric_csv(forces, "lbm_forces_nonempty_finite"))
            artifacts.append((forces, "text/csv"))
        else:
            checks.append(_check("lbm_forces_nonempty_finite", False, "file not found"))
        return checks, artifacts

    return {
        "name": "lbm",
        "run_id": f"{scenario_id}-lbm",
        "repo_dir": repo, "repo_key": "cxs503/TensorLBM",
        "command": [str(venv_lbm), "examples/cylinder_flow.py",
                    "--nx", "64", "--ny", "24", "--radius", "4",
                    "--n-steps", "20", "--output-interval", "10",
                    "--run-name", "smoke", "--overwrite", "--device", "cpu",
                    "--output-root", str(out_dir)],
        "cwd": repo,
        "extra_env": {"PYTHONPATH": str(repo / "src")},
        "allowed_codes": (0,),
        "collect": collect,
        "plan_parameters": {
            "solver": "TensorLBM", "case": "examples/cylinder_flow.py smoke variant",
            "nx": 64, "ny": 24, "radius": 4, "n_steps": 20, "output_interval": 10,
            "run_name": "smoke", "device": "cpu", "venv": str(venv_lbm),
        },
        "validation_check_names": ("lbm_metadata_valid_json", "lbm_forces_nonempty_finite"),
    }


def _leg_fvm(scenario_id: str, eco_root: Path, venv_eco: Path, work_dir: Path) -> dict[str, Any]:
    repo = eco_root / "TensorFVM"
    out_dir = work_dir / "solver-output"

    def collect() -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
        checks: list[dict[str, Any]] = []
        artifacts: list[tuple[Path, str]] = []
        summary = _rglob_one(out_dir, "summary.json")
        ok, payload, detail = _load_json(summary)
        converged = bool(payload.get("converged")) if isinstance(payload, dict) else False
        checks.append(_check("fvm_summary_valid_json_finite", ok and _numbers_finite(payload),
                             f"{detail}; finite={_numbers_finite(payload) if ok else False}; "
                             f"converged={converged}"))
        if summary is not None and ok:
            artifacts.append((summary, "application/json"))
        history = _rglob_one(out_dir, "history.json")
        ok_h, payload_h, detail_h = _load_json(history)
        checks.append(_check("fvm_history_valid_json_finite",
                             ok_h and _numbers_finite(payload_h)
                             and (_first_list_length(payload_h) or 0) > 0,
                             f"{detail_h}; entries={_first_list_length(payload_h) if ok_h else None}"))
        if history is not None and ok_h:
            artifacts.append((history, "application/json"))
        fields = _rglob_one(out_dir, "fields.csv")
        checks.append(_check_numeric_csv(fields, "fvm_fields_nonempty_finite"))
        if fields is not None:
            artifacts.append((fields, "text/csv"))
        return checks, artifacts

    return {
        "name": "fvm",
        "run_id": f"{scenario_id}-fvm",
        "repo_dir": repo, "repo_key": "cxs503/TensorFVM",
        "command": [str(venv_eco), "-m", "tensorfvm",
                    "--mesh-type", "body-fitted", "--nx", "80", "--ny", "40",
                    "--reynolds", "20", "--max-iterations", "1000",
                    "--device", "cpu", "--output", str(out_dir)],
        "cwd": repo,
        "extra_env": None,
        "allowed_codes": (0, 2),
        "collect": collect,
        "plan_parameters": {
            "solver": "TensorFVM", "case": "body-fitted cylinder Re=20 (python -m tensorfvm)",
            "mesh_type": "body-fitted", "nx": 80, "ny": 40, "reynolds": 20,
            "max_iterations": 1000, "device": "cpu", "venv": str(venv_eco),
            "allowed_returncodes": [0, 2],
        },
        "validation_check_names": ("fvm_summary_valid_json_finite", "fvm_history_valid_json_finite",
                                   "fvm_fields_nonempty_finite"),
    }


def _leg_fem(scenario_id: str, eco_root: Path, venv_eco: Path, work_dir: Path) -> dict[str, Any]:
    repo = eco_root / "TensorFEM"
    out_dir = work_dir / "solver-output"
    evidence = out_dir / "benchmark-evidence.json"

    def collect() -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
        ok, payload, detail = _load_json(evidence if evidence.exists() else None)
        registry_passed = isinstance(payload, dict) and payload.get("passed") is True
        entries = _first_list_length(payload) if ok else None
        checks = [_check("fem_registry_all_passed", ok and registry_passed,
                         f"{detail}; registry_passed="
                         f"{payload.get('passed') if isinstance(payload, dict) else None}; "
                         f"entries={entries}")]
        artifacts: list[tuple[Path, str]] = []
        if evidence.exists():
            artifacts.append((evidence, "application/json"))
        return checks, artifacts

    return {
        "name": "fem",
        "run_id": f"{scenario_id}-fem",
        "repo_dir": repo, "repo_key": "cxs503/TensorFEM",
        "command": [str(venv_eco.parent / "tensorfem"), "verify", "--output", str(evidence)],
        "cwd": repo,
        "extra_env": None,
        "allowed_codes": (0,),
        "collect": collect,
        "plan_parameters": {
            "solver": "TensorFEM", "case": "tensorfem verify (analytic benchmark registry)",
            "command": "tensorfem verify", "venv": str(venv_eco),
        },
        "validation_check_names": ("fem_registry_all_passed",),
    }


def _leg_dem(scenario_id: str, eco_root: Path, venv_eco: Path, work_dir: Path) -> dict[str, Any]:
    repo = eco_root / "TensorDEM"
    out_dir = work_dir / "solver-output"

    def collect() -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
        history = out_dir / "history.csv"
        trajectory = out_dir / "trajectory.pt"
        checks = [_check_dem_history(history)]
        if history.exists():
            checks.append(_check_dem_trajectory(trajectory))
        else:
            checks.append(_check("dem_trajectory_finite_and_time_ordered", False,
                                 "history missing; trajectory validation skipped"))
        artifacts: list[tuple[Path, str]] = []
        if history.exists():
            artifacts.append((history, "text/csv"))
        if trajectory.exists():
            artifacts.append((trajectory, "application/octet-stream"))
        return checks, artifacts

    return {
        "name": "dem",
        "run_id": f"{scenario_id}-dem",
        "repo_dir": repo, "repo_key": "cxs503/TensorDEM",
        "command": [str(venv_eco), "-m", "tensordem",
                    "--nx", "21", "--ny", "5", "--steps", "2000", "--save-every", "20",
                    "--device", "cpu", "--output", str(out_dir)],
        "cwd": repo,
        "extra_env": None,
        "allowed_codes": (0,),
        "collect": collect,
        "plan_parameters": {
            "solver": "TensorDEM", "case": "default 2D floe field 21x5",
            "nx": 21, "ny": 5, "steps": 2000, "save_every": 20,
            "device": "cpu", "venv": str(venv_eco),
        },
        "validation_check_names": ("dem_history_finite_and_monotonic",
                                   "dem_trajectory_finite_and_time_ordered"),
    }


def run_scenario(
    work_dir: Path, scenario_id: str, eco_root: Path,
    venv_eco: Path, venv_lbm: Path,
) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    physics_dir = eco_root / "TensorPhysics"
    goal = ResearchGoal(
        goal_id=f"{scenario_id}-goal",
        objective=(
            "S0 joint-debug pipeline smoke: invoke four independently verified solver "
            "entrypoints under one fail-closed controller with independent "
            "output-integrity gates and full RunManifest/ArtifactRef lineage."
        ),
        acceptance_criteria=(
            "each solver process exits within its allowed return codes",
            "solver outputs load, are finite, and are time ordered where applicable",
            "every leg manifest and artifact passes checksum verification",
            "the controller stops at the first failed gate",
        ),
        constraints={"stage": "S0", "coupling": "none", "device": "cpu",
                     "eco_root": str(eco_root)},
        budget={"max_legs": 4},
    )
    (work_dir / "research_goal.json").write_text(
        json.dumps(goal.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    builders = (
        lambda: _leg_lbm(scenario_id, eco_root, venv_lbm, work_dir / "lbm"),
        lambda: _leg_fvm(scenario_id, eco_root, venv_eco, work_dir / "fvm"),
        lambda: _leg_fem(scenario_id, eco_root, venv_eco, work_dir / "fem"),
        lambda: _leg_dem(scenario_id, eco_root, venv_eco, work_dir / "dem"),
    )
    legs: list[dict[str, Any]] = []
    for build in builders:
        spec = build()
        result = _run_leg(
            name=spec["name"], work_dir=work_dir / spec["name"], run_id=spec["run_id"],
            parent_run_id=scenario_id, goal_id=goal.goal_id, command=spec["command"],
            cwd=spec["cwd"], extra_env=spec["extra_env"], repo_dir=spec["repo_dir"],
            repo_key=spec["repo_key"], physics_dir=physics_dir,
            allowed_codes=spec["allowed_codes"], collect_outputs=spec["collect"],
            plan_parameters=spec["plan_parameters"],
            validation_check_names=spec["validation_check_names"],
        )
        legs.append({
            "leg": spec["name"], "run_id": spec["run_id"],
            "validation_status": result["validation_report"]["status"],
            "checks": result["validation_report"]["checks"],
            "solver_returncode": result["solver_returncode"],
            "manifest_artifact": result["manifest_artifact"],
            "next_action": result["research_memory"]["next_action"],
        })
        # Fail closed: do not spend the remaining budget after a failed gate.
        if result["validation_report"]["status"] != "passed":
            break

    all_passed = len(legs) == 4 and all(
        item["validation_status"] == "passed" for item in legs
    )
    report = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "stage": "S0",
        "goal": goal.to_dict(),
        "design": "four-leg sequential solver smoke; no physics coupling",
        "legs": legs,
        "decision": ("s0_smoke_complete_integrity_gates_passed" if all_passed
                     else "stop_s0_and_repair_failed_gate"),
        "scientific_status": "not_a_physics_benchmark",
        "next_stage": ("S1: DEM ice sheet + prescribed wedge hull (one-way)"
                       if all_passed else "repair the failed leg before retry"),
    }
    (work_dir / "s0_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/wedge-icebreaking-s0"))
    parser.add_argument("--scenario-id", default="wedge-icebreaking-s0-001")
    parser.add_argument("--eco-root", type=Path, default=DEFAULT_ECO_ROOT,
                        help="six-repo workspace root holding the Tensor* checkouts")
    parser.add_argument("--venv-eco", type=Path, default=DEFAULT_VENV_ECO,
                        help="interpreter with TensorPhysics/TensorFVM/TensorFEM/TensorDEM installed")
    parser.add_argument("--venv-lbm", type=Path, default=DEFAULT_VENV_LBM,
                        help="interpreter used for the TensorLBM leg")
    args = parser.parse_args()
    report = run_scenario(args.work_dir, args.scenario_id, args.eco_root,
                          args.venv_eco, args.venv_lbm)
    print(json.dumps({
        "scenario_id": report["scenario_id"],
        "leg_count": len(report["legs"]),
        "decision": report["decision"],
        "report": str((args.work_dir.resolve()) / "s0_report.json"),
    }, indent=2, sort_keys=True))
    if report["decision"] != "s0_smoke_complete_integrity_gates_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
