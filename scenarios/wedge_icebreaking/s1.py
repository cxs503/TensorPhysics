"""S1 DEM ice sheet + prescribed wedge hull for the wedge-icebreaking staged integration ladder.

Single-leg DEM scenario: generate a wedge hull polyline CSV inline, run TensorDEM
with --hull-profile, and validate three gates:
(a) output integrity (history.csv, trajectory.pt, accounting.json),
(b) impulse ledger balance (tool/support/drag/external impulses vs Δp),
(c) fracture energy ledger (fracture_release_J ≥ 0, closure with failure_energy_sum_J).

This scenario exercises the integration pipeline and accounting ledger checks.
It is NOT a physics benchmark and must not be promoted as one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tensorphysics import (
    LocalArtifactStore, RunManifest, RunStatus, ValidationStatus,
)
from tensorphysics.research_contracts import (
    ExperimentPlan, ResearchGoal, ResearchMemoryRecord, ValidationReport,
)

# Import helpers from s0.py (same directory)
# Use absolute import for direct script execution
from s0 import (
    _check, _load_json, _numbers_finite, _check_dem_history, _check_dem_trajectory,
    _git_revision, _run_leg,
)

DEFAULT_ECO_ROOT = Path("/nfs/wangxi/tensor_eco")
DEFAULT_VENV_ECO = Path("/nfs/wangxi/venvs/tensor_eco/bin/python")


def _generate_wedge_hull_csv(work_dir: Path) -> Path:
    """Generate a preset wedge hull polyline CSV (≥3 vertices) inline."""
    hull_path = work_dir / "wedge_hull.csv"
    # Simple wedge: 3 vertices forming a wedge shape
    # Vertex 1: front tip (0.0, 0.0)
    # Vertex 2: upper back (-0.3, 0.15)
    # Vertex 3: lower back (-0.3, -0.15)
    vertices = [
        (0.0, 0.0),
        (-0.3, 0.15),
        (-0.3, -0.15),
    ]
    with hull_path.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in vertices:
            f.write(f"{x},{y}\n")
    return hull_path


def _check_accounting_json(path: Path | None) -> dict[str, Any]:
    """Check accounting.json is loadable and all numerics are finite."""
    if path is None:
        return _check("accounting_json_loadable_finite", False, "file not found")
    ok, payload, detail = _load_json(path)
    if not ok:
        return _check("accounting_json_loadable_finite", False, detail)
    finite = _numbers_finite(payload)
    return _check("accounting_json_loadable_finite", finite,
                  f"{detail}; finite={finite}")


def _check_impulse_ledger(accounting_path: Path, trajectory_path: Path) -> dict[str, Any]:
    """Check impulse ledger balance: |Σimpulse - Δp| / max(|Σimpulse|, |Δp|, 1e-12) ≤ 1e-3."""
    ok_acc, accounting, detail_acc = _load_json(accounting_path)
    if not ok_acc:
        return _check("impulse_ledger_balance", False, f"accounting load failed: {detail_acc}")

    try:
        traj = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        velocities = traj["velocities"]  # (T, N, 2)
        if velocities.ndim != 3 or velocities.shape[2] != 2:
            raise ValueError(f"velocities shape {velocities.shape} != (T, N, 2)")
        v_initial = velocities[0]  # (N, 2)
        v_final = velocities[-1]   # (N, 2)
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        return _check("impulse_ledger_balance", False, f"trajectory load failed: {exc}")

    # Read impulses from accounting
    try:
        tool_impulse = accounting["tool_impulse_on_ice_Ns"]
        support_impulse = accounting["support_impulse_Ns"]
        drag_impulse = accounting["drag_impulse_Ns"]
        external_impulse = accounting["external_impulse_Ns"]
        total_mass_kg = accounting["total_mass_kg"]
    except KeyError as exc:
        return _check("impulse_ledger_balance", False, f"missing accounting field: {exc}")

    # Check all are [x, y] lists
    for name, val in [("tool_impulse", tool_impulse), ("support_impulse", support_impulse),
                      ("drag_impulse", drag_impulse), ("external_impulse", external_impulse)]:
        if not isinstance(val, list) or len(val) != 2:
            return _check("impulse_ledger_balance", False, f"{name} must be [x, y] list")
        if not all(math.isfinite(v) for v in val):
            return _check("impulse_ledger_balance", False, f"{name} contains non-finite values")

    if not math.isfinite(total_mass_kg) or total_mass_kg <= 0:
        return _check("impulse_ledger_balance", False, f"invalid total_mass_kg: {total_mass_kg}")

    # Sum impulses
    sum_impulse_x = tool_impulse[0] + support_impulse[0] + drag_impulse[0] + external_impulse[0]
    sum_impulse_y = tool_impulse[1] + support_impulse[1] + drag_impulse[1] + external_impulse[1]

    # Calculate Δp = m * (v_final - v_initial), summed over all particles
    delta_p_x = total_mass_kg * float((v_final[:, 0] - v_initial[:, 0]).sum())
    delta_p_y = total_mass_kg * float((v_final[:, 1] - v_initial[:, 1]).sum())

    # Check balance per component: |Σimpulse - Δp| / max(|Σimpulse|, |Δp|, 1e-12) ≤ 1e-3
    tol = 1e-3
    for comp_name, sum_imp, delta_p in [("x", sum_impulse_x, delta_p_x), ("y", sum_impulse_y, delta_p_y)]:
        error = abs(sum_imp - delta_p)
        scale = max(abs(sum_imp), abs(delta_p), 1e-12)
        rel_error = error / scale
        if rel_error > tol:
            return _check("impulse_ledger_balance", False,
                          f"component {comp_name}: |Σimpulse-Δp|/max={rel_error:.6e} > {tol} "
                          f"(Σimpulse={sum_imp:.6e}, Δp={delta_p:.6e})")

    return _check("impulse_ledger_balance", True,
                  f"impulse ledger balanced: x_err={abs(sum_impulse_x - delta_p_x):.6e}, "
                  f"y_err={abs(sum_impulse_y - delta_p_y):.6e}")


def _check_fracture_energy_ledger(accounting_path: Path) -> dict[str, Any]:
    """Check fracture energy ledger: fracture_release_J ≥ 0, closure with failure_energy_sum_J."""
    ok, accounting, detail = _load_json(accounting_path)
    if not ok:
        return _check("fracture_energy_ledger", False, f"accounting load failed: {detail}")

    try:
        fracture_release_J = accounting["fracture_release_J"]
        failure_energy_sum_J = accounting["failure_energy_sum_J"]
        broken_bonds = accounting["broken_bonds"]
        total_bonds = accounting["total_bonds"]
    except KeyError as exc:
        return _check("fracture_energy_ledger", False, f"missing accounting field: {exc}")

    # Check all are finite
    for name, val in [("fracture_release_J", fracture_release_J),
                      ("failure_energy_sum_J", failure_energy_sum_J),
                      ("broken_bonds", broken_bonds), ("total_bonds", total_bonds)]:
        if not math.isfinite(val):
            return _check("fracture_energy_ledger", False, f"{name} is not finite: {val}")

    # Check fracture_release_J ≥ 0
    if fracture_release_J < 0:
        return _check("fracture_energy_ledger", False,
                      f"fracture_release_J={fracture_release_J} < 0")

    # Check closure: |fracture_release_J - failure_energy_sum_J| ≤ 1e-10
    energy_diff = abs(fracture_release_J - failure_energy_sum_J)
    if energy_diff > 1e-10:
        return _check("fracture_energy_ledger", False,
                      f"|fracture_release_J - failure_energy_sum_J| = {energy_diff:.6e} > 1e-10")

    # Check broken_bonds ≤ total_bonds
    if broken_bonds > total_bonds:
        return _check("fracture_energy_ledger", False,
                      f"broken_bonds={broken_bonds} > total_bonds={total_bonds}")

    return _check("fracture_energy_ledger", True,
                  f"fracture_release_J={fracture_release_J:.6e}, "
                  f"failure_energy_sum_J={failure_energy_sum_J:.6e}, "
                  f"broken_bonds={broken_bonds}/{total_bonds}")


def _leg_dem_wedge(scenario_id: str, eco_root: Path, venv_eco: Path, work_dir: Path) -> dict[str, Any]:
    """Single-leg DEM with wedge hull profile."""
    repo = eco_root / "TensorDEM"
    out_dir = work_dir / "solver-output"

    # Generate wedge hull CSV
    hull_csv = _generate_wedge_hull_csv(work_dir)

    def collect() -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
        history = out_dir / "history.csv"
        trajectory = out_dir / "trajectory.pt"
        accounting = out_dir / "accounting.json"

        checks: list[dict[str, Any]] = []
        artifacts: list[tuple[Path, str]] = []

        # (a) Output integrity gates
        checks.append(_check_dem_history(history))
        if history.exists():
            artifacts.append((history, "text/csv"))

        if history.exists():
            checks.append(_check_dem_trajectory(trajectory))
            if trajectory.exists():
                artifacts.append((trajectory, "application/octet-stream"))
        else:
            checks.append(_check("dem_trajectory_finite_and_time_ordered", False,
                                 "history missing; trajectory validation skipped"))

        checks.append(_check_accounting_json(accounting))
        if accounting.exists():
            artifacts.append((accounting, "application/json"))

        # (b) Impulse ledger balance gate
        if accounting.exists() and trajectory.exists():
            checks.append(_check_impulse_ledger(accounting, trajectory))
        else:
            checks.append(_check("impulse_ledger_balance", False,
                                 "accounting.json or trajectory.pt missing; impulse check skipped"))

        # (c) Fracture energy ledger gate
        if accounting.exists():
            checks.append(_check_fracture_energy_ledger(accounting))
        else:
            checks.append(_check("fracture_energy_ledger", False,
                                 "accounting.json missing; fracture energy check skipped"))

        return checks, artifacts

    return {
        "name": "dem_wedge",
        "run_id": f"{scenario_id}-dem-wedge",
        "repo_dir": repo, "repo_key": "cxs503/TensorDEM",
        "command": [str(venv_eco), "-m", "tensordem",
                    "--hull-profile", str(hull_csv),
                    "--hull-start-x", "0.0",
                    "--hull-velocity-x", "0.2",
                    "--nx", "21", "--ny", "5", "--steps", "2000", "--save-every", "20",
                    "--device", "cpu", "--output", str(out_dir)],
        "cwd": repo,
        "extra_env": None,
        "allowed_codes": (0,),
        "collect": collect,
        "plan_parameters": {
            "solver": "TensorDEM", "case": "wedge hull 21x5 ice sheet",
            "hull_profile": str(hull_csv), "hull_start_x": 0.0, "hull_velocity_x": 0.2,
            "nx": 21, "ny": 5, "steps": 2000, "save_every": 20,
            "device": "cpu", "venv": str(venv_eco),
        },
        "validation_check_names": ("dem_history_finite_and_monotonic",
                                   "dem_trajectory_finite_and_time_ordered",
                                   "accounting_json_loadable_finite",
                                   "impulse_ledger_balance",
                                   "fracture_energy_ledger"),
    }


def run_scenario(
    work_dir: Path, scenario_id: str, eco_root: Path, venv_eco: Path,
) -> dict[str, Any]:
    """Run the S1 scenario: single-leg DEM with wedge hull."""
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    physics_dir = eco_root / "TensorPhysics"
    goal = ResearchGoal(
        goal_id=f"{scenario_id}-goal",
        objective=(
            "S1 DEM ice sheet + prescribed wedge hull: validate impulse ledger balance "
            "and fracture energy ledger closure under fail-closed controller."
        ),
        acceptance_criteria=(
            "solver process exits within allowed return codes",
            "solver outputs load, are finite, and are time ordered where applicable",
            "impulse ledger balance: |Σimpulse - Δp| / max(|Σimpulse|, |Δp|, 1e-12) ≤ 1e-3",
            "fracture energy ledger: fracture_release_J ≥ 0, closure with failure_energy_sum_J",
            "every leg manifest and artifact passes checksum verification",
            "the controller stops at the first failed gate",
        ),
        constraints={"stage": "S1", "coupling": "none", "device": "cpu",
                     "eco_root": str(eco_root)},
        budget={"max_legs": 1},
    )
    (work_dir / "research_goal.json").write_text(
        json.dumps(goal.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    spec = _leg_dem_wedge(scenario_id, eco_root, venv_eco, work_dir / "dem_wedge")
    result = _run_leg(
        name=spec["name"], work_dir=work_dir / spec["name"], run_id=spec["run_id"],
        parent_run_id=scenario_id, goal_id=goal.goal_id, command=spec["command"],
        cwd=spec["cwd"], extra_env=spec["extra_env"], repo_dir=spec["repo_dir"],
        repo_key=spec["repo_key"], physics_dir=physics_dir,
        allowed_codes=spec["allowed_codes"], collect_outputs=spec["collect"],
        plan_parameters=spec["plan_parameters"],
        validation_check_names=spec["validation_check_names"],
    )
    legs = [{
        "leg": spec["name"], "run_id": spec["run_id"],
        "validation_status": result["validation_report"]["status"],
        "checks": result["validation_report"]["checks"],
        "solver_returncode": result["solver_returncode"],
        "manifest_artifact": result["manifest_artifact"],
        "next_action": result["research_memory"]["next_action"],
    }]

    all_passed = len(legs) == 1 and legs[0]["validation_status"] == "passed"
    report = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "stage": "S1",
        "goal": goal.to_dict(),
        "design": "single-leg DEM with prescribed wedge hull; impulse and fracture energy ledger checks",
        "legs": legs,
        "decision": ("s1_impulse_and_energy_ledgers_passed" if all_passed
                     else "stop_s1_and_repair_failed_gate"),
        "scientific_status": "implemented_not_benchmark_verified",
        "next_stage": ("S2: +FEM structure one-way coupling" if all_passed
                       else "repair the failed gate before retry"),
    }
    (work_dir / "s1_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/wedge-icebreaking-s1"))
    parser.add_argument("--scenario-id", default="wedge-icebreaking-s1-001")
    parser.add_argument("--eco-root", type=Path, default=DEFAULT_ECO_ROOT,
                        help="six-repo workspace root holding the Tensor* checkouts")
    parser.add_argument("--venv-eco", type=Path, default=DEFAULT_VENV_ECO,
                        help="interpreter with TensorPhysics/TensorDEM installed")
    args = parser.parse_args()
    report = run_scenario(args.work_dir, args.scenario_id, args.eco_root, args.venv_eco)
    print(json.dumps({
        "scenario_id": report["scenario_id"],
        "leg_count": len(report["legs"]),
        "decision": report["decision"],
        "report": str((args.work_dir.resolve()) / "s1_report.json"),
    }, indent=2, sort_keys=True))
    if report["decision"] != "s1_impulse_and_energy_ledgers_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
