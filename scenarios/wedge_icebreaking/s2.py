"""S2 DEM→FEM 单向耦合：S1 冰-船反力历史 → TensorFEM 梁准静态。

三腿顺序：leg_loads（提取载荷）→ leg_fem（准静态求解）→ leg_gates（账本裁判）。
验证力、力矩、功三共轭账本闭合（管道级耦合正确性，非结构 Benchmark）。

成熟度声明：Implemented 级证据，不得凭 S2 晋级 Benchmark-verified 或更高。
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
from typing import Any

import torch

from tensorphysics import (
    LocalArtifactStore, RunManifest, RunStatus, ValidationStatus,
)
from tensorphysics.research_contracts import (
    ExperimentPlan, ResearchGoal, ResearchMemoryRecord, ValidationReport,
)

# Import helpers from s0.py
from s0 import _check, _load_json, _numbers_finite, _git_revision

DEFAULT_ECO_ROOT = Path("/nfs/wangxi/tensor_eco")
DEFAULT_VENV_ECO = Path("/nfs/wangxi/venvs/tensor_eco/bin/python")

# S1 history path (read-only)
S1_HISTORY_PATH = Path("/nfs/wangxi/runs/wedge_s1_20261018/outputs/wedge-icebreaking-s1-repair/dem_wedge/solver-output/history.csv")


def _leg_loads(history_path: Path, work_dir: Path) -> dict[str, Any]:
    """Leg 1: 从 S1 history.csv 提取载荷历史到 loads.json。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    loads_path = work_dir / "loads.json"

    # Read history.csv
    try:
        with history_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError as exc:
        return {
            "passed": False,
            "loads_path": None,
            "frames": 0,
            "detail": f"history.csv read failed: {exc}",
        }

    if not rows:
        return {
            "passed": False,
            "loads_path": None,
            "frames": 0,
            "detail": "history.csv is empty",
        }

    # Extract time, reaction_x, reaction_y
    times = []
    reaction_x = []
    reaction_y = []

    for i, row in enumerate(rows):
        try:
            t = float(row["time"])
            rx = float(row["reaction_x"])
            ry = float(row["reaction_y"])
        except (KeyError, ValueError, TypeError) as exc:
            return {
                "passed": False,
                "loads_path": None,
                "frames": i,
                "detail": f"row {i} parse failed: {exc}",
            }

        if not (math.isfinite(t) and math.isfinite(rx) and math.isfinite(ry)):
            return {
                "passed": False,
                "loads_path": None,
                "frames": i,
                "detail": f"row {i} non-finite: t={t}, rx={rx}, ry={ry}",
            }

        times.append(t)
        reaction_x.append(rx)
        reaction_y.append(ry)

    # Check time monotonic
    monotonic = all(times[i] <= times[i+1] for i in range(len(times)-1))
    if not monotonic:
        return {
            "passed": False,
            "loads_path": None,
            "frames": len(times),
            "detail": "time is not monotonic",
        }

    # Write loads.json
    loads = {
        "frames": len(times),
        "time": times,
        "reaction_x": reaction_x,
        "reaction_y": reaction_y,
    }
    loads_path.write_text(json.dumps(loads, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    return {
        "passed": True,
        "loads_path": loads_path,
        "frames": len(times),
        "detail": f"extracted {len(times)} frames from history.csv",
    }


def _build_cantilever_model(L: float = 1.0, E: float = 2.1e11, A: float = 1e-4, I: float = 1e-8):
    """Build a simple cantilever beam model: 2 nodes, 1 element.
    Node 0 at (0, 0) fixed, node 1 at (L, 0) free.
    """
    nodes = torch.tensor([[0.0, 0.0], [L, 0.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1]], dtype=torch.int64)
    young_modulus = torch.tensor(E, dtype=torch.float64)
    area = torch.tensor(A, dtype=torch.float64)
    inertia = torch.tensor(I, dtype=torch.float64)
    # Fixed DOFs at node 0: u_x=0, u_y=0, theta_z=0
    fixed_dofs = torch.tensor([0, 1, 2], dtype=torch.int64)
    return nodes, elements, young_modulus, area, inertia, fixed_dofs


def _solve_frame_static(nodes, elements, young_modulus, area, inertia, forces, fixed_dofs):
    """Solve a 2D frame static problem. Returns displacement and reaction."""
    # Add TensorFEM src to path
    tensorfem_src = Path("/nfs/wangxi/tensor_eco/TensorFEM/src")
    if str(tensorfem_src) not in sys.path:
        sys.path.insert(0, str(tensorfem_src))
    
    from tensorfem.frame2d import FrameModel, solve_frame_static

    model = FrameModel(
        nodes=nodes,
        elements=elements,
        young_modulus=young_modulus,
        area=area,
        inertia=inertia,
        forces=forces,
        fixed_dofs=fixed_dofs,
    )
    result = solve_frame_static(model)
    return result.displacement, result.reaction, result.strain_energy


def _leg_fem(loads_path: Path, work_dir: Path) -> dict[str, Any]:
    """Leg 2: 准静态求解，逐步施加载荷，输出 fem_out.json。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    fem_out_path = work_dir / "fem_out.json"

    # Load loads.json
    ok, loads, detail = _load_json(loads_path)
    if not ok:
        return {
            "passed": False,
            "fem_out_path": None,
            "steps": 0,
            "detail": f"loads.json load failed: {detail}",
        }

    frames = loads["frames"]
    times = loads["time"]
    reaction_x = loads["reaction_x"]
    reaction_y = loads["reaction_y"]

    # Build cantilever model
    L = 1.0  # beam length [m]
    E = 2.1e11  # Young's modulus [Pa]
    A = 1e-4  # cross-section area [m^2]
    I = 1e-8  # second moment of area [m^4]
    nodes, elements, young_modulus, area, inertia, fixed_dofs = _build_cantilever_model(L, E, A, I)

    # Reference point for moment computation: node 0 (fixed end) at (0, 0)
    # External force applied at node 1 (L, 0)
    # Moment about node 0: M_ext = L * F_y (since force acts at (L, 0))

    fem_steps = []
    prev_u = (0.0, 0.0)  # Start from zero displacement
    prev_F = (0.0, 0.0)  # Start from zero force
    W_ext_cumulative = 0.0

    for step_idx in range(frames):
        F_x = reaction_x[step_idx]
        F_y = reaction_y[step_idx]

        # External force vector: 6 DOFs (3 per node)
        # Node 0: [0, 1, 2] fixed
        # Node 1: [3, 4, 5] free, apply F_x at DOF 3, F_y at DOF 4
        forces = torch.zeros(6, dtype=torch.float64)
        forces[3] = F_x
        forces[4] = F_y

        try:
            u, reaction, U_strain = _solve_frame_static(
                nodes, elements, young_modulus, area, inertia, forces, fixed_dofs
            )
        except Exception as exc:
            return {
                "passed": False,
                "fem_out_path": None,
                "steps": step_idx,
                "detail": f"step {step_idx} solve failed: {exc}",
            }

        # Extract quantities
        F_ext_x = F_x
        F_ext_y = F_y
        M_ext = L * F_y  # moment of external force about node 0

        # Reaction at fixed end (node 0): DOFs 0, 1, 2
        R_x = float(reaction[0])
        R_y = float(reaction[1])
        M_R = float(reaction[2])  # reaction moment at node 0

        # Displacement at reference node (node 1): DOFs 3, 4
        u_ref_x = float(u[3])
        u_ref_y = float(u[4])

        U_strain_J = float(U_strain)

        # Compute external work using trapezoidal integration
        # Trapezoidal: 0.5 * (F_{i-1} + F_i) · (u_i - u_{i-1})
        dF_x = 0.5 * (prev_F[0] + F_x)
        dF_y = 0.5 * (prev_F[1] + F_y)
        du_x = u_ref_x - prev_u[0]
        du_y = u_ref_y - prev_u[1]
        dW = dF_x * du_x + dF_y * du_y
        W_ext_cumulative += dW
        W_ext_J = W_ext_cumulative

        prev_u = (u_ref_x, u_ref_y)
        prev_F = (F_x, F_y)

        fem_steps.append({
            "step": step_idx,
            "F_ext_x": F_ext_x,
            "F_ext_y": F_ext_y,
            "M_ext": M_ext,
            "R_x": R_x,
            "R_y": R_y,
            "M_R": M_R,
            "u_ref_x": u_ref_x,
            "u_ref_y": u_ref_y,
            "U_strain_J": U_strain_J,
            "W_ext_J": W_ext_J,
        })

    # Write fem_out.json
    fem_out = {
        "sampling_factor_k": 1,  # no subsampling
        "steps": fem_steps,
    }
    fem_out_path.write_text(json.dumps(fem_out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    return {
        "passed": True,
        "fem_out_path": fem_out_path,
        "steps": len(fem_steps),
        "detail": f"solved {len(fem_steps)} load steps",
    }


def _leg_gates(loads_path: Path, fem_out_path: Path) -> dict[str, Any]:
    """Leg 3: 检查力、力矩、功三共轭账本。"""
    # Load loads.json
    ok_loads, loads, detail_loads = _load_json(loads_path)
    if not ok_loads:
        return {
            "passed": False,
            "checks": [_check("loads_loadable", False, detail_loads)],
            "detail": "loads.json load failed",
        }

    # Load fem_out.json
    ok_fem, fem_out, detail_fem = _load_json(fem_out_path)
    if not ok_fem:
        return {
            "passed": False,
            "checks": [_check("fem_out_loadable", False, detail_fem)],
            "detail": "fem_out.json load failed",
        }

    checks = []

    # Check frames match
    frames = loads["frames"]
    steps = fem_out["steps"]
    if len(steps) != frames:
        checks.append(_check("frames_match", False, f"frames={frames} != steps={len(steps)}"))
        return {"passed": False, "checks": checks, "detail": "frames mismatch"}
    checks.append(_check("frames_match", True, f"frames={frames}"))

    # Check all values finite
    all_finite = True
    for step in steps:
        for key in ["F_ext_x", "F_ext_y", "M_ext", "R_x", "R_y", "M_R", "u_ref_x", "u_ref_y", "U_strain_J", "W_ext_J"]:
            if not math.isfinite(step[key]):
                all_finite = False
                break
    checks.append(_check("all_finite", all_finite, "all FEM output values finite"))

    # Gate 3.3: Force balance |F_ext + R| / max(|F_ext|, |R|, 1e-12) ≤ 1e-6
    force_balance_pass = True
    max_force_error = 0.0
    for step in steps:
        F_ext_x = step["F_ext_x"]
        F_ext_y = step["F_ext_y"]
        R_x = step["R_x"]
        R_y = step["R_y"]

        for comp_name, F_ext, R in [("x", F_ext_x, R_x), ("y", F_ext_y, R_y)]:
            error = abs(F_ext + R)
            scale = max(abs(F_ext), abs(R), 1e-12)
            rel_error = error / scale
            max_force_error = max(max_force_error, rel_error)
            if rel_error > 1e-6:
                force_balance_pass = False

    checks.append(_check("force_balance", force_balance_pass,
                         f"max |F_ext+R|/max(|F_ext|,|R|,1e-12) = {max_force_error:.6e}"))

    # Gate 3.4: Moment balance |M_ext + M_R| / max(|M_ext|, |M_R|, 1e-12) ≤ 1e-6
    moment_balance_pass = True
    max_moment_error = 0.0
    for step in steps:
        M_ext = step["M_ext"]
        M_R = step["M_R"]

        error = abs(M_ext + M_R)
        scale = max(abs(M_ext), abs(M_R), 1e-12)
        rel_error = error / scale
        max_moment_error = max(max_moment_error, rel_error)
        if rel_error > 1e-6:
            moment_balance_pass = False

    checks.append(_check("moment_balance", moment_balance_pass,
                         f"max |M_ext+M_R|/max(|M_ext|,|M_R|,1e-12) = {max_moment_error:.6e}"))

    # Gate 3.5: Work conjugacy |W_ext - U_strain| / max(|W_ext|, |U_strain|, 1e-12) ≤ 1e-5
    work_conjugacy_pass = True
    max_work_error = 0.0
    for step in steps:
        W_ext = step["W_ext_J"]
        U_strain = step["U_strain_J"]

        error = abs(W_ext - U_strain)
        scale = max(abs(W_ext), abs(U_strain), 1e-12)
        rel_error = error / scale
        max_work_error = max(max_work_error, rel_error)
        if rel_error > 1e-5:
            work_conjugacy_pass = False

    checks.append(_check("work_conjugacy", work_conjugacy_pass,
                         f"max |W_ext-U_strain|/max(|W_ext|,|U_strain|,1e-12) = {max_work_error:.6e}"))

    all_passed = all(c["passed"] for c in checks)
    return {
        "passed": all_passed,
        "checks": checks,
        "detail": f"force_error={max_force_error:.6e}, moment_error={max_moment_error:.6e}, work_error={max_work_error:.6e}",
    }


def run_scenario(
    work_dir: Path, scenario_id: str, eco_root: Path, venv_eco: Path,
) -> dict[str, Any]:
    """Run the S2 scenario: DEM→FEM one-way coupling with conjugacy gates."""
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Record git revisions
    tensorfem_rev = _git_revision(eco_root / "TensorFEM")
    tensorphysics_rev = _git_revision(eco_root / "TensorPhysics")

    # Research goal
    goal = ResearchGoal(
        goal_id=f"{scenario_id}-goal",
        objective=(
            "S2 DEM→FEM one-way coupling: validate force, moment, and work conjugacy "
            "under fail-closed controller."
        ),
        acceptance_criteria=(
            "loads.json loadable with frames == history.csv rows, time monotonic, all finite",
            "fem_out.json loadable with all values finite, steps == frames",
            "force balance: |F_ext + R| / max(|F_ext|, |R|, 1e-12) ≤ 1e-6",
            "moment balance: |M_ext + M_R| / max(|M_ext|, |M_R|, 1e-12) ≤ 1e-6",
            "work conjugacy: |W_ext - U_strain| / max(|W_ext|, |U_strain|, 1e-12) ≤ 1e-5",
            "every leg manifest and artifact passes checksum verification",
            "the controller stops at the first failed gate",
        ),
        constraints={"stage": "S2", "coupling": "one-way DEM→FEM", "device": "cpu",
                     "eco_root": str(eco_root)},
        budget={"max_legs": 3},
    )
    (work_dir / "research_goal.json").write_text(
        json.dumps(goal.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    legs = []
    all_checks = []
    artifacts = []
    failed_gate = None

    # Leg 1: loads extraction
    leg1_dir = work_dir / "leg_loads"
    leg1_result = _leg_loads(S1_HISTORY_PATH, leg1_dir)
    legs.append({
        "leg": "leg_loads",
        "passed": leg1_result["passed"],
        "detail": leg1_result["detail"],
    })
    if not leg1_result["passed"]:
        failed_gate = "leg_loads"
        all_checks.append(_check("loads_extraction", False, leg1_result["detail"]))
    else:
        loads_path = leg1_result["loads_path"]
        all_checks.append(_check("loads_extraction", True, leg1_result["detail"]))
        artifacts.append((loads_path, "application/json"))

    # Leg 2: FEM solve (only if leg 1 passed)
    if leg1_result["passed"]:
        leg2_dir = work_dir / "leg_fem"
        leg2_result = _leg_fem(loads_path, leg2_dir)
        legs.append({
            "leg": "leg_fem",
            "passed": leg2_result["passed"],
            "detail": leg2_result["detail"],
        })
        if not leg2_result["passed"]:
            failed_gate = "leg_fem"
            all_checks.append(_check("fem_solve", False, leg2_result["detail"]))
        else:
            fem_out_path = leg2_result["fem_out_path"]
            all_checks.append(_check("fem_solve", True, leg2_result["detail"]))
            artifacts.append((fem_out_path, "application/json"))

    # Leg 3: gates (only if leg 2 passed)
    if leg1_result["passed"] and leg2_result["passed"]:
        leg3_result = _leg_gates(loads_path, fem_out_path)
        legs.append({
            "leg": "leg_gates",
            "passed": leg3_result["passed"],
            "checks": leg3_result["checks"],
            "detail": leg3_result["detail"],
        })
        all_checks.extend(leg3_result["checks"])
        if not leg3_result["passed"]:
            failed_gate = "leg_gates"

    # Validation report
    passed = all(c["passed"] for c in all_checks)
    report = ValidationReport(
        run_id=scenario_id,
        status="passed" if passed else "failed",
        checks=tuple(all_checks),
        summary=("All DEM→FEM conjugacy gates passed." if passed
                 else f"Failed gate: {failed_gate}" if failed_gate
                 else "One or more gates failed."),
    )
    report_path = work_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Artifact store
    store = LocalArtifactStore(work_dir / "artifact-store")
    output_refs = []
    for path, media_type in artifacts:
        ref = store.put_file(path, f"{scenario_id}/{path.name}", media_type=media_type)
        if not store.verify(ref):
            raise RuntimeError(f"artifact store verification failed: {path.name}")
        output_refs.append(ref)
    report_ref = store.put_file(report_path, f"{scenario_id}/validation_report.json",
                                media_type="application/json")
    if not store.verify(report_ref):
        raise RuntimeError("validation report artifact verification failed")

    # Run manifest
    config = {
        "scenario_id": scenario_id,
        "stage": "S2",
        "s1_history_path": str(S1_HISTORY_PATH),
        "tensorfem_revision": tensorfem_rev,
    }
    manifest = RunManifest(
        run_id=scenario_id,
        status=RunStatus.SUCCEEDED if passed else RunStatus.FAILED,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        config_sha256=RunManifest.digest_config(config),
        code_revisions={"cxs503/TensorPhysics": tensorphysics_rev,
                        "cxs503/TensorFEM": tensorfem_rev},
        environment={"python": sys.version.split()[0], "platform": sys.platform},
        resources={"execution": "local", "device": "cpu"},
        outputs=output_refs,
        validation_status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
        validation_report=report_ref,
        parent_run_ids=["wedge-icebreaking-s1"],
    )
    manifest_path = work_dir / "run_manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    manifest_ref = store.put_file(manifest_path, f"{scenario_id}/run_manifest.json",
                                  media_type="application/json")
    if not store.verify(manifest_ref):
        raise RuntimeError("run manifest artifact verification failed")

    # Research memory
    memory = ResearchMemoryRecord(
        goal_id=goal.goal_id,
        run_id=scenario_id,
        finding=report.summary,
        evidence_uris=tuple(ref.uri for ref in [*output_refs, report_ref, manifest_ref]),
        next_action="proceed_to_S3" if passed else f"repair_{failed_gate}",
    )
    (work_dir / "research_memory.json").write_text(
        json.dumps(memory.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Decision
    decision = "s2_structural_one_way_passed" if passed else "stop_s2_and_repair_failed_gate"
    scientific_status = "implemented_not_benchmark_verified"

    # Final report
    final_report = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "stage": "S2",
        "decision": decision,
        "scientific_status": scientific_status,
        "legs": legs,
        "validation_report": report.to_dict(),
        "research_memory": memory.to_dict(),
        "manifest_artifact": manifest_ref.to_dict(),
        "tensorfem_revision": tensorfem_rev,
        "tensorphysics_revision": tensorphysics_rev,
        "next_stage": ("S3: further coupling stages" if passed
                       else "repair the failed gate before retry"),
    }
    (work_dir / "s2_report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/wedge-icebreaking-s2"))
    parser.add_argument("--scenario-id", default="wedge-icebreaking-s2")
    parser.add_argument("--eco-root", type=Path, default=DEFAULT_ECO_ROOT,
                        help="six-repo workspace root holding the Tensor* checkouts")
    parser.add_argument("--venv-eco", type=Path, default=DEFAULT_VENV_ECO,
                        help="interpreter with TensorPhysics/TensorFEM installed")
    args = parser.parse_args()
    report = run_scenario(args.work_dir, args.scenario_id, args.eco_root, args.venv_eco)
    print(json.dumps({
        "scenario_id": report["scenario_id"],
        "leg_count": len(report["legs"]),
        "decision": report["decision"],
        "report": str((args.work_dir.resolve()) / "s2_report.json"),
    }, indent=2, sort_keys=True))
    if report["decision"] != "s2_structural_one_way_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
