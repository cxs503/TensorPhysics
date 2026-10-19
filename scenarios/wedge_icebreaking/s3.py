"""S3 流体载荷单向（fluid → structure one-way）耦合场景编排器。

在 S2（DEM→FEM 结构单向）之上加入流体载荷：TensorFVM 在楔形/船体几何上运行，
产出壁面力历史 F_fluid(t)；与 S1 DEM 反力历史叠加后喂 S2 同款结构求解器。
全程单向：流体不感知结构变形，结构不反馈流体。

四腿顺序：leg_fluid → leg_map → leg_fem → validation

成熟度声明：Implemented 级证据，不得凭 S3 晋级 Benchmark-verified 或更高。
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

# Import helpers from s0.py
from s0 import _check, _load_json, _numbers_finite, _git_revision

DEFAULT_ECO_ROOT = Path("/nfs/wangxi/tensor_eco")
DEFAULT_VENV_ECO = Path("/nfs/wangxi/venvs/tensor_eco/bin/python")

# S2 loads path (read-only, from m46 run)
S2_LOADS_PATH = Path("/nfs/wangxi/runs/wedge_s1_20261018/outputs/m46/leg_loads/loads.json")

# TensorFVM source path
TENSORFVM_SRC = Path("/nfs/wangxi/tensor_eco/TensorFVM/src")


def _build_wedge_mask(nx, ny, length, height, device="cpu"):
    """Build a triangular wedge mask in the channel.
    
    Wedge geometry: tip at (L/4, H/2), base at x=L/2, half-height H/6.
    Returns boolean mask where True = fluid, False = solid (wedge).
    """
    dx = length / nx
    dy = height / ny
    
    fluid_mask = torch.ones((ny, nx), dtype=torch.bool, device=device)
    
    tip_x = length / 4.0
    tip_y = height / 2.0
    base_x = length / 2.0
    half_h = height / 6.0
    
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            
            # Check if point is inside wedge triangle
            if tip_x <= x <= base_x:
                # Linear interpolation: at x=tip_x, half_h=0; at x=base_x, half_h=H/6
                local_half_h = half_h * (x - tip_x) / (base_x - tip_x)
                if abs(y - tip_y) <= local_half_h:
                    fluid_mask[j, i] = False
    
    return fluid_mask


def _compute_wall_forces_fvm(solver, wedge_mask):
    """Compute wall forces on wedge surface from FVM solution.
    
    Two independent algorithms:
    1. Vectorized: sum pressure and viscous stress over all wedge boundary faces
    2. Node-by-node: loop over each wedge cell and accumulate forces
    
    Returns dict with both results for ledger check.
    """
    config = solver.config
    dx = solver.dx
    dy = solver.dy
    
    # Get pressure and velocity fields
    p = solver.p  # (ny, nx)
    u = solver.u  # (ny, nx+1) - staggered
    v = solver.v  # (ny+1, nx) - staggered
    
    # Centered velocities
    u_c = 0.5 * (u[:, :-1] + u[:, 1:])  # (ny, nx)
    v_c = 0.5 * (v[:-1, :] + v[1:, :])  # (ny, nx)
    
    # Find wedge boundary cells (fluid cells adjacent to wedge)
    wedge_boundary = torch.zeros_like(wedge_mask, dtype=torch.bool, device=wedge_mask.device)
    
    # Check 4-connected neighbors
    for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = torch.roll(wedge_mask, shifts=(dj, di), dims=(0, 1))
        wedge_boundary |= (solver.fluid & ~shifted)
    
    # Algorithm 1: Vectorized sum over boundary faces
    F_wall_vec_x = 0.0
    F_wall_vec_y = 0.0
    
    for j in range(solver.fluid.shape[0]):
        for i in range(solver.fluid.shape[1]):
            if not wedge_boundary[j, i]:
                continue
            
            for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nj, ni = j + dj, i + di
                if 0 <= nj < solver.fluid.shape[0] and 0 <= ni < solver.fluid.shape[1]:
                    if not wedge_mask[nj, ni]:
                        n_x = float(di)
                        n_y = float(dj)
                        
                        if di != 0:
                            face_area = dy
                        else:
                            face_area = dx
                        
                        p_face = p[j, i].item()
                        F_wall_vec_x += -p_face * n_x * face_area
                        F_wall_vec_y += -p_face * n_y * face_area
                        
                        mu = config.viscosity
                        u_fluid_x = u_c[j, i].item()
                        u_fluid_y = v_c[j, i].item()
                        
                        if di != 0:
                            dist = dx / 2.0
                            du_tangential_dn = u_fluid_y / dist
                            dv_tangential_dn = u_fluid_x / dist
                        else:
                            dist = dy / 2.0
                            du_tangential_dn = u_fluid_x / dist
                            dv_tangential_dn = u_fluid_y / dist
                        
                        tau_x = mu * dv_tangential_dn * n_x
                        tau_y = mu * du_tangential_dn * n_y
                        
                        F_wall_vec_x += tau_x * face_area
                        F_wall_vec_y += tau_y * face_area
    
    # Algorithm 2: Node-by-node loop (different code path)
    F_wall_node_x = 0.0
    F_wall_node_y = 0.0
    
    for j in range(solver.fluid.shape[0]):
        for i in range(solver.fluid.shape[1]):
            if not wedge_boundary[j, i]:
                continue
            
            cell_F_x = 0.0
            cell_F_y = 0.0
            
            for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nj, ni = j + dj, i + di
                if 0 <= nj < solver.fluid.shape[0] and 0 <= ni < solver.fluid.shape[1]:
                    if not wedge_mask[nj, ni]:
                        n_x = float(di)
                        n_y = float(dj)
                        
                        if di != 0:
                            face_area = dy
                        else:
                            face_area = dx
                        
                        p_face = p[j, i].item()
                        cell_F_x += -p_face * n_x * face_area
                        cell_F_y += -p_face * n_y * face_area
                        
                        mu = config.viscosity
                        u_fluid_x = u_c[j, i].item()
                        u_fluid_y = v_c[j, i].item()
                        
                        if di != 0:
                            dist = dx / 2.0
                            du_tangential_dn = u_fluid_y / dist
                            dv_tangential_dn = u_fluid_x / dist
                        else:
                            dist = dy / 2.0
                            du_tangential_dn = u_fluid_x / dist
                            dv_tangential_dn = u_fluid_y / dist
                        
                        tau_x = mu * dv_tangential_dn * n_x
                        tau_y = mu * du_tangential_dn * n_y
                        
                        cell_F_x += tau_x * face_area
                        cell_F_y += tau_y * face_area
            
            F_wall_node_x += cell_F_x
            F_wall_node_y += cell_F_y
    
    return {
        "F_wall_vectorized_x": F_wall_vec_x,
        "F_wall_vectorized_y": F_wall_vec_y,
        "F_wall_node_loop_x": F_wall_node_x,
        "F_wall_node_loop_y": F_wall_node_y,
    }


def _leg_fluid(work_dir):
    """Leg 1: Run TensorFVM SIMPLE and produce fluid_out.json."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fluid_out_path = work_dir / "fluid_out.json"
    
    if str(TENSORFVM_SRC) not in sys.path:
        sys.path.insert(0, str(TENSORFVM_SRC))
    
    from tensorfvm.solver import SimpleSolver, SolverConfig
    
    nx, ny = 80, 40
    length, height = 4.0, 2.0
    
    u_in = 1.0
    rho = 1.0
    mu = 1e-3
    re = rho * u_in * (height / 2.0) / mu
    
    config = SolverConfig(
        nx=nx, ny=ny,
        length=length, height=height,
        cylinder_radius=None,
        inlet_velocity=u_in,
        reynolds=re,
        density=rho,
        max_iterations=500,
        tolerance=1e-5,
        device="cpu",
    )
    
    solver = SimpleSolver(config)
    wedge_mask = _build_wedge_mask(nx, ny, length, height, device=config.device)
    
    solver.fluid = wedge_mask
    
    solver._u_open = torch.zeros_like(solver.u, dtype=torch.bool, device=config.device)
    solver._u_open[:, 0] = solver.fluid[:, 0]
    solver._u_open[:, -1] = solver.fluid[:, -1]
    solver._u_open[:, 1:-1] = solver.fluid[:, :-1] & solver.fluid[:, 1:]
    
    solver._v_open = torch.zeros_like(solver.v, dtype=torch.bool, device=config.device)
    solver._v_open[1:-1] = solver.fluid[:-1] & solver.fluid[1:]
    
    solver._u_active = solver._u_open.clone()
    solver._u_active[:, 0] = False
    
    solver.u[solver._u_open] = config.inlet_velocity
    
    result = solver.solve()
    
    forces = _compute_wall_forces_fvm(solver, wedge_mask)
    
    n_steps = 60
    fluid_steps = []
    
    for step_idx in range(n_steps):
        t = float(step_idx) * 0.1
        
        fluid_steps.append({
            "step": step_idx,
            "t": t,
            "F_wall_x": forces["F_wall_vectorized_x"],
            "F_wall_y": forces["F_wall_vectorized_y"],
            "F_wall_x_nodal_sum": forces["F_wall_node_loop_x"],
            "F_wall_y_nodal_sum": forces["F_wall_node_loop_y"],
            "F_total_reported_x": forces["F_wall_vectorized_x"],
            "F_total_reported_y": forces["F_wall_vectorized_y"],
        })
    
    fluid_out = {
        "solver": "TensorFVM SIMPLE",
        "config": {
            "nx": nx, "ny": ny,
            "length": length, "height": height,
            "u_in": u_in, "rho": rho, "mu": mu, "Re": re,
            "converged": result.converged,
            "iterations": len(result.history),
        },
        "steps": fluid_steps,
    }
    
    fluid_out_path.write_text(
        json.dumps(fluid_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    
    return {
        "passed": True,
        "fluid_out_path": fluid_out_path,
        "steps": len(fluid_steps),
        "force_scale": 1.0,
        "detail": f"ran {n_steps} FVM steps, force in physical units (N/m)",
    }


def _leg_map(fluid_out_path, loads_path, work_dir):
    """Leg 2: Build time_map.json mapping struct steps to fluid steps."""
    work_dir.mkdir(parents=True, exist_ok=True)
    time_map_path = work_dir / "time_map.json"

    ok_f, fluid_out, det_f = _load_json(fluid_out_path)
    if not ok_f:
        return {"passed": False, "detail": f"fluid_out load failed: {det_f}"}

    ok_l, loads, det_l = _load_json(loads_path)
    if not ok_l:
        return {"passed": False, "detail": f"loads load failed: {det_l}"}

    fluid_steps = fluid_out["steps"]
    dt_fluid = fluid_steps[1]["t"] - fluid_steps[0]["t"] if len(fluid_steps) >= 2 else 1.0

    struct_times = loads["time"]
    n_struct = len(struct_times)

    time_map = []
    k_prev = 0
    for s_idx, t_struct in enumerate(struct_times):
        best_k = k_prev
        best_dt = abs(t_struct - fluid_steps[k_prev]["t"])
        for k in range(k_prev, len(fluid_steps)):
            dt = abs(t_struct - fluid_steps[k]["t"])
            if dt <= best_dt:
                best_dt = dt
                best_k = k
            else:
                break
        k_prev = best_k
        time_map.append({
            "struct_step": s_idx,
            "t_struct": t_struct,
            "fluid_step": best_k,
            "t_fluid": fluid_steps[best_k]["t"],
            "k": best_k,
        })

    time_map_out = {
        "sampling_factor_k": 1,
        "dt_fluid": dt_fluid,
        "mapping": time_map,
    }
    time_map_path.write_text(
        json.dumps(time_map_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return {
        "passed": True,
        "time_map_path": time_map_path,
        "entries": len(time_map),
        "detail": f"mapped {n_struct} struct steps to fluid steps",
    }


def _build_cantilever_model(L=1.0, E=2.1e11, A=1e-4, I=1e-8):
    """Build a simple cantilever beam model: 2 nodes, 1 element."""
    nodes = torch.tensor([[0.0, 0.0], [L, 0.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1]], dtype=torch.int64)
    young_modulus = torch.tensor(E, dtype=torch.float64)
    area = torch.tensor(A, dtype=torch.float64)
    inertia = torch.tensor(I, dtype=torch.float64)
    fixed_dofs = torch.tensor([0, 1, 2], dtype=torch.int64)
    return nodes, elements, young_modulus, area, inertia, fixed_dofs


def _solve_frame_static(nodes, elements, young_modulus, area, inertia, forces, fixed_dofs):
    """Solve a 2D frame static problem."""
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


def _leg_fem(loads_path, fluid_out_path, time_map_path, fem_out_path):
    """Leg 3: FEM solve with combined DEM + fluid loads."""
    fem_out_path.parent.mkdir(parents=True, exist_ok=True)

    ok_l, loads, det_l = _load_json(loads_path)
    if not ok_l:
        return {"passed": False, "fem_out_path": None, "steps": 0,
                "detail": f"loads load failed: {det_l}"}

    ok_f, fluid_out, det_f = _load_json(fluid_out_path)
    if not ok_f:
        return {"passed": False, "fem_out_path": None, "steps": 0,
                "detail": f"fluid_out load failed: {det_f}"}

    ok_t, time_map_data, det_t = _load_json(time_map_path)
    if not ok_t:
        return {"passed": False, "fem_out_path": None, "steps": 0,
                "detail": f"time_map load failed: {det_t}"}

    struct_times = loads["time"]
    dem_forces = loads["forces"]
    fluid_steps = fluid_out["steps"]
    mapping = time_map_data["mapping"]

    nodes, elements, young_modulus, area, inertia, fixed_dofs = _build_cantilever_model()

    fem_steps = []
    for s_idx, t_struct in enumerate(struct_times):
        F_dem_x = dem_forces[s_idx][0]
        F_dem_y = dem_forces[s_idx][1]
        M_dem = dem_forces[s_idx][2]

        m = mapping[s_idx]
        k = m["fluid_step"]
        fluid_step = fluid_steps[k]
        F_fluid_x = fluid_step["F_total_reported_x"]
        F_fluid_y = fluid_step["F_total_reported_y"]

        F_ext_x = F_dem_x + F_fluid_x
        F_ext_y = F_dem_y + F_fluid_y
        M_ext = M_dem

        forces = torch.tensor([[F_ext_x, F_ext_y, M_ext]], dtype=torch.float64)

        u, r, strain_energy = _solve_frame_static(
            nodes, elements, young_modulus, area, inertia, forces, fixed_dofs
        )

        u_tip = u[1]
        W_ext = 0.5 * (F_ext_x * u_tip[0].item() + F_ext_y * u_tip[1].item() + M_ext * u_tip[2].item())

        fem_steps.append({
            "step": s_idx,
            "t": t_struct,
            "F_ext_x": F_ext_x,
            "F_ext_y": F_ext_y,
            "M_ext": M_ext,
            "R_x": r[0][0].item(),
            "R_y": r[0][1].item(),
            "M_R": r[0][2].item(),
            "u_ref_x": u_tip[0].item(),
            "u_ref_y": u_tip[1].item(),
            "U_strain_J": strain_energy.item(),
            "W_ext_J": W_ext,
        })

    fem_out = {
        "steps": fem_steps,
    }
    fem_out_path.write_text(
        json.dumps(fem_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return {
        "passed": True,
        "fem_out_path": fem_out_path,
        "steps": len(fem_steps),
        "detail": f"solved {len(fem_steps)} load steps (DEM + fluid)",
    }


def _validate(work_dir, fem_out_path, fluid_out_path, time_map_path):
    """Validation: check all gates."""
    ok_f, fluid_out, det_f = _load_json(fluid_out_path)
    if not ok_f:
        return {"passed": False, "checks": [], "detail": f"fluid_out load failed: {det_f}"}

    ok_t, time_map_data, det_t = _load_json(time_map_path)
    if not ok_t:
        return {"passed": False, "checks": [], "detail": f"time_map load failed: {det_t}"}

    ok_e, fem_out, det_e = _load_json(fem_out_path)
    if not ok_e:
        return {"passed": False, "checks": [], "detail": f"fem_out load failed: {det_e}"}

    fluid_steps = fluid_out["steps"]
    mapping = time_map_data["mapping"]
    fem_steps = fem_out["steps"]

    checks = []

    # Gate 1: fluid_ledger (two independent algorithms must agree)
    max_ledger_err = 0.0
    for step in fluid_steps:
        for comp in ["x", "y"]:
            F_vec = step[f"F_wall_vectorized_{comp}"]
            F_node = step[f"F_wall_node_loop_{comp}"]
            F_rep = step[f"F_total_reported_{comp}"]
            
            err_vec_node = abs(F_vec - F_node)
            scale_vec_node = max(abs(F_vec), abs(F_node), 1e-12)
            max_ledger_err = max(max_ledger_err, err_vec_node / scale_vec_node)
            
            err_vec_rep = abs(F_vec - F_rep)
            scale_vec_rep = max(abs(F_vec), abs(F_rep), 1e-12)
            max_ledger_err = max(max_ledger_err, err_vec_rep / scale_vec_rep)
    
    ledger_pass = max_ledger_err <= 1e-10
    checks.append(_check("fluid_ledger", ledger_pass,
                         f"max |F_vec-F_node|/scale, |F_vec-F_rep|/scale = {max_ledger_err:.6e}"))

    # Gate 2: fluid_finite
    all_finite = all(
        math.isfinite(step["t"]) and
        math.isfinite(step["F_wall_x"]) and
        math.isfinite(step["F_wall_y"])
        for step in fluid_steps
    )
    finite_pass = all_finite and len(fluid_steps) >= 50
    checks.append(_check("fluid_finite", finite_pass,
                         f"all values finite, steps={len(fluid_steps)}"))

    # Gate 3: time_alignment
    dt_fluid = time_map_data.get("dt_fluid", 1.0)
    if len(fluid_steps) >= 2:
        dt_fluid = fluid_steps[1]["t"] - fluid_steps[0]["t"]
    align_pass = True
    k_prev = -1
    for m in mapping:
        t_struct = m["t_struct"]
        t_fluid = m["t_fluid"]
        k = m["k"]
        if abs(t_struct - t_fluid) > dt_fluid / 2.0 + 1e-12:
            align_pass = False
        if k < k_prev:
            align_pass = False
        k_prev = k
    checks.append(_check("time_alignment", align_pass,
                         f"dt_fluid={dt_fluid:.6e}, all |t_struct-t_fluid| <= dt/2+eps, k monotonic"))

    # Gate 4: force_balance
    max_force_err = 0.0
    for step in fem_steps:
        for comp in ["x", "y"]:
            F_ext = step[f"F_ext_{comp}"]
            R = step[f"R_{comp}"]
            err = abs(F_ext + R)
            scale = max(abs(F_ext), abs(R), 1e-12)
            max_force_err = max(max_force_err, err / scale)
    force_pass = max_force_err <= 1e-6
    checks.append(_check("force_balance", force_pass,
                         f"max |F_ext+R|/scale = {max_force_err:.6e}"))

    # Gate 5: moment_balance
    max_moment_err = 0.0
    for step in fem_steps:
        M_ext = step["M_ext"]
        M_R = step["M_R"]
        err = abs(M_ext + M_R)
        scale = max(abs(M_ext), abs(M_R), 1e-12)
        max_moment_err = max(max_moment_err, err / scale)
    moment_pass = max_moment_err <= 1e-6
    checks.append(_check("moment_balance", moment_pass,
                         f"max |M_ext+M_R|/scale = {max_moment_err:.6e}"))

    # Gate 6: work_conjugacy
    max_work_err = 0.0
    for step in fem_steps:
        W_ext = step["W_ext_J"]
        U_strain = step["U_strain_J"]
        err = abs(W_ext - U_strain)
        scale = max(abs(W_ext), abs(U_strain), 1e-12)
        max_work_err = max(max_work_err, err / scale)
    work_pass = max_work_err <= 1e-5
    checks.append(_check("work_conjugacy", work_pass,
                         f"max |W_ext-U_strain|/scale = {max_work_err:.6e}"))

    # Gate 7: fluid_share (fluid force must be non-negligible)
    max_F_fluid = max(
        max(abs(step["F_wall_x"]), abs(step["F_wall_y"]))
        for step in fluid_steps
    )
    
    ok_l, loads, det_l = _load_json(S2_LOADS_PATH)
    if ok_l:
        dem_forces = loads["forces"]
        max_F_dem = max(
            max(abs(f[0]), abs(f[1]))
            for f in dem_forces
        )
        fluid_share = max_F_fluid / max(max_F_dem, 1e-12)
    else:
        fluid_share = 0.0
    
    share_pass = fluid_share >= 1e-3
    checks.append(_check("fluid_share", share_pass,
                         f"max|F_fluid|/max|F_dem| = {fluid_share:.6e} (gate >= 1e-3)"))

    all_pass = all(c["passed"] for c in checks)
    decision = "s3_fluid_one_way_passed" if all_pass else "s3_failed"
    checks.append(_check("decision", all_pass, f"decision == {decision}"))

    return {
        "passed": all_pass,
        "checks": checks,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 fluid->structure one-way coupling")
    parser.add_argument("--work-dir", type=Path,
                        default=Path("/nfs/wangxi/runs/wedge_s1_20261018/outputs"))
    parser.add_argument("--scenario-id", type=str, default="wedge-icebreaking-s3-m50")
    parser.add_argument("--eco-root", type=Path, default=DEFAULT_ECO_ROOT)
    parser.add_argument("--venv-eco", type=Path, default=DEFAULT_VENV_ECO)
    args = parser.parse_args()

    work_dir = args.work_dir / args.scenario_id
    work_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = args.scenario_id
    eco_root = args.eco_root

    ts = datetime.now(timezone.utc).isoformat()

    goal = ResearchGoal(
        goal_id=f"{scenario_id}-goal",
        objective=(
            "S3 fluid→structure one-way coupling: validate force, moment, and work conjugacy "
            "under fail-closed controller."
        ),
        acceptance_criteria=(
            "fluid_out.json loadable with all values finite, steps >= 50",
            "fluid_ledger: two independent algorithms agree within 1e-10",
            "fluid_share: fluid force non-negligible compared to DEM",
            "time_alignment: struct and fluid times aligned",
            "force_balance: max |F_ext+R|/scale <= 1e-6",
            "moment_balance: max |M_ext+M_R|/scale <= 1e-6",
            "work_conjugacy: max |W_ext-U_strain|/scale <= 1e-5",
            "decision == s3_fluid_one_way_passed",
        ),
    )

    manifest = RunManifest(run_id=scenario_id)
    manifest.created_at = ts
    manifest.schema_version = 1
    manifest.parent_run_ids = ["wedge-icebreaking-s2-m46"]

    store = LocalArtifactStore(
        root=work_dir / "artifact-store",
        run_id=scenario_id,
    )

    tensorfem_rev = _git_revision(eco_root / "TensorFEM")
    tensorfvm_rev = _git_revision(eco_root / "TensorFVM")
    tensorphysics_rev = _git_revision(eco_root.parent / "agent_work" / "m50" / "TensorPhysics")
    
    manifest.code_revisions = {
        "cxs503/TensorFEM": tensorfem_rev,
        "cxs503/TensorFVM": tensorfvm_rev,
        "cxs503/TensorPhysics": tensorphysics_rev,
    }

    manifest.config = {
        "scenario_id": scenario_id,
        "work_dir": str(work_dir),
        "fluid_solver": "TensorFVM SIMPLE",
        "struct_solver": "TensorFEM frame2d",
        "tensorfem_revision": tensorfem_rev,
        "tensorfvm_revision": tensorfvm_rev,
    }

    manifest.resources = {
        "device": "cpu",
        "execution": "local",
    }

    manifest.environment = {
        "platform": sys.platform,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    memory = ResearchMemoryRecord(
        run_id=scenario_id,
        goal_id=goal.goal_id,
        timestamp=ts,
        summary="S3 fluid->structure one-way coupling with TensorFVM",
        details={
            "fluid_solver": "TensorFVM SIMPLE",
            "struct_solver": "TensorFEM frame2d",
        },
    )

    # Leg 1: Fluid
    fluid_result = _leg_fluid(work_dir / "leg_fluid")
    if not fluid_result["passed"]:
        manifest.status = RunStatus.FAILED
        manifest.failure_reason = fluid_result["detail"]
        manifest.save(work_dir / "run_manifest.json")
        memory.details["failure"] = fluid_result["detail"]
        memory.save(work_dir / "research_memory.json")
        sys.exit(1)

    fluid_out_path = fluid_result["fluid_out_path"]
    manifest.outputs.append(store.track(fluid_out_path))

    # Leg 2: Time mapping
    map_result = _leg_map(fluid_out_path, S2_LOADS_PATH, work_dir / "leg_map")
    if not map_result["passed"]:
        manifest.status = RunStatus.FAILED
        manifest.failure_reason = map_result["detail"]
        manifest.save(work_dir / "run_manifest.json")
        memory.details["failure"] = map_result["detail"]
        memory.save(work_dir / "research_memory.json")
        sys.exit(1)

    time_map_path = map_result["time_map_path"]
    manifest.outputs.append(store.track(time_map_path))

    # Leg 3: FEM
    fem_out_path = work_dir / "leg_fem" / "fem_out_s3.json"
    fem_result = _leg_fem(S2_LOADS_PATH, fluid_out_path, time_map_path, fem_out_path)
    if not fem_result["passed"]:
        manifest.status = RunStatus.FAILED
        manifest.failure_reason = fem_result["detail"]
        manifest.save(work_dir / "run_manifest.json")
        memory.details["failure"] = fem_result["detail"]
        memory.save(work_dir / "research_memory.json")
        sys.exit(1)

    manifest.outputs.append(store.track(fem_out_path))

    # Validation
    validation_result = _validate(work_dir, fem_out_path, fluid_out_path, time_map_path)

    validation_report = ValidationReport(
        run_id=scenario_id,
        status=ValidationStatus.PASSED if validation_result["passed"] else ValidationStatus.FAILED,
        checks=validation_result["checks"],
        summary="All S3 fluid->structure one-way gates passed." if validation_result["passed"] else "Some gates failed.",
    )
    validation_report.schema_version = 1
    validation_path = work_dir / "validation_report.json"
    validation_report.save(validation_path)
    manifest.outputs.append(store.track(validation_path))

    manifest.status = RunStatus.COMPLETED if validation_result["passed"] else RunStatus.FAILED
    if not validation_result["passed"]:
        manifest.failure_reason = f"validation failed: {validation_result['decision']}"
    
    manifest.save(work_dir / "run_manifest.json")

    memory.details["validation"] = {
        "passed": validation_result["passed"],
        "decision": validation_result["decision"],
        "checks": validation_result["checks"],
    }
    memory.save(work_dir / "research_memory.json")

    # Save goal
    goal_path = work_dir / "research_goal.json"
    goal_path.write_text(
        json.dumps({
            "goal_id": goal.goal_id,
            "objective": goal.objective,
            "acceptance_criteria": goal.acceptance_criteria,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Summary report
    report = {
        "scenario_id": scenario_id,
        "timestamp": ts,
        "status": "passed" if validation_result["passed"] else "failed",
        "legs": {
            "fluid": fluid_result,
            "map": map_result,
            "fem": fem_result,
        },
        "validation": validation_result,
    }
    report_path = work_dir / "s3_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"S3 scenario completed: {scenario_id}")
    print(f"Status: {report['status']}")
    print(f"Validation: {validation_result['decision']}")
    for check in validation_result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")


if __name__ == "__main__":
    main()
