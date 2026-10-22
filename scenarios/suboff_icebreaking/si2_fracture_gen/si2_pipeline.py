"""SI-2 断裂生成管线 —— FEM cohesive 破坏 → DEM 键结颗粒冰场

端到端管线：
1. FEM cohesive 破坏单元检测 (cohesive_failure.py)
2. 破坏面 → DEM 颗粒参数映射 (fracture_gen.py)
3. 双账本对账：质量守恒 + 断裂能量守恒 (fracture_ledger.py)
4. 生成 TensorDEM 可初始化的键结颗粒冰场文件

输出：
- particles.json: DEM 颗粒集（位置/半径/质量/速度/键结信息）
- ledger_a.json: FEM 侧账本（核销质量 + 释放断裂能量）
- ledger_b.json: DEM 侧账本（颗粒质量 + 断裂能量）
- consistency_report.json: 双账本对账报告
- bonded_ice_field.json: TensorDEM 键结颗粒冰场初始化文件
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

# 添加项目路径 (repo root = 4 levels up from this file)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, _REPO_ROOT)

from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    solve_plate_on_foundation,
    structured_square_mesh,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.cohesive_failure import (
    identify_failed_faces,
    save_failed_faces,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.fracture_gen import (
    generate_particles_from_failed_faces,
    compute_fem_ledger,
    compute_dem_ledger,
    save_particles,
    save_ledger_a,
    save_ledger_b,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.fracture_ledger import (
    reconcile_ledgers,
    save_reconciliation_report,
)


def utcnow_z() -> str:
    """返回 UTC 时间戳（Z 后缀）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_bonded_ice_field(
    particles_json_path: str,
    output_path: str,
    bond_stiffness: float = 1e6,
    bond_strength: float = 1e4,
    bond_factor: float = 3.0,
) -> dict:
    """从颗粒集生成键结颗粒冰场初始化文件。

    键结模型：相邻颗粒间建立键结（bond），模拟冰的连续性。
    当键结受力超过 bond_strength 时断裂，对应宏观裂纹扩展。

    Parameters
    ----------
    particles_json_path : str
        颗粒文件路径。
    output_path : str
        输出键结冰场文件路径。
    bond_stiffness : float
        键结刚度 (N/m)。
    bond_strength : float
        键结强度 (N)。
    bond_factor : float
        键结距离因子：两颗粒中心距 < (r1 + r2) * bond_factor 时建立键结。

    Returns
    -------
    dict
        键结冰场信息。
    """
    with open(particles_json_path, "r", encoding="utf-8") as f:
        particles_data = json.load(f)

    particles = particles_data["particles"]
    n_particles = len(particles)

    # 提取颗粒位置
    positions = np.array([p["position"] for p in particles])
    radii = np.array([p["radius"] for p in particles])

    # 建立键结：基于距离的邻近搜索
    # 两颗粒中心距 < (r1 + r2) * bond_factor 时建立键结
    bonds = []
    bond_id = 0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            contact_dist = (radii[i] + radii[j]) * bond_factor
            if dist < contact_dist:
                bonds.append({
                    "bond_id": bond_id,
                    "particle_i": i,
                    "particle_j": j,
                    "stiffness": bond_stiffness,
                    "strength": bond_strength,
                    "initial_length": float(dist),
                    "status": "intact",  # intact / broken
                })
                bond_id += 1

    # 构建键结冰场
    ice_field = {
        "metadata": {
            "timestamp": utcnow_z(),
            "description": "TensorDEM 键结颗粒冰场（从 FEM 断裂生成管线生成）",
            "n_particles": n_particles,
            "n_bonds": len(bonds),
            "bond_model": "linear_elastic",
            "bond_stiffness": bond_stiffness,
            "bond_strength": bond_strength,
        },
        "particles": particles,
        "bonds": bonds,
    }

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ice_field, f, indent=2)

    return ice_field


def run_end_to_end_pipeline(output_dir: Path) -> dict:
    """运行端到端管线。

    Returns
    -------
    dict
        管线结果摘要。
    """
    print("=" * 70)
    print("SI-2 断裂生成管线：端到端 FEM → DEM")
    print("=" * 70)
    print(f"时间戳: {utcnow_z()}")
    print()

    # -------------------------------------------------------------------------
    # 参数设置
    # -------------------------------------------------------------------------
    mesh_res = 4
    side = 1.0
    E = 1e7
    nu = 0.3
    t = 0.01
    rho = 917.0
    sigma_c = 1.0   # 极低强度，确保有破坏
    G_c = 100.0
    q0 = 1e5

    print(f"参数:")
    print(f"  网格分辨率: {mesh_res}×{mesh_res}")
    print(f"  板边长: {side} m")
    print(f"  弹性模量: {E} Pa")
    print(f"  泊松比: {nu}")
    print(f"  板厚度: {t} m")
    print(f"  密度: {rho} kg/m³")
    print(f"  cohesive 强度: {sigma_c} Pa")
    print(f"  断裂能密度: {G_c} J/m²")
    print()

    # -------------------------------------------------------------------------
    # 步骤 1：FEM 板求解
    # -------------------------------------------------------------------------
    print("步骤 1：FEM 板求解")
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)
    n_nodes = len(nodes_2d)

    # 正弦载荷
    forces = np.zeros((n_nodes, 3))
    for i, (x, y) in enumerate(nodes_2d):
        forces[i, 2] = -q0 * np.sin(np.pi * x / side) * np.sin(np.pi * y / side)

    result = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, t, 0.0, forces, "simply_supported"
    )
    print(f"  节点数: {n_nodes}")
    # displacement is (ndof,) with 3 dofs per node (ux, uy, uz)
    max_deflection = float(np.max(np.abs(result.displacement[2::3])))
    print(f"  最大挠度: {max_deflection:.4e} m")
    print()

    # -------------------------------------------------------------------------
    # 步骤 2：Cohesive 破坏检测
    # -------------------------------------------------------------------------
    print("步骤 2：Cohesive 破坏检测")
    failure_result = identify_failed_faces(
        nodes_2d, elements, result.displacement, E, nu, t, sigma_c, G_c
    )
    print(f"  总内部 face 数: {failure_result.n_total_faces}")
    print(f"  破坏 face 数: {failure_result.n_failed}")

    if failure_result.n_failed == 0:
        raise ValueError("无破坏面，测试参数有问题")

    failed_faces_path = output_dir / "failed_faces.json"
    save_failed_faces(failure_result, str(failed_faces_path))
    print(f"  破坏面已保存: {failed_faces_path}")
    print()

    # -------------------------------------------------------------------------
    # 步骤 3：破坏面 → DEM 颗粒映射
    # -------------------------------------------------------------------------
    print("步骤 3：破坏面 → DEM 颗粒映射（等体积球）")

    with open(failed_faces_path, "r", encoding="utf-8") as f:
        failed_faces_data = json.load(f)

    gen_result = generate_particles_from_failed_faces(
        failed_faces_data, nodes_2d, rho
    )

    print(f"  生成颗粒数: {len(gen_result.particles)}")
    print(f"  总质量: {gen_result.total_mass:.6e} kg")
    print(f"  总断裂能: {gen_result.total_fracture_energy:.6e} J")
    print()

    # -------------------------------------------------------------------------
    # 步骤 4：双账本对账
    # -------------------------------------------------------------------------
    print("步骤 4：双账本对账")

    ledger_a = compute_fem_ledger(failed_faces_data, rho)
    ledger_b = compute_dem_ledger(gen_result)

    # 保存账本
    ledger_a_path = output_dir / "ledger_a.json"
    ledger_b_path = output_dir / "ledger_b.json"
    save_ledger_a(ledger_a, str(ledger_a_path))
    save_ledger_b(ledger_b, str(ledger_b_path))

    # 对账
    consistency_report = reconcile_ledgers(ledger_a, ledger_b)
    consistency_path = output_dir / "consistency_report.json"
    save_reconciliation_report(consistency_report, str(consistency_path))

    print(f"  质量守恒:")
    print(f"    FEM 核销: {ledger_a['total_mass']:.10e} kg")
    print(f"    DEM 颗粒: {ledger_b['total_mass']:.10e} kg")
    print(f"    相对误差: {consistency_report['mass_conservation']['relative_error']:.3e}")
    print(f"    判据: |Δm|/Σm ≤ 1e-12")
    print(f"    状态: {'PASS' if consistency_report['mass_conservation']['passed'] else 'FAIL'}")
    print()
    print(f"  断裂能量守恒:")
    print(f"    FEM 释放: {ledger_a['total_fracture_energy']:.10e} J")
    print(f"    DEM 继承: {ledger_b['total_fracture_energy']:.10e} J")
    print(f"    相对误差: {consistency_report['energy_conservation']['relative_error']:.3e}")
    print(f"    判据: |ΔE|/ΣE ≤ 1e-12")
    print(f"    状态: {'PASS' if consistency_report['energy_conservation']['passed'] else 'FAIL'}")
    print()

    # -------------------------------------------------------------------------
    # 步骤 5：保存颗粒文件
    # -------------------------------------------------------------------------
    print("步骤 5：保存颗粒文件")
    particles_path = output_dir / "particles.json"
    save_particles(gen_result, str(particles_path))
    print(f"  颗粒文件已保存: {particles_path}")
    print()

    # -------------------------------------------------------------------------
    # 步骤 6：生成键结颗粒冰场
    # -------------------------------------------------------------------------
    print("步骤 6：生成 TensorDEM 键结颗粒冰场")
    ice_field_path = output_dir / "bonded_ice_field.json"
    ice_field = generate_bonded_ice_field(
        str(particles_path),
        str(ice_field_path),
        bond_stiffness=1e6,
        bond_strength=1e4,
    )
    print(f"  颗粒数: {ice_field['metadata']['n_particles']}")
    print(f"  键结数: {ice_field['metadata']['n_bonds']}")
    print(f"  键结冰场已保存: {ice_field_path}")
    print()

    # -------------------------------------------------------------------------
    # 汇总
    # -------------------------------------------------------------------------
    summary = {
        "timestamp": utcnow_z(),
        "pipeline": "SI-2: FEM cohesive 破坏 → DEM 键结颗粒冰场",
        "steps": {
            "fem_solve": {"n_nodes": n_nodes, "max_deflection": max_deflection},
            "cohesive_failure": {"n_total_faces": failure_result.n_total_faces, "n_failed": failure_result.n_failed},
            "particle_generation": {"n_particles": len(gen_result.particles), "total_mass": gen_result.total_mass, "total_fracture_energy": gen_result.total_fracture_energy},
            "ledger_consistency": {"mass_rel_error": consistency_report['mass_conservation']['relative_error'], "energy_rel_error": consistency_report['energy_conservation']['relative_error'], "status": consistency_report['status']},
            "bonded_ice_field": {"n_particles": ice_field['metadata']['n_particles'], "n_bonds": ice_field['metadata']['n_bonds']},
        },
        "output_files": {
            "failed_faces": str(failed_faces_path),
            "particles": str(particles_path),
            "ledger_a": str(ledger_a_path),
            "ledger_b": str(ledger_b_path),
            "consistency_report": str(consistency_path),
            "bonded_ice_field": str(ice_field_path),
        },
    }

    return summary


if __name__ == "__main__":
    # Output to repo-root benchmarks/icebreaking/si2_end_to_end_output/
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = _repo_root / "benchmarks" / "icebreaking" / "si2_end_to_end_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_end_to_end_pipeline(output_dir)

    # 保存汇总
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("管线完成")
    print("=" * 70)
    print(f"汇总已保存: {summary_path}")
