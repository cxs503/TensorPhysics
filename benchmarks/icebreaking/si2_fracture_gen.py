"""SI-2 断裂生成验证脚本。

验证三个锚点：
1. 锚1（质量守恒）：破坏单元核销质量 vs 生成颗粒质量之和，|Δ|/Σ ≤ 1e-12
2. 锚2（断裂能量守恒）：cohesive face 释放能 ∑G·A vs 账本记录断裂能量，|Δ|/Σ ≤ 1e-12
3. 锚3（解析枚举锚）：预设均匀应力场 + 给定强度阈值——破坏 face 集合可解析手算
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    solve_plate_on_foundation,
    structured_square_mesh,
)
from scenarios.suboff_icebreaking.si2_fracture_gen.cohesive_failure import (
    identify_failed_faces,
    identify_failed_faces_uniform_stress,
    save_failed_faces,
    extract_internal_faces,
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


def test_anchors_1_and_2(output_dir: Path) -> tuple:
    """锚1+锚2：质量与断裂能量守恒。

    完整管线：
    1. 创建网格 + 施加载荷
    2. 求解板 (SI-1 基础设施)
    3. 计算应力 → 检测破坏面
    4. 生成颗粒
    5. 计算双账本
    6. 对账
    """
    print("=" * 60)
    print("锚1+锚2：质量与断裂能量守恒")
    print("=" * 60)

    # 参数 - 使用极低 cohesive 强度确保有破坏
    mesh_res = 4
    side = 1.0
    E = 1e7
    nu = 0.3
    t = 0.01
    rho = 917.0
    sigma_c = 1.0   # 极低强度，确保有破坏
    G_c = 100.0
    q0 = 1e5

    # 网格
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)
    n_nodes = len(nodes_2d)

    # 正弦载荷
    forces = np.zeros((n_nodes, 3))
    for i, (x, y) in enumerate(nodes_2d):
        forces[i, 2] = -q0 * np.sin(np.pi * x / side) * np.sin(np.pi * y / side)

    # 求解板（无地基）
    result = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, t, 0.0, forces, "simply_supported"
    )

    # 检测破坏
    failure_result = identify_failed_faces(
        nodes_2d, elements, result.displacement, E, nu, t, sigma_c, G_c
    )
    print(f"  总内部 face 数: {failure_result.n_total_faces}")
    print(f"  破坏 face 数: {failure_result.n_failed}")
    print(f"  σ_c = {sigma_c} Pa")

    if failure_result.n_failed == 0:
        print("  错误：仍无破坏面，测试参数有问题")
        raise ValueError("No failed faces detected")

    # 保存破坏面
    failed_faces_path = output_dir / "failed_faces.json"
    save_failed_faces(failure_result, str(failed_faces_path))

    # 加载破坏面（模拟文件传递）
    with open(failed_faces_path, "r", encoding="utf-8") as f:
        failed_faces_data = json.load(f)

    # 生成颗粒
    gen_result = generate_particles_from_failed_faces(
        failed_faces_data, nodes_2d, rho
    )
    print(f"  生成颗粒数: {len(gen_result.particles)}")
    print(f"  总质量 (DEM): {gen_result.total_mass:.6e} kg")
    print(f"  总断裂能 (DEM): {gen_result.total_fracture_energy:.6e} J")

    # 计算账本
    ledger_a = compute_fem_ledger(failed_faces_data, rho)
    ledger_b = compute_dem_ledger(gen_result)

    # 保存账本和颗粒
    save_ledger_a(ledger_a, str(output_dir / "ledger_a.json"))
    save_ledger_b(ledger_b, str(output_dir / "ledger_b.json"))
    save_particles(gen_result, str(output_dir / "particles.json"))

    # 对账
    report = reconcile_ledgers(ledger_a, ledger_b)
    save_reconciliation_report(report, str(output_dir / "consistency_report.json"))

    # 锚1：质量守恒
    mass_info = report["mass_conservation"]
    mass_passed = mass_info["passed"]
    print(f"\n  锚1 质量守恒:")
    print(f"    FEM 核销质量: {mass_info['ledger_a_mass']:.15e}")
    print(f"    DEM 颗粒质量: {mass_info['ledger_b_mass']:.15e}")
    print(f"    相对误差: {mass_info['relative_error']:.3e}")
    print(f"    判据: |Δm|/Σm ≤ 1e-12")
    print(f"    状态: {'PASS' if mass_passed else 'FAIL'}")

    # 锚2：断裂能量守恒
    energy_info = report["energy_conservation"]
    energy_passed = energy_info["passed"]
    print(f"\n  锚2 断裂能量守恒:")
    print(f"    FEM 释放能量: {energy_info['ledger_a_energy']:.15e}")
    print(f"    DEM 断裂能量: {energy_info['ledger_b_energy']:.15e}")
    print(f"    相对误差: {energy_info['relative_error']:.3e}")
    print(f"    判据: |ΔE|/ΣE ≤ 1e-12")
    print(f"    状态: {'PASS' if energy_passed else 'FAIL'}")
    print()

    mass_result = {
        "anchor": "锚1：质量守恒",
        "criterion": "|Δm|/Σm ≤ 1e-12",
        "value": float(mass_info["relative_error"]),
        "ledger_a_mass": float(mass_info["ledger_a_mass"]),
        "ledger_b_mass": float(mass_info["ledger_b_mass"]),
        "passed": bool(mass_passed),
        "source_file": str(output_dir / "consistency_report.json"),
        "source_key": "mass_conservation.relative_error",
    }

    energy_result = {
        "anchor": "锚2：断裂能量守恒",
        "criterion": "|ΔE|/ΣE ≤ 1e-12",
        "value": float(energy_info["relative_error"]),
        "ledger_a_energy": float(energy_info["ledger_a_energy"]),
        "ledger_b_energy": float(energy_info["ledger_b_energy"]),
        "passed": bool(energy_passed),
        "source_file": str(output_dir / "consistency_report.json"),
        "source_key": "energy_conservation.relative_error",
    }

    return mass_result, energy_result


def test_anchor_3(output_dir: Path) -> dict:
    """锚3：解析枚举锚。

    预设均匀应力场 σ_x = σ0 > σ_c, σ_y = τ_xy = 0。
    在结构化方网格上：
    - 垂直内部边（法向 ±x）：σ_n = σ_x = σ0 > σ_c → 破坏
    - 水平内部边（法向 ±y）：σ_n = σ_y = 0 < σ_c → 不破坏

    解析预测：
    - n_failed = (mesh_res - 1) × mesh_res（垂直内部边数）
    - 每条边长度 = side / mesh_res
    - 每个面面积 = (side / mesh_res) × thickness
    - 总面积 = n_failed × (side / mesh_res) × thickness
    - 总释放能 = n_failed × G_c × (side / mesh_res) × thickness
    """
    print("=" * 60)
    print("锚3：解析枚举锚")
    print("=" * 60)

    # 参数
    mesh_res = 3
    side = 1.0
    t = 0.01
    sigma_c = 1e5
    G_c = 100.0
    sigma0 = 2e5  # σ_x = 2e5 > σ_c = 1e5

    # 网格
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)

    # 均匀应力场
    uniform_stress = np.array([sigma0, 0.0, 0.0])

    # 检测破坏
    failure_result = identify_failed_faces_uniform_stress(
        nodes_2d, elements, uniform_stress, t, sigma_c, G_c
    )

    # 解析计算
    n_vertical_expected = (mesh_res - 1) * mesh_res  # 垂直内部边数
    n_horizontal_expected = mesh_res * (mesh_res - 1)  # 水平内部边数
    n_total_expected = n_vertical_expected + n_horizontal_expected
    n_failed_expected = n_vertical_expected  # 只有垂直边破坏

    edge_length = side / mesh_res
    face_area_expected = edge_length * t
    total_area_expected = n_failed_expected * face_area_expected
    total_energy_expected = n_failed_expected * G_c * face_area_expected

    # 实际结果
    n_failed_actual = failure_result.n_failed
    n_total_actual = failure_result.n_total_faces
    total_area_actual = sum(f.area for f in failure_result.failed_faces)
    total_energy_actual = sum(f.released_energy for f in failure_result.failed_faces)

    # 验证
    count_match = n_failed_actual == n_failed_expected
    total_count_match = n_total_actual == n_total_expected

    if total_area_expected > 0:
        area_rel_error = abs(total_area_actual - total_area_expected) / total_area_expected
    else:
        area_rel_error = 0.0

    if total_energy_expected > 0:
        energy_rel_error = abs(total_energy_actual - total_energy_expected) / total_energy_expected
    else:
        energy_rel_error = 0.0

    passed = count_match and area_rel_error <= 1e-12 and energy_rel_error <= 1e-12

    print(f"  网格: {mesh_res}×{mesh_res}, side={side}m, t={t}m")
    print(f"  均匀应力: σ_x={sigma0:.2e} Pa, σ_y=0, τ_xy=0")
    print(f"  cohesive 强度: σ_c={sigma_c:.2e} Pa")
    print(f"  断裂能密度: G_c={G_c:.2e} J/m²")
    print()
    print(f"  总内部 face 数: expected={n_total_expected}, actual={n_total_actual}, match={total_count_match}")
    print(f"  破坏 face 数: expected={n_failed_expected}, actual={n_failed_actual}, match={count_match}")
    print(f"  总面积: expected={total_area_expected:.15e}, actual={total_area_actual:.15e}")
    print(f"  总面积相对误差: {area_rel_error:.3e}")
    print(f"  总释放能: expected={total_energy_expected:.15e}, actual={total_energy_actual:.15e}")
    print(f"  总释放能相对误差: {energy_rel_error:.3e}")
    print(f"  判据: 计数精确相等, 面积/能量相对误差 ≤ 1e-12")
    print(f"  状态: {'PASS' if passed else 'FAIL'}")
    print()

    return {
        "anchor": "锚3：解析枚举锚",
        "criterion": "破坏 face 计数精确相等、总面积与总释放能 vs 闭式值 ≤1e-12",
        "mesh_res": mesh_res,
        "side": side,
        "thickness": t,
        "sigma_x": sigma0,
        "sigma_c": sigma_c,
        "G_c": G_c,
        "n_total_faces_expected": n_total_expected,
        "n_total_faces_actual": n_total_actual,
        "n_failed_expected": n_failed_expected,
        "n_failed_actual": n_failed_actual,
        "count_match": bool(count_match),
        "total_area_expected": float(total_area_expected),
        "total_area_actual": float(total_area_actual),
        "area_rel_error": float(area_rel_error),
        "total_energy_expected": float(total_energy_expected),
        "total_energy_actual": float(total_energy_actual),
        "energy_rel_error": float(energy_rel_error),
        "passed": bool(passed),
        "source_file": "analytical",
        "source_key": "closed_form",
    }


def main():
    """主函数。"""
    print("\n" + "=" * 60)
    print("SI-2 断裂生成验证")
    print("=" * 60)
    print(f"时间戳: {utcnow_z()}")
    print()

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent / "benchmarks" / "icebreaking" / "si2_fracture_gen_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行锚点测试
    results = []

    mass_result, energy_result = test_anchors_1_and_2(output_dir)
    results.append(mass_result)
    results.append(energy_result)

    anchor3_result = test_anchor_3(output_dir)
    results.append(anchor3_result)

    # 生成验证报告
    report = {
        "timestamp": utcnow_z(),
        "scenario": "SI-2: FEM→DEM 断裂生成",
        "description": "cohesive 破坏面→DEM 颗粒集生成管线验证",
        "anchors": results,
        "summary": {
            "total_anchors": len(results),
            "passed_anchors": sum(1 for r in results if r["passed"]),
            "failed_anchors": sum(1 for r in results if not r["passed"]),
        },
    }

    # 保存报告
    report_path = output_dir / "validation_report.json"

    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    report_python = convert_to_python_types(report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_python, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("验证报告摘要")
    print("=" * 60)
    print(f"总锚点数: {report['summary']['total_anchors']}")
    print(f"通过: {report['summary']['passed_anchors']}")
    print(f"失败: {report['summary']['failed_anchors']}")
    print(f"\n报告已保存: {report_path}")
    print()

    if report["summary"]["failed_anchors"] > 0:
        print("验证失败！")
        return 1
    else:
        print("验证通过！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
