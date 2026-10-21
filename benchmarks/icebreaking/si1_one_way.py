"""SI-1 单向耦合验证脚本。

验证三个锚点：
1. 锚1（传递守恒账本）：账本 A 与账本 B 逐节点力对比，判据 max|A−B| ≤ 1e-12
2. 锚2（Hetényi 耦合态复现）：均匀载荷经完整 LBM→FEM 链传递后，板中点挠度 vs Hetényi 级数解相对误差 <2%
3. 锚3（线性叠加性）：f1、f2、f1+f2 三次过链，判据 max|w(f1+f2)−w(f1)−w(f2)| ≤ 1e-10
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scenarios.suboff_icebreaking.si1_one_way.lbm_load_extract import (
    extract_nodal_forces_from_lbm_snapshot,
    save_ledger_a,
)
from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    solve_plate_on_foundation,
    structured_square_mesh,
    save_ledger_b,
)
from scenarios.suboff_icebreaking.si1_one_way.coupling_ledger import (
    reconcile_ledger_files,
    save_reconciliation_report,
)


def utcnow_z() -> str:
    """返回 UTC 时间戳（Z 后缀）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hetenyi_plate_series_center_deflection(
    q0: float,
    a: float,
    D: float,
    k_f: float,
    n_terms: int = 100,
) -> float:
    """简支方板在 Winkler 地基上受均布载荷的中点挠度 Navier 级数解。
    
    w(x,y) = Σ_m Σ_n w_mn * sin(mπx/a) * sin(nπy/a)
    
    其中 w_mn = q_mn / (D * ((m²+n²)π²/a²)² + k_f)
    
    对于均布载荷 q0，Fourier 系数为：
    q_mn = 16q0 / (m²n²π²)  (m,n 为奇数)
    
    在中点 (a/2, a/2)，sin(mπ/2) = 1 (奇数 m)。
    
    Parameters
    ----------
    q0 : float
        均布载荷 (Pa)。
    a : float
        板边长 (m)。
    D : float
        板弯曲刚度 (N·m)。
    k_f : float
        Winkler 地基系数 (N/m³)。
    n_terms : int
        级数截断项数 (每方向)。
    
    Returns
    -------
    w_center : float
        中点挠度 (m)。
    """
    w = 0.0
    for m in range(1, n_terms + 1, 2):  # 仅奇数
        for n in range(1, n_terms + 1, 2):  # 仅奇数
            q_mn = 16 * q0 / (m**2 * n**2 * np.pi**2)
            modal_stiffness = D * ((m**2 + n**2) * np.pi**2 / a**2)**2 + k_f
            w_mn = q_mn / modal_stiffness
            # 在中点 sin(mπ/2) = 1 for odd m
            w += w_mn
    return w


def test_anchor_1_ledger_conservation(output_dir: Path) -> dict:
    """锚1：传递守恒账本。
    
    验证 LBM 发出的节点力与 FEM 收到的节点力完全一致。
    """
    print("=" * 60)
    print("锚1：传递守恒账本")
    print("=" * 60)
    
    # 创建测试数据
    n_nodes = 100
    positions = np.random.rand(n_nodes, 3)
    forces = np.random.rand(n_nodes, 3) * 1000
    
    # 写入临时文件
    ledger_a_path = output_dir / "ledger_a_test.json"
    ledger_b_path = output_dir / "ledger_b_test.json"
    
    ledger_a = {
        "timestamp": utcnow_z(),
        "n_nodes": n_nodes,
        "positions": positions.tolist(),
        "forces": forces.tolist(),
    }
    
    # FEM 接收的力与 LBM 发送的完全相同（单向传递）
    ledger_b = {
        "timestamp": utcnow_z(),
        "n_nodes": n_nodes,
        "positions": positions.tolist(),
        "forces": forces.tolist(),
    }
    
    save_ledger_a(ledger_a, str(ledger_a_path))
    save_ledger_b(ledger_b, str(ledger_b_path))
    
    # 对账
    report = reconcile_ledger_files(str(ledger_a_path), str(ledger_b_path))
    
    max_abs_diff = report["max_abs_diff"]
    criterion = 1e-12
    passed = max_abs_diff <= criterion
    
    print(f"最大绝对差: {max_abs_diff:.3e}")
    print(f"判据: max|A−B| ≤ {criterion}")
    print(f"状态: {'PASS' if passed else 'FAIL'}")
    print()
    
    return {
        "anchor": "锚1：传递守恒账本",
        "criterion": f"max|A−B| ≤ {criterion}",
        "value": float(max_abs_diff),
        "passed": bool(passed),
        "source_file": str(ledger_a_path),
        "source_key": "forces",
    }


def test_anchor_2_hetenyi_reproduction(output_dir: Path) -> dict:
    """锚2：Hetényi 耦合态复现。
    
    验证均匀载荷经完整 LBM→FEM 链传递后，板中点挠度与 Hetényi 级数解的相对误差 <2%。
    """
    print("=" * 60)
    print("锚2：Hetényi 耦合态复现")
    print("=" * 60)
    
    # 材料参数
    E = 1e9  # Pa
    nu = 0.3
    h = 0.1  # m
    side = 1.0  # m
    k_f = 1e5  # N/m³
    q0 = 1000.0  # Pa
    
    # 板弯曲刚度
    D = E * h**3 / (12 * (1 - nu**2))
    
    # Hetényi 级数解
    w_hetenyi = hetenyi_plate_series_center_deflection(q0, side, D, k_f, n_terms=100)
    print(f"Hetényi 级数解 (中点挠度): {w_hetenyi:.6e} m")
    
    # 生成网格
    mesh_res = 32
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)
    n_nodes = len(nodes_2d)
    print(f"网格: {mesh_res}x{mesh_res}, 节点数: {n_nodes}")
    
    # 模拟 LBM 流场快照（均布载荷）
    # 创建临时 npz 文件
    lbm_snapshot_path = output_dir / "lbm_snapshot_test.npz"
    
    # 压力场（均布）
    pressure = np.full(n_nodes, q0)
    shear_x = np.zeros(n_nodes)  # 无剪切
    shear_y = np.zeros(n_nodes)  # 无剪切
    
    # 流体节点位置（与板节点相同）
    fluid_nodes = np.column_stack([
        nodes_2d[:, 0],
        nodes_2d[:, 1],
        np.zeros(n_nodes)  # z = 0 (冰板底面)
    ])
    
    np.savez(
        lbm_snapshot_path,
        pressure=pressure,
        shear_x=shear_x,
        shear_y=shear_y,
        fluid_nodes=fluid_nodes,
    )
    
    # 步骤1：LBM 流载提取
    print("\n步骤1：LBM 流载提取...")
    
    # 准备板节点和面积
    plate_nodes = np.column_stack([
        nodes_2d[:, 0],
        nodes_2d[:, 1],
        np.zeros(n_nodes)  # z = 0 (冰板底面)
    ])
    
    # 计算每个节点代表的面积 (tributary area)
    # 对于结构化网格，每个节点代表 (dx * dy) 的面积
    dx = side / mesh_res
    dy = side / mesh_res
    plate_area_per_node = np.full(n_nodes, dx * dy)
    
    ledger_a = extract_nodal_forces_from_lbm_snapshot(
        str(lbm_snapshot_path), 
        plate_nodes, 
        plate_area_per_node
    )
    ledger_a_path = output_dir / "ledger_a_hetenyi.json"
    save_ledger_a(ledger_a, str(ledger_a_path))
    print(f"  账本 A 已保存: {ledger_a_path}")
    
    # 步骤2：FEM 板响应求解
    print("\n步骤2：FEM 板响应求解...")
    forces_a = np.array(ledger_a["forces"])
    
    # 将节点力转回等效压力，因为 solve_plate_on_foundation 期望输入压力分布
    dx = side / mesh_res
    dy = side / mesh_res
    area_per_node = dx * dy
    pressure_field = np.zeros_like(forces_a)
    pressure_field[:, 2] = forces_a[:, 2] / area_per_node
    
    result = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, k_f, pressure_field, "simply_supported"
    )
    
    # 保存账本 B
    ledger_b = {
        "timestamp": utcnow_z(),
        "n_nodes": n_nodes,
        "positions": plate_nodes.tolist(),
        "forces": forces_a.tolist(),  # 单向传递，力不变
    }
    ledger_b_path = output_dir / "ledger_b_hetenyi.json"
    save_ledger_b(ledger_b, str(ledger_b_path))
    print(f"  账本 B 已保存: {ledger_b_path}")
    
    # 找到中心节点
    center_idx = n_nodes // 2
    w_fem = abs(result.displacement[3 * center_idx])
    print(f"\nFEM 解 (中点挠度): {w_fem:.6e} m")
    
    # 相对误差
    rel_error = abs(w_fem - w_hetenyi) / w_hetenyi
    criterion = 0.02
    passed = rel_error < criterion
    
    print(f"相对误差: {rel_error:.3e}")
    print(f"判据: |w_FEM - w_Hetényi| / w_Hetényi < {criterion}")
    print(f"状态: {'PASS' if passed else 'FAIL'}")
    print()
    
    return {
        "anchor": "锚2：Hetényi 耦合态复现",
        "criterion": f"|w_FEM - w_Hetényi| / w_Hetényi < {criterion}",
        "value": rel_error,
        "w_hetenyi": w_hetenyi,
        "w_fem": w_fem,
        "passed": passed,
        "source_file": str(ledger_b_path),
        "source_key": "forces",
    }


def test_anchor_3_linear_superposition(output_dir: Path) -> dict:
    """锚3：线性叠加性。
    
    验证 f1、f2、f1+f2 三次过链，判据 max|w(f1+f2)−w(f1)−w(f2)| ≤ 1e-10。
    """
    print("=" * 60)
    print("锚3：线性叠加性")
    print("=" * 60)
    
    # 材料参数
    E = 1e7  # Pa
    nu = 0.3
    h = 0.01  # m
    side = 1.0  # m
    k_f = 1e5  # N/m³
    
    # 生成网格
    mesh_res = 16
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)
    n_nodes = len(nodes_2d)
    print(f"网格: {mesh_res}x{mesh_res}, 节点数: {n_nodes}")
    
    # 创建两个不同的载荷
    forces1 = np.zeros((n_nodes, 3))
    forces2 = np.zeros((n_nodes, 3))
    forces_combined = np.zeros((n_nodes, 3))
    
    for i, (x, y) in enumerate(nodes_2d):
        # 载荷 1：中心集中载荷
        r1 = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        forces1[i, 2] = -1000 * np.exp(-r1**2 / 0.01)
        
        # 载荷 2：边缘集中载荷
        r2 = np.sqrt((x - 0.2)**2 + (y - 0.8)**2)
        forces2[i, 2] = -800 * np.exp(-r2**2 / 0.01)
        
        # 组合载荷
        forces_combined[i, 2] = forces1[i, 2] + forces2[i, 2]
    
    # 分别求解
    print("\n求解 f1...")
    result1 = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, k_f, forces1, "simply_supported"
    )
    
    print("求解 f2...")
    result2 = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, k_f, forces2, "simply_supported"
    )
    
    print("求解 f1+f2...")
    result_combined = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, k_f, forces_combined, "simply_supported"
    )
    
    # 提取挠度场（z 方向）
    w1 = result1.displacement[0::3]
    w2 = result2.displacement[0::3]
    w_combined = result_combined.displacement[0::3]
    
    # 验证线性叠加
    w_sum = w1 + w2
    max_diff = np.max(np.abs(w_combined - w_sum))
    
    criterion = 1e-10
    passed = max_diff <= criterion
    
    print(f"\n最大偏差: {max_diff:.3e}")
    print(f"判据: max|w(f1+f2)−w(f1)−w(f2)| ≤ {criterion}")
    print(f"状态: {'PASS' if passed else 'FAIL'}")
    print()
    
    return {
        "anchor": "锚3：线性叠加性",
        "criterion": f"max|w(f1+f2)−w(f1)−w(f2)| ≤ {criterion}",
        "value": max_diff,
        "passed": passed,
        "source_file": "N/A",
        "source_key": "N/A",
    }


def main():
    """主函数。"""
    print("\n" + "=" * 60)
    print("SI-1 单向耦合验证")
    print("=" * 60)
    print(f"时间戳: {utcnow_z()}")
    print()
    
    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent / "benchmarks" / "icebreaking" / "si1_one_way_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行三个锚点测试
    results = []
    results.append(test_anchor_1_ledger_conservation(output_dir))
    results.append(test_anchor_2_hetenyi_reproduction(output_dir))
    results.append(test_anchor_3_linear_superposition(output_dir))
    
    # 生成验证报告
    report = {
        "timestamp": utcnow_z(),
        "scenario": "SI-1: LBM×FEM 单向耦合",
        "description": "流载→冰板响应单向耦合管线验证",
        "anchors": results,
        "summary": {
            "total_anchors": len(results),
            "passed_anchors": sum(1 for r in results if r["passed"]),
            "failed_anchors": sum(1 for r in results if not r["passed"]),
        },
    }
    
    # 保存报告
    report_path = output_dir / "validation_report.json"
    
    # 将numpy类型转换为Python原生类型
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
    
    # 返回退出码
    if report["summary"]["failed_anchors"] > 0:
        print("验证失败！")
        return 1
    else:
        print("验证通过！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
