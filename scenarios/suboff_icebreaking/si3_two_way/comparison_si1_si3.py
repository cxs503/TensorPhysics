"""SI-3 vs SI-1 对照分析。

比较双向耦合（SI-3）与单向耦合（SI-1）的结果差异，量化双向效应。
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加 SI-1 模块路径
si1_path = Path(__file__).parent.parent / "si1_one_way"
sys.path.insert(0, str(si1_path))

from lbm_load_extract import generate_synthetic_lbm_snapshot, extract_nodal_forces_from_lbm_snapshot
from plate_response import structured_square_mesh, solve_plate_on_foundation


def run_si1_baseline(output_dir: Path) -> dict:
    """运行 SI-1 单向耦合基线。"""
    print("\n[SI-1 基线] 运行单向耦合...")
    
    # 生成网格
    nodes, elements = structured_square_mesh(n=5, side=1.0)
    nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
    
    # 生成 LBM 快照
    snapshot_path = output_dir / "si1_lbm_snapshot.npz"
    generate_synthetic_lbm_snapshot(
        output_path=str(snapshot_path),
        plate_nodes=nodes_3d,
        pressure_magnitude=1000.0,
        seed=42
    )
    
    # 提取载荷
    area_per_node = np.ones(len(nodes)) * (1.0 / 5) ** 2  # tributary area
    result_forces = extract_nodal_forces_from_lbm_snapshot(
        str(snapshot_path), nodes_3d, area_per_node
    )
    forces = np.array(result_forces["forces"])  # (N, 3)
    
    # 求解板响应
    plate_result = solve_plate_on_foundation(
        nodes=nodes,
        elements=elements,
        young=2.1e11,
        poisson=0.3,
        thickness=0.01,
        foundation_coefficient=1e5,
        nodal_forces=forces,
        boundary_conditions="simply_supported",
    )
    
    # 提取挠度
    deflection = plate_result.displacement[0::3]  # 每3个DOF取第一个（w）
    
    result = {
        "max_deflection": float(np.max(np.abs(deflection))),
        "mean_deflection": float(np.mean(np.abs(deflection))),
        "center_deflection": float(deflection[len(deflection)//2]),
    }
    
    print(f"  最大挠度: {result['max_deflection']:.3e} m")
    print(f"  平均挠度: {result['mean_deflection']:.3e} m")
    
    return result


def run_si3_two_way(output_dir: Path) -> dict:
    """运行 SI-3 双向耦合。"""
    print("\n[SI-3 双向] 运行双向耦合...")
    
    # 生成网格
    nodes, elements = structured_square_mesh(n=5, side=1.0)
    nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
    
    # 生成 LBM 快照
    snapshot_path = output_dir / "si3_lbm_snapshot.npz"
    generate_synthetic_lbm_snapshot(
        output_path=str(snapshot_path),
        plate_nodes=nodes_3d,
        pressure_magnitude=1000.0,
        seed=42
    )
    
    # 板配置
    area_per_node = np.ones(len(nodes)) * (1.0 / 5) ** 2  # tributary area
    plate_config = {
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "area_per_node": area_per_node.tolist(),
        "young": 2.1e11,
        "poisson": 0.3,
        "thickness": 0.01,
        "foundation_coefficient": 1e5,
        "mass": 1000.0,
    }
    
    # 运行双向耦合
    from coupling_loop import run_two_way_coupling
    summary = run_two_way_coupling(
        lbm_snapshot_path=str(snapshot_path),
        plate_config=plate_config,
        n_coupling_steps=10,
        n_sub_steps=5,
        dt_coupling=0.001,
        output_dir=str(output_dir / "si3_run"),
    )
    
    # 读取结构历史（从保存的 JSON 文件）
    struct_history_path = output_dir / "si3_run" / "struct_history.json"
    with open(struct_history_path) as f:
        struct_history = json.load(f)
    
    # 提取最终挠度
    final_w = struct_history[-1]["w_mean"]
    max_w = max(abs(s["w_mean"]) for s in struct_history)
    
    result = {
        "max_deflection": float(max_w),
        "mean_deflection": float(abs(final_w)),
        "center_deflection": float(final_w),
    }
    
    print(f"  最大挠度: {result['max_deflection']:.3e} m")
    print(f"  平均挠度: {result['mean_deflection']:.3e} m")
    
    return result


def compare_si1_si3(si1_result: dict, si3_result: dict) -> dict:
    """比较 SI-1 与 SI-3 结果。"""
    print("\n[对照分析] SI-1 vs SI-3...")
    
    # 计算差异
    delta_max = abs(si3_result["max_deflection"] - si1_result["max_deflection"])
    delta_mean = abs(si3_result["mean_deflection"] - si1_result["mean_deflection"])
    
    # 相对差异
    rel_max = delta_max / max(abs(si1_result["max_deflection"]), 1e-20)
    rel_mean = delta_mean / max(abs(si1_result["mean_deflection"]), 1e-20)
    
    comparison = {
        "si1": si1_result,
        "si3": si3_result,
        "delta": {
            "max_deflection_m": float(delta_max),
            "mean_deflection_m": float(delta_mean),
        },
        "relative_delta": {
            "max_deflection": float(rel_max),
            "mean_deflection": float(rel_mean),
        },
    }
    
    print(f"  最大挠度差异: {delta_max:.3e} m ({rel_max*100:.2f}%)")
    print(f"  平均挠度差异: {delta_mean:.3e} m ({rel_mean*100:.2f}%)")
    
    # 物理解释
    if rel_max < 1e-6:
        explanation = "双向耦合效应可忽略（准静态小变形，流体载荷变化微小）"
    elif rel_max < 0.01:
        explanation = "双向耦合效应微弱（结构变形对流场反馈有限）"
    else:
        explanation = "双向耦合效应显著（结构变形显著改变流场分布）"
    
    comparison["physical_interpretation"] = explanation
    
    return comparison


def main():
    """主函数。"""
    print("=" * 60)
    print("SI-3 vs SI-1 对照分析")
    print("=" * 60)
    
    output_dir = Path("outputs/si3_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行 SI-1 基线
    si1_result = run_si1_baseline(output_dir)
    
    # 运行 SI-3 双向耦合
    si3_result = run_si3_two_way(output_dir)
    
    # 对照分析
    comparison = compare_si1_si3(si1_result, si3_result)
    
    # 保存报告
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 对照报告已保存: {report_path}")
    
    return comparison


if __name__ == "__main__":
    result = main()
    sys.exit(0)
