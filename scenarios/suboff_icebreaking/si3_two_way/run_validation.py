#!/usr/bin/env python3
"""SI-3 双向耦合验证脚本。

运行双向耦合管线，生成验证报告。
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加 SI-1 模块路径
si1_path = Path(__file__).parent.parent / "si1_one_way"
sys.path.insert(0, str(si1_path))

from lbm_load_extract import generate_synthetic_lbm_snapshot
from plate_response import structured_square_mesh
from coupling_loop import run_two_way_coupling
from conservation_ledger import load_conservation_ledger, aggregate_conservation_ledgers


def generate_validation_report(output_dir: Path, aggregation: dict) -> dict:
    """生成验证报告。
    
    Args:
        output_dir: 输出目录
        aggregation: 守恒账本聚合结果
    
    Returns:
        验证报告字典
    """
    # 判据（与 wedge S4 同标尺）
    momentum_threshold = 1e-12
    energy_threshold = 1e-12
    
    # 检查动量守恒
    momentum_passed = aggregation["momentum_max_residual"] <= momentum_threshold
    
    # 检查能量守恒
    energy_passed = aggregation["energy_max_residual"] <= energy_threshold
    
    # 总体判定
    all_passed = momentum_passed and energy_passed
    
    report = {
        "scenario": "SI-3: LBM×FEM 双向耦合",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "momentum_conservation": {
                "passed": momentum_passed,
                "value": aggregation["momentum_max_residual"],
                "threshold": momentum_threshold,
                "description": "动量守恒残差 ≤ 1e-12"
            },
            "energy_conservation": {
                "passed": energy_passed,
                "value": aggregation["energy_max_residual"],
                "threshold": energy_threshold,
                "description": "能量守恒残差 ≤ 1e-12"
            }
        },
        "summary": {
            "total_checks": 2,
            "passed_checks": int(momentum_passed) + int(energy_passed),
            "all_passed": all_passed
        },
        "details": {
            "n_coupling_steps": aggregation["n_coupling_steps"],
            "n_sub_steps": aggregation["n_sub_steps"],
            "momentum_residuals": {
                "max": aggregation["momentum_max_residual"],
                "mean": aggregation["momentum_mean_residual"]
            },
            "energy_residuals": {
                "max": aggregation["energy_max_residual"],
                "mean": aggregation["energy_mean_residual"]
            }
        }
    }
    
    # 保存报告
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def main():
    """主函数。"""
    print("=" * 60)
    print("SI-3: LBM×FEM 双向耦合验证")
    print("=" * 60)
    
    # 输出目录
    output_dir = Path("outputs/si3_two_way")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成合成 LBM 快照
    print("\n[1/4] 生成合成 LBM 快照...")
    nodes, elements = structured_square_mesh(n=5, side=1.0)
    snapshot_path = output_dir / "synthetic_lbm_snapshot.npz"
    # 将 2D 节点扩展为 3D (z=0)
    nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
    generate_synthetic_lbm_snapshot(
        output_path=str(snapshot_path),
        plate_nodes=nodes_3d,
        pressure_magnitude=1000.0,
        seed=42
    )
    print(f"  ✓ 快照已生成: {snapshot_path}")
    
    # 2. 准备板配置
    print("\n[2/4] 准备板配置...")
    plate_config = {
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "young": 2.1e11,  # 钢，Pa
        "poisson": 0.3,
        "thickness": 0.01,  # 1cm
        "foundation_coefficient": 1e6,  # Winkler 地基系数
        "mass": 7800 * 0.01 * 1.0 * 1.0,  # 密度 × 厚度 × 面积
        "area_per_node": (1.0 / 4) * np.ones(len(nodes))  # 每个节点代表面积
    }
    print(f"  ✓ 板配置: {len(nodes)} 节点, {len(elements)} 单元")
    
    # 3. 运行双向耦合
    print("\n[3/4] 运行双向耦合...")
    summary = run_two_way_coupling(
        lbm_snapshot_path=str(snapshot_path),
        plate_config=plate_config,
        n_coupling_steps=10,
        n_sub_steps=5,
        dt_coupling=0.001,
        output_dir=str(output_dir)
    )
    print(f"  ✓ 耦合完成")
    
    # 4. 生成验证报告
    print("\n[4/4] 生成验证报告...")
    
    # 加载所有守恒账本
    ledger_files = sorted(output_dir.glob("conservation_ledger_step*.json"))
    ledgers = [load_conservation_ledger(str(f)) for f in ledger_files]
    
    # 聚合统计
    aggregation = aggregate_conservation_ledgers(ledgers)
    aggregation["n_coupling_steps"] = 10
    aggregation["n_sub_steps"] = 5
    
    # 生成报告
    report = generate_validation_report(output_dir, aggregation)
    
    print(f"\n{'=' * 60}")
    print("验证报告")
    print("=" * 60)
    print(f"总检查数: {report['summary']['total_checks']}")
    print(f"通过数: {report['summary']['passed_checks']}")
    print(f"总体结果: {'✓ 通过' if report['summary']['all_passed'] else '✗ 失败'}")
    print(f"\n详细结果:")
    for check_name, check_data in report['checks'].items():
        status = "✓" if check_data['passed'] else "✗"
        print(f"  {status} {check_name}: {check_data['value']:.3e} (阈值: {check_data['threshold']:.0e})")
    
    print(f"\n报告已保存: {output_dir / 'validation_report.json'}")
    print("=" * 60)
    
    return 0 if report['summary']['all_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
