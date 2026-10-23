"""SI-4 全链演示运行脚本。

运行 LBM⇄FEM 双向 × cohesive 断裂 × DEM 键结颗粒四库全链，
生成论文级演示 artifacts。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scenarios.suboff_icebreaking.si4_full_chain.full_chain_orchestrator import (
    run_full_chain,
)
from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    structured_square_mesh,
)
from scenarios.suboff_icebreaking.si1_one_way.lbm_load_extract import (
    generate_synthetic_lbm_snapshot,
)


def main():
    """运行 SI-4 全链演示。"""
    print("=" * 80)
    print("SI-4: 全链演示——LBM⇄FEM 双向 × cohesive 断裂 × DEM 键结颗粒")
    print("=" * 80)
    
    # 输出目录
    output_dir = project_root / "outputs" / "si4_full_chain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成板网格
    print("\n[1/3] 生成板网格...")
    nodes, elements = structured_square_mesh(n=4, side=1.0)
    print(f"  节点数: {len(nodes)}")
    print(f"  单元数: {len(elements)}")
    
    # 生成 LBM 快照
    print("\n[2/3] 生成 LBM 流场快照...")
    nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
    snapshot_path = output_dir / "lbm_snapshot_initial.npz"
    generate_synthetic_lbm_snapshot(
        output_path=str(snapshot_path),
        plate_nodes=nodes_3d,
        pressure_magnitude=5000.0,  # 增大压力以触发断裂
        seed=42,
    )
    print(f"  快照路径: {snapshot_path}")
    
    # 计算节点面积
    side = 1.0
    n = 4
    dx = side / n
    area_per_node = np.full(len(nodes), dx**2)
    
    # 板配置
    plate_config = {
        "nodes": nodes.tolist(),
        "elements": elements.tolist(),
        "area_per_node": area_per_node.tolist(),
        "young": 2.1e11,      # 钢的杨氏模量 (Pa)
        "poisson": 0.3,        # 泊松比
        "thickness": 0.01,     # 板厚 (m)
    }
    
    # 运行全链
    print("\n[3/3] 运行全链演示...")
    summary = run_full_chain(
        lbm_snapshot_path=str(snapshot_path),
        plate_config=plate_config,
        n_chain_steps=5,        # 5 个全链步
        n_dem_substeps=3,       # 每步 3 个 DEM 子步
        dt_coupling=0.001,      # 耦合时间步 (s)
        dt_dem=1e-5,            # DEM 时间步 (s)
        sigma_c=5e5,            # cohesive 强度 (Pa) - 匹配应力水平以触发断裂
        G_c=100.0,              # 断裂能密度 (J/m²)
        density=917.0,          # 冰密度 (kg/m³)
        output_dir=str(output_dir),
    )
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("全链演示摘要")
    print("=" * 80)
    print(f"全链步数: {summary['n_chain_steps']}")
    print(f"DEM 子步数/全链步: {summary['n_dem_substeps']}")
    print(f"\n最终状态:")
    print(f"  颗粒数: {summary['final_state']['n_particles']}")
    print(f"  键结数: {summary['final_state']['n_bonds']}")
    print(f"  断裂键结: {summary['final_state']['n_bonds_broken']}")
    print(f"  总质量: {summary['final_state']['total_mass_kg']:.6e} kg")
    print(f"  总能量: {summary['final_state']['total_energy_J']:.6e} J")
    
    print("\n双账本门检查:")
    all_passed = True
    for k, ledger in enumerate(summary['conservation_ledgers']):
        mass_passed = ledger['full_chain_conservation']['mass_passed']
        energy_passed = ledger['full_chain_conservation']['energy_passed']
        si3_passed = ledger['si3_coupling']['passed']
        si2_passed = ledger['si2_fracture']['passed']
        step_passed = ledger['all_gates_passed']
        
        status = "✓ PASS" if step_passed else "✗ FAIL"
        print(f"  步 {k+1}: {status}")
        print(f"    SI-3: {'✓' if si3_passed else '✗'}  "
              f"SI-2: {'✓' if si2_passed else '✗'}  "
              f"质量: {'✓' if mass_passed else '✗'}  "
              f"能量: {'✓' if energy_passed else '✗'}")
        
        if not step_passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 全链演示成功：所有账本门通过")
    else:
        print("✗ 全链演示失败：部分账本门未通过")
    print("=" * 80)
    
    print(f"\n输出目录: {output_dir}")
    print(f"  - full_chain_summary.json (全链摘要)")
    print(f"  - lbm_snapshot_step*.npz (LBM 快照时序)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
