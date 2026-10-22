"""SI-3 双向耦合管线测试。

测试内容：
1. 守恒账本：动量守恒残差 ≤ 1e-12
2. 守恒账本：能量守恒残差 ≤ 1e-12
3. 双向数据交换：流体→结构 和 结构→流体 数据完整
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scenarios.suboff_icebreaking.si3_two_way.conservation_ledger import (
    compute_conservation_ledger,
)
from scenarios.suboff_icebreaking.si3_two_way.immersed_boundary import (
    compute_immersed_boundary_from_plate_deflection,
)


def test_momentum_conservation():
    """测试动量守恒：流体冲量 = 结构动量变化 + 反力冲量。"""
    # 模拟数据：稳态载荷
    fluid_force = np.array([0.0, 0.0, -1000.0])  # N
    reaction_force = -fluid_force  # 准静态平衡
    struct_displacement = np.array([0.0, 0.0, -1e-5])  # m
    struct_strain_energy = 0.5 * np.dot(fluid_force, struct_displacement)
    
    # 计算守恒账本
    ledger = compute_conservation_ledger(
        coupling_step=0,
        fluid_force=fluid_force,
        struct_displacement=struct_displacement,
        struct_strain_energy=struct_strain_energy,
        reaction_force=reaction_force,
        dt=0.001,
    )
    
    # 验证：动量守恒残差应为机器精度
    momentum_residual = ledger["momentum_balance"]["relative_residual"]
    assert momentum_residual <= 1e-12, f"动量守恒失败: residual = {momentum_residual}"


def test_energy_conservation():
    """测试能量守恒：应变能 = 0.5 * 外力功。"""
    # 模拟数据：线性弹性系统
    fluid_force = np.array([0.0, 0.0, -1000.0])  # N
    reaction_force = -fluid_force
    struct_displacement = np.array([0.0, 0.0, -1e-5])  # m
    struct_strain_energy = 0.5 * np.dot(fluid_force, struct_displacement)
    
    # 第一步（无增量信息）
    ledger = compute_conservation_ledger(
        coupling_step=0,
        fluid_force=fluid_force,
        struct_displacement=struct_displacement,
        struct_strain_energy=struct_strain_energy,
        reaction_force=reaction_force,
        dt=0.001,
    )
    
    # 第一步能量残差应为 0（无增量信息）
    energy_residual = ledger["energy_balance"]["relative_residual"]
    assert energy_residual == 0.0, f"第一步能量残差应为 0: {energy_residual}"
    
    # 第二步（有增量信息）
    fluid_force_prev = fluid_force * 0.5
    struct_displacement_prev = struct_displacement * 0.5
    struct_strain_energy_prev = 0.5 * np.dot(fluid_force_prev, struct_displacement_prev)
    
    ledger2 = compute_conservation_ledger(
        coupling_step=1,
        fluid_force=fluid_force,
        struct_displacement=struct_displacement,
        struct_strain_energy=struct_strain_energy,
        reaction_force=reaction_force,
        dt=0.001,
        fluid_force_prev=fluid_force_prev,
        struct_displacement_prev=struct_displacement_prev,
        struct_strain_energy_prev=struct_strain_energy_prev,
    )
    
    # 验证：能量守恒残差应为机器精度
    energy_residual2 = ledger2["energy_balance"]["relative_residual"]
    assert energy_residual2 <= 1e-12, f"能量守恒失败: residual = {energy_residual2}"


def test_immersed_boundary():
    """测试浸入边界计算：结构变形→流场边界修正。"""
    # 生成简单板网格
    n_nodes = 25
    plate_nodes = np.zeros((n_nodes, 3))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            plate_nodes[idx, 0] = i * 0.25
            plate_nodes[idx, 1] = j * 0.25
            plate_nodes[idx, 2] = 0.0
    
    # 模拟板挠度（中心下沉）
    plate_deflection = np.zeros(n_nodes)
    center_idx = 12  # 5x5 网格中心
    plate_deflection[center_idx] = -1e-5
    
    # 模拟板速度
    plate_velocity = np.zeros(n_nodes)
    
    # 流体节点（板上方）
    fluid_nodes = plate_nodes.copy()
    fluid_nodes[:, 2] = 0.1  # 板上方 0.1m
    
    # 计算浸入边界
    dt = 0.001
    ib_data = compute_immersed_boundary_from_plate_deflection(
        plate_nodes=plate_nodes[:, :2],  # 只传 2D
        plate_deflection=plate_deflection,
        plate_velocity=plate_velocity,
        fluid_nodes=fluid_nodes,
        coupling_step=0,
        dt=dt,
    )
    
    # 验证：返回数据结构完整
    assert "boundary_nodes" in ib_data
    assert "boundary_velocity" in ib_data
    assert "boundary_displacement" in ib_data
    
    # 验证：边界节点包含挠度信息
    boundary_nodes = np.array(ib_data["boundary_nodes"])
    assert boundary_nodes.shape == (n_nodes, 3)
    
    # 中心节点应有最大挠度
    center_deflection = abs(boundary_nodes[center_idx, 2])
    corner_deflection = abs(boundary_nodes[0, 2])
    assert center_deflection > corner_deflection, "中心挠度应大于角点"


def test_two_way_coupling_integration():
    """集成测试：运行完整双向耦合循环。"""
    from scenarios.suboff_icebreaking.si3_two_way.coupling_loop import (
        run_two_way_coupling,
    )
    from scenarios.suboff_icebreaking.si1_one_way.lbm_load_extract import (
        generate_synthetic_lbm_snapshot,
    )
    from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
        structured_square_mesh,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 生成网格
        nodes, elements = structured_square_mesh(n=5, side=1.0)
        nodes_3d = np.column_stack([nodes, np.zeros(len(nodes))])
        
        # 生成 LBM 快照
        snapshot_path = output_dir / "test_snapshot.npz"
        generate_synthetic_lbm_snapshot(
            output_path=str(snapshot_path),
            plate_nodes=nodes_3d,
            pressure_magnitude=1000.0,
            seed=42,
        )
        
        # 板配置
        area_per_node = np.ones(len(nodes)) * (1.0 / 5) ** 2
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
        
        # 运行双向耦合（3步快速测试）
        summary = run_two_way_coupling(
            lbm_snapshot_path=str(snapshot_path),
            plate_config=plate_config,
            n_coupling_steps=3,
            n_sub_steps=2,
            dt_coupling=0.001,
            output_dir=str(output_dir / "run"),
        )
        
        # 验证：守恒账本通过
        assert summary["conservation_summary"]["all_passed"], "守恒账本未通过"
        
        # 验证：动量残差 ≤ 1e-12
        momentum_residual = summary["conservation_summary"]["momentum_max_residual"]
        assert momentum_residual <= 1e-12, f"动量残差过大: {momentum_residual}"
        
        # 验证：能量残差 ≤ 1e-12
        energy_residual = summary["conservation_summary"]["energy_max_residual"]
        assert energy_residual <= 1e-12, f"能量残差过大: {energy_residual}"
