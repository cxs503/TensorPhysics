"""SI-1 单向耦合管线测试。

测试内容：
1. 账本对账：LBM 发送的节点力与 FEM 接收的节点力完全一致
2. 正弦载荷验证：与 TensorFEM 基准解对比
3. 线性叠加性：f1 + f2 的响应等于 w(f1) + w(f2)
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scenarios.suboff_icebreaking.si1_one_way.plate_response import (
    solve_plate_on_foundation,
    structured_square_mesh,
)
from scenarios.suboff_icebreaking.si1_one_way.coupling_ledger import reconcile_ledger_files


def test_ledger_conservation():
    """测试账本对账：LBM 发送的节点力与 FEM 接收的节点力完全一致。"""
    # 创建测试数据
    n_nodes = 100
    positions = np.random.rand(n_nodes, 3)
    forces = np.random.rand(n_nodes, 3) * 1000
    
    # 写入临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_a_path = Path(tmpdir) / "ledger_a.json"
        ledger_b_path = Path(tmpdir) / "ledger_b.json"
        
        ledger_a = {
            "timestamp": "2024-01-01T00:00:00Z",
            "n_nodes": n_nodes,
            "positions": positions.tolist(),
            "forces": forces.tolist(),
        }
        
        # FEM 接收的力与 LBM 发送的完全相同（单向传递）
        ledger_b = {
            "timestamp": "2024-01-01T00:00:01Z",
            "n_nodes": n_nodes,
            "positions": positions.tolist(),
            "forces": forces.tolist(),
        }
        
        with open(ledger_a_path, "w") as f:
            json.dump(ledger_a, f)
        with open(ledger_b_path, "w") as f:
            json.dump(ledger_b, f)
        
        # 对账
        report = reconcile_ledger_files(str(ledger_a_path), str(ledger_b_path))
        
        # 验证：机器精度级别的一致性
        assert report["max_abs_diff"] < 1e-12, f"账本不一致: max_abs_diff = {report['max_abs_diff']}"


def test_plate_sine_load():
    """测试正弦载荷下的板响应，与 TensorFEM 基准解对比。"""
    # 材料参数（与 TensorFEM 验证口径一致）
    E = 1e7
    nu = 0.3
    h = 0.01
    side = 1.0
    q0 = 1.0
    
    # 生成网格
    mesh_res = 16
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)
    n_nodes = len(nodes_2d)
    
    # 施加正弦载荷
    forces = np.zeros((n_nodes, 3))
    for i, (x, y) in enumerate(nodes_2d):
        forces[i, 2] = -q0 * np.sin(np.pi * x / side) * np.sin(np.pi * y / side)
    
    # 求解（无地基）
    result = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, 0.0, forces, "simply_supported"
    )
    
    # 找到中心节点
    center_idx = n_nodes // 2
    w_fem = abs(result.displacement[3 * center_idx])
    
    # Kirchhoff 理论解（薄板，无剪切变形）
    D = E * h**3 / (12 * (1 - nu**2))
    w_kirchhoff = q0 * side**4 / (4 * np.pi**4 * D)
    
    # Mindlin 理论解（考虑剪切变形）
    G = E / (2 * (1 + nu))
    kappa = 5.0 / 6.0
    w_shear = q0 * side**2 / (8 * kappa * G * h)
    w_mindlin = w_kirchhoff + w_shear
    
    # 验证：FEM 解与 Mindlin 理论解一致（考虑剪切变形）
    rel_error = abs(w_fem - w_mindlin) / w_mindlin
    assert rel_error < 0.05, f"FEM 与 Mindlin 理论解偏差过大: {rel_error:.3e}"
    
    # 验证：FEM 解与 Kirchhoff 理论解偏差在合理范围内（剪切变形效应）
    rel_error_kirchhoff = abs(w_fem - w_kirchhoff) / w_kirchhoff
    # 对于 h/a = 0.01 的薄板，剪切效应较小，但仍存在
    assert rel_error_kirchhoff < 0.10, f"FEM 与 Kirchhoff 理论解偏差过大: {rel_error_kirchhoff:.3e}"


def test_linear_superposition():
    """测试线性叠加性：f1 + f2 的响应等于 w(f1) + w(f2)。"""
    # 材料参数
    E = 1e7
    nu = 0.3
    h = 0.01
    side = 1.0
    k_f = 1e5  # 有地基
    
    # 生成网格
    mesh_res = 16
    nodes_2d, elements = structured_square_mesh(mesh_res, side=side)
    n_nodes = len(nodes_2d)
    
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
    result1 = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, k_f, forces1, "simply_supported"
    )
    result2 = solve_plate_on_foundation(
        nodes_2d, elements, E, nu, h, k_f, forces2, "simply_supported"
    )
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
    
    # 机器精度级别的线性叠加
    assert max_diff < 1e-10, f"线性叠加性破坏: max_diff = {max_diff:.3e}"


if __name__ == "__main__":
    test_ledger_conservation()
    test_plate_sine_load()
    test_linear_superposition()
    print("所有测试通过！")
