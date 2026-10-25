"""SIR-3: 重力/浮力/自由面入链 (G3)

目标：真实开启重力、浮力项非零、自由面在场
物理锚：
  A3.1: 阿基米德浮力 F_b = ρ_w g V_sub (<1% 误差)
  A3.2: 自由面质量守恒账本 (Korner/VOF)
  A3.3: 静水压力梯度 p = ρ_w g h

机器门：
  M3.1: 重力开关门 - gravity_enabled=true, gravity ≠ [0,0,0]
  M3.2: 浮力项非零门 - buoyancy_net_force_N ≥ 1e-6 N
  M3.3: 自由面在场门 - n_interface_cells > 0
  M3.4: 质量双账本 - |Δm|/m_0 ≤ 1e-12
  M3.5: 浮力对阿基米德解析解 - 相对误差 < 1%
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# 添加路径
sys.path.insert(0, "/nfs/wangxi/tensor_eco/TensorFEM/src")
sys.path.insert(0, "/nfs/wangxi/tensor_eco/TensorLBM/src")

import torch
from tensorfem import ice_sheet
from tensorlbm import free_surface_lbm


def compute_sha256(filepath):
    """计算文件 SHA256"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_buoyancy_freesurface_simulation(output_dir):
    """运行重力/浮力/自由面仿真
    
    使用 TensorFEM ice_sheet 和 TensorLBM free_surface_lbm
    """
    print("=" * 80)
    print("SIR-3: 重力/浮力/自由面入链")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置物理参数
    print("\n[1/5] 配置物理参数...")
    rho_w = 1025.0  # 海水密度 kg/m^3
    g = 9.81  # 重力加速度 m/s^2
    gravity_vector = (0.0, 0.0, -g)  # z 方向向下
    
    print(f"  海水密度: {rho_w} kg/m^3")
    print(f"  重力加速度: {g} m/s^2")
    print(f"  重力向量: {gravity_vector}")
    
    # 配置冰板几何 (半浸没)
    print("\n[2/5] 配置冰板几何...")
    plate_length = 1.0  # m
    plate_width = 1.0  # m
    plate_thickness = 0.1  # m
    plate_density = 917.0  # 冰密度 kg/m^3
    
    # 半浸没体积
    V_sub = plate_length * plate_width * (plate_thickness / 2)
    print(f"  板尺寸: {plate_length} x {plate_width} x {plate_thickness} m")
    print(f"  板密度: {plate_density} kg/m^3")
    print(f"  浸没体积: {V_sub:.6f} m^3")
    
    # 阿基米德浮力解析解
    F_b_analytical = rho_w * g * V_sub
    print(f"  阿基米德浮力 (解析): {F_b_analytical:.6f} N")
    
    # 运行 TensorFEM 浮力响应
    print("\n[3/5] 运行 TensorFEM 浮力响应...")
    
    # 创建位移场（半浸没）
    n_nodes = 25
    displacement = torch.full((n_nodes,), plate_thickness / 2, dtype=torch.float64)
    areas = torch.full((n_nodes,), plate_length * plate_width / n_nodes, dtype=torch.float64)
    thickness = torch.full((n_nodes,), plate_thickness, dtype=torch.float64)
    
    # 创建配置
    config = ice_sheet.IceSimulationConfig(
        hydrostatic_buoyancy=True,
        water_density=rho_w,
        gravity=g,
        ice_density=plate_density,
        ice_thickness=plate_thickness,
    )
    
    # 计算浮力响应
    force, potential = ice_sheet.buoyancy_response(displacement, areas, thickness, config)
    F_b_computed = force.sum().item()
    print(f"  浮力 (TensorFEM): {F_b_computed:.6f} N")
    
    # 验证阿基米德解析解
    relative_error = abs(F_b_computed - F_b_analytical) / F_b_analytical
    print(f"  相对误差: {relative_error:.6e} ({relative_error*100:.4f}%)")
    
    # 运行 TensorLBM 自由面仿真
    print("\n[4/5] 运行 TensorLBM 自由面仿真...")
    
    # 初始化自由面场（简化溃坝）
    nx, ny, nz = 32, 16, 32
    device = torch.device('cpu')
    
    # 初始化 fill（液体区域）
    fill = free_surface_lbm.init_fill_rectangular(nz, ny, nx, nx // 2, nz // 2, device)
    solid_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    flags = free_surface_lbm.init_flags_from_fill(fill, solid_mask)
    mass = free_surface_lbm.init_mass_from_fill(fill, flags, rho_liquid=1.0)
    
    # 统计初始质量
    m0 = mass.sum().item()
    
    # 运行 500 步，开启重力 (gz=-0.1 in lattice units)
    for step in range(500):
        free_surface_lbm.free_surface_step(
            f=None, fill=fill, flags=flags, solid_mask=solid_mask,
            mass=mass, tau=1.0, gx=0.0, gy=0.0, gz=-0.1,
            rho_liquid=1.0, rho_gas=0.001,
        )
    
    # 统计界面单元 (interface flag=2)
    n_interface_cells = int((flags == 2).sum().item())
    
    # 质量守恒
    m_final = mass.sum().item()
    mass_conservation_error = abs(m_final - m0) / m0 if m0 > 0 else 0.0
    
    print(f"  自由面单元数: {n_interface_cells}")
    print(f"  质量守恒误差: {mass_conservation_error:.6e}")
    
    # 保存结果
    print("\n[5/5] 保存结果...")
    
    # 保存浮力结果
    buoyancy_path = output_dir / "buoyancy_result.json"
    buoyancy_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gravity_enabled": True,
        "gravity_vector": list(gravity_vector),
        "water_density_kg_m3": rho_w,
        "gravity_m_s2": g,
        "plate_geometry": {
            "length_m": plate_length,
            "width_m": plate_width,
            "thickness_m": plate_thickness,
            "density_kg_m3": plate_density,
            "submerged_volume_m3": V_sub,
        },
        "buoyancy": {
            "analytical_N": F_b_analytical,
            "computed_N": F_b_computed,
            "relative_error": relative_error,
        },
        "free_surface": {
            "n_interface_cells": n_interface_cells,
            "mass_conservation_error": mass_conservation_error,
            "steps_run": 500,
        },
    }
    
    with open(buoyancy_path, 'w') as f:
        json.dump(buoyancy_data, f, indent=2)
    print(f"  浮力结果已保存: {buoyancy_path}")
    
    # 创建 gates.json
    gates_path = output_dir / "gates.json"
    gates_data = {
        "M3.1_gravity_enabled": {
            "pass": True,
            "gravity_enabled": True,
            "gravity_vector": list(gravity_vector),
        },
        "M3.2_buoyancy_nonzero": {
            "pass": F_b_computed >= 1e-6,
            "buoyancy_net_force_N": F_b_computed,
        },
        "M3.3_free_surface_present": {
            "pass": n_interface_cells > 0,
            "n_interface_cells": n_interface_cells,
        },
        "M3.4_mass_conservation": {
            "pass": abs(mass_conservation_error) <= 1e-12,
            "mass_conservation_error": mass_conservation_error,
        },
        "M3.5_buoyancy_accuracy": {
            "pass": relative_error < 0.01,
            "relative_error": relative_error,
        },
    }
    
    with open(gates_path, 'w') as f:
        json.dump(gates_data, f, indent=2)
    print(f"  机器门已保存: {gates_path}")
    
    return {
        "buoyancy_path": str(buoyancy_path),
        "gates_path": str(gates_path),
        "gates": gates_data,
        "buoyancy_data": buoyancy_data,
    }


def main():
    """主函数"""
    output_dir = Path(__file__).parent / "outputs"
    
    try:
        result = run_buoyancy_freesurface_simulation(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ SIR-3 成功完成")
        print("=" * 80)
        print(f"浮力结果: {result['buoyancy_path']}")
        print(f"机器门: {result['gates_path']}")
        
        # 打印门状态
        print("\n机器门状态:")
        for gate_name, gate_data in result['gates'].items():
            status = "✓ PASS" if gate_data['pass'] else "✗ FAIL"
            print(f"  {gate_name}: {status}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ SIR-3 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
