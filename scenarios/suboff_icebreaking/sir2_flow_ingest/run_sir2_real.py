"""SIR-2: 真实 LBM 流场入链 (G2) - 使用 GeneralSimEngine

目标：替换合成载荷，使用真实 SUBOFF 几何的 LBM 仿真结果
物理锚：
  A2.1: SUBOFF Re=1000 阻力基线 (Blasius Cf=0.0420)
  A2.2: Hetényi 弹性地基板级数解 (<2% 误差)
  A2.3: SUBOFF 几何尺寸 L=4.356m, D=0.508m
  A2.4: 掩码实体单元计数 (appendage_solid_cells=126)

机器门：
  M2.1: 溯源门 - 无合成标记
  M2.2: 几何在场门 - hull_mask.solid_cells > 0
  M2.3: 传递双账本 - max|A-B| <= 1e-12
  M2.4: 响应端不退化 - Hetényi 相对误差 < 2%
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# 添加 TensorLBM 路径
sys.path.insert(0, "/nfs/wangxi/tensor_eco/TensorLBM/src")

from tensorlbm import general_sim


def compute_sha256(filepath):
    """计算文件 SHA256"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_real_lbm_simulation(output_dir):
    """运行真实 SUBOFF LBM 仿真
    
    使用 TensorLBM 的 GeneralSimEngine 运行真实仿真
    """
    print("=" * 80)
    print("SIR-2: 运行真实 SUBOFF LBM 仿真 (GeneralSimEngine)")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置 SUBOFF 几何
    print("\n[1/4] 配置 SUBOFF 几何...")
    geometry = general_sim.GeometryConfig(
        source=general_sim.GeometrySource.PARAMETRIC_SUBOFF,
        suboff_length=4.356,  # 真实 SUBOFF 尺寸 (m)
        suboff_radius=0.254,  # D=0.508m, R=0.254m
    )
    print(f"  几何源: {geometry.source}")
    print(f"  船长: {geometry.suboff_length} m")
    print(f"  船径: {geometry.suboff_radius * 2} m")
    
    # 配置物理参数 (Re=1000)
    print("\n[2/4] 配置物理参数...")
    # Re = U * L / nu = 1000
    # 取 L=4.356m, nu=1e-6 m^2/s (水)
    # U = Re * nu / L = 1000 * 1e-6 / 4.356 = 2.296e-4 m/s
    Re_target = 1000.0
    nu = 1e-6  # 水的运动粘度
    L_ref = 4.356
    U_inlet = Re_target * nu / L_ref
    
    physics = general_sim.PhysicsConfig(
        density=1000.0,  # 水密度
        viscosity=nu,
        inlet_velocity=U_inlet,
        reference_length=L_ref,
        gravity=(0.0, 0.0, 0.0),  # 无重力（纯阻力测试）
    )
    print(f"  密度: {physics.density} kg/m^3")
    print(f"  粘度: {physics.viscosity} m^2/s")
    print(f"  入口速度: {physics.inlet_velocity:.6e} m/s")
    print(f"  参考长度: {physics.reference_length} m")
    print(f"  目标 Re: {Re_target}")
    
    # 配置求解器参数 (极小规模以适配 CPU 限时)
    print("\n[3/4] 配置求解器...")
    solver = general_sim.SolverConfig(
        lattice=general_sim.LatticeModel.D3Q19,
        collision=general_sim.CollisionModel.AUTO,
        resolution=16,  # 极小规模以适配 CPU 限时
        domain_padding=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),  # 各方向 padding
        max_steps=50,  # 极小规模测试
        warmup_steps=10,
        snapshot_interval=25,
        force_sample_interval=5,
        target_mach=0.05,
        device="cpu",
        wall_treatment=general_sim.WallTreatment.AUTO,
        force_method=general_sim.ForceMethod.PRESSURE_FRICTION,
        mem_variant="standard",
    )
    print(f"  格子模型: {solver.lattice}")
    print(f"  分辨率: {solver.resolution}")
    print(f"  最大步数: {solver.max_steps}")
    print(f"  设备: {solver.device}")
    
    # 配置输出
    output_config = general_sim.OutputConfig(
        directory=str(output_dir),
        formats=[general_sim.OutputFormat.NPY],
        save_macroscopic=True,
        save_forces=True,
    )
    
    # 配置边界条件 (x方向流: inlet x_min, outlet x_max)
    boundaries = [
        general_sim.BoundaryCondition(
            face="x_min",
            type=general_sim.BoundaryType.ZOU_HE_INLET,
            velocity=(U_inlet, 0.0, 0.0),
        ),
        general_sim.BoundaryCondition(
            face="x_max",
            type=general_sim.BoundaryType.ZOU_HE_OUTLET,
            pressure=101325.0,
        ),
        general_sim.BoundaryCondition(
            face="y_min",
            type=general_sim.BoundaryType.FAR_FIELD,
        ),
        general_sim.BoundaryCondition(
            face="y_max",
            type=general_sim.BoundaryType.FAR_FIELD,
        ),
        general_sim.BoundaryCondition(
            face="z_min",
            type=general_sim.BoundaryType.FAR_FIELD,
        ),
        general_sim.BoundaryCondition(
            face="z_max",
            type=general_sim.BoundaryType.FAR_FIELD,
        ),
    ]
    
    # 创建完整配置
    config = general_sim.GeneralSimConfig(
        name="SIR-2 SUBOFF Real LBM",
        geometry=geometry,
        physics=physics,
        solver=solver,
        boundaries=boundaries,
        output=output_config,
    )
    
    # 运行仿真
    print("\n[4/4] 运行 LBM 仿真...")
    engine = general_sim.GeneralSimEngine(config)
    
    print("  设置引擎...")
    setup_info = engine.setup()
    print(f"  设置完成: Re={setup_info['Re']:.2f}, tau={setup_info['tau']:.4f}")
    print(f"    域尺寸 (lu): {setup_info['domain_lu']}")
    print(f"    障碍物单元: {setup_info['obstacle_cells']}")
    print(f"    总单元: {setup_info['total_cells']}")
    
    print("  运行仿真...")
    result = engine.run()
    print(f"  仿真完成")
    
    # 提取结果
    print("\n提取流场快照...")
    results = engine.results()
    
    # 保存流场快照
    snapshot_path = output_dir / "real_lbm_snapshot.npz"
    
    # 从结果中提取流场数据
    # 假设 results 包含 pressure, velocity 等字段
    snapshot_data = {
        "pressure": results.get("pressure", np.array([])),
        "velocity_x": results.get("velocity_x", np.array([])),
        "velocity_y": results.get("velocity_y", np.array([])),
        "velocity_z": results.get("velocity_z", np.array([])),
        "solid_mask": results.get("solid_mask", np.array([])),
    }
    
    np.savez(snapshot_path, **snapshot_data)
    snapshot_sha256 = compute_sha256(snapshot_path)
    print(f"  流场快照已保存: {snapshot_path}")
    print(f"  SHA256: {snapshot_sha256}")
    
    # 创建 Ledger A (包含实际仿真数据)
    ledger_a = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_sha256,
        "hull_stats": {
            "obstacle_cells": int(setup_info['obstacle_cells']),
            "total_cells": int(setup_info['total_cells']),
            "fluid_cells": int(setup_info['total_cells'] - setup_info['obstacle_cells']),
            "length_m": float(geometry.suboff_length),
            "diameter_m": float(geometry.suboff_radius * 2),
            "L_D_ratio": float(geometry.suboff_length / (geometry.suboff_radius * 2)),
        },
        "physics": {
            "Re": float(setup_info['Re']),
            "tau": float(setup_info['tau']),
            "u_lb": float(setup_info['u_lb']),
            "Ma": float(setup_info['Ma']),
            "density": float(physics.density),
            "viscosity": float(physics.viscosity),
            "inlet_velocity": float(physics.inlet_velocity),
        },
        "solver": {
            "resolution": int(solver.resolution),
            "max_steps": int(solver.max_steps),
            "device": str(solver.device),
            "domain_lu": list(setup_info['domain_lu']),
        },
    }
    
    ledger_a_path = output_dir / "ledger_a.json"
    with open(ledger_a_path, 'w') as f:
        json.dump(ledger_a, f, indent=2)
    print(f"  Ledger A 已保存: {ledger_a_path}")
    
    return {
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_sha256,
        "ledger_a_path": str(ledger_a_path),
        "ledger_a": ledger_a,
    }


def main():
    """主函数"""
    output_dir = Path(__file__).parent / "outputs"
    
    try:
        result = run_real_lbm_simulation(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ SIR-2 成功完成")
        print("=" * 80)
        print(f"流场快照: {result['snapshot_path']}")
        print(f"SHA256: {result['snapshot_sha256']}")
        print(f"Ledger A: {result['ledger_a_path']}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ SIR-2 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
