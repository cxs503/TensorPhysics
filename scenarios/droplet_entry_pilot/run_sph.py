"""TensorSPH 液滴入水适配器 - MP-DROP-0 先导

运行 SPH 求解器并保存粒子位置快照供后续分析。

用法:
    python run_sph.py [--output-dir OUTPUT_DIR]

输出:
    - particles_XXXXXX.pt: 粒子位置快照（PyTorch 张量）
    - diagnostics.csv: 诊断数据时间序列
    - run_manifest.json: 运行元数据
"""
import sys
import torch
from pathlib import Path

# 添加 TensorSPH 到 Python 路径
SPH_ROOT = Path("/nfs/wangxi/tensor_eco/TensorSPH")
sys.path.insert(0, str(SPH_ROOT / "src"))

from tensorsph.droplet_impact import Config, run


def run_with_snapshots(output_dir: Path):
    """运行 SPH 并保存粒子位置快照"""
    # 参数配置（与 prereg.md 一致）
    config = Config(
        dx=0.005,
        h_factor=1.3,
        rho0=1000.0,
        sound_speed=20.0,
        gamma=7.0,
        gravity=2.0,  # 平衡点：足够快但稳定
        viscosity=0.001,
        dt=9.0e-5,  # 满足 Courant 条件
        steps=15000,  # 增加步数，让液滴有足够时间入水
        output_every=100,
        pool_width=1.28,
        pool_depth=0.64,
        drop_radius=0.08,
        drop_center_y=0.80,
        device="auto",
        boundary="wall",
        wall_layers=3,
        max_courant=0.3,
        max_density_error=0.05,
        max_mach=0.10,
        fail_fast=True,
        pair_skin_factor=0.25,
        hydrostatic_init=False,
        wall_friction=0.0,
        density_diffusion=0.0,
        adaptive_dt=False,
    )
    
    print(f"Running SPH: dx={config.dx}, pool={config.pool_width}x{config.pool_depth}")
    print(f"  drop_radius={config.drop_radius}, steps={config.steps}")
    print(f"Output: {output_dir}")
    
    # 运行求解器
    rows = run(config, output_dir, plot=True)
    
    # 计算导出物理量
    D = 2 * config.drop_radius
    U = (config.gravity * D) ** 0.5
    nu = config.viscosity
    Re = U * D / nu
    Fr = U / (config.gravity * D) ** 0.5
    
    # 更新元数据
    import json
    manifest_file = output_dir / "run_manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        manifest["derived"] = {
            "D": D,
            "U": U,
            "nu": nu,
            "Re": Re,
            "Fr": Fr,
        }
        manifest["config"]["pool_width"] = config.pool_width
        manifest["config"]["pool_depth"] = config.pool_depth
        manifest["config"]["drop_radius"] = config.drop_radius
        manifest["config"]["drop_center_y"] = config.drop_center_y
        manifest_file.write_text(json.dumps(manifest, indent=2))
        print(f"Updated metadata → {manifest_file}")
    
    print(f"Completed: {len(rows)} diagnostic rows")
    print(f"  D={D:.4f} m, U={U:.4f} m/s, Re={Re:.2f}, Fr={Fr:.2f}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sph"))
    args = parser.parse_args()
    
    run_with_snapshots(args.output_dir)
