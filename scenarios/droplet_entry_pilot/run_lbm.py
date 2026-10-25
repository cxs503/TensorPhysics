"""TensorLBM 液滴入水适配器 - MP-DROP-0 先导

运行 LBM 多相流求解器（流体液滴版本）并保存密度场快照供后续分析。

用法:
    python run_lbm.py [--output-dir OUTPUT_DIR]

输出:
    - snapshots/density_XXXXXX.pt: 密度场快照（PyTorch 张量）
    - run_metadata.json: 运行元数据
    - forces.csv: 力时间序列（此处为空，因为无刚体）
"""
import sys
import json
import torch
from pathlib import Path

# 添加 TensorLBM 到 Python 路径
LBM_ROOT = Path("/nfs/wangxi/tensor_eco/TensorLBM")
sys.path.insert(0, str(LBM_ROOT / "src"))

from tensorlbm.d2q9 import equilibrium
from tensorlbm.multiphase import color_gradient_step
from tensorlbm.solver import stream
from tensorlbm.utils import resolve_device


def _smooth_profile_y(ny, nx, water_level, width, device):
    """tanh profile: 1.0 in water (y < water_level), 0.0 in air."""
    y = torch.arange(ny, dtype=torch.float32, device=device)
    prof = 0.5 * (1.0 - torch.tanh((y - water_level) / width))
    return prof.view(ny, 1).expand(ny, nx)


def _circle_mask(ny, nx, cx, cy, r, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def _wall_mask_2d(ny, nx, device):
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


def _init_fluid_droplet_cg_2d(
    ny, nx, water_level, drop_cx, drop_cy, drop_radius,
    rho_water, rho_air, device,
):
    """初始化流体液滴（不是刚体）"""
    # 基础水/空气分布
    prof = _smooth_profile_y(ny, nx, water_level, 3.0, device)
    frac = 0.05
    rho_r = rho_water * prof + rho_water * frac * (1.0 - prof)
    rho_b = rho_air * frac * prof + rho_air * (1.0 - prof)
    
    # 在液滴区域增加水相
    drop_mask = _circle_mask(ny, nx, drop_cx, drop_cy, drop_radius, device)
    rho_r[drop_mask] = rho_water
    rho_b[drop_mask] = rho_air * frac
    
    # 初始速度（向下）
    vy_init = -0.01  # 格子单位/步
    zero_x = torch.zeros((ny, nx), device=device)
    vy = torch.full((ny, nx), 0.0, device=device)
    vy[drop_mask] = vy_init
    
    return (
        equilibrium(rho_r, zero_x, vy),
        equilibrium(rho_b, zero_x, vy),
    )


def run_with_snapshots(output_dir: Path):
    """运行 LBM 并保存密度场快照"""
    # 参数配置
    nx = 128
    ny = 256
    water_level = 128
    drop_radius = 8.0
    drop_cx = nx / 2
    drop_cy = water_level + 20  # 液滴中心在水面上方
    rho_water = 0.8
    rho_air = 0.4
    tau = 1.0
    g_lattice = 5e-5
    n_steps = 1500
    output_interval = 150
    
    device = resolve_device("cpu")
    
    # 创建快照目录
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化
    wall = _wall_mask_2d(ny, nx, device)
    
    f1, f2 = _init_fluid_droplet_cg_2d(
        ny, nx, water_level, drop_cx, drop_cy, drop_radius,
        rho_water, rho_air, device,
    )
    
    gy = -g_lattice
    A_surface = 0.9 * 0.04
    
    print(f"Running LBM (fluid droplet): {nx}x{ny}, drop_radius={drop_radius}, steps={n_steps}")
    print(f"Output: {output_dir}")
    
    # 主循环
    for step in range(1, n_steps + 1):
        # 碰撞
        f1, f2 = color_gradient_step(
            f1, f2,
            tau=tau,
            A=A_surface,
            gy=gy,
            solid_mask=wall,  # 只有墙壁，无刚体
        )
        
        # 流动
        f1 = stream(f1)
        f2 = stream(f2)
        
        # 诊断和快照
        if step % output_interval == 0 or step == n_steps:
            # 保存密度场快照
            rho_water_field = f1.sum(dim=0)
            rho_air_field = f2.sum(dim=0)
            rho_total = rho_water_field + rho_air_field
            
            snapshot_file = snapshot_dir / f"density_{step:06d}.pt"
            torch.save({
                "step": step,
                "rho_water": rho_water_field.detach().cpu(),
                "rho_air": rho_air_field.detach().cpu(),
                "rho_total": rho_total.detach().cpu(),
                "wall_mask": wall.detach().cpu(),
                "config": {
                    "nx": nx,
                    "ny": ny,
                    "drop_radius": drop_radius,
                    "drop_center": (float(drop_cx), float(drop_cy)),
                    "water_level": water_level,
                },
            }, snapshot_file)
            
            print(f"  Step {step}: saved snapshot")
    
    # 保存元数据
    D = 2 * drop_radius
    U = (g_lattice * D) ** 0.5
    nu = (tau - 0.5) / 3.0
    Re = U * D / nu
    Fr = 1.0
    
    metadata = {
        "solver": "TensorLBM",
        "model": "cg",
        "mode": "fluid_droplet",
        "config": {
            "nx": nx,
            "ny": ny,
            "drop_radius": drop_radius,
            "drop_center": (float(drop_cx), float(drop_cy)),
            "water_level": water_level,
            "rho_water": rho_water,
            "rho_air": rho_air,
            "density_ratio": rho_water / rho_air,
            "tau": tau,
            "g": g_lattice,
            "u0": abs(float(vy_init)),  # 液滴初始下落速度（格子/步），v2 抽取器运动学接触时刻用
            "n_steps": n_steps,
            "output_interval": output_interval,
        },
        "derived": {
            "D": D,
            "U": U,
            "nu": nu,
            "Re": Re,
            "Fr": Fr,
        },
        "forces": [],
        "snapshots_saved": len(list(snapshot_dir.glob("*.pt"))),
    }
    
    metadata_file = output_dir / "run_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata → {metadata_file}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lbm"))
    args = parser.parse_args()
    
    run_with_snapshots(args.output_dir)
