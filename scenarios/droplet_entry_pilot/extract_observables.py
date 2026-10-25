"""从双求解器输出中抽取观测量 - MP-DROP-0 先导

抽取的观测量：
1. 穿透深度 d(t)/D @ t* = 0.5, 1.0, 1.5, 2.0
2. 形状比 λ(t) = W/H @ t* = 0.5, 1.0, 1.5
3. （可选）空腔/冠最大宽度

输出：
    observables_lbm.json
    observables_sph.json
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def extract_lbm_observables(run_dir: Path) -> Dict:
    """从 LBM 输出中抽取观测量"""
    snapshot_dir = run_dir / "snapshots"
    metadata_file = run_dir / "run_metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"LBM metadata not found: {metadata_file}")
    
    metadata = json.loads(metadata_file.read_text())
    config = metadata["config"]
    derived = metadata["derived"]
    
    D = derived["D"]  # 特征长度（格子）
    U = derived["U"]  # 特征速度（格子/步）
    
    # 液滴初始中心位置
    cx = config["nx"] // 2
    # 支持新的 fluid_droplet 模式（使用 drop_center）和旧的刚体模式
    if "drop_center" in config:
        cx, cy_initial = config["drop_center"]
    else:
        cy = config["water_level"] + config.get("clearance", 8) + config["radius"] + 1
        cy_initial = cy
    
    # 抽取快照
    snapshots = sorted(snapshot_dir.glob("density_*.pt"))
    
    penetration_depth = []
    shape_ratio = []
    
    for snap_file in snapshots:
        data = torch.load(snap_file, map_location="cpu")
        step = data["step"]
        rho_water = data["rho_water"].numpy()
        rho_air = data["rho_air"].numpy()
        
        # 计算物理时间
        t = step  # 格子单位：步
        t_star = t * U / D  # 无量纲时间
        
        # 识别液滴/圆柱区域（水相主导区域）
        # 阈值：rho_water > 0.5 * (rho_water_max + rho_air_max)
        rho_threshold = 0.5 * (rho_water.max() + rho_air.max())
        drop_mask = rho_water > rho_threshold
        
        if drop_mask.sum() == 0:
            print(f"  Warning: no drop region at step {step}")
            continue
        
        # 计算质心
        y_coords, x_coords = np.where(drop_mask)
        if len(y_coords) == 0:
            continue
        
        cy_current = y_coords.mean()
        cx_current = x_coords.mean()
        
        # 穿透深度（y 向下为正，所以 cy_initial - cy_current）
        d = cy_initial - cy_current
        d_normalized = d / D
        
        penetration_depth.append({
            "step": step,
            "t_star": float(t_star),
            "d/D": float(d_normalized),
            "cy": float(cy_current),
        })
        
        # 形状比 W/H
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        W = x_max - x_min
        H = y_max - y_min
        
        if H > 0:
            lambda_val = W / H
            shape_ratio.append({
                "step": step,
                "t_star": float(t_star),
                "lambda": float(lambda_val),
                "W": float(W),
                "H": float(H),
            })
    
    # 插值到标记时刻
    t_star_markers = [0.5, 1.0, 1.5, 2.0]
    penetration_at_markers = _interpolate_to_markers(
        penetration_depth, t_star_markers, "t_star", "d/D"
    )
    
    t_star_markers_shape = [0.5, 1.0, 1.5]
    shape_at_markers = _interpolate_to_markers(
        shape_ratio, t_star_markers_shape, "t_star", "lambda"
    )
    
    return {
        "solver": "TensorLBM",
        "D": D,
        "U": U,
        "Re": derived["Re"],
        "Fr": derived["Fr"],
        "penetration_depth": penetration_at_markers,
        "shape_ratio": shape_at_markers,
        "raw": {
            "penetration_depth": penetration_depth,
            "shape_ratio": shape_ratio,
        },
    }


def extract_sph_observables(run_dir: Path) -> Dict:
    """从 SPH 输出中抽取观测量"""
    metadata_file = run_dir / "run_manifest.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"SPH metadata not found: {metadata_file}")
    
    metadata = json.loads(metadata_file.read_text())
    config = metadata["config"]
    derived = metadata["derived"]
    
    D = derived["D"]  # 特征长度（米）
    U = derived["U"]  # 特征速度（米/秒）
    dt = config["dt"]
    
    # 液滴初始中心位置
    drop_center_y_initial = config["drop_center_y"]
    
    # 抽取粒子快照
    snapshot_files = sorted(run_dir.glob("particles_*.pt"))
    
    penetration_depth = []
    shape_ratio = []
    
    for snap_file in snapshot_files:
        data = torch.load(snap_file, map_location="cpu")
        step = data["step"]
        pos = data["position"].numpy()  # (N, 2)
        
        # 计算物理时间
        t = step * dt  # 秒
        t_star = t * U / D  # 无量纲时间
        
        # 识别液滴粒子（y > pool_depth）
        pool_depth = config["pool_depth"]
        drop_mask = pos[:, 1] > pool_depth
        
        if drop_mask.sum() == 0:
            print(f"  Warning: no drop particles at step {step}")
            continue
        
        drop_pos = pos[drop_mask]
        
        # 计算质心
        cy_current = drop_pos[:, 1].mean()
        cx_current = drop_pos[:, 0].mean()
        
        # 穿透深度（y 向上为正，所以 drop_center_y_initial - cy_current）
        # 注意：SPH 中 y 向上为正，液滴从高处下落
        d = drop_center_y_initial - cy_current
        d_normalized = d / D
        
        penetration_depth.append({
            "step": step,
            "t_star": float(t_star),
            "d/D": float(d_normalized),
            "cy": float(cy_current),
        })
        
        # 形状比 W/H
        x_min, x_max = drop_pos[:, 0].min(), drop_pos[:, 0].max()
        y_min, y_max = drop_pos[:, 1].min(), drop_pos[:, 1].max()
        W = x_max - x_min
        H = y_max - y_min
        
        if H > 0:
            lambda_val = W / H
            shape_ratio.append({
                "step": step,
                "t_star": float(t_star),
                "lambda": float(lambda_val),
                "W": float(W),
                "H": float(H),
            })
    
    # 插值到标记时刻
    t_star_markers = [0.5, 1.0, 1.5, 2.0]
    penetration_at_markers = _interpolate_to_markers(
        penetration_depth, t_star_markers, "t_star", "d/D"
    )
    
    t_star_markers_shape = [0.5, 1.0, 1.5]
    shape_at_markers = _interpolate_to_markers(
        shape_ratio, t_star_markers_shape, "t_star", "lambda"
    )
    
    return {
        "solver": "TensorSPH",
        "D": D,
        "U": U,
        "Re": derived["Re"],
        "Fr": derived["Fr"],
        "penetration_depth": penetration_at_markers,
        "shape_ratio": shape_at_markers,
        "raw": {
            "penetration_depth": penetration_depth,
            "shape_ratio": shape_ratio,
        },
    }


def _interpolate_to_markers(
    data: List[Dict],
    t_star_markers: List[float],
    t_key: str,
    value_key: str,
) -> List[Dict]:
    """将数据插值到标记时刻"""
    if not data:
        return []
    
    t_values = [d[t_key] for d in data]
    values = [d[value_key] for d in data]
    
    result = []
    for t_marker in t_star_markers:
        # 线性插值
        if t_marker <= t_values[0]:
            interp_val = values[0]
        elif t_marker >= t_values[-1]:
            interp_val = values[-1]
        else:
            # 找到相邻点
            for i in range(len(t_values) - 1):
                if t_values[i] <= t_marker <= t_values[i + 1]:
                    t0, t1 = t_values[i], t_values[i + 1]
                    v0, v1 = values[i], values[i + 1]
                    alpha = (t_marker - t0) / (t1 - t0)
                    interp_val = v0 + alpha * (v1 - v0)
                    break
        
        result.append({
            "t_star": t_marker,
            value_key: float(interp_val),
        })
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbm-dir", type=Path, required=True, help="LBM 输出目录")
    parser.add_argument("--sph-dir", type=Path, required=True, help="SPH 输出目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    
    print("Extracting LBM observables...")
    obs_lbm = extract_lbm_observables(args.lbm_dir)
    
    print("Extracting SPH observables...")
    obs_sph = extract_sph_observables(args.sph_dir)
    
    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    lbm_file = args.output_dir / "observables_lbm.json"
    sph_file = args.output_dir / "observables_sph.json"
    
    lbm_file.write_text(json.dumps(obs_lbm, indent=2))
    sph_file.write_text(json.dumps(obs_sph, indent=2))
    
    print(f"Saved: {lbm_file}")
    print(f"Saved: {sph_file}")
