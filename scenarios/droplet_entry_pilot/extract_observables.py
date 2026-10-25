"""从双求解器输出中抽取观测量 - MP-DROP-0 先导（v2 勘误版）

v1 缺陷（2026-10-25 owner 亲目动画异常触发勘误，见本目录 ERRATUM.md）：
1. LBM：drop_mask 阈值用逐帧全场 max → 整个水池（均值 0.8 > 阈 0.6）被选为
   「液滴」，穿透深度实为全池质心位移（5.4–8.1D 垃圾值，水池深仅 8D）；
2. SPH：drop_mask = 当前时刻 y > pool_depth → 液滴没入后从观察量中消失，
   且两侧水舌粒子被误标为液滴；
3. 两者都缺接触时刻自检：marker 时刻 {0.5,1.0,1.5,2.0} 中 LBM 前两个、
   SPH 第一个在运动学接触之前，对比数字无意义。

v2 语义：
- LBM 液滴 = 配置常数阈值 rho_water > 0.5*(rho_water_cfg + rho_air_cfg)
  且位于初始水面之上（y >= water_level）的连通区域（x 限制在初始液滴
  2 倍半径邻带内抗气相污染）；区域像素数跌破初始液滴像素 10% 后
  观察量标记 valid=false（融合/消散，不给质心垃圾值）；
- SPH 液滴 = 首帧 y > pool_depth + dx 的粒子索引块，全程按索引追踪
  （粒子序号不变，块不丢失、不误纳水舌）；
- 穿透深度统一以初始水面为零点、向下为正：d/D = (surface - cy)/D，
  接触前为负值（=液滴质心在水面以上高度）；
- 运动学接触时刻 = 液滴最低点首次抵达水面（LBM 掩膜 ymin / SPH 块 ymin），
  输出 contact_t_star，并对每个 marker 标注 pre/post-contact；
- 物理上界护栏：|d/D| 超过水池深度（以 D 计）+ 1D 即判 mask 语义破裂
  （v1 的 8D 值会被此护栏当场拦下）。

输出：
    observables_lbm.<tag>.json / observables_sph.<tag>.json（--tag，默认 v2）
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------- LBM ------- #
def extract_lbm_observables(run_dir: Path, u0: Optional[float] = None) -> Dict:
    """从 LBM 输出中抽取观测量（v2：初始水面以上区域 + 接触自检 + 气相纯度门）

    u0 = 液滴初始下落速度（格子/步）。新工件请写入 config["u0"]；对旧工件可用
    CLI --u0 回溯指定（m303 的 run_lbm.py:67 硬编码 -0.01，config 未记录）。
    无 u0 时接触时刻只能经验探测，界面涂抹场里不可靠 → 标 low-confidence。
    """
    snapshot_dir = run_dir / "snapshots"
    metadata_file = run_dir / "run_metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"LBM metadata not found: {metadata_file}")

    metadata = json.loads(metadata_file.read_text())
    config = metadata["config"]
    derived = metadata["derived"]

    D = derived["D"]  # 特征长度（格子）
    U = derived["U"]  # 特征速度（格子/步）
    water_level = int(config["water_level"])
    rho_thr = 0.5 * (config["rho_water"] + config["rho_air"])  # 配置常数阈值
    u0 = u0 if u0 is not None else config.get("u0")

    if "drop_center" in config:
        cx0, cy0 = config["drop_center"]
        r = config["drop_radius"]
    else:
        raise ValueError("v2 需要 config.drop_center/drop_radius（v1 刚体模式已废弃）")

    # 运动学接触时刻（首选）：gap = u0·t + g·t²/2
    kin_contact_t_star = None
    if u0 is not None:
        gap = (cy0 - r) - water_level
        g = config["g"]
        t_contact = (-u0 + np.sqrt(u0 * u0 + 2 * g * gap)) / g
        kin_contact_t_star = float(t_contact * U / D)

    # x 邻带（初始液滴 ±2r），抗气相污染杂点
    x_lo = max(0, int(cx0 - 2 * r))
    x_hi = min(config["nx"], int(cx0 + 2 * r) + 1)
    n_init_ref = max(1, int(np.pi * r * r))  # 初始液滴像素量级（圆形近似）

    # 气相纯度监测框：远离液滴路径的右上角（行 water_level+42 起、避开界面涂抹带）
    air_r0 = min(water_level + 42, config["ny"] - 10)
    air_r1 = min(water_level + 122, config["ny"])
    air_c1 = min(40, config["nx"])
    air_impure_threshold = 0.25 * rho_thr  # 健康场应保持 ~初始残值（远低于此）

    snapshots = sorted(snapshot_dir.glob("density_*.pt"))
    penetration_depth = []
    shape_ratio = []
    empirical_contact_t_star = None
    air_mean_series = []

    for snap_file in snapshots:
        data = torch.load(snap_file, map_location="cpu", weights_only=True)
        step = data["step"]
        rho_water = data["rho_water"].numpy()
        t_star = step * U / D

        air_mean = float(rho_water[air_r0:air_r1, 0:air_c1].mean())
        air_mean_series.append({"t_star": float(t_star), "air_mean": air_mean})

        # 液滴区域 = 密度超配置阈值 且 在初始水面之上 且 在初始液滴 x 邻带内
        drop_mask = (
            (rho_water > rho_thr)
            & (np.arange(rho_water.shape[0])[:, None] >= water_level)
        )
        drop_mask[:, :x_lo] = False
        drop_mask[:, x_hi:] = False

        n_pix = int(drop_mask.sum())
        y_coords, x_coords = np.where(drop_mask)

        if n_pix == 0 or n_pix < 0.10 * n_init_ref:
            # 液滴已融合/消散：不给质心值，标记无效
            penetration_depth.append({
                "step": step, "t_star": float(t_star),
                "d/D": None, "valid": False, "air_mean": air_mean,
                "reason": f"above-surface region {n_pix}px < 10% of initial droplet",
            })
            continue

        if empirical_contact_t_star is None and int(y_coords.min()) <= water_level:
            # 经验探测：掩膜最低像素触到初始水面行。注意在界面涂抹严重的场里，
            # 池面本身的涂抹层会先于液滴触线造成假接触 —— 以运动学值为准。
            empirical_contact_t_star = float(t_star)

        cy_current = float(y_coords.mean())
        # 以初始水面为零点、向下为正；接触前为负（质心在水面以上）
        d_normalized = (water_level - cy_current) / D

        # 物理上界护栏：水池深 water_level 格 = water_level/D 倍 D
        pool_depth_D = water_level / D
        valid = abs(d_normalized) <= pool_depth_D + 1.0

        penetration_depth.append({
            "step": step, "t_star": float(t_star),
            "d/D": float(d_normalized), "valid": valid, "air_mean": air_mean,
            "reason": None if valid else
                f"|d/D|={abs(d_normalized):.2f} 超过水池深度+1D={pool_depth_D + 1.0:.1f}（mask 语义破裂）",
        })

        if valid:
            W = float(x_coords.max() - x_coords.min())
            H = float(y_coords.max() - y_coords.min())
            if H > 0:
                shape_ratio.append({
                    "step": step, "t_star": float(t_star),
                    "lambda": float(W / H), "W": W, "H": H,
                })

    # ---- 气相纯度门：污染超阈值 → 全部观察量判无效（诚实输出而非垃圾数） ---- #
    air_max = max((a["air_mean"] for a in air_mean_series), default=0.0)
    field_pure = air_max <= air_impure_threshold
    if not field_pure:
        for rec in penetration_depth:
            if rec.get("valid"):
                rec["valid"] = False
                rec["reason"] = (
                    f"气相纯度门未过：监测框均值峰值 {air_max:.3f} > {air_impure_threshold:.3f}"
                    f"（阈值=0.25×界面阈值）——水组分已污染气相，『液滴区域』语义不可信"
                )

    contact_t_star = kin_contact_t_star if kin_contact_t_star is not None \
        else empirical_contact_t_star
    contact_source = ("kinematic" if kin_contact_t_star is not None
                      else "empirical-low-confidence")

    t_star_markers = [0.5, 1.0, 1.5, 2.0]
    penetration_at_markers = _interpolate_to_markers(
        penetration_depth, t_star_markers, "t_star", "d/D",
        contact_t_star=contact_t_star,
    )
    shape_at_markers = _interpolate_to_markers(
        shape_ratio, t_star_markers[:3], "t_star", "lambda",
        contact_t_star=contact_t_star,
    )

    return {
        "solver": "TensorLBM",
        "extractor": "v2",
        "D": D, "U": U, "Re": derived["Re"], "Fr": derived["Fr"],
        "contact_t_star": contact_t_star,
        "contact_source": contact_source,
        "kinematic_contact_t_star": kin_contact_t_star,
        "empirical_contact_t_star": empirical_contact_t_star,
        "field_purity": {
            "air_monitor_box": [air_r0, air_r1, 0, air_c1],
            "air_mean_max": air_max,
            "threshold": air_impure_threshold,
            "pure": bool(field_pure),
        },
        "penetration_depth": penetration_at_markers,
        "shape_ratio": shape_at_markers,
        "raw": {
            "air_mean_series": air_mean_series,
            "penetration_depth": penetration_depth,
            "shape_ratio": shape_ratio,
        },
    }


# ---------------------------------------------------------------- SPH ------- #
def extract_sph_observables(run_dir: Path) -> Dict:
    """从 SPH 输出中抽取观测量（v2：首帧索引块全程追踪 + 接触自检）"""
    metadata_file = run_dir / "run_manifest.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"SPH metadata not found: {metadata_file}")

    metadata = json.loads(metadata_file.read_text())
    config = metadata["config"]
    derived = metadata["derived"]

    D = derived["D"]
    U = derived["U"]
    dt = config["dt"]
    pool_depth = config["pool_depth"]
    dx = config["dx"]
    drop_center_y_initial = config["drop_center_y"]

    snapshot_files = sorted(run_dir.glob("particles_*.pt"))
    if not snapshot_files:
        raise FileNotFoundError(f"no particles_*.pt under {run_dir}")

    # 首帧锁定液滴粒子索引块（初始在水面以上），全程按索引追踪
    first = torch.load(snapshot_files[0], map_location="cpu", weights_only=True)
    drop_ids = np.where(first["position"][:, 1].numpy() > pool_depth + dx)[0]
    if drop_ids.size == 0:
        raise ValueError("首帧未找到水面以上液滴粒子（配置或输出异常）")

    penetration_depth = []
    shape_ratio = []
    contact_t_star = None

    for snap_file in snapshot_files:
        data = torch.load(snap_file, map_location="cpu", weights_only=True)
        step = data["step"]
        pos = data["position"].numpy()
        t_star = step * dt * U / D

        drop_pos = pos[drop_ids]
        ymin = float(drop_pos[:, 1].min())
        if contact_t_star is None and ymin <= pool_depth:
            contact_t_star = float(t_star)

        cy_current = float(drop_pos[:, 1].mean())
        # 与 LBM 同一口径：水面零点、向下为正；接触前为负
        d_normalized = (pool_depth - cy_current) / D

        pool_depth_D = pool_depth / D
        valid = abs(d_normalized) <= pool_depth_D + 1.0

        penetration_depth.append({
            "step": step, "t_star": float(t_star),
            "d/D": float(d_normalized), "valid": valid,
            "above_fraction": float((drop_pos[:, 1] > pool_depth).mean()),
            "reason": None if valid else
                f"|d/D|={abs(d_normalized):.2f} 超过水池深度+1D={pool_depth_D + 1.0:.1f}",
        })

        if valid:
            W = float(drop_pos[:, 0].max() - drop_pos[:, 0].min())
            H = float(drop_pos[:, 1].max() - drop_pos[:, 1].min())
            if H > 0:
                shape_ratio.append({
                    "step": step, "t_star": float(t_star),
                    "lambda": float(W / H), "W": W, "H": H,
                })

    t_star_markers = [0.5, 1.0, 1.5, 2.0]
    penetration_at_markers = _interpolate_to_markers(
        penetration_depth, t_star_markers, "t_star", "d/D",
        contact_t_star=contact_t_star,
    )
    shape_at_markers = _interpolate_to_markers(
        shape_ratio, t_star_markers[:3], "t_star", "lambda",
        contact_t_star=contact_t_star,
    )

    return {
        "solver": "TensorSPH",
        "extractor": "v2",
        "D": D, "U": U, "Re": derived["Re"], "Fr": derived["Fr"],
        "n_drop_particles": int(drop_ids.size),
        "drop_id_range": [int(drop_ids.min()), int(drop_ids.max())],
        "contact_t_star": contact_t_star,
        "penetration_depth": penetration_at_markers,
        "shape_ratio": shape_at_markers,
        "raw": {
            "penetration_depth": penetration_depth,
            "shape_ratio": shape_ratio,
        },
    }


# --------------------------------------------------------------- shared ----- #
def _interpolate_to_markers(
    data: List[Dict],
    t_star_markers: List[float],
    t_key: str,
    value_key: str,
    contact_t_star: Optional[float] = None,
) -> List[Dict]:
    """将数据插值到标记时刻；标注 marker 是否在运动学接触之后"""
    usable = [d for d in data if d.get(value_key) is not None and d.get("valid", True)]
    if not usable:
        return [{"t_star": t_marker, value_key: None,
                 "post_contact": None, "note": "no valid samples"}
                for t_marker in t_star_markers]

    t_values = [d[t_key] for d in usable]
    values = [d[value_key] for d in usable]

    result = []
    for t_marker in t_star_markers:
        if t_marker <= t_values[0]:
            interp_val = values[0]
        elif t_marker >= t_values[-1]:
            interp_val = values[-1]
        else:
            interp_val = values[-1]
            for i in range(len(t_values) - 1):
                if t_values[i] <= t_marker <= t_values[i + 1]:
                    t0, t1 = t_values[i], t_values[i + 1]
                    alpha = (t_marker - t0) / (t1 - t0)
                    interp_val = values[i] + alpha * (values[i + 1] - values[i])
                    break

        post_contact = None if contact_t_star is None else bool(t_marker >= contact_t_star)
        result.append({
            "t_star": t_marker,
            value_key: float(interp_val),
            "post_contact": post_contact,
        })
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbm-dir", type=Path, required=True, help="LBM 输出目录")
    parser.add_argument("--sph-dir", type=Path, required=True, help="SPH 输出目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tag", type=str, default="v2",
                        help="输出文件后缀（observables_<solver>.<tag>.json）")
    parser.add_argument("--u0", type=float, default=None,
                        help="LBM 液滴初始下落速度（格子/步，旧工件回溯指定；"
                             "m303 的 run_lbm.py 硬编码 0.01 但未写入 config）")
    args = parser.parse_args()

    print("Extracting LBM observables (v2: above-surface region + contact check)...")
    obs_lbm = extract_lbm_observables(args.lbm_dir, u0=args.u0)

    print("Extracting SPH observables (v2: index-block tracking + contact check)...")
    obs_sph = extract_sph_observables(args.sph_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lbm_file = args.output_dir / f"observables_lbm.{args.tag}.json"
    sph_file = args.output_dir / f"observables_sph.{args.tag}.json"

    lbm_file.write_text(json.dumps(obs_lbm, indent=2))
    sph_file.write_text(json.dumps(obs_sph, indent=2))

    # ---- 自检摘要（机器可读，勘误证据） ---------------------------------- #
    for name, obs in (("LBM", obs_lbm), ("SPH", obs_sph)):
        ct = obs.get("contact_t_star")
        src = obs.get("contact_source", "")
        print(f"[{name}] contact_t_star = {ct} ({src})" if src else f"[{name}] contact_t_star = {ct}")
        if "field_purity" in obs:
            fp = obs["field_purity"]
            print(f"  field_purity: air_mean_max={fp['air_mean_max']:.3f} "
                  f"threshold={fp['threshold']:.3f} pure={fp['pure']}")
        for m in obs["penetration_depth"]:
            pc = m.get("post_contact")
            if pc is None:
                flag = "INVALID(观察量未定义)"
            elif pc:
                flag = "OK"
            else:
                flag = "PRE-CONTACT(无效对比点)"
            val = m.get("d/D")
            val_s = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  t*={m['t_star']:.2f}  d/D={val_s}  {flag}")
    print(f"Saved: {lbm_file}")
    print(f"Saved: {sph_file}")
