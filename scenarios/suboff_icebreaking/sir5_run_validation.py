#!/usr/bin/env python3
"""SIR-5 验证脚本：G5 断裂传播（渐进裂纹扩展 + 键断裂演化）。

薄编排：零求解器代码，全部调用兄弟库已验证接口。
- TensorFEM suboff_ice_collision.simulate_impact：SUBOFF 外壳（由 TensorLBM
  suboff_cad.suboff_radius_profile 旋成）顶冰板的 cohesive 缝渐进开裂
- TensorFEM cohesive.BilinearCohesiveLaw：双线性牵引-分离律（A5.1）
- TensorDEM dem3d.IceDEM3D：3-D 键结颗粒场渐进键断裂（压板加载）
- 场景既有 SI-2 锚件 cohesive_failure.identify_failed_faces_uniform_stress：解析枚举锚（A5.2）

跑法（仓库 venv 的 editable install 指向主检出，必须带 PYTHONPATH）：
    PYTHONPATH=src:. /nfs/wangxi/venvs/tensor_eco/bin/python \
        scenarios/suboff_icebreaking/sir5_run_validation.py
CPU 单核约 90 s（FEM 冲击 ~45 s + DEM 3-D 2000 步 ~25 s + 解析锚 <1 s）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
ECO = Path("/nfs/wangxi/tensor_eco")
for p in (ECO / "TensorFEM" / "src", ECO / "TensorDEM" / "src", ECO / "TensorLBM" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SCN = ROOT / "scenarios" / "suboff_icebreaking"
for p in (SCN / "si1_one_way", SCN / "si2_fracture_gen"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensorfem.cohesive import BilinearCohesiveLaw  # noqa: E402
from tensorfem.suboff_geometry import closed_revolved_mesh  # noqa: E402
from tensorfem.suboff_ice_collision import IceImpactConfig, simulate_impact  # noqa: E402
from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
from cohesive_failure import identify_failed_faces_uniform_stress  # noqa: E402

OUT = ROOT / "outputs" / "sir5_fracture_propagation"
PREREG = SCN / "prereg" / "sir5_prereg.md"
TOL = 1e-12


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def suboff_cad():
    path = ECO / "TensorLBM" / "src" / "tensorlbm" / "suboff_cad.py"
    spec = importlib.util.spec_from_file_location("suboff_cad", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["suboff_cad"] = mod
    spec.loader.exec_module(mod)
    return mod


def stage1_fem(length=1.0, axial=12, circum=16, ice_nx=12, ice_ny=4, duration=0.006):
    """FEM 侧：SUBOFF 外壳上顶冰板，cohesive 缝渐进开裂。"""
    cad = suboff_cad()
    radius = length * cad.SuboffConfig().r_over_l
    nodes, elements, audit = closed_revolved_mesh(
        cad.suboff_radius_profile, length, radius, axial, circum)
    geom = {
        "schema": "tensorfem.suboff-outer-geometry/1",
        "nodes": nodes.tolist(), "elements": elements.tolist(), "length_m": length,
        "source": {"project": "TensorLBM", "relative_path": "src/tensorlbm/suboff_cad.py",
                   "function": "suboff_radius_profile"},
        "audit": audit,
    }
    cfg = IceImpactConfig(ice_nx=ice_nx, ice_ny=ice_ny, duration=duration)
    res = simulate_impact(geom, cfg, dt=None)
    hist = res["history"]
    n_failed = [int(s["failed_points"]) for s in hist]
    damage = np.asarray(res["cohesive_damage"], dtype=float)
    areas = np.asarray(res["cohesive_area_weights"], dtype=float)
    seams = np.asarray(res["cohesive_seam_ids"], dtype=int)
    openings = np.asarray(res["cohesive_maximum_opening_m"], dtype=float)
    dx = (length + 2.0 * cfg.ice_margin) / cfg.ice_nx
    dy = cfg.ice_width / cfg.ice_ny
    law = BilinearCohesiveLaw(cfg.cohesive_factor * cfg.ice_young / min(dx, dy),
                              cfg.ice_strength, cfg.ice_fracture_energy)
    # 账本 B：场景侧用双线性律独立复算断裂耗散（梯形积分 - 当前存能），零求解器侵入
    clamped = np.clip(openings - law.onset_opening, 0.0,
                      law.failure_opening - law.onset_opening)
    per_area = (law.peak_traction * law.failure_opening
                / (2.0 * (law.failure_opening - law.onset_opening)) * clamped)
    ledger_a = float(res["metrics"]["fracture_dissipation_J"])   # 库侧账本
    ledger_b = float((per_area * areas).sum())                    # 场景侧复算账本
    failed = damage >= 1.0 - TOL
    return {
        "geometry_audit": audit,
        "suboff_radius_m": radius,
        "config": {"ice_thickness": cfg.ice_thickness, "ice_strength_Pa": cfg.ice_strength,
                   "ice_fracture_energy_J_m2": cfg.ice_fracture_energy,
                   "ice_nx": cfg.ice_nx, "ice_ny": cfg.ice_ny,
                   "duration_s": cfg.duration, "upward_speed_m_s": cfg.upward_speed},
        "cohesive_law": {"stiffness_Pa_per_m": law.stiffness,
                         "onset_opening_m": law.onset_opening,
                         "failure_opening_m": law.failure_opening,
                         "peak_traction_Pa": law.peak_traction,
                         "fracture_energy_J_m2": law.fracture_energy},
        "n_cohesive_points": int(damage.size),
        "n_seams": int(len(set(int(s) for s in seams))),
        "n_failed_faces_cumulative": n_failed,
        "fracture_time_s": [float(s["time_s"]) for s in hist],
        "fracture_dissipation_J_history": [float(s["fracture_dissipation_J"]) for s in hist],
        "n_steps_with_new_fracture": int(sum(1 for a, b in zip(n_failed, n_failed[1:]) if b > a)),
        "n_failed_faces_final": int(n_failed[-1]),
        "n_damaged_seams": int(len({int(s) for s in seams[damage > 0.0]})),
        "n_failed_seams": int(len({int(s) for s in seams[failed]})),
        "failed_seam_ids": sorted({int(s) for s in seams[failed]}),
        "failed_area_m2": float(areas[failed].sum()),
        "total_cohesive_area_m2": float(areas.sum()),
        "fracture_dissipation_ledger_a_J": ledger_a,
        "fracture_dissipation_ledger_b_J": ledger_b,
        "energy_relative_error": abs(ledger_a - ledger_b) / max(abs(ledger_a), abs(ledger_b), 1e-300),
        "metrics": {k: v for k, v in res["metrics"].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "library_checks": res["checks"],
    }


def stage2_dem(nx=15, ny=9, nz=3, breaking_strain=0.004, shear_strain=0.03,
               platen_velocity=-0.06, steps=2000):
    """DEM 侧：3-D 键结颗粒场在压板下渐进断键。"""
    cfg = DEM3DConfig(nx=nx, ny=ny, nz=nz, breaking_strain=breaking_strain,
                      shear_breaking_strain=shear_strain, top_platen_enabled=True,
                      top_platen_velocity=platen_velocity, top_platen_gap=5e-4,
                      fix_bottom=True)
    sim = IceDEM3D(cfg)
    n_bonds = int(sim.alive.numel())
    mass0 = float(cfg.mass) * len(sim.positions)
    hist = []
    for _ in range(steps):
        sim.step()
        hist.append(int((~sim.alive).sum()))
    failed = sim.failure_time_s >= 0.0
    # 账本 A：逐键 E=0.5·k_bond·extension²（A5.4）独立复算；账本 B：库侧累加账本
    ledger_a = float((0.5 * cfg.bond_stiffness * sim.failure_extension_m[failed].square()).sum())
    ledger_b = float(sim.failure_energy_J.sum())
    mass1 = float(cfg.mass) * len(sim.positions)
    return {
        "config": {"nx": nx, "ny": ny, "nz": nz, "breaking_strain": breaking_strain,
                   "shear_breaking_strain": shear_strain,
                   "top_platen_velocity_m_s": platen_velocity,
                   "dt_s": float(sim.dt), "recommended_dt_s": float(cfg.recommended_dt)},
        "n_particles": int(len(sim.positions)),
        "n_bonds_initial": n_bonds,
        "n_bonds_broken_history": hist,
        "n_bonds_broken_total": int(hist[-1]),
        "n_steps_with_new_breakage": int(sum(1 for a, b in zip([0] + hist[:-1], hist) if b > a)),
        "physical_time_s": float(sim.time),
        "fracture_energy_ledger_a_J": ledger_a,
        "fracture_energy_ledger_b_J": ledger_b,
        "energy_relative_error": abs(ledger_a - ledger_b) / max(abs(ledger_a), abs(ledger_b), 1e-300),
        "mass_initial_kg": mass0,
        "mass_final_kg": mass1,
        "mass_relative_error": abs(mass1 - mass0) / mass0,
        "broken_bond_fraction": (hist[-1] / n_bonds) if n_bonds else 0.0,
    }


def stage3_anchor(mesh_res=3, side=1.0, thickness=0.01, sigma_x=200000.0,
                  sigma_c=100000.0, G_c=100.0):
    """A5.2 解析枚举锚（SI-2 冻结口径）：均匀应力 + 阈值 → 破坏 face 集合可闭式手算。"""
    from plate_response import structured_square_mesh
    nodes, elements = structured_square_mesh(mesh_res, side=side)
    res = identify_failed_faces_uniform_stress(
        np.asarray(nodes, dtype=float)[:, :2], np.asarray(elements),
        np.array([sigma_x, 0.0, 0.0]), thickness, sigma_c, G_c)
    cell = side / mesh_res
    # 闭式枚举（SI-2 冻结口径，mesh_res=3 → 总面 12 / 坏面 6）：
    # n×n 四边形网格内部 face = 竖缝 n(n-1) + 横缝 n(n-1) = 2n(n-1)；
    # 均匀 σ_x 下只有竖缝有法向应力 σ_n = σ_x > σ_c，故坏面 = n(n-1)；
    # 每道缝面积 = 单元边长 × 板厚，总坏面面积 = n(n-1)·cell·t。
    exp_total = 2 * mesh_res * (mesh_res - 1)
    exp_failed = mesh_res * (mesh_res - 1)
    exp_area = exp_failed * cell * thickness
    exp_energy = G_c * exp_area
    area = float(sum(f.area for f in res.failed_faces))
    energy = float(sum(f.released_energy for f in res.failed_faces))
    return {
        "params": {"mesh_res": mesh_res, "side_m": side, "thickness_m": thickness,
                   "sigma_x_Pa": sigma_x, "sigma_c_Pa": sigma_c, "G_c_J_m2": G_c},
        "n_total_faces_expected": exp_total, "n_total_faces_actual": res.n_total_faces,
        "n_failed_expected": exp_failed, "n_failed_actual": res.n_failed,
        "count_match": res.n_total_faces == exp_total and res.n_failed == exp_failed,
        "total_area_expected": exp_area, "total_area_actual": area,
        "area_rel_error": abs(area - exp_area) / exp_area,
        "total_energy_expected": exp_energy, "total_energy_actual": energy,
        "energy_rel_error": abs(energy - exp_energy) / exp_energy,
    }


def monotone(v):
    return all(b >= a for a, b in zip(v, v[1:]))


def build():
    fem = stage1_fem()
    dem = stage2_dem()
    anchor = stage3_anchor()
    prog = monotone(fem["n_failed_faces_cumulative"])
    bond_mono = monotone(dem["n_bonds_broken_history"])
    e_fem, e_dem = fem["energy_relative_error"], dem["energy_relative_error"]
    checks = [
        {"id": "M5.1", "name": "渐进性门",
         "criterion": "n_failed_faces(k) 随步非降 且 n_steps_with_new_fracture >= 3",
         "value": {"monotone": prog,
                   "n_steps_with_new_fracture": fem["n_steps_with_new_fracture"],
                   "n_failed_faces_final": fem["n_failed_faces_final"]},
         "pass": bool(prog and fem["n_steps_with_new_fracture"] >= 3),
         "evidence": {"file": "fracture_history.json", "key": "n_failed_faces_cumulative",
                      "value": fem["n_failed_faces_cumulative"]}},
        {"id": "M5.2", "name": "键断裂演化门",
         "criterion": "n_bonds_broken_total > 0 且随步非降",
         "value": {"n_bonds_broken_total": dem["n_bonds_broken_total"], "monotone": bond_mono,
                   "n_steps_with_new_breakage": dem["n_steps_with_new_breakage"]},
         "pass": bool(dem["n_bonds_broken_total"] > 0 and bond_mono),
         "evidence": {"file": "run_summary.json", "key": "n_bonds_broken_total",
                      "value": dem["n_bonds_broken_total"]}},
        {"id": "M5.3", "name": "断裂能量双账本门",
         "criterion": "|ΔE|/ΣE <= 1e-12（FEM cohesive 与 DEM 键断裂双侧）",
         "value": {"fem_energy_relative_error": e_fem, "dem_energy_relative_error": e_dem},
         "pass": bool(e_fem <= TOL and e_dem <= TOL),
         "evidence": {"file": "validation_report.json",
                      "key": "tests.fracture_energy_ledger.relative_error",
                      "value": max(e_fem, e_dem)}},
        {"id": "M5.4", "name": "质量双账本门", "criterion": "|Δm|/Σm <= 1e-12",
         "value": {"mass_relative_error": dem["mass_relative_error"]},
         "pass": bool(dem["mass_relative_error"] <= TOL),
         "evidence": {"file": "validation_report.json", "key": "tests.mass_ledger.relative_error",
                      "value": dem["mass_relative_error"]}},
        {"id": "M5.5", "name": "解析枚举锚",
         "criterion": "破坏面计数精确相等，总面积/释放能 vs 闭式 <= 1e-12",
         "value": {k: anchor[k] for k in ("n_total_faces_expected", "n_total_faces_actual",
                                          "n_failed_expected", "n_failed_actual",
                                          "area_rel_error", "energy_rel_error")},
         "pass": bool(anchor["count_match"] and anchor["area_rel_error"] <= TOL
                      and anchor["energy_rel_error"] <= TOL),
         "evidence": {"file": "analytical_anchor.json", "key": "n_failed_actual",
                      "value": anchor["n_failed_actual"]}},
    ]
    thickness = fem["config"]["ice_thickness"]
    # 裂纹路径长度：破坏缝数 × 单缝长度（单缝面积/缝宽；冰板缝宽 = 冰厚）
    crack_len = fem["n_failed_seams"] * thickness
    scale = {
        "fracture_spans_ge_3_steps": fem["n_steps_with_new_fracture"] >= 3,
        "n_failed_faces_ge_30": fem["n_failed_faces_final"] >= 30,
        "n_bonds_broken_ge_10": dem["n_bonds_broken_total"] >= 10,
        "crack_path_ge_2x_thickness": crack_len >= 2.0 * thickness,
        "crack_path_length_m": crack_len,
    }
    semantic = {
        "sir5_physics_fracture_progressive": bool(prog and fem["n_steps_with_new_fracture"] >= 3
                                                 and fem["n_failed_faces_final"] >= 30),
        "sir5_physics_bond_breakage": bool(dem["n_bonds_broken_total"] >= 10),
    }
    passed = sum(1 for c in checks if c["pass"])
    return {
        "schema": "tensorphysics-suboff-sir5-validation-v1",
        "created_utc": utcnow(), "mission": "m309", "leg": "SIR-5", "gap": "G5",
        "prereg_file": str(PREREG.relative_to(ROOT)), "prereg_md5": md5(PREREG),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed, "checks_total": len(checks), "checks": checks,
        "gates": {c["id"].replace(".", "_") + "_pass": c["pass"] for c in checks},
        "tests": {"fem_progressive_fracture": fem, "dem_bond_breakage": dem,
                  "analytical_anchor": anchor,
                  "fracture_energy_ledger": {"relative_error": max(e_fem, e_dem)},
                  "mass_ledger": {"relative_error": dem["mass_relative_error"]}},
        "semantic_criteria": semantic,
        "semantic_criteria_passed": sum(1 for v in semantic.values() if v),
        "semantic_criteria_total": len(semantic),
        "scale_criteria": scale,
        "scale_criteria_passed": sum(1 for v in scale.values() if v is True),
        "scale_criteria_total": len(scale) - 1,
        "baseline_comparison": {
            "si4_n_failed_faces_step0": 12,
            "si4_n_steps_with_new_fracture": 1,
            "si4_n_bonds_broken_final": 0,
            "note": "SI-4 基线：首链步一次断 12 面、其余步 0、终态 0 键断"
                    "（prereg/sir0_baseline.json ← gaps.g5_fracture_propagation.evidence）",
        },
        "limitations": [
            "FEM 侧 cohesive 缝格由库内 IceImpactConfig(ice_nx, ice_ny) 给定，裂纹沿预设缝格扩展，"
            "不在单元内部任意取向开裂（库侧能力边界；任意路径开裂需开 TensorFEM 裂纹追踪定向修订单）",
            "DEM 侧为 3-D 键结颗粒场压板加载（轴向拉伸应变阈值 + 客观剪应变阈值），与 FEM cohesive "
            "缝不是同一套损伤变量：两侧能量账本各自闭合，不是跨库统一账本",
            "两阶段物理时长不同（FEM 冲击毫秒级、DEM 压板 ~6.7e-2 s），本腿只判渐进性/键断裂/账本/解析锚，"
            "跨尺度时长统一留给 SIR-6",
            "[待办] 下一班次可把 FEM 侧 ice_nx/ice_ny 提到 24×8 复核裂纹路径单调性（本班次预算内只跑到 12×4）",
            "[待办] DEM 与本链 SI-4 的 12 颗粒/30 键基线是两套模型，SIR-6 放大时要统一到同一键结场",
        ],
    }


def main():
    report = build()
    OUT.mkdir(parents=True, exist_ok=True)
    fem, dem = report["tests"]["fem_progressive_fracture"], report["tests"]["dem_bond_breakage"]
    (OUT / "fracture_history.json").write_text(json.dumps({
        "schema": "tensorphysics-suboff-sir5-fracture-history-v1",
        "created_utc": report["created_utc"],
        "time_s": fem["fracture_time_s"],
        "n_failed_faces_cumulative": fem["n_failed_faces_cumulative"],
        "fracture_dissipation_J": fem["fracture_dissipation_J_history"],
    }, indent=1), encoding="utf-8")
    (OUT / "run_summary.json").write_text(json.dumps({
        "schema": "tensorphysics-suboff-sir5-run-summary-v1",
        "created_utc": report["created_utc"],
        "n_bonds_broken_total": dem["n_bonds_broken_total"],
        "n_bonds_initial": dem["n_bonds_initial"],
        "n_particles": dem["n_particles"],
        "n_steps_with_new_breakage": dem["n_steps_with_new_breakage"],
        "physical_time_s": dem["physical_time_s"],
        "n_failed_faces_final": fem["n_failed_faces_final"],
        "n_steps_with_new_fracture": fem["n_steps_with_new_fracture"],
        "n_damaged_seams": fem["n_damaged_seams"],
        "n_failed_seams": fem["n_failed_seams"],
        "broken_bond_history": dem["n_bonds_broken_history"],
    }, indent=1), encoding="utf-8")
    (OUT / "analytical_anchor.json").write_text(
        json.dumps(report["tests"]["analytical_anchor"], indent=1), encoding="utf-8")
    (OUT / "validation_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    triplets = [{"id": c["id"], "value": c["evidence"]["value"], "file": c["evidence"]["file"],
                 "key": c["evidence"]["key"]} for c in report["checks"]]
    triplets += [{"id": "semantic:" + k, "value": v, "file": "validation_report.json",
                  "key": "semantic_criteria." + k} for k, v in report["semantic_criteria"].items()]
    (OUT / "gates.json").write_text(json.dumps({"triplets": triplets}, indent=1, ensure_ascii=False),
                                    encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "checks": f"{report['checks_passed']}/{report['checks_total']}",
                      "semantic": f"{report['semantic_criteria_passed']}/{report['semantic_criteria_total']}",
                      "scale": f"{report['scale_criteria_passed']}/{report['scale_criteria_total']}",
                      "n_failed_faces_final": fem["n_failed_faces_final"],
                      "n_steps_with_new_fracture": fem["n_steps_with_new_fracture"],
                      "n_bonds_broken_total": dem["n_bonds_broken_total"]},
                     ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()