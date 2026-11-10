"""Publish the paper-grade progress report of node p380w2-n06 to TensorPlatform.

Reads ``artifacts/ice_structure_dem_fem/verification.json`` (numbers injected,
never hand-copied) plus the generated figures, logs in with the agent
credentials from the environment, and POSTs to ``/tensor/reports``.

Usage::

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
    PYTHONPATH=src python benchmarks/ice_structure_dem_fem/publish_report.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "ice_structure_dem_fem"
MISSION_ID = 10884


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _pct(x: float) -> str:
    return f"{100.0 * x:+.2f}%"


def build_markdown(data: dict) -> str:
    crit = data["criteria"]
    a, b, c, d, e, f = (crit[k] for k in (
        "a_dem_peak_vs_lindqvist", "b_elastoplastic_plate", "c_interface",
        "d_iacs_margin", "e_mesh_convergence", "f_negative_control"))
    ship = data["ship_panel"]
    el = b["elastic_limit"]
    pb = b["plastic_branch_demonstration"]
    L = []
    add = L.append
    add("## 1. 案例介绍")
    add("")
    add("冰-结构耦合 DEM-FEM 交叉验证面：DEM 冰载荷谱驱动弹塑性船体板。")
    add(f"极级 {ship['polar_class']}，设计冰厚 {ship['h_I_m']} m；局部板格 "
        f"{ship['panel_m'][0]} m × {ship['panel_m'][1]} m，板厚 "
        f"{1000*ship['plate_thickness_m']:.0f} mm，σ_y = {ship['material']['sigma_y_Pa']/1e6:.0f} MPa，"
        f"E = {ship['material']['E_Pa']/1e9:.0f} GPa；IACS 冰斑 {ship['patch_m']} m。"
        "工况 (h, v) = " + "、".join(f"({r['h_m']} m, {r['speed_ms']} m/s)" for r in a["rows"]) + "。")
    add("")
    add("## 2. 怎么做的")
    add("")
    add("1) DEM（`dem_ice.py`）：二维竖直切片键合粒子（LDEM）冰板，Euler-Bernoulli 梁键，"
        "弯扭破坏判据 `|N|/A + 6|M|/d² ≥ σ_f`；Winkler 浮力 + 冰重；艏柱以固定外飘角推进，"
        "法向力被压溃强度 σ_c 饱和 + 库仑摩擦；累计压溃功超阈值即侵蚀（形成压溃坑）；显式中心差分。"
        "输出单位水线长冰阻力 F_x(t) [N/m]。")
    add("2) 参考律（`lindqvist_ref.py`）：薄适配 `tensorphysics.ice.resistance.level_ice_resistance`，"
        "按文档化的“线力”读法（R_crush 无 B 因子 → N/m；R_break、R_submersion 除 B）。")
    add("3) 耦合（`coupling.py`）：冰斑压力 p(t)=F_x(t)/h_p（h_p=0.6h）；到结构增量的传递为冲量守恒"
        "（分段线性精确子循环），并给同步误差与零阶保持峰值损失。")
    add("4) FEM（`fem_plate.py`）：Mindlin Q4（剪切选择性减缩积分）、厚度 5 层 Simpson、每层三维 J2 "
        "径向回映 + 平面应力凝聚（ε_z 迭代使 σ_z=0）、层切线取自该映射的中心差分；沿传递载荷历史增量 "
        "Newton（带回溯线搜索）。Krichhoff Navier 级数作为独立弹性基准。")
    add("5) 验收机（`verification.py`）跑全部判据并落 JSON，本文数字全部由该 JSON 程序化注入。")
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 DEM 冰载荷谱 vs Lindqvist（判据 a）")
    add("")
    add("| h [m] | v [m/s] | DEM 峰值 [kN/m] | DEM 稳态均值 [kN/m] | Lindqvist 峰值/m [kN/m] | 偏差 | 浮梁解析 [kN/m] |")
    add("|---|---|---|---|---|---|---|")
    for r in a["rows"]:
        add(f"| {r['h_m']} | {r['speed_ms']} | {r['dem_peak_N_per_m']/1e3:.1f} | "
            f"{r['dem_steady_mean_N_per_m']/1e3:.1f} | "
            f"{r['lindqvist_peak_per_metre_N_per_m']/1e3:.1f} | "
            f"{_pct(r['dev_peak_vs_lindqvist'])} | "
            f"{r['analytic_floating_beam_break_N_per_m']/1e3:.1f} |")
    add("")
    add(f"最大偏差 {_pct(a['max_abs_deviation'])}（目标 <15%）→ "
        f"{'PASS' if a['passed'] else 'honest FAIL'}；幂律指数 DEM "
        f"{a['scaling_exponent_dem']:.2f} vs Lindqvist {a['scaling_exponent_lindqvist']:.2f}。")
    add("")
    add("### 3.2 弹塑性板（判据 b）")
    add("")
    add(f"弹性极限（网格 {el['mesh']}，p={el['probe_pressure_Pa']/1e3:.0f} kPa）：FEM max|σx| = "
        f"{el['fem_max_abs_sigma_x_Pa']/1e6:.2f} MPa vs Navier "
        f"{el['navier_max_surface_stress_Pa']/1e6:.2f} MPa（{_pct(el['deviation_sigma'])}）；"
        f"跨中挠度 {el['fem_centre_deflection_m']*1e3:.3f} mm vs "
        f"{el['navier_deflection_m']*1e3:.3f} mm（{_pct(el['deviation_deflection'])}）。")
    add(f"塑性支路（1.5× 首屈服压力 = {pb['applied_peak_pressure_Pa']/1e6:.2f} MPa）：von Mises 峰值 "
        f"{pb['peak_von_mises_stress_Pa']/1e6:.1f} MPa，等效塑性应变 "
        f"{pb['max_equivalent_plastic_strain']:.3e}，收敛={pb['converged']}。")
    add(f"HSVA 板响应对比：**{b['hsva_comparison']['status']}**——{b['hsva_comparison']['reason']}")
    add("")
    add("### 3.3 耦合接口（判据 c）")
    add("")
    add("| h [m] | v [m/s] | 同步冲量误差 | 子循环比 | 接口能量残差 | W_ext [J] |")
    add("|---|---|---|---|---|---|")
    for r in c["rows"]:
        add(f"| {r['h_m']} | {r['speed_ms']} | {r['sync_impulse_error']:.2e} | "
            f"{r['subcycling_factor']:.1f} | {r['interface_energy_residual']:.2e} | "
            f"{r['W_external_J']:.3f} |")
    add("")
    add(f"最大同步误差 {c['max_sync_impulse_error']:.2e}（目标 <1%），最大能量残差 "
        f"{c['max_interface_energy_residual']:.2e}（目标 <5%）→ "
        f"{'PASS' if c['passed'] else 'FAIL'}。")
    add("")
    add("### 3.4 IACS 安全裕度（判据 d）")
    add("")
    add(f"PC6 设计压力（冰斑面积 {d['patch_area_m2']:.3f} m²）p_IACS = "
        f"{d['iacs_design_pressure_Pa']/1e6:.3f} MPa。")
    add("")
    add("| h [m] | v [m/s] | DEM 峰值斑压 [MPa] | 压力裕度 | FEM 峰值应力 [MPa] | 应力裕度 |")
    add("|---|---|---|---|---|---|")
    for r in d["rows"]:
        add(f"| {r['h_m']} | {r['speed_ms']} | {r['dem_peak_patch_pressure_Pa']/1e6:.3f} | "
            f"{r['pressure_margin']:.2f} | {r['fem_peak_stress_Pa']/1e6:.1f} | "
            f"{r['stress_margin']:.2f} |")
    add("")
    add(f"最小应力裕度 {d['min_stress_margin']:.2f}。")
    add("")
    add("### 3.5 网格收敛（判据 e）")
    add("")
    add("| 网格 | 单元数 | 峰值 |σx| [MPa] | 相对变化 |")
    add("|---|---|---|---|")
    for r in e["rows"]:
        add(f"| {r['mesh'][0]}×{r['mesh'][1]} | {r['n_elements']} | "
            f"{r['peak_abs_sigma_x_Pa']/1e6:.2f} | {_pct(r['relative_change_vs_coarse'])} |")
    add("")
    add(f"细网格 vs 粗网格 {_pct(e['change_fine_vs_coarse'])}；细 vs 中 "
        f"{_pct(e['change_fine_vs_medium'])}（目标 <5%）→ {'PASS' if e['passed'] else 'FAIL'}。")
    add("")
    add("### 3.6 负控（判据 f）")
    add("")
    add(f"零冰厚 → 零载荷：峰值斑压 {f['zero_load_peak_pressure_Pa']:.1f} Pa，跨中挠度 "
        f"{f['zero_load_centre_deflection_m']:.3e} m，峰值应力 {f['zero_load_peak_stress_Pa']:.3e} Pa → "
        f"{'PASS' if f['passed'] else 'FAIL'}（目标 <1e-9）。")
    add("")
    add("## 4. 结果图")
    add("")
    add("见内嵌 figures：载荷谱五工况、峰值对比、耦合响应、弹性验证 1:1、网格收敛。")
    add("")
    add("## 5. 注意点")
    add("")
    add("* 判据 (a) 为 **honest FAIL**：DEM 峰值系统性低于 Lindqvist（最差 "
        f"{abs(a['max_abs_deviation'])*100:.0f}%）。归因：Lindqvist R_crush 的接触斑宽 b_c=0.32h 是"
        "半经验常数，而 DEM 的接触斑由几何+强度涌现，且 DEM 不重现 h² 标度（实测指数 "
        f"{a['scaling_exponent_dem']:.2f}）。")
    add("* 俯视（水平面内）格构 DEM 先试后弃：该面内不存在面外弯曲与浸水两大机制，低估约两个数量级；"
        "否定结论已记录，避免重复。")
    add("* 板模型无薄膜耦合（小挠度），后屈服薄膜强化与大挠度重分布缺失，塑性支路仅小应变有效。")
    add("* 分层应变能积分存在常数因子偏差（JSON 中 `layer_quadrature_residual`），故接口能量判据用"
        "离散功-能闭合（`work_energy_closure`），**该未决项如实披露**。")
    add("* 载荷传递为单向（DEM→FEM），无结构对破冰过程的反馈；双向分区的稳定性面见 "
        "`tensorphysics.ice_structure`。")
    add("* 冰载荷绝对精度 vs 全尺度为 **WITHHELD**（仓内无数字化实验记录）。")
    add("")
    add("## 6. 误差判定（关键量 vs 基准，逐项是否 <3%）")
    add("")
    add("| 量 | 基准 | 偏差 | <3%？ |")
    add("|---|---|---|---|")
    add(f"| 板峰值 |σx|（弹性极限） | Navier 级数 | {_pct(el['deviation_sigma'])} | "
        f"{'是' if abs(el['deviation_sigma'])<0.03 else '否'} |")
    add(f"| 板跨中挠度（弹性极限） | Navier 级数 | {_pct(el['deviation_deflection'])} | "
        f"{'是' if abs(el['deviation_deflection'])<0.03 else '否'} |")
    add(f"| DEM 冰载荷峰值 | Lindqvist 峰值/m | 最差 {abs(a['max_abs_deviation'])*100:.1f}% | 否 |")
    add(f"| 接口冲量 | 精确积分 | {c['max_sync_impulse_error']:.2e} | 是 |")
    add(f"| 板峰值应力（网格加密） | 中网格 | {_pct(e['change_fine_vs_medium'])} | "
        f"{'是' if abs(e['change_fine_vs_medium'])<0.03 else '否'} |")
    add("")
    add("判定汇总：" + json.dumps(data["summary"], ensure_ascii=False))
    return "\n".join(L)


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not base:
        raise SystemExit("TENSOR_API_BASE is not set")
    data = json.loads((ART / "verification.json").read_text())
    markdown = build_markdown(data)
    figures = []
    captions = {
        "fig_load_spectra.png": "五工况 DEM 冰阻力时程与 Lindqvist 峰值/均值线",
        "fig_peak_comparison.png": "峰值对比：DEM vs Lindqvist vs 浮梁解析",
        "fig_coupled_response.png": "传递斑压与板面应力响应（h=1 m, v=3 m/s）",
        "fig_elastic_verification.png": "弹性极限验证：FEM vs Navier 1:1",
        "fig_mesh_convergence.png": "板网格收敛（耦合载荷历史）",
    }
    for name, cap in captions.items():
        p = ART / name
        if p.exists():
            figures.append({
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            })
    payload = {
        "title": "冰-结构耦合 DEM-FEM 交叉验证面：Lindqvist 冰载荷谱 × 弹塑性板响应",
        "summary": (
            "键合粒子 DEM 冰载荷谱驱动弹塑性 Mindlin 板（分层 J2）；五工况验收："
            f"{data['summary']['n_passed']}/{data['summary']['n_criteria']} 判据通过，"
            f"honest FAIL={data['summary']['honest_fail']}；弹性极限 vs Navier "
            f"{_pct(data['criteria']['b_elastoplastic_plate']['elastic_limit']['deviation_sigma'])}，"
            f"接口同步误差 {data['criteria']['c_interface']['max_sync_impulse_error']:.1e}。"
        ),
        "scenario_key": "ice-structure-dem-fem-coupling",
        "mission_id": MISSION_ID,
        "tags": ["ice-structure", "DEM", "FEM", "elastoplastic", "Lindqvist",
                 "IACS", "coupling", "p380w2-n06"],
        "markdown": markdown,
        "figures": figures,
    }
    if user and password:
        token = _post(f"{base}/auth/login", {"username": user, "password": password}).get("access_token")
    else:
        token = None
    resp = _post(f"{base}/tensor/reports", payload, token)
    print(json.dumps(resp)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())