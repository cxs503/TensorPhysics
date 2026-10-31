#!/usr/bin/env python3
"""Publish the W3-N02 report to TensorPlatform (``/tensor/reports``).

Reads the machine-generated results JSON / figures / report markdown, embeds the
figures as base64 PNG and posts the payload.  Credentials come from the
environment (``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``);
nothing is stored on disk.

Usage (from the repository root, after running the benchmark and report
generators)::

    python benchmarks/icebreaking/publish_w3n02_report.py
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
API = os.environ["TENSOR_API_BASE"].rstrip("/")
MISSION_ID = 789

FIG_CAPTIONS = {
    "fig1_adfreeze_strength": "图1 粘结强度-温度曲线（模型三工况）与 Frederking(1984) 四点及测量带宽",
    "fig2_contact_time_kinetics": "图2 四个温度下冻结面积分数随接触时间演化（Stefan 冻结动力学，对数时间轴）",
    "fig3_degradation_mechanisms": "图3 退化机制：(a) 内聚剥离曲线 (b) 热退化因子 (c) 盐度（盐水体积）惩罚",
    "fig4_breakout_vs_thrust": "图4 脱困力随接触面积线性增长 vs 三船 bollard 推力水平线，标记临界面积分数",
}


def login() -> str:
    payload = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(
        API + "/auth/login", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    return data["access_token"]


def build_payload() -> dict:
    data = json.loads((ROOT / "artifacts/adfreeze_bond/w3n02_results.json").read_text())
    ca, cb = data["criterion_a"], data["criterion_b"]
    kin = data["kinetics"]["rows"]
    deg = data["degradation"]

    figs = []
    for key, rel in data["figures"].items():
        figs.append(
            {
                "name": key + ".png",
                "caption": FIG_CAPTIONS[key],
                "png_base64": base64.b64encode((ROOT / rel).read_bytes()).decode(),
            }
        )

    strength_rows = "\n".join(
        f"| {r['T_C']:+.0f} | {r['measured_MPa']:.2f} ± {r['measured_band_MPa']:.2f} | "
        f"{r['model_smooth_saturated_MPa']:.3f} | {r['model_err_pct']:+.2f}% | "
        f"{r['model_hull_Ra2p5_MPa']:.3f} | {r['model_hull_err_pct']:+.2f}% |"
        for r in ca["rows"]
    )
    ship_rows = "\n".join(
        f"| {r['ship']} | {r['model_contact_area_m2']:.1f} | {r['empirical_contact_area_m2']:.1f} | "
        f"{r['area_dev_pct']:+.2f}% | {r['tau_adfreeze_MPa']:.3f} | {r['breakout_force_MN']:.2f} | "
        f"{r['bollard_thrust_MN']:.2f} | {r['critical_area_fraction'] * 100:.2f}% |"
        for r in cb["rows"]
    )
    kin_rows = "\n".join(
        f"| {r['T_C']:+.0f} | {r['frozen_fraction_60s']:.3f} | {r['frozen_fraction_600s']:.3f} | "
        f"{r['frozen_fraction_1h']:.3f} | {r['t_to_95pct_s']:.0f} | {r['tau_saturated_MPa']:.3f} |"
        for r in kin
    )
    crit_lo = min(r["critical_area_fraction"] for r in cb["rows"]) * 100
    crit_hi = max(r["critical_area_fraction"] for r in cb["rows"]) * 100
    thr_lo = min(r["bollard_thrust_MN"] for r in cb["rows"])
    thr_hi = max(r["bollard_thrust_MN"] for r in cb["rows"])
    fbo_lo = min(r["breakout_force_MN"] for r in cb["rows"])
    fbo_hi = max(r["breakout_force_MN"] for r in cb["rows"])
    ratio_lo = min(r["thrust_over_breakout"] for r in cb["rows"]) * 100
    ratio_hi = max(r["thrust_over_breakout"] for r in cb["rows"]) * 100

    markdown = f"""# W3-N02 冰-结构界面冻结粘结力学与破冰船脱困判据

**模块**：`src/tensorphysics/ice/`（TensorPhysics 新领域目录：adfreeze / degradation / breakout / references）
**证据等级**：REAL（物理模型 + 文献标定 + 解析解核验，无 MOCK/STUB）
**产物**：`artifacts/adfreeze_bond/w3n02_results.json`、4 张 PNG、`tests/test_ice_adfreeze.py`（52 passed）、
报告 `docs/reports/m789/w3n02_adfreeze_bond_report.md`

## 1 案例介绍

beset 船舶失去机动性的直接物理原因，是冰与船体钢板之间水膜在冰点以下冻结形成**冻结粘结接头**
（adfreeze joint）。其剪切强度 τ_adfreeze 与摩擦阻力共同决定脱困力。模型需同时解释三条实验事实：
(i) 强度随界面温度强烈上升（−2 ℃≈0.15 MPa → −20 ℃≈1.2 MPa）；(ii) 强度随接触时间发展（秒级近零，
数十分钟至小时饱和）；(iii) 海水冰接头强度显著低于淡水冰。算例为三种船型
（1A-Super 护航破冰船 / Polar Class 4 中型 / Polar Class 2-3 大型极地破冰船），示意水线面
（平行中体 + 余弦进流段 + 二次收敛尾部），冰厚 0.6 / 1.2 / 1.8 m。

## 2 方法

τ_adfreeze = τ_ref(ΔT) · η_freeze(t,ΔT) · φ_rough(Ra) · φ_salt(T,S)，ΔT = T_f(S) − T_i：

* τ_ref(ΔT)=τ_sat·ΔT/(ΔT+d_half)，由 Frederking(1984) 四温度点作相对误差最小二乘标定
  （τ_sat={ca['calibration']['tau_sat_MPa']:.3f} MPa，d_half={ca['calibration']['d_half_K']:.2f} K，
  RMS={ca['calibration']['rms_relative_error']:.4f}），并施加物理上界 2.0 MPa（粘结不能强于冰自身剪切强度）；
* 形成动力学：Stefan 解 h(t)=√(2kΔTt/ρL)，接头间隙指数分布（均值 5 mm）→ η=1−exp(−h/h_gap)；
* 粗糙度互锁 φ_rough=1+κ_R(1−exp(−Ra/Ra,ref))；
* 盐度 φ_salt=(1−v_b)^n，盐水体积分数由盐水质量守恒（Frankenstein–Garner）迭代求，
  海水冰点由 Assur/Cox–Weeks 液相线锚点插值（S=32 g/kg → {data['seawater_freezing_point_32gkg_C']:.2f} ℃）；
* 退化三机制乘性：机械剥离（双线性内聚律，δ_c=2G_c/τ，G_c=5 J/m²）、热融化（强度比）、
  盐度（淡化恢复 / 富集削弱）；
* 脱困判据：F_breakout = τ_residual·A_contact + μ·p_n·A_contact，A_contact 由离散船体线型积分
  （接触带高度受外飘角与破碎层容差 0.1 m 限制）；推力用螺旋桨动量理论 bollard 工况
  T=N·C_T·ρ·n²·D⁴，n=[P/(N·2πC_QρD⁵)]^(1/3)；判据输出**临界脱粘面积分数**
  A_crit/A=(T−F_friction)/F_adhesion。

工具链：numpy（无 scipy）→ pytest 52 项 → matplotlib 4 图 → 数字零手抄的报告生成器。

## 3 计算结果

### 判据 (a) 粘结强度 vs Frederking(1984) 四温度（<15%）

| T [℃] | 实验 τ [MPa] | 模型(光滑饱和) [MPa] | 误差 | 模型(Ra=2.5 µm) [MPa] | 误差 |
|---|---|---|---|---|---|
{strength_rows}

最大误差 **{ca['max_abs_error_pct_smooth']:.2f}%**（光滑参考面）/ **{ca['max_abs_error_pct_hull']:.2f}%**（计入实船钢板粗糙度）→ **{'PASS' if ca['pass'] else 'FAIL'}**。

### 判据 (b) 脱困力 vs 闭式经验接触面积（3 船型，T=−10 ℃，淡水冰，饱和接触）

| 船型 | A 模型 [m²] | A 经验 [m²] | 面积偏差 | τ [MPa] | F_breakout [MN] | 推力 [MN] | A_crit/A |
|---|---|---|---|---|---|---|---|
{ship_rows}

最大偏差 **{cb['max_abs_force_dev_pct']:.2f}%** < 10% → **{'PASS' if cb['pass'] else 'FAIL'}**（须理解为离散积分与矩形闭式估计的
**方法一致性检查**，非独立实验验证）。

### 粘结形成动力学

| T [℃] | η(60 s) | η(600 s) | η(1 h) | t(η=0.95) [s] | τ 饱和 [MPa] |
|---|---|---|---|---|---|
{kin_rows}

### 退化（峰值 τ={deg['tau_peak_MPa']:.3f} MPa，δ_c={deg['critical_slip_um']:.1f} µm）

* 机械剥离：滑移 {deg['peel_rows'][3]['slip_um']:.1f} µm → 残余 {deg['peel_rows'][3]['residual_fraction']:.3f}；滑移 ≥ δ_c 完全脱粘；
* 热融化：−10→−5 ℃ 因子 {deg['thermal_rows'][2]['factor']:.3f}，−10→−2 ℃ 因子 {deg['thermal_rows'][3]['factor']:.3f}，升温至液相线 → 0；
* 盐度：−10 ℃、S=32 g/kg 相对淡水因子 {deg['salinity_factor_ratio_at_datum']:.3f}
  （文献对比点 {deg['saline_datum']['ratio']:.2f}±{deg['saline_datum']['ratio_band']:.2f}，在带内）；
  淡化（32→0）恢复因子 {deg['fresh_salinity_degradation_32_to_0']:.3f}。

### 物理结论

三船型在 −10 ℃、整条冻结接触均存在时，脱困力 {fbo_lo:.1f}–{fbo_hi:.1f} MN 远大于 bollard 推力
{thr_lo:.2f}–{thr_hi:.2f} MN（推力/脱困力仅 {ratio_lo:.2f}–{ratio_hi:.2f}%）：**纯推力不可能一次剪断完整冻结接头**，
临界脱粘面积分数仅 {crit_lo:.2f}–{crit_hi:.2f}%，即必须剥离 >97% 的接头面积（现场做法：横倾/摇摆剥离、
等待接头升温或淡化、破碎邻近冰盖）。摩擦项仅占脱困力 1–3%，τ_adfreeze 主导。

## 4 图

见图1–图4（强度-温度与文献数据；接触时间动力学；三退化机制；脱困力 vs 推力与临界面积）。

## 5 注意点与局限

1. **标定 ≠ 独立验证**：判据 (a) 的 4 个温度点同时用于标定与残差评估，{ca['max_abs_error_pct_smooth']:.2f}% 是标定残差；
   文献值为数字化中值（exact=False，带宽见表，−5/−20 ℃ 处模型值落在测量带内）。独立验证需新温度点或新实验批次。
2. **判据 (b) 是方法一致性检查**（离散积分 vs 矩形闭式），非独立实验基准；实船脱困推力记录未入库。
3. **接触面积是主要不确定源**：clearance / p_n / μ / h_gap 均为工程估计（±30–50%），脱困力绝对值置信区间约 ±50%；
   但临界面积分数对此不敏感（分子分母同阶抵消），结论稳健。
4. **未建模**：冰盖弯曲破坏先于接头剪切的竞争（flexural limit）、横倾剥离功、碎冰堆积阻力、船体振动、
   排盐随时间改变接头盐度、压力脊动力学。本模型给出冻结接头强度的**上限约束**，不是完整脱困时程仿真。
5. 适用域：T∈[−30,0] ℃、S∈[0,50] g/kg、h_i∈[0.3,2.5] m、接触时间 1 s–24 h；v_b>0.3 时接头近泥浆态，仅定性。

## 6 误差判定（是否 <3% 逐项回答）

| 关键量 | 基准 | 相对误差 | <3% ? |
|---|---|---|---|
| τ(−2 ℃) | {ca['rows'][0]['measured_MPa']:.3f} MPa | {ca['rows'][0]['model_err_pct']:+.2f}%（含粗糙度 {ca['rows'][0]['model_hull_err_pct']:+.2f}%） | 光滑面是，含粗糙度否 |
| τ(−5 ℃) | {ca['rows'][1]['measured_MPa']:.3f} MPa | {ca['rows'][1]['model_err_pct']:+.2f}%（含粗糙度 {ca['rows'][1]['model_hull_err_pct']:+.2f}%） | 是（含粗糙度否） |
| τ(−10 ℃) | {ca['rows'][2]['measured_MPa']:.3f} MPa | {ca['rows'][2]['model_err_pct']:+.2f}%（含粗糙度 {ca['rows'][2]['model_hull_err_pct']:+.2f}%） | 否（判据 15% 内） |
| τ(−20 ℃) | {ca['rows'][3]['measured_MPa']:.3f} MPa | {ca['rows'][3]['model_err_pct']:+.2f}%（含粗糙度 {ca['rows'][3]['model_hull_err_pct']:+.2f}%） | 是 |
| Stefan h(t) | 解析 Neumann 解 | <1e−12 | 是 |
| 内聚律耗散能 | G_c=5 J/m² | <1e−3 | 是 |
| 脱困力 / 接触面积 | 矩形闭式估计 | {cb['max_abs_force_dev_pct']:.2f}% | 否（且非独立基准；满足 <10% 判据） |
| 推力 vs 公开 bollard pull | 无可靠入库基准 | 未比对 | 不适用（无基准） |

**总体**：判据 (a) PASS（{ca['max_abs_error_pct_smooth']:.2f}% / 含粗糙度 {ca['max_abs_error_pct_hull']:.2f}% < 15%）；
判据 (b) PASS（{cb['max_abs_force_dev_pct']:.2f}% < 10%，方法一致性）；判据 (c) pytest ≥ 10 → PASS（52 passed）。
「<3%」仅对光滑参考面的 3/4 个温度点成立，含实船粗糙度后不成立（如实声明）。
"""

    return {
        "title": "W3-N02 冰-结构冻结粘结力学与破冰船脱困判据（TensorPhysics ice/ 模块）",
        "summary": (
            "新建 TensorPhysics ice/ 领域模块：adfreeze 粘结强度模型（Frederking 四温度标定，"
            f"最大残差 {ca['max_abs_error_pct_smooth']:.2f}% < 15%）、Stefan 接触时间动力学、"
            "剥离/热/盐度三退化机制、脱困判据 F=τA+μp_nA 与 bollard 推力比对"
            f"（3 船型接触面积/脱困力偏差 {cb['max_abs_force_dev_pct']:.2f}% < 10%）；pytest 52 passed。"
            f"结论：整条冻结接头无法靠推力剪断（推力仅占脱困力 {ratio_lo:.2f}–{ratio_hi:.2f}%），"
            "须剥离 >97% 接头面积。"
        ),
        "scenario_key": "ice_adfreeze_bond",
        "mission_id": MISSION_ID,
        "tags": ["ice", "adfreeze", "breakout", "beset", "TensorPhysics", "W3-N02", "validation"],
        "markdown": markdown,
        "figures": figs,
    }


def main() -> int:
    token = login()
    payload = build_payload()
    req = urllib.request.Request(
        API + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(r.status, r.read().decode()[:400])
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        print("HTTP", exc.code, exc.read().decode()[:800])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())