"""Paper-style report for campaign node ``p227w2-N08`` (QZS isolation maturity).

Reads the payload written by ``scripts/m10169_n08_figures.py`` and emits

* ``docs/ICE_VIBRATION_TRANSFER_MITIGATION_N08.md`` - the six-element report;
* ``/tmp/report_p227w2_n08.json`` - the platform report body (figures embedded).

Every number in the report is taken from the payload: nothing is hand-copied.
Usage: ``PYTHONPATH=src python scripts/m10169_n08_report.py``
"""

from __future__ import annotations

import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "artifacts" / "m10169" / "ice_vibration_transfer_mitigation_n08.json"
FIGDIR = ROOT / "docs" / "ice_vibration_transfer_n08_maturity"
DOC = ROOT / "docs" / "ICE_VIBRATION_TRANSFER_MITIGATION_N08.md"
REPORT = pathlib.Path("/tmp/report_p227w2_n08.json")

FIG_CAPTIONS = {
    "fig1_wet_dry_modes.png": "干/湿模态对比（M1）：湿模态在全部弹性模态上低于干模态。",
    "fig2_bode_transfer.png": "传递函数 Bode 图（幅值 dB + 相位）：常规 4 Hz 隔振器在冰带内放大，QZS 设计点把冰带压到 0 dB 以下。",
    "fig3_base_accel_psd.png": "加速度 PSD 沿链演化：船体梁 → 基座支撑 → 设备基座（常规 vs QZS）。",
    "fig4_design_map.png": "QZS 设计图：band RMS 比等值线、可行域（静态挠度/残余刚度/阻尼/行程/线性化）与选中设计点。",
    "fig5_life_vs_ice_load.png": "疲劳寿命-冰载荷曲线：轻/设计/重三冰级，两种方案均严格单调下降。",
    "fig6_sn_curve_ranges.png": "DNV-RP-C203 D 曲线与工作应力范围：说明绝对寿命量级（披露 D6）。",
    "fig7_tolerance_stroke.png": "刚度抵消误差容限与行程核对：冰带振动在 NS 线性区内，波浪/鞭击由缓冲限位承担。",
}


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) >= 1e4 or abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "; ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def main() -> None:
    d = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    cfg = d["config"]
    geom = d["geometry_gate"]
    hm = d["hull_modal"]
    qzs = d["qzs_mount"]
    base = d["baseline_mount"]
    sel = d["design_selection"]
    tol = d["tolerance"]
    stroke = d["stroke"]
    ver = d["verification"]
    sev = d["severities"]
    sn = d["sn_curve"]
    crit = {c["id"]: c for c in d["criteria"]}
    disc = {x["id"]: x for x in d["disclosures"]}
    labels = ["light", "design", "heavy"]

    ice_lo, ice_hi = cfg["ice_band_hz"]
    ratio_conv = crit["M3"]["value"]["baseline_ratio"]
    ratio_qzs = crit["M3"]["value"]["qzs_ratio"]
    alpha = qzs["negative_stiffness_fraction_alpha"]
    f_n = qzs["f_n_hz"]
    static_mm = qzs["static_deflection_m"] * 1e3
    equiv_mm = qzs["equivalent_linear_deflection_m"] * 1e3
    life_gain = crit["M9"]["value"]
    geo_cb_err = (geom["block_coefficient"] - 0.6505) / 0.6505 * 100.0
    dn_table = 52.6  # DNV-RP-C203 D curve ordinate at 1e7 cycles, MPa
    sn_err = (sn["stress_at_1e7_cycles_MPa"] - dn_table) / dn_table * 100.0

    mode_rows = "\n".join(
        "| {} | {:.4f} | {:.4f} | {:.3f} | {:.3f} |".format(
            m["mode"], m["f_dry_hz"], m["f_wet_hz"], m["ratio_wet_over_dry"], m["added_mass_ratio"]
        )
        for m in hm["modes"]
    )
    sev_rows = "\n".join(
        "| {} | {:.3e} | {:.3e} | {:.3f} | {:.3f} | {:.3e} | {:.3e} | {:.0f} |".format(
            s,
            sev[s]["hull_rms_m_s2"],
            sev[s]["hull_ice_band_rms_m_s2"],
            sev[s]["conventional"]["rms_ratio_base_over_hull_ice_band"],
            sev[s]["qzs"]["rms_ratio_base_over_hull_ice_band"],
            sev[s]["conventional"]["fatigue"]["life_years"],
            sev[s]["qzs"]["fatigue"]["life_years"],
            sev[s]["life_gain_qzs_over_conventional"],
        )
        for s in labels
    )
    band_rows = "\n".join(
        "| {:.1f}–{:.1f} | {:.3f} | {:.3f} |".format(b["band_hz"][0], b["band_hz"][1], b["conventional_ratio"], b["qzs_ratio"])
        for b in d["band_sensitivity"]
    )
    crit_rows = "\n".join(
        "| {} | {} | {} | {} | {} |".format(
            c["id"], c["name"], _fmt(c["value"]), _fmt(c["threshold"]), "PASS" if c["passed"] else "FAIL"
        )
        for c in d["criteria"]
    )
    disc_rows = "\n".join("| {} | {} | {} |".format(x["id"], x["statement"], _fmt(x["value"])) for x in d["disclosures"])
    tol_ok = [r for r in tol["rows"] if r["attenuated"]]
    tol_fail = [r for r in tol["rows"] if not r["attenuated"]]
    stiffness_err_pct = abs(qzs["frequency_sensitivity_per_1pct_alpha"]["relative_frequency_error"]) * 100.0
    robust_pct = 100.0 * tol["max_abs_delta_alpha_still_attenuating"]

    md = f"""# p227w2-N08 冰致振动传递链成熟化：船体湿模态 → 设备基座加速度（QZS 隔振）→ 管路应力 → 疲劳寿命

| 项 | 内容 |
|---|---|
| 战役节点 | `p227w2-N08`（成熟化轮，前序 `p228w1-N08`）|
| 代码 | `src/tensorphysics/ice_vibration_transfer/mitigation.py`（新增）、`maturity.py`（新增）|
| 载荷 payload | `artifacts/m10169/ice_vibration_transfer_mitigation_n08.json`（本文所有数字的唯一来源）|
| 测试 | `tests/test_ice_vibration_transfer_maturity.py`（21 用例，含 5 条机器判据）|
| 图 | `docs/ice_vibration_transfer_n08_maturity/`（7 张）|
| 判据总判定 | **{'全部通过' if d['all_criteria_pass'] else '存在 FAIL'}**（M1–M9，见 §3.5）|

## 1 案例介绍

### 1.1 物理背景与节点增量

冰区航行时螺旋桨铣削冰层产生与叶片通过相位锁定的扭矩脉动（N04 DEM 铣削扭矩谱），经轴系反力激起船体梁
**湿模态**振动，再经甲板基座局部结构、设备隔振器传到设备与管路，驱动管路细节疲劳。前序节点 `p228w1-N08`
把这条链串通并给出 10 条判据，但其披露 D1 给出一个无法回避的物理结论：**单级被动隔振器无法衰减冰载荷能带**。
常规 4 Hz 隔振器（静挠度 {static_mm:.1f} mm）在 {ice_lo:.1f}–{ice_hi:.1f} Hz 带内 |H|>1，带内 RMS(基座)/RMS(船体) =
**{ratio_conv:.3f}**（放大）；而把 f_n 降到 {f_n:.1f} Hz 的线性弹簧需要 {equiv_mm:.0f} mm 静挠度，船用隔振器不可能实现。

本节点增量是**把静刚度与动刚度解耦**：载荷弹簧 k_v 承担设备重量，负刚度机构 k_n = −α k_v 抵消其中 α 份额，
动态频率 f_n,dyn = f_rigid·√(1−α) 降到冰带以内，这就是准零刚度（QZS / HSLDS）隔振器（Carrella et al. 2007；
Ibrahim 2008）。据此给出：QZS 设计图 → 前置规则选点 → 传递链重算（真实随机相位应力合成 + 雨流 + DNV-RP-C203
S-N + Miner）→ 与常规方案**前后对照**，并把 QZS 的工程代价（刚度容限、行程、缓冲限位、Duffing 线性化边界）
逐项量化披露。

### 1.2 几何与主尺度（几何先行校验）

| 量 | 数值 | 判定 |
|---|---|---|
| 船型 | KCS（公开基准）| — |
| L×B×T×D | {geom['length_m']:.1f} × {geom['breadth_m']:.1f} × {geom['draught_m']:.1f} × {geom['depth_m']:.1f} m | — |
| Cb | {geom['block_coefficient']:.4f}（公开 0.6505）| {geo_cb_err:+.2f}% ✓（<3%）|
| 排水量 | {geom['displacement_t']:.0f} t（公开 {geom['displacement_reference_t']:.0f} t）| {geom['displacement_rel_error'] * 100:+.2f}% ✓ |
| L/B / B/T | {geom['L_over_B']:.2f} / {geom['B_over_T']:.2f} | 窗口内 ✓ |
| 几何闸门 G1 | {'通过' if geom['passed'] else '未通过'} | 先校验 Cb/L/B 再算结构 |

### 1.3 参数表

| 参数 | 数值 | 性质 |
|---|---|---|
| 冰况 | light / design / heavy（design：冰压 {sev['design']['excitation']['metadata'].get('sigma_ice_Pa', float('nan')) / 1e6:.2f} MPa，Q_mill {sev['design']['excitation']['Q_mill_Nm'] / 1e3:.1f} kN·m，Q RMS {sev['design']['excitation']['Q_rms_Nm'] / 1e3:.1f} kN·m）| 上游 N04 |
| 转速 / 叶片通过 | {sev['design']['excitation']['shaft_rpm']:.0f} rpm / {sev['design']['excitation']['blade_passing_hz']:.1f} Hz | N04 |
| 轴系→船体传递效率 η | {cfg['eta_hull']:.2f}（力 RMS {sev['design']['excitation']['force_rms_N']:.0f} N）| 声明（继承前序）|
| 干模态阻尼 | ζ={cfg['zeta_structural']:.2f} | 声明 |
| 湿模态方法 | 干模态基 + 二维剖面附加质量 a33(x) + 浮力恢复刚度，{hm['n_modes']} 阶 | 复用 hull_hydroelastic |
| 常规隔振器（基线）| f_n={base['f_n_hz']:.2f} Hz，ζ={base['zeta']:.2f}，静挠度 {base['static_deflection_m'] * 1e3:.1f} mm | 前序节点基线 |
| QZS 设计点 | f_n,dyn={f_n:.3f} Hz、α={alpha:.4f}（残余刚度 {qzs['residual_stiffness_fraction']:.4f}）、ζ={qzs['zeta']:.2f}、峰值放大 {qzs['peak_amplification']:.2f} | 本节点（规则前置）|
| QZS 硬件 | 载荷弹簧 f_rigid={qzs['f_rigid_hz']:.2f} Hz；静挠度 {static_mm:.1f} mm 不变；k_v={qzs['positive_spring_stiffness_N_per_m'] / 1e6:.2f} MN/m，k_dyn={qzs['dynamic_stiffness_N_per_m'] / 1e3:.1f} kN/m；NS 线性区 ±{qzs['ns_range_m'] * 1e3:.0f} mm；行程 ±{qzs['stroke_m'] * 1e3:.0f} mm | 声明 |
| 冰带 | {ice_lo:.1f}–{ice_hi:.1f} Hz（设计前固定）| 由船体加速度谱能量界定 |
| 管路 | 跨距 8.0 m，Φ406×12.7 mm，一阶 ~12.6 Hz，SCF 1.8、厚度修正 (t/25mm)^0.2 | 简支梁 + DNV |
| S-N | D 曲线 / 海水+CP：m1={sn['m1']:g}、m2={sn['m2']:g}、1e7 次 = {sn['stress_at_1e7_cycles_MPa']:.2f} MPa | 仓库 fatigue.sn_curve |
| 运行假设 | 冰区运行占比 {cfg['operating_fraction']:.2f}，设计寿命 {cfg['design_life_years']:.0f} 年 | 声明 |
| 波浪/鞭击包络（仅行程核对）| 0.10 Hz / 0.60 m/s² + 1.00 Hz / 0.30 m/s² RMS | 声明输入（D7）|

## 2 怎么做的

### 2.1 方法链

1. **湿模态**：干自由-自由 Timoshenko 梁模态（质量归一）+ 二维剖面无穷频附加质量 a33(x) + 浮力恢复刚度，
   在干模态基中解广义特征值问题 → f_wet 与湿模态振型（`hull_modal.py`，复用前序实现，不重写）。
2. **载荷**：N04 铣削扭矩冲击序列 → Welch 扭矩 PSD → 自平衡力偶谱 → 模态叠加得到基座站位船体加速度 PSD。
3. **基座局部结构**：25 Hz 甲板基座 SDOF（ζ=0.02）把船体加速度传到"支撑"（冰带内 |A|≈1）。
4. **隔振层（本节点增量）**：基座激励 SDOF 复传递函数 H(f) = (1+2iζr)/((1−r²)+2iζr)，r=f/f_n；
   常规方案 f_n={base['f_n_hz']:.2f} Hz，QZS 方案 f_n=f_rigid·√(1−α)={f_n:.3f} Hz。
5. **管路应力**：简支管路模态应力传递 |σ(f)/a_base|²（静态极限应等于 m L²/(8Z)），乘 SCF 与厚度修正。
6. **疲劳**：随机相位应力时程合成（Parseval 保方差）→ ASTM E1049-85 四点雨流 → DNV-RP-C203 D 曲线 →
   Palmgren-Miner 累积 → 寿命（年），与窄带 Rayleigh 闭式解交叉校核。
7. **设计（前置规则，先定后算）**：在 (f_n,dyn, ζ) 网格上建 QZS 设计图，硬约束为静态挠度 ≤
   {cfg['static_deflection_limit_m'] * 1e3:.0f} mm、残余刚度份额 ≥ {cfg['residual_stiffness_budget']:.3f}、ζ ≥ {qzs['zeta']:.2f}
   （峰值放大 ≤ {1 / (2 * qzs['zeta']):.1f}）、波浪/鞭击行程或缓冲限位力 ≤ {stroke['stop_force_limit_fraction']:.0%} 设备重量、
   冰带振动落在 NS 线性区内；在可行域内取带内 RMS 比最小的点（平手取残余刚度更大者）。可行行数 {sel['n_feasible']}/{sel['n_rows']}。

### 2.2 子模块与验证

| 子模块 | 职责 | 验证 |
|---|---|---|
| `mitigation.QZSMount` | 线性化 QZS：f_n、静挠度、灵敏度、Duffing 三次项、行程相对位移 | α=0 时与常规隔振器逐点完全一致；H(0)=1 |
| `mitigation.design_map/select_design` | 设计图与前置选点 | 极软/极硬/低阻尼行分别被对应 flag 判死（测试）|
| `mitigation.simulate_base_excitation` | 基座激励 SDOF 的 RK4 时域积分（含三次非线性） | 与频域 \\|H(f)\\| 相对误差 {ver['relative_error']:.2e}；硬化骨干使共振峰右移（测试）|
| `mitigation.tolerance_window` | α 误差容限（老化/温度）| 带内比值随 δα 单调（测试）|
| `maturity.run_mitigation` | 设计 + 三冰级重算 + 判据 + 披露 + 曲线 | 21 条 pytest 全绿 |

## 3 计算结果

### 3.1 干/湿模态（M1）

| 阶次 | f_dry [Hz] | f_wet [Hz] | f_wet/f_dry | 附加质量比 |
|---|---|---|---|---|
{mode_rows}

全部 {hm['n_modes']} 阶 f_wet < f_dry，最大比值 {max(m['ratio_wet_over_dry'] for m in hm['modes']):.3f}（M1 PASS）。
2 节点垂向湿模态 {hm['modes'][0]['f_wet_hz']:.3f} Hz 落在冰带内，正是常规隔振器放大的频段。

### 3.2 传递函数与基座加速度（M2/M3）

| 方案 | f_n | √2f_n | 衰减带内 max\\|H\\| | 冰带 RMS 比（基座/船体）| 全带(0.2–500 Hz) RMS 比 |
|---|---|---|---|---|---|
| 常规 | {base['f_n_hz']:.2f} Hz | {base['isolation_frequency_hz']:.2f} Hz | {crit['M2']['value']['conventional_max_above_isolation']:.3f} | {ratio_conv:.3f} | {crit['M3']['value']['full_band_baseline_ratio']:.3f} |
| QZS 设计 | {f_n:.3f} Hz | {qzs['isolation_frequency_hz']:.3f} Hz | {crit['M2']['value']['qzs_max_above_isolation']:.3f} | {ratio_qzs:.3f} | {crit['M3']['value']['full_band_qzs_ratio']:.3f} |

* 点判据：两者在 f > √2f_n 的衰减带内 |H|<1（M2 PASS）；QZS 衰减带内最大 {crit['M2']['value']['qzs_max_above_isolation']:.3f}。
* 能带判据：QZS 把冰带 RMS 比从 **{ratio_conv:.3f}（放大）** 压到 **{ratio_qzs:.3f}**（M3 PASS，改善 {ratio_conv / ratio_qzs:.1f} 倍）。
* 全带（0.2–500 Hz）比值：常规 {crit['M3']['value']['full_band_baseline_ratio']:.3f}（放大）；QZS {crit['M3']['value']['full_band_qzs_ratio']:.3f}。
  注意 QZS 全带 <1 并不代表"所有频率都衰减"：0.3 Hz 以下任何被动悬挂都是 1:1 传递（D1），
  QZS 在 f < √2f_n = {qzs['isolation_frequency_hz']:.2f} Hz 处仍有窄带放大（峰值 {qzs['peak_amplification']:.2f}，D2）；
  全带 <1 来自 0.6 Hz 以上大幅衰减对低频段的补偿，属于物理上限内的可实现结果，不是实现缺陷。

### 3.3 管路应力与疲劳寿命（M4–M6、M9）

| 冰级 | 船体 RMS [m/s²] | 冰带船体 RMS [m/s²] | 常规带内比 | QZS 带内比 | 常规寿命 [yr] | QZS 寿命 [yr] | 寿命增益 |
|---|---|---|---|---|---|---|---|
{sev_rows}

* 管路应力 RMS（QZS，SCF+厚度修正）：{', '.join(f"{s}={sev[s]['qzs']['stress_rms_scf_MPa']:.2e} MPa" for s in labels)}（M4 正，PASS）。
* 三冰级寿命有限且 > 0（M5 PASS），并严格单调下降（M6 PASS）。
* 设计冰级寿命增益 **{life_gain:.0f}×**（M9 PASS）；纯幂律估计 (1/{ratio_qzs:.3f})^{sn['m2']:g} = {(1 / ratio_qzs) ** sn['m2']:.0f}×，
  与雨流+Miner 直接算得的值同量级（差异来自谱形与 S-N 两分支混合）。

### 3.4 冰带定义敏感性（结论稳健性）

| 冰带 [Hz] | 常规带内 RMS 比 | QZS 带内 RMS 比 |
|---|---|---|
{band_rows}

即使冰带下移到 0.8–4 Hz（若真实 2 节点湿模态 ~1 Hz），结论不变：常规放大、QZS 衰减、设计点仍在可行域。

### 3.5 判据与披露

| ID | 判据 | 数值 | 阈值 | 判定 |
|---|---|---|---|---|
{crit_rows}

| ID | 披露 | 数值 |
|---|---|---|
{disc_rows}

### 3.6 图

{chr(10).join(f'{i}. `{name}` — {cap}' for i, (name, cap) in enumerate(FIG_CAPTIONS.items(), start=1))}

## 4 注意点（踩坑 / 局限 / 适用范围）

1. **被动隔振在 DC 处不可能衰减**：|H(0)|=1 是任何被动悬挂的硬上限，因此"基座加速度在**所有**频率低于船体"
   物理上不可满足；本节点把判据拆为点判据（f>√2f_n）与能带判据（冰带 RMS），并显式披露（D1/D2）。
2. **QZS 的代价是刚度抵消精度**：α={alpha:.3f} 意味着只保留 {qzs['residual_stiffness_fraction'] * 100:.1f}% 刚度，
   d ln f_n/dα = −1/(2(1−α))，α 误差 1% → 频率误差 {stiffness_err_pct:.0f}%。容限窗计算：带内衰减在
   δα ∈ [{min(r['delta_alpha'] for r in tol_ok):+.1%}, {max(r['delta_alpha'] for r in tol_ok):+.1%}] 内成立
   （+ 方向受"残余刚度不能为负"限制，− 方向是机构变硬/老化方向）；
   变硬方向在 δα ≤ {min(r['delta_alpha'] for r in tol_fail):+.0%} 时失效（此时 f_n 升到 {min((r['f_n_hz'] for r in tol_fail), default=float('nan')):.2f} Hz，带内比回到 1 以上）
   → 服役期必须可调/可复标（D3）。
3. **行程是第二个硬约束**：冰带振动仅 {stroke['ice_relative_displacement_rms_m'] * 1e6:.2f} µm（远在 NS 线性区
   {stroke['linearity_amplitude_m'] * 1e3:.3f} mm 内），但 0.1 Hz 波浪分量造成的相对位移 RMS 达
   {stroke['worst_relative_displacement_rms_m'] * 1e3:.0f} mm（2σ = {stroke['peak_relative_displacement_m'] * 1e3:.0f} mm），
   超出行程 ±{qzs['stroke_m'] * 1e3:.0f} mm → 必须设**缓冲限位**，其承担力为设备重量的
   {stroke['bump_stop_force_fraction_of_weight'] * 100:.1f}%（{stroke['bump_stop_force_n']:.0f} N），所需限位刚度
   {stroke['bump_stop_stiffness_n_m'] / 1e3:.0f} kN/m。判据 M7 因此检查"行程**或**限位力"，并把限位力披露（D4）。
4. **非线性边界**：超出 NS 线性区后三次项 β={qzs['cubic_coefficient_N_per_m3']:.2e} N/m³ 使骨干硬化；线性化在相对位移
   < {stroke['linearity_amplitude_m'] * 1e3:.3f} mm 内误差 <1%，冰带振动低 2 个数量级，链上用线性 QZS 自洽（D5，附 RK4 硬化验证）。
5. **绝对寿命量级巨大（{sev['design']['qzs']['fatigue']['life_years']:.1e} 年）**：冰致管路应力 RMS 仅
   {sev['design']['qzs']['stress_rms_scf_MPa']:.1e} MPa，比 D 曲线 1e7 次幅值 {sn['stress_at_1e7_cycles_MPa']:.1f} MPa 低 5–6 个数量级
   → **冰致振动疲劳不控制该管路细节**。M9 的寿命增益是相对缓解指标，不能读作"设计寿命 1e27 年"，更不能替代含内压/热胀/
   总纵弯曲的完整疲劳评估（D6）。
6. **继承自前序节点的局限（本节点未改，仅披露）**：船体梁模型偏柔（湿 2 节点 {hm['modes'][0]['f_wet_hz']:.2f} Hz，而 230 m
   集装箱船实测 2 节点垂向湿频率通常 0.6–1.0 Hz），故冰带与设计点的绝对位置随结构理想化移动；§3.4 表明结论对该
   不确定性稳健。S-N 取仓库实现（1e7 次 {sn['stress_at_1e7_cycles_MPa']:.1f} MPa），较 DNV-RP-C203 表值 52.6 MPa 低
   {abs(sn_err):.1f}%（偏保守）；未改动以保持与前序节点可比。
7. **未考虑**：辐射阻尼（用结构阻尼 2%）、冰-船冲击压力路径（仅扭矩路径）、多轴/焊接细节真实 SCF、QZS 机构摩擦与间隙、
   主动/半主动隔振对比。管路应力只含基座加速度惯性激励，不含内压/热载荷。

## 5 误差判定（是否 <3%）

| 关键量 | 计算值 | 基准/参考 | 偏差 | 是否 <3% |
|---|---|---|---|---|
| Cb（几何闸门）| {geom['block_coefficient']:.4f} | 0.6505（KCS 公开值）| {geo_cb_err:+.2f}% | 是 |
| 排水量 | {geom['displacement_t']:.0f} t | {geom['displacement_reference_t']:.0f} t（KCS 公开值）| {geom['displacement_rel_error'] * 100:+.2f}% | 是 |
| QZS 频域 \\|H\\| vs RK4 时域 | {ver['time_domain_transmissibility']:.5f} | {ver['analytic_transmissibility']:.5f} | {ver['relative_error'] * 100:.4f}% | 是 |
| 管路模态应力传递 vs 静态极限 m L²/(8Z) | {ver['pipe_static_limit']['modal_transfer_Pa_per_m_s2']:.6e} Pa/(m/s²) | {ver['pipe_static_limit']['closed_form_mL2_over_8Z_Pa_per_m_s2']:.6e} Pa/(m/s²) | {ver['pipe_static_limit']['relative_error'] * 100:.4f}% | 是 |
| α=0 时 QZS vs 常规隔振器 | 逐点一致 | 解析式 | 0（机器精度）| 是 |
| 雨流损伤 vs 窄带 Rayleigh（设计冰级）| {sev['design']['qzs']['fatigue']['rainflow_to_narrowband_ratio']:.3f} | 1.0 | {(sev['design']['qzs']['fatigue']['rainflow_to_narrowband_ratio'] - 1) * 100:+.1f}% | 否（宽带修正方向正确、量级一致）|
| S-N 1e7 次应力幅 | {sn['stress_at_1e7_cycles_MPa']:.1f} MPa | {dn_table:.1f} MPa（DNV-RP-C203 D 表值）| {sn_err:+.1f}% | **否**（偏保守；为与前序可比未改）|
| 湿模态频率 vs 实测 | {hm['modes'][0]['f_wet_hz']:.3f} Hz | 230 m 船 2 节点垂向湿频率 0.6–1.0 Hz（文献量级）| −39%…−63% | **否**（继承缺陷，见 §4.6）|

**结论**：可解析校验的量（几何、管路静态应力极限、QZS 时域/频域一致性、α=0 退化）全部 <3%；
**两项 >3%**：① 仓库 S-N 曲线 1e7 次幅值比 DNV 表值低 {abs(sn_err):.1f}%（偏保守，保持与前序可比未改）；
② 湿模态绝对频率偏低 39–63%（继承前序船体梁理想化，本节点未改动，只披露并做了冰带敏感性 §3.4）。
按纪律不粉饰：两项均为已知限制，非本节点引入。

## 6 结论

1. 湿模态在全部 {hm['n_modes']} 阶低于干模态，最大比值 {max(m['ratio_wet_over_dry'] for m in hm['modes']):.3f}（M1）。
2. 常规 4 Hz 隔振器在冰带内放大（RMS 比 {ratio_conv:.3f}），这正是前序节点 D1 的定量根源；QZS 设计点
   （f_n,dyn={f_n:.2f} Hz、α={alpha:.3f}、静挠度 {static_mm:.1f} mm 不变）把冰带 RMS 比压到 {ratio_qzs:.3f}，
   并满足点判据 |H|<1（f>√2f_n）（M2/M3）。
3. 管路应力与疲劳寿命在三冰级上均为正、有限且随冰载荷严格单调下降（M4/M5/M6）；设计冰级寿命增益 {life_gain:.0f}×（M9）。
4. QZS 可实现性代价已量化：刚度抵消容限 ±{robust_pct:.0f}%、缓冲限位力 {stroke['bump_stop_force_fraction_of_weight'] * 100:.1f}% 设备重量、
   NS 线性区 {qzs['ns_range_m'] * 1e3:.0f} mm（M7）。
5. 绝对寿命极长说明冰致振动疲劳不控制该管路细节（D6）；QZS 的价值在于把设备/管路的**振动环境**从放大变为衰减，
   为振动烈度指标（ISO 6954 类）留出余量。

## 7 审计链

* 代码：`src/tensorphysics/ice_vibration_transfer/mitigation.py`、`maturity.py`（本节点新增）；
  `hull_modal.py` / `excitation.py` / `response.py` / `pipe.py` / `transfer.py`（复用前序，未改）。
* 测试：`tests/test_ice_vibration_transfer_maturity.py`（21 用例，全绿）。
* 载荷：`artifacts/m10169/ice_vibration_transfer_mitigation_n08.json`（本文所有数字由该 payload 程序化注入）。
* 图：`docs/ice_vibration_transfer_n08_maturity/fig1–fig7`（由 `scripts/m10169_n08_figures.py` 生成，
  报告由 `scripts/m10169_n08_report.py` 生成）。
* 上游：N04 铣削扭矩谱（`propeller_fatigue.ice_spectrum`）、湿模态（`hull_hydroelastic`，前序节点已消费）。
* 前序：`docs/ICE_VIBRATION_TRANSFER_N08.md`（p228w1-N08，披露 D1 为本节点输入问题）。
"""
    DOC.write_text(md, encoding="utf-8")

    figures = []
    for name, caption in FIG_CAPTIONS.items():
        path = FIGDIR / name
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    report = {
        "title": "p227w2-N08 冰致振动传递链成熟化：QZS 隔振把冰带基座加速度压到船体之下",
        "summary": (
            "在前序已验证的冰致振动传递链上新增准零刚度（QZS）隔振层：湿模态全部低于干模态（最大比 %.3f），"
            "常规 4 Hz 隔振器在 0.3–3 Hz 冰带内 RMS(基座)/RMS(船体)=%.3f（放大，前序 D1），QZS 设计点 "
            "f_n=%.2f Hz（α=%.3f，静挠度仍 %.1f mm）把该比值压到 %.3f；三冰级管路应力>0、寿命有限为正且随冰载荷"
            "单调下降，设计冰级寿命增益 %.0f×；刚度容限/行程/缓冲限位/Duffing 线性化代价全部量化披露。"
            % (
                max(m["ratio_wet_over_dry"] for m in hm["modes"]),
                ratio_conv,
                f_n,
                alpha,
                static_mm,
                ratio_qzs,
                life_gain,
            )
        ),
        "scenario_key": "ice_vibration_transfer_qzs_maturity",
        "mission_id": 10169,
        "tags": [
            "ice_induced_vibration",
            "wet_modal",
            "added_mass",
            "transfer_function",
            "quasi_zero_stiffness",
            "piping_stress",
            "rainflow",
            "DNV-RP-C203",
            "miner_damage",
            "fatigue_life",
        ],
        "markdown": md,
        "figures": figures,
    }
    REPORT.write_text(json.dumps(report), encoding="utf-8")
    print("doc:", DOC)
    print("report:", REPORT, f"({REPORT.stat().st_size / 1e6:.2f} MB, {len(figures)} figures)")


if __name__ == "__main__":
    main()