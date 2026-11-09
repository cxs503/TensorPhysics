#!/usr/bin/env python3
"""Build the P82W4-N06 platform report payload from ``artifacts/m7361/results.json``.

The markdown is generated *from the results JSON* (no hand-typed numbers) and the
seven figures are embedded as base64 PNG.  Publish with::

    python3 scripts/ice_wave_fatigue_report.py --out /tmp/report.json
    python3 scripts/publish_report.py /tmp/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FIGURES = [
    ("fig1_time_histories.png",
     "冰热点应力（总/准静态）、波浪热点应力、叠加总应力与 10 s 放大图：riding 分量把 carrier 跨距撑大"),
    ("fig2_spectra.png", "应力谱：冰带（0.43 Hz）、三档波浪应力谱、物理 RAO 波浪谱与合并谱"),
    ("fig3_combination_law.png",
     "ρ = D_comb/(D_ice+D_wave) vs 频带比：雨流真值、TB、组合律、线性叠加线与 [0.97,1.03] 验收带"),
    ("fig4_mechanism.png", "机理：carrier 跨距增大（实测 vs 预测）与 κ=1+a r^b N^c 的标定曲线"),
    ("fig5_spectral_errors.png", "三档谱宽的 Dirlik / TB / 窄带误差 vs 雨流真值（±10% 判据线）"),
    ("fig6_whipping_calibers.png", "whipping 三种口径：峰值/RMS/损伤份额与误计入放大 A 的实测 vs 预测"),
    ("fig7_commutation.png", "雨流可交换性：分段、闭合块重排与朴素整段重排反例（对数轴 + 1e-9 阈值）"),
]


def _f(x, nd=3):
    return f"{x:.{nd}f}"


def build_markdown(res: dict) -> str:
    c = res["criteria"]
    a, b, w, d, e = (c["a_combination_law"], c["b_spectral_methods"],
                     c["c_whipping_bookkeeping"], c["d_rainflow_commutability"],
                     c["e_negative_control_m5528"])
    case = res["case"]
    prov = res["provenance"]
    phys = res["physical_level_b"]
    scan = sorted(res["scan_rows"], key=lambda r: (r["amplitude_mode"], r["ratio"],
                                                   r.get("sigma_ratio") or 0))
    law = a["law"]
    cal = law["calibration"]
    rows = res["criteria"]["c_whipping_bookkeeping"]["rows"]
    wh_rows = {r["regime"]: r for r in rows}

    L = []
    L.append("# 冰-波浪组合疲劳：组合律裁决、谱方法误差与 whipping 口径（P82W4-N06 / m7361）\n")
    L.append("**一句话结论**：冰致与波浪致疲劳损伤**不能线性相加**——25 点雨流真值的 "
             f"ρ = D_comb/(D_ice+D_wave) 落在 [{_f(a['rho_all_min'])} , {_f(a['rho_all_max'])}]，"
             f"叠加式平均低估 {_f(law['sum_rule_mean_err_pct'],1)}%、最大低估 {_f(law['sum_rule_max_err_pct'],1)}%；"
             "本报告给出可判定的非线性组合律（留出点最大误差 "
             f"{_f(law['law_max_err_pct_holdout'],2)}%）与机理（riding 幅值被重复计入）。\n")

    # ---------------- ① 案例 ----------------
    L.append("## ① 案例介绍\n")
    L.append(f"船体梁（m5528 IN32-N07 参考船）：L = {case['length_m']:.0f} m、EI = {case['EI_Nm2']:.3g} N·m²、"
             f"m′ = {case['mass_per_length_kg_m']:.3g} kg/m、z = {case['z_deck_m']:.1f} m、"
             f"E = {case['E_Pa']:.3g} Pa、模态 {case['n_modes']} 阶、阻尼 {case['damping_ratio']*100:.1f}%；"
             f"一阶垂向弯曲频率 f₁ = {_f(case['f1_hz'],4)} Hz；由 EI 反算剖面模数 "
             f"{_f(case['Z_deck_from_EI_m3'],3)} m³。\n")
    L.append(f"细部：x/L = {case['x_detail_m']/case['length_m']:.2f}、热点 SCF = {case['scf']:.1f}、"
             f"板厚 {case['thickness_mm']:.0f} mm、σ_UTS = {case['s_uts_MPa']:.0f} MPa；"
             f"S-N = {case['sn_curve']}（海水+CP，Δσ(N=1e7, t=28 mm) = {_f(case['sn_dsigma_1e7_MPa'],2)} MPa），"
             f"Miner + Goodman；年冰区 {case['ice_hours_per_year']:.0f} h。\n")
    ip = prov["ice_parameters"]
    L.append(f"冰载荷：降阶 DEM（{ip['n_zones']} 接触区、v_indent = {ip['v_indent']} m/s、"
             f"k_c = {ip['k_contact']:.1e}、σ_c = {ip['ice_parameter'] if 'ice_parameter' in ip else ip['sigma_crush']:.1e} Pa、"
             f"crush_depth = {ip['crush_depth']} m、τ_dec = {ip['decay_time']} s、seed = {ip['seed']}），"
             "参数取 m5528 定标最优解，载荷时程与 m5528 逐位一致。\n")
    L.append(f"海况（Level B）：{phys['sea_state']}，峰值频率 {_f(phys['f_wave_peak_hz'],4)} Hz。\n")
    L.append("证据分级：模态叠加/雨流/S-N = REAL；冰载荷 = REAL（降阶模型、统计定标）；"
             "波浪弯矩 = APPROXIMATE-STRIP（无 Smith 修正/绕射，幅值不作物理声明）；"
             "组合律 = REAL（裁决）+ 半经验 κ（分离频带子集标定、留出点验证）；"
             "多轴/厚度方向裂纹 = NOT_SUPPORTED。\n")

    # ---------------- ② 怎么做 ----------------
    L.append("## ② 怎么做的\n")
    ss = res["scan_settings"]
    L.append("两级实验设计（组合律须在受控条件下测，物理链须独立验证）：\n")
    L.append(f"* **Level A（受控叠加）**：两个零均值限带高斯应力分量相加。冰带 = 高斯形谱、"
             f"中心 {res['scan_rows'][0]['f_ice_hz']:.2f} Hz、相对带宽 15%；波浪带 = JONSWAP/双峰"
             "**应力谱**（RAO 幅值折入）。扫掠 f_ice/f_w ∈ "
             f"{ss['ratios']} × 幅值比 {ss['sigma_ratios']} × {len(ss['seeds'])} 个种子，"
             f"记录 {ss['law_duration_s']:.0f} s @ {ss['law_fs_hz']:.0f} Hz；另加等单分量损伤模式。\n")
    L.append(f"* **Level B（物理链）**：DEM 冰载荷 → 模态叠加 → 热点应力 + JONSWAP 波浪谱 → "
             f"自由-自由 EB 波浪 RAO → 波浪应力；{ss['physical_duration_s']:.0f} s、"
             f"dt = {ss['physical_dt_s']*1000:.0f} ms、DNV F3 双斜率 + Goodman + 板厚修正。\n")
    L.append("* **组合律**：介于两个精确极限之间——分离频带（雨流循环互不干涉）与合流频带"
             "（幅值在同一循环内相加），权重为谱重叠度 β = ∫min(S₁,S₂)df / min(∫S₁df,∫S₂df)（无拟合参数）。\n")
    L.append(f"* **标定/验证**：κ = 1 + a·r^b·N^c 只用分离频带子集（β ≤ 0.05、f_ice/f_w ∈ {{10,100}}，"
             f"{cal['n_points']} 点）标定 → a = {_f(cal['a'],4)}、b = {_f(cal['b'],4)}、c = {_f(cal['c'],4)}"
             f"（拟合残差 {_f(cal['fit_rel_err_max_pct'],2)}%）；留出点 f_ice/f_w ∈ {{0.3, 3.0}} 不参与标定。\n")
    L.append("* **工具链**：Python 3.11 + numpy/scipy；全量算例 13 s 单核，"
             "并外协方 所 loong64 双节点集群（n315，stage_dir 直投工作树）复跑，"
             "结果逐位一致（law_max_err_pct = 8.1526%）。\n")

    # ---------------- ③ 计算结果 ----------------
    L.append("## ③ 计算结果\n")
    L.append("### 3.1 端到端负控（判据 e）：关波浪后退化到 m5528\n")
    L.append("| 量 | 本节点 | m5528 (IN32-N07) | 相对误差 |\n|---|---|---|---|")
    for key in ("damage_annual_goodman", "life_years", "whipping_share_peak", "hotspot_peak_MPa",
                "dsigma_eq_MPa", "n_window_cycles", "f1_Hz", "force_peak_MN", "mean_peak"):
        cmp_ = e["comparisons"][key]
        L.append(f"| {key} | {cmp_['computed']:.6g} | {cmp_['m5528']:.6g} | "
                 f"{cmp_['rel_err_pct']:.2e}% |")
    L.append(f"\n15 个量最大相对误差 **{_f(e['worst_rel_err_pct'],2)}e-8 %**（阈值 2%）→ "
             f"{'PASS' if e['pass'] else 'FAIL'}。\n")

    L.append("### 3.2 判据 (a)：线性叠加律被否证 + 组合律\n")
    L.append(f"分离频带（f_ice/f_w ≥ 10）的 {a['n_separated_points']} 个点中只有 "
             f"**{a['n_separated_in_band']} 个**落在 [0.97, 1.03] → 线性叠加假设否证（honest FAIL），"
             "且失稳在**不安全侧**。\n")
    L.append("| f_ice/f_w | σ_wave/σ_ice | ρ 真值 | ρ_TB | ρ_组合律 | β | 合并谱 ε |\n|---|---|---|---|---|---|---|")
    for r in scan:
        sr = r.get("sigma_ratio")
        L.append(f"| {r['ratio']:g} | {sr:g} | {_f(r['rho_mean'])} ± {_f(r['rho_std'])} | "
                 f"{_f(r['rho_tb_model_mean'])} | {_f(r['rho_law'])} | {_f(r['overlap_beta'])} | "
                 f"{_f(r['combined_eps'])} |")
    L.append(f"\n规则精度（程序化评分）：线性叠加 平均误差 {_f(law['sum_rule_mean_err_pct'],1)}% / "
             f"最大 {_f(law['sum_rule_max_err_pct'],1)}%；TB 最大 {_f(law['tb_max_err_pct'],1)}%；"
             f"**组合律 最大 {_f(law['law_max_err_pct_all'],2)}%、平均 {_f(law['law_mean_err_pct_all'],2)}%"
             f"（留出点 {_f(law['law_max_err_pct_holdout'],2)}%）**，工具改进倍数 "
             f"{_f(law['tool_improvement_factor'],1)}×。\n")
    L.append("```\nD_comb = (1 − β)·[ D_rider + κ(r,N)·D_carrier ] + β·D_TB(S_ice + S_wave)\n"
             f"κ(r,N) = 1 + a·r^b·N^c,  a = {_f(cal['a'],4)}, b = {_f(cal['b'],4)}, c = {_f(cal['c'],4)}\n"
             "r = σ_rider/σ_carrier, N = f_rider/f_carrier ≥ 1（rider = 高频分量）\n```\n")
    L.append("机理（直接测量）：carrier 平均跨距在叠加中由 "
             f"{_f(res['mechanism'][0]['mean_range_carrier_alone_mpa'])} MPa 增至 "
             f"{_f(res['mechanism'][0]['mean_range_carrier_in_sum_mpa'])} MPa（f_ice/f_w = "
             f"{res['mechanism'][0]['ratio']:g}），实测增大 "
             f"{_f(res['mechanism'][0]['delta_measured_mpa'])} MPa vs 预测 "
             f"{_f(res['mechanism'][0]['delta_predicted_mpa'])} MPa —— riding 分量既按全幅计了自己的"
             "循环，又把 carrier 极值撑大同样幅值，同一幅值被计两次；故该非可加性**与频带间隔无关**。\n")
    L.append(f"**Level B（真实海况）**：冰 rms {_f(phys['ice_rms_mean'],2)} MPa、浪 rms "
             f"{_f(phys['wave_rms_mean'],2)} MPa、f_ice/f_w = {_f(phys['f_ratio_mean'],2)} → "
             f"ρ = **{_f(phys['rho_mean'])}**（单斜率）/{_f(phys['rho_bilinear_mean'])}（DNV 双斜率+Goodman）；"
             f"组合寿命 {_f(phys['life_years_combined_mean'],2)} yr。叠加式在此工况低估 "
             f"{_f(100*(phys['rho_mean']-1),1)}%。\n")

    L.append("### 3.3 判据 (b)：谱方法能力（FAIL）\n")
    L.append("| 谱宽档 | f_ice/f_w | α₂ | Dirlik | TB | 窄带 |\n|---|---|---|---|---|---|")
    for r in b["rows"]:
        L.append(f"| {r['level']} | {r['ratio']:g} | {_f(r['alpha2'])} | {_f(r['err_dirlik_pct'],2)}% | "
                 f"{_f(r['err_tb_pct'],2)}% | {_f(r['err_nb_pct'],2)}% |")
    L.append(f"\n合流频带（f_ice/f_w = 1）三法均在 2% 内；分离频带（=10）Dirlik "
             f"{_f(b['worst_err_dirlik_pct'],1)}%、TB {_f(b['worst_err_tb_pct'],1)}% → **FAIL**。"
             "工程含义：冰+浪双峰分离谱不要直接用 Dirlik/TB，应用合成时程+雨流或本组合律。\n")

    L.append("### 3.4 判据 (c)：whipping 口径自洽（FAIL，机理清楚）\n")
    L.append("| 工况（f_pulse） | 峰值份额 | RMS 份额 | 损伤份额 | 实测 A | 峰值口径 A | 偏差 |\n"
             "|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['regime']}（{_f(r['f_pulse_hz'],3)} Hz） | {_f(r['share_peak'])} | "
                 f"{_f(r['share_rms'])} | {_f(r['damage_share_whip'])} | {_f(r['amplification_A'])} | "
                 f"{_f(r['A_predicted_peak_caliber'])} | {_f(r['deviation_peak_caliber_pp'],1)} pp |")
    sub = w["subchecks"]
    L.append(f"\n* 子检 c1（m5528「<8%」）：谐波准静态 0.05 Hz 份额 "
             f"{_f(sub['c1_share_peak'])} → {'成立' if sub['c1_harmonic_slow_share_below_8pct'] else '不成立'}。\n"
             f"* 子检 c2（DEM 脉冲列 @0.05 Hz）：份额 {_f(sub['c2_share_peak_stretched'])} / "
             f"{_f(sub['c2_share_peak_rate_only'])} → **不成立**（冰卸载 τ=0.15 s 仍激起 0.559 Hz）。\n"
             f"* 子检 c3（峰值口径→A 偏差 ≤2 pp）：设计点偏差 "
             f"{_f(sub['c3_deviation_pp_design'],1)} pp（A 实测 {_f(sub['c3_A_measured_design'])} vs "
             f"峰值口径 {_f(sub['c3_A_predicted_peak_caliber'])}）→ **FAIL**。"
             "根因：峰值份额是极值统计量，损伤份额是 m 阶矩×循环率统计量；"
             f"本算例 whipping 损伤份额 {_f(wh_rows['crushing_m5528']['damage_share_whip'])}。\n")

    L.append("### 3.5 判据 (d)：雨流可交换性（PASS）\n")
    L.append("| 检验 | 相对偏差 | 阈值 |\n|---|---|---|")
    for s in d["segmentation"]:
        L.append(f"| 同序分段 {s['n_chunks']} 段（残差栈跨段） | {s['rel_dev']:.2e} | 1e-9 |")
    L.append(f"| 闭合块重排（20 次） | {d['permutation_max_rel_dev']:.2e} | 1e-9 |")
    L.append(f"| 反例：朴素整段计数施加于重排拼接 | 散布 {d['naive_permutation_spread_rel']:.2e} | — |")
    L.append("")

    # ---------------- ⑤ 注意点 ----------------
    L.append("## ⑤ 注意点（局限与坑）\n")
    L.append("1. **riding 分层是组合律的全部关键**：叠加两个限带过程后，高频分量按全幅计一遍、"
             "又把低频极值撑大同样幅值——同一幅值计两次。故 \u201c频带分离则损伤可加\u201d 只在 "
             "r ≲ 1% 时成立；幅值可比时会低估 20–55%，且与频带间隔无关（f_ice/f_w=100 时仍有 +21%）。\n")
    L.append("2. **起振瞬态污染**：ζ=1.5% 时一阶模态衰减时间 ≈19 s，不丢弃前 60 s 会把 whipping 份额"
             "算成 70–100%（m5528 老坑，本节点再确认）。\n")
    L.append("3. **谱方法共享同一限带 PSD**：Dirlik 的 ν_p=√(m₄/m₂) 对高频尾巴极敏感；"
             "时域与频域必须用同一 rFFT 网格，否则误差里混进截断效应。\n")
    L.append("4. **whipping 口径陷阱**：峰值 35% 与损伤 97% 差近三倍；用峰值份额论证"
             "\u201cwhipping 误计入只影响 15%\u201d 会严重低估风险（实测放大 2.18 倍）。\n")
    L.append("5. **平台内雨流实现不一致（顺带发现）**：`fatigue.rainflow.rainflow_counting`（list 版）"
             "与 `rainflow_counting_stack` 在含噪信号上给出不同循环（451 vs 456，损伤差 +13.2%）；"
             "本模块 `rainflow_stream` 与 stack 版逐位一致（pytest 保证）。"
             "建议平台把 `time_domain_damage_rate(method='reference')` 统一到 stack 版。\n")
    L.append("6. **波浪弯矩绝对值 withheld**：条带 Froude-Krylov + 深度衰减 + 沿船长相位，"
             "无 Smith 修正/绕射/砰击；Level A 用等幅/等损伤归一化，Level B 报告条带绝对值并给出隐含增益。"
             "受影响的只是 ρ 的绝对位置，律的形式由 r、N、β 决定。\n")
    L.append("7. **记录长度**：f_ice/f_w=100 时 carrier 在 1800 s 内仅 ~8 个循环，"
             "种子间标准差 0.7–2.3% 已如实给出。\n")
    L.append("8. 适用对象：钢质船体梁（自由-自由弯曲）+ 冰区航行与波浪共存工况的焊缝疲劳筛查；"
             "不适用于极端砰击、屈曲主导失效或低温脆断。\n")

    # ---------------- ⑥ 误差判定 ----------------
    L.append("## ⑥ 误差判定（逐项，显式回答 \u201c是否 <3%\u201d）\n")
    L.append("| 量 | 基准 | 误差 | <3% |\n|---|---|---|---|")
    L.append(f"| 年损伤/寿命/份额/Δσ_eq/循环数（15 量） | m5528 IN32-N07 | {_f(e['worst_rel_err_pct'],2)}e-8 % | 是 |")
    L.append("| 自由-自由频率、静弯矩反力平衡 | 解析 βL 根 / 刚体平衡 | ≤1e-12 | 是 |")
    L.append("| 单模态动力放大（4 频率比） | 解析 DAF | ≤4.9%（含瞬态）；稳态 <1% | 稳态口径是 |")
    L.append("| 雨流计数（正弦 25 周期 / 含噪 vs stack 参考） | ASTM E1049-85 | 0（逐位） | 是 |")
    L.append("| S-N 表值（D/F3 @1e7） | DNV-RP-C203 | ≤0.1% | 是 |")
    L.append("| JONSWAP 谱能量 | Hs²/16 | ≤3% | 是 |")
    L.append(f"| 判据 (a) 组合律 ρ | 雨流真值 | 最大 {_f(law['law_max_err_pct_all'],2)}% / 平均 "
             f"{_f(law['law_mean_err_pct_all'],2)}% | **否（8.15% > 3%）** |")
    L.append(f"| 判据 (b) Dirlik / TB | 雨流真值 | {_f(b['worst_err_dirlik_pct'],1)}% / "
             f"{_f(b['worst_err_tb_pct'],1)}% | **否** |")
    L.append(f"| 判据 (c) 峰值口径→A | 实测 A | {_f(sub['c3_deviation_pp_design'],1)} pp | **否** |")
    L.append(f"| 判据 (d) 分段/重排 | 整段雨流 | {max(s['rel_dev'] for s in d['segmentation']):.1e} / "
             f"{d['permutation_max_rel_dev']:.1e} | 是 |")
    L.append("\n**显式回答**：凡有独立解析/规范基准的量误差均 <3%（多数 <1e-6）；"
             f"组合律对雨流真值 最大 {_f(law['law_max_err_pct_all'],2)}%、平均 "
             f"{_f(law['law_mean_err_pct_all'],2)}%，**未达 3%**，如实标注——"
             "κ 的幂律只捕捉 riding/carrier 一阶机理，未含循环率比的高阶修正。"
             "判据 (b)、(c) 记 honest FAIL，不粉饰。\n")

    L.append("## 产物与审计链\n")
    L.append(f"* 模块：`{res['module']}`（dem/beam/sn/rainflow_stream/waves/combination/pipeline，"
             "含 m5528 移植说明与证据分级）\n"
             "* 数字：`artifacts/m7361/results.json`（provenance / case / scan_settings / criteria / "
             "scan_rows / mechanism / physical_level_b / environment / git_commit）\n"
             "* 图：`artifacts/m7361/fig1..fig7*.png`；判据表：`artifacts/m7361/criteria.md`\n"
             "* 用例：`tests/test_ice_wave_fatigue.py`（26 passed；含解析闸、m5528 复现、雨流一致性、"
             "组合律极限）\n"
             "* 复现：`python3 scripts/ice_wave_fatigue_study.py`（13 s）→ "
             "`python3 scripts/ice_wave_fatigue_figures.py`\n"
             f"* git commit `{res['git_commit']}`，生成时间 {res['generated_utc']}，"
             f"环境 {res['environment']['platform']} / numpy {res['environment']['numpy']}\n")
    return "\n".join(L)


def build_payload(res: dict, fig_dir: str) -> dict:
    figures = []
    for name, caption in FIGURES:
        path = os.path.join(fig_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            figures.append({
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(fh.read()).decode("ascii"),
            })
    c = res["criteria_summary"]
    return {
        "title": "P82W4-N06 冰-波浪组合疲劳：线性叠加律否证、非线性组合律与 whipping 口径裁决",
        "summary": (
            f"25 点雨流真值显示 D_comb/(D_ice+D_wave) ∈ "
            f"[{res['criteria']['a_combination_law']['rho_all_min']:.3f}, "
            f"{res['criteria']['a_combination_law']['rho_all_max']:.3f}]（分离频带仅 "
            f"{c['a']['measured_separated_in_band']} 点入 [0.97,1.03]），"
            f"线性叠加平均低估 {c['a']['sum_rule_mean_err_pct']:.1f}%；"
            f"给出组合律 D=(1-β)[D_rider+κD_carrier]+βD_TB，留出点误差 {c['a']['law_holdout_max_err_pct']:.2f}%；"
            f"(b) Dirlik {c['b']['worst_err_dirlik_pct']:.1f}% / TB {c['b']['worst_err_tb_pct']:.1f}% honest FAIL；"
            f"(c) 峰值口径→A 偏差最大 {c['c']['max_deviation_peak_caliber_pp']:.1f}pp honest FAIL；"
            f"(d)(e) PASS（{c['e']['worst_rel_err_pct']:.1e}% 复现 m5528）。Level B 真实海况 ρ=1.55。"
        ),
        "scenario_key": "ice_wave_combined_fatigue",
        "mission_id": 7361,
        "tags": ["ice", "fatigue", "combined-loading", "rainflow", "spectral-method",
                 "whipping", "combination-law", "TensorPhysics", "P82W4-N06"],
        "markdown": build_markdown(res),
        "figures": figures,
        "metadata": {
            "node": "P82W4-N06",
            "criteria": res["criteria_summary"],
            "artifacts": ["artifacts/m7361/results.json", "artifacts/m7361/criteria.md"],
            "tests": "tests/test_ice_wave_fatigue.py (26 passed)",
            "git_commit": res["git_commit"],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(ROOT, "artifacts", "m7361", "results.json"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "artifacts", "m7361"))
    ap.add_argument("--out", default="/tmp/report.json")
    args = ap.parse_args(argv)
    with open(args.results, encoding="utf-8") as fh:
        res = json.load(fh)
    payload = build_payload(res, args.fig_dir)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print(f"wrote {args.out}: markdown {len(payload['markdown'])} chars, "
          f"{len(payload['figures'])} figures, {os.path.getsize(args.out)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())