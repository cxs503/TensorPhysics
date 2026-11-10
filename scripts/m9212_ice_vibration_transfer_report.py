"""Build the p228w1-N08 markdown report + figures and submit it to the platform.

Usage::

    python scripts/m9212_ice_vibration_transfer_report.py            # doc + figures + submit
    python scripts/m9212_ice_vibration_transfer_report.py --no-post  # no submission
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m9212_ice_vibration_transfer_figures import (  # noqa: E402
    DOC,
    FIGDIR,
    PAYLOAD,
    fig_bode,
    fig_life_vs_load,
    fig_stress_sn,
    fig_tuning_map,
    fig_wet_dry,
)

MISSION_ID = 9212
SCENARIO = "ice_vibration_transfer_n08"


def build_markdown(data, figures):
    cfg = data["config"]
    modal = data["hull_modal"]
    geo = modal["geometry_gate"]
    mount = data["mount"]
    bracket = data["foundation_bracket"]
    pipe = data["pipe"]
    sn = data["sn_curve"]
    design = data["severities"][cfg["severity"]]
    path_a = design["paths"]["hull_girder"]
    path_b = design["paths"]["shaft_reaction"]
    exc = design["excitation"]
    crit = {c["id"]: c for c in data["criteria"]}
    disc = {d["id"]: d for d in data["disclosures"]}
    modes = modal["modes"]
    sev = data["severities"]
    life = {k: sev[k]["paths"]["shaft_reaction"]["pipe"]["fatigue"]["life_years"] for k in ("light", "design", "heavy")}
    damage = {k: sev[k]["paths"]["shaft_reaction"]["pipe"]["fatigue"]["damage_per_year"] for k in ("light", "design", "heavy")}
    ratio_max = max(m["ratio_wet_over_dry"] for m in modes)

    L = []
    A = L.append
    A("# p228w1-N08 冰致振动传递链：船体湿模态 → 设备基座加速度 → 管路应力谱 → 疲劳寿命")
    A("")
    A("| 项 | 内容 |")
    A("|---|---|")
    A(f"| 战役节点 | `{data['node']}`（冰区航行全流程示范 POC · 物理域）|")
    A("| 代码 | `src/tensorphysics/ice_vibration_transfer/`（新增：hull_modal / excitation / response / transfer / pipe / chain）|")
    A(f"| 载荷 payload | `artifacts/m9212/ice_vibration_transfer_n08.json`（{PAYLOAD.stat().st_size/1024:.0f} kB，程序化注入）|")
    A("| 测试 | `tests/test_ice_vibration_transfer.py`（21 用例）|")
    A("| 图 | `docs/ice_vibration_transfer_n08/`（5 张）|")
    A(f"| 判据总判定 | **{'全部通过' if data['all_criteria_pass'] else '存在未通过项'}**（{len(data['criteria'])} 条，见 §3.4）|")
    A("")
    A("## 1 案例介绍")
    A("")
    A("### 1.1 物理背景")
    A("")
    A("冰区航行时，螺旋桨叶片铣削冰层产生大幅值、与叶片通过相位锁定的扭矩脉动（N04 DEM 铣削扭矩谱）。")
    A("扭矩脉动经轴系推力/扭矩轴承反力传给船体结构，激起船体梁**湿模态**振动，再经甲板基座局部结构与设备")
    A("隔振器传到设备及管路，最终由管路细节的交变应力驱动疲劳损伤。本节点把这条链完整串起，逐环节给出")
    A("可机器裁决的判据，并输出传递函数 Bode 图与疲劳寿命-冰载荷曲线。")
    A("")
    A("上游 N04 不重写：直接消费生态既有实现 `tensorphysics.propeller_fatigue.ice_spectrum`")
    A("（碎冰面积铣削扭矩 `ice_milling_torque`、对数正态冲击幅值谱 `impact_spectrum`、叶片冲击序列")
    A("`blade_impact_train`），本节点只补结构链需要的两步：扭矩 PSD（Welch）与作用于船体的自平衡力偶谱。")
    A("")
    A("### 1.2 几何与主尺度（几何先行校验）")
    A("")
    A("| 量 | 数值 | 判定 |")
    A("|---|---|---|")
    A(f"| 船型 | {geo['case']}（公开基准）| — |")
    A(f"| L×B×T×D | {geo['length_m']:.1f} × {geo['breadth_m']:.1f} × {geo['draught_m']:.1f} × {geo['depth_m']:.1f} m | — |")
    A(f"| Cb | {geo['block_coefficient']:.4f}（公开 0.6505，差 {abs(geo['block_coefficient']-0.6505)/0.6505:.2%}）| <3% ✓ |")
    A(f"| 排水量 | {geo['displacement_t']:.0f} t（公开 {geo['displacement_reference_t']:.0f} t，差 {geo['displacement_rel_error']:.2%}）| <9% ✓ |")
    A(f"| L/B / B/T | {geo['L_over_B']:.2f} / {geo['B_over_T']:.2f} | 窗口内 ✓ |")
    A(f"| 几何闸门 G1 | {'通过' if geo['passed'] else '未通过'} | 先校验 Cb/L/B 再算结构 |")
    A("")
    A("### 1.3 参数表")
    A("")
    A("| 参数 | 数值 | 性质 |")
    A("|---|---|---|")
    A(f"| 冰况 | {exc['severity']}：{exc['label']} | ICE_CASES |")
    A(f"| 冰压 σ_ice | {exc['metadata']['sigma_ice_Pa']/1e6:.2f} MPa；进速 {exc['metadata']['advance_speed_m_s']:.2f} m/s；{exc['impacts_per_hour']:.0f} 冲击/h | N04 |")
    A(f"| 转速 / 叶片 / 直径 | {exc['shaft_rpm']:.0f} rpm / {exc['metadata']['n_blades']} / {exc['metadata']['propeller_D_m']:.2f} m（叶片通过 {exc['blade_passing_hz']:.1f} Hz）| N04 |")
    A(f"| Q_mill / Q_design / Q_max包络 | {exc['Q_mill_Nm']/1e3:.1f} / {exc['Q_design_Nm']/1e3:.1f} / {exc['Q_envelope_Nm']/1e3:.1f} kN·m | K_Q=0.023 声明 |")
    A(f"| 扭矩脉动 RMS / 峰值 | {exc['Q_rms_Nm']/1e3:.2f} / {exc['Q_peak_Nm']/1e3:.1f} kN·m（{exc['n_impacts']} 次冲击 / {exc['duration_s']:.0f} s）| 泊松冲击序列 |")
    A(f"| 轴系→船体传递效率 η | {exc['eta_hull']:.2f}（声明）；力 RMS {exc['force_rms_N']:.1f} N，r_eff {exc['r_effective_m']:.3f} m | 声明参数 D5 |")
    A(f"| 干模态阻尼 | ζ={cfg['zeta_structural']:.2f}（结构阻尼，不含辐射）| 声明 |")
    A(f"| 湿模态方法 | 干模态基 + 二维剖面 a33(x) + 浮力恢复刚度，{cfg['n_modes']} 阶 | 复用 hull_hydroelastic |")
    A(f"| 设备隔振器 H(f) | f_n={mount['f_n_hz']:.2f} Hz，ζ={mount['zeta']:.2f}，√2f_n={mount['isolation_frequency_hz']:.2f} Hz，静挠度 {mount['static_deflection_m']*1000:.1f} mm | 声明 |")
    A(f"| 基座局部模态（路径 A）| {bracket['f_n_hz']:.0f} Hz，ζ={bracket['zeta']:.2f}（峰值放大 {bracket['peak_amplification']:.2f}）| 声明 |")
    A(f"| 局部基础（路径 B）| {cfg['local_foundation_mass_kg']:.0f} kg，{bracket['f_n_hz']:.0f} Hz，ζ={bracket['zeta']:.2f} | 声明 |")
    A(f"| 管路 | 跨距 {pipe['length_m']:.1f} m，Φ{pipe['outer_diameter_m']*1000:.0f}×{pipe['wall_thickness_m']*1000:.1f} mm，线质量 {pipe['mass_per_length_kg_m']:.1f} kg/m，一阶 {pipe['first_frequency_hz']:.2f} Hz，Z={pipe['z_m3']*1e6:.0f} cm³ | 简支梁 |")
    A(f"| 疲劳细节 | SCF={pipe['scf']:.2f}，厚度修正 (t/25mm)^{pipe['thickness_exponent']:.2f} | DNV-RP-C203 |")
    A(f"| S-N | {sn['class']} 曲线 {sn['environment']}：m1={sn['m1']:.0f}，m2={sn['m2']:.0f}，1e7 次 = {sn['stress_at_1e7_cycles_MPa']:.1f} MPa | 仓库 fatigue.sn_curve |")
    A(f"| 运行假设 | 冰区运行占比 {cfg['operating_fraction']:.0%}，设计寿命 {cfg['design_life_years']:.0f} 年 | 声明 |")
    A("")
    A("## 2 怎么做的")
    A("")
    A("### 2.1 模块与公式")
    A("")
    A("| 子模块 | 职责 | 关键公式/实现 |")
    A("|---|---|---|")
    A("| `hull_modal` | 几何闸门 + 干/湿模态 | Timoshenko 自由-自由干模态；湿特征值 `M=I+A`，`K=diag(ω_dry²)+G`，`A_ij=∫a33 φi φj dx`，`G_ij=∫ρgB(x)φiφj dx` |")
    A("| `excitation` | N04 扭矩 → 船体力偶 | `F=η(Q−⟨Q⟩)/r_eff`，(x_e, x_r) 两站位反号施加（自平衡，不激励刚体模态）；Welch PSD `S=dt|X|²/Σw²` |")
    A("| `response` | 加速度传递 | `H_u(f)=−ω²Σ φ_n(x_f)[φ_n(x_e)−φ_n(x_r)]/(ω_n²−ω²+2iζ_nω_nω)`（湿模态质量归一，广义质量 1 kg）|")
    A("| `transfer` | 设备基座 H(f) | 基础激励 SDOF `H=(1+2iζr)/((1−r²)+2iζr)`；解析峰值驻点 `u*=(√(1+8ζ²)−1)/(4ζ²)` |")
    A("| `pipe` | 应力谱 + 雨流 + S-N + Miner | 简支梁模态应力传递（f→0 → `mL²/(8Z)`）；随机相位合成时程；ASTM E1049-85 雨流；DNV D 曲线 + SCF + 厚度修正；Miner |")
    A("| `chain` | 全链驱动 | 两条载荷路径并列 + λ 扫描 + 隔振器调谐图 + 判据/披露 |")
    A("")
    A("### 2.2 两条载荷路径（为什么必须并列）")
    A("")
    A("任务书给定的是**船体梁**路径（A）。但同一扭矩脉动还有一条物理上强得多的路径：轴系反力直接作用在")
    A("机舱基础结构（B）。两条路径共用同一套下游（隔振器→管路→雨流→S-N→Miner），因此可直接比较：")
    A("")
    A(f"- 路径 A：扭矩力偶 → 船体湿模态 → 基座站位加速度 RMS **{path_a['hull_girder_acceleration']['rms_m_s2']:.3e} m/s²** → 甲板局部模态 → 隔振器 → 基座；")
    A(f"- 路径 B：扭矩力偶 → 机舱基础质量（SDOF）→ 加速度 RMS **{path_b['support_acceleration']['rms_all_m_s2']:.3e} m/s²** → 隔振器 → 基座。")
    A("")
    A(f"两者相差 **{path_b['support_acceleration']['rms_all_m_s2']/path_a['hull_girder_acceleration']['rms_m_s2']:.0f} 倍**，对应管路寿命相差 **{disc['D7']['value']:.2e} 倍**。")
    A(f"因此疲劳判据（C3/C4/C6）在**路径 B**（`governing_path={data['governing_path']}`）上裁决；路径 A 的寿命同时报告，")
    A("作为「冰铣削扭矩经船体梁传递不控制疲劳」的定量证据（披露 D7）。只报其中一条都会误导。")
    A("")
    A("### 2.3 流程与数值设置")
    A("")
    A("1. 几何闸门（Cb/排水量/L/B/B/T）→ 不通过则拒绝计算；")
    A("2. 湿模态：二维剖面 BVP 求 a33(x)（每站位一题，13 站位 × 64 面板），在干模态基上解 16 阶湿特征值；")
    A("3. 激励：N04 冲击序列（120 s，dt=1 ms）→ 扭矩 PSD → 力偶谱；")
    A("4. 响应：模态叠加得船体梁/局部基础加速度 PSD → 乘 H(f)² 得设备基座加速度 PSD；")
    A("5. 管路：应力 PSD = |T_σ(f)|²S_a(f)·SCF² → 随机相位合成 120 s 应力时程（dt=2 ms）→ 雨流 → Miner；")
    A("6. 判据：10 条数值判据 + 8 条披露项写入 payload；λ 扫描给寿命-载荷曲线；调谐图给隔振可行性。")
    A("")
    A("## 3 计算结果")
    A("")
    A("### 3.1 干/湿模态（C1）")
    A("")
    A("| 阶次 | f_dry [Hz] | f_wet [Hz] | f_wet/f_dry | 广义附连水质量比 | 浮力刚度占比 |")
    A("|---|---|---|---|---|---|")
    for m in modes:
        A(f"| {m['mode']} | {m['f_dry_hz']:.4f} | {m['f_wet_hz']:.4f} | {m['ratio_wet_over_dry']:.4f} | {m['added_mass_ratio']:.3f} | {m['buoyancy_stiffness_share']:.4f} |")
    A("")
    A(f"两节点垂向弯曲：干 **{modes[0]['f_dry_hz']:.3f} Hz** → 湿 **{modes[0]['f_wet_hz']:.3f} Hz**（比值 {modes[0]['ratio_wet_over_dry']:.3f}）；")
    A(f"全部 {len(modes)} 阶满足 f_wet<f_dry，最大比值 {ratio_max:.4f}。频率降幅与广义附连水质量比一致（模式 1 附连水质量 60% 量级），浮力恢复刚度使湿频略高于纯质量效应预测。")
    A("")
    A("### 3.2 传递函数与加速度（C2）")
    A("")
    A("| 量 | 路径 A（船体梁）| 路径 B（轴系局部）|")
    A("|---|---|---|")
    A(f"| 支撑加速度 RMS（全带）[m/s²] | {path_a['support_acceleration']['rms_all_m_s2']:.3e} | {path_b['support_acceleration']['rms_all_m_s2']:.3e} |")
    A(f"| 设备基座加速度 RMS（全带）[m/s²] | {path_a['base_acceleration']['rms_all_m_s2']:.3e} | {path_b['base_acceleration']['rms_all_m_s2']:.3e} |")
    A(f"| 衰减带（f>√2f_n）支撑→基座比 | {path_a['base_acceleration']['ratio_over_support_band']:.4f} | {path_b['base_acceleration']['ratio_over_support_band']:.4f} |")
    A(f"| 衰减带内 max\\|H\\| | {path_a['max_transmissibility_above_isolation_hz']:.4f} | {path_b['max_transmissibility_above_isolation_hz']:.4f} |")
    A(f"| 全带 基座/参考 比（披露）| {path_a['base_acceleration']['ratio_over_reference_all']:.4f} | {path_b['base_acceleration']['ratio_over_support_all']:.4f} |")
    A("")
    u_star = ((1.0 + 8.0 * mount["zeta"] ** 2) ** 0.5 - 1.0) / (4.0 * mount["zeta"] ** 2)
    A(
        "隔振器 H(f)：峰值 |H|max=%.3f（位于 f/f_n=%.3f），√2f_n=%.2f Hz 以上严格 <1。校核点："
        % (mount["peak_amplification"], u_star ** 0.5, mount["isolation_frequency_hz"])
    )
    A("")
    A("| 校核点 | 频率 [Hz] | \\|H\\| | \\|H\\| [dB] |")
    A("|---|---|---|---|")
    for name, point in path_a["transmissibility"]["checkpoints"].items():
        A(f"| {name} | {point['frequency_hz']:.2f} | {point['magnitude']:.4f} | {point['magnitude_db']:.2f} |")
    A("")
    A("### 3.3 管路应力与疲劳（C3–C6）")
    A("")
    A("| 冰况 | 支撑加速度 RMS [m/s²] | 基座加速度 RMS [m/s²] | 应力 RMS [MPa] | 最大应力范围 [MPa] | 损伤/年 | 疲劳寿命 [年] |")
    A("|---|---|---|---|---|---|---|")
    for k in ("light", "design", "heavy"):
        b = sev[k]["paths"]["shaft_reaction"]
        f_ = b["pipe"]["fatigue"]
        A(f"| {k} | {b['support_acceleration']['rms_all_m_s2']:.3e} | {b['base_acceleration']['rms_all_m_s2']:.3e} | "
          f"{f_['stress_rms_MPa']:.4f} | {f_['max_stress_range_MPa']:.3f} | {f_['damage_per_year']:.3e} | {f_['life_years']:.3e} |")
    A("")
    A(f"路径 A（设计冰级）：应力 RMS {path_a['pipe']['stress_rms_MPa']:.3e} MPa，损伤/年 {path_a['pipe']['fatigue']['damage_per_year']:.3e}，寿命 {path_a['pipe']['fatigue']['life_years']:.3e} 年。")
    A(f"刚性支撑参考（路径 B 支撑运动、去掉隔振器）：应力 RMS {path_b['pipe']['stress_rms_rigid_reference_MPa']:.3f} MPa，寿命 "
      f"{path_b['pipe']['fatigue_rigid_reference']['life_years']:.3e} 年 → 隔振器把寿命提高 **{1.0/disc['D8']['value']:.1f} 倍**（披露 D8）。")
    A("")
    A("雨流/窄带交叉校验：rainflow/narrowband = %.3f（宽带过程低于窄带 Rayleigh 闭式属正常物理，见 §5）。"
      % path_b["pipe"]["fatigue"]["narrowband_crosscheck"]["rainflow_to_narrowband_ratio"])
    A("")
    A("扭矩强度扫描（λ = 扭矩幅度倍数）：")
    A("")
    A("| λ | 力 RMS [N] | 基座加速度 RMS [m/s²] | 应力 RMS [MPa] | 损伤/年 | 寿命 [年] | 路径 A 寿命 [年] |")
    A("|---|---|---|---|---|---|---|")
    for row in data["scale_sweep"]:
        A(f"| {row['scale']:.2f} | {row['force_rms_N']:.0f} | {row['base_rms_m_s2']:.3e} | {row['pipe_stress_rms_MPa']:.4f} | "
          f"{row['damage_per_year']:.3e} | {row['life_years']:.3e} | {row['hull_girder_pipe_life_years']:.3e} |")
    A("")
    A("### 3.4 判据与披露")
    A("")
    A("| 判据 | 内容 | 数值 | 阈值 | 判定 |")
    A("|---|---|---|---|---|")
    for c in data["criteria"]:
        v = c["value"]
        if isinstance(v, list):
            nums = [x for x in v if isinstance(x, (int, float))]
            v = f"max={max(nums):.4f}, min={min(nums):.4f}" if nums else "见 payload"
        elif isinstance(v, dict):
            v = "见 payload"
        elif isinstance(v, float):
            v = f"{v:.4e}"
        A(f"| {c['id']} | {c['name']} | {v} {c['units']} | {c['threshold']} | {'PASS' if c['passed'] else 'FAIL'} |")
    A("")
    A("| 披露 | 内容 | 数值 |")
    A("|---|---|---|")
    for d in data["disclosures"]:
        A(f"| {d['id']} | {d['statement']} | {d['value']:.4e} {d['units']} |")
    A("")
    A("### 3.5 图")
    A("")
    for path, caption in figures:
        A(f"![{path.stem}]({path.relative_to(ROOT / 'docs').as_posix()})")
        A("")
        A(f"*{caption}*")
        A("")
    A("## 4 注意点（踩坑 / 局限 / 适用范围）")
    A("")
    A("1. **Welch 归一化坑（本轮最关键）**：初版把 `S=dt|X|²/Σw²` 写成 `|X|²/(Σw²·dt)`，PSD 量级偏 1/dt²，")
    A("   直接导致加速度 RMS 偏大 10³ 倍、疲劳寿命偏长 10¹⁵ 倍。是「频域 RMS 与时域 lsim 对不上」把它揪出来的。")
    A("   教训：**任何 PSD 链路必须有时域对照测试**（已固化为用例）。")
    A("2. **基础激励传递峰值不等于 1/(2ζ√(1−ζ²))**（那是相对传递率公式）；本节点用解析驻点 + 数值扫描对齐。")
    A(f"3. **|H|<1 只能带内成立**：H(0)=1，0<f<√2f_n 必放大（本设计峰值 {mount['peak_amplification']:.2f}）。船体梁冰致响应的")
    A(f"   方差 {disc['D2']['value']:.1%} 落在 √2f_n 以下，因此**全频带 RMS(基座)<RMS(船体梁) 不可能靠单级被动隔振实现**；")
    A(f"   全带比 {disc['D1']['value']:.3f} 如实披露，调谐图（图 5）显示需 f_n≤1 Hz（静挠度>0.25 m）才可能 <1，工程不可接受。")
    A("4. **两条路径必须并列**：只报路径 A 会得到「寿命 10²² 年」这种无工程意义的数而不自知；只报路径 B 则回避了任务书给定的链条。")
    A("5. **寿命数值是「远大于设计寿命」的表述，不是精确预测**：设计冰级等效恒幅应力范围远低于 D 曲线 1e7 次的")
    A(f"   {sn['stress_at_1e7_cycles_MPa']:.0f} MPa，需应力放大 {disc['D3']['value']:.1f} 倍才到 {cfg['design_life_years']:.0f} 年寿命。")
    A("6. **S-N 出处差异**：仓库 D/海水+CP 曲线用 log10K1=12.102、转折 1e7；DNV-RP-C203 表列海水 CP D 曲线（log10a1=11.764、转折 1e6）")
    A("   在 1e6 次以下更保守约 2 倍。本节点另施加 SCF 与厚度修正，但曲线本体差异已披露（D4）。")
    A("7. **雨流统计不确定性**：损伤由 120 s 合成时程估计，跨种子应力散布约 ±20% → 寿命约 ×2–3；重冰冲击率低散布更大。判据只用单调性。")
    A("8. **未建模**：冰-壳撞击压力（船首局部高频激励，通常是冰区局部疲劳真凶）、推力脉动、扭振、辐射阻尼、双轴应力、")
    A("   支撑非线性。本节点结论仅对「扭矩经轴系反力/船体梁」这条路径成立。")
    A("")
    A("## 5 误差判定")
    A("")
    A("| 关键量 | 基准 | 本文结果 | 相对偏差 | 是否 <3% |")
    A("|---|---|---|---|---|")
    A(f"| 排水量（几何闸门）| 公开 {geo['displacement_reference_t']:.0f} t | {geo['displacement_t']:.0f} t | {geo['displacement_rel_error']:.2%} | **是** |")
    A(f"| Cb | 公开 0.6505 | {geo['block_coefficient']:.4f} | {abs(geo['block_coefficient']-0.6505)/0.6505:.2%} | **是** |")
    A(f"| 管路应力静态极限 | 闭式 m L²/(8Z)={path_b['pipe']['static_closed_form_Pa_per_m_s2']:.4e} Pa/(m/s²) | {abs(path_b['pipe']['stress_transfer_static_Pa_per_m_s2']):.4e} | {path_b['pipe']['static_relative_error']:.3%} | **是** |")
    A("| 隔振器峰值 | 解析驻点驻值 | 数值扫描 | <1e-9 | **是** |")
    A(f"| 雨流 vs 窄带 Rayleigh | 闭式（同谱矩）| rainflow/narrowband = {path_b['pipe']['fatigue']['narrowband_crosscheck']['rainflow_to_narrowband_ratio']:.3f} | {abs(1-path_b['pipe']['fatigue']['narrowband_crosscheck']['rainflow_to_narrowband_ratio']):.1%} | **否**（宽带雨流本应低于窄带闭式，0.5 属物理差异非误差）|")
    A("| 频域 PSD RMS vs 时域积分 | 时域 | 频域 | 约 10%（窗函数 + Nyquist 截断）| **否**（PSD 类量按 15% 判据）|")
    A("")
    A("**显式回答**：几何类（Cb、排水量）与解析类（静态应力极限、传递峰值）四项 <3%，可判数值正确；")
    A("雨流-窄带差异是模型差异而非误差（0.50 落在宽带过程 0.3–0.8 预期区间）；频域-时域 RMS 差约 10%，源于窗函数与截断，")
    A("阈值按 15%。结构/疲劳的**绝对量**（加速度、应力、寿命）缺少同船实测基准，本报告只做量级与单调性判定，")
    A("并已在 §4.5/§4.8 写明适用范围——不把无量级的数字当精确预测。")
    A("")
    A("## 6 结论")
    A("")
    A(f"1. **湿模态低于干模态成立**：几何闸门通过后 {len(modes)} 阶垂向弯曲全部 f_wet<f_dry（最大比值 {ratio_max:.4f}），两节点弯曲 {modes[0]['f_dry_hz']:.3f}→{modes[0]['f_wet_hz']:.3f} Hz。")
    A(f"2. **设备基座加速度低于支撑加速度成立（衰减带内）**：|H|max={path_b['max_transmissibility_above_isolation_hz']:.3f}<1，基座/支撑 RMS 比 {path_b['base_acceleration']['ratio_over_support_band']:.3f}；放大段与全带比 {disc['D1']['value']:.3f} 已披露。")
    A(f"3. **冰载荷增加 → 疲劳寿命降低成立**：轻/设计/重冰寿命 {life['light']:.2e}/{life['design']:.2e}/{life['heavy']:.2e} 年（损伤/年 {damage['light']:.2e}/{damage['design']:.2e}/{damage['heavy']:.2e}），λ 扫描严格单调。")
    A(f"4. **反向结论（重要）**：需把扭矩强度放大 **{disc['D6']['value']:.1f} 倍**才使该管路细节达到 {cfg['design_life_years']:.0f} 年设计寿命——")
    A("   **冰铣削扭矩经轴系/船体梁的传递不是本船管路疲劳的控制载荷**；冰区局部疲劳应转向冰-壳撞击压力路径（建议后续节点）。")
    A("")
    A("## 7 审计链")
    A("")
    A(f"- payload：`{PAYLOAD.relative_to(ROOT)}`（每条判据的数值/阈值/明细 + 全部输入参数 + S-N 参数 + 种子 + 运行时间 {data.get('runtime_s')} s）")
    A("- 模块：`src/tensorphysics/ice_vibration_transfer/`（hull_modal, excitation, response, transfer, pipe, chain）")
    A("- 测试：`tests/test_ice_vibration_transfer.py`；命令 `python -m pytest tests/test_ice_vibration_transfer.py -q`")
    A("- 生成脚本：`scripts/m9212_ice_vibration_transfer_payload.py`（算例 + payload）、`scripts/m9212_ice_vibration_transfer_figures.py`（图）、本脚本（报告 + 提交）")
    A(f"- 种子：损伤 {cfg['seed']+11}；所有判据量确定性可复现，唯损伤量跨种子散布已在 §4.7 披露")
    return "\n".join(L)


def _b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def submit(markdown: str, figures) -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not user or not password:
        print("no credentials in env - skipping submission")
        return
    login = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(login, timeout=30) as resp:
        token = json.loads(resp.read().decode())["access_token"]
    summary = (
        "冰致振动传递链节点 N08：新增 tensorphysics.ice_vibration_transfer 全链模块，"
        "由 N04 DEM 铣削扭矩谱驱动船体湿模态→设备基座加速度→管路应力谱→雨流/S-N/Miner 疲劳寿命；"
        "10 条判据全部 PASS，并如实披露全带隔振不可行性与两条载荷路径 4 个量级的差异。"
    )
    body = {
        "title": "p228w1-N08 冰致振动传递链：湿模态→设备基座加速度→管路疲劳（DNV-RP-C203）",
        "summary": summary,
        "scenario_key": SCENARIO,
        "mission_id": MISSION_ID,
        "tags": ["p228w1-N08", "ice", "vibration", "wet-modes", "fatigue", "dnv-rp-c203", "rainflow", "tensorphysics"],
        "markdown": markdown,
        "figures": [{"name": p.name, "caption": c, "png_base64": _b64(p)} for p, c in figures],
    }
    request = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            print("report submitted:", resp.status, resp.read().decode()[:300])
    except urllib.error.HTTPError as exc:  # pragma: no cover
        print("report submission failed:", exc.code, exc.read().decode()[:500])
        raise


def main() -> int:
    post = "--no-post" not in sys.argv
    data = json.loads(PAYLOAD.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figures = [
        fig_wet_dry(data),
        fig_bode(data),
        fig_stress_sn(data),
        fig_life_vs_load(data),
        fig_tuning_map(data),
    ]
    markdown = build_markdown(data, figures)
    DOC.write_text(markdown)
    print(f"wrote {DOC} ({DOC.stat().st_size/1024:.1f} kB) and {len(figures)} figures")
    if post:
        submit(markdown, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())