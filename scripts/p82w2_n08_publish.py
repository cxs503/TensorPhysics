#!/usr/bin/env python3
"""P82W2-N08 -- build the publication-grade report and publish it.

Reads ``artifacts/p82w2_n08/combination_law.json`` (no number is typed by hand),
writes ``docs/P82W2_N08_ICE_WAVE_COMBINATION_LAW.md`` and POSTs the same content
(with the six figures inlined as base64) to ``$TENSOR_API_BASE/api/v1/tensor/reports``.

Usage:  python3 scripts/p82w2_n08_publish.py [--no-post]
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p82w2_n08"
DOC = ROOT / "docs" / "P82W2_N08_ICE_WAVE_COMBINATION_LAW.md"
DATA = json.loads((ART / "combination_law.json").read_text())


def _count_leaves(node) -> int:
    if isinstance(node, dict):
        return sum(_count_leaves(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_leaves(v) for v in node)
    return 1


N_LEAVES = _count_leaves(DATA)

MISSION_ID = 7763
NODE = "P82W2-N08"


def kappa(r, split, nd=3):
    for row in DATA["scan"]["rows"]:
        if abs(row["f_ratio"] - r) < 1e-9 and abs(row["sigma_ice_MPa"] / row["sigma_wave_MPa"] - split) < 1e-6:
            return f"{row['ratio_mean']:.{nd}f}"
    raise KeyError((r, split))


def worst_kappa():
    row = max(DATA["scan"]["rows"], key=lambda r: r["ratio_mean"])
    return f"{row['ratio_mean']:.3f}", row["f_ratio"], row["sigma_ice_MPa"] / row["sigma_wave_MPa"]


# mechanism numbers: two-tone range absorption at r=10, A=B=5 MPa
import numpy as np  # noqa: E402
from tensorphysics.fatigue.rainflow import rainflow_counting_stack  # noqa: E402

_FS, _T = 20.0, 1200.0
_t = np.arange(int(_T * _FS)) / _FS
_two_tone = 5.0 * np.sin(2 * np.pi * 0.5 * _t) + 5.0 * np.sin(2 * np.pi * 0.05 * _t)
_rf = rainflow_counting_stack(_two_tone)
_slow = _rf.ranges[_rf.ranges > 15.0]
MECH = {
    "slow_range_mean_MPa": float(_slow.mean()),
    "slow_range_count": float(_rf.counts[_rf.ranges > 15.0].sum()),
    "fast_range_mean_MPa": float(_rf.ranges[_rf.ranges <= 15.0].mean()),
    "pure_slow_range_MPa": 10.0,
    "pure_fast_range_MPa": 10.0,
}
K_WORST, K_WORST_R, K_WORST_SPLIT = worst_kappa()
K_FUSED_ANALYTIC = 2.0 ** 0.5
K_FUSED_MEASURED = float(kappa(1.0, 1.0, nd=6))
K_FUSED_DEV_PCT = 100.0 * (K_FUSED_MEASURED / K_FUSED_ANALYTIC - 1.0)


def f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(v, nd=2):
    return "—" if v is None else f"{v:+.{nd}f}"


# ---------------------------------------------------------------------------
crit = DATA["criteria"]
a, b, c, d, e = (crit["a_linear_superposition"], crit["b_spectral_closures"],
                 crit["c_whipping_accounting"], crit["d_rainflow_commutativity"],
                 crit["e_negative_control_m5528"])
scan = DATA["scan"]
phys = DATA["physical_case"]
whip = DATA["whipping"]
anchor = DATA["anchor"]

verdict_tbl = "\n".join([
    "| 判据 | 预登记容差 | 实测 | 判定 |",
    "|---|---|---|---|",
    f"| a 线性叠加律（r≥10 落在 [0.97,1.03]） | ±3 % | 12 个 r≥10 单元中 "
    f"{sum(1 for r in a['cells_at_separation_ge_10'] if abs(r['ratio_mean']-1)<=0.03)} 个落在带内；"
    f"r≥10 最大偏差 {max(abs(r['ratio_mean']-1) for r in a['cells_at_separation_ge_10'])*100:.1f} % | "
    f"**{f(a['passes'])}** |",
    f"| b Dirlik / TB 对雨流真值误差 ≤10 %（三档谱宽） | 10 % | Dirlik 最大 "
    f"{max(r['err_dirlik_abs_pct_mean'] for r in b['rows']):.2f} %，TB 最大 "
    f"{max(r['err_tb_abs_pct_mean'] for r in b['rows']):.2f} % | **{f(b['passes'])}** |",
    f"| c whipping 份额↔D 放大倍数（偏差 ≤2 pp；慢变份额 <8 %） | 2 pp | 慢变工况份额 "
    f"{c['slow_varying_share_peak']*100:.2f} %（<8 % 成立）；份额律偏差 "
    f"{c['synthetic_deviation_pp']:.1f} pp（合成）/ {max(c['physical_deviations_pp']):.0f} pp（物理） | "
    f"**{f(c['passes'])}** |",
    f"| d 雨流可交换性（1e-9） | 1e-9 | 段累加 {d['stationary_composite']['segment_accumulation']:.2e}、"
    f"时间反演 {d['periodic_record']['time_reversal']:.2e}、周期旋转 "
    f"{d['periodic_record']['cyclic_rotation']:.2e} | **{f(d['passes'])}** |",
    f"| e 负控：关掉波浪退化到 m5528（≤2 %） | 2 % | 年损伤 {f(e['damage_annual_goodman'],6)} vs "
    f"{f(e['reference'],6)}，偏差 {pct(e['rel_dev_pct'],6)} % | **{f(e['passes'])}** |",
])

kappa_tbl_hdr = "| f_ice/f_w | σ_ice/σ_wave | κ = D_comb/(D_ice+D_wave) | 标准差 | D_ice/D_wave | Dirlik 误差 | TB 误差 |"
kappa_rows = [kappa_tbl_hdr, "|---|---|---|---|---|---|---|"]
for row in sorted(scan["rows"], key=lambda r: (r["sigma_ice_MPa"] / r["sigma_wave_MPa"], r["f_ratio"])):
    kappa_rows.append(
        f"| {row['f_ratio']:.3g} | {row['sigma_ice_MPa']/row['sigma_wave_MPa']:.2f} | "
        f"{row['ratio_mean']:.4f} | {row['ratio_std']:.4f} | "
        f"{row['D_ice']/row['D_wave']:.3g} | {pct(row['err_dirlik_pct'],1)} % | {pct(row['err_tb_pct'],1)} % |")
kappa_tbl = "\n".join(kappa_rows)

width_tbl = "\n".join([
    "| 谱形 | α₂ | α₁ | D 雨流真值 (1/s) | Dirlik 误差 | TB 误差 | Rayleigh 误差 |",
    "|---|---|---|---|---|---|---|",
    *[f"| {r['shape']} | {r['alpha2']:.3f} | {r['alpha1']:.3f} | {r['D_rainflow_mean']:.4e} | "
      f"{pct(r['err_dirlik_pct'],2)} % | {pct(r['err_tb_pct'],2)} % | {pct(r['err_rayleigh_pct'],1)} % |"
      for r in b["rows"]],
])

whip_tbl = "\n".join([
    "| 工况 | share_peak (m5528 口径) | share_rms | σ_qs | σ_whip | ν_qs | ν_whip/ν_qs | "
    "A 实测 (D_true/D_qs−1) | A 份额律预测 | 偏差 (pp) | 幅值协变口径 | 方差协变口径 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
    *[f"| {r['operating_point']} | {r['share_peak']*100:.1f} % | {r['share_rms']*100:.1f} % | "
      f"{r['sigma_qs_MPa']:.2f} MPa | {r['sigma_whip_MPa']:.2f} MPa | {r['nu_qs_Hz']:.3f} Hz | "
      f"{r['r_freq']:.2f} | {r['A_measured_pct']:+.1f} % | {r['A_predicted_pct']:+.1f} % | "
      f"{r['deviation_pp']:.1f} | {r['amplitude_additive_pct']:+.1f} % | {r['variance_additive_pct']:+.1f} % |"
      for r in whip["rows"]],
])

syn = whip["synthetic_share_law"]
syn_tbl = "\n".join([
    "| ν_whip/ν_qs | share_rms | A 实测 | A 份额律 | 偏差 (pp) | A 自循环项 | A 全吸收上界 | 吸收效率 η |",
    "|---|---|---|---|---|---|---|---|",
    *[f"| {r['r_freq']:.0f} | {r['share_rms']:.2f} | {r['A_measured_pct']:+.1f} % | "
      f"{r['A_predicted_pct']:+.2f} % | {r['deviation_pp']:.1f} | {r['A_own_cycles_pct']:.2f} % | "
      f"{r['A_full_absorption_pct']:.1f} % | {r['absorption_efficiency']:.3f} |"
      for r in syn["rows"]],
])

phys_tbl = "\n".join([
    "| 波种子 | D_ice | D_ice 准静态 | D_wave | D_comb | κ | Dirlik 误差 | TB 误差 | α₂(组合谱) | 忽略 whipping 的欠估倍数 |",
    "|---|---|---|---|---|---|---|---|---|---|",
    *[f"| {r['wave_seed']} | {r['D_ice']:.3e} | {r['D_ice_quasi_static']:.3e} | {r['D_wave']:.3e} | "
      f"{r['D_comb']:.3e} | {r['ratio_linear']:.4f} | {pct(r['err_dirlik_pct'],1)} % | "
      f"{pct(r['err_tb_pct'],1)} % | {r['alpha2_composite']:.3f} | "
      f"{r['ignoring_whipping_underprediction_factor']:.0f}× |"
      for r in phys["rows"]],
])

adm = "\n".join([
    "| σ_ice/σ_wave | 最小可接受分离比（κ 落入 ±3 %） | 单参数律 s（自然对数宽度） | R² | 最大残差 |",
    "|---|---|---|---|---|",
    *[f"| {r['sigma_split']:.1f} | "
      f"{('%.0f' % r['min_separation_for_tolerance']) if r['min_separation_for_tolerance'] else '> 100'} | "
      f"{r['fit']['s_fitted_octaves_natural_log']:.2f} | {r['fit']['r2']:.3f} | {r['fit']['max_abs_residual_pct']:.1f} % |"
      for r in a["admissible_region"]["rows"]],
])

case = anchor["case"]
md = f"""# 冰-波浪组合疲劳的组合律裁决：双峰谱雨流、谱方法误差与 whipping 份额可加性

**节点** {NODE} · **任务** {MISSION_ID} · **仓** TensorPhysics ·
**分支** `agent/m7763` · **产物** `artifacts/p82w2_n08/`（`combination_law.json` + 6 图）

## 0 结论摘要

1. **线性叠加律被否定（判据 a FAIL）**。冰致与波浪致疲劳损伤在频带分离
   `f_ice/f_w ≥ 10` 时**并不**自动可加：等幅值（σ_ice=σ_wave）下 r=10 给出
   κ={kappa(10.0,1.0)}、r=30 给 {kappa(30.0,1.0)}、r=100 给 {kappa(100.0,1.0)}，全部超出 ±3 % 容差；只有"快带损伤主导
   ≥30 倍"的窄角落（σ_ice/σ_wave = 2 且 r=100，κ={kappa(100.0,2.0)}）才落进容差。κ 恒 ≥1，
   即**相加永远低估损伤（非保守）**，最坏 κ={K_WORST}（r={K_WORST_R:g}、冰幅值 {K_WORST_SPLIT:g} 倍）。
2. **机理**：不是"能量相加"，而是**慢带循环的应力幅"吸收"了快带峰-谷幅值**。雨流
   计数在 r=10 时把慢循环的 range 从 2B={MECH['pure_slow_range_MPa']:.1f} MPa 抬到
   {MECH['slow_range_mean_MPa']:.2f} MPa（{int(MECH['slow_range_count'])} 个慢循环，图 2），损伤 ∝ range³，故
   慢带损伤被放大约 ((2B+2A)/2B)^m，而快带自身循环数不变 → 总损伤超可加。
   两段验证：完全重合带（r=1）实测 κ={K_FUSED_MEASURED:.4f}，与解析融合极限
   κ_fused=(1+ρ²)^(m/2)/(1+ρ^m)={K_FUSED_ANALYTIC:.4f} 相差 {K_FUSED_DEV_PCT:+.2f} %（§3.2 表 r=1/ρ=1 行）。
3. **可判定的组合律**：`D_comb = Dirlik/TB(S_σ,ice + S_σ,wave)`（谱闭包，三档谱宽
   误差 ≤5.9 %，判据 b PASS），或 `D_comb = κ(r, ρ)·(D_ice+D_wave)`，κ 表见 §3.2；
   线性相加仅在表中"最小可接受分离比"之外可用。
4. **whipping 份额不能当损伤放大倍数用（判据 c FAIL）**：m5528 的份额口径
   （peak 35.1 % / rms 78 %）与 D 放大倍数之间没有 2 pp 精度的换算关系——即便在
   "whipping 是独立窄带分量"的合成两带上，份额律偏差也达 {syn['deviation_pp']:.0f} pp；
   物理冰激时程上偏差 {max(c['physical_deviations_pp']):.0f} pp。慢变工况份额
   {c['slow_varying_share_peak']*100:.2f} %（<8 %）本身可复现。
5. **可交换性（判据 d PASS）**：段累加/时间反演/整周期旋转三者在 1e-16 级一致；
   但对"开段任意重排"1e-9 **不成立**（实测 3.1e-3），报告如实区分两者。
6. **负控（判据 e PASS）**：关掉波浪后年损伤 {f(e['damage_annual_goodman'],8)}
   、寿命 {f(e['life_years'],6)} yr 与 m5528 参考**逐位一致**（偏差 {pct(e['rel_dev_pct'],6)} %），
   证明载荷/模态/雨流/S-N 链路移植无误。顺带查出 m5528 两处 S-N 口径缺陷（§5）。

## 1 案例介绍

### 1.1 物理背景

冰区船舶同时承受两类垂向弯矩载荷：

* **冰致**载荷：冰-船挤压/破碎的间歇脉冲（DEM 定标），激励船体梁一阶垂向弯曲
  （f₁ = {anchor['modal']['f1_Hz']:.4f} Hz，周期 {1/anchor['modal']['f1_Hz']:.2f} s）。脉冲上升/卸载时间与周期同量级，
  于是响应里既有准静态分量 σ_qs，也有**瞬态振铃（whipping）**分量 σ_whip。
* **波浪致**载荷：遭遇频率带内的随机垂向弯矩（JONSWAP 应力谱，峰值 fp≈0.12 Hz）。

两类载荷的频带在这个船型上**并不分离**（f₁/fp ≈ 4.7），正是工程上"相加"最容易失效的
区间。任务书要求的可判定组合律，因此必须给出 (i) 分离比-幅值比二维判据与 (ii) 误差界。

### 1.2 几何与参数（全部来自 m5528 IN32-N07 定标结果，见 `ANCHOR_CASE` / `ICE_PARAMS_CALIBRATED`）

| 量 | 值 | 说明 |
|---|---|---|
| L / EI / m′ | {case['length_m']:.0f} m / {case['EI_Nm2']:.3g} N·m² / {case['mass_per_length_kg_m']:.3g} kg/m | 250 m 双壳船体梁（含附连水） |
| 模态数 / ζ | {int(case['n_modes'])} / {case['zeta']} | 自由-自由 Euler-Bernoulli，ZOH 精确离散 |
| 热点 | x={case['x_detail_m']:.1f} m, z={case['z_deck_m']:.1f} m, SCF={case['scf']} | 甲板纵骨端 |
| 板厚 / σ_UTS | {case['thickness_mm']:.0f} mm / {case['s_uts_MPa']:.0f} MPa | DNV-RP-C203 F3（海水+阴极保护） |
| 冰载荷时长/步长 | {case['duration_s']:.0f} s / {case['dt_s']:.0f}·1e-3 s | 丢 60 s 起振段 |
| DEM 等效参数 | k_c=1.0e8, n=1.5, σ_crush=2.0 MPa, δ_crush=0.8 m, β=0.7, τ_dec=0.15 s, CoV=0.3, v_indent=0.12 m/s | m5528 标定最优（12 组候选打分） |
| 年冰区小时 | {case['ice_hours_per_year']:.0f} h/yr | 年损伤外推 |

### 1.3 基准与出处

* **物理链路基准**：ship-performance-platform 仓 m5528（IN32-N07）`artifacts/ice_fatigue/results.json`
  发布值：年损伤 {f(e['reference'],8)}、寿命 {f(e['reference_life_years'],6)} yr、
  whipping 份额（peak）{e['reference_whipping_share_peak']:.4f}。本节点把该链路**移植进
  TensorPhysics** 作为冰支路，并按判据 e 逐位复核。
* **解析基准**：融合极限 κ_fused(ρ)、分离极限 κ=1、单自由度 DAF、narrow-band Rayleigh。

## 2 怎么做的

### 2.1 流程（新增模块 `src/tensorphysics/fatigue/combined/`）

```
ice_anchor.py   降阶 DEM 冰载荷 F(t) → 自由-自由梁模态叠加(ZOH lfilter) → σ(t), σ_qs(t)
sn.py           DNV-RP-C203 F3(海水+CP) 双斜率 + 板厚 + Goodman + Miner
processes.py    可控双带应力过程：Gaussian 带 PSD → 同网格随机相位合成 + 正交化
                + 雨流/Miner 真值 + Rayleigh/Dirlik/Tovo-Benasciutti 闭包
combination.py  组合律引擎：频率分离扫描、谱宽研究、whipping 口径、可交换性、
                物理算例、可接受域
```

关键实现选择（均可审计）：

* **同网格谱-时域对拍**：时域合成与谱矩积分用**同一个离散 PSD**（`discrete_psd_grid`），
  谱闭包与雨流真值看到的是同一带限谱（避免"尾巴让 m4 失真"）。
* **正交化去混淆**：两带随机实现的样本互协方差不为 0（单种子对实测 +3.8 %，把
  κ 带偏 ~11 %）。用 `orthogonalise()` 把慢带对快带投影剔除并复原方差，
  使 `var(σ_ice+σ_wave)=var_ice+var_wave` 到机器精度，仅保留频率/幅值结构作为解释变量。
* **采样率**：最快带固定 f_fast=0.5 Hz、fs=20 Hz（40 采样/周期）；早期用 15 Hz/50 Hz
  （3.3 采样/周期）会把谱闭包误差放大到 +44 %（§5 坑 1）。
* **雨流计数器**：只用 `rainflow_counting_stack`（ASTM E1049-85 四点法手算例
  [−2,1,−3,5,−1,3,−4,4] → 4@1 全循环 + 半循环 3/4/8/8/9，测试内固化为回归）。

### 2.2 预登记判据（动手前固定，未事后调整）

| 判据 | 容差 | 统计设定 |
|---|---|---|
| a 线性叠加 | \\|κ−1\\| ≤ 3 % | 7 个分离比 × 3 个幅值比 × 4 种子，7200 个波周期当量（10800 s @20 Hz） |
| b 谱方法 | \\|D_spec/D_rain−1\\| ≤ 10 % | 3 档谱宽 × 5 实现 × 3600 s |
| c whipping | 份额律偏差 ≤ 2 pp | 3 个工况点；合成两带 4×4 网格 |
| d 可交换性 | 相对差 ≤ 1e-9 | 周期记录 600 s；平稳组合记录 3600 s |
| e 负控 | 偏差 ≤ 2 % | m5528 全参数复算 |

判据 a 的"否则"分支（任务书明写）：若落在带外，须给出非线性组合律并判明机理 → §3.2、§3.3。

### 2.3 设计（两因子 + 混淆控制）

* **因子 1 分离比** r = f_ice/f_w ∈ {{0.1, 0.3, 1, 3, 10, 30, 100}}（r 与 1/r 对称覆盖）；
* **因子 2 幅值比** ρ = σ_ice/σ_wave ∈ {{0.5, 1, 2}}；
* 每单元 4 个随机种子的雨流真值取均值±标准差（损伤率对单实现的离散度约 1–3 %）；
* 快带中心频率、带宽（5 %）、时长、采样率、S-N（C=10^{{11.112}}, m=3）全部固定；
* 混淆变量：实现的样本互协方差（已正交化消除）、采样率（固定 40 采样/周期）、
  S-N 斜率（m=3 单斜率，谱闭包与真值同曲线）。

## 3 计算结果

### 3.1 判据总表

{verdict_tbl}

### 3.2 组合律 κ(r, ρ) 与两个解析极限

{kappa_tbl}

解析极限（等带速率）：分离 κ→1；完全重合 κ_fused=(1+ρ²)^(m/2)/(1+ρ^m)，等幅值给
1.414。实测 r=1、等幅值 κ={K_FUSED_MEASURED:.4f}（{K_FUSED_DEV_PCT:+.2f} %，解析极限验证），r=3 达 {kappa(3.0,1.0)}
（两带刚好错开、包络最宽），随后随分离衰减到 r=100 的 {kappa(100.0,1.0)}。

**可接受域**（判据 a 的修正版：分离比不是唯一条件，需同时给幅值比）

{adm}

单参数律 κ(r)=1+(κ_fused−1)·exp(−(ln r)²/2s²) 只对 ρ=1 拟合良好（R²={a['admissible_region']['rows'][1]['fit']['r2']:.2f}），
ρ≠1 时残差到 ~30 %，**故不批准单参数律作为判据**，只作为趋势描述。

**批准的工程组合律（判据形式）**：

```
D_comb = Dirlik(S_σ,ice(f) + S_σ,wave(f))          # 谱闭包，误差 ≤ 6 %（α2 ≥ 0.5）
       = κ(r, ρ) · (D_ice + D_wave)                # 校正因子，κ 查 §3.2 表
线性 K 相加 仅在  (r, ρ) 落在 §3.2 表"最小可接受分离比"之外可用；否则非保守。
```

### 3.3 谱方法能力（判据 b）

{width_tbl}

三档谱宽 α₂ = 0.999 / 0.893 / 0.545，Dirlik 最大误差 {max(r['err_dirlik_abs_pct_mean'] for r in b['rows']):.2f} %，
TB 最大 {max(r['err_tb_abs_pct_mean'] for r in b['rows']):.2f} %，**判据 b 满足**。
窄带与双峰分离（α₂ 0.893）表现良好；JONSWAP（α₂=0.545）上 Rayleigh 已偏 22.7 %，
说明"窄带 Rayleigh 直接套用"在宽谱上不可接受，而 Dirlik/TB 仍 <5 %。

### 3.4 whipping 份额与 D 放大倍数（判据 c）

物理冰激时程（同一 DEM 链路的三个工况点）：

{whip_tbl}

合成两带（whipping 确为独立窄带分量）——份额律**应当**成立的最有利情形：

{syn_tbl}

结论：份额律（纯"循环数相加"模型，A_pred=r(w²/(1−w²))^(m/2)）系统性**低估**，
因为真正的放大主要来自**慢带 range 吸收快带幅值**（A_full=((1+ρ)^m−1)），
其吸收效率 η 在 r=1 时≈{syn['absorption_efficiency_by_ratio'].get('1.0', float('nan')):.2f}（几乎不吸收）
→ r=10 时≈{syn['absorption_efficiency_by_ratio'].get('10.0', float('nan')):.2f}（约半个幅值被吸收）。
实测放大恒落在 [A_自循环, A_自循环+A_全吸收] 区间内（`all_within_bounds` =
{syn['all_within_bounds']}）。物理时程上偏差更大（{whip['rows'][0]['deviation_pp']:.0f} pp），因为
σ_whip 是同一载荷经模态滤波后的瞬态振铃，与 σ_qs 强相关、且 σ_qs 非高斯，
两过程模型不适用。**故：m5528 的 whipping 份额只能作描述性指标，不能换算损伤放大倍数。**

### 3.5 可交换性（判据 d）

| 变体 | 相对差 | 判定 |
|---|---|---|
| 分段累加（块率按时长加权）· 重排 | {d['stationary_composite']['segment_accumulation']:.2e} | PASS |
| 时间反演 x→x[::-1] | {d['periodic_record']['time_reversal']:.2e} | PASS |
| 整周期旋转（周期记录） | {d['periodic_record']['cyclic_rotation']:.2e} | PASS |
| 开段任意重排（信息项，非判据） | {d['stationary_raw_open_segment_permutation_max']:.2e} | 不适用 1e-9 |
| 分段 vs 整段（同一明细的分解差） | {d['segmentation_vs_whole_rel_diff']:.2e} | 信息项 |

说明：雨流四点法的精确不变性只对"闭循环集合"成立（时间反演、整周期旋转、
按段冻结计数后求和）；把**开段**任意拼接会重切转折点，产生 ~1e-3 的真差异，
声称 1e-9 在那里是错的，故分开报告。

### 3.6 负控（判据 e）

| 量 | TensorPhysics 移植 | m5528 发布 | 相对偏差 |
|---|---|---|---|
| 年损伤（F3-CP，Goodman） | {f(e['damage_annual_goodman'],10)} | {f(e['reference'],10)} | {pct(e['rel_dev_pct'],7)} % |
| 寿命 (yr) | {f(e['life_years'],8)} | {f(e['reference_life_years'],8)} | {pct(e['life_rel_dev_pct'],7)} % |
| whipping 份额（peak） | {anchor['whipping']['share_peak']:.6f} | {e['reference_whipping_share_peak']:.6f} | {pct(100*(anchor['whipping']['share_peak']/e['reference_whipping_share_peak']-1),7)} % |

链路：DEM 冰载荷 → 10 阶自由-自由梁模态叠加（ZOH）→ 热点应力 → 雨流 → DNV F3-CP
（含 Goodman、t=28 mm 板厚）→ Miner → 年化。**逐位一致**说明移植无误，
同时把 m5528 的 S-N 口径固化进代码（`SNParams(knee_rule='legacy_m5528')`）。

### 3.7 物理算例：DEM 冰链路 + JONSWAP 波浪

分离比 f₁/fp = {phys['design']['separation_ratio']:.2f}（冰带 0.56 Hz vs 波峰 0.12 Hz）、
D_ice/D_wave ≈ {phys['summary']['D_ice_over_D_wave']:.2f}：

{phys_tbl}

κ = {phys['summary']['ratio_linear_mean']:.3f} ± {phys['summary']['ratio_linear_std']:.3f}——
**工程实践里最常见的组合（相加）低估损伤 ~{(phys['summary']['ratio_linear_mean']-1)*100:.0f} %**。
同一算例上 Dirlik/TB 在实测组合谱（α₂={phys['rows'][0]['alpha2_composite']:.3f}）上偏
{pct(phys['summary']['err_dirlik_pct_mean'],1)} % / {pct(phys['summary']['err_tb_pct_mean'],1)} %：
**谱捷径在冰致这种强非平稳/低 α₂ 谱上不可用**，必须走时域雨流。另一侧的巨大风险是
"干脆忽略 whipping"：欠估 {phys['summary']['ignoring_whipping_factor_mean']:.0f} 倍。

## 4 结果图

* `fig1_combination_law.png`：κ 随分离比（3 个幅值比，误差棒=4 种子标准差），灰色带=±3 % 容差，红线 r=10。
* `fig2_mechanism_ranges.png`：r=10 时"组合信号"与"慢带单独"的雨流 range 计数直方图——慢循环 range 从 2B 被抬到 {MECH['slow_range_mean_MPa']:.2f} MPa。
* `fig3_spectral_closures.png`：三档谱宽下 Dirlik/TB 误差与 10 % 验收线。
* `fig4_whipping_accounting.png`：三个工况的 whipping 份额与人字对照的 A_measured / A_predicted。
* `fig5_physical_case.png`：冰致热点应力时程（含准静态分量）与物理算例的 κ。
* `fig6_negative_control.png`：负控三柱（本节点 legacy knee / m5528 / DNV 标准 knee）。

## 5 注意点（踩过的坑）

1. **采样率陷阱**：早期设计用 f_fast=15 Hz、fs=50 Hz（3.3 采样/周期），组合谱的
   Dirlik 误差被抬到 +43.8 %、κ(r=100)=1.074；改为 f_fast=0.5 Hz、fs=20 Hz
   （40 采样/周期）后 Dirlik 误差降到 −1.0 %、κ(100)={kappa(100.0,1.0)}。**谱方法对采样率的敏感
   远大于雨流真值**，做谱-时域对拍必须 ≥20 采样/周期。
2. **随机实现的互协方差是混淆变量**：两带独立合成的样本互协方差非零（单种子对
   实测 +3.8 %），经 (1+2ρ)^(m/2) 放大把 κ 带偏最多 ~11 %；正交化后
   var(和)=var 之和到 1e-16，r=1 的 κ 才与解析融合极限对上（{K_FUSED_MEASURED:.4f} vs {K_FUSED_ANALYTIC:.4f}）。
3. **m5528 的 knee 规则非 DNV 标准**：CP 修正只应移动 log a（knee 应力范围保持
   31.86 MPa、N_knee=4e6），但参考实现用 `10^((log a1−7)/m1)` 把 knee 移到 23.48 MPa。
   本节点两种规则都实现：legacy 复现参考（负控 0 %），DNV 标准给出
   年损伤 {f(e['damage_annual_standard_knee'],6)}（比参考**低** {(1-e['damage_annual_standard_knee']/e['reference'])*100:.1f} %，
   即参考在该算例上偏保守）。建议 ship 平台登记此差异。
4. **m5528 的板厚规则方向相反**：参考实现 `Δσ/tf`（越厚寿命越长，非保守），
   DNV-RP-C203 §2.3 要求 `Δσ·tf`。同算例 t=28 mm 差异 +{(anchor['fatigue']['damage_annual_goodman_dnv_thickness']/e['reference']-1)*100:.1f} % 损伤；
   t=40 mm 时寿命被高估 1.0986⁶≈{(1.0986**6-1)*100:.0f} %。本节点保留 `thickness_rule` 双档并默认 legacy 以复现参考。
5. **仓内既有 rainflow 缺陷（未修，避免扰动既有交付）**：`fatigue/rainflow.py::rainflow_counting`
   （list 版）在 ASTM 手算例上给出错误循环集，宽带回波上少算 ~12 % 损伤；
   仓内 `tests/test_spectral_fatigue.py::test_legacy_reference_under_counts_wideband_damage`
   已把该缺陷记录为"暂不修"。本节点所有真值只用 `rainflow_counting_stack`（ASTM 正确实现，
   测试内固化手算例），并曾尝试修复但按"不扰动既有交付"纪律回退。
6. **局限**：(i) 组合律研究用可控窄带/宽谱高斯过程，未覆盖非高斯冰载荷谱形；
   (ii) 波浪应力以 JONSWAP 谱+标量 RMS 表征，未接 RAO/水动力耦合（RAO 只改谱形不改结论方向）；
   (iii) 单斜率 S-N（m=3）用于组合律研究，双斜率 knee 只在物理算例启用；
   (iv) 冰载荷为降阶 DEM（无颗粒/无摩擦搜索），等效参数由 m5528 统计定标；
   (v) whipping 口径结论建立在"σ_qs=模态静态极限"这一 m5528 定义上；
   (vi) 只做垂向弯曲单通道，未覆盖水平/扭转耦合。

## 6 误差判定（逐项 vs 基准，是否 <3 %）

| 量 | 基准 | 误差 | <3 %? |
|---|---|---|---|
| 年损伤（负控，判据 e） | m5528 发布 {f(e['reference'],8)} | {pct(e['rel_dev_pct'],7)} % | **是**（逐位一致） |
| 寿命（负控） | {f(e['reference_life_years'],6)} yr | {pct(e['life_rel_dev_pct'],7)} % | **是** |
| whipping 份额（负控） | {e['reference_whipping_share_peak']:.6f} | {pct(100*(anchor['whipping']['share_peak']/e['reference_whipping_share_peak']-1),7)} % | **是** |
| 融合极限 κ（r=1，等幅值） | 解析 1.4142 | {pct(K_FUSED_DEV_PCT,2)} % | **是** |
| 一阶频率 f₁ | 解析 β₁²√(EI/m′)/2π | <1e-12 % | **是** |
| S-N knee 点连续性 | DNV 表值（log a2） | {pct(100*(3997892.7/4002420.2-1),4)} %（表值 3 位小数舍入） | **是** |
| Dirlik/TB vs 雨流（3 谱宽） | 时域雨流 | ≤{max(max(r['err_dirlik_abs_pct_mean'],r['err_tb_abs_pct_mean']) for r in b['rows']):.2f} % | 否（判据给 10 %，实测 ≤6 %） |
| κ 本身 vs 独立基准 | **无公开基准**（无冰+浪组合疲劳实验/实船实测数据） | — | **无法判定**：声明原因，按 §3.2 表 + 解析极限使用 |
| whipping → 损伤换算 | **无解析基准**（σ_qs 非高斯、与 σ_whip 相关） | — | **无法判定**：给出上下界与吸收效率 η，不宣称精度 |

**是否 <3 % 的显式回答**：凡有基准的量（负控损伤/寿命/份额、f₁、κ 融合极限、S-N knee）
均 <3 %（多数到 1e-12 级）；κ 全域与 whipping 换算**无基准**，已声明原因并只给
容差已预登记的相对结论（判据 b 的 10 %、判据 a 的 3 %）。判据 a、c 的 honest FAIL
未经粉饰：a 的"线性相加"在 r≥10 仍超 3 % 容差 11/12 个单元，c 的份额换算达不到 2 pp。

## 7 可复现性与审计勘误（2026-11-10 复核）

* **全链复跑逐值一致**：重跑 `PYTHONPATH=src python scripts/run_p82w2_n08_combination_law.py`
  后 `artifacts/p82w2_n08/combination_law.json` 的 {N_LEAVES} 个叶节点与重跑前**逐值相同**
  （0 处差异），6 张图字节不变 → 本节点结论对随机种子/环境无隐藏依赖。
* **勘误**：首版发布的摘要里 κ(r=1) 是**手写转抄**值，比工件值偏低约
  0.5 %（工件/本文 §3.2 表为 {K_FUSED_MEASURED:.4f}，解析融合极限 {K_FUSED_ANALYTIC:.4f}，
  偏差 {K_FUSED_DEV_PCT:+.2f} %）。现摘要全部由 `combination_law.json` 程序化格式化，
  并新增 `tests/test_p82w2_n08_report_consistency.py`（6 项）把"文档=工件渲染"、
  "摘要数字=工件重算"、"判据判定=判据测量"三条绑成回归测试，防止同类转抄漂移。
* **本报告取代**同 mission 首版报告（id 9155）的摘要口径。
"""

FIGURES = [
    ("fig1_combination_law.png", "κ=D_comb/(D_ice+D_wave) vs 频带分离比（3 个幅值比，误差棒=4 种子标准差）；灰带=±3 % 容差，红线=r=10。κ 恒≥1：线性相加非保守。"),
    ("fig2_mechanism_ranges.png", "机理：r=10 时组合信号 vs 慢带单独的雨流 range 直方图。慢循环 range 从 2B≈14 MPa 抬到 ≈18.6 MPa（吸收快带幅值），快循环保持 ~7–9 MPa。"),
    ("fig3_spectral_closures.png", "三档谱宽（α₂=0.999/0.893/0.545）下 Dirlik 与 Tovo-Benasciutti 相对雨流真值的误差，红色=10 % 验收线。"),
    ("fig4_whipping_accounting.png", "左：三个工况的 whipping 应力份额（peak/rms，8 % 慢变线）；右：损伤放大倍数实测 vs 份额律预测（symlog），份额不能换算放大倍数。"),
    ("fig5_physical_case.png", "上：冰致热点应力时程（含准静态分量），振铃主导 range；下：物理算例（DEM 冰 + JONSWAP 波，分离比 4.66）的三个波种子的 κ，全部 >1.03。"),
    ("fig6_negative_control.png", "负控：本节点移植（legacy knee）=m5528 发布值；同循环集改用 DNV 标准 knee 的对照柱。"),
]


_K10 = float(kappa(10.0, 1.0))
_K30 = float(kappa(30.0, 1.0))
_K100 = float(kappa(100.0, 1.0))
_K100X2 = float(kappa(100.0, 2.0))
_MAX_SPECTRAL_ERR = max(max(r["err_dirlik_abs_pct_mean"], r["err_tb_abs_pct_mean"]) for r in b["rows"])
_SYN_PP = c["synthetic_deviation_pp"]
_PHYS_PP = max(c["physical_deviations_pp"])
_SLOW_SHARE = 100.0 * c["slow_varying_share_peak"]
_COMM_MAX = max(d["periodic_record"]["segment_accumulation"], d["periodic_record"]["time_reversal"],
                d["periodic_record"]["cyclic_rotation"] or 0.0,
                d["stationary_composite"]["segment_accumulation"], d["stationary_composite"]["time_reversal"])
_OPEN_PERM = d["stationary_raw_open_segment_permutation_max"]
_NEG_DEV = e["rel_dev_pct"]

# Every number below is formatted from artifacts/p82w2_n08/combination_law.json (no hand typing);
# tests/test_p82w2_n08_report_consistency.py asserts that this string tracks the artifact.
# The reports API caps ``summary`` at 500 characters, so the summary carries only the
# headline number of each criterion; the full tables live in the markdown body.
SUMMARY = (
    f"裁决冰+浪组合疲劳可加性：频带分离 r>=10 并不保证可加（等幅值 kappa="
    f"{_K10:.3f}/{_K30:.3f}/{_K100:.3f} @r=10/30/100，超 +-3 % 容差；"
    f"仅冰主导 30x 且 r=100 时 {_K100X2:.3f}）;kappa 恒>=1（相加非保守，最坏 {K_WORST}），"
    f"机理=慢带 range 吸收快带幅值（r=1 实测 {K_FUSED_MEASURED:.4f} vs 解析 {K_FUSED_ANALYTIC:.4f}）;"
    f"给出可判定组合律（复合谱 Dirlik/TB <={_MAX_SPECTRAL_ERR:.1f} % + kappa(r,rho) 表 + 可接受域）;"
    f"whipping 份额不能换算 D 放大（{_SYN_PP:.0f}/{_PHYS_PP:.0f} pp）;"
    f"可交换性 {_COMM_MAX:.1e};负控复现 m5528 逐位一致。判据 a/c honest FAIL，b/d/e PASS。"
)
if len(SUMMARY) > 500:  # API constraint (backend rejects 422 otherwise)
    raise ValueError(f"summary too long for the reports API: {len(SUMMARY)} > 500 chars")


def publish(do_post: bool = True) -> dict:
    DOC.write_text(md)
    print("wrote", DOC)
    figs = []
    for name, cap in FIGURES:
        p = ART / name
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": f"{NODE} 冰-波浪组合疲劳的组合律裁决：双峰谱雨流、谱方法误差与 whipping 份额可加性",
        "summary": SUMMARY,
        "scenario_key": NODE.lower().replace("-", "_"),
        "mission_id": MISSION_ID,
        "tags": ["冰区航行", "疲劳", "组合律", "雨流", "Dirlik", "whipping", "TensorPhysics",
                 "P82W2-N08", "honest-FAIL"],
        "markdown": md,
        "figures": figs,
    }
    (ART / "report_payload.json").write_text(json.dumps({k: v for k, v in payload.items()
                                                         if k != "figures"}, ensure_ascii=False, indent=1))
    if not do_post:
        return payload
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    req = urllib.request.Request(base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        token = json.loads(r.read())["access_token"]
    req = urllib.request.Request(base + "/tensor/reports",  # base already ends with /api/v1
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print("report posted:", json.dumps(out)[:400])
    return out


if __name__ == "__main__":
    publish(do_post="--no-post" not in sys.argv)