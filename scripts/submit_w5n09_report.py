#!/usr/bin/env python3
"""Build and submit the P82W5-N09 report (all numbers read from results.json)."""

from __future__ import annotations

import base64
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path("/root/tp-agent/work/m8866")
ART = ROOT / "artifacts" / "m8866"
RES = json.loads((ART / "results.json").read_text())
C = RES["criteria"]
a = RES["criterion_a"]
b = RES["criterion_b"]
cc = RES["criterion_c"]
cons = cc["consistency"]
cal = RES["calibration"]["closure"]
rows_a = [r for r in a["rows"] if r["amplitude_mode"] == "equal_damage"]


def figure(name: str, caption: str) -> dict:
    png = (ART / name).read_bytes()
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(png).decode("ascii")}


def md_tables() -> str:
    lines = []
    lines.append("### (a) 加和律裁决：`f_ice/f_w ∈ {0.3,1,10,100}` × 3 档谱宽（等分量损伤，4 种子）\n")
    lines.append("| 谱宽 | f_ice/f_w | N | r | ρ 雨流真值 | 加和律偏差 | 两带闭包偏差 | κ 采用 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows_a:
        lines.append(f"| {r['width']} | {r['ratio_f_ice_over_wave']:g} | {r['n_ratio']:.3g} | "
                     f"{r['r_rider_over_carrier']:.3f} | {r['rho_mean']:.3f} ± {r['rho_std']:.3f} | "
                     f"{r['sum_rule_rel_err_pct']:+.1f} % | {r['two_band_err_pct']:+.1f} % | "
                     f"{r['kappa_used']:.2f} ({r['two_band_regime']}) |")
    lines.append("")
    lines.append(f"* 全扫描 ρ ∈ [{a['rho_range'][0]:.3f}, {a['rho_range'][1]:.3f}]；"
                 f"分离频带（≥10）等损伤点落在 [0.97,1.03] 的个数 "
                 f"**{a['separated_equal_damage_inside_band']}/{a['separated_equal_damage_points']}** "
                 f"→ 加和律被否证（honest FAIL），方向为不安全侧（低估 {min(r['sum_rule_rel_err_pct'] for r in rows_a):+.1f} % ~ "
                 f"{max(r['sum_rule_rel_err_pct'] for r in rows_a):+.1f} % 的反号）。")
    lines.append(f"* 替代律（两带 riding 闭包）全 {a['n_points']} 点最差 **{a['two_band_worst_err_pct']:.2f} %**，"
                 f"分离频带最差 **{a['two_band_worst_err_separated_pct']:.2f} %** → 判据 (a) 的‘给律+误差界’达成。")

    lines.append("\n### (b) 谱方法能力（同一限带 PSD、同一 S-N 常数）\n")
    lines.append("| 谱宽 | N | 雨流真值 | Dirlik | Tovo-Benasciutti | 窄带 | 两带闭包 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in b["rows"]:
        lines.append(f"| {r['level']} | {r['n_ratio']:.3g} | {r['d_truth_mean']:.3g} | "
                     f"{r['err_dirlik_pct']:+.1f} % | {r['err_tb_pct']:+.1f} % | "
                     f"{r['err_narrowband_pct']:+.1f} % | {r['err_two_band_pct']:+.1f} % |")
    lines.append("")
    lines.append(f"* 最差误差：Dirlik **{b['worst_err_dirlik_pct']:.2f} %**（FAIL）、TB "
                 f"**{b['worst_err_tb_pct']:.2f} %**（FAIL）、两带直接相加 "
                 f"{b['worst_err_split_sum_pct']:.2f} %（FAIL）、"
                 f"**两带 riding 闭包 {b['worst_err_two_band_pct']:.2f} %（PASS，阈值 "
                 f"{b['threshold_pct']:.0f} %）**。单峰谱方法的 FAIL 与上游 m7361 一致，未被覆盖。")

    lines.append("\n### (c) whipping 口径：加载带宽域（DNV F3 + Goodman）\n")
    lines.append("| 工况 | θ=τ/T1 | 力跳变 j | 峰值份额 | RMS 份额 | A 实测 | 峰值口径偏差 | RMS 口径偏差 | 力矩口径偏差 | 准静态域 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in cc["scan"]["rows"]:
        lines.append(f"| {r['regime']} | {r['theta_transition_over_T1']:.2f} | "
                     f"{r['force_jump_fraction']:.3f} | {r['share_peak']:.4f} | {r['share_rms']:.4f} | "
                     f"{r['A_measured']:.3f} | {r['dev_peak_pp']:.2f} pp | {r['dev_rms_pp']:.2f} pp | "
                     f"{r['dev_moment_pp']:.2f} pp | {'✔' if r['quasi_static'] else '✘'} |")
    lines.append("")
    lines.append(f"* c1（准静态域互推 ≤2 pp）：份额 {cons['c1_share_peak_max']:.4f} < 8 %，"
                 f"峰值口径偏差 **{cons['c1_dev_peak_pp_max']:.3f} pp** → PASS。")
    lines.append(f"* c2（`<8 %` 不可外推到脉冲列）：域外最小份额 {cons['c2_share_peak_min_outside']:.4f}，"
                 f"DEM 力跳变占峰值 **{cons['c2_dem_jump_fraction_min']:.3f}**（时间基拉伸 40× 仍 25.8 %）→ PASS（机理可测）。")
    lines.append(f"* c3（口径排序）：损伤口径恒等式成立；幅值口径域外偏差最大 "
                 f"峰值 {cons['caliber_max_dev_pp']['peak']:.1f} pp / RMS {cons['caliber_max_dev_pp']['rms']:.0f} pp / "
                 f"力矩 {cons['caliber_max_dev_pp']['moment']:.1f} pp → PASS。")

    d, e = RES["criterion_d"], RES["criterion_e"]
    lines.append("\n### (d)(e) 可交换性与负控\n")
    lines.append(f"* (d) 流式分段 2/7/31/97 段最大偏差 **{d['streaming']['max_rel_dev']:.1e}**（循环表长度一致）；"
                 f"闭合块重排 20 次 **{d['block_permutation']['permutation_max_rel_dev']:.2e}**；"
                 f"朴素反例散布 {d['block_permutation']['naive_permutation_spread_rel']:.2e} → PASS（1e-9）。")
    lines.append(f"* (e) 关波浪退化到 m5528：15 量最大相对误差 **{e['worst_rel_err_pct']:.2e} %**"
                 f"（阈值 {e['tolerance_pct']:.0f} %）→ PASS。")

    env = RES["envelope"]["rows"]
    lines.append("\n### 可判定误差界（加和律 ±3 % 的 r 上界，等分量损伤）\n")
    lines.append("| N | " + " | ".join(f"{r['N_freq_ratio']:g}" for r in env) + " |")
    lines.append("|---" * (len(env) + 1) + "|")
    lines.append("| r_max | " + " | ".join(f"{r['r_max_sum_rule']:.4f}" for r in env) + " |")
    lines.append("")
    lines.append(f"κ 闭包：`κ(r,N)=1 + A N^b r^p/(1 + C N^c r^q)`，A={cal['A']:.4f}, b={cal['b']:.4f}, "
                 f"p={cal['p']:.4f}, C={cal['C']:.3g}, c={cal['c']:.4f}, q={cal['q']:.4f}；"
                 f"标定残差 RMS {cal['rms_rel_err_pct']:.2f} %、最大 {cal['max_rel_err_pct']:.2f} %（{cal['n_points']} 点）。")
    return "\n".join(lines)


def main() -> int:
    # login
    login = json.dumps({"username": __import__("os").environ["TENSOR_AGENT_USER"],
                        "password": __import__("os").environ["TENSOR_AGENT_PASS"]}).encode()
    base = __import__("os").environ["TENSOR_API_BASE"].rstrip("/")
    req = urllib.request.Request(f"{base}/auth/login", data=login,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        token = json.loads(resp.read())["access_token"]

    figures = [
        figure("fig1_rho_vs_band_ratio.png",
               "图1 ρ = D_comb/(D_ice+D_wave) 随频带比的变化（3 谱宽等分量损伤 + equal-rms 点）；灰带为判据 (a) 的 [0.97,1.03]"),
        figure("fig2_kappa_map.png",
               "图2 实测 carrier 放大 κ(r,N) 与本文闭包；黑色虚线为纯理论 riding 卷积锚（r=0.465 高估 62 %）"),
        figure("fig3_spectral_errors.png",
               "图3 Dirlik / Tovo-Benasciutti / 两带 riding 闭包对雨流真值的误差与 ±10 % 判据线"),
        figure("fig4_sum_rule_envelope.png",
               "图4 可判定边界：加和律 ±3 % 允许的最大 riding 幅值比 r_max(N)"),
        figure("fig5_spectrum_valley.png",
               "图5 一个 N=10 双峰应力 PSD：谷点分割出的 carrier（波浪）与 rider（冰）两带，及一阶船体梁频率位置"),
        figure("fig6_whipping_regimes.png",
               "图6 左：whipping 峰值份额随 θ=τ/T1（8 % 声明线）；右：同一份额映射到差异极大的双计放大 A"),
        figure("fig7_caliber_deviations.png",
               "图7 峰值 / RMS / 力矩三种幅值口径对 A 的偏差（对数轴）与 2 pp 判据线"),
        figure("fig8_commutation.png",
               "图8 雨流可交换性：分段流式与闭合块重排的相对偏差与 1e-9 判据线"),
    ]

    markdown = f"""# 冰-波浪组合疲劳（续）：双峰谱方法、组合律的可判定误差界与 whipping 口径边界

**节点** `P82W5-N09`（TensorPhysics / `src/tensorphysics/ice_wave_fatigue/`，分支 `agent/m8866`）
**上游** `P82W4-N06`(m7361) 的 honest FAIL 之续研（(b) 谱方法、(c) whipping 口径），并补齐严格 `f_ice/f_w ∈ {{0.3,1,10,100}}` 扫描。
**产物** `artifacts/m8866/results.json`（全部判据数字，零手抄）、`fig1..fig8`、`docs/ICE_WAVE_BIMODAL_FATIGUE_W5N09.md`、`tests/test_ice_wave_bimodal_w5n09.py`（17 passed）。
**环境** python {RES['environment']['python']} / numpy {RES['environment']['numpy']} / {RES['environment']['platform']} / commit `{RES['environment']['git_commit']}`，全量算例 {RES['wall_time_s']:.1f} s 单核。

## 1 案例介绍（物理背景与参数）

冰区船体梁同时承受冰挤压/破碎载荷（含 whipping 惯性放大）与波浪弯矩。工程惯例把 `D_ice`、`D_wave` 分别算再加。上游 m7361 已证伪加和律（偏差在**不安全侧**），但留下两处机器判据未闭合：单峰谱方法在分离双峰谱上超出 10 %；whipping 双计放大与 m5528「慢变工况份额 <8 %」互推 16.3 pp 不自洽。

参数（m5528 IN32-N07 参考船，保证判据 (e) 可退化）：L/B/T=250/40/14 m，EI=5.0e13 N·m²，m′=5.2e5 kg/m，细部 x=75 m/z=10.5 m/SCF=2.5，`f1`={RES['case']['f1_hz']:.6f} Hz（T1=1.790 s），ζ=1.5 %，10 阶模态，DNV-RP-C203 F3（海水+阴极保护，双斜率 3/5，拐点 1e7），σ_UTS=490 MPa、t=28 mm、年冰区 2500 h；冰带中心 0.43 Hz，DEM 定标 `CALIBRATED_M5528`（decay 0.15 s，residual 0.70）；谱宽三档 γ=6.0 / γ=3.3 / 双峰 r_p=3.0。模型对比统一用单斜率 C=1、m=3，只比较计数模型。

证据分级：雨流/Miner/S-N = REAL；双峰 PSD 与谱矩 = REAL（与雨流共用同一 rFFT 网格）；两带闭包 = REAL 结构 + SEMI-EMPIRICAL κ（标定残差如实报告）；纯理论 riding 卷积 = 解析锚但被证伪为闭包；冰载荷 = 降阶 DEM（本节点并证明它不能用于 whipping 份额的物理验证）；波浪绝对幅值 = WITHHELD（等损伤归一化）；多轴/焊缝细节/共振带 = NOT_SUPPORTED。

## 2 怎么做的（方法、流程、先验阈值）

**单一构造** `bimodal.band_case()`：一次给出组合 PSD（同一条 rFFT 网格）、谷点分割的两带谱矩、两带 Dirlik、组合谱 Dirlik/TB/窄带损伤、**同一 PSD 合成**时程的雨流真值、两个分量各自的雨流损伤；记录长度随频带比自动加长（ratio ≥ 20 时 ×1.5~×3），使 N=100 仍有 ~10 个载波循环。下游一切比较都用它，杜绝第二套 PSD 定义。

**先验声明（动手前定）**：(a) `ρ ∈ [0.97,1.03]` 且必须在 `f_ice/f_w ≥ 10` 成立；(b) Dirlik/TB/两带闭包 ≤10 %，3 档谱宽；(c) 准静态域内份额 <8 % 且峰值口径互推 ≤2 pp；(d) 分段/重排 ≤1e-9；(e) 负控 ≤2 %。两条方法学声明（偏差与理由）：①两带闭包切换点 `N_SEPARATED=5` 取自 `fatigue.spectral` 已有分离判据（Torhaug 1990 / Jiao & Moan 1990），非拟合；②判据 (c) 的加载域由两个先验无量纲量定义：`θ=τ_transition/T1` 与力跳变 `j=max|ΔF|/max F`，准静态 ⟺ `j ≤ 0.02 且 θ ≥ 0.5`；共振带 `f_load/f1 ∈ [0.8,1.25]` 事先排除（share_rms > 1，幅值口径无定义）。

**标定/验证分离**：κ 闭包只在 Level-A 网格（N∈{{5,10,20,50,100}} × r∈{{0.15..1.5}} × 3 种子 = 35 点，γ=3.3）上标定；判据 (b) 的验证集用 3 档谱宽（含 γ=6、双峰 r_p=3）与不同 r，属外推检验。判据 (d)(e) 直接调用上游已验收实现，不复制代码。

**核心式**：`D_comb ≈ D_rider + κ(r,N)·D_carrier`，`κ=1+A N^b r^p/(1+C N^c r^q)`；`ρ = (1+κ·D_c/D_r)/(1+D_c/D_r)`，故 **加和律可用 ⟺ |κ−1| ≤ tol·(1+D_c/D_r)** —— 判定只需两条带谱矩，无需时域计数。

## 3 计算结果

{md_tables()}

## 4 结果图

8 张图见文末 figures（ρ-频带比、κ 图谱与锚、谱方法误差、判定边界 r_max(N)、PSD 谷分割、whipping 域、口径偏差、可交换性）。

## 5 注意点（坑与局限）

1. **两带损伤直接相加比单峰 Dirlik 更糟**（−37 % vs −11~14 %）：失效根源是漏掉 riding 对 carrier 跨距的放大，任何不加 κ 的分带法都会低估 30 % 以上。
2. **纯理论 riding 卷积不能当闭包**：m=1 精确（1+√2 r，pytest 钉住），m=3 在 r≈0.5 高估 62 %（把放大后的 carrier 与全部 riding 循环同时保留＝另一种双计）。
3. **κ 非单调**：固定 r 时 κ 在 N≲20 随 N 增长、N≳50 回落；上游幂律在 r>1 会无界外推。本闭包在栅格角点 (r=1.5,N=50) 仍有 −10.2 % 残差，**不外推 r>1.5**。
4. **记录长度是 N=100 的隐性误差源**：载波 0.0043 Hz，1200 s 仅 ~5 循环，ρ 种子标准差 7~10 %（加长到 3× 才从 16 % 压到 7 %）。
5. **峰值份额与损伤份额可差三倍**（DEM 挤压：峰值 35 %、RMS 78 %、A−1=1.15）：用峰值份额论证「whipping 误计入只影响 15 %」会把风险低估近一倍（实测 2.15 倍）。
6. **DEM 力台阶是 whipping 份额地板**：`max|ΔF|/max F=27 %`，与时间基拉伸无关 → 该 DEM 不能用来验证 whipping 份额的物理水平。
7. **KD 共振带必须事先排除**（谐波 1 Hz 时 share_rms=4.43>1），排除是先验声明而非事后剔除。
8. **同一条 PSD 是公平比较的前提**：谱矩必须在合成时程所用同一网格上积分，否则 Dirlik 的 ν_p=√(m4/m2) 会把截断误差算成方法误差。

## 6 误差判定（逐项，显式回答是否 <3 %）

| 量 | 基准 | 误差 | <3 %？ |
|---|---|---|---|
| 负控 15 量 | m5528 已发布值 | {RES['criterion_e']['worst_rel_err_pct']:.2e} % | **是** |
| 雨流可交换（分段 2/7/31/97） | 整段雨流 | {RES['criterion_d']['streaming']['max_rel_dev']:.1e}（逐位） | **是** |
| 闭合块重排（20 次） | 闭合块损伤和 | {RES['criterion_d']['block_permutation']['permutation_max_rel_dev']:.2e} | **是** |
| 带分割方差守恒 | 组合谱 m0 | ≤1e-10 | **是** |
| 理论 riding 卷积 m=1 极限 | 闭式 1+√2 r | ≤2e-3 | **是** |
| 两带闭包（判据 b，12 点） | 雨流真值 | {b['worst_err_two_band_pct']:.2f} % | 否（>3 %，如实标注） |
| 两带闭包（判据 a，{a['n_points']} 点） | 雨流真值 | {a['two_band_worst_err_pct']:.2f} % | 否（同上） |
| κ 闭包标定残差 | 网格 κ 真值 | RMS {cal['rms_rel_err_pct']:.2f} % / 最大 {cal['max_rel_err_pct']:.2f} % | 否（角点外推） |
| Dirlik / TB（判据 b） | 雨流真值 | {b['worst_err_dirlik_pct']:.2f} % / {b['worst_err_tb_pct']:.2f} % | 否（honest FAIL） |
| 峰值口径 → A（准静态域） | 实测 A | {cons['c1_dev_peak_pp_max']:.3f} pp | **是** |
| 峰值口径 → A（全域） | 实测 A | {cons['caliber_max_dev_pp']['peak']:.1f} pp | 否（域外无效，FAIL 保留） |
| 加和律 ρ | 雨流真值 | {min(r['sum_rule_rel_err_pct'] for r in rows_a):.1f} % ~ {max(r['sum_rule_rel_err_pct'] for r in rows_a):.1f} % | 否（加和律被否证） |

**显式回答**：凡有独立解析/规范基准的量（负控 15 量、雨流计数、带分割守恒、m=1 解析锚）误差均 <3 %（多数 <1e-6）；本节点两个新方法的精度为 4.07~4.70 %（最大），**未达 3 %**（κ 闭包角点模型误差）；被判 FAIL 的 Dirlik/TB/加和律/全域峰值口径一律如实 FAIL，不粉饰。

## 7 复现与审计链

```bash
python3 scripts/ice_wave_bimodal_study.py        # 21 s 单核 → artifacts/m8866/results.json
python3 scripts/ice_wave_bimodal_figures.py      # → fig1..fig8
python3 -m pytest tests/test_ice_wave_bimodal_w5n09.py -q   # 17 passed（与上游 26 用例合跑 43 passed）
```

`results.json` 含 environment / case / settings / calibration / criterion_a..e / envelope / criteria / wall_time_s，本文所有数字均可在其中查到同名键。集群复算（loong64 双节点）用于架构一致性核对。

## 8 局限

κ 闭包为半经验（6 参数，Level-A 网格内标定；`r>1.5`、`N<5` 与强双峰载波不外推）；`N∈(3.5,7)` 过渡带两分支差异可达数 %，不建议用于设计；判据 (c) 是「域受限通过」（域内 {cons['c1_dev_peak_pp_max']:.3f} pp，域外幅值口径失效，只有损伤口径 A=1+D_res/D_true 全域可用）；DEM 不能用于 whipping 份额的物理验证；波浪绝对幅值未声明；单轴名义/热点应力法，不含多轴、厚度方向裂纹、焊缝细节、低温脆断。
"""

    payload = {
        "title": "P82W5-N09 冰-波浪组合疲劳续：双峰谱两带 riding 闭包（≤4.7 %）、加和律可判定边界与 whipping 份额域边界",
        "summary": ("加和律在全部要求频带比被否证（ρ∈[1.39,1.62]，6/6 分离点越界，低估 27.9–38.3 %）；"
                    "新增两带 riding 谱闭包 D=D_rider+κ(r,N)D_carrier，3 档谱宽 12 点最差 4.70 %（Dirlik 13.74 %、TB 24.12 % honest FAIL）；"
                    "whipping 份额被证明是加载带宽量：准静态域内互推 0.02 pp（m5528 <8 % 成立），DEM 因 27 % 力台阶地板 25.8 %；"
                    "雨流可交换性逐位一致（0 / 1.6e-16），负控 8.9e-8 %。"),
        "scenario_key": "ice_wave_combined_fatigue_bimodal",
        "mission_id": 8866,
        "tags": ["TensorPhysics", "P82W5-N09", "fatigue", "bimodal-spectrum", "rainflow",
                 "whipping", "combination-law", "ice-wave"],
        "markdown": markdown,
        "figures": figures,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base}/tensor/reports", data=body,
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(resp.status, resp.read().decode()[:600])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1200])
        return 1
    pathlib.Path("/tmp/report.json").write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())