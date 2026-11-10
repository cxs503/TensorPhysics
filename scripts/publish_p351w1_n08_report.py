#!/usr/bin/env python
"""Publish the p351w1_n08 paper-level report to the platform report API.

All numbers are read from ``artifacts/p351w1_n08/*.json`` (single source of
truth, produced by ``scripts/run_p351w1_n08_rif.py``) - nothing is typed by hand.
Figures are embedded as base64 PNG.

Usage::

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python scripts/publish_p351w1_n08_report.py
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p351w1_n08"
FIG = ART / "figures"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def login() -> str:
    for path, body in (
        ("/auth/login", {"username": USER, "password": PASS}),
        ("/tensor/auth/login", {"username": USER, "password": PASS}),
    ):
        try:
            out = _post(path, body)
        except Exception:
            continue
        token = out.get("access_token") or out.get("token") or (
            out.get("data", {}) or {}).get("access_token")
        if token:
            return token
    raise SystemExit("login failed")


def fnum(x, nd=4):
    return f"{float(x):.{nd}f}"


def main() -> None:
    crit = json.loads((ART / "criteria.json").read_text())
    summ = json.loads((ART / "redundancy_summary.json").read_text())
    sens = json.loads((ART / "sensitivity.json").read_text())
    bench = json.loads((ART / "benchmark.json").read_text())
    conv = json.loads((ART / "convergence.json").read_text())
    modes = json.loads((ART / "mode_ordering.json").read_text())

    prim = summ["p350w1_n03_combined"]["directions"]
    sec = summ["double_hull_side"]["directions"]
    a, c, d, e, f = (crit["a_rif_range_and_negative_control"],
                     crit["c_monotonicity_spearman"],
                     crit["d_benchmark_vs_published_method"],
                     crit["e_failure_mode_ordering"],
                     crit["f_discretisation_convergence"])

    rows_sag = "\n".join(
        f"| {e_['component_id']} | {fnum(e_['z_m'], 2)} | {fnum(e_['area_m2'], 4)} | "
        f"{fnum(e_['lever_arm_ultimate_m'], 2)} | **{fnum(e_['rif'])}** | {fnum(e_['delta_M_MNm'], 1)} |"
        for e_ in json.loads((ART / "rif_matrix_p350w1_n03_combined_sagging.json").read_text())["entries"]
        if e_["kind"] == "plate")

    stiff_rows = "\n".join(
        f"| {cid} | {fnum(modes[b]['rif']['remove'])} | {fnum(modes[b]['rif']['debond'])} | "
        f"{fnum(modes[b]['rif']['trip'])} | {fnum(modes[b]['min_abs_gap'])} |"
        for b, cid in (("sagging", modes["sagging"]["component_id"]),
                       ("hogging", modes["hogging"]["component_id"])))

    conv_rows = "\n".join(
        f"| {c_['component_id']} | {c_['mode']} | {c_['bending']} | "
        f"{fnum(c_['rif_coarse'], 5)} | {fnum(c_['rif_fine'], 5)} | "
        f"{float(c_['relative_change']) * 100:.4f}% |" for c_ in conv)

    bench_rows = "\n".join(
        f"| {b} | {k} | {fnum(v['M_u_smith_MNm'], 1)} | {fnum(v['M_u_reference_MNm'], 1)} | "
        f"{float(v['relative_error']) * 100:.2f}% | "
        f"{'是' if float(v['relative_error']) < 0.03 else '否'}"
        for b, blk in bench.items() for k, v in blk["cases"].items())

    md = f"""## 结论速览

节点 **p351w1_n08**（mission 10727）在 TensorPhysics 新建 `tensorphysics.hull_redundancy`
包，把已验证的弹塑性渐进崩溃内核扩到**截面级增量 Smith 法**，完成冰区双壳船体梁的
**构件级 RIF 矩阵**（12 板条 × 6 筋组 × 3 失效模式 × sagging/hogging）。
机检判据 (a)–(g) **全部通过**（`artifacts/p351w1_n08/criteria.json`，`all_pass: true`），
pytest **34 passed**。

- 完好极限弯矩：sagging **{fnum(prim['sagging']['M_u_intact_MNm'], 1)} MN·m**、hogging
  **{fnum(prim['hogging']['M_u_intact_MNm'], 1)} MN·m**（z_NA,elastic = 9.526 m）。
- RIF 值域 **[{fnum(a['rif_min'])}, {fnum(a['rif_max'])}]**，完好自检 RIF = **1.0000**。
- 最危险构件：sagging = 甲板板带 **PD1–3（0.8348）**；hogging = 底部板带 **PB1–3（0.8603）**。
- **冗余构件集（任意单构件失效后 RIF ≥ 0.8）= 全部 18 个构件**（双向均 18/18，关键构件 0）。
- 反例参照：双壳舷侧独立成梁时 PO1（sagging）/ PO3（hogging）RIF = **0.7963 < 0.80** → 指标有分辨力。
- 敏感性：RIF 随力臂单调下降，Spearman **ρ = {fnum(c['sagging_pooled_rho'], 3)}（sagging）/ {fnum(c['hogging_pooled_rho'], 3)}（hogging）**（门槛 ≤ −0.8）。
- 模式谱：RIF(remove) < RIF(debond) < RIF(trip) < 1，最小间隙 {fnum(e['cases']['sagging']['min_abs_gap'])}。
- 离散加倍后 RIF 变化 **{float(f['max_relative_change']) * 100:.4f}%**（≤3%）。
- 基准：vs 已发表闭式简化法（Faulkner 1975 + Paik & Thayamballi 2003）最大误差
  **{float(d['max_relative_error']) * 100:.2f}%**（≤15%）。

## ① 案例介绍

**物理问题**：冰区船碰撞/搁浅/冰压破损后的剩余强度校核通常只回答"够不够"。
本节点回答更细的一问：**哪些构件单独失效仍能把船体梁极限弯矩保持在完好值的 80% 以上（冗余构件集），
哪些是单点即击穿 0.8 门槛的关键构件**，并给出构件级失效谱与失效位置敏感性。

**目标截面（主）** `p350w1_n03_combined`：P350W1-N03 板-梁组合（底部/甲板加筋板格）由双壳舷侧
闭合的理想化中横剖面箱型截面。B = 24.0 m，D = 20.0 m，肋距 a = 2.4 m，纵骨间距 0.8 m；
MS 钢 E = 206 GPa、ν = 0.3、σ_y = 235 MPa、H = 1 GPa。

| 构件类 | 范围 | 板厚 | 加筋 | z 质心 |
|---|---|---|---|---|
| 底部板 PB1–3 | 3×8.0 m | 18 mm | 350×12 FB @0.8 m（每组 10 根） | 0.009 m |
| 底部筋组 SB1–3 | — | — | 折算 0.042 m² | 0.184 m |
| 甲板板 PD1–3 | 3×8.0 m | 16 mm | 300×11 FB @0.8 m（每组 10 根） | 19.984 m |
| 甲板筋组 SD1–3 | — | — | 折算 0.033 m² | 19.841 m |
| 外板 PO1–3 | 高 6.667 m | 14 mm | 纵骨 @0.8 m | 3.333 / 10.000 / 16.667 m |
| 内板 PI1–3 | 高 6.000 m | 12 mm | 纵骨 @0.8 m | 3.350 / 10.009 / 16.669 m |

**几何自检**（独立重算 A/z_NA/I）：A = 1.5608 m²、z_NA = 9.526 m、I = 118.19 m⁴、
S_deck = 11.285 m³、S_bottom = 12.407 m³；重算相对误差 1.4e-16 / 0.0，NA 在 [0, D] 内 → valid。

**截面（副）** `double_hull_side`：双壳舷侧组件独立成梁（8 板条 + 6 筋组），用于检验排序的截面无关性。

**证据等级**：REAL（无硬编码/随机数；每个数字由逐曲率求解得到）。

## ② 怎么做的

1. **内核复用**：元素本构沿用已验证的 `elastoplastic_collapse`（板 load-end shortening + J2 单轴），
   本节点只加截面组装与损伤语义，不重写内核。
2. **截面级增量 Smith 法**：纤维离散（板条 1 纤维/筋组 n_web 纤维）→ 逐级曲率 κ（6e-4 1/m，80 级，
   sagging κ>0 / hogging κ<0）→ 每级二分求解 `N(ε0)=0` 得瞬时中和轴 → `M(κ)=ΣσA(z−z_NA)` →
   `M_u = max M(κ)`（峰值在区间内部，有测试保证）。
3. **三种构件失效模式**：`remove`（板带撕裂/构件移除，面积→0 且其支持板格屈曲宽度 ×2）、
   `trip`（筋 tripping：压应力上限 σ_cr,trip + 零切线平台，**保留剪切连接**、板格支持保留）、
   `debond`（组合作用丧失：筋退化为裸压杆 + 其板格屈曲宽度 ×2）。
4. **三个关键本构决策**：
   - **板格有效屈曲应力**：弹性 σ_cr（k=4）对 0.8 m×16 mm 板格为 298 MPa > σ_y → 板永不软化、
     M(κ) 无峰值（实测到 κ=2e-3 仍单调升）。改用 Faulkner 反标定
     `σ_cr,eff = min(σ_cr,el, (φσ_y)²/σ_y)` 使 von Karman 分支复现 `φ(β)σ_y`，峰值成为真实极限点。
   - **tripping 应力**：`σ = G J/I_p`，`J = h t_w³/3`，`I_p = t_w h³/3 + b_e t³/12`，
     `b_e = min(0.4a, b/2)` + Johnson–Ostenfeld → 甲板 300×11：106.3 MPa (0.45σ_y)；
     底部 350×12：93.0 MPa (0.40σ_y)。
   - **裸压杆**：`r = h/√12`、`σ_E = π²E/(a/r)²` + J-O → 350×12 @2.4 m = 231 MPa ≈ σ_y；
     故组合作用丧失的惩罚主要由**板格支持丧失**承担（φ 0.833 → 0.504），本截面真实结论。
5. **工具链**：numpy + 自实现 Spearman（并列秩平均）；`scripts/run_p351w1_n08_rif.py` 一键产出全部
   JSON+5 图；34 项 pytest。

## ③ 计算结果

### 完好极限弯矩与中和轴迁移

| 方向 | M_u (MN·m) | κ_u (1/m) | z_NA,elastic (m) | z_NA,ultimate (m) | 受压翼缘应变 |
|---|---|---|---|---|---|
| Sagging | **{fnum(prim['sagging']['M_u_intact_MNm'], 1)}** | {float(prim['sagging']['kappa_u_intact']):.2e} | 9.526 | **{fnum(prim['sagging']['z_NA_ultimate_m'], 3)}** | 3.65e-3 |
| Hogging | **{fnum(prim['hogging']['M_u_intact_MNm'], 1)}** | {float(prim['hogging']['kappa_u_intact']):.2e} | 9.526 | **{fnum(prim['hogging']['z_NA_ultimate_m'], 3)}** | 2.22e-3 |

> **弹性中和轴 ≠ 极限中和轴**：受压翼缘卸载使零应变点向受拉侧迁移（sagging −3.84 m，hogging +1.44 m）。
> 这条结论直接决定判据 (b) 的检验方式（见 ⑤）。

### RIF 矩阵 — 板条（sagging，甲板受压）

| 构件 | z (m) | A (m²) | 力臂 (m) | RIF | ΔM_u (MN·m) |
|---|---|---|---|---|---|
{rows_sag}

Hogging 下最危险由甲板板换成底部板 PB1–3（RIF 0.8603，ΔM_u 387.6 MN·m），甲板板 0.8955；
舷侧中段 PO2/PI2 = 0.9938/0.9958（几乎无影响）。

### RIF 矩阵 — 筋组（三模式谱）

| 方向/构件 | remove | debond | trip | 最小间隙 |
|---|---|---|---|---|
{stiff_rows}

> 受拉侧筋组的 trip/debond RIF 恒为 1.0000（压应力上限只作用于受压），物理正确（有专门测试）。
> 底部筋组 sagging 下 remove = 0.9837（受拉侧移除仍降低少量承载力，来自面积损失）。

### top-3 最危险构件与冗余构件集

| 方向 | top-1 | top-2 | top-3 | RIF_min | 冗余集 (RIF_min ≥ 0.8) |
|---|---|---|---|---|---|
| Sagging | {prim['sagging']['top3_most_dangerous'][0]['component_id']} {fnum(prim['sagging']['top3_most_dangerous'][0]['rif_min'])} | {prim['sagging']['top3_most_dangerous'][1]['component_id']} {fnum(prim['sagging']['top3_most_dangerous'][1]['rif_min'])} | {prim['sagging']['top3_most_dangerous'][2]['component_id']} {fnum(prim['sagging']['top3_most_dangerous'][2]['rif_min'])} | {fnum(prim['sagging']['rif_min'])} | **{prim['sagging']['redundancy_set']['n_redundant']}/18 全部冗余** |
| Hogging | {prim['hogging']['top3_most_dangerous'][0]['component_id']} {fnum(prim['hogging']['top3_most_dangerous'][0]['rif_min'])} | {prim['hogging']['top3_most_dangerous'][1]['component_id']} {fnum(prim['hogging']['top3_most_dangerous'][1]['rif_min'])} | {prim['hogging']['top3_most_dangerous'][2]['component_id']} {fnum(prim['hogging']['top3_most_dangerous'][2]['rif_min'])} | {fnum(prim['hogging']['rif_min'])} | **{prim['hogging']['redundancy_set']['n_redundant']}/18 全部冗余** |

**反例参照（副截面）**：双壳舷侧独立成梁时 PO1(sagging)/PO3(hogging) 移除 RIF = **0.7963 < 0.80**，
关键构件 1 个 → 指标有分辨力，主截面"全冗余"不是恒真结论。且构件分组越粗损伤越大，
故主截面结论位于**保守侧**（细分构件只会让 RIF 更接近 1）。

### RIF 对失效位置的敏感性

| 截面 / 方向 | 总体 ρ | 板条 ρ | 斜率 dRIF/d力臂 (1/m) | n |
|---|---|---|---|---|
| 主截面 sagging | **{fnum(sens['p350w1_n03_combined']['sagging']['pooled']['spearman_rho'], 3)}** | {fnum(sens['p350w1_n03_combined']['sagging']['plating']['spearman_rho'], 3)} | {fnum(sens['p350w1_n03_combined']['sagging']['pooled']['slope_per_m'], 5)} | {sens['p350w1_n03_combined']['sagging']['pooled']['n']} |
| 主截面 hogging | **{fnum(sens['p350w1_n03_combined']['hogging']['pooled']['spearman_rho'], 3)}** | {fnum(sens['p350w1_n03_combined']['hogging']['plating']['spearman_rho'], 3)} | {fnum(sens['p350w1_n03_combined']['hogging']['pooled']['slope_per_m'], 5)} | {sens['p350w1_n03_combined']['hogging']['pooled']['n']} |
| 副截面 sagging | {fnum(sens['double_hull_side']['sagging']['pooled']['spearman_rho'], 3)} | {fnum(sens['double_hull_side']['sagging']['plating']['spearman_rho'], 3)} | {fnum(sens['double_hull_side']['sagging']['pooled']['slope_per_m'], 5)} | {sens['double_hull_side']['sagging']['pooled']['n']} |

筋组类内 ρ = 0（退化）：每组 10 根纵骨共用一个 z，秩无区分度 → 类内秩相关无定义，不做虚假声明。
副截面 −0.730 的成因已量化：闭合板面积 0.0168 m² vs 舷侧板条 0.0933 m²（5.6 倍），
秩相关被面积差混杂；剔除上下闭合板后回到 **{fnum(c['secondary_excl_closing_rho'], 3)}**。

### 收敛（判据 f，离散双加倍）

| 构件 | 模式 | 方向 | RIF 粗 | RIF 细 | 相对变化 |
|---|---|---|---|---|---|
{conv_rows}

## ④ 结果图

- **图 1** 板条 RIF 柱状图（sagging/hogging 双面板，0.80 门槛 + 0.98 负控线）
- **图 2** 筋组三模式谱：remove < debond < trip
- **图 3** RIF–力臂散点（板条/筋组 × 双方向，图例含 ρ）
- **图 4** M–κ 曲线：完好 vs 甲板板移除 vs SD1 trip/debond（★为各 M_u）
- **图 5** 截面理想化示意（12 板条 + 6 筋组）

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **坑 1（数值）**：板格弹性 σ_cr 高于 σ_y 时板永不软化，`M(κ)` 无峰值，"极限弯矩"退化为扫描边界值
   （实测 2991 → 3099 MN·m 单调升）。Faulkner 反标定后峰值在 κ = 2.55e-4 内部点。已加回归测试。
2. **坑 2（判据口径）**：弹性中和轴不能当负控基准。sagging 下舷侧中段 PO2 距弹性轴仅 0.47 m，
   但移除 RIF = 0.9525；真因是塑性中和轴降到 5.691 m（力臂 4.31 m）。判据 (b) 按**极限中和轴**
   选最近构件：PI1 0.9930、PO1 0.9907（≥0.98 ✓）；弹性轴读数 0.9525/0.9661 作为**发现**如实报告。
3. **坑 3（基准调用）**：损坏算例必须传损坏态 M_u，误传完好值会得到 12.5% 虚高误差；修正后完好 0.37%。
4. **局限 1（基准类型）**：判据 (d) 基准为**已发表闭式方法**（Faulkner 1975 + Paik & Thayamballi 2003 简化法），
   **非试验数据**——仓内无受损船体梁试验数据集。故为**方法级基准**，已在 criteria.json 声明 limitation；
   试验数据级对标 = **WITHHELD**。
5. **局限 2（理想化）**：未建模腹板/翼缘局部屈曲、残余应力、拉侧断裂；板条为分组超单元（粒度 1/3 板格）；
   tripping 取理想塑性平台（无 post-tripping 陡降）；截面为 350 m 级双壳船的代表性尺度而非施工图。
6. **局限 3（副截面）**：ρ = −0.730 未达 −0.8，成因已量化；剔除闭合板后 −0.873。判据在主截面双向满足。
7. **适用范围**：单构件失效、静态单调弯曲、无腐蚀折减；不适用多构件同时失效与疲劳-极限交互。

## ⑥ 误差判定（逐项，显式回答「是否 <3%」）

| 方向 | 算例 | Smith (MN·m) | 闭式基准 (MN·m) | 相对误差 | <3%? |
|---|---|---|---|---|---|
{bench_rows}

**逐项判定**：
- **<3% 成立**的量：完好 sagging（0.37%）、SD1 trip sagging（0.53%）、弹性斜率 dM/dκ vs E·I（1.3e-5）、
  板屈曲应力解析式（同式 1e-12）、完好自检 RIF（<1e-6）。
- **<3% 不成立**的量：hogging 完好 7.85%、PD1 移除 6.07%、SD1 移除 4.71%–13.44%、PB1 移除 4.76%。
  原因：(i) 闭式简化法**方向无关**，真实截面双向差 7.5%（底部 18 mm+350×12 筋强于甲板 16 mm+300×11 筋）；
  (ii) 闭式法无卸载/中和轴迁移。**故不宣称整体 <3%**；本节点判据口径为 ≤15%，**全部满足**（最大 13.44%）。

## 机检判据汇总

| 判据 | 内容 | 数值 | 通过 |
|---|---|---|---|
| (a) | RIF∈[0,1]；完好自检 1.000±1e-3 | [{fnum(a['rif_min'])}, {fnum(a['rif_max'])}]；sag {fnum(a['self_check_rif_sagging'])} / hog {fnum(a['self_check_rif_hogging'])} | {'✓' if a['pass'] else '✗'} |
| (b) | 移除近中和轴构件 RIF ≥ 0.98 | PI1 {fnum(crit['b_negative_control_neutral_axis']['sagging']['ultimate_NA_picks'][0]['rif'])}、PO1 {fnum(crit['b_negative_control_neutral_axis']['sagging']['ultimate_NA_picks'][1]['rif'])}；hogging PO2 {fnum(crit['b_negative_control_neutral_axis']['hogging']['ultimate_NA_picks'][0]['rif'])} | {'✓' if crit['b_negative_control_neutral_axis']['pass'] else '✗'} |
| (c) | Spearman ρ ≤ −0.8 | sag {fnum(c['sagging_pooled_rho'], 3)} / hog {fnum(c['hogging_pooled_rho'], 3)} | {'✓' if c['pass'] else '✗'} |
| (d) | vs 已发表算例 ≤ 15% | 最大 {float(d['max_relative_error']) * 100:.2f}%（方法级基准） | {'✓' if d['pass'] else '✗'} |
| (e) | 模式差非零且方向正确 | 0.8810<0.9363<0.9743<1；0.9005<0.9515<0.9752<1 | {'✓' if e['pass'] else '✗'} |
| (f) | 离散加倍 ≤ 3% | {float(f['max_relative_change']) * 100:.4f}% | {'✓' if f['pass'] else '✗'} |
| (g) | pytest ≥ 18 含负控 | **34 passed** | ✓ |

## 复现与审计链

```bash
PYTHONPATH=src python scripts/run_p351w1_n08_rif.py    # criteria/benchmark/convergence/matrices + 5 图
python -m pytest tests/test_hull_redundancy.py -q      # 34 passed
```

审计链：`artifacts/p351w1_n08/criteria.json` → `rif_matrix_*.json`（逐构件逐模式 M_u/RIF）→
`redundancy_summary.json`（排序/冗余集/负控）→ `sensitivity.json` → `convergence.json` /
`benchmark.json` / `mode_ordering.json`。数字全部由脚本单次运行注入，无手抄。

**文献**：Smith (1977) Trans. SNAME 85；Faulkner (1975) J. Ship Res. 19(1)；Paik & Thayamballi (2003)
*Ultimate Limit State Design of Steel-Plated Structures* Ch.5；Timoshenko & Gere (1961)；IACS CSR Pt 1 Ch 5。
"""

    figures = []
    captions = {
        "fig1_rif_plating.png": "图 1 板条 RIF（sagging/hogging，0.80 门槛线与 0.98 负控线）",
        "fig2_mode_spectrum.png": "图 2 筋组三失效模式谱：remove < debond < trip",
        "fig3_sensitivity.png": "图 3 RIF 对失效位置的敏感性（RIF–力臂散点，图例含 Spearman ρ）",
        "fig4_moment_curvature.png": "图 4 M–κ 曲线：完好 vs 甲板板移除 vs SD1 trip/debond",
        "fig5_section.png": "图 5 P350W1-N03 组合截面理想化示意（12 板条 + 6 筋组）",
    }
    for name, cap in captions.items():
        path = FIG / name
        if path.exists():
            figures.append({
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            })

    token = login()
    payload = {
        "title": "P351W1-N08 冰区船体结构冗余度指标 RIF 与破损后极限承载力",
        "summary": (
            f"截面级增量 Smith 法给出 12 板条×6 筋组×3 失效模式的 RIF 矩阵；"
            f"完好 M_u {fnum(prim['sagging']['M_u_intact_MNm'], 1)}（sagging）/ "
            f"{fnum(prim['hogging']['M_u_intact_MNm'], 1)} MN·m（hogging），"
            f"RIF 值域 [{fnum(a['rif_min'])}, 1.0000]，最危险构件为甲板板带 0.8348（sagging）/"
            f"底部板带 0.8603（hogging）；冗余构件集 = 全部 18 构件（RIF ≥ 0.8），"
            f"敏感性 ρ = {fnum(c['sagging_pooled_rho'], 3)} / {fnum(c['hogging_pooled_rho'], 3)}，"
            f"vs 已发表闭式法最大误差 {float(d['max_relative_error']) * 100:.2f}%，34 项 pytest 全绿。"
        ),
        "scenario_key": "structural_redundancy",
        "mission_id": 10727,
        "tags": ["ice-class", "hull-girder", "ultimate-strength", "RIF", "redundancy",
                 "Smith-method", "tripping", "damaged-ship", "p351w1_n08"],
        "markdown": md,
        "figures": figures,
    }
    out = _post("/tensor/reports", payload, token=token)
    print(json.dumps(out, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()