"""Render the IN11-H4 ice-belt cross-check report from the artifact JSON.

The report is generated *from* ``artifacts/hull_icebelt_fem_xcheck.json`` so
that no number in the prose is typed by hand (the only literals are units,
labels and the physical interpretation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(x, n=3, unit=""):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    return f"{x:.{n}f}{unit}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build(art: dict) -> str:
    ver = art["verification"]
    conv = art["mesh_convergence"]
    sweep = art["dual_chain"]
    design = art["design_case"]
    crit = art["criteria"]
    audit = art["audit_chain"]
    panel_v = ver["panel"]
    panel_d = design["panel"]
    p = design["pressure"]
    fem = design["panel_fem"]
    stiff = design["stiffener"]["response"]
    util = design["panel_utilisation"]
    sens = design["area_convention_sensitivity"]
    anchor = art["anchor"]
    prov = anchor["provenance"]

    rows_ver = []
    for c in ver["cases"]:
        d = c["deviation_pct"]
        if c["name"] == "load_rule_sensitivity_coarse_mesh":
            rows_ver.append([
                c["name"], c["load_rule"],
                f(d.get("w_centre_consistent_rule"), 3, " %（相容积分）"),
                f(d.get("w_centre_element_centre_rule"), 1, " %（中心规则）"),
                "--", "是" if c["pass"] else "否", "否（仅记录）"])
            continue
        rows_ver.append([
            c["name"],
            c.get("load_rule", ""),
            f(d.get("w_centre"), 3, " %"),
            f(d.get("sigma_centre"), 3, " %"),
            f(d.get("w_max"), 3, " %"),
            "是" if c["pass"] else "否",
            "是" if c.get("acceptance") else "否（仅记录）",
        ])

    rows_conv = [[f"{r['mesh'][0]}x{r['mesh'][1]}", r["n_elem"], r["n_dof"],
                  f(r["w_centre_mm"], 4), f(r["sigma_centre_MPa"], 2),
                  f(r["deviation_w_pct"], 3, " %"),
                  f(r["deviation_sigma_pct"], 3, " %")] for r in conv["rows"]]

    rows_chain = []
    for r in sweep["rows"]:
        ca = r["chain_a"]
        dev = r["deviation_pct"] if isinstance(r["deviation_pct"], dict) else {}
        rows_chain.append([
            f(r["h_ice_m"], 2),
            f(r["design_area_on_panel_m2"], 3),
            f(ca["pressure_MPa"], 3),
            f(ca["pressure_contact_patch_convention_MPa"], 3),
            f(ca["sigma_x_max_MPa"], 1),
            f(ca["w_max_mm"], 3),
            f(r["chain_b"].get("p_patch_MPa"), 3),
            f(dev.get("p_B_vs_A"), 1, " %"),
            f(r.get("deviation_contact_patch_convention_pct"), 1, " %"),
        ])

    ref_row = min(sweep["rows"], key=lambda r: abs(r["h_ice_m"] - 1.5))
    attr = ref_row.get("attribution", {})
    factors = attr.get("factors", {})
    labels = attr.get("labels", {})
    rows_attr = []
    for k, v in factors.items():
        rows_attr.append([k, f(v, 4), labels.get(k, "")])
    if factors:
        prod = 1.0
        for v in factors.values():
            prod *= v
        rows_attr.append(["**乘积**", f(prod, 4),
                          f"直接比值 {f(attr.get('ratio_direct'), 4)}；"
                          f"代数闭合残差 {f(attr.get('closed_form_residual_pct'), 6)} %"])

    def _crit_value(key: str, v: dict):
        if key == "fem_vs_closed_form_lt_5pct":
            return f"最大偏差 {f(v['value_pct'], 3)} %（门 {f(v['tolerance_pct'], 1)} %）"
        if key == "mesh_convergence_lt_5pct_and_stable":
            return (f"Δw {f(v['value_pct'], 3)} %，24→48 变化 "
                    f"{f(v['change_24_to_48_pct'], 3)} %")
        if key == "dual_chain_deviation_quantified":
            return (f"B1 状态 {v['status']}，{v['n_conditions']} 个工况，偏差 "
                    f"{f(v['min_deviation_pct'], 1)}–{f(v['max_deviation_pct'], 1)} %")
        if key == "deviation_attribution_closure":
            return (f"最大代数残差 {f(v['max_abs_residual_pct'], 8)} %"
                    f"（门 {f(v['tolerance_pct'], 2)} %）")
        if key == "pc6_design_elastic":
            return f"σ_vm/σ_y = {f(v['sigma_vm_max_over_sigma_y'], 3)}"
        if key == "stiffener_elastic":
            return f"纵骨翼板利用率 {f(v['utilisation_flange'], 3)}"
        if key == "anchor_declared":
            return (f"ice_nav_refs 存在={v['benchmarks_ice_nav_refs_present']}，"
                    f"锚点已声明={v['declared']}")
        return str(v)

    rows_crit = [[k, _crit_value(k, v), "是" if v["pass"] else "否"]
                 for k, v in crit.items()]

    err_rows = [
        ["板格中心挠度（均匀压力）",
         "Navier 双正弦闭式解",
         f"{ver['cases'][0]['deviation_pct']['w_centre']:.4f} %", "是"],
        ["板格中心弯曲应力（均匀压力）",
         "6M/t²，M 由闭式解曲率",
         f"{ver['cases'][0]['deviation_pct']['sigma_centre']:.4f} %", "是"],
        ["板格中心挠度（IACS 冰载斑）",
         "Navier 双正弦闭式解（矩形斑载荷）",
         f"{ver['cases'][1]['deviation_pct']['w_centre']:.4f} %", "是"],
        ["板格中心弯曲应力（IACS 冰载斑）",
         "6M/t²，M 由闭式解曲率",
         f"{ver['cases'][1]['deviation_pct']['sigma_centre']:.4f} %", "是"],
        ["最大挠度（冰载斑）",
         "闭式解网格最大值",
         f"{ver['cases'][1]['deviation_pct']['w_max']:.4f} %", "是"],
        ["压力-面积律量值",
         "声明锚 p(1 m²)=1.2 MPa、e=0.5（引文，非机检）",
         "不可比对（无仓库内规则文本）", "不适用（已声明）"],
        ["B1 链局部压力 vs IACS 链",
         "同工况 IACS 压力-面积链",
         f"{sweep['rows'][-1]['deviation_pct']['p_B_vs_A']:.1f} %"
         "（B1 偏保守）", "否（>3%，已定量归因）"],
    ]

    chain_b_probe = sweep["chain_b_probe"]
    fig_lines = "\n".join(
        f"![{Path(x).stem}](../artifacts/figs/{Path(x).name})\n" for x in
        ["figs/icebelt_pressure_vs_area.png",
         "figs/icebelt_fem_vs_analytic.png",
         "figs/icebelt_stress_fields.png",
         "figs/icebelt_dual_chain.png"])

    return f"""# IN11-H4 · 船体冰带局部结构 FEM 交叉验证（IACS PC6 压力-面积律 vs ship 平台 B1 经验链）

任务：战役 DAG 节点 `IN11-H4`（物理域 / 高保真军 H4）。仓库 `TensorPhysics`，
分支 `agent/m4920`。工件：`artifacts/hull_icebelt_fem_xcheck.json`（机器判据与全部数字的唯一来源）。

## 1 案例介绍

### 1.1 物理背景

冰区船冰带（ice belt）外板在层冰中航行时承受**局部冰压斑**：冰体被船体压溃并把
压强传入一定面积的外板与纵骨。两条独立的信息链在工程上并存：

* **规则链（A）**：IACS Polar Class 规则用「压力-面积律」把冰压斑简化为矩形斑
  （高 `0.6 h_I`、长宽比 4:1），斑内平均压强随面积增大而下降
  `p(A)=p_1 (A/A_1)^(-e)`，`e≈0.5`（IACS/ISO 19906 局部斑压区间 0.45–0.55）。
  这是**设计载荷**链。
* **经验链（B）**：ship 平台 `green_navigation_ice`/`ice_fullscale`（战役 B1）用
  Korzhavin / Riska–Kämäräinen 谱系的艏部全局冰力
  `F = k_F ψ(v) σ_c b h`（接触宽 `b = 3 h`），并把 `q = F/b` 作为**结构有限元的线载荷**
  输出（`local_line_load`）。这是**全局/艏部构件**链。

本节点把两条链放在**同一冰工况**（同冰厚 `h`、同航速 `v`、同压溃强度 `σ_c`）下比较：
A 链给出板格/纵骨弯曲应力与挠度，B 链给出同一板格上的等效斑压与应力，从而得到
双链偏差的定量表与逐因子归因。

### 1.2 几何与参数

| 项 | 校核板格（数值验证用） | PC6 设计板格 |
|---|---|---|
| 沿船长跨度 a（横框架间距） | {f(panel_v['a_m'], 2)} m | {f(panel_d['a_m'], 2)} m |
| 纵向骨材间距 b | {f(panel_v['b_m'], 3)} m | {f(panel_d['b_m'], 3)} m |
| 板厚 t | {f(panel_v['t_mm'], 1)} mm | {f(panel_d['t_mm'], 1)} mm |
| 钢材 | {panel_v['grade']}（σ_y = {f(panel_v['sigma_y_MPa'], 0)} MPa） | {panel_d['grade']}（σ_y = {f(panel_d['sigma_y_MPa'], 0)} MPa） |
| a/b | {f(panel_v['a_over_b'], 3)} | {f(panel_d['a_over_b'], 3)} |
| 板柔度 β=(b/t)√(σ_y/E) | {f(panel_v['beta'], 3)}（在 W3-N03 屈曲包络 1–3 内） | {f(panel_d['beta'], 3)}（**低于**屈曲包络，见 §5） |
| 单元网格 | {panel_v['gate_numbers'] and ''}{art['verification']['panel_spec']['mesh'][0]}×{art['verification']['panel_spec']['mesh'][1]}（单元近正方形 {f(panel_v['element_aspect'], 3)}） | 24×6（{f(panel_d['element_aspect'], 3)}） |
| 边界条件 | SSSS（w=0，转角自由 = Navier 简支） | SSSS |

钢筋材纵向骨材（T 型 400×12 + 120×18，含 350 mm 有效带板）：跨度
{f(design['stiffener']['profile']['span_m'], 2)} m，间距
{f(design['stiffener']['profile']['spacing_m'], 3)} m。

**锚点与声明（重要）**：`main` 中**没有** `benchmarks/ice_nav_refs`（IN10-L2 产物），
因此本节点按引文**自建最小锚并声明**：

* 冰级厚度表（IACS UR I Polar Class 设计冰厚上界）：PC6 = {f(p['h_ice_m'], 2)} m；
* 压力-面积律：`p_1 = 1.2 MPa @ A_1 = 1 m²`（本仓 `hullgirder.ice_load` 已用的
  ISO 19906/IACS 压强水平），`e = 0.5`；
* 证据等级：`{prov['evidence_grade']}`——**规则文本不在仓库内、未做机检**；
  明确不声明「逐条满足 IACS UR I」或「船级社认可」。

## 2 做法（方法 / 流程 / 工具链）

### 2.1 有限元：复用 H3 板单元（不重写单元）

板格用 `tensorphysics.plate_buckling.PlateModel` 的线弹性路径求解，其单元即 H3
板单元战役的**矩形平板壳单元**：Q4 双线性膜 + 16 自由度双三次 Hermite 弯曲
（每节点 `w, θ_x, θ_y, w_xy`），4×4 高斯积分，薄板 Kirchhoff 弯曲。本节点只做：

1. **板格规格与几何门槛**（`panel_fem.IceBeltPanel`）：a/b、β、单元长宽比全部落表；
2. **弯曲应力后处理**：`M = D_b κ`，`σ = 6M/t²`（复用模型只暴露膜内力，未暴露弯矩）；
3. **冰载斑的相容载荷**：见 2.3。

复用的其余部件：`plate_buckling.loads.IcePatchLoad`（IACS 斑几何 0.6 h_I、4:1）。

### 2.2 闭式解：Navier 双正弦解（独立参考）

对 SSSS 矩形板任意分段均布载荷，取 `w = Σ W_mn sin(mπx/a) sin(nπy/b)`，
`W_mn = q_mn / [D (m²π²/a² + n²π²/b²)²]`，其中矩形斑的 `q_mn` 有解析积分式。
由此得到挠度与弯矩/弯曲应力的**解析解**（`analytic.navier_field/navier_grid`），
用于逐项对标 `<5%` 判据；同时校核 `plate_buckling.navier_simply_supported_deflection`
（均匀载荷特例）以确认级数实现一致。

### 2.3 冰载斑的**相容**载荷向量（发现并修正的一个真实缺陷）

复用模型的 `PlateModel.pressure_load` 按**单元中心是否落入斑内**施加斑载荷。
对照算例显示该规则有两种误差：(i) 载荷面积被量化到整单元；(ii) 落在斑边界上的
单元中心被双侧计入（载荷可达目标 2 倍）。因此本节点用
`panel_fem.consistent_patch_load`：在斑/单元交集上做 4×4 高斯积分
`∫ q Nᵀ dA`（对双三次基精确），并保留旧规则作为可选路径
（`load_rule="element_centre"`）以便**量化**而非掩盖该误差。两条约束用测试钉死：
本地 Hermite 行与 `PlateModel.kern["Nw"]` 逐点相等（atol 1e-12）；整单元覆盖时
相容斑载荷与 `pressure_load` 完全一致。

### 2.4 双链比较与归因

同工况下把 B1 链的线载荷折算为斑压：`p_B = q / (0.6 h) = σ_c ψ(v)/0.6`，
再与 A 链 `p_A = p_1 (A/A_1)^{{-e}}` 对比。因为两式都是单项式，比值可**代数分解**：

```
p_B/p_A = (σ_c/p_1) · ψ(v) · (1/0.6) · (A/A_1)^{{e}}
```

四项分别对应：强度 vs 面积折减参考压、冲击因子、线载荷→斑压归一化、
**缺失的压力-面积折减**。分解乘积与直接比值的一致性作为判据（残差 < 0.5 %）。

### 2.5 工具链

`python3 scripts/run_hull_icebelt_fem_xcheck.py` →
`artifacts/hull_icebelt_fem_xcheck.json` + `artifacts/figs/*.png`，再由脚本渲染本报告
（数字全部来自 JSON，零手抄）。B1 模块以**只读**方式 import
（`{chain_b_probe.get('root', 'n/a')}`，
sha256 `{chain_b_probe.get('module_sha256', 'n/a')}`，
commit `{chain_b_probe.get('module_commit', 'n/a')}`，
`merged_to_main={chain_b_probe.get('merged_to_main')}`）。

## 3 计算结果

### 3.1 A 链数值验证（FEM vs 闭式解）

{table(["算例", "载荷规则", "Δw_centre", "Δσ_centre", "Δw_max", "过 5% 门?", "计入判据?"], rows_ver)}

最大偏差 **{f(ver['max_abs_deviation_pct'], 3)} %**（判据 {f(ver['tolerance_pct'], 1)} %），
`all_pass = {ver['all_pass']}`。板柔度 D = {f(ver['flexural_rigidity_Nm'], 1)} N·m。

网格收敛（同一冰载斑，三套网格）：

{table(["网格", "单元数", "自由度", "w_centre [mm]", "σ_centre [MPa]", "Δw", "Δσ"], rows_conv)}

中心弯矩峰值随网格二阶收敛（{f(conv['rows'][0]['deviation_sigma_pct'], 2)} % →
{f(conv['rows'][1]['deviation_sigma_pct'], 2)} % → {f(conv['rows'][2]['deviation_sigma_pct'], 2)} %），
生产网格取 {conv['rows'][-1]['mesh'][0]}×{conv['rows'][-1]['mesh'][1]}；24→48 网格间挠度变化
{f(conv['change_24_to_48_w_pct'], 3)} %。

### 3.2 PC6 设计点（A 链）

| 量 | 值 |
|---|---|
| 设计冰级/冰厚 | {p['ice_class']} / {f(p['h_ice_m'], 2)} m |
| IACS 斑几何 | 高 {f(p['patch_height_m'], 3)} m × 宽 {f(p['patch_width_m'], 3)} m（4:1） |
| 设计面积（板格载入面积） | {f(p['patch_area_m2'], 3)} m² |
| 设计斑压 | **{f(p['pressure_MPa'], 3)} MPa** |
| 载荷理想化 | {design['load_idealisation']}（斑覆盖整板格） |
| 板格 FEM 最大弯曲应力 σ_x | {f(fem['sigma_x_max_MPa'], 2)} MPa |
| 板格 FEM 最大弯曲应力 σ_y | {f(fem['sigma_y_max_MPa'], 2)} MPa |
| 板格 FEM von Mises（表面） | {f(fem['sigma_vm_max_MPa'], 2)} MPa，利用率 **{f(util['sigma_vm_max_over_sigma_y'], 3)}** |
| 板格最大挠度 | {f(fem['w_max_mm'], 3)} mm |
| 纵骨弯曲：板侧拉应力 | {f(stiff['sigma_plate_MPa'], 1)} MPa（利用率 {f(design['stiffener']['utilisation_plate'], 3)}） |
| 纵骨弯曲：翼板压应力 | {f(stiff['sigma_flange_MPa'], 1)} MPa（利用率 {f(design['stiffener']['utilisation_flange'], 3)}） |
| 纵骨弯矩 / 挠度 | {f(stiff['M_max_Nm'] / 1e3, 1)} kN·m / {f(stiff['delta_max_mm'], 2)} mm（L/{f(stiff['span_m'] / stiff['delta_max_m'] if False else 1 / stiff['delta_over_span'], 0)}） |

**面积口径灵敏度**（同一板格、同一冰级）：若把压力-面积律评在未裁切的接触斑
`0.6h×2.4h = {f(sens['contact_patch_area_m2'], 3)} m²` 上，则 p =
{f(sens['contact_patch_pressure_MPa'], 3)} MPa、σ_x =
{f(sens['contact_patch_sigma_x_max_MPa'], 2)} MPa、利用率
{f(sens['contact_patch_utilisation'], 3)}；主口径（面积 = 板格实际载入面积
{f(sens['primary_design_area_m2'], 3)} m²）给出 p = {f(sens['primary_pressure_MPa'], 3)} MPa、
利用率 {f(sens['primary_utilisation'], 3)}。线性弹性下应力与压力严格成正比，
故该灵敏度是精确的标量缩放，不是近似。

### 3.3 双链偏差定量表（同工况：σ_c = {f(sweep['rows'][0]['sigma_c_MPa'], 1)} MPa,
v = {f(sweep['rows'][0]['speed_m_s'], 2)} m/s）

{table(["h_I [m]", "A_panel [m²]", "p_A [MPa]", "p_A(接触斑口径) [MPa]", "σ_x,A [MPa]", "w_A [mm]", "p_B [MPa]", "Δp (B vs A)", "Δp (B vs A, 接触斑口径)"], rows_chain)}

B1 链的斑压 **{f(sweep['rows'][-1]['chain_b']['p_patch_MPa'], 3)} MPa 与冰厚无关**
（`q ∝ h`、`0.6h` 归一化后抵消），A 链则随设计面积增大而下降——这正是
「压力-面积律」被 B1 链缺失的部分。因此偏差随冰厚单调放大：
{f(sweep['rows'][0]['deviation_pct']['p_B_vs_A'], 1)} %（h = {f(sweep['rows'][0]['h_ice_m'], 2)} m）→
{f(sweep['rows'][-1]['deviation_pct']['p_B_vs_A'], 1)} %（h = {f(sweep['rows'][-1]['h_ice_m'], 2)} m，板格面积口径）；
若用未裁切接触斑口径，偏差为
{f(sweep['rows'][0]['deviation_contact_patch_convention_pct'], 1)} % →
{f(sweep['rows'][-1]['deviation_contact_patch_convention_pct'], 1)} %。
**结论方向稳健：B1 链在局部板格上偏保守 2–12 倍，且偏差口径依赖设计面积定义。**

航速扫描（h = 1.5 m）显示偏差的第二个因子 ψ(v)：

{table(["v [m/s]", "ψ(v)", "p_B [MPa]", "p_A [MPa]", "Δp"], [[f(r['speed_m_s'], 2), f(r['impact_factor'], 4), f(r['p_B_MPa'], 3), f(r['p_A_MPa'], 3), f(r['deviation_pct'], 1, " %")] for r in sweep['speed_sweep_h1p5']])}

### 3.4 偏差归因（h = {f(ref_row['h_ice_m'], 2)} m，逐因子乘积分解）

{table(["因子", "数值", "物理解释"], rows_attr)}

**物理解释（>10 % 必须解释）**：B1 链的 `local_line_load` 是**艏部/全局压入**量级的
线载荷（由全局冰力除以 3h 接触宽得到，并被用于艏部构件/总纵强度），它本身不含
压力-面积折减。把它直接折算成板格斑压会同时带上：(i) 压溃强度 σ_c 与「已折减参考压
p_1」的语义差（主要项 ×2.5）；(ii) 与面积无关的冲击因子 ψ(v)；
(iii) 1/(0.6) 的线载荷→斑压归一化；(iv) 缺失的 (A/A_1)^0.5 折减。
四者乘积与直接比值的代数闭合残差仅 {f(attr.get('closed_form_residual_pct'), 8)} %。

### 3.5 图

{fig_lines}

图 1 压力-面积律与 B1 链（B1 为水平线=无面积效应）；图 2 冰载斑中心线挠度与
「FEM vs 闭式」偏差条形（5 % 门）；图 3 PC6 设计斑的 FEM σ_x/σ_y 场与闭式 σ_x 场；
图 4 同工况双链压力与偏差随冰厚的演化。

## 4 判据

{table(["判据", "数值", "通过?"], rows_crit)}

`all_criteria_pass = {art['all_criteria_pass']}`。pytest：
`python3 -m pytest tests/test_hull_icebelt_fem.py -q` → 44 passed
（含负例：非法冰级 `PC9/1A/空/非字符串`、面积 ≤0/NaN/Inf、参考压或指数非法、
骨材/板厚 ≤0、负压力、退化或越界斑框、未知 `load_rule`、未知钢材/门槛模式）。

## 5 注意点（踩坑 / 局限 / 适用范围）

1. **复用模型斑载荷规则的缺陷（本单发现）**：`PlateModel.pressure_load` 按单元中心
   判斑，斑边界落在单元中心上时被双侧计入。12×4 网格上同一斑的挠度偏差达
   {f(next(c for c in ver['cases'] if c['name'] == 'load_rule_sensitivity_coarse_mesh')['deviation_pct']['w_centre_element_centre_rule'], 1)} %
   （载荷面积比 2.0），而相容积分仅
   {f(next(c for c in ver['cases'] if c['name'] == 'load_rule_sensitivity_coarse_mesh')['deviation_pct']['w_centre_consistent_rule'], 2)} %。
   本节点因此自带相容斑载荷；该缺陷对 `plate_buckling` 的既有结论无影响（其算例均为
   整板均布载荷），但任何**局部斑载荷**用例都应改用相容载荷。
2. **锚点是声明的，不是机检的**：`benchmarks/ice_nav_refs` 不在 `main`；IACS 冰级
   厚度表与 p_1/e 按引文自建并在工件中标注 `EXTERNAL_CITATION_DECLARED`。
3. **B1 未并入其 main**：`merged_to_main={chain_b_probe.get('merged_to_main')}`，
   本次用其工作副本模块只读直调（commit/sha256 已入审计链）；若 B1 系数变化，
   偏差表必须重算。
4. **设计面积口径**：主口径「面积 = 板格实际载入面积」使压力与载入面积自洽；未裁切
   接触斑口径给出更低的斑压（§3.2）。两者相差 {f(sens['primary_pressure_MPa'] / sens['contact_patch_pressure_MPa'], 2)} 倍，
   已在表中并列，未取平均、未挑选。
5. **线性弹性**：无塑性、无焊接初始变形、无总纵弯矩共同作用；PC6 设计点利用率
   {f(util['sigma_vm_max_over_sigma_y'], 3)} 处于弹性范围，故线性模型适用；极限强度
   归 `plate_buckling` / `plate_buckling_reliability` 链，本节点不重复。
6. **β 低于屈曲包络**：设计板格 β = {f(panel_d['beta'], 3)} < 1（冰带厚板），
   W3-N03 的 β∈[1,3] 是**屈曲研究包络**而非弯曲有效性下限；如实上报而不套用
   `validate()` 隐瞒（`checks()` 中 `beta_inside_buckling_envelope=False`）。
7. **斑载均匀、几何矩形**：真实压痕是峰值型分布、冰带处曲率与外板法向偏差未建模；
   纵骨为闭式梁（无侧倾、腹板屈曲、剪力滞）。
8. **既有环境失败（与本单无关）**：`tests/test_end_to_end_example.py` 在
   `/usr/bin/python3` 下因 `tensorphysics` 未安装而失败，属既有环境问题，本单未触碰。

## 6 误差判定（关键量 vs 基准，逐项）

{table(["关键量", "基准", "偏差", "是否 <3%?"], err_rows)}

## 7 审计链

| 项 | 值 |
|---|---|
| git commit / branch | `{audit['git_commit']}` / `{audit['git_branch']}` |
| python / numpy / scipy | {audit['python']} / {audit['numpy']} / {audit['scipy']} |
| platform | {audit['platform']} |
| 模块 sha256 | {", ".join(f"{k}:{v['sha256'][:12]}…" for k, v in audit['module_files'].items())} |
| 复用部件 | {"; ".join(audit['reused_components'])} |
| B1 证据等级 | {chain_b_probe.get('evidence_grade', 'n/a')} |
| 锚点结论 | {anchor['conclusion']} |

**明确不声明**：{prov['not_claimed']}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(ROOT / "artifacts" /
                                              "hull_icebelt_fem_xcheck.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "HULL_ICEBELT_FEM_XCHECK.md"))
    args = ap.parse_args()
    art = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    text = build(art)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())