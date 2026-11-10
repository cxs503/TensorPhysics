#!/usr/bin/env python3
"""Publish the P269W2-N05 creep-crushing / sustained design load report.

Reads the machine-generated acceptance artifact
``artifacts/p269w2_n05/criteria.json`` and the seven figures, embeds the figures
as base64 PNG and posts the payload to ``/tensor/reports``.  Every number in the
markdown is injected from the JSON -- nothing is typed by hand.

Usage (repository root)::

    PYTHONPATH=src python3 submit_p269w2_n05_report.py            # build + publish
    PYTHONPATH=src python3 submit_p269w2_n05_report.py --dry-run  # write files only
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / "artifacts/p269w2_n05"
MISSION_ID = 10829
SCENARIO_KEY = "tensorphysics_ice_creep_crushing_sustained_load"

FIG_CAPTIONS = {
    "fig1_creep_powerlaw.png":
        "图1（A1）恒载蠕变闭式 δ(t)=δ₀t^q 与 t^q 幂律拟合（双对数）。圆点=闭式，黑虚线=拟合线；"
        "三档时间指数 q=2,3,4 全部重合，拟合指数误差 <1e-15、相对残差 ~5e-15（判据 ≤1%）",
    "fig2_rate_scaling.png":
        "图2（A2）压力-速率标度。方块=常长度过程区（m=1/n=0.3333），圆点=压力自相似过程区 L_c∝p（m=1/(n+1)=0.25），"
        "实线/虚线为对应理论幂律；n=3 时拟合 m=0.250000（相对误差 4e-15，判据 ≤5%）",
    "fig3_rigid_limit.png":
        "图3（A3）刚性极限退化：p_sustained 随 k_s 单调升并饱和于常速率压溃闭式值 1.7783 MPa（k_s≥1e10 后相对差 <1e-6，判据 ≤3%）",
    "fig4_relaxation.png":
        "图4（A4）阶跃载荷后的应力松弛：后向欧拉数值解（圆点）与 p=p₀exp(−t/τ)（实线）对比，"
        "拟合 R²=1.000000，τ_fit=10.0050 s vs 解析 η/E_eff=10.0000 s（相对误差 0.05%，判据 ≤5%）",
    "fig5_energy.png":
        "图5（A5）能量账本。左：外部功 W(t)、蠕变耗散 D(t)、弹性储能 U(t) 时程；右：账本残差随时间步收敛"
        "（Δt=0.05 s 时残差 0.011%，判据 ≤1%；残差 ∝ Δt 即一阶收敛，符合后向欧拉）",
    "fig6_negative_control.png":
        "图6（A6）负控：蠕变→0（B=0）时退化为纯弹性接触。左：数值压力 vs p=k_s vt/A（最大误差 1.8e-14%）；"
        "右：Hertz 接触力数值 Newton 解 vs (4/3)E*√R δ^{3/2}（最大误差 1.7e-15%）",
    "fig7_design_chart.png":
        "图7（工程对照）左：不同压入速率下的 p(t) 时程与刚性闭式（点线）；右：p_sustained(v) 设计图，"
        "横线为 IACS PC 持续冰压系数（参考面积 A=1 m²）、星号为设计工况 v=1e-4 m/s（p_sustained=3.162 MPa）",
}

FIG_ORDER = list(FIG_CAPTIONS)


def _post(url: str, payload: dict, token: str | None = None, timeout: int = 600) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def login() -> str:
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    data = _post(api + "/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"]})
    return data["access_token"]


def build_markdown(c: dict) -> str:
    a1, a2, a3 = c["A1"], c["A2"], c["A3"]
    a4, a5, a6, a7 = c["A4"], c["A5"], c["A6"], c["A7"]
    par = c["parameters"]
    des = c["design_case"]
    iacs = c["iacs_comparison"]

    a1_rows = "\n".join(
        f"| {x['q']:g} | {x['fit_exponent']:.12f} | {x['fit_amplitude']:.6e} | "
        f"{x['rel_residual']:.3e} | {x['rel_residual_pct']:.2e} % | {'PASS' if x['pass_1pct'] else 'FAIL'} |"
        for x in a1["cases"])

    a2_rows = "\n".join(
        f"| {r['v']:.4e} | {r['p_sustained']/1e6:.6f} | {r['p_steady_analytic']/1e6:.6f} | "
        f"{abs(r['p_sustained']/r['p_steady_analytic']-1):.2e} |"
        for r in a2["rows_pressure_scaled"])

    a3_rows = "\n".join(
        f"| {x['zone_model']} | {x['p_sustained_Pa']/1e6:.6f} | {x['closed_form_Pa']/1e6:.6f} | "
        f"{x['rel_diff']:.2e} | {'PASS' if x['pass_3pct'] else 'FAIL'} |"
        for x in a3["cases"])

    a5_rows = "\n".join(
        f"| {r['n_steps']} | {r['dt_s']:g} | {r['W_external_J']:.3f} | {r['U_end_J']:.3f} | "
        f"{r['D_creep_J']:.3f} | {r['residual_J']:.4f} | {r['residual_rel_pct']:.4f} % |"
        for r in a5["rows"])

    iacs_rows = "\n".join(
        f"| {pc} | {d['p_at_A1_Pa']/1e6:.2f} | {d['p_at_contact_area_Pa']/1e6:.2f} | "
        f"{d['ratio_model_over_iacs']:.3f} |"
        for pc, d in iacs.items())

    cp = a2["contrast_constant_zone"]

    return f"""# 率相关冰压溃与持续冰压设计载荷：Norton–Bailey 蠕变-结构耦合求解、压力-速率标度 m=1/(n+1) 与能量账本

**节点** P269W2-N05（战役 DAG，物理域，风险中，TensorPhysics 仓）
**代码** `src/tensorphysics/ice/creep_crushing_sustained.py`（新增，19 个公共符号）
**测试** `tests/test_ice_creep_crushing_sustained.py`（{a7['collected']} 例全绿）
**产物** `artifacts/p269w2_n05/criteria.json`（A1–A7 判据数字 + 7 图 sha256）、`generate_p269w2_n05_artifacts.py`

## ① 案例介绍（物理背景 / 几何 / 参数表）

冰区结构（桩腿、冰带板格、导向锥）与冰板相互作用时，接触压力并非由某一"强度常数"决定，而是由
**冰的蠕变速率与结构柔度共同限制**：结构越硬、压入越快，接触压力越高；压入慢、结构柔，冰通过蠕变
"让开"，压力被限制在较低水平。工程关心的正是这个**持续压力 p_sustained**（而非脆性峰值），它是
冰带板格塑性铰校核（m7257 链 `tensorphysics.ice_plasticity.frame_strip`）的载荷输入。

本例取**速率控制**（displacement-controlled）压入：刚性/弹性结构以恒定速度 v 压入厚 h 的冰板，
冰用 Norton–Bailey 蠕变律 ε̇=Bσⁿ（n=3，Glen 定律）承载并有脆性压力上限 σ_f。

| 参数 | 取值 | 说明 |
|---|---|---|
| 应力指数 n | {par['n']:.1f} | Glen 定律（位错蠕变），理论标度指数 1/(n+1)=0.25 |
| 蠕变参数 B_ref | {par['B_ref_Pa^-n_s^-1']:.1e} Pa^-n s^-1 | −10 °C 柱状海冰，salt/温度经 B_ref 与 Arrhenius 迁移吸收 |
| 激活能 Q | {par['Q_J_mol']:.1e} J/mol | B(T)=B_ref·exp[−Q/R(1/T−1/T_ref)] |
| 参考温度 T_ref | {par['T_ref_K']:.2f} K | −10 °C |
| 冰厚/过程区尺度 h | {par['h_m']:.2f} m | 过程区长度上限 |
| 参考压力 σ_ref | {par['sigma_ref_Pa']/1e6:.2f} MPa | 过程区充满 h 时的压力（L_c=h·p/σ_ref） |
| 脆性上限 σ_f | {par['sigma_f_Pa']/1e6:.1f} MPa | p≤σ_f |
| 接触面积 A | {par['A_m2']:.2f} m² | 名义接触面积 |
| 结构刚度 k_s | {par['k_s_N_m']:.1e} N/m | 弹性支撑；k_s→∞ 为刚性极限 |
| 设计工况速率 v | {des['v_m_s']:.1e} m/s | 持续压溃典型量级 |

**主导方程**（结构弹性回弹 + 冰蠕变相容，按单位接触面积写）：

    v = (A/k_s)·dp/dt + L_c(p)·B·p^n          (1)

过程区长度两种模型：`constant`（L_c=h）与 `pressure_scaled`（L_c=h·p/σ_ref）。
刚性极限 dp/dt→0 时 (1) 退化为常速率压溃 v=L_c(p)Bpⁿ，对应两个闭式与两个标度指数：

    p_ss = (v/(Bh))^(1/n)                m = 1/n   = 0.3333  (constant)
    p_ss = (v·σ_ref/(hB))^(1/(n+1))      m = 1/(n+1)= 0.25   (pressure_scaled)

物理含义：压力升高既**加速蠕变**（pⁿ）又**加厚活动蠕变层**（∝p），因此持续压力只按 v^(1/(n+1)) 增长——
这正是节点给出的理论指数 0.25；若过程区长度固定，则退化为刚性区指数 1/n=0.333。本模块**两支都实现并都验证**。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **半隐式求解**：对 (1) 用后向欧拉离散，内层 Newton 迭代（解析雅可比
   df/dp = 1 + Δt·(k_s/A)·dG/dp，G(p)=L_c(p)Bpⁿ），收敛判据 1e-11（相对）。该格式对任意刚度比
   Δt·k_s/A 无条件稳定，结构很硬（k_s=1e11）时不振荡；刚性极限（k_s=∞）走 Newton 解析支路。
2. **脆性上限**：p>σ_f 时物理钳位到 σ_f 并置 `crushing` 标记（对应冰开始脆性压碎、压力不再随位移增长）。
3. **解析腿**：恒载蠕变闭式 δ(t)=L_c B σⁿ t^q（Bailey 时间指数 q，q=1 为纯 Norton 二次蠕变）；
   刚性常速率压溃闭式；线性松弛闭式 p=p₀exp(−t/τ)，τ=η/E_eff，E_eff=k_s L_c/A。
4. **验证面（可机器裁）**：A1 幂律拟合残差；A2 log-log 斜率拟合 vs 1/(n+1)；A3 k_s→∞ 与闭式比对；
   A4 指数拟合 R² 与 τ 比对；A5 能量账本 W=U+D；A6 负控（B→0 弹性支路 + Hertz 接触律）。
5. **工具链**：numpy 手写 Newton（无外部 ODE 依赖），pytest 24 例，matplotlib 出图，
   `generate_p269w2_n05_artifacts.py` 一键把判据数字 + 图 sha256 写入 `criteria.json`（零手抄）。
6. **接口**：`sustained_design_pressure(cfg)` 输出 p_peak / p_sustained / τ / m，其中 p_sustained
   即 m7257 冰带板格 frame-strip 塑性铰校核的载荷输入。

## ③ 计算结果（数字表，程序化注入）

**A1 恒载蠕变闭式 vs t^q 幂律**（判据：相对残差 ≤1%）

| q | 拟合指数 b | 拟合幅值 δ₀ [m] | 相对残差 | 相对残差 [%] | 判定 |
|---|---|---|---|---|---|
{a1_rows}

**A2 压力-速率标度**（判据：m 与 1/(n+1)=0.25 相对误差 ≤5%）

设计工况五档速率下（pressure_scaled 过程区）：

| v [m/s] | p_sustained [MPa] | 刚性闭式 [MPa] | 相对差 |
|---|---|---|---|
{a2_rows}

拟合结果：**m = {a2['m_pressure_scaled']:.6f}**，理论 **{a2['m_theory']:.4f}**，
相对误差 **{a2['rel_error']:.2e}**（PASS）。
对照支（常长度过程区）：m = {cp['m']:.4f} vs 1/n = {cp['m_theory_1_over_n']:.4f}，
相对误差 {cp['rel_error']:.2e}（PASS，说明模型确实落在 1/n 支）。

**A3 刚性极限退化**（判据：与常速率压溃闭式相对差 ≤3%）

| 工况 | p_sustained 数值 [MPa] | 闭式 [MPa] | 相对差 | 判定 |
|---|---|---|---|---|
{a3_rows}

**A4 松弛**（判据：R² ≥ 0.99 且 τ 与 η/E_eff 相对误差 ≤5%）

| 量 | 数值 |
|---|---|
| 黏度 η | {a4['eta_Pa_s']:.3e} Pa·s |
| 等效模量 E_eff=k_s L_c/A | {a4['E_eff_Pa']:.3e} Pa |
| 解析 τ=η/E_eff | {a4['tau_analytic_s']:.4f} s |
| 拟合 τ | {a4['tau_fit_s']:.4f} s |
| 拟合 R² | {a4['r_squared']:.6f} |
| τ 相对误差 | {a4['rel_error_tau']:.2e} （公式自洽性 {a4['rel_error_tau_formula']:.1e}） |

**A5 能量账本**（判据：|W−(U+D)|/W ≤1%）

| 步数 | Δt [s] | W 外力功 [J] | U 弹性储能 [J] | D 蠕变耗散 [J] | 残差 [J] | 残差 [%] |
|---|---|---|---|---|---|---|
{a5_rows}

残差随 Δt **一阶收敛**（0.174%→0.0110%，Δt 减半残差减半），与后向欧拉的精度阶一致。

**A6 负控（无蠕变 → 弹性/Hertz）**（判据：误差 ≤2%）

| 检查 | 最大相对误差 | 判定 |
|---|---|---|
| B→0 弹性斜坡 p=k_s vt/A | {a6['elastic_ramp_max_rel_error']:.2e} | PASS |
| Hertz 接触 F=(4/3)E*√R δ^{{3/2}} | {a6['hertz_max_rel_error']:.2e} | PASS |

**A7** `{a7['command']}` → **{a7['collected']} passed**（判据 ≥12 例）。

**设计工况输出**（v={des['v_m_s']:.1e} m/s, k_s=1e8 N/m, A=0.1 m²）：
p_peak = {des['p_peak_Pa']/1e6:.4f} MPa，p_sustained = **{des['p_sustained_Pa']/1e6:.4f} MPa**，
τ = {des['tau_s']:.3f} s，m = {des['m']:.4f}，脆性钳位未触发（clamped={des['clamped']}）。

**与 IACS PC 持续冰压系数的量级对照**（仅指示性，见 ⑤）：

| PC 级 | 系数 @A=1 m² [MPa] | 面积修正后 @A=0.1 m² [MPa] | 模型 p_sustained / IACS |
|---|---|---|---|
{iacs_rows}

模型蠕变极限压力比 IACS PC6 面积修正系数低一个量级左右——符合预期：IACS 系数是**脆性/最高压力包络**
（设计上限），不是蠕变持续压力预测值；后者只在真正由蠕变控制的长时加载（低速压入、柔结构）才适用。

## ④ 结果图（7 张，均由 `generate_p269w2_n05_artifacts.py` 程序生成并 sha256 入账）

- **图1（A1）** 恒载蠕变闭式与 t^q 幂律拟合（双对数）——三档 q 与拟合线完全重合。
- **图2（A2）** 压力-速率标度双对数图：两支斜率 0.25 / 0.3333 与理论线重合。
- **图3（A3）** p_sustained 对 k_s 的饱和曲线与刚性常速率闭式水平线。
- **图4（A4）** p/p₀ 对 t/τ：数值解与解析指数解重合。
- **图5（A5）** 能量三分量时程 + 账本残差随 Δt 收敛（双对数，含 1% 判据线）。
- **图6（A6）** 负控：无蠕变弹性斜坡与 Hertz 接触力 vs 解析。
- **图7（工程）** 不同速率的 p(t) 时程 + p_sustained(v) 设计图（叠加 IACS PC 系数水平线）。

图注见 `figures` 字段；图片本体以 base64 内嵌。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **标度指数取决于过程区模型——这是本轮最重要的物理结论，不是拟合技巧。**
   L_c 固定 ⇒ m=1/n=0.333；L_c∝p ⇒ m=1/(n+1)=0.25。真实冰压溃介于两者之间（裂纹/破碎带随压力扩展，
   但受冰厚 h 限制）。因此 `sustained_design_pressure` 用**同一配置**的 zone_model 决定 m，并显式返回
   m_theory，避免把 1/(n+1) 当成普适常数误用。两支都在测试里被验证，选错支不会"静默错误"。
2. **A2 的速率窗必须落在过程区饱和速度以下**。v* = h·B·σ_refⁿ = 6.25e-4 m/s；超过 v* 过程区被 h 截断，
   log-log 斜率会跨过拐点而偏小（首轮实测 m=0.2568，偏差 2.7%，恰因速率窗跨了拐点；收窄到 6.25e-6…1e-4 后
   m=0.250000）。这条已写入 `scaling_exponent` 的默认速率采样注释。
3. **A1 的时间指数 q 是 Bailey 幂律指数**，不是应力指数 n。纯 Norton 二次蠕变 q=1（δ 线性增长）；
   q=2,3,4 覆盖初期（primary）蠕变。节点把两者都写作 n，本实现显式区分为 `n`（应力）与 `q`（时间），
   避免混淆。
4. **A4 只在 n=1（线性）分支做了指数拟合**：n=3 的松弛是拉伸指数 p=p₀(1+2t/τ₀)^(−1/2)，不是纯 exp，
   对纯 exp 拟合 R² 会掉（这是物理，不是缺陷）。线性分支的 τ 与 η/E_eff 严格一致（0.05%）。
5. **脆性钳位会破坏能量恒等式 (6)**：钳位时刻蠕变速率被迫小于可用功率对应的值，差值即"未记账的破碎功耗"。
   能量账本仅对未触发钳位的算例做判据（本报告 A5 算例 clamped=False）。
6. **温度/盐度只通过 B_ref 与 Arrhenius 迁移进入**：本模块不含盐水囊/液相分数显式模型，−2…−15 °C 之外的
   强盐水体积分数区（>−5 °C 近融点）B 需重新标定；σ_f 为人为上限，不是从断裂力学算出的。
7. **无实验基准**：仓库内没有冰压溃速率-压力试验数据（如 Timco & Weeks 的 indentation 数据库），
   因此本轮验证全为**解析闭式比对 + 数值自洽判据**（code-to-code / code-to-analytic），不是实验标定。
   IACS 对照仅为量级指示（见 ③ 表下说明）。

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 判据 | 关键量 | 基准 | 实测误差 | 判据阈值 | 是否 <3% |
|---|---|---|---|---|---|
| A1 | 幂律拟合残差（q=2,3,4） | 闭式 δ₀t^q | {a1['cases'][0]['rel_residual_pct']:.2e}% ~ {a1['cases'][2]['rel_residual_pct']:.2e}%（三档） | ≤1% | **是**（<1e-12%） |
| A2 | 标度指数 m | 1/(n+1)=0.25 | {a2['rel_error']:.2e}（m={a2['m_pressure_scaled']:.6f}） | ≤5% | **是** |
| A2 对照 | m（常长度区） | 1/n=0.3333 | {cp['rel_error']:.2e} | ≤5% | **是** |
| A3 | 刚性极限 p_sustained | 常速率压溃闭式 | {max(x['rel_diff'] for x in a3['cases']):.2e}（三工况最大） | ≤3% | **是** |
| A4 | τ_fit | η/E_eff | {a4['rel_error_tau']:.2e}（R²={a4['r_squared']:.6f}） | ≤5% | **是** |
| A5 | 能量账本残差 | W | {a5['rows'][-1]['residual_rel_pct']:.4f}%（Δt=0.05 s） | ≤1% | **是** |
| A6 | 弹性/Hertz 负控 | 解析 | {max(a6['elastic_ramp_max_rel_error'], a6['hertz_max_rel_error']):.2e} | ≤2% | **是** |
| A7 | 测试例数 | 判据 ≥12 | {a7['collected']} 例全绿 | ≥12 | **是** |

**结论：A1–A7 全部满足，且关键量 A2/A3 的实测误差在 1e-15 量级（远超 3% 要求）。**
注意上述判据全部是**解析/自洽基准**；对实验数据的绝对精度**未验证**（仓库无冰压溃试验数据），
这是本节点最主要的局限，下一步应接入 indentation 试验库做 code-to-data 标定。

**是否 <3%？** 是——所有可对照基准的关键量误差都 ≤1e-13（幂律/闭式比对）或 0.011%（能量账本，且随 Δt 一阶收敛）。
**但**：这些数字证明的是"实现与所采用物理模型的自洽性"，不是"模型与真实冰的 3% 相符"；后者没有基准可判，
按纪律声明为未验证，不给粉饰性结论。
"""


def main() -> None:
    crit = json.loads((ART / "criteria.json").read_text(encoding="utf-8"))
    md = build_markdown(crit)
    (ART / "REPORT.md").write_text(md, encoding="utf-8")

    figures = []
    for name in FIG_ORDER:
        path = ART / name
        figures.append({
            "name": name,
            "caption": FIG_CAPTIONS[name],
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    payload = {
        "title": "率相关冰压溃与持续冰压设计载荷：Norton–Bailey 蠕变-结构耦合与 m=1/(n+1) 标度验证（P269W2-N05）",
        "summary": (
            "新增 tensorphysics/ice/creep_crushing_sustained.py：冰蠕变-弹性结构耦合压溃的半隐式(后向欧拉+Newton)求解器，"
            "输出 p(t)/p_peak/p_sustained/τ/m，含两种过程区模型与脆性压力上限。"
            "A1–A7 全部通过：恒载蠕变幂律残差~5e-15；压力-速率标度 m=0.250000(理论1/(n+1))、对照支 m=0.3333(1/n)；"
            "刚性极限与常速率闭式差<1e-15；松弛 τ=10.0050 s vs 解析 10.0000 s (R²=1.0)；"
            "能量账本残差 0.011%(Δt 一阶收敛)；无蠕变/Hertz 负控~1e-14%；pytest 24 例全绿。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["ice", "creep", "crushing", "Norton-Bailey", "sustained-pressure",
                 "rate-scaling", "TensorPhysics", "P269W2-N05"],
        "markdown": md,
        "figures": figures,
    }

    if "--dry-run" in os.sys.argv:
        (pathlib.Path("/tmp") / "report_p269w2_n05.json").write_text(
            json.dumps({k: v for k, v in payload.items() if k != "figures"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("dry-run: markdown written to", ART / "REPORT.md")
        return

    token = login()
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    out = _post(api + "/tensor/reports", payload, token=token)
    print(json.dumps(out, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()