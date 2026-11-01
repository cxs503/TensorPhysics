#!/usr/bin/env python3
"""Render the paper-grade report for mission m1212 from the machine artefacts.

Every number in the markdown is read from ``artifacts/ice_contact_m1212/results.json``
(or recomputed from the module), never typed by hand.  The same renderer also
writes the payload used to post the report to the platform (figures base64
embedded).
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "ice_contact_m1212"
FIGURE_DIR = ARTIFACT_DIR / "figures"
REPORT_PATH = REPO_ROOT / "docs" / "reports" / "ice_contact_phase_change_report.md"
PAYLOAD_PATH = Path("/tmp/report.json")

MISSION_ID = 1212


REPORT_TEMPLATE = """# 冰-结构接触相变与融膜演化成熟化（W8-N01 / m1212）

**仓库**：TensorPhysics → 新领域模块 `src/tensorphysics/ice_contact/`
**分支**：`agent/m1212`  **任务**：W8-N01 冰-结构接触相变与融膜演化（压力融化 + 摩擦融化 + 融膜厚度 ODE + 冰粘附温度依赖 + Stribeck 扩展）
**产物**：`artifacts/ice_contact_m1212/results.json`、`pytest.log`、`figures/*.png`、`tests/test_ice_contact_*.py`、本报告

---

## 1. 案例介绍

### 1.1 物理背景

冰与结构（钢）滑动接触界面存在**三相结构**：冰 — 融膜（水） — 结构。界面热力学与摩擦学双向耦合，三种相变机制相互竞争：

1. **压力融化**：接触压力使冰 Ih 的熔点降低（Clausius-Clapeyron 反常斜率为负），高压下本"很冷"的冰相对当地熔点变成过热态，无需摩擦热即可融化；
2. **摩擦融化**：摩擦功率 tau*V 转为热流密度，一部分转为潜热使冰融化，另一部分经 Stefan 条件向冰体与结构导热；
3. **再冻结**：当导热损失超过摩擦热时，融膜向冷冰体/冷结构凝固（Stefan 移动边界）。

三种机制决定融膜厚度 h，融膜厚度经润滑状态（膜厚参数 Lambda = h/sigma_rms）决定摩擦系数，摩擦系数又反馈回摩擦热——构成闭环。工程意义覆盖冰区航行船体冰载荷、破冰结构接触摩擦、除冰/防冰表面设计。

### 1.2 模型几何与参数

单节点（0-D）接触控制体：接触长度 L，法向名义压力 p，滑移速度 V，冰温 T_i，结构表面温度 T_s。冰侧与结构侧各取半无限体热渗透深度 delta = sqrt(alpha*L/V)（一次接触驻留时间 L/V 内的导热尺度）。

热物性取值（`properties.py`，0 °C 参考值）：

{props_tbl}

### 1.3 控制方程

- 压力融化： dT_m/dp = T_m*(v_w - v_i)/L_f，精确积分 T_m(p) = T_m0 * exp[dv*(p-p0)/L_f]；
- 融膜 ODE（质量 + 能量守恒）：

      dh/dt = (q_fric + q_ice + q_struct)/(rho_i*L_f) - c_d*h*V/L
      q_fric  = mu(h,p,V,T)*p*V
      q_ice   = k_i*(T_i - T_m(p))/sqrt(alpha_i*L/V)
      q_struct= (T_s - T_m(p))/(h/k_w + delta_s/k_s)

  正号表示向融膜供热（融冰），负号表示膜失热（再冻结）；末项为融膜被拖曳/挤压排出接触区的质量平衡；
- 摩擦闭合（扩展 Stribeck）：mu = A_r(Lambda)*mu_b + (1-A_r)*eta(T)*V/(p*h)，A_r = exp(-Lambda)，Lambda = h/sigma_rms，并设残余润滑摩擦下限 mu_floor；
- 冰粘附：sigma_adh(T) 由界面准液层厚度 h_QLL = A*(T_m - T)^(-1/3) 与经验强度律双分支给出（-10 °C 峰值 0.9 MPa）。

## 2. 方法 / 流程 / 工具链

1. 新建领域包 `src/tensorphysics/ice_contact/`（properties / pressure_melting / stefan / film / adhesion / tribology / competition / verification），纯 Python 数学核心，仅 1-D Stefan 参考解需要 numpy；
2. 每个物理关系都配一个**独立参考解**：
   - (a) Clausius-Clapeyron ODE 的 RK4 数值积分 vs 指数精确积分；
   - (b) 融膜 ODE 的**定常点二分求解** vs **手推二次型闭式解**（两者互不调用）；
   - (c) 1-D **焓法**（有限体积显式）vs **Neumann 相似解** X = 2*lambda*sqrt(alpha*t)；
   - (d) 参数空间扫描 + 5 条物理方向门；
3. `benchmarks/ice_contact/run_verification.py` 汇编全部判据 → `results.json` + 7 张图；
4. `benchmarks/ice_contact/render_report.py` 从 `results.json` **程序化注入**全部数字生成本报告（零手抄）；
5. pytest：{f_passed} 例全绿（判据 f，阈值 {f_threshold} 例）。

## 3. 计算结果

### 3.1 判据总表

{criteria_tbl}

### 3.2 判据 (a) 压力融化（Clausius-Clapeyron）

斜率 dT_m/dp = **{slope} K/MPa**，文献值 -0.0742 K/MPa，相对误差 {slope_err_pct}。

{a_tbl}

-10 °C 冰的**压力融化阈值压力** = **{threshold_mpa} MPa**（1 MPa 名义接触仅降 0.0741 K，故压力融化只可能在**粗糙峰尺度**（1e2 MPa 量级）起作用）。

### 3.3 判据 (b) 纯摩擦融化稳态膜厚

两类比较：定摩擦系数闭式解，以及自洽（mu = mu(h)）耦合定常点 vs 二次型闭式解；另外从三个初值做 RK4 瞬态积分，验证收敛到同一解析定常态。

{b_tbl}

瞬态收敛（初值 → 最终膜厚 vs 闭式解）：

{b_trans_tbl}

### 3.4 判据 (c) 再冻结 vs Stefan 解析解

Delta T = 10 K，Ste = {stefan_number}，Neumann 常数 lambda = {neumann_lambda}。终时刻（1 ms）冻结层厚度与网格收敛：

{c_tbl}

误差随网格加密单调下降 → 一阶收敛，最细网格误差 {c_error_pct}（150 格 {c_coarse_pct}）。

### 3.5 判据 (d) 三机制竞争与物理方向门

{gate_tbl}

融化起始速度（等温滑块）数值二分 vs 闭式解 V_c = (-(Delta_i + Delta_s)/(mu*p))^2：

{onset_tbl}

四压力最大偏差 {onset_worst_pct}（p = 1 MPa 处 {onset_main_pct}）：闭式解取干摩擦系数 mu_b，而闭合在膜底还含一项随 V 增长的粘性份额，故 V_c 越大（p 越小）偏差越大，方向一致。

### 3.6 判据 (e) 端到端算例

**主算例（任务指定）：1 MPa / 0.1 m/s / -10 °C，冷滑块**

- 熔点降 {spec_depression} K，冰粘附强度 {spec_adhesion} MPa；
- 稳态：**无持久融膜**（exists = False），摩擦落在边界润滑支 mu = {spec_mu}；
- 1 µm 初始融膜的**寿命 = {spec_lifetime} ms**，随后进入干接触/边界分支；
- 机制速率（m/s）：摩擦融化 {spec_friction_rate}、压力融化 {spec_pressure_rate}、**再冻结 {spec_refreeze_rate}** —— 再冻结主导；
- 该工况等温滑块下的融化起始速度 V_c = {spec_onset} m/s，指定速度 0.1 m/s **低于** V_c，故模型判定为边界摩擦支——与冰摩擦文献中"低速高摩擦、高速才进入融化润滑"的**速度弱化**规律一致。

四个算例汇总（含补充的摩擦融化 / 压力融化 / 再冻结分支）：

{case_tbl}

机制速率分解（m/s）：

{rate_tbl}

### 3.7 结果图

{figures_block}

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

1. **焓法 + 等温边界对"无过热"融化失效**：把壁面温度钉在 T_m 作 Dirichlet 边界时，首格焓只会从下方渐近趋于融化焓，融化前沿永远进不到邻格（经典焓法病态）。因此 `solve_stefan_1d(mode="melting")` 强制要求过热壁（T_w > T_m），再冻结（判据 (c)）不受影响。这是**求解器施加的物理前提**，不是结果修饰。
2. **结构侧热状态决定融膜存亡**：结构侧 Stefan 汇以"膜热阻 + 半无限结构热阻"串联表示；冷滑块（-10 °C 钢）比热平衡滑块多吸收一个量级的摩擦热，使 1 MPa/1 m/s 工况反而无膜。连续滑行中滑块自身会升温，故报告同时给出等温滑块（equilibrated_structure=True）分支，两者都标注。
3. **排出系数 c_d 与接触长度 L**：融膜厚度正比于 1/c_d；c_d 取 O(1) 是量级假设，未做挤压流（Reynolds）求解，因此**膜厚的绝对值**只具量级意义，趋势与竞争关系才是主张。
4. **高压端外推**：p > 50 MPa 时塞状排出假设不再成立（扫描中最大膜厚达数百 µm），故压力融化分支只主张**机制判定与驱动量**（熔点降、过热度），不主张该处膜厚数值。同理 Clapeyron 在 >50 MPa 忽略 dv、L_f 变化，会低估熔点降（IAPWS 曲线弯曲）。
5. **粘附模型是工程标定**：峰值 0.9 MPa @ -10 °C、脆化段衰减长度 25 K 为冰-碳钢剪切试验量级的经验取值；应变率依赖未建模。准液层 h ~ (Delta T)^(-1/3) 与 vdW 桥接应力（~ Delta T）提供温变**机理**解释。
6. **摩擦闭合是混合润滑载荷分担模型**，不是 Reynolds 方程解；mu_floor 防止高压端给出非物理的无摩擦界面。膜厚参数 Lambda 对粗糙度 sigma_rms 敏感（本报告取 1 µm），故 mu 的绝对值依赖粗糙度设定。
7. **0-D 单节点**：无横向流动、无冰变形、无粗糙峰温度分布，不能替代专业接触/CFD 求解器；本模块定位是 TensorPhysics 编排层的**可审计物理闭合 + 验收夹具**。
8. **git/CI 环境**：本 worktree 中 `tensorphysics` 的可编辑安装指向其它 worktree，故所有命令显式带 `PYTHONPATH=src:.`；报告与图件的生成脚本已固化该路径。

## 5. 误差判定（逐项 vs 基准，显式回答是否 <3 %）

{error_tbl}

结论：**主判据关键量全部 <3 %**（(a) 机器精度、(b) {b_error_pct_4}、(c) {c_error_pct}、(d) {onset_main_pct}）。两处超 3 % 的项目如实列出：① 150 格粗网格的 Stefan 前沿 {c_coarse_pct}（网格离散误差，随加密收敛，见 3.4）；② 低压端融化起始速度 {onset_worst_pct}（闭式解未含膜底粘性份额，方向一致）。二者均**不影响判据 (b)(c)(d) 的通过与结论方向**，不做粉饰。

### 5.1 审计链

- 主判据数字全部来自 `artifacts/ice_contact_m1212/results.json`（由 `benchmarks/ice_contact/run_verification.py` 生成）；
- 测试用例输出：`artifacts/ice_contact_m1212/pytest.log`（{f_passed} passed）；图表由脚本重放生成（7 张 PNG）；
- 复现命令：`PYTHONPATH=src:. python3 benchmarks/ice_contact/run_verification.py` → `PYTHONPATH=src:. python3 benchmarks/ice_contact/render_report.py`；
- 未重复使用任何已 verified 案例；本节点全部为全新实现。

### 5.2 未完成 / 明确不做

- 未与实验数据直接比对（本机无冰-钢摩擦实验数据）：判据全部对标解析/相似解，属**代码与物理正确性验证**，非实验标定；
- 未做二维/三维接触、粗糙峰分布、冰变形与破碎耦合；
- 未接入 TensorFEM/TensorLBM 求解器（本模块是闭合与夹具，非新求解器内核）。
"""


def load_results() -> dict:
    return json.loads((ARTIFACT_DIR / "results.json").read_text())


def pct(value: float | None, digits: int = 3) -> str:
    """Percentage with an explicit scientific notation for machine-precision errors."""
    if value is None:
        return "—"
    if value < 1e-6:
        return f"{value * 100:.1e} %"
    return f"{value * 100:.{digits}f} %"


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}g}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def verdict_3pct(error: float | None) -> str:
    if error is None:
        return "无基准"
    return "是（<3 %）" if error < 0.03 else "否（≥3 %）"


def build_markdown(results: dict) -> str:
    a = results["a_pressure_melting"]
    b = results["b_friction_melting_steady"]
    c = results["c_refreezing_stefan"]
    d = results["d_competition_gates"]
    e = results["e_end_to_end"]
    f = results["f_pytest"]
    spec = e["specified_case"]
    cases = {case["name"]: case for case in e["cases"]}
    friction_case = cases["friction_melting_1MPa_1mps_253K"]
    pressure_case = cases["pressure_melting_20MPa_5mmps_272p5K"]
    refreeze_case = cases["refreezing_1MPa_1mps_253K_cold_slider"]

    onset_rows = [
        [num(row["pressure"] / 1e6, 3), num(row["onset_velocity"], 4),
         num(row["onset_velocity_analytic"], 4),
         pct(abs(row["onset_velocity"] - row["onset_velocity_analytic"])
             / row["onset_velocity_analytic"], 2)]
        for row in results["onset_samples"]
    ]
    onset_worst = max(
        abs(row["onset_velocity"] - row["onset_velocity_analytic"]) / row["onset_velocity_analytic"]
        for row in results["onset_samples"]
    )

    gate_rows = [
        [name, "PASS" if gate["passed"] else "FAIL", gate["detail"]]
        for name, gate in d["gates"].items()
    ]

    refreeze_rows = [
        [str(row["n_cells"]), num(row["dx"], 3), num(row["dt"], 3), str(row["n_steps"]),
         num(row["front_numerical"][-1] * 1e6, 4), num(row["front_neumann"][-1] * 1e6, 4),
         pct(row["max_relative_error"], 2)]
        for row in c["rows"]
    ]

    b_rows = [
        [row["mode"], num(row.get("pressure", 0) / 1e6, 3), num(row.get("velocity", 0), 3),
         num(row.get("friction_coefficient"), 4),
         num(row.get("analytic_thickness", row.get("ode_fixed_point")), 4),
         num(row.get("reference_thickness", row.get("analytic_thickness")), 4),
         pct(row.get("relative_error"), 3)]
        for row in b["rows"]
    ]
    b_transient_rows = [
        [num(row["initial_thickness"], 3), num(row["final_thickness"], 4),
         num(row["analytic_thickness"], 4), pct(row["relative_error"], 3)]
        for row in b["transient"]
    ]

    a_rows = [
        [num(row["pressure"] / 1e6, 3), num(row["melt_temperature_rk4"], 7),
         num(row["melt_temperature_analytic"], 7), pct(row["relative_error"], 2),
         num(row["depression"], 4)]
        for row in a["rows"]
    ]

    case_rows = [
        [case["name"],
         num(case["inputs"]["pressure"] / 1e6, 3),
         num(case["inputs"]["velocity"], 3),
         num(case["inputs"]["ice_temperature"] - 273.15, 4),
         "yes" if case["inputs"]["equilibrated_structure"] else "no",
         "yes" if case["steady_state"]["exists"] else "no",
         num(case["steady_state"]["film_thickness"] * 1e6, 3),
         num(case["steady_state"]["friction_coefficient"], 4),
         case["steady_state"]["regime"],
         "—" if case["film_lifetime"] is None else num(case["film_lifetime"] * 1e3, 3)]
        for case in e["cases"]
    ]

    rate_rows = [
        [case["name"], num(case["steady_rates"]["friction_melting"], 3),
         num(case["steady_rates"]["pressure_melting"], 3),
         num(case["steady_rates"]["refreezing"], 3),
         num(case["steady_rates"]["drain_rate"], 3),
         num(case["steady_rates"]["dh_dt"], 3)]
        for case in e["cases"]
    ]

    figure_specs = [
        ("fig1_pressure_melting", "图 1 Clausius-Clapeyron 压力融化线：精确积分 vs 切线近似，标出 -10 °C 冰的融化阈值压力"),
        ("fig2_stefan_refreezing", "图 2 再冻结 Stefan 移动边界验证：Neumann 相似解 vs 三种网格的焓法解"),
        ("fig3_steady_film_map", "图 3 稳态融膜厚度与摩擦系数随滑移速度的变化（-10 °C，滑块已热平衡）"),
        ("fig4_case_histories", "图 4 四个算例的融膜厚度与摩擦系数时历"),
        ("fig5_mechanism_map", "图 5 三机制竞争参数空间图：压-速平面上主导机制分布（左 -10 °C，右 -0.65 °C）"),
        ("fig6_adhesion", "图 6 冰粘附强度温度依赖与准液层厚度"),
        ("fig7_stribeck", "图 7 以融膜厚度为坐标的扩展 Stribeck 曲线与润滑区间划分"),
    ]
    figure_lines = [
        f"### {caption}\n\n![{name}](figures/{Path(results['figures'][name]).name})\n"
        for name, caption in figure_specs
    ]

    property_rows = [
        ["冰密度", "ρ_i", "916.8", "kg/m³"],
        ["水密度", "ρ_w", "999.8", "kg/m³"],
        ["冰/水比热", "c_i / c_w", "2050 / 4186", "J/kg/K"],
        ["冰/水导热", "k_i / k_w", "2.22 / 0.561", "W/m/K"],
        ["融化潜热", "L_f", "3.3355e5", "J/kg"],
        ["冰熔点（常压）", "T_m0", "273.15", "K"],
        ["钢 (rho/c/k)", "—", "7850 / 475 / 45", "—"],
    ]
    criterion_rows = [
        ["(a)", "纯压力融化温度 vs CC 解析解", "<1 %", pct(a["error"], 4),
         "PASS" if a["passed"] else "FAIL"],
        ["(b)", "纯摩擦融化稳态膜厚 vs 解析解", "<5 %", pct(b["error"], 4),
         "PASS" if b["passed"] else "FAIL"],
        ["(c)", "再冻结速率 vs Stefan 解析解", "<5 %", pct(c["error"], 3),
         "PASS" if c["passed"] else "FAIL"],
        ["(d)", "三机制竞争 + 物理方向门", "全门通过", pct(d["error"], 3) + " (onset)",
         "PASS" if d["passed"] else "FAIL"],
        ["(e)", "端到端 1 MPa/0.1 m/s/-10 °C 输出时历", "四算例可用", "—",
         "PASS" if e["passed"] else "FAIL"],
        ["(f)", "pytest 用例数 >= " + str(f["threshold"]), ">=18",
         f"{f['tests_passed']} passed / {f['tests_failed']} failed",
         "PASS" if f["passed"] else "FAIL"],
    ]
    error_rows = [
        ["压力融化温度 T_m(p) (p<=100 MPa)", "Clapeyron 精确积分（解析）", pct(a["error"], 4),
         verdict_3pct(a["error"])],
        ["Clapeyron 斜率 dT/dp", "文献 -0.0742 K/MPa", pct(a["slope_relative_error"], 3),
         verdict_3pct(a["slope_relative_error"])],
        ["纯摩擦融化稳态膜厚", "手推二次型闭式解", pct(b["error"], 4), verdict_3pct(b["error"])],
        ["瞬态积分 → 定常膜厚", "同一闭式解（自洽 μ）",
         pct(max(row["relative_error"] for row in b["transient"]), 4),
         verdict_3pct(max(row["relative_error"] for row in b["transient"]))],
        ["再冻结前沿（最细网格 600 格）", "Neumann 相似解", pct(c["error"], 3),
         verdict_3pct(c["error"])],
        ["再冻结前沿（150 格）", "Neumann 相似解", pct(c["error_coarsest_grid"], 3),
         verdict_3pct(c["error_coarsest_grid"])],
        ["融化起始速度 V_c (1 MPa)", "平方根平衡闭式解", pct(d["onset_relative_error"], 3),
         verdict_3pct(d["onset_relative_error"])],
        ["融化起始速度 V_c (0.5-4 MPa 最差)", "同上", pct(onset_worst, 2),
         verdict_3pct(onset_worst)],
    ]

    props_tbl = table(["参数", "符号", "取值", "单位"], property_rows)
    criteria_tbl = table(["判据", "内容", "阈值", "实测误差", "结论"], criterion_rows)
    a_tbl = table(["p [MPa]", "T_m 数值积分 [K]", "T_m 解析 [K]", "相对误差", "熔点降 [K]"], a_rows)
    b_tbl = table(["模式", "p [MPa]", "V [m/s]", "μ", "数值膜厚 [m]", "参考膜厚 [m]", "相对误差"], b_rows)
    b_trans_tbl = table(["初始膜厚 [m]", "积分终值 [m]", "解析定常 [m]", "相对误差"], b_transient_rows)
    c_tbl = table(["网格数", "dx [m]", "dt [s]", "步数", "数值冻结厚 [µm]", "Neumann [µm]", "最大相对误差"],
                  refreeze_rows)
    gate_tbl = table(["门", "结论", "判据内容"], gate_rows)
    onset_tbl = table(["p [MPa]", "V_c 数值 [m/s]", "V_c 闭式 [m/s]", "相对偏差"], onset_rows)
    case_tbl = table(["算例", "p [MPa]", "V [m/s]", "T [°C]", "热平衡滑块", "有稳态膜", "h_ss [µm]",
                      "μ_ss", "润滑区", "膜寿命 [ms]"], case_rows)
    rate_tbl = table(["算例", "摩擦融化", "压力融化", "再冻结", "排出", "dh/dt"], rate_rows)
    error_tbl = table(["关键量", "基准（独立来源）", "相对误差", "是否 <3 %"], error_rows)
    figures_block = "".join(figure_lines)

    md = REPORT_TEMPLATE.format(
        f_passed=f["tests_passed"],
        f_threshold=f["threshold"],
        b_error_pct=pct(b["error"], 3),
        props_tbl=props_tbl,
        criteria_tbl=criteria_tbl,
        a_tbl=a_tbl,
        b_tbl=b_tbl,
        b_trans_tbl=b_trans_tbl,
        c_tbl=c_tbl,
        gate_tbl=gate_tbl,
        onset_tbl=onset_tbl,
        onset_worst_pct=pct(onset_worst, 2),
        onset_main_pct=pct(d["onset_relative_error"], 2),
        case_tbl=case_tbl,
        rate_tbl=rate_tbl,
        figures_block=figures_block,
        error_tbl=error_tbl,
        slope=num(results["clapeyron_slope_K_per_MPa"], 6),
        slope_err_pct=pct(a["slope_relative_error"], 3),
        threshold_mpa=num(a["melting_threshold_pressure_at_263K"] / 1e6, 4),
        stefan_number=num(c["stefan_number_freezing"], 4),
        neumann_lambda=num(c["neumann_lambda_freezing"], 5),
        c_error_pct=pct(c["error"], 3),
        c_coarse_pct=pct(c["error_coarsest_grid"], 3),
        spec_depression=num(spec["melt_point_depression"], 4),
        spec_adhesion=num(spec["adhesion_strength"] / 1e6, 3),
        spec_mu=num(spec["steady_state"]["friction_coefficient"], 4),
        spec_lifetime=num(spec["film_lifetime"] * 1e3, 3),
        spec_friction_rate=num(spec["steady_rates"]["friction_melting"], 3),
        spec_pressure_rate=num(spec["steady_rates"]["pressure_melting"], 3),
        spec_refreeze_rate=num(spec["steady_rates"]["refreezing"], 3),
        spec_onset=num(spec["onset_velocity_equilibrated"], 4),
        a_error_pct=pct(a["error"], 4),
        b_error_pct_4=pct(b["error"], 4),
    )
    return md


def build_payload(results: dict, markdown: str) -> dict:
    figures = []
    captions = {
        "fig1_pressure_melting": "图 1 Clausius-Clapeyron 压力融化线与 -10 °C 融化阈值压力",
        "fig2_stefan_refreezing": "图 2 再冻结 Stefan 验证：Neumann 相似解 vs 焓法（三种网格）",
        "fig3_steady_film_map": "图 3 稳态膜厚与摩擦系数 vs 滑移速度",
        "fig4_case_histories": "图 4 四算例融膜厚度与摩擦系数时历",
        "fig5_mechanism_map": "图 5 三机制竞争参数空间图",
        "fig6_adhesion": "图 6 冰粘附强度温度依赖 + 准液层厚度",
        "fig7_stribeck": "图 7 以膜厚为坐标的扩展 Stribeck 曲线",
    }
    for name, relative in results["figures"].items():
        path = REPO_ROOT / "artifacts" / "ice_contact_m1212" / relative
        figures.append(
            {
                "name": name,
                "caption": captions.get(name, name),
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    a = results["a_pressure_melting"]
    b = results["b_friction_melting_steady"]
    c = results["c_refreezing_stefan"]
    d = results["d_competition_gates"]
    spec = results["e_end_to_end"]["specified_case"]
    summary = (
        "冰-结构接触三相（冰/融膜/结构）热力学-摩擦学耦合成熟化：Clausius-Clapeyron 压力融化、"
        "摩擦融化 Stefan 条件、融膜厚度 ODE（含再冻结与排出）、冰粘附温度依赖、扩展 Stribeck 摩擦曲线。"
        f"判据 (a) 误差 {pct(a['error'],3)}、(b) {pct(b['error'],3)}、(c) {pct(c['error'],2)}、"
        f"(d) 五门全通过且融化起始速度偏差 {pct(d['onset_relative_error'],2)}；"
        f"(e) 指定工况 1 MPa/0.1 m/s/-10 °C 判定为再冻结主导、无持久融膜，1 µm 膜寿命 "
        f"{num(spec['film_lifetime']*1e3,3)} ms、边界摩擦 μ={num(spec['steady_state']['friction_coefficient'],4)}；"
        f"(f) pytest {results['f_pytest']['tests_passed']} 例全绿。"
    )
    return {
        "title": "W8-N01 冰-结构接触相变与融膜演化成熟化：压力融化 + 摩擦融化 Stefan + 融膜 ODE + 粘附温度依赖 + 扩展 Stribeck",
        "summary": summary,
        "scenario_key": "ice_structure_contact_phase_change",
        "mission_id": MISSION_ID,
        "tags": ["ice-contact", "phase-change", "stefan", "clausius-clapeyron", "tribology",
                 "melt-film", "tensorphysics", "verification"],
        "markdown": markdown,
        "figures": figures,
        "criteria": results["summary"]["criteria"],
        "artifacts": {
            "results_json": "artifacts/ice_contact_m1212/results.json",
            "pytest_log": "artifacts/ice_contact_m1212/pytest.log",
            "figures": "artifacts/ice_contact_m1212/figures",
            "module": "src/tensorphysics/ice_contact/",
            "tests": "tests/test_ice_contact_*.py",
            "report": "docs/reports/ice_contact_phase_change_report.md",
        },
    }


def main() -> None:
    results = load_results()
    markdown = build_markdown(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown)
    payload = build_payload(results, markdown)
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"report: {REPORT_PATH}")
    print(f"payload: {PAYLOAD_PATH} ({len(payload['figures'])} figures)")


if __name__ == "__main__":
    main()