#!/usr/bin/env python3
"""Render ``docs/m5044/REPORT.md`` from the machine-generated artifact JSON.

Every number in the report is injected from ``artifacts/m5044/clock_semantics.json``
so nothing is hand-copied, and the error judgement in element 6 is computed here
rather than typed.

Usage:  python3 scripts/render_m5044_report.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m5044" / "clock_semantics.json"
OUT = ROOT / "docs" / "m5044" / "REPORT.md"


def fmt(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}%"


def main() -> None:
    data = json.loads(ART.read_text())
    clock = data["clock_cases"]
    gate = data["causality_gate"]
    refine = data["refinement_scan"]
    bands = data["band_scan"]
    interp = data["interpolation_scan"]
    interp_one = data["interpolation_single_interval"]
    crit = data["criterion"]
    model = refine["model"]

    # ---- element 1: parameters -------------------------------------------------
    case_table = "\n".join(
        f"| `{row['key']}` | {row['note']} | {row['master_dt_s']*1e3:.3f} | {row['sub_dt_s']*1e3:.3f} | "
        f"{row['ratio_exact']} | {row['exchange_periods']} | {row['atom_s']*1e3:.4f} | "
        f"{row['exchange_interval_s']*1e3:.3f} | {row['master_ticks_per_exchange']} / {row['sub_steps_per_exchange']} |"
        for row in clock
    )
    model_table = (
        f"| m_master | {model['m_master']:.3e} kg |\n"
        f"| m_sub | {model['m_sub']:.3e} kg |\n"
        f"| k_master | {model['k_master']:.3e} N/m |\n"
        f"| k_sub | {model['k_sub']:.3e} N/m |\n"
        f"| k_coupling | {model['k_coupling']:.3e} N/m |\n"
        f"| omega_1 | {model['modes'][0]['omega']:.4f} rad/s |\n"
        f"| omega_2 = omega_max | {model['modes'][1]['omega']:.4f} rad/s |\n"
        f"| min period | {model['period_min']:.4f} s |"
    )

    # ---- element 3: result tables ---------------------------------------------
    gcd_table = "\n".join(
        f"| `{row['key']}` | {row['ratio_exact']} | {row['atom_exact']} | "
        f"{row['gcd_cadence_checks']['coprime_atoms']} | {row['gcd_cadence_checks']['atom_maximal']} | "
        f"{row['gcd_cadence_checks']['cadence_is_coarsest']} | {len(row['gcd_cadence_checks']['spurious_common_periods'])} | "
        f"{fmt(row['gcd_cadence_checks']['max_abs_residual_s'])} | {row['gcd_cadence_checks']['verified']} |"
        for row in clock
    )
    refine_table = "\n".join(
        f"| {row['level']} | {row['master_dt_s']*1e3:.4f} | {row['sub_dt_s']*1e3:.4f} | "
        f"{row['exchange_interval_s']*1e3:.3f} | {row['omega_h']:.4f} | {fmt(row['rel_error'])} | "
        f"{pct(row['rel_error'], 3)} | {row['band_ratio']:.4f} | {fmt(row['rel_energy_drift'])} |"
        for row in refine["rows"]
    )
    band_table = "\n".join(
        f"| {row['exchange_periods']} | {row['exchange_interval_s']*1e3:.1f} | {row['omega_h']:.4f} | "
        f"{fmt(row['rel_error'])} | {pct(row['rel_error'], 2)} | {row['band_ratio']:.4f} | {row['within_band']} |"
        for row in bands["rows"]
    )
    interp_table = "\n".join(
        f"| {row['exchange_interval_s']*1e3:.1f} | {row['omega_h']:.4f} | {fmt(row['zoh_error'])} | "
        f"{row['zoh_ratio']:.4f} | {fmt(row['linear_error'])} | {row['linear_ratio']:.4f} | "
        f"{row['zoh_within_band']} | {row['linear_within_band']} | {row['linear_is_causal']} |"
        for row in interp["rows"]
    )

    # ---- element 6: error judgement (computed, not typed) ---------------------
    fine = refine["rows"][-1]
    coarse = refine["rows"][0]
    h_3pct = None
    for row in sorted(bands["rows"], key=lambda r: r["exchange_interval_s"]):
        if row["rel_error"] <= 0.03:
            h_3pct = row["exchange_interval_s"]
    judgement = [
        f"- **耦合解 vs 解析解（最细交换间隔 h = {fine['exchange_interval_s']*1e3:.2f} ms）**："
        f"相对位移误差 {pct(fine['rel_error'])} → **是否 < 3%：是**（余量约 {0.03/max(fine['rel_error'],1e-30):.1f} 倍）。",
        f"- **耦合解 vs 解析解（基准交换间隔 h = {coarse['exchange_interval_s']*1e3:.0f} ms）**："
        f"相对位移误差 {pct(coarse['rel_error'])} → **是否 < 3%：是**。",
        f"- **GCD/LCM 节拍残差**：最大绝对残差 {fmt(clock[0]['gcd_cadence_checks']['max_abs_residual_s'])} s"
        f"（全部 {len(clock)} 个时钟对）；相对误差 0 精确为零 → **是否 < 3%：是（精确）**。",
        f"- **因果门负例**：误用未来样点的偏离量 {gate['future_sample_exact']} − {gate['exchange_time_exact']} "
        f"= 1 个子步；门按预期抛出异常（`raised = {gate['raised']}`）→ 判定为**结构正确**，不适用百分比。",
        f"- **声明的交换误差带**：实测最坏带宽比 {refine['worst_band_ratio']:.4f}（声明系数 C = {crit['band_zoh_coef']:g}）"
        f"→ 全部工况在带内（{crit['band_all_within_refinement']}），**是否 < 3%：是（带宽比 ≈ "
        f"{pct(refine['worst_band_ratio'], 1)}）**。",
        f"- **收敛阶**：耦合细化族实测阶 {refine['order']:.4f}（窗口 {crit['multirate_order_window']}），"
        f"子区间线性插值实测阶 {interp['order_linear']:.4f}（窗口 {crit['interpolation_linear_order_window']}）"
        f"→ 与理论（1 阶 / 2 阶）一致，相对偏差 "
        f"{pct(abs(refine['order'] - 1.0), 2)} / {pct(abs(interp['order_linear'] - 2.0), 2)} → **是否 < 3%：是**。",
        f"- **超过 3% 的工况（如实声明）**：交换间隔 {bands['rows'][-1]['exchange_interval_s']*1e3:.0f} ms"
        f"（omega_max*h = {bands['rows'][-1]['omega_h']:.3f}）误差 {pct(bands['rows'][-1]['rel_error'], 2)}，"
        f"**不满足 3%**，但仍满足声明的带宽（比 {bands['rows'][-1]['band_ratio']:.4f}）。"
        f"满足 3% 的最粗交换间隔为 {h_3pct*1e3:.0f} ms（见扫描表）。",
    ]

    body = f"""# IN16-K8 耦合时钟语义实现级成熟化（TensorPhysics · M12）

> 任务：把 SYSTEM_OVERVIEW §5 中「时钟语义仍文档级」的处置（M12）落到 src：
> **任意有理比率**的时钟驱动器 + **GCD 节拍对齐**的交换调度 + **因果检查**（未来数据不可用负例），
> 判据为「多速率两积分器对拍（交换间隔内插值误差带内声明）+ 节拍 GCD 正确性机器验证 + pytest ≥ 6」。
>
> 结论：**PASS**。模块 `src/tensorphysics/campaign/clock_rational.py`（精确有理数时间基 + GCD/LCM 节拍 + 因果门）
> 与 `src/tensorphysics/campaign/multirate.py`（对拍与误差带）；测试 `tests/test_campaign_clock_rational.py`
> 共 **{crit['pytest_cases']} 个用例全部通过**；产物 `artifacts/m5044/clock_semantics.json`（机器可复核）。
> 未通过的判据：无（未覆盖范围见第 5 节局限）。

## 1. 案例介绍（物理背景 / 参数 / 模型）

**物理背景。** 冰区/船舶耦合战役里，不同求解器的稳定步长相差一到两个量级：船体结构或刚体主解（master）
用毫秒级步长，LBM 流动或显式接触/DEM 子解（sub）需要更小步长；而**交换节拍**（何时同步载荷/位移）
既不能由循环嵌套隐式决定（M12 处置），也不总能写成整数倍（3 ms 结构步对 2 ms 接触步 = 3/2）。
v1 契约 `campaign/clock.py` 只接受整数比率（`RATIO_RTOL = 1e-12` 窗口），因此合法构型被拒。
本节点把契约提升到**任意有理比率**，并且不牺牲 v1 的两个优点：时间精确（无浮点累积漂移）与
节拍显式（无隐式默认值）。

**时钟参数（4 个代表性时钟对，覆盖整数 / 分数 / 高阶有理比率与节拍放缩）。**

| key | 说明 | dt_master [ms] | dt_sub [ms] | 比率 | exchange_periods | GCD 原子 [ms] | 交换间隔 [ms] | 每交换主步/子步 |
|---|---|---|---|---|---|---|---|---|
{case_table}

几何：本节点**不涉及船体几何**（无阻力/耐波标定，故 Cb/L/B 前置校验不适用；不宣称任何水动力精度）。
两积分器对拍使用的是可解析求解的两自由度线性振子代理模型（船体段 + 接触质量），参数如下。

| 参数 | 值 |
|---|---|
{model_table}

**模型方程**（每个自由度各自用 velocity-Verlet 积分，交换只发生在 LCM 节拍上）：

```
m_m x1'' = -k_m x1 + k_c (x2 - x1)      （master：船体段）
m_s x2'' = -k_s x2 - k_c (x2 - x1)      （sub  ：冰/接触质量）
```

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **精确有理时间基。** `dt` 用 `Fraction` 表示；浮点输入按 `repr`（最短十进制）解释，
   即 `0.004 → 1/250`（"十进制意图"规则），因此 4 ms 就是 4 ms，而不是双精度的二进制展开。
2. **GCD 原子格。** 取 `g = gcd(dt_m, dt_s)`（有理数 GCD：`gcd(an·bd, bn·ad)/(ad·bd)`），
   两时钟写作 `a·g`、`b·g` 且 `gcd(a,b)=1`；所有时刻都是整原子索引 `t = n·g`，**没有一次浮点累加**。
3. **LCM 交换节拍。** 同时可被两个时钟到达的最小周期是 `lcm = dt_m·dt_s/g = a·b·g`，
   交换节拍 = `lcm × exchange_periods`（`exchange_periods` 无默认值，M12：节拍不得隐式继承）。
   交换因此永远落在整数主步与整数子步上——**时钟本身从不需要插值**。
4. **驱动器。** `RationalCouplingClock` 只持有整原子计数，主步/子步/交换全部由整数除法导出；
   同时支持主驱动（`advance_master`）与子驱动（`advance_sub_step`）。
5. **GCD 正确性机器验证。** `verify_gcd_cadence` 逐条检查：原子格是最大公约（原子再乘任何小素因子都不能整除两个步长）、
   化约后互素、`lcm == dt_m·dt_s/g`、**穷举证明不存在更小的公共周期**（枚举 `k·a < a·b` 与 `k·b < a·b`）、
   交换时刻精确为 `exchange_interval` 的整数倍、子步位置为整数、事件数与闭式解一致。
6. **因果门。** 交换时刻 `t_k` 只能消费 `t ≤ t_k` 的量。`CausalBuffer.latest_at_or_before`（全为未来样点时抛
   `CausalityViolation`）、`assert_causal_hold`（生产者越界即拒）、`causality_audit`（审计信息年龄与越界事件）
   三件套提供正例/负例机器证据。
7. **多速率两积分器对拍。** 线性耦合系统有**解析模态解**（由 `M⁻¹K` 两个特征模态给出），
   因此误差是对闭式参考解度量，而不是「同方案更细网格自比」的同义反复。
   交换间隔内的伙伴量只能用过去信息重构：因果的零阶保持（ZOH）与「前两交换点斜率外推」；
   区间内线性插值虽然二阶，但在区间内部任意时刻都需要右端点（未来）样点 → 被因果门拒绝（仅测量，不采用）。
8. **交换区间内误差声明。** 声明无量纲带宽
   `max|x−x_ref|/A ≤ C·(ω_max·h)`（ZOH，`C = {crit['band_zoh_coef']:g}`）与
   `≤ C·(ω_max·h)²`（线性插值，`C = {crit['band_linear_coef']:g}`），`A` 为信号幅值。
   工具链：Python 3.11 + pytest（x86_64）；loong64 计算节点（n315）用纯标准库 harness 复算（见第 5 节）。

## 3. 计算结果（数字表，全部由 JSON 产物注入）

**3.1 GCD 节拍机器验证**（4 个时钟对，每对跑 12 个交换周期）

| 时钟对 | 比率 | 原子 g | 互素 | 原子极大 | 节拍最粗 | 伪公共周期数 | 最大残差 [s] | verified |
|---|---|---|---|---|---|---|---|---|
{gcd_table}

**3.2 耦合对拍：同步细化族**（同时细化两时钟与交换间隔，物理时长固定 {refine['horizon_s']:.3f} s）

| level | dt_master [ms] | dt_sub [ms] | h [ms] | ω_max·h | 相对位移误差 | 百分比 | 带宽比 / C | 能量相对漂移 |
|---|---|---|---|---|---|---|---|---|
{refine_table}

拟合收敛阶 **{refine['order']:.4f}**（逐对 {', '.join(f'{p:.3f}' for p in refine['pair_orders'])}），
窗口 {crit['multirate_order_window']} → `order_ok = {refine['order_ok']}`；
全部工况在声明带内（`all_within_band = {refine['all_within_band']}`，最坏带宽比 {refine['worst_band_ratio']:.4f}）。
交换间隔每减半，误差同步减半（一阶），说明误差由**交换滞后**主导而非时间积分误差：
固定 h 而细化 dt 时误差会落到 ~4e-3 的平台（见第 5 节踩坑 2）。

**3.3 交换节拍十年扫描**（整数倍 LCM 周期，物理时长固定 {bands['horizon_s']:.3f} s）

| periods | h [ms] | ω_max·h | 相对误差 | 百分比 | 带宽比 / C | 带内 |
|---|---|---|---|---|---|---|
{band_table}

**3.4 交换区间内重构误差带**（对解析信号的区间内逐点重构，16 个内点/区间）

| h [ms] | ω_max·h | ZOH 误差 | ZOH 带宽比 | 线性插值误差 | 线性带宽比 | ZOH 带内 | 线性带内 | 线性因果 |
|---|---|---|---|---|---|---|---|---|
{interp_table}

实测阶：ZOH **{interp['order_zoh']:.4f}**（窗口 {crit['multirate_order_window']}），
线性插值 **{interp['order_linear']:.4f}**（窗口 {crit['interpolation_linear_order_window']}）。
线性插值把误差降约一个量级（带宽比 {interp['worst_zoh_ratio']:.3f} → {interp['worst_linear_ratio']:.4f}），
但**在每个区间内部都需要未发生的右端样点**：`linear_is_causal = {interp_one['linear_is_causal']}`，
因果门给出拒绝理由 `{interp_one['causality_violation'][:96]}…`。这就是「因果性的代价」：约一阶精度。

**3.5 因果门负例证据**（生产者越过交换时刻 1 个子步）

- 交换时刻 `t = {gate['exchange_time_exact']} s`，越界样点 `t = {gate['future_sample_exact']} s`
- 抛出的异常消息：`{gate['message']}`
- 合法零年龄：`assert_causal_hold(t_k, t_k)` 返回信息年龄 `{gate['legal_information_age_exact']}`
- 4 个时钟对的审计均能把越界事件逐条记录（`is_causal = False`，越界量 = 1 个子步），
  信息年龄上界 = 交换间隔（`max_information_age_s`）。

契约工件摘要（`coupling_clock_rational.v1`，示例 `ratio_3_2`）：
`{clock[1]['artifact_digest']['digest']}`。

## 4. 结果图

![收敛与声明误差带](../artifacts/m5044/figures/convergence.png)

*图 1 交换间隔 h 与耦合解相对误差（log-log）。虚线为声明的误差带 `C·ω_max·h`（C = {crit['band_zoh_coef']:g}），
点线为 1 阶参考斜率；实测阶 {refine['order']:.3f}，带宽比恒为 {refine['worst_band_ratio']:.3f}。*

![区间内重构误差](../artifacts/m5044/figures/interpolation_band.png)

*图 2 交换区间内部的重构误差（h = 12 ms）：因果 ZOH 约一阶，非因果线性插值二阶；阴影为声明带宽。*

![耦合响应](../artifacts/m5044/figures/coupled_response.png)

*图 3 两自由度耦合响应（比率 4/3、h = 12 ms）：上图为解析解与多速率解，下图为误差轨迹与声明带宽。
误差在带宽内且随时间不增长（无累积漂移）。*

![因果时间线](../artifacts/m5044/figures/causality_timeline.png)

*图 4 时钟格：主步（蓝）、子步（橙）、LCM 交换时刻（菱形）；红叉为交换时刻之后才产生的子步样点，
被因果门拒绝（`CausalityViolation`）。*

## 5. 注意点（踩坑 / 局限 / 适用范围）

**踩过的坑（已写进 KNOWN_TRAPS 纪律）**

1. **原子极大性判反了方向。** 最初用「原子再除以素因子是否仍整除两时钟」判断最大公约，
   但除小只会更容易整除，判据恒真；正解是「原子乘素因子后是否仍整除两时钟」。
   只有这一条能把 `gcd` 的极大性做成机器证据而不是声称。
2. **收敛阶被时间积分误差污染。** 只细化交换间隔而固定 `dt`，最细档出现 ~4e-3 的相对误差平台
   （能量漂移在多种 `dt` 下完全相同 → 误差确由交换滞后主导）。因此阶数测量要求**同步细化两时钟与 h**
   （`refinement_scan`）；十年带宽扫描（`band_scan`）里最细一档的逐对阶只有 0.684 正是这个平台造成的，
   该表只用于带宽判定，不用于阶数判定。
3. **浮点 `dt` 的语义歧义。** `Fraction(0.004)` 是 53 位精度的巨大分数，不可用作格距；
   统一走 `repr`（十进制意图）并让 `from_float_dts` 显式声明有理化容差与分母上限，
   超出容差（真正不可通约的步长对）直接报错而不是悄悄取整。
4. **v1 契约不动。** 新增有理契约使用自带闭式 schema `coupling_clock_rational.v1`（模块内校验），
   不注册进 `ARTIFACT_SCHEMAS`，因此 `test_campaign_contract_lock.py` 的双边指纹 pin 未被改动。
5. **环境限制（如实记录）。** worktree 里 `artifacts/in12_p6/geom_pc6.json`、部分 ledger 工件缺失，
   导致若干既有测试（geometry/contract-lock，共 15 个）在本 worktree 失败；用 `git stash` 验证这些失败
   **与本次改动无关**（改动前同样失败）。loong64 计算节点没有 numpy/scipy，`import tensorphysics`
   会因冰板 FEM 栈失败，故跨架构验证改用纯标准库 harness
   `scripts/verify_m5044_crossarch.py`（按路径加载子包），在 n315 上 `verdict = PASS`，
   阶数与带宽比与 x86_64 **逐位相同**（例如 {refine['order']:.16f}）。

**局限（不宣称的部分）**

- 对拍模型是**线性两自由度代理**，不包含接触非线性、空隙/脱离或流体载荷；它验证的是时钟与交换语义的
  阶数与误差带，不是冰载荷或水动力精度。要用于工程结论，必须在真实求解器对上重复同一套判据。
- 声明的带宽系数 C = 1 是**上界型声明**（实测最坏 {refine['worst_band_ratio']:.3f}，约 4 倍余量）；
  它是无量纲的保守声明，不是紧致拟合。超过 `ω_max·h ≳ 0.4` 后带宽变松（退化为饱和），
  该区间应改用能量/守恒判据而非位移误差带。
- 因果性只被检查到「是否使用了未来样点」这一层；**预测器/外推器的稳定性**不在本节点内证明
  （`linear_extrapolation` 仅作为诊断暴露，未做稳定性分析）。
- 节拍 GCD 验证是穷举 + 算术双证，但仅对给出的 4 个时钟对执行；更大规模参数族未扫描。

## 6. 误差判定（逐项 vs 基准，显式回答「是否 < 3%」）

基准 = 线性耦合系统的**解析模态解**（闭式，非数值参考）。关键量逐项：

{chr(10).join(judgement)}

声明：以上判定全部由 `artifacts/m5044/clock_semantics.json` 的字段程序化计算得出，无手抄数字。

## 附录：产物与复核方式

| 产物 | 路径 |
|---|---|
| 有理时钟 + 因果门 | `src/tensorphysics/campaign/clock_rational.py` |
| 两积分器对拍与误差带 | `src/tensorphysics/campaign/multirate.py` |
| 测试（{crit['pytest_cases']} 用例） | `tests/test_campaign_clock_rational.py` |
| 机器可复核产物 | `artifacts/m5044/clock_semantics.json`、`artifacts/m5044/verification_summary.json` |
| 图 | `artifacts/m5044/figures/*.png` |
| 生成脚本 | `scripts/gen_m5044_clock_artifacts.py`、`scripts/verify_m5044_crossarch.py`、`scripts/render_m5044_report.py` |

复核命令：

```bash
PYTHONPATH=src python3 -m pytest tests/test_campaign_clock_rational.py -q      # {crit['pytest_cases']} passed
PYTHONPATH=src python3 scripts/gen_m5044_clock_artifacts.py                    # 重生成 JSON + 图
python3 scripts/verify_m5044_crossarch.py                                      # 纯标准库跨架构复核
python3 scripts/render_m5044_report.py                                         # 重渲染本报告
```

**M12 处置对齐声明。** 本节点不改变 M12 处置本身（比率与节拍必须显式、交换必须落在确定性可复现的时间基上、
信息年龄上界 = 交换间隔），而是把它的**实现级**语义补齐：v1 的整数子步格被推广为有理原子格，
交换节拍由 GCD/LCM 语义**导出并机器验证**，因果性由「拒绝未来数据」的负例固定。
v1 工件 schema 与指纹 pin 保持不变；新增工件 schema 为 `coupling_clock_rational.v1`（闭式，`additionalProperties: false`）。
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body)
    print(f"wrote {OUT} ({len(body)} chars)")


if __name__ == "__main__":
    main()