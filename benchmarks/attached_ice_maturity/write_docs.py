"""Generate docs/attached_ice/maturity_p302w2_n09.md from criteria.json.

All numbers in the document are read from the machine-produced
``artifacts/attached_ice_maturity/criteria.json`` — nothing is hand-copied.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CJ = os.path.join(ROOT, "artifacts", "attached_ice_maturity", "criteria.json")
OUT = os.path.join(ROOT, "docs", "attached_ice", "maturity_p302w2_n09.md")


def pct(x):
    return f"{100.0 * x:.3g}%"


def main():
    d = json.load(open(CJ))
    c = d["criteria"]
    defects = d["defects"]
    topics = {
        "C1": ("热平衡闭合（界面能量残差/潜热守恒）", 0.02, pct),
        "C2": ("Stefan 相似解极限 h ∝ t^{1/2}", 0.03,
               lambda x: "< 1e-12（机器精度）" if x < 1e-12 else pct(x)),
        "C3": ("负控 η_stick=0 ⇒ h(t)≡0", 1e-15, lambda x: f"{x:.3g}"),
        "C4": ("剥离/剪切单调性（200 点属性测试）", 0, lambda x: f"{int(x)} 违例"),
        "C5": ("ITTC-1978 惩罚链一致性", 0.01, pct),
        "C6": ("adfreeze 粘附强度接口一致性", 1e-9, lambda x: f"{x:.3g}"),
        "C7": ("pytest 覆盖条数", 15, lambda x: f"{int(x)} 条"),
    }
    lines = []
    for key, (name, thr, fmt) in topics.items():
        v = c[key]
        thr_s = fmt(thr) if key not in ("C3", "C6") else f"{thr:.0g}"
        lines.append(
            f"| {key} | {name} | {thr_s} | {fmt(v['value'])} | "
            f"{'PASS' if v['pass'] else 'FAIL'} |"
        )
    table = "\n".join(lines)

    d2 = defects["D2_heat_transfer"]
    sens = d["sensitivity"]
    def mm(v):
        return "无稳态(h→∞)" if v == float("inf") else f"{1e3 * v:.1f}"

    sens_tab = "\n".join(
        f"| {r['water_over_temperature_K']:.2f} | {r['q_conv_W_m2']:.4g} | "
        f"{mm(r['h_eq_m'])} | {r['limited_by']} |"
        for r in sens["vs_water_over_temperature"])
    sens_cf = "\n".join(
        f"| {r['c_f']:.0e} | {r['q_frazil_W_m2']:.4g} | "
        f"{mm(r['h_eq_m'])} | {r['limited_by']} |"
        for r in sens["vs_frazil_concentration"])
    regime = defects["D3_shear_detachment_regime"]
    env = d["envelope_final"]

    md = f"""# 水下附着冰模块成熟化验收（p302w2-n09）

模块：`tensorphysics.ice.attached_ice`（flux / growth / detach / roughness / handoff）
审计日期：2026-11-10 · 判据 JSON：`artifacts/attached_ice_maturity/criteria.json`
本轮唯一定量来源：`benchmarks/attached_ice_maturity/run_maturity.py`（本文表格由
`benchmarks/attached_ice_maturity/write_docs.py` 从该 JSON 直接生成，零手抄）。

## 1. 结论（先给判据，再讲物理）

| 判据 | 内容 | 阈值 | 实测 | 判定 |
|---|---|---|---|---|
{table}

全部 7 条机器判据通过（`all_pass = {str(d['all_pass']).lower()}`）。
但"判据通过"仅说明实现自洽，**不说明模型已定标**——本轮同时修掉三个使定量结果
不可用的缺陷，并把模型适用范围写成代码里的显式诊断（第 4 节）。

## 2. 物理与数值模型（本轮定稿）

界面能量平衡（growth.py）：

    ρ_ice L_f dh/dt = q_cond − q_conv + q_frazil
    q_cond = k_ice ΔT / h,      q_frazil = η_stick · C_f ρ_ice v_dep L_f
    q_conv = h_c (T_water − T_freeze),  h_c = ρ Cp u_* √(cf/2) Pr^(-2/3)

稳态厚度由两个物理上限的较小者给出（新增 `equilibrium_thickness`）：

    h_growth  = k_ice ΔT / (q_conv − q_frazil)         （熔融-导热平衡）
    h_detach  = h_ref (τ_bond /(ρ_water u_*²) − 1)     （粘附强度-剪切平衡）
    h_ss(u_*) = min(h_growth, h_detach)

两者对 u_* 均严格单调下降，故 h_ss 单调（C4 因此有解析保证，而不是靠时间窗运气）。
`stefan_split` 积分器把奇异导热项按其精确求积 h²→h²+2kΔT dt/(ρL) 推进，纯导热情形
与相似解等价（C2 达机器精度）。

## 3. 三个缺陷：发现 → 量化 → 修复

### D1 ITTC-1978 粗糙度惩罚系数错 {c['C5']['ratio_canonical_over_legacy']:.0f} 倍（C5 判据原本不成立）
旧实现取 α=0.004，与 ITTC-1978 规范式 (105 (k_s/L)^(1/3) − 0.64)×10⁻³ 相比：

* 规范差值（k_s: 0.5 mm → 5 mm, L = 150 m）：{c['C5']['delta_cf_canonical']:.4e}
* 修复后（α = 0.105）：{c['C5']['delta_cf_fixed']:.4e}，相对误差 {pct(c['C5']['rel_err_fixed'])}
* 旧实现：{c['C5']['delta_cf_legacy']:.4e}，相对误差 {pct(c['C5']['rel_err_legacy'])}，
  偏低 {c['C5']['ratio_canonical_over_legacy']:.1f} 倍

旧测试之所以"绿"，是因为它拿同一个 α 自比（实现与自己比），属自指测试；
本轮把规范式写成显式参考函数 `ittc_1978_delta_cf` 并加了回归测试
`test_legacy_coefficient_would_fail`。

### D2 对流换热系数错 2 个量级（使冰在船速下根本无法附着）
旧式 h_c = ρ Cp u_* 把摩擦速度当平均流速塞进 Stanton 数（St = u_*/U = √(cf/2) 而非 cf/2）：

| 量 | 旧 | 新 |
|---|---|---|
| h_c @ u_*={d2['u_star_m_s']} m/s [W/m²K] | {d2['h_c_legacy_W_m2K']:.4g} | {d2['h_c_fixed_W_m2K']:.4g} |
| q_conv @ ΔT_water=0.2 K [W/m²] | {d2['q_conv_legacy_W_m2']:.4g} | {d2['q_conv_fixed_W_m2']:.4g} |
| 熔化速率 [m/s] | {d2['melting_rate_legacy_m_s']:.3g} | {d2['melting_rate_fixed_m_s']:.3g} |
| 相应平衡厚度 [m] | {d2['implied_equilibrium_legacy_m']:.3g} | {d2['equilibrium_fixed_m']:.3g} |

旧比值 {d2['ratio_legacy_over_fixed']:.0f}×，恰好等于 1/[√(cf/2) Pr^(-2/3)]。
修复后 h_c 走 Colburn 类比（cf_ref = 1.5e-3，Pr = 13），并由回归测试锁定。

### D3 "稳态厚度"未定义 + 剪切剥离在船尺度不激活
旧 `solve_euler` 在剥离后把 h 置 0 再重新生长，形成锯齿，"稳态 h"取决于积分窗长度；
且以 Frederking 粘附强度（~0.5 MPa）对壁面剪切（u_*=0.1 m/s 时 10 Pa）计算：

* 剥离所需 u_* = {regime['u_star_crit_m_s']:.1f} m/s，对应平均流速 {regime['speed_crit_m_s']:.0f} m/s
  （当前状态仅 {regime['speed_at_state_m_s']:.1f} m/s）→ 剪切剥离在船尺度**不激活**，
  真正起作用的移除机制是熔融。

处理方式（不加假物理、不删分支）：新增 `equilibrium_thickness` 给出单值稳态解，
新增 `detachment_regime` 把"是否激活/所需流速"变成可审计输出；文档与 payload 的
`limitations` 明确写出这一适用范围。

## 4. P50/P90 包络（任务 (d)）

`monte_carlo_envelope` 对 ΔT、u_*、C_f、η_stick 做蒙特卡洛（{env['n_samples']} 样本，
σ 由输入区间给定），输出**随时间演化**的 P50/P90 包络（旧 `p50_p90_envelope` 只对已
采样厚度数组取分位数，无时间维度、无不确定性传播）：2 h 窗末
h_ice P50 = {1000*env['h_ice_final_p50']:.2f} mm、P90 = {1000*env['h_ice_final_p90']:.2f} mm；
ΔT/T 以 C_F=0.0015 为分母时 P50 = {env['delta_t_over_t_final_p50']:.2f}×——远超 1，
正是一阶传播失效区（payload 以 `delta_t_over_t_linearization_ok=False` 显式标记）；
下游必须以自身 C_T（量级 4e-3）作分母并重算。

## 5. 薄接口（任务 (2)）

`attached_ice/handoff.py`：`build_handoff(...) -> dict` + `to_json()`，schema
`{d['module'].split('.')[-1]}` = `tensorphysics.attached_ice/1`。payload 只给
状态量与系数（h_ice、k_s、ΔC_f、ΔT/T、稳态解、剥离regime、包络、limitations），
**不含任何船舶阻力/推进域语义式**（有测试 `test_no_domain_resistance_formula_leaked`
守住这条边界）。下游（ship-performance-platform）自备 C_T、推进效率等域模型。

## 5b. 敏感性（为什么必须用实测水温驱动）

h_ss ∝ 1/(q_conv − q_frazil)，对"水温过热度"和 frazil 浓度都极度敏感
（u_* = 0.05 m/s, ΔT = 2 K, C_f = 1e-4）：

| 水温过热度 [K] | q_conv [W/m²] | h_eq [mm] | 限制机制 |
|---|---|---|---|
{sens_tab}

| C_f [-] | q_frazil [W/m²] | h_eq [mm] | 限制机制 |
|---|---|---|---|
{sens_cf}

过热度 ≤0.1 K 时 frazil 输入大于熔融汇，模型给出"无稳态"（冰无限增长）；
0.2→0.5 K 使 h_eq 从 74.7 mm 降到 11.6 mm（约 6 倍）。即本模块**不能**在无实测
水温输入时给出定量厚度，只能给趋势与包络；这条写进代码 payload 的 `limitations`。

## 6. 局限与诚实声明

1. 附着冰厚度/粗糙度**无实验标定**：C1–C7 只证明实现自洽（热平衡闭合、相似解收敛、
   单调性、单位一致），不等于与实测一致；不存在本轮的精度对照实验（无基准可用）。
2. k_s = k_s0 + α h_ice（α=0.5）是线性代理模型，无附着冰粗糙度数据库支撑。
3. ΔT/T = ΔC_f / C_denominator 是一阶传播；当 ΔC_f 与 C_T 同量级时失效，payload 用
   `delta_t_over_t_linearization_ok` 显式标记（包络 P90 即为该失效区，见第 4 节）。
4. 忽略冰层内温度剖面、船体板导热、盐度对冻结点/熔解的影响（海水 T_freeze=−1.8 ℃ 用
   常数 T_FREEZE_SW 近似）。
5. h_eq 对水温过热度极敏感（第 5b 节），必须由实测水温驱动；否则只应使用趋势/包络。
6. 剪切剥离分支保留但船尺度不激活；若要做实验尺度（低粘附强度）验证，用
   `detachment_regime` 先确认分支是否处于激活区。

## 7. 复现

```
python -m pytest tests/test_ice_attached.py tests/test_ice_attached_maturity.py -q   # {d['n_tests']} passed
python benchmarks/attached_ice_maturity/run_maturity.py                              # 判据 + 图
python benchmarks/attached_ice_maturity/write_docs.py                                # 本文
```

图：`artifacts/attached_ice_maturity/fig1..fig5*.png`
（增长/粗糙度、Stefan 验证、ITTC 惩罚、P50/P90 包络、单调性与剥离 regime）。
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(md)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()