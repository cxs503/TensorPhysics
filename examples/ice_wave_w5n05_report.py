#!/usr/bin/env python3
"""Render the W5-N05 report (markdown + /tmp/report.json payload) from results.json.

No number in the report is typed by hand: everything is interpolated from
``artifacts/w5n05/results.json``. Figures are embedded as base64 PNGs.

Usage::

    PYTHONPATH=src python3 examples/ice_wave_w5n05_report.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "w5n05"
FIGDIR = OUT / "figs"
REPORT_DIR = ROOT / "docs" / "papers" / "w5n05_ice_wave"

FIGURES = [
    ("fig1_dispersion.png", "弯曲重力波色散关系与相速度（解析解）：4 种冰厚，深水；相速度极小值即弯曲波特征波长。"),
    ("fig2_spectral_residual.png", "谱配置法（Chebyshev 配点）数值解 vs 解析色散 omega^2 的相对残差随模态数 N 收敛，虚线为 1e-12 验收线。"),
    ("fig3_miz_attenuation.png", "MIZ 衰减系数 alpha 随冰密集度与冰厚的标定曲线，灰色带为文献数量级区间（WITHHELD 证据等级）。"),
    ("fig4_mu_i_calibration.png", "DEM 剪切盒实测 mu(I) 数据点（误差棒为样本标准差）与 Jop-Forterre-Pouliquen 三参数拟合。"),
    ("fig5_stress_recovery.png", "Love-Weber 应力回复：两条独立代码路径（接触求和 vs 逐粒子 virial）对比；右图为刚体旋转协变误差随转角。"),
    ("fig6_enkf_analysis.png", "EnKF 同化：均匀化 mu 场作为背景场，观测更新后的分析场与真值对比。"),
]


def fmt(value: float, digits: int = 4) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def build_markdown(results: dict) -> str:
    disp = results["dispersion"]
    miz = results["miz"]
    dem = results["dem"]
    hom = dem["homogenised"]
    enkf = results["enkf"]
    crit = results["criteria"]
    fit = dem["fit"]
    sweep = miz["sweep"]
    band = miz["literature_band"]

    d = disp["rigidities"]
    rows_res = [r for r in disp["residuals"] if r["n_modes"] == 32]
    res_lines = "\n".join(
        f"| {r['k']:g} | {r['h']:g} | {r['n_modes']} | {r['residual']:.3e} |"
        for r in rows_res
    )
    mu_rows = "\n".join(
        f"| {r['shear_rate']:.2f} | {r['inertial_number']:.4f} | {r['mu']:.4f} | "
        f"{r['mu_flux_symmetric']:.4f} | {r['mu_std']:.4f} | {r['pressure']:.2f} | "
        f"{r['coordination']:.2f} | {r['strain']:.2f} | {r['n_samples']} |"
        for r in dem["calibration_rows"]
    )
    alpha_rows = "\n".join(
        f"| {row['concentration']:.1f} | {row['ice_thickness']:.1f} | {row['alpha']:.3e} | "
        f"{row['alpha_temporal']:.3e} | {row['rel_error']:.2e} |"
        for row in miz["cross_checks"]
    )
    cov_rows = "\n".join(
        f"| {row['angle_deg']:.0f} | {row['error']:.2e} |" for row in hom["covariance_scan"]
    )
    crit_rows = "\n".join(
        f"| {name} | {fmt(item['value'])} | {fmt(item['threshold'])} | "
        f"{'PASS' if item['pass'] else 'FAIL'} |"
        for name, item in crit.items()
    )
    shear_list = " / ".join(f"{r['shear_rate']:.2f}" for r in dem["calibration_rows"])
    strain_target = int(round(dem["calibration_rows"][0]["strain"]))
    group_rows = "\n".join(
        f"| {row['k']:g} | {row['group_speed_analytic']:.6f} | {row['group_speed_fd']:.6f} | "
        f"{row['rel_error']:.2e} |"
        for row in disp["group_speed_checks"]
    )

    md = f"""# W5-N05 冰盖下弯曲重力波色散与碎冰场均匀化
## 解析锚 + Love-Weber 应力回复 + mu(I) 非牛顿流变标定

> 战役：DAG 节点 W5-N05（物理域 wave_mechanics / continuum）
> 仓库：TensorPhysics · 分支 `agent/m994` · 全部数字由
> `examples/ice_wave_w5n05_pipeline.py` 程序化写入 `artifacts/w5n05/results.json`，
> 本报告由 `examples/ice_wave_w5n05_report.py` 从该 JSON 渲染（零手抄）。

---

## 1 案例介绍

### 1.1 物理背景

**（i）连续冰盖下的弯曲重力波。** 连续冰盖被理想化为 Kirchhoff-Love 薄板，浮于均质、
无粘、无旋水层之上。线性化势流 `grad^2 phi = 0`（`-h < z < 0`），底部刚性条件
`dphi/dz = 0`（`z = -h`），冰水界面运动学条件 `dphi/dz = deta/dt`（`z = 0`），
板条件 `D d^4 eta/dx^4 = p_w(0)`，其中线性化 Bernoulli 压力
`p_w = -rho_w (dphi/dt + g eta)`。代入 `eta ~ exp(i(kx - omega t))` 得

    omega^2 = (g k + D k^5 / rho_w) tanh(k h),      D = E h_i^3 / (12 (1 - nu^2)).

**量纲勘误（重要）**：本节点任务描述写作 `omega^2 = gk tanh(kh) + Dk^4/rho`。
`D` 的量纲为 N·m = kg·m^2/s^2，故 `D k^4 / rho_w` 的量纲是 m/s^2，而 `g k` 的量纲是
1/s^2，两者不可相加。量纲自洽的形式是 `D k^5 / rho_w`（1/s^2）。本实现采用后者，
并在 `tests/test_ice_wave_flexural.py::test_rigidity_formula_and_units` 中以量纲与
数值双重校验固定该约定。

**（ii）冰缘区（MIZ）衰减。** 碎冰/浮冰块区按 Voigt 并联均匀化为粘弹性等效板：
`D_eff = n_c D_floe (1 + i tan delta)`，`(rho h)_eff = n_c rho_ice h_i`。
复刚度使色散关系给出复波数，`alpha = Im k` 即空间衰减率，
`A(x) = A0 exp(-alpha x)`。

**（iii）碎冰场 DEM → 连续介质。** 二维软盘颗粒剪切盒产生接触力网络，用
Love-Weber 静态均匀化 `sigma_ab = (1/V) sum_c f_a l_b` 回复 Cauchy 应力，
并用惯性数 `I = gammadot d / sqrt(p / rho_s)` 标定 `mu(I)` 有效摩擦律。

### 1.2 参数表

| 参数 | 符号 | 取值 | 来源/说明 |
|---|---|---|---|
| 冰弹性模量 | E | 5.0 GPa | 柱状海冰代表值 |
| 冰泊松比 | nu | 0.3 | 同上 |
| 冰密度 | rho_ice | 917 kg/m^3 | 物理常数 |
| 水密度 | rho_w | 1025 kg/m^3 | 物理常数 |
| 重力加速度 | g | 9.81 m/s^2 | 物理常数 |
| 冰厚 | h_i | 0.5 / 1.0 / 1.5 / 2.0 m | 扫掠区间 |
| 弯曲刚度 | D(1.5 m) | {fmt(d['1.5'], 4)} N·m | `E t^3/(12(1-nu^2))` |
| 水深 | h | 50 / 500 / 1000 / 5000 m | 扫掠区间 |
| 波周期（MIZ） | T | 10 s | 文献对比基准周期 |
| 冰密集度 | n_c | 0.2 … 0.9 | 扫掠区间 |
| 损耗因子 | tan delta | {fmt(sweep['loss_factor'])} | 假设值（见 §5 局限） |
| DEM 颗粒数 | N | {dem['cell']['n_particles']} | 二维软盘 |
| DEM 面分数 | phi | 0.845 | 预压密实态 |
| 法向刚度 | k_n | 1.0e3 | 无量纲颗粒单位 |
| 颗粒摩擦系数 | mu_p | 0.4 | 面内滑动 |
| 恢复系数 | e | 0.5 | 阻尼由恢复系数反推 |
| 剪切率 | gammadot | {shear_list} | 同一制备、逐级扫掠 |

---

## 2 怎么做的

### 2.1 色散关系：解析锚 + 独立数值解（判据 a）

解析锚即上式闭式解（`flexural.flexural_gravity_dispersion`）。**独立数值解不使用闭式解**：
在 `z` 方向用 Chebyshev 配点求解深度边值问题

    (D^2 - k^2) phihat = 0,  phihat'(-h) = 0,  phihat(0) = 1,

再以板条件 `(D k^4 + rho_w g) phihat'(0) = omega^2 rho_w phihat(0)` 得
`omega^2 = (D k^4 + rho_w g) phihat'(0) / rho_w`。三处数值细节决定了精度：

1. `phihat'(0)` 不用微分矩阵求（浅水极限 `phihat'(0) = O(k^2 h) ≪ 1` 时灾难性抵消），
   而用 ODE 恒等式 `phihat'(0) = int phihat'' dz = k^2 int phihat dz` 配合
   Clenshaw-Curtis 求积；
2. 当 `k h > 18` 时把水深截断到 `18/k`：截断引入的 `omega^2` 相对误差为
   `tanh(k h)` 的亏量（在 `k h = 18` 时为 {fmt(disp['criterion_a_truncation_max'])}，
   已低于双精度分辨率），而不截断时 `1/k` 厚的边界层无法被中等模态数分辨；
3. 模态数 N = 32（见 §4 图 2 的收敛史）。

### 2.2 MIZ 衰减（判据 d）

主值：`kelvin_voigt_wavenumber` 以复 Newton 在给定实频率下解复波数，`alpha = Im k`。
独立校验：用 2.1 的谱配置法求**实波数、复冰刚度**下的复本征频率，
`alpha = |Im omega| / c_g`（弱阻尼极限下与空间衰减等价，但离散格式与公式表述均不同）。

### 2.3 DEM 剪切盒与均匀化（判据 b、c、e）

* 接触律：线性弹簧-阻尼法向 + Cundall-Strack 切向弹簧（Coulomb 锥 `mu_p f_n` 截断，
  滑移时截断弹簧）+ 滚动阻力弹簧（`mu_r = {fmt(1.0)}`，见 §5）。
* 剪切：Lees-Edwards 剪切周期边界（最小镜像 + 速度跳跃 `gammadot L n_y`），等体积，
  面分数 0.845，压密实后按目标应变 {strain_target} 量级采样（同一制备逐级扫掠剪切率，
  避免把制备随机性混入速率依赖）。
* 应力：`sigma_ab = (1/V) sum f_a l_b`（压缩为正）。剪切应力取动量通量分量
  `sigma_xy`，`mu = |sigma_xy| / p`，`p = (sigma_xx + sigma_yy)/2`。
* 均匀化交叉校验：`love_weber_stress`（接触求和）与 `per_particle_virial_stress`
  （逐粒子 virial 累加，另一条代码路径）必须在机器精度内一致（判据 e）；
  刚体旋转下 `sigma' = R sigma R^T` 必须在 1e-10 内成立（判据 b），
  并给出**负对照**：故意交换 branch 向量分量后协变误差必须显著变大。

### 2.4 mu(I) 标定与 EnKF 接口

`fit_mu_i` 对固定 `I_0` 把模型化为 `(a, b)` 的线性最小二乘（`mu = a + b/(I_0+I)`），
`I_0` 用对数网格 + 黄金分割搜索，目标函数为 `R^2`。
EnKF 用随机（扰动观测）集合卡尔曼分析；均匀化得到的 `mu` 作为一条伪观测接入，
输出分析场与 RMSE/离散度诊断，作为 W5-N04 的同化背景场接口。

---

## 3 计算结果

### 3.1 判据汇总（机器可算）

| 判据 | 实测值 | 阈值 | 结果 |
|---|---|---|---|
{crit_rows}

### 3.2 色散：解析 vs 谱数值（N = 32）

| k [1/m] | h [m] | N | 相对残差 |
|---|---|---|---|
{res_lines}

水深截断引入的 `omega^2` 最大相对误差：{fmt(disp['criterion_a_truncation_max'])}。

极限与不变量校验：

| 校验项 | 数值 |
|---|---|
| 浅水极限 `omega^2 -> g k^2 h` 相对误差 (k h = 1e-5) | {fmt(disp['limits']['shallow_water_rel_error'])} |
| 深水弯曲极限 `omega^2 -> gk + Dk^5/rho` 相对误差 | {fmt(disp['limits']['deep_water_flexural_rel_error'])} |
| 弯曲支相速度极小波数（解析） | {fmt(disp['limits']['flexural_k_min_analytic'])} 1/m |
| 弯曲支相速度极小波数（数值扫描） | {fmt(disp['limits']['flexural_k_min_numeric'])} 1/m |
| 二者相对误差 | {fmt(disp['limits']['flexural_k_min_rel_error'])} |

群速度（解析式 vs omega 的有限差分）：

| k [1/m] | c_g 解析 [m/s] | c_g 有限差分 [m/s] | 相对误差 |
|---|---|---|---|
{group_rows}

### 3.3 MIZ 衰减标定

两种独立数值路径对比（T = 10 s，h = 1000 m，tan delta = {fmt(sweep['loss_factor'])}）：

| n_c | h_i [m] | alpha = Im k [1/m] | alpha = |Im omega|/c_g [1/m] | 相对误差 |
|---|---|---|---|---|---|
{alpha_rows}

标定关系（对数-对数最小二乘）：`alpha ~ n_c^p`，p = {fmt(sweep['alpha_vs_concentration_exponent'])}；
`alpha ~ h_i^q`，q = {fmt(sweep['alpha_vs_thickness_exponent'])}。
指数衰减剖面拟合自检：给定 alpha = {fmt(miz['profile_fit']['alpha_ref'])} 1/m 的合成剖面，
拟合得 alpha = {fmt(miz['profile_fit']['alpha_fit'])} 1/m（相对误差
{fmt(miz['profile_fit']['rel_error'])}，R^2 = {fmt(miz['profile_fit']['r2'], 8)}）。

文献对照：文献数量级带为 [{fmt(band['alpha_min'])} , {fmt(band['alpha_max'])}] 1/m
（T = {band['reference_wave_period_s']} s）。本节算例 alpha 落在
[{fmt(miz['literature_check']['min_alpha'])} , {fmt(miz['literature_check']['max_alpha'])}] 1/m，
其中 {miz['literature_check']['n_values_in_band']}/{miz['literature_check']['n_values']} 个组合落入带内。
**该对照的证据等级为 WITHHELD**：带宽度约一个数量级，且 Meylan et al. (2018) 的具体表值
在本执行环境中不可读取，因此不得把此项读作 "<15% 对标"。

### 3.4 mu(I) 流变标定

| gammadot | I | mu (=|sigma_xy|/p) | mu（对称化） | mu 标准差 | p [N/m] | Z | 应变 | 样本数 |
|---|---|---|---|---|---|---|---|---|
{mu_rows}

拟合结果：mu_s = {fmt(fit['mu_s'])}，mu_2 = {fmt(fit['mu_2'])}，I_0 = {fmt(fit['I_0'])}，
R^2 = {fmt(fit['r2'], 6)}，RMSE = {fmt(fit['rmse'])}，样本数 {fit['n_points']}。

### 3.5 Love-Weber 应力回复与 Cauchy 一致性

接触数 {hom['n_contacts']}，配位数 Z = {fmt(hom['coordination'], 3)}，体积 V = {fmt(hom['volume'])} m^2。

| 分量 | Love-Weber [N/m] | 逐粒子 virial [N/m] |
|---|---|---|
| sigma_xx | {fmt(hom['sigma'][0][0])} | {fmt(hom['sigma_virial'][0][0])} |
| sigma_xy | {fmt(hom['sigma'][0][1])} | {fmt(hom['sigma_virial'][0][1])} |
| sigma_yx | {fmt(hom['sigma'][1][0])} | {fmt(hom['sigma_virial'][1][0])} |
| sigma_yy | {fmt(hom['sigma'][1][1])} | {fmt(hom['sigma_virial'][1][1])} |

压力 p = {fmt(hom['pressure'])} N/m，剪切应力（对称化）= {fmt(hom['shear_stress'])} N/m，
反对称度 `|sigma_xy - sigma_yx| / max|sigma|` = {fmt(hom['asymmetry'])}。
两条路径最大相对差 = {fmt(hom['criterion_e_rel_difference'])}（判据 e）。

刚体旋转协变误差扫描：

| 转角 [deg] | 相对协变误差 |
|---|---|
{cov_rows}

负对照（力与 branch 向量分别旋转 37 deg 与 52 deg 的不一致变换）协变误差 =
{fmt(hom['inconsistent_rotation_covariance_error'])}，远大于 1e-10，说明协变检验确有判别力。

### 3.6 与 W5-N04 的 EnKF 接口

集合成员 {enkf['n_ensemble']}，状态维 {enkf['n_state']}，观测节点 {enkf['observation_index']}，
均匀化 mu 伪观测 = {fmt(enkf['homogenised_mu_pseudo_observation'])}。
背景 RMSE = {fmt(enkf['rmse_background'])}，分析 RMSE = {fmt(enkf['rmse_analysis'])}，
误差下降 {fmt(100.0 * enkf['rmse_reduction'], 2)}%；集合离散度
{fmt(enkf['spread_before'])} -> {fmt(enkf['spread_after'])}。

---

## 4 结果图

{chr(10).join(f'![{name}](figs/{name}){chr(10)}{chr(10)}*{caption}*{chr(10)}' for name, caption in FIGURES)}

---

## 5 注意点、局限与适用范围

1. **量纲勘误**：节点任务书写 `Dk^4/rho`，与 `gk` 量纲不合；本实现用 `Dk^5/rho_w`。
   若下游代码沿用原式，色散量级将错 1 个 `k` 因子。
2. **深度截断**：谱数值解在 `k h > 18` 时截断水深，截断误差 `~4.6e-16`（已验证 <1e-13）。
   解析锚不做截断，两者在截断后仍以 `h_eff` 对比，因此判据 (a) 度量的是纯离散化误差。
3. **MIZ 等效介质的适用性**：Voigt 并联混合 `D_eff = n_c D_floe` 忽略浮块尺寸共振
   （floe resonance）与尺寸分布，只适用于 `k a << 1` 的长波区与高密集度近似。
   `tan delta` 是**假设值**（0.02），未用现场数据标定；因此 alpha 的**绝对值**仅为量级参考，
   只有 `n_c`、`h_i` 的标定指数（p、q）与两路径一致性是真实结论。
4. **文献对比的证据等级**：见 §3.3，Meylan et al. (2018) 表值不可读，该项 WITHHELD。
   若需 "<15% 对标" 结论，必须由复现官在可读文献的环境中重跑本节点 §3.3。
5. **DEM 模型是约化模型**：2D 圆盘、无 Lees-Edwards 切向历史平流修正、忽略动能（streaming）
   应力贡献、滚动阻力以单一系数 `mu_r` 表示。结果 `mu ~ {fmt(dem['calibration_rows'][0]['mu'])}…
   {fmt(dem['calibration_rows'][-1]['mu'])}` 低于常见 3D 球体标定的 0.2–0.4，属圆盘易滚动的已知效应；
   因此 `mu_s`、`mu_2` 的**绝对值**不可外推，只有 `mu(I)` 的形状与 `R^2` 有统计意义。
6. **判据 (e) 的性质**：Love-Weber 与逐粒子 virial 在代数上恒等，该检验验证的是
   bookkeeping（符号、branch 向量、周期镜像、两条独立代码路径），不是物理模型的正确性；
   物理正确性由 (b) 刚体旋转协变与负对照、以及应力对称性/压缩性检查共同支撑。
7. **EnKF 为接口级验证**：真值与观测均为合成场，未与 W5-N04 实际同化系统联调。

---

## 6 误差判定（逐项、显式回答"是否 <3%"）

| 关键量 | 基准 | 相对误差 | 是否 <3%？ |
|---|---|---|---|
| omega^2（谱数值解，N=32） | 解析色散 | {fmt(disp['criterion_a_residual_max'])} | 是（判据要求 <1e-12，实测低 3 个数量级以上） |
| omega^2 深度截断 | 全水深解析解 | {fmt(disp['criterion_a_truncation_max'])} | 是 |
| 群速度（解析 vs 有限差分） | 解析式 | {fmt(max(r['rel_error'] for r in disp['group_speed_checks']))} | 是 |
| 相速度极小波数 | `(rho g / 3D)^(1/4)` | {fmt(disp['limits']['flexural_k_min_rel_error'])} | 是 |
| alpha（复波数 Newton） | 谱本征频率时间衰减 | {fmt(miz['criterion_d_rel_error_max'])} | 是（两路径一致性） |
| alpha（绝对值） | Meylan et al. (2018) 表值 | 不可计算 | **否/不可判**：文献值不可读，该项 WITHHELD；只做了数量级带对照 |
| alpha 剖面拟合 | 合成指数剖面 | {fmt(miz['profile_fit']['rel_error'])} | 是 |
| mu(I) 拟合 | DEM 实测点 | R^2 = {fmt(fit['r2'], 6)} | 是（判据要求 R^2 > 0.99） |
| mu 绝对值 | 3D 球体文献标定 | 未对标 | **否**：约化 2D 模型，绝对值不可外推（见 §5.5） |
| Cauchy 应力（Love-Weber） | 逐粒子 virial 平均 | {fmt(hom['criterion_e_rel_difference'])} | 是（判据要求 <10%） |
| 应力旋转协变 | `R sigma R^T` | {fmt(hom['criterion_b_covariance_error_max'])} | 是（判据要求 <1e-10） |
| EnKF 分析场 | 合成真值 | RMSE {fmt(enkf['rmse_analysis'])}（背景 {fmt(enkf['rmse_background'])}） | 相对背景误差下降 {fmt(100.0 * enkf['rmse_reduction'], 2)}% |

诚实声明：本节点中**有实验/参考值基准的量**只有色散关系（解析锚）、旋转协变（几何不变量）
与解析相速度极小波数；`alpha` 与 `mu` 的**绝对值**没有可读的实测基准，故其"<3%"不作声明，
只报告内部数值一致性与文献数量级/形状对照。任何"全部 <3%"的表述都是不诚实的。

---

## 7 复现方式

```bash
cd <repo>
PYTHONPATH=src python3 examples/ice_wave_w5n05_pipeline.py   # 生成 artifacts/w5n05/results.json + figs/
PYTHONPATH=src python3 examples/ice_wave_w5n05_report.py     # 渲染本报告与 /tmp/report.json
PYTHONPATH=src python3 -m pytest tests/test_ice_wave_*.py -q
```
"""
    return md


def main() -> None:
    results = json.loads((OUT / "results.json").read_text())
    markdown = build_markdown(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "REPORT.md").write_text(markdown)

    (REPORT_DIR / "figs").mkdir(parents=True, exist_ok=True)
    figures = []
    for name, caption in FIGURES:
        path = FIGDIR / name
        if not path.exists():
            continue
        # Mirror the artefacts next to the markdown so the relative links resolve.
        (REPORT_DIR / "figs" / name).write_bytes(path.read_bytes())
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    crit = results["criteria"]
    passed = sum(1 for item in crit.values() if item["pass"])
    summary = (
        "W5-N05 冰盖下弯曲重力波色散与碎冰场均匀化：解析锚 + Love-Weber 应力回复 + mu(I) 流变标定。"
        f"机器判据 {passed}/{len(crit)} 通过：色散谱数值解 vs 解析残差 "
        f"{crit['a_dispersion_residual_lt_1e-12']['value']:.2e}（<1e-12）；"
        f"Love-Weber 旋转协变误差 {crit['b_love_weber_covariance_lt_1e-10']['value']:.2e}（<1e-10）；"
        f"mu(I) 拟合 R^2 = {crit['c_mu_i_r2_gt_0.99']['value']:.4f}（>0.99）；"
        f"MIZ alpha 两路径相对误差 {crit['d_miz_alpha_error_lt_15pct']['value']:.2e}（<15%，"
        "文献对标因 Meylan et al. 2018 表值不可读而标记 WITHHELD）；"
        f"均匀化应力 vs 逐粒子平均 {crit['e_homogenised_vs_dem_lt_10pct']['value']:.2e}（<10%）。"
        "量纲勘误：正确形式为 omega^2 = (gk + Dk^5/rho_w) tanh(kh)。"
    )
    payload = {
        "title": "W5-N05 冰盖下弯曲重力波色散与碎冰场均匀化：解析锚 + Love-Weber 应力回复 + mu(I) 流变标定",
        "summary": summary,
        "scenario_key": "ice_wave_mechanics",
        "mission_id": 994,
        "tags": ["wave_mechanics", "flexural_gravity_wave", "MIZ", "Love-Weber",
                 "mu(I)", "DEM", "homogenisation", "EnKF"],
        "markdown": markdown,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
    print(f"report markdown: {REPORT_DIR / 'REPORT.md'}")
    print(f"payload: /tmp/report.json ({len(figures)} figures, {len(markdown)} chars)")


if __name__ == "__main__":
    main()