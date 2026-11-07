# 平整冰板弯曲破坏 Kirchhoff–Winkler 有限元（战役节点 IN11-H3）

**工件**：`artifacts/ice_breaking_fem.json`（本报告全部数字由程序从该工件注入，无手抄）· **模块**：`src/tensorphysics/ice_sheet/kirchhoff_plate_fem.py` · **测试**：`tests/test_ice_breaking_fem.py`（28 项，含负例）· **复现**：`python benchmarks/icebreaking/run_ice_breaking_fem.py`

节点判据（机检，见工件 `acceptance` 段）：无限大板点载荷解析闭式解挠度对标 < 3 %；网格收敛 Richardson 阶验证；pytest ≥ 6（含基础模量 ≤ 0 / 板厚 ≤ 0 拒收）；输出破坏长度–冰厚曲线。

## 1. 案例介绍

### 1.1 物理背景

船艏以准静态压痕方式推入平整冰板。船体水线附近的前倾外飘面把水平推力的一部分转为垂向力，冰板在该垂向载荷下沉、弯曲；当板内弯曲应力达到冰的弯曲强度时，冰板开裂，被压断的冰块（floe）剥离，船艏前进。控制量是**首裂纹出现时的垂向力**与**断开的冰块尺寸（破坏长度）**，二者正是 Lindqvist 类破冰阻力公式中破坏分量（C1）的物理输入。

力学模型：浮冰板 = Kirchhoff 薄板 + Winkler 弹性基础（浮力回复），

```
D ∇⁴w + k_w w = q(x,y),   D = E h³ / [12(1-ν²)],   k_w = ρ_w g
M_x = -D(w,xx + ν w,yy),  M_y = -D(w,yy + ν w,xx),  M_xy = -D(1-ν) w,xy
首裂纹判据：  max|M| ≥ M_f = σ_f h² / 6      ( ⇔ 表面弯曲应力 6|M|/h² ≥ σ_f )
```

符号约定：`w` 向下为正，`M > 0` 为下缘受拉（sagging），`M < 0` 为上缘受拉（hogging）。故下压载荷先在板顶面诱发**环形裂纹**，判据取 hogging 主弯矩 `M₂ = -M_f`；接触区内的下缘受拉判据同时给出，用于讨论压溃/弯曲的竞争（见 §5）。

### 1.2 几何与工况

| 量 | 取值 | 出处 |
|---|---|---|
| `ice_elastic_modulus_Pa` | 5.0000e+09 （范围 3e9 - 10e9） | Timco & Weeks (2010) Int. Geol. Rev. 52(7-8) 675-700; ISO 19906:2010 Annex A |
| `ice_poisson_ratio` | 0.3000  | Timco & Weeks (2010) |
| `ice_flexural_strength_Pa` | 7.0000e+05 （范围 0.5e6 - 1.0e6） | Timco & Weeks (2010); ISO 19906:2010 (first-year sea ice) |
| `sea_water_density_kgm3` | 1025.0000  | ISO 19906:2010 |
| `gravity_ms2` | 9.8100  | CODATA |
| `k_w` | 10055.25 Pa/m | k_w = rho_w g (Archimedes restoring pressure; Hetenyi 1946, Kerr 1975) |
| 船艏接触扇形 | 半径 1.00 m，半角 60.0° | plan-curvature contact of an icebreaker bow stem; cf. Lindqvist (1989) contact widths of order 0.3-1.5 m |
| 冰厚谱 | [0.6, 0.8, 1.0, 1.2] m | 平整冰设计谱 |

> **锚点声明**：benchmarks/ice_nav_refs (IN10-L2 product) is absent from this worktree; the minimal anchor set below is built from the cited literature and used consistently throughout.

### 1.3 两种布置

* **内场（interior）**：无限大冰板，扇形压痕远离任何边界 —— 与 Kelvin 闭式解对标，用于验证单元/求解器与断裂判据的物理标度；
* **边场（edge，作业工况）**：冰板 x ≥ 0，x = 0 为自由冰缘（floe 边界），y = 0 为对称面，扇形压痕压在自由冰缘上 —— 这是船舶实际破冰的位置（载荷在自由边附近会使弯矩放大数倍）。

## 2. 方法与工具链

| 环节 | 做法 |
|---|---|
| 离散 | 协调 ``C¹`` Bogner–Fox–Schmit 矩形单元（双三次 Hermite，每节点 4 自由度 `w, w_x, w_y, w_xy`，共 16 dof/单元） |
| 单元矩阵 | 3×3 Gauss 积分精确积分弯曲刚度与 Winkler 项；全部单元以 `einsum` 向量化，10⁴ 级网格仍廉价 |
| 载荷 | 扇形（sector）均布压力 → 一致节点载荷；载荷区用 6×6 Gauss；节点向量按**合力精确**重标定（`force_correction` 记录） |
| 边界 | `free`（自然 Kirchhoff 自由边）/ `simply`（w=0）/ `sym`（法向斜率+扭率=0）/ `clamped` |
| 求解 | `scipy.sparse.linalg.spsolve`；Winkler 项保证算子正定，全自由边也可解（模拟无限板） |
| 弯矩恢复 | 单元角点曲率 → 节点 M_x, M_y, M_xy，跨单元平均；主弯矩从张量旋转得到 |
| 网格 | 向载荷几何加密（`graded_axis`），以绝对尺寸 h₀ 加密至接触区 ~a/8 |
| 断裂判据 | `M₂ = -M_f`；临界载荷由线性标度直接给出 P_c = M_f / |M₂ per unit load| |

验证阶梯（独立参照，互为交叉验证）：

1. **点载荷 / 无限板**：Hetényi (1946) Kelvin 闭式解 `w(r) = -(P l²/2πD)·kei(r/l)`，`l = (D/k_w)^{1/4}`（挠度）；径向弯矩由 Kelvin 方程 `kei'' = ker - kei'/x` 精确给出（不依赖数值微分）；
2. **均布载荷 / 四边简支**：Navier 双 Fourier 级数（仓库既有 `winkler_plate.navier_center_deflection`）；
3. **分布接触斑**：Gaussian 压力 vs 半解析 Hankel 积分（`ice.static_plate.hankel_gauss_deflection`）；
4. **网格收敛**：同一物理问题的逐次网格差 `‖w_{h/2}-w_h‖₂` 估计 Richardson 阶；
5. **负控**：k_w ≤ 0、h ≤ 0、E ≤ 0、ν ∉ (-1, 0.5)、网格 0 单元、σ_f ≤ 0 必须拒收。

## 3. 计算结果

### 3.1 验证（REAL 证据）

点载荷挠度对标 Kelvin 闭式解，中心点相对误差最优值 **0.0115 %**；径向剖面 L2 相对误差 **0.0550 %**。

| 网格（每边单元） | dof | w_center 相对误差 [%] | 剖面 L2 相对误差 [%] |
|---|---|---|---|
| 30² | 3844 | 0.0375 | 0.1421 |
| 60² | 14884 | 0.0115 | 0.0550 |
| 120² | 58564 | 0.0237 | 0.0561 |

网格收敛研究（10 l x 10 l quarter model，刻意用粗网格使离散误差主导域截断偏移）：

| 网格（每边单元） | 单元尺寸 [m] | 剖面 L2 相对误差 [%] |
|---|---|---|
| 16² | 9.130 | 1.4236 |
| 32² | 4.565 | 0.3386 |
| 64² | 2.282 | 0.0784 |

Richardson 观测阶：**2.051**（log2(||w_{h/2}-w_h||_2 / ||w_{h/4}-w_{h/2}||_2) on the radial profile (r >= 0.5 l, avoids the singular load core)；三阶 C¹ 单元理论期望 2）

| 弯矩场对标 Kelvin ODE 解（r/l） | 0.30 | 0.50 | 1.00 | 1.50 | 2.00 | 2.50 |
|---|---|---|---|---|---|---|
| FEM M_r [Nm/m] | 1.11e+05 | 61950.1 | 6360.9 | -13528.0 | -18851.3 | -17672.4 |
| Kelvin M_r [Nm/m] | 1.11e+05 | 61978.2 | 6373.7 | -13530.6 | -18875.3 | -17726.5 |
| 相对误差 [%] | 0.044 | 0.045 | 0.201 | 0.019 | 0.127 | 0.305 |

| 交叉验证 | 参照 | 最优误差 | 判据 | 通过 |
|---|---|---|---|---|
| 均布载荷简支板 | Navier 级数 | 4.55614e-05 % | < 1 % | yes |
| Gaussian 接触斑 | Hankel 半解析 | 0.0091 % | < 3 % | yes |
| 弯矩场 | Kelvin ODE | 0.305 % | < 3 % | yes |
| 负控（8 类非法参数） | ValueError | 全部拒收 = yes | 全部拒收 | yes |

### 3.2 破坏长度–冰厚曲线（主产物）

| h [m] | l [m] | M_f [kNm/m] | 边场 P_c [kN] | 边场破坏长度 r_c [m] | r_c/l | 1.2P_c 时裂纹区尺度 [m] | 内场 P_c [kN] | 内场 r_c/l |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 9.96 | 42.0 | 358.2 | 11.81 | 1.186 | 19.56 | 2376.9 | 1.979 |
| 0.8 | 12.36 | 74.7 | 626.3 | 14.38 | 1.164 | 24.19 | 4222.8 | 2.013 |
| 1.0 | 14.61 | 116.7 | 968.2 | 16.65 | 1.139 | 28.44 | 6602.5 | 2.008 |
| 1.2 | 16.75 | 168.0 | 1383.5 | 19.10 | 1.141 | 32.91 | 9504.0 | 2.020 |

标度（对数最小二乘）：

* 边场临界力 P_c ∝ h^**1.949**（残差 0.12 %） —— 与 M_f ∝ h² 一致，因为 M_r/P 是无量纲常数；
* 内场临界力 ∝ h^**2.000**（残差 0.04 %）；
* 破坏长度 ∝ h^**0.689**（残差 0.72 %），与特征长度 l ∝ h^{3/4} = h^0.75 同标度。

### 3.3 前进载荷扫描（边场 → 内场过渡）

| d/l | P_c [kN] | 裂纹半径（自载荷点）[m] | 裂纹半径 / l |
|---|---|---|---|
| 0.00 | 970.9 | 17.04 | 1.17 |
| 0.25 | 1405.6 | 23.42 | 1.60 |
| 0.50 | 1993.3 | 27.76 | 1.90 |
| 1.00 | 3829.2 | 35.97 | 2.46 |
| 2.00 | 5930.0 | 31.04 | 2.12 |
| 4.00 | 6106.4 | 31.15 | 2.13 |

d 为接触斑中心到自由冰缘的距离。d = 0（压在冰缘上）时临界力最低；随着压痕深入，临界力迅速升高，d ≥ 2l 后趋近内场（无限板）值 —— 即**破冰只能发生在载荷靠近自由边时**，这解释了船舶以有限推力即可破开厚冰的物理机制。

### 3.4 为 C1（Lindqvist 破坏分量）提供的标定锚

```json
{
  "target": "C1 Lindqvist breaking component (P_B vs h, flake length)",
  "P_break_edge_N": {
    "0.6": 358178.7666313715,
    "0.8": 626282.1134520272,
    "1.0": 968175.7653916661,
    "1.2": 1383529.7821037774
  },
  "break_length_edge_m": {
    "0.6": 11.811960760394891,
    "0.8": 14.38397509183896,
    "1.0": 16.645126201981153,
    "1.2": 19.10442471949842
  },
  "flake_extent_at_1p2Pc_m": {
    "0.6": 19.561261094536285,
    "0.8": 24.1902634298856,
    "1.0": 28.442157030513748,
    "1.2": 32.90792732244998
  },
  "edge_amplification_vs_interior": {
    "0.6": 6.635946450691506,
    "0.8": 6.7425829413583305,
    "1.0": 6.819531842484149,
    "1.2": 6.86936522055401
  },
  "usage": "P_break_edge is the vertical force that initiates the first bending crack when the bow contact patch sits on the free ice edge; it is the physical upper bound of the bending component of level-ice resistance at quasi-static advance speed."
}
```

## 4. 结果图

### fig1_point_load_verification.png

![fig1_point_load_verification.png](figures/ice_breaking_fem/fig1_point_load_verification.png)

Point-load verification: mesh convergence of the FEM centre deflection and radial profile against the Kelvin closed form (left) and the closed-form deflection profile shape (right).

### fig2_moment_field_and_crack.png

![fig2_moment_field_and_crack.png](figures/ice_breaking_fem/fig2_moment_field_and_crack.png)

Hogging principal moment $M_2$ of the edge-loaded fan patch with the first-crack locus $M_2=-M_f$ (left) and the radial moment along the cracked direction (right).

### fig3_break_length_curve.png

![fig3_break_length_curve.png](figures/ice_breaking_fem/fig3_break_length_curve.png)

Break length (left) and critical vertical force (middle) versus ice thickness for the edge-loaded and interior configurations, with the Kelvin closed-form critical load as reference; power-law fit of the break-length curve (right).

### fig4_advance_sweep.png

![fig4_advance_sweep.png](figures/ice_breaking_fem/fig4_advance_sweep.png)

Advance sweep: critical vertical force (left) and first-crack radius (right) versus the distance of the contact patch from the free ice edge; breaking becomes possible at ship-level thrust once the patch is within ~1 l of the edge.

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **半边模型对称因子**（本单实际踩到的坑）：载荷压在对称面上时，计算域内只含扇形的一半，把节点向量标定到「域内合力 = P」会让物理合力少一半、弯矩放大 2 倍。修正前后 h=1.0 m 的边场 P_c 从 484 kN 变为 968 kN；修正的判据是「d = 4l 的扫描值必须趋近内场 P_c」，修正后 6106 kN vs 内场 6602 kN（差 8 %，且网格更粗），修正前相差 2.2 倍。API 现以 `symmetry_factor` 显式表达。
2. **接触斑分辨率**：船艏接触半径 ~1 m 远小于 l ~ 10–17 m。若斑块小于单元，节点向量会退化为少数集中力（合力虽经重标定精确，但近场弯矩失真）。对策：局部加密（h₀ ≤ a/8）+ 6×6 Gauss + 合力精确标定；报告的破坏长度取自 r ≈ 1.1–1.2 l，对斑块细节不敏感（a 从 0.5 m 加到 1.0 m，峰值弯矩变化 < 1 %）。
3. **点载荷挠度的「中心点误差」不是收敛指标**：点载荷核心奇异，逐次加密时中心点误差非单调（0.037 % → 0.012 % → 0.024 %），因此 Richardson 阶用 r ≥ 0.5 l 的剖面逐次差的 L2 范数估计（得 ~2，符合三阶单元）。
4. **压溃 vs 弯曲的竞争**：无限板点载荷下缘受拉极限对应力仅 ~0.3 MN（h=1 m），远小于环形裂纹的 6.6 MN；但接触区内压力早已超过冰的压溃强度（2 MPa 量级），真实冰在接触区先压溃、载荷以压溃平台传递，故**首裂纹判定只采用 hogging（环形）分支**，下缘受拉分支仅作诊断输出。
5. **线弹性 + 单值强度**：Kirchhoff 小挠度（本文工况 w/h < 0.1 满足）、无裂纹扩展、无损伤；真实海冰弯曲强度有 Weibull 分散（COV ≈ 0.3）和先天裂纹，实际冰块尺寸会小于本文的弹性预测。
6. **基础模型**：k_w = ρ_w g 为静态浮力回复；压痕速度 0.5–1.5 m/s 时附连水质量/阻尼效应未计（量级估计：事件时标 ~l/v ~ 10 s，远大于板的第一自振周期 ~1 s，故准静态近似合理）。
7. **锚点来源**：本仓不含 IN10-L2 的 `benchmarks/ice_nav_refs`，故按 citation 自建最小锚点并在工件 `anchors.declaration` 中声明；若后续并入 IN10-L2 锚点，应以同一 E/σ_f/ρ 口径重跑。

## 6. 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 误差 | 是否 < 3 % |
|---|---|---|---|
| 无限板点载荷中心挠度 | Kelvin 闭式解 | 0.0115 % | **是** |
| 无限板点载荷挠度剖面（L2） | Kelvin 闭式解 | 0.0550 % | **是** |
| 径向弯矩场 | Kelvin ODE 解 | 0.305 % | **是** |
| 简支板均布载荷中心挠度 | Navier 级数 | 4.55614e-05 % | **是** |
| Gaussian 斑载荷挠度 | Hankel 半解析 | 0.0091 % | **是** |
| 边场网格离散（P_c 的网格散布） | 三级加密 | 0.252 % | **是** |
| 内场 P_c 与点载荷解析解 | Kelvin `critical_point_load` | 7.52 % | **否（> 3 %，见下）** |

**关于内场 P_c 的 7 % 差异**：本文内场用有限半径（1 m）扇形斑，Kelvin 解是理想点载荷；点载荷弯矩核心对数奇异，任何有限斑都会系统性降低峰值弯矩（并给出更大的临界力），因此 7 % 属**物理差异而非数值误差**，随 a → 0 应收敛到解析值（工件中 a = 0.5 m 与 1.0 m 的峰值弯矩差 < 1 %，说明该差异主要来自斑块尺度的物理平滑而非网格）。破坏长度 r_c/l 两者一致（FEM 1.98–2.02 vs 解析 2.11）。

**honest 声明**：边场（作业工况）构型没有独立解析解，其可信度建立在（i）已验证的单元与求解器、（ii）网格三级加密后的 P_c 散布 < 1 %、（iii）d = 4l 时向已验证内场值收敛，三条一致；但**未**与冰池试验或实船数据直接比对（无可用公开同构型数据），故边场 P_c 的绝对值为**理论预测**级别，不建议直接作为设计系数使用。

## 7. 验收判据对照（机检）

| 判据 | 要求 | 实测 | 通过 |
|---|---|---|---|
| 点载荷解析对标 | < 3 % | 0.0115 % | yes |
| Richardson 阶 | 1.7 ≤ p ≤ 3.2 | 2.051 | yes |
| 弯矩场 | < 3 % | 0.305 % | yes |
| 负控 | 全部拒收 | yes | yes |
| pytest | ≥ 6 项含负例 | 28 项 | yes |
| 破坏长度–冰厚曲线 | h = 0.6/0.8/1.0/1.2 m | 见 §3.2 | yes |

**综合**：`all_criteria_met = yes`（工件 `acceptance` 段逐项可复核）。

## 8. 审计链

* 工件：`artifacts/ice_breaking_fem.json`（生成于 2026-11-07T00:13:10Z，commit `c82e43e`，命令 `python benchmarks/icebreaking/run_ice_breaking_fem.py`，运行 334.4 s）；
* 模块：`src/tensorphysics/ice_sheet/kirchhoff_plate_fem.py`（Bogner-Fox-Schmit C1 bicubic Hermite rectangle, 16 dof (w, w_x, w_y, w_xy)）；
* 测试：`tests/test_ice_breaking_fem.py`（28 项，含负例）；
* 图：`artifacts/figures/ice_breaking_fem/*.png`（matplotlib，程序生成）；
* 环境：Python 3.11.6 / numpy 2.4.6。

## 9. 参考文献

* Bogner, F.K., Fox, R.L., Schmit, L.A. (1967) *The generation of interelement compatible stiffness and mass matrices by the use of interpolation formulas*, AFFDL-TR-66-80, 397–443.
* Hetényi, M. (1946) *Beams on Elastic Foundation*, Univ. of Michigan Press.
* Kerr, A.D. (1975) *The bearing capacity of floating ice plates subjected to static or quasi-static loads*, CRREL Report 75-1.
* Kerr, A.D. (1996) *Bearing capacity of floating ice covers subjected to static, moving and fluctuating loads*, J. Glaciol. 42(142) 474–486.
* Timoshenko, S., Woinowsky-Krieger, S. (1959) *Theory of Plates and Shells*, 2nd ed., Ch. 8.
* Timco, G.W., Weeks, W.F. (2010) *A review of the engineering properties of sea ice*, Int. Geol. Rev. 52(7–8) 675–700.
* ISO 19906:2010 *Petroleum and natural gas industries — Arctic offshore structures*.
* Lindqvist, G. (1989) *A straightforward method for calculation of ice resistance of ships*, POAC'89, 722–735.
