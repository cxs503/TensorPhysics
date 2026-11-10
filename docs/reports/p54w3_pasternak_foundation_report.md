# P54-W3-N02｜Pasternak 双参数弹性地基替代 Winkler 占位：Hetényi 解析锚 + 冰板弯曲破坏 FEM 的真实地基模型
**节点** P54-W3-N02 ｜ **mission** 9110 ｜ **仓** TensorPhysics ｜ **生成时间** 2026-11-10T08:25:31+00:00 UTC

**判据总览：全部通过 = PASS**

| 判据 | 内容 | 结论 |
|---|---|---|
| (a) | Winkler 退化（G_p=0）相对差 < 0.1 % | PASS |
| (b) | Hetényi/Vlasov 解析验证：r=0 < 3 %，[0,5l_c] RMS < 5 % | PASS |
| (c) | 剪切层效应单调（Spearman ρ=-1，≥5 档） | PASS |
| (d) | 破坏载荷 Pasternak vs Winkler 偏差 > 5 % | PASS |
| (e) | 负控：k=G_p=0 退化为自由板；P→0 则 w→0 | PASS |
| (f) | 交付：pytest ≥ 15 例 + results.json 全表 | PASS |

所有数字由 `scripts/p54w3_pasternak_foundation_verify.py` 程序化写入 `artifacts/p54w3_pasternak/results.json`，本报告由 `scripts/p54w3_pasternak_report.py` 从该 JSON 注入（零手抄）。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

**物理背景。** 平整冰层的弯曲破坏（level-ice bending failure）是冰区船舶冰载荷链的核心环节：船首挤压使冰板受弯，当弯曲正应力达到冰的抗弯强度时出现第一条裂纹（环裂）。该链的历史实现把海冰板放在 **Winkler 单参数地基** 上——地基反力 `p = k·w`，等价于一排彼此独立的弹簧，不传递剪切。物理上这意味着载荷正下方的冰只对正上方的载荷负责，邻近冰面与水的剪切相互作用被完全丢弃（缺口登记 **GAP-FEM-P0**）。本节点用 **Pasternak 双参数地基** 替代该占位：

```
p(x,y) = k·w(x,y) − G_p·∇²w(x,y)
D·∇⁴w − G_p·∇²w + k·w = q(x,y),   D = E·h³/[12(1−ν²)]
```

- `G_p = 0` → 精确退化为 Winkler；`k = G_p = 0` → 退化为无地基自由板；
- 无量纲耦合 `γ = G_p/√(D·k)`：`γ<2` 复根（阻尼振荡，类 Hetényi `kei`），`γ>2` 双实根（两个衰减长度，剪切层主导）。

**几何。** 无限大冰板近似：正方形域 L×L，四边自由（free），向加载点几何分级加密；破坏载荷算例用半径 R_c = 0.5000 m 的均布圆接触面（船首接触斑）。

**参数表（SI）。**

| 量 | 符号 | 值 | 来源 |
|---|---|---|---|
| 冰厚 | h | 1.0000 m | 配置 |
| 弹性模量 | E | 5.0e+09 Pa | 配置 |
| 泊松比 | ν | 0.3000 | 配置 |
| 抗弯强度 | σ_f | 7.0e+05 Pa | 配置 |
| 海水密度 / 重力 | ρ_w, g | 1025 kg/m³, 9.81 m/s² | 配置 |
| 弯曲刚度 | D | 4.579e+08 N·m | E h³/[12(1−ν²)] |
| Winkler 模量 | k = ρ_w g | 10055.25 Pa/m | 浮力（Archimedes） |
| 冰剪切模量 | G_ice = E/[2(1+ν)] | 1.923e+09 Pa | 公式 |
| 剪切层模量 | G_p = β·G_ice·h | 1.923e+09 N/m | β=1（任务书标定） |
| 特征长度 | l_c = (D/k)^¼ | 14.6079 m | 公式 |
| 远场衰减长度 | 1/min Re(a,b) | 437.3223 m | 公式 |
| 无量纲耦合 | γ | 896.2 | 公式（real 根区） |
| 首裂弯矩 | M_f = σ_f h²/6 | 1.167e+05 N·m/m | 公式 |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**2.1 闭式锚（Hetényi / Vlasov–Kerr Green 函数）。** 对无限大板，算子可分解为 `D(∇²−m₁)(∇²−m₂)`，`m₁,₂` 是 `D m² − G_p m + k = 0` 的两根；借助二维基本解 `(∇²−m)K₀(√m r) = −2πδ(r)` 得集中载荷 `P` 的挠度

```
w(r) = P/(2π√Δ)·[K₀(b r) − K₀(a r)],  a=√m₁, b=√m₂, Δ=G_p²−4Dk
w(0) = Re[P/(2π√Δ)·ln(a/b)]        （两支对数奇点相消 → 中心挠度有限）
w,r  = C[a K₁(ar) − b K₁(br)]
w,rr = C[b²(K₀(br)+K₁(br)/(br)) − a²(K₀(ar)+K₁(ar)/(ar))]
M_r  = −D(w,rr + ν w,r/r),   M_t = −D(w,r/r + ν w,rr)
```

`γ = 0` 时该式严格回到经典 `w = −(P l_c²/2πD)·kei(r/l_c)`：实现上 `G_p = 0` 直接委派给既有 `kelvin_point_load`，逐位一致（判据 a）。斜幅角 Bessel 用自实现 **向量化** 级数/渐近展开（`|z| ≤ 8.5` 升阶级数、否则 DLMF 10.40.2 渐近式），对 mpmath 30 位参考值实测最大相对误差 **5.3e-9**；mpmath 仅作测试锚保留（`bessel_k_reference`），不进生产路径（逐点 15 ms 太慢）。

**2.2 有限接触面（第二套独立实现）。** 点载荷的 *sagging*（底面受拉）弯矩在 `r→0` 对数发散，因此“压力真实、接触面有限”的破坏算例不能只用点载荷。用 Graf 加法定理把环向积分解析化

```
(1/2π)∫₀^{2π} K₀(a|Δx|) dθ = I₀(a r₋)·K₀(a r₊)
w(r) = q∫₀^{R_c} 2π r' C₁[I₀K₀(b) − I₀K₀(a)] dr'
```

并把该核的 `∂/∂r`、`∂²/∂r²`（`I₀'=I₁`、`I₀''=I₀−I₁/z`、`K₀'=−K₁`、`K₀''=K₀+K₁/z`）在 Gauss 源点上解析求积，得到无奇点的弯矩场。于是**破坏载荷有两条互相独立的计算通道**：点载荷 Green 函数与环积分圆盘场（§3d 交叉校验 0.03 %）。

**2.3 FEM 实现（最小侵入扩展）。** 在既有 BFS（Bogner–Fox–Schmit，C¹ Hermite）Kirchhoff 板求解器 `BFSPlateSolver` 上新增 `foundation_model ∈ {winkler, pasternak, none}` 与 `g_p`：

- 剪切层以 `∫ G_p ∇N·∇N dA` 进入单元矩阵（Pasternak 压力 `−G_p∇²w` 正是 `½G_p∫|∇w|²dA` 的 Euler–Lagrange 形式）；
- 3×3 Gauss 积分；`g_p = 0` 时该分支代码根本不执行 → Winkler 组装逐位不变（判据 a）；
- `foundation_model='none'` 令 `k_eff = G_eff = 0`，用于 k=G_p=0 自由板负控（判据 e），而不必放宽 `KirchhoffWinklerPlate` 对 `k_w > 0` 的既有校验；
- `PlateField` 携带 `foundation_model / g_p` 溯源自证；新增 `element_shear_matrix()` 供测试单独检查剪切贡献。

**2.4 参数标定与无量纲化。** `k = ρ_w g`（浮力，物理确定）；`G_p = β·G_ice·h`（Pasternak 土壤经验式 `G_p = G·H` 套到冰层），β 为**显式可审计**标定因子，并输出 β→(γ, 衰减长度, w(0), P_crit) 灵敏度表（§3e）。

**2.5 工具链。** Python 3.11 / numpy 2.4 / scipy（`spsolve` 稀疏直接解）/ matplotlib（Agg）/ pytest；分级网格 `graded_axis_centered`（中心 h₀ 至远场 ~100–200 m）。

## 3. 计算结果

### (a) Winkler 退化（判据：相对差 < 0.1 %%）

| 项 | 值 | 判据 | 结论 |
|---|---|---|---|
| FEM 刚度矩阵差异元素数（winkler vs pasternak g_p=0） | 0 / 234256 | =0 | PASS |
| FEM 挠度场最大相对差 | 0.0000 | <1e-3 | PASS |
| FEM 弯矩场最大相对差 | 0.0000 | <1e-3 | PASS |
| 闭式解 vs 经典 Hetényi（w，绝对差 m） | 0.0000 | =0 | PASS |
| 闭式解 vs 经典 Hetényi（w′, w″, M_r, M_t 绝对差） | 0.0000 | =0 | PASS |

**结论**：`G_p = 0` 时 FEM 刚度矩阵逐位相同（0 个差异元素），挠度/弯矩场逐位相同；闭式解直接复用经典解，差为精确 0——退化不是“近似”，是恒等。

### (b) Hetényi / Vlasov 解析验证（FEM vs 闭式解，r ∈ [0, 5l_c]）

| 算例 | γ | 域 L [m] | 每边单元 | w(0) FEM [m] | w(0) 解析 [m] | r=0 相对差 | RMS 相对差 | 去零点 RMS | 归一化 L2 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| gamma=1 balanced, fine mesh | 1.0 | 240 | 80 | 0.045 | 0.045 | 4.9640e-06 | 1.93 % | 0.33 % | 0.14 % | PASS |
| gamma=1 balanced, coarse mesh | 1.0 | 240 | 60 | 0.045 | 0.045 | 2.4374e-05 | 3.14 % | 0.50 % | 0.20 % | PASS |
| beta=1 calibration, L=2000 m | 896.2 | 2000 | 100 | 5.644e-04 | 5.626e-04 | 0.0032 | 1.10 % | 1.10 % | 0.93 % | PASS |
| beta=1 calibration, L=3000 m | 896.2 | 3000 | 110 | 5.624e-04 | 5.626e-04 | 4.3252e-04 | 0.13 % | 0.13 % | 0.12 % | PASS |

最差 RMS 相对差 = **3.14 %**，最差 r=0 相对差 = **0.0032**（域收敛研究：β=1 算例 L=2000 m → L=3000 m 把 r=0 误差从 0.0032 压到 4.3252e-04，说明剩余误差主要来自自由边截断而非格式）。

### (c) 剪切层效应（单调性）

| γ | 根区 | 衰减长度 [m] | w_max / w_max,Winkler | 峰值 hogging M [kN·m/m] | 峰值 \|w″\| [1/m] |
|---|---|---|---|---|---|
| 0.00 | winkler | 20.66 | 1.0000 | 18.99 | 6.825e-04 |
| 0.25 | complex | 19.48 | 0.9275 | 17.26 | 6.667e-04 |
| 0.50 | complex | 18.48 | 0.8667 | 15.87 | 6.530e-04 |
| 1.00 | complex | 16.87 | 0.7698 | 13.79 | 6.301e-04 |
| 2.00 | complex | 14.61 | 0.6366 | 11.21 | 5.958e-04 |
| 4.00 | real | 28.22 | 0.4841 | 8.75 | 5.508e-04 |
| 8.00 | real | 40.99 | 0.3392 | 6.98 | 4.982e-04 |

Spearman ρ：w_max = **-1.0**，峰值 hogging 弯矩 = **-1.0**，峰值曲率 = **-1.0**，FEM（3 档）w_max = **-1.0**；档数 7。

FEM 交叉校验（同一网格，仅改 G_p）：γ=0.0: w_max=0.0583 m、γ=1.0: w_max=0.0449 m、γ=4.0: w_max=0.0284 m → 挠度场随 G_p 单调变平，最大挠度单调下降。

### (d) 破坏载荷（R_c = 0.5000 m 圆盘接触面，M_f = 1.167e+05 N·m/m）

| γ | G_p [N/m] | P_crit [MN] | P_crit/P_crit,Winkler | 偏差 vs Winkler | 峰值半径 r/l_c |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 6.1424 | 1.0000 | 0.00 % | 2.103 |
| 0.5 | 1.073e+06 | 7.3504 | 1.1967 | 1966.63 % | 1.970 |
| 1.0 | 2.146e+06 | 8.4606 | 1.3774 | 3774.15 % | 1.850 |
| 2.0 | 4.291e+06 | 10.4029 | 1.6936 | 6936.32 % | 1.651 |
| 4.0 | 8.583e+06 | 13.3380 | 2.1715 | 11714.72 % | 1.384 |
| 8.0 | 1.717e+07 | 16.7245 | 2.7228 | 17228.12 % | 1.065 |

任务书标定（β=1，γ=896.2）：P_crit = **24.635 MN**，相对 Winkler **30106.66 %**（比值 4.011）。

FEM 交叉校验（同一网格、同一圆盘载荷，峰值 hogging 弯矩）：

| γ | FEM 峰值 M_r [N·m/m] | 闭式解峰值 M_r [N·m/m] | 相对差 | FEM 反演 P_crit [MN] |
|---|---|---|---|---|
| 0.0 | -18894.75 | -18993.80 | 0.52 % | 6.175e-06 |
| 1.0 | -13688.76 | -13789.33 | 0.73 % | 8.523e-06 |

接触面半径敏感性（峰值 hogging 位于接触斑之外，故结果对 R_c 不敏感）：

| R_c [m] | P_crit Winkler [MN] | P_crit γ=1 [MN] | 偏差 |
|---|---|---|---|
| 0.25 | 6.1412 | 8.4585 | 3773.42 % |
| 0.50 | 6.1424 | 8.4606 | 3774.15 % |
| 1.00 | 6.1471 | 8.4689 | 3777.07 % |
| 2.00 | 6.1659 | 8.5022 | 3789.10 % |

**结论**：剪切层把载荷摊开、降低峰值弯矩，首裂载荷显著升高；γ=1 已达 3774.15 %，任务书标定 β=1 达 30106.66 %，远超过 5 % 判据。

### (e) 负控

| 项 | 值 | 判据 | 结论 |
|---|---|---|---|
| 自由板（k=G_p=0，四边简支）中心挠度 FEM | 0.014195 m | — | — |
| Navier 双三角级数解 | 0.014195 m | — | — |
| 相对差 | 9.0143e-07 | < 5e-3 | PASS |
| 地基矩阵最大绝对值（k=0） | 0.0000 | =0 | PASS |
| 剪切层矩阵最大绝对值（G_p=0） | 0.0000 | =0 | PASS |
| 线性性：P=1 MN / P=1 kN 挠度比 | 1000.0000 | =1000 | PASS |
| P=0 时最大 |w| | 0.0000 m | =0 | PASS |

注：FEM 中心挠度与 Navier 级数解相对差 9.0143e-07，与教科书经典系数 `w_max = 0.00406 q a⁴/D`（ν=0.3）一致（本算例 q=10000 Pa, a=20 m, D=4.579e+08 N·m）。

### (f) 参数标定表（β 灵敏度）

| β | G_p [N/m] | γ | 根区 | 剪切长度 [m] | 衰减长度 [m] | w(0)/w(0),Winkler | P_crit/P_crit,Winkler | P_crit 偏差 |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.000 | 0.000 | winkler | 0.00 | 20.66 | 1.0000 | 1.0000 | 0.00 % |
| 0.0010 | 1.923e+06 | 0.896 | complex | 13.83 | 17.17 | 0.7877 | 1.3412 | 3412.11 % |
| 0.0030 | 5.769e+06 | 2.689 | real | 23.95 | 21.88 | 0.5723 | 1.8793 | 8793.05 % |
| 0.0100 | 1.923e+07 | 8.962 | real | 43.73 | 43.46 | 0.3178 | 2.8068 | 18067.63 % |
| 0.0300 | 5.769e+07 | 26.887 | real | 75.75 | 75.69 | 0.1562 | 3.3555 | 23554.71 % |
| 0.1000 | 1.923e+08 | 89.624 | real | 138.29 | 138.28 | 0.0639 | 3.5476 | 25476.01 % |
| 0.3000 | 5.769e+08 | 268.873 | real | 239.53 | 239.53 | 0.0265 | 3.6684 | 26683.72 % |
| 1.0000 | 1.923e+09 | 896.245 | real | 437.32 | 437.32 | 0.0097 | 4.0107 | 30106.66 % |

**关键读数**：γ=1 的“弯剪同尺度”工作在 β≈0.0011 附近，而任务书标定 β=1 对应 γ≈896.2——比弯剪平衡区高约 3 个数量级，衰减长度被拉到 ~437.3 m，已超出单块冰排的物理尺度。该表的用途正是把这一标定决策显式化、可审计化。

### (g) 测试与交付

- `pytest tests/test_pasternak_foundation.py -q` → **35 passed / 0 failed**（returncode 0，耗时 51.2 s）；
- `results.json` 含：判据 a–f、Hetényi 验证 4 算例全指标、剪切层 7 档扫描、破坏载荷表 + FEM 交叉校验 + 接触半径扫描、参数标定 8 行、6 张图 base64 索引、pytest 记录；
- 图（`artifacts/p54w3_pasternak/`）：
  - `fig_a_winkler_degeneration.png` — (a) G_p=0 Pasternak point load coincides with the Winkler Hetenyi solution; the lower panel shows the double precision residual (exact zero).
  - `fig_b_hetenyi_fem_vs_analytic.png` — (b) FEM radial deflection versus the Pasternak/Vlasov Green function for the balanced regime (gamma=1) and the task-prescribed calibration (beta=1, gamma~896); lower row is the pointwise relative error.
  - `fig_c_shear_sweep.png` — (c) Shear-layer effect: centre deflection, peak hogging moment and peak curvature all decrease monotonically with the dimensionless coupling gamma (Spearman rho = -1).
  - `fig_d_calibration_failure.png` — (d) Ice calibration: gamma spans three decades between beta=1e-3 and the task-prescribed beta=1; the first-crack load of a 0.5 m contact patch departs from Winkler by more than 5 % already near gamma~1.
  - `fig_e_winkler_vs_pasternak_field.png` — (e) Deflection contours (left) and radial moments (right) for the same 0.5 m disc load: the Pasternak sheet spreads the load, the hogging peak shrinks and moves towards the contact edge.
  - `fig_f_free_plate_control.png` — (f) Negative control: with k = G_p = 0 the solver reproduces the simply supported (Navier) free-plate deflection to better than 0.1 %.

## 4. 结果图

![fig_a_winkler_degeneration.png](../artifacts/p54w3_pasternak/fig_a_winkler_degeneration.png)

*(a) G_p=0 Pasternak point load coincides with the Winkler Hetenyi solution; the lower panel shows the double precision residual (exact zero).*

![fig_b_hetenyi_fem_vs_analytic.png](../artifacts/p54w3_pasternak/fig_b_hetenyi_fem_vs_analytic.png)

*(b) FEM radial deflection versus the Pasternak/Vlasov Green function for the balanced regime (gamma=1) and the task-prescribed calibration (beta=1, gamma~896); lower row is the pointwise relative error.*

![fig_c_shear_sweep.png](../artifacts/p54w3_pasternak/fig_c_shear_sweep.png)

*(c) Shear-layer effect: centre deflection, peak hogging moment and peak curvature all decrease monotonically with the dimensionless coupling gamma (Spearman rho = -1).*

![fig_d_calibration_failure.png](../artifacts/p54w3_pasternak/fig_d_calibration_failure.png)

*(d) Ice calibration: gamma spans three decades between beta=1e-3 and the task-prescribed beta=1; the first-crack load of a 0.5 m contact patch departs from Winkler by more than 5 % already near gamma~1.*

![fig_e_winkler_vs_pasternak_field.png](../artifacts/p54w3_pasternak/fig_e_winkler_vs_pasternak_field.png)

*(e) Deflection contours (left) and radial moments (right) for the same 0.5 m disc load: the Pasternak sheet spreads the load, the hogging peak shrinks and moves towards the contact edge.*

![fig_f_free_plate_control.png](../artifacts/p54w3_pasternak/fig_f_free_plate_control.png)

*(f) Negative control: with k = G_p = 0 the solver reproduces the simply supported (Navier) free-plate deflection to better than 0.1 %.*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **符号陷阱（被独立通道抓出）**：`w,rr` 里 `K₀''` 组合的 `a²` 与 `b²` 次序写反会给出几乎等幅反号的弯矩场（`w`、`w,r` 却完全正确，很隐蔽）。是 *有限差分 vs 解析导数* 的交叉校验把它揪出来；修复后点载荷弯矩与环积分圆盘场互相吻合到 1e-6。**教训：只验挠度不足以验弯矩。**
2. **点载荷的 sagging 弯矩对数发散**：`M_r(r→0) → +∞`（底面受拉），因此“底面裂纹/接触压溃”类判据必须用有限接触面；而环裂（hogging）峰值是**内部极值**，故点载荷与圆盘两通道的 P_crit 一致到 0.03 %——两条通道互为验证，而非二选一。
3. **自由边截断**：无限板近似要求域 ≳ 6–7 个衰减长度。β=1 算例 L=2000 m（≈4.6 衰减长度）中心误差 0.32 %，L=3000 m 降到 0.04 %；γ=1 算例 240 m 已足够（RMS 1.9 %）。域选择必须与 γ 联动。
4. **G_p 标定的物理边界**：`k = ρ_w g` 有浮力一阶原理支撑；`G_p = β·G_ice·h` 只是 Pasternak 土壤经验式的迁移。开阔水域**不能传递静剪切**（β→0 才是冰-水界面的保守取值），β>0 更适合代表碎冰/搁浅/浅滩等可剪切的下卧层。因此本节点交付的是**机制 + 可审计标定表**，不是“β=1 是真理”。
5. **板理论层级**：实现的是 Kirchhoff 薄板（BFS C¹ 单元）。节点文本提到的 `plate_mindlin` 在 TensorPhysics 中不存在（冰板 FEM 为 `ice_sheet/kirchhoff_plate_fem.py`），故按最小侵入原则扩展既有求解器，横向剪切变形（Mindlin/Reissner）仍未覆盖——这是与节点描述的一处显式偏差。
6. **小挠度线性理论**：`M_f = σ_f h²/6` 首裂判据、无接触压溃/塑性、无裂纹扩展；破坏载荷是“首裂”载荷而非极限承载。
7. **Bessel 核精度**：复幅角 `K₀/K₁` 用级数+渐近，实测最差 5.3e-9（mpmath 30 位参考）；解析层总误差远低于 FEM 离散误差，不构成瓶颈。
8. **数值成本**：环积分场 `(n_field × n_source)` 全向量化，601×48 已使 P_crit 收敛到 1e-6；无需 mpmath 进生产路径。
9. **第二个被交叉校验抓出的坑**：极坐标求积版圆盘载荷曾漏掉径向 Jacobian（`dr' = R_c/2 dξ`），结果恰好偏大 `1/R_c = 2` 倍且随 `R_c` 漂移；与环积分解析场逐点对比（现为回归测试 `test_disc_load_quadrature_matches_ring_averaged_field`，3 档 γ）后修复，两套圆盘实现现吻合 1.5e-4–2.4e-4（残差为场插值误差，与求积加密无关）。

## 6. 误差判定（关键量 vs 基准，逐项显式回答“是否 < 3 %”）

| 关键量 | 基准 | 相对差 | 是否 < 3 % |
|---|---|---|---|
| FEM 中心挠度 w(0)（γ=1，细网格） | 闭式 Pasternak/Vlasov Green 函数 | 4.9640e-06 | **是** |
| FEM 中心挠度 w(0)（β=1，L=3000 m） | 同上 | 4.3252e-04 | **是** |
| FEM 全场挠度 RMS（γ=1，r∈[0,5l_c]） | 同上 | 1.93 % | **是** |
| FEM 全场挠度 RMS（β=1，L=3000 m） | 同上 | 0.13 % | **是** |
| FEM 峰值 hogging 弯矩（圆盘载荷，Winkler） | 环积分闭式场 | 0.52 % | **是** |
| FEM 峰值 hogging 弯矩（圆盘载荷，γ=1） | 环积分闭式场 | 0.73 % | **是** |
| 自由板中心挠度（k=G_p=0） | Navier 双三角级数解 | 9.0143e-07 | **是** |
| Pasternak G_p=0 挠度/弯矩 | 经典 Hetényi/Kelvin 解 | 0.0000 | **是（精确 0）** |
| 点载荷 P_crit vs 圆盘 P_crit（两条独立实现） | 互为基准 | < 1e-3（测试断言） | **是** |
| 复幅角 Bessel K₀/K₁ | mpmath 30 位 | 5.3e-9（测试断言 <1e-7） | **是** |

**没有基准的量**：Pasternak 地基下冰板首裂载荷本身无实验/文献对照值（`G_p` 的冰-水取值本身是建模决策），本报告只声明“与 Winkler 的偏差”这一相对量（判据 d），并给出 β 灵敏度表；未做任何“与实测吻合”的断言。

## 7. 审计链与复现

- 模块：`src/tensorphysics/ice_sheet/pasternak_foundation.py`（闭式锚 / 环积分圆盘场 / 标定）
- FEM 扩展：`src/tensorphysics/ice_sheet/kirchhoff_plate_fem.py`（`foundation_model`, `g_p`, `element_shear_matrix`, `PlateField` 溯源字段）
- 测试：`tests/test_pasternak_foundation.py`（35 例）
- 驱动：`scripts/p54w3_pasternak_foundation_verify.py` → `artifacts/p54w3_pasternak/results.json` + `fig_*.png`
- 报告生成：`scripts/p54w3_pasternak_report.py`（本文件，全部数字来自 results.json）
- 复现：`PYTHONPATH=src python scripts/p54w3_pasternak_foundation_verify.py && PYTHONPATH=src python scripts/p54w3_pasternak_report.py`
- 缺口：本节点关闭 **GAP-FEM-P0**（Winkler 弹性地基占位 → 含剪切相互作用的双参数地基）；`docs/fem_dem_capability_inventory.md` 已同步登记。
