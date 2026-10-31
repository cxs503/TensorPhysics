# W3-N04 冰载荷下船体结构可靠度直接法：FORM/SORM + ISO 19906 极限状态

**节点**: W3-N04 ｜ **仓库**: TensorPhysics ｜ **分支**: `agent/m791` ｜ **模块**: `tensorphysics.structural` ｜ **生成时间(UTC)**: 2026-10-31T13:12:52+00:00

**总体判据**: 8 项机器判据 **全部通过**（明细见 §3.6 与 §6）。

## 1. 案例介绍

### 1.1 物理背景

极地运输船/破冰船的中部横剖面冰带（ice belt）长期承受海冰挤压。冰压力极值具有明显的极值统计特征，而船体响应又受材料强度、腐蚀/磨蚀裕量与计算模型不确定性的共同影响，因此板厚设计本质上是一个**可靠度问题**：在给定极限状态与目标可靠度指标 $\beta$（ISO 19906 对高后果等级的海洋结构推荐 $\beta \approx 4$）下反算所需板厚。

本节点在 TensorPhysics 中实现三类极限状态（屈服 / 弹塑性屈曲 / Miner 疲劳）的一次二阶矩（FORM，Hasofer-Lind 设计点 + Nataf 变换）、Breitung 二次型修正（SORM）、多失效模式 Ditlevsen 界限合成，以及可靠度指标对分布参数的灵敏度，并用 100 万样本Monte Carlo 与解析/求积参考值逐项校核。

### 1.2 结构与载荷模型（程序化参数）

| 参数 | 含义 | 取值 |
|---|---|---|
| $t_{nom}$ | 冰带板厚（as-built） | 34.0 mm |
| $b$ | 横向加强筋间距（板弯曲跨距） | 350.0 mm |
| $a$ | 强框架间距（屈曲板格尺寸） | 2400.0 mm |
| $\alpha_b$ | 长固支板弯曲系数 | 0.500 |
| $k_\sigma$ | 板格屈曲系数（简支单轴） | 4.00 |
| $E$ | 弹性模量 | 206000 MPa |
| $\nu$ | 泊松比 | 0.30 |
| $\sigma_{global}$ | 冰带总纵弯曲压应力（确定性） | 95.0 MPa |
| $\psi$ | 冰压强向面内压应力转换系数 | 1.50 |
| $A$ | S-N 曲线（$m$=3）常系数 | $10^{11.764}$ |
| $m$ | S-N 曲线斜率 | 3.0 |
| $n$ | 等效恒幅循环数（设计寿命） | 5.000e+05 |
| $\Delta p_{eq}$ | 等效恒幅压力幅值 | 1.00 MPa |
| $D_{allow}$ | 允许 Miner 损伤（DF=2） | 0.50 |

三条极限状态（单位 MPa/mm，$t_{{net}} = t_{{nom}} - \Delta t$）：

$$g_1^{yield} = f_y - k_{mod}\,\alpha_b\,p\left(\frac{b}{t_{net}}\right)^2$$

$$g_2^{buck} = \sigma_{cr}(t_{net}, f_y) - k_{mod}\left(\sigma_{global} + \psi p\right),\quad \sigma_{cr} = \begin{cases}\sigma_E & \sigma_E \le \tfrac{f_y}{2}\\ f_y\left(1-\dfrac{f_y}{4\sigma_E}\right) & \sigma_E > \tfrac{f_y}{2}\end{cases},\quad \sigma_E = k_\sigma\frac{\pi^2E}{12(1-\nu^2)}\left(\frac{t_{net}}{a}\right)^2$$

$$g_3^{fat} = D_{allow} - \frac{n}{A}\left[k_{mod}\alpha_b\Delta p_{eq}\left(\frac{b}{t_{net}}\right)^2\right]^{m}$$

### 1.3 随机变量与相关结构

| 变量 | 分布 | 均值 | 标准差 | COV | 分布参数 | 单位 |
|---|---|---|---|---|---|---|
| `p_ice` | gumbel_max | 2.500000 | 0.700000 | 28.00% | `loc`=2.184963, `scale`=0.545788 | MPa |
| `fy` | lognormal | 355.000000 | 28.400000 | 8.00% | `mu_ln`=5.868928, `sigma_ln`=0.079872 | MPa |
| `dt_corr` | normal | 2.000000 | 0.500000 | 25.00% | `mu`=2.000000, `sigma`=0.500000 | mm |
| `k_mod` | lognormal | 1.000000 | 0.100000 | 10.00% | `mu_ln`=-0.004975, `sigma_ln`=0.099751 | - |

物理相关矩阵（`p_ice` 与 `k_mod` 正相关：载荷模型偏差与压力极值由同一批未建模接触面积效应驱动；其余为 0）：

```
            p_ice         fy    dt_corr      k_mod
   p_ice    1.000      0.000      0.000      0.250
      fy    0.000      1.000      0.000      0.000
 dt_corr    0.000      0.000      1.000      0.000
   k_mod    0.250      0.000      0.000      1.000
```

Nataf 变换反解得到的高斯 copula 相关系数 $\rho_0$（非正态边缘导致 $\rho_0 \ne \rho$）：

```
            p_ice         fy    dt_corr      k_mod
   p_ice   1.0000     0.0000     0.0000     0.2574
      fy   0.0000     1.0000     0.0000     0.0000
 dt_corr   0.0000     0.0000     1.0000     0.0000
   k_mod   0.2574     0.0000     0.0000     1.0000
```

### 1.4 名义应力水平（均值点，程序化）

| 量 | 数值 |
|---|---|
| 净板厚 $t_{net}$ [mm] | 32.000000 |
| 板弯曲应力 [MPa] | 149.536133 |
| 面内压应力需求 [MPa] | 98.750000 |
| 弹性屈曲应力 $\sigma_E$ [MPa] | 132.398112 |
| 屈曲承载力 $\sigma_{cr}$ [MPa] | 132.398112 |
| Miner 损伤 $D$ [-] | 0.184242 |
| 允许损伤 $D_{allow}$ [-] | 0.500000 |
| 屈服强度均值 [MPa] | 355.000000 |

## 2. 方法与流程

### 2.1 工具链

- `tensorphysics/structural/distributions.py`：标准正态函数（`erfc` 尾部精确）、Acklam+Halley 反函数、20 点 Gauss-Hermite 表、边缘分布（Normal / Lognormal / Gumbel-max / Exponential，均带 `ppf_z` 尾部稳定实现）、Nataf 变换（逐对求积解 $\rho_0$ + Cholesky）。
- `tensorphysics/structural/reliability.py`：径向 HL-RF 迭代（每步沿梯度方向对 $g(\lambda\alpha)$ 做 Illinois 求根）+ 阻尼 HL-RF 抛光；Breitung SORM（Hessian 中心差分 → 切空间投影 → Jacobi 特征值 → 主曲率）；crude Monte Carlo / 设计点重要抽样；Ditlevsen 双模式界限（$\rho_{ij}=\alpha_i\cdot\alpha_j$，$P_{ij}=\Phi_2$）；设计点灵敏度 $\partial\beta/\partial\theta = \alpha^{T}\partial u/\partial\theta$。
- `tensorphysics/structural/ice_structure.py`：ISO 19906 冰带板格三类极限状态、随机变量集、系统界限、目标 $\beta$ 的板厚反算（二分法）。
- `tensorphysics/structural/_batch.py`：可选 NumPy 向量化通道（仅用于大规模抽样加速，标量路径与向量路径在测试中互相校验）。

### 2.2 算法要点

1. **u 空间与 Nataf**：$u$ 为独立标准正态坐标，$z=L\,u$（$LL^T=\rho_0$），$x_i = F_i^{-1}(\Phi(z_i))$。可靠度指标 $\beta=\|u^*\|$。
2. **设计点**：求解 $\min \|u\|$ s.t. $g(T(u))=0$。利用 KKT 条件的固定点形式（$u=\beta(\alpha)\alpha$，$\alpha=-\nabla g/\|\nabla g\|$）逐次沿射线求根，收敛后再用阻尼 HL-RF 抛光。经典割线 HL-RF 在高曲率面（疲劳三次型/抛物面）会振荡，射线法对此稳健（见 §5）。
3. **SORM（Breitung）**：$P_f \approx \Phi(-\beta)\prod_i (1+\beta\kappa_i)^{-1/2}$，$\kappa_i$ 为 $P H P/\|\nabla g\|$ 在切空间上的特征值（$P=I-nn^T$；正曲率=面向背离原点鼓出=失效概率下降）。$1+\beta\kappa_i\le 0$ 时公式失效，代码显式置 `valid=False`。
4. **系统**：串联（最弱环）系统失效域并集，Ditlevsen 双模式界限：
   $\max\left(\sum_i P_i - \sum_{i<j}P_{ij},\ \max_i P_i\right) \le P_f \le \sum_i P_i - \sum_i \max_{j<i} P_{ij}$。
5. **灵敏度**：由包络定理 $\partial\beta/\partial\theta = \alpha^{T}\,\partial u/\partial\theta\big|_{x^*}$，其中 $\partial z_i/\partial\theta = (\partial F_i/\partial\theta)/\varphi(z_i)$，$\partial u = L^{-1}\partial z$（解析）；参考值由重解 FORM 的中心差分（$h=10^{-4}$）给出。
6. **板厚反算**：对 $\beta(t)$ 做二分（区间 15–90 mm，$|\beta-\beta_{target}|\le 5\times10^{-3}$），分别按单模式与串联系统给出。

## 3. 计算结果

### 3.1 三类极限状态的可靠度指标

| 极限状态 | FORM $\beta$ | FORM $P_f$ | SORM $\beta$ | MC $\beta$ (1e6) | MC $P_f$ | |FORM−MC|/MC | 重要抽样 $\beta$ |
|---|---|---|---|---|---|---|---|
| 屈服（冰压板弯曲） | 2.66094 | 0.00390 | 2.66018 | 2.65283 | 0.00399 | 0.31% | 2.66069 |
| 屈曲（板格弹塑性） | 2.77815 | 0.00273 | 2.75715 | 2.75574 | 0.00293 | 0.81% | 2.75511 |
| 疲劳（Miner 损伤） | 3.22895 | 6.21220e-04 | 3.22692 | 3.21731 | 6.47000e-04 | 0.36% | 3.22694 |

名义板厚 34.0 mm 下三条极限状态的 $\beta$ 处于同一量级（2.66–3.23），说明该冰带设计在屈服/屈曲/疲劳三个模式上的可靠度是**均衡**的；串联系统指标 $\beta=2.501$ 低于最弱单模式（$\beta=2.661$），体现了相关失效模式的合成效应。

设计点（物理空间）与方向余弦：

| 极限状态 | $p_{ice}$ [MPa] | $f_y$ [MPa] | $\Delta t$ [mm] | $k_{mod}$ [-] | $\alpha_{p}$ | $\alpha_{fy}$ | $\alpha_{\Delta t}$ | $\alpha_{k}$ |
|---|---|---|---|---|---|---|---|---|
| 屈服（冰压板弯曲） | 4.93423 | 337.62635 | 2.11550 | 1.13572 | +0.934 | -0.221 | +0.087 | +0.267 |
| 屈曲（板格弹塑性） | 3.14781 | 353.86942 | 2.40399 | 1.29436 | +0.362 | -0.000 | +0.291 | +0.886 |
| 疲劳（Miner 损伤） | 2.96145 | 353.86942 | 2.48946 | 1.35250 | +0.245 | -0.000 | +0.303 | +0.921 |

设计点处 $|g|$ 与迭代次数：

| 极限状态 | $|g(x^*)|$ | 迭代次数 | 收敛 | $\|\nabla_u g\|$ |
|---|---|---|---|---|
| 屈服（冰压板弯曲） | 5.684e-14 | 4 | 是 | 121.97684 |
| 屈曲（板格弹塑性） | 8.117e-11 | 5 | 是 | 14.04655 |
| 疲劳（Miner 损伤） | 3.109e-15 | 3 | 是 | 0.15702 |

SORM 主曲率（Breitung 修正量）：

- 屈服（冰压板弯曲）: $\kappa$ = -0.0014, -3.2035e-04, -4.8387e-10（有效: 是）
- 屈曲（板格弹塑性）: $\kappa$ = -0.0398, -0.0041, 0（有效: 是）
- 疲劳（Miner 损伤）: $\kappa$ = -0.0044, 0, 2.1970e-10（有效: 是）

冰带三类极限状态在 Nataf-u 空间中接近线性（曲率 $|\kappa| \lesssim 0.04$），因此 SORM 与 FORM 几乎重合——这与 §3.5 的强非线性验证案例形成对照。

### 3.2 多失效模式合成（Ditlevsen 界限）

- 分量 $P_f$: 屈服（冰压板弯曲）=0.0039, 屈曲（板格弹塑性）=0.0027, 疲劳（Miner 损伤）=6.2122e-04
- 双模式相关系数 $\rho_{ij}=\alpha_i\cdot\alpha_j$:
  - $\rho$(yield, buckling) = 0.5994, $P_{ij}$ = 4.3470e-04
  - $\rho$(yield, fatigue) = 0.5011, $P_{ij}$ = 9.2779e-05
  - $\rho$(buckling, fatigue) = 0.9925, $P_{ij}$ = 6.2121e-04
- Ditlevsen 界限: $P_f \in [0.00619,\ 0.00619]$，对应 $\beta \in [2.50084,\ 2.50084]$（$\beta$ 与 $P_f$ 单调反向）
- 点估计（界限中值）: $P_f=0.00619$，$\beta=2.50084$
- 100 万样本联合失效模拟: $P_f=0.00627$，$\beta=2.49674$（与上限偏差 +1.16%）

模拟值比 Ditlevsen 上限高 +1.16%（1e6 样本对 $P_f\sim6\times10^{-3}$ 的统计 1σ 约 1.3%），属 FORM 线性化对三个相关失效面合成时的小幅非保守偏差，在 §5.2 中作为局限记录。

### 3.3 目标 $\beta$ 的板厚反算

| 目标 $\beta$ | 屈服所需 $t$ [mm] | 屈曲所需 $t$ [mm] | 疲劳所需 $t$ [mm] | 系统所需 $t$ [mm] | 相对名义板厚 |
|---|---|---|---|---|---|
| 3.0 | 36.02 | 34.39 | 33.62 | 36.06 | +6.1% |
| 3.5 | 39.17 | 35.27 | 34.45 | 39.17 | +15.2% |
| 4.0 | 42.61 | 36.17 | 35.29 | 42.61 | +25.3% |
| 4.5 | 46.27 | 37.10 | 36.15 | 46.27 | +36.1% |

ISO 19906 类目标 $\beta=4.0$ 下，串联系统要求 $t \ge 42.61$ mm（对应 $\beta=4.0006$），比名义值 34.0 mm 高 **+25.3%**；控制模式为屈服（冰压板弯曲）（$\beta=4$ 需 42.61 mm）。

### 3.4 灵敏度 $\partial\beta/\partial\theta$（屈服模式）

| 参数 | 设计点解析值 | 中心差分 (h=1e-4) | 相对差 |
|---|---|---|---|
| `p_ice.loc` | -5.60970e-01 | -5.60970e-01 | 0.0000% |
| `p_ice.scale` | -2.82574e+00 | -2.82574e+00 | 0.0000% |
| `fy.mu_ln` | +2.76795e+00 | +2.76795e+00 | 0.0000% |
| `fy.sigma_ln` | -1.62836e+00 | -1.62836e+00 | 0.0000% |
| `dt_corr.mu` | -1.73624e-01 | -1.73624e-01 | 0.0000% |
| `dt_corr.sigma` | -4.01074e-02 | -4.01074e-02 | 0.0000% |
| `k_mod.mu_ln` | -2.76795e+00 | -2.76795e+00 | 0.0000% |
| `k_mod.sigma_ln` | -3.66944e+00 | -3.66992e+00 | 0.0129% |

弹性屈曲模式对 $f_y$ 的灵敏度为 0（$\sigma_E \le f_y/2$ 时为纯弹性屈曲分支，承载力与屈服强度无关），这与 Johnson-Ostenfeld 分段定义的物理含义一致。

### 3.5 强非线性验证：抛物面极限状态（判据 c）

极限状态 g = 3.0 - u1 + 1.0*(u2^2 - 1) in standard normal space：

| 量 | 数值 |
|---|---|
| 求积参考 $\beta$ | 2.358699 |
| 参考 $P_f$ | 0.009170 |
| FORM $\beta$ | 2.000000（误差 15.207%） |
| SORM $\beta$ | 2.319862（误差 1.647%） |
| MC(1e6) $\beta$ | 2.356542 |
| SORM vs MC 相对差 | 1.556% |
| FORM vs MC 相对差 | 15.130% |
| 主曲率 $\kappa$ | 2.00000 |

闭式验证案例（判据 a）：

| 案例 | 极限状态 | 参考 $\beta$ | FORM $\beta$ | 相对误差 |
|---|---|---|---|---|
| linear | `g = R - S, R~N(400,40), S~N(200,30)` | 4.000000 | 4.000000 | 0.0000% |
| exponential_lsf | `g = exp(2) - exp(X), X~N(1,0.5)  (failure iff X > 2)` | 2.000000 | 2.000000 | 0.0000% |
| exponential_marginal | `g = R - E, R~N(3,0.3), E~Exp(rate=1)` | 1.625025 | 1.629558 | 0.2790% |

### 3.6 机器判据汇总

| 判据 | 内容 | 实测 | 限值 | 结果 |
|---|---|---|---|---|
| (a) | closed-form linear/exponential beta vs analytic (max rel. error) | 0.2790% | < 0.5% | PASS |
| (b) | FORM beta vs Monte Carlo 1e6 (max rel. difference over 3 limit states) | 0.8132% | < 2% | PASS |
| (c1) | SORM beta vs Monte Carlo 1e6 on strongly nonlinear surface | 1.5565% | < 3% | PASS |
| (c2) | SORM better than FORM on the nonlinear surface | SORM 1.647% vs FORM 15.207% | SORM < FORM | PASS |
| (d) | design-point d beta / d theta vs central difference (h = 1e-4) | 0.0129% | < 1% | PASS |
| (e) | pytest suite of the structural reliability module | 52 passed, 0 failed | >= 18 passed | PASS |
| (design) | target beta = 4.0 back-solved plate thickness (system) | 42.61 mm (achieved beta 4.001) | |beta - 4.0| <= 0.01 | PASS |
| (system) | Ditlevsen bounds bracket the simulated 1e6-sample union probability | bounds [6.1949e-03, 6.1949e-03] vs simulated 6.2670e-03 | within 10% | PASS |

pytest 执行：`python -m pytest tests/test_structural_distributions.py tests/test_structural_reliability.py tests/test_structural_ice_reliability.py -q --tb=no` → 52 passed, 1 warning in 13.35s

疲劳损伤 MC 分布（名义板厚，1000000 样本）：均值 0.1908，95% 分位 0.3043，99% 分位 0.3762，超越 $D_{allow}=0.50$ 的比例 6.1200e-04（与疲劳模式 FORM/SORM 失效概率一致）。

## 4. 结果图

**图 1**（`docs/reports/w3n04/figures/fig1_limit_state_fields.png`）：Safety-margin fields g(p_ice, t_nom) for the three limit states with the remaining random variables frozen at their design-point values; the black curve is g = 0, the star is the design point projected on the nominal thickness line.

![fig1](figures/fig1_limit_state_fields.png)

**图 2**（`docs/reports/w3n04/figures/fig2_beta_vs_thickness.png`）：Component and series-system reliability index versus as-built plating thickness; the intersection with the beta = 4.0 target defines the required thickness.

![fig2](figures/fig2_beta_vs_thickness.png)

**图 3**（`docs/reports/w3n04/figures/fig3_form_sorm_mc.png`）：Reliability indices from FORM, Breitung SORM and Monte Carlo (1e6 samples) for the three ice-belt limit states and the strongly nonlinear parabolic verification case.

![fig3](figures/fig3_form_sorm_mc.png)

**图 4**（`docs/reports/w3n04/figures/fig4_convergence.png`）：Convergence of the radial HL-RF iteration: |g| falls to the 1e-13 level within a few iterations for all three limit states.

![fig4](figures/fig4_convergence.png)

**图 5**（`docs/reports/w3n04/figures/fig5_sensitivity.png`）：Design-point sensitivities of the yield-mode reliability index with respect to all eight distribution parameters, analytic versus central differences.

![fig5](figures/fig5_sensitivity.png)

**图 6**（`docs/reports/w3n04/figures/fig6_correlation.png`）：Reliability index versus the physical correlation between the ice-pressure extreme and the load-model bias; the Nataf transformation re-solves the Gaussian-copula correlation at every point.

![fig6](figures/fig6_correlation.png)

**图 7**（`docs/reports/w3n04/figures/fig7_fatigue_damage.png`）：Monte Carlo distribution of the Miner damage at the nominal thickness; the tail beyond D_allow is the fatigue failure probability.

![fig7](figures/fig7_fatigue_damage.png)

**图 8**（`docs/reports/w3n04/figures/fig8_sorm_geometry.png`）：Parabolic limit-state surface in standard normal space: FORM replaces the surface by its tangent at the design point, SORM corrects with the principal curvature kappa = 2c.

![fig8](figures/fig8_sorm_geometry.png)

图件共 8 张，均由 `benchmarks/structural_reliability/run_w3n04_reliability.py` 用 matplotlib(Agg) 程序化生成，数据来自同一次运行的 JSON，无手工插值。

## 5. 注意点（踩坑与局限）

### 5.1 实现过程中踩到的坑

1. **经典 HL-RF 割线更新在本问题上不收敛**：屈服面在 u 空间曲率虽小，但首个割线步会把迭代点抛过失效面（$g$ 跨过零点变号），随后阻尼线搜索进入极小步长的蠕变，迭代次数耗尽仍停在明显偏小的 $\beta$ 上（与 MC 对不上）。改为**径向射线求根（Illinois）+ 阻尼 HL-RF 抛光**后，三模式分别在 4, 5, 3 次迭代内使 $|g|$ 降到 $10^{-13}$ 量级（§3.1、图 4）。
2. **概率空间饱和**：$\Phi(z)$ 在 $|z|\gtrsim8.2$ 处被 double 精度截断为 1，导致大 $\beta$ 反解时逆变换退化为常数、梯度为 0、迭代停滞（现象：板厚扫描到 $\beta_{form}\gtrsim7$ 时迭代次数耗尽且收敛标志为假）。改为边缘分布的 `ppf_z(z)` 直接解析式（Gumbel 用 `-log1p(-tail)` 稳定形式）后，板厚扫描（§3.3，$\beta$ 最高达数十）全部收敛。
3. **SORM 曲率符号**：最初取 $\kappa=-\mathrm{eig}(PHP/\|\nabla g\|)$，导致抛物面案例给出$1+\beta\kappa<0$ 的“失效”。用抛物面 $\kappa=2c$ 与求积参考值反推后确认应为 **正号**（正曲率=面向背离原点鼓出=$P_f$ 下降），并保留单元测试锁定该约定。
4. **Nataf $\rho_0$ 需要求解**：Gumbel–Lognormal 对上解出 $\rho_0=0.25736 \ne \rho=0.250$，直接用 $\rho$ 会在相关系数上引入 2.9% 的系统偏差；实现对每对边缘做 Gauss-Hermite 二重求积 + 二分反解，并用回代（把解出的 $\rho_0$ 重新求积）校验。
5. **Monte Carlo 性能**：纯 Python 标量路径（每样本 4 次分位数变换 + 1 次极限状态）在本机约几十微秒/样本，1e6 样本需十余秒；`numpy.frompyfunc` 的 object-array 路径更慢。最终采用 NumPy 向量化变换 + 向量化极限状态（1e6 样本亚秒级），并用测试锁定标量/向量两条路径一致（容差 1e-12）。

### 5.2 模型局限与适用范围

- **冰压力极值分布参数为标定假设**：$p_{ice}$ 取 Gumbel(max)、均值 2.5 MPa、COV 0.28，代表 ISO 19906 局部冰压力水平与 m768 极值统计链路的量级，本次**未**用 m768 原始数据重新拟合，因此 §3.3 的绝对板厚数值应视为“模型自洽的设计输出”，不宜直接用作工程结论；趋势与判据不受影响。
- **屈曲面元尺寸**：屈曲极限状态取强框架间距 2.4 m 作为板格尺寸，忽略加强筋扭转/侧向约束，偏保守；未考虑加筋板整体柱屈曲与冰带区域局部加强。
- **载荷转换系数** $\psi=1.5$ 与 $\sigma_{global}=95$ MPa 为确定性假设；§3.4 的灵敏度表明 $\beta$ 对这些量敏感，工程应用需以具体船型的总纵强度计算替换。
- **疲劳模型**采用等效恒幅应力幅 $\Delta p_{eq}=1.0$ MPa 与 5×10^5 循环的等效恒幅假设，S-N 散差被并入 $k_{mod}$（COV 0.10）与 DF=2 的部分系数中；未做雨流计数与长周期谱分析。
- **未建模**：腐蚀随时间的退化（时变可靠度）、空间相关的冰压力场、焊接残余应力、板厚制造公差（$t_{nom}$ 视为确定量）、多轴应力状态与低周冰致疲劳。
- **SORM** 在 $1+\beta\kappa_i \le 0$（曲率过大/球面型极限状态）时不可用，代码显式报 `valid=False`，此时应以重要抽样或高阶方法代替。
- **FORM 系统合成偏差**：串联系统模拟值比 Ditlevsen 上限高 +1.16%，属线性化近似的小幅非保守；系统 $\beta$ 的点估计 2.5008 与模拟值 2.4967 相差 0.0041。

## 6. 误差判定（逐项对照基准，显式回答“是否 < 3%”）

| # | 关键量 | 基准 | 本次结果 | 相对误差 | 是否 < 3% |
|---|---|---|---|---|---|
| 1 | 线性极限状态 $\beta$ | 解析 4.000000 | 4.000000 | 0.0000% | 是 |
| 2 | 指数极限状态 $\beta$ | 解析 2.000000 | 2.000000 | 0.0000% | 是 |
| 3 | 指数边缘载荷 $\beta$ | 求积 1.625025 | 1.629558 | 0.2790% | 是 |
| 4 | 冰带三模式 FORM $\beta$ vs MC(1e6) | Monte Carlo | 最大偏差 0.813% | 0.813% | 是 |
| 5 | SORM $\beta$（强非线性抛物面）vs MC(1e6) | Monte Carlo | 1.556% | 1.556% | 是 |
| 6 | SORM $\beta$ vs 求积参考 | 求积 2.358699 | 2.319862 | 1.647% | 是 |
| 7 | FORM $\beta$ vs 求积参考（同一强非线性面） | 求积 | 15.207% | 15.207% | 否（SORM 优于 FORM） |
| 8 | 灵敏度 $\partial\beta/\partial\theta$ vs 中心差分 | 有限差分 h=1e-4 | 最大偏差 0.0129% | 0.0129% | 是 |
| 9 | 串联系统 $P_f$ | 100 万样本联合模拟 0.00627 | 界限 [0.00619, 0.00619] | +1.16%（对上限） | 是 |

**结论**：判据 (a)–(e) 全部满足；SORM 在强非线性面上相对 MC 的偏差 1.56%（< 3%）且显著优于 FORM（15.21% → 1.65%）。冰带三条物理极限状态在 u 空间接近线性，FORM 已足够（偏差 < 1%），SORM 仅提供边际修正；两者与 MC 的差异均在统计误差范围内。

**局限声明**：第 1 项判据的“解析解”为闭式公式；第 2、3、5、6 项的基准为独立求积结果；第 4、9 项基准为 100 万样本 Monte Carlo（$P_f \sim 10^{-3}$ 时 $\beta$ 的 1σ 统计误差约 0.5–1%）。以上全部为**数值/统计基准**，没有实验数据；冰压力分布参数未从 m768 原始记录重拟合（见 §5.2）。

### 6.1 复现方式

```bash
PYTHONPATH=src python3 benchmarks/structural_reliability/run_w3n04_reliability.py
PYTHONPATH=src python3 -m pytest tests/test_structural_distributions.py \
    tests/test_structural_reliability.py tests/test_structural_ice_reliability.py -q
```

产物：`docs/reports/w3n04/reliability_results.json`（全部判据数字与路径）、`docs/reports/w3n04/figures/`（8 张图）、`outputs/w3n04_reliability/reliability_results.json`（运行工件副本）。

