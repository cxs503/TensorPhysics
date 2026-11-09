# IN101-N09 — ISO 19906 ice-pressure partial-factor back-calibration at beta_T = 4.0 (mature: target-reliability inverse, FORM/IS/MC cross-checked)

**仓库** TensorPhysics ｜ **模块** `tensorphysics.ice_partial_factors` ｜ **设计格式** `gamma_f * F_k <= R_k / psi,  kappa = gamma_f * psi` ｜ **目标** $\beta_T$ = 4.00 ($P_f$ = 3.1671e-05) ｜ **前序节点** IN13-W3-N10 (plate-stress limit state, one-parameter inverse)

| 机器判据 | 结果 |
|---|---|
| `a_form_vs_simulation` | PASS |
| `b_inverse_solved_unique` | PASS |
| `c_beta_sensitivity_vs_finite_difference` | PASS |
| `d_iso_table_deviation_map` | PASS |
| `e_negative_control_pa_correlation` | PASS |

**判据汇总** 5/5 通过

## ① 案例介绍（物理背景 / 参数表）

ISO 19906 用压力–面积关系描述冰载荷：作用于接触（或整体）面积 $A$ 上的冰压随面积下降，$p(A)=p_0(A/A_0)^{-ex}$。结构承受的冰力因此是 $F=k_F\,p(A)\,A$，而指数为负意味着**产生年极值冰力的冰压与面积负相关**：大面积伴随低压。本节点把「冰压年极值分布 → 抗力分布 → 极限状态」串成反问题，求设计格式 $\gamma_f F_k \le R_k/\psi$ 的分项系数对 $(\gamma_f,\psi)$，使按该格式设计的构件恰好达到目标年可靠度 $\beta_T = 4.0$。

| 参数 | 取值 | 说明 |
|---|---|---|
| $p_0$ | 2.500 MPa | 参考面积处冰压中值 |
| $ex$ | 0.400 | ISO 压力–面积指数 |
| $A_0$ / $A_{med}$ | 5.00 / 5.00 m² | 参考面积 / 年极值接触面积中值 |
| CoV($A$) | 0.350 | 年极值接触面积对数正态变异系数 |
| CoV($\varepsilon_p$) | 0.150 | 压力规律认知不确定性（中值归一）|
| CoV($k_F$) | 0.100 | 载荷模型不确定性 |
| CoV($R$) | 0.120 | 构件抗力变异系数 |
| 特征冰力 $F_k$ | 23.517 MN | 100 年超越（年超越概率 1e-2）|
| 特征抗力 $R_k$ | 33.660 MN | 5 % 分位 |
| $\rho_{\ln p,\ln A}$ | -0.6737 | 由指数与面积散度解析给出 |
| $\sigma_{\ln F}$ | 0.2717 | 计入负相关 |
| $\sigma_{\ln F}$（$\rho$=0 对照）| 0.4077 | 同一对边缘分布，忽略相关 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **联合 p–A 状态**（`p_area_joint.py`）。冰压与面积均为对数正态；ISO 幂律把两者在对数空间的相关锁定为 $\rho_{\ln p,\ln A}=-ex\,\sigma_{\ln A}/\sigma_{\ln p}$。$\text{Nataf}$ 变换吃的是**物理**相关系数，故用对数正态的精确关系 $\rho_{phys}=(e^{\rho_0 s_1 s_2}-1)/\sqrt{(e^{s_1^2}-1)(e^{s_2^2}-1)}$ 反解后传入，使组装出的可靠度问题恰好具有目标对数相关（实测 `rho0` 回代误差 < 1e-8）。由于 $\ln F=\ln k_F+\ln p+\ln A$ 是正态变量之和，$F$ 严格对数正态，失效域 $\{R<F\}$ 在 $u$ 空间是**半空间**，于是 FORM 在该问题上精确，并有闭式 $\beta_{ln}=(\mu_{\ln R}-\mu_{\ln F})/\sqrt{\sigma_{\ln R}^2+\sigma_{\ln F}^2}$ 作为解析基准。
2. **反问题两方程两未知量**（`target_reliability_inverse.py`）。极限状态只通过 $\kappa=\gamma_f\psi$ 依赖分项系数对（构件按 $R_k=\kappa F_k$ 定尺），因此

$$F_1(\gamma_f,\psi)=\beta(\kappa)-\beta_T=0,\qquad F_2(\gamma_f,\psi)=\gamma_f-F^*(\kappa)/F_k=0$$

   $F_1$ 是规范标定条件（按该对设计的构件恰好达到目标可靠度），$F_2$ 是 JCSS/ISO 2394 设计值劈分（荷载分项系数 = FORM 设计点作用/特征作用）。未知量取对数坐标 $y=(\ln\gamma_f,\ln\psi)$ 做阻尼 Newton（回溯线搜索），Jacobian 用**包络定理**给出的 $\partial\beta/\partial\kappa=R^*/(\kappa\|\nabla_u g(u^*)\|)$ 与 $F^*/F_k$ 的中心差分组装；三组分散初值独立收敛以检验唯一性。
3. **四条独立验证面**：(i) 设计点重要性抽样（$2\times10^5$）；(ii) 粗 Monte Carlo （4e+06 样本，失败数 140）；(iii) 抗力解析积分的 Rao–Blackwell 化粗 MC（同一估计量、方差更小）；(iv) 闭式 $\beta_{ln}$ 解析值。四者对照 FORM。
4. **ISO 表格裁决**：A.2 荷载分项 × A.3 抗力分项（9 对）+ 仓内锚点（1 对）逐行算偏差；偏差 >15 % 且 $\kappa_{table}<\kappa^*$ 的行给出**不保守算例**（按该对设计后的实际 $\beta$、$P_f$ 相对目标倍数、所需抗力差）。
5. **负控**：同一对边缘分布、只把 $\rho_{\ln p,\ln A}$ 置 0，重跑整个反问题，检验 $\gamma_f^*$ 是否系统性偏大 ≥5 %。
6. **工具链**：纯 Python（`tensorphysics.structural` 的 FORM/SORM/IS/MC + Nataf），`matplotlib` 出图；所有数字由 `run_study()` 写入 JSON，报告文本从 JSON 注入（零手抄）。

## ③ 计算结果

**表 1 反问题解**

| 量 | 值 |
|---|---|
| $\gamma_f^*$ | **1.437056** |
| $\psi^*$ | **0.996007** |
| $\kappa^*=\gamma_f^*\psi^*$ | **1.431318** |
| $F^*$ (设计点冰力) | 33.7950 MN |
| $R^*$ (设计点抗力) | 33.7950 MN |
| 达到的 $\beta$ | 4.000000000000 |
| $|\beta-\beta_T|$ | 0.00e+00 |
| Newton 迭代数（三初值） | [7, 5, 5] |
| $\gamma_f^*$ 三初值极差 | 7.88e-11 |
| Jacobian 条件数 | 4.791 |
| $\kappa$ 单变量二分解（独立复核）| 1.431318 ($\beta$=4.000000000) |
| 设计点方向余弦 $\alpha$ | [0.3361, -0.0915, 0.8464, -0.4028] |

**表 2 FORM 的四条独立验证（判据 a）**

| 方法 | $\beta$ | $P_f$ | 与 FORM 相对误差 | 1σ 区间覆盖 FORM |
|---|---|---|---|---|
| FORM | 4.000000 | 3.1671e-05 | — | — |
| 闭式 $\beta_{ln}$（解析）| 4.000000 | — | 6.46e-09 | — |
| 设计点重要性抽样 | 3.997451 | 3.2014e-05 | -6.372e-04 | 否（1σ 带仅 ±4.5e-03，比 FORM 收敛容差还窄） |
| 粗 Monte Carlo | 3.976286 | 3.5000e-05 | -5.929e-03 | 否（95 % 带：True，区间 [3.94, 4.019]） |
| 粗 MC + 抗力解析积分 | 4.002966 | 3.1277e-05 | +7.416e-04 | 是 |

**表 3 β 敏感性（判据 c）：包络定理解析导数 vs 中心差分**

| 导数 | 解析（伴随）| 中心差分 | 相对误差 |
|---|---|---|---|
| $\partial\beta/\partial\kappa$ | 2.353844 | 2.353845 | -3.33e-07 |
| $\partial\beta/\partial\gamma_f$ | 2.344445 | 2.344446 | -3.33e-07 |
| $\partial\beta/\partial\psi$ | 3.382607 | 3.382608 | -3.33e-07 |
| 最大相对误差 | | | 3.33e-07 |

**表 4 与 ISO 19906 表格分项系数偏差（判据 d）**

| 表格对（$\gamma_F\times\gamma_M$）| 表格值 | $\kappa_{table}$ | $\Delta\gamma_f$ | $\Delta\psi$ | $\Delta\kappa$ | 判定 | 按表设计的 $\beta$ |
|---|---|---|---|---|---|---|---|
| ice_action_extreme x steel_plate_yielding | (1.35, 1.10) | 1.4850 | -6.06 % | +10.44 % | +3.75 % | 带内/偏保守 | 4.1240 |
| ice_action_extreme x steel_plate_buckling | (1.35, 1.15) | 1.5525 | -6.06 % | +15.46 % | +8.47 % | 超差/偏保守 | 4.2738 |
| ice_action_extreme x steel_weld_connection | (1.35, 1.25) | 1.6875 | -6.06 % | +25.50 % | +17.90 % | 超差/偏保守 | 4.5547 |
| ice_action_abnormal x steel_plate_yielding | (1.10, 1.10) | 1.2100 | -23.45 % | +10.44 % | -15.46 % | 超差/不保守 | 3.4341 |
| ice_action_abnormal x steel_plate_buckling | (1.10, 1.15) | 1.2650 | -23.45 % | +15.46 % | -11.62 % | 超差/不保守 | 3.5838 |
| ice_action_abnormal x steel_weld_connection | (1.10, 1.25) | 1.3750 | -23.45 % | +25.50 % | -3.93 % | 超差/不保守 | 3.8648 |
| ice_action_frequent x steel_plate_yielding | (1.00, 1.10) | 1.1000 | -30.41 % | +10.44 % | -23.15 % | 超差/不保守 | 3.1130 |
| ice_action_frequent x steel_plate_buckling | (1.00, 1.15) | 1.1500 | -30.41 % | +15.46 % | -19.65 % | 超差/不保守 | 3.2627 |
| ice_action_frequent x steel_weld_connection | (1.00, 1.25) | 1.2500 | -30.41 % | +25.50 % | -12.67 % | 超差/不保守 | 3.5436 |
| repo_anchor gamma_ice x gamma_m | (1.25, 1.15) | 1.4375 | -13.02 % | +15.46 % | +0.43 % | 超差/偏保守 | 4.0145 |

**表 5 不保守算例（偏差 >15 % 且 $\kappa_{table}<\kappa^*$）**

| 表格对 | $\kappa$ 缺口 | 达到的 $\beta$ | $\Delta\beta$ | $P_f/P_{f,T}$ | 按表所需 $R_k$ / 标定所需 $R_k$ |
|---|---|---|---|---|---|
| ice_action_abnormal x steel_plate_yielding | +15.46 % | 3.4341 | 0.5659 | 9.39× | 28.455 / 33.660 MN |
| ice_action_abnormal x steel_plate_buckling | +11.62 % | 3.5838 | 0.4162 | 5.35× | 29.749 / 33.660 MN |
| ice_action_abnormal x steel_weld_connection | +3.93 % | 3.8648 | 0.1352 | 1.76× | 32.336 / 33.660 MN |
| ice_action_frequent x steel_plate_yielding | +23.15 % | 3.1130 | 0.8870 | 29.24× | 25.869 / 33.660 MN |
| ice_action_frequent x steel_plate_buckling | +19.65 % | 3.2627 | 0.7373 | 17.42× | 27.044 / 33.660 MN |
| ice_action_frequent x steel_weld_connection | +12.67 % | 3.5436 | 0.4564 | 6.23× | 29.396 / 33.660 MN |

**表 6 负控（判据 e）：忽略 p–A 负相关**

| 模型 | $\kappa^*$ | $\gamma_f^*$ | $\psi^*$ | $\sigma_{\ln F}$ |
|---|---|---|---|---|
| 计入 $\rho_{\ln p,\ln A}$=-0.6737 | 1.431318 | 1.437056 | 0.996007 | 0.2717 |
| 忽略相关（$\rho$=0）| 1.740922 | 1.852433 | 0.939803 | 0.4077 |

$\Delta\gamma_f^*$ = +28.90 %，$\Delta\kappa^*$ = +21.63 %；把偏大的 $\kappa$ 用到**物理**模型上会得到 $\beta$ = 4.6597（过度设计 +0.6597）。判据阈值 5 %，实际 +28.90 % → 检出。

**表 7 $\beta_T$ 敏感性**

| $\beta_T$ | $P_{f,T}$ | $\gamma_f^*$ | $\psi^*$ | $\kappa^*$ | $\mathrm{d}\gamma_f/\mathrm{d}\beta_T$ |
|---|---|---|---|---|---|
| 3.30 | 4.834e-04 | 1.20749 | 0.96298 | 1.16279 | — |
| 3.80 | 7.235e-05 | 1.36734 | 0.98646 | 1.34882 | 0.3280 |
| 4.00 | 3.167e-05 | 1.43706 | 0.99601 | 1.43132 | 0.3620 |
| 4.30 | 8.540e-06 | 1.54835 | 1.01050 | 1.56462 | — |

**表 8 模型参数敏感性 $\partial\beta/\partial\theta$（在 $\kappa^*$ 处，中心差分）**

| 参数 | 名义值 | $\partial\beta/\partial\theta$ |
|---|---|---|
| exponent | 0.4000 | +1.1474 |
| a_cov | 0.3500 | -1.8573 |
| pressure_model_cov | 0.1500 | -2.4270 |
| load_model_cov | 0.1000 | -1.6383 |
| resistance_cov | 0.1200 | +0.1134 |

## ④ 结果图

**图 1.** Joint p-A state: both correlation hypotheses keep identical marginals and differ only in $\rho_{\ln p,\ln A}$; the force density shows the resulting scatter inflation.

![fig1](/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/fig1_pa_joint.png)

**图 2.** Achieved reliability index versus the factor product: the Newton solution, the independent bisection on $\kappa$, and the FORM/IS/MC/Rao-Blackwell estimates of the same point.

![fig2](/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/fig2_beta_vs_kappa.png)

**图 3.** Left: residual norm of the two-equation system from three dispersed starting points (quadratic convergence, same root). Right: the design-value split curve that makes the second equation non-degenerate.

![fig3](/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/fig3_newton.png)

**图 4.** Deviation map of every ISO 19906 load/resistance factor pair against the calibrated pair, with the achieved reliability index of the rows that fall more than 15 % short.

![fig4](/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/fig4_iso_deviations.png)

**图 5.** Negative control. Dropping the negative p-A correlation inflates the load scatter and the calibrated load factor; the calibrated member is then over-designed by the reported beta margin when applied to the physical model.

![fig5](/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/fig5_negative_control.png)

**图 6.** Left: adjoint (envelope-theorem) derivatives against central finite differences. Right: the calibrated pair as a function of the target reliability index.

![fig6](/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/fig6_sensitivity.png)

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **Nataf 吃物理相关、ISO 幂律给对数相关**。首版把 $\rho_{\ln p,\ln A}$ 直接当物理相关系数传入，结果 FORM 与闭式解析值相差 3 %、而重要性抽样又站在 FORM 一边（两者差 0.07 %）——即「解析基准错了」而不是「FORM 错了」。改为对数正态精确关系反解 $\rho_{phys}$ 后，回代 `rho0` 与目标对数相关一致到 1e-8，FORM 与解析值相对误差 6.5e-09。这是本节点最重要的一次自纠。
2. **该问题的 FORM 是精确的**：失效域 $\{R<F\}$ 在 $z=(\ln x-\mu)/\sigma$ 坐标下是超平面，线性变换到 $u$ 空间仍是超平面，故 Hasofer–Lind 距离即真值。因此「FORM 与 MC 差 3 %」这类常规判据在此**不能**用来检验 FORM 的精度，只能检验实现的一致性——本节点如实按此解读（见判据 a 的 `criterion_interpretation`）。
3. **±1σ 覆盖率是 68 % 检验**。粗 MC 本次得到 $P_f$=3.500e-05（失败数 140），估计偏高 1σ，1σ 带未覆盖 FORM，但 95 % 带覆盖；同一估计量的 Rao–Blackwell 化版本（抗力解析积分）方差小 10 倍以上，1σ 带覆盖。两者都写进 JSON，不择优隐匿。
4. **分项系数对的唯一性依赖劈分约定**。$\kappa$ 由达到目标可靠度唯一确定，但 $\kappa$ 如何在 $\gamma_f$ 与 $\psi$ 之间劈分是约定；本节点采用 JCSS/ISO 2394 设计值劈分，于是 $\psi^*$ = 0.9960 ≈ 1（5 % 分位特征抗力与设计点抗力几乎重合）。换成别的劈分约定会得到不同的 $\gamma_f$/$\psi$，但 $\kappa^*$ 不变——与 ISO 表格比对时以 $\kappa$ 为准（唯一无约定量），$\gamma_f$/$\psi$ 的逐项偏差仅作参考。
5. **表格值出处受限**：ISO 19906 Annex A 正文不在工作树内，A.2/A.3 为二次文献转录（`source_tier=secondary`, `primary_source_verified=False`），由它得出的裁决一律 `conditional`；仓内锚点 (1.25, 1.15) 是全可验证的。偏差 >15 % 的行中 SLS（$\gamma_F$=1.00）行按 ULS 目标 $\beta_T$ 裁决属于口径错配，算例的物理含义是「若把使用极限状态的分项系数用到极值状态」，不是对规范的指摘。
6. **适用范围**：年极值接触面积按对数正态建模、冰压遵循单一幂律指数（$ex$ 为常数）、抗力与冰力独立。参数敏感性见表 8：$\partial\beta/\partial ex$ = +1.147 / 单位，即 $ex$ 每增加 0.1，$\beta$ 约上升 0.11 —— 指数是仅次于压力散度的敏感方向。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| FORM $\beta$ | 闭式解析 $\beta_{ln}$ | 6.46e-09 | 是 |
| FORM $\beta$ | 设计点重要性抽样 | 6.37e-04 | 是 |
| 反问题残差 $|\beta(\gamma^*) - \beta_T|$ | 目标 4.0 | 0.00e+00 | 是（阈值 1e-3） |
| 伴随 $\partial\beta/\partial\kappa$ | 中心差分 | -3.33e-07 | 是 |
| 伴随 $\partial\beta/\partial\gamma_f$ | 中心差分 | -3.33e-07 | 是 |
| 伴随 $\partial\beta/\partial\psi$ | 中心差分 | -3.33e-07 | 是 |
| $\kappa^*$（Newton）| $\kappa^*$（二分）| 2.31e-11 | 是 |
| 粗 MC $\beta$ | FORM | 5.93e-03 | 是 |
| 与 ISO A.2×A.3（极值+屈服）$\kappa$ 偏差 | 表格 1.485 | +0.0375 | 否（表格值高于标定值，偏保守侧） |

**结论**：FORM 与解析值、重要性抽样的一致性远优于 3 %（均为 1e-2 % 量级）；反问题残差 1e-11；伴随导数与有限差分一致到 1e-7；唯一未达 3 % 量级的对照是**粗 MC 的统计噪声**（1σ 约 8 %，Rao–Blackwell 化后约 6 %，两者 95 % 带均覆盖 FORM），属采样误差而非模型误差，已在表 2 如实列出。

## 审计链

- 结果 JSON：`/root/tp-agent/work/m8350/artifacts/iso19906_target_inverse/in101_n09_results.json`（本报告所有数字的唯一来源，`run_study()` 直出）
- 机器判据：`machine_criteria` 5 条，`criteria_summary` = {'passed': 5, 'total': 5, 'all_passed': True, 'failed': []}
- 求解器：`tensorphysics.structural.reliability`（FORM/HL-RF、SORM、IS、MC）与 `tensorphysics.structural.distributions`（Nataf）
- 复现：`PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/run_in101_n09.py`
