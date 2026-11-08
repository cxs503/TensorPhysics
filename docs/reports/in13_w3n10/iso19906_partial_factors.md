# IN13-W3-N10 — ISO 19906 冰压设计分项系数的目标可靠度反标定（$\beta_T=4.0$）

**节点**: IN13-W3-N10 ｜ **仓库**: TensorPhysics ｜ **模块**: `tensorphysics.ice_partial_factors` ｜ **生成时间(UTC)**: 2026-11-08T09:30:40+00:00

**总体判据**: 11/11 项机器判据通过（skipped=0），于 `artifacts/iso19906_partial_factors/calibration_results.json` 程序化读取。

## 1 案例介绍

### 1.1 物理背景

极地结构冰带板（ice belt）承受海冰挤压，设计格式为 $\gamma_F S_k \le R_k/\gamma_M$（ISO 19906 / Polar Code 分项系数格式，本仓库 `tensorphysics.ice_plasticity.iso19906` 采用同一格式）。$S_k$ 为特征冰载荷（年超越概率 $10^{-2}$，即 100 年重现期），$R_k$ 为特征抗力（5% 分位）。本节点回答一个规范编制层面的问题：**给定 $\beta_T=4.0$（年失效概率 $3.17\times10^{-5}$）与载荷/抗力的统计模型，分项系数 $\gamma_F$、$\gamma_M$ 应取多少？与 ISO 19906 表 A.1/A.2/A.3 的一致性如何裁决？**

### 1.2 几何/材料/模型参数（程序化）

| 参数 | 含义 | 取值 |
|---|---|---|
| $b$ | 横向加强筋间距（板弯曲跨距） | 350.0 mm |
| $t_{nom}$ | 冰带板厚（初始基准） | 26.0 mm |
| $\Delta t$ | 腐蚀/磨蚀裕量（正态，均值/COV） | 2.0 mm / 0.25 |
| $\alpha_b$ | 板弯曲系数 | 0.500 |
| $f_y$ | 屈服强度（对数正态） | 355 MPa / COV 0.08 |
| $k_S$ | 载荷模型不确定性（对数正态，均值 1） | COV 0.15 |
| $k_R$ | 抗力模型不确定性（对数正态，均值 1） | COV 0.10 |
| $A$ | 参考接触面积 | 5.0 m$^2$ |
| $\beta_T$ | 目标年可靠度指标 | 4.00 |
| $P_f$ | 目标年失效概率 $\Phi(-\beta_T)$ | 3.167124e-05 |

### 1.3 p-A 极值统计输入

ISO 19906 压力-面积律 $p(A)=p_0 (A/A_0)^{-ex}$ 描述压力中位趋势；设计作用量是**年最大值**，必须用极值分布描述。本仓库不携带实测冰压记录，因此年最大值由**文档化的生成模型**（风暴峰值 Gumbel + 面积散差对数正态 + 每年 12 次独立风暴）仿真，生成模型的参数即拟合的验证靶：

| 量 | 生成模型（plant） | 拟合（数据驱动，进入标定） |
|---|---|---|
| $p_0$ @1 m$^2$（事件级趋势） | 2.500 MPa | — |
| $p_0$（年最大值趋势，log-log 拟合） | 3.566 MPa | 3.595 MPa |
| 指数 $ex$ | 0.400 | 0.3991 ± 0.0023 |
| 年最大值 Gumbel $loc$ | 1.7430 MPa | 1.7655 MPa |
| 年最大值 Gumbel $scale$ | 0.2253 MPa | 0.2220 MPa |
| 年最大值 COV | 0.154 | 0.150 |
| GEV 形状 $\xi$（L 矩 / MLE） | 0（Gumbel） | 0.019 / -0.010 |
| KS 距离（MLE 拟合 vs 样本） | — | 0.0521 |

样本量：300 年 × 12 次风暴，面积网格 [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0] m$^2$，随机种子 20261031（可复现）。Gumbel 与 GEV 的似然比统计量 0.051 < 3.841（$\chi^2_1$ 5% 分位）⇒ Gumbel 限制不被拒绝。

## 2 方法

① **极限状态**（与 W3-N04 冰带链条一致的屈服型板极限状态）：

$$S(X)=k_S\,\alpha_b\,p\left(\frac{b}{t_{net}}\right)^2,\qquad R(X)=k_R f_y,\qquad g(X)=R-S$$

随机向量 $X=(p,f_y,\Delta t,k_S,k_R)$，经 `tensorphysics.structural` 的 Nataf 变换（$p$ 与 $k_S$ 相关性 0.25）进入 Hasofer-Lind/Rackwitz-Fiessler FORM、Breitung SORM、设计点重要性抽样与粗 Monte Carlo。

② **反问题（route A：规范标定逆问题）**：按格式设计的结构满足 $R_k=k S_k$（$k=\gamma_F\gamma_M$），由此反解设计板厚 $t_{nom}(k)=b\sqrt{\alpha_b p_k k/R_k}+\Delta t$；$\beta(k)$ 单调递增，用**二分法**求 $k^*$ 使 $\beta(k^*)=\beta_T$。

③ **分项系数拆分（route B：JCSS/ISO 2394 设计值法）**：在 $k^*$ 对应结构上取 FORM 设计点 $s^*=r^*$，令 $\gamma_F=s^*/S_k$、$\gamma_M=R_k/s^*$，则 $\gamma_F\gamma_M=R_k/S_k=k^*$ 恒成立（格式在设计点处取等号）。

④ **一致性裁决**：对 ISO 19906 表 A.1（目标可靠度）、A.2（载荷系数）、A.3（抗力系数）逐条裁决；因子对另经**达成可靠度** $\beta_{achieved}(\gamma_F\gamma_M)$ 判定（唯一与拆分约定无关的量是乘积）。

## 3 计算结果

### 3.1 反标定主结果（$\beta_T=4.0$）

| 量 | 数值 | 说明 |
|---|---|---|
| $S_k$（设计板厚处，100 年） | 162.41 MPa | 特征载荷 |
| $R_k$（5% 分位） | 285.49 MPa | 特征抗力 |
| $k^*=\gamma_F\gamma_M$ | 1.7578 | 二分法解（22 次迭代） |
| $t_{design}$ | 34.42 mm | 满足格式的最小板厚 |
| $s^*=r^*$（设计点） | 288.93 MPa | FORM 设计点载荷效应 |
| $\gamma_F$ | 1.7790 | 设计值拆分 |
| $\gamma_M$ | 0.9881 | 设计值拆分 |
| $\gamma_F\gamma_M$ | 1.757789 | 与 $k^*$ 一致（相对误差 1.26e-16） |
| $\beta_{achieved}$ | 4.000000 | 目标 4.00 |
| $\alpha$（$p,f_y,\Delta t,k_S,k_R$） | 0.807, -0.242, 0.094, 0.436, -0.302 | 设计点方向余弦 |

固定 $\gamma_M$ 时的 $\gamma_F$ 反推（工程上材料系数常由材料规范给定）：

| $\gamma_M$（固定） | 所需 $\gamma_F=k^*/\gamma_M$ |
|---|---|
| 1.00 | 1.758 |
| 1.10 | 1.598 |
| 1.15 | 1.529 |

### 3.2 可靠度方法交叉校核（设计点处）

| 方法 | $\beta$ | $P_f$ | 与 FORM 之差 |
|---|---|---|---|
| FORM | 4.0000 | 3.167e-05 | — |
| SORM（Breitung） | 3.9948 | 3.237e-05 | -0.0052 |
| 设计点重要性抽样（200000 样本） | 3.9939 | 3.250e-05 | -0.0061 |
| 粗 Monte Carlo（400000 样本） | 3.9298 | 4.250e-05 | -0.0702 |

主曲率：-0.0093, -0.0014, -0.0000, 0.0000；IS 相对标准差 0.45%。

### 3.3 目标可靠度扫描与面积情景

| $\beta_T$ | $P_{f,T}$ | $k^*$ | $\gamma_F$ | $\gamma_M$ | $t_{design}$ [mm] |
|---|---|---|---|---|---|
| 3.30 | 4.834e-04 | 1.3995 | 1.462 | 0.957 | 30.93 |
| 3.80 | 7.235e-05 | 1.6463 | 1.681 | 0.979 | 33.37 |
| 4.00 | 3.167e-05 | 1.7578 | 1.779 | 0.988 | 34.42 |
| 4.30 | 8.540e-06 | 1.9402 | 1.936 | 1.002 | 36.06 |

| $A$ [m$^2$] | $p_k$ [MPa] | $k^*$ | $\gamma_F$ | $\gamma_M$ |
|---|---|---|---|---|
| 2.5 | 3.672 | 1.7588 | 1.781 | 0.987 |
| 5.0 | 2.787 | 1.7578 | 1.779 | 0.988 |
| 10.0 | 2.200 | 1.7766 | 1.807 | 0.983 |
| 20.0 | 1.632 | 1.7819 | 1.813 | 0.983 |

$\gamma_F$ 在面积族上的相对散布 +1.89%（判据 C10，容差 +10%）。

### 3.4 与 ISO 19906 表 A.1 的裁决

| 表 | 行 | 表值 $\beta_T$ | 标定 $\beta$ | 相对偏差 | 裁决 | 证据等级 |
|---|---|---|---|---|---|---|
| A.1 | consequence_class_S1 | 3.30 | 4.00 | +21.21% | CONSERVATIVE | conditional |
| A.1 | consequence_class_S2 | 3.80 | 4.00 | +5.26% | CONSERVATIVE | conditional |
| A.1 | consequence_class_S3 | 4.00 | 4.00 | +0.00% | CONSISTENT | conditional |
| A.1 | consequence_class_S4 | 4.30 | 4.00 | -6.98% | MINOR_UNCONSERVATIVE | conditional |

### 3.5 与 ISO 19906 表 A.2/A.3 的逐条裁决

| 表 | 行 | 量 | 表值 | 反标定值 | 相对偏差 | 裁决 | 证据等级 |
|---|---|---|---|---|---|---|---|
| A.2 | ice_action_extreme | gamma_F | 1.350 | 1.779 | +31.78% | NON_CONSERVATIVE | conditional |
| A.2 | ice_action_abnormal | gamma_F | 1.100 | 1.779 | +61.73% | NON_CONSERVATIVE | conditional |
| A.2 | ice_action_frequent | gamma_F | 1.000 | 1.779 | +77.90% | NON_CONSERVATIVE | conditional |
| A.3 | steel_plate_yielding | gamma_M | 1.100 | 0.988 | -10.17% | CONSERVATIVE | conditional |
| A.3 | steel_plate_buckling | gamma_M | 1.150 | 0.988 | -14.08% | CONSERVATIVE | conditional |
| A.3 | steel_weld_connection | gamma_M | 1.250 | 0.988 | -20.95% | CONSERVATIVE | conditional |
| repo:ice_plasticity.uls_check | gamma_ice | gamma_F | 1.250 | 1.779 | +42.32% | NON_CONSERVATIVE | real |
| repo:ice_plasticity.uls_check | gamma_m | gamma_M | 1.150 | 0.988 | -14.08% | CONSERVATIVE | real |

### 3.6 设计组合的达成可靠度裁决

| 组合 | $\gamma_F$ | $\gamma_M$ | 乘积 | 目标 $\beta_T$ | 达成 $\beta$ | $\Delta\beta$ | 裁决 |
|---|---|---|---|---|---|---|---|
| iso_extreme_yield_high | 1.35 | 1.10 | 1.485 | 4.00 | 3.483 | -0.517 | NON_CONSERVATIVE |
| iso_extreme_buckling_high | 1.35 | 1.15 | 1.552 | 4.00 | 3.620 | -0.380 | NON_CONSERVATIVE |
| iso_extreme_yield_normal | 1.35 | 1.10 | 1.485 | 3.80 | 3.483 | -0.317 | NON_CONSERVATIVE |
| iso_abnormal_yield_high | 1.10 | 1.10 | 1.210 | 4.00 | 2.844 | -1.156 | NON_CONSERVATIVE |
| repo_anchor_yield_high | 1.25 | 1.15 | 1.438 | 4.00 | 3.383 | -0.617 | NON_CONSERVATIVE |

裁决计数：{"CONSERVATIVE": 6, "CONSISTENT": 1, "MINOR_UNCONSERVATIVE": 1, "NON_CONSERVATIVE": 9}（共 17 行）。

### 3.7 反问题：规范因子对对应什么特征值约定

| 目标乘积（规范组合） | 需要多长重现期才对得上 |
|---|---|
| iso_table_pair_yield = 1.4850 | 999 年 |
| iso_table_pair_buckling = 1.5525 | 535 年 |
| repo_anchor_pair = 1.4375 | 扫描范围外（不需要这么长的重现期） |

| 重现期 [年] | $p_k$ [MPa] | $S_k$ [MPa] | $k^*$ | $\gamma_F$ |
|---|---|---|---|---|
| 50 | 2.632 | 279.9 | 1.8613 | 1.884 |
| 100 | 2.787 | 296.3 | 1.7578 | 1.779 |
| 200 | 2.941 | 312.8 | 1.6655 | 1.686 |
| 500 | 3.145 | 334.4 | 1.5576 | 1.576 |
| 1000 | 3.299 | 350.8 | 1.4849 | 1.503 |

| 年最大值 COV | $k^*$ | $\gamma_F$ |
|---|---|---|
| 0.08 | 1.6689 | 1.611 |
| 0.12 | 1.7155 | 1.710 |
| 0.16 | 1.7714 | 1.800 |
| 0.20 | 1.8259 | 1.879 |
| 0.24 | 1.8762 | 1.947 |
| 0.28 | 1.9217 | 2.008 |

## 4 结果图

![p-A trend: simulated annual-max means against the ISO 19906 power law](../../../artifacts/iso19906_partial_factors/fig1_pA_trend.png)

*p-A trend: simulated annual-max means against the ISO 19906 power law*

![return-level plot of the annual-maximum ice pressure with Gumbel/GEV fits](../../../artifacts/iso19906_partial_factors/fig2_return_levels.png)

*return-level plot of the annual-maximum ice pressure with Gumbel/GEV fits*

![achieved reliability index against the factor product with the bisection solution](../../../artifacts/iso19906_partial_factors/fig3_beta_vs_factor.png)

*achieved reliability index against the factor product with the bisection solution*

![calibrated partial factors against the target reliability index](../../../artifacts/iso19906_partial_factors/fig4_factors_vs_beta.png)

*calibrated partial factors against the target reliability index*

![ISO 19906 table adjudication: factor deviations and achieved reliability per design case](../../../artifacts/iso19906_partial_factors/fig5_iso_adjudication.png)

*ISO 19906 table adjudication: factor deviations and achieved reliability per design case*

![inversion of the required factor product over the characteristic-value convention and load COV](../../../artifacts/iso19906_partial_factors/fig6_inversions.png)

*inversion of the required factor product over the characteristic-value convention and load COV*

## 5 注意点与局限

1. **p-A 数据来源**：仓库不含实测冰压记录，年最大值来自文档化生成模型（种子固定、可复现）。因此本文的 Gumbel/GEV 拟合是「参数回收」验证（plant → 拟合 → 与 plant 比对），不是对实测极值分布的标定；接真实观测数据只需替换 `simulate_annual_maxima` 的输入。
2. **ISO 表格文本**：工作树内无 ISO 19906:2010 正文，表 A.1/A.2/A.3 的数值取自二手文献概述，每行标注 `primary_source_verified = False`、`source_tier = secondary`，其裁决一律打 `evidence_tier = conditional`；唯一「实」锚点是仓库内 `ice_plasticity` 默认因子对（1.25, 1.15），裁决为 `real`。
3. **拆分约定的非唯一性**：一个方程两个未知数，$\gamma_F\gamma_M$ 唯一而拆分不唯一。设计值法给出的 $\gamma_M=0.988<1$ 反映该情景由载荷不确定性主导（$|\alpha_S|$ 显著大于 $|\alpha_R|$）；工程上若材料系数由材料规范锁定，应改用 3.1 节的 $\gamma_F=k^*/\gamma_M$ 反推表。
4. **特征值约定的支配作用**：$k^*$ 对特征载荷的重现期极其敏感（100 年 $\to$ 1000 年使 $k^*$ 从 1.758 降到 1.485）。因此「规范表格与反标定不一致」首先应归因于特征值定义差异，而非直接判定规范不安全。
5. **模型不确定性预算**：$k_S$（COV 0.15）在 $\beta_T=4$ 的设计点取 1.4 量级，它把 $\gamma_F$ 抬高约 30%。若把 $k_S$ 视为已在特征载荷中计入，$\gamma_F$ 会显著下降——这是与规范表格对比时最大的单点敏感性。
6. **极限状态范围**：仅覆盖冰带板屈服的单一失效模式；屈曲/疲劳/焊接等其他模式的抗力统计不同，需各自标定。

## 6 误差判定（是否 <3%）

| 关键量 | 基准 | 本文结果 | 相对误差 | <3%? |
|---|---|---|---|---|
| $\Phi(-4)$ 年失效概率 | 解析 3.167124e-05 | 3.167124e-05 | 0.00e+00 | 是 |
| $p_0$（年最大值趋势） | plant 3.566 MPa | 3.595 MPa | +0.83% | 是 |
| 指数 $ex$ | plant 0.400 | 0.3991 | -0.23% | 是 |
| Gumbel $loc$ | plant 1.7430 MPa | 1.7655 MPa | +1.29% | 是 |
| Gumbel $scale$ | plant 0.2253 MPa | 0.2220 MPa | -1.44% | 是 |
| FORM vs 重要性抽样 $\beta$ | IS 3.9939 | FORM 4.0000 | +0.153% | 是 |
| FORM vs SORM $\beta$ | SORM 3.9948 | FORM 4.0000 | +0.130% | 是 |
| 反问题闭合 $\beta(k^*)$ | 目标 4.0 | 4.000000 | +0.00001% | 是 |
| 格式闭合 $\gamma_F\gamma_M$ vs $k^*$ | $k^*$=1.757789 | 1.757789 | 1.26e-16 | 是 |

**无解析基准的量**：$\gamma_F$、$\gamma_M$ 本身（规范表值为二手文献转录，属条件性对比而非基准值），故不对其声明 <3%；其「误差」以与表格/锚点因子的相对偏差与达成 $\beta$ 呈现（3.5/3.6 节）。粗 Monte Carlo 在 $P_f\sim3\times10^{-5}$ 下仅数十个失效样本，其 $\beta$ 偏差+1.75% 属抽样噪声，不计入精度声明。

## 7 机器判据明细

| 判据 | 结果 | 关键数字 |
|---|---|---|
| C1_target_pf_mapping | PASS | beta_target=4; pf_target=3.16712e-05; rel_error_vs_Phi_minus_4=0 |
| C2_pA_trend_recovery | PASS | p_ref_fit=3.59507; p_ref_plant_event_level=2.5; p_ref_plant_annual=3.56561; rel_error_p_ref=0.00826165 |
| C3_gumbel_parameter_recovery | PASS | loc_mle=1.76547; loc_moments=1.76656; loc_plant=1.74301; scale_mle=0.222014 |
| C4_gev_gumbel_shape_tenable | PASS | gev_shape_lmoments=0.0189473; gev_shape_mle=-0.010126; loglik_gev_mle=-23.0444; loglik_gumbel_mle=-23.07 |
| C5_form_vs_importance_sampling | PASS | beta_form=4; beta_is=3.99389; abs_delta_beta=0.00610732; is_cov=0.00449051 |
| C6_form_vs_sorm_curvature | PASS | beta_form=4; beta_sorm=3.99482; abs_delta_beta=0.00518349; principal_curvatures=[-0.009345936632478728, -0.0014161486279843608, -4.798805914438036e-10, 8.422967297074419e-11] |
| C7_form_vs_monte_carlo | PASS | beta_form=4; beta_mc=3.92985; pf_mc=4.25e-05; samples=400000 |
| C8_format_closure | PASS | gamma_F=1.77898; gamma_M=0.988086; product=1.75779; k_star=1.75779 |
| C9_beta_target_inverse_solved | PASS | iterations=22; converged=True; g_design=-5.97993e-11; beta_achieved=4 |
| C10_area_scenario_stability | PASS | gamma_F_min=1.77898; gamma_F_max=1.8129; relative_spread=0.0188949; areas=[2.5, 5.0, 10.0, 20.0] |
| C11_iso_rows_all_adjudicated | PASS | rows=17; counts={'CONSERVATIVE': 6, 'CONSISTENT': 1, 'MINOR_UNCONSERVATIVE': 1, 'NON_CONSERVATIVE': 9} |

## 8 复现与产物

```bash
PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/run_in13_w3n10.py
python3 -m pytest tests/test_ice_partial_factors.py -q
```

- 数值结果：`artifacts/iso19906_partial_factors/calibration_results.json`
- 图：`artifacts/iso19906_partial_factors/fig*.png`
- 模块：`src/tensorphysics/ice_partial_factors/`（pressure_area / limit_state / calibration / iso_tables / api）
- 测试：`tests/test_ice_partial_factors.py`
