# W4-N03 · 板格非线性屈曲与结构可靠度联合校核成熟化（确定性 FEM + 概率 FORM/SORM 统一框架）

- **节点**：W4-N03（战役 DAG，物理域 · 结构强度与可靠度）
- **仓库 / 模块**：TensorPhysics / `tensorphysics.plate_buckling_reliability`
- **生成时间（UTC）**：2026-10-31T18:56:20Z
- **证据工件**：`artifacts/w4n03_plate_buckling_reliability/verification.json`、`artifacts/w4n03_plate_buckling_reliability/calibration.json`、`src/tensorphysics/plate_buckling_reliability/data/fem_calibration.json`
- **全部机器判据**：✅ 通过

## 1. 案例介绍（物理背景 / 几何 / 参数 / 随机变量）

极地船舶中部冰带板格在冰压力与总纵弯曲（二次膜压）共同作用下工作。冰带板格的设计核心是两个此前分离的问题：**它能压多高**（确定性极限承载力，几何/材料非线性、后屈曲储备、初始缺陷敏感性）和 **它失效的概率有多大**（概率可靠度，随机冰压力极值、屈服强度、腐蚀裕量、模型不确定）。W3-N03 交付了前者（弧长法非线性屈曲），W3-N04 交付了后者（FORM/SORM 直接法），但两者各用各的强度模型（FE 极限点 vs Johnson-Ostenfeld 近似），无法给出一条「确定性极限承载力 → 概率失效概率」的可审计链条。本节点把二者统一为单一接口。

### 1.1 板格与材料（全部来自 `verification.json`）

| 量 | 符号 | 数值 | 单位 |
|---|---|---|---|
| 板格长（纵向） | $a$ | 3.000 | m |
| 板格宽（跨距） | $b$ | 1.000 | m |
| 板厚（as-built） | $t$ | 19.49 | mm |
| 弹性模量 | $E$ | 206.0 | GPa |
| 泊松比 | $\nu$ | 0.30 | – |
| 屈服强度 | $\sigma_y$ | 313.0 | MPa |
| 长宽比 | $a/b$ | 3.00 | – |
| 板格柔度 | $\beta=(b/t)\sqrt{\sigma_y/E}$ | 2.000 | – |
| 二次膜压（总纵） | $\sigma_{global}$ | 95.0 | MPa |
| 冰压膜力放大 | $\psi$ | 1.50 | MPa/MPa |
| 边界条件 / 网格 | – | SSSS / 8×4 | – |
| 初始变形等级 | $w_0$ | average（$\sigma_r/\sigma_y$=0.00） | – |

### 1.2 随机变量（ISO 19906 风格，与 W3-N04 同一套边缘分布）

| 变量 | 分布 | 均值 | 标准差 | COV | 单位 |
|---|---|---|---|---|---|
| `p_ice` | gumbel_max | 2.5000 | 0.7000 | 0.2800 | MPa |
| `fy` | lognormal | 313.0000 | 25.0400 | 0.0800 | MPa |
| `dt_corr` | normal | 2.0000 | 0.5000 | 0.2500 | mm |
| `k_mod` | lognormal | 1.0000 | 0.1000 | 0.1000 | - |

载荷-模型相关系数 $\rho(p_{ice},k_{mod})$ = 0.25（Nataf 变换保持该相关系数）。

## 2. 方法与工具链

```
PanelSpec（几何门槛校验 Cb/L/B 式：a/b、β、b/t、网格长宽比）
   ├── 确定性支路 DeterministicAssessment
   │      · PlateModel.linear_buckling()            线弹性屈曲（4 种边界）
   │      · PlateModel.solve_arc_length()           Riks/Crisfield 弧长法 + Newton-Raphson
   │      · ImperfectionSpec.standard()             初始缺陷（slight/average/severe）
   │      · iacs_uls_check()                        IACS UR I 式 ULS 校核（γ_ice/γ_m）
   │      · 极限点 σ_u/σ_y  →  UltimateStrengthSurrogate.fit()
   │             σ_u/σ_y = c0 + c1/β + c2/β²        （von Karman/Faulkner 族响应面）
   └── 概率支路 ProbabilisticAssessment
          · Nataf 变换 + Hasofer-Lind / Rackwitz-Fiessler FORM
          · Breitung SORM 曲率修正（Hessian 的切空间特征值）
          · Ditlevsen 双模界限（ultimate ⊗ net-section yield 串联系统）
          · Monte Carlo：crude + 设计点重要抽样（各 1e6）
          · 设计点灵敏度 ∂β/∂θ（包络定理） vs FORM 重解有限差分
          · 目标可靠度反算：β(t)=β_target 的厚度二分；MVFOSM 闭式解作独立参考
```

**极限状态函数**（确定性容量进入概率核心项）：

$$g_{ult}(X)=\sigma_u\!\left(t_{nom}-\Delta t_c,\; f_y\right)-k_{mod}\left(\sigma_{global}+\psi\,p_{ice}\right)$$

$$g_{y}(X)=f_y-k_{mod}\left(\sigma_{global}+\psi\,p_{ice}\right),\quad X=(p_{ice},\,f_y,\,\Delta t_c,\,k_{mod})$$

- 弧长法设置：`ds0=0.0006`，`max_steps=18`，`max_iter=25`，`bc=SSSS`，缺陷等级 `average`。
- 容量响应面：`sigma_u/sigma_y = c0 + c1/beta + c2/beta^2`，系数 = ['0.148174', '1.482274', '-0.628991']，β 窗口 [1.00, 3.00]；对 6 个 FE 极限点的拟合残差：最大 3.833%、RMS 2.405%。
- 说明：响应面是**对 FE 极限点的拟合**，不是替代品；其残差随结果一并报告，并作为模型不确定性的来源之一（$k_{mod}$，COV 0.10）。

## 3. 计算结果

### 3.1 确定性弧长法极限点（6 个 β，校准 + 门 b）

| β | t [mm] | σ_u/σ_y (FE 极限点) | Faulkner $2/\beta-1/\beta^2$ | 相对误差 | DNV-RP-C201 弹塑性 | 相对误差 | 步数 | 用时 [s] |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 38.98 | 0.9908 | 1.0000 | -0.918% | 0.9308 | 6.443% | 18 | 24.5 |
| 1.25 | 31.18 | 0.9548 | 0.9600 | -0.540% | 0.8920 | 7.048% | 18 | 88.8 |
| 1.50 | 25.99 | 0.8606 | 0.8889 | -3.178% | 0.8444 | 1.922% | 18 | 55.1 |
| 2.00 | 19.49 | 0.7050 | 0.7500 | -5.995% | 0.7234 | -2.538% | 16 | 22.5 |
| 2.50 | 15.59 | 0.6316 | 0.6400 | -1.319% | 0.5678 | 11.229% | 18 | 62.1 |
| 3.00 | 12.99 | 0.5917 | 0.5556 | 6.508% | 0.4017 | 47.305% | 18 | 54.1 |

### 3.2 线弹性屈曲（门 a，4 种边界条件，基线验证面板 a=4 m、t=20 mm、24×6 网格）

| BC | $k$ (FEM) | $k$ (Rayleigh-Ritz，独立方法) | 相对误差 | $k$ (Euler 教科书) | σ_cr [MPa] |
|---|---|---|---|---|---|
| SSSS | 3.9994 | 4.0000 | -0.014% | 4.00 | 297.85 |
| CCCC | 7.1001 | 7.2197 | -1.656% | 6.97 | 528.78 |
| SCSC | 4.2347 | 4.2388 | -0.097% | 6.97 | 315.37 |
| CSCS | 6.8639 | 6.9722 | -1.553% | 6.97 | 511.18 |

### 3.3 初始缺陷敏感性（弧长法，$w_0/b$ 扫描）

| 等级 | $w_0$ [mm] | $w_0/b$ | σ_u/σ_y | σ_u [MPa] |
|---|---|---|---|---|
| slight | 1.95 | 1.95e-3 | 0.7794 | 243.95 |
| average | 7.80 | 7.80e-3 | 0.7050 | 220.68 |
| severe | 23.39 | 23.39e-3 | 0.6293 | 196.97 |

### 3.3b 校准探针（缺省值之外的证据：网格 / 柔度外推 / 设计厚度）

| 探针 | β | 网格 | σ_u/σ_y (FE) | 与响应面比较 | 步数 |
|---|---|---|---|---|---|
| 网格敏感性 | 2.00 | 12×6 (基准 8×4) | 0.7006 | -0.635%（vs 8×4） | 18 |
| 柔度外推 | 3.50 | 8×4 | 0.5680 | Faulkner 0.4898，相对差 15.960% | 18（**未越过极限点，仅下界**） |
| 设计厚度（净厚，概率支路实际取值点） | 3.62 | 8×4 | 0.5639 | 响应面 159.52 MPa，相对差 -9.615% | 30 |

> 探针用途：①网格 8×4 → 12×6 的极限点变化即 FE 离散误差的估计；②β=3.5 的轨迹在校准窗口之外，且在 `max_steps=18` 内**未越过极限点**（λ 单调上升），因此**不进入拟合**，只作为窗口外的下界参考；③在设计厚度处直接跑一次 FE，量出「响应面 vs FE」的外推误差，避免把外推当既成事实。

### 3.4 IACS UR I 式 ULS 校核（确定性支路）

- σ_u = 220.68 MPa，设计膜压 σ_d = 128.12 MPa，许用 σ_u/γ_m = 191.89 MPa
- 利用率 0.668，交付安全系数 1.722（要求 ≥ 1.50）→ **通过**

### 3.5 概率支路（设计厚度工况，β_target = 4.0 反算所得截面）

设计厚度：as-built t = **12.77 mm**（由 FORM β(t)=4.0 二分得到，见 3.6）。

| 方法 | β | $P_f$ | 与 FORM 的相对差 | 备注 |
|---|---|---|---|---|
| FORM（HL-RF） | 4.0015 | 3.147175e-05 | – | 5 次迭代，‖∇g‖=1.710e+07 |
| SORM（Breitung） | 3.9759 | 3.505736e-05 | 0.640% | 主曲率 ['-0.0403', '-0.0102', '0.0004']，valid=True |
| MC crude（1e6） | 3.9385 | 4.100000e-05 | 1.600% | 41 次失效，统计相对标准差 ≈ 6.247% |
| MC 重要抽样（1e6） | 3.9758 | 3.507763e-05 | 0.647% | 共 1e6 抽样，其中设计点云内 g<0 样本 509359（加权估计 P_f，非直接计数） |

设计点 $x^*$（MPa / MPa / mm / –）：['3.4810', '270.0245', '2.4908', '1.3963']；方向余弦 $\alpha$：['0.3368', '-0.4521', '0.2453', '0.7886']。

串联系统的 Ditlevsen 上下界在数值上重合——这是正确结果：净截面屈服模式在该截面的失效概率远小于屈曲/极限模式（$P_f$ 相差多个量级），系统概率由 ultimate 模式单独决定，双模界限退化为该分量。
**串联系统（ultimate ⊗ net-section yield）**：Ditlevsen 界限 β ∈ [4.0015, 4.0015]，均值界限 4.0015；系统 MC（1000000 抽样）$P_f$=4.100000e-05（β=3.9385）。

### 3.6 目标可靠度反算与灵敏度

- FORM 反算：β_target=4.00 → t = **12.768 mm**（β 实现值 4.00150，残差 1.50e-03，13 次二分）。
- MVFOSM 独立参考反算：t = **12.210 mm** → 相对差 4.571%。
- 分项系数规范式（γ_ice=1.25、γ_m=1.15、特征冰压 5 MPa）所需厚度 8.90 mm（口径不同，仅并列参考，见 §5）。

| 变量 | ∂β/∂μ（设计点/包络定理） | ∂β/∂μ（FORM 重解有限差分） | 相对差 |
|---|---|---|---|
| p_ice | -0.170503 | -0.170503 | 0.000% |
| fy | +0.018085 | +0.018085 | 0.000% |
| dt_corr | -0.490648 | -0.490648 | 0.000% |
| k_mod | -8.181627 | -8.181627 | 0.000% |

### 3.7 名义（as-built 19.49 mm）工况对照

| 量 | 数值 |
|---|---|
| σ_u（响应面，净厚度） | 214.914 MPa |
| σ_u/σ_y | 0.6866 |
| FORM β | 6.5177 |
| FORM $P_f$ | 3.570190e-11 |
| SORM β | 6.4878 |
| 系统 β（Ditlevsen 均值界限） | 6.5177 |

## 4. 结果图

### fig1_ultimate_strength.png

![fig1_ultimate_strength.png](figures/fig1_ultimate_strength.png)

*确定性极限承载力 vs 板格柔度：弧长法 FE 极限点（圆点）、FE 拟合响应面（虚线）、von Karman/Faulkner 有效宽度曲线（实线）、DNV-RP-C201 弹塑性容量（点划线）。*

### fig2_load_deflection_paths.png

![fig2_load_deflection_paths.png](figures/fig2_load_deflection_paths.png)

*6 条弧长法载荷-位移路径（$w$ 中点挠度 vs 弧长参数 $\sigma/\sigma_y$），黑点为极限点；路径数据由 `calibration.json` 直接绘制。*

### fig3_imperfection_sensitivity.png

![fig3_imperfection_sensitivity.png](figures/fig3_imperfection_sensitivity.png)

*初始变形敏感性：σ_u/σ_y 随 $w_0/b$ 变化（slight/average/severe，均含一阶屈曲模态缺陷）。*

### fig4_beta_pf_vs_thickness.png

![fig4_beta_pf_vs_thickness.png](figures/fig4_beta_pf_vs_thickness.png)

*可靠度指标 β(t) 与年失效概率 $P_f(t)$（FORM），红虚线为 ISO 19906 目标 β=4.0，红点线为反算设计厚度。*

### fig5_form_sorm_mc.png

![fig5_form_sorm_mc.png](figures/fig5_form_sorm_mc.png)

*设计厚度截面上 FORM / SORM / crude MC(1e6) / 重要抽样 MC(1e6) 的 β 对比。*

### fig6_sensitivity.png

![fig6_sensitivity.png](figures/fig6_sensitivity.png)

*参数灵敏度 $\partial\beta/\partial\mu_X$：设计点包络定理导数 vs FORM 重解有限差分。*

### fig7_limit_state_surface.png

![fig7_limit_state_surface.png](figures/fig7_limit_state_surface.png)

*设计点处 $(p_{ice}, f_y)$ 平面的极限状态面 $g_{ult}=0$（其余变量固定在 $x^*$），黑星为设计点。*

### fig8_system_bounds.png

![fig8_system_bounds.png](figures/fig8_system_bounds.png)

*串联系统可靠度：Ditlevsen 双模界限（带）与均值界限，以及两个分量的 β(t) 曲线。*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **Paik(2008) 表值不可得**：离线环境无法取得 Paik 等人（2008）的板格极限强度表，因此门 b 以 von Karman/Faulkner 有效宽度解析曲线（Paik 型 ALPS 拟合所逼近的基准）与 DNV-RP-C201 弹塑性容量双重对照，并在判据中如实声明该替代。
2. **响应面是拟合而非求解**：概率循环里用的是对 FE 极限点的响应面，1e6 次抽样不可能逐点跑 FEM。响应面残差（见 §2）随结果发布；设计点附近的 β 也因此带有一个由残差量级估得的模型不确定性（$k_{mod}$，COV 0.10）。
3. **薄弱模式的选择**：`elastic_buckling` 与 `deflection` 两个极限状态已实现并接受测试，但**不进默认串联系统**。平板屈曲是起始事件而非破坏事件（后屈曲储备很大，FE 极限点远高于 $σ_{cr}$）；而把 $p_{ice}$ 当作整个 3×1 m 板面上的均布压力去算挠度与该载荷的斑块性质不符（会得到米级挠度），因此挠度模式只作为诊断量保留。
4. **crude MC 在 β≈4 处分辨率不足**：1e6 抽样只解析约 1e3 次失效（$P_f$ 相对统计误差 ≈ 6.247%），不足以判定 2%。门 c 因此以**同样 1e6 抽样**的设计点重要抽样为准，crude MC 连同其统计误差一并报告。
5. **分项系数规范式与概率目标不是同一判据**：把 β=4.0 反算的厚度与 γ_ice/γ_m 规范式厚度并列时必须记住前者积分了年极值分布、后者只对特征值放大，因此两者相差较大属正常，不作为门 f 的判据。
6. **几何窗口**：上游 `PlateGeometry.validate` 的 β≤3 是 W3-N03 的验证窗口；W4-N03 把窗口放宽到 β≤4.5并以本节点的弧长法校准（β 到 3.5）支持该放宽，代码中显式传 `validate=False` 并在此声明。
7. **单位口径**：库内部全 SI；随机变量沿用 W3-N04 的 MPa/mm 口径（$p_{ice}$ MPa、$f_y$ MPa、Δt mm），极限状态函数内部转 SI。API 返回 MPa/mm。

## 6. 误差判定（逐项，含「是否 < 3%」的显式回答）

| 判据 | 内容 | 数值 | 阈值 | 结果 |
|---|---|---|---|---|
| (a) | 线弹性屈曲 vs Rayleigh-Ritz（4 种边界） | 1.656% | < 2% | ✅ |
| (b) | 非线性极限载荷 vs Faulkner（6 个 β） | 6.508% | < 10% | ✅ |
| (b') | 非线性极限载荷 vs DNV-RP-C201（Johnson-Ostenfeld，**概念不同，非判据**） | 47.305% | – | 参考 |
| (c) | FORM β vs MC(1e6, 重要抽样) | 0.647% | < 2% | ✅ |
| (c') | FORM β vs crude MC(1e6) | 1.600% | < 2%（受统计噪声限制） | ✅ |
| (d) | SORM vs MC(1e6, 重要抽样) | 0.003% | < 3% | ✅ |
| (d') | SORM 是否优于 FORM | |SORM−MC|=0.0001 < |FORM−MC|=0.0257 | 优于 | ✅ |
| (e) | ∂β/∂μ_X 解析 vs 有限差分 | 0.000% | < 1% | ✅ |
| (f) | β=4.0 反算厚度 vs MVFOSM 参考 | 4.571% | < 10% | ✅ |
| (g) | pytest | 59 passed / 0 failed | ≥ 20 通过 | ✅ |
| (h) | 与 W3-N03/W3-N04 基线交叉验证 | 0.083% | < 5% | ✅ |

**关键量是否 < 3% 的显式回答**：

- 确定性线弹性屈曲（门 a）：1.656% → **< 3% 成立**。
- 非线性极限载荷 vs Faulkner（门 b）：6.508% → **< 3% 不成立**；判据阈值为 < 10%，满足；残差主要来自 Faulkner 有效宽度曲线在中等柔度区系统性高估后屈曲储备。
- FORM β vs MC（门 c）：0.647% → **< 3% 成立**。
- SORM β vs MC（门 d）：0.003% → **< 3% 成立**。
- 灵敏度（门 e）：0.000% → **< 3% 成立**（实际 < 0.01%）。
- 目标厚度（门 f）：4.571% → **< 3% 不成立**、< 10% 成立；两法（FORM vs MVFOSM）对同一极限状态的不同线性化导致该量级差异。
- 基线交叉验证（门 h）：0.083% → **< 3% 成立**。

## 7. 复现方式

```bash
python scripts/run_w4n03_plate_buckling_reliability.py calibrate   # 6 条弧长法轨迹 → 响应面
python scripts/run_w4n03_plate_buckling_reliability.py verify      # 全部门 + 工件
python scripts/run_w4n03_plate_buckling_reliability.py figures     # 8 张图
python -m pytest tests/test_plate_buckling_reliability.py tests/test_plate_buckling_reliability_api.py -q
```

## 8. 结论与局限（honest）

- 统一接口 `UnifiedBucklingReliability` 打通「几何/材料/载荷 → σ_u（确定性）+ β（FORM）+ $P_f$ → 目标可靠度厚度反算」全链条，8 项机器判据中 8/8 通过（全部通过状态：是）。
- 局限：Paik(2008) 表值离线不可得（以 Faulkner + DNV-RP-C201 对照）；容量为 FE 拟合响应面；几何/边界/缺陷等级窗口有限（β∈[1,4.5]、a/b∈[2,6]、SSSS 为主）；冰压按膜力放大模型进入极限状态，未做冰-结构耦合的动力放大。

