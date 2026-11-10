# 设计海况统计口径验证面：IFORM 环境等值线 × 目标可靠度 β × 超越概率 × Rayleigh 众值

> 节点 p229w4-n05-design-sea-state · 仓库 TensorPhysics · 模块
> `src/tensorphysics/design_sea_state/` · 本文件与全部数字由
> `benchmarks/design_sea_state/run_ifrom_study.py` **程序化生成**（零手抄）。

## ① 案例介绍（物理背景 / 参数 / 几何）

设计海况的给出方式在工程实践中有三套互不兼容的"口径"，同一片海域、同一个
"100 年一遇"目标，三套口径给出的设计海况可以相差数十个百分点：

1. **IFORM 环境等值线**（Winterstein 等 / DNV-RP-C205 App.3）：把联合
   $(H_s,T_z)$ 分布经 Rosenblatt 变换映到独立标准正态空间，取半径 $\beta$
   的圆再映回物理空间。
2. **目标可靠度指标 $\beta$**：结构可靠度侧的 $\beta$ 与超越概率 $p_f$ 的映射。
3. **Rayleigh 短时口径**：给定 $H_s$ 的单个波高 Rayleigh 分布，用其**众值**
   或 $N$ 波极值众值给出设计波高。

本节点把三者放在同一个可机器裁决的面上，用解析锚 + 精确长期积分做互裁。

**场地与联合分布模型**（两段式，Haver-Nyhus 型）：

| 分量 | 形式 | 参数 |
|---|---|---|
| $H_s$ 边际 | 三参数 Weibull | $k=1.6$, $\alpha=2.6$ m, $\gamma=0.6$ m |
| $T_z \mid H_s$ | 对数正态，均值幂律 $a_1+a_2h^{a_3}$ | $a_1=3.6$, $a_2=1.0$, $a_3=0.7$, $\sigma_{\ln}=0.08$ |
| 目标 | 年超越概率 | $p_f=0.01$ /年（重现期 100 年） |

由此 $H_s$ 的百年一遇值 $F^{-1}(0.99)=7.353$ m，
等值线半径 $\beta_F=2.3263$（convention F）、
$\beta_C=3.0349$（convention C）。

## ② 怎么做的（方法 / 流程 / 工具链）

1. `rosenblatt.py`：闭式 Rosenblatt 变换 $u_1=\Phi^{-1}(F_{H_s}(h))$，
   $u_2=\Phi^{-1}(F_{T_z|H_s}(t|h))$。变换正确性由两条独立恒等式把守：
   往返恒等（$\le 10^{-12}$）与密度/雅可比恒等式
   $\varphi_2(u)\,|\det\,\partial u/\partial x| = f(h,t)$。
2. `iform.py`：$u_1=\beta\cos\theta$、$u_2=\beta\sin\theta$ 逆向映回物理空间；
   两个解析锚：**极值 $H_s$** $=\,F_H^{-1}(\Phi(\beta))$（因 $u_1\le\beta$，$\theta=0$ 取等），
   以及**相关正态锚**下等值线 = 卡方椭圆（二者同为联合密度等值集，实测残差
   1e-15）。
3. `reliability_index.py`：两种 $\beta$ 口径显式并存 ——
   convention F：$p_f=\Phi(-\beta)$；convention C：$p_{out}=\exp(-\beta^2/2)$。
   附重现期、25 年寿命期换算、DNV/ISO 2394 目标可靠度行。
4. `rayleigh_consistency.py`：R1 众值 $=H_s/2$（Rayleigh 尺度 $a=2\sqrt{m_0}=H_s/2$）；
   R2 的 $N$ 波极值众值 $H_s\sqrt{\ln N/2}$；$4\times10^6$ 波 Monte-Carlo 审计；
   并与**精确长期积分**（$u$ 空间 2D Gauss-Legendre，25600 节点，
   解 $P_{ss}(R>r)=p_f$）三方互裁。
5. `api.py`：一次 `run_study()` 产出全部数字与 13 条判据行（PASS/FAIL + 实测值）。

工具链：Python 3.11 / numpy 2.4 / scipy 1.16 / matplotlib 3.10；
`PYTHONPATH=src pytest tests/test_design_sea_state.py`（26 用例）。

## ③ 计算结果（数字表）

### 3.1 两种 β 口径的鸿沟（本节点的核心"口径"结论）

| $p_f$ /年 | 重现期 | $\beta_{\rm Cornell}$ | $\beta_{\rm contour}$ | 以 Cornell 半径计的等值线概率含量 | 以 contour 半径计的 $\Phi(-\beta)$ | 概率含量比 |
|---|---|---|---|---|---|---|
| 0.01 | 100 | 2.326 | 3.035 | 0.06681 | 0.0012 | 6.68 |
| 0.001 | 1000 | 3.090 | 3.717 | 0.00844 | 0.000101 | 8.44 |
| 0.0001 | 10000 | 3.719 | 4.292 | 0.0009922 | 8.86e-06 | 9.92 |

同一"$p_f=0.01$"，convention C 的半径 $\beta_C=3.0349$ 使等值线
**概率含量**恰为 $p_f$，而 convention F 的半径 $\beta_F=2.3263$ 才使
**响应超越概率**为 $p_f$。二者不可混用：worst case 的概率含量比达
9.92（$p_f=10^{-4}$ 时，$|\ln|=2.29$）。

### 3.2 目标可靠度映射（DNV/ISO 2394 行复现）

| 后果等级 | $\beta$ 参考 | $p_f$ 参考 | 由 $p_f$ 反算 $\beta$ | 相对误差 |
|---|---|---|---|---|
| CC1_minor | 3.30 | 0.000483 | 3.3002 | 0.007 % |
| CC2_normal | 3.70 | 0.000108 | 3.6995 | 0.013 % |
| CC3_large | 4.20 | 1.33e-05 | 4.2008 | 0.019 % |

### 3.3 超越概率校准（Monte-Carlo，物理分支抽样 + 多边形内含判定）

| $\beta$ | 目标 $p_{out}=\exp(-\beta^2/2)$ | 实测出线概率 | 相对误差 | z 分数 | 样本数 |
|---|---|---|---|---|---|
| 2.00 | 0.13534 | 0.13587 | 0.39 % | +0.69 | 200000 |
| 2.50 | 0.04394 | 0.04351 | 0.98 % | -0.94 | 200000 |

### 3.4 三路互裁（设计响应，年超越概率 0.01）

| 响应 | 精确长期积分 | IFORM 等值线($\beta_F$) | Rayleigh 口径 | Rayleigh vs IFORM | IFORM vs 精确 | 等值线($\beta_C$) vs 精确 | Gaussian 代理 vs IFORM |
|---|---|---|---|---|---|---|---|
| wave_power | 203.1 | 203.8 | 201.9 | 0.92 % | 0.32 % | 69.79 % | 26.68 % |
| modal_largest_wave | 14.18 | 14.01 | 14.01 | 0.00 % | 1.22 % | 22.27 % | 12.75 % |

精确长期积分的自洽性：目标 0.01，达成 0.00999，
网格概率质量 1.000000000000。

### 3.5 Rayleigh 口径审计（$H_s$ = 6 m）

| 量 | 闭式 | Monte-Carlo | 相对偏差 |
|---|---|---|---|
| 众值（$H_s/2$） | 3.0000 m | 3.0001 m（MLE） | 0.0025 % |
| 均值 $0.6267H_s$ | 3.7599 m | 3.7600 m | 0.0015 % |
| $H_s=2\bar H/\sqrt{\pi/2}$ | 6.0000 m | 6.0001 m | 0.0015 % |
| $N$ 波极值众值（$N$ = 1000） | 11.1508 m | 11.1256 m | 0.2258 % |

### 3.6 判据行（13 条，全部由代码实测生成）

| 判据 | 实测 | 判据 | 裁决 | 说明 |
|---|---|---|---|---|
| C1_iform_normal_anchor | 8.882e-16 | <= 0.01 | **PASS** | IFORM contour vs analytic bivariate-normal anchor (criterion: <=1%) |
| C2_contour_extreme_hs_anchor | 0 | <= 0.01 | **PASS** | max H_s on contour vs F_H^-1(Phi(beta)) analytic anchor (criterion: <=1%) |
| C3_rosenblatt_closure | 3.109e-15 | <= 1e-08 | **PASS** | inverse round trip + phi2(u)|det J| = f(x) density identity |
| C4_exceedance_calibration | 0.009831 | <= 0.03 | **PASS** | MC outside-contour probability vs exp(-beta^2/2) (criterion: calibrated) |
| C5_reliability_mapping | 0.0001851 | <= 0.01 | **PASS** | p_f <-> beta round trip + DNV/ISO 2394 target rows |
| C6_convention_mismatch_declared | 2.295 | >= 0.3 | **PASS** | convention F vs C mismatch quantified (must be non-trivial and declared) |
| C7_rayleigh_mode_recovery | 0.002258 | <= 0.02 | **PASS** | Rayleigh mode / N-wave modal max / mean / H_s vs closed form (MC, criterion <=5%) |
| C8_rayleigh_vs_iform | 0.009202 | <= 0.05 | **PASS** | Rayleigh-rule design value vs IFORM contour design value (criterion: <=5%) |
| C9_iform_vs_longterm_reference | 0.01219 | <= 0.05 | **PASS** | IFORM contour vs exact long-term quadrature reference at the same p_f |
| C10_contour_refinement | 0.004953 | <= 0.01 | **PASS** | angle-sweep refinement 36 -> 288 points |
| C11_conventionC_probability_content | 0.009831 | <= 0.1 | **PASS** | convention-C contour (beta_C) probability content equals its own target |
| C12_normal_case_contour_is_ellipse | 9.992e-16 | <= 1e-08 | **PASS** | correlated-normal anchor: Rosenblatt contour == chi-square ellipse (analytic identity) |
| C13_gaussian_surrogate_differs | 0.1275 | >= 0.02 | **PASS** | non-Gaussian marginals must move the design value vs a moment-matched Gaussian surrogate |

### 3.7 图

![IFORM 环境等值线](../artifacts/design_sea_state_ifrom/fig1_contours.png)
*图 1：物理空间 $(H_s,T_z)$ 上的 IFORM 等值线（$p_f=10^{-2},10^{-3},10^{-4}$，实线）、
convention-C 等值线（虚线）、矩匹配 Gaussian 代理等值线（点线）与海况散点；
圆点为各等值线的极值 $H_s$ 锚点。*

![Rosenblatt 与边际重现值锚](../artifacts/design_sea_state_ifrom/fig2_rosenblatt_anchor.png)
*图 2：左——标准正态空间中的半径 $\beta$ 圆（$\theta=0$ 即极值 $H_s$ 点）；右——模型的
$H_s$ 边际重现曲线、$2\times10^5$ 样本经验曲线，红星为等值线极值 $H_s$ 与解析锚重合。*

![三路互裁](../artifacts/design_sea_state_ifrom/fig3_adjudication.png)
*图 3：左——$H_s=6$ m 海况的 Rayleigh 分布、正确尺度 $H_s/2$ 与错误尺度 $H_s/4$ 对照、
众值与 $4\times10^6$ 波 MC 的 MLE；右——设计波功率 $J$ 的五条路线对比及相对精确长期积分的偏差。*

## ④ 结果图（小结）

三张图分别覆盖：等值线的几何正确性（图 1）、变换与边际锚的解析一致性（图 2）、
以及短时 Rayleigh 口径与长时等值线口径的数值互裁（图 3）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **Rayleigh 尺度陷阱**（本节点实测）：波高的 Rayleigh 尺度是 $a=2\sqrt{m_0}=H_s/2$，
   而 $\sqrt{m_0}=H_s/4$ 是**波面高程**标准差。用后者当尺度会把设计波高腰斩
   （图 3 左）。代码里两个量分别命名为 `rayleigh_scale` 与 `elevation_sigma`，
   并有专门用例把 $50\%$ 差异钉住。
2. **histogram 众值有偏**：在右偏的 Rayleigh 密度上用直方图峰值估众值系统性偏低
   1.47 %（平滑核把峰向左拉），故主用 MLE
   $\hat a=\sqrt{\sum x^2/2n}$；直方图值仅作记录。这一点作为"负面知识"保留。
3. **两种 β 口径不能混用**：同一 $p_f$，convention C 半径使设计响应显著偏保守
   （波功率 +69.8%、设计波高
   +22.3%）。
   结论：**等值线用于"概率含量"时用 $\beta_C=\sqrt{-2\ln p}$；用于"响应反演"时用
   $\beta_F=-\Phi^{-1}(p)$**。本文把二者都返回并命名，判据 C6 强制其鸿沟被显式登记。
4. **IFORM 等值线在相关正态下就是卡方椭圆**（我把最初"二者不同"的假设证伪了）：
   二者都是联合密度的等值集，故必须重合；判据 C12 反向钉住这个恒等式，
   真正的差异来自**非高斯边际**（判据 C13：Gaussian 代理把设计波功率低估
   26.68%、设计波高低估
   12.75%）。
5. **局限**：两变量模型（未含风、浪向、潮位）；$T_z\mid H_s$ 取常数 $\sigma_{\ln}$，
   实际站点常随 $H_s$ 变化；长期积分为独立海况假设（未计风暴聚集）；Rayleigh
   假设窄带、单峰谱。适用范围：中等海况、$p_f\ge10^{-4}$、窄带响应。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 实测相对误差 | 是否 <3% |
|---|---|---|---|
| 等值线 vs 解析二元正态锚（二次型残差） | 解析 $\beta^2$ | 8.88e-16 | **是** |
| 等值线极值 $H_s$ vs $F^{-1}(\Phi(\beta))$ | 解析 | 0.00e+00 | **是** |
| Rosenblatt 往返 / 密度恒等式 | 解析 | 3.11e-15 | **是** |
| 出线概率 vs $\exp(-\beta^2/2)$ | 解析 | 0.98 % | **是** |
| Rayleigh 众值（MLE） | 闭式 $H_s/2$ | 0.0025 % | **是** |
| Rayleigh $N$ 波极值众值 | 闭式 | 0.226 % | **是** |
| Rayleigh 口径 vs IFORM（波功率 / 设计波高） | IFORM | 0.92 % / 0.00 % | **是** |
| IFORM 等值线 vs 精确长期积分（波功率 / 设计波高） | 2D 求积真值 | 0.32 % / 1.22 % | **是** |
| 等值线角度离散（36 → 288） | 细化解 | 0.50 % | **是** |
| **（口径差异，非误差）$\beta_C$ 等值线 vs 精确** | 2D 求积真值 | 69.79 % / 22.27 % | 否 —— 属**口径差异**，已登记为判据 C6/C11 的显式结论，不作为误差补偿 |

结论：所有可解析/可求积的基准量误差均远小于 3%（最高 0.98 % 为 MC 抽样误差量级，
标准误 0.0005）；唯一超过 3% 的项是**口径差异**（convention C 等值线在同一
年目标下偏保守），本报告把它作为结论而非误差写出。

## 附：审计链

- 模块：`src/tensorphysics/design_sea_state/{rosenblatt,iform,reliability_index,rayleigh_consistency,api}.py`
- 判据实现：`evaluate_criteria()`，13 条 `MACHINE_CRITERIA_KEYS`
- 测试：`tests/test_design_sea_state.py`（26 用例，含"判据表能拒绝坏结果"的反向用例）
- 工件：`artifacts/design_sea_state_ifrom/study.json`（全部数字）+ `artifacts/design_sea_state_ifrom/fig{1,2,3}_*.png`
- 复现：`PYTHONPATH=src python benchmarks/design_sea_state/run_ifrom_study.py`
