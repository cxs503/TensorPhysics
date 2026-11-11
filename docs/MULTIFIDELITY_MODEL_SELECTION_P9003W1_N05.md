# 多保真度模型选择引擎：误差预算 + 计算成本自动推荐（阻力预报案例）

**Mission m13845 · Campaign node p9003w1_n05 · 仓库 TensorPhysics · 物理域 · 全新前沿节点**

新增模块 `tensorphysics.multifidelity`（stats / model / selection / calibration / backtest / verification / figures，7 个源文件 + 1 个注册表配置），24 个 pytest 用例全绿（`24 passed, 1 warning in 2.97s`）。本报告全部数字由 `benchmarks/multifidelity/run_verification.py` 现场计算注入（`artifacts/multifidelity_selection/p9003w1_n05_results.json`，配置 sha256 `b7f34928fc5e80c4…`，git `498e828a3`，生成于 2026-11-11T06:03:24+00:00，零手抄），图 1–6 由同一脚本生成。

> **诚实边界（先声明）**：本节点验收的对象是**选择引擎的统计承诺**（误差预算、成本预算、区间覆盖率），不是 Holtrop 公式本身的精度。所谓「真值」是**显式声明的随机真值生成器**（先验乐观化 + 重尾跳变 + 分层异方差），不是实测阻力数据；注册表里的误差/成本先验是**文献量级先验**（Holtrop & Mennen 1982 / ITTC-1978 / 网格收敛 RANS 2–3% / 水池重复性 0.5–1%），未做任何拟合。真实 EFD 对照本工作树内不可得，故本报告**不对阻力绝对值作实验对标**，只在 §6 显式声明该项 WITHHELD。

## 1. 案例介绍

### 1.1 物理背景与问题

船舶总体性能设计里，同一个物理量（这里是静水总阻力 R_T）有多个保真度的获取手段：经验回归公式、半经验预报程序、CFD RANS、水池试验。四者的精度差 2–5 倍、成本差 10⁵–10⁶ 倍。工程决策的真实形式不是「哪个最准」，而是「在给定误差预算 ε 与计算/设施预算 C 下，**哪个最划算且可信**」。

把这件事做成可验证的机制，需要三个区分：

1. **准确度是分布，不是数**：Holtrop 的误差不是 6%，而是「中位 6%、对数标准差 0.40」的一条分布。拿中位数去比预算，会有一半场合违约。
2. **预算也是概率约束**：`P(error ≤ ε) ≥ α` 与 `P(cost ≤ C) ≥ β`，α、β 进入合同（本节点 0.95）。
3. **报出的不确定性必须被回测**：引擎给每个查询一个 90% 单侧误差上界，这个界的经验覆盖率必须接近 90%——这恰好是最容易被「先验很好看」骗过去的地方（§3.2：先验直接用时覆盖率只有 64.22%，名义 90%）。

### 1.2 注册模型与参数表（数据，非代码）

注册表 `src/tensorphysics/multifidelity/configs/drag_multifidelity.v1.json`：

| 模型 | 保真级 | 误差先验（中位 / σ） | 成本（中位 s / σ） | 适用域 Fn | Cb | Re |
|---|---|---|---|---|---|---|
| `holtrop_mennen_1982` | 1 | 6.00% / 0.40 | 0.4 / 0.25 | 0.10–0.45 | 0.50–0.85 | 1.0e+08–5.0e+09 |
| `ittc1978_prediction` | 2 | 4.00% / 0.35 | 4 / 0.3 | 0.10–0.40 | 0.50–0.85 | 1.0e+08–5.0e+09 |
| `cfd_rans_sst` | 3 | 2.50% / 0.30 | 7.2e+03 / 0.4 | 0.05–0.48 | 0.48–0.88 | 1.0e+08–6.0e+09 |
| `efd_towing_tank` | 4 | 1.20% / 0.20 | 1.73e+05 / 0.25 | 0.05–0.45 | 0.48–0.88 | 1.0e+07–1.0e+10 |

误差定义：`r = |R_T,model − R_T,ref| / R_T,ref`（无量纲相对误差幅值，恒正）。成本单位为秒（CPU 秒或设施占用秒），是量级而非价格。

### 1.3 物理锚：把 (Fn, Cb, Re) 盒子钉在一条真船上

注册表的适用域与 ε 的量级必须对应真实阻力，否则「ε=3%」是空话。用本仓库的 Holtrop-1984 核（`tensorphysics.physics_kernels_holtrop1984.holtrop1984_resistance`）对公开几何 KCS (KRISO container ship, public geometry) 计算：

| 航速 [kn] | Fn | Re | R_T [kN] | ε=1% 对应 ΔR_T [kN] | ε=2% [kN] | P_E [kW] |
|---|---|---|---|---|---|---|
| 18 | 0.1949 | 1.790e+09 | 933.6 | 9.3 | 18.7 | 8645 |
| 21 | 0.2274 | 2.088e+09 | 1420.0 | 14.2 | 28.4 | 15341 |
| 24 | 0.2599 | 2.386e+09 | 2136.7 | 21.4 | 42.7 | 26381 |

几何：Lpp=230 m、B=32.2 m、T=10.8 m、Cb=0.6505、Cm=0.9770、Cwp=0.8596、∇=52030 m³。
边界声明：resistance magnitudes and (Fn, Re) ranges only; no EFD comparison is claimed here；该表只用于证明 (Fn, Re) 采样区间与 ε 的物理量级自洽。

### 1.4 工具链

`numpy`（采样/统计）+ 纯 Python 决策核心（无隐藏状态、可在任意 host 复现）+ `matplotlib`（图）+ `pytest`（24 例）。选定轨迹：queries 采样子种子 20261111、真值潜变量种子 7、验证残差种子 11；n_queries=2000，ε∈[0.02, 0.35] 对数均匀，C∈[500, 10000000] s 对数均匀，α=β=0.95，区间名义水平 0.9。

## 2. 怎么做的

### 2.1 选择问题（两个对称的优化）

对查询点 p（Fn, Cb, Re）与预算 (ε, C)：

```
min_cost :  argmin_m E[cost_m]   s.t.  P(r_m ≤ ε) ≥ α  且  P(cost_m ≤ C) ≥ β
min_error:  argmin_m median(r_m) s.t.  同上
m ∈ A(p) = { 模型 : p 落在 m 的适用域内 }
```

`P(r ≤ ε)` 直接由误差律给出：先验时是 log-normal 的解析 CDF，重标定后是残差经验 CDF（保守，不加平滑）。成本概率同理。**拒绝是合法输出**：

| 拒绝码 | 含义 |
|---|---|
| `NO_ADMISSIBLE_MODEL` | 所有注册模型在该查询点都出适用域（逐轴报出越界轴） |
| `EPSILON_BELOW_MODEL_FLOOR` | ε 低于最准可用模型的中位误差——物理上不可能达成（负控） |
| `COST_BELOW_MODEL_FLOOR` | C 低于最便宜可用模型的中位成本 |
| `NO_MODEL_MEETS_ERROR_BUDGET` | 无模型能在 α 置信度下承诺 ε |
| `NO_MODEL_MEETS_COST_BUDGET` | 无模型能在 β 置信度下承诺 C |
| `NO_MODEL_MEETS_BOTH_BUDGETS` | 每个模型都至少违反一个预算 |

拒绝时仍给出**标注为 advisory 的最优努力候选**（`advisory_only=True`），避免下游把它误当保证。

### 2.2 不确定性量化：保形重标定（conformal recalibration）

注册先验回答的是「模型在标定集上有多准」，新船不在标定集里：重尾爆发、域外误差增长都看不见。所以引擎把「验证残差样本」当作一等输入：

* 单侧上界取**保形次序统计量** `k = ceil((n+1)·level)` 的残差值；在可交换性下有限样本覆盖 ≥ level，不需要任何分布假设；
* 残差可按轴盒子**分层**（`StratumSpec`）；查询命中某层且该层样本 ≥ min_n 时用该层，否则退回池化样本——分层修的是「误差随 Fn 增长」的残余异方差，而不是假装建模它；
* 层划分是**配置里声明的设计选择**，不是拟合结果；因此本报告额外跑了一个**故意分错层**的对照（按 Cb 而非 Fn 分层）来量化这条假设的代价（§3.3）。

### 2.3 回测协议（承诺必须被证伪过）

四个注册表变体 × 四个真值场景，同一批 2000 查询，配对比较：

| 变体 | 说明 |
|---|---|
| `prior_only` | 只用注册先验（基线） |
| `conformal_pooled` | 用池化验证残差重标定 |
| `conformal_stratified` | 按声明分层（Fn）重标定——**验收主配置** |
| `conformal_mispartitioned` | 故意按 Cb 分层（错误划分）——假设的证伪探针 |

| 真值场景 | 机制 | 作用 |
|---|---|---|
| `prior_consistent` | 真值=注册先验本身 | 机制对照：必须复现 α=0.95 与 90% 覆盖 |
| `deployment` | σ×1.15 + 12% 重尾跳变(×2.2) + 高 Fn 段 ×1.35 | 主验收场景 |
| `heteroscedastic_only` | 仅高 Fn 段 ×1.70 | 消融：分层该赢的场景 |
| `heavy_tail_only` | 仅 15% 重尾跳变 | 消融：分层**不该**赢的场景（方法自证伪） |

潜变量按 (query, model) 预生成，全部变体共用同一批实现，因此比较是配对的。每次推荐都额外计算**事后 oracle**（同查询下真实同时满足 ε 与 C 的最便宜模型），用「推荐成本 / oracle 成本」量化决策质量——这是拒绝率之外唯一能说明「引擎不是在躺平」的指标。

## 3. 计算结果

### 3.1 验收判据（主配置 = deployment 真值 + conformal_stratified）

| 判据 | 要求 | 实测 | 判定 |
|---|---|---|---|
| (a) 推荐准确率 P(实际误差 ≤ ε) | > 90% | 97.855% | PASS |
| (b) 成本效率 P(实际成本 ≤ C) | > 90% | 100.000% | PASS |
| (c) 区间覆盖率（名义 90%） | > 85% | 89.275% | PASS |
| (d) 负控：ε<1% 必须 FAIL | 全部 FAIL/`EPSILON_BELOW_MODEL_FLOOR` | [0.005, 0.008, 0.01] | PASS |
| (e) pytest 用例数 | ≥ 10 | 24 | PASS |
| (f) 最差分层覆盖率（透明性附加项） | ≥ 85% | 86.975% | PASS |

**总判定：全部通过。**

### 3.2 主结果表：四个变体 × 四个真值场景

| 真值场景 | 变体 | 推荐率 | 准确率 (a) | 成本效率 (b) | 覆盖率 (c) | 最差分层 | oracle 一致率 | 成本/oracle | 平均界 |
|---|---|---|---|---|---|---|---|---|---|
| prior_consistent | `prior_only` | 67.5% | 99.41% | 99.85% | 89.48% | 88.89% | 67.0% | 1.00 | 0.0581 |
| prior_consistent | `conformal_pooled` | 48.9% | 100.00% | 100.00% | 99.49% | 99.06% | 42.2% | 8.65 | 0.0770 |
| prior_consistent | `conformal_stratified` | 48.9% | 99.80% | 100.00% | 98.88% | 97.69% | 45.4% | 7.33 | 0.0761 |
| prior_consistent | `conformal_mispartitioned` | 51.7% | 100.00% | 100.00% | 99.81% | 99.78% | 48.4% | 6.07 | 0.0786 |
| deployment | `prior_only` | 67.5% | 92.00% | 99.85% | 64.22% | 54.25% | 72.0% | 1.00 | 0.0581 |
| deployment | `conformal_pooled` | 48.9% | 98.26% | 100.00% | 89.66% | 88.36% | 47.2% | 5.97 | 0.0770 |
| deployment | `conformal_stratified` | 48.9% | 97.85% | 100.00% | 89.27% | 86.97% | 49.0% | 4.48 | 0.0761 |
| deployment | `conformal_mispartitioned` | 51.7% | 97.87% | 100.00% | 88.88% | 85.98% | 53.4% | 1.00 | 0.0786 |
| heteroscedastic_only | `prior_only` | 67.5% | 92.44% | 99.85% | 57.56% | 33.59% | 73.7% | 1.00 | 0.0581 |
| heteroscedastic_only | `conformal_pooled` | 48.9% | 99.90% | 100.00% | 90.58% | 84.00% | 48.3% | 5.49 | 0.0770 |
| heteroscedastic_only | `conformal_stratified` | 48.9% | 99.69% | 100.00% | 93.05% | 88.67% | 48.5% | 5.27 | 0.0761 |
| heteroscedastic_only | `conformal_mispartitioned` | 51.7% | 99.71% | 100.00% | 89.56% | 81.71% | 55.9% | 1.00 | 0.0786 |
| heavy_tail_only | `prior_only` | 67.5% | 94.30% | 99.85% | 79.63% | 78.12% | 67.9% | 1.00 | 0.0581 |
| heavy_tail_only | `conformal_pooled` | 48.9% | 98.87% | 100.00% | 92.63% | 90.40% | 44.6% | 7.67 | 0.0770 |
| heavy_tail_only | `conformal_stratified` | 48.9% | 98.47% | 100.00% | 91.42% | 87.18% | 47.7% | 5.96 | 0.0761 |
| heavy_tail_only | `conformal_mispartitioned` | 51.7% | 98.65% | 100.00% | 92.94% | 91.54% | 50.5% | 1.00 | 0.0786 |

读法（三件事）：①**机制对照成立**——`prior_consistent` 下先验直接用即可复现 准确率 99.41% 与覆盖率 89.48%（名义 90%），说明引擎本身没有系统偏差；②**deployment 下先验直接用的覆盖率崩到 64.22%**（最差层 54.25%），而 conformal 重标定后回到 89.27%（最差层 86.97%）——这正是判据 (c) 要抓的东西，且**未标定的基线是 honest FAIL，没有被覆盖**；③**保证有价格**：重标定把推荐率从 67.5% 压到 48.9%，成本/oracle 中位比 4.48×——引擎宁可拒绝，也不在 α=0.95 下押注重尾。（α 可调，价目见 §3.5。）

### 3.3 消融：分层到底值不值？

| 真值场景 | 池化最差分层覆盖率 | 分层最差分层覆盖率 | 分层收益 | 错分层最差分层覆盖率 |
|---|---|---|---|---|
| heteroscedastic_only | 84.00% | 88.67% | +4.67 pts | 81.71% |
| heavy_tail_only | 90.40% | 87.18% | -3.21 pts | 91.54% |
| deployment | 88.36% | 86.97% | -1.39 pts | 85.98% |

结论按声明读：**当误设是异方差时分层赢**（`heteroscedastic_only` 最差层覆盖率 84.00% → 88.67%），**当误设里重尾主导时分层不再占优**（`heavy_tail_only`：90.40% → 87.18%；混合场景 `deployment`：88.36% → 86.97% —— 每层的保形次序统计量都退化到同一条重尾上，分层只是额外增加了小样本噪声）。**分错层等于白分**——`conformal_mispartitioned` 的最差层覆盖率在异方差场景掉到 81.71%，这就是声明式分层的代价。此段是方法自证伪，不允许事后改口径。

### 3.4 误差-成本 Pareto 前沿（示例查询点 Fn=0.30, Cb=0.72, Re=1.5e9）

| 模型 | 成本中位 [s] | 成本 90% 上界 [s] | 误差中位 | 误差 90% 上界 |
|---|---|---|---|---|
| `holtrop_mennen_1982` | 0.4 | 0.551 | 8.78% | 18.80% |
| `ittc1978_prediction` | 4 | 5.88 | 5.42% | 9.93% |
| `cfd_rans_sst` | 7.2e+03 | 1.2e+04 | 3.66% | 6.54% |
| `efd_towing_tank` | 1.73e+05 | 2.38e+05 | 1.70% | 2.83% |

四个模型在 (成本, 误差) 平面上互不支配，前沿呈阶梯状（图 2）：成本跨 5.6 个数量级，注册先验的误差中位 6.00% → 1.20%，重标定（同一查询点、90% 保形界）后 8.78% → 1.70%。Pareto 前沿随重标定整体抬高：同一 ε 需要的保真级上升，这就是保证的价格在图上的形态。

### 3.5 置信度扫描：保证的价格（同一批 2000 查询）

| 变体 | α | 推荐率 | 准确率 | 覆盖率 | 成本/oracle 中位比 |
|---|---|---|---|---|---|
| `prior_only` | 0.80 | 74.0% | 84.66% | 64.19% | 1.00 |
| `conformal_stratified` | 0.80 | 68.0% | 93.38% | 89.71% | 1.00 |
| `prior_only` | 0.90 | 70.7% | 88.61% | 63.58% | 1.00 |
| `conformal_stratified` | 0.90 | 60.5% | 95.70% | 89.59% | 1.00 |
| `prior_only` | 0.95 | 67.5% | 92.00% | 64.22% | 1.00 |
| `conformal_stratified` | 0.95 | 48.9% | 97.85% | 89.27% | 4.48 |

α=0.90 是个甜点：推荐率回升、成本/oracle 比回到 1.0（决策与事后最优一致），覆盖率仍 89.59%。α=0.95（验收所用）买的是尾部安全，代价是 4–5× 的成本膨胀。

### 3.6 负控明细（ε<1%）

| ε | 状态 | 拒绝码 | 理由 |
|---|---|---|---|
| 0.5% | FAIL | `EPSILON_BELOW_MODEL_FLOOR` | epsilon=0.005 is below the accuracy floor 0.01241 (median error of the best admissible model); no model can meet it |
| 0.8% | FAIL | `EPSILON_BELOW_MODEL_FLOOR` | epsilon=0.008 is below the accuracy floor 0.01241 (median error of the best admissible model); no model can meet it |
| 1.0% | FAIL | `EPSILON_BELOW_MODEL_FLOOR` | epsilon=0.01 is below the accuracy floor 0.01241 (median error of the best admissible model); no model can meet it |

三个负控都返回 FAIL：ε 低于「最准可用模型的中位误差」时，任何推荐都是幻想。

## 4. 结果图

![fig1_recommendation_histogram.png](artifacts/multifidelity_selection/figures/fig1_recommendation_histogram.png)

*图 1｜推荐模型分布直方图（deployment truth，2000 查询）。纵轴为各配置推荐结果中该模型的占比：未标定先验（红）大量落到便宜的 Holtrop/ITTC；conformal 重标定（橙/绿）把推荐推到 CFD/EFD，即「保证」的代价。*

![fig2_pareto_frontier.png](artifacts/multifidelity_selection/figures/fig2_pareto_frontier.png)

*图 2｜误差-成本 Pareto 前沿（实线=中位数，虚线=90% 上界）。灰点为注册模型；横轴 log 成本 [s]，纵轴 R_T 相对误差 [%]；点线为示例精度预算 ε=4% 与计算预算 C=2×10⁴ s。*

![fig3_interval_coverage.png](artifacts/multifidelity_selection/figures/fig3_interval_coverage.png)

*图 3｜单侧 90% 误差界覆盖率（柱）+ 最差分层覆盖率（黑色横线）+ Wilson 95% 置信区间。绿虚线=名义水平 90%，红点线=验收下限 85%。*

![fig4_budget_response.png](artifacts/multifidelity_selection/figures/fig4_budget_response.png)

*图 4｜按精度预算 ε 分箱的准确率与成本效率（deployment truth）。左：P(实际误差 ≤ ε)；右：P(实际成本 ≤ C)。红点线为 90% 验收线。*

![fig5_error_ratio.png](artifacts/multifidelity_selection/figures/fig5_error_ratio.png)

*图 5｜误差界标定图：实际误差 / 预测 90% 上界（箱=IQR，须=5–95%）。虚线以上=界被突破。*

![fig6_confidence_sweep.png](artifacts/multifidelity_selection/figures/fig6_confidence_sweep.png)

*图 6｜置信度 α 扫描（同一批 2000 查询）：推荐率、准确率、以及「保证的价格」——中位成本 / 事后 oracle 成本。*

## 5. 注意点（踩过的坑 / 局限）

### 5.1 三个真实的坑（都影响了数字，均保留记录）

1. **跳变指示变量用了正态分布**（已修，`backtest.py` 有注释钉住）：重尾跳变概率 0.12 本应比较 `U(0,1) < 0.12`，但潜变量整块按标准正态生成，于是「负值即跳变」→ 实际跳变率 50%。**是机制对照（prior_consistent 真值）抓到的**：该对照下准确率/覆盖率本应 0.95/0.90，实测 0.841/0.554——没有对照真值就发现不了。修复后该对照指标见 §3.2 表。
2. **残差池化被饿死**：最初池化样本只收集「不属于任何分层」的样本，而当声明分层的轴盒子覆盖整个采样域时，池化样本永远凑不够，循环跑到上界、给每个模型攒了 9–14 万个残差（先验估计被无限精确化，覆盖率虚高到 97%）。修复：池化=全部样本，分层=子集；残差量回到 1052 量级，覆盖率回到 89%。
3. **`eps < floor` 与 `α 不可达` 是两种不同的拒绝**：前者是物理不可能（负控要 FAIL 的那种），后者是「有更准的模型但预算不允许」，两者合一会让负控判据失去意义。现在分别是 `EPSILON_BELOW_MODEL_FLOOR` 与 `NO_MODEL_MEETS_ERROR_BUDGET`。

### 5.2 局限（适用范围）

* **真值是显式随机模型，不是物理**。它证明的是选择机制在「先验会错、会重尾、会异方差」时仍守承诺，不证明 Holtrop/ITTC/RANS 的真实精度等级；真实标定需要一条带留出集的验证战役（残差样本即接口）。
* **覆盖是边际的**：保形保证是总体（或分层）可交换意义下的，不是逐查询条件覆盖；异方差未被声明的区间（如 Fn∈[0.20,0.25] 内部）不会有额外保障。
* **分层是声明式假设**：分错层的代价已量化（§3.3），引擎不做自动分层发现。
* **成本模型是量级**：CPU 秒/设施占用秒，不含排队、失败重试与人力；跨设施价格不在本节点范围。
* **α=0.95 下拒绝率高**（推荐率 48.9%）：这是诚实而非缺陷；若业务能接受 α=0.90，推荐率与成本比显著改善（§3.5）。
* **单一物理量**（R_T）：换量（如功率、耐波 RAO、冰载荷）需重新注册先验与残差，引擎本身不含领域语义。

## 6. 误差判定（逐项，是否 <3%）

「基准」分两类：**名义统计水平**（覆盖率名义 90%、准确率承诺 α=0.95、成本承诺 β=0.95）与**实验值**（阻力绝对值）。前者逐项给出相对偏差，后者显式声明 WITHHELD。

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 <3% | 说明 |
|---|---|---|---|---|---|
| 机制对照：覆盖率 | 0.90 | 89.48% | -0.58% | 是 | 先验即真值时引擎必须复现名义水平 |
| 机制对照：准确率 | 0.95 | 99.41% | +4.64% | **否** | 约束是下界，高保真模型 P≈1 → 安全方向偏高 |
| 主配置：覆盖率 (c) | 0.90 | 89.27% | -0.81% | 是 | 验收判据 (c)，>85% 且贴近名义 90% |
| 主配置：最差分层覆盖率 | 0.90 | 86.97% | -3.36% | **否** | 两层中的较差者，n≈300/层，Wilson 半宽约 3.5 pts |
| 主配置：准确率 (a) | 0.95 | 97.85% | +3.01% | **否** | 验收判据 (a) |
| 主配置：成本效率 (b) | 0.95 | 100.00% | +5.26% | **否** | min_cost 目标优先选便宜模型 → 结构性偏高（合同单侧） |
| 未标定基线：覆盖率 | 0.90 | 64.22% | -28.64% | **否** | honest FAIL：先验未标定的真实代价 |
| 阻力绝对值 vs EFD | 实验 R_T | WITHHELD | — | **不适用** | 工作树内无可溯源 EFD，见下 |

显式回答：

* **覆盖率 <3%**：机制对照 -0.58%、主配置 -0.81%，两项均 <3%，保形界标定正确。
* **最差分层覆盖率 -3.36%**（86.97% vs 90%）：**不 <3%**，如实报。原因是取两层最小值、每层约 300 个体，Wilson 95% 半宽约 3.5 pts，−3.4% 落在抽样噪声内；但它同时说明**边际保形保证不等于逐层条件保证**——所以我们把它单列为附加判据 (f) 而不是藏进总覆盖率。
* **准确率 +3.01%**：约束是 `P(r≤ε) ≥ α`（下界），而 min_cost 会在满足下界的前提下选最便宜的模型，在多数量级预算区间里其 P 接近 1，因此实测高于承诺属**安全方向**，不是标定缺陷；单侧合同的正确读法是「违约率 ≤ 1−α」，实测违约率 2.145% < 5%。
* **成本效率 +5.26%**：同理为单侧合同的安全方向；且验证真值的成本分布与注册成本先验同族（成本是可知量，误差才是不可知量），所以这里的偏高不是「成本先验被低估」的证据。
* **未标定基线覆盖率 64.22%（-28.6%）**：远超 3%，**honest FAIL 保留**，正是判据 (c) 与重标定层的存在理由，也说明「先验好看」不等于「界可信」。
* **阻力绝对值对实验的误差：WITHHELD**。原因有两条：①本离线工作树内没有可溯源的水池 EFD 数字（不凭记忆引用 KCS 试验值）；②本节点验收对象是选择引擎的统计承诺，不是 Holtrop 公式精度。§1.3 只把 Holtrop 核用作量级锚（R_T 与 ε 的物理量级自洽）。因此本报告**不作任何「<3%」的阻力精度声明**。

## 7. 可复用知识（蒸馏条目，knowledge-distillation 交付）

**规则（Title / rule）**：凡「按误差预算选模型」的机制，误差必须作为**分布**注册、预算必须写成**概率约束**、报出的不确定度必须是**可回测的保形界**；只登记中位误差就把覆盖率当成承诺，在域外船上会系统性违约。

**适用边界（context / applicability）**：适用于任何「同一量有多个保真度获取手段 + 预算为合同」的场景（阻力/功率/耐波/冰载荷/结构响应均可，只需换注册表数据）。不适用：①无验证残差样本时（此时只能报先验界，并声明覆盖率未经验证）；②查询点落在所有模型适用域之外（应直接拒绝）；③真值不可交换（时间序列强相关）时保形次序统计量的有限样本保证失效。

**证据（evidence / provenance）**：节点 p9003w1_n05 / mission m13845 / TensorPhysics；工件 `artifacts/multifidelity_selection/p9003w1_n05_results.json`（配置 sha256 `b7f34928fc5e80c4…`）；代码 `src/tensorphysics/multifidelity/`；测试 `tests/test_multifidelity.py`（24 例）；生成时间 2026-11-11T06:03:24+00:00。

**机制（mechanism）**：log-normal 先验给 `P(r ≤ ε)` 与解析 90% 界 → 保形次序统计量把「验证残差样本」换成无分布假设的有限样本界 → 轴盒子分层吸收残余异方差 → 回测用预生成潜变量的配对比较证伪承诺（含 oracle 成本比，防止「靠拒绝刷准确率」）。

**推荐流程（procedure）**：①按轴盒子注册每个保真级的误差/成本先验并标注出处；②采一条**独立于部署查询**的验证残差样本（含域外船）；③默认用分层保形界，层数控制在每层 ≥30 样本；④每条推荐都输出 `prob_error_le_epsilon`、`prob_cost_le_cost`、`error_upper`、advisory 拒绝理由；⑤上线前必须跑对照组（先验即真值）与消融组（重尾/异方差分离），并把覆盖率与 oracle 成本比一起公布。

**反例与局限（counterexamples / limitations）**：先验直接用（`prior_only`）在 deployment 真值下覆盖率 64.22%（名义 90%）——「先验好看」不是承诺；分层在**全局重尾主导**时反而更差（heavy_tail_only 最差层 90.40% → 87.18%）；**分错层**在异方差场景掉到 81.71%；α=0.95 的保证价格是中位成本/oracle 4.48×。

**回归护栏（regression guard）**：`test_conformal_bound_finite_sample_coverage`（保形界平均覆盖率 0.885–0.915）、`test_prior_consistent_truth_reproduces_prior`（真值=先验时 0.5/0.9 分位偏差 <2.5%）、`test_jump_indicator_is_uniform_and_rate_is_honoured`（跳变指示变量必须均匀，防 5.1 坑复发）、`test_negative_control_epsilon_below_floor`（ε<1% 必须 FAIL）、`test_calibrated_selection_is_more_conservative`（残差变宽不得让引擎更胆大）、`test_end_to_end_verification_acceptance`（未标定基线覆盖率必须 <85%，防止把 honest FAIL 洗成 PASS）。

**评审状态 / 最后核验**：未经外部同行评审（提交平台独立席位按 S×E 复判）；本文件由 `generate_p9003w1_n05_report.py` 从工件生成，最后核验 2026-11-11。

## 8. 复现与审计链

```bash
python benchmarks/multifidelity/run_verification.py      # 产物 + 6 图 + 验收表
python -m pytest tests/test_multifidelity.py -q          # 24 例
python generate_p9003w1_n05_report.py --publish          # 本报告 + 平台投递
```

* 注册表配置 sha256：`b7f34928fc5e80c42d30cabd49321efe6f8538d125823ae6e14ecfde63c2d7b6`
* 结果工件：`artifacts/multifidelity_selection/p9003w1_n05_results.json`（含逐查询紧凑记录、标定口径、残差摘要 digest）
* git：代码提交 `498e828a3e93e3d75a2d8638b4c4b7a4efeea74e`（本模块最后一次被修改的提交，稳定标识）；工件生成于 HEAD `498e828a3e93e3d75a2d8638b4c4b7a4efeea74e`；python 3.11.6；Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38
* 非确定性来源：无（全部 rng 均有显式种子）；重跑应逐位复现。