# 作业窗口包络引擎：探测尺度与子网格特征 (N09)

**节点** IN68W2-N09 · 仓 TensorPhysics · 模块 `tensorphysics.envelope`

**一句话** 窗口引擎过去用「每个区间一个中点采样」判可行，遇到比节点间距更窄的特征时，同一份代码会给出两种都错的答案：把整个跨度宣布为可行（本例真窗宽 0.002，v1 报 1），或者把窗口整段丢掉。两种答案过去都带着 `contracts.ok == True`，因为没有任何字段记录「到底在哪个尺度上探过」。本节点把探测尺度变成显式变量：按声明的间距打点、不一致就地细分、细分不掉的跨度登记为 unresolved 并让契约失败。

## ① 案例介绍

**物理背景** 三个既有作业窗（搁浅临界潮位 / 压载置换率 / STS 泵率）把「参数可用的取值区间」交给同一个引擎求解；窗宽跨越两个数量级，参数面包含 1 维（单参数扫描）与 2 维（双参数平面）两种形态。本节点不引入新的领域公式：三窗仍是 `scenarios/envelope_cases/` 里的普通配置数据，本节点的靶子是**求解机制**本身。

**几何 / 参数** 1 维窗：轴 `alpha ∈ [0,1]`，默认 96 个扫描节点（节点间距 0.01053），约束是不等式保证金。子网格特征用解析窗构造：口袋窗 `|alpha - c| ≤ w`（真窗宽 `2w`）与窄空隙窗 `|alpha - c| > w`（真可行集为两段）。2 维窗：单位正方形，对角通道 `|beta - alpha| ≤ eps`，解析面积 `2·eps − eps²`。

**参数表**

| 量 | 值 |
|---|---|
| 默认扫描节点数 (1 维) | 96 |
| 节点间距 | 0.01053 |
| 每扫描格探测点数 `samples_per_segment` | 8 |
| 保证探测间距 τ（默认） | 0.001316 |
| 子格数 / 最大深度 | 8 / 6 |
| 细分预算上限（次约束评估） | 2000000 |
| 参与迁移的三窗条件数 | 9 |

## ② 怎么做的

1. **探测（`subgrid._probe_positions`）** 每个区间的判定不再取一个中点，而是在目标间距 `τ = (轴跨度/扫描格数)/samples_per_segment` 上打点，并同时采两端点。保证：宽度 ≥ τ 的特征（可行口袋或不可行缝隙）必落在至少一个探测点上。
2. **判定（`subgrid._scan_span`）** 采样按 `+1 可行 / −1 不可行 / 0 在容差内贴边界` 分类；内部混号或相邻异号即视为不一致。
3. **细分** 不一致的区间按 `sub_cells` 等分，子节点上对每条约束做符号变化 bracket + Brent/二分精化，根位置并入分割边（因此边界由根定位，不会停在探测点上）；子区间递归重探，直到无异议、深度用尽或预算用尽。
4. **契约（`contracts.check_feature_resolution`）** 输出 `resolution:<label>`：`min_probed_width`（实际探到的最细间距）、`suspect_spans`、`refined_spans`、`unresolved_spans`、`budget_exhausted`。有 unresolved 或预算耗尽即判失败；案例若声明 `contract.resolution.min_feature_width`，还要求 τ ≤ 该值，否则失败。
5. **2 维** 逐行 1 维 marching 继承同一机制；平面结果额外登记 `line_spacing` 与`single_line_components`，据此给出 `area_reliable`（特征比线间距还窄时面积为不可解，不许当成数字报出去）。
6. **零域语义** 重扫包内源码与测试的领域词表命中数：0 条。所有门限、间距、格点来自 case 配置。

## ③ 计算结果

### 判据总表

| criterion | what it establishes | verdict |
|---|---|---|
| **C1** | migration consistency vs the closed-form per-window path | yes |
| **C10** | shipped windows are bit-identical with and without the probe pass | yes |
| **C11** | envelope test suite | yes |
| **C2** | boundary reconstruction order (interpolated boundary vs analytic) | yes |
| **C3** | contract checker negative control: declared monotone, actually folded | yes |
| **C4** | infeasible semantics: empty window, nearest feasible point, no extrapolation | yes |
| **C5** | zero domain semantics in package sources and tests | yes |
| **C6** | cost per window and marched area vs brute-force grid scan | yes |
| **C7** | sub-grid window recovery against the analytic window | yes |
| **C8** | the v1 single-midpoint failure is reproducible and no longer silent | yes |
| **C9** | cost of the probe pass on windows that have no sub-grid feature | yes |

### 子网格窗恢复（C7，解析窗为基准）

| 特征 | 真窗 | 引擎窗 | 宽度相对差 | 探测间距 τ | 约束评估 |
|---|---|---|---|---|---|
| pocket at the scan midpoint | [0.499, 0.501] | [0.499, 0.501] | 3.331e-13 | 3.333e-04 | 7308 |
| pocket off the scan midpoint | [0.299, 0.301] | [0.299, 0.301] | 0.000e+00 | 4.000e-04 | 7310 |
| pocket, 3.5x narrower | [0.4163, 0.4177] | [0.4163, 0.4177] | 3.965e-14 | 3.500e-04 | 7311 |

偏差最大 3.331e-13（判据 ≤ 1e-12）。窄空隙窗（真可行集两段）恢复：yes，引擎给出 [[0.0, 0.36919999999999986], [0.3708, 1.0]]。

### v1 基线对照（C8）

| 特征 | v1 判定 | v1 宽度 | v1 是否静默出错 | 修复后 | 修复后宽度 | 粗探测(1/格) 时是否登记 unresolved |
|---|---|---|---|---|---|---|
| pocket at the scan midpoint | feasible | 1 | yes | feasible | 0.002 | yes |
| pocket off the scan midpoint | infeasible | — | yes | feasible | 0.002 | yes |
| pocket, 3.5x narrower | infeasible | — | yes | feasible | 0.0014 | no |

v1 静默出错 3/3，修复后静默出错 0。v1 基线不是回忆，是脚本里逐字复现的「网格根 + 每段一个中点采样」实现（`_v1_window`），与修复后的引擎同案对跑。

### 迁移一致性（C1 / C10）

* C1：三窗 × 三条件共 27 个数字对「闭式分窗路径」比较，最大相对差 3.608e-15（判据 ≤ 1e-12）。
* C10：同一批窗在默认策略与「每格 1 个探测点、不细分」策略下逐位比较，18 个边界数字 + 状态 + 活跃约束标签，不一致 0 处。新增的探测不允许挪动任何一个它不必挪动的数字。

### 边界阶次与空窗语义（C2 / C4）

* 解析抛物带边界误差实测阶最小 2.0000（判据 ≥ 1.9）。
* 空窗返回 `infeasible`、无边界、无面积、带最近可行点残差：yes。

### 成本（C6 / C9）

* C6：二维窗 474740 次约束评估、5.343 s 单核（≈ 88860 评估/s），判据「5·10⁴ 次评估 ≤ 60 s」对应0.563 s。
* C9：三窗九条件下探测开销相对「每格 1 点」：

| 窗 | 条件 | 默认评估数 | 单点评估数 | 开销倍数 | suspect_spans |
|---|---|---|---|---|---|
| exchange_rate_window | exchange_open_sea | 10811 | 2815 | 3.84 | 0 |
| exchange_rate_window | exchange_coastal | 10822 | 2814 | 3.85 | 0 |
| exchange_rate_window | exchange_estuary | 10779 | 2771 | 3.89 | 0 |
| stranding_tide_window | transit_full | 5384 | 1382 | 3.90 | 0 |
| stranding_tide_window | transit_medium | 5384 | 1382 | 3.90 | 0 |
| stranding_tide_window | transit_light | 5382 | 1382 | 3.89 | 0 |
| sts_transfer_window | sts_calm | 10898 | 2882 | 3.78 | 0 |
| sts_transfer_window | sts_moderate | 10871 | 2867 | 3.79 | 0 |
| sts_transfer_window | sts_rough | 10841 | 2833 | 3.83 | 0 |

最大开销 3.90×（门限 40×），且三窗一次细分都没触发（它们的窗边界本来就宽于节点间距）。

## ④ 结果图

![subgrid_baseline_vs_fixed.png](../../../artifacts/envelope_n09/figures/subgrid_baseline_vs_fixed.png)

*Three windows narrower than the node spacing. The analytic window is shaded blue, the v1 single-midpoint claim in orange (the whole span, or nothing), the fixed engine's bounds are the dark vertical lines.*

![probe_spacing_vs_error.png](../../../artifacts/envelope_n09/figures/probe_spacing_vs_error.png)

*Relative error of the recovered pocket bounds against the probed spacing for 1, 2, 3, 4, 8, 16, 32 probes per scan cell: coarser than the window the answer is wrong or absent, finer it is exact to rounding.*

![channel_area_vs_lines.png](../../../artifacts/envelope_n09/figures/channel_area_vs_lines.png)

*Diagonal channel |beta - alpha| <= 0.002 (analytic area 3.996e-3). Below one line spacing per channel width the window is found and flagged area-unreliable; above it the area converges to the analytic value.*

![probe_cost.png](../../../artifacts/envelope_n09/figures/probe_cost.png)

*Constraint evaluations per one-dimensional window against the probe density, for a window with and without a sub-grid pocket: the cost is linear in the probes and the refinement adds a bounded amount on top.*

## ⑤ 注意点、局限与适用范围

* **保证是「相对于探测间距」的，不是绝对的。** 引擎保证 τ 以上的特征不漏；τ 以下的黑盒采样原则上仍可漏（C8 第三行：真窗宽 0.0014 < τ=0.0105 时，每格 1 点的探测依旧丢窗，但不谎报——要么登记 unresolved，要么状态为 infeasible）。要更强，就让案例声明 `min_feature_width`，让契约在 τ 不够细时失败。
* **2 维的面积分辨力受线间距限制。** 比线间距还窄的对角通道：状态判为可行（不再谎报不可行），但每个扫描行自成一个单行分量，面积为 0 且被标 `area_reliable=False`；把线数加密到每通道宽至少一条线后面积才收敛到解析值（见图 3）。
* **探测密度与成本是线性关系**：默认每扫描格 8 点，1 维窗评估数随探测密度线性增长，三窗实际开销 3.78–3.90×；对小预算场景可降到 2–4 点，代价是保证间距变粗。
* **引擎不再用「把网格加密到结构复现」来掩盖不确定结构**：一旦某段被登记为 unresolved，`window_1d` 直接返回该结果而不再升级网格（升级网格会改变 τ 却不留痕）。
* **物理量本身未被验证**：本节点的判据全部是算法与几何判据（解析窗、闭式代数、守恒的空窗语义），三窗里的经验系数是配置数据；把它们对模型试验做标定不在本节点范围。

## ⑥ 误差判定

| 关键量 | 基准 | 实测误差 | 是否 < 3 % |
|---|---|---|---|
| 三窗边界/宽度（C1，27 个数） | 闭式分窗路径 | 3.608e-15 | 是（好 12 个数量级） |
| 三窗默认 vs 单点策略（C10） | 逐位相等 | 0 处不一致 | 是（精确相等） |
| 子网格窗边界/宽度（C7） | 解析窗 | 3.331e-13 | 是（好 11 个数量级） |
| 边界重建阶次（C2） | 解析抛物带 | 阶 2.000 | 是（二阶） |
| 二维面积（C6） | 闭式圆面积 | 1.527e-04 | 是 |
| 探测开销（C9） | 每格 1 点 | 3.90× | —（成本项，非精度项） |

本节点没有任何量对实验或实船数据比较，因此不宣称物理精度；表中的「< 3%」是相对解析/闭式基准的回答，物理相关式的标定仍为开放项（见 ⑤）。

## 审计链

* 代码摘要（包内源码 SHA-256）：`f18699d3210bb2a4f9b86e446e2e39f4de18a78950cc3c832540799a66142ca6`
* 迁移三窗摘要：`exchange_rate_window` = `146740aa98bc00b8…`，`stranding_tide_window` = `84a70cb5fccd415e…`，`sts_transfer_window` = `354f4397206b9bd2…`
* 证据工件：`artifacts/envelope_n09/criteria_evidence.json`
* 环境：Python 3.11.6，loongarch64，Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38，整轮 120.4 s
* 复现：`python scripts/envelope_subgrid_acceptance.py --out artifacts/envelope_n09` 然后 `python scripts/generate_envelope_subgrid_report.py`
* 测试：76 passed, 1 warning in 40.43s
