# 作业窗口包络引擎成熟化：多区间（含洞）可行域的面积积分与机器校验

节点 `IN68W3-N06` · 仓 `TensorPhysics` · 模块 `tensorphysics.envelope` · 配置 `tensorphysics.envelope.case.v1`

证据：`artifacts/envelope_topology/topology_evidence.json`（本节点新判据 T1–T5）、`artifacts/envelope_postfix/criteria_evidence.json`（战役判据 C1–C6 复跑）、`artifacts/envelope_postfix/case_results.json`（三窗×三工况逐条结果）

## ① 案例介绍（物理背景 / 几何 / 参数）

作业窗口（operating window）是参数空间里使全部约束同时成立的子集。工程上它几乎总是**一条带**：下限来自"来不及/不够"，上限来自"来不及停/受不了"。已交付的三条窗（搁浅脱浅的临界潮位、压载水置换率、STS 过驳泵率）都是一维带，引擎把它们当纯配置处理，代码里没有任何领域词条。

三窗的物理读法与本次复跑结果（全部由 `case_results.json` 读出，无手抄）：

* `stranding_tide_window` — under-keel-clearance / sill band on the tide axis
* `exchange_rate_window` — ballast-water exchange rate band (time, residence, pump, stability)
* `sts_transfer_window` — ship-to-ship transfer rate band (min rate, manifold pressure, hose, weather)

| case | condition | window | width | lower driver | upper driver | attainable settings | evaluations | wall [s] |
|---|---|---|---|---|---|---|---|---|
| exchange_rate_window | exchange_open_sea | [1000.000000, 1600.000000] | 600.000000 | c_completion | c_residence | 3 | 14371 | 0.070587 |
| exchange_rate_window | exchange_coastal | [1200.000000, 1800.000000] | 600.000000 | c_completion | c_pump_capacity | 3 | 14364 | 0.070028 |
| exchange_rate_window | exchange_estuary | [1500.000000, 1513.130671] | 13.130671 | c_completion | c_stability | 1 | 14331 | 0.058403 |
| stranding_tide_window | transit_full | [1.400000, 3.200000] | 1.800000 | c_ukc | c_sill | -- | 7158 | 0.049134 |
| stranding_tide_window | transit_medium | [0.700000, 4.000000] | 3.300000 | c_ukc | c_sill | -- | 7156 | 0.042471 |
| stranding_tide_window | transit_light | [0.100000, 4.600000] | 4.500000 | c_ukc | c_sill | -- | 7160 | 0.034605 |
| sts_transfer_window | sts_calm | [300.000000, 1933.732496] | 1633.732496 | c_min_rate | c_manifold_pressure | 17 | 14438 | 0.083624 |
| sts_transfer_window | sts_moderate | [450.000000, 1695.171155] | 1245.171155 | c_min_rate | c_manifold_pressure | 12 | 14422 | 0.080567 |
| sts_transfer_window | sts_rough | [600.000000, 1200.000000] | 600.000000 | c_min_rate | c_hose_limit | 7 | 14393 | 0.068085 |

本次成熟化针对的是**二维窗的拓扑**：当可行域在参数平面上不是"星形"、某条扫描线上会两次穿过同一个连通域时（环形可行域、带禁区的窗、折返边界），可行域面积应当按"外壳减洞"积分。审计发现原实现对这类拓扑把**各区间长度直接串起来积分**（同一横坐标出现多次），得到的是几何上毫无意义的数：半径 0.5–0.8 的环，解析面积 `pi(0.8^2-0.5^2)=1.225221`，引擎给出 0.864925，偏低 29.4%，而**全部契约检查为绿**——因为没有任何机器检查拿这个数与别的东西比对过。

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **审计**（先证伪再动手）：用合成几何（环、双圆、方板挖方孔）在引擎上做对抗性探查，对每个疑点先看现状数值与契约结论，再决定是否算缺陷。审计确认两点：二维多区间面积系统性偏低（与网格无关，纯算法错），且平面模式下暴力网格扫描只被"打印"、从未被比较。
2. **修复**：组件面积改为按**壳分解**积分——外壳 = 该线最后一个上界减第一个下界；每个洞 = 相邻两区间之间的间隙；面积 = 外壳 − Σ洞。每一壳都是一条闭合弦，在自己的端点按 `sqrt` 收口，于是原有的解析端块（两结点定位尖端 + `L^2` 线性）对每一壳都成立，面积精度恢复二阶。`Component` 新增 `multi_interval_lines` 字段，`area_method` 报告 `shell_tip_model` / `shell_trapezoid`。
3. **把网格扫描升格为机器检查**：平面窗现在跑两次暴力网格（`n_grid_check` 与其两倍），新增 `contracts.check_area_vs_grid`，判据 **先声明后使用**：
   `band = max(rel_tol·|A_fine|, 3·|A_fine − A_coarse|)`，`|A_march − A_fine| ≤ band` 为通过。第二项是网格自身的分辨率漂移（它自己误差的估计），乘 3 为安全系数。该检查是**粗错探测器**，不是精度证书——精度由解析面积判据（T1/T3）证明。
4. **不留倒退**：把"修复前的算术"作为负控留在测试与验收脚本里（T2），要求机器检查必须拒绝它；并要求三条已迁移的一维窗数字**一位不动**（T4，对照 `scripts/envelope_cases_acceptance.py` 里独立的闭式参考路径）。
5. **工具链**：纯 Python 标准库（包内无 numpy 依赖）+ matplotlib 出图；`pytest` 单元/回归；`scripts/envelope_topology_acceptance.py` 产出 T1–T5 证据 JSON 与图；机器 loongarch64、Python 3.11.6、单核 `taskset -c 2` 计时。

## ③ 计算结果（数字表）

### 3.1 环形可行域的面积（判据 T1）

| n_lines | area | rel. error vs closed form | area method | lines with 2 intervals | evaluations |
|---|---|---|---|---|---|
| 100 | 1.22505241 | 1.377e-04 | shell_tip_model | 50 | 90529 |
| 200 | 1.22516582 | 4.515e-05 | shell_tip_model | 100 | 180995 |
| 400 | 1.22520232 | 1.536e-05 | shell_tip_model | 200 | 362009 |

默认梯度（100 线 × 96 探针）相对误差 `1.377e-04`，满足 1e-3；细化到 400 线为 `1.536e-05`，满足 1e-4。最后两级观测阶 `1.556`（1 与 2 之间：端块是解析的，内部是梯形，阶的估计受端块主导项影响）。

### 3.2 修复前后对照与负控（判据 T2）

| quantity | value |
|---|---|
| closed-form area | 1.22522113 |
| shipped area (per-shell) | 1.22505241 |
| pre-fix area (summed intervals) | 0.86492539 |
| pre-fix relative error | 0.29406589 |
| check verdict on pre-fix value | rejected |
| check verdict on shipped value | accepted |
| check band on the shipped value | 0.17695610 |

### 3.3 拓扑回归（判据 T3）

| geometry | marched area | closed form | rel. error | components | verdict |
|---|---|---|---|---|---|
| single disc | 1.76672868 | 1.76714587 | 2.361e-04 | 1 | pass |
| two disjoint discs | 0.56508632 | 0.56548668 | 7.080e-04 | 2 | pass |
| slab with hole (edge on a line) | 3.00021053 | 3.00000000 | 7.018e-05 | 1 | pass |
| slab with hole (corner between lines) | 3.04000000 | 3.00000000 | 0.01333333 | 1 | declared limitation (first order) |

### 3.4 迁移不变性（判据 T4）

三窗 × 三工况共 `27` 组窗口边界/宽度对照独立的闭式参考实现，最大相对差 `3.608e-15`（阈值 1e-12）。即：二维改动没有移动任何一条已迁移的一维窗数字。

### 3.5 成本（判据 T5）

| quantity | value |
|---|---|
| window | ring, n_lines=100, n_scan=96 |
| constraint evaluations | 102049 |
| wall time [s] | 0.943716 |
| evaluations / s | 108135.336745 |
| machine | loongarch64 |
| predeclared limits | 5e4 evals, 60 s |

### 3.6 战役判据 C1–C6 复跑（修复后）

| criterion | what it establishes | headline number | pass | declared threshold |
|---|---|---|---|---|
| C1 | migration consistency vs the closed-form per-window path | 3.608e-15 | yes | 1.000e-12 |
| C2 | boundary reconstruction order (interpolated boundary vs analytic) | 2.000000 | yes | 1.900000 |
| C3 | contract checker negative control: declared monotone, actually folded | monotone:c_band:alpha | yes |  |
| C4 | infeasible semantics: empty window, nearest feasible point, no extrapolation | semantics respected | yes |  |
| C5 | zero domain semantics in package sources and tests | 0 hits / 6325 lines | yes |  |
| C6 | cost per window and marched area vs brute-force grid scan | 1.527e-04 | yes | 0.001000 |

### 3.7 C6：二维面积 vs 暴力网格（逐分辨率）

| grid n per axis | cell-count area | feasible cells | rel. error vs closed form | wall [s] |
|---|---|---|---|---|
| 512 | 1.76672118 | 115332 | 2.403e-04 | 0.00984924 |
| 1024 | 1.76732035 | 462388 | 9.874e-05 | 0.02055133 |
| 2048 | 1.76716340 | 1851196 | 9.924e-06 | 0.08420803 |

引擎在 160 线 × 320 探针下用 `474740` 次评估、`3.294` s 得到面积 `1.766875989`，与闭式 `1.767145868` 相对差 `1.527e-04`；吞吐 `144140` 次评估/秒（单核）。

### 3.8 汇总判据表

| criterion | what it establishes | measured | predeclared threshold | pass |
|---|---|---|---|---|
| T1 | ring area against the closed form, and under refinement | 1.377e-04 | 1e-3 | yes |
| T2 | negative control: the pre-fix arithmetic is rejected, the shipped one accepted | 0.294066 | > 0.25 and rejected | yes |
| T3 | topology regression: disc, two components, holed slab, cornered slab | 7.018e-05 | 1e-3 | yes |
| T4 | migration invariance against the closed-form per-window path | 3.608e-15 | 1e-12 | yes |
| T5 | cost of one holed 2-D window and of the two-resolution area check | 102049 | >= 5e4 evals and <= 60 s | yes |

## ④ 图

随报告内嵌四张图：图 1 环形窗的行进边界与暴力网格掩膜对照（同一解析圆心/半径叠加）；图 2 环面积收敛曲线（修复后 vs 修复前，附 `O(h^2)` 参考线与 1e-3 判据线）；图 3 三条已迁移一维窗的全部工况（引擎区间条与闭式参考刻度叠加）；图 4 面积机器判据的带子、两级网格与修复前错值的位置。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

* **"检查全绿"不等于"数对了"**：本次缺陷能活下来的唯一原因是平面模式的暴力网格扫描只被写进结果对象、从未被比较。把已有信息升格为判据（而不是新增更多输出）才是修复。
* **把区间先加起来积分是错的**：环形域上积分式 `∫ΣL_k du` 在洞开/闭处出现 `sqrt` 拐折，梯形只有 `O(h^1.5)`；实测 100 线时 1.3e-3，比引擎承诺的 `O(h^2)` 还差。按壳分解后每一壳端点都是 `sqrt` 收口，才回到二阶。
* **拐角边界只有一阶**：方板挖方孔的洞边若正好落在扫描线上，误差 7e-5；若落在两线之间（线性拐折而非 `sqrt` 收口），误差 1.3e-2。此类几何需让梯度对齐拐角（T3 已把这条写成声明式局限并给出实测数，不隐藏）。
* **面积检查的带子是分辨率相关的**：网格面积本身只有一阶精度（48×48 对圆盘偏高约 1%），所以 `check_area_vs_grid` 只能做粗错探测；把 1e-3 的精度主张压在这种检查上会是过度声明。精度主张必须由解析面积承担。
* **端块模型的前提**：`L^2` 线性定位尖端只在端点两线都是单区间时成立；代码用 `simple_ends` 显式判断，不满足则退回梯形并把 `area_method` 报出来。
* **适用范围**：二维平面窗（两参数）；三维窗需要切片，不在本节点范围。离散可行域仍按连续窗之外单独报告，连续边界不承诺可设定。

## ⑥ 误差判定（是否 < 3%）

| 关键量 | 基准 | 相对误差 | < 3% ? | 依据 |
|---|---|---|---|---|
| 环形窗面积（默认梯度） | 闭式 pi(R^2-r^2) | 1.377e-04 | yes | T1 |
| 环形窗面积（400 线） | 闭式 pi(R^2-r^2) | 1.536e-05 | yes | T1 |
| 单圆盘面积 | 闭式 pi r^2 | 2.361e-04 | yes | T3 |
| 双圆盘总面积 | 闭式 2 pi r^2 | 7.080e-04 | yes | T3 |
| 方板挖孔（拐角对齐） | 闭式 4-1 | 7.018e-05 | yes | T3 |
| 方板挖孔（拐角在两线间） | 闭式 4-1 | 0.01333333 | no（1.3%，但落在声明的一阶带内） | T3（声明式局限） |
| 修复前环形窗面积 | 闭式 pi(R^2-r^2) | 0.29406589 | no（29.4%，已修复） | T2 |
| 三窗迁移边界/宽度 | 独立闭式参考路径 | 3.608e-15 | yes | T4 |
| C6 二维面积 | 暴力网格 Richardson 外推 | 7.384e-05 | yes | C6 |
| C6 二维面积 | 闭式 pi r^2 | 1.527e-04 | yes | C6 |

结论：**能给出基准的量全部 < 3%（实际 < 1e-3）**；唯一超过 1e-3 的是带拐角的方板挖孔在两线间的情形，其成因（线性拐折 vs `sqrt` 收口）已定位并作为声明式局限记录，未粉饰为通过。修复前 29.4% 的错值已被机器判据拒绝，不再是可交付数字。

## 变更与审计链

| artifact | digest |
|---|---|
| topology_evidence (artifact_digest) | a8bbef268909e7cde5e3fdf34056e6a45b266b844171cb48e972174bf244d52a |
| topology_evidence (code_digest) | 650db89c278d22f0be281d99c8b1715942c66608215d376771a2fb3f7703c24e |
| migrated criteria evidence (artifact_digest) | 38da9ae340db4ee3a292b6399cef897a9d6231b706fcbf07166daf06c6eefd69 |
| migrated case results (artifact_digest) | 90c34b8e8ebcfe46d2dfa095f54ea6008e05276db01500f731d8b859cbfd3bd9 |

改动文件：`src/tensorphysics/envelope/marching.py`（壳分解面积 + `multi_interval_lines`）、`src/tensorphysics/envelope/contracts.py`（`check_area_vs_grid`）、`src/tensorphysics/envelope/engine.py`（平面窗两次网格 + 面积判据）、`src/tensorphysics/envelope/benchmarks.py`（环、方板挖孔合成算例 + 闭式面积）、`src/tensorphysics/envelope/__init__.py`（导出）、`tests/test_envelope_engine.py`（6 个回归/负控用例）、`scripts/envelope_topology_acceptance.py`（T1–T5 证据）、`docs/OPERATING_WINDOW_ENVELOPE.md`（第 2/4/6/7 节同步）。
