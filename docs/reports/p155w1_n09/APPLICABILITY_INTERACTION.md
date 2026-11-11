# 适用域机器包络的联合（多轴）探测：把「离线门在角点上写 REAL」的漏洞堵上

节点 `P155W1-N09` · 仓 `TensorPhysics` · 模块 `tensorphysics.applicability.interaction` · 配置 `src/tensorphysics/applicability/configs/default_capabilities.v1.json`（sha256 `a3daea28b05f361a`）

证据工件：`artifacts/applicability_interaction/evidence.json`（本文件所有数字由 `tools/make_applicability_interaction_report.py` 从探测运行注入，零手抄）；图：`artifacts/applicability_interaction/figures/`。

## ① 案例介绍（物理背景 / 参数）

适用域包络机制回答的是「这个能力的数在哪个参数范围内还算数」。已交付的机制沿**单轴**扫描：扫某一轴时，其余轴固定在其标称值。对真实能力，标称值几乎总是取在「工作正常」的一侧，于是两个参数**同时**变坏才会出现的失效，单轴扫描看不到：两个一维区间都是满的，离线门（`artifact.gate`，按设计不重跑评估算子）就把越界点写成 `REAL`。这不是精度问题，而是 fail-closed 契约被绕过——越界点被判为合格。

三个合成能力把这种「只在联合方向上失效」的闭式真值写下来（全部为无物理语义的代数标量）：

| capability | 观测量 | 判据 | 标称点 | 闭式真值（失效集） |
|---|---|---|---|---|
| `syn-joint-product-corner` | product | \|value\| ≤ 4 | — | value = u*v, tolerance 4, benign nominal (0.5, 0.5): the failure region is the hyperbolic corner u*v > 4, which neither 1-D scan can see |
| `syn-joint-sum-diagonal` | sum | \|value\| ≤ 6 | — | value = a+b, tolerance 6, nominal (0, 0): the failure region is the half plane a+b > 6, a diagonal that lies entirely between the two 1-D intervals |
| `syn-joint-ratio-hyperbola` | hyperbola | \|value\| ≤ 1 | — | value = s*t/(s+t), tolerance 1, nominal (0.05, 0.05): 1-D scans pass everywhere, while a curved band of the plane fails jointly |

参数表（全部由配置注入，代码中没有任何轴名/量程/阈值）：

| capability | axis | searched span | 1-D in-domain interval | samples (joint) | boxes | witnesses |
|---|---|---|---|---|---|---|
| `syn-joint-product-corner` | `u` | [0.5, 4] | [0.5, 4] | 13 | 102 | 388 |
| `syn-joint-product-corner` | `v` | [0.5, 8] | [0.5, 8] | 13 | 102 | 388 |
| `syn-joint-sum-diagonal` | `a` | [0, 6] | [0, 6] | 13 | 78 | 300 |
| `syn-joint-sum-diagonal` | `b` | [0, 6] | [0, 6] | 13 | 78 | 300 |
| `syn-joint-ratio-hyperbola` | `s` | [0.05, 5] | [0.05, 5] | 13 | 69 | 258 |
| `syn-joint-ratio-hyperbola` | `t` | [0.05, 5] | [0.05, 5] | 13 | 69 | 258 |

## ② 怎么做的（方法 / 流程 / 关键设置）

**阶 2b 联合探测算子**（`interaction.py`）。对每个轴对，用与单轴相同的采样律（Chebyshev-Lobatto 几何梯度）建 n×n 格点，其余轴固定在标称；**见证点（witness）** 的定义是「两个一维区间都判它在域内，而评估算子判它失败」——正是离线门会误判为合格的点。见证点落在格胞（相邻格点围成的矩形）内时，整个格胞被标成域外**保守胞**：格胞平铺整个搜索面，因此相邻见证之间不会留缝；胞内必定含有一个真实失效点，所以「胞内判 OUT」永远不会放过坏点，代价是边界处至多一格厚的过度拒绝（over-refusal），它随格距线性收缩。

**阶 3b 见证射线二分**。每个保守胞再从其最近的「通过点」沿直线二分到见证点，收敛到参数区间的 `1e-3`（与单轴同标准），给出联合边界点与置信度（`boundary` 字段；为控成本按 `max_rays` 截断，截断事实写进 `notes`，保守胞不受截断影响）。

**阶 4 判定接线**。`ApplicabilityEnvelope.classify` 在单轴全 IN 之后追加一次联合检查：点落入任一保守胞即 `domain_status=OUT`，`grade` 强制为 `OUT_OF_DOMAIN`，`REAL` 被拒；`Verdict.axis_status['interaction']` 记录轴对、胞界、见证点与失效类型。联合区随工件（`interactions` 字段）落盘并参与 `suite_digest`，离线消费端重放同一判定。

**零领域语义**：轴名、量程、采样数、`max_pairs`/`max_rays`、评估算子全部来自配置；模块与测试只用合成域名（`u`/`v`、`a`/`b`、`s`/`t`）。

配置新增块（缺省即启用，可用于关断以复现缺陷基线）：

```json
{
  "enabled": true,
  "samples": 13,
  "max_pairs": 3,
  "max_rays": 24
}
```

## ③ 计算结果（判据数字）

验收判据 **C6**（本节点新增，`verification.verify_c6_interaction_coverage`）：在 61×61 闭式真值格点上，只用一维包络的离线门（基线）**认证的失效点数**与联合包络（成熟态）认证的失效点数对比；要求成熟态为 0、基线 > 0、成熟态过度拒绝 ≤ 10% 搜索面、且随格点加密单调下降。

判定：**PASS**，成熟态误认证 = 0，基线误认证 = 6425。

| capability | 基线误认证 (占比) | 成熟态误认证 | 成熟态过度拒绝（面占比） | coarse→fine 过度拒绝 |
|---|---|---|---|---|
| `syn-joint-product-corner` | 2754/2754 (100.00%) | 0/2754 | 7.61% | 11.96% → 3.28% |
| `syn-joint-sum-diagonal` | 1830/1830 (100.00%) | 0/1830 | 6.32% | 8.57% → 3.57% |
| `syn-joint-ratio-hyperbola` | 1841/1841 (100.00%) | 0/1841 | 8.57% | 14.59% → 5.62% |

格点加密的收敛性（同一个能力的联合扫描代价同步上升）：

| capability | lattice | boxes | joint evaluations | over-refusal [% box] | false-IN |
|---|---|---|---|---|---|
| `syn-joint-product-corner` | 9 | 48 | 529 | 11.96 | 0 |
| `syn-joint-product-corner` | 13 | 102 | 865 | 7.61 | 0 |
| `syn-joint-product-corner` | 25 | 385 | 2641 | 3.28 | 0 |
| `syn-joint-sum-diagonal` | 9 | 36 | 529 | 8.57 | 0 |
| `syn-joint-sum-diagonal` | 13 | 78 | 865 | 6.32 | 0 |
| `syn-joint-sum-diagonal` | 25 | 300 | 2641 | 3.57 | 0 |
| `syn-joint-ratio-hyperbola` | 9 | 33 | 529 | 14.59 | 0 |
| `syn-joint-ratio-hyperbola` | 13 | 69 | 865 | 8.57 | 0 |
| `syn-joint-ratio-hyperbola` | 25 | 260 | 2641 | 5.62 | 0 |

**真实能力不变性（机制边界不宽于人工声明）**：三个已注冊（上账）真实能力均为单轴能力，联合探测对其不产生任何区域，一维包络逐字段不变（见下表）；`declaration_compat` 仍为不宽于人工声明。

| capability | axes unchanged | interaction regions | evaluations before → after | compat verdict |
|---|---|---|---|---|
| `multigrid-contraction-vs-damping` | yes | 0 | 15 → 15 | damping:NOT_WIDER |
| `stationary-iteration-stability` | yes | 0 | 15 → 15 | relaxation:NOT_WIDER |
| `stationary-solve-error-vs-grid` | yes | 0 | 12 → 12 | grid_n:CONSERVATIVE |

**离线门话语记录（抽样 1/20 真值格点）**：

| capability | 真值失效点数 | 基线写 REAL 数 | 成熟态写 REAL 数 |
|---|---|---|---|
| `syn-joint-product-corner` | 9 | 9 | 0 |
| `syn-joint-sum-diagonal` | 6 | 6 | 0 |
| `syn-joint-ratio-hyperbola` | 8 | 8 | 0 |

## ④ 结果图

1. `figures/joint_plane_before_after.png` — 同一个能力的参数平面：左为单轴包络（离线门全绿，包含真实失效区），右为联合包络（保守胞覆盖失效区，离线门拒绝）。绿=允许 REAL，红=OUT_OF_DOMAIN，深蓝框=保守胞，黑点=闭式真值失效集。
2. `figures/convergence_and_cost.png` — 过度拒绝随格点加密下降（左）与联合扫描代价（右），三个合成能力六条曲线。
3. `figures/gate_transcript.png` — 离线门在真值失效点上的 `REAL` 写入次数，修复前后对比。

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **残留别名效应（honest limitation）**：联合探测不是连续证明。保守胞只能保证「胞内含已采样的失效点」，失效区若薄于一个格胞（叶宽 < 格距）可能采样不到而漏检；本报告的 false-IN=0 是在 61×61 闭式真值格点上验证的，不是连续意义上的证明。降低风险的办法是加密 `samples`（代价见表 3）或对被判 IN 的关键点再用 `classify(..., verify=True)` 重算。
* **代价**：过度拒绝是有意付出的代价（边界一格厚）；它随格距线性下降，本报告给出 coarse→fine 的实测曲线，不做「零代价」的虚假声明。
* **轴对数上限**：`max_pairs` 默认 3，超出的轴对不扫（高维能力需分组），且顺序由声明顺序决定以保证可复现。
* **序列律不参与联合二分**：单调性律是序列性质，只能在扫描序列上判；联合面只做逐点判据/病态/符号律。首末两个端点在真实孤立点效应处仍按采样分辨率报告。
* **修复不移动单轴结果**：三个真实能力的一维包络逐字段不变（上表），因此历史上已发布的数不受影响。

## ⑥ 误差判定（关键量 vs 基准，逐项回答是否 <3%）

| 关键量 | 本报告值 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 联合边界（见证射线二分）剩余区间宽度 | ≤ 1e-3·射线参数 | 判据要求 1e-3 | ≤ 1.0×10⁻³ | 是（判据本身） |
| `syn-joint-product-corner` 成熟态误认证率 | 0 | 闭式真值 0 | 绝对差 0 | 是 |
| `syn-joint-product-corner` 基线误认证率（缺陷幅度） | 100.00% | 闭式真值 0 | 100.00% | 否（这正是被修复的缺陷） |
| `syn-joint-sum-diagonal` 成熟态误认证率 | 0 | 闭式真值 0 | 绝对差 0 | 是 |
| `syn-joint-sum-diagonal` 基线误认证率（缺陷幅度） | 100.00% | 闭式真值 0 | 100.00% | 否（这正是被修复的缺陷） |
| `syn-joint-ratio-hyperbola` 成熟态误认证率 | 0 | 闭式真值 0 | 绝对差 0 | 是 |
| `syn-joint-ratio-hyperbola` 基线误认证率（缺陷幅度） | 100.00% | 闭式真值 0 | 100.00% | 否（这正是被修复的缺陷） |

**结论**：本节点的关键量是「越界点被写成 REAL 的次数」。以闭式真值为基准，成熟态在 61×61 格点 × 3 个合成能力（共 11163 个格点，其中真值失效点 6425 个）上为 **0/6425**（0%），基线为 6425 次（真值失效点全部误认证，100%）；过度拒绝 8.57% 搜索面（上限 10%），并随格点加密降至 5.62%。真实能力的一维边界与修复前逐字段相同（相对变化 0%），满足「机制边界不宽于人工声明」。

## 审计链

* 配置：`src/tensorphysics/applicability/configs/default_capabilities.v1.json` sha256 `a3daea28b05f361a0d51663f8818d658882a1ec4f9b8a9d9087d4616cf132cab`
* 合成能力注册表：`tensorphysics.applicability.interaction.synthetic_interaction_registry`（3 个联合算例，闭式真值见 `INTERACTION_CASES`）
* 包络摘要 `suite digest` `syn-joint-product-corner` = `360a5aa63ee389c3`
* 包络摘要 `suite digest` `syn-joint-sum-diagonal` = `4e91cbf5210e1728`
* 包络摘要 `suite digest` `syn-joint-ratio-hyperbola` = `36bbc1645af0796f`
* 判据 C6 详细：`evidence.json.criterion_c6`（含每个能力的 baseline/matured/coarse/fine 四组数字）
* 离线门话语：`evidence.json.cases[*].transcript`（逐点 value/真值/前后判级）

## 复现

```bash
PYTHONPATH=src python tools/make_applicability_interaction_report.py
PYTHONPATH=src python -m pytest tests/test_applicability_interaction.py -q
PYTHONPATH=src python -c "from tensorphysics.applicability.verification import verify_c6_interaction_coverage as c; print(c().passed)"
```
