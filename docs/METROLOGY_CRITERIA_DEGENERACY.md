# 判据度量学：相对误差判据的三类退化与自适应判据族

**节点** P266W2-N09 · **仓** TensorPhysics · **模块** `tensorphysics.metrology.criteria` · **语料** 仓库历史判据（只读回放）

## ① 案例介绍：为什么“相对误差 < τ”这句话不是一个判据

平台里几乎所有验收判据都写成一句自然语言：`相对误差 < 2%`。这句话缺少四样东西中最关键的一样——**基准量的量级**。同一个字符串，在 `y_ref = 1.0` 时是 2% 的物理容差，在 `y_ref = 1e-9` 且被测量为 `1e-3` 时是 1e9 倍的噪声放大；在 `y` 与 `y_ref` 反号时它永远 ≥ 100 %，无论阈值多宽；在同一量上并列两个口径时，谁赢取决于读报告的人，这就是“部分通过”的真正来源。

本节点把这类判据从字符串提升为带度量语义的结构 `(y, y_ref, τ, metric + unit + condition)`，把不可判的情形显式降级为 `FAIL`/`WITHHELD`，并对仓库历史判据做一次只读回放。

语料与规模（程序化注入，来自 `artifacts/m9942/criteria_replay.json → harvest`）：

| 项 | 数值 |
|---|---|
| 扫描 JSON 文档 | 217 |
| 扫描 markdown 文档 | 210 |
| JSON 来源判据记录 | 55 |
| markdown 来源判据记录 | 641 |
| 去重后回放判据 | 301 |
| 其中已归档结果的 | 129 |
| 仅归档判据、无结果的 | 172 |

**只读承诺**：`build_corpus` 只打开文件读取，不写任何领域工件；回放产物全部落在 `artifacts/m9942/`，且产物本身带 `schema` 戳被采集器显式跳过（见 ⑤ 踩坑 4）。

## ② 怎么做的：判据四元组 → 三型退化 → 自适应族 → 回放重判

**2.1 判据解析器**（`parser.py`）。用有序模式表把判据串解析为语义：

| 度量 | 形式 | 例 |
|---|---|---|
| `relative` | `|y−y_ref| / |y_ref|` | `相对误差 < 2%`、`|Δm|/Σm ≤ 1e-12`、`|w_fem−w_het|/w_het < 0.02` |
| `absolute` | `|y−y_ref|`（工程单位） | `max|A−B| ≤ 1e-12`、`eps_max < 1e-3` |
| `log_ratio` | `|ln(y/y_ref)|` | `|ln(y/y_ref)| ≤ ln(1+0.03)` |
| `signed` | `y − y_ref`（保号） | `带符号偏差 ≤ 0.5 kN` |

`extract_quadruple()` 组装 `(y, y_ref, τ, metric+unit+condition)`；`parse_goal_criteria()` 处理任务书里的 `≥90%`、`≥200 条`、`12/12`、`≥1000 个随机样本` 形态。解析失败不抛异常，返回 `UNPARSED` 哨兵，下游因此只能 `WITHHELD`——**解析失败永远不等于通过**。

**2.2 三型退化分类器**（`degeneracy.py`）。每型三值判定 `hit / clear / undecidable`：

| 型 | 判据 | 说明 |
|---|---|---|
| T1 基准趋零 | `|y_ref| ≤ 3σ` 或 `|y_ref| < 0.1·scale` | σ 缺省 `1e-6·scale`（六位有效数字的工程报告精度） |
| T2 跨零/变号 | `sign(y)·sign(y_ref) < 0` 或 `y_ref = 0 ∧ y ≠ 0` | 仅对分母归一/保号度量有意义 |
| T3 多口径无优先级 | 同一量族并列 ≥2 种度量且无 `priority` 字段 | “部分通过”的机器成因 |

三值是关键设计：证据缺失（只归档了相对误差、丢掉了基准量级）判 `undecidable`，与 `hit` 一起计入 `flagged`，**缺失证据不能被当成健康判据**。

**2.3 自适应判据族**（`adaptive.py`），四个成员都有界：

```
scaled_delta   |y − y_ref| / max(|y_ref|, 0.1·scale, 3σ)         量级归一，分母不再塌陷
log_ratio      |ln(y / y_ref)| ≤ ln(1 + τ)                       对称带，高估/低估同权
signed_abs     |y − y_ref| ≤ τ·scale（工程单位）                  保号，变号不可能通过
merge          优先级 absolute → log_ratio → relative；任一强制口径 FAIL 即 FAIL
```

合并规则的关键词是**禁止双标准**：只要有一个强制口径判 FAIL，整条记录就是 FAIL；结论词表被限制为 `PASS / FAIL / WITHHELD` 三元，`PARTIAL` 在数据结构里不可表达。
当工件只归档了相对误差 ε（无 `y_ref`）时，自适应族仍可判定：以对称带的最坏方向 `max(|ln(1+ε)|, |ln(1−ε)|)` 与 `ln(1+τ)` 比较——保守、方向无关，但不再让“只有相对误差”变成不可判。

**2.4 回放重判器**（`replay.py`）。每条记录同时给出：现行基线（朴素相对误差，`y_ref = 0` 时按 IEEE 语义得到 `inf` → FAIL）、三型退化、自适应判定、**决定性替代判据的数值与阈值**、`flip`（PASS↔FAIL）、`downgrade`（判定 → WITHHELD）、`upgrade`（WITHHELD → 判定）。

**2.5 负控与标注集**（`controls.py` + `data/controls.json`）。12 个退化样本（T1/T2/T3 各 4）+ 12 个良性样本（含 1e-6 级一致性、零基准但被测量也为零且用绝对带的合法情形）+ 2 个补充样本，外加 16 条补充标注凑成 40 条标注集，用于预注册的停止规则。

**2.6 工具链**。`python -m tensorphysics.metrology.criteria all --root . --out-dir artifacts/m9942` 一键产出回放/负控/标注三份 JSON；`artifacts/m9942/generate_figures.py` 出图；`build_acceptance_report.py` 把六条机器判据逐一落成 JSON。

## ③ 计算结果

**3.1 机器验收判据（全部来自 `artifacts/m9942/acceptance_report.json`）**

| 判据 | 阈值 | 实测 | 结论 | 证据 |
|---|---|---|---|---|
| (a) degeneracy classification agreement >= 90% aga | ≥ 90 % | 一致率 100.0 %，κ = 1（n=40） | **PASS** | `artifacts/m9942/criteria_labeled40.json` |
| (b) replay >= 200 historical criteria, >= 5% flagg | ≥ 200 条且 ≥ 5 % | 301 条，退化 206 条（68.4 %），附替代数值 70 条，结论变更 18 条 | **PASS** | `artifacts/m9942/criteria_replay.json` |
| (c) 4 typical cases with alternative numbers, dire | 4 例 / 4-4 | 4 例，方向一致 4/4，偏差 > 50 % 的 2 例全部显式 FAIL | **PASS** | `artifacts/m9942/criteria_replay.json` |
| (d) negative controls: 12/12 degenerate hit, 0/12  | 12/12 且 0/12 | 退化命中 12/12，良性误报 0/12 | **PASS** | `artifacts/m9942/criteria_controls.json` |
| (e) invariant: y_ref -> 0 never returns an unbound | ≥ 1000 | 1200 个随机样本，`y_ref → 0` 只给有限值或 WITHHELD | **PASS** | `tests/test_metrology_criteria_acceptance.py` |
| (f) pytest suite green, NaN/None propagation safe, | ≥ 80 % | 覆盖 95.34 %，pytest 80 项全绿 | **PASS** | `artifacts/m9942/coverage.json` |

**总体：PASS**

**3.2 回放分解**

| 量 | 数值 |
|---|---|
| 回放判据 | 301 |
| 判为退化/不可判 | 206（68.4 %） |
| 其中可证退化 T1/T2/T3（有数值证据） | 0 |
| 其中仅因证据缺失不可判 | 206 |
| 附带数值替代判据的退化条目 | 70（34.0 %） |
| 结论翻转 flip | 1 |
| 降级为 WITHHELD | 11 |
| 由不可判升级为可判 | 6 |
| 基线 PASS / FAIL / WITHHELD | 110 / 19 / 172 |
| 自适应 PASS / FAIL / WITHHELD | 108 / 16 / 177 |

读法：语料中**没有**一条记录能“可证”T1/T2（`n_degraded_provable = 0`）——不是因为判据都健康，而是因为仓库历史上几乎从不把 `y` 与 `y_ref` 一起归档，只有一句相对误差。可证退化由负控样本 12/12 证明（它们带完整数值）；真实语料里能证明的是**证据缺口**本身：206 条判据在归档形式下已永久不可复核。

**3.3 四个典型案例（判据 c）**

| 角色 | 出处 | 判据 | 基线 | 自适应 | 替代判据数值 | 偏差/尺度 | 方向一致 |
|---|---|---|---|---|---|---|---|
| corpus_metric_misparse | corpus | `相对误差 < 2%` | FAIL | FAIL | log_ratio 2.1679 ≤ 0.0198026 | 7.74 | ✓ |
| sign_flip | controls | `相对误差 < 5%` | FAIL | FAIL | signed 2 ≤ 0.05 | 2 | ✓ |
| multi_metric_merge | controls | `相对误差 < 2%` | PASS | PASS | log_ratio 0.0009995 ≤ 0.0198026（族合并 FAIL；单条相对口径 PASS） | 0.001 | ✓ |
| legit_zero_reference | controls | `相对误差 < 1%` | FAIL | PASS | signed 0.005 ≤ 0.1 | 0.0005 | ✓ |

第一例取自真实工件 `benchmarks/icebreaking/results/validation_report.json` 的 Hetényi 锚：相对误差 7.74（774 %），基线与自适应族都判 FAIL——但**同一个字符串**若被 `|…| < τ` 模式误吃成绝对带，就会得到 PASS（见 ⑤ 踩坑 1），这条案例正是“判据是字符串”的缺陷证据。第二例 T2 变号：`y = −1` 对 `y_ref = +1`，相对判据结构上永远 ≥ 100 %，自适应族保号后判 FAIL。第三例 T3：相对口径 0.10 % 通过、绝对带 1e-3 > 1e-4 不通过，族合并给出单一 FAIL，消灭“部分通过”。第四例是近零基准（`y_ref = 1e-12`、`scale = 10`）：基线因 5e9 的相对误差判 FAIL，自适应族按工程带 0.1 判 PASS——这是一个 **FAIL → PASS 的翻转**，证明该族不会把“基准趋零”的近零情形一刀切误杀。

**3.4 负控矩阵（判据 d）**

| 类型 | 样本 | 命中 | 判定符合预期 |
|---|---|---|---|
| T1_zero_reference | 4 | 4 | — |
| T2_sign_flip | 4 | 4 | — |
| T3_multi_metric_no_priority | 4 | 4 | — |
| benign（含 1e-6 级一致性与合法零基准） | 12 | 误报 FAIL 0 | 结构性 flagged 0 |

**3.5 标注集与预注册停止规则**

- 40 条标注集：分类器与标注 A 一致率 **100.0 %**，Cohen κ = **1**；
- 盲文本标注器（只看判据字符串、看不到数值）一致率 60.0 %，κ = 0.229——**光凭字符串无法判断退化**，这正是本节点的论点；
- 停止规则：一致率 < 70 % 判度量族不可用；实测 100.0 %，触发 = False。

## ④ 结果图

![回放判定与退化分解](../artifacts/m9942/fig1_replay_verdicts.png)

*图 1* 左：同一批判据在基线（相对误差 < τ）与自适应族下的判定分布；右：退化分解——“可证退化 / 证据缺失不可判 / 有数值替代 / 结论变更”的规模差，是本节点最重要的定量结论。

![度量平面](../artifacts/m9942/fig2_metric_plane.png)

*图 2* 基线相对误差（横轴）与自适应决定性度量值（纵轴）的散点，虚线为 `y = x`，点线为各族阈值；红圈为 4 个案例。落在虚线之上且低于阈值带的点，就是“相对判据不明智、自适应判据通过”的区域。

![零基准不变量](../artifacts/m9942/fig3_zero_reference_invariant.png)

*图 3* 判据 (e) 的数值证据：固定 `y = 1`、`scale = 1`，扫描 `y_ref ∈ [1e-18, 10]`。相对误差（红）在 `y_ref → 0` 时无界发散；`scaled_delta`（蓝）始终有界；低于 3σ 噪声线的部分返回 `WITHHELD` 而不是 `inf`。

![负控矩阵](../artifacts/m9942/fig4_controls.png)

*图 4* 左：三型负控 12/12 命中；右：良性样本误报 0/12，并标注 40 条标注集的一致率与 κ。

## ⑤ 注意点与局限

**踩过的坑（全部已落成测试）**

1. **竖线比值被当成绝对量**：`|w_FEM − w_Hetényi| / w_Hetényi < 0.02` 最初被 `|…| < τ` 模式吃成“绝对值 0.02”，于是 774 % 的偏差被判 PASS。修法=在绝对模式之前插入 `quantity_ratio` 模式并加用例；这条坑说明“判据是字符串”本身就是缺陷面。
2. **百分号字符串值**：工件里 `"value": "0.2790%"` 与 `"limit": "< 0.5%"` 是字符串，朴素 `float()` 全部丢弃——补 `_parse_number` 后语料从 183 条涨到 200+ 条；同时必须防止“% 误差”去比“绝对带”（单位不匹配一律跳过）。
3. **行号不是量名**：表首列 `C1`、`锚2` 被当成 measurand 后，同表不同口径的行被误判 T3（假阳性 27 条）。修法=标识符型 measurand 不再构成量族。
4. **采集器吃自己的输出**：首次运行把 `artifacts/m9942/criteria_replay.json` 也当成历史判据扫描，语料自指增长（408 → 425）。修法=产物带 `schema` 戳，采集器显式跳过本包产物。
5. **`None` 与 `NaN` 的默认方向**：`nan < τ` 在 Python 里是 `False`，任何“NaN 即失败”的直觉都会把证据缺失误判为 FAIL；本实现让缺失走 `WITHHELD`，只有真的 `0/0` 才按 IEEE 语义记 FAIL。

**局限（诚实列出）**

1. 语料中**没有**可证 T1/T2 记录（`n_degraded_provable = 0`）：仓库历史工件几乎不归档基准量级，因此 206 条退化属“证据缺失不可判”，只有 70 条能给出数值替代判据（覆盖率 34.0 %）。可证 T1/T2/T3 的判别力由负控 12/12 承担，两者不可混为一谈。
2. 基线对历史“自报结论”的复现率只有 49.0 %（49/100）——说明历史结论本身就不能从归档的数值复算出来（容差缺失、口径漂移、结论与数字不同源）。这是发现，不是回放器的精度问题，但它也意味着“基线 vs 自适应”的翻转统计只在可复算子集上有意义。
3. 标注集由本节点构造（标注 A = 构造真值），因此 100 % 一致率是**实现正确性检查**，不是无偏精度估计；无偏证据是盲文本标注器（一致率见 3.5）与 200+ 条真实语料回放。
4. 无人类标注者参与（本节点为自主运行），“两人独立标注”以两条独立规则路径近似，混淆项（标注者偏见）无法真实测量，只能给出量级。
5. 判据语义依赖字符串模式表：新的措辞可能需要补模式；解析失败一律 `WITHHELD`，因此召回不足时表现为“不可判”增多，而不是错判为通过。
6. `noise = 1e-6·scale` 是工程报告精度的建模假设（六位有效数字），可由工件 `noise` 字段覆盖；对更高精度要求的场景应显式注入 σ。

## ⑥ 误差判定（逐项，含是否 < 3 %）

| 关键量 | 基准 | 实测 | 误差 | 是否 < 3 % |
|---|---|---|---|---|
| 退化分类一致率 | 40 条标注集（判定 A） | 100.0 %，κ=1 | 0.0 % | **是** |
| 退化负控命中 | 12 个构造样本 | 12/12 | 0 % | **是** |
| 良性误报 | 12 个构造样本 | 0/12 | 0 % | **是** |
| 案例方向一致性 | 4 例 | 4/4 | 0 % | **是** |
| 零基准不变量 | 解析上界（有限） | 1200 样本，0 次无界 | 0 % | **是** |
| 基线复现历史自报结论 | 工件 `status`/`passed` | 49.0 % | 51.0 % | **否**（原因见 ⑤-局限 2：历史结论不可从归档数值复算） |
| 模块测试覆盖率 | 80 % 门槛 | 95.34 % | — | **是** |

结论：六条机器判据全部 PASS（`artifacts/m9942/acceptance_report.json → overall`）；唯一“未 < 3 %”的量是“基线复现历史自报结论”，它是语料证据缺口的度量，不是本模块的精度指标，按 honest 原则原样列出，不做粉饰。

## 附录：复现命令

```bash
PYTHONPATH=src python -m tensorphysics.metrology.criteria all --root . --out-dir artifacts/m9942
python artifacts/m9942/generate_figures.py
python artifacts/m9942/build_acceptance_report.py
PYTHONPATH=src python -m pytest tests/test_metrology_criteria_*.py -q
PYTHONPATH=src python -m coverage run --source=src/tensorphysics/metrology/criteria \
    -m pytest tests/test_metrology_criteria_*.py -q && python -m coverage json -o artifacts/m9942/coverage.json
```
