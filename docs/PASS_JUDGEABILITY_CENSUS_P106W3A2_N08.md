# 验收判据可判性普查与 PASS 依赖判决（P106W3A2-N08）

**仓库** TensorPhysics · **分支** agent/m12400 · **基线** main@86e2c70
**节点** P106W3A2-N08（plan 9249）· **mission** 12400
**日期** 2026-11-10

---

## 1. 案例介绍（对象与数据面）

本节点审计的"物理对象"不是船型也不是工况，而是**平台判据本身**：
战役节点用自由文本声明验收判据，节点一经 `done` 就成为下游继承的 PASS。
问题是：**一个 PASS 里有多少是"能判错的"？**

数据面（只读，零手抄）：

| 项 | 数值 | 来源 |
|---|---|---|
| 已枚举计划 | **462** 个（plan-id 窗口 `1–2306` ∪ `9125–9400`，覆盖 plan 库的密集区） | `GET /tensor/campaigns/plans/{id}/nodes` |
| 节点 | **4,409** | 同上 |
| 含可抽取判据的节点 | **4,141**（抽取率 **93.92%**） | `parse_acceptance_criteria` |
| 判据条款 | **24,849** 条 | 同上 |
| 语料冻结 | `artifacts/acceptance_judgeability/corpus.jsonl`，sha256 `a3d736b2b4b502649eeea886a0bc4e732c720926f9075c7c5b8809bc6971185a` | `corpus_manifest.json` |
| 其中 `status=done` 的节点 | **2,735** | 语料 `status` 字段 |

语料由 `scripts/harvest_acceptance_criteria.py` 唯一生产（可 `--offline` 从 per-plan 缓存逐字节重建），
之后所有统计都是**离线纯函数**——同一语料两次运行必须给出同一 KS 表摘要（见 §3.4 A5）。

## 2. 怎么做的（方法 / 流程 / 工具链）

### 2.1 判据分类学（四类，闭合）

| 类 | 定义 | 计入 PASS 支撑？ |
|---|---|---|
| `JUDGEABLE` | 有**被测量** + 有**基准**（外部参考，或自参照闭合目标，如残差=0）+ 有**容差** | 是（且须已实测） |
| `THRESHOLD_ONLY` | 有被测量 + 有界，但**无基准**（阈值本身不指向任何真值） | 否 |
| `RECORD_ONLY` | 只要求记录 / 不崩 / 单调 / 定性声明 | 否 |
| `WITHHELD_OK` | 显式声明缺数据 | 否 |

子类进一步指出**缺哪一件**：`EXTERNAL_REFERENCE` / `SELF_REFERENCE`（JUDGEABLE）、
`NO_BASELINE` / `NO_TOLERANCE`（THRESHOLD_ONLY）、`MONOTONIC` / `QUALITATIVE` / `UNCLASSIFIED`
（RECORD_ONLY）、`DECLARED_MISSING`（WITHHELD_OK）——升级路径据此生成。

判定是**有序规则集**（`classify.py` R1–R7，先匹配先赢，写在 docstring 里即契约）：
R1 显式缺数据（且非判决上下文）→ WITHHELD_OK；R2 数值界 + 基准/自参照 → JUDGEABLE；
R3 数值界无基准 → THRESHOLD_ONLY；R4 极限箭头 + 数值无容差 → THRESHOLD_ONLY；
R5 单调 → RECORD_ONLY；R6 记录/定性 → RECORD_ONLY；R7 其余 → RECORD_ONLY + `needs_review`。

**零领域语义**：所有识别词表（容差/基准/缺数据/记录/单调/区间/算子…）都在
`data/lexicon_default.json`（元数据学词汇，无船/冰/求解器名词），逻辑与词表分离，
`set_lexicon()` 可整体替换（测试 `test_lexicon_is_injectable_and_restored` 钉住这一点）。

### 2.2 可判支撑度 KS

```
KS(node) = Σ w(条款 : JUDGEABLE 且已实测) / Σ w(全部条款)
```

* 权重默认**均匀**（`UNIFORM_WEIGHTS`，此时 KS 就是条款份额），另给 `TOLERANCE_WEIGHTS`
  （硬判据 ×2）作敏感性；
* 无条款的节点 `KS = None`（**未定义**，不是 0）——"没判据"与"判据判不了"不能混淆；
* "已实测"用节点级证据代理：`status=done` 且 `consecutive_fails=0`（§5 给出该代理的抽样校验）。

### 2.3 机制化（把分类器接进验收门）

* `support.adjudicate()` 是**判决函数**：无条款 → `NO_CRITERIA`；无 JUDGEABLE → `HOLLOW_PASS`
  （`RECORD_ONLY_ONLY`）；KS < `ks_min` → `HOLLOW_PASS`；有可判条款但未实测 → `UNMEASURED`；
  否则 `SUPPORTED`。**代码里没有任何路径让 RECORD_ONLY 单独支撑 PASS**。
* `campaign/wave3_gate.NodeRecord` 新增可选字段 `judgeability`（默认 `None`，历史记录行为不变）：
  带 flag 且 `supports_pass=False` 时，`verdict` 的 `ACCEPT` **降级为 `WITHHELD`**；
  `FAIL` 永不被软化（honest FAIL 不可覆盖）。测试 `test_node_record_gate_downgrades_hollow_accept`
  与 `test_node_record_fail_is_never_overridden_by_the_guard` 钉住两条边界。
* `gate.py` 提供 `clauses_from_node_metadata()`（复用平台唯一抽取器）与 `judgeability_flag()`。

### 2.4 工具链与审计链

| 环节 | 工件 |
|---|---|
| 语料生产 | `scripts/harvest_acceptance_criteria.py` → `artifacts/acceptance_judgeability/corpus.jsonl` + `corpus_manifest.json`（sha256） |
| 普查 + 图 | `scripts/acceptance_judgeability_census.py` → `census.json`（含逐节点 scorecard 与空心 PASD 台账）/ `census_sensitivity.json` / `judge_criteria.json` / `figs/*.png` |
| 金标一致性 | `artifacts/acceptance_judgeability/gold_labels_100.json` → `gold_kappa.json` |
| 对照 | `controls.json`（负控 20 条 + 假阳性 20 条） |
| 证据代理校验 | `scripts/validate_evidence_proxy.py` → `evidence_proxy.json` |
| 回归保护 | `tests/test_acceptance_judgeability.py`（34 测试，全绿） |

审计链：`corpus sha256 → 每条款 (cid,text) → 分类 (class, subtype, features) → KS 表 sha256 →
空心台账 (node_id, KS, 逐条升级建议)`，任一层可复算。

## 3. 计算结果（判据数字，程序化注入）

### 3.1 判据分类分布（24,849 条）

| 类 | 条数 | 份额 |
|---|---|---|
| JUDGEABLE | **6,248** | **25.14%** |
| THRESHOLD_ONLY | 12,987 | 52.26% |
| RECORD_ONLY | 5,524 | 22.23% |
| WITHHELD_OK | 90 | 0.36% |

子类：`NO_BASELINE` 12,672 · `EXTERNAL_REFERENCE` 4,403 · `UNCLASSIFIED` 2,576 ·
`QUALITATIVE` 2,572 · `SELF_REFERENCE` 1,845 · `MONOTONIC` 376 · `NO_TOLERANCE` 315 ·
`DECLARED_MISSING` 90。

**读数**：四条判据里只有一条同时具备"被测量 + 基准 + 容差"；一半以上只有阈值（无基准），
两成是过程/记录要求。

### 3.2 可判支撑度 KS（4,409 节点）

| 统计量 | 数值 |
|---|---|
| 定义域（≥1 条款） | 4,141 节点；未定义 268 |
| 中位数 | **0.0** |
| 均值 | 0.1571 |
| p10 / 最大 | 0.0 / 1.0 |

严重度分层（全体节点）：`TOTAL`(KS=0) **2,411** · `SEVERE`(<0.25) 565 · `PARTIAL`(0.25–0.5) 620 ·
`SUPPORTED`(≥0.5) **545** · `NO_CRITERIA` 268。

### 3.3 PASS 依赖判决：空心 PASS 台账

以 `done` 节点为 PASS 总体（2,735 个），`ks_min=0.5`：

| 项 | 数值 |
|---|---|
| 空心 PASS 节点 | **2,190** |
| 空心率 | **80.07%** |
| 升级建议条目 | 10,799 条（覆盖 2,190 个节点的每一条不可判条款） |

按仓（前 5，节点数 / 空心数 / 空心比 / KS 中位数）：

| 仓 | 节点 | 空心 | 空心比 | KS 中位数 |
|---|---|---|---|---|
| ship-performance-platform | 1,974 | 976 | 0.4944 | 0.0 |
| TensorPlatform | 606 | 354 | 0.5842 | 0.0 |
| TensorLBM | 487 | 222 | 0.4559 | 0.0 |
| TensorDEM | 463 | 212 | 0.4579 | 0.0 |
| TensorPhysics | 336 | 165 | 0.4911 | 0.0 |

**缺陷是系统性的**：没有任何一个仓的空心率低于 40%，最好的仓也接近一半。

升级建议样例（真实语料，逐条自动生成）：

| 原条款 | 类/子类 | 缺什么 | 建议动作（节选） |
|---|---|---|---|
| `unit=ratio; metric=center_deflection_relative_error; operator=<; threshold=1e-3` | THRESHOLD_ONLY / NO_BASELINE | reference | 补基准：写明对照物 y_ref（解析解/文献/实测参考值）与相对容差 τ |
| `AC3: GM 增大→参数横摇风险指标单调不增 (符号100%正确)` | THRESHOLD_ONLY / NO_BASELINE | reference | 同上；单调性本身需给自变量网格与斜率下界 |
| `C5: 推力崩溃单调下降` | RECORD_ONLY / MONOTONIC | tolerance, reference | 单调性判据可量化：给出采样点与斜率下界（或参考区间） |
| `验收汇总表落盘` | RECORD_ONLY / QUALITATIVE | quantity, reference, tolerance | 把定性要求改写为可判量；若确实无法量化，改标 WITHHELD_OK |

### 3.4 机器验收判据（本节点自身，全部满足）

| 判据 | 门槛 | 实测 | 判定 | 工件 |
|---|---|---|---|---|
| A1 金标 κ（100 条分层抽样人工标注） | ≥0.80 | **0.9666**（holdout 0.9351，dev 1.0，一致率 98%） | PASS | `gold_kappa.json` |
| A2 负控：20 条纯 RECORD_ONLY 伪造 PASS | 20/20 判空心 | **20/20 = 1.000** | PASS | `controls.json` |
| A3 假阳性：20 条已确认可判条不得判 RECORD_ONLY | 误报 ≤1 | **1** | PASS（压线，见 §5） | `controls.json` |
| A4 抽取率 | ≥85% | **93.92%** | PASS | `census.json` |
| A5 两次运行 KS 表 sha256 一致 | 相同 | `7164c422…02a1ce5` == 第二次运行 | PASS | `census_sensitivity.json` |

### 3.5 敏感性（结论对政策选择的依赖）

| 口径 | 空心率 | 说明 |
|---|---|---|
| 默认（均匀权重，ks_min=0.5） | **0.8007** | 主口径 |
| 严格（自参照不算基准） | 0.8735 | 只认真外部锚 |
| 乐观（未写基准的"误差<x"按隐含对标计入） | 0.5890 | 上界 |
| 权重（硬判据 ×2） | 0.6399 | 强调硬判据后改善 |
| ks_min 扫描 0.1→0.75 | 0.3678 → 0.9558 | 见 fig3 |

**结论的稳健部分**：无论取哪种口径，空心 PASS 都占 **59%–87%**，"多数 PASS 缺少可判支撑"不是阈值人为制造的。

### 3.6 证据代理校验（抽样 40 个 done 节点）

| 项 | 数值 |
|---|---|
| 有已发布报告的节点 | **35/40 = 87.5%** |
| mission 状态正常 | 40/40 = 100% |
| 同一子集空心率（宽松证据） | 0.65（26/40） |

即"`done` ⇒ 已实测"这一代理在抽样上成立（87.5% 有报告，剩余 12.5% 是"报告缺失"而不是"未测量"），
但它只给出**下界**：报告缺失不证明没测。

## 4. 结果图

![fig1](artifacts/acceptance_judgeability/figs/fig1_class_distribution.png)
**图 1** 判据分类分布（24,849 条）：JUDGEABLE 25.1%，THRESHOLD_ONLY 52.3%，RECORD_ONLY 22.2%。

![fig2](artifacts/acceptance_judgeability/figs/fig2_ks_histogram.png)
**图 2** 节点 KS 直方图与 `ks_min=0.5` 门限：中位数 0，右侧长尾只有 545 个节点越过门限。

![fig3](artifacts/acceptance_judgeability/figs/fig3_threshold_sweep.png)
**图 3** 空心率对 `ks_min` 的敏感性：门槛是政策不是物理，故同时报告扫描曲线。

![fig4](artifacts/acceptance_judgeability/figs/fig4_repo_hollow.png)
**图 4** 分仓空心率：缺陷跨仓分布，非单仓局部问题。

> 本节点是**判据度量学/门禁机制**任务，物理量云图不适用；图为分类学与分布的统计图，
> 这是本任务唯一诚实的可视化形式（无场量可绘）。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **抽取器缺陷（本次修复）**：`metadata.acceptance_criteria` 有 145 个节点是 **JSON 字符串**，
   旧 `parse_acceptance_criteria` 直接 `enumerate(str)` → **每字符一条判据**，制造 28,204 条碎片条款
   （占首轮语料 63%），并让 wave-4 门的判据账目虚高。已修：字符串先 `json.loads` 再枚举；
   结构化（dict）判据按声明顺序渲染 `key=value; …`（确定性、可哈希）。回归测试
   `test_json_encoded_criteria_are_decoded_not_iterated_per_character` / `test_structured_criteria_render_deterministically`。
2. **条款 ID 不是量**：`AC6 pytest全绿` 里的 `6` 曾把纯文字条款抬成"有界条款"；已加 `strip_id_prefix`
   （覆盖 `C1:`/`AC6 `/(a)/3) 等形态），并有用例钉住。
3. **自参照算不算基准**：残差/守恒/恒等这类**自参照目标**（目标值由条款自身定义）默认计入 JUDGEABLE，
   但单列 `SELF_REFERENCE` 子类与严格口径（空心率 87.4%）。这是一条**定义边界**，不是测量结果。
4. **阈值是政策**：`ks_min=0.5` 会让"6 条判据里 2 条硬判据"的节点也进空心台账。因此本报告同时给
   扫描曲线与三种口径；引用单一数字时必须带上口径。
5. **A3 压线**：唯一误报是 `漂移速度变化10-40%…显式回答是否落入观测带`——机器把祈使句"显式回答"
   读成记录要求，人读作"与观测带比对"。同一条也是 κ 的两处不一致之一（另一处是判决词上下文的
   `WITHHELD 异常数 0`）。**如实记录，不追调至 0**。
6. **作者自标注**：100 条金标由本节点作者逐条阅读标注，规则微调用前 50 条（dev，κ=1.0），
   报告口径取后 50 条 holdout（κ=0.9351）与全体 100（κ=0.9666）。作者自标注带来的乐观偏差**未消除**，
   只做了 dev/holdout 拆分；独立第二标注者缺失是本报告最大局限。
7. **语料范围**：plan-id 窗口 `1–2306 ∪ 9125–9400`（462 个真实存在的计划）。中间段经抽样为 404 空段，
   未逐 id 穷举——报告不声称覆盖"平台有史以来全部计划"。
8. **证据代理是节点级**：平台不在条款级存测量值，KS 分子用 `done` 代理；`require_report` 更严模式可切换，
   但需先补报告台账的时序一致性。

## 6. 误差判定（逐项，显式回答"是否 <3%"）

| 关键量 | 基准 | 实测误差 | 是否 <3% |
|---|---|---|---|
| 机器 vs 人工分类一致率（100 条） | 人工金标 | 2/100 = 2.00% 不一致 | **是（2.00% < 3%）** |
| holdout 50 条不一致率 | 人工金标 | 2/50 = 4.00% | 否（4.00%）—— 声明：小样本，κ 才是主口径（0.9351 ≥ 0.80） |
| 负控漏判率（应全判空心） | 设计真值 20/20 | 0.00% | **是** |
| 假阳性率（可判条被误判 RECORD_ONLY） | 人工确认 20 条 | 1/20 = 5.00% | 否（5.00%，门槛为"≤1 条"，判据以条数计） |
| 抽取率 | 应有判据节点 | 6.08% 未抽到（=1−93.92%） | 否，但判据是"≥85% 抽取"（实测 93.92% 通过） |
| KS 表可复算 | 两次运行 | 0 差异（sha256 相同） | **是** |

**"是否 <3%"的诚实回答**：本任务是**分类与统计**任务，不存在连续物理量的解析解，
因此"误差 <3%"只在"与人工金标的一致率"一项上有意义（2.00% < 3%，是）；
其余判据是**计数型门槛**（20/20、≤1 条、≥85%、sha256 相同），按条数判定，
不能也不该换算成百分比误差——此处不做粉饰。κ 的绝对值受"作者自标注"影响，见 §5.6。

## 7. 适用边界

* **适用**：任何把验收判据写成自然语言、按节点汇总 PASS 的流程（本生态 12 仓、全部战役）。
* **不适用**：判据已是结构化数值 schema（`metric/threshold/unit` 三元组 + 参考值字段）的场景——
  那里应当直接做数值门禁，本分类器的价值退化为"参考值是否缺失"的检查。
* **不可用于**：以 KS 高低替代物理正确性判断。KS 只回答"这条 PASS 能不能判错"，
  不回答"物理对不对"。