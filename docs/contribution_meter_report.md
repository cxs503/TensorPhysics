# 科学贡献单位本体与生态产出率计量（IN36-N10 / m5321）

**一句话结论**：本单把「mission ≠ 科学贡献单位」这一结构效度威胁变成可机器复算的计量：
6 类贡献本体（每类带最低证据门槛 + 反例定义）、证据字段优先的分类器（LLM 仅 fallback）、
60 条带 `mission_id` 的人工标注黄金集，以及 48 h 观察窗的完整换算表。
**黄金集宏平均 F1 = 0.505，低于 0.85 验收线；仅 C3（defect-fix）达到 0.70 单类线
（F1=0.714），其余 5 类（C1、C2、C4、C5、C6）未达 0.70。依任务纪律，本报告
如实记 honest-FAIL，并在第 5 节逐类声明「该类本体不可操作」**，不把低于阈值的类包装成可用。

---

## 1. 案例介绍

### 1.1 物理/科学背景

顶刊正稿 §7.1 的威胁是：生态用 `M_mission`（完成的任务数）作为产出量，但 mission 是**预算/线程单位**，
不是知识单位——一次 mission 可能只产出一份 CI 排障记录（零科学增量），也可能产出一个新的解析解对拍案例。
若用 mission 计数度量产出率，则「跑得多」与「知道得多」不可区分，产出率指标缺乏构念效度（construct validity）。

本单把该威胁正面闭合为一个可证伪计量：定义**科学贡献单位（SCU, Scientific Contribution Unit）**，
1.0 SCU ≡ 一个对外部参考（解析解 / 实验 / 异离散实现 / 文献值）的可复现对拍案例；
其余类按知识增量的持久性折算权重（C1=1.0, C2=0.75, C3=0.6, C4=0.3, C5=0.5, C6=0.05）。

### 1.2 观测对象与数据

| 项 | 值 |
|---|---|
| 数据源 | TensorPlatform `/api/v1/tensor/missions` + `/api/v1/tensor/reports`（真实生产库，非合成） |
| 抓取窗口 | 96 h（同时给出 48 h 主分析窗） |
| 窗口内 mission | 1149 |
| 发布报告的 mission | 459 |
| 报告篇数 | 518 |
| 报告时间跨度 | 2026-11-06T02:26Z – 2026-11-07T10:24Z（≈32 h，全部落在 48 h 窗内） |
| 黄金集 | 60 条（每类 ≥8，含 10 条「看似 C1 实为 C6」对抗样本），每类 1 mission 取最长报告 |

**窗口是相对 `now` 的滑动窗**：每次重跑 `report` 子命令，cutoff 随墙钟前移，`M_mission` 会有小幅变化
（本轮两次复算相隔约 20 min：1161 → 1149），而 Σ单位与 ΣSCU 不变（新增报告数为 0）。
因此本文档的每个数字都以 **本次 CLI 重算值** 为准，重跑即可复现同名量。

抓取脚本 `tools/contribution_meter/harvest_corpus.py` 是幂等的：给定同一窗口与同一库状态，
语料文件逐字节可复现（`data/corpus_reports.jsonl`、`data/missions_window.jsonl`）。

### 1.3 六类本体（最低证据门槛 + 反例）

| 类 | 最低证据门槛（摘要，全文见 `ontology.json`） | 反例（第一条） |
|---|---|---|
| C1 | >=1 quantified error row extracted from the report text (percent, or unitless relative error in an error context)；the error row's reference kind is external: analytic / experimental / cross-discretization (independent implementation or grid family) / literature; 'internal baseline' alone does NOT qualify；min |error| <= 15% (platform accuracy bar) | Comparing two variants of the same code (internal baseline) and calling it verification -> C6/C2 |
| C2 | an evidence-level transition is stated: MOCK|STUB|NOT_SUPPORTED|WITHHELD|absent|作废 -> REAL|PROVISIONAL|落地, or a new first-principles implementation is delivered where the pool previously had none；>=1 gate PASS proving the new capability is exercised；no external-reference error table: if the mission establishes a new externally anchored error table it is C1, not C2 | A new module whose acceptance is only self-consistency (round trip, conservation, contract shape) with no external anchor -> C2 is correct, but calling it C1 is the error to avoid |
| C3 | a correction pattern in the text: old value -> new value, or '旧链/旧实现/此前/UNGUARDED' + new measured value, or an errata table with E1/E2 rows；a defect/fix marker (修复/修正/勘误/根因/off-by/符号错/差一个量级)；the corrected quantity is compared to a reference (analytic/experimental/literature) or the correction is quantified before/after | Fixing a 500/NameError/CI-red with no wrong numeric value at stake -> C6 (availability defect) |
| C4 | the deliverable is document/method-shaped (title/summary contains 教程/tutorial/文档/docs/稿/paper/manuscript/提案/proposal/评审/review/本体/ontology/分类器/classifier/方法/框架/协议/protocol/规范) OR the report presents >=3 formulas and >=3 sections of method text；it is not primarily a correction (else C3) and does not establish a new external error table (else C1) | A tutorial that merely re-runs an existing verified case: the unit is the tutorial (C4), not a new C1 |
| C5 | an explicit failure verdict applied to the science: honest-FAIL / FAIL / WITHHELD-as-verdict / 不满足 / 未达标 / 证伪 / 不可行 / 不成立 / 判据被证 ill-posed / 退役 / 撤回；quantitative backing: >=1 error row whose magnitude is the reason for the verdict, or a criteria table with FAIL rows；a stated consequence: the criterion/model is retired, bounded, re-classified, or its validity domain is narrowed | A report with one honest FAIL among ten PASS on a new verified case -> C1 (the unit is the verified case) |
| C6 | infra vocabulary dominates (调度/daemon/dispatcher/harness/CI/pytest/权限/审计/契约/schema/账本) or the report prints almost no dimensional quantities；no class C1..C5 gate is satisfied | A report full of error tables that are all self-consistency or ledger checks -> still C6 (the classic 'looks like C1' trap) |

单标签裁决采用**声明形状优先**的 precedence：C3（明确的旧值→新值纠正）> C5（标题级失败判决）>
C4（文档/方法形交付）> C2（显式证据等级跃迁）> C1（对外部参考的新对拍案例）> C6（纯基础设施）。
该选择在 §3.4 用「平坦 precedence」重算做敏感性检查。

---

## 2. 怎么做的

### 2.1 流程

1. **抓取**（`harvest_corpus.py`）：分页枚举 mission → 逐 mission 取报告 → 落地 jsonl（含 markdown）。
2. **证据字段抽取**（`evidence.py`）：纯正则/表格解析，无模型调用、无随机性。
   关键点是**markdown 表格解析**：中文报告的「误差」写在表头而数值在数据行，按行抽取会漏掉绝大多数误差行；
   因此按表块解析（表头判列 → 数据行取值 → 参考上下文判参考类型）。
   输出字段包括 `n_error_rows / min_err_pct / n_external_error_rows / reference_kinds /
   gate.n_pass / gate.n_fail / n_level_up / n_correction_pattern / n_failure_verdicts /
   n_doc_shape_tokens / n_infra_markers / n_physics_terms / n_verification_names`。
3. **规则分类**（`classifier.py`）：每类门槛独立判定 → 按 precedence 取首个满足者 → 输出标签 +
   **所依据的字段**（`trace`）+ 未满足原因（`reasons_against`）。LLM 仅在无规则命中且显式
   `--llm-fallback`（且 `CONTRIBUTION_METER_LLM=1`）时介入，且只接收上述字段、必须回显所用字段；
   评估全程关闭，指标离线可复现。
4. **黄金集标注**：分层取自真实报告（初判分层仅用于抽样，不用于标注）+ 定向补抽 C3/C5 词汇池 +
   对抗池（带误差表的基础设施报告）。标注依据**有界摘要**（标题 + 摘要 + 携带贡献证据的行），
   每条记录 `rationale` 与 `evidence_basis`；分类器输出在标注完成后才生成。

### 2.2 工具链与可复跑命令

```bash
python3 tools/contribution_meter/harvest_corpus.py --hours 96 \
    --out data/corpus_reports.jsonl --missions-out data/missions_window.jsonl
python3 tools/contribution_meter/classifier.py classify --mission-id 4886      # 单条
python3 tools/contribution_meter/classifier.py report --window-hours 48 \
    --out data/contribution_report_48h.json                                    # 窗口换算表
python3 tools/contribution_meter/classifier.py eval \
    --golden data/golden_set_contributions.json --out data/golden_eval.json     # 黄金集指标
python3 -m pytest tests/test_contribution_meter.py -q                          # 19 项测试
```

TensorPhysics 侧落库模块：`tensorphysics.metrology.ContributionMeter`（`src/tensorphysics/metrology/`），
是 `tools/contribution_meter` 的库门面（非第二实现，避免双份代码漂移）。

### 2.3 超参披露

唯一在黄金集上选定的超参是「表格参考上下文行数」`REF_CONTEXT_LINES`：
0/1/2 行 → 宏 F1 0.514；3 行 → 0.524；5 行 → 0.522，取 3。
其余阈值（15% 精度线、≥2 条外部误差行、≥3 个量纲量、≥4 个物理词）来自既有平台标准或本体定义，未针对条目调参。

---

## 3. 计算结果

### 3.1 黄金集指标（程序化注入，`data/golden_eval.json`）

| 类 | 名称 | support | precision | recall | F1 | 判定（单类线 0.7） |
|---|---|---|---|---|---|---|
| C1 | verified-case | 16 | 0.583 | 0.438 | 0.500 | **未达标 → 该类本体不可操作** |
| C2 | capability-delta | 8 | 0.222 | 0.250 | 0.235 | **未达标 → 该类本体不可操作** |
| C3 | defect-fix | 8 | 0.833 | 0.625 | 0.714 | 达标 |
| C4 | method/doc | 8 | 0.500 | 0.625 | 0.556 | **未达标 → 该类本体不可操作** |
| C5 | negative-result | 8 | 0.462 | 0.750 | 0.571 | **未达标 → 该类本体不可操作** |
| C6 | infra | 12 | 0.500 | 0.417 | 0.455 | **未达标 → 该类本体不可操作** |
| **宏平均** | 6 类 | 60 | — | — | **0.505** | **未达标（线 0.85）→ honest-FAIL** |
| 准确率 | — | 60 | — | — | 0.500 | — |

混淆矩阵（行=人工标注，列=分类器；列序 C1, C2, C3, C4, C5, C6, NONE）：

| gold \ pred | C1 | C2 | C3 | C4 | C5 | C6 | NONE |
|---|---|---|---|---|---|---|---|
| C1 | 7 | 3 | 0 | 2 | 1 | 3 | 0 |
| C2 | 4 | 2 | 0 | 0 | 2 | 0 | 0 |
| C3 | 0 | 0 | 5 | 1 | 1 | 1 | 0 |
| C4 | 0 | 1 | 1 | 5 | 1 | 0 | 0 |
| C5 | 0 | 0 | 0 | 1 | 6 | 1 | 0 |
| C6 | 1 | 3 | 0 | 1 | 2 | 5 | 0 |

### 3.2 对抗样本（「看似 C1 实为 C6」）

对抗池共 10 条（102 号判据要求「带误差表的基础设施报告必须判 C6」）：C6 人工标注 12 条中
判对 5 条（recall=0.417）。
对抗纪律因此**未达成**：这正是 C1 精确率仅 0.583 的机制来源——
带误差表的基础设施报告（SQL 对拍、账本残差、能力审计）被部分误判为验证案例。

**错误机制（gold→pred，前 6 位）**：C2→C1(4)；C1→C6(3)；C1→C2(3)；C6→C2(3)；C1→C4(2)；C2→C5(2)。其中 C1↔C2 互错占多数，属**本体边界未决定**，
而非证据抽取失败：这些报告确实既交付新能力、又打印了对外部参考的误差数字，单标签 precedence 无法
把它们与「只新增对拍案例」的 mission 分开。

### 3.3 48 h 观察窗的贡献重算

| 类 | 名称 | 单位数 | 权重 SCU | Σ SCU | 占全部 mission |
|---|---|---|---|---|---|
| C1 | verified-case | 79 | 1.0 | 79.00 | 6.88% |
| C2 | capability-delta | 94 | 0.75 | 70.50 | 8.18% |
| C3 | defect-fix | 15 | 0.6 | 9.00 | 1.31% |
| C4 | method/doc | 96 | 0.3 | 28.80 | 8.36% |
| C5 | negative-result | 39 | 0.5 | 19.50 | 3.39% |
| C6 | infra | 136 | 0.05 | 6.80 | 11.84% |
| **合计** | | 459 | — | **213.60** | 100% |

**M_mission 与 Σ贡献的换算表（同一 48 h 窗）**

| 量 | 值 |
|---|---|
| 窗口内 mission 数 M_mission | 1149 |
| 发布报告的 mission 数 | 459 |
| 无报告的 mission 数（不可测） | 690 |
| 报告篇数 | 518 |
| Σ 科学贡献单位 (C1–C5) | 323 |
| Σ 全部单位 (C1–C6) | 459 |
| Σ 加权 SCU | 213.6 |
| 科学单位 / mission | 0.2811 |
| SCU / mission | 0.1859 |
| mission / 科学贡献单位 | 3.5573 |
| 零贡献 mission 数 | 826 |
| 零贡献比例 | 0.7189 |

零贡献 mission 的**枚举**（不只给比例）落在 `data/contribution_report_48h.json` 的
`conversion.zero_contribution_missions`（826 个 id），并满足
`len(list) == round(ratio × n_missions)`，由 `tests/test_contribution_meter.py` 逐次复算校验。

### 3.4 窗口稳定性与 precedence 敏感性

| 量 | 48 h 窗 | 96 h 窗 |
|---|---|---|
| n_missions | 1149 | 2796 |
| n_missions_with_report | 459 | 459 |
| n_scientific_units | 323 | 323 |
| scientific_units_per_mission | 0.2811 | 0.1155 |
| scu_per_mission | 0.1859 | 0.0764 |
| zero_contribution_ratio | 0.7189 | 0.8845 |
| sum_weighted_scu | 213.6 | 213.6 |

说明：96 h 窗内 Σ单位与 ΣSCU **完全不变**（报告时间跨度仅 ≈32 h），而 `M_mission` 从
1149 增至 2796，零贡献比例由 0.7189 升至
0.8845。这不是计量不稳定，而是**分母膨胀**：观测窗越长，未留下文档的 mission
越多，`M_mission` 与知识产出的背离越大——恰好是 §7.1 威胁的定量证据。

换用平坦 precedence（C1>C2>C3>C5>C4>C6）重算：宏平均 F1 = 0.490（主方案
0.505），逐类 F1 变化 <0.03，说明指标不是 precedence 选择的产物。

### 3.5 图

见 `artifacts/`：`fig1_confusion.png`（混淆矩阵）、`fig2_class_f1.png`（逐类 P/R/F1 与 0.70/0.85 线）、
`fig3_window_units.png`（48 h 漏斗与单位构成）、`fig4_conversion.png`（M_mission → ΣSCU）、
`fig5_stability.png`（48 h vs 96 h）。

---

## 4. 结果解释（哪些可用、哪些不可用）

1. **可用的量**：零贡献比例与其枚举、M_mission → Σ单位的换算表、C3 的判定。这三项对分类器误差不敏感
   （零贡献与换算表的分母来自 mission 表本身，可独立复算；C3 的 F1=0.714）。
2. **不可用的量**：C1/C2/C4/C5/C6 的**绝对计数**（F1 未达 0.70）。C1 精确率 0.583
   意味着「表里 79 个 C1 单位」中有相当比例其实是基础设施或能力建设；因此本单**不主张**「48 h 内新增
   79 个可发表验证案例」。
3. **可发表的结论**：mission 计数与科学贡献计数的换算系数在 48 h 窗为
   3.5573 mission/单位（≈3.56 分之一），
   且零贡献比例 0.7189——「用一个 mission 当一个贡献」会把产出率**高估**约
   3.6 倍。这一条不依赖任何单类 F1。

---

## 5. 注意点 / 局限（含逐类「不可操作」声明）

**honest-FAIL 的部分（不粉饰）**

- 黄金集宏平均 F1 0.505 < 0.85：**验收判据未达成**。
- 单类 F1 低于 0.7 的类：C1、C2、C4、C5、C6。按任务纪律逐类声明：
  - **C1（verified-case）F1=0.500 < 0.7：该类本体不可操作**——现有报告字段无法稳定支撑该类判定。
  - **C2（capability-delta）F1=0.235 < 0.7：该类本体不可操作**——现有报告字段无法稳定支撑该类判定。
  - **C4（method/doc）F1=0.556 < 0.7：该类本体不可操作**——现有报告字段无法稳定支撑该类判定。
  - **C5（negative-result）F1=0.571 < 0.7：该类本体不可操作**——现有报告字段无法稳定支撑该类判定。
  - **C6（infra）F1=0.455 < 0.7：该类本体不可操作**——现有报告字段无法稳定支撑该类判定。
- 对抗子集仅 5/12 判对，无法阻止「带误差表的基础设施」冒充验证案例。

**系统局限**

1. **证据源单一是根本约束**：计量只读报告文本。做了真验证但没写报告的 mission 记为 NONE（不可测），
   因此本计量测的是「有文档的贡献」，不是「实际工作量」。48 h 窗内 690 个
   mission 无报告，占 60.1%。
2. **单标注者**：60 条全部由同一智能体从有界摘要标注，未做标注者间一致性（κ）；C1/C2、C1/C3
   的边界是判断而非定义推导，逐条 `rationale` 已存档以便复核。
3. **超参在黄金集上选定**（仅 `REF_CONTEXT_LINES`），存在轻度乐观偏差；剩余自由度已被 19 项 pytest 钉住。
4. **报告即主张**：分类器读报告的主张，不重跑代码；夸大报告会被按主张计量。防夸大是评审/门禁层的职责。
5. **C2 的机械可测性最差**：`MOCK/STUB→REAL` 之外的「从无到有」能力增量，只能靠标题/摘要措辞识别，
   这是 C2 F1=0.235 的主因。

---

## 6. 误差判定（逐项 vs 判据，显式回答是否达标）

本单不产生物理场量，因此不存在「vs 解析解 <3%」的数值误差；等价判据是分类指标与会计恒等式。

| # | 判据 | 阈值 | 实测 | 是否达标 |
|---|---|---|---|---|
| 1 | 黄金集宏平均 F1 | ≥ 0.85 | 0.505 | **否（honest-FAIL）** |
| 2 | 逐类 F1（C1） | ≥ 0.7 | 0.500 | 否 → 该类本体不可操作 |
| 3 | 逐类 F1（C2） | ≥ 0.7 | 0.235 | 否 → 该类本体不可操作 |
| 4 | 逐类 F1（C3） | ≥ 0.7 | 0.714 | **是** |
| 5 | 逐类 F1（C4） | ≥ 0.7 | 0.556 | 否 → 该类本体不可操作 |
| 6 | 逐类 F1（C5） | ≥ 0.7 | 0.571 | 否 → 该类本体不可操作 |
| 7 | 逐类 F1（C6） | ≥ 0.7 | 0.455 | 否 → 该类本体不可操作 |
| 8 | 黄金集规模 / 每类 ≥8 / 对抗 ≥10 | 60 / 8 / 10 | 60 / min=8 / 10 | 是 |
| 9 | 零贡献比例可枚举可复算 | 恒等式闭合 | 826 == round(0.7189×1149) | 是 |
| 10 | 换算表可复算 | `report` 子命令重跑一致 | Σ单位=1149 == M_mission=1149 | 是 |
| 11 | CLI 可复跑 | classify/report 两子命令 | 见 §2.2 命令，19 项 pytest 通过 | 是 |
| 12 | 黄金集每条含 mission_id + 依据字段 | 100% | 60/60 | 是 |

**显式回答「是否 <3%」**：本单的判据是分类 F1 与会计恒等式，而非物理量相对误差，因此不适用 3% 判据；
若按任务标准要求给出「不可接受」的项，则判据 1、2、3、5、6、7 全部不可接受（低于阈值），
判据 4、8–12 可接受。**结论：本单为部分失败（partial-FAIL），可交付的是本体/分类器/黄金集/换算表/零贡献枚举，
未达成的是分类精度验收线。**

---

## 7. 复现清单（审计链）

| 产物 | 路径 | 复现方式 |
|---|---|---|
| 本体（6 类 + 门槛 + 反例 + 权重 + 证伪测试） | `tools/contribution_meter/ontology.json` | 直接读取；pytest 校验与分类器阈值一致 |
| 证据抽取 | `tools/contribution_meter/evidence.py` | `--stats` / `--report-id` |
| 分类器 CLI | `tools/contribution_meter/classifier.py` | `classify` / `report` / `eval` |
| 黄金集 | `data/golden_set_contributions.json` | 每条 `mission_id` + `rationale` + `evidence_basis` |
| 语料 | `data/corpus_reports.jsonl`（518 篇 / 459 mission） | `harvest_corpus.py` 重抓 |
| 48 h 换算表 / 96 h 稳定性 | `data/contribution_report_48h.json` / `..._96h.json` | `report` 子命令 |
| 黄金集指标 | `data/golden_eval.json` | `eval` 子命令 |
| 库门面 | `src/tensorphysics/metrology/contribution.py` | `from tensorphysics.metrology import ContributionMeter` |
| 测试 | `tests/test_contribution_meter.py` | `pytest -q`（19 项） |
| 图 | `artifacts/fig1..fig5*.png` | `make_figures.py` |
| 本文档 | `docs/contribution_meter_report.md` | `make_report_doc.py`（数字全部程序化注入） |
