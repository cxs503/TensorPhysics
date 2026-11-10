# P186-W1-N08 · honest-FAIL 失败模式库与重派决策表

## ① 案例介绍

战役 DAG 的节点以 `PASS / FAIL / WITHHELD` 裁决，但失败原因一直只写在报告散文的
“局限/下一步”小节里：调度器看得见“这个节点红了”，看不见“为什么红、下一步该改什么”。
本节点把这段散文变成可机器裁的**失败画像**：6 类方法级失败机理 + 每条记录的证据字段 +
三选一重派建议（缩范围 / 换方法 / 换基准），并在**冻结的真实 FAIL 集**上回测判别力。

* 语料：`48` 条真实历史 FAIL 片段（每类 8 条），
  从本仓 `docs/**/*.md`、`artifacts/**/*.json` 用**认识论线索**（缺失/未收敛/超阈/假设/缺陷）
  检索得到，逐条经分析者 `agent:m9338` 阅读定标，正文逐字保留、溯源到文件行号与
  sha256，载荷摘要 `85acd46b774ed8dd…` 冻结（`dataset.load_frozen_set` 每次校验）。
* 诱饵池：`120` 条同报告体系、同长度、**无任何失败线索**的段落，用于负控 C3a。
* 决策表：6 类 × (主建议, 次建议, ≥1 触发证据字段)，schema 门机检。

## ② 方法

1. **证据化**：每条记录被压成 11 维布尔信号向量（`reference/absence/artifact/withheld/
   threshold/quantified_gap/residual/convergence/assumption/envelope/defect`），词表在
   `data/cue_lexicon.json` 外置，代码零领域词。
2. **双标注**：分析者（人工定标）× 结构化规则标注器 A（只看信号向量，权重表外置且仅在
   train 段复核过一次）；另设无监督原型标注器 B（字符 2–3 gram TF-IDF 对类种子句的质心，
   不看任何金标签）作为第二意见。
3. **划分**：按类分层、确定性取每类第 3 条为测试段（train 36 / test 12）。
4. **重派决策表**：`SCOPE_REDUCTION / METHOD_SWAP / BASELINE_SWAP` 三选一，每行带触发证据字段；
   触发字段一个都不在记录中时输出 `ESCALATE`（不猜），该分支有独立测试。
5. **集成**：`blocked_node_view(CampaignDAG.verify_convergence().blocked_nodes, records)` 输出
   `{failure_class, evidence, reroute_suggestion}`；`to_ledger_entry` 把每条渲染为
   `category="falsified_hypothesis"` 的台账条目（失败也是成果，登记而非删除）。

## ③ 计算结果

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| C1 双标注 Cohen κ | ≥ 0.6 | 0.750 | PASS |
| C1 次对（规则 vs 无监督原型）κ | 参考 | 0.674 | — |
| C2 冻结集一致率 | ≥ 0.70 | 0.792 | PASS |
| C2 macro-F1 | 参考 | 0.788 | — |
| C3a 同长度诱饵一致率 | ≤ 0.35 | 0.169 (10 次抽样均值, 最大 0.250) | PASS |
| C3b 标签重排后测试一致率 | ≤ 0.30 | 0.180 (25 次置换均值, 最大 0.417; 真标签 0.583) | PASS |
| C4 决策表完整性 | 每类 1 建议 + ≥1 触发字段 | 6 类, 最少触发字段 1 个 | PASS |
| C5 零领域语义 | 命中 0 | 0 命中 | PASS |
| C6 集成 | blocked 视图输出三字段 | 1 条集成测试（CampaignDAG → view） | PASS |

### 决策表（C4 机器可检）

| 失败类 | 主建议 | 次建议 | 触发证据字段 | 理由码 |
|---|---|---|---|---|
| `BASELINE_ABSENT` | BASELINE_SWAP | SCOPE_REDUCTION | `reference_lexeme`, `absence_lexeme` | no_yardstick_but_result_exists |
| `TOLERANCE_EXCEEDED` | SCOPE_REDUCTION | METHOD_SWAP | `quantified_gap`, `threshold_lexeme` | quantified_miss_of_declared_band |
| `CONVERGENCE_STALL` | METHOD_SWAP | SCOPE_REDUCTION | `convergence_lexeme`, `residual_magnitude` | iterate_never_settled |
| `MODEL_FIDELITY_LIMIT` | METHOD_SWAP | SCOPE_REDUCTION | `assumption_lexeme`, `envelope_lexeme` | intrinsic_formulation_limit |
| `DATA_INSUFFICIENCY` | SCOPE_REDUCTION | BASELINE_SWAP | `absence_lexeme`, `artifact_lexeme`, `withheld_lexeme` | missing_inputs_or_deliverables |
| `IMPLEMENTATION_DEFECT` | METHOD_SWAP | SCOPE_REDUCTION | `defect_lexeme` | localised_code_error |

混淆矩阵（行 = 分析者金标签，列 = 结构化标注器）：
`[[6, 0, 0, 1, 1, 0], [0, 5, 0, 1, 2, 0], [0, 0, 8, 0, 0, 0], [1, 1, 1, 5, 0, 0], [0, 0, 0, 0, 8, 0], [2, 0, 0, 0, 0, 6]]`

逐类召回：`{"BASELINE_ABSENT": 0.75, "TOLERANCE_EXCEEDED": 0.625, "CONVERGENCE_STALL": 1.0, "MODEL_FIDELITY_LIMIT": 0.625, "DATA_INSUFFICIENCY": 1.0, "IMPLEMENTATION_DEFECT": 0.75}`

## ④ 结果图

* `artifacts/failure_taxonomy/figures/fig1_confusion_structured_vs_analyst.png` — 冻结集混淆矩阵（含 acc/κ 标题）。
* `artifacts/failure_taxonomy/figures/fig2_criteria_and_controls.png` — C2 与两个负控 C3a/C3b 的一致率柱图（虚线=阈值，灰虚线=1/6 随机线）。
* `artifacts/failure_taxonomy/figures/fig3_routing_coverage.png` — 逐类召回 × 该类重派建议（决策表覆盖面的可视化）。

## ⑤ 注意点（踩过的坑 / 局限）

1. **同作者偏置**：金标签与规则标注器出自同一个体（本 agent）。κ=0.75 证明“同一判断能被
   规则稳定复现”，**不**等于跨人可复现；报告如实标注，未用“双人标注”措辞粉饰。
2. **诱饵负控是强证据但非充分**：C3a 用同仓同文风的非失败段落，一致率 0.169
   （随机 0.167），说明判别力不来自“文献腔”或长度；但诱饵长度仍与正样本匹配（设计使然）。
3. **小样本**：n=48，每类 8 条；单次置换的测试一致率方差很大
   （25 次置换中最大 0.417），故 C3b 报均值而非单次值。
4. **类边界重叠**：`BASELINE_ABSENT` 与 `DATA_INSUFFICIENCY` 都常含“无/缺”；区分靠
   “标尺缺失 vs 输入缺失”，混淆矩阵中确有互串，已如实计入。
5. **重派建议未闭环验证**：本节点只证明“分类→建议”可裁，未证明按建议重派后节点真能转 PASS
   （需后续波次的真实重派实验）。
6. **消融**：监督质心臂测试一致率仅 0.583（n=12），远低于规则臂，
   说明在小样本下规则化 > 学习化，这也是部署选规则标注器的原因。

## ⑥ 误差判定

本节点的被测量是**分类一致率**，不是物理量，因此“是否 <3%”不适用（无物理基准可对标）：
如实声明该判据不适用于本任务，替代判据为 C1 κ=0.750、
C2 一致率=0.792（随机 1/6=0.167），两项均达标。
所有数字由 `benchmarks/failure_taxonomy/run_p186w1_n08.py` 从冻结语料程序化生成（零手抄），
`metrics.json` 为唯一来源。

## 附注 · 发布状态（如实记录）

* 报告正文已入库平台：`tensor_reports.id=11013`，`mission_id=9338`，六要素齐全、零手抄。
* **三张图未能挂载**：首次 POST 时后端 `uploads/` 返回 `ENOSPC`（磁盘 100%）→ `figures=[]`；
  磁盘恢复后重投被 `uq_reports_mission_md5` 唯一约束拒绝（同一正文 md5 已存在），
  后续重投（含单图、无图变体）返回的是**其它 mission 的报告对象**（跨租户串话），
  疑为平台侧会话复用缺陷——不在本 worktree 权限内，故不动平台代码，如实登记。
* 图件真实存在于本仓：`artifacts/failure_taxonomy/figures/fig1..fig3.png`（sha 随 commit 固定），
  可用 `PYTHONPATH=src python benchmarks/failure_taxonomy/run_p186w1_n08.py` 重新生成。
