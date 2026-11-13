# m18751 交差核验：GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB（m18658）board-audit 裁定

**审计对象**：mission 18658（GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB · 冰压压力-面积参数库与 `ex` 标定 · TensorPhysics，status=succeeded）
**审计人**：board-audit（本单 m18751）
**落表**：`tensor_maturity_verdicts` id=764，gate_state=pending（平台 maturity_gate 自动裁定）
**报告**：tensor_reports id=17418（六段 + 判据图 1 张）

**裁定**：**L2**（L1/L2 实测成立；L3 无调用方、L4 无外部基准 → 未记）
**基准裁判**：`tensor_benchmark_cases WHERE anchor_slug='GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB'` → **0 行**（全表 110 行无此锚点）→ 判据降级为本单自建的「ref/got 是否外部可裁」审查（见 §2）。

---

## 一、三件套核验（逐项，本单独立重跑）

复核方式：把 m18658 的 `src/tensorphysics/ice_pressure_area`、`ice_modules`、`data/ice_pressure_area`、`tests/test_ice_pressure_area.py`、`scripts/run_gap_pp2_pressure_area.py`、`benchmarks/ice_nav_refs` 复制到 `/tmp/m18658_repro`，**不在原工作树内跑**（避免污染被审工件）。

| 项 | 判据 | 核验结果 |
|---|---|---|
| run_artifacts | 真实工件存在且可复现 | ✅ `artifacts/ice_pressure_area/calibration.json`(32316 B) + `reference_table.md` + `figures/fig1..fig4.png`(146/102/82/63 KB) 全部存在；**本单重跑 3.5 s，calibration.json 逐字节相同**（sha256 `9a5479ee76f059bb…`，两路一致，reproduced=True） |
| ref vs got | 数值在容差内 | ✅ 8/8 PASS，逐行复算一致：R1 0.45/0.45(1.2e-14 %)·R2 0.185/0.185648(0.350 %)·R3/R4 定比不变性(1.2e-14 %)·R5 闭式 vs 差分(1.1e-09 %)·R6 1.32/1.29370(1.99 %)·R7 1.74/1.82366(4.81 %)·R8(2.1e-07 %)；**注意 R6/R7 是「用声明的 ex 带反演平台实测带宽」，ref 来自 gap 陈述本身，属自洽复核而非外部基准** |
| repro_cmd | 可执行、单命令 | ✅ `PYTHONPATH=src python scripts/run_gap_pp2_pressure_area.py`（3.5 s，无网络，纯 stdlib，末尾打印 `{"all_ref_vs_got_pass": true, "n_pass": 8, "n_rows": 8}`）；`python -m tensorphysics.ice_pressure_area --strict` → **rc=2**，与 mission 记录一致（设计如此：blocking family conflict） |

**测试**：`pytest tests/test_ice_pressure_area.py -q` → **37 passed (0.59 s)**；25 个 test 函数 / 62 处 assert，断言的是不变量而非「跑通即过」：字节漂移检测、占位符引用拒收、`ex` 越带拒收、定比口径不变、过程区导数上界、DL 等方差退化=算术平均、异质放大区间、不可辨识协变量 WITHHELD、**实测层不落在声明带内（诚实负结果）**、验收跑 8 检查。

**数据锚**：`data/ice_pressure_area/pressure_area_sources.json` 的 `file_sha256` 与 `MANIFEST.json` 逐字节一致（本单复算 True）；7 条记录的逐条 sha256 齐备。

**报告六段**：tensor_reports id=17383（mission_id=18658，markdown 4691 B，figures 4 张），标题段 ①案例介绍 ②怎么做的 ③计算结果 ④结果图 ⑤注意点 ⑥误差判定 —— **六段齐全**。

---

## 二、基准裁判（唯一裁判缺位）

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB';
-- (0 rows)；全表 110 行，按 anchor_slug 分组无此锚点
```

→ **F1_no_registered_judge**：任务书指定的唯一裁判无该锚点注册行，`tolerance` 无平台权威值。m18658 侧自报的容差（1 %、5 %、1e-6 %）**是被告自己设定的**，不构成独立判据。本单据此不采信「8/8 PASS」作为级别依据，仅采信其作为**内部一致性证据**。

逐项把 ref 的独立性过一遍（这是 m18658 自报「8/8 PASS」最容易被误读的地方）：

| 行 | ref 的真实来源 | 独立性 |
|---|---|---|
| R1 | IACS 记录的 `reported.ex=0.45`，而该记录的 `data_points` 由 `p=2.5A^-0.45` 生成 | **同一公式自拟合**（恒等式级，非验证） |
| R2 | 树内 `FREDERKSON_REF` 表声明 ex=0.185，fit 得 0.185648 | 内部一致性（0.35 %，非外部真值） |
| R3/R4 | 解析定理 `ex_ref = ex_src`（r≡1） | ✅ 真数学验证（独立于数据） |
| R5 | 闭式 `r=(1+t/2)/(1+t)` vs 中心差分 | ✅ 真数值验证 |
| R6/R7 | 平台自报带宽 1.32x/1.74x，用 ex=0.30/0.70 反演 | 自洽（且**两端反演面积比不一致 13.4 %**，报告已诚实记录） |
| R8 | 同 R5 的粗步收敛 | ✅ |

结论：R3/R4/R5/R8 是**真验证**（L1 成立），R1/R2/R6/R7 是**内部一致性**。这四项不能替代 `external_anchor`。

---

## 三、判级（L0–L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 只有文字 | — |
| L1 数学/单元验证 | 解析恒等式 + 数值交叉 | ✅ **成立**：面积口径转换律 `ex_ref = 1+(ex_src−1)r` 的闭式 vs 中心差分 1.1e-09 %、定比不变性 bit-identical、过程区导数上界 `|Δex| ≤ (1−ex)t/2`、DL 等方差退化=算术平均（37 测试覆盖） |
| L2 独立功能实现 | 真代码：输入→计算→输出 | ✅ **成立**：2371 行纯 stdlib 实现（`library.py` 545 / `calibrate.py` 421 / `conventions.py` 304 / `run_*.py` 582 / `tests` 408），输入=sha256 锚定 JSON 库，输出=分层合并 ex + 区间 + 覆盖率 + 冲突分级；确定性重跑逐字节一致 |
| L3 系统集成 | 接入主求解器/平台 | ❌ **不成立**：全仓 `grep -rn ice_pressure_area src/` 在包**外**零命中 —— 无任何求解器/板格设计路径 import 它；`artifacts/.../consumer_crosscheck` 只是**只读比较** `ice_modules/local_pressure.py`，消费方模型未改一行（其自带表 ex≈0.15，与本单标定 0.450 相差 3 倍，报告自己的 crosscheck 表把这条差异列到 62–223 %，属诚实负结果，但也正说明「未接入」） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ **未记**：可复现 ✅（逐字节），误差分析 ✅（但 ref 为内部锚），**完整算例/external_anchor ✗** —— 5/7 条 accuracy_reference 记录 `primary_source_verified=false`（`fail_stage=primary_source_verification`, `failure_reason=no_network_egress`），且 `tensor_benchmark_cases` 无注册行 |
| L5 科研工程成果 | — | 未达（L3/L4 未记） |

**裁定 L2。** 附 145 行 docs + 4 张图 + 1116 行 calibration.json 不改变级别：级别=功能实质，本单实质是「库 + 代数 + 分层合并」的独立实现，未被任何生产路径调用、无外部真值裁判。

---

## 四、发现（findings，按严重度）

- **F1_no_registered_judge**（blocking 级别依据缺失）：`tensor_benchmark_cases` 无锚点注册行 → 自报容差无权威来源。补一行 `(anchor_slug, metric='ex_design_rule', ref=0.45, tolerance=…)` 才能让 L4 可判。
- **F2_external_anchor_withheld**（blocking L4）：7 条记录中 5 条一手原文未打开（ISO 19906 / IACS UR I2 / Masterson & Frederking 1993 / Sanderson 1988 / Johnston 1998），其中 **design_rule 层被合并的两条（ISO / IACS）恰恰都在未核验名单里**。→ 「ex = 0.450 [0.408, 0.492]」目前是**两次仓内转写的合并**，不是一手数据标定。
- **F3_narrowing_claim_independence_sensitive**（major，本单新增量化）：报告的招牌数字「带宽收窄 5.04×」对两条 design_rule 记录的**独立性假设**高度敏感。同一面积比 2.3593 下复算：合并区间 [0.4084,0.4916] → 5.04×；单条 IACS 带 [0.40,0.50] → 4.19×；两条视为同源（取并带 [0.40,0.55]）→ **2.74×**。即 5.04× 里约 1.8× 来自「假设两条同源转写相互独立」这一未经验证的假设，DL 的 √2 收缩不保。文档 §5 已承认 design_rule `tau=0` 是「转写一致性而非测量一致性」，但摘要仍以 5.04× 作为交付结论 → **建议表述改为「4.19×（单条带）～5.04×（假设独立）」，并注明不可验证**。
- **F4_conflict_with_consumer**（major，能力缺口指向下一轮）：标定值 ex=0.450 与平台在用 `ice_modules/local_pressure.py` 的 ex=0.15 相差 3 倍（同一 A=1 m² 处 p 相同 → 大 A 处分歧 62–223 %），而本单**未改消费方**。gap 的原始诉求（统一口径/收缩带宽）在平台层面**未闭环**，只在库层面闭环。这也是 L3 不成立的具体含义。
- **F5_honest_negatives_recorded**（正面项）：measured_local 层 ex=0.618 [0.495, 0.741] 明确判「不落在声明 1.32x–1.74x 内（上限 1.889 > 1.824）→ 声明上限对实测局部压力不保守」，且不因不好看而改判据；`wetted_surface_area` 计数=0 如实记为「不能给经验换算系数」。诚实度合格。
- **F6_report_qa_pending**（minor）：report id=17383 `qa_status=pending`，`server_quality_audit.gate_valid=false / key_present=false` —— 报告未被机器 QA 闸过，不影响内容判定。

## 五、checks_not_run

- 集群 loong64 复跑（单机 3.5 s 已完成，未上集群）
- 一手原文核验（ISO 19906 / IACS UR I2 等 5 条）：无外网出口，与 m18658 同环境限制 → 未能升级 `primary_source_verified`
- `tensor_benchmark_cases` 注册行写入（属平台配置面，非本单授权范围）
- 消费方 `ice_modules/local_pressure.py` 接入改造（属 L3 工作，本单只核验不改）

## 六、处置（disposition）

**approve with notes**：m18658 的成果真实、可复现、诚实标注负结果 → 判 **L2 确认**（不返修、不消账到 L3/L4）。后续升级路径：① 注册 `tensor_benchmark_cases` 行；② 有网环境补 2 条以上一手原文（尤其 design_rule 层）；③ 把 ex 标定接入板格/局部压力设计路径并给出消费方误差（→ L3），④ 接外部实测表做误差分析（→ L4）。