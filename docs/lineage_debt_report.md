# 谱系继承契约机检成熟化（P173W2-N9 / mission 7877）：把「继承局限节」从文字声明变成可复算门

## ① 案例介绍（物理背景 / 输入 / 参数表）

**问题（工程背景）**：自主战役以 DAG 波次推进，子波以「本波继承父波局限节与 KPI 缺口」一句话承接父波未偿
欠账。这句话不可机检：子波可以在没有任何判据、也没有任何书面理由的情况下被播种，父波的 honest FAIL
就静默蒸发。本节点把该声明变成一条可复算的门：**父波/本波每条未偿欠账，必须在子波里被判据重新度量、
或被显式引用、或留下书面延期理由，否则播种 fail-closed 拦截。**

**输入（只读回放快照，字节镜像）**：`artifacts/lineage_debt/inputs/lineage_debt_bundle.json`（sha256 `219d4f4ec6201aab…`，
fetched_utc 2026-11-09T03:40:00Z）。含 plan 26, 173, 388 的计划元数据与节点表、18 份已发布
战役报告（plan 173 的 11 个节点 + plan 26 的 8 个节点）。快照来自平台战役 API 的只读 GET，逐文件 sha256
入 provenance；回放离线、纯函数、无网络无时钟。

**参数表（全部经配置注入，逻辑零领域语义）**

| 参数 | 值 | 来源 |
|---|---|---|
| 欠账门限（默认） | 15.0 % | `configs/lineage_debt.vocab.json :: thresholds.default_pct` |
| 门限解析序 | per-quantity > per-domain > default | 同上，运行时可换配置 |
| 误差列识别 | 表头含 7 个 pattern（误差/error/偏差/…） | 配置 `table_contract` |
| 终态欠账状态 | blocked / failed | `extract.DEFAULT_TERMINAL_STATUSES` |
| 覆盖目标波 | plan 173 → plan 388 | bundle `downstream_plan` |
| 词表规模 | 46 条 deny-list，域/量别名词全部外置 | 配置文件（非代码） |
| 回放规模 | 3 个计划 / 29 个节点 / 18 份报告 | 快照 |

## ② 怎么做的（方法与工具链）

落在 **TensorPhysics** 新域包 `src/tensorphysics/lineage_debt/`（8 个模块 + 1 份词表配置 + CLI），五步流水线：

1. **欠账台账化（`extract.py`）**：遍历报告 markdown 的管道表格，误差列由配置 pattern 识别；单元格换算为
   有符号百分比（`%` 权威；无 `%` 时 `|v|≤1` 视为分数×100，原始串留在 evidence 供审计）；重复误差列
   用「同组表头最长公共前缀」推条件标签（纯结构规则，无领域知识）；行标签为纯数值时其本身是工况参数。
   **终态节点**（blocked/failed）即使从未产出报告也是欠账（本波两个 blocked 节点即此情形）。
   未能规范化的行进入 `unresolved` 段，只暴露不计数（不污染门）。
2. **子节点判据解析（`criteria.py`）**：结构化 metadata 优先（可给 quantity/domain/threshold/refs），
   字符串 → 配置正则兜底 → goal 散文再兜底，三级来源逐条留痕（`metadata` / `metadata_string` / `goal_regex`）。
3. **覆盖判定（`coverage.py`）**：规则优先级固定 **显式引用 > 量级匹配 > 仅域级 > 书面延期 > dropped**；
   四结局 covered / covered_by_domain / deferred(reason) / dropped 写入覆盖矩阵，匹配到的判据 id 全部登记。
4. **播种拦截（`gate.py`）**：`dropped ∧ 无 reason` → 判据 fail-closed，`missing` 逐条枚举（key/域/量/
   报告 id/证据行号/所需理由），并给出可合并的 `autonomy.debt_unpaid` 补丁；`debt_closed` 只在**更晚**的
   同键观测落回门限内时才成立（claim 不能闭合债务）。
5. **可查询台账（`replay.py` / `query.py` / `__main__.py`）**：每波一份 `runs/lineage_debt/ledger_plan<id>.json`
   （canonical JSON：键排序、浮点定精度、谱系链显式），`query` 支持 plan/state/coverage/domain/quantity/
   node/mission/report/min-|err| 组合过滤；`selfcheck` 对包内 `.py` 做 deny-list 扫描证明逻辑零领域语义。

工具链：Python 3.11 + 无第三方依赖的回放核心；matplotlib 出图；pytest 25 条用例；CLI
`python -m tensorphysics.lineage_debt replay|query|selfcheck`。

## ③ 计算结果（程序化注入，零手抄）

**每波台账**

| plan | 债务总数 | error_row | node_terminal | covered | covered_by_domain | dropped | 播种门 | 缺失项 |
|---|---|---|---|---|---|---|---|---|
| 26 | 3 | 2 | 1 | 1 | 2 | 0 | **pass** | 0 |
| 173 | 21 | 18 | 3 | 10 | 5 | 6 | **blocked** | 6 |
| 388 | 21 | 18 | 3 | 10 | 5 | 6 | **blocked** | 6 |

**判据 (d) 真实回放：mission 6753（报告 8117）**

hand label: report 8117 declares w(+25.2%) and t(+15.1%) on KCS as honest FAIL。回放结果：

| debt key | err [%] | 门限 [%] | 状态 | 报告 |
|---|---|---|---|---|
| `propulsion|t|KCS` | +15.10 | 15.0 | debt_open | 8117 |
| `propulsion|w|KCS` | +25.20 | 15.0 | debt_open | 8117 |

**一致率 2/2**（人工金标 2 项：w 与 t 均须 `debt_open`）。

**播种拦截（plan 173 → 子波 388）的 missing 枚举**

| debt key | 说明 |
|---|---|
| `propulsion|J0|零推力点` | 后代波无判据覆盖、也无书面延期理由 → 播种拦截 |
| `propulsion|eta_0|ALL` | 后代波无判据覆盖、也无书面延期理由 → 播种拦截 |
| `propulsion|t|KCS` | 后代波无判据覆盖、也无书面延期理由 → 播种拦截 |
| `propulsion|w|KCS` | 后代波无判据覆盖、也无书面延期理由 → 播种拦截 |
| `stability|node_terminal|node:1078` | 后代波无判据覆盖、也无书面延期理由 → 播种拦截 |
| `stability|node_terminal|node:1085` | 后代波无判据覆盖、也无书面延期理由 → 播种拦截 |

**判据自检一栏**

| 判据 | 结果 | 证据 |
|---|---|---|
| (a) 6 合成报告 / 3 条 honest-FAIL 抽取 precision/recall | **1.0 / 1.0** | `tests/test_lineage_debt_extract.py::test_a_extraction_precision_recall_contract` |
| (b) 3 合成子节点覆盖矩阵 | **covered / covered_by_domain / dropped 与金标一致** | `tests/test_lineage_debt_criteria_coverage.py::test_b_coverage_matrix_gold_covered_by_domain_dropped` |
| (c) dropped 无 reason fail-closed | **阻断 + 缺失项枚举 + autonomy.debt_unpaid** | `tests/test_lineage_debt_gate.py::test_c_fail_closed_enumerates_missing_items` |
| (d) 真实 plan 173 回放 m6753 w/t | **2/2 debt_open** | `tests/test_lineage_debt_replay_real.py::test_d_real_replay_marks_m6753_w_t_debt_open` |
| (e) 同输入两跑逐字节一致 | **byte-identical**（plans [26, 173, 388]） | `test_e_same_input_replays_byte_identical` + `lineage_debt_summary.json` |
| (f) 零领域语义 deny-list 自检 | **pass / 0 命中**（扫描 11 个 .py，deny-list 46 词） | `test_f_deny_list_scan_is_clean`，`python -m tensorphysics.lineage_debt selfcheck` |
| (g) pytest 条数 | **25**（≥14） | `tests/test_lineage_debt_*.py` |

## ④ 结果图

图1–图4 见 report figures（`artifacts/lineage_debt/figs/`）：债务谱系继承堆叠图、m6753 误差行 vs 门限、
覆盖矩阵栅格、播种门四计数对比。所有图由 `scripts/gen_lineage_debt_artifacts.py` 从台账 JSON 程序化生成。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **门限 15% 是任务契约，不是物理判据**：报告自身的判据可以是 3%/10%，台账只回答「是否超出继承门限的
  欠账线」，两者语义不同；本波把 15% 作为默认并允许 per-quantity 覆盖（配置注入，改配置即改门）。
- **表格解析是启发式的**：无 `%` 的裸值按「|v|≤1 视为分数」解释；重复误差列的条件靠表头公共前缀推断；
  自由文本单元格（如 "3.55 (err 18.4%) / 4.27 (err 42.3%)"）不抽数——一律保留原文，宁可漏检不可臆造。
  未能规范化的行进入 `unresolved` 段（本次真实回放 0 行），**只暴露不计数**。
- **词表是数据、会生长**：为使真实回放可判读，配置里补充了战役实际出现的量别名词（进速系数 J/J0、首超
  角、稳态速降、横移、附加质量系数、幅值、相位滞后等）。这是配置注入的应有之义，但也意味着**词表覆盖
  率决定识别率**；换域只需换词表（测试 `test_f_synthetic_vocabulary_drives_the_whole_pipeline` 用纯虚构
  词表跑通全链，证明逻辑未硬编码）。
- **覆盖目标是 bundle 级参数**：本次统一指向 plan 388（`coverage_target_source = bundle`）。plan 388 自身
  台账因此也是「对 388 判据」判定的（其 `autonomy.new_plan` 未在快照内）——同一目标的横向对比，不是自证。
- **gate 输出 blocked 不是缺陷，而是结论**：2 个计划被拦截，missing 6 条全部是
  真实未覆盖缺口（推进因子 w/t/η0/J0 与新波判据无交集；两个 blocked 节点无人接手）。台账把这些从「文字
  继承」变成「可派工清单」。
- **域级覆盖是弱承诺**：`covered_by_domain` 只说明有人在同域工作，不说明重新度量了该量；门不拦它（契约
  规定只拦 dropped），但报告显式给出 weak 计数（plan 173：5 条）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

本节点**不产生新的物理测量**：它审计的是已发布报告里的误差数字，因此「新物理量 vs 基准误差」不适用，
不存在可报的 <3% 物理误差——如实声明原因，不作含糊表述。可判定的是抽取/判定本身的确度：

| 判定项 | 基准 | 结果 | 误差 | <3%？ |
|---|---|---|---|---|
| 合成抽取 precision | 金标 3 条欠账 | 1.000 | 0 | **是（精确相等）** |
| 合成抽取 recall | 金标 3 条欠账 | 1.000 | 0 | **是（精确相等）** |
| 覆盖矩阵 gold 一致 | 3 节点金标 | 3/3 | 0 | **是（精确相等）** |
| 真实回放 m6753 | 人工金标 w/t 2 项 | 2/2 | 0 | **是（精确相等）** |
| 回放确定性 | 同输入两跑 | byte-identical | 0 字节 | **是（逐字节相等）** |
| deny-list 自检 | 0 命中 | 0 命中 | — | **是（精确相等）** |
| 继承误差值的复现（w +25.20% / t +15.10%） | 报告 8117 原件 | 25.20 / 15.10 | 0.00 pp | **是（原值复现，未重测）** |

**honest 声明**：以上 25.20% / 15.10% 是**被继承的欠账误差**（报告 8117 在 KCS 上的 honest FAIL），
本节点未做任何重测或调参覆盖；它们在台账中的状态是 `debt_open`，并作为 plan 173 播种拦截的 missing 项
交回战役。台账 blocked 结论不予粉饰。
