# 验收判据可证伪性台账：全台账 mutant kill-rate 与"新判据必须自带 killer test"门禁

**节点** IN50-W3-N08 ｜ **仓** TensorPhysics ｜ **工件** `tools/audit/criteria_mutation.py`、
`tests/test_criteria_mutation.py`、`artifacts/criteria_mutation/` ｜ **门禁**
`.github/workflows/criteria-mutation-gate.yml`

---

## ① 案例介绍（对象、参数、判定口径）

TensorPhysics 的每一个验收门都是一个**判决器**：它对交付的工件说 PASS / FAIL /
WITHHELD。单元测试全绿只证明判决器会跑，不证明它会抓——一个分支被删掉的验收门和
一个合格的验收门在 CI 上是同一副面孔。`tests/test_contract_mutation.py` 曾在**单门**上
做过这件事（阻力链契约门）；本节点把它升级为**全台账 KPI + 合并门禁**。

对象不是物理场，而是"判据本身"：对每一条判据，喂给它**它本来要守住的证据的损坏
副本（mutant）**，统计它拒绝了多少。

| 参数 | 取值 |
|---|---|
| 台账规模 | 5 个验收门（4 个 TensorPhysics 代码门 + 1 个外部判决器）、22 条判据（17 条实测判据 + 5 条恒真负控） |
| 代码门 | `p20_hydrostatic`（CAD→水动力交接，H1–H6）、`p20_seakeeping`（BEM→耐波/砰击交接，S1–S5）、`vv_gci`（网格收敛 GCI 门，5 个检查）、`vv_validation`（ASME V&V 20 验证门） |
| 外部判决器 | TensorPlatform `tools/paper_gate/traceability_gate.py`（PAPER-TRACE 稿件溯源门，596 行）+ `tools/audit/paper_gate_mutation_audit.py`，186 个 AST 单点变异体 |
| mutant 类 | M1 数值扰动 ε∈{1 %,3 %,15 %}、M2 单位/量纲错位、M3 常数替身、M4 样本伪装、M5 基准替换 |
| KPI | 台账 kill-rate ≥ 90 %；逐门 kill-rate；weakest-mutant 清单；M1 ε 阶梯单调 |
| 门禁 | 相对 `baseline_ledger.json` 的**新增**判据必须 kills ≥ 1，否则 CI 退出码 2 |
| 精度口径 | 平台规则：>15 % 误差不可接受；M1 的 ε 阶梯上限即取 15 % |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **台账 adapter（消费，不重写）**。`Criterion = {id, gate, callable, expected_verdict,
   evidence_ref}` + 声明式 guard。`callable_ref = "module:function"` 直接 `import`
   已发货的门函数，adapter 只负责把声明式 case 变成函数调用，再从返回的判决对象里
   读取**指定的子检查**（`ContractCheck.check_id` / `vv` 检查 dict 的 `check` 字段）。
   门函数一行未改。外部判决器更是通过它**自己的 per-mutant 记录**消费
   （`--record-external` 把两份审计 JSON 压成 `artifacts/criteria_mutation/paper_trace_audit.json`，
   台账只做 kill-rate 重算）。
2. **mutant 引擎（域无关）**。`# BEGIN-DOMAIN-FREE-ENGINE … # END-DOMAIN-FREE-ENGINE`
   之间的代码只认识"数、单位因子表、样本元数据、provenance 字符串"，
   `scan_engine_lexicon()` 用 19 个领域词（hull/ship/draught/seakeeping/…）扫描该区间并
   断言为 0。M1 用 `x→x(1+ε)`，M2 用 `UNIT_FACTOR[dim]`（m→mm、m³→litre、rad/s→Hz、
   无量纲→percent），M3 用"看起来合理的替身常数"（错版本号、他船吃水、满秩常数数组、
   空标签），M4 改样本元数据不改结果（删条目、删声明键、复制样本、改网格标号 h ×2），
   M5 把基准替换成被判决值或恒等式（`identity:Cb*L*B*T`、`identity:f_sim`）。
3. **两种判决口径，都报**。`gate_only` = 只调发货门函数；`ledger` = 门函数 + 判据声明的
   **provenance guard**（基准不得是被判决值 / 声明常数必须与审计值一致）。差额就是
   "门本身的盲点"，不能拿 guard 把盲点糊过去。
4. **负控**。5 条恒真判据（`AlwaysAccept`，`assert True` 家族）各挂 1 个 M1 mutant，
   预期 0/5 且被标 `unfalsifiable`；负控不计入 KPI 分子分母。
5. **合并门禁**。`pr_gate(current, baseline)`：`baseline_ledger.json` 是被接受过的
   判据集合（棘轮），任何新 id 若 `n_killed < 1` 即 REJECT → 退出码 2。红绿对照由
   `--pr-demo` 生成：green = 现有台账（ACCEPT, exit 0），red = 现有台账 + 1 条
   "无 killer test 的新判据"（REJECT, exit 2），两对 JSON 落在
   `artifacts/criteria_mutation/pr_gate_demo/`。
6. **工具链**：纯 Python（无新依赖）、`pytest` 18 条、GitHub Actions 两条工作流步骤
   （测试 + 重新生成台账后跑 PR 门禁）。重计算（外部判决器 186 变异体）走集群。

## ③ 计算结果（数字表，全部由程序写入 `ledger.json`）

**表 1 台账总账**（`artifacts/criteria_mutation/ledger.json`）

| 指标 | 数值 |
|---|---|
| 验收门 | 5（4 代码门 + 1 外部判决器） |
| 判据可实例化 | 22/22 = 100 % |
| mutant 覆盖（有 ≥1 mutant 的判据） | 22/22 = 100 % |
| mutant 总数（含在带内 4 个） | 219 |
| kill | **204/219 = 93.15 %**（KPI ≥ 90 % → 达标） |
| gate-only kill（去掉 provenance guard） | 201/219 = 91.78 % |
| 恒真负控 | **0/5**，5 条全部标记 `unfalsifiable` |
| M1 ε 阶梯 kill | ε=1 %: 4/5、3 %: 5/5、15 %: 5/5 → **违例 0** |
| 台账运行时长 | 0.74 s（预算 300 s） |

**表 2 逐门 kill-rate**

| 门 | 判据 | mutant | gate-only | ledger | 说明 |
|---|---|---|---|---|---|
| `p20_hydrostatic` | 6 | 11 | 90.9 % | **100 %** | H6 的 M5（`identity:Cb*L*B*T`）只有 provenance guard 抓得住 |
| `p20_seakeeping` | 5 | 6 | 100 % | **100 %** | 单位错位（rad/s→Hz、条目数≠频率数）被 S3/S4 直接杀 |
| `vv_gci` | 5 | 11 | 100 % | **100 %** | 网格标号 h 被改（M4）→ 观测阶数与 h 不符即 FAIL |
| `vv_validation` | 1 | 5 | 40 % | **80 %** | 见 ⑤：无 U 值膨胀盲点（M3 只被一致性 guard 杀）；1 % 扰动在 U_val 内属等价变异体 |
| `paper_trace`（外部） | 186 | 186 | 92.5 % | **92.5 %** | 硬化的 3 文件测试集 172/186；原始单文件 92/186 = 49.5 % |

**表 3 PAPER-TRACE 基线复算（外部判决器自报记录 → 本台账重算，非转抄）**

| 运行 | 测试集 | mutant | killed | kill-rate | 记录自洽 |
|---|---|---|---|---|---|
| baseline | `test_paper_traceability_gate.py`（21 条） | 186 | **92** | **49.46 %** | True |
| hardened | +`..._kills.py`(15) + `..._kills_v2.py`(36) | 186 | **172** | **92.47 %** | True |

与战役验收判据要求的 92/186 = 49.5 %（±0）与 ≥168/186 逐条对齐。

**表 4 集群预算实测**（外协方 loong64 双节点）

| 分片 | 节点 | 命令 | mutant | 墙钟 |
|---|---|---|---|---|
| 外部判决器 baseline | n315 | `paper_gate_mutation_audit.py --jobs 32 --tests <21 条>` | 186 | **184.4 s** |
| 外部判决器 hardened | n315（本次 n330 取源超时，另在 99 控制面复算） | 同上 + 2 个 killer 测试文件 | 186 | **185.2 s**（99, `--jobs 24`）/ 分片并行墙钟取 max ≈ **185 s < 300 s** |
| TensorPhysics 台账本体 | 任意 | `criteria_mutation.py` | 33 本仓 mutant | **0.74 s** |

## ④ 结果图

| 图 | 文件 | 读法 |
|---|---|---|
| 图 1 | `artifacts/criteria_mutation/fig1_gate_killrate.png` | 逐门 kill-rate，蓝=ledger、橙=gate-only、红虚线=90 % KPI、绿点线=台账总账 93.2 %。差值即"门本身的盲点" |
| 图 2 | `artifacts/criteria_mutation/fig2_epsilon_ladder.png` | M1 ε 阶梯 kill 数与 in-scope mutant 数，标注违例数 0 |
| 图 3 | `artifacts/criteria_mutation/fig3_class_matrix.png` | 门 × mutant 类热图（n/a = 该类未被该门声明），空白处正是"未声明覆盖" |

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **等价变异体不是盲点**。`VVVAL-ratio` 的 M1 ε=1 % 存活：`f_sim=1.008→1.01808`，
   误差 0.01808 ≤ 声明的 `U_val=0.02`，ASME V&V 20 明文判 PASS。台账把它单列为
   `equivalent_survivors`（附 ε 与容差），不计入 `blind_spots`；`blind_spots` 为空。
2. **门本身的 M5 盲点是真的**。`p20_hydrostatic` H6 的闭合式 `Cb·L·B·T ≈ ∇` 在
   ∇ 由同一式子反算时是恒等式，门无法分辨；`vv_validation` 的 `f_exp := f_sim` 同理。
   台账用"基准必须独立来源"的 provenance guard 补上，并用 `gate_only` 列把差距显式留证
   （90.9 % / 40 %）。
3. **`u_num` 膨胀是 `vv_validation` 的真实盲点**：把 `u_num=0.02` 改成 0.2 后门照样
   PASS（ASME 只比对 `|e| ≤ U_val`）。台账加了"声明常数必须等于审计常数"的一致性
   guard 才杀掉；这属于平台已有的"declared_formula_matches_impl"模式，不是新语义。
4. **`permute` 是等价变异体**：`gci` 的层集会被 `_sorted_levels` 按 h 重排，打乱列表
   顺序不产生任何差异（实测 PASS），故 M4 改用"网格标号 ×2 / 复制层"这类真正改样本
   元数据的操作。这条坑已固化为测试 `test_every_mutant_changes_its_case`（任何
   mutant 若不改变 case 即失败）。
5. **"域无关"只覆盖引擎区**。判据 case 里必然出现 `displacement_m3`、`frame.x` 等领域
   字段——这属于 adapter 声明，不属于 mutant 引擎。扫描严格限定在两行 marker 之间，
   并用"投毒自检"证明扫描能失败（测试里插入 `# propeller` 后必须报出）。
6. **集群**：本次 n315 成功复算 baseline（184.4 s），n330 在该时刻 GitHub 取源超时
   （`Failed to connect to github.com port 443 after 134056 ms`）导致分片未完成；
   台账本体的预算（<5 min）不受影响（0.74 s）；外部判决器复算按"两节点各一片、
   取 max"设计，实测两片各 184.4 s / 185.2 s，并行墙钟 ≈185 s < 300 s（预算达标），
   但**本次未能在两节点上同时跑完**，分片墙钟为单节点实测 + 串/并行推断，如实标注。
7. **适用范围**：M1–M5 是**数据层**变异，不覆盖源码层算符变异（那是 PAPER-TRACE
   的 `ast` 变异负责的域）；两者在台账里并存（代码门用 M1–M5，外部判决器用它自己的
   186 个算符变异体）。

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 量 | 基准 | 实测 | 相对偏差 | <3 % ? |
|---|---|---|---|---|
| PAPER-TRACE 基线 kill-rate | 92/186 = 0.494624 | 92/186 = 0.494624 | 0（逐位相同，±0） | **是** |
| PAPER-TRACE 硬化 kill-rate | 172/186 = 0.924731 | 172/186 = 0.924731 | 0 | **是** |
| 台账 kill-rate vs KPI 90 % | 0.90 | 204/219 = 0.931507 | +3.5 %（相对 KPI，方向上优于阈值） | 是（优于阈值） |
| 恒真负控 kill-rate | 0/5 | 0/5 | 0 | **是** |
| M1 扰动幅值 vs 声明 ε | 1 %/3 %/15 % | 实测比值偏差 < 1e-12 | ≤1e-10 % | **是** |
| 台账运行时 vs 预算 300 s | 300 s | 0.74 s（本仓 mutant）；外部判决器 baseline 184.4 s（n315） | 远低于预算 | **是** |

> 结论：所有有基准的量都 <3 %；`vv_validation` 单门 kill-rate 80 % 低于 90 % 门线，
> 这是**如实记录的 weak gate**（1 个等价变异体，见 ⑤-1），不改口径、不剔除。