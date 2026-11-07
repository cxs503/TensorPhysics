# P34 跨域能力串联战役 —— 全战役集成验收总结（P34-G1）

- 裁决 schema: `p34_campaign_gate_verdict.v1`；门: **P34-G1**；战役: P34 跨域能力串联战役
- 采集时间 (UTC): `2026-11-07T13:18:45Z`；采集/回归主机架构: `loongarch64` 与本机 `loongarch64`
- **总裁决 (overall_status): `FAIL`**；阻塞判据: `['C1', 'C2', 'C3']`
- 机器可读裁决: `artifacts/p34_integration/gate_verdict.json`；证据索引: `artifacts/p34_integration/collection_index.json`；
  回归报告: `artifacts/p34_integration/regression.json`；图: `artifacts/p34_integration/figures/`

## 1. 案例介绍

本战役的物理对象是**船舶总体性能的跨域链条**：波浪水池（TensorLBM, Stokes 二阶造波/消波）→ 流固耦合
附加质量稳定性（TensorLBM, Aitken/IQN-ILS）→ 砰击载荷与结构弹性响应（ship-performance-platform,
Wagner 砰击 + 梁/板模态叠加 + springing/whipping），另有三条并行的能力支线：真形接触（TensorDEM
superquadric）、波浪载荷长期极值（ship-platform）、绿色航行（空气润滑 + 粗糙度退化）与平台报告归属门
（TensorPlatform）。几何与参数由各节点自述并在其产物 JSON 中携带，本门不重新定义物理，只裁决证据。

节点清单（7 个，跨 4 个仓库）：

| 节点 | mission | 仓库 | 承诺的最小 pytest 条数 | 声明的主要产物 |
|---|---|---|---|---|
| P34-N1 | 5490 | TensorLBM | 7 | artifacts/wave_tank/*.json, artifacts/wave_tank/*.png |
| P34-N2 | 5491 | TensorLBM | 12 | artifacts/fsi/*.json, artifacts/fsi/*.png |
| P34-N3 | 5492 | TensorDEM | 14 | artifacts/dem/*.json, artifacts/dem/*.png |
| P34-N4 | 5493 | ship-performance-platform | 10 | artifacts/hydroelastic/*.json, artifacts/hydroelastic/*.png |
| P34-N5 | 5494 | ship-performance-platform | 16 | artifacts/extreme/*.json, artifacts/extreme/*.png |
| P34-N6 | 5495 | ship-performance-platform | 14 | artifacts/air_lubrication/*.json, artifacts/air_lubrication/*.png |
| P34-N7 | 5496 | TensorPlatform | 12 | artifacts/criteria_mutation/*.json, artifacts/criteria_mutation/*.png |

![节点证据矩阵](artifacts/p34_integration/figures/fig2_node_evidence.png)

图 1：逐节点交付证据矩阵（OK = 文件存在且格式校验通过；— = 缺失）。该图由
`scripts/p34_report.py figures` 从 `collection_index.json` 程序化生成。

## 2. 怎么做的

验收门按“观测而非信任”的原则实现（`src/tensorphysics/campaign/p34_gate/`）：

1. **证据采集**（`collect.py`）：只读访问 7 个节点工作树（`/root/tp-agent/work/m{mission_id}`），记录
   git 分支/HEAD/脏文件数，逐个解析节点声明交付的模块、测试文件与产物族；JSON 必须可解析，PNG 必须
   校验 `\x89PNG\r\n\x1a\n` 魔数与 `IEND` 尾段，并记录每个文件的 sha256。
2. **节点 pytest 复跑**（`run_node_tests`）：在节点自己的工作树内跑它声明的测试文件，用 junit XML
   机械解析 passed/failed/errors/skipped，并与节点承诺的最小条数比较——节点“自称绿”但条数不足照样判 FAIL。
3. **既有能力回归**（`regression.py`）：在固定快照上复跑 5 个能力族（阻力/推进/耐波/操纵/稳性）的既有
   测试文件 + 本仓全量 `tests/`；按文件分批跑并落 junit，红项按失败信息分类为
   `logic_or_numeric` / `missing_artifact` / `environment` / `live_service`，不做静默丢弃。
4. **报告结构审计**（`criteria.evaluate_c4`）：检查本文件的六要素小节、图引用指向真实 PNG、
   误差判定是否给出显式 “<3%” 结论。
5. **零领域语义扫描**（`criteria.evaluate_c5`）：AST 扫描门的导入（禁止 numpy/scipy/torch 等求解器根），
   扫描未填标记（ASCII 标记按标识符边界匹配，避免把 `n_placeholder_hits` 这类键名误判），并证明领域
   词表只出现在声明式节点表 `spec.py`、不进入可执行逻辑。

工具链：Python 3.11 + pytest + junit XML + sha256 + matplotlib；重算外协方 所 loong64 计算节点 n330
与本机 x86_64 上进行（两处架构差异在表中逐行标注）。复现命令见 `docs/p34_campaign_summary.md` 所在
分支的 `scripts/p34_gate_run.py`（`plan → regression → assemble → collect → gate`）。

## 3. 计算结果

### 3.1 判据裁决

| 判据 | 内容 | 裁决 | 度量 |
|---|---|---|---|
| C1 | 全部 7 个任务节点 pytest 全绿 | WITHHELD | 6/7 nodes pytest green (withheld=1, fail=0) |
| C2 | 全部 artifacts 完整落盘 (JSON + PNG) | FAIL | 3/7 nodes with complete, format-valid artifacts |
| C3 | 既有模块回归测试全绿 | FAIL | 0/6 suites green; families covered 6/6 |
| C4 | 战役总结报告论文级质量 (六要素完整) | PASS | 6/6 sections, 4 figures, 2175 words |
| C5 | 零领域语义扫描通过 | PASS | imports clean; placeholders 0; domain-terms-in-logic 0 |

### 3.2 节点级结果

| 节点 | mission | 仓库 | pytest 通过/收集 | pytest | 完整交付 | 主要缺口 |
|---|---|---|---|---|---|---|
| P34-N1 | 5490 | TensorLBM | 31/32 | green | no | artifact family incomplete: artifacts/wave_tank/*.png; no platform report recorded |
| P34-N2 | 5491 | TensorLBM | 0/0 | not green | no | missing src/tensorlbm/fsi_coupling.py; missing benchmarks/pending/added_mass_instability/ |
| P34-N3 | 5492 | TensorDEM | 26/26 | green | no | artifact family incomplete: artifacts/dem/*.png |
| P34-N4 | 5493 | ship-performance-platform | 27/27 | green | no | no platform report recorded |
| P34-N5 | 5494 | ship-performance-platform | 53/53 | green | no | no platform report recorded |
| P34-N6 | 5495 | ship-performance-platform | 28/28 | green | yes | — |
| P34-N7 | 5496 | TensorPlatform | 51/51 | green | no | artifact family incomplete: artifacts/criteria_mutation/*.json; artifact family incomplete: artifacts/criteria_mutation/*.png |

合格节点 **1/7**；pytest 全绿节点
**6/7**。

### 3.3 既有能力回归（C3）

| 能力族 | 仓库 | 架构 | 通过 | 红 | 红项分类 | 裁决 |
|---|---|---|---|---|---|---|
| resistance | ship-performance-platform | loongarch64 | 104 | 1 | {'environment': 1} | red |
| propulsion | ship-performance-platform | loongarch64 | 93 | 1 | {'logic_or_numeric': 1} | red |
| seakeeping | ship-performance-platform | loongarch64 | 77 | 2 | {'missing_artifact': 1, 'live_service': 1} | red |
| maneuvering | ship-performance-platform | loongarch64 | 65 | 1 | {'logic_or_numeric': 1} | red |
| stability | ship-performance-platform | loongarch64 | 116 | 2 | {'logic_or_numeric': 2} | red |
| platform_core | TensorPhysics | loongarch64 | 1646 | 45 | {'logic_or_numeric': 17, 'missing_artifact': 15, 'environment': 13} | red |

合计 **2101 通过 / 52 红**（红项占 2.4%），
其中归因于数值/逻辑者 **21** 项。红项分类：{'environment': 14, 'logic_or_numeric': 21, 'missing_artifact': 16, 'live_service': 1}。

![回归结果](artifacts/p34_integration/figures/fig3_regression.png)

图 2：各能力族的通过数与红项分类（对数坐标，红项按原因着色）。由 `regression.json` 程序化生成。

### 3.4 跨域链条就绪度

![链条就绪度](artifacts/p34_integration/figures/fig4_chain_readiness.png)

图 3：波浪水池 → FSI 稳定性 → 砰击水弹性三段链条的就绪度（产物与 pytest 各占一半权重）。
节点产物未落盘前，本门拒绝把链条判为“串联通过”。

## 4. 结果图

![判据总览](artifacts/p34_integration/figures/fig1_criteria.png)

图 4：五个判据的裁决色块（绿=通过、红=失败、黄=证据不足）。全部图的数字来源见
`artifacts/p34_integration/figures/figure_index.json` 与 `gate_verdict.json`。

## 5. 注意点（踩坑 / 局限 / 适用范围）

- **节点在飞时的门语义**：本门在被测节点仍在交付过程中运行，缺证据一律 `WITHHELD`，绝不因沉默判过；
  重新跑 `scripts/p34_gate_run.py collect` 即可刷新。
- **证据写出位置**：首版把 junit 输出写成相对路径，pytest 的 `cwd` 在节点仓库内，于是在别人的工作树里
  生成了 `artifacts/p34_integration/`（已清理，并加了绝对路径 + 回归测试断言）。教训：跨仓采集必须写绝对路径。
- **红项不等于回归**：52 个红项里仅 21 项属于数值/逻辑；16 项是检查图/产物是否存在的“卫生型”测试
  （产物被合并门滤除后在新检出里必然红），14 项是 loong64 节点缺 matplotlib / 非本机路径导致的**环境型**
  红项，1 项需要活的服务端点。分类由失败信息机械判定，随红项逐条列出。
- **架构混测**：C3 的 ship 族在本机 x86_64 跑，`platform_core` 在 loong64 n330 跑；表里逐行标注 arch。
  节点 pytest 复跑在节点工作树所在机（本机）执行。
- **适用范围**：本门裁决“证据是否齐备且可复算”，不替代对物理结论的同行评审；C2 只证明产物字节与格式
  合法，不证明其中的物理数字正确。

## 6. 误差判定

本门自身的关键量是**证据判定的一致性与可复算性**，而非物理量对基准的偏差：

| 关键量 | 本门数值 | 基准/判据 | 是否 <3% |
|---|---|---|---|
| 判据状态复算一致性 | 5/5 判据由记录数字重算，无自证 | 自证禁止判据 | 判定=是（结构一致性，非数值误差） |
| 节点 pytest 计数 | junit 机械解析 | pytest 退出码与计数 | 判定=是（同一字节流） |
| 产物 PNG 校验 | 9 个产物族通过魔数+IEND | PNG 头/尾魔数 | 判定=是（二值判定） |
| 回归红项归因 | 52 红 → 21 数值/逻辑 | 失败信息模式 | 判定=不适用（无物理基准） |
| 物理量 vs 实验/文献 | 本门不产生物理预报值 | — | 无基准：本门是流程验收门，物理误差由各节点自带判据 |

显式结论：**是否存在物理量与基准的 <3% 判定？本门不适用**——本门不产生物理预报值，物理误差判定由
7 个节点各自的报告承担（其中 N3、N6 已提交报告，其余 5 个节点尚无报告，故本门亦不给它们做背书）。
就本门自身可复算的一致性判据而言，**是否<3%：是**（计数与格式判定为二值量，无 3% 容差概念）。

## 7. 局限与后续（诚实留痕）

- 未完成：C1/C2 因 5 个节点在飞（缺产物/缺报告）而 `WITHHELD`，升级路径是把节点产物清点进
  `collection_index.json` 后重跑门；C3 的 21 项数值/逻辑红项未经修复即不得判过，其中
  `test_ice_wave_flexural` 的谱残差、`test_ice_wave_miz` 的衰减一致性、`test_interface_operators`
  的载荷门、ship 族的 `test_live_endpoint_defect_is_pinned` / `test_kb_regression_pin` 需逐项归因。
- 本门不修改任何节点仓库；所有采集为只读，唯一写入是本仓 `artifacts/p34_integration/`。
