# P35 全战役集成验收门 (P35-G1) — 跨域能力串联与回归基线确认

- 产物 schema: `P35-G1` → `campaign_readiness.v1` (fingerprint `0a06491ea401e5b2423910992992c8418f14fc955518b80dda9cb5f41d804ba8`)
- 战役计划: plan_id=179 · P35·多域求解器前沿与船舶深域拓展 · 10 任务节点 + 本门
- 生成时间 (UTC): `2026-11-08T09:25:35.159592+00:00`
- **总裁决: `FAIL` · dispatcher_ready = `false` · 阻塞项 25 条**

## 1. 案例介绍（这是什么、为什么需要一道门）

P35 是跨四仓的十个并行研究节点：求解器前沿（TensorLBM 热多相相变 N1、造波探针反演 N3、LBM-DEM-FSI 三向耦合 N9）、颗粒湿冰（TensorDEM N2）、船舶深域（空气润滑减阻 N4、轴系校中 N5、LNG 舱 rollover N6、替代燃料 EEXI/CII N7、散货自热 N8）与跨求解器串联账本（TensorPhysics N10）。节点之间靠**产物**而非代码互相消费，因此战役是否可交付不能用「PR 是否合并」判断，只能用「产物是否落盘、能力是否登记、基线是否退化、依赖是否可执行」判断。本门（P35-G1）把这四件事做成机器可算判据，并输出派发器可直接消费的 `campaign_readiness.v1`。

| 节点 | mission | 仓库 | 物理域 | 成熟化 |
|---|---|---|---|---|
| P35-N1 | 6795 | TensorLBM | multiphase_flow | 否 |
| P35-N2 | 6796 | TensorDEM | ice_navigation | 否 |
| P35-N3 | 6797 | TensorLBM | wave_basin_inversion | 否 |
| P35-N4 | 6798 | ship-performance-platform | green_navigation | 是 |
| P35-N5 | 6799 | ship-performance-platform | shaft_alignment | 是 |
| P35-N6 | 6800 | ship-performance-platform | cargo_safety | 是 |
| P35-N7 | 6801 | ship-performance-platform | green_navigation | 是 |
| P35-N8 | 6802 | ship-performance-platform | cargo_safety | 否 |
| P35-N9 | 6803 | TensorLBM | ice_navigation | 否 |
| P35-N10 | 6804 | TensorPhysics | hydroelasticity | 否 |

## 2. 怎么做的（判据、工具链、可复现命令）

判据实现于 `src/tensorphysics/campaign/p35_gate/`（stdlib-only，不 import 任何求解器栈）：`spec.py` 只声明身份/产物族/依赖，`collect.py` 采集平台证据与磁盘产物，`criteria.py` 判 (a)–(g)，`regression.py` 复跑既有验证面，`gate.py` 组装并校验 `campaign_readiness.v1`（schema 注册在 `campaign/schema_defs.py`）。

| 判据 | 内容 | 证据来源 | 非绿语义 |
|---|---|---|---|
| (a) | 每节点：平台报告 ≥1 且声明的产物族在交付检出内解析通过（JSON 可解析 / PNG 魔数+IEND 尾校验） | 平台 reports API + 节点交付 worktree | FAIL |
| (b) | 台账登记必须携带 ext_id、physics_domain，且 mission_repo 等于声明仓库 | 平台 capability-delta API | FAIL |
| (c) | 复跑已晋升案例，|Δ|/max(|baseline|,floor) ≤ 1%；机器地板量级指标用声明 floor | 生产者自有 runner 输出 + 案例 result.json 基线 | WITHHELD |
| (d) | 声明 DAG 无环、边可解析、无孤立节点，且与 campaign_plans.dependencies 一致 | spec 声明（节点 goal 原文）+ 平台 DAG 记录 | FAIL |
| (e) | 成熟化节点须有成熟化声明，TRL≥6 且门禁记录全通过 | 节点产物目录内 maturity.json | FAIL |
| (f) | 载荷通过注册 schema 校验（closed schema，fingerprint 固定） | campaign/schema_defs.py | PASS |
| (g) | 任一判据非 PASS → overall=FAIL 且逐条落阻塞清单；WITHHELD 永不等于 PASS | 本门自身 | FAIL |

复现命令（证据全部落在 `artifacts/p35_gate/`）：

```bash
# 1) 采集平台证据 + 逐节点绑定交付检出 + 复跑回归 + 出裁决
python -m tensorphysics.campaign.p35_gate \
    --base-url "$TENSOR_API_BASE" --plan-id 179 \
    --node-root P35-N1=/path/to/m6795 ... --node-root P35-N10=/path/to/m6804 \
    --run-probes --probe-timeout 1800 --out artifacts/p35_gate
# 2) 长时回归探针（120 s shell 上限之外单独跑，逐例追加记录）
python3 scripts/p35_gate_probes.py --node-root TensorLBM=/repo/TensorLBM \
    --only couette_3d --timeout 900 --out artifacts/p35_gate
# 3) 图与报告
python3 scripts/p35_gate_figures.py && python3 scripts/p35_gate_report.py
```

## 3. 计算结果（全部由产物注入，零手抄）

### 3.1 判据裁决

| 判据 | 内容 | 裁决 | 说明 |
|---|---|---|---|
| (a) | 10 个节点产物落盘率 100% | ❌ FAIL | 4/10 节点产物落盘 (落盘率 40.0%) |
| (b) | 能力台账登记率 100% (ext_id/repo/physics_domain) | ❌ FAIL | 0/10 节点能力台账登记 (登记率 0.0%); 台账契约要求 evidence 携带 ext_id 与 physics_domain |
| (c) | 回归基线偏差 <1% | ⚪ WITHHELD | 1/3 已复跑案例 PASS；已复跑 1 例，最大偏差 1.262e-19 (阈值 1.0%) |
| (d) | 依赖链完整且无孤立节点 | ❌ FAIL | 平台 campaign_plans.dependencies 未记录 1 条声明边 (如 [('P35-N1', 'P35-N10')]); 平台记录了未声明的依赖边: [('P35-N3', 'P35-N10')] |
| (e) | 成熟化节点 (N4/N5/N6/N7) 门禁通过率 100% | ❌ FAIL | 0/4 成熟化节点门禁通过 (通过率 0.0%)；要求 TRL>=6 且成熟化声明中门禁全通过 |
| (f) | campaign_readiness JSON 合法且通过 schema 校验 | ✅ PASS | campaign_readiness.v1 通过注册 schema 校验 |
| (g) | honest-FAIL 语义：任一节点未通过 → FAIL + 阻塞清单 | ❌ FAIL | 5 条判据未通过 → honest FAIL，25 条阻塞项落清单 |

### 3.2 判据数字

| 指标 | 值 | 阈值/目标 | 是否达标 |
|---|---|---|---|
| 产物落盘率 (a) | 40.0% (4/10) | 100% | 未达标 |
| 能力台账登记率 (b) | 0.0% (0/10) | 100% | 未达标 |
| 回归最大偏差 (c) | 1.262e-19 | < 1.0% | 达标 |
| 回归案例复跑数 (c) | 1/3 | 3 | 未达标 |
| DAG 无环 (d) | True | True | 达标 |
| 孤立节点 (d) | 无 | 无 | 达标 |
| 平台依赖边一致 (d) | 缺失 1 / 多余 1 | 0 / 0 | 未达标 |
| 成熟化门禁通过率 (e) | 0.0% (0/4) | 100% | 未达标 |
| 链就绪节点 (d) | 4/10 | 10/10 | 未达标 |

### 3.3 回归基线复跑明细 (criterion c)

| 案例 | 基线 | 复跑值 | 相对偏差 | 阈值 | floor | 耗时 | 裁决 |
|---|---|---|---|---|---|---|---|
| couette_3d | 9.596263e-14 | 9.596263e-14 | 1.262177e-19 | 1.0% | 1.000000e-10 | 260.3 s | ✅ PASS |
| cavity_natural_convection | 2.1854 | — | — | 1.0% | 0 | 1800.1 s | ⚪ WITHHELD |
| cavity_3d_full | -0.27142 | — | — | 1.0% | 0 | — s | ⚪ WITHHELD |

- `cavity_natural_convection`：probe exceeded 1800s budget
- `cavity_3d_full`：案例未启动：测算成本 19400 s 超出本门预算 3600 s (declared estimate; 见 spec 实测标定)

### 3.4 节点就绪矩阵

| 节点 | 产物(a) | 台账(b) | 成熟化(e) | 链就绪(d) | 阻塞摘要 |
|---|---|---|---|---|---|
| P35-N1 | ❌ FAIL | ❌ FAIL | ⚪ WITHHELD | ❌ | deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/thermal_multiphase/**/*.json; 产物族未落盘/无效: png artifacts/thermal_multiphase/**/*.png; registrati |
| P35-N2 | ✅ PASS | ❌ FAIL | ⚪ WITHHELD | ✅ | registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N2'); physics_domain 缺失; repo 不符 (ledger=None != 'TensorDEM') |
| P35-N3 | ❌ FAIL | ❌ FAIL | ⚪ WITHHELD | ❌ | deliverable:FAIL: 平台未收到该节点报告 (reports=0); registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N3'); physics_domain 缺失; repo 不符 (ledger=None != 'TensorLBM') |
| P35-N4 | ❌ FAIL | ❌ FAIL | ❌ FAIL | ❌ | deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/air_lubrication/**/*.json; 产物族未落盘/无效: png artifacts/air_lubrication/**/*.png; registration:FAI |
| P35-N5 | ❌ FAIL | ❌ FAIL | ❌ FAIL | ❌ | deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/shaft_alignment/**/*.json; 产物族未落盘/无效: png artifacts/shaft_alignment/**/*.png; registration:FAI |
| P35-N6 | ✅ PASS | ❌ FAIL | ❌ FAIL | ✅ | registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N6'); physics_domain 缺失; repo 不符 (ledger=None != 'ship-performance-platform'); maturity:FAIL: 缺失成熟化声明 artifacts/lng_rollover_m6 |
| P35-N7 | ✅ PASS | ❌ FAIL | ❌ FAIL | ✅ | registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N7'); physics_domain 缺失; repo 不符 (ledger=None != 'ship-performance-platform'); maturity:FAIL: 缺失成熟化声明 artifacts/alt_fuel_propul |
| P35-N8 | ✅ PASS | ❌ FAIL | ⚪ WITHHELD | ✅ | registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N8'); physics_domain 缺失; repo 不符 (ledger=None != 'ship-performance-platform') |
| P35-N9 | ❌ FAIL | ❌ FAIL | ⚪ WITHHELD | ❌ | deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/lbm_dem_fsi/**/*.json; 产物族未落盘/无效: png artifacts/lbm_dem_fsi/**/*.png; registration:FAIL: 能力台账无 |
| P35-N10 | ❌ FAIL | ❌ FAIL | ⚪ WITHHELD | ❌ | deliverable:FAIL: 产物族未落盘/无效: json artifacts/p35_chain/**/*.json; 产物族未落盘/无效: png artifacts/p35_chain/**/*.png; registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N10'); physics_do |

## 4. 结果图

![fig1](figures/fig1_node_readiness.png)

*图 1 · 十节点就绪矩阵：绿=判据通过，红=未通过，灰=证据未取得（WITHHELD）。成熟化列对非成熟化节点为 WITHHELD（该节点本无成熟化要求）。*

![fig2](figures/fig2_regression_deviation.png)

*图 2 · 回归复跑偏差（对数轴）。绿柱为实际复跑并达标的案例；灰位为未复跑案例（超预算或缺 runner 入口），按 WITHHELD 记账而不是当作未退化。*

![fig3](figures/fig3_couette_profile.png)

*图 3 · 图 2 中唯一复跑案例的物理内容：Couette 三维剖面与解析线性解重合，逐点偏差停在 float64 舍入地板（~1e-17 量级），说明观测到的 l2 是舍入噪声而非离散误差——这正是该案例用 floor 而非百分比比较的原因。*

## 5. 注意点（踩过的坑与适用范围）

- **产物是否落盘只能在节点自己的交付检出里判**：门用 `--node-root EXT_ID=PATH` 绑定节点 worktree，缺失即 FAIL；若只绑定仓库级 `--repo-root`，缺失只能是 WITHHELD（可能看的是别的分支）。这一区分是本门第一版刻意加的，避免把「看错检出」写成「节点没交」。
- **成熟化声明不计入落盘率**：N4–N7 的 `maturity.json` 由判据 (e) 判，重复计入 (a) 会让一个成熟化缺陷同时打两次，属双重计分。
- **平台 DAG 用 node_id 而非 ext_id**：`campaign_plans.dependencies` 存的是 1095..1105，直接与 `P35-N*` 比较会造出一堆假 divergence；门先做 id→ext_id 映射再比对。
- **机器地板量级指标必须用 floor**：Couette fp64 的 l2≈9.6e-14，复跑 5.7e-14 会是「40% 偏差」的假回归，物理上却是同一台机器噪声；门要求逐案声明 floor 并写进裁决。
- **回归 runner 的算力成本要事前标定**：cavity_3d_full 64³ 实测 500 步 97 s（x86 CPU/8 线程），100k 步约 1.94e4 s ≈ 5.4 h，超出本门 1 h 预算，按 WITHHELD 记账并给出成本估计——不跑也不假装跑过。
- **判据的时效性**：战役节点仍在并行推进，本裁决是 `generated_utc` 时刻的快照（快照 sha256 见溯源表），重跑即得新状态；门本身不做「乐观推测」。

## 6. 误差判定（逐项回答「是否 <3%」）

- couette_3d: |Δ|/max(|baseline|,floor) = 1.262e-19 → **<3% 达标**（阈值 1%）
- cavity_natural_convection: 未取得复跑值 → **<3% 不可判定**（probe exceeded 1800s budget）
- cavity_3d_full: 未取得复跑值 → **<3% 不可判定**（案例未启动：测算成本 19400 s 超出本门预算 3600 s (declared estimate; 见 spec 实测标定)）

- 结论（回归面）：1/3 例取得复跑值，其中 1 例偏差低于阈值（最大 1.262177e-19），2 例无法判定——因此 (c) 不判 PASS。
- 结论（跨域交付面）：产物/台账/成熟化三项判据的「误差」不是数值误差而是缺件，按缺件计数（第 3.2 节），不做数值百分比讹代。
- honest-FAIL 声明：本门当前为 **FAIL**，不作为可覆盖意见；阻塞清单见下文第 7 节。

## 7. 阻塞清单（dispatcher 消费）

- `a:FAIL:10 个节点产物落盘率 100%`
- `b:FAIL:能力台账登记率 100% (ext_id/repo/physics_domain)`
- `c:WITHHELD:回归基线偏差 <1%`
- `d:FAIL:依赖链完整且无孤立节点`
- `e:FAIL:成熟化节点 (N4/N5/N6/N7) 门禁通过率 100%`
- `P35-N1: deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/thermal_multiphase/**/*.json; 产物族未落盘/无效: png artifacts/thermal_multiphase/**/*.png`
- `P35-N1: registration:FAIL: 能力台账无登记 (capability delta not booked)`
- `P35-N2: registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N2'); physics_domain 缺失; repo 不符 (ledger=None != 'TensorDEM')`
- `P35-N3: deliverable:FAIL: 平台未收到该节点报告 (reports=0)`
- `P35-N3: registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N3'); physics_domain 缺失; repo 不符 (ledger=None != 'TensorLBM')`
- `P35-N4: deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/air_lubrication/**/*.json; 产物族未落盘/无效: png artifacts/air_lubrication/**/*.png`
- `P35-N4: registration:FAIL: 能力台账无登记 (capability delta not booked)`
- `P35-N4: maturity:FAIL: 缺失成熟化声明 artifacts/air_lubrication/maturity.json`
- `P35-N5: deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/shaft_alignment/**/*.json; 产物族未落盘/无效: png artifacts/shaft_alignment/**/*.png`
- `P35-N5: registration:FAIL: 能力台账无登记 (capability delta not booked)`
- `P35-N5: maturity:FAIL: 缺失成熟化声明 artifacts/shaft_alignment/maturity.json`
- `P35-N6: registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N6'); physics_domain 缺失; repo 不符 (ledger=None != 'ship-performance-platform')`
- `P35-N6: maturity:FAIL: 缺失成熟化声明 artifacts/lng_rollover_m6800/maturity.json`
- `P35-N7: registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N7'); physics_domain 缺失; repo 不符 (ledger=None != 'ship-performance-platform')`
- `P35-N7: maturity:FAIL: 缺失成熟化声明 artifacts/alt_fuel_propulsion/maturity.json`
- `P35-N8: registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N8'); physics_domain 缺失; repo 不符 (ledger=None != 'ship-performance-platform')`
- `P35-N9: deliverable:FAIL: mission状态=failed; 平台未收到该节点报告 (reports=0); 产物族未落盘/无效: json artifacts/lbm_dem_fsi/**/*.json; 产物族未落盘/无效: png artifacts/lbm_dem_fsi/**/*.png`
- `P35-N9: registration:FAIL: 能力台账无登记 (capability delta not booked)`
- `P35-N10: deliverable:FAIL: 产物族未落盘/无效: json artifacts/p35_chain/**/*.json; 产物族未落盘/无效: png artifacts/p35_chain/**/*.png`
- `P35-N10: registration:FAIL: ext_id 缺失/不符 (ledger=None != 'P35-N10'); physics_domain 缺失; repo 不符 (ledger=None != 'TensorPhysics')`

## 8. 溯源（审计链）

| 输入 | 路径 | 存在 | sha256 |
|---|---|---|---|
| | `snapshot:artifacts/p35_gate/platform_snapshot.json` | True | `e8c1a55f315b551d02fbc9038ddd3c7d` |
| | `regression_records:artifacts/p35_gate/regression_records.json` | True | `d2e445e31e9d4e7629d5f50d9cd9cbe3` |
| | `checkout:P35-N1:/root/tp-agent/work/m6795` | True | `None` |
| | `checkout:P35-N2:/root/tp-agent/work/m6796` | True | `None` |
| | `checkout:P35-N3:/root/tp-agent/work/m6797` | True | `None` |
| | `checkout:P35-N4:/root/tp-agent/work/m6798` | True | `None` |
| | `checkout:P35-N5:/root/tp-agent/work/m6799` | True | `None` |
| | `checkout:P35-N6:/root/tp-agent/work/m6800` | True | `None` |
| | `checkout:P35-N7:/root/tp-agent/work/m6801` | True | `None` |
| | `checkout:P35-N8:/root/tp-agent/work/m6802` | True | `None` |
| | `checkout:P35-N9:/root/tp-agent/work/m6803` | True | `None` |
| | `checkout:P35-N10:/root/tp-agent/work/m6804` | True | `None` |

- 生成器: `tensorphysics.campaign.p35_gate/gate.py`
- schema: `campaign_readiness.v1`，fingerprint `0a06491ea401e5b2423910992992c8418f14fc955518b80dda9cb5f41d804ba8`（`schema.py` 的 closed-schema 校验器逐字段复核）

## 9. 局限

- 节点产物落盘以平台报告 + 声明产物族的磁盘解析为准：未绑定仓库根的仓库记为 WITHHELD，不当作空产物；报告存在而产物族缺失仍是 FAIL。
- 回归判据只复跑生产者自己发布的 runner，门不重写任何求解器内核；runner 入口缺失或超出计算预算的案例记 WITHHELD 并写明原因，绝不代之以「未发现退化」。
- 机器地板量级指标（如 fp64 Couette 的 l2≈1e-13）用声明的绝对 floor 参与比较，否则噪声会被百分比放大成伪回归。
- 依赖链以节点 goal 的声明为准并同时记录平台 campaign_plans.dependencies；两者不一致记为阻塞项而不是择一采信。
- 能力台账登记要求 evidence 携带 ext_id/physics_domain（平台 delta 行的第一类字段只有mission_id/repo/category），这是本门对台账契约的显式要求，属契约缺口而非门杜撰。
- 本门不复算节点内部物理（那是各节点自己的验收面）；它只确认产物存在、可解析、能力登记、基线与依赖链，因此 PASS 不代表物理正确，FAIL 也不代表物理错误。

## 10. 与战役纪律的对应

- 判据 (a) 目标 100%，实测 40.0%；判据 (b) 目标 100%，实测 0.0%；判据 (c) 目标 <1%，已复跑最大偏差 1.262177e-19；判据 (e) 目标 100%，实测 0.0%。
- 全部非绿项均进入阻塞清单，未做任何「部分通过」折算；`--require-pass` 使 CI 在本条件下以非零码退出（退出码 3）。

