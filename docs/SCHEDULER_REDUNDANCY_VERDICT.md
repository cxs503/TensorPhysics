# FEAT-TPHYS-ADJUDICATOR · 编排层冗余实践检验：wedge_icebreaking 双路对比

**任务** m18417 · **仓** TensorPhysics · **锚点** FEAT-TPHYS-ADJUDICATOR（编排层疑点）·
**场景** `scenarios/wedge_icebreaking` S1 级（DEM 冰层 + 预定楔形船体）·
**判据** s1.py 的 5 条 gate（同一 judge 代码作用于全部路径，C1）·
**工件** `runs/m18417_scheduler_redundancy/`（`comparison.json` 为本文全部数字的程序化来源）

> **结论（三选一）：降级为可选工具（keep-as-optional, do not wire into the production path）**
> 单求解器单进程场景下，编排路径与裸路径**物理逐比特相同**（Δpos=0、Δvel=0、账本差=0），
> 但编排路径要多写 64 行胶合代码（裸路径最小 loop 13 行、手写等价工件 51 行），
> 慢 6.7%，且它的「检查点链」不含任何物理状态（payload 232 B、
> 物理字段 []、`restartable=False`），
> 无法据其重启；真正能重启的是底层求解器自己的快照（164733 B，重启后与连续run**逐比特相等**）。
> 因此：**没有「多余应退役」，也没有「保留并接入生产路径」**——它对单求解器场景净增值为负/零，
> 但对多求解器耦合（S3/S4 双向交换）这一未测边界仍可能是必要的，故降级为可选工具并冻结其接入。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

`wedge_icebreaking` 是生态的跨库联调阶梯：二维楔形船首连续破冰是目标应用（破冰船-水-冰闭环）的
2D 截片。本单打的是阶梯里**最小可跑的一级 S1**：DEM 冰层 + 预定楔形船体，判据是 s1.py 预注册的
五条机器可判 gate（输出完整性、轨迹有限/时序、账本可载、**冲量账本平衡 ≤1e-3**、**断裂能账本闭合**）。

几何与参数（与 `s1.py` 的 `plan_parameters` 逐项一致，程序化注入自 `comparison.json` 的 `scenario` 段）：

| 参数 | 值 |
| --- | --- |
| 冰层离散 | nx=21 × ny=5（105 个键合颗粒，`DEMConfig` 默认半径/厚度/密度） |
| 船体折线 | (0,0)、(-0.3,0.15)、(-0.3,-0.15)（3 顶点楔形，m） |
| 船体平移速度 | (+0.2, 0) m/s |
| 原生时间步 | dt = 6.87879766796e-05 s；由 IceDEM 自动定步（`dt=None`） |
| 积分步数 | 2000 原生步（物理时程 t_end = 0.1375759533592065 s） |
| 采样间隔 | 每 20 步（轨迹 101 帧 = history.csv 101 行） |

四路同场景同判据（C1 可比性）：A=编排路径、B1=生产 CLI 裸路、B2=进程内最小直驱、B3=B2+手写等价工件。

## 2. 怎么做的（方法 / 流程 / 工具链）

| 路径 | 驱动方式 | 胶合代码 |
| --- | --- | --- |
| A 编排 | `tensorphysics.Simulation(SimulationConfig(dt=native_dt, end_time=2000·dt), [TensorDEMAdapter(IceDEM)])`，按 `run(steps=20)` 分块驱动（100 个 chunk，块间采样 + 写检查点），诊断走 `Diagnostics`，检查点走 `CheckpointStore`，溯源走 `RunManifest`/`ArtifactRef` | `runs/m18417_scheduler_redundancy/harness/run_path_a_orchestrated.py` |
| B1 裸·生产 | `python -m tensordem --hull-profile … --nx 21 --ny 5 --steps 2000 --save-every 20`（s1.py 现行调用式） | `run_path_b1_cli.py` |
| B2 裸·最小 | 进程内 `for step in range(2000): model.step()` + 采样写出；脚本内断言 `tensorphysics` 未进入 `sys.modules` | `run_path_b2_bare_loop.py` |
| B3 裸·等价 | B2 + 手写 `diagnostics.jsonl`、每 20 步一个检查点信封、含 SHA-256 的 `run_manifest.json` | `run_path_b3_bare_plus.py` |

判据执行器：`harness/judge.py`（**不 import tensorphysics / tensordem**，四条路径共用同一份代码）。
全部实验在本地控制面完成（单路 ≤ 27 s，远低于 10 min 判据线，故未上集群）。

## 3. 计算结果（数字表）

### 3.1 同判据 gate 结果（C1）

| 路径 | gate 通过 | 结论 | 冲量账本相对误差 [x, y] | 工件数 | 工件字节 |
| --- | --- | --- | --- | --- | --- |
| A 编排路径（Simulation + TensorDEMAdapter） | 5/5 | PASS | [1.6765152407001042e-14, 4.717399258250804e-14] | 110 | 848689 |
| B1 裸路径·生产 CLI（`python -m tensordem`） | 5/5 | PASS | [1.6765152407001042e-14, 4.717399258250804e-14] | 5 | 422639 |
| B2 裸路径·进程内直驱（最小 loop） | 5/5 | PASS | [1.6765152407001042e-14, 4.717399258250804e-14] | 5 | 422629 |
| B3 裸路径·直驱 + 手写等价工件 | 5/5 | PASS | [1.6765152407001042e-14, 4.717399258250804e-14] | 107 | 770484 |

**四路全过，且冲量账本相对误差逐位相同**（[1.6765152407001042e-14, 4.717399258250804e-14]），
断裂能账本闭合到 5.87e-16。

### 3.2 跨路径物理一致性（逐比特比对）

| 路径对 | max\|Δpos\| (m) | max\|Δvel\| (m/s) | 账本 max\|Δ\| | history 最大差 | 逐比特相等 |
| --- | --- | --- | --- | --- | --- |
| A_orchestrated vs B2_bare_loop | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True |
| A_orchestrated vs B1_bare_cli | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True |
| A_orchestrated vs B3_bare_plus | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True |
| B2_bare_loop vs B1_bare_cli | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | True |

### 3.3 成本与产物

| 路径 | 胶合行数（非注释、marker 内） | 2000 步墙钟 | 工件数 | 工件字节 |
| --- | --- | --- | --- | --- |
| A 编排路径（Simulation + TensorDEMAdapter） | 64 | 22.04 s | 110 | 848689 |
| B3 裸路径·直驱 + 手写等价工件 | 51 | 20.65 s | 107 | 770484 |
| B2 裸路径·进程内直驱（最小 loop） | 13 | 20.93 s | 5 | 422629 |
| B1 裸路径·生产 CLI（`python -m tensordem`） | 23 | 26.96 s | 5 | 422639 |

micobench（无采样、无工件的纯驱动开销）：裸 loop 最小 1.675 s vs 编排最小
1.875 s ⇒ **+11.9%，约 998 µs/步**。
编排路径加载成本：`tensorphysics`（scheduler+adapters+checkpoint+provenance+diagnostics）
import 1.25 s（torch 3.22 s、tensordem 0.062 s，后两者两路共有）。

## 4. 结果图

- `runs/m18417_scheduler_redundancy/figures/fig1_physics_identity.png`：两路冲量账本状态与断键计数随时间的曲线（完全重合，图题标注 max|Δpos|）。
- `runs/m18417_scheduler_redundancy/figures/fig2_cost_and_restart.png`：胶合行数 / 墙钟 / 重启载荷体积（232 B vs 164733 B，对数轴）三联图。

## 5. 净增值 / 净负担逐条实测（C2）

### 5.1 净增值（逐条带实测证据）

| # | 主张 | 实测证据 | 判定 |
| --- | --- | --- | --- |
| V1 | **分块控制**：外部驱动可在块间插采样/检查点，`run(steps=20)` 后状态回到 `initialized`，第 5 块后 `finished` | `probes.chunked_control.chunk_states = ['initialized', 'initialized', 'initialized', 'initialized', 'finished']`，时钟漂移 2.14e-15 | 真实；但裸 loop 天然支持块间插代码（0 成本），单求解器下净值为 0 |
| V2 | **诊断事件流**：一次性拿到 2002 条结构化事件（init 2 条 + `step_completed` 2000 条）（`simulation_initialized`/`solver_initialized`/`step_completed`） | `probes.terminal_state_persistence.diagnostic_event_names = ['simulation_initialized', 'solver_initialized', 'step_completed']`；A 的 `diagnostics.jsonl` 由平台生成 | 真实但可替代：B3 手写同格式 38 行（51-13 行）即等价 |
| V3 | **溯源信封**：`RunManifest`/`ArtifactRef` 做 schema 校验（run_id/时间戳/64 位 sha256/size） | A 的 `run_manifest.json` 5 条 output；B3 手写同字段 | 真实但弱：**sha256 由调用方计算**，平台不提供哈希/校验执行（两侧代码量相同） |
| V4 | **检查点链**：每 20 步一个原子写信封（共 100 个） | `path_a/checkpoints/` 100 个文件；`CheckpointStore.load` 可读回 | **无重启价值**：payload 232 B、物理字段 []、`restartable=False`；按检查点续跑 → `SimulationError: initialization failed: native DEM time 0.0 does not match simulation time 0.0013757595335920927` |
| V5 | **fail-closed 生命周期**：非法步长比时在推进任何原生步之前拒绝；运行中异常包成 `SimulationError` 并 emit `solver_failed` | `probes.dt_integrality`: state=`failed`，原生推进 0 步；`probes.midrun_solver_failure`: `SimulationError: advance/coupling failed at step 40, time 0.…`，cause=RuntimeError，事件已发=True | 真实且是**唯一无法零成本复制的项**（类型化失败 + 事件），但仅在本场景外才有价值 |

### 5.2 净负担（逐条带实测证据）

| # | 负担 | 实测证据 |
| --- | --- | --- |
| C1 | **胶合成本更高**：编排 64 行 vs 手写等价 51 行 / 最小直驱 13 行 | §3.3 表；差值 +13 行（编排反而更多） |
| C2 | **墙钟更慢**：22.04 s vs B3 20.65 s（+6.7%）；纯驱动开销 +11.9% | §3.3 + micobench |
| C3 | **步长刚性**：全局步长必须是原生步长的整数倍，非整数比在 step 0 直接 fail-closed | `probes.dt_integrality`: 1.7× → `SimulationError: advance/coupling failed at step 0, time 0.0: global dt 0.000116…`；3× 才能跑（30 全局步 = 90 原生步） |
| C4 | **检查点无读者**：库内 `CheckpointStore.load` 消费者 0 个（仅测试）；`Simulation`/`TensorDEMAdapter` 均无 restore/resume API | `probes.checkpoint_restore.simulation_public_restore_api = []`、`adapter_public_restore_api = []`；`probes.consumer_audit` |
| C5 | **失败不自动清理**：推进中异常后 `finalize` 未被调用（需调用方 `close()`） | `probes.midrun_solver_failure.finalize_called_before_close = 0`，after_close = 1；例外：init 失败路径会清理 |
| C6 | **终端状态不落盘**：`finished`/`limit_reached` 既不是事件也不写文件，只存在于内存 | `probes.terminal_state_persistence.terminal_state_events = []`、`files_emitted_by_scheduler = []`；B3 手写 1 行即落盘 |
| C7 | **不在生产路径上**：wedge 阶梯 S0–S4 对 scheduler/adapter 的引用数 = 0（S1 走 CLI 子进程）；库内引用仅 `capability_routing/integration.py`（构造 Simulation）与 `fail_revival`（类型导入） | `probes.consumer_audit.all_hits`；`grep -rn "Simulation(\|TensorDEMAdapter\|scheduler" scenarios/wedge_icebreaking/` 空 |
| C8 | **平台代码体量**：被评估的 5 个模块合计 540 行（非空非注释） | §2 表 / `comparison.json.glue.platform_modules_touched` |

被评估模块行数（程序化统计）：

| 模块 | 总行 | 非空非注释 |
| --- | --- | --- |
| `adapters/tensordem.py` | 113 | 95 |
| `checkpoint.py` | 149 | 136 |
| `diagnostics.py` | 32 | 26 |
| `provenance.py` | 164 | 141 |
| `scheduler.py` | 159 | 142 |

## 6. 结论：降级为可选工具（三选一，C3）

**判定：降级为可选工具，冻结其生产接入。**

理由（全部来自 §5 实测，不含宣传语）：

1. **物理零贡献**：四路同判据同数字，编排路径对物理结果的最大影响是 0（逐比特相同）。
2. **收益可被手写复现且更省**：平台提供的诊断流/检查点信封/溯源清单，用手写 51 行即可得到等价工件，
   比编排路径的 64 行还少 13 行、快 6.7%。
3. **检查点链是空壳**：232 B 元数据、0 个物理字段、`restartable=False`、库内 0 个
   `CheckpointStore.load` 消费者；真正可重启的是求解器自带的 `IceDEM.snapshot()/from_snapshot()`
   （164733 B，重启后 position/velocity/账本差 = 0.0e+00，
   `restart_bitwise_equal=True`）。
4. **唯一实打实的增值是类型化失败语义**（V5），价值不足以抵消 C1–C6 的胶合/刚性/空壳成本。
5. **不退役的理由**：多求解器耦合（S3/S4 双向交换、`CouplingManager`）本单**未测**（见 §7 边界），
   而分块控制 + 耦合阶段钩子在该情形下可能真有必要；退役是不可逆动作，证据不足。

本单**不改动平台代码**（零代码改动）：判决落文档 + 工件，是否执行「冻结接入」由平台级任务/owner 采纳后另行落地。

配套建议（不改变本单结论，供后续单采纳）：
- 若后续确有耦合需求，接入前必须先补 (a) adapter 级 `restore(snapshot)` 契约与 `Simulation.resume(checkpoint)`，(b) 终端状态落盘事件；
  否则「检查点链」只能算审计日志，不能算断点续跑能力。
- 单求解器场景请继续用 CLI/直驱（B1/B2），并用 B3 式手写工件补溯源，成本更低。

## 7. 注意点与边界（含 honest-FAIL 留痕，C4）

- **未按原样跑 s1.py**：脚本硬编码 `/nfs/wangxi/...`（tensor_eco 工作区、venv、历史 loads.json），本机不存在；
  本单**按其 `plan_parameters` 逐项复刻**场景参数（nx/ny/steps/save-every/hull-profile/hull-velocity），
  用本机 TensorDEM 工作树（`/root/tensor-ecosystem/TensorDEM`）执行。差异：s1.py 用 venv+子进程，本单 B1 用同版本工作树 + 子进程。
- **S2/S3/S4 不可达（honest FAIL 边界）**：s2/s3/s4 依赖 TensorFVM 源码路径与 `/nfs/wangxi/runs/...` 历史载荷，
  本机均不存在，故**多求解器/双向耦合这一条编排层最可能正当的用途未能实测**。本单结论的适用范围
  = 单求解器单进程（S1 级及等价场景），越界不外推。
- **检查点续跑的另一条路（已实测）**：原生快照 → 新建 `Simulation` 会因 `initialize` 时钟不匹配而拒绝
  （`SimulationError: initialization failed: native DEM time 0.0 does not match simulation time 0.0013757595335920927`），只有直接改写 `simulation.step/time` 私有属性才可能续跑——属未文档化体操，不作推荐。
- **micobench 噪声**：裸路径 3 次重复 1.68–1.96 s，故开销百分比（+11.9%）取 min-of-3 的差值，
  置信区间较宽；但 §3.3 的端到端差（+6.7%）与 micobench 同向。
- **判据强度**：五条 gate 只保证账本闭合与产物完整（scientific_status = not_a_physics_benchmark），
  本单比较的是**管道**而非物理精度，与 s1.py 的预注册精神一致。

## 8. 误差判定（关键量 vs 基准，逐项回答是否 <3%）

| 关键量 | 基准 | 偏差 | 是否 <3% |
| --- | --- | --- | --- |
| 编排 vs 裸 末态位置 | B1 CLI 轨迹 | 0.000e+00 m（逐比特相同） | 是（=0） |
| 编排 vs 裸 末态速度 | B1 CLI 轨迹 | 0.000e+00 m/s | 是（=0） |
| 冲量账本相对误差 | s1 判据 1e-3 | 1.677e-14（A）/ 1.677e-14（B1） | 是（≪1e-3） |
| 断裂能账本闭合 | 1e-3 | 5.869e-16 | 是 |
| 原生重启 vs 连续 run 位置 | 连续 200 步 | 0.000e+00 m（逐比特相等） | 是（=0） |
| 编排墙钟开销 | B3 等价裸路径 | +6.7%（22.04 vs 20.65 s） | **否**（刻意测成本，非精度量） |

说明：本单为「编排层是否多余」的工程判决，无物理基准量（不是精度任务）；凡有基准的量（轨迹/账本/重启）
偏差均为 0 或 ≪1e-3，唯一 >3% 的量是**成本**，且该 >3% 正是结论依据。

## 9. 复现

```bash
cd runs/m18417_scheduler_redundancy/harness
python run_path_a_orchestrated.py      # A 编排
python run_path_b1_cli.py              # B1 生产 CLI
python run_path_b2_bare_loop.py        # B2 最小直驱
python run_path_b3_bare_plus.py        # B3 直驱+等价工件
python probe_orchestration.py          # 编排行为探针（分块/重启/步长/终态/失败）
python probe_costs.py                  # 原生重启、开销 micobench、消费者审计
python compare.py                      # -> ../comparison.json（本文数字来源）
python make_figures.py && python make_verdict.py
```
