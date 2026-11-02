# TensorPhysics 核心测试基线漂移巡更（m1830）

- 仓库：`TensorPhysics` @ `1b67e7b`（分支 `agent/m1830`）
- 巡更时刻（UTC）：2026-11-02T00:25:32Z
- 运行时：Python 3.11.6 / pytest 9.2.0
- 变更面：`tests/test_sir_phase3_contract.py`（+63/−4）、`tools/test_drift.py`（新增）、
  `docs/reports/m1830/`（本报告与插图脚本）、`artifacts/tests/`（junit 与签名工件）

## 1. 案例介绍（被巡更对象与物理/工程背景）

TensorPhysics 是「后端中立的 HPC+AI 多物理场编排」仓库，核心测试套件（`tests/`）
覆盖三类可机裁事实：

1. **SIR 破冰战役契约**（`test_sir_phase3_contract.py`、`test_sir5_contract.py`、
   `test_sir3_contract.py`）：把 SUBOFF 破冰真实化战役的 prereg 冻结件钉成
   值-文件-键三元组，要求每个声明值当场从工件复算，禁手抄；
2. **流固耦合链**（`test_si1..si4`、`test_coupling.py`）：平板/梁在流体载荷下的
   单向/双向耦合与四库联合编排，含守恒量与双账本门；
3. **平台与工件面**（`test_artifact_store*`、`test_platform_*`、`test_worker*`、
   `test_scheduler*`）：工件 URI/校验和/谱系、注册表与调度限额。

物理背景与几何（与三元组相关的实船/模型尺度）：战役对象为 SUBOFF 全附体潜艇
模型（含指挥台围壳与尾附体）在冰区航行的水动力/接触问题；G1 门要求最细网格档
阻力系数误差 |err| ≤ 3%（对标 ITTC-1957 平板参考 Cf）。该门的量化证据（`G1_err_finest_pct`、
`G1_pass`、`OVERALL`、`Cf_ref`）落在集群 NFS 的 `suboff_sail` 运行目录里，
本研发容器**不可见**，构成本次巡更的核心分类问题（见 §5）。

巡更目标：重跑核心套件 → 产出通过/失败签名 → 区分**环境红**（外部证据/算力不可得）
与**真红**（代码/契约回归）→ 新增真红即修 → 立基线供后续巡更 diff。

## 2. 怎么做的（方法、流程、工具链）

**仓库内既有基线记录**：巡更前检索 `artifacts/`、`docs/reports/`、`runs/`、git 历史，
**未发现任何测试结果基线工件**（仅有 m175 的 FVM 交换验证报告，与测试签名无关）。
因此本次先建基线，再以基线为参照做漂移判定。

流程（全部可复跑）：

1. **基线采集**：`python3 -m pytest tests -q --junitxml=artifacts/tests/junit_<stamp>.xml`，
   全量 2 轮（轮次间不加缓存 `-p no:cacheprovider`），确认失败签名确定性；
2. **签名压缩**：新增 `tools/test_drift.py`，把 junit XML 压成
   `artifacts/tests/baseline_signature.json`（`nodeid → passed/failed/skipped/env_red`，
   附 summary、git rev、版本、durations、history），**数字只从 junit 读**；
3. **红/环境红分类规则**（写入测试代码，非人工判断）：
   - `repo` 类三元组（工件在仓内）→ 任何环境都必须复算，值不符即 `AssertionError`（真红）；
   - `external` 类三元组（工件前缀 `/nfs/`）→ 文件可见即复算；不可见抛 `LookupError`，
     由 `test_external_evidence_recomputes` 显式 `skip` 并报清单（环境红），
     `TP_STRICT_EXTERNAL_EVIDENCE=1` 时改判红（集群/CI 严格模式）；
   - 反空心门：`repo` 可复算三元组数必须 ≥ 12（基线实际 14），
     外部证据不得成为基线主体；
4. **探针验证**（不靠自述）：严格模式判红探针、仓内三元组篡改探针、还原后重跑、
   巡更幂等性 diff；
5. **插图**：`docs/reports/m1830/make_figures.py` 从 junit/基线 JSON 直读绘图。

## 3. 计算结果（数字全部程序化注入）

### 3.1 通过/失败签名（run1 = 建档，run2 = 巡更）

| 关键量 | 基线 run1 | 本次 run2 | Δ | 判定 | 相对误差 | 是否 <3% |
|---|---|---|---|---|---|---|
| total testcases | 263 | 263 | 0 | exact | 0.00% | 是 |
| passed | 262 | 262 | 0 | exact | 0.00% | 是 |
| failed（真红） | 0 | 0 | 0 | exact | 0.00% | 是 |
| env-gated（环境红） | 1 | 1 | 0 | exact | 0.00% | 是 |
| 仓内可复算三元组 | 14 | 14 | 0 | exact | 0.00% | 是 |
| 外部三元组（本环境不可执行） | 7 | 7 | 0 | exact | 0.00% | 否（本环境无基准可复算） |

注：`run1` 建档于本单修复之后，故 run1/run2 已一致；**修复前**状态由
「run1 签名 − 本单新增用例 + 契约用例按前缀探针置红」程序化派生：
total=262, passed=261, failed=1。
前缀探针（`artifacts/tests/probe_prefix_contract.xml`，用 HEAD 版测试文件回放）实测
25 passed / 1 failed，
唯一失败用例为 `tests/test_sir_phase3_contract.py::TestBaselineTriples::test_every_triple_recomputes`，失败原因=外部证据文件不存在（环境红，非代码逻辑红）。

### 3.2 非通过用例清单（本次唯一非绿项，已分类为环境红）

| nodeid | run1 | run2 | run2 t(s) |
|---|---|---|---|
| tests/test_sir_phase3_contract.py::TestBaselineTriples::test_external_evidence_recomputes | env_red | env_red | 0.002 |

缺失外部工件（3 个）：
- `/nfs/wangxi/runs/suboff_sail_20261008/e5_verdict.json`
- `/nfs/wangxi/runs/suboff_sail_20261008/prereg.md`
- `/nfs/wangxi/runs/suboff_sail_20261008/verdict.json`

### 3.3 基线三元组按证据可达性分类

- 总计 **21** 组：仓内硬复算 **14** 组（本环境每次都执行）；
  外部证据 **7** 组（本环境环境红档）。
- 完全依赖外部证据的缺口：g1_fluid_load_precision, g7_reference_anchors
  —— 即 G1（流体载荷精度）与 G7（参考锚/掩码事实）在本容器内**不可复算**。

### 3.4 耗时分布（巡更预算定位，run1 直读）

| testcase（run1 实耗） | s |
|---|---|
| tests.test_si1_one_way_coupling::test_linear_superposition | 103.34 |
| tests.test_si1_one_way_coupling::test_plate_sine_load | 22.14 |
| tests.test_si3_two_way_coupling::test_two_way_coupling_integration | 12.4 |
| tests.test_si4_full_chain.TestFullChainOrchestration::test_full_chain_runs_three_steps | 5.62 |
| tests.test_si4_full_chain.TestFullChainOrchestration::test_full_chain_dual_ledger_gates | 5.4 |
| tests.test_si4_full_chain.TestFullChainOrchestration::test_full_chain_four_libraries_participate | 4.2 |
| tests.test_end_to_end_example::test_end_to_end_example_writes_checksummed_artifact_and_manifest | 0.48 |
| tests.test_end_to_end_example::test_platform_registration_requires_shared_artifact_uri | 0.28 |

## 4. 结果图

![suite status](fig_suite_status.png)

**图 1** 套件状态签名：两轮全量运行的 passed/env-gated/failed/skipped 构成
（262/1/0/0
与 262/1/0/0）。
失败项归零、环境红项显式可见。

![triple availability](fig_triple_availability.png)

**图 2** 契约三元组按证据可达性分类（14 仓内 vs 7 外部）。
G1/G7 两个缺口全部落在 `/nfs/`，本环境只能登记为环境红。

![top durations](fig_top_durations.png)

**图 3** 最慢 15 个用例：单一耦合用例
`tests.test_si1_one_way_coupling::test_linear_superposition`（103.34 s）主导套件时延，巡更预算应主要控此处（见 §5 局限）。

## 5. 注意点（踩过的坑、局限、适用范围）

1. **环境红 ≠ 真红**：`test_every_triple_recomputes` 原先对 21 组三元组一律硬断言，
   而其中 7 组指向集群 NFS 产物（`/nfs/` 在本容器根本不存在）。
   结果是任何容器/CI 环境下套件恒红，且红因与代码无关 —— 这正是基线漂移巡更最难
   分辨的一类噪声。修法不是删断言，而是**给证据分类**：仓内硬门 + 外部显式环境红档，
   并保留 `TP_STRICT_EXTERNAL_EVIDENCE=1` 让集群侧恢复全硬模式。
2. **skip 的残余风险**：外部证据档在本环境的复算能力为零，若 `verdict.json` 里的
   `G1_err_finest_pct` 在集群侧漂移，本容器**检测不到**。已用反空心门（repo ≥ 12）
   与严格模式探针约束，但根治仍需把外部产物镜像进仓（受「禁手抄数字」纪律约束，
   本轮不做：没有原始工件就无法造可信校验和，抄 README 表格即等于手抄）。
3. **机器噪音**：本轮巡更与其他 agent 的全量 pytest 并发（`/proc/loadavg` 峰值约 700，
   128 核），单轮 wall clock 由 122 s 涨到 177 s。故**时长不可跨轮比较**，
   签名（passed/failed 计数）是不变量，也是漂移判定的唯一可靠量。
4. **环境红的口径边界**：`EXTERNAL_EVIDENCE_ROOTS = ("/nfs/",)` 是显式白名单，
   新增外部证据根必须改代码，避免「未知路径悄悄跳过」变成漏检后门。
5. **并发重复**：巡更开始时池内 m1831/m1832/m1836 亦在跑同类基线任务（见 `pgrep` 快照），
   存在算力与结论重复，后续建议按仓分派而非按 agent 抢跑。
6. **图注字体**：渲染机无 CJK 字体（仅 DejaVu/Noto Sans），故图内标签用英文，
   避免豆腐块；报告正文保持中文。

## 6. 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）

| 关键量 | 基线 run1 | 本次 run2 | Δ | 判定 | 相对误差 | 是否 <3% |
|---|---|---|---|---|---|---|
| total testcases | 263 | 263 | 0 | exact | 0.00% | 是 |
| passed | 262 | 262 | 0 | exact | 0.00% | 是 |
| failed（真红） | 0 | 0 | 0 | exact | 0.00% | 是 |
| env-gated（环境红） | 1 | 1 | 0 | exact | 0.00% | 是 |
| 仓内可复算三元组 | 14 | 14 | 0 | exact | 0.00% | 是 |
| 外部三元组（本环境不可执行） | 7 | 7 | 0 | exact | 0.00% | 否（本环境无基准可复算） |

**结论**：可机裁的计数类关键量（total/passed/failed/env-gated、仓内与外部三元组数）
与基线**逐项完全相等，相对误差 0.00%，全部 < 3%**，漂移=0（新增真红 0、消失用例 0）。
唯一**无基准可判**的量是外部证据组（G1 精度证据 `G1_err_finest_pct` 等，
3 个文件 / 7 组三元组）——本环境无 `/nfs/` 挂载，无法复算，
因此如实标注「否（本环境无基准可复算）」，不以 skip 冒充通过，也不以历史数字冒充实测。

### 探针结论表

| 探针 | 判据 | 实测 | 结论 |
|---|---|---|---|
| 严格模式 TP_STRICT_EXTERNAL_EVIDENCE=1 | 外部证据不可见 → 判红 | 1 failed（pytest.fail 触发） | PASS（门未被空心化） |
| 仓内三元组篡改探针（g3 值改为 CORRUPTED_DRIFT_PROBE） | 值-文件-键复算不符 → 判红 | 1 failed（三元组值不符） | PASS（真红可捕获） |
| 篡改后 git checkout 还原 + 契约自测重跑 | 26 passed / 1 skipped | 262 passed 全量无红 | PASS（可安全还原） |
| 巡更幂等性（run2 vs run1 签名 diff） | 新增真红 = 0 | new_red=0 | PASS |

## 7. 复跑命令

```bash
cd TensorPhysics
python3 tools/test_drift.py --update-baseline   # 建档/刷新基线签名
python3 tools/test_drift.py                     # 巡更并 diff（新增真红 → exit 1）
TP_STRICT_EXTERNAL_EVIDENCE=1 python3 -m pytest tests -q   # 集群严格模式
python3 docs/reports/m1830/make_figures.py      # 重绘本报告插图
```
