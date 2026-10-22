# S1 预注册：DEM 冰层 + 预设楔形船体

> Append-only 注册文件。先注册后派发，后续只允许追加（以 `## Addendum` 段落形式）。
> 编排者：tensor-agent mission m39，architect 角色。

## 1. 科学假设

S1 验证以下假设：在 2-D 中心力键合粒子冰层上施加预设楔形船体（polyline hull）
以规定速度平移贯穿，DEM 求解器产出的冲量账本闭合、断键能量账本非负且可复核。

- **物理范围**：DEM-only（无流体/结构耦合）。船体为刚体预设运动，不受冰反力影响。
- **成熟度声明**：Implemented 级证据。不得凭 S1 通过晋级 Benchmark-verified 或更高标签
  （见 `docs/SYSTEM_OVERVIEW.md` §7）。
- **fail-closed 语义**：沿用 S0。任一机器门校验失败即停，不消耗剩余预算。

## 2. 机器可裁判据

### 2.1 反力冲量账本平衡门

DEM 求解器在每步累积四类冲量（单位 N·s，全局 x/y 分量）：

| 账本字段 | 含义 |
|---|---|
| `tool_impulse_on_ice_Ns` | 船体对冰层的冲量（反力的负累积） |
| `support_impulse_Ns` | 固定边界支反冲量 |
| `drag_impulse_Ns` | 阻力冲量 |
| `external_impulse_Ns` | 外部载荷冲量（S1 中为 0） |

**判据**：冲量守恒闭合——

```
tool_impulse_on_ice + support_impulse + drag_impulse + external_impulse ≈ Δp_ice
```

其中 Δp_ice = m_total × (v_final - v_initial) 为冰层总动量变化。
平衡误差上限：|Σimpulse - Δp_ice| / max(|Σimpulse|, |Δp_ice|, 1e-12) ≤ 1e-3（相对容差）。

### 2.2 断键能量账本门

DEM 求解器在键断裂时记录弹性应变能释放量 `failure_energy_J`（单位 J）：
E_release = 0.5 × k_bond × extension²，在首次断裂时一次性记录。

**判据**：
1. `fracture_release_J`（累计释放能）≥ 0（物理要求：释放能不能为负）。
2. `fracture_release_J` == Σ(per-bond `failure_energy_J`)（账本闭合：累计值等于逐键求和）。
   容差：绝对差 ≤ 1e-10 J。
3. `broken_bonds`（断键数）== Σ(per-bond failure events with failure_time_s ≥ 0)。

### 2.3 输出完整性门（沿用 S0 模式）

- 求解器进程退出码在允许集 {0} 内。
- `history.csv` 可载入、至少一行数值数据、所有数值有限、时间单调非递减。
- `trajectory.pt` 可 torch.load、positions 张量非空且有限、times 单调。
- `accounting.json` 可载入 JSON 且所有数值有限。
- RunManifest + LocalArtifactStore 校验和验证通过。

### 2.4 fail-closed 停机语义

- 腿按顺序执行（S1 为单腿 DEM）。
- 任一腿校验失败即停，不消耗剩余预算。
- 判词表：`decision ∈ {s1_impulse_and_energy_ledgers_passed,
  stop_s1_and_repair_failed_gate}`。
- `scientific_status = implemented_not_benchmark_verified`。

## 3. 子任务拆分

### 子任务 A：TensorDEM 账本暴露（implementer）

- **目标**：修改 `TensorDEM/src/tensordem/dem.py` 的 `diagnostics()` 方法，
  将 `self.accounting` 中的累计冲量/能量账本字段（`tool_impulse_on_ice_Ns`、
  `support_impulse_Ns`、`drag_impulse_Ns`、`external_impulse_Ns`、
  `fracture_release_J`、`tool_work_J`、`damping_dissipation_J`、`external_work_J`）
  扁平化输出到 diagnostics 字典（每个 x/y 分量为标量字段）。
  修改 `TensorDEM/src/tensordem/cli.py`，在运行结束时额外写出 `accounting.json`
  （完整 `self.accounting` 字典 + `total_bonds` + `broken_bonds` +
  per-bond `failure_energy_J` 求和）。
- **验收**：现有 TensorDEM 测试全绿；新测试验证 `diagnostics()` 包含
  `tool_impulse_on_ice_x_Ns` 等字段且数值有限；`accounting.json` 存在且可载入。
- **仓库路径**：`TensorDEM/src/tensordem/dem.py`、`TensorDEM/src/tensordem/cli.py`。

### 子任务 B：TensorPhysics s1.py 场景编排器（implementer）

- **目标**：创建 `TensorPhysics/scenarios/wedge_icebreaking/s1.py`。
  单腿 DEM 场景：用 `--hull-profile` 传入预设楔形船体 CSV（由脚本内联生成
  楔形 polyline，写入 work-dir），调用 `python -m tensordem` 运行。
  运行后检查：
  (a) 输出完整性（history.csv / trajectory.pt / accounting.json）；
  (b) 反力冲量账本平衡门（从 accounting.json 读取四类冲量 + trajectory.pt
  末态动量变化，计算闭合误差 ≤ 1e-3）；
  (c) 断键能量账本门（fracture_release_J ≥ 0 且 == Σ failure_energy_J）。
  全部接线到 ValidationReport → LocalArtifactStore → RunManifest →
  ResearchMemoryRecord（与 s0.py 同一管道模式）。
  产出 `s1_report.json`，结构对齐 s0_report.json：
  `schema_version / scenario_id / stage / goal / design / legs / decision /
  scientific_status / next_stage`。
- **验收**：`python scenarios/wedge_icebreaking/s1.py --work-dir outputs/wedge-icebreaking-s1`
  退出码 0 且 `s1_report.json` 中 `decision == s1_impulse_and_energy_ledgers_passed`。
- **仓库路径**：`TensorPhysics/scenarios/wedge_icebreaking/s1.py`。

### 子任务 C：集成验证（verifier）

- **目标**：在子任务 A/B 完成后，运行 `s1.py` 全门，产出 `s1_report.json`，
  验证报告结构与判词正确。
- **验收**：`s1_report.json` 存在、`legs` 数组非空、`decision` 字段为
  `s1_impulse_and_energy_ledgers_passed`、`scientific_status` 为
  `implemented_not_benchmark_verified`。
- **仓库路径**：`TensorPhysics/scenarios/wedge_icebreaking/`。

## 4. 执行顺序

1. 先注册本文件（已完成）。
2. 派发子任务 A（TensorDEM 账本暴露）和子任务 B（TensorPhysics s1.py）并行。
   - 子任务 B 依赖子任务 A 的 `accounting.json` 输出，但可先基于现有 dem.py
     accounting 结构编写检查逻辑，待 A 合入后联调。
3. await_children 等全部终态。
4. 派发子任务 C（verifier 集成验证）。
5. 收尾汇总。
