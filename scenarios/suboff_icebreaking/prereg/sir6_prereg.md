# SIR-6 预注册：G6 规模化

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G6 —— 1 m 板 / 4×4 网格 / 毫秒量级耦合步 → 物理尺度
- **依赖**：**SIR-1..SIR-5 全部腿可运行**（本腿只做尺度放大，不改物理）
- **算力/预算**：**GPU5 长算 + launch-harvest 分离**；XL（多班次、多波）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

在物理量级上跑通同一条链：物理时长 ≥1 s、板单元 ≥1e4、颗粒 ≥1e3、断键 >0，
且守恒门不退化、规模放大后数值残差如实登记（非零属正常）。这是 id=51 局限 L-1/L-2 的正解。

## 2. 物理锚

| 锚 | 内容 | 出处 / 复用处 |
|---|---|---|
| A6.1 | 网格收敛单调性（≥2 档，|err| 沿阶梯非增） | `/nfs/wangxi/runs/suboff_sail_20261008/prereg.md` G2 门 |
| A6.2 | 时间步稳定性：DEM `dt ≤ dt_crit`（Gershgorin 界）；LBM `tau−0.5 > 0` | `TensorFEM/src/tensorfem/explicit_dynamics.py` `critical_time_step`；`sir0_baseline.json` `derived_frozen_table` |
| A6.3 | 无量纲不变量（Fr / Re）放大前后相对差 ≤5% | 量纲分析；记录于 `run_summary.json` |
| A6.4 | 响应端不失真：Hetényi <2%（SIR-1/2 同口径） | `benchmarks/icebreaking/si1_one_way.py:39` |

## 3. 机器门

| 门 | 判据 |
|---|---|
| M6.1 | 规模门（四项）：`total_physical_time_s ≥ 1.0`、`n_elements ≥ 1e4`、`n_particles ≥ 1e3`、`n_bonds_broken > 0` |
| M6.2 | 守恒门不退化：质量/能量双账本残差 ≤ 门限（数值残差允许非零，门限为标尺） |
| M6.3 | 稳定性门：无 NaN/发散；`dt ≤ dt_crit`；`tau − 0.5 > 0` |
| M6.4 | 资源披露门：wall time / 峰值显存 / `CUDA_VISIBLE_DEVICES=5` 三项齐全 |

## 4. 规模判据（相对 id=51 smoke 规模的量化下限）

| 维度 | id=51 现状 | SIR-6 下限 | 倍数 |
|---|---|---|---|
| 物理时长 | 毫秒量级耦合步（5 链步） | **≥ 1 s** | ≳ 1e2 量级 |
| 板单元数 | 16（4×4） | **≥ 1e4** | ≳ 6e2 |
| 颗粒数 | 12 | **≥ 1e3** | ≳ 8e1 |
| 键断裂数 | 0 | **> 0** | ∞（从零到有） |

（现状列的 id=51 数字来源：`sir0_baseline.json` ← `gaps.g6_scale.evidence` 与
`gaps.g5_fracture_propagation.evidence`。）

## 5. 内容符合判据（语义门 → `semantic_criteria.json` → `SIR-6`）

- `sir6_scale_physical_duration`（scale_realism，`numeric_lower_bound`，≥1.0 s）。
- `sir6_scale_grid_resolution`（scale_realism，≥1e4）。
- `sir6_scale_particle_count`（scale_realism，≥1e3）。

## 6. 交付物

`sir6_scale/validation_report.json` + `gates.json` + `run_summary.json`（规模量与资源披露）
+ `DONE.json` 完成清单（launch-harvest 分离用）。

## 7. 停机条件与预算

- 规模门任一项不足 → 该波判未达标，登记差距后按「放大策略」下一波再试（不得降下限）。
- 长算一律 `nohup`/`setsid` 分离 + `GPU5_BUSY` 标记 + `DONE.json`；发射班与收割班分离。

## Addendum 记录

（暂无）
