# SIR-2 预注册：G2 真流场入链

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G2 —— 链路当前用 `generate_synthetic_lbm_snapshot(seed=42)` 合成载荷；SUBOFF 几何未进链
- **依赖**：SIR-0；可与 SIR-1 / SIR-3 / SIR-5 并行；**是 SIR-4 的硬前置**
- **算力/预算**：GPU5；M–L（1–3 班次）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

把全链的流体载荷来源从合成函数换成**真实求解器产出的流场工件**，并让 SUBOFF 几何在链内
真实在场：链上的每个节点力都能溯源到某个 `GeneralSimEngine` 运行的输出文件，且该运行的
掩码含真实 SUBOFF 实体单元。

## 2. 物理锚

| 锚 | 内容 | 出处 / 路径 |
|---|---|---|
| A2.1 | SUBOFF Re=1000 阻力基线（Blasius Cf 0.0420 口径，历史 Cd_tot=0.0436） | `TensorLBM/benchmarks/pending/suboff_re1000/README.md` |
| A2.2 | Hetényi 弹性地基板级数解（载荷→响应端复现，<2%） | Hetényi 1946；`benchmarks/icebreaking/si1_one_way.py:39`；库实现 `TensorFEM/src/tensorfem/hetenyi_reference.py`（分支 `agent/m136`） |
| A2.3 | SUBOFF 几何尺寸 `L=4.356 m, D=0.508 m` | `TensorLBM/src/tensorlbm/suboff_cad.py:19`（`L = 14.292 ft = 4.356 m`） |
| A2.4 | 掩码实体单元计数（appendage_solid_cells=126） | `/nfs/wangxi/runs/suboff_sail_20261008/e5_verdict.json` ← `arms_L128.mrt.mask.appendage_solid_cells` |

## 3. 机器门

| 门 | 判据 |
|---|---|
| M2.1 | 溯源门：链上工件与场景源码不得出现 `generate_synthetic` / `seed=` / `placeholder`（PL-SEM `no_synthetic_marker` 同类） |
| M2.2 | 几何在场门：流场工件 `hull_mask.solid_cells > 0`，且几何参数三元组 = `4.356` / `0.508` |
| M2.3 | 传递双账本：流场→结构节点力 `max|A−B| ≤ 1e-12` |
| M2.4 | 响应端不退化：Hetényi 相对误差 < 2%（A2.2 复用） |

## 4. 规模判据

- 域胞数 ≥ 1e7（L≥112 同族下限）。
- 流场贯通步数 ≥ 150·L。
- 至少 1 个完整力采样窗（≥1000 样本）。

## 5. 内容符合判据（语义门 → `semantic_criteria.json` → `SIR-2`）

- `sir2_provenance_load_source`（provenance）：载荷源 = 流场文件（工件中记录源文件路径 + sha256）。
- `sir2_physics_hull_geometry_in_chain`（physics_switch/field_present）：`hull_mask.solid_cells > 0`。

## 6. 交付物

`sir2_flow_ingest/validation_report.json` + `gates.json` + 流场工件清单（含 sha256）；
场景侧改动仅限编排（替换合成载荷调用为读取真实快照），不得写求解器代码。

## 7. 停机条件

溯源门红 → 直接停（这是 G2 的定义性判据，不得降级为「待办」）。

## Addendum 记录

（暂无）
