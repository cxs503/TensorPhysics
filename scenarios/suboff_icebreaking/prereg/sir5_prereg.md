# SIR-5 预注册：G5 断裂传播（渐进裂纹扩展 + 键断裂演化）

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G5 —— 当前一步断 12 面、其余链步 0、终态 0 键断；断裂传播路径未被演练
- **依赖**：SIR-0；可与 SIR-1 / SIR-2 / SIR-3 并行；**是 SIR-7 的硬前置**
- **算力/预算**：**CPU 锚优先**；M（1–2 班次）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

把断裂从「一次性事件」变成「演化过程」：破坏面数随链步**渐进增加**，裂纹在板内扩展，
DEM 侧的键断裂数随之非零且单调累积；断裂释放能/质量在 FEM→DEM 交接处双账本闭合。

## 2. 物理锚

| 锚 | 内容 | 出处 / 已验证锚路径 |
|---|---|---|
| A5.1 | cohesive 双线性牵引-分离律；`failure_opening = 2·G_c/σ_max` | `TensorFEM/src/tensorfem/cohesive.py:33`（`BilinearCohesiveLaw`）、`:39`；库测试 `TensorFEM/tests/test_cohesive.py`（6 测试，含梯形积分=G_c） |
| A5.2 | 解析枚举锚：均匀应力场 + 阈值 → 破坏 face 集合可解析手算 | SI-2 冻结：`benchmarks/icebreaking/si2_fracture_gen_output/validation_report.json` ← `anchors[2].*`（期望 12 / 实际 6） |
| A5.3 | 断裂能量守恒双账本 `ΣG_c·A` vs released_energy ≤1e-12 | SI-2 锚2 口径（同文件 `anchors[1].*`） |
| A5.4 | 键断裂能量释放 `E = 0.5·k_bond·extension²` | wedge S1 断键能量账本范式（`scenarios/wedge_icebreaking/prereg/s1_prereg.md` §2.2） |

## 3. 机器门

| 门 | 判据 |
|---|---|
| M5.1 | 渐进性门：`n_failed_faces(k)` 随步非降 + `n_steps_with_new_fracture ≥ 3`（对照 SI-4「一步 12 面其余 0」） |
| M5.2 | 键断裂演化门：`n_bonds_broken_total > 0` 且随步非降（对照 SI-4 终态 0） |
| M5.3 | 断裂能量双账本 `|ΔE|/ΣE ≤ 1e-12` |
| M5.4 | 质量双账本 `|Δm|/Σm ≤ 1e-12` |
| M5.5 | 解析枚举锚：破坏面计数精确相等，总面积/释放能 vs 闭式 ≤1e-12（A5.2） |

## 4. 规模判据

- 断裂事件跨越 ≥ 3 个链步。
- 破坏面总数 ≥ 30（对照 SI-4 的 12）。
- 键断裂数 ≥ 10（对照 SI-4 的 0）。
- 裂纹路径长度 ≥ 2 倍板厚。

## 5. 内容符合判据（语义门 → `semantic_criteria.json` → `SIR-5`）

- `sir5_physics_fracture_progressive`（field_integrity，`numeric_monotone_nondecreasing`）：
  逐步 `n_failed_faces` 非降且新增步数 ≥3。
- `sir5_physics_bond_breakage`（physics_switch，`numeric_lower_bound`）：`n_bonds_broken_total ≥ 10`。

## 6. 交付物

`sir5_fracture_propagation/validation_report.json` + `gates.json` + 逐步断裂面/键断裂时序工件；
cohesive 与键断裂演化如有改动 → 落 TensorFEM/TensorDEM `src/` + `tests/`。

## 7. 停机条件

渐进性门红（仍是一步断完）→ 停，判「G5 未达标」，不得以「断了就行」通过。

## Addendum 记录

（暂无）
