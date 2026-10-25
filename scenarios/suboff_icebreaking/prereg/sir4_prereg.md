# SIR-4 预注册：G4 船体上浮运动 + 船-冰接触（顶升过程）

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G4 —— hull 运动学、接触检测、顶升过程全缺（Phase 1/2 无一实装）
- **依赖**：**SIR-2（真流场）+ SIR-3（浮力/自由面）**；本腿是唯一串行咽喉
- **算力/预算**：**CPU 锚优先**；M（1–2 班次）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

让船体成为**会动的物体**：在真流场载荷与浮力共同作用下，hull 有竖直方向运动学（上浮/下沉），
能检测与冰板的接触，并演示顶升（hull 抬升 + 接触力传递）过程——而不是冰板被动受一个固定压力场。

## 2. 物理锚

| 锚 | 内容 | 出处 / 已验证锚路径 |
|---|---|---|
| A4.1 | 罚函数接触闭式解 `penetration=(P−k·g0)/(k+k_p)`、`λ=k_p·penetration` | `TensorFEM/docs/contact-validation.md:21` |
| A4.2 | 船-冰接触等反力（polyline 2D），`ΣF` 等值反向 | `TensorDEM/src/tensordem/hull_contact.py:7`；库测试 `TensorDEM/tests/test_hull_contact.py`（含 `test_known_force_and_action_reaction`） |
| A4.3 | 恢复系数↔阻尼解析映射 `e = exp(−ηπ/√(4mk−η²))` | Hunt & Crossley 1975 / Thornton 1997；`TensorDEM/src/tensordem/dem3d.py` `contact_damping`；判据 `|v_after/v_before − e|/e < 5%` |
| A4.4 | 浮力平衡（hull 静态浮态） | 复用 A3.1（`ice_sheet.py:145`）；hull 排水体积 vs 重量 |

## 3. 机器门

| 门 | 判据 |
|---|---|
| M4.1 | 运动学账本：`|Δz_hull − ∫v_z dt| ≤ 1e-12`（相对） |
| M4.2 | 接触等反力门：`|ΣF_hull_on_ice + ΣF_ice_on_hull|` 相对 ≤ 1e-10 |
| M4.3 | 顶升事件门：`hull_rise_m > 0`，`n_contact_pairs` 随步非降 |
| M4.4 | 接触解析锚：罚函数穿透量与闭式解相对误差 ≤ 1e-10（A4.1） |
| M4.5 | 恢复系数标定门：`|v_after/v_before − e|/e < 5%`（A4.3） |

## 4. 规模判据

- 物理时长 ≥ 1 s（接触/顶升演化量级）。
- 接触检测步数 ≥ 1000。
- 接触对峰值 ≥ 100。

## 5. 内容符合判据（语义门 → `semantic_criteria.json` → `SIR-4`）

- `sir4_physics_hull_motion`（physics_switch）：hull 速度/位移工件非零且带时间序列。
- `sir4_physics_contact_present`（field_integrity）：`n_contact_pairs > 0` 且随步非降
  （`numeric_monotone_nondecreasing`）。

## 6. 交付物

`sir4_hull_motion_contact/validation_report.json` + `gates.json` + 运动学时序工件；
hull 运动学与接触检测如需新增能力 → 落 TensorDEM/TensorFEM `src/` + `tests/`，场景只编排。

## 7. 停机条件

hull 运动学账本红或接触等反力门红 → 停。

## Addendum 记录

（暂无）
