# SIR-3 预注册：G3 重力 / 浮力 / 自由面入链

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G3 —— 当前全链禁用重力（`gravity=[0,0,0]`）；FVM VOF 与 LBM 自由面单库达标未入链
- **依赖**：SIR-0；可与 SIR-1 / SIR-2 / SIR-5 并行；**是 SIR-4 的硬前置**
- **算力/预算**：**CPU 锚优先**；S–M（<1–2 班次）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

让链路里真实存在**重力、浮力、自由面**三件事：重力开关在运行工件里为开；碎冰/冰板上的净
浮力项非零且对上阿基米德解析值；自由面在场并有质量账本闭环。此后链路才可能演示「上浮/顶升」。

## 2. 物理锚

| 锚 | 内容 | 出处 / 已验证锚路径 |
|---|---|---|
| A3.1 | 阿基米德浮力：半浸单块 `F_b = ρ_w g V_sub`，相对误差 <1% | `TensorFEM/src/tensorfem/ice_sheet.py:145 buoyancy_response`；库测试 `TensorFEM/tests/test_ice_sheet.py:40`；常数 `ρ_w=1025`、`g=9.81`（`ice_sheet.py:26-27`） |
| A3.2 | 自由面质量守恒账本（Korner/VOF） | `TensorLBM/src/tensorlbm/free_surface_lbm.py` + `free_surface_ownership_ledger.py`；库测试 `TensorLBM/tests/test_dam_break_3d_free_surface_campaign.py:18`（101 步质量曲线） |
| A3.3 | 静水压力梯度 `p = ρ_w g h` 线性剖面 | 解析解，与 A3.1 互证 |

## 3. 机器门

| 门 | 判据 |
|---|---|
| M3.1 | 重力开关门：运行工件 `gravity_enabled = true`，`gravity ≠ [0,0,0]`（对照 SI-4 的 `gravity=[0,0,0]`） |
| M3.2 | 浮力项非零门：`buoyancy_net_force_N` 非零且 \|值\| ≥ 1e-6 N |
| M3.3 | 自由面在场门：`n_interface_cells > 0` |
| M3.4 | 质量双账本 `|Δm|/m_0 ≤ 1e-12` |
| M3.5 | 浮力对阿基米德解析解：相对误差 < 1%（A3.1） |

## 4. 规模判据

- 自由面演化 ≥ 500 步。
- 域高向 ≥ 32 网格（自由面分辨率下限）。
- 浸没体积积分点数 ≥ 1e3。

## 5. 内容符合判据（语义门 → `semantic_criteria.json` → `SIR-3`）

- `sir3_physics_gravity_enabled`（physics_switch，`config_key_true`）。
- `sir3_physics_buoyancy_nonzero`（physics_switch，`numeric_nonzero`）。
- `sir3_physics_free_surface_present`（field_integrity，`field_present`）。

## 6. 交付物

`sir3_buoyancy_freesurface/validation_report.json` + `gates.json`；场景薄编排只做开关接线与账本，
浮力/自由面求解全部来自库（TensorFEM `ice_sheet`、TensorLBM `free_surface_*`）。

## 7. 停机条件

重力开关门红 → 停（G3 的定义性判据）。

## Addendum 记录

（暂无）
