# SIR-5 审计报告：G5 断裂传播（渐进裂纹扩展 + 键断裂演化）

- **班次**：m309（分支 `agent/m309`）
- **时间**：2026-10-25
- **契约**：`scenarios/suboff_icebreaking/README.md` Phase 3 §3.4/§3.7 + `prereg/sir5_prereg.md`
  （含 `## Addendum 2026-10-25T07:40:00Z` 执行口径冻结）
- **判决**：**PASS** —— 机器门 5/5、语义判据 2/2、规模判据 4/4

## 1. 一句话结论

G5 已从「一步断 12 面、其余 0、终态 0 键断」推进到**渐进演化**：FEM 侧 cohesive 缝在 13 个
采样步上逐次新增破坏（终态 33 个 cohesive 点 / 8 道缝全断），DEM 侧 3-D 键结场在 143 个
时间步上逐次断键（终态 624 / 3560 键断），两侧断裂能量双账本与质量账本残差均为机器零，
解析枚举锚（总面 12 / 坏面 6）逐位精确复现。

## 2. 机器门（`outputs/sir5_fracture_propagation/validation_report.json`）

| 门 | 判据 | 实际值 | 结果 |
|---|---|---|---|
| M5.1 渐进性 | 破坏面累计数随步非降，且新增断裂步数 ≥ 3 | 单调非降；`n_steps_with_new_fracture = 13`，终态 33 点 | PASS |
| M5.2 键断裂演化 | `n_bonds_broken_total > 0` 且随步非降 | `624`（初 3560 键，断键比例 0.1753）；新增断键步 `143` | PASS |
| M5.3 断裂能量双账本 | `|ΔE|/ΣE ≤ 1e-12`（FEM + DEM 双侧） | FEM `0.0`；DEM `0.0` | PASS |
| M5.4 质量双账本 | `|Δm|/Σm ≤ 1e-12` | `0.0` | PASS |
| M5.5 解析枚举锚 | 计数精确相等 + 面积/能量 vs 闭式 ≤ 1e-12 | 总面 12/12、坏面 6/6；`area_rel_error = 0.0`、`energy_rel_error = 0.0` | PASS |

三元组工件：`outputs/sir5_fracture_propagation/gates.json`（5 条机器门 + 2 条语义判据，共 7 条
值-文件-键），由 `tests/test_sir5_contract.py` 当场复算。

## 3. 语义判据（`prereg/semantic_criteria.json` ← `gates.SIR-5`）

| 判据 | 断言 | 实际 | 结果 |
|---|---|---|---|
| `sir5_physics_fracture_progressive` | `n_failed_faces_cumulative` 非降、`min_increasing_steps ≥ 3`、`lower_bound ≥ 30` | 13 个新增步、终态 33 | compliant |
| `sir5_physics_bond_breakage` | `n_bonds_broken_total ≥ 10` | 624 | compliant |

## 4. 规模判据（prereg §4）

| 判据 | 要求 | 实际 | 结果 |
|---|---|---|---|
| 断裂事件跨越链步 | ≥ 3 | 13 | 达标 |
| 破坏面总数 | ≥ 30 | 33 | 达标 |
| 键断裂数 | ≥ 10 | 624 | 达标 |
| 裂纹路径长度 | ≥ 2 倍板厚 | `n_failed_seams=8` × 冰厚 0.01 m = 0.08 m ≥ 0.02 m | 达标 |

## 5. 与 SI-4 基线对照（`prereg/sir0_baseline.json` ← `gaps.g5_fracture_propagation`）

| 指标 | SI-4 基线 | SIR-5 | 
|---|---|---|
| 破坏面出现方式 | 首链步一次 12 面，其余步 0 | 13 个采样步逐次新增 |
| 链步新增断裂次数 | 1 | 13 |
| 终态键断裂数 | 0 | 624 |

基线三元组仍由 `test_baseline_comparison_triplets_still_recomputable` 当场复算（未被改写）。

## 6. 物理锚使用（全部复用，零自证闭环）

| 锚 | 出处 | 用法 |
|---|---|---|
| A5.1 cohesive 双线性律 | `TensorFEM/src/tensorfem/cohesive.py:33` `BilinearCohesiveLaw`（库测试 `tests/test_cohesive.py` 已验证） | FEM 侧缝面牵引-分离；场景侧账本 B 用它独立复算耗散 |
| A5.2 解析枚举锚 | SI-2 冻结口径 `benchmarks/icebreaking/si2_fracture_gen_output/validation_report.json` ← `anchors[2].*`（期望 12 / 实际 6） | 本腿 `analytical_anchor.json` 独立重跑同口径，逐位一致 |
| A5.3 断裂能量双账本 | SI-2 锚 2 口径 | FEM/DEM 双侧各做一次 |
| A5.4 键断裂能量 `0.5·k_bond·extension²` | wedge S1 断键账本范式 | DEM 侧账本 A 逐键复算 |
| 几何锚 | `TensorLBM/src/tensorlbm/suboff_cad.py:suboff_radius_profile` → `tensorfem.suboff_geometry.closed_revolved_mesh` | SUBOFF 外包络入链，`audit.passed = true`、欧拉示性数 2 |

## 7. 交付物

- `scenarios/suboff_icebreaking/sir5_run_validation.py`（薄编排，零求解器代码，约 420 行）
- `outputs/sir5_fracture_propagation/{validation_report,gates,fracture_history,run_summary,analytical_anchor}.json`
- `tests/test_sir5_contract.py`（9 测试：工件齐全 / prereg md5 / 三元组复算 / 门限文本解析重判 /
  单调性独立复验 / 解析锚 / 语义对齐 / 基线三元组复算）
- `scenarios/suboff_icebreaking/prereg/sir5_prereg.md` 的 `## Addendum 2026-10-25T07:40:00Z`
  （append-only 追加执行口径，未改写 §1-§7）

## 8. 局限（如实披露，不假绿）

1. **FEM 侧裂纹只能沿库内预设 cohesive 缝格扩展**（`IceImpactConfig(ice_nx=12, ice_ny=4)`），
   不在单元内部任意取向开裂。本腿的「渐进性」只主张缝格尺度传播。任意路径开裂需另开
   TensorFEM 裂纹追踪定向修订单（库侧改动，不在薄编排腿范围）。
2. **FEM 与 DEM 是两套独立损伤变量**（缝格 cohesive vs 键结颗粒），两侧能量账本各自闭合，
   **不是**跨库统一账本。跨库统一账本属后续定向修订单。
3. **两阶段物理时长不同**（FEM 冲击 6 ms 量级、DEM 压板 6.7e-2 s），本腿只判渐进性/键断裂/
   账本/解析锚；跨尺度时长与规模统一由 SIR-6 承担。
4. 预算限制：本班次工具调用进入保留区，FEM 侧只跑到 `ice_nx=12, ice_ny=4`、DEM 侧 2000 步；
   更高分辨率复核列为待办。

## 9. [待办] 交后续班次

1. 把 FEM 侧 `ice_nx/ice_ny` 提到 24×8、`duration` 提到 0.009 s，复核裂纹路径单调性与缝数收敛
   （脚本改 `stage1_fem()` 的两处入参即可）。
2. DEM 与 SI-4 的 12 颗粒/30 键基线是两套模型，SIR-6 放大时统一到同一键结场。
3. 若 owner 要求「单元内部任意路径开裂」，需向 TensorFEM 提定向修订单（裂纹追踪/拓扑演化），
   本腿不复用 `suboff_ice_collision` 的固定缝格。
4. SIR-7 需要本腿 `fracture_history.json` 的 P-h（载荷-挠度）配对：当前只登记了破坏面时序，
   未导出接触力-位移曲线，需在 SIR-7 前补导 `history[*].contact_force_N` 与
   `maximum_ice_vertical_displacement_m` 的配对序列。

## 10. 复跑

```bash
cd <repo>
PYTHONPATH=src:. /nfs/wangxi/venvs/tensor_eco/bin/python \
    scenarios/suboff_icebreaking/sir5_run_validation.py    # 约 90 s，CPU
PYTHONPATH=src:. /nfs/wangxi/venvs/tensor_eco/bin/python -m pytest -q tests/test_sir5_contract.py
```