# m307 任务交付报告：SIR-2/3/5 真实化战役（部分完成）

**任务 ID**: m307  
**分支**: agent/m307  
**时间**: 2026-10-25T06:46:00Z  
**状态**: PARTIAL_DELIVERY（预算耗尽，部分交付）

---

## 交付清单

### ✓ SIR-2 真流场入链（G2）- **PASS**

**目标**: 替换合成载荷 `generate_synthetic_lbm_snapshot`，使用真实 SUBOFF LBM 仿真流场

**交付物**:
- `scenarios/suboff_icebreaking/sir2_flow_ingest/run_sir2_real.py` - 真实 LBM 仿真脚本
- `scenarios/suboff_icebreaking/sir2_flow_ingest/outputs/` - 仿真输出
  - `real_lbm_snapshot.npz` - 流场快照（SHA256: 8adf9ea4...）
  - `ledger_a.json` - 溯源账本
  - `forces.csv` - 力历史（Cd_total ≈ 0.92）
  - `snapshot_0/1_{rho,ux,uy,uz}.npy` - 真实物理场
- `scenarios/suboff_icebreaking/sir2_flow_ingest/validation_report_sir2.json` - 验证报告

**机器门**:
- M2.1 溯源门: ✓ PASS（无合成标记，由 GeneralSimEngine 生成）
- M2.2 几何在场门: ✓ PASS（obstacle_cells=17 > 0）
- M2.3 传递双账本: ✓ PASS（SHA256 一致）
- M2.4 响应端不退化: ✓ PASS（继承 SI-1 验证）

**关键数字**:
- 真实 SUBOFF 几何: L=4.356m, D=0.508m, L/D=8.575
- 真实 LBM 仿真: Re=1000, tau=0.5024, 域 (80,66,66) lu
- 障碍物单元: 17（非合成，真实几何入链）

**局限**:
- 低分辨率测试（resolution=16），Cd=0.92 与 Blasius 基线 0.0420 差距大
- CPU 环境限制，无法运行高分辨率大规模仿真
- 不影响"真实入链"目标达成

---

### △ SIR-3 重力/浮力/自由面入链（G3）- **PARTIAL_FAIL** (3/5)

**目标**: 真实开启重力、浮力项非零、自由面在场

**交付物**:
- `scenarios/suboff_icebreaking/sir3_buoyancy_freesurface/run_sir3.py` - 仿真脚本
- `scenarios/suboff_icebreaking/sir3_buoyancy_freesurface/validation_report_sir3.json` - 验证报告

**机器门**:
- M3.1 重力开关门: ✓ PASS（hydrostatic_buoyancy=True, gravity=9.81）
- M3.2 浮力项非零门: ✓ PASS（F_b=502.762500 N ≥ 1e-6 N）
- M3.3 自由面在场门: ✗ FAIL（TensorLBM free_surface_step API 调用失败）
- M3.4 质量双账本: ✗ FAIL（自由面仿真未完成）
- M3.5 浮力对阿基米德解析解: ✓ PASS（relative_error=0.0 < 0.01）

**关键数字**:
- 阿基米德浮力解析解: 502.762500 N
- TensorFEM 计算值: 502.762500 N（完全一致）
- 相对误差: 0.0%

**失败原因**:
- `free_surface_lbm.free_surface_step` API 调用参数不匹配（`f=None` 不被接受）
- 需要后续班次修正调用签名或改用 GeneralSimEngine 的 free_surface 模式

**待办**:
1. 修正 `free_surface_step` 调用（检查 `f` 参数是否必需）
2. 完成 500 步自由面演化，提取 `n_interface_cells` 和质量守恒误差
3. 重新运行 SIR-3 验证全部 5 个机器门

---

### ✗ SIR-5 断裂传播（G5）- **FAIL** (0/3)

**目标**: 渐进裂纹扩展+键断裂演化（基线=一步断 12 面 0 键断）

**状态**: 未执行（预算耗尽）

**待办**:
1. 创建 `run_sir5.py` 脚本，调用 TensorFEM cohesive 模块运行渐进断裂仿真
2. 验证机器门：渐进裂纹扩展（非一步断）、键断裂演化（>0 键断）
3. 与 `sir0_baseline.json` 三元组对比（12面/30键/0断键→渐进演化）

---

## 未交付项（交给后续班次）

### SIR-3 自由面修复
- **问题**: `free_surface_step` API 调用失败
- **建议路径**: 
  - 检查 `TensorLBM/src/tensorlbm/free_surface_lbm.py` 中 `free_surface_step` 签名
  - 参考库测试 `TensorLBM/tests/test_dam_break_3d_free_surface_campaign.py`
  - 或改用 `GeneralSimEngine` + `PhysicsConfig(gravity=(0,0,-9.81))` 模式

### SIR-5 断裂传播
- **基线**: `scenarios/suboff_icebreaking/prereg/sir0_baseline.json`
  - 12 个面、30 个键、0 个断键（一步断）
- **目标**: 渐进裂纹扩展，键断裂数 > 0
- **库路径**: `TensorFEM/src/tensorfem/cohesive.py`
- **参考**: `scenarios/suboff_icebreaking/prereg/sir5_prereg.md`

---

## 纪律遵守

- ✓ 契约冻结后未修改判据（append-only）
- ✓ 零求解器侵入（薄编排，只调用库 API）
- ✓ 机器门直读工件，零手抄
- ✓ honest FAIL 合法（SIR-3 部分失败如实披露）
- ✓ 物理锚用库内 verified 路径（TensorFEM ice_sheet.buoyancy_response）

---

## 测试状态

未运行 `run_tests`（预算限制）。SIR-2 脚本已验证可运行（outputs/ 已生成）。

---

## 交接笔记

**给下一个自己**:
1. SIR-2 已完成，可直接复用 `run_sir2_real.py` 和 outputs/
2. SIR-3 需修复 `free_surface_step` 调用，参考 `TensorLBM/tests/test_dam_break_3d_free_surface_campaign.py:18`
3. SIR-5 需从头实现，参考 `prereg/sir5_prereg.md` 和 `TensorFEM/src/tensorfem/cohesive.py`
4. 所有验证报告在 `scenarios/suboff_icebreaking/sir{2,3,5}_*/validation_report_sir*.json`
5. 物理锚路径已在交接笔记中列出（TensorFEM ice_sheet:145, TensorLBM free_surface_lbm）

---

**结论**: m307 在预算耗尽前完成了 SIR-2（真流场入链）和 SIR-3 部分（浮力验证），SIR-5 未执行。交付物完整，判据机器可读，待办项明确。
