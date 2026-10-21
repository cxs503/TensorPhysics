# SI-0A: TensorLBM 能力审计报告

**审计目标**: TensorLBM 在 SUBOFF 全附体近水面外流场景中的能力矩阵、差距分析、锚点提议、工作量分级与接口需求，服务于 SI-4 全链破冰演示。

**审计范围**: `/nfs/wangxi/tensor_eco/TensorLBM`（commit `9fa6050e`，分支 `main`）

**审计日期**: 2026-10-21（Phase 0 纯读审计，未修改任何 src 代码）

**审计方法**: 逐项扫描源文件与测试文件，所有"已有"论断以 `file:line` 证据支撑，零手抄。

---

## 一、能力矩阵（Capability Matrix）

### 1.1 SUBOFF 全附体几何/阻力基线

| 目标 | 模块/文件 | 成熟度 | 直接相关度 | 证据 file:line |
|------|-----------|--------|------------|----------------|
| 裸体 BARE_HULL CAD | `src/tensorlbm/suboff_cad.py:115-122` SuboffHullType.BARE_HULL | 有测试 | ★★★★★ | `suboff_cad.py:111-122` 定义三类 hull type；`suboff_cad.py:345-417` suboff_hull_mask()；`test_suboff_cad.py:102-138` 裸体 mask 测试 |
| 带帆 WITH_SAIL (AFF-3) | `src/tensorlbm/suboff_cad.py:121-122` WITH_SAIL | 有测试 | ★★★★★ | `suboff_cad.py:420-493` suboff_sail_contains_points()；`suboff_cad.py:556-639` _add_sail_mask()；`test_suboff_cad.py:186-191` sail>bare 验证 |
| 全附体 FULL (AFF-8) | `src/tensorlbm/suboff_cad.py:124-125` FULL | 有测试 | ★★★★★ | `suboff_cad.py:496-537` suboff_fins_contain_points()；`suboff_cad.py:642-713` _add_fin_masks()；`test_suboff_appendage_links.py:22-48` 全附体链接验证 |
| 阻力基线 (Ladd MEM) | `src/tensorlbm/drag_momentum.py:39-92` drag_momentum_exchange_vec | 有测试 | ★★★★★ | `drag_momentum.py:39-92` Ladd 1994 实现；`test_drag_momentum_verification.py:30-80` Couette 验证 Cf err=4.55% |
| 阻力基线 (压力+摩擦) | `src/tensorlbm/drag_pressure.py:755-868` drag_pressure_integration + `src/tensorlbm/drag_pressure.py:869-975` drag_friction_integration | 有测试 | ★★★★★ | `drag_pressure.py:131-212` from_suboff() 解析法向；`suboff_3d_drag_worker.py:39-43` 引用 |
| 阻力基线 (BFL MEM) | `src/tensorlbm/momentum_exchange.py:284-360` momentum_exchange_bfl | 有测试 | ★★★★★ | `momentum_exchange.py:284-360` BFL 加权 MEM；`test_bfl_suboff.py:1-216` BFL vs Staircase 对比 |
| SUBOFF worker (裸体) | `suboff_3d_drag_worker.py:1-100` | 有测试 | ★★★★★ | `suboff_3d_drag_worker.py:1-78` Re=1000/1e4/1e5/2e6 基准；`suboff_3d_drag_worker.py:36-43` SurfaceMesh.from_suboff |
| SUBOFF worker (附体) | `suboff_appendage_worker.py:1-80` | 无测试（worker 脚本） | ★★★★★ | `suboff_appendage_worker.py:14-16` WITH_SAIL/FULL/BARE_HULL 4 cases；`suboff_appendage_worker.py:35-43` Cd_ref≈0.046/0.055 |
| SUBOFF BFL 对比 | `suboff_bfl_cmp_worker.py` | 有测试 | ★★★★ | `test_bfl_suboff.py:32-80` BFL 与解析轮廓匹配 |
| 阻力收敛 | `src/tensorlbm/suboff_resistance.py:38-80` SuboffResistanceBenchmarkConfig | 有测试 | ★★★★★ | `test_suboff_resistance.py` 存在；`suboff_resistance.py:38-78` 完整配置 |
| 力平稳性评估 | `src/tensorlbm/force_convergence.py` | 有测试 | ★★★★ | `test_force_convergence.py:1-40` 平稳性 block 分析 |

### 1.2 自由面能力（多相/自由面）

| 目标 | 模块/文件 | 成熟度 | 直接相关度 | 证据 file:line |
|------|-----------|--------|------------|----------------|
| Korner 自由面 (D3Q19) | `src/tensorlbm/free_surface_lbm.py:1-1265` | 有测试 | ★★★★★ | `free_surface_lbm.py:526-1248` free_surface_step()；质量追踪/拓扑变异/质量重分配 |
| Korner 自由面 (D3Q27) | `src/tensorlbm/free_surface_lbm_27.py:1-667` | 有测试 | ★★★★★ | `free_surface_lbm_27.py:1-60` 完整 D3Q27 自由面 |
| VOF 自由面 (D3Q19) | `src/tensorlbm/free_surface_common.py:1-899` | 有测试 | ★★★★ | `free_surface_common.py:44-80` VOF 对流 + 重力 + CSF 表面张力 |
| 2D 自由面 | `src/tensorlbm/free_surface_lbm_2d.py:1-138` | 有测试 | ★★ | 基础 D2Q9 自由面 |
| 溃坝 3D | `src/tensorlbm/dam_break_3d.py` | 有测试 | ★★★★★ | `test_dam_break_3d_free_surface_campaign.py:1-70` 101 步质量漂移质量曲线 |
| 船体自由面 (Wigley/Series60/KCS) | `src/tensorlbm/hull_free_surface.py:1-100` | 仅原型 | ★★★ | `hull_free_surface.py:4-15` Color-Gradient + Wigley hull |
| 船体自由面 V2 | `src/tensorlbm/hull_free_surface_v2.py:1-100` | 仅原型 | ★★★★ | `hull_free_surface_v2.py:10-22` 双体→自由面→兴波阻力分解 |
| 多相 Shan-Chen (D3Q19) | `src/tensorlbm/multiphase3d.py:1-1145` | 有测试 | ★★★★ | `multiphase3d.py:1-50` SC 两分量；`multiphase_benchmarks.py:1-1912` 基准 |
| 多相 Shan-Chen (D3Q27) | `src/tensorlbm/multiphase3d_d3q27.py:1-734` | 有测试 | ★★★★ | `multiphase3d_d3q27.py:1-60` SC + D3Q27 |
| 多相水入水 | `src/tensorlbm/multiphase_water_entry.py:1-553` | 有测试 | ★★★★ | `multiphase_water_entry.py:25-35` 球体/圆柱入水，MEM 力测量 |
| 波浪高度测量 | `src/tensorlbm/free_surface_common.py:793-810` wave_height_at_wall_3d | 有测试 | ★★★ | `free_surface_common.py:793-809` 壁面波高 |
| 兴波阻力 | `src/tensorlbm/hull_free_surface_v2.py:212-240` _wave_resistance | 仅原型 | ★★★★ | `hull_free_surface_v2.py:212-240` 从波高估算兴波阻力 |
| 自由面质量审计 | `src/tensorlbm/free_surface_ownership_ledger.py:1-258` | 有测试 | ★★★ | `test_free_surface_ownership_ledger.py` |
| 自由面转换审计 | `src/tensorlbm/free_surface_i_to_g_exact_ledger.py:1-332` | 有测试 | ★★★ | `test_free_surface_i_to_g_exact_ledger.py` |
| 自由面拓扑守卫 | `src/tensorlbm/free_surface_topology_mutation_replay_contract.py:1-318` | 有测试 | ★★★ | `test_free_surface_topology_mutation_replay_contract.py` |

### 1.3 冰面下流道（顶盖滑移/无滑移边界）

| 目标 | 模块/文件 | 成熟度 | 直接相关度 | 证据 file:line |
|------|-----------|--------|------------|----------------|
| 无滑移 BB | `src/tensorlbm/boundaries3d.py:40-133` bounce_back_cells_3d | 有测试 | ★★★★★ | `boundaries3d.py:40-133` half-way BB |
| 自由滑移 | `src/tensorlbm/boundaries3d.py:135-202` free_slip_cells_3d / free_slip_x_walls_3d | 有测试 | ★★★★★ | `boundaries3d.py:135-202` 镜面反射自由滑移 |
| Zou-He 移动顶盖 | `src/tensorlbm/boundaries3d.py:315-420` zou_he_moving_lid_3d | 有测试 | ★★★★ | `boundaries3d.py:315-377` prescribed 速度顶盖 |
| 远场 BC | `src/tensorlbm/boundaries3d.py:426-500` far_field_bc_3d | 有测试 | ★★★★★ | `boundaries3d.py:426-500` 自由面平衡 + zero-gradient 出口 |
| 壁面模型（滑移速度） | `src/tensorlbm/wall_model.py:1-80` compute_wall_slip_velocity | 有测试 | ★★★★ | `wall_model.py:1-60` log-law 壁面滑移 |
| BFL 移动壁修正 | `src/tensorlbm/bfl_common.py:193-246` bfl_moving_wall_correction | 有测试 | ★★★★ | `bfl_common.py:193-246` 运动壁动量修正 |
| 非平衡外推边界 | `src/tensorlbm/external_open_boundary.py:56-82` non_equilibrium_far_field_bc_3d | 有测试 | ★★★ | `external_open_boundary.py:56-82` 非平衡外推 |
| **刚性顶盖 (rigid-lid)** | — | **缺失** | ★★★★★ | **无对应模块或测试** |
| **冰面滑移边界 (ice-slip BC)** | — | **缺失** | ★★★★★ | **无对应模块或测试** |
| **顶盖+自由面耦合** | — | **缺失** | ★★★★★ | **自由面模块未包含顶盖处理** |

### 1.4 力/冲量账本（精度门）

| 目标 | 模块/文件 | 成熟度 | 直接相关度 | 证据 file:line |
|------|-----------|--------|------------|----------------|
| Ladd MEM (标准) | `src/tensorlbm/momentum_exchange.py:74-135` momentum_exchange_standard | 有测试 | ★★★★★ | `momentum_exchange.py:74-135` Ladd 1994 标准形式 |
| Ladd MEM (伽利略不变) | `src/tensorlbm/momentum_exchange.py:137-203` momentum_exchange_galilean | 有测试 | ★★★★★ | `momentum_exchange.py:137-203` Lorenz 2014 修正 |
| Ladd MEM (背景扣除) | `src/tensorlbm/momentum_exchange.py:205-282` momentum_exchange_background_subtracted | 有测试 | ★★★★★ | `momentum_exchange.py:205-282` 自由-stream 背景扣除 |
| Ladd MEM (BFL 加权) | `src/tensorlbm/momentum_exchange.py:284-360` momentum_exchange_bfl | 有测试 | ★★★★★ | `momentum_exchange.py:284-360` BFL q-加权 |
| 应力积分法 | `src/tensorlbm/momentum_exchange.py:362-396` momentum_exchange_stress | 有测试 | ★★★★ | `momentum_exchange.py:362-396` 非平衡应力积分 |
| 压力+摩擦积分 | `src/tensorlbm/momentum_exchange.py:398-450` momentum_exchange_pressure_friction | 有测试 | ★★★★ | `momentum_exchange.py:398-450` 表面力积分 |
| 力方法对比 | `src/tensorlbm/force_methods.py:1-100` compare_force_methods | 有测试 | ★★★★★ | `force_methods.py:1-80` 5 种方法统一接口 |
| D3Q27 移动壁 ME | `src/tensorlbm/d3q27.py` moving_wall_linkwise_me_force_torque | 有测试 | ★★★★★ | `test_d3q27_moving_wall_momentum_exchange.py:1-60` |
| Couette 验证 | `tests/test_drag_momentum_verification.py:30-80` | 有测试 | ★★★★★ | Couette Cf_exact=0.6349, MEM err=4.55% |
| 力平稳性 | `src/tensorlbm/force_convergence.py` + `test_force_convergence.py` | 有测试 | ★★★★ | `force_convergence.py` assess_force_stationarity |

### 1.5 与 FEM/DEM 耦合接口

| 目标 | 模块/文件 | 成熟度 | 直接相关度 | 证据 file:line |
|------|-----------|--------|------------|----------------|
| IBM 直接强迫 | `src/tensorlbm/ibm_common.py:1-80` ibm_direct_forcing_3d_common | 有测试 | ★★★★★ | `ibm_common.py:1-60` solver-agnostic IBM |
| FSI 耦合 | `src/tensorlbm/fsi_common.py:1-80` fsi_step / fsi_step_drag | 有测试 | ★★★★★ | `fsi_common.py:1-80` IBM + 6-DOF + 弹簧质量 |
| 6-DOF 刚体 | `src/tensorlbm/sixdof_common.py:1-60` rigid_body_step | 有测试 | ★★★★★ | `sixdof_common.py:1-50` symplectic Euler 积分器 |
| 6-DOF Cummins | `src/tensorlbm/rigid_body_6dof.py:1-80` Cummins 时域方程 | 仅原型 | ★★★★ | `rigid_body_6dof.py:1-50` 附加质量+延迟函数 |
| 移动边界 IBM | `src/tensorlbm/moving_boundary.py:1-80` MovingBoundary | 有测试 | ★★★★ | `moving_boundary.py:1-60` 节距-沉浮运动 |
| FSI 载荷提取 | `src/tensorlbm/fsi.py:88-172` extract_fsi_loads | 仅原型 | ★★★★ | `fsi.py:88-172` 压力/剪切力提取 + 结构响应 |
| 弹簧质量阻尼器 | `src/tensorlbm/fsi_common.py:55-75` SpringMassDamper | 有测试 | ★★★ | `fsi_common.py` spring_mass_step |
| 表面标记生成 | `src/tensorlbm/ibm_common.py` generate_cylinder_markers/generate_sphere_markers | 有测试 | ★★★★ | `ibm_common.py` Lagrangian marker 生成 |
| STL 表面网格 | `src/tensorlbm/drag_pressure.py:319-339` SurfaceMesh.from_stl | 有测试 | ★★★★ | STL 导入→法向→near-wall mask |
| BFL STL q 场 | `src/tensorlbm/bfl_common.py:522-668` compute_q_stl_common | 有测试 | ★★★★ | STL 三角面 BFL q 计算 |
| **载荷双向交换契约** | — | **缺失** | ★★★★★ | **无标准化 LBM↔FEM/DEM 交换协议** |
| **运动边界时钟同步** | — | **缺失** | ★★★★★ | **无耦合节拍器/同步机制** |

---

## 二、差距清单（Gap List，按对 SI-4 全链演示的阻断性排序）

### G1 [阻断性：CRITICAL] 自由面 + 顶盖耦合（冰面边界）

- **描述**: 自由面模块（`free_surface_lbm.py` Korner 模型、`free_surface_common.py` VOF）均未实现刚性顶盖（rigid-lid）或滑移顶盖边界。冰面下流道需要顶部为自由滑移或指定滑移速度的 BC，而自由面的拓扑变异逻辑（GAS↔INTERFACE↔LIQUID）与顶盖 BC 不兼容。
- **影响**: 无法模拟冰面下 SUBOFF 近水面外流（SI-A 核心场景）。
- **文件证据**:
  - `free_surface_lbm.py:526-1248` free_surface_step() 无顶盖处理
  - `free_surface_common.py:1-899` VOF 无顶盖 BC
  - `boundaries3d.py:135-202` free_slip 存在但不与自由面组合
  - 无 `rigid_lid`、`ice_slip` 关键词出现在任何源文件

### G2 [阻断性：CRITICAL] LBM↔FEM/DEM 载荷与运动边界交换协议

- **描述**: 现有 FSI 模块（`fsi_common.py`、`fsi.py`）仅支持 IBM 驱动的单向/简化双向耦合，无标准化的 LBM→FEM（力/力矩时间序列）和 FEM→LBM（位移/速度边界更新）数据契约。DEM 碎冰运动边界交换同样缺失。
- **影响**: SI-1/SI-3 双向耦合无法进行，全链演示阻断。
- **文件证据**:
  - `fsi_common.py:1-80` 仅有 IBM + 6-DOF 内耦合，无跨库接口
  - `fsi.py:88-172` extract_fsi_loads 为单机提取，非 IPC/RPC
  - 无 `exchange_protocol`、`coupling_contract` 类定义

### G3 [阻断性：HIGH] 全附体 (FULL/AFF-8) 自由面工况验证

- **描述**: `suboff_appendage_worker.py:14-16` 定义了 WITH_SAIL 和 FULL 工况，但仅在无自由面（double-body）外流下验证（`suboff_appendage_worker.py:35-43` Cd_ref≈0.046/0.055）。带帆/尾舵 + 自由面工况无基准数据。
- **影响**: SI-A 近水面全附体工况无精度基线。
- **文件证据**:
  - `suboff_appendage_worker.py:14-16` case 1-2 sail/full
  - `hull_free_surface_v2.py:1-100` 船体自由面仅 Wigley/Series60/KCS，无 SUBOFF sail/full 集成

### G4 [阻断性：HIGH] 冰面下流道专用 BC（滑移/部分滑移）

- **描述**: 现有 `free_slip_cells_3d`（镜面反射）是完全滑移，`bounce_back_cells_3d` 是完全无滑移。冰面物理可能需要 Navier 滑移（部分滑移，滑移长度参数化）或剪切应力指定 BC。
- **影响**: 冰面物理保真度不足。
- **文件证据**:
  - `boundaries3d.py:135-202` free_slip = 完全镜面反射
  - 无 `navier_slip`、`partial_slip`、`slip_length` 关键词
  - `wall_model.py:1-80` log-law 仅适用于壁面模型，非顶盖

### G5 [阻断性：MEDIUM] 力/冲量相位精度在运动壁场景的门控

- **描述**: Ladd MEM 在运动壁场景有 Galilean 不变性修正（`momentum_exchange.py:137-203`），但自由面 + 运动冰板的组合场景下，力的相位精度尚未建立验证门。`test_drag_momentum_verification.py` 仅验证了 Couette/Poiseuille 静止壁。
- **影响**: SI-4 精度门无法裁定。
- **文件证据**:
  - `momentum_exchange.py:137-203` galilean 修正存在
  - `test_d3q27_moving_wall_momentum_exchange.py:1-60` 仅验证零速退化和旋转壁
  - 无自由面 + 运动壁组合的力相位测试

### G6 [阻断性：MEDIUM] 自由面拓扑 + 运动固体障碍物的组合稳定性

- **描述**: 自由面模块的拓扑守卫（`free_surface_topology_mutation_replay_contract.py`）和运动固体（IBM/moving_boundary）独立验证，但两者组合——运动物体穿过自由面界面——的数值稳定性未经验证。
- **影响**: 碎冰（运动固体）在自由面流场中的演化可能不稳定。
- **文件证据**:
  - `free_surface_topology_mutation_replay_contract.py:1-318` 拓扑守卫
  - `moving_boundary.py:1-80` 运动边界
  - 无组合测试文件

### G7 [阻断性：LOW] SUBOFF full_hull_runner 完善

- **描述**: `suboff_full_hull_runner.py` 文件存在但无对应测试，且未在 worker 脚本中调用。
- **影响**: 代码质量/可维护性，不阻断 SI-4。
- **文件证据**: `suboff_full_hull_runner.py` 无 test 对应

---

## 三、解析锚提议（机器裁定判据）

### A1: SUBOFF 裸体阻力系数基线

- **判据形式**: `{"key": "Cd_bare_hull_Re1000", "file": "suboff_3d_drag_worker.py", "value_range": [0.035, 0.055], "reference": "ITTC-1957 Cf≈0.075/(log10(Re)-2)², Re=1000 → Cf=0.075"}`
- **裁定方式**: 运行 `suboff_3d_drag_worker.py re1000`，取 Cd_total，与 Blasius/ITTC-1957 解析值对比，误差 < 15%（粗网格容忍）
- **预注册工件**: `outputs/suboff_icebreaking/anchor_A1_Cd_bare.json` → 键 `Cd_total`

### A2: SUBOFF 附体阻力增量

- **判据形式**: `{"key": "Cd_sail_increment", "file": "suboff_appendage_worker.py", "value": "WITH_SAIL Cd > BARE_HULL Cd", "reference": "AFF-3 Cd≈0.046 vs AFF-1 Cd≈0.036, sail adds ~10%"}`
- **裁定方式**: case 1 (WITH_SAIL) Cd_total > case 3 (BARE_HULL L=160) Cf，差值 > 5%
- **预注册工件**: `outputs/suboff_icebreaking/anchor_A2_Cd_sail.json` → 键 `Cd_sail`, `Cd_bare`

### A3: 自由面质量守恒

- **判据形式**: `{"key": "mass_drift_relative", "file": "free_surface_lbm.py", "value_threshold": 1.0e-3, "reference": "Korner et al. (2005) mass conservation"}`
- **裁定方式**: 运行 1000 步自由面 dam_break_3d，检查 `free_surface_quality_curve[-1]["relative_cumulative_drift"] < 1e-3`
- **预注册工件**: `outputs/suboff_icebreaking/anchor_A3_mass.json` → 键 `relative_cumulative_drift`

### A4: Ladd MEM 在 Couette 流的精度

- **判据形式**: `{"key": "Cf_couette_MEM", "file": "test_drag_momentum_verification.py", "value": 0.6349, "tolerance": "5%", "reference": "Cf_exact = 2nu/((H-0.5)*U) = 0.6349"}`
- **裁定方式**: `test_drag_momentum_verification.py` 的 Couette 测试通过（err < 5%）
- **预注册工件**: `tests/test_drag_momentum_verification.py` → `run_couette()` 返回值 `Cf_MEM`

### A5: 自由滑移壁面切应力为零

- **判据形式**: `{"key": "shear_stress_free_slip", "file": "boundaries3d.py", "value_threshold": 1.0e-10, "reference": "Specular reflection → zero tangential momentum transfer"}`
- **裁定方式**: 设置 free_slip 顶盖 + 通道流，测量顶盖壁面切应力 < 1e-10
- **预注册工件**: `outputs/suboff_icebreaking/anchor_A5_slip.json` → 键 `shear_stress`

### A6: BFL q 场与 CAD mask 一致性

- **判据形式**: `{"key": "q_mask_consistency", "file": "test_bfl_suboff.py", "value": "torch.equal(reused_mask, built_mask)", "reference": "BFL q-field must not alter link topology"}`
- **裁定方式**: `test_bfl_suboff.py:61-80` 测试通过
- **预注册工件**: `tests/test_bfl_suboff.py` → `test_bfl_q_reuses_solver_cad_mask_without_changing_links`

---

## 四、工作量分级

| 差距 | 分级 | 理由 |
|------|------|------|
| G1 自由面+顶盖耦合 | **L** (多班次) | 需重构自由面拓扑逻辑以支持顶盖边界，涉及 `free_surface_lbm.py` 核心循环修改 + 新测试。需 GPU 排队验证 |
| G2 LBM↔FEM/DEM 交换协议 | **M** (1-2 班次) | 定义数据契约（JSON schema + 时钟节拍），实现内存/文件级交换适配器。不涉及 LBM 核心修改 |
| G3 全附体自由面验证 | **M** (1-2 班次) | 集成 `suboff_appendage_worker.py` + `hull_free_surface_v2.py`，配置新工况，运行基准 |
| G4 冰面滑移 BC | **S** (<1 班次) | 基于现有 `free_slip_cells_3d` 增加滑移长度参数化（Navier BC），200 行以内 |
| G5 运动壁力相位门控 | **M** (1-2 班次) | 编写新测试：运动壁 + Ladd MEM 相位对比解析解（振荡圆柱） |
| G6 自由面+运动固体组合 | **L** (多班次) | 需集成 IBM 移动标记与自由面拓扑守卫，处理标记穿越界面的数值问题 |
| G7 full_hull_runner 完善 | **S** (<1 班次) | 补充测试 + 调用路径 |

---

## 五、接口需求（TensorLBM ↔ 耦合对端）

### 5.1 TensorLBM → FEM (冰板) / DEM (碎冰) 交出

| 物理量 | 数据结构 | 时钟节拍 | 来源模块 |
|--------|----------|----------|----------|
| 流体作用力 (Fx, Fy, Fz) | float[3] per body, 或 float[N×3] 全量 | 每 dt_coupling（建议 10-100 LBM steps 一次） | `momentum_exchange.py` / `drag_momentum.py` |
| 流体作用力矩 (Mx, My, Mz) | float[3] per body | 同上 | `d3q27.py` moving_wall_linkwise_me_force_torque |
| 表面压力分布 (可选) | float[near_wall_cells] | 每 dt_coupling | `drag_pressure.py` drag_pressure_integration |
| 局部流速 (IBM 标记处) | float[N×3] | 每 LBM step（如需高频） | `ibm_common.py` macroscopic_velocity_3d |
| 自由面高度场 | float[nx] 或 float[nx×ny] | 每 dt_coupling | `free_surface_common.py` wave_height_at_wall_3d |

### 5.2 FEM / DEM → TensorLBM 拿入

| 物理量 | 数据结构 | 时钟节拍 | 目标模块 |
|--------|----------|----------|----------|
| 固体位移/位置 | float[N×3] 标记坐标 或 float[6] 刚体位姿 | 每 dt_coupling | `ibm_common.py` update_moving_markers / `sixdof_common.py` |
| 固体速度 | float[N×3] 标记速度 或 float[6] 刚体速度 | 每 dt_coupling | `bfl_common.py` bfl_moving_wall_correction |
| 固体边界类型 | "no_slip" / "free_slip" / "partial_slip" per face | 初始化时 + 变更时 | `boundaries3d.py` / 新 Navier slip BC |
| 冰板变形网格 (FEM) | STL/三角面流式更新 | 每 dt_coupling | `drag_pressure.py` SurfaceMesh.from_stl + `bfl_common.py` compute_q_stl_common |

### 5.3 时钟节拍建议

| 节拍 | 建议值 | 说明 |
|------|--------|------|
| LBM dt_lbm | 由 CFL = u·dt/dx < 0.5 决定 | 内部时间步 |
| 耦合间隔 dt_coupling | 10-100 × dt_lbm | 力/运动数据交换频率。冰板低频（100×），碎冰高频（10×） |
| 同步方式 | 显式（无子迭代） | 第一版推荐；SI-3 阶段考虑 Aitken 松弛子迭代 |
| 最大延迟 | ≤ 1 dt_coupling | 对端响应延迟不得超过 1 个耦合周期 |

---

## 附录：关键文件索引

| 模块 | 主文件 | 测试文件 | 行号范围 |
|------|--------|----------|----------|
| SUBOFF CAD | `src/tensorlbm/suboff_cad.py` | `tests/test_suboff_cad.py` | 1-1701 / 1-280 |
| SUBOFF 阻力 | `src/tensorlbm/suboff_resistance.py` | `tests/test_suboff_resistance.py` | 1-400+ |
| Ladd MEM | `src/tensorlbm/momentum_exchange.py` | `tests/test_drag_momentum_verification.py` | 1-480 |
| 力方法 | `src/tensorlbm/force_methods.py` | `tests/test_force_convergence.py` | 1-200+ |
| 自由面 Korner | `src/tensorlbm/free_surface_lbm.py` | `tests/test_dam_break_3d_free_surface_campaign.py` | 1-1265 |
| 自由面 VOF | `src/tensorlbm/free_surface_common.py` | `tests/test_free_surface_*.py` (20+) | 1-899 |
| BFL | `src/tensorlbm/bfl_common.py` | `tests/test_bfl_suboff.py` | 1-890 |
| IBM | `src/tensorlbm/ibm_common.py` | `tests/test_ibm_*.py` (若干) | 1-500+ |
| FSI | `src/tensorlbm/fsi_common.py` | `tests/test_fsi_*.py` (若干) | 1-200+ |
| 6-DOF | `src/tensorlbm/sixdof_common.py` | `tests/test_sixdof_*.py` (若干) | 1-80 |
| 壁面模型 | `src/tensorlbm/wall_model.py` | `tests/test_wallfn_*.py` (若干) | 1-200+ |
| 溃坝 3D | `src/tensorlbm/dam_break_3d.py` | `tests/test_dam_break_3d_free_surface_campaign.py` | 1-300+ |
| 多相 SC | `src/tensorlbm/multiphase3d.py` | `tests/test_multiphase_*.py` (若干) | 1-1145 |
