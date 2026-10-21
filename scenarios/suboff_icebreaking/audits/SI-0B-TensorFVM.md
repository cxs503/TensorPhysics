# SI-0B: TensorFVM 能力审计报告

> 审计日期：2026-01-20
> 代码基线：TensorFVM main 分支（140 个 .py 模块，见 `find src -name "*.py" | wc -l`）
> 审计范围：纯读+文档，零行功能代码修改
> 目标：SUBOFF 全附体外流（SI-B 腿）五维能力基线

---

## 1. 能力矩阵（Capability Matrix）

### 1.1 贴体网格与外流求解（body_fitted / conservative3d）

| 目标 | 模块 | 成熟度 | 直接相关度 | 证据 file:line |
|------|------|--------|-----------|----------------|
| 2D O-Grid 圆柱贴体 | `body_fitted.py:59` (class BodyFittedMesh) | 有测试 | 间接——圆柱几何，非 SUBOFF 外形 | `body_fitted.py:59-232` O-Grid 构建，`solver.py:59-70` SolverConfig mesh_type 支持 body-fitted |
| 2D C-Grid 翼型贴体 | `body_fitted.py:234` (class CGridMesh) | 有测试 | 低——翼型网格 | `body_fitted.py:234-241` C-Grid |
| 2D 平板 | `body_fitted.py:243` (class FlatPlateMesh) | 有测试 | 低——边界层验证基准 | `body_fitted.py:248-320` |
| 3D 挤出 O-Grid | `body_fitted3d.py:20` (class BodyFittedCylinderMesh3D) | 有测试 | 中——圆柱 3D，非 SUBOFF | `body_fitted3d.py:20-60` |
| 3D 非结构贴体求解 | `body_fitted_transient3d.py` | 仅原型 | 低——仅限非结构 transient | `body_fitted_transient3d.py:80` (def integrate) |
| SUBOFF 几何定义 | `suboff_geometry.py:1` | 仅原型（无物理测试） | 高——SUBOFF 4 段 Groves 公式 | `suboff_geometry.py:1` 文档字符串；`suboff_geometry.py:27-90` suboff_radius_profile 实现 4 段公式（段1 指数 1/2.1、段2 平行中体、段3 6 阶多项式、段4 椭圆尾尖） |
| SUBOFF Gmsh 网格生成 | `suboff_mesh.py:1` | 仅原型（无验证测试） | 高——OCC 旋转体 + 远场盒切割 | `suboff_mesh.py:1-2` "No Cartesian immersed approximation. Boundary layers are not generated"；`suboff_mesh.py:16` generate_suboff_mesh |
| SUBOFF 混合网格（结构化+四面体） | `suboff_hybrid_mesh.py:1` | 仅原型 | 高——conforming hex 层 + 纯 Gmsh 四面体外部 | `suboff_hybrid_mesh.py:1` 文档；`suboff_hybrid_mesh.py:30` adaptive_meridian_stations 自适应母线采样；`suboff_hybrid_mesh.py:92` wall_facet_geometry_audit |
| SUBOFF 验证对标 | `suboff_validation.py:1` | 仅原型 | 高——阻力积分、Cp 采样 | `suboff_validation.py:25` drag_comparison；`suboff_validation.py:61` sample_upper_meridian_pressure |
| 3D 笛卡尔外流 | `solver3d.py:1` | 有测试（cylinder3d benchmark） | 中——仅限圆柱，非 SUBOFF | `solver3d.py:19` Cylinder3DConfig；`solver3d.py:89-280` Cylinder3DSolver |
| 3D 保守动量 | `conservative3d.py:1` | 无测试 | 中——柱对称动量，非 SUBOFF | `conservative3d.py:10` ConservativeCylinder3DSolver |
| 3D 原生多面体网格 | `native_mesh3d.py:1` | 无独立测试 | 高——tet/hex/wedge/pyramid 通用 FV 几何 | `native_mesh3d.py:20-33` NativeMesh3D dataclass；支持 4 种单元类型 |
| 已验证工况清单 | `verification/cases.py` + benchmarks | — | 见下表 |

**已验证工况（按验证覆盖度排序）：**

| 工况 | 验证状态 | Re 域 | 维度 | 湍流模型 | 证据 |
|------|----------|-------|------|----------|------|
| Poiseuille 层流 | **Benchmark-verified**（3% 门） | ~1 | 2D | laminar | `verification/cases.py:17-52` channel()；`verification/core.py:15` ERROR_LIMIT=0.03 |
| Taylor-Green 解析解 | **Benchmark-verified** | ~1 | 2D/3D | laminar | `verification/cases.py` taylor_green；MAC 投影验证 |
| 环形管层流 | **Benchmark-verified** | ~100 | 2D | laminar | `annular.py:1`；`verification/engineering.py:17` |
| DFG 2D-1/2/3 | **Benchmark-verified** | 20/100/20 | 2D | laminar | `benchmark_dfg.py` |
| 平板 SA RANS | **有测试** | 1e5 | 2D | Spalart-Allmaras | `benchmark_flat_plate.py:1`；`benchmark_flat_plate.py:32` compare_flat_plate 对比 Schlichting 公式 |
| 圆柱 URANS Re=3900 | **有测试**（筛选级） | 3900 | 2D | SA-URANS | `benchmark_cylinder_urans.py:1`；明确声明"only a screening result" |
| NACA 0012 层流 | **有测试** | ~几百 | 2D | laminar | `benchmark_airfoil.py`；3% 门 |
| 30P30N 多翼型 | **有测试** | 9e6 | 2D | SA | `benchmark_30p30n.py:224` drag 输出；`data/30p30n-hlpw4/` 实验数据 |
| 圆柱 3D Re=3900 | **有测试** | 3900 | 3D | LES/SGS Smagorinsky | `benchmark_cylinder3d.py` |

### 1.2 雷诺数域与湍流模型

| 目标 | 模块 | 成熟度 | 证据 file:line |
|------|------|--------|----------------|
| Spalart-Allmaras 稳态 | `sa_newton.py:1` | 有测试 | `sa_newton.py:1` exact frozen-velocity Newton；Jacobian 经 autodiff 验证 `sa_newton.py:48` defect>2e-10 门 |
| SA 2D 瞬态 | `solver.py` + `sa_newton.py` | 有测试 | `benchmark_cylinder_urans.py:22` SA-URANS Re=3900 |
| SST-2003 稳态 | `native_sst3d.py:1` | 仅原型 | `native_sst3d.py:1` NASA TMR SST-2003；`native_sst3d.py:40` sst2003_closure 点态闭包；`native_sst3d.py:12` "does not establish an integrated RANS solver" |
| SST 混合输运 | `native_hybrid_sst3d.py:1` | 仅原型 | `native_hybrid_sst3d.py:1` "A step does not establish nonlinear RANS convergence" |
| SST/动量耦合 | `native_hybrid_rans3d.py:1` | 仅原型 | `native_hybrid_rans3d.py:1` "Experimental segregated SST/momentum coupling"；`native_hybrid_rans3d.py:141` "Numerical convergence does not qualify a physical benchmark" |
| 壁面函数（Spalding） | `wall_functions.py:1` | **独立验证** | `wall_functions.py:1` Spalding all-y-plus；`wall_functions.py:43` wall_traction 摩擦速度迭代 90 步收敛；`verification/engineering.py:22` wall_case 最大误差 < 1e-8 |
| 壁面函数线性化 | `wall_functions.py:53` | 独立验证 | `wall_functions.py:53` wall_traction_linearization 精确雅可比 |
| Spalding 壁面模型 3D | `native_spalding3d.py:1` | 仅原型 | `native_spalding3d.py:12` 调用 wall_traction_linearization |
| SST 壁面处理 | `native_sst_wall3d.py` | 仅原型 | `native_hybrid_rans3d.py:21` wall_treatment='resolved' 或 'spalding_blended' |

**SUBOFF 典型 Re 1e6-1e7 可达性评估：**
- SA 在 Re=1e5 平板上有验证（`benchmark_flat_plate.py:23` REFERENCE_REYNOLDS=100000）
- SA 在 Re=9e6 30P30N 上有 benchmark 模块（`benchmark_30p30n.py`），但未见明确物理精度对标通过记录
- SST 尚未在物理 Re 域完成验证（仅 closure 单元级测试）
- **结论：SUBOFF Re=1e6-1e7 域目前仅具备 SA 单方程工具链，无 SST 双方程验证通过记录**

### 1.3 外开放边界与自由面处理

| 目标 | 模块 | 成熟度 | 证据 file:line |
|------|------|--------|----------------|
| 压力开放边界 | `external_open_boundary.py:1` | 有测试 | `external_open_boundary.py:1` "Pressure-open NACA far boundaries"；`external_open_boundary.py:15` OpenBoundaryNacaSolver |
| 入口 Dirichlet | `body_fitted.py` / `solver.py` | 有测试 | `body_fitted.py:356` BodyFittedSolver；入口速度在 `solver.py` config 设置 |
| 出口零梯度 | `body_fitted.py:346` | 有测试 | `body_fitted.py:792` 壁面力积分 mask 排除 outlet |
| 回流处理 | `external_open_boundary.py:13` | 有测试 | `external_open_boundary.py:17` backflow 注入 freestream 速度 |
| **自由面/近水面** | **无模块** | **缺失** | 全文搜索 `free_surface\|level_set\|VOF\|multiphase\|air_water` 零结果（仅 `interface` 出现在耦合/网格上下文） |
| **气-液两相** | **无模块** | **缺失** | TensorFVM 为单相不可压 N-S 求解器 |

### 1.4 阻力积分口径

| 目标 | 模块 | 成熟度 | 证据 file:line |
|------|------|--------|----------------|
| 壁面应力积分 | `body_fitted.py:792` _surface_force | 有测试 | `body_fitted.py:792-811` 压力 + 分子粘性剪切；SA 壁面 nu_t=0 的精确重建 |
| 气动力系数 | `body_fitted.py:813` _aerodynamic_coefficients | 有测试 | `body_fitted.py:813-844` drag/lift 旋转至气动轴；scale=0.5*rho*U²*L_ref |
| SUBOFF 阻力对标 | `suboff_validation.py:25` drag_comparison | 仅原型 | `suboff_validation.py:25-43` 力单位 rho·U²·L² → Cd = drag/(0.5*rho*U²*A)；`suboff_validation.py:37` relative_error < 3% 门 |
| SUBOFF Cp 采样 | `suboff_validation.py:61` | 仅原型 | `suboff_validation.py:61-85` 三角形最近邻采样，无拟合 |
| 动量积分 | `benchmark_flat_plate.py:45` wall_shear | 有测试 | `benchmark_flat_plate.py:45` wall_shear = drag / length |
| 力平衡门 | `verification/cases.py:42` force_balance | 有测试 | `verification/cases.py:42` abs(wallforce+pressureforce)/abs(pressureforce) < 3% |
| 3D 力积分 | `native_hybrid_ns3d.py` + `native_hybrid_rans3d.py` | 仅原型 | `native_hybrid_ns3d.py:157` interface_traction_defect_max |

**不确定性来源：**
1. SUBOFF 网格非结构化（Gmsh 四面体 + 结构化六面体），力积分基于 `NativeMesh3D` 面法向（`native_mesh3d.py`），未独立验证
2. SUBOFF 验证模块 `suboff_validation.py` 声明 `nominal_error_below_3_percent` 门，但无已通过的物理迭代记录
3. `suboff_mesh.py:3` "Boundary layers are not generated"——BL 层缺失直接影响高 Re 壁面力精度

### 1.5 可交换流体腿接口（SI-5 目标）

| 接口维度 | 现状 | 证据 |
|---------|------|------|
| 求解器注册 | `backend_registry.py` | `backend_registry.py` SolverBackend dataclass + register_backend/registered_backends |
| 统一配置 | `SolverConfig` (2D) / `Cylinder3DConfig` (3D) | 无统一 3D 通用 Config |
| 场交换 | 无标准化场交换协议 | 各求解器内部 field dict |
| 载荷契约 | `_surface_force` 返回 2D/3D 力矢量 | `body_fitted.py:792`；3D 无统一输出格式 |
| 时钟节拍 | `time_step` 可选参数 | `solver.py:44` time_step: float | None；`body_fitted.py:896` _transient_step |
| LBM 替换兼容 | **缺失** | `verification/lbm.py` 仅存在审计模块，无实时交换协议 |

---

## 2. 差距清单（按对 SI-4 全链演示的阻断性排序）

### G1. 无自由面/气液两相模块 【阻断性：致命】
**影响：** SUBOFF 近水面航行、顶破冰板场景核心依赖。当前 TensorFVM 为单相不可压 N-S 求解器，无任何 VOF/Level-Set/两相处理代码。
**证据：** 全库搜索 `free_surface\|level_set\|VOF\|multiphase\|air_water\|interface_tracking` 结果仅含"interface"（指代网格/耦合接口，非相界面）。
**阻断链：** SI-5 全链耦合演示 → 无自由面 = 无水面破冰场景 = 无法演示。

### G2. SUBOFF 物理验证未通过 【阻断性：致命】
**影响：** SUBOFF 阻力/流场无实验对标通过记录。`suboff_validation.py` 存在但无完成记录。
**证据：** `suboff_validation.py:25` drag_comparison 存在 3% 门，但无调用记录证明通过；`suboff_mesh.py:3` "Boundary layers are not generated"；`suboff_hybrid_mesh.py:1` "No y+ or viscous-flow qualification is implied"。
**阻断链：** SI-B 互证门 → 无物理验证数据 = 无法通过互证。

### G3. SST 双方程未验证 【阻断性：高】
**影响：** SUBOFF Re=1e6-1e7 需 SST 或等价 RANS。SA 在 Re=1e5 已验证但 Re=1e6+ 无记录。
**证据：** `native_sst3d.py:1` "does not establish an integrated RANS solver"；`native_hybrid_rans3d.py:1` "Experimental"；无 SST benchmark 脚本。
**阻断链：** SI-B Re 域可达性 → 无 SST 验证 = 高 Re 流场可信度不足。

### G4. 3D 非结构外流基准缺失 【阻断性：高】
**影响：** SUBOFF 是 3D 旋转体，当前 3D 验证仅限圆柱笛卡尔网格（Re=3900）和平板。
**证据：** `benchmark_cylinder3d.py` 仅圆柱；`body_fitted_transient3d.py` 为原型；无 3D 非结构化 benchmark。
**阻断链：** SI-B 几何覆盖 → 3D 非结构化 = SUBOFF 网格求解未验证。

### G5. 自由面边界条件缺失 【阻断性：高】
**影响：** 即使有了两相模型，开放边界/自由表面 BC 也未定义。
**证据：** `external_open_boundary.py` 仅处理 far-field 压力开放，非自由面。
**阻断链：** G1 的子项。

### G6. 可交换流体接口契约缺失 【阻断性：中】
**影响：** SI-5 要求 FVM 可作为 LBM 替换件。当前无统一的场/载荷/时钟交换协议。
**证据：** `verification/lbm.py` 仅审计模块；无 TensorFVM↔TensorLBM 实时耦合代码。
**阻断链：** SI-5 互操作性。

### G7. SUBOFF 附体（桨/舵/围壳）几何缺失 【阻断性：中】
**影响：** 任务目标为"全附体 SUBOFF"。当前仅 bare hull 几何。
**证据：** `suboff_geometry.py` 仅含 mother-hull radius；无桨、舵、围壳参数化。
**阻断链：** SUBOFF 全附体完整性。

---

## 3. 差距解析锚（Pre-registered Acceptance Criteria）

每个差距提议可机器裁定的解析锚，格式为 `(值-文件-键)` 三元组：

| 差距 | 锚描述 | 值 | 文件 | 键 |
|------|--------|-----|------|-----|
| G1 | 自由面模块存在性 | `import tensorfvm.free_surface` 可执行 | `scenarios/suboff_icebreaking/audits/validation_report.json` | `checks.free_surface_module_exists` |
| G2a | SUBOFF 阻力对标误差 < 3% | `relative_error < 0.03` | `suboff_validation.py:37` | `nominal_error_below_3_percent` |
| G2b | SUBOFF 网格 BL 层存在 | `min_first_cell_height < target_y+ * nu / u_tau` | `suboff_hybrid_mesh.py` | `wall_facet_geometry_audit` |
| G3a | SST channel 对标 | `wall_shear_relative_error < 0.03` | `verification/sst_channel_reference.py` | `wall_shear_relative_error` |
| G3b | SST flat plate 对标 | `Cf_relative_error < 0.20` | 新建 benchmark_sst_flat_plate.py | `cf_relative_error` |
| G4 | 3D SUBOFF 流场 MMS 验证 | `L2_velocity_error < 1e-4` | `verification/native_manufactured3d.py` | `velocity_l2_error` |
| G5 | 自由面 BC 守恒门 | `mass_imbalance < 1e-6` | 新建 | `free_surface_mass_imbalance` |
| G6 | FVM-LBM 场交换协议 | 定义 `FluidLegContract` 类 | `src/tensorfvm/leg_contract.py` | `FluidLegContract` |
| G7 | 附体几何参数化 | `appendage_count >= 3` | 新建 `suboff_appendages.py` | `appendage_count` |

---

## 4. 工作量分级

| 差距 | 分级 | 估算 | 说明 |
|------|------|------|------|
| G1 自由面/两相 | **L** | 多班次 + GPU 排队 | VOF/Level-Set 核心模块开发 + 两相 BC + 基准验证 |
| G2 SUBOFF 物理验证 | **M** | 1-2 班次 | 生成 SUBOFF 网格 + BL 层 + 跑 SA + 对标实验数据 |
| G3 SST 验证 | **M** | 1-2 班次 | SST channel/flat plate benchmark 脚本 + 对标文献 |
| G4 3D 非结构基准 | **M** | 1-2 班次 | SUBOFF MMS 或圆柱 O-grid 3D 基准 |
| G5 自由面 BC | **L** | 多班次 | 依赖 G1 |
| G6 可交换接口 | **S** | <1 班次 | 定义契约数据结构 + 文档 |
| G7 附体几何 | **S** | <1 班次 | 参数化桨/舵/围壳（Groves 报告已有定义） |

---

## 5. 接口需求（FVM ↔ 耦合对端）

### 5.1 需从对端拿入的物理量

| 物理量 | 来源 | 数据结构 | 时钟节拍 | 说明 |
|--------|------|----------|----------|------|
| 壁面位移/速度 | TensorDEM/TensorFEM | `dict[str, np.ndarray]` 节点速度 (N,3) | 每耦合步 | 用于动网格/壁面运动边界 |
| 冰板几何/位置 | TensorDEM | `dict` 冰板位姿 | 每耦合步 | 自由面/浸入边界 |
| 压力/速度场 | TensorLBM | `dict` 边界 patch 速度/压力 | 每子步 | 若 LBM 为流体腿则 FVM 不消费 |

### 5.2 需交出的物理量

| 物理量 | 接收方 | 数据结构 | 时钟节拍 | 说明 |
|--------|--------|----------|----------|------|
| 壁面剪切力矢量 | TensorDEM/TensorFEM | `np.ndarray` (N_faces, 3) + 面索引 | 每耦合步 | `body_fitted.py:792` _surface_force 输出 |
| 总阻力/升力 | TensorFEM | `dict` drag, lift, side_force | 每耦合步 | `body_fitted.py:813` 气动力系数 |
| 表面压力分布 | TensorFEM | `np.ndarray` (N_faces,) | 每耦合步 | 结构载荷分布 |
| 流场速度/压力 | TensorDEM | `np.ndarray` 边界层外速度 | 每耦合步 | DEM 冰-流交互 |

### 5.3 建议时钟节拍

| 节拍 | 建议值 | 说明 |
|------|--------|------|
| FVM 内步 | `dt_fvm ≈ 0.001-0.01 s` (CFL 约束) | 取决于网格尺寸和流速 |
| 耦合交换 | `dt_coupling ≈ 10-100 * dt_fvm` | 子循环聚合 |
| DEM 内步 | `dt_dem ≈ 1e-5-1e-4 s` | 接触刚度约束，通常比流体步小 |
| 数据对齐 | 线性插值或最近邻 | 不同时钟节拍需时间插值 |

### 5.4 数据结构契约建议

```python
@dataclass(frozen=True)
class FluidLegSnapshot:
    """FVM 向耦合管理器交出的单一快照。"""
    timestamp: float          # 物理时间 (s)
    wall_force: np.ndarray    # (N_wall_faces, 3) 壁面力矢量 (Pa·m²)
    wall_pressure: np.ndarray # (N_wall_faces,) 面压力 (Pa)
    face_centers: np.ndarray  # (N_wall_faces, 3) 壁面中心坐标 (m)
    total_drag: float         # 总阻力 (N)
    total_lift: float         # 总升力 (N)
    domain_extent: np.ndarray # (2, 3) 流场域边界 (min, max)

@dataclass(frozen=True)
class FluidLegBoundary:
    """FVM 从耦合对端接收的边界条件。"""
    timestamp: float
    wall_velocity: np.ndarray  # (N_wall_faces, 3) 壁面速度 (m/s)
    free_surface: dict | None  # 自由面几何参数 (若适用)
```

---

*审计结束。本文件仅记录现有代码基线的审计事实，不包含任何功能代码变更。*
