# SI-0D: TensorDEM 能力审计（Phase 0 — 纯读）

> 审计对象：`/nfs/wangxi/tensor_eco/TensorDEM`（104 个 .py 模块）
> 审计日期：2026-10-21
> 审计方法：只读源码 + 已有测试用例，不修改任何 src/ 代码
> 所有"已有"论断均附 `file:line` 证据，零手抄。

---

## 1. Capability Matrix

### 1-A 碎冰块接触模型（dem3d）

| 子目标 | 模块 | 成熟度 | 直接相关度 | 证据（file:line） |
|--------|------|--------|------------|-------------------|
| 颗粒形状族（仅球体/无 clump） | `dem3d.py` | **有测试** | 直接 | `dem3d.py:19` "3-D bonded-sphere model"；`dem3d.py:130-133` 仅支持 box/cylinder_xz，粒子坐标=格点×2×radius，纯球体；`dem3d.py:355` 接触检测 `overlap = (2 * cfg.radius - cdistance)` 纯球-球 |
| 接触本构（法向弹簧+阻尼，无切向摩擦） | `dem3d.py` | **有测试** | 直接 | `dem3d.py:364` `contact_mag = cfg.contact_stiffness * overlap - cfg.contact_damping * vn` 仅法向；`dem3d.py:29` 默认 `contact_damping=5.0`；无 `friction`/`restitution` 参数（全库搜索零结果） |
| 恢复系数标定 | 无专门模块 | **仅原型** | 间接 | `dem3d.py:364` 法向阻尼通过 `contact_damping` 参数实现能量耗散，但无 restitution→damping 的解析映射函数；`dem3d_energy.py:55-84` `_dissipation_rates()` 可量化阻尼耗散率，用于后验拟合但不提供预标定接口 |
| 键合断裂（拉/剪应变准则） | `dem3d.py` | **有测试** | 直接 | `dem3d.py:27-28` `breaking_strain=0.015, shear_breaking_strain=0.03`；`dem3d.py:304-319` `_commit_fracture()` 拉/剪双准则；测试 `test_dem3d.py:40-48` 验证断裂不可逆 |
| 接触空间搜索 | `dem3d.py` | **有测试** | 直接 | `dem3d.py:227-260` `_contact_candidate_pairs()` CPU cell list（cell width = 2r，27 邻域）；`dem3d.py:110` 注释 "CPU-side cell hashing is a prototype and can synchronize CUDA execution" |

### 1-B 船体-冰接触（hull_contact）

| 子目标 | 模块 | 成熟度 | 直接相关度 | 证据（file:line） |
|--------|------|--------|------------|-------------------|
| 2D polyline 船体轮廓接触 | `hull_contact.py` | **有测试** | 直接 | `hull_contact.py:1` "Penalty contact between DEM disks and a prescribed 2-D polyline hull"；`hull_contact.py:7-17` `polyline_contact_forces()` 返回粒子力和反作用力，含端点接触；测试 `test_hull_contact.py:9-17` 验证已知力和反作用平衡 |
| IceDEM 集成 hull_profile | `dem.py` | **有测试** | 直接 | `dem.py:84-86` `hull_profile, hull_start, hull_velocity` 构造参数；`dem.py:279-281`  hull 存在时用 polyline 替代圆形压头；测试 `test_hull_contact_integration.py:8-22` 验证水平运动反应 |
| 3D 船体曲面接触（wedge 破冰线） | `dem3d_wedge.py` | **有测试** | 直接 | `dem3d_wedge.py:17-67` `MovingWedgeBowIceDEM3D` 楔形船首，2D 楔沿 y 方向挤压；`dem3d_wedge.py:74-108` `_wedge_contact()` 法向/侧向接触力；`dem3d_wedge.py:86-96` 可选有限半宽（transverse slab）；测试 `test_dem3d_moving_wedge_bow.py:13-68` 有限/无限宽均覆盖 |
| 全附体 SUBOFF 曲面接触 | 无 | **不存在** | 阻断 | 当前 `hull_contact.py` 仅 2D polyline；`dem3d_wedge.py` 仅理想楔形+有限宽 slab；无三角网格曲面（STL/OBJ）接触，无 SUBOFF 几何导入 |

### 1-C 浮沉与湿耦合（calibrated_wet_coupling）

| 子目标 | 模块 | 成熟度 | 直接相关度 | 证据（file:line） |
|--------|------|--------|------------|-------------------|
| LBM 流-固双向耦合（2D） | `calibrated_wet_coupling.py` | **有测试** | 直接 | `calibrated_wet_coupling.py:6` `from tensorlbm.ice_coupling_2d import CoupledIce2D`；`calibrated_wet_coupling.py:7-66` 继承 `CoupledIce2D` 实现 2D 流-DEM 耦合，含 subcycling（`calibrated_wet_coupling.py:23` `self.substeps=math.ceil(config.fluid_dt_s/dc.recommended_dt)`）；测试 `test_calibrated_wet_coupling.py:11-23` 验证交换/重启一致性 |
| 浮力 | 无 | **不存在** | 阻断 | 全库搜索 "buoyancy\|hydrostatic\|water_level\|submerged" 零结果 |
| 湿阻力（浸水阻尼） | 无 | **不存在** | 阻断 | `dem3d.py:345` `-cfg.drag * self.velocities` 为全局粘性阻尼，非浸水面积相关的湿阻力；无 `rho_water`/`rho_ice` 或冰-水密度比参数 |
| 冰-水密度比浮稳态 | 无 | **不存在** | 阻断 | 密度 `density=917.0`（冰密度）硬编码在 `dem3d.py:24`；无水体密度参数；无浮力平衡计算 |

### 1-D continuum_boundary_dem（FEM→DEM 交接）

| 子目标 | 模块 | 成熟度 | 直接相关度 | 证据（file:line） |
|--------|------|--------|------------|-------------------|
| 连续体弹簧网络（nu=1/3 平面应力） | `continuum_boundary_dem.py` | **有测试** | 直接 | `continuum_boundary_dem.py:1-5` "Linear central-spring control-volume network"；`continuum_boundary_dem.py:10-46` 方网+对角线+边界半权弹簧；测试 `test_continuum_boundary.py:25-33` 验证平面应力本构恢复 E=1e5 |
| 罚函数接触（移动平面） | `moving_plane_boundary.py` | **有测试** | 直接 | `moving_plane_boundary.py:11-32` `MovingPlaneNetwork.moving_step()` 半步 Verlet+端点力二次积分功；`smooth_plane()` C2 五次曲线接触 |
| 断裂生成的直接胚（continuum→DEM 交接） | 无 | **不存在** | 高差距 | `continuum_boundary_dem.py` 为纯连续体网络，独立于 disk DEM（`continuum_boundary_dem.py:1` "isolated from disk DEM"）；无 continuum→DEM 的断裂碎片传递机制 |
| 重启/守恒验证 | `continuum_boundary_dem.py` + `moving_plane_boundary.py` | **有测试** | 间接 | `continuum_boundary_dem.py:81` `snapshot()`/`restore()`；测试 `test_continuum_boundary.py:46-56` 验证重启一致性和冲量/能量守恒 |

### 1-E 碎冰场规模

| 子目标 | 模块 | 成熟度 | 直接相关度 | 证据（file:line） |
|--------|------|--------|------------|-------------------|
| 颗粒数上限 | `dem3d.py` | **有测试**（小规模） | 直接 | 基准测试默认 `nx=7, ny=5, nz=3` → 105 颗粒（`benchmark_dem3d.py:195-197`）；测试用例最大 `nx=4, ny=3, nz=2`（`test_dem3d.py:58`）；无大规模压测（>10k 颗粒） |
| GPU/CPU 路线 | `dem3d.py` | **有测试** | 直接 | `dem3d.py:73-74` 支持 "cpu"/"cuda"；`dem3d.py:116-117` CUDA 可用性检查；但 `dem3d.py:110` "CPU-side cell hashing is a prototype and can synchronize CUDA execution" 暗示 GPU 下 cell list 可能同步阻塞 |
| 与流体双向交换节拍 | `calibrated_wet_coupling.py` | **有测试**（2D） | 直接 | `calibrated_wet_coupling.py:23` substeps = ceil(fluid_dt / dem_dt)；`calibrated_wet_coupling.py:62` `exchange_sample_time` 按 `config.exchange_steps * config.fluid_dt_s` 节拍采样；仅 2D（LBM 2D） |
| 3D DEM↔流体双向交换 | 无 | **不存在** | 阻断 | 3D DEM (`dem3d.py`) 无 `exchange_state()` 方法（2D `dem.py:389-400` 有）；无 3D 耦合框架 |

---

## 2. 差距清单（按对 SI-4 全链演示的阻断性排序）

| # | 差距描述 | 阻断性 | 影响 SI 阶段 |
|---|---------|--------|-------------|
| G1 | **无 3D DEM↔流体双向耦合框架** — `dem3d.py` 无 `exchange_state()`，无 TensorLBM 3D 耦合层 | **阻断** | SI-3 基础，SI-4 全链 |
| G2 | **无浮力/湿耦合模型** — 无浮力计算、无水体密度、无冰-水密度比浮稳态 | **阻断** | SI-4 碎冰浮沉演化 |
| G3 | **无全附体 SUBOFF 曲面接触** — 仅 wedge 理想楔形+2D polyline，无三角网格曲面导入 | **阻断** | SI-4 船体-碎冰接触 |
| G4 | **无切向摩擦/滚动阻力** — 仅法向弹簧阻尼，无摩擦系数 μ、无滚动摩擦、无恢复系数标定映射 | **高** | SI-4 碎冰接触物理保真度 |
| G5 | **continuum→DEM 断裂碎片传递机制缺失** — 连续体网络与 disk DEM 隔离 | **高** | SI-2 断裂→碎冰衔接 |
| G6 | **CPU cell list 在 GPU 下可能同步阻塞** | **中** | 大规模碎冰场（>10k）性能 |
| G7 | **无大规模颗粒压测** — 基准最大 ~105 颗粒，未验证 10k-100k 规模 | **中** | SI-3 碎冰场规模可信度 |
| G8 | **无 GPU 原生空间搜索** — cell hashing 在 CPU 上执行 | **低** | 与 G6 相关 |

---

## 3. 解析锚表（预注册判据）

每个差距的可机器裁定锚：

| 差距 | 锚类型 | 判据（值-文件-键三元组） | 裁定来源 |
|------|--------|--------------------------|---------|
| G1 | 接口存在性 | `dem3d.py` 中 `exchange_state` 方法存在 → `False`（grep 结果为空） | 机器 grep |
| G2 | 守恒律 | 浮力公式 `F_b = rho_water * g * V_submerged`；取 `rho_water=1025 kg/m³`（海水），`g=9.81 m/s²`，V_submerged 由冰粒浸没深度计算。判据：单球半浸浮力误差 < 1% 相对解析值 | 文献解析解 |
| G3 | 几何导入 | SUBOFF 母型坐标文件（STL/OBJ）导入后，曲面法向与接触法向偏差 < 2° | 几何复现 |
| G4 | 恢复系数标定 | 单球碰撞：`e = exp(-ηπ/√(4mk-η²))`（damped harmonic oscillator）；给定 restitution e=0.3（冰-冰），反算 `contact_damping`。判据：碰撞后速度比 `v_after/v_before` 与目标 e 误差 < 5% | 文献解析解（Hunt & Crossley 1975 / Thornton 1997） |
| G5 | 碎片传递 | continuum_boundary_dem 断裂键位坐标 → dem3d 新碎冰颗粒位置，质量守恒误差 < 0.1% | 质量守恒律 |
| G6 | 性能基准 | GPU 上 cell list 构建时间占比 < 总步时间的 20% | 性能测试 |
| G7 | 规模上限 | 固定内存下，10k/50k/100k 颗粒可运行且不 OOM | 机器复现 |
| G8 | GPU 原生搜索 | 全 GPU cell list 构建（无 CPU 同步） | 架构验证 |

---

## 4. 工作量分级

| # | 差距 | 分级 | 理由 |
|---|------|------|------|
| G1 | 3D DEM↔流体双向耦合 | **L** | 需新建耦合层、TensorLBM 3D 端适配、exchange_state 接口实现、subcycling 节拍、重启一致性；需多班次 + GPU 验证 |
| G2 | 浮力/湿耦合 | **M** | 浮力计算本身不复杂（阿基米德原理），但需确定浸没体积计算方法、水密度参数、浮稳态测试；1-2 班次 |
| G3 | SUBOFF 曲面接触 | **L** | 需网格导入、三角面-球接触检测、法向/切向力分解、多波段船体曲面支持；多班次 |
| G4 | 切向摩擦/滚动阻力 | **M** | 法向已有，添加切向弹簧+库仑摩擦+滚动阻力，1-2 班次；restition 标定需额外测试脚本 |
| G5 | continuum→DEM 碎片传递 | **M** | 已有 continuum 断裂事件和 DEM 颗粒初始化，需搭桥接层；1-2 班次 |
| G6 | GPU cell list 阻塞 | **M** | 需重写 cell list 为 GPU kernel（CUDA/PyTorch scatter）；1-2 班次 |
| G7 | 大规模压测 | **M** | 编写大规模基准脚本 + 运行（可能需 GPU 排队）；1-2 班次 |
| G8 | GPU 原生搜索 | **M** | 与 G6 合并实施 |

---

## 5. 接口需求（耦合对端交互）

### 5.1 本库需从耦合对端（TensorLBM / TensorFVM / TensorFEM）拿入的物理量

| 物理量 | 数据结构 | 建议节拍 | 来源库 |
|--------|---------|---------|--------|
| 流体速度场 `u(x,y,z)` | `torch.Tensor[3, Nx, Ny, Nz]` 或 SPH 粒子列表 | 每 `exchange_steps * fluid_dt` | TensorLBM / TensorFVM |
| 流体压力场 `p(x,y,z)` | `torch.Tensor[Nx, Ny, Nz]` | 同上 | TensorLBM / TensorFVM |
| 连续体位移场 `U(x,y,z)`（FEM 冰板） | `torch.Tensor[N_nodes, 3]` | 每 FEM 求解步 | TensorFEM |
| 连续体断裂面位置 | 顶点列表 + 法向 | SI-2 断裂事件触发时 | TensorFEM |

### 5.2 本库需交出的物理量

| 物理量 | 数据结构 | 建议节拍 | 目标库 |
|--------|---------|---------|--------|
| DEM 颗粒位置 `x_i` | `torch.Tensor[N, 3]` | 同拿入节拍 | TensorLBM / TensorFVM |
| DEM 颗粒速度 `v_i` | `torch.Tensor[N, 3]` | 同上 | TensorLBM / TensorFVM |
| DEM 颗粒半径 `r_i` | `torch.Tensor[N]` 或标量 | 初始化时 + 碎裂更新 | TensorLBM / TensorFVM |
| 颗粒-船体接触力合力 `F_hull` | `torch.Tensor[3]` | 每 DEM 步（可用于船体六自由度积分） | TensorFVM / 船体运动模块 |
| 碎冰断裂事件列表 | `list[dict{particle_id, time, position, mode}]` | 断裂发生时 | TensorLBM / TensorFVM / TensorFEM |

### 5.3 时钟节拍建议

- **流体主步**：`dt_fluid`（LBM 典型 1e-5~1e-4 s）
- **DEM 子步**：`dt_dem = dt_fluid / substeps`，`substeps = ceil(dt_fluid / dt_dem_recommended)`
  - 参照现有 `calibrated_wet_coupling.py:23` 的模式
- **交换节拍**：每 `exchange_steps` 个流体步交换一次（参照 `calibrated_wet_coupling.py:62`）
- **建议**：3D 耦合沿用 2D 已验证的 subcycling 架构，将 `exchange_state()` 从 `dem.py:389-400` 模式扩展到 `dem3d.py`

---

*审计完毕。所有"已有"条目均可在当前源码 + 测试文件中直接定位验证。*
