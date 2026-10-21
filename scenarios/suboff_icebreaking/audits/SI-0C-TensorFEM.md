# SI-0C：TensorFEM 能力审计报告

> **审计目标**：评估 TensorFEM（`/nfs/wangxi/tensor_eco/TensorFEM`，174 src 模块）在
> SUBOFF 破冰战役 SI-C 腿中的就绪度。  
> **审计日期**：2026-10-21  
> **审计人**：tensor-agent (m127)  
> **基线提交**：agent/m127 分支  
> **方法**：逐目标阅读源模块 + 对应测试文件 + 导入依赖链，论断一律带 file:line。

---

## 一、能力矩阵（Capability Matrix）

### 目标 C-1：连续冰板——弹性地基板（Winkler 地基上的板单元）

| 条目 | 模块/路径 | 成熟度 | 相关度 | 证据 (file:line) |
|------|-----------|--------|--------|-----------------|
| Q4 Mindlin 板单元 | `src/tensorfem/plate.py` | ✅ 有测试 | 高 | `plate.py:36` `q4_mindlin_stiffness()`；`test_plate.py` 覆盖 |
| 简支正弦解验证 | `src/tensorfem/plate.py:108` | ✅ 有测试 | 高 | `solve_simply_supported_sine_square()` |
| Kirchhoff/Mindlin 解析挠度 | `src/tensorfem/plate.py:149-156` | ✅ 有测试 | 中 | 参考解口径 |
| Shell4 MITC 单元 | `src/tensorfem/shell4_mitc.py` | ✅ 有测试 | 高 | `shell4_mitc.py` `_membrane_sri()`；`test_shell4_mitc.py` |
| 球面投影 Shell4 | `src/tensorfem/spherical_shell.py` | ⚠️ 仅原型 | 中 | `spherical_shell.py:11` `projected_shell4_stiffness()`，无独立测试文件 |
| Winkler 地基板 **直接** 实现 | — | ❌ 缺失 | — | `grep -rli "winkler\|elastic_foundation"` 仅在 `surface_contact3d.py:120` 出现 `winkler_sphere_load()`——用于球体压入的简化接触模型，**非**弹性地基板连续模型 |
| Hetényi 级数解 | — | ❌ 缺失 | — | 全仓库搜索 Hetényi/Hetenyi/级数 → 零匹配 |

**论断 C-1**：板/壳单元基础成熟，但 **Winkler 弹性地基板**（连续冰板的核心物理模型）尚未实现。
现有 `winkler_sphere_load`（`surface_contact3d.py:120`）是离散接触惩罚模型，不等价于连续弹性地基板。

### 目标 C-2：Cohesive 断裂

| 条目 | 模块/路径 | 成熟度 | 相关度 | 证据 (file:line) |
|------|-----------|--------|--------|-----------------|
| Mode-I 双线性牵引-分离律 | `src/tensorfem/cohesive.py:32` | ✅ 有测试 | 高 | `BilinearCohesiveLaw`；`test_cohesive.py` 6 个测试 |
| 能量释放率（∫traction·dopening = Gc）| `src/tensorfem/cohesive.py:55-59` | ✅ 有测试 | 高 | `envelope_traction()` + `test_cohesive.py:26` trapezoid 积分验证 |
| 两节点界面单元力/刚度 | `src/tensorfem/cohesive.py:95` | ✅ 有测试 | 高 | `two_node_interface_response()`；`test_cohesive.py:50` |
| Mixed-Mode BK 准则 | `src/tensorfem/mixed_mode_cohesive.py:23` | ✅ 有测试 | 中 | `BenzeggaghKenaneLaw`；`test_mixed_mode_cohesive.py` |
| cohesive 在完整 SUBOFF 碰撞中集成 | `src/tensorfem/suboff_ice_collision.py:164` | ✅ 有测试 | 高 | `cohesive_response()`；`test_suboff_ice_collision.py:37` |
| 起始准则（traction-based 峰值） | `src/tensorfem/cohesive.py:45-48` | ✅ 有测试 | 高 | `onset_opening = peak_traction / stiffness` |
| 损伤不可逆性 | `src/tensorfem/cohesive.py:78` | ✅ 有测试 | 高 | `maximum = torch.maximum(old_max, tensile_opening)` |

**论断 C-2**：cohesive 断裂核心能力 **成熟**。Mode-I 有完整测试；Mixed-Mode BK 有比例路径验证。
局限：`cohesive.py` 仅实现两节点法向界面，尚无面内 cohesive 单元族（如 wedge S1 3D 实体 cohesive）。

### 目标 C-3：破冰瞬态动力学（板-水附加质量处理）

| 条目 | 模块/路径 | 成熟度 | 相关度 | 证据 (file:line) |
|------|-----------|--------|--------|-----------------|
| Newmark-β 线性瞬态 | `src/tensorfem/dynamics.py:13` | ✅ 有测试 | 高 | `newmark_linear()`；`test_beam_dynamics.py:32` 能量守恒验证 |
| 显式中心差分 | `src/tensorfem/explicit_dynamics.py:30` | ✅ 有测试 | 高 | `central_difference()`；`test_explicit_dynamics.py` |
| 临界时间步 | `src/tensorfem/explicit_dynamics.py:17` | ✅ 有测试 | 高 | `critical_time_step()` Gershgorin 界 |
| 机械能双账本 | `src/tensorfem/dynamics.py:40` | ✅ 有测试 | 高 | `mechanical_energy()` 动能+势能；`test_beam_dynamics.py` |
| SUBOFF 碰撞全链瞬态 | `src/tensorfem/suboff_ice_collision.py:321` | ✅ 有测试 | 高 | `simulate_impact()`；`test_suboff_ice_collision.py` |
| 水动力附加质量（ice_sheet） | `src/tensorfem/ice_sheet.py:28` | ⚠️ 无测试 | 中 | `water_added_mass_per_area: float = 0.` **声明但未集成到质量矩阵**，仅在 IceSimulationConfig 中作为占位字段 |
| 静水浮力 | `src/tensorfem/ice_sheet.py:145` | ✅ 有测试 | 高 | `buoyancy_response()`；`test_ice_sheet.py:71` 阿基米德验证 |
| 冰碎片诊断 | `src/tensorfem/ice_sheet.py:163` | ✅ 有测试 | 中 | `ice_fragments()`；`test_ice_sheet.py:89` 质量守恒验证 |

**论断 C-3**：显式/隐式瞬态积分器成熟，SUBOFF 碰撞全链有测试。**附加质量**字段（`ice_sheet.py:28`）存在但未
接入动力学系统——是零值占位，未对质量矩阵做修正。

### 目标 C-4：coupled_hull.py 能力评估

| 条目 | 模块/路径 | 成熟度 | 相关度 | 证据 (file:line) |
|------|-----------|--------|--------|-----------------|
| LinearHullReceiver 类 | `src/tensorfem/coupled_hull.py:44` | ✅ 有测试 | 高 | 构造函数；`test_surface_coupling.py:66` 使用 |
| 船体质量组装 | `src/tensorfem/coupled_hull.py:57-66` | ⚠️ 间接测试 | 高 | 钢壳质量 + 总质量缩放 |
| Shell4 刚度集成 | `src/tensorfem/coupled_hull.py:63` | ✅ 间接测试 | 高 | `projected_shell4_stiffness` 逐单元组装 |
| Leapfrog 时间推进 | `src/tensorfem/coupled_hull.py:80` | ⚠️ 间接测试 | 高 | `advance()` Verlet/leapfrog 半步格式 |
| 表面耦合接口 | `src/tensorfem/coupled_hull.py:77` | ✅ 有测试 | 高 | `surface()` 返回 `QuadSurfaceExchange` |
| 能量/动量守恒双账本 | `src/tensorfem/coupled_hull.py:91-101` | ⚠️ 间接测试 | 高 | `diagnostics()` 返回 kinetic/strain/external_work/impulse |
| 物理精度限定 | `src/tensorfem/coupled_hull.py:102` | ❌ 已知局限 | — | `physical_accuracy_qualified: False` 硬编码 |
| 大转动/非线性壳 | — | ❌ 缺失 | — | `coupled_hull.py:2` 注释："No FEM ice is assembled...finite-rotation shell dynamics and coupled wet accuracy are open" |

**论断 C-4**：`coupled_hull.py` 是一个 **线性小应变船体接收器**，用于外部耦合荷载（来自 LBM 流体）。
已有船体-结构耦合雏形，但：① 仅限小应变线弹性；② 无大转动几何非线性；③ 物理精度标志硬编码为 False；
④ 无独立测试文件（仅被 `test_surface_coupling.py` 间接使用）。

### 目标 C-5：破坏→离散交接（cohesive 失效后交 DEM 数据通道）

| 条目 | 模块/路径 | 成熟度 | 相关度 | 证据 (file:line) |
|------|-----------|--------|--------|-----------------|
| 碎片连通性分析 | `src/tensorfem/ice_sheet.py:163` | ✅ 有测试 | 高 | `ice_fragments()` BFS 连通域分解 |
| 碎片质量守恒验证 | `src/tensorfem/ice_sheet.py:184` | ✅ 有测试 | 高 | `total_retained_mass_kg`；`test_ice_sheet.py:89-98` |
| 碎片运动学（质量/质心/速度/动能） | `src/tensorfem/ice_sheet.py:179-183` | ✅ 有测试 | 高 | 每碎片 `mass_kg/center_m/mean_velocity_m_s/translational_kinetic_energy_J` |
| cohesive 失效阈值 | `src/tensorfem/ice_sheet.py:169` | ✅ 有测试 | 高 | `damage[sheet['seam_ids']==sid]>=.999` 判定断裂 |
| DEM 数据导出接口 | — | ❌ 缺失 | — | 无任何 `export_to_dem` / `dem_fragment_input` / `to_json_dem` 函数 |
| 质量守恒门 | `src/tensorfem/suboff_ice_simulation.py:245` | ✅ 有测试 | 高 | `ice_mass_retained: abs(fragments['total_retained_mass_kg']-ice_mass)<1e-9*max(ice_mass,1)` |

**论断 C-5**：**碎片数据结构已就绪**（连通域分析+质量守恒+运动学），但 **无 DEM 数据导出接口**。
`ice_fragments()` 的输出结构是 Python dict，需序列化+格式约定才能被 DEM 消费。

---

## 二、差距清单（按 SI-4 全链演示阻断性排序）

| 优先级 | 差距描述 | 阻断级别 | 当前状态 | 影响链路 |
|--------|----------|---------|---------|---------|
| **P0** | Winkler 弹性地基板连续模型缺失（冰板-水耦合的核心物理） | 🔴 阻断 | 零实现 | SI-4 全链演示 → 无法定义连续冰板的初始状态和边界响应 |
| **P1** | 板-水附加质量未集成到质量矩阵（`ice_sheet.py:28` 为零占位） | 🔴 阻断 | 声明未实现 | SI-4 瞬态动力学 → 破冰频率/周期误差 |
| **P2** | DEM 数据导出接口缺失（碎片 dict → DEM 输入文件/结构） | 🟠 半阻断 | 数据结构就绪 | SI-2 质量守恒门 → 碎片交接断链 |
| **P3** | coupled_hull 大转动/非线性能力缺失 | 🟡 降级 | 仅线弹性小应变 | SI-3 船体-冰碰撞 → 大变形工况降级 |
| **P4** | Mixed-Mode cohesive 仅支持比例路径 | 🟡 降级 | 非比例路径不支持 | SI-1 混合断裂 → 复杂载荷路径降级 |
| **P5** | Hetényi 级数解参考未实现 | 🟢 非阻断 | 零实现 | SI-C 验证锚 → 可用数值解替代 |
| **P6** | wedge S1 3D 实体 cohesive 单元族缺失 | 🟢 非阻断 | 仅两节点界面 | SI-C 三维断裂 → 可用板单元 cohesive 替代 |

---

## 三、解析锚（可机器裁定判据）

### 锚 A-1：Winkler 弹性地基板解析解

**预注册判据**：
```
值-文件-键三元组：
  - 值: center_deflection_m (板中心挠度)
  - 文件: scenarios/suboff_icebreaking/audits/validation_SI0C_anchor_A1.json
  - 键: "winkler_plate_center_deflection"
判据形式:
  abs(winkler_plate_center_deflection - analytical_hetenyi_deflection) / analytical < 5%
解析锚来源:
  Hetényi, M. (1946) Beams on Elastic Foundation, Sec. 4.1
  圆形板 Winkler 地基中心挠度: w₀ = q·a⁴/(64·D) · f(k·a⁴/D)
  其中 D = E·h³/(12(1-ν²)), k = 水地基模量
```

### 锚 A-2：附加质量系数基准

**预注册判据**：
```
值-文件-键三元组:
  - 值: added_mass_coefficient
  - 文件: scenarios/suboff_icebreaking/audits/validation_SI0C_anchor_A2.json
  - 键: "added_mass_per_area_SI"
判据形式:
  abs(added_mass_per_area - rho_water * pi * a² / 4) / expected < 10%
解析锚来源:
  Lamb (1932) Hydrodynamics §371, 无限深水中平板附加质量系数
  m_added ≈ (π/4) · ρ_water · a² (等效半径 a 的圆盘)
```

### 锚 A-3：cohesive 断裂能量守恒（双账本）

**预注册判据**（已有测试覆盖，直接复用）:
```
值-文件-键三元组:
  - 值: test_cohesive.py:26 test_monotonic_envelope_integrates_to_fracture_energy
  - 文件: TensorFEM/tests/test_cohesive.py
  - 键: relative_error < 1e-8
判据形式:
  trapezoid(envelope_traction, opening) / fracture_energy - 1 < 1e-8
已有证据: test_cohesive.py:26-29
```

### 锚 A-4：碎片质量守恒

**预注册判据**（已有测试覆盖，直接复用）:
```
值-文件-键三元组:
  - 值: total_retained_mass_kg
  - 文件: TensorFEM/tests/test_ice_sheet.py
  - 键: test_precrack_separates_two_fragments_without_losing_mass
判据形式:
  abs(total_retained_mass_kg - initial_mass) / initial_mass < 1e-12
已有证据: test_ice_sheet.py:94
```

### 锚 A-5：Newmark 能量守恒

**预注册判据**（已有测试覆盖，直接复用）:
```
值-文件-键三元组:
  - 值: mechanical_energy(M,K,u,v)
  - 文件: TensorFEM/tests/test_beam_dynamics.py
  - 键: test_newmark_sdof_free_vibration_and_energy_conservation
判据形式:
  abs(E(t)/E(0) - 1) < 1e-10 (无阻尼自由振动)
已有证据: test_beam_dynamics.py:32-47
```

---

## 四、工作量分级

| 差距 | 分级 | 理由 | 预估工时 |
|------|------|------|---------|
| Winkler 弹性地基板连续模型 | **L** | 需新模块（地基刚度矩阵推导 + 板-地基耦合集成 + Hetényi 验证脚本 + 测试），涉及连续介质力学推导 | 2-3 班次 |
| 板-水附加质量集成 | **M** | 需修改质量矩阵组装逻辑，对接 `water_added_mass_per_area` 到节点质量，+ 验证 | 1-2 班次 |
| DEM 数据导出接口 | **M** | 已有碎片数据结构，需定义 JSON/VTK 输出格式 + 序列化工具 + DEM 侧协议对齐 | 1-2 班次 |
| coupled_hull 大转动能力 | **L** | 需 corotational shell 与 coupled_hull 集成 + 大变形接触 + 新测试套件 | 多班次 |
| Mixed-Mode 非比例路径 | **M** | 需扩展 BenzeggaghKenaneLaw 处理非比例加载路径 + 历史变量管理 | 1-2 班次 |
| Hetényi 级数参考解 | **S** | 纯 Python 脚本，级数求和，已有文献公式可直译 | <1 班次 |
| wedge S1 3D cohesive | **L** | 新单元族（6/15 节点 wedge），需要形状函数 + 数值积分 + cohesive 本构嵌入 + 测试 | 多班次 |

---

## 五、接口需求

### 5.1 从耦合对端拿入（TensorFEM ← 外部）

| 物理量 | 来源 | 数据结构建议 | 时钟节拍 |
|--------|------|-------------|---------|
| 流体压力场 p(x,y,z,t) | TensorLBM 流体求解器 | `SurfaceLoad`（见 `surface_coupling.py:38`），含 `pressure_pa` + `time_s` | 每 Δt_LBM 同步，建议 Δt_LBM ≤ Δt_FEM_stable |
| 碎冰初始位置/速度/质量 | TensorDEM | 与 `ice_fragments()` 输出同构的 JSON：`fragments: [{mass_kg, center_m, mean_velocity_m_s}]` | 破冰事件触发（damage≥0.999 时） |
| 水地基模量 k_w | 外部波浪/海况数据 | 标量或场 `k_w(x,y)` [Pa/m]，Winkler 模型参数 | 初始化时一次加载 |
| 附加质量场 | 势流/面板法 | `added_mass_per_area(x,y)` [kg/m²]，对角加到节点质量 | 初始化时一次加载 |

### 5.2 向耦合对端交出（TensorFEM → 外部）

| 物理量 | 去向 | 数据结构建议 | 时钟节拍 |
|--------|------|-------------|---------|
| 船体表面位移/速度 | TensorLBM（移动壁面边界） | `QuadSurfaceExchange`（见 `surface_coupling.py:57`）：nodes+elements+face_offset | 每 Δt_FEM |
| 冰碎片运动学 | TensorDEM | `ice_fragments()` 输出的 JSON 序列化，含每片质量/质心/速度/动能 | 断裂事件触发（异步） |
| 结构内力/应力 | 后处理/损伤评估 | von Mises 场（`coupled_hull.py:100`），节点位移场 | 每 Δt_FEM 或按需 |
| 能量双账本 | 全局耦合监控 | `{kinetic_J, strain_J, external_work_J, energy_residual_J}` | 每 Δt_FEM |

### 5.3 时钟节拍建议

| 物理过程 | 特征时间尺度 | 建议 Δt | 依据 |
|----------|-------------|---------|------|
| 船体弹性振动 | T₁ ~ 0.01-0.1 s | Δt ≤ 0.5·T₁/N_modes | `coupled_hull.py:74` `conservative_step_bound()` |
| 冰板断裂 | T_fracture ~ 0.001-0.01 s | Δt ≤ 0.5·critical_time_step | `explicit_dynamics.py:17` |
| 流体-结构耦合 | T_LBM ~ CFL 限制 | Δt_FEM ≥ Δt_LBM（子循环） | 稳定性要求 |
| 碎冰 DEM 交接 | 事件驱动 | 断裂触发即交接 | 质量守恒门 |
