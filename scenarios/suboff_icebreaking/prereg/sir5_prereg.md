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

## Addendum 2026-10-25T07:40:00Z —— 执行口径冻结（SIR-5 开腿）

本节只**追加**判据的执行口径，不改写 §1-§7 任何条款。冻结后的判据仍以 §3/§4/§5 为准，
本节说明「用哪两个兄弟库接口、按什么参数落地」，以免跑后回填。

### 执行装置（薄编排，零求解器代码）

| 阶段 | 兄弟库接口（只读引用） | 承载的锚/门 |
|---|---|---|
| FEM 渐进开裂 | `TensorFEM/src/tensorfem/suboff_ice_collision.py` `simulate_impact` + `IceImpactConfig` | A5.1、M5.1、M5.3(FEM 侧) |
| SUBOFF 几何 | `TensorLBM/src/tensorlbm/suboff_cad.py` `suboff_radius_profile` → `tensorfem.suboff_geometry.closed_revolved_mesh` | G2 语义（几何进链，接口沿用冻结语义） |
| cohesive 律 | `TensorFEM/src/tensorfem/cohesive.py:33` `BilinearCohesiveLaw`（A5.1 已验证锚） | A5.1、A5.3 |
| DEM 键断裂 | `TensorDEM/src/tensordem/dem3d.py` `IceDEM3D` + `DEM3DConfig(top_platen_*)` | A5.4、M5.2、M5.4 |
| 解析枚举锚 | 场景内 SI-2 冻结件 `si2_fracture_gen/cohesive_failure.py` `identify_failed_faces_uniform_stress` | A5.2、M5.5 |

### 参数（跑前冻结）

- FEM 阶段：`length=1.0 m`、`axial=12`、`circum=16`（SUBOFF 外包络，`r_over_l` 由库内 `SuboffConfig` 给）、
  `IceImpactConfig(ice_nx=12, ice_ny=4, duration=0.006 s)`，其余取库默认。
- DEM 阶段：`DEM3DConfig(nx=15, ny=9, nz=3, breaking_strain=0.004, shear_breaking_strain=0.03,
  top_platen_enabled=True, top_platen_velocity=-0.06 m/s, top_platen_gap=5e-4, fix_bottom=True)`，
  推进 2000 步。
- 解析锚：`mesh_res=3, side=1.0 m, thickness=0.01 m, σ_x=2e5 Pa, σ_c=1e5 Pa, G_c=100 J/m²`
  （与 SI-2 冻结锚同口径：期望总面 12 / 坏面 6 → 本口径下总面 12、坏面 6）。

### 双账本口径

- FEM 断裂能量：账本 A = 库侧 `metrics.fracture_dissipation_J`；账本 B = 场景侧按 A5.1
  双线性律逐点独立复算 `Σ (梯形积分 − 当前存能) · A_face`。
- DEM 键断裂能量：账本 A = 逐键 `0.5·k_bond·extension²`（A5.4）复算；账本 B = 库侧
  `failure_energy_J.sum()`。
- 质量：DEM 侧颗粒数 × 单颗粒质量，初末对比（本模型无质量损失机制，残差应为机器零）。

### 已知限制（跑前声明，不许跑后拿来解释失败）

1. FEM 侧裂纹沿库内预设 cohesive 缝格扩展，不在单元内部任意取向开裂 —— 这是库能力边界，
   本腿的「渐进性」只主张**缝格尺度**的渐进传播。
2. FEM 与 DEM 是两套独立损伤变量（缝格 cohesive vs 键结颗粒），两侧账本各自闭合，
   **不是**跨库统一账本；跨库统一留给后续定向修订单。
3. 规模判据「裂纹路径长度 ≥ 2 倍板厚」以「破坏缝数 × 冰厚」为长度代理（每道缝贯通板厚）。
