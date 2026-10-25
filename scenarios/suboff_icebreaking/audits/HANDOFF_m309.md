# m309 班次交接笔记（SIR-5 交付）

> 常驻记忆 `suboff-icebreak-full/context.md` 在工作区外，本班次工具无法写入（路径越界被拒），
> 故把交接笔记随交付物一起入库，请下个班次把它并回 `context.md`。

## 当前状态

**SIR-5 已交付**：G5 断裂传播（渐进裂纹扩展 + 键断裂演化）。
机器门 5/5、语义判据 2/2、规模判据 4/4。证据：
`outputs/sir5_fracture_propagation/validation_report.json`；叙述：`validation_report_SIR5.md`。

## 三块可用引擎（已踩通，都能薄编排调用）

1. **FEM 渐进开裂真引擎** —— `TensorFEM/src/tensorfem/suboff_ice_collision.py`
   `simulate_impact(geometry, IceImpactConfig(...))` 本身就是渐进断裂：
   `history[*]` 含 `failed_points`（累计）、`fracture_dissipation_J`、`contact_force_N`、
   `maximum_penetration_m`、`hull_vertical_velocity_m_s`、`maximum_damage`；
   顶层含 `cohesive_damage` / `cohesive_area_weights` / `cohesive_seam_ids` /
   `cohesive_maximum_opening_m` / `seam_damage`。
   **这是 G5 的主引擎，也是 G4（船-冰接触 + 顶升）的现成起点**（M4.1/M4.2/M4.3 三门基本现成）。
   几何：`tensorfem.suboff_geometry.closed_revolved_mesh(cad.suboff_radius_profile, L, r, axial, circum)`。
   wall：12×16 外壳 + `ice_nx=12, ice_ny=4, duration=0.006` ≈ 45 s/CPU 核；`ice_nx≥20` 会 >120 s。
2. **DEM 3-D 键断裂真引擎** —— `TensorDEM/src/tensordem/dem3d.py`，类名 **`IceDEM3D`**
   （不是 `DEM3D`）。压板加载：`DEM3DConfig(top_platen_enabled=True,
   top_platen_velocity=-0.06, top_platen_gap=5e-4, fix_bottom=True)`；
   逐键 `failure_extension_m/failure_time_s/failure_energy_J`，账本机器精度闭合。
   15×9×3 + 2000 步 ≈ 25 s。
3. **2-D 快引擎** —— `TensorDEM/src/tensordem/dem.py: IceDEM`（wedge 工具加载，1–3 s），
   有 `fracture_events()` 与 `accounting["fracture_release_J"]`，适合小规模快速判据。

## 坑（新增，务必记）

- `importlib` 加载 `suboff_cad.py`：**必须先 `sys.modules['suboff_cad']=mod` 再 `exec_module`**，
  否则 py3.12 下 `@dataclass` 抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
- `closed_revolved_mesh`：**circum 必须是 8 的倍数**，axial ≥ 8。
- `structured_square_mesh(n)` 内部 face 总数 = **2n(n-1)**；均匀 σ_x 下只有竖缝受拉，
  坏面 = **n(n-1)**。n=3 → 12/6，与 SI-2 冻结锚一致（我第一版闭式写成 `(n-1)n` + `//2`，
  导致 M5.5 假红）。
- 必须 `PYTHONPATH=src:.`（venv editable install 指向主检出）。
- **预算教训**：本班次前 150 次调用几乎都花在参数探索上，撞到保留区才收尾。
  正确顺序：先落「可跑的薄编排脚本 + 一个最小可判工件」→ 再谈高分辨率复核。

## 待办（交后续班次）

1. FEM 侧 `ice_nx/ice_ny` 提到 24×8、`duration=0.009` 复核裂纹路径单调性与缝数收敛
   （改 `sir5_run_validation.py::stage1_fem()` 两处入参）。
2. DEM 与 SI-4 的 12 颗粒/30 键基线是两套模型，SIR-6 放大时统一到同一键结场。
3. 若要「单元内部任意路径开裂」，需向 TensorFEM 提定向修订单（裂纹追踪/拓扑演化）；
   本腿复用固定缝格是库能力边界，已如实披露。
4. **SIR-7 硬待办**：本腿只导出了破坏面时序，**未导出 contact_force–位移配对曲线**；
   SIR-7 做 P-h 对照前需补导 `history[*].contact_force_N` ×
   `metrics.maximum_ice_vertical_displacement_m` 的配对序列。
5. 把本文件内容并回 `agent_memory/suboff-icebreak-full/context.md`。

## 依赖图状态

SIR-5 完成 → SIR-7 的一个前置已解除。剩余：SIR-1/SIR-2/SIR-3 → SIR-4；
SIR-1..SIR-5 → SIR-6；SIR-5+SIR-6 → SIR-7。