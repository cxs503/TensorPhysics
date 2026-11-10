# FEM/DEM 能力盘点现状表（m605 产物）

> 审计日期：2026-10-31 | 审计方法：只读源码 + 已有测试 | 证据附 file:line

---

## A. TensorFEM 模块现状

| 模块 | 单元/功能 | 成熟度 | 验证 | 关键证据 | 缺口ID |
|------|-----------|--------|------|----------|--------|
| beam2.py | 2节点 Euler-Bernoulli 梁 | ✅有测试 | PASS | test_beam.py | — |
| shell4.py | 4节点 MITC4 壳 | ✅有测试 | PASS | test_shell4.py | — |
| solid3d.py | 8节点六面体 | ✅有测试 | PASS | test_solid3d.py | — |
| cohesive.py | 两节点 cohesive 界面 (Mode-I) | ✅有测试 | PASS | test_cohesive.py (6 tests) | GAP-FEM-3D-COHESIVE |
| mixed_mode_cohesive.py | BK 混合模式准则 | ✅有测试 | PASS | test_mixed_mode_cohesive.py | GAP-FEM-P4 |
| ice_sheet.py | 冰板 (Winkler 占位) | ✅Pasternak 已闭环 (2026-11-10) | PASS | TensorPhysics `ice_sheet/pasternak_foundation.py` + `kirchhoff_plate_fem.py` (`foundation_model='pasternak'`, `g_p`): FEM vs Pasternak/Vlasov Green 函数 r=0 5e-6 / RMS 1.9 %; Winkler 退化 0 差异; 32 pytest; docs/reports/p54w3_pasternak_foundation_report.md | GAP-FEM-P0 **已关闭** |
| elastic_foundation (TensorPhysics) | 弹性地基梁/板解析验证面 | ✅验证面成熟 (2026-11-10) | PASS | `src/tensorphysics/elastic_foundation/` (beams: Hetényi / Selvadurai Pasternak 复根统一解 + Hermite FEM; plates: Kelvin kei 闭式 + Hankel 积分复核 + 轴对称 Kirchhoff 环单元 FEM + 有限域精确解): 梁 FEM 内部 L∞ 3.5e-6 / Pasternak 7.9e-5, 板 FEM 斑载剖面 L2 2.5e-7, 弯矩 0.6 %, 边界效应指数 1.35 (≈√2), 标度律指数 0.75 (FEM 反算); 27 pytest; 15 机器判据全 PASS; docs/reports/p270w1_n09_elastic_foundation.md | GAP-FEM-P0 **验证面补齐**（缺口本体由 p54w3 关闭） |
| dynamics.py | Newmark-β 隐式瞬态 | ✅有测试 | PASS | test_beam_dynamics.py | — |
| explicit_dynamics.py | 中心差分显式 | ✅有测试 | PASS | test_explicit_dynamics.py | — |
| coupled_hull.py | 线性船体接收器 | ⚠️间接 | 间接 | physical_accuracy=False | GAP-FEM-P3 |
| suboff_ice_collision.py | SUBOFF 碰撞全链 | ✅有测试 | PASS | test_suboff_ice_collision.py | — |
| surface_contact3d.py | 球体压入 Winkler 近似 | ⚠️原型 | — | winkler_sphere_load() | GAP-FEM-WINKLER |

## B. TensorDEM 模块现状

| 模块 | 功能 | 成熟度 | 验证 | 关键证据 | 缺口ID |
|------|------|--------|------|----------|--------|
| dem3d.py | 3D 键合球体 DEM | ✅有测试 | PASS | 105 颗粒基准 | GAP-DEM-G6/G7 |
| dem.py | 2D IceDEM | ✅有测试 | PASS | hull_contact 集成 | — |
| dem3d_wedge.py | 楔形船首接触 | ✅有测试 | PASS | 有限/无限宽 | GAP-DEM-G3 |
| hull_contact.py | 2D polyline 船体 | ✅有测试 | PASS | 反力平衡验证 | GAP-DEM-G3 |
| continuum_boundary_dem.py | 连续体弹簧网络 | ✅有测试 | PASS | 平面应力本构 | GAP-DEM-G5 |
| calibrated_wet_coupling.py | 2D 流-DEM 耦合 | ✅有测试 | PASS | subcycling | GAP-DEM-G1 |
| dem3d_energy.py | 能量跟踪 | ✅有测试 | PASS | 耗散率量化 | — |
| moving_plane_boundary.py | 罚函数接触(移动平面) | ✅有测试 | PASS | 冲量/能量守恒 | — |

## C. 缺口矩阵（三层交叉）

### C-1: FEM vs deal.II/CalculiX

| 能力维度 | deal.II/CalculiX | TensorFEM 现状 | 差距 |
|----------|------------------|----------------|------|
| 单元库 | 梁/板/壳/实体/cohesive/接触单元 | 梁/板/壳/实体/cohesive(2节点) | 缺 3D cohesive, 接触单元 |
| 材料非线性 | 弹性/弹塑性/蠕变/损伤 | 线弹性 + cohesive 损伤 | 缺弹塑性 |
| 接触 | 罚函数/Lagrange/mortar/AD | 罚函数(简化) | 缺 mortar/AD |
| 自适应 | h/p/refinement/Goal-oriented | 无 | 🔴阻断 |
| 并行 | MPI/domain decomp/MG | 无 | 🔴阻断 |

### C-2: DEM vs LIGGGHTS/ESyS-Particle

| 能力维度 | LIGGGHTS/ESyS | TensorDEM 现状 | 差距 |
|----------|---------------|----------------|------|
| 颗粒形状 | 球/clump/polyhedron | 仅球体 | 缺 clump/polyhedron |
| 接触模型 | Hertz/Mindlin/JKR/摩擦 | 法向弹簧阻尼(无法向摩擦) | 🔴缺摩擦 |
| 破碎 | bonded-particle/AFEM | bonded-sphere(拉/剪准则) | 基本覆盖 |
| 参数标定 | 恢复系数/摩擦系数标定流程 | 无自动标定 | 🟠高 |
| 大规模并行 | MPI+GPU/Domain decomp | CPU cell list(GPU 阻塞) | 🔴阻断 |

### C-3: 船冰应用需求 (SIR/G6)

| 需求 | 当前状态 | 差距 |
|------|----------|------|
| SIR 七腿同时接触 | 仅 wedge/polyline | 🔴缺 SUBOFF 曲面 |
| G6 规模化(100k+) | 最大 405 颗粒 | 🔴缺大规模 |
| FEM→DEM 断裂交接 | 碎片数据结构就绪, 无导出 | 🔴阻断 |
| 3D 流-固-冰全耦合 | 仅 2D 耦合 | 🔴阻断 |
| 破冰力实验对标 | 无实验数据 | 🟡降级 |
