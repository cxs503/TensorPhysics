# FEM/DEM 能力盘点与成组任务树 — m605 进展报告

> **Mission ID**: m605  
> **Branch**: agent/m605  
> **Date**: 2026-10-31  
> **Owner**: 2026-10-31  

---

## 1. 案例介绍（物理背景与几何参数）

### 1.1 物理背景

船舶破冰是极区航运的核心工程问题。SIR（Suboff Icebreaking Research）线是当前 TensorPhysics 的主战场，
涉及 **四库耦合**：TensorLBM（流场）→ TensorFEM（船体/冰板结构）→ TensorDEM（碎冰块离散）→ TensorPhysics（编排）。

当前进展锚点：
- **wedge S1-S4** 已收官：S1 DEM 冰层+反力账本 / S2 FEM 单向共轭 / S3 流体载荷守恒 / S4 闭环时钟
- **m579 G4** 在跑：两自由度刚体 + FEM 船体 + DEM 冰层断裂-再接触
- **m598** 破冰锚库刚成
- **G5/G6** 未派（G6 规模化等 G4 终态）

### 1.2 关键参数表

| 参数 | 值 | 来源 |
|------|-----|------|
| 冰密度 ρ_ice | 917 kg/m³ | dem3d.py:24 硬编码 |
| 键合断裂应变 | 0.015 (拉) / 0.03 (剪) | dem3d.py:27-28 |
| 接触刚度 | 用户定义 | dem3d.py:364 |
| 接触阻尼 | 5.0 (默认) | dem3d.py:29 |
| FEM 板单元 | Shell4 (4节点壳) | tensorfem/shell4.py |
| cohesive 模型 | 双线性牵引-分离律 | tensorfem/cohesive.py:32 |
| 瞬态积分 | Newmark-β / 中心差分 | tensorfem/dynamics.py, explicit_dynamics.py |
| SI-4 演示规模 | 25节点×16单元板, 12颗粒×30键 | validation_report.json |
| SIR5 断裂规模 | 405颗粒, 3560键, 624断裂 | sir5 run_summary.json |

---

## 2. 方法（资产盘点 + 三层交叉缺口分析）

### 2.1 资产盘点方法

对 TensorFEM 和 TensorDEM 进行 **只读源码审计**，逐模块评估：
- 单元类型 / 颗粒形状族
- 材料模型 / 接触本构
- 断裂/破碎算法
- 时间积分方案
- 并行度 / GPU 支持
- 验证状态（有测试 / 仅原型 / 不存在）

证据全部附 `file:line`，零手抄。

### 2.2 三层交叉方法

**Layer 1 — FEM 对标 deal.II/CalculiX**：
- 单元库（梁/板/壳/实体/cohesive）
- 材料非线性（弹性/弹塑性/损伤）
- 接触算法（罚函数/Lagrange/mortar）
- 自适应（h/p/refinement）
- 并行（MPI/domain decomposition）

**Layer 2 — DEM 对标 LIGGGHTS/ESyS-Particle**：
- 接触模型（法向/切向/滚动摩擦）
- 破碎（bonded-particle / clump）
- 参数标定（恢复系数映射）
- 大规模并行（GPU/Domain decomposition）

**Layer 3 — 船冰应用需求（SIR 七腿 + G6 规模化）**：
- 七腿同时接触力平衡
- 100k+ 颗粒碎冰场
- FEM→DEM 断裂碎片交接
- 流-固-热全耦合

---

## 3. 计算结果（现状表 + 缺口矩阵）

### 3.1 TensorFEM 能力现状

| 模块 | 单元/功能 | 成熟度 | 验证状态 | 关键证据 |
|------|-----------|--------|----------|----------|
| beam2.py | 2节点Euler-Bernoulli梁 | ✅有测试 | PASS | test_beam.py |
| shell4.py | 4节点MITC4壳 | ✅有测试 | PASS | test_shell4.py |
| solid3d.py | 8节点六面体 | ✅有测试 | PASS | test_solid3d.py |
| cohesive.py | 两节点cohesive界面 | ✅有测试 | PASS | test_cohesive.py (6 tests) |
| mixed_mode_cohesive.py | BK混合模式 | ✅有测试 | PASS | test_mixed_mode_cohesive.py |
| ice_sheet.py | 冰板(Winkler占位) | ⚠️部分 | 部分PASS | 地基板未实现,浮力有测试 |
| dynamics.py | Newmark-β隐式 | ✅有测试 | PASS | test_beam_dynamics.py |
| explicit_dynamics.py | 中心差分显式 | ✅有测试 | PASS | test_explicit_dynamics.py |
| coupled_hull.py | 线性船体接收器 | ⚠️间接 | 间接PASS | physical_accuracy=False |
| suboff_ice_collision.py | SUBOFF碰撞全链 | ✅有测试 | PASS | test_suboff_ice_collision.py |

### 3.2 TensorDEM 能力现状

| 模块 | 功能 | 成熟度 | 验证状态 | 关键证据 |
|------|------|--------|----------|----------|
| dem3d.py | 3D键合球体DEM | ✅有测试 | PASS | 105颗粒基准 |
| dem.py | 2D IceDEM | ✅有测试 | PASS | hull_contact集成 |
| dem3d_wedge.py | 楔形船首接触 | ✅有测试 | PASS | 有限/无限宽 |
| hull_contact.py | 2D polyline船体 | ✅有测试 | PASS | 反力平衡验证 |
| continuum_boundary_dem.py | 连续体弹簧网络 | ✅有测试 | PASS | 平面应力本构 |
| calibrated_wet_coupling.py | 2D流-DEM耦合 | ✅有测试 | PASS | subcycling |
| dem3d_energy.py | 能量跟踪 | ✅有测试 | PASS | 耗散率量化 |

### 3.3 缺口矩阵（按阻断性排序）

| 优先级 | 缺口ID | 描述 | 阻断级别 | 影响 |
|--------|--------|------|---------|------|
| P0 | GAP-FEM-P0 | Winkler弹性地基板缺失 | 🔴阻断 | SI-4全链 |
| P0 | GAP-DEM-G1 | 3D DEM↔流体双向耦合缺失 | 🔴阻断 | SI-3/SI-4 |
| P0 | GAP-DEM-G2 | 浮力/湿耦合缺失 | 🔴阻断 | SI-4碎冰浮沉 |
| P0 | GAP-DEM-G3 | SUBOFF曲面接触缺失 | 🔴阻断 | SI-4船体-冰 |
| P0 | GAP-INTEGRATION | FEM→DEM碎片交接缺失 | 🔴阻断 | SI-2断链 |
| P1 | GAP-FEM-P1 | 附加质量未接入质量矩阵 | 🔴阻断 | 频率误差 |
| P1 | GAP-DEM-G4 | 无切向摩擦/滚动阻力 | 🟠高 | 物理保真度 |
| P2 | GAP-FEM-P3 | coupled_hull大转动缺失 | 🟡降级 | 大变形工况 |
| P2 | GAP-DEM-G6 | GPU cell list同步阻塞 | 🟡中 | 大规模性能 |
| P3 | GAP-DEM-G7 | 无大规模压测(>10k) | 🟡中 | 规模可信度 |

### 3.4 能力雷达图

![能力雷达图](runs/m605/fig1_capability_radar.png)

**解读**：TensorFEM 在时间积分(7)和断裂(6)方面较强，但在自适应(1)、并行(2)、工程规模(2)方面严重不足。
TensorDEM 在接触(4)和断裂(5)有基础，但单元库(2)、流固耦合(2)、工程规模(2)是短板。
与 deal.II/CalculiX 和 LIGGGHTS 参考代码相比，整体差距在 4-6 分（10分制）。

### 3.5 三层交叉缺口热图

![缺口热图](runs/m605/fig3_gap_heatmap.png)

**解读**：
- **FEM→DEM交接** 三个视角均为 BLOCKING（红色），是最紧迫的缺口
- **流固耦合** DEM侧为 BLOCKING（3D耦合缺失），FEM侧为 Gap（Winkler/附加质量）
- **并行/规模化** 应用需求为 BLOCKING（G6 需 100k+ 颗粒），当前两侧均为 Gap
- **时间积分** 是唯一接近覆盖的维度（FEM/DEM 均有成熟积分器）

### 3.6 任务树依赖 DAG

![依赖DAG](runs/m605/fig2_dependency_dag.png)

**解读**：
- **L1 基础能力层**（8节点）：可立即启动的并行组 A（F01/F03/F04/F05/V04）
- **L2 验证锚层**（4节点）：依赖 L1 部分完成
- **L3 应用集成层**（4节点）：I01/I03 依赖多个 L1 节点，I02/I04 等 G4 终态
- **红框节点**（7个）阻塞 m579 G4 关键路径

---

## 4. 成组任务树（16 节点，278 工时预算）

### 4.1 任务总览

| 层 | 节点数 | 总工时 | 并行组 |
|----|--------|--------|--------|
| L1 基础能力 | 8 | 136h | A(5并行), B(4), C(3) |
| L2 验证锚 | 4 | 50h | B(2), C(1), A(1) |
| L3 应用集成 | 4 | 92h | D(2), E(2等G4) |
| **合计** | **16** | **278h** | — |

### 4.2 关键路径

```
F01(Winkler) → F02(附加质量) → F08(大转动) → I01(SIR七腿) → I04(实验对标)
F05(摩擦) → F06(3D耦合) → F07(浮力) → I01
F04(曲面接触) → I01
F03(碎片导出) → I03(FEM→DEM闭环) → I04
```

### 4.3 并行策略

- **立即启动（组A）**：F01, F03, F04, F05, V04 — 5个任务可并行
- **第二批（组B）**：F02, F06, V01, V02 — 依赖A组部分完成
- **第三批（组C）**：F07, F08, V03 — 依赖B组
- **集成批（组D）**：I01, I03 — 依赖L1/L2大部分完成
- **等G4批（组E）**：I02, I04 — 等m579 G4终态

### 4.4 任务树 JSON

完整任务树已入库：`artifacts/fem_dem_task_tree.json`

验证结果：
- ✅ 节点数 = 16 ≥ 12
- ✅ 每节点含 goal/acceptance_criteria/budget_hours/depends_on/gap_refs
- ✅ DAG 无环（拓扑排序验证通过）
- ✅ 所有判据可失败（数值阈值明确）

---

## 5. 注意点（踩过的坑/局限/适用范围）

### 5.1 审计局限

1. **只读审计**：未修改任何 src/ 代码，所有论断基于源码 + 已有测试
2. **规模限制**：SI-4 演示为 smoke 级（12颗粒），非工程规模
3. **GPU 未验证**：dem3d.py 声称支持 CUDA，但 cell list 可能同步阻塞
4. **物理精度**：coupled_hull.py 的 `physical_accuracy_qualified` 硬编码为 False

### 5.2 任务树局限

1. **工时估算**：基于经验估算，未考虑代码重构难度
2. **依赖 m579**：I01/I02/I04 依赖 G4 终态确认接口
3. **实验数据**：I04 实验对标需要模型试验数据（目前缺失）
4. **GPU 硬件**：V03 GPU 加速需要 CUDA 硬件环境

### 5.3 适用范围

- 本任务树适用于 TensorPhysics FEM/DEM 能力补齐
- 不适用于 LBM/FVM 求解器内核开发
- 不重复 G4 在跑范围（两自由度刚体+FEM船体+DEM冰层断裂-再接触）

---

## 6. 误差判定

本单为 **能力盘点与任务规划**，无数值计算结果，故无误差判定。

任务树中每个节点的 acceptance_criteria 均包含明确的数值阈值（如 <5%, <1e-12），
可在后续开发单中逐项验证。

---

## 7. 结论与展望

### 7.1 结论

1. **TensorFEM** 在板/壳单元、cohesive 断裂、瞬态积分方面成熟，但 Winkler 地基板、附加质量、大转动是阻断级缺口
2. **TensorDEM** 在键合颗粒、断裂、楔形接触方面有基础，但 3D 流固耦合、浮力、SUBOFF 曲面接触、摩擦模型缺失
3. **FEM→DEM 交接** 是最紧迫的三方 BLOCKING 缺口
4. **16 节点任务树** 覆盖基础能力/验证锚/应用集成三层，总预算 278 工时

### 7.2 展望

- **短期（1-2月）**：启动组A（F01/F03/F04/F05/V04），补齐基础能力
- **中期（3-4月）**：完成 L1/L2，启动 I01/I03 集成
- **长期（5-6月）**：等 G4 终态后启动 I02/I04，实现 G6 规模化和实验对标

---

## 附录：文件清单

| 文件 | 描述 |
|------|------|
| `artifacts/fem_dem_task_tree.json` | 任务树 JSON（16节点，机器可读） |
| `runs/m605/fig1_capability_radar.png` | 能力雷达图 |
| `runs/m605/fig2_dependency_dag.png` | 依赖 DAG 图 |
| `runs/m605/fig3_gap_heatmap.png` | 缺口矩阵热图 |
| `runs/m605/generate_figures.py` | 图表生成脚本 |
| `docs/reports/m605_fem_dem_audit_report.md` | 本报告 |
