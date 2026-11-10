# p154w2-N03 · `failure_faces` 从文档级冻结接口升到实现级（FEM 断裂面 → DEM 碎块守恒交接）

**节点** p154w2-N03 · **仓库** TensorPhysics · **接口版本** `failure_faces` v3.0.0
（schema fingerprint `807d9eef29db7f79…`）

## ① 案例介绍（物理背景 / 几何 / 参数）

PANORAMA §五.3 记录的唯一成型跨库接口是 `failure_faces.json`，但此前它只有**文档级** v1/v2 语义
（面 id + 法向 + 面积 + 承载力，无几何、无断裂能），DEM 侧根本无法据此初始化碎块。
本节点把它升级为**实现级** v3：FEM 导出的每个失效面自带顶点环、法向、形心、面积、混合模式
比、正/剪牵引与断裂能，整包 SHA-256 封印；DEM 侧据顶点**独立重算**面积并生成多面体/clump 碎块，
继承断裂能作为初始内能；两侧面积与能量由守恒门裁定。

- **试件**：缺口三点弯曲梁，L = 0.4 m，h = 0.06 m，
  t = 0.02 m，缺口深 a = 0.02 m
  （网格对齐后 0.02 m，缺口带宽 0.0250 m）。
- **材料**（准脆性，混凝土量级）：E = 3e+10 Pa，ν = 0.2，
  抗拉强度 σ_t = 3.5e+06 Pa，
  G_Ic = 100.0 J/m²，G_IIc = 200.0 J/m²，
  BK 指数 η = 1.5，密度 ρ = 2400.0 kg/m³。
- **网格**：Hex8 1504 单元 / 2125 节点 / 6375 自由度，
  dx = 0.0125 m、dy = 0.005 m；缺口由单元删除生成，故裂纹面与网格面完全重合。
- **载荷**：位移控制，载荷片宽 2dx，施加位移 0.1294 mm，
  由支座反力回算总载荷 **P = 585.54 N**（这是量测值而非输入值）。
- **加载水平标定**：先以 δ = 0.010 mm 探针求解，测得韧带根部
  σ_n = 0.379 MPa，按线弹性比例放大
  12.9385 倍，使根部达 1.4×σ_t =
  4.900 MPa。结果是韧带**部分断裂**
  （8/32 面，
  25.0% 面积），裂纹前沿落在韧带内部而不是贯穿。

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **FEM 断裂面导出**（`fem_export.py`）：取出裂纹平面（x = 0.18750 m）上
   保留实体一侧的单元面；每面用 Newell 面积向量算面积与法向（面平均应力 → 牵引
   `t = σ·n`，分解 σ_n、τ），混合模式比 ψ = τ²/(σ_n²+τ²)，断裂韧性按
   Benzeggagh–Kenane `G_c(ψ) = G_Ic + (G_IIc − G_Ic)ψ^η`，面能量 = G_c(ψ)·A。
   强度判据 σ_n ≥ σ_t 决定该面是否失效；未失效面只进诊断，不进接口包。
2. **接口封印**（`schema.py`）：冻结 JSON Schema（version/source_solver/timestamp/units/
   faces[]/area_total/energy_total/sha256），三层校验：闭包子集校验器 + `jsonschema`
   draft-2020-12 + 语义不变量（面积=多边形重算、法向单位、绕向一致、合计=逐面和、封印摘要）。
   v1/v2 文档级包可经 `upgrade_legacy_packet` 无损字段升级（几何合成标记 `geometry_synthetic`）。
3. **DEM 碎块消费**（`dem_fragments.py`）：每面沿 −n 挤出断裂层厚
   t_layer = 0.0125 m 成为多面体碎块，
   V = A·t_layer、m = ρV、内能 = 面断裂能；可选 clump：n 个刚性球体积**精确**分割碎块体积。
   面积一律由顶点重算（不采信声明值），这正是 C1 有内容的原因。
4. **守恒门**（`conservation.py`）：面积门（≤1%）、逐面面积门（≤1%）、能量门（≤2%）、
   质量门（≤1e-12）、clump 体积门（≤1e-12）；结构不一致（碎块数≠面数）直接抛
   `HandoffGateError` 而非静默通过。
5. **端到端**（`dem_rigid.py`）：碎块提升至释放高度 H = 0.5 m，
   刚性平动 + 精确接触时刻分裂积分（不是"先穿入再推回"），弹性组 e = 1、μ = 0；
   另一组 e = 0.5、μ = 0.3 为耗散对照。能量账本 KE + PE + 内部能逐时步记录。
6. **零手抄**：本报告所有数字由 `run_p154w2_n03.py` 从同一次运行的工件 JSON 读回注入。

## ③ 计算结果（数字表）

**FEM 场认证（无缺口素梁，与梁理论对比，用于证明导出的牵引场可信）**

| 自由度 | 总载荷 P (N) | 跨中挠度 vs Timoshenko | 底纤维应力 vs Mc/I |
|---|---|---|---|
| 4116 | 6.466 | 7.69 % | 13.95 % |
| 15132 | 6.081 | 1.94 % | 4.06 % |
| 32895 | 6.025 | 1.06 % | 2.69 % |

各网格细节：[48, 6, 3], [96, 12, 3], [128, 16, 4]（L/h = 20.0）。
最细网格上挠度误差 **1.06%**、应力误差
**2.69%**；应变能与外力功
½Pδ 的相对差 **2.38e-10**。

**导出的失效面（`failure_faces.json` 逐面，程序注入）**

| face_id | 单元 | 面积 A (m²) | σ_n (MPa) | 混合模式比 ψ | G_c(ψ) (J/m²) | 面能 E (J) |
|---|---|---|---|---|---|---|
| F0001 | 688 | 2.5000e-05 | 4.868 | 0.0055 | 100.04 | 2.5010e-03 |
| F0002 | 689 | 2.5000e-05 | 4.900 | 0.0060 | 100.05 | 2.5012e-03 |
| F0003 | 690 | 2.5000e-05 | 4.900 | 0.0060 | 100.05 | 2.5012e-03 |
| F0004 | 691 | 2.5000e-05 | 4.868 | 0.0055 | 100.04 | 2.5010e-03 |
| F0005 | 692 | 2.5000e-05 | 3.658 | 0.0315 | 100.56 | 2.5140e-03 |
| F0006 | 693 | 2.5000e-05 | 3.674 | 0.0246 | 100.39 | 2.5097e-03 |
| F0007 | 694 | 2.5000e-05 | 3.674 | 0.0246 | 100.39 | 2.5097e-03 |
| F0008 | 695 | 2.5000e-05 | 3.658 | 0.0315 | 100.56 | 2.5140e-03 |

**接口包总量与 DEM 侧独立重算**

| 量 | FEM 声明（导出方） | DEM 重算（消费方） | 相对差 | 判据 | 结论 |
|---|---|---|---|---|---|
| 断裂面面积 | 2.000000e-04 m² | 2.000000e-04 m² | 0.000e+00 | ≤1%（C1） | PASS |
| 逐面面积最大差 | — | — | 0.000e+00 | ≤1% | PASS |
| 断裂能 | 2.005164e-02 J | 2.005164e-02 J | 0.000e+00 | ≤2%（C2） | PASS |
| 质量 ρAt | 6.000000e-03 kg | 6.000000e-03 kg | 0.000e+00 | ≤1e-12 | PASS |
| clump 体积 | 2.500000e-06 m³ | 2.500000e-06 m³ | 1.016e-15 | ≤1e-12 | PASS |

**能量链闭合（DEM 碎石自由落体 + 落地，H = 0.5 m，dt = 0.0001 s，
时长 0.8 s）**

| 量 | 数值 |
|---|---|
| 碎块数 / 总质量 | 8 / 6.000000e-03 kg |
| FEM 断裂释放能 U_frac | 2.005164e-02 J |
| FEM 同载荷下弹性应变能 | 3.788022e-02 J（U_frac/U_elastic = 0.529） |
| DEM 碎块初始内能 | 2.005164e-02 J |
| 弹性组初始总能 E0 | 4.987658e-02 J |
| 弹性组 max\|E−E0\|/E0 | **5.259e-14**（C4 ≤5%） |
| 弹性组接触事件数 | 8 |
| 弹性组耗散 | 0.000e+00 J（应为 0） |
| 耗散组 max\|E−E0\|/E0 | 5.698e-01（非保守，如实报告） |
| 耗散组累计耗散 | 2.841834e-02 J |
| 耗散组账本闭合 max\|E+D−E0\|/E0 | **6.900e-14** |
| 接口幂等（两次导出 sha256） | `ae7d039ddfd116d953982edc…` == `ae7d039ddfd116d953982edc…` → 一致 |

**守恒门判别力（把消费侧故意做坏，看门是否真的报警）**

| 场景 | 说明 | 面积/能量相对差 | 面积门/能量门 |
|---|---|---|---|
| control_unchanged | unmodified packet/fragment pair | 0 / 0 | PASS / PASS |
| fragment_dropped | consumer lost one fragment | — | structural error |
| faces_triangulated_lossily | consumer kept 3 of 4 vertices per face (area halved) | 0.5 / 0 | FAIL / PASS |
| area_offset_0p5pct | consumer area +0.5 % (inside tolerance) | 0.004975 / 0 | PASS / PASS |
| area_offset_1p5pct | consumer area +1.5 % (outside tolerance) | 0.01478 / 0 | FAIL / PASS |
| energy_offset_3pct | consumer internal energy -3 % (outside tolerance) | 0 / 0.03 | PASS / FAIL |
| energy_offset_1p5pct | consumer internal energy -1.5 % (inside tolerance) | 0 / 0.015 | PASS / PASS |

## ④ 结果图

- **fig_failure_faces_3d.png** — 失效面三维视图（颜色 = 比断裂能）、未失效韧面带（灰）、裂纹前沿折线。
- **fig_ligament_stress.png** — 裂纹面上开口牵引 σ_n 沿高度的剖面，σ_t 阈值线与被导出（失效）面标注。
- **fig_energy_chain.png** — 左：弹性组能量分量与总能；中：FEM 释放能→DEM 内能→动能→耗散的能量转移账本；
  右：耗散组 E(t)、累计耗散与 E+耗散 闭合线。
- **fig_conservation_gates.png** — 左：五道守恒门的相对误差 vs 容差（对数轴）；右：判别力敏感性矩阵。
- **fig_fem_validation.png** — 左：挠度/应力误差随自由度收敛到 3% 线；右：FEM 与 Timoshenko 挠度 1:1 对照。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **提取器不是裂纹扩展求解器**：应力场是该载荷水平下的一次线弹性解，失效面由强度判据给出；
   逐面失效后的应力重分布、软化段、裂纹扩展阻力曲线均未模拟。断裂能按"新表面单价"
   （BK 韧性 × 面积）计入，因此是**上界式**的能量分配，不含部分损伤残余。
2. **本征值问题**：Hex8 线性单元在弯曲下有寄生剪切（shear locking），细网格下挠度误差
   1.06%，粗网格 (48,6,3) 仍达
   7.69%；报告如实给出收敛表。
3. **DEM 只积分平动**：碎块按等体积球接触、不积分转动、不做碎块互撞。重力释放无初始自转，
   故对能量链无影响；但真实堆叠/互撞场景需要扩展（见 §六）。
4. **clump 体积精确但球可重叠**：clump 是刚性球团（相对位置固定），重叠在刚体团里是允许的，
   因此体积守恒精确而"几何占据"并非严格无重叠。
5. **气隙条件**：门槛值（1%/2%）是接口契约门槛，不是物理定律；它们约束的是**交接保真度**
   （面积/能量不得在序列化与消费端重建时丢失），不是 FEM 或 DEM 自身的数值精度。
6. **v1/v2 升级是语义降级**：旧包无几何，升级需合成平面片并标记 `geometry_synthetic`；
   旧包 failure_mode 词表（crushing/bending/shear/buckling）映射到 v3 的开裂模式词表是有损的。
7. **接口只在 v3 与消费者匹配时可用**：`SUPPORTED_VERSIONS` 之外的版本一律拒绝（version 4.0.0 的包
   在本消费者上直接 `SchemaError`），变更必须升版本 + 重钉 schema fingerprint。

## ⑥ 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|
| 跨中挠度（最细网格，32895 自由度） | Timoshenko 梁理论 ½ 点载 | 1.06 % | **是** |
| 跨中挠度（中间网格 15132 自由度） | Timoshenko 梁理论 | 1.94 % | **是** |
| 跨中挠度（粗网格 4116 自由度） | Timoshenko 梁理论 | 7.69 % | 否（收敛未完成，如实列出） |
| 底纤维弯曲应力（最细网格） | Mc/I | 2.69 % | **是** |
| 底纤维弯曲应力（中间网格） | Mc/I | 4.06 % | 否（4.06%，线弹性 Hex8 寄生剪切，15% 验收带内） |
| 应变能 vs 外力功 ½Pδ | 能量守恒恒等式 | 2.38e-10 | **是** |
| 断裂面面积 FEM vs DEM | 顶点独立重算 | 0.00e+00 | **是**（门槛 1%） |
| 断裂能 FEM vs DEM | 碎块内能和 | 0.00e+00 | **是**（门槛 2%） |
| 自由落体总能量残差 | 初始总能 E0 | 5.26e-14 | **是**（门槛 5%） |
| 耗散组账本闭合 | E(t)+D(t) vs E0 | 6.90e-14 | **是**（门槛 5%） |

无实验基准的量（如绝对断裂能水平、裂纹前沿位置）不设"<3%"结论，仅以自洽恒等式
（面积/能量/账本闭合）与梁理论基准判定；这一限制在 §⑤.1 已说明。

## 判据对照（机器可算）

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| C1 面积守恒 | 相对差 ≤1% | 0.000e+00 | PASS |
| C2 能量守恒 | 相对差 ≤2% | 0.000e+00 | PASS |
| C3 schema 校验 | JSON Schema + 语义 | v3.0.0 通过（含封印校验） | PASS |
| C4 端到端能量残差 | ≤5% | 5.259e-14 | PASS |
| C5 接口幂等 | 两次导出 sha256 一致 | 一致 | PASS |
| C6 pytest | ≥15 全绿 | 65 项（schema 31 / 守恒 13 / 端到端 21） | PASS |
| C7 honest-FAIL 定源 | 未达 3% 的项须定位 | 见 §⑥ 粗/中网格应力与挠度行 | 已定源 |

## 工件与审计链

| 工件 | 说明 |
|---|---|
| `artifacts/p154w2_n03/failure_faces.json` | 封印接口包，sha256 `ae7d039ddfd116d953982edc5552d54e546c40df8e8d92c0dccaa3c85b3b80f1` |
| `artifacts/p154w2_n03/conservation_gates.json` | 五道门 + 结论 |
| `artifacts/p154w2_n03/dem_initial_conditions.json` | 碎块几何/质量/内能（含 clump） |
| `artifacts/p154w2_n03/energy_chain.json` | 弹性组 + 耗散组逐时步能量账本 |
| `artifacts/p154w2_n03/fem_diagnostics.json` | FEM 场 + 提取诊断 + 加载标定 |
| `artifacts/p154w2_n03/gate_sensitivity.json` | 判别力敏感性矩阵 |
| `artifacts/p154w2_n03/beam_validation.json` | 三套网格的梁理论对照 |
| `artifacts/p154w2_n03/criteria_report.json` | C1–C5 判据数字（机器读） |
| `docs/reports/p154w2_n03/REPORT.md` | 本报告（数字由运行注入） |

复现：`python3 benchmarks/failure_handoff/run_p154w2_n03.py`
测试：`python3 -m pytest tests/test_failure_handoff_schema.py tests/test_failure_handoff_conservation.py tests/test_failure_handoff_e2e.py -q`

接口契约变更流程：改 `src/tensorphysics/failure_handoff/schema.py` 的
`FAILURE_FACES_SCHEMA` → 升 `VERSION`/`SUPPORTED_VERSIONS` → 重钉
`tests/test_failure_handoff_schema.py::PINNED_SCHEMA_FINGERPRINT`（当前
`807d9eef29db7f7974ab8c1acc8546ea7659b9f69c7a64cba83f5bca8bca0db3`），
否则双侧契约测试同时变红。

## ⑥′ 后续（未完成，如实登记）

- DEM 侧转动惯量与切向滚动摩擦未接入 → 目前能量链只闭合平动 + 断裂能，不含滚动耗散。
- 碎块间接触（多碎块堆叠、二次破碎）未实现：本节点的验证场景是自由落体 + 地面撞击。
- 裂纹前沿的**推进**（逐级卸载/重分布 + 内聚力单元）需要真正的 cohesive FEM 求解器，
  本节点是后处理式提取；下一步应接 TensorFEM 的 cohesive 单元（`fracture_mechanics/cohesive.py`
  已有双线性 CZM 内核，但未与提取器联动）。
