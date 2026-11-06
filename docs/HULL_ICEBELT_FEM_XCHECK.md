# IN11-H4 · 船体冰带局部结构 FEM 交叉验证（IACS PC6 压力-面积律 vs ship 平台 B1 经验链）

任务：战役 DAG 节点 `IN11-H4`（物理域 / 高保真军 H4）。仓库 `TensorPhysics`，
分支 `agent/m4920`。工件：`artifacts/hull_icebelt_fem_xcheck.json`（机器判据与全部数字的唯一来源）。

## 1 案例介绍

### 1.1 物理背景

冰区船冰带（ice belt）外板在层冰中航行时承受**局部冰压斑**：冰体被船体压溃并把
压强传入一定面积的外板与纵骨。两条独立的信息链在工程上并存：

* **规则链（A）**：IACS Polar Class 规则用「压力-面积律」把冰压斑简化为矩形斑
  （高 `0.6 h_I`、长宽比 4:1），斑内平均压强随面积增大而下降
  `p(A)=p_1 (A/A_1)^(-e)`，`e≈0.5`（IACS/ISO 19906 局部斑压区间 0.45–0.55）。
  这是**设计载荷**链。
* **经验链（B）**：ship 平台 `green_navigation_ice`/`ice_fullscale`（战役 B1）用
  Korzhavin / Riska–Kämäräinen 谱系的艏部全局冰力
  `F = k_F ψ(v) σ_c b h`（接触宽 `b = 3 h`），并把 `q = F/b` 作为**结构有限元的线载荷**
  输出（`local_line_load`）。这是**全局/艏部构件**链。

本节点把两条链放在**同一冰工况**（同冰厚 `h`、同航速 `v`、同压溃强度 `σ_c`）下比较：
A 链给出板格/纵骨弯曲应力与挠度，B 链给出同一板格上的等效斑压与应力，从而得到
双链偏差的定量表与逐因子归因。

### 1.2 几何与参数

| 项 | 校核板格（数值验证用） | PC6 设计板格 |
|---|---|---|
| 沿船长跨度 a（横框架间距） | 2.40 m | 2.00 m |
| 纵向骨材间距 b | 0.800 m | 0.350 m |
| 板厚 t | 25.0 mm | 30.0 mm |
| 钢材 | AH36（σ_y = 355 MPa） | AH36（σ_y = 355 MPa） |
| a/b | 3.000 | 5.714 |
| 板柔度 β=(b/t)√(σ_y/E) | 1.328（在 W3-N03 屈曲包络 1–3 内） | 0.484（**低于**屈曲包络，见 §5） |
| 单元网格 | 48×16（单元近正方形 1.000） | 24×6（1.429） |
| 边界条件 | SSSS（w=0，转角自由 = Navier 简支） | SSSS |

钢筋材纵向骨材（T 型 400×12 + 120×18，含 350 mm 有效带板）：跨度
2.00 m，间距
0.350 m。

**锚点与声明（重要）**：`main` 中**没有** `benchmarks/ice_nav_refs`（IN10-L2 产物），
因此本节点按引文**自建最小锚并声明**：

* 冰级厚度表（IACS UR I Polar Class 设计冰厚上界）：PC6 = 1.50 m；
* 压力-面积律：`p_1 = 1.2 MPa @ A_1 = 1 m²`（本仓 `hullgirder.ice_load` 已用的
  ISO 19906/IACS 压强水平），`e = 0.5`；
* 证据等级：`EXTERNAL_CITATION_DECLARED`——**规则文本不在仓库内、未做机检**；
  明确不声明「逐条满足 IACS UR I」或「船级社认可」。

## 2 做法（方法 / 流程 / 工具链）

### 2.1 有限元：复用 H3 板单元（不重写单元）

板格用 `tensorphysics.plate_buckling.PlateModel` 的线弹性路径求解，其单元即 H3
板单元战役的**矩形平板壳单元**：Q4 双线性膜 + 16 自由度双三次 Hermite 弯曲
（每节点 `w, θ_x, θ_y, w_xy`），4×4 高斯积分，薄板 Kirchhoff 弯曲。本节点只做：

1. **板格规格与几何门槛**（`panel_fem.IceBeltPanel`）：a/b、β、单元长宽比全部落表；
2. **弯曲应力后处理**：`M = D_b κ`，`σ = 6M/t²`（复用模型只暴露膜内力，未暴露弯矩）；
3. **冰载斑的相容载荷**：见 2.3。

复用的其余部件：`plate_buckling.loads.IcePatchLoad`（IACS 斑几何 0.6 h_I、4:1）。

### 2.2 闭式解：Navier 双正弦解（独立参考）

对 SSSS 矩形板任意分段均布载荷，取 `w = Σ W_mn sin(mπx/a) sin(nπy/b)`，
`W_mn = q_mn / [D (m²π²/a² + n²π²/b²)²]`，其中矩形斑的 `q_mn` 有解析积分式。
由此得到挠度与弯矩/弯曲应力的**解析解**（`analytic.navier_field/navier_grid`），
用于逐项对标 `<5%` 判据；同时校核 `plate_buckling.navier_simply_supported_deflection`
（均匀载荷特例）以确认级数实现一致。

### 2.3 冰载斑的**相容**载荷向量（发现并修正的一个真实缺陷）

复用模型的 `PlateModel.pressure_load` 按**单元中心是否落入斑内**施加斑载荷。
对照算例显示该规则有两种误差：(i) 载荷面积被量化到整单元；(ii) 落在斑边界上的
单元中心被双侧计入（载荷可达目标 2 倍）。因此本节点用
`panel_fem.consistent_patch_load`：在斑/单元交集上做 4×4 高斯积分
`∫ q Nᵀ dA`（对双三次基精确），并保留旧规则作为可选路径
（`load_rule="element_centre"`）以便**量化**而非掩盖该误差。两条约束用测试钉死：
本地 Hermite 行与 `PlateModel.kern["Nw"]` 逐点相等（atol 1e-12）；整单元覆盖时
相容斑载荷与 `pressure_load` 完全一致。

### 2.4 双链比较与归因

同工况下把 B1 链的线载荷折算为斑压：`p_B = q / (0.6 h) = σ_c ψ(v)/0.6`，
再与 A 链 `p_A = p_1 (A/A_1)^{-e}` 对比。因为两式都是单项式，比值可**代数分解**：

```
p_B/p_A = (σ_c/p_1) · ψ(v) · (1/0.6) · (A/A_1)^{e}
```

四项分别对应：强度 vs 面积折减参考压、冲击因子、线载荷→斑压归一化、
**缺失的压力-面积折减**。分解乘积与直接比值的一致性作为判据（残差 < 0.5 %）。

### 2.5 工具链

`python3 scripts/run_hull_icebelt_fem_xcheck.py` →
`artifacts/hull_icebelt_fem_xcheck.json` + `artifacts/figs/*.png`，再由脚本渲染本报告
（数字全部来自 JSON，零手抄）。B1 模块以**只读**方式 import
（`/root/ship-performance-platform-incoming/ship-performance-platform`，
sha256 `2fbb1ac4682c9ff7cea4afd67bf6a37de269e492f2668ab04d7fb1c2e195dc95`，
commit `76c7b91f10b68ef026f5b1866c7facd6a24583ac`，
`merged_to_main=False`）。

## 3 计算结果

### 3.1 A 链数值验证（FEM vs 闭式解）

| 算例 | 载荷规则 | Δw_centre | Δσ_centre | Δw_max | 过 5% 门? | 计入判据? |
|---|---|---|---|---|---|---|
| uniform_pressure_SSSS | consistent (uniform over whole elements) | -0.102 % | -0.571 % | -- | 是 | 是 |
| iacs_patch_SSSS | consistent (4x4 Gauss on the patch/element intersection) | -0.161 % | -1.508 % | -0.161 % | 是 | 是 |
| iacs_patch_offmesh_sensitivity | consistent (4x4 Gauss on the patch/element intersection) | -0.164 % | -1.673 % | -- | 是 | 否（仅记录） |
| load_rule_sensitivity_coarse_mesh | element_centre vs consistent, 12x4 mesh | 0.042 %（相容积分） | 82.9 %（中心规则） | -- | 是 | 否（仅记录） |

最大偏差 **1.508 %**（判据 5.0 %），
`all_pass = True`。板柔度 D = 294757.3 N·m。

网格收敛（同一冰载斑，三套网格）：

| 网格 | 单元数 | 自由度 | w_centre [mm] | σ_centre [MPa] | Δw | Δσ |
|---|---|---|---|---|---|---|
| 12x4 | 48 | 390 | 1.1082 | 23.83 | 0.042 % | -18.647 % |
| 24x8 | 192 | 1350 | 1.1077 | 27.76 | -0.008 % | -5.223 % |
| 48x16 | 768 | 4998 | 1.1060 | 28.85 | -0.161 % | -1.508 % |

中心弯矩峰值随网格二阶收敛（-18.65 % →
-5.22 % → -1.51 %），
生产网格取 48×16；24→48 网格间挠度变化
-0.152 %。

### 3.2 PC6 设计点（A 链）

| 量 | 值 |
|---|---|
| 设计冰级/冰厚 | PC6 / 1.50 m |
| IACS 斑几何 | 高 0.900 m × 宽 3.600 m（4:1） |
| 设计面积（板格载入面积） | 0.700 m² |
| 设计斑压 | **1.434 MPa** |
| 载荷理想化 | uniform over the panel（斑覆盖整板格） |
| 板格 FEM 最大弯曲应力 σ_x | 52.32 MPa |
| 板格 FEM 最大弯曲应力 σ_y | 147.83 MPa |
| 板格 FEM von Mises（表面） | 131.36 MPa，利用率 **0.370** |
| 板格最大挠度 | 0.549 mm |
| 纵骨弯曲：板侧拉应力 | 69.4 MPa（利用率 0.195） |
| 纵骨弯曲：翼板压应力 | 176.3 MPa（利用率 0.497） |
| 纵骨弯矩 / 挠度 | 251.0 kN·m / 1.11 mm（L/1803） |

**面积口径灵敏度**（同一板格、同一冰级）：若把压力-面积律评在未裁切的接触斑
`0.6h×2.4h = 3.240 m²` 上，则 p =
0.667 MPa、σ_x =
24.32 MPa、利用率
0.172；主口径（面积 = 板格实际载入面积
0.700 m²）给出 p = 1.434 MPa、
利用率 0.370。线性弹性下应力与压力严格成正比，
故该灵敏度是精确的标量缩放，不是近似。

### 3.3 双链偏差定量表（同工况：σ_c = 3.0 MPa,
v = 1.00 m/s）

| h_I [m] | A_panel [m²] | p_A [MPa] | p_A(接触斑口径) [MPa] | σ_x,A [MPa] | w_A [mm] | p_B [MPa] | Δp (B vs A) | Δp (B vs A, 接触斑口径) |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.360 | 2.000 | 2.000 | 66.2 | 0.736 | 6.156 | 207.8 % | 207.8 % |
| 0.75 | 0.630 | 1.512 | 1.333 | 53.2 | 0.579 | 6.156 | 307.2 % | 361.7 % |
| 1.00 | 0.700 | 1.434 | 1.000 | 52.3 | 0.549 | 6.156 | 329.2 % | 515.6 % |
| 1.50 | 0.700 | 1.434 | 0.667 | 52.3 | 0.549 | 6.156 | 329.2 % | 823.4 % |
| 2.00 | 0.700 | 1.434 | 0.500 | 52.3 | 0.549 | 6.156 | 329.2 % | 1131.1 % |

B1 链的斑压 **6.156 MPa 与冰厚无关**
（`q ∝ h`、`0.6h` 归一化后抵消），A 链则随设计面积增大而下降——这正是
「压力-面积律」被 B1 链缺失的部分。因此偏差随冰厚单调放大：
207.8 %（h = 0.50 m）→
329.2 %（h = 2.00 m，板格面积口径）；
若用未裁切接触斑口径，偏差为
207.8 % →
1131.1 %。
**结论方向稳健：B1 链在局部板格上偏保守 2–12 倍，且偏差口径依赖设计面积定义。**

航速扫描（h = 1.5 m）显示偏差的第二个因子 ψ(v)：

| v [m/s] | ψ(v) | p_B [MPa] | p_A [MPa] | Δp |
|---|---|---|---|---|
| 0.30 | 0.8579 | 4.290 | 0.667 | 543.4 % |
| 0.50 | 1.0000 | 5.000 | 0.667 | 650.0 % |
| 1.00 | 1.2311 | 6.156 | 0.667 | 823.4 % |
| 1.50 | 1.3904 | 6.952 | 0.667 | 942.8 % |
| 2.00 | 1.5157 | 7.579 | 0.667 | 1036.8 % |

### 3.4 偏差归因（h = 1.50 m，逐因子乘积分解）

| 因子 | 数值 | 物理解释 |
|---|---|---|
| strength_to_reference_pressure | 2.5000 | sigma_c / p_1: chain B multiplies the ice crushing strength, while the chain-A reference p_1 = 1.2 MPa at 1 m^2 is already the area-reduced pressure level (not a strength) |
| impact_factor | 1.2311 | psi(v): velocity/impact enhancement present in chain B and absent in the quasi-static IACS patch pressure |
| line_load_to_patch_pressure | 1.6667 | 1/0.6: chain B's line load is not normalised to the rule patch height, so converting it to a patch pressure introduces the 1/(0.6 h_I) factor |
| missing_area_reduction | 0.8367 | (A/A_1)^(1/2) = (A)^(1/2): chain B has no pressure-area reduction; the IACS chain lowers the pressure as the patch area grows |
| **乘积** | 4.2919 | 直接比值 4.2919；代数闭合残差 0.000000 % |

**物理解释（>10 % 必须解释）**：B1 链的 `local_line_load` 是**艏部/全局压入**量级的
线载荷（由全局冰力除以 3h 接触宽得到，并被用于艏部构件/总纵强度），它本身不含
压力-面积折减。把它直接折算成板格斑压会同时带上：(i) 压溃强度 σ_c 与「已折减参考压
p_1」的语义差（主要项 ×2.5）；(ii) 与面积无关的冲击因子 ψ(v)；
(iii) 1/(0.6) 的线载荷→斑压归一化；(iv) 缺失的 (A/A_1)^0.5 折减。
四者乘积与直接比值的代数闭合残差仅 0.00000000 %。

### 3.5 图

![icebelt_pressure_vs_area](../artifacts/figs/icebelt_pressure_vs_area.png)

![icebelt_fem_vs_analytic](../artifacts/figs/icebelt_fem_vs_analytic.png)

![icebelt_stress_fields](../artifacts/figs/icebelt_stress_fields.png)

![icebelt_dual_chain](../artifacts/figs/icebelt_dual_chain.png)


图 1 压力-面积律与 B1 链（B1 为水平线=无面积效应）；图 2 冰载斑中心线挠度与
「FEM vs 闭式」偏差条形（5 % 门）；图 3 PC6 设计斑的 FEM σ_x/σ_y 场与闭式 σ_x 场；
图 4 同工况双链压力与偏差随冰厚的演化。

## 4 判据

| 判据 | 数值 | 通过? |
|---|---|---|
| fem_vs_closed_form_lt_5pct | 最大偏差 1.508 %（门 5.0 %） | 是 |
| mesh_convergence_lt_5pct_and_stable | Δw -0.161 %，24→48 变化 -0.152 % | 是 |
| dual_chain_deviation_quantified | B1 状态 AVAILABLE，5 个工况，偏差 207.8–329.2 % | 是 |
| deviation_attribution_closure | 最大代数残差 0.00000000 %（门 0.50 %） | 是 |
| pc6_design_elastic | σ_vm/σ_y = 0.370 | 是 |
| stiffener_elastic | 纵骨翼板利用率 0.497 | 是 |
| anchor_declared | ice_nav_refs 存在=False，锚点已声明=True | 是 |

`all_criteria_pass = True`。pytest：
`python3 -m pytest tests/test_hull_icebelt_fem.py -q` → 44 passed
（含负例：非法冰级 `PC9/1A/空/非字符串`、面积 ≤0/NaN/Inf、参考压或指数非法、
骨材/板厚 ≤0、负压力、退化或越界斑框、未知 `load_rule`、未知钢材/门槛模式）。

## 5 注意点（踩坑 / 局限 / 适用范围）

1. **复用模型斑载荷规则的缺陷（本单发现）**：`PlateModel.pressure_load` 按单元中心
   判斑，斑边界落在单元中心上时被双侧计入。12×4 网格上同一斑的挠度偏差达
   82.9 %
   （载荷面积比 2.0），而相容积分仅
   0.04 %。
   本节点因此自带相容斑载荷；该缺陷对 `plate_buckling` 的既有结论无影响（其算例均为
   整板均布载荷），但任何**局部斑载荷**用例都应改用相容载荷。
2. **锚点是声明的，不是机检的**：`benchmarks/ice_nav_refs` 不在 `main`；IACS 冰级
   厚度表与 p_1/e 按引文自建并在工件中标注 `EXTERNAL_CITATION_DECLARED`。
3. **B1 未并入其 main**：`merged_to_main=False`，
   本次用其工作副本模块只读直调（commit/sha256 已入审计链）；若 B1 系数变化，
   偏差表必须重算。
4. **设计面积口径**：主口径「面积 = 板格实际载入面积」使压力与载入面积自洽；未裁切
   接触斑口径给出更低的斑压（§3.2）。两者相差 2.15 倍，
   已在表中并列，未取平均、未挑选。
5. **线性弹性**：无塑性、无焊接初始变形、无总纵弯矩共同作用；PC6 设计点利用率
   0.370 处于弹性范围，故线性模型适用；极限强度
   归 `plate_buckling` / `plate_buckling_reliability` 链，本节点不重复。
6. **β 低于屈曲包络**：设计板格 β = 0.484 < 1（冰带厚板），
   W3-N03 的 β∈[1,3] 是**屈曲研究包络**而非弯曲有效性下限；如实上报而不套用
   `validate()` 隐瞒（`checks()` 中 `beta_inside_buckling_envelope=False`）。
7. **斑载均匀、几何矩形**：真实压痕是峰值型分布、冰带处曲率与外板法向偏差未建模；
   纵骨为闭式梁（无侧倾、腹板屈曲、剪力滞）。
8. **既有环境失败（与本单无关）**：`tests/test_end_to_end_example.py` 在
   `/usr/bin/python3` 下因 `tensorphysics` 未安装而失败，属既有环境问题，本单未触碰。

## 6 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 偏差 | 是否 <3%? |
|---|---|---|---|
| 板格中心挠度（均匀压力） | Navier 双正弦闭式解 | -0.1016 % | 是 |
| 板格中心弯曲应力（均匀压力） | 6M/t²，M 由闭式解曲率 | -0.5710 % | 是 |
| 板格中心挠度（IACS 冰载斑） | Navier 双正弦闭式解（矩形斑载荷） | -0.1608 % | 是 |
| 板格中心弯曲应力（IACS 冰载斑） | 6M/t²，M 由闭式解曲率 | -1.5078 % | 是 |
| 最大挠度（冰载斑） | 闭式解网格最大值 | -0.1608 % | 是 |
| 压力-面积律量值 | 声明锚 p(1 m²)=1.2 MPa、e=0.5（引文，非机检） | 不可比对（无仓库内规则文本） | 不适用（已声明） |
| B1 链局部压力 vs IACS 链 | 同工况 IACS 压力-面积链 | 329.2 %（B1 偏保守） | 否（>3%，已定量归因） |

## 7 审计链

| 项 | 值 |
|---|---|
| git commit / branch | `cdcac64b8db3d9b72db80f39b0f29feca0619004` / `agent/m4920` |
| python / numpy / scipy | 3.11.6 / 2.4.6 / 1.16.0 |
| platform | Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 |
| 模块 sha256 | iacs_pressure.py:59d293ea1e74…, analytic.py:e4acaf81f9ff…, panel_fem.py:5517f0253918…, stiffener.py:88fc129e5758…, xcheck.py:51f057273dc2… |
| 复用部件 | tensorphysics.plate_buckling.PlateModel (H3 Hermite flat shell element, Q4 membrane + bicubic Hermite bending); tensorphysics.plate_buckling.loads.IcePatchLoad (IACS patch geometry 0.6 h_I, 4:1); tensorphysics.plate_buckling.navier_simply_supported_deflection (uniform-load sanity of the analytic series) |
| B1 证据等级 | REAL_CODE_RUN_EXTERNALLY_IMPORTED |
| 锚点结论 | absent -> self-built IACS/ISO pressure-area anchor, declared in iacs_pressure.ANCHOR_PROVENANCE |

**明确不声明**：Verbatim IACS UR I rule compliance or class-society approval of the resulting scantlings is NOT claimed; only the physics chain and its numerical verification are.
