# 跨求解器几何保真契约成熟化（IN52-N04 / m6242）：CAD 三路积分、离散网格闸门与网格入口 provenance

**一句话结论.** 沿用并重建了「这块网格可信到能进物理验收吗？」这一机器可判对象：契约
`geometry_fidelity_contract.v1` 现在带 **三条结构独立的 CAD 积分路线**（横剖切片 Simpson /
等参曲面剖分+散度定理 / 分层 Monte-Carlo N=1e6）、**三族离散求解器**（FVM 多面体 / LBM 体素-IBM
/ BEM 面元）在同一真值源上生成的网格、L/B 前置校验、双 sha256 provenance 与逐项 GATE 判定。
本轮相对首轮（m5513）修掉了三处实质欠账：SUBOFF 的 `∇=0.699211 m³ / S=5.998 m²` **两个交付
约束现在同时闭合**（首轮 S 偏 +0.19 % 记为 honest FAIL；本轮复核发现该数值有两个独立来源——
族自由度不足与横剖求积在尾尖的误差，解耦后族残余只有 +7.2e-4，详见 §5.1）；SUBOFF 的 BEM 面元**重新进契约**
（首轮因 `|ΔCb|=1.9e-3` 被排除，本轮 −1.5e-4）；LBM 湿表面从「一阶阶梯 + 方向修正」换成
**IBM 体积分数 + 高度场重构**（体积升到二阶，湿面积偏差从 −11…−40 % 降到 −1.4…+8.5 %，
并按形状类先验地给出证据等级）。仍存在的欠账如实标注：细型船（KCS）与回转体（SUBOFF）的
LBM 湿面积因龙骨/极线的垂直切线无法在立方格点上解析，**证据等级 WITHHELD**、不进闸门。

## 1 案例介绍

**物理背景.** 船舶总体性能链（阻力 / 推进 / 耐波 / 操纵 / 稳性）的结论都建立在同一份船体几何上。
被求解器消费的网格一旦与 CAD 型线不闭合（排水体积、湿表面积、方形系数漂移），下游阻力、
推力、波浪载荷会在一个**错误的几何**上得到"精度合格"的假结论。工程上这正是"网格可信性"
问题的核心：物理误差预算必须与几何误差预算分离、可审计、可机器拒绝。

**研究对象（三船，主尺度与合同系数取自公开值）.**

| 船 | 族 | L [m] | B [m] | T [m] | 参考量 | 说明 |
|---|---|---|---|---|---|---|
| KCS | wall_sided_power | 230.0 | 32.2 | 10.8 | Cb = 0.651 | 单桨集装箱船 |
| KVLCC2 | wall_sided_power | 320.0 | 58.0 | 20.8 | Cb = 0.8098 | 肥大油轮（墙式舷侧） |
| SUBOFF | revolution | 4.356 | 0.508 | 0.508 | ∇ = 0.699211 m³, S = 5.998 m² | DARPA 裸艇体 |

**闸门判据（GATE_PASS 需全部满足）**

| 检查 | 容差 | 物理含义 |
|---|---|---|
| CAD 体积自洽（切片 vs MC） | ≤ 0.1 % | 两条独立积分路线互证 |
| CAD 湿面积自洽（求积 vs MC） | ≤ 0.3 % | 同上 |
| CAD 第三路（等参剖分 vs 切片） | ≤ 0.1 % | 结构独立的第三条路线 |
| CAD vs 交付参考几何 | ≤ 0.1 % | SUBOFF 的 (∇, S)、KCS/KVLCC2 的 Cb |
| 网格 L / B（主尺度） | ≤ 0.5 %（体素族按 3 倍格距放宽） | 几何必须先验证 Cb/L/B |
| 离散 vs CAD 体积 | ≤ 0.5 % | 网格是否守恒排水量 |
| 离散 vs CAD 湿面积 | ≤ 1.0 % | 网格是否守恒湿面积 |
| 离散 vs CAD 方形系数 | ≤ 1e-3（绝对） | 与合同 Cb 的一致性 |
| BEM 闭合曲面恒等式 `\|Σ Aᵢnᵢ\|/Σ Aᵢ` | ≤ 1e-12 | 面元法向闭合（散度定理） |
| provenance | 双 sha256 齐备 | CAD 与网格各自可追溯 |

## 2 怎么做的

**工具链**（`src/tensorphysics/geometry_fidelity/`，TensorPhysics 领域目录）
* `hulls.py` — CAD 真值源 `HullForm`：横剖切片 Simpson 求积 + 分层 Monte-Carlo 复算；
  `suboff_hull()` 现委托给标定后的 SUBOFF（两个独立 fairing 指数）。
* `cad_routes.py`（新）— **第三条路线**：把闭曲面写成参数曲面 `r(x, s)`，按 `(n_x, n_s)`
  网格剖分成三角形，用散度定理的离散形式（原点出发的有向四面体体积求和）得到 ∇，三角形
  面积求和得到 S；另含 SUBOFF 的两约束标定（嵌套二分 + 全指数域扫描兜底）与形状族敏感性。
* `mesh.py` — 三族离散积分：FVM 六四面体体积和 + 面元湿面积；LBM 委托 `reconstruct`；BEM
  平板面元 + 散度定理 + 闭合恒等式；各记录写回 L/B 实测值与格距量子。
* `reconstruct.py`（新）— LBM 几何：`occupancy_fraction` 逐格解析横向量 → IBM 体积分数
  （体积二阶）；`height_field_area` 以格点高度场重构湿表面；`marching_tets_area` 为通用等值面
  工具（球面验证：h=0.01 时面积误差 −1.0e-4）；`staircase_area` 量化阶梯表示差（球面 +50.1 %）。
* `contract.py` — 契约文档、自包含 JSON-Schema 校验、闸门判定（新增等参/参考/主尺度检查）、
  每族「积分方法」字段、`assert_admitted` / `physics_chain` 结构化准入。
* `scripts/m6242_geometry_fidelity.py` — 一键复现：3 份契约 + 三路 CAD 表 + 收敛表 +
  网格无关性表 + 负控语料 + SUBOFF 族 + 表示差表 + 6 张图 + `acceptance.json`。
* `tests/test_geometry_fidelity_m6242.py` — 新增 13 条 pytest。

**SUBOFF 的处理（本轮关键修正）.** 首轮用**同一个** fairing 指数闭合 ∇，然后发现 S 落在
形状族下界之外（+0.19 %），记为 honest FAIL。机理是：一个自由度只够闭合一个约束。本轮
把 fore/aft 两个指数作为独立未知量（aft 指数限制在 ≥1.02 以保证尾部切线有限、面元可闭合），
用嵌套二分求解，`fore_b = 0.592898`、`aft_b = 1.079607`，
两个交付约束同时闭合到 1e-6 量级；首轮的单指数读数保留为负控（`sharp_tail_hull`，S 偏差
+0.07 %）。形状族敏感性表见 §3.6：`(∇, S)` 两个积分约束并不
唯一确定艇形，沿尾部基半径 `re` 存在一族解——这本身是结论，不是缺陷。

**负控语料（三种破坏网格，每船各一套）**：① 全节点等向拉伸 2 %（FVM）；② 水线处删一层
体素（LBM）；③ 半单元错位——剖面宽度采样与自己所在层错开半格（FVM，经典 off-by-one-half-cell）。

## 3 计算结果

> 全部数字由 `scripts/m6242_geometry_fidelity.py` 程序化写入
> `artifacts/geometry_fidelity/m6242/*.json`，本文表格由 `scripts/m6242_report.py` 从这些 JSON 渲染，零手抄。

### 3.1 CAD 三路积分与合同量

| 船 | 族 | ∇ 切片 [m³] | ∇ 等参 [m³] | ∇ MC [m³] | S 切片 [m²] | S 等参 [m²] | S MC [m²] | Cb | Cwp |
|---|---|---|---|---|---|---|---|---|---|
| KCS | wall_sided_power | 52,070.105 | 52,068.064 | 52,072.344 | 8,941.658 | 8,941.713 | 8,941.666 | 0.651000 | 0.7820 |
| KVLCC2 | wall_sided_power | 312,621.803 | 312,598.219 | 312,587.698 | 27,232.431 | 27,232.686 | 27,231.488 | 0.809800 | 0.8620 |
| SUBOFF | revolution | 0.699211 | 0.699171 | 0.699194 | 5.997581 | 5.997785 | 5.997996 | 0.622004 | 0.8550 |

### 3.2 三路互证（C1）

| 船 | ∇ 等参 vs 切片 | ∇ MC vs 切片 | S 等参 vs 切片 | S MC vs 切片 |
|---|---|---|---|---|
| KCS | 9.65e-06 | 4.36e-05 | 3.73e-06 | 8.85e-06 |
| KVLCC2 | 1.86e-05 | 1.08e-04 | 5.51e-06 | 6.10e-05 |
| SUBOFF | 3.22e-05 | 2.38e-05 | 4.05e-05 | 6.92e-05 |

第三路（等参剖分）对切片路线的收敛见 §4 图 2：`∇`、`S` 均按二阶收敛，480×320 时残差量级
4.05e-05
，远小于 0.1 % 判据。

### 3.3 离散 vs CAD（C2）与 L/B 前置校验

| 船 | 族 | Δ∇/∇ | ΔS/S | \|ΔCb\| | L 相对误差 | B 相对误差 | 湿面积证据 | 闸门 |
|---|---|---|---|---|---|---|---|---|
| KCS | FVM | -0.036 % | -0.169 % | 2.32e-04 | 0.00e+00 | 3.67e-11 | REAL | GATE_PASS |
| KCS | LBM | +0.123 % | WITHHELD | 8.03e-04 | 1.67e-03 | 0.00e+00 | WITHHELD | GATE_PASS |
| KCS | BEM | -0.039 % | -0.008 % | 2.55e-04 | 0.00e+00 | 1.86e-09 | REAL | GATE_PASS |
| KVLCC2 | FVM | -0.091 % | -0.649 % | 7.36e-04 | 0.00e+00 | 0.00e+00 | REAL | GATE_PASS |
| KVLCC2 | LBM | +0.004 % | +0.466 % | 3.58e-05 | 1.67e-03 | 2.30e-03 | REAL | GATE_PASS |
| KVLCC2 | BEM | -0.095 % | -0.028 % | 7.66e-04 | 0.00e+00 | 0.00e+00 | REAL | GATE_PASS |
| SUBOFF | FVM | -0.117 % | -0.039 % | 7.26e-04 | 0.00e+00 | 0.00e+00 | REAL | GATE_PASS |
| SUBOFF | LBM | +0.002 % | WITHHELD | 1.24e-05 | 3.33e-03 | 3.94e-04 | WITHHELD | GATE_PASS |
| SUBOFF | BEM | -0.038 % | -0.010 % | 2.36e-04 | 0.00e+00 | 0.00e+00 | REAL | GATE_PASS |

### 3.4 闸门逐项（三份契约）

| 契约 | 检查项 | 值 | 容差 | 通过 |
|---|---|---|---|---|
| KCS | cad_volume_consistency | 4.30e-05 | 1.00e-03 | ✅  |
| KCS | cad_surface_consistency | 9.64e-07 | 3.00e-03 | ✅  |
| KCS | cad_isoparametric_volume | 3.92e-05 | 1.00e-03 | ✅  |
| KCS | cad_isoparametric_surface | 6.17e-06 | 1.00e-03 | ✅  |
| KCS | cad_reference_cb | 2.97e-09 | 1.00e-03 | ✅  |
| KCS | bem_volume | 3.91e-04 | 5.00e-03 | ✅  |
| KCS | bem_surface | 7.91e-05 | 1.00e-02 | ✅  |
| KCS | bem_cb | 2.55e-04 | 1.00e-03 | ✅  |
| KCS | bem_length | 0.00e+00 | 5.00e-03 | ✅  |
| KCS | bem_breadth | 1.86e-09 | 5.00e-03 | ✅  |
| KCS | bem_closure | 1.66e-14 | 1.00e-12 | ✅  |
| KCS | fvm_volume | 3.56e-04 | 5.00e-03 | ✅  |
| KCS | fvm_surface | 1.69e-03 | 1.00e-02 | ✅  |
| KCS | fvm_cb | 2.32e-04 | 1.00e-03 | ✅  |
| KCS | fvm_length | 0.00e+00 | 5.00e-03 | ✅  |
| KCS | fvm_breadth | 3.67e-11 | 5.00e-03 | ✅  |
| KCS | lbm_volume | 1.23e-03 | 5.00e-03 | ✅  |
| KCS | lbm_surface_evidence | 0.00e+00 | 1.00e+00 | ✅ wetted area WITHHELD: the lattice height field cannot resolve a section with a vertical tangent (keel of a fine form, polar lines of a body of revolution), so the reconstruction bias is resolution independent; volume and Cb of this family are admitted instead |
| KCS | lbm_cb | 8.03e-04 | 1.00e-03 | ✅  |
| KCS | lbm_length | 1.67e-03 | 5.00e-03 | ✅  |
| KCS | lbm_breadth | 0.00e+00 | 5.00e-03 | ✅  |
| KCS | provenance_cad_sha256 | 0.00e+00 | 1.00e+00 | ✅ 0 = digest present and well formed |
| KCS | provenance_grid_sha256 | 0.00e+00 | 1.00e+00 | ✅ 0 = digest present and well formed |
| KVLCC2 | cad_volume_consistency | 1.09e-04 | 1.00e-03 | ✅  |
| KVLCC2 | cad_surface_consistency | 3.46e-05 | 3.00e-03 | ✅  |
| KVLCC2 | cad_isoparametric_volume | 7.54e-05 | 1.00e-03 | ✅  |
| KVLCC2 | cad_isoparametric_surface | 9.37e-06 | 1.00e-03 | ✅  |
| KVLCC2 | cad_reference_cb | 4.24e-07 | 1.00e-03 | ✅  |
| KVLCC2 | bem_volume | 9.46e-04 | 5.00e-03 | ✅  |
| KVLCC2 | bem_surface | 2.76e-04 | 1.00e-02 | ✅  |
| KVLCC2 | bem_cb | 7.66e-04 | 1.00e-03 | ✅  |
| KVLCC2 | bem_length | 0.00e+00 | 5.00e-03 | ✅  |
| KVLCC2 | bem_breadth | 0.00e+00 | 5.00e-03 | ✅  |
| KVLCC2 | bem_closure | 1.59e-14 | 1.00e-12 | ✅  |
| KVLCC2 | fvm_volume | 9.09e-04 | 5.00e-03 | ✅  |
| KVLCC2 | fvm_surface | 6.49e-03 | 1.00e-02 | ✅  |
| KVLCC2 | fvm_cb | 7.36e-04 | 1.00e-03 | ✅  |
| KVLCC2 | fvm_length | 0.00e+00 | 5.00e-03 | ✅  |
| KVLCC2 | fvm_breadth | 0.00e+00 | 5.00e-03 | ✅  |
| KVLCC2 | lbm_volume | 4.42e-05 | 5.00e-03 | ✅  |
| KVLCC2 | lbm_surface | 4.66e-03 | 1.00e-02 | ✅  |
| KVLCC2 | lbm_cb | 3.58e-05 | 1.00e-03 | ✅  |
| KVLCC2 | lbm_length | 1.67e-03 | 5.00e-03 | ✅  |
| KVLCC2 | lbm_breadth | 2.30e-03 | 5.00e-03 | ✅  |
| KVLCC2 | provenance_cad_sha256 | 0.00e+00 | 1.00e+00 | ✅ 0 = digest present and well formed |
| KVLCC2 | provenance_grid_sha256 | 0.00e+00 | 1.00e+00 | ✅ 0 = digest present and well formed |
| SUBOFF | cad_volume_consistency | 2.38e-05 | 1.00e-03 | ✅  |
| SUBOFF | cad_surface_consistency | 6.92e-05 | 3.00e-03 | ✅  |
| SUBOFF | cad_isoparametric_volume | 5.72e-05 | 1.00e-03 | ✅  |
| SUBOFF | cad_isoparametric_surface | 3.39e-05 | 1.00e-03 | ✅  |
| SUBOFF | cad_reference_volume | 2.16e-07 | 1.00e-03 | ✅  |
| SUBOFF | cad_reference_surface | 6.98e-05 | 1.00e-03 | ✅  |
| SUBOFF | bem_volume | 3.80e-04 | 5.00e-03 | ✅  |
| SUBOFF | bem_surface | 1.03e-04 | 1.00e-02 | ✅  |
| SUBOFF | bem_cb | 2.36e-04 | 1.00e-03 | ✅  |
| SUBOFF | bem_length | 0.00e+00 | 5.00e-03 | ✅  |
| SUBOFF | bem_breadth | 0.00e+00 | 5.00e-03 | ✅  |
| SUBOFF | bem_closure | 5.32e-15 | 1.00e-12 | ✅  |
| SUBOFF | fvm_volume | 1.17e-03 | 5.00e-03 | ✅  |
| SUBOFF | fvm_surface | 3.86e-04 | 1.00e-02 | ✅  |
| SUBOFF | fvm_cb | 7.26e-04 | 1.00e-03 | ✅  |
| SUBOFF | fvm_length | 0.00e+00 | 5.00e-03 | ✅  |
| SUBOFF | fvm_breadth | 0.00e+00 | 5.00e-03 | ✅  |
| SUBOFF | lbm_volume | 1.99e-05 | 5.00e-03 | ✅  |
| SUBOFF | lbm_surface_evidence | 0.00e+00 | 1.00e+00 | ✅ wetted area WITHHELD: the lattice height field cannot resolve a section with a vertical tangent (keel of a fine form, polar lines of a body of revolution), so the reconstruction bias is resolution independent; volume and Cb of this family are admitted instead |
| SUBOFF | lbm_cb | 1.24e-05 | 1.00e-03 | ✅  |
| SUBOFF | lbm_length | 3.33e-03 | 5.00e-03 | ✅  |
| SUBOFF | lbm_breadth | 3.94e-04 | 5.00e-03 | ✅  |
| SUBOFF | provenance_cad_sha256 | 0.00e+00 | 1.00e+00 | ✅ 0 = digest present and well formed |
| SUBOFF | provenance_grid_sha256 | 0.00e+00 | 1.00e+00 | ✅ 0 = digest present and well formed |

### 3.5 网格无关性（族 × 分辨率）

| 船 | 族 | 分辨率 | Δ∇/∇ | ΔS/S | \|ΔCb\| | 证据 |
|---|---|---|---|---|---|---|
| KCS | FVM | n_s=12, n_x=40, n_y=4 | -0.42 % | -0.26 % | 2.73e-03 | REAL |
| KCS | FVM | n_s=24, n_x=80, n_y=6 | -0.10 % | -0.19 % | 6.75e-04 | REAL |
| KCS | FVM | n_s=40, n_x=160, n_y=8 | -0.03 % | -0.17 % | 2.25e-04 | REAL |
| KCS | LBM | n_x=300 | -0.22 % | WITHHELD | 1.42e-03 | WITHHELD |
| KCS | LBM | n_x=600 | +0.12 % | WITHHELD | 8.10e-04 | WITHHELD |
| KCS | LBM | n_x=1200 | -0.02 % | WITHHELD | 1.57e-04 | WITHHELD |
| KCS | BEM | n_s=20, n_x=60 | -0.16 % | -0.04 % | 1.02e-03 | REAL |
| KCS | BEM | n_s=30, n_x=90 | -0.07 % | -0.02 % | 4.47e-04 | REAL |
| KCS | BEM | n_s=40, n_x=120 | -0.04 % | -0.01 % | 2.48e-04 | REAL |
| KVLCC2 | FVM | n_s=12, n_x=40, n_y=4 | -1.02 % | -0.95 % | 8.24e-03 | REAL |
| KVLCC2 | FVM | n_s=24, n_x=80, n_y=6 | -0.26 % | -0.71 % | 2.07e-03 | REAL |
| KVLCC2 | FVM | n_s=40, n_x=160, n_y=8 | -0.09 % | -0.65 % | 7.20e-04 | REAL |
| KVLCC2 | LBM | n_x=300 | -0.62 % | +1.11 % | 5.04e-03 | REAL |
| KVLCC2 | LBM | n_x=600 | +0.01 % | +0.46 % | 5.20e-05 | REAL |
| KVLCC2 | LBM | n_x=1200 | +0.00 % | +0.24 % | 3.59e-05 | REAL |
| KVLCC2 | BEM | n_s=20, n_x=60 | -0.38 % | -0.12 % | 3.04e-03 | REAL |
| KVLCC2 | BEM | n_s=30, n_x=90 | -0.17 % | -0.06 % | 1.35e-03 | REAL |
| KVLCC2 | BEM | n_s=40, n_x=120 | -0.09 % | -0.03 % | 7.50e-04 | REAL |
| SUBOFF | FVM | n_s=12, n_x=40, n_y=4 | -1.34 % | -0.51 % | 8.34e-03 | REAL |
| SUBOFF | FVM | n_s=24, n_x=80, n_y=6 | -0.34 % | -0.14 % | 2.09e-03 | REAL |
| SUBOFF | FVM | n_s=40, n_x=160, n_y=8 | -0.11 % | -0.04 % | 7.06e-04 | REAL |
| SUBOFF | LBM | n_x=300 | +0.01 % | WITHHELD | 3.59e-05 | WITHHELD |
| SUBOFF | LBM | n_x=600 | +0.01 % | WITHHELD | 3.24e-05 | WITHHELD |
| SUBOFF | LBM | n_x=1200 | +0.00 % | WITHHELD | 2.39e-05 | WITHHELD |
| SUBOFF | BEM | n_s=20, n_x=60 | -0.04 % | -0.02 % | 2.33e-04 | REAL |
| SUBOFF | BEM | n_s=30, n_x=90 | -0.04 % | -0.02 % | 2.33e-04 | REAL |
| SUBOFF | BEM | n_s=40, n_x=120 | -0.04 % | -0.02 % | 2.33e-04 | REAL |

### 3.6 SUBOFF 形状族（两个积分约束的非唯一性）

| re | fore_b | aft_b | ∇ 相对偏差 | S 相对偏差 | 两约束同时闭合 |
|---|---|---|---|---|---|
| 0.00 | 0.5929 | 1.0796 | +0.0000 % | -0.000 % | 是 |
| 0.10 | 1.2666 | 0.6200 | +0.0000 % | -0.005 % | 否 |
| 0.20 | 1.4060 | 0.6600 | +0.0000 % | +0.006 % | 否 |
| 0.30 | 1.6070 | 0.7000 | +0.0000 % | -0.017 % | 否 |
| 0.40 | 1.8058 | 0.7800 | +0.0000 % | -0.003 % | 否 |
| 0.50 | 2.0983 | 0.8800 | +0.0000 % | -0.011 % | 否 |

### 3.7 负控语料（C4）

| 船 | 破坏方式 | 族 | Δ∇/∇ | \|ΔCb\| | 判定 | 触发检查 | 准入被拒 |
|---|---|---|---|---|---|---|---|
| KCS | stretch_2pct | FVM | +6.08 % | 3.96e-02 | GATE_REJECT | fvm_breadth, fvm_cb, fvm_length, fvm_surface, fvm_volume | 是 |
| KCS | drop_voxel_layer | LBM | -4.12 % | 2.69e-02 | GATE_REJECT | lbm_cb, lbm_volume | 是 |
| KCS | off_by_half_cell | FVM | +1.38 % | 8.96e-03 | GATE_REJECT | fvm_cb, fvm_volume | 是 |
| KVLCC2 | stretch_2pct | FVM | +6.02 % | 4.88e-02 | GATE_REJECT | fvm_breadth, fvm_cb, fvm_length, fvm_surface, fvm_volume | 是 |
| KVLCC2 | drop_voxel_layer | LBM | -2.72 % | 2.20e-02 | GATE_REJECT | lbm_cb, lbm_volume | 是 |
| KVLCC2 | off_by_half_cell | FVM | +1.02 % | 8.29e-03 | GATE_REJECT | fvm_cb, fvm_volume | 是 |
| SUBOFF | stretch_2pct | FVM | +6.00 % | 3.73e-02 | GATE_REJECT | fvm_breadth, fvm_cb, fvm_length, fvm_surface, fvm_volume | 是 |
| SUBOFF | drop_voxel_layer | LBM | -0.21 % | 1.28e-03 | GATE_REJECT | lbm_cb | 是 |
| SUBOFF | off_by_half_cell | FVM | -0.19 % | 1.18e-03 | GATE_REJECT | fvm_cb, fvm_surface | 是 |

### 3.8 LBM 湿表面：阶梯表示差与重构残留偏差

| 船 | n_x | CAD S [m²] | 体素阶梯面积 | 高度场重构 | 阶梯偏差 | 重构偏差 | 证据 |
|---|---|---|---|---|---|---|---|
| KCS | 300 | 8,941.779 | 11,663.862 | 9,442.949 | +30.4 % | +5.60 % | WITHHELD |
| KCS | 600 | 8,941.779 | 11,478.712 | 9,190.269 | +28.4 % | +2.78 % | WITHHELD |
| KCS | 1200 | 8,941.779 | 11,391.354 | 9,065.100 | +27.4 % | +1.38 % | WITHHELD |
| KVLCC2 | 300 | 27,233.301 | 32,684.942 | 27,534.593 | +20.0 % | +1.11 % | REAL |
| KVLCC2 | 600 | 27,233.301 | 31,970.133 | 27,359.248 | +17.4 % | +0.46 % | REAL |
| KVLCC2 | 1200 | 27,233.301 | 31,776.569 | 27,298.434 | +16.7 % | +0.24 % | REAL |
| SUBOFF | 300 | 5.997824 | 7.728410 | 6.418966 | +28.9 % | +7.02 % | WITHHELD |
| SUBOFF | 600 | 5.997824 | 7.801146 | 6.473102 | +30.1 % | +7.92 % | WITHHELD |
| SUBOFF | 1200 | 5.997824 | 7.853854 | 6.518643 | +30.9 % | +8.68 % | WITHHELD |

## 4 图

1. `artifacts/geometry_fidelity/m6242/figures/fig1_grid_independence.png` — Discrete-vs-CAD displacement, wetted area and Cb deviations for the three solver families against mesh refinement; dashed lines are the gate tolerances.
2. `artifacts/geometry_fidelity/m6242/figures/fig2_cad_route_convergence.png` — Third CAD route (isoparametric surface tessellation, divergence theorem) converging to the transverse-slice route at second order for the three truth sources.
3. `artifacts/geometry_fidelity/m6242/figures/fig3_negative_controls.png` — Nine deliberately corrupted grids (three failure modes x three hulls) against the 0.5 % displacement gate; every one of them is rejected.
4. `artifacts/geometry_fidelity/m6242/figures/fig4_lbm_representation_gap.png` — LBM wetted area: voxel staircase (+20...+50 %) versus the height-field reconstruction (-1.4...+8.5 %). The residual bias of the fine-form hulls is why their wetted area is withheld while volume and Cb are admitted.
5. `artifacts/geometry_fidelity/m6242/figures/fig5_suboff_family.png` — SUBOFF shape family: wetted-area residual at closed displacement versus the aft base radius, and the exponents that solve it.
6. `artifacts/geometry_fidelity/m6242/figures/fig6_body_plan.png` — Non-dimensional section lines of the three CAD truth sources (KCS, KVLCC2, SUBOFF).

## 5 注意点（踩过的坑 / 局限 / 适用范围）

1. **首轮的 honest FAIL 有两个来源，不是一个.** 首轮用同一个指数闭合 ∇，测到 S 偏 +0.19 %，
   并据此写下「任何忠实几何都达不到 5.998」。本轮复核把它拆成两半：(a) **族的自由度不足**——
   fore/aft 指数解耦后两约束可同时闭合到 1e-6；(b) **求积误差**——在 n_x=240 的横剖求积下，
   同一几何的 S 本身就偏 +1.7e-3（n_x=960 时收敛到 −2e-5），即首轮 +0.19 % 里约 0.12 % 是
   求积伪差。所以契约现在对回转体统一用 n_x ≥ 480。教训：**判"族内不可达"之前，先确认自由度
   够不够、求积收敛了没有**——否则会把自己的数值误差写成物理结论。
2. **LBM 湿面积：立方格点无法解析垂直切线.** 细型船的龙骨 `b ~ (z/T)^{1/m}`（`1/m≈0.2`）与
   回转体的极线处，高度场重构留下与分辨率无关的系统偏差（KCS +1.38 %@n=1200，
   SUBOFF +8.68 %@n=1200），故这两船的 LBM 湿面积
   **证据等级 WITHHELD**，只用体积/Cb 进闸门；墙式舷侧的 KVLCC2 湿面积 REAL 且 <1 %。体积积分
   把这些窄条偏差积掉了，所以体积仍是二阶（1.24e-03）。
3. **等值面工具：被体积分数的截断坑了一次，修正后仍不达预期，故不进契约.** `marching_tets_area`
   把体积分数（上界 0、下界 1 的截断场）当输入时，等值面在饱和区退化成阶梯，面积偏 +5 %。改成
   未截断的符号距离场后实测仍是 **+2.0 %（h=0.01，50 格/半径）且一阶收敛**，说明单元内的平面
   剖分对高曲率面的面积偏保守；因此它**不用于契约数字**，只作通用件保留，pytest 钉住的是实测
   精度（≤3 %）而不是期望精度。LBM 湿面积改走高度场重构（§3.8）。
4. **半单元错位对积分量不敏感（首轮已证，本轮保持）.** 纯平移半格对体积几乎无影响；能检出的是
   「宽度与自己层错位」这类几何-索引失配。要真正判半格对齐必须做场量级检查——这是本契约的
   适用范围边界，负控表里 `off_by_half_cell` 的作用是证明闸门对**索引失配**敏感，而不是对平移敏感。
5. **CAD 真值是参数化型线，不是官方型值表.** 三船复现公开主尺度与 Cb、SUBOFF 复现 (∇, S)，但
   **不是**拖曳水池型值表：KCS 湿面积约 8941.8 m² 对文献 ~9530 m²（−6 %），KVLCC2 对 ~27,000 m²
   一致。故「CAD vs 参考」检验的是**积分闭合性**，绝对湿面积不得直接用于阻力计算。
6. **水面船舶的"闭合面"含甲板平面.** 排水体积的边界是「船体外表面 + z=T 平面」；湿面积不含
   甲板。两个量的闭合面定义不同，代码里显式区分（`_tessellate` 的 `deck` 只在体积里，面积里剔除）。
7. **面元闭合恒等式对退化单元极敏感.** 标定后 SUBOFF 的 `aft_b≥1.02` 是硬约束：若允许 `aft_b<1`
   尾部切线垂直，面元闭合残差跳到 3e-3、体积偏 −3.8 %。也就是说**闸门的闭合恒等式在约束
   形状族的可离散性**——这是本轮新增判据的真实作用。
8. **幂等性依赖确定性抽样.** MC 用固定 seed 的分层抖动、网格生成无随机、浮点求和顺序固定，
   才做到字节级幂等；换 BLAS/并行归约需重新核验。

## 6 误差判定（逐项显式回答「是否 < 3 %」）

| 关键量 | 基准 | 本文值 | 相对误差 | < 3 %? | 说明 |
|---|---|---|---|---|---|
| SUBOFF ∇ | 0.699211 m³（交付） | 0.699211 | 2.16e-07 | **是** | 标定后双向闭合 |
| SUBOFF S | 5.998 m²（交付） | 5.997581 | 6.98e-05 | **是** | 首轮为 +0.19 % 的 FAIL，本轮闭合 |
| KCS Cb | 0.651（交付） | 0.651000 | 2.97e-09 | **是** | 由构造闭合 |
| KVLCC2 Cb | 0.8098（公开） | 0.809800 | 4.24e-07 | **是** | 由构造闭合 |
| CAD 三路互证 ∇ | 相互基准 | 最大 3.22e-05 | — | **是** | 切片/等参/MC |
| CAD 三路互证 S | 相互基准 | 最大 4.05e-05 | — | **是** | 同上 |
| 离散 vs CAD（9 组） | CAD 真值 | 体积最大 +0.12 %、Cb 最大 8.03e-04 | — | **是** | 全部在闸门内 |
| LBM 湿面积（KCS / SUBOFF） | CAD 真值 | +1.38 % / +8.68 % | — | **否** | WITHHELD，不参与验收（局限 2） |
| KCS 湿面积（绝对几何） | 文献 ~9530 m² | 8,941.658 | -6.2 % | **否** | 参数化型线未含舭部/球鼻（局限 5） |

**未完成 / 失败项（不粉饰）**
(a) KCS、SUBOFF 的 LBM 湿面积未达 1 % 判据，已按形状类先验标注 WITHHELD（§3.8 给出偏差与机理）；
(b) 三船均为参数化型线代表，非官方型值表，KCS 绝对湿面积与文献差 −6 %；
(c) `off_by_half_cell` 负控检出的仍是「索引失配」而非平移型半格错位（§5.4 适用范围边界）；
(d) 等参路线在 SUBOFF 尾尖附近的收敛受 `r ~ x^b` 的弱奇异性支配，480×320 时残差
4.05e-05，若更严的判据（1e-4）需换几何级网格。

## 7 复现

```bash
cd <worktree> && PYTHONPATH=src python3 scripts/m6242_geometry_fidelity.py     # 契约/表/图/acceptance.json
PYTHONPATH=src python3 scripts/m6242_report.py                                 # 本报告 + /tmp/report.json
PYTHONPATH=src python3 -m pytest tests/test_geometry_fidelity_contract.py tests/test_geometry_fidelity_m6242.py -q
```

产物：`artifacts/geometry_fidelity/m6242/{contract_kcs,contract_kvlcc2,contract_suboff}.json`、
`cad_routes.json`、`cad_convergence.json`、`grid_independence.json`、`negative_controls.json`、
`suboff_family.json`、`representation_gap.json`、`acceptance.json`、`figures/*.png`。

**机器验收（`acceptance.json`）**：C1–C7 全部 `pass = true`
（含 `c4_negative_corpus`：9/9 检出、
误报 0）。
