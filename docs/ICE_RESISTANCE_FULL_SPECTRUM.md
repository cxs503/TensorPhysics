# 冰阻力全谱与速度-功率曲线（IN10-C1 / mission 4911）

> 三态阻力模型族：平整冰 Lindqvist 三分量 · 碎冰覆盖衰减律 · 冰脊冲撞经验项（Keinonen 锚）。
> 工件：`artifacts/vp_curves.json`（PC6 全冰况速度-功率矩阵）、`artifacts/ice_resistance/validation_report.json`（锚语料判定 + 误差表）、`src/tensorphysics/ice/resistance.py`（模型实现）、`benchmarks/ice_resistance/`（语料 + 评测器 + 运行器）、`tests/test_ice_resistance.py`（pytest 判据）。

生成时间（UTC）：`2026-11-06T23:19:21Z`　语料版本：`1.0.0`　证据级别：`L2`

## 1. 案例介绍（物理背景 / 几何 / 参数）

极区运输船的总体性能谱由三种冰况构成，三者的阻力物理机制不同，不能用一条经验律覆盖：

| 冰况 | 物理机制 | 主导量 |
|---|---|---|
| 平整冰 level ice | 冰缘在艏部被准静态压溃 + 冰尖弯曲断裂 + 破冰下潜/摩擦 | `σ_c h²`、`σ_f h B`、`ρ V² h B` |
| 碎冰 brash / 部分覆盖 | 浮冰清道 + 堆积挤压；阻力随覆盖率衰减，并在覆盖率 ~0.6 以上陡增 | 覆盖率 `c`、`D(c)` |
| 冰脊 ridge | 冲撞排开龙骨固体冰体积（浮力做功）+ 碎块动量通量 | `(1-e) ρ g h_k² B`、`(1-e) ρ h_k B V²` |

参考船（Polar Class 6 量级，几何/功率为公开设计量级，非某条实船）：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| 垂线间长 `L_pp` | 150.0 m | 型宽 `B` | 24.0 m |
| 吃水 `T` | 8.5 m | 排水量 | 21000 t |
| 水线面艏半角 `φ` | 30.0° | 艏部外飘角 `α` | 25.0° |
| 装机交付功率 | 22.0 MW | 总推进效率 `η_D` | 0.55 |

冰况谱：平整冰 `h = 0.3/0.5/0.7/0.9/1.2 m`；碎冰 `h = 0.6 m, c = 0.3/0.6/0.9`；冰脊 `h_k = 1.5/2.5/3.5 m, 间距 200 m, 背景平整冰 0.5 m, 孔隙率 e = 0.30`；航速网格 0.5–5.0 m/s（1–9.7 kn），共 132 个速度-阻力-功率点。

## 2. 怎么做的（方法 / 关键设置 / 工具链）

**实现**：`src/tensorphysics/ice/resistance.py`，纯闭式、组件可分解、无随机项、无代理模型。

### 2.1 平整冰：Lindqvist (1989) 三分量

```text
R = R_crush + R_break + R_submersion
R_crush      = 0.5 σ_c h² (tan α + μ/cos α)                     α = 艏部外飘角（水平面起算）
R_break      = 0.5 σ_f h B (sin(φ/2) + μ cos(φ/2)) / k_bend     k_bend = 2.0（冰尖长度 ~ k_bend·h）
R_submersion = (0.15 + 0.3 Fr_h) ρ_i V² h B                     Fr_h = V/√(gh)
```

**关键修正（审计发现）**：仓内既有转写 `ice_anchor/anchors.py:lindqvist_1989` 把*水线面*艏半角当成*板面倾角*用（`α = 90° − φ/2`，30° 艏半角 → 75°），使 `tan α` 从 0.47 放大到 3.73，压溃分量单独就给出 10779 kN、总阻力 10907 kN（相对两条独立文献律均值 +271%）。本实现改用外飘角 `α = 25°`，同一工况总阻力 2876 kN（-2.1%）。同一约定错误也存在于 `riska_1997`（本次仅记录，不改动冻结范围外的模块）。

### 2.2 碎冰：覆盖衰减律 `D(c)`

```text
R_brash(c) = D(c)·R_level(h,V) + 4c(1-c)·R_floe_impact
D(c) = c                                   (c ≤ c_crit = 0.6，自由清道区)
D(c) = c + (1-c)·A·w(c),  w = ((c-0.6)/0.4)²  (c > 0.6，堆积区，A = 1)
```
端点严格：`D(0)=0`、`D(1)=1`（c=1 严格退化为同厚度平整冰），`0 ≤ D(c) ≤ 1` 且单调不减；堆积区局部斜率 > 1 反映覆盖率 0.6 以上通道无法清空、阻力陡增。

### 2.3 冰脊：Keinonen 锚冲撞经验项 + 能量摊薄

```text
R_ridge = C1(1-e)ρ_i g h_k² B + C2(1-e)ρ_i h_k B V²      C1=0.35, C2=0.5
W(h_k)  = C1(1-e)ρ_i g B h_k³/3 + C2(1-e)ρ_i B V² h_k²/2   （穿脊做功，闭式）
R_field(s) = R_level(h_bg) + W(h_k)/s                     （间距 s 摊薄成场平均）
```

### 2.4 适用域（拒绝外推）

| 限制 | 值 | 越界处理 |
|---|---|---|
| 冰厚 | `h ≤ min(0.03L, 0.10B) = 2.40 m`，且 `h ≥ 0.05 m` | strict 抛 `IceDomainError`；lenient 截断并打 `H_TRUNCATED` 标志 |
| 航速 | `Fr_L ≤ 0.25` → `V ≤ 9.59 m/s` | 同上（`V_TRUNCATED`） |
| 覆盖率 | `c ∈ [0, 1]` | 抛异常 |
| 冰脊龙骨 | `h_k ≤ min(0.15B, 4.0 m) = 3.60 m` | 同上（`HK_TRUNCATED`） |
| 冰脊孔隙率 | `e ∈ [0.05, 0.6]`（实测区间） | 抛异常 |

**工具链**：`benchmarks/ice_resistance/anchors_eval.py`（锚判据唯一实现）→ `run_ice_resistance.py`（工件 + 图）→ `write_docs.py`（本文档，数字全部程序注入）→ `tests/test_ice_resistance.py`（pytest）。跨定律参考直接用仓内 `ice_anchor/anchors.py` 的独立转写。

## 3. 计算结果

### 3.1 锚语料判定

- 参考锚：**25/25 全过**（判据要求 ≥10）
- 负例（域外拒绝/如实截断）：**7/7 全过**
- 总判定：`all_passed = True`

| 锚 | 类型 | 判据 | 容差 | 实测误差 | 结果 |
|---|---|---|---|---|---|
| `IR-L01` | identity | R_total = R_crush + R_break + R_submersion（Lindqvist | 1e-12 | 0.00e+00 | PASS |
| `IR-L02` | analytic_property | 压溃分量与航速无关（半静态压溃），V=0.5 与 V=5 m/s 的 R_crush 相等 | 1e-12 | 0.00e+00 | PASS |
| `IR-L03` | analytic_derivative | R_submersion 对数-对数斜率 = (2+3x)/(1+x)，x = 2·Fr_h（R_s ∝ | 1e-06 | 1.51e-12 | PASS |
| `IR-L04` | scaling_law | R_crush ∝ sigma_c·h²（对 sigma_c 的一次、对 h 的二次标度） | 1e-12 | 0.00e+00 | PASS |
| `IR-L05` | scaling_law | R_break ∝ sigma_f·h·B（冰尖弯曲分量对标度各一次） | 1e-12 | 0.00e+00 | PASS |
| `IR-L06` | physics_sign | R_total 对 h、V、B、phi（艏越钝）、alpha（外飘越陡）、mu 均单调递增 | 0.0 | 0.00e+00 | PASS |
| `IR-L07` | cross_law | 在 h={0.5,0.75,1.0} m、V=3 m/s 时本模型总阻力落在两条独立文献律（Keinon | 2e-01 | 1.09e-01 | PASS |
| `IR-L08` | order_of_magnitude | 1.0 m 平整冰、5 kn 时所需交付功率落在 5–20 MW（冰级船实船推进量级带） | 0.0 | 0.00e+00 | PASS |
| `IR-L09` | domain | h = 0.999·h_limit（h_limit = min(0.03L, 0.10B)）在 stri | 1e-12 | 0.00e+00 | PASS |
| `IR-B01` | endpoint | 覆盖率 c=1 时碎冰阻力严格退化为同厚度平整冰总阻力 | 1e-12 | 0.00e+00 | PASS |
| `IR-B02` | endpoint | 覆盖率 c=0 时冰阻力严格为 0（开水域） | 1e-09 | 0.00e+00 | PASS |
| `IR-B03` | decay_law | 清除区（c ≤ 0.6）衰减因子 D(c)=c 精确线性 | 1e-12 | 0.00e+00 | PASS |
| `IR-B04` | decay_law | 衰减律 D(c) 在 [0,1] 上单调不减（101 点网格） | 0.0 | 0.00e+00 | PASS |
| `IR-B05` | physics_bound | 对任意 c<1，碎冰阻力 < 同厚度平整冰阻力（D(c) ≤ 1） | 0.0 | 0.00e+00 | PASS |
| `IR-B06` | decay_law | 堆积区（c>0.6）衰减律局部斜率 > 1：(D(0.9)-D(0.6))/0.3 ≥ 1.05 | 5e-02 | 0.00e+00 | PASS |
| `IR-R01` | scaling_law | 冰脊浮力排开项 R_keel ∝ h_k²（R(2h_k)/R(h_k)=4），碎块动量项 ∝ h_k（ | 1e-12 | 0.00e+00 | PASS |
| `IR-R02` | scaling_law | 冰脊阻力 ∝ (1-e)（孔隙率 e 越大固体冰越少、阻力越小） | 1e-12 | 1.48e-16 | PASS |
| `IR-R03` | scaling_law | 冰脊阻力 ∝ B（宽度） | 1e-12 | 0.00e+00 | PASS |
| `IR-R04` | scaling_law | 碎块动量项 ∝ V²（速度加倍该项 4 倍） | 1e-12 | 0.00e+00 | PASS |
| `IR-R05` | identity | 穿脊功 W(h_k)=∫R dh_k 的中点求积与闭式解相对差 < 1e-4 | 1e-04 | 4.63e-08 | PASS |
| `IR-R06` | identity | 冰脊场平均阻力 R_mean = R_level(1-w_r/s) + R_ridge(w_r/s) 恒 | 1e-12 | 0.00e+00 | PASS |
| `IR-R07` | physics_sign | 冰脊场平均阻力随间距 s 单调递减（h_k=3.5 m、背景平整冰 0.2 m、s=100/200/40 | 0.0 | 0.00e+00 | PASS |
| `IR-P01` | identity | 交付功率恒等式 P_D = R·V/eta_D 及其逆运算 R = P_D·eta_D/V | 1e-12 | 0.00e+00 | PASS |
| `IR-P02` | numerical | speed_for_power(22 MW) 求得的 V* 代回功率链，残差 |P(V*)-P_targ | 1e-08 | 6.44e-12 | PASS |
| `IR-P03` | physics_sign | P_D(V) 在域内单调递增（网格 0.5–5 m/s） | 0.0 | 0.00e+00 | PASS |

负例：

| 负例 | 内容 | 结果 |
|---|---|---|
| `IR-N01` | h=5.0 m > 0.03L(=4.5 m)（且 > 0.10B）时 strict 模式必须拒绝外推 | PASS（正确拒绝/截断） |
| `IR-N02` | V=12 m/s 超 Fr_L ≤ 0.25 上限（v_limit≈9.59 m/s）必须拒绝 | PASS（正确拒绝/截断） |
| `IR-N03` | 覆盖率 c=1.35 越界必须拒绝 | PASS（正确拒绝/截断） |
| `IR-N04` | 冰脊龙骨 6.0 m > min(0.15B, 4.0 m)=3.6 m 时 strict 必须拒绝 | PASS（正确拒绝/截断） |
| `IR-N05` | 冰厚 h=-0.1 m 负值必须拒绝 | PASS（正确拒绝/截断） |
| `IR-N06` | 冰脊孔隙率 e=0.9 超出实测区间 [0.05,0.60] 必须拒绝 | PASS（正确拒绝/截断） |
| `IR-N07` | 非 strict 模式对 h=5.0 m 必须如实截断到 h_limit 并打 H_TRUNCATED 标志（禁 | PASS（正确拒绝/截断） |

### 3.2 跨定律与实尺度量级对照

| 模型 | h=1 m, V=3 m/s 平整冰阻力 [kN] | 相对 Keinonen/Enkvist 均值 |
|---|---|---|
| 本模型（三分量族） | 2876 | -2.1% |
| Keinonen (1983) | 2998 | +2.1% |
| Enkvist (1972) | 2877 | -2.1% |
| 仓内旧转写 Lindqvist（A1，`α=90°−φ/2`） | 10907 | +271.3% |
| 仓内旧转写 Riska（A5） | 8714 | +196.6% |

### 3.3 PC6 速度-功率表（摘要）

| 冰况 | R@3 kn [kN] | R@5 kn [kN] | P_D@5 kn [MW] | 22 MW 可达航速 [kn] |
|---|---|---|---|---|
| level ice h=0.30 m | 512 | 531 | 2.5 | 17.20 |
| level ice h=0.50 m | 1009 | 1036 | 4.8 | 14.20 |
| level ice h=0.70 m | 1633 | 1666 | 7.8 | 11.45 |
| level ice h=0.90 m | 2383 | 2422 | 11.3 | 8.96 |
| level ice h=1.20 m | 3744 | 3792 | 17.7 | 6.13 |
| brash ice h=0.60 m, coverage c=0.30 | 398 | 419 | 2.0 | 18.64（受适用域封顶） |
| brash ice h=0.60 m, coverage c=0.60 | 790 | 822 | 3.8 | 15.41 |
| brash ice h=0.60 m, coverage c=0.90 | 1251 | 1285 | 6.0 | 12.91 |
| ridge field keel h_k=1.50 m, spacing 200 m, background level ice 0.30 m † | 142 | 188 | 0.9 | 17.24 |
| ridge field keel h_k=2.50 m, spacing 200 m, background level ice 0.30 m † | 367 | 442 | 2.0 | 17.13 |
| ridge field keel h_k=3.50 m, spacing 200 m, background level ice 0.30 m † | 695 | 801 | 3.6 | 16.90 |

> † 冰脊三行为**脊内瞬时穿透阻力/瞬时功率**（载荷口径）：航程加权后的场平均阻力由背景平整冰主导，故另列场平均口径如下。冰脊的危害主要是局部峰值载荷，载荷设计应使用瞬时口径。

| 冰脊场（h_k） | 场平均 R@5 kn [kN] | 脊内瞬时 R@5 kn [kN] | 脊内瞬时 P_D@5 kn [MW] |
|---|---|---|---|
| ridge field keel h_k=1.50 m, spacing 200 m, background level ice 0.30 m | 522 | 188 | 0.9 |
| ridge field keel h_k=2.50 m, spacing 200 m, background level ice 0.30 m | 526 | 442 | 2.0 |
| ridge field keel h_k=3.50 m, spacing 200 m, background level ice 0.30 m | 544 | 801 | 3.6 |

完整矩阵（12 档航速 × 11 种冰况，含分量分解与域标志）见 `artifacts/vp_curves.json`。

冲撞容量（能量法，含 10% 附加质量）：

| 冲撞速度 [kn] | 动能 [MJ] | 可穿透龙骨 [m] |
|---|---|---|
| 1.94 | 11.6 | 3.60 |
| 3.89 | 46.2 | 3.60 |
| 5.83 | 104.0 | 3.60 |

## 4. 结果图

| 图 | 内容 |
|---|---|
| `artifacts/ice_resistance/fig1_resistance_spectrum.png` | 11 条冰况的阻力-航速族（三态同图） |
| `artifacts/ice_resistance/fig2_power_spectrum.png` | 交付功率-航速族 + 22 MW 装机线（速度-功率曲线） |
| `artifacts/ice_resistance/fig3_coverage_decay.png` | 覆盖衰减律 `D(c)` 与碎冰阻力 vs 覆盖率 |
| `artifacts/ice_resistance/fig4_audit_deviation.png` | 审计：旧转写 vs 修正族 vs 两条文献律（±25% 带） |

## 5. 注意点（坑 / 局限 / 适用范围）

- 绝对精度 vs 实尺度测量 = WITHHELD：仓内无数字化实尺度冰阻力记录；本报告只给出解析恒等式、已发表定律标度关系、跨定律一致性与推进功率量级带。
- Lindqvist 三分量中压溃项与船体宽度无关（σ_c h² 结构），这是原式的特性；跨定律一致区间接吸收了这一点。
- k_bend 闭合常数按跨定律量级标定（非实验标定），其 ±50% 影响已在 uncertainty 中给出。
- 冰脊项为半静态穿透阻力，动态冲撞峰值可比其高 2-4 倍；能量法 ram 容量忽略推力限制，因此 ram 容量全部被适用域上限封顶（能量 12-104 MJ 远大于准静态穿脊功 0.2-1.0 MJ），真实冲撞受推力/动态堆积压力限制，本模型不覆盖。
- 冰脊场平均阻力按航程加权，可能低于同背景厚度的平整冰阻力（准静态口径）；冰脊的真实危害是局部峰值载荷与卡船，属峰值/动力问题，不在本阻力谱口径内。
- 碎冰衰减律的 c_crit=0.6 与堆积闭合系数 A=1 为工程值，无本仓数据标定。

**过程中踩到的坑**：

1. 幂律斜率锚的解析式一开始写错（误用 `2 + x/(0.5+x)`），有限差分给 2.657、解析给 2.365，锚报 FAIL——正是判据该起的作用：解析导数应为 `(2+3x)/(1+x)`，`x = 2Fr_h`。
2. 冰脊阻力总项不是单一幂律：浮力项 ∝ `h_k²`、动量项 ∝ `h_k`，最初用总阻力验证 `h_k²` 得到 14% 偏差；锚改为分项验证（`R_keel` 4 倍、`R_moment` 2 倍）。
3. 碎冰 c=0 端点最初不为 0（浮冰撞击项权重用 `(1-c)`），改为 `4c(1-c)` 后端点严格为 0。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 三分量叠加恒等式 | 定义式 | 0（机器精度） | 是 |
| 压溃项速度无关性 | Lindqvist 解析性质 | 0（机器精度） | 是 |
| 下潜项对数斜率 | 解析导数 | 1.5e-12 | 是 |
| 压溃/弯曲项标度 | 解析指数 | 0（机器精度） | 是 |
| 覆盖衰减律端点与线性区 | 解析端点 | 0（机器精度） | 是 |
| 冰脊分项标度 / 穿脊功积分 | 解析式 | 4.6e-08 | 是 |
| 功率恒等式/反解残差 | P=R·V/η_D | 6.4e-12 | 是 |
| 1 m 平整冰 h/V=3 m/s 总阻力 | Keinonen/Enkvist 均值 | -2.1% | **否**（文献半经验律间散布 ±25% 量级，非实现误差） |
| 1 m 冰 5 kn 交付功率 | 冰级船实船量级带 5–20 MW | 落在带内 | 不适用（区间判据） |

**结论**：所有*解析类*判据（恒等式、解析导数、标度律、端点、功率自洽）误差在机器精度～1e-6 量级，严格 <3%。**绝对精度 vs 实尺度测量不可判定（WITHHELD）**：仓内没有可用的实尺度冰阻力数字化记录，因此不宣称 <3% 的实尺度精度；跨文献半经验律一致性为 ±11%（本工况），属文献律固有散布范围而非本实现误差。`k_bend` 闭合常数按跨定律量级标定（非实验标定），其杠杆见下表。

闭合常数敏感性（h=1 m, V=3 m/s，各 ±50%）：

| 常数 | +50% 时 R_total 相对变化 | −50% 时 R_total 相对变化 |
|---|---|---|
| `sigma_c` | +27.5% | -27.5% |
| `sigma_f` | +21.1% | -21.1% |
| `k_bend` | -14.0% | +42.1% |
| `mu` | +14.8% | -14.8% |

参数来源与证据等级见 `artifacts/vp_curves.json: parameter_provenance`（LITERATURE-INTERVAL / SEMI-EMPIRICAL 逐项标注）。

