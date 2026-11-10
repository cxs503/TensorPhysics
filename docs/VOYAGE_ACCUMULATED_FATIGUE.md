# 航次累积疲劳损伤评估：从单窗口到全航次 Miner 累积、剩余寿命与冰致振动疲劳

**任务** m9281 · 节点 `p257w2-n10-voyage-accumulated-fatigue` · 仓库 TensorPhysics
**模块** `tensorphysics.voyage_fatigue` · **基准** `benchmarks/pending/voyage_fatigue/`
**测试** `tests/test_voyage_fatigue.py`（26 项，全绿）
**provenance** git `72ab09f`，Python 3.11.6，NumPy 2.4.6，
结果由 `run_voyage_fatigue.py` 程序化生成（`results.json` + sha256 记入 `figures[]`），本文所有数字均自该 JSON 插值，零手抄。

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

冰区航行时船体梁承受冰载荷脉冲串激励，应力循环沿整个航次累积。m7698 只做了**单窗口**评估
（一段应力时程 → 一次雨流计数 → 一个损伤数），无法回答三个工程问题：(i) 一个**航次**累积多少损伤；
(ii) 其中**冰致振动**贡献多少；(iii) 按当前损伤还剩多少**航次 / 运营年限**。本节点把链条补全。

**船体（几何先验证 Cb/L/B）**

| 量 | 值 |
|---|---|
| L × B × T | 150 m × 25 m × 9.0 m |
| Cb（声明 / 由排水量反算 / 相对误差 / 容差） | 0.75 / 0.7500 / 0.000 % / 2.0 % |
| 排水量 Δ | 26 t |
| 舯剖面惯性矩 I = EI/E | 19.42 m^4（EI = 4.000e+12 N·m²） |
| 甲板剖面模数 Z = I/(D/2) | 3.107 m³（D = 12.5 m） |
| 一阶垂向弯曲频率 f1 / 阻尼比 ζ | 0.761 Hz / 0.03 |
| 一阶模态质量 m1 / 模态刚度 k1 | 6.486e+06 kg / 1.483e+08 N/m |
| 舯曲率型函数 φ''_mid / 弓部型函数 φ_bow | 7.896e-04 1/m / 1.0 |

**参考航线（一次北极往返：10.0 h 冰区 + 敞水，日历 12 天）**

| 航段 | 时长 [h] | 冰压脉冲频率 f_ice [Hz] | 峰值冰力 [MN] | 说明 |
|---|---|---|---|---|
| level_ice_transit | 2.0 | 0.30 | 2.00 | continuous level-ice crushing at U/s = 0.30 Hz |
| brash_ice_channel | 1.0 | 0.45 | 1.00 | brash-ice channel, high pulse rate, low amplitude |
| ridge_ramming | 0.5 | 0.20 | 3.50 | ridge ramming: sparse high-amplitude impacts |
| lock_in_channel | 1.5 | 0.68 | 1.60 | pulse frequency close to f1 -> ice-induced vibration dominant |
| open_water_background | 5.0 | 0.12 | 0.35 | residual open-water / wave-induced load at the same station |

S-N 曲线：`power_law_fit_DNV_D_seawater_cp`，σ_a^3·N = C，log10 C = 11.1989
（由 DNV-RP-C203 D 曲线（海水 + 阴极保护）陡段拟合，见 §2.1）。

---

## 2. 方法与流程

### 2.1 S-N 曲线（C2）

幂律 σ_a^m·N = C 用 log-log 最小二乘拟合：`fit_power_law_sn(σ_a, N)`。
对 DNV-RP-C203 D 曲线陡段（N = 10^4…10^7）拟合得
m = 3.0000（规范值 3.0）、log10 C = 11.1989、
R² = 1.000000、log N 残差 RMSE = 1.27e-15。
**跨拐点（N = 1e7 处斜率由 3 变 5）用单一幂律拟合会把 m 抬到
3.157（R² = 0.99609）**，
所以模块同时接受双段 DNV 曲线（`fatigue.sn_curve.DNCurve`），幂律只用于解析校核与快速拟合。

### 2.2 冰载荷 → 船体梁应力（准静态 + 冰致振动）

对每个航段取 60 s 代表窗口：

1. **载荷**：半正弦压碎脉冲串，周期抖动 ±15 %、幅值对数正态（CV 0.30–0.45）、脊背撞击尾部（概率 p、放大 1.8）；
   幅值 f_ice = U/s 由漂移速度与碎冰长度决定。
2. **准静态**：自由-自由梁集中力静力学解 M_mid = F(L−a)/2（a = L/4 → M = 0.375 F L），σ_qs = M/Z。
3. **冰致振动**：一阶垂向弯曲模态 m1 q̈ + c1 q̇ + k1 q = φ_bow F(t)，RK4 积分（dt = 0.02 s），
   σ_vib = EI φ''_mid [q(t) − φ_bow F(t)/k1] / Z —— **扣掉模态静态偏置**，只保留振荡部分（whipping / lock-in 残余）。
4. σ_total(t) = σ_qs(t) + σ_vib(t)。

分解假设（显式声明）：高阶垂向模态按**准静态**处理（忽略其惯性），一阶模态承载全部被保留的振动动力学。
对本船，一阶模态携带弓部集中力静态舯弯矩的
37.9 %
（6.85 MPa/MN vs 精确静解 18.11 MPa/MN），其余 62.1 %
经精确静解进入。

### 2.3 雨流计数 + Miner 累积（C1/C3/C4）

- 雨流：ASTM E1049-85 四点法，采用符合标准的栈实现 `fatigue.rainflow.rainflow_counting_stack`；
  另在本模块内**独立重写**一份四点计数（`rainflow_counting_reference`，穷举闭合检查 + 环删除）作交叉验证。
- Miner：D = Σ n_i/N_i（`damage_from_ranges`），无任何修正系数。
- 航段损伤按**代表块假设**线性放大：D_leg = D_window·(T_leg/T_window)（同航段冰况统计定常）。
- 航次：D_voyage = Σ_legs D_leg；逐航次累积 D(t)，无修复时单调不减。

### 2.4 剩余寿命（C5）

N_total = 1/D_voyage，N_remaining = (1 − D)/D_voyage，剩余运营时间 = N_remaining × 航次日历天数。
极限精确成立：D = 0 → N_remaining = N_total；D = 1 → 0。

### 2.5 工具链

`src/tensorphysics/voyage_fatigue/voyage_fatigue.py`（1421 行）
复用 `fatigue.rainflow`、`fatigue.sn_curve`、`ice_vibration.modal`（自由-自由梁模态）、
`ice_vibration.integrators.rk4_sdof`、`ice_vibration.ice_load.half_sine_pulse`。
基准脚本 `run_voyage_fatigue.py`（判据 + 7 张图 + results.json），报告生成器 `make_report.py`。

---

## 3. 计算结果

### 3.1 航次损伤分解

| 航段 | 时长 [h] | f_ice [Hz] | 峰冰力 [MN] | D_leg | D_准静态 | D_振动分量 | 振动占比 | 峰值应力 [MPa] |
|---|---|---|---|---|---|---|---|---|
| level_ice_transit | 2.0 | 0.30 | 2.00 | 3.879e-04 | 1.579e-04 | 2.329e-04 | 59.3 % | 53.8 |
| brash_ice_channel | 1.0 | 0.45 | 1.00 | 1.798e-05 | 1.450e-05 | 1.168e-05 | 19.4 % | 25.2 |
| ridge_ramming | 0.5 | 0.20 | 3.50 | 1.109e-04 | 1.019e-04 | 6.536e-05 | 8.1 % | 71.4 |
| lock_in_channel | 1.5 | 0.68 | 1.60 | 3.060e-04 | 1.004e-04 | 1.308e-04 | 67.2 % | 54.4 |
| open_water_background | 5.0 | 0.12 | 0.35 | 3.959e-07 | 1.213e-07 | 3.347e-07 | 69.4 % | 4.1 |

- **航次累积损伤 D_voyage = 8.233e-04**；
  准静态 3.749e-04；纯振动分量 4.410e-04；
  **冰致振动占比 54.47 %**（判据 C6）。
- 峰值总应力 71.4 MPa（脊背撞击段），低于钢材屈服，线弹性 S-N 适用。
- 单窗口损伤（最严航段 60 s 窗口）3.697e-06，
  航次/窗口比 222.7（判据 C7：航次包含窗口）。

### 3.2 剩余寿命

| 量 | 值 |
|---|---|
| N_total = 1/D_voyage | 1214.7 航次 |
| 对应运营年限（12 天/航次） | 39.9 年 |
| 模拟到失效的航次数（逐航次累加） | 1215 航次（解析 1214.70，差 ≤ 1 航次） |
| D = 0.5 时剩余航次 | 607.3 |
| D = 0.99 时剩余航次 | 12.1 |
| D = 1.0 时剩余航次 | 0.0 |

### 3.3 判据总表

| 判据 | 内容 | 结果 | 门限 | 判定 |
|---|---|---|---|---|
| C1 | Miner cumulative damage vs hand calculation | 0.0000 % vs hand | < 0.1 % | PASS |
| C2 | Power-law S-N fit sigma_a^m * N = C | R^2 = 1.000000, m = 3.0000 | > 0.99 | PASS |
| C3 | Rainflow counting vs ASTM E1049-85 / independent implementation | analytic 0.000 %; cross-check 0.000 % | < 1 % | PASS |
| C4 | Cumulative damage monotone non-decreasing (no repair) | min increment 8.233e-04; severity profile min increment 4.940e-04 | no negative step | PASS |
| C5 | Residual life limits: D->1 => N_remaining->0, D=0 => N_total | N_rem(D=0) = 1214.70; N_rem(D=1) = 0.00 | exact | PASS |
| C6 | Ice-induced vibration share of the voyage damage | 54.47 % of the voyage damage (range over legs 8.1 % - 69.4 %) | 0 < f < 1 | PASS |
| C7 | Voyage damage >= single-window damage | D_voyage / D_window = 222.7 | >= 1 | PASS |
| C8 | pytest coverage of C1-C8 | 26 tests, all green | all pass | PASS |

### 3.4 频域工况图（冰压频率 / 一阶模态）

| f/f1 | f_ice [Hz] | 解析放大 | D_leg | 振动占比 |
|---|---|---|---|---|
| 0.10 | 0.076 | 1.01 | 3.676e-05 | 16.9 % |
| 0.20 | 0.152 | 1.04 | 1.549e-04 | 47.5 % |
| 0.30 | 0.228 | 1.10 | 1.964e-04 | 46.3 % |
| 0.40 | 0.304 | 1.19 | 4.402e-04 | 64.1 % |
| 0.50 | 0.381 | 1.33 | 1.051e-03 | 76.0 % |
| 0.60 | 0.457 | 1.56 | 3.468e-04 | 9.3 % |
| 0.70 | 0.533 | 1.95 | 6.298e-04 | 46.7 % |
| 0.80 | 0.609 | 2.75 | 8.895e-04 | 58.8 % |
| 0.90 | 0.685 | 5.06 | 1.464e-03 | 73.0 % |
| 0.95 | 0.723 | 8.85 | 4.331e-03 | 90.4 % |
| 1.00 | 0.761 | 16.67 | 1.186e-02 | 96.4 % |
| 1.10 | 0.837 | 4.54 | 6.525e-04 | 32.3 % |
| 1.25 | 0.951 | 1.76 | 3.200e-04 | -58.6 % |

### 3.5 时间步收敛

| dt [s] | D_leg（lock-in 航段） | 相对最细网格 | 峰值应力 [MPa] |
|---|---|---|---|
| 0.020 | 3.060e-04 | 0.270 % | 54.44 |
| 0.010 | 3.067e-04 | 0.039 % | 54.45 |
| 0.005 | 3.069e-04 | 0.000 % | 54.44 |

---

## 4. 结果图

1. `benchmarks/pending/voyage_fatigue/figures/fig1_stress_histories.png`
   — 两段代表窗口（lock-in 段 / 脊背撞击段）的应力分解：准静态、振动残余、总应力。
2. `fig2_rainflow_and_damage.png` — 雨流幅值谱与每循环损伤（m = 3 的 σ³ 敏感性）。
3. `fig3_leg_damage_split.png` — 逐航段损伤的准静态/振动堆叠（对数轴）+ 振动占比折线。
4. `fig4_damage_trajectory.png` — D(t) 航次累积与 D = 1 失效线，标注 10/25/50 航次处的剩余寿命。
5. `fig5_frequency_sweep.png` — 工况图：损伤与振动占比随 f_ice/f1 的变化，叠加解析放大因子 |H(r)|。
6. `fig6_sn_fit.png` — DNV D 双段曲线与幂律拟合（m、R² 标注）。
7. `fig7_dt_convergence.png` — 时间步收敛曲线与 1 % 接受带。

---

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **共享雨流工具的缺陷（本轮发现并修复）**：`fatigue.rainflow._find_peaks_valleys` 用
   `diff[i]·diff[i-1] < 0` 判极值，遇到**平台段**（冰压脉冲之间的严格零间隙、削顶信号）时驻点相邻差分
   为 0，符号变化永不触发，极值被吞掉。4 个脉冲的合成时程只能数出 3.0 个循环（真实 4.0）。
   回归用例 `tests/test_voyage_fatigue.py::test_c3_plateaus_do_not_swallow_extrema` 把该行为钉死
   （极值数 8 → 10，循环数 3.0 → 4.0），既有 fatigue / spectral / propeller 共 192 项测试回归全绿。
   对本节点的影响是实质性的：准静态应力时程正是「脉冲 + 严格零间隙」形态，修复前其循环被系统性吞掉，
   准静态损伤（§3.1 的 D_准静态 列）会被低估一个量级以上。
2. **`rainflow_counting`（列表版）是"外层闭合"变体**：该实现取四点中的**最大**内范围闭合循环，与
   ASTM E1049-85 相反（仓内测试已把它标注为 legacy）。在本文真实应力时程上，相对独立四点实现
   最大偏差 17.5 %；因此本模块一律用
   `rainflow_counting_stack`（与独立实现偏差 0.000 %）。
3. **一阶模态截断**：高阶模态按准静态处理，振动只保留一阶。当 f_ice 高于 f1 时一阶残余与准静态反相，
   振动占比会变负（fig5 已标注），故占比只在 f/f1 < 1 的子临界带内解释；参考航线各段 f/f1 ≤ 0.9。
4. **未建模**：速度弱化自激 lock-in 反馈（会使振幅自限）、裂纹扩展/检查信用、载荷次序相互作用、
   平均应力 Goodman 修正、局部焊接细节（本节点是船体梁总体应力级，细节级见 `fatigue.detail_fatigue`）。
5. **代表块假设**：航段内冰况统计定常，损伤率线性外推；若冰况在 60 s 尺度内剧变则该假设失效，
   实际使用应按冰况切分航段。
6. **绝对寿命不可用于设计**：σ 与 Z、EI、m1、ζ 都对船体理想化（自由-自由等截面梁）敏感；
   寿命是模型输出而非经过实验标定的预测值（见 §6 误差判定）。

---

## 6. 误差判定（关键量 vs 基准，逐项回答"是否 < 3%"）

| 关键量 | 基准/参照 | 偏差 | 是否 < 3% |
|---|---|---|---|
| Miner 累积 D（三块谱 20/40/60 MPa，1000/500/100 循环） | 手算 Σn_i/N_i | 0.00000 % | **是** |
| S-N 拟合 m | DNV-RP-C203 D 规范值 m = 3.0 | 0.000000 % | **是**（R² = 1.000000） |
| 雨流循环数（30 周期等幅余弦） | 解析 29 个 range = 2A 循环 | 0.00000 % | **是** |
| 雨流损伤（5 段真实应力时程 × 3 分量） | 独立四点实现（穷举闭合） | 0.00000 % | **是** |
| 共振应力量级（谐波激励 @ f1） | 解析 F/(k·2ζ)·|EI φ''/Z| | 0.161 % | **是**（114.42 vs 114.24 MPa） |
| 航次损伤对时间步 | dt = 0.005 s 最细解 | 0.270 %（dt = 0.02 s） | **是** |
| 剩余寿命极限 D=0 / D=1 | 解析 N_total / 0 | 0.000000 % / 0 航次 | **是**（精确） |
| 冰致振动损伤占比 | **无实验/文献基准可比**（公开数据不足，且依赖船体理想化） | 模型输出 54.47 % | **不适用**：按纪律声明无基准，给出的是模型内数字与跨段趋势（fig5），不作精度声称 |
| 绝对寿命（1215 航次 / 39.9 年） | **无连接实船疲劳裂纹台账** | — | **不适用**：结构理想化（等截面梁 + 名义 Z + 3 % 阻尼）带来的不确定度远大于 3 %，数值仅用于相对比较与流程演示 |

**结论**：所有可与解析解/独立算法/规范值对照的量均在 3 % 以内（多数在 1e-3 量级以下，C1/C3 为 0）；
与实验基准无关的量（振动占比、绝对寿命）已显式声明无基准，不作精度声称。
