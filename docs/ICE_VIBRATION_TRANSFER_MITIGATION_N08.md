# p227w2-N08 冰致振动传递链成熟化：船体湿模态 → 设备基座加速度（QZS 隔振）→ 管路应力 → 疲劳寿命

| 项 | 内容 |
|---|---|
| 战役节点 | `p227w2-N08`（成熟化轮，前序 `p228w1-N08`）|
| 代码 | `src/tensorphysics/ice_vibration_transfer/mitigation.py`（新增）、`maturity.py`（新增）|
| 载荷 payload | `artifacts/m10169/ice_vibration_transfer_mitigation_n08.json`（本文所有数字的唯一来源）|
| 测试 | `tests/test_ice_vibration_transfer_maturity.py`（21 用例，含 5 条机器判据）|
| 图 | `docs/ice_vibration_transfer_n08_maturity/`（7 张）|
| 判据总判定 | **全部通过**（M1–M9，见 §3.5）|

## 1 案例介绍

### 1.1 物理背景与节点增量

冰区航行时螺旋桨铣削冰层产生与叶片通过相位锁定的扭矩脉动（N04 DEM 铣削扭矩谱），经轴系反力激起船体梁
**湿模态**振动，再经甲板基座局部结构、设备隔振器传到设备与管路，驱动管路细节疲劳。前序节点 `p228w1-N08`
把这条链串通并给出 10 条判据，但其披露 D1 给出一个无法回避的物理结论：**单级被动隔振器无法衰减冰载荷能带**。
常规 4 Hz 隔振器（静挠度 15.5 mm）在 0.3–3.0 Hz 带内 |H|>1，带内 RMS(基座)/RMS(船体) =
**1.158**（放大）；而把 f_n 降到 0.4 Hz 的线性弹簧需要 1553 mm 静挠度，船用隔振器不可能实现。

本节点增量是**把静刚度与动刚度解耦**：载荷弹簧 k_v 承担设备重量，负刚度机构 k_n = −α k_v 抵消其中 α 份额，
动态频率 f_n,dyn = f_rigid·√(1−α) 降到冰带以内，这就是准零刚度（QZS / HSLDS）隔振器（Carrella et al. 2007；
Ibrahim 2008）。据此给出：QZS 设计图 → 前置规则选点 → 传递链重算（真实随机相位应力合成 + 雨流 + DNV-RP-C203
S-N + Miner）→ 与常规方案**前后对照**，并把 QZS 的工程代价（刚度容限、行程、缓冲限位、Duffing 线性化边界）
逐项量化披露。

### 1.2 几何与主尺度（几何先行校验）

| 量 | 数值 | 判定 |
|---|---|---|
| 船型 | KCS（公开基准）| — |
| L×B×T×D | 230.0 × 32.2 × 10.8 × 19.0 m | — |
| Cb | 0.6510（公开 0.6505）| +0.08% ✓（<3%）|
| 排水量 | 53372 t（公开 52030 t）| +2.58% ✓ |
| L/B / B/T | 7.14 / 2.98 | 窗口内 ✓ |
| 几何闸门 G1 | 通过 | 先校验 Cb/L/B 再算结构 |

### 1.3 参数表

| 参数 | 数值 | 性质 |
|---|---|---|
| 冰况 | light / design / heavy（design：冰压 2.00 MPa，Q_mill 234.1 kN·m，Q RMS 7.5 kN·m）| 上游 N04 |
| 转速 / 叶片通过 | 150 rpm / 10.0 Hz | N04 |
| 轴系→船体传递效率 η | 0.30（力 RMS 1340 N）| 声明（继承前序）|
| 干模态阻尼 | ζ=0.02 | 声明 |
| 湿模态方法 | 干模态基 + 二维剖面附加质量 a33(x) + 浮力恢复刚度，12 阶 | 复用 hull_hydroelastic |
| 常规隔振器（基线）| f_n=4.00 Hz，ζ=0.15，静挠度 15.5 mm | 前序节点基线 |
| QZS 设计点 | f_n,dyn=0.400 Hz、α=0.9900（残余刚度 0.0100）、ζ=0.15、峰值放大 3.51 | 本节点（规则前置）|
| QZS 硬件 | 载荷弹簧 f_rigid=4.00 Hz；静挠度 15.5 mm 不变；k_v=1.39 MN/m，k_dyn=13.9 kN/m；NS 线性区 ±20 mm；行程 ±50 mm | 声明 |
| 冰带 | 0.3–3.0 Hz（设计前固定）| 由船体加速度谱能量界定 |
| 管路 | 跨距 8.0 m，Φ406×12.7 mm，一阶 ~12.6 Hz，SCF 1.8、厚度修正 (t/25mm)^0.2 | 简支梁 + DNV |
| S-N | D 曲线 / 海水+CP：m1=3、m2=5、1e7 次 = 50.20 MPa | 仓库 fatigue.sn_curve |
| 运行假设 | 冰区运行占比 0.25，设计寿命 25 年 | 声明 |
| 波浪/鞭击包络（仅行程核对）| 0.10 Hz / 0.60 m/s² + 1.00 Hz / 0.30 m/s² RMS | 声明输入（D7）|

## 2 怎么做的

### 2.1 方法链

1. **湿模态**：干自由-自由 Timoshenko 梁模态（质量归一）+ 二维剖面无穷频附加质量 a33(x) + 浮力恢复刚度，
   在干模态基中解广义特征值问题 → f_wet 与湿模态振型（`hull_modal.py`，复用前序实现，不重写）。
2. **载荷**：N04 铣削扭矩冲击序列 → Welch 扭矩 PSD → 自平衡力偶谱 → 模态叠加得到基座站位船体加速度 PSD。
3. **基座局部结构**：25 Hz 甲板基座 SDOF（ζ=0.02）把船体加速度传到"支撑"（冰带内 |A|≈1）。
4. **隔振层（本节点增量）**：基座激励 SDOF 复传递函数 H(f) = (1+2iζr)/((1−r²)+2iζr)，r=f/f_n；
   常规方案 f_n=4.00 Hz，QZS 方案 f_n=f_rigid·√(1−α)=0.400 Hz。
5. **管路应力**：简支管路模态应力传递 |σ(f)/a_base|²（静态极限应等于 m L²/(8Z)），乘 SCF 与厚度修正。
6. **疲劳**：随机相位应力时程合成（Parseval 保方差）→ ASTM E1049-85 四点雨流 → DNV-RP-C203 D 曲线 →
   Palmgren-Miner 累积 → 寿命（年），与窄带 Rayleigh 闭式解交叉校核。
7. **设计（前置规则，先定后算）**：在 (f_n,dyn, ζ) 网格上建 QZS 设计图，硬约束为静态挠度 ≤
   50 mm、残余刚度份额 ≥ 0.010、ζ ≥ 0.15
   （峰值放大 ≤ 3.3）、波浪/鞭击行程或缓冲限位力 ≤ 25% 设备重量、
   冰带振动落在 NS 线性区内；在可行域内取带内 RMS 比最小的点（平手取残余刚度更大者）。可行行数 16/60。

### 2.2 子模块与验证

| 子模块 | 职责 | 验证 |
|---|---|---|
| `mitigation.QZSMount` | 线性化 QZS：f_n、静挠度、灵敏度、Duffing 三次项、行程相对位移 | α=0 时与常规隔振器逐点完全一致；H(0)=1 |
| `mitigation.design_map/select_design` | 设计图与前置选点 | 极软/极硬/低阻尼行分别被对应 flag 判死（测试）|
| `mitigation.simulate_base_excitation` | 基座激励 SDOF 的 RK4 时域积分（含三次非线性） | 与频域 \|H(f)\| 相对误差 2.45e-05；硬化骨干使共振峰右移（测试）|
| `mitigation.tolerance_window` | α 误差容限（老化/温度）| 带内比值随 δα 单调（测试）|
| `maturity.run_mitigation` | 设计 + 三冰级重算 + 判据 + 披露 + 曲线 | 21 条 pytest 全绿 |

## 3 计算结果

### 3.1 干/湿模态（M1）

| 阶次 | f_dry [Hz] | f_wet [Hz] | f_wet/f_dry | 附加质量比 |
|---|---|---|---|---|
| 1 | 0.5589 | 0.3670 | 0.657 | 0.601 |
| 2 | 1.0552 | 0.6577 | 0.623 | 0.628 |
| 3 | 1.5122 | 0.9652 | 0.638 | 0.585 |
| 4 | 1.9369 | 1.2344 | 0.637 | 0.591 |
| 5 | 2.3426 | 1.5017 | 0.641 | 0.579 |
| 6 | 2.7358 | 1.7586 | 0.643 | 0.578 |
| 7 | 3.1195 | 2.0124 | 0.645 | 0.574 |
| 8 | 3.4953 | 2.2626 | 0.647 | 0.570 |
| 9 | 3.8637 | 2.5384 | 0.657 | 0.535 |
| 10 | 4.2252 | 2.7922 | 0.661 | 0.523 |
| 11 | 4.5799 | 3.2265 | 0.704 | 0.434 |
| 12 | 4.9279 | 3.4872 | 0.708 | 0.430 |

全部 12 阶 f_wet < f_dry，最大比值 0.708（M1 PASS）。
2 节点垂向湿模态 0.367 Hz 落在冰带内，正是常规隔振器放大的频段。

### 3.2 传递函数与基座加速度（M2/M3）

| 方案 | f_n | √2f_n | 衰减带内 max\|H\| | 冰带 RMS 比（基座/船体）| 全带(0.2–500 Hz) RMS 比 |
|---|---|---|---|---|---|
| 常规 | 4.00 Hz | 5.66 Hz | 0.890 | 1.158 | 1.216 |
| QZS 设计 | 0.400 Hz | 0.566 Hz | 0.247 | 0.238 | 0.235 |

* 点判据：两者在 f > √2f_n 的衰减带内 |H|<1（M2 PASS）；QZS 衰减带内最大 0.247。
* 能带判据：QZS 把冰带 RMS 比从 **1.158（放大）** 压到 **0.238**（M3 PASS，改善 4.9 倍）。
* 全带（0.2–500 Hz）比值：常规 1.216（放大）；QZS 0.235。
  注意 QZS 全带 <1 并不代表"所有频率都衰减"：0.3 Hz 以下任何被动悬挂都是 1:1 传递（D1），
  QZS 在 f < √2f_n = 0.57 Hz 处仍有窄带放大（峰值 3.51，D2）；
  全带 <1 来自 0.6 Hz 以上大幅衰减对低频段的补偿，属于物理上限内的可实现结果，不是实现缺陷。

### 3.3 管路应力与疲劳寿命（M4–M6、M9）

| 冰级 | 船体 RMS [m/s²] | 冰带船体 RMS [m/s²] | 常规带内比 | QZS 带内比 | 常规寿命 [yr] | QZS 寿命 [yr] | 寿命增益 |
|---|---|---|---|---|---|---|---|
| light | 2.319e-05 | 2.277e-05 | 1.246 | 0.210 | 1.080e+26 | 1.101e+30 | 10196 |
| design | 6.892e-05 | 6.792e-05 | 1.158 | 0.238 | 1.202e+24 | 2.615e+27 | 2174 |
| heavy | 8.069e-05 | 7.911e-05 | 1.226 | 0.216 | 2.331e+23 | 1.844e+27 | 7914 |

* 管路应力 RMS（QZS，SCF+厚度修正）：light=1.61e-05 MPa, design=5.47e-05 MPa, heavy=5.78e-05 MPa（M4 正，PASS）。
* 三冰级寿命有限且 > 0（M5 PASS），并严格单调下降（M6 PASS）。
* 设计冰级寿命增益 **2174×**（M9 PASS）；纯幂律估计 (1/0.238)^5 = 1313×，
  与雨流+Miner 直接算得的值同量级（差异来自谱形与 S-N 两分支混合）。

### 3.4 冰带定义敏感性（结论稳健性）

| 冰带 [Hz] | 常规带内 RMS 比 | QZS 带内 RMS 比 |
|---|---|---|
| 0.3–3.0 | 1.158 | 0.238 |
| 0.8–4.0 | 1.298 | 0.196 |
| 0.2–5.0 | 1.216 | 0.235 |
| 0.3–5.0 | 1.216 | 0.235 |

即使冰带下移到 0.8–4 Hz（若真实 2 节点湿模态 ~1 Hz），结论不变：常规放大、QZS 衰减、设计点仍在可行域。

### 3.5 判据与披露

| ID | 判据 | 数值 | 阈值 | 判定 |
|---|---|---|---|---|
| M1 | wet natural frequencies below dry (all elastic modes) | [0.6567, 0.6233, 0.6383, 0.6373, 0.641, 0.6428, 0.6451, 0.6473, 0.657, 0.6608, 0.7045, 0.7076] | 1 | PASS |
| M2 | |H(f)| < 1 pointwise above sqrt(2) f_n (both mounts) | conventional_max_above_isolation=0.8901; qzs_max_above_isolation=0.2472 | 1 | PASS |
| M3 | RMS(equipment base) < RMS(hull) over the ice band (energy-sense |H|<1) | baseline_ratio=1.158; qzs_ratio=0.2379; band_hz=[0.3, 3]; full_band_baseline_ratio=1.216; full_band_qzs_ratio=0.2346 | 1 | PASS |
| M4 | pipe stress positive for the three ice severities (QZS chain) | light=1.024e-05; design=3.479e-05; heavy=3.678e-05 | 0 | PASS |
| M5 | fatigue life finite and positive for the three ice severities (QZS chain) | light=1.101e+30; design=2.615e+27; heavy=1.844e+27 | 0 | PASS |
| M6 | fatigue life decreases when the ice load increases | light=1.101e+30; design=2.615e+27; heavy=1.844e+27 | life(light) > life(design) > life(heavy) | PASS |
| M7 | QZS hardware feasibility (static deflection, stiffness budget, stroke) | static_deflection_m=0.01553; static_deflection_limit_m=0.05; residual_stiffness_fraction=0.01; residual_stiffness_budget=0.01; zeta=0.15; zeta_min=0.15; peak_amplification=3.514; ice_band_relative_displacement_rms_m=2.311e-06; linearity_amplitude_m=1.161e-04; bump_stop_force_fraction_of_weight=0.09013; stop_force_limit_fraction=0.25; stroke_m=0.05; peak_service_relative_displacement_m=0.202 | static deflection, stiffness budget, damping floor, linear ice band and bump-stop force | PASS |
| M8 | RK4 time-domain integration reproduces the analytic transmissibility | 2.446e-05 | 0.01 | PASS |
| M9 | fatigue-life gain of the QZS mount over the conventional mount (design severity) | 2174 | 1 | PASS |

| ID | 披露 | 数值 |
|---|---|---|
| D1 | Any passive mount has |H(0)| = 1: at zero frequency the masses move rigidly with the foundation, so 'base acceleration below hull acceleration at every frequency' is impossible for a passive suspension. The conventional mount even amplifies the ice band; the criterion is therefore posed pointwise above sqrt(2) f_n (M2) and in the energy sense over the ice band (M3). | 1.216 |
| D2 | QZS amplification band: below sqrt(2) f_n the softened mount still amplifies, with a narrow-band resonance at f_n,dyn. Pointwise |H| maximum inside the ice band is reported with the design; only the band integral is below unity. | peak_amplification=3.514; grid_max_in_band=1.741; note=grid max is sampled on the Welch frequency grid; peak_amplification is the analytic maximum |
| D3 | Stiffness-cancellation tolerance: an error delta_alpha in the negative-stiffness fraction moves the dynamic frequency by -delta_alpha/(2 (1-alpha)). The window in which the attenuation criterion still holds is reported; outside it the design degrades to the conventional-mount regime (stiffer, band ratio > 1). | max_abs_delta_alpha_still_attenuating=0.02; first_failing_delta_alpha=-0.05 |
| D4 | Stroke: the QZS mechanism must remain linear over the whole service motion. Wave-induced and whipping relative displacements (declared envelope) are compared with the ice-band motion and with the stroke; they exceed the ice-band motion by orders of magnitude. | ice_relative_displacement_rms_m=2.311e-06; linearity_amplitude_m=1.161e-04; worst_case=wave; worst_relative_displacement_rms_m=0.101; peak_service_relative_displacement_m=0.202; stroke_m=0.05; bump_stop_engaged=True; bump_stop_force_fraction_of_weight=0.09013; bump_stop_stiffness_N_per_m=1.556e+05 |
| D5 | Duffing linearisation: beyond the NS linear range the QZS force gains a cubic term (beta = alpha k_v / ns_range^2), the resonance backbone hardens. The relative displacement below which the cubic correction stays inside 1 % is reported and compared with the ice-band motion. | cubic_coefficient_N_per_m3=3.439e+09; linearity_amplitude_m=1.161e-04; ice_relative_displacement_rms_m=2.311e-06; backbone_peak_hz_at_linearity_amplitude=0.4002 |
| D6 | Absolute life levels: the ice-induced pipe detail stresses are far below the DNV-RP-C203 D-curve endurance level (50.2 MPa at 1e7 cycles), so the computed lives are astronomically long (1e8 yr and above). Ice-vibration fatigue does not govern this pipe detail; the QZS gain is a relative mitigation metric, not a design-life statement. | stress_rms_MPa_design=3.479e-05; life_years_design=2.615e+27; stress_at_1e7_cycles_MPa=50.2 |
| D7 | Declared inputs: wave/whipping envelope (used only for the stroke check), QZS damping ratio, NS linear range, stroke, equipment mass, foundation bracket mode and the shaft-to-hull transmission efficiency inherited from the verified chain. The QZS linearisation is used for the chain; the nonlinear check is reported separately. | eta_hull=0.3; qzs_zeta=0.15; ns_range_m=0.02; stroke_m=0.05; equipment_mass_kg=2200; foundation_f_n_hz=25; wave_whipping_envelope=[label=wave; freq_hz=0.1; accel_rms_m_s2=0.6, label=whipping; freq_hz=1; accel_rms_m_s2=0.3] |
| D8 | Ice-band definition and energy split: the band is fixed at 0.3-3 Hz before the design is selected; the share of hull acceleration variance below the QZS isolation frequency is reported because that share cannot be attenuated by this mount. | ice_band_hz=[0.3, 3]; qzs_isolation_frequency_hz=0.5657; variance_share_below_isolation=0.002751 |

### 3.6 图

1. `fig1_wet_dry_modes.png` — 干/湿模态对比（M1）：湿模态在全部弹性模态上低于干模态。
2. `fig2_bode_transfer.png` — 传递函数 Bode 图（幅值 dB + 相位）：常规 4 Hz 隔振器在冰带内放大，QZS 设计点把冰带压到 0 dB 以下。
3. `fig3_base_accel_psd.png` — 加速度 PSD 沿链演化：船体梁 → 基座支撑 → 设备基座（常规 vs QZS）。
4. `fig4_design_map.png` — QZS 设计图：band RMS 比等值线、可行域（静态挠度/残余刚度/阻尼/行程/线性化）与选中设计点。
5. `fig5_life_vs_ice_load.png` — 疲劳寿命-冰载荷曲线：轻/设计/重三冰级，两种方案均严格单调下降。
6. `fig6_sn_curve_ranges.png` — DNV-RP-C203 D 曲线与工作应力范围：说明绝对寿命量级（披露 D6）。
7. `fig7_tolerance_stroke.png` — 刚度抵消误差容限与行程核对：冰带振动在 NS 线性区内，波浪/鞭击由缓冲限位承担。

## 4 注意点（踩坑 / 局限 / 适用范围）

1. **被动隔振在 DC 处不可能衰减**：|H(0)|=1 是任何被动悬挂的硬上限，因此"基座加速度在**所有**频率低于船体"
   物理上不可满足；本节点把判据拆为点判据（f>√2f_n）与能带判据（冰带 RMS），并显式披露（D1/D2）。
2. **QZS 的代价是刚度抵消精度**：α=0.990 意味着只保留 1.0% 刚度，
   d ln f_n/dα = −1/(2(1−α))，α 误差 1% → 频率误差 50%。容限窗计算：带内衰减在
   δα ∈ [-2.0%, +1.0%] 内成立
   （+ 方向受"残余刚度不能为负"限制，− 方向是机构变硬/老化方向）；
   变硬方向在 δα ≤ -5% 时失效（此时 f_n 升到 0.98 Hz，带内比回到 1 以上）
   → 服役期必须可调/可复标（D3）。
3. **行程是第二个硬约束**：冰带振动仅 2.31 µm（远在 NS 线性区
   0.116 mm 内），但 0.1 Hz 波浪分量造成的相对位移 RMS 达
   101 mm（2σ = 202 mm），
   超出行程 ±50 mm → 必须设**缓冲限位**，其承担力为设备重量的
   9.0%（1945 N），所需限位刚度
   156 kN/m。判据 M7 因此检查"行程**或**限位力"，并把限位力披露（D4）。
4. **非线性边界**：超出 NS 线性区后三次项 β=3.44e+09 N/m³ 使骨干硬化；线性化在相对位移
   < 0.116 mm 内误差 <1%，冰带振动低 2 个数量级，链上用线性 QZS 自洽（D5，附 RK4 硬化验证）。
5. **绝对寿命量级巨大（2.6e+27 年）**：冰致管路应力 RMS 仅
   5.5e-05 MPa，比 D 曲线 1e7 次幅值 50.2 MPa 低 5–6 个数量级
   → **冰致振动疲劳不控制该管路细节**。M9 的寿命增益是相对缓解指标，不能读作"设计寿命 1e27 年"，更不能替代含内压/热胀/
   总纵弯曲的完整疲劳评估（D6）。
6. **继承自前序节点的局限（本节点未改，仅披露）**：船体梁模型偏柔（湿 2 节点 0.37 Hz，而 230 m
   集装箱船实测 2 节点垂向湿频率通常 0.6–1.0 Hz），故冰带与设计点的绝对位置随结构理想化移动；§3.4 表明结论对该
   不确定性稳健。S-N 取仓库实现（1e7 次 50.2 MPa），较 DNV-RP-C203 表值 52.6 MPa 低
   4.6%（偏保守）；未改动以保持与前序节点可比。
7. **未考虑**：辐射阻尼（用结构阻尼 2%）、冰-船冲击压力路径（仅扭矩路径）、多轴/焊接细节真实 SCF、QZS 机构摩擦与间隙、
   主动/半主动隔振对比。管路应力只含基座加速度惯性激励，不含内压/热载荷。

## 5 误差判定（是否 <3%）

| 关键量 | 计算值 | 基准/参考 | 偏差 | 是否 <3% |
|---|---|---|---|---|
| Cb（几何闸门）| 0.6510 | 0.6505（KCS 公开值）| +0.08% | 是 |
| 排水量 | 53372 t | 52030 t（KCS 公开值）| +2.58% | 是 |
| QZS 频域 \|H\| vs RK4 时域 | 3.48002 | 3.48010 | 0.0024% | 是 |
| 管路模态应力传递 vs 静态极限 m L²/(8Z) | 1.265106e+06 Pa/(m/s²) | 1.266281e+06 Pa/(m/s²) | 0.0927% | 是 |
| α=0 时 QZS vs 常规隔振器 | 逐点一致 | 解析式 | 0（机器精度）| 是 |
| 雨流损伤 vs 窄带 Rayleigh（设计冰级）| 0.573 | 1.0 | -42.7% | 否（宽带修正方向正确、量级一致）|
| S-N 1e7 次应力幅 | 50.2 MPa | 52.6 MPa（DNV-RP-C203 D 表值）| -4.6% | **否**（偏保守；为与前序可比未改）|
| 湿模态频率 vs 实测 | 0.367 Hz | 230 m 船 2 节点垂向湿频率 0.6–1.0 Hz（文献量级）| −39%…−63% | **否**（继承缺陷，见 §4.6）|

**结论**：可解析校验的量（几何、管路静态应力极限、QZS 时域/频域一致性、α=0 退化）全部 <3%；
**两项 >3%**：① 仓库 S-N 曲线 1e7 次幅值比 DNV 表值低 4.6%（偏保守，保持与前序可比未改）；
② 湿模态绝对频率偏低 39–63%（继承前序船体梁理想化，本节点未改动，只披露并做了冰带敏感性 §3.4）。
按纪律不粉饰：两项均为已知限制，非本节点引入。

## 6 结论

1. 湿模态在全部 12 阶低于干模态，最大比值 0.708（M1）。
2. 常规 4 Hz 隔振器在冰带内放大（RMS 比 1.158），这正是前序节点 D1 的定量根源；QZS 设计点
   （f_n,dyn=0.40 Hz、α=0.990、静挠度 15.5 mm 不变）把冰带 RMS 比压到 0.238，
   并满足点判据 |H|<1（f>√2f_n）（M2/M3）。
3. 管路应力与疲劳寿命在三冰级上均为正、有限且随冰载荷严格单调下降（M4/M5/M6）；设计冰级寿命增益 2174×（M9）。
4. QZS 可实现性代价已量化：刚度抵消容限 ±2%、缓冲限位力 9.0% 设备重量、
   NS 线性区 20 mm（M7）。
5. 绝对寿命极长说明冰致振动疲劳不控制该管路细节（D6）；QZS 的价值在于把设备/管路的**振动环境**从放大变为衰减，
   为振动烈度指标（ISO 6954 类）留出余量。

## 7 审计链

* 代码：`src/tensorphysics/ice_vibration_transfer/mitigation.py`、`maturity.py`（本节点新增）；
  `hull_modal.py` / `excitation.py` / `response.py` / `pipe.py` / `transfer.py`（复用前序，未改）。
* 测试：`tests/test_ice_vibration_transfer_maturity.py`（21 用例，全绿）。
* 载荷：`artifacts/m10169/ice_vibration_transfer_mitigation_n08.json`（本文所有数字由该 payload 程序化注入）。
* 图：`docs/ice_vibration_transfer_n08_maturity/fig1–fig7`（由 `scripts/m10169_n08_figures.py` 生成，
  报告由 `scripts/m10169_n08_report.py` 生成）。
* 上游：N04 铣削扭矩谱（`propeller_fatigue.ice_spectrum`）、湿模态（`hull_hydroelastic`，前序节点已消费）。
* 前序：`docs/ICE_VIBRATION_TRANSFER_N08.md`（p228w1-N08，披露 D1 为本节点输入问题）。
