# 冰致振动锁频与冰压循环低周疲劳验证面（P180W2-N07）

*生成时间 2026-11-10T18:35:06Z，TensorPhysics `src/tensorphysics/`；本文件全部数字由 `benchmarks/ice_vibration_fatigue/analysis.py` 程序化注入，零手抄。*

## 1. 案例介绍（物理背景 / 参数）

冰排以漂移速度 $U$ 挤压柔性结构时，破碎/剥落过程以 $f_{ice}=U/s$ 的频率施力（$s$ 为特征剥落长度）。当结构速度反馈使冰力随相对压入速度**下降**（延性—脆性转变、压力—面积软化）时，冰力向振型注入负阻尼；若冰提供的负阻尼超过结构阻尼，静平衡失稳，结构进入**自激极限环**。此即冰致振动锁频（lock-in）的力学内核：响应频率被结构固有频率俘获，振幅由冰力饱和而非 $1/(2\zeta)$ 决定。船体/冰区结构的该类振动是低周疲劳（LCF）与高周疲劳（HCF）的分界地带：局部缺口处应变幅可达 0.3–1 %，寿命落入 $10^3$–$10^5$ 周。

**参数表（NOMINAL 工况）**

| 量 | 值 | 来源 |
|---|---|---|
| 固有频率 $f_n$ | 1 Hz | 算例输入 |
| 模态质量 $m$ | 1e+06 kg | 算例输入 |
| 阻尼比 $\zeta$ | 0.02 | 算例输入 |
| 模态刚度 $k=m\omega_n^2$ | 3.948e+07 N/m | 派生 |
| 结构阻尼 $c_s=2\zeta m\omega_n$ | 2.513e+05 N·s/m | 派生 |
| 冰力幅值上限 $F_{max}$ | 1e+05 N | 算例输入 |
| 速度尺度 $v_c$ | 0.05 m/s | 算例输入 |
| 剥落长度 $s$ | 0.05 m | 算例输入 |
| 弱化指数 $p$ | 2 | 算例输入 |
| 自激速度窗口 | 0.003167 – 0.1113 m/s | 解析求解 |
| 冰力速度斜率 @ $U=v_c$ | -1e+06 N·s/m | 解析导数 |

**材料与缺口（代表值，非实测曲线）**

| 量 | 值 |
|---|---|
| $E$ | 2.06e+05 MPa |
| $\sigma'_f$ / $b$ | 1000 MPa / -0.1 |
| $\varepsilon'_f$ / $c$ | 0.5 / -0.5 |
| $n'=b/c$（Morrow 相容） | 0.2 |
| $K'=\sigma'_f/\varepsilon_f'^{n'}$ | 1149 MPa |
| 缺口常数（Neuber/Peterson） | 0.5 mm |

## 2. 怎么做的（方法 / 工具链）

三块新代码，全部落在 `src/tensorphysics/`（冰振动域 + 疲劳域），无外部数据依赖：

1. `ice_vibration/lockin_coupling.py`：速度弱化冰力 $F=F_{max}/(1+(v_{rel}/v_c)^p)$ 的**描述函数**负阻尼 $c_{ice}(a)=\frac{1}{\pi a\omega_n}\int_0^{2\pi}F(U-a\omega_n\cos\theta)\cos\theta\,d\theta$，解 $c_{ice}(a^*)=c_s$ 得极限环振幅；判锁频需**同时**满足频率俘获带 $0.8\le f_{ice}/f_n\le1.2$ 与自激条件 $c_{ice}(0)>c_s$。
2. `ice_vibration/integrators.rk4_sdof`（既有）做**独立时间域**验证：直接积分非线性方程测极限环振幅与频率。
3. `fatigue/strain_life.py`：Coffin-Manson-Morrow 应变寿命（含 Morrow 平均应力）、Ramberg-Osgood 循环曲线（$n'=b/c$、$K'=\sigma'_f/\varepsilon_f'^{n'}$ 由四常数相容导出）、Neuber 缺口方程 $(K_t\Delta S)^2/E=\Delta\sigma\Delta\varepsilon$、Glinka 能量法（独立闭包）、$K_t\to K_f$（Neuber/Peterson）、过渡寿命 $2N_T=[\varepsilon'_fE/(\sigma'_f-\sigma_m)]^{1/(b-c)}$。
4. `fatigue/ice_pressure_lowcycle.py`：冰压循环载荷谱（锁频载荷 $\Delta F=2ka$、曝光 $T$ 下 $n=f_n\cdot3600T$、Weibull 脉冲幅值离散）→ 简化梁名义应力 $\Delta S=\Delta F L/(4W)$ → Neuber 缺口应变 → $N_f$ → Miner 累积与 LCF/HCF 分账、$D=1$ 允许曝光。

验证工具链：`tests/test_ice_vibration_fatigue.py`（24 用例）、`benchmarks/ice_vibration_fatigue/analysis.py`（判据 C1–C7 + 机器可读工件）、本报告脚本（图 + 数字注入）。

## 3. 计算结果

### 3.1 锁频与振幅放大（C1 / C2）

| $r=f_{ice}/f_n$ | 锁频 | 极限环振幅 [m] | 放大因子 $X_{dyn}/X_{static}$ | 线性受迫放大（无冰反馈） |
|---|---|---|---|---|
| 0.8 | 是 | 0.03785 | **24.51** | 2.767 |
| 0.85 | 是 | 0.03791 | **25.78** | 3.577 |
| 0.9 | 是 | 0.03796 | **27.13** | 5.171 |
| 0.95 | 是 | 0.03801 | **28.55** | 9.556 |
| 1 | 是 | 0.03806 | **30.05** | 25 |
| 1.05 | 是 | 0.03809 | **31.62** | 9.028 |
| 1.1 | 是 | 0.03813 | **33.26** | 4.661 |
| 1.15 | 是 | 0.03815 | **34.98** | 3.07 |
| 1.2 | 是 | 0.03818 | **36.77** | 2.259 |

- 带内最小放大因子 **24.51**（判据 > 3）→ 满足；带内最大 36.77。
- 锁频带内放大因子的变化幅度 1.5 倍，而线性受迫响应在同一带内变化 11.07 倍；两者之比（俘获不敏感度）**0.1356**。
- **判别性证据**：强迫频率取 0.8 / 1.0 / 1.2 Hz（跨 40 %）时，时间域响应频率为 0.9999 Hz / 0.9999 Hz / 0.9999 Hz，相对 $f_n$ 的最大偏差 **0.01466 %**（频率俘获）；描述函数振幅与 RK4 时间域振幅的最大偏差 **0.00923 %**（两种独立方法互校）。
- 带外（$r$ = 0.3, 0.5, 2.0, 3.0）全部**不锁频**，放大因子退化为线性受迫值 1.099、1.333、0.3332、0.125。

### 3.2 应变寿命与过渡寿命（C3 / C5）

- 解析过渡寿命 $N_T$ = **5.383e+04 周**（$2N_T$ = 1.077e+05 反向）；独立数值求根 $2N_T$ = 1.077e+05，相对偏差 1.352e-16。
- 落在文档化钢铁 LCF/HCF 量级窗口 $10^4$–$10^5$ 周之内：**是**。
- 制造数据回归往返（2 %/3 % 噪声，seed 20261110）：总应变幅 RMS 偏差 **3.446 %**（判据 < 20 %），最大偏差 7.966 %；四常数回收偏差 σ'_f -3.183 %、b -2.228 %、ε'_f 3.794 %、c 1.44 %。

### 3.3 Neuber 缺口修正（C4）

| $\Delta S$ [MPa] | $\Delta\sigma\Delta\varepsilon$ 残差 | 缺口 $\Delta\varepsilon$（Neuber） | 缺口 $\Delta\varepsilon$（Glinka） | Neuber vs Glinka | 缺口塑性应变占比 |
|---|---|---|---|---|---|
| 20 | 6.949e-14 | 0.0002913 | 0.0002913 | +0.00 % | 8.341e-05 |
| 60 | 4.588e-14 | 0.0008767 | 0.0008748 | +0.22 % | 0.006624 |
| 150 | 1.988e-14 | 0.00238 | 0.002253 | +5.65 % | 0.1577 |
| 300 | 1.614e-13 | 0.006221 | 0.005137 | +21.11 % | 0.5068 |
| 600 | 2.665e-14 | 0.0184 | 0.01347 | +36.57 % | 0.7745 |

设计应力范围 $\Delta S$ = 150 MPa 处两法偏差 **5.649 %**（判据 < 15 %）；600 MPa（缺口塑性占比 0.7745）时增至 36.57 %，即 Neuber 规则在大范围屈服的已知局限，如实列出不下判据。

### 3.4 冰压循环载荷谱 → 疲劳损伤（C5 / C7）

锁频载荷（$r=1$）：振幅 $a$ = 0.03806 m，模态反力幅 $ka$ = 1.502e+06 N，载荷范围 $\Delta F = 2ka$ = 3.005e+06 N；曝光 10 h → $n$ = 3.6e+04 周。

| 工况 | $W$ [m$^3$] | $\Delta S$ [MPa] | 缺口 $\Delta\varepsilon/2$ | $N_f$ | 区段 | 10 h 损伤 $D$ | $D_{LCF}/D_{HCF}$ | 允许曝光 $D=1$ [h] |
|---|---|---|---|---|---|---|---|---|
| NOMINAL | 0.006 | 150.2 | 0.001192 | 3.505e+06 | **HCF** | 0.08122 | 0 / 0.08122 | 123.1 |
| SEVERE | 0.003 | 300.5 | 0.003118 | 4.992e+04 | **LCF** | 2.2 | 2.148 / 0.05174 | 4.546 |
| EXTREME | 0.0015 | 600.9 | 0.009224 | 2446 | **LCF** | 32.84 | 32.83 / 0.01245 | 0.3045 |

- 同一锁频载荷下 $\Delta S$ 从 150.2 MPa 到 300.5 MPa，寿命跨越 $N_T$，区段由 HCF 翻转为 LCF：$N_f$ = 3.505e+06 → 4.992e+04 周。
- 脉冲幅值离散（Weibull/Rayleigh，CoV=0.523）使 SEVERE 工况损伤由常幅值 0.7212 升至 2.2（谱效应 ×3.05）。
- 设计校核取 0.5×允许曝光：三个工况损伤 0.5、0.5、0.5 全部 ≤ 1（Miner 准则满足）。
- 负控：载荷范围 → 1 N 时损伤 = 0、允许曝光 = inf、全部块落入 run-out。

### 3.5 灵敏度（自激阈值的存在）

带内最小放大因子随阻尼比与冰力水平的变化（$r$ = 0.8/1.0/1.2 三点）：

| $\zeta$ \ $F_{max}$ [N] | 5e+04 | 1e+05 | 2e+05 |
|---|---|---|---|
| 0.01 | 49.02 | 51.1 | 51.85 |
| 0.02 | 21.86 | 24.51 | 25.55 |
| 0.03 | 12.65 | 15.49 | 16.71 |
| 0.05 | 0（未锁频） | 8.174 | 9.557 |
| 0.08 | 0（未锁频） | 3.88（未锁频） | 5.466 |

## 4. 结果图

![fig1](../artifacts/ice_vibration_fatigue/fig1_lockin_amplification.png)

**图 1** 放大因子—频率比：灰虚线为无冰反馈的线性受迫响应（在 $r=1$ 达 25），蓝点为自激极限环响应（带内 24.51–36.77），红点线为 C1 的 3 倍阈值，阴影为 0.8–1.2 俘获带。

![fig2](../artifacts/ice_vibration_fatigue/fig2_strain_life.png)

**图 2** Coffin-Manson-Morrow 应变寿命曲线（弹性/塑性/总应变）与 $N_T$ = 5.383e+04 周；三个圆点为三个缺口工况的 $(N_f, \Delta\varepsilon/2)$，可见由 HCF 侧移入 LCF 侧。

![fig3](../artifacts/ice_vibration_fatigue/fig3_neuber.png)

**图 3** 缺口求解：四条 Neuber 双曲线与 Ramberg-Osgood 循环曲线的交点（圆=Neuber，叉=Glinka 能量法），标注为两法偏差百分比。

![fig4](../artifacts/ice_vibration_fatigue/fig4_damage.png)

**图 4** 损伤—曝光时长（三个局部细节刚度），虚线竖线为 $D=1$ 的允许曝光，黑色虚线为 Miner 极限。

![fig5](../artifacts/ice_vibration_fatigue/fig5_entrainment.png)

**图 5** 俘获证据：左图为三个强迫频率（0.8/1.0/1.2 Hz）下的稳态位移时程，周期相同（= $T_n$）；右图同一时段的冰力时程，周期明显不同。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **“放大 > 3 倍”本身不是锁频证据。** 算例 $\zeta=2\%$ 的线性受迫响应在 $r=1$ 已达 25 倍，足以让“C1 判据”被纯共振冒充。本报告因此补了两条判别证据：响应频率被 $f_n$ 俘获（三组强迫频率下偏差 < 0.02 %）与带内放大因子对失谐不敏感（不敏感度 0.1356）。C1 若只按字面数字判，会把共振误判为锁频。
2. **负阻尼机制的符号约定**是这类实现最容易写反的地方：速度弱化力 $dF/dv<0$ 才能给出负阻尼；本模块把 $c_{ice}(a)$ 定义为“冰供给的负阻尼幅值”，方程写作 $c_{eff}=c_s-c_{ice}(a)$，并用 $c_{ice}(0)=|dF/dv|$ 的线性极限做机检（相对误差 < 1e-6）。
3. **极限环振幅的标度**由饱和而非刚度决定：$a^*$ 随 $F_{max}/(c_s\omega_n)$ 放大，在带内几乎不随 $U$ 变化（变化仅 1.5 倍），这正是锁频振动的特征，也是它对失谐不敏感的物理原因。
4. **Glinka 能量法的塑性项系数**易错：$\int_0^{\sigma_a}\sigma\,d\varepsilon$ 的塑性部分系数是 $K'/(n'+1)$，误写成 $n'K'/(n'+1)$ 会让 Glinka 反而高估缺口应变（本实现在自检中发现并修正）。
5. **$K_f\to1$ 必须走几何通道。** 固定 $K_t$ 时 Neuber/Peterson 公式给出的 $K_f$ 随缺口半径**减小**而趋近 1（缺口敏感性下降），所以“半径→∞ ⇒ $K_f$→1”只能通过几何 $K_t=1+2\sqrt{a/r}\to1$ 实现；本实现显式提供 `notch_factor_from_geometry` 以免把极限写反。
6. **适用范围**：速度弱化力模型是一维等效（不含接触面积演化、破碎尺度统计）；描述函数法假定响应近简谐，强非简谐（Duffing 跳跃区）需与时间域并用（本报告已用 RK4 互校）；Miner 线性累积不含载荷顺序与裂纹扩展阶段；缺口分析限于单轴、比例加载。
7. **材料常数是代表值**，不是实测曲线：$\sigma'_f$/$\varepsilon'_f$/$b$/$c$ 的取值区间会整体平移 $N_f$ 与 $N_T$。凡涉及“与实验对标”的判据（C2/C3/C4/C5 的实测部分）本报告一律记 WITHHELD，不粉饰。

## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3 %）

| 判据 | 阈值 | 达成值 | 基准类型 | 结论 | 是否 < 3 % |
|---|---|---|---|---|---|
| C1 锁频带内最小放大 | > 3 | 24.51 | 模型 | PASS | — （阈值型判据） |
| C2 放大因子 vs Frederking(1984) | < 30 % | 无可用实测锚 | 实验文献 | WITHHELD | 无法回答（无实测值） |
| C2' 描述函数 vs RK4 时间域 | < 30 % | 0.00923 % | 独立数值方法 | PASS | **是** |
| C3 ε-N vs 实验 | < 20 % RMS | 无实测数据集 | 实验 | WITHHELD | 无法回答（无实测值） |
| C3' 制造数据回归往返 | < 20 % RMS | 3.446 % | 制造解 | PASS | 否（3.446 %，为噪声量级） |
| C4 缺口应变 vs 实验 | < 15 % | 无实测数据 | 实验 | WITHHELD | 无法回答（无实测值） |
| C4' Neuber vs Glinka | < 15 % | 5.649 % | 独立闭包 | PASS | 否（5.649 %） |
| C4'' Neuber 恒等式残差 | 0 | 1.614e-13 | 解析恒等式 | PASS | **是** |
| C5 过渡寿命 vs 文献点值 | < 25 % | 5.383e+04 周 | 材料专属文献点 | WITHHELD | 无法回答（无点值） |
| C5' N_T 解析 vs 数值 | 0 | 1.352e-16 | 独立求根 | PASS | **是** |
| C6a 远离 $f_n$ 无锁频 | 不锁频 | 全部不锁频 | 模型负控 | PASS | — |
| C6b 应力幅→0 寿命→∞ | $D$=0 | $D$=0，允许曝光 inf | 模型负控 | PASS | — |
| C6c 缺口半径→∞ $K_f$→1 | $K_f$→1 | 1.01（渐近 1） | 模型负控 | PASS | — |
| C7 Miner $D\le1$ | ≤ 1 | 设计曝光下 0.5、0.5、0.5 | 解析恒等式 | PASS | **是**（D=1 处精确到 1e-9） |

**计数**：PASS 8 · WITHHELD 4 · FAIL 0。

### 审计链

- `C1` (PASS, model + RK4 time-domain entrainment evidence): Amplification alone is not evidence of lock-in (the linear forced response already reaches 25.0 at r = 1); the discriminating evidence is that the response frequency stays at f_n for forcing frequencies 0.8/1.0/1.2 Hz (max deviation 0.015 %) and that the amplification spread across the band is 0.136 of the linear forced-response spread.
- `C2` (WITHHELD, experimental literature anchor): No digitised Frederking (1984) ice-induced-vibration amplification record could be obtained: a Crossref author+date sweep 1983-1985 returns his adfreeze / flexural strength / wharf-loading papers (e.g. doi:10.1139/l84-114, 'Ice action on Nanisivik wharf, winter 1979-1980') without retrievable numeric amplification data. Reported WITHHELD, not passed.
- `C2-surrogate` (PASS, independent numerical method (RK4 time integration of the nonlinear EOM)): Two independent methods for the same limit cycle; agreement is far tighter than the 30 % criterion.
- `C3` (WITHHELD, experimental dataset): No strain-controlled epsilon-N dataset is available offline; the constants used are a representative structural-steel set, not a measured curve. Reported WITHHELD.
- `C3-surrogate` (PASS, manufactured_roundtrip): Verifies the curve, the Basquin/Coffin-Manson reduction and the life inversion; it says nothing about the physical validity of the constants.
- `C4` (WITHHELD, experimental dataset (notched-specimen strain measurements)): No notched-specimen notch-strain measurements available offline. Reported WITHHELD.
- `C4-surrogate` (PASS, independent notch closure (Glinka strain-energy density)): Both closures collapse onto the elastic notch strain in small-scale yielding and diverge to 36.6 % at 600 MPa where the notch plastic fraction reaches 0.77 -- the documented limitation of Neuber's rule, reported rather than hidden.
- `C5` (WITHHELD, material-specific literature point anchor): The analytic N_T matches an independent root find of the elastic/plastic crossing to 1.35e-16 relative and lands inside the documented 1e4-1e5 cycle window for steels; a material-specific literature N_T could not be verified offline, so the literature point comparison stays WITHHELD.
- `C6a` (PASS, model negative control): Ratios 0.3/0.5/2.0/3.0 fall outside the entrainment band and, at the low end, outside the self-excited window.
- `C6b` (PASS, model negative control): Run-out defined at 2e8 reversals (1e8 cycles); the infinite allowable exposure is serialised as Infinity.
- `C6c` (PASS, model negative control (Inglis K_t + Neuber/Peterson K_f)): For a fixed K_t the notch *sensitivity* falls as the root sharpens, so K_f -> 1 must be driven through the geometry K_t = 1 + 2 sqrt(a/r); stated explicitly to avoid the inverted-limit trap.
- `C7` (PASS, model + analytic Miner identity): Damage is linear in exposure at fixed spectrum, so D = 1 at the allowable exposure is an exact identity (tested to 1e-9).

工件：`artifacts/ice_vibration_fatigue/{results,criteria,literature_anchors}.json`、`fig1..fig5 *.png`；测试 `tests/test_ice_vibration_fatigue.py`（24 用例）；复现 `python3 benchmarks/ice_vibration_fatigue/analysis.py && python3 benchmarks/ice_vibration_fatigue/report.py`。
