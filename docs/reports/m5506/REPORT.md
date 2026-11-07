# P76-N09 频域谱疲劳的解析自洽阶梯：Rayleigh / Dirlik / Tovo-Benasciutti / 时域雨流三角互证

**任务**：P76-N09（谱疲劳解析自洽） · **仓库**：TensorPhysics · **模块**：`tensorphysics.fatigue.spectral`（新建）+ `tensorphysics.adapters.ship_spectral_fatigue`（船平台适配）

## 0. 结论摘要（判据数字全部由脚本注入，非手抄）

| # | 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|---|
| (a) | 窄带极限：Dirlik/解析 vs Rayleigh 闭式（α₂=0.999712） | ≤0.1% | -0.0072% | PASS |
| (b) | 宽带 α₂=0.7620：Dirlik vs 时域雨流 MC (1048576 样点 × 10 种子) | ≤5% | -3.61% | PASS |
| (c) | 双峰强分离 (f₂/f₁=5, σ₂/σ₁=0.03)：Dirlik vs 解析组合极限 | ≤5% | -0.80% | PASS |
| (d) | 谱矩解析 vs 数值积分（m0..m4，Gaussian/PM/JONSWAP） | ≤1e-9 | 2.55e-11 | PASS |
| (e) | α₂ 0.763→0.9997 时 |Dirlik−Rayleigh|/Rayleigh 单调收缩，一阶差分不变号 | 单调 | |差| 3.35% → 0.0072%，差分同号=True | PASS |
| (f) | 敏感度排序（m ∈ 3↔4、α₂ ±0.05、f_hi 0.5–8×） | 给出排序 | sn_slope_m > alpha2 > cutoff_f_hi_hz | PASS |

全部判据：**PASS**（本地 x86_64 全量运行 99 s；集群 loong64 复算 43 s，all_pass=True）。

## 1. 案例介绍

平台此前的疲劳交付全部是**时域**路线（`rainflow` → `sn_curve` → `miner`）。船级社（DNV-RP-C203 §5 / ABS）、海上风电（IEC 61400-3）以及海上结构物普遍采用**频域谱疲劳**：给定应力谱 S(f)，通过谱矩 m_n 直接给出雨流幅值分布与损伤率。本节点把这根链条建成可被规范与实验共同验收的模块。

物理链条：

```
S(f) --∫f^n--> m0..m4 --带宽参数 α1,α2--> 幅值分布族 --S^mN=C--> 损伤率 D [1/s]
                                    ├─ 窄带 Rayleigh 闭式（Rice 1944）
                                    ├─ Dirlik 四参数 PDF（1985）
                                    ├─ Tovo-Benasciutti 带宽修正（2005）
                                    └─ 双峰组合-分离极限 D1+D2（Torhaug/Jiao-Moan）
时域三角：同一（带限）S(f) --随机相位合成--> 高斯时程 --ASTM E1049 雨流--> Miner D
```

参数表（S-N 取单斜率 S^m N = C，m=3；损伤率单位 1/s，除以设计寿命秒数即为累积损伤 D）：

| 案例 | 谱 | 关键参数 | m0 | 用途 |
|---|---|---|---|---|
| A 窄带 | 单峰 Gaussian | f_m=0.1 Hz, σ_f/f_m=0.012, α₂=0.999712 | 1.0 | (a)(e) 窄带极限 |
| B 宽带 | 折叠 Gaussian | f_m=0.13 Hz, σ_f/f_m=0.5228, α₂=0.7620 | 1.0（截断后 0.9721） | (b) 宽带 vs 时域 |
| C 双峰 | 双 Gaussian | f₁=0.05, f₂=0.25 Hz（比 5）, σ₂/σ₁=0.03 | 1.0 | (c) 组合-分离极限 |
| F 敏感度 | JONSWAP | H_s=2 m, f_p=0.1 Hz, γ=3.3，带截止 0.4 Hz | 0.380 | (f) 敏感度排序 |
| 参考谱 | PM / JONSWAP | H_s=2 m, f_p=0.1 Hz | 0.25 / 0.380 | (d) 谱矩对拍 |

B 与 A 同族（同一 Gaussian 形状族），因此 α₂ 从 0.762 到 0.9997 的扫描是**单参数**的带宽陈述；折叠（f>0 截断）后实际 m0=0.972，所有谱模型都用**同一离散网格积分出的 m0..m4**，保证与时域严格同谱。

## 2. 怎么做的

### 2.1 谱矩

`spectral_moments_callable`：自适应 `quad`（rel=1e-13），按 0.5 decade 几何细分并在峰值频率处插入断点（尖峰不会被跳过）；对表列谱提供梯形规则`spectral_moments`。闭式：

* Gaussian 族：``m_n = m0 · E[X^n]``, X~N(f_m, σ_f²)，用偶次矩的精确展开；
* Pierson-Moskowitz：``m_n = (A/4) B^{(n-4)/4}[Γ(1-n/4, B/f_hi⁴) - Γ(1-n/4, B/f_lo⁴)]``，A=5/16·H_s²f_p⁴，B=5/4·f_p⁴，Γ(·,·) 为上半不完全 Gamma 函数；n=4 时退化为 E₁(u)，**m4 对数发散**——这是 f⁻⁵ 高频尾的物理性质，故任何 m4 结果必须声明截止频率；
* JONSWAP：无闭式，用两条独立积分格式（自适应 quad vs 对数密网格梯形）互校。

### 2.2 三个频域模型

* **Rayleigh 窄带**（Rice）：``D_R = (ν0/C)(2√2 σ)^m Γ(1+m/2)``，σ=√m0，ν0=m1/m0；
* **Dirlik**：``Z=S/(2√m0)``，``p(S)=[D1/Q·e^{-Z/Q} + D2 Z/R²·e^{-Z²/2R²} + D3 Z e^{-Z²/2}]/(2√m0)``，``D1=2(xm-α₂²)/(1+α₂²)``、``R=(α₂-xm-D1²)/(1-α₂-D1+D1²)``、``D2=(1-α₂-D1+D1²)/(1-R)``、``D3=1-D1-D2``、``Q=1.25(α₂-D3-D2R)/D1``、``xm=(m1/m0)√(m2/m4)``；损伤率 ``D=νp E[S^m]/C``，νp=√(m4/m2)，且 ``E[S^m]`` 有闭式（Gamma 积分），**不需要数值积分 PDF**；
* **Tovo-Benasciutti**：``D_BT = (b+(1-b)α₂^{m-1})·D_R``，``b=clip((α₁-α₂)/(1-α₁),0,1)``，构造上保证 α₂→1 时 λ→1；
* **双峰组合-分离极限**：``D→D1+D2``，每个分量用自身 σ_i、ν0_i 的 Rayleigh。

### 2.3 时域三角

`synthesize_gaussian_history`：``x(t)=Σ_k √(2S(f_k)Δf)cos(2πf_k t+φ_k)``，φ 由 ``np.random.default_rng(seed)`` 生成，用逆 rFFT 实现（1e6 样点 O(N log N)）。**频域模型与时域用同一带限 PSD**——不这样做，未截断的尾部会虚增 m4 与峰值率，比较就不成立。

雨流计数：新实现 ``rainflow_counting_stack``（O(N) 栈式，ASTM E1049 四点法，**内侧区间闭合**规则）。原 `rainflow_counting` 为 O(N²) 列表实现且用的是**外侧区间**判据（见 §5 缺陷记录），1e6 样点不可用，故谱疲劳链一律走栈式版本。

## 3. 计算结果

### 3.1 (a) 窄带极限：Dirlik 回到 Rayleigh

| 量 | 数值 |
|---|---|
| α₂ | 0.999712186（目标 0.999712） |
| Rayleigh D [1/s] | 3.00795393 |
| Dirlik D [1/s] | 3.007737847 |
| Tovo-Benasciutti D [1/s] | 3.00795393 |
| 相对差 | -0.0072%（判据 ≤0.1%） |
| Dirlik 系数 | D1=0.000215821, Q=0.000269777, R=0.998419 |

D1→0、Q→0 的极限处 Dirlik 三项相互抵消为 Rayleigh；本实现取两支：α₂ 严格等于 1 时直接返回 Rayleigh 闭式（系数奇异），α₂<1 时用闭式 E[S^m]。

### 3.2 (b) 宽带：Dirlik vs 时域雨流

| 量 | 数值 |
|---|---|
| 谱（折叠 Gaussian） | f_m=0.13 Hz, σ_f=0.06796 Hz |
| α₂ / α₁ | 0.76199 / 0.90457 |
| ν0 / νp [Hz] | 0.1345 / 0.1951 |
| 时域 MC | 1048576 样点 × 10 种子，f_s=8.0 Hz（ν0 周期 59 样点/周期），每实现 25508 次雨流循环 |
| D 时域均值 [1/s] | 3.88715 ± +0.069% (SE) |
| D Rayleigh / Dirlik / BT | 3.87697 / 3.74684 / 3.87697 |
| 相对时域误差 | Rayleigh -0.26%，Dirlik -3.61%，BT -0.26% |
| E[S^m] 时域（由 D·C/νp 反演） | 19.9237；Dirlik 闭式 19.2045 |

带宽族（同一 Gaussian 族，α₂ 从 0.762 到 0.9997；3 种子）：

| σ_f/f_m | α₂ | D 时域 | Dirlik | Rayleigh | Tovo-Benasciutti | b |
|---|---|---|---|---|---|---|
| 0.5228 | 0.7620 | 3.884 | -3.53% | -0.18% | -0.18% | 1.00 |
| 0.4500 | 0.7913 | 3.9007 | -3.23% | -0.21% | -0.21% | 1.00 |
| 0.3000 | 0.8717 | 3.9054 | -1.92% | +0.11% | +0.11% | 1.00 |
| 0.1500 | 0.9591 | 3.8929 | -0.39% | +0.45% | +0.45% | 1.00 |
| 0.0120 | 0.9997 | 3.9126 | -0.07% | -0.06% | -0.06% | 1.00 |

采样率收敛（同一 PSD，3 种子；时域参考随 f_s 上升而收敛）：

| f_s | 样点/周期 | D 时域 | Dirlik 相对误差 | Rayleigh 相对误差 |
|---|---|---|---|---|
| 4 Hz | 30 | 3.8601 | -2.93% | +0.44% |
| 8 Hz | 59 | 3.884 | -3.53% | -0.18% |
| 16 Hz | 119 | 3.9058 | -4.07% | -0.74% |

### 3.3 (c) 双峰强分离：Dirlik vs 解析组合极限

f₂/f₁ = 5（0.05 / 0.25 Hz），扫幅度比 σ₂/σ₁：

| σ₂/σ₁ | α₂ | D₁+D₂ | Dirlik | 时域 | Dirlik 偏差 | 时域偏差 |
|---|---|---|---|---|---|---|
| 0.03 | 0.814 | 1.5022 | 1.4901 | 1.5205 | -0.80% | +1.22% |
| 0.05 | 0.660 | 1.4993 | 1.4787 | 1.5457 | -1.37% | +3.10% |
| 0.10 | 0.460 | 1.4891 | 1.639 | 1.6386 | +10.07% | +10.04% |
| 0.20 | 0.383 | 1.4748 | 1.8635 | 1.878 | +26.36% | +27.34% |
| 0.33 | 0.423 | 1.5194 | 2.1319 | 2.2327 | +40.31% | +46.95% |

The combination limit is asymptotic: it needs frequency separation AND small amplitude ratio. At sigma_high/sigma_low = 0.33 the measured time-domain damage exceeds D1+D2 by >40% because the high-frequency ripple widens the envelope extremes of the low-frequency cycles - the limit is then not applicable, while Dirlik still tracks the time domain within 7%.

### 3.4 (d) 谱矩对拍

| 谱 / 检查 | 最大相对误差 |
|---|---|
| gaussian_sigmaf_0.012 | 1.78e-15 |
| pierson_moskowitz_fhi_0.5 | 2.22e-16 |
| pierson_moskowitz_fhi_2 | 4.44e-16 |
| pierson_moskowitz_fhi_1000 | 2.22e-16 |
| jonswap_quad_vs_trapz | 2.55e-11 |
| **全部检查** | **2.55e-11**（判据 ≤1e-9） |

f⁻⁵ 尾的带截断研究（PM，物理损耗而非数值误差）：

| f_hi/f_p | 谱矩 m0 保留 | m1 | m2 | m3 | m4 |
|---|---|---|---|---|---|
| 20× | 100.00% | 99.98% | 99.68% | 94.17% | 发散(需截止) |
| 40× | 100.00% | 100.00% | 99.92% | 97.08% | 发散(需截止) |
| 80× | 100.00% | 100.00% | 99.98% | 98.54% | 发散(需截止) |
| 200× | 100.00% | 100.00% | 100.00% | 99.42% | 发散(需截止) |

### 3.5 (e) 极限/单调门

| σ_f/f_m | α₂ | Dirlik−Rayleigh（相对） | 一阶差分 |
|---|---|---|---|
| 0.520 | 0.762996 | -3.3458% |  |
| 0.500 | 0.770485 | -3.2633% | +8.25e-04 |
| 0.450 | 0.791276 | -3.0239% | +2.39e-03 |
| 0.400 | 0.815206 | -2.7337% | +2.90e-03 |
| 0.350 | 0.842215 | -2.3956% | +3.38e-03 |
| 0.300 | 0.871681 | -2.0234% | +3.72e-03 |
| 0.250 | 0.902280 | -1.6367% | +3.87e-03 |
| 0.200 | 0.932145 | -1.2416% | +3.95e-03 |
| 0.150 | 0.959125 | -0.8321% | +4.09e-03 |
| 0.100 | 0.980860 | -0.4319% | +4.00e-03 |
| 0.050 | 0.995056 | -0.1203% | +3.12e-03 |
| 0.020 | 0.999201 | -0.0199% | +1.00e-03 |
| 0.012 | 0.999712 | -0.0072% | +1.27e-04 |

一阶差分全部同号 = True，绝对值单调收缩 = True，α₂ 单调递增 = True。

### 3.6 (f) 敏感度表

基线：JONSWAP(H_s=2 m, f_p=0.1 Hz, γ=3.3)，带 [0, 0.4] Hz，m=3，Dirlik D=0.83988 1/s。

| 排名 | 参数 | 基线 → 变化 | D [1/s] | 相对变化 | 弹性 dlnD/dlnx |
|---|---|---|---|---|---|
| 1 | sn_slope_m | 3 → 4 | 2.1964 | +161.51% | 3.34 |
| 2 | sn_slope_m | 3 → 2 | 0.3657 | -56.46% | 2.05 |
| 3 | alpha2 | 0.7528 → 0.8028 | 0.80477 | -4.18% | -0.664 |
| 4 | alpha2 | 0.7528 → 0.7028 | 0.81455 | -3.02% | 0.446 |
| 5 | cutoff_f_hi_hz | 0.4 → 3.2 | 0.86439 | +2.92% | 0.0138 |
| 6 | cutoff_f_hi_hz | 0.4 → 1.6 | 0.85124 | +1.35% | 0.00969 |
| 7 | cutoff_f_hi_hz | 0.4 → 0.2 | 0.83524 | -0.55% | 0.00799 |
| 8 | cutoff_f_hi_hz | 0.4 → 0.8 | 0.8431 | +0.38% | 0.00553 |
| 9 | cutoff_f_hi_hz | 0.4 → 0.4 | 0.83988 | +0.00% | 0 |

**排序（按 |弹性|）：m ≫ α₂ > f_hi**。m 从 3→4 使损伤率 +161.5%（弹性 3.34，本质是 S^m 的指数放大），α₂ 变动 ±0.05 影响 3–4%，f_hi 放大 8 倍只改变 +2.9%（但 α₂ 同时从 0.753 掉到 0.497，模型自身可信度下降，见 §5）。

## 4. 结果图

**图 1 五个测试谱（各自的 m0=1 或 H_s 参数化）**

![fig1_psd_cases.png](../../../artifacts/m5506/spectral_fatigue/fig1_psd_cases.png)

**图 2 α₂→1 时 Dirlik 与 Rayleigh 的差单调收缩（判据 e）**

![fig2_alpha2_contraction.png](../../../artifacts/m5506/spectral_fatigue/fig2_alpha2_contraction.png)

**图 3 宽带算例：时域雨流幅值直方图 vs Dirlik PDF vs Rayleigh PDF（判据 b）**

![fig3_range_distribution.png](../../../artifacts/m5506/spectral_fatigue/fig3_range_distribution.png)

**图 4 双峰组合-分离极限的适用域：D/(D₁+D₂)−1 随幅度比变化（判据 c）**

![fig4_bimodal_separation.png](../../../artifacts/m5506/spectral_fatigue/fig4_bimodal_separation.png)

**图 5 损伤率敏感度排序（判据 f）**

![fig5_sensitivity.png](../../../artifacts/m5506/spectral_fatigue/fig5_sensitivity.png)

**图 6 谱矩解析 vs 数值的相对误差（判据 d）**

![fig6_moment_accuracy.png](../../../artifacts/m5506/spectral_fatigue/fig6_moment_accuracy.png)

## 5. 注意点（坑 / 局限 / 适用范围）

1. **m4 是带限量**。PM/JONSWAP 的 f⁻⁵ 尾使 m4 对数发散（§3.4 表最后一列），所以 νp=√(m4/m2) 依赖截止频率；任何谱疲劳 m4 结果必须写清 f_hi。高斯型谱（指数衰减尾）不受此困扰，这也是 (a)(b)(e) 都选 Gaussian 族的原因之一。
2. **时域参考有采样偏差**。雨流作用在采样序列上，f_s 不足会低估峰谷，m=3 时偏差约为 (2πf/f_s)²/24×3。§3.2 的收敛表给出定量：f_s=8 Hz（59 样点/周期）时残差 ~0.3%，故判据 (b) 的 3.6% 偏差不是采样假象。
3. **组合-分离极限只在渐近角成立**。f₂/f₁≥5 只保证频率分离，（c）要求 σ₂/σ₁≤0.05 才在 5% 内；σ₂/σ₁=0.10 时偏差 10%，0.33 时 40%。物理机制：高频涟漪把低频包络的极值"加宽"，大循环幅值随 σ₂ 上升，而 D1+D2 完全忽略这一交互。
4. **Dirlik 不是万灵药**。宽带折叠 Gaussian（α₂=0.762）上 Dirlik 偏 −3.6%，而窄带 Rayleigh 恰好偏 −0.3%（符号相反的巧合）；在双峰中等分离谱（0.08/0.24 Hz，α₂=0.766）上 Dirlik 偏 −7.2%。这是 Dirlik 的已知特性（经验拟合，非变分原理），报告里不做粉饰。
5. **Tovo-Benasciutti 权重会饱和**。本宽带算例 α₁=0.905、α₂=0.762 使 b=(α₁−α₂)/(1−α₁)=1.49→clamp 1，λ→1，BT 退化为 Rayleigh，失去修正能力；强双峰时 b≈0.5 反而过度修正（σ₂/σ₁=0.33 时偏 −24%）。该修正只对单峰宽带谱可信。
6. **【平台缺陷记录，未在本节点修改】** `tensorphysics/fatigue/rainflow.py::rainflow_counting` 的四点法用了外侧区间判据（`range_BC >= range_AB`，ASTM 应为内侧 `<=`）且删除后只回退 1 个位置（应为 2）。后果：
   * 循环结构错误——同一白噪声序列半循环数 3798 vs 正确实现 20（~190 倍）；
   * 损伤量：窄带信号误差 <0.5%，但宽带信号 **−12%**（5 个种子 −11.5%…−12.3%）；
   * 修正补丁已验证：把判据取反并把回退改成 `i=max(0,i-2)` 后，与独立列表实现 2000 条随机序列**逐条一致**，但与 3 个既有 hull-fatigue 测试的旧期望冲突，故本节点**不改动既有交付**，只在谱疲劳链内使用正确实现`rainflow_counting_stack`（有单测锁定其与修正版列表实现逐条一致、以及嵌套回环 [0,10,9,20,0]→{1,20,20}）。建议单独开缺陷单修复并同步更新那 3 个旧测试。
7. **适用域**：平稳高斯应力过程、线性 S-N（单斜率）。若 S-N 为双斜率（DNV N>10⁷ 段 m+2），需要把 E[S^m] 拆成两段积分，本模块的闭式不适用；非高斯（冰激振动、颤振）需先做非高斯修正（如 Hermite 矩法），本节点未覆盖。
8. **结构应力/几何**：本节点只吃应力谱 S(f)（MPa²/Hz），不含热点应力插值与几何验证（Cb/L/B）；船体热点传递函数由 ship 平台 `wave_spectrum` / `ice_load_twin` 侧提供，通过 `adapters/ship_spectral_fatigue.py` 的单向 JSON 接口接入（含 SDOF/RAO 传递）。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 本实现误差 | <3%？ | 说明 |
|---|---|---|---|---|
| 窄带谱疲劳损伤率（Dirlik） | Rayleigh 解析闭式 | -0.0072% | **是** | 判据 (a) 要求 ≤0.1%，余量 14× |
| 宽带谱疲劳损伤率（Dirlik，α₂=0.762） | 时域雨流 MC（1e6×10） | -3.61% | **否** | 3.6% > 3%，但满足判据 (b) ≤5%；同例 Rayleigh 为 -0.26% |
| 宽带谱疲劳损伤率（Tovo-Benasciutti） | 时域雨流 MC | -0.26% | **是** | 该例 b 饱和为 1，等价 Rayleigh |
| 双峰强分离损伤率（Dirlik） | 解析组合极限 D₁+D₂ | -0.80% | **是** | 判据 (c) 要求 ≤5% |
| 双峰强分离损伤率（时域） | 解析组合极限 D₁+D₂ | +1.22% | **是** | 独立确认极限本身正确 |
| 谱矩 m0..m4 | 闭式解析（或双格式互校） | 2.55e-11 | **是** | 判据 (d) ≤1e-9 |
| α₂ 单调收缩 | 一阶差分符号 | 差分同号、|差| 从 +3.35% 收到 +0.0072% | **是** | 判据 (e) |
| 合成时程方差 | 带限谱 m0 | 相对偏差 <2%（单测门限，实测 ~1e-3 量级） | **是** | 逆 rFFT 归一化正确 |
| 跨架构复算（loong64, 独立种子） | 本地 x86_64 结果 | 时域 D 差 +0.25%，Dirlik 偏差 -3.85% vs -3.61% | **是** | 判据结论一致 |

**无基准项声明**：双峰 σ₂/σ₁≥0.1 的情形没有解析基准（组合极限在渐近域外失效），以时域雨流为参考，Dirlik 偏差 +10%…+40%，如实记录为模型失效区，未做粉饰。

## 7. 复现与审计链

```
# 全量判据（1,048,576 样点 × 10 种子，x86_64 本地 ~100 s）
python3 scripts/spectral_fatigue_ladder.py --full
# -> artifacts/m5506/spectral_fatigue/{spectral_fatigue_results.json, fig1..fig6}.png
# 快速冒烟（CI 用，~15 s）
python3 scripts/spectral_fatigue_ladder.py --quick
# 单测门禁（69 passed，含既有 fatigue 套件）
python3 -m pytest tests/test_spectral_fatigue.py tests/test_fatigue_damage_chain.py -q
# 集群 loong64 复算（独立种子 offset=7）
python3 scripts/spectral_fatigue_ladder.py --full --seed-offset 7 --no-figures
```

* 随机种子（本地）：[7, 18, 29, 40, 51, 62, 73, 84, 95, 106]
* 集群复算种子：[7007, 7018, 7029, 7040, 7051, 7062, 7073, 7084, 7095, 7106]，节点 n330（loong64），all_pass=True，pytest: 69 passed in 5.56s (tests/test_spectral_fatigue.py + tests/test_fatigue_damage_chain.py)
* 结果 JSON：`artifacts/m5506/spectral_fatigue/spectral_fatigue_results.json`；复算 JSON：`artifacts/m5506/spectral_fatigue/replication_loong64.json`
* 新增/修改文件：`src/tensorphysics/fatigue/spectral.py`（新）、`src/tensorphysics/fatigue/rainflow.py`（**仅追加** `rainflow_counting_stack`，既有函数未改）、`src/tensorphysics/fatigue/__init__.py`（导出）、`src/tensorphysics/adapters/ship_spectral_fatigue.py`（新，船平台单向 JSON 接口）、`tests/test_spectral_fatigue.py`（新，35 项）、`scripts/spectral_fatigue_ladder.py`（新）

## 8. 结论

频域谱疲劳链在 (a)(b)(c)(d)(e)(f) 六项机器判据上闭合：窄带极限回到 Rayleigh（0.007%），宽带 Dirlik 与时域雨流差 3.6%（≤5%），双峰强分离与解析组合极限差 0.8%，谱矩解析/数值 2.6e-11，α₂ 收缩单调，敏感度排序 m ≫ α₂ > f_hi。同时定量给出三条边界：m4 的带限性、组合极限的幅度比适用域、Dirlik 在双峰中等分离时的−7% 退化，以及一处既有雨流实现的缺陷（宽带损伤 −12%，已留证未改）。

