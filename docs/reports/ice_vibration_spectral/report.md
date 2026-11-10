# 冰致振动频谱分析与共振风险判据（P233W2-N07）

> 战役节点 P233W2-N07 · 物理域 ice/cfd/platform · 落库 `src/tensorphysics/ice_vibration_spectral/`
> 全部数字由 `generate_p233w2_n07_ice_vibration.py` 从 `node_payload.json` 程序化注入，无手抄。

## 1. 案例介绍（物理背景 / 参数表）

冰排以漂移速度 U 挤压柔性结构时，韧性-脆性破碎交替进行，载荷以重复频率 $f_{ice}=U/s$（s 为两次破碎之间累积的侵入深度）周期性释放。若 $f_{ice}$ 落入结构某阶模态的半功率带宽，结构响应被放大 >3 dB，且对速度弱化型冰力可自激锁定（lock-in）。本节点建立「冰激励谱 → 结构频响 → 响应 PSD → 共振风险判据」的完整管线，并在两类结构上实施：

**案例 A：细长灯塔摆振模态（文献锚点）**。Gulf of Bothnia 灯塔冰致振动（Norrström 1978；Kärnä & Turunen 1990），文献报道主导响应频率 1.0–1.2 Hz，取 1.10 Hz 作为锚点。

| 参数 | 值 | 来源/说明 |
|---|---|---|
| 模态质量 m | 3e+05 kg | 声明参数（细长钢塔量级） |
| 固有频率 f_n | 1.1000 Hz | 文献锚点 1.10 Hz |
| 阻尼比 ζ | 0.02 | 焊接钢结构 2 % |
| 模态刚度 k = m(2πf_n)² | 1.433e+07 N/m | 由 m、f_n 导出 |
| 冰漂移速度 U | 0.16 m/s | 声明参数 |
| 破碎侵入深度 s | 0.15 m | 约 0.5×冰厚 |
| 接触宽度 | 2.0 m | 声明参数 |
| 有效冰压 p_eff | 0.30 MPa | Timco & Weeks (2010) 0.3–2.0 MPa |
| 破碎频率 f_ice = U/s | 1.0667 Hz | 由 U、s 导出 |
| 载荷时程 | 180.0 s @ 0.002 s | 90000 点 |
| 半功率带宽 | 0.0440 Hz | $2\zeta f_n$ 精确解 |

**案例 B：船体梁 5 自由度集中质量链**。前 5 阶垂向弯曲模态（0.91、2.64、4.17、5.36、6.11 Hz），节点质量 5e+06 kg、节点刚度 2e+09 N/m，首末模态阻尼 2 %（Rayleigh 阻尼），冰力作用于自由端（船首）。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **冰激励建模**（`excitation.py`）：韧性加载-脆性卸载的间歇破碎时程，周期 $T=s/U$ 带对数正态抖动（cv = 0.18）、幅值抖动 0.25、慢漂移调制 0.08 @ 0.11 Hz 与 12 % 宽带背景；Welch 单边 PSD（Hann 窗，50 % 重叠，nperseg = 16384）。另给解析窄带谱 $S_{ff}$（高斯形，半功率带宽为参数），用作 Welch 估计的闭式锚。
2. **结构动力学**（`structural.py`）：SDOF 复频响 $H(\omega)=1/(k-m\omega^2+i c\omega)$；MDOF 固定-自由均匀链，闭式特征频率 $\omega_r = 2\sqrt{k/m}\sin\frac{(2r-1)\pi}{2(2n+1)}$，Rayleigh 阻尼 $\zeta_r=(\alpha+\beta\omega_r^2)/(2\omega_r)$；频响由**直接复数求解** $(K-\omega^2M+i\omega C)^{-1}$ 与**模态叠加**两条独立代码路径给出。**FEM 模态基接口**：`response.modal_response_psd(f, f_n, phi_n, zeta_n, S_ff, dof_in, dof_out)` 接受任意质量归一化模态基（TensorFEM 特征值导出/实测模态）直接产出响应 PSD，无需 M、K 矩阵；与链式系统模态叠加结果一致到 1e-12。
3. **频响与响应 PSD**（`response.py`）：$S_{yy}(f)=|H_a(\omega)|^2 S_{ff}(f)$（位移/速度/加速度），RMS 由细网格积分获得；激励 PSD 通过线性插值细化 64×（Welch 分辨率 0.031 Hz 粗于结构带宽 0.044 Hz，直接在 Welch 网格上积分使 RMS 偏高 2.2 %，见 §5）。
4. **共振风险判据**（`criterion.py`）：由半功率带精确解$f_\pm=f_n\sqrt{1-2\zeta^2\pm 2\zeta\sqrt{1-\zeta^2}}$ 分级：`lock_in`（带内）/ `near_resonance`（3 倍半功率带宽内）/ `clear`；并与幅值判据 $|H(f_{ice})|\ge \max|H|/\sqrt2$ 独立互检。
5. **校核**：时域精确 ZOH 递推（piecewise-constant 力的状态转移矩阵 + 特解 $[F/k,0]$，无积分截断误差）独立复算响应 RMS 与谐响应幅值。

## 3. 计算结果（关键数字）

| 判据 | 内容 | 实测 | 基准 | 相对误差 | 容差 | 判定 |
|---|---|---|---|---|---|---|
| AC1 | SDOF 固有频率 | 1.086800 Hz | 1.087235 Hz | 0.040 % | 1 % | PASS |
| AC1 | MDOF（3/5/8 自由度） | — | 闭式解 | max 2.27e-15 (<1e-2 %) | 5 % | PASS |
| AC2 | SDOF 频响幅值 | — | 状态空间解 | max 2.76e-15 (<1e-2 %) | 1 % | PASS |
| AC2 | 共振峰附近（0.95–1.05 f_n） | — | 状态空间解 | 2.76e-15 (<1e-2 %) | 5 % | PASS |
| AC2 | 相位 | — | 状态空间解 | 2.29e-13 deg | 1 deg | PASS |
| AC2 | 峰值幅值 1/(2ζk√(1-ζ²)) | 1.785639e-06 m/N | 1.786072e-06 m/N | 0.024 % | 1 % | PASS |
| AC2 | MDOF 模态叠加 vs 直接求解 | — | 直接求解 | 4.14e-14 (<1e-2 %) | 1 % | PASS |
| AC3 | 白噪声响应 RMS（谱积分） | 4.667272e-03 m | 4.667278e-03 m | 1.24e-06 (<1e-2 %) | 5 % | PASS |
| AC3 | 白噪声响应 RMS（时时域 6 种子） | 4.664483e-03 m | 4.667278e-03 m | 0.060 % | 5 % | PASS |
| AC4 | 判据一致性（27 点） | 27/27 一致 | 幅值判据 | 0.00e+00 (<1e-2 %) | 0 % | PASS |
| AC5 | 响应主频 vs 文献 | 1.0987 Hz | 1.10 Hz | 0.122 % | 20 % | PASS |

**案例 A 响应量**：位移 RMS 14.556 mm（时限 14.221 mm，差异 2.298 %）；加速度 RMS 0.6820 m/s²；响应 PSD 主频 1.0987 Hz。

**共振风险**：$f_{{ice}}$ = 1.0667 Hz 与 f_n = 1.1000 Hz 的失谐 3.030 %，半功率带 [1.0773, 1.1213] Hz，放大率 $|H(f_{ice})|/\max|H|$ = 0.562（< 0.707）→ 分级 **near_resonance**。

**锁频（lock-in）演示**：同一机制在 U 使 $f_{ice}=f_n$ 且破碎周期抖动被结构速度夹带收窄（cv 0.18 → 0.03）时，位移 RMS 由 14.556 mm 增至 29.998 mm（放大 2.06×），判据由 near_resonance 变为 lock_in。

## 4. 结果图

![fig1_excitation](docs/reports/ice_vibration_spectral/figures/fig1_excitation.png)

*图 1 冰破碎激励：时程片段（左）与 Welch 单边 PSD（右），标注 f_ice 与结构半功率带。*

![fig2_frf](docs/reports/ice_vibration_spectral/figures/fig2_frf.png)

*图 2 SDOF 位移频响：闭式解与状态空间独立解重合；右图为相位，共振处穿越 −90°。*

![fig3_response_psd](docs/reports/ice_vibration_spectral/figures/fig3_response_psd.png)

*图 3 案例 A 响应 PSD（位移实线 / 加速度虚线），文献频带与响应峰标注。*

![fig4_mode_shapes](docs/reports/ice_vibration_spectral/figures/fig4_mode_shapes.png)

*图 4 案例 B 船体梁 5 自由度链的归一化振型与模态频率。*

![fig5_resonance_map](docs/reports/ice_vibration_spectral/figures/fig5_resonance_map.png)

*图 5 共振风险图：f_ice(U) 扫过各阶模态半功率带（左）与分级结果（右）。*

![fig6_hull_chain](docs/reports/ice_vibration_spectral/figures/fig6_hull_chain.png)

*图 6 案例 B 船首加速度响应 PSD 与各阶模态频率。*

![fig7_grid_study](docs/reports/ice_vibration_spectral/figures/fig7_grid_study.png)

*图 7 Welch 网格分辨率相对结构半功率带宽的响应 RMS 偏差（分辨率坑的定量演示）。*

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **网格分辨率坑**：响应 RMS 由细网格（64×）积分给出；若直接在 Welch 网格上积分 $|H|^2S_{ff}$，偏差随 $\Delta f/(2\zeta f_n)$ 变化（图 7）：最细档 nperseg=32768（Δf=0.0153 Hz，带宽比 0.35）偏差 +4.2 %，最粗档 nperseg=1024（Δf=0.488 Hz，带宽比 11.10）偏差 -61.4 %。当前算例谱/时域差异 2.30 %。此类偏差会伪装成「物理放大」。
2. **共振峰识别**：响应 PSD 峰位（1.0987 Hz）落在 f_ice 与 f_n 之间，因为激励窄带与结构窄带部分重叠；判据必须用 $f_{ice}$ 与 $f_n$ 直接比较，不能仅凭响应谱峰反推激励频率。
3. **锁频建模的简化**：本案锁频通过「破碎周期抖动收窄」参数化（Määttänen 1978 的速度夹带机制），未求解速度弱化冰力的自激微分方程（该能力在 `ice_vibration/lockin.py`，本节点不重复）。因此放大倍数 2.06× 是**下界量级的演示**，不是锁频极限幅值预测。
4. **激励模型层级**：$f_{ice}=U/s$ 的一维间歇破碎参数化，不是 TensorDEM 颗粒接触直接输出；接口契约：任何 DEM 冰载荷时序只要给出 `(t, F)` 等间隔采样即可经 `welch_psd` 接入本管线（`excitation.intermittent_crushing_series` 是可复现的合成替代）。
5. **验证层级**：案例 A 只验证**频率**（文献仅公开频带 1.0–1.2 Hz，原始时程未公开），**幅值未验证**；案例 B 无实船实测对照，仅作 MDOF 管线与判据扫图的演示。
6. **适用范围**：线性粘性阻尼、小变形、单点激励；未含非线性冰-结构耦合、水动附加质量与船体梁剪切变形（案例 B 为集中质量近似）。

## 6. 误差判定（逐项，显式回答是否 <3 %）

- **SDOF 固有频率**：相对误差 0.040 % → 是否 <3 %：**是**
- **MDOF 特征频率（3/5/8 DOF）**：相对误差 2.27e-15 (<1e-2 %) → 是否 <3 %：**是**
- **SDOF 频响幅值（全频段）**：相对误差 2.76e-15 (<1e-2 %) → 是否 <3 %：**是**
- **SDOF 频响相位**：相对误差 1.27e-15 (<1e-2 %) → 是否 <3 %：**是**
- **SDOF 峰值幅值**：相对误差 0.024 % → 是否 <3 %：**是**
- **MDOF 频响（模态 vs 直接）**：相对误差 4.14e-14 (<1e-2 %) → 是否 <3 %：**是**
- **白噪声响应 RMS（谱积分）**：相对误差 1.24e-06 (<1e-2 %) → 是否 <3 %：**是**
- **白噪声响应 RMS（时域）**：相对误差 0.060 % → 是否 <3 %：**是**
- **案例 A 谱/时域 RMS 交叉校核**：相对误差 2.298 % → 是否 <3 %：**是**
- **AC5 响应主频 vs 文献**：相对误差 0.122 % → 是否 <3 %：**是**

**免责声明**：案例 A 的响应**幅值**无公开实测可对标，故未列入误差表；判据 AC5 的容差为 20 %（预注册），实测 0.122 %，满足且 <3 %。

## 7. 审计链

- 模块：`src/tensorphysics/ice_vibration_spectral/`（excitation / structural / response / criterion / case）
- 测试：`tests/test_ice_vibration_spectral.py`（18 项，含 AC1–AC5、守恒/归一化与 FEM 模态基一致性校核）
- 载荷数据：`node_payload.json`（sha256 见 provenance），图表 `figures/`
- 生成器：`generate_p233w2_n07_ice_vibration.py`（数字全部自 payload 注入）
- 全部判据：ALL PASS

### 参考文献

- Norrström, S. (1978) *Ice-induced vibration of a lighthouse*, POAC.
- Kärnä, T., Turunen, R. (1990) *A straightforward technique for analysing structural response to dynamic ice action*, IAHR Ice Symposium.
- Määttänen, M. (1978) *On conditions for the rise of self-excited ice-induced motional forces*, IAHR Ice Symposium.
- Timco, G.W., Weeks, W.F. (2010) *A review of the engineering properties of sea ice*, Cold Reg. Sci. Technol. 60, 107–129.
- Clough, R.W., Penzien, J. (1993) *Dynamics of Structures*, 2nd ed.
- Rice, S.O. (1944–45) *Mathematical analysis of random noise*.
