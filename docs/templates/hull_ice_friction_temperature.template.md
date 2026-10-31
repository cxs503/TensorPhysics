# 船体-冰摩擦系数温度依赖性：标定与验证（W3-N07）

> 模块：`src/tensorphysics/hull_ice_friction/`
> 产物：`artifacts/hull_ice_friction/calibration_report.json` + `artifacts/hull_ice_friction/figures/*.png`
> 复现：`python examples/run_hull_ice_friction_calibration.py --output artifacts/hull_ice_friction`
> 文档生成：`python examples/render_hull_ice_friction_tables.py`（本文件由模板 + JSON 自动生成，数字零手抄）
> 测试：`pytest tests/test_hull_ice_friction.py`（{{n_tests}} 用例全绿）
> 证据等级：**REAL**（物理接触律 + 真实数值实验 + 可复现判据）。参考数据为**文献锚定中值基准**，
> 不是受控实验原始数据，见 §7 局限。

---

## 1 目标与判据

在 TensorPhysics 中建立船体-冰（钢-冰）摩擦系数与温度的函数关系

$$\mu(T)=\mu_0+\alpha\,|T-T_{melt}|^{\beta}$$

并以**冰-钢滑动摩擦数值实验**给出宏观标定与验证。机器可算判据：

| 判据 | 阈值 | 本报告结果 | 判决 |
|---|---|---|---|
| (a) μ(T) 幂律拟合 R² | > {{r2_threshold}} | {{r2}} | {{pass_a}} |
| (b) 逐点 μ 相对误差（vs 实验数据，5 点） | < {{err_threshold_pct}} | 最大 {{max_rel_err_pct}}（{{max_rel_err_temp}} °C） | {{pass_b}} |
| (c) 收敛性（FD 网格 / 不动点 / 集合规模 / 重复数） | 见 §6 | 全部达标 | {{pass_c}} |
| (d) pytest | ≥ 10 通过 | {{n_tests}} passed | {{pass_d}} |

判据数字与产物路径全部写入 `calibration_report.json` 的 `criteria` / `verdict` 字段，
由程序判定，不接受人工改数。

---

## 2 物理背景与参数

### 2.1 工程背景

冰区船舶（含破冰船）船体外板与冰排的相对滑移摩擦系数直接决定冰阻力中的摩擦分量
（破冰阻力—摩擦项、冰排沿船壳滑移）以及冰载荷极值。摩擦系数随温度强相关：暖端（接近熔点）
接触斑点上出现融水膜，μ 可降至 0.05–0.1；冷端（-25 °C 以下）冰的剪切强度上升、融膜消失，
μ 升至 0.3–0.45。工程上常用 μ=0.1–0.3 的常数取值，忽略了这一温度依赖，是冰阻力预报
（如 Lindqvist / Riska 系列公式）的主要不确定度来源之一。

### 2.2 几何与工况（数值等效于实验室“钢滑块在冰面恒速滑移”试验）

| 项目 | 取值 | 依据 / 说明 |
|---|---|---|
| 名义接触压力 p_nom | {{p_nom_MPa}} MPa | 船体-冰典型量级 0.5–2 MPa |
| 滑移速度 v | {{velocity}} m/s | 船体相对冰面滑移 0.5–2 m/s 的中值 |
| 温度扫描 | {{temperature_list}} °C | 过冷度 θ = \|T−T_melt\| = {{theta_range}} K |
| 重复次数 | 每个温度 {{n_reps}} 次 | 独立随机微凸体集合 → 均值/标准差/标准误 |
| 微凸体数量 | {{n_asperities}} | 集合规模收敛性见 §6.2 |
| 冰类型 | 淡水冰（T_melt = {{t_melt}} °C） | 模型支持盐度（-0.054 °C/psu），本次验证用淡水冰 |
| 钢侧热物性 | k={{steel_k}} W/mK, ρ={{steel_rho}} kg/m³, c={{steel_c}} J/kgK | 低合金高强钢 |
| 冰侧热物性 | k={{ice_k}} W/mK, ρ={{ice_rho}} kg/m³, c={{ice_c}} J/kgK | 冰的热物性文献值 |

### 2.3 冰力学性质的温度相关性（任务要求 (2)）

<!-- AUTO_MATERIALS -->

E_ice(T) 通过赫兹接触 `a ∝ (F/E*)^{1/3}` 反馈到接触斑尺寸，H(T) 作为微凸体局部压力上限，
二者把“温度 → 冰力学性质 → 接触状态”的链条显式接入摩擦计算（不是装饰项，见 §5）。

---

## 3 方法

### 3.1 接触律（任务要求 (3)）

摩擦系数分解为并联的两承载机制（Bowden–Tabor 型面积加权）：

$$\mu(\theta)=(1-\lambda)\,\underbrace{\frac{\tau_s(\theta)}{p_{real}}}_{\mu_{dry}}+\lambda\,\mu_{wet}$$

* **干摩擦分支** μ_dry = τ_s(θ)/p_real：低速滑移下冰由蠕变/界面剪切破坏主导，随过冷度增大而上升；
* **融水润滑分支** μ_wet = {{mu_wet}}（文献 0.02–0.05，文献给定、不标定）；
* **润滑覆盖度** λ 由每个微凸体界面闪温 ΔT_i 与当地熔点的比较给出

$$s_i=\frac{1}{1+\exp[-(\Delta T_i-\theta)/w]},\qquad \lambda=\frac{\sum_i A_i s_i}{\sum_i A_i}$$

  其中 w = {{w}} K 为融膜转变宽度，A_i ∝ a_i² 为接触斑点面积。

### 3.2 界面闪温（热-力耦合）

摩擦热流 q_i = μ_i p_i v，接触时间 t_c,i = 2a_i/v。两体接触的界面温升（Blok）：

$$\Delta T_i=\frac{2q_i}{S}\sqrt{\frac{t_c}{\pi}},\qquad S=\sqrt{k\rho c}\big|_{ice}+\sqrt{k\rho c}\big|_{steel}$$

由此**热分配系数 ζ_ice = √(kρc)_ice / S = {{zeta_ice}}**：摩擦热只有 {{zeta_ice_pct}} % 进入冰侧，
其余由钢侧导走。这一点决定了闪温量级（§5.1 的机理发现），是本工作最容易被忽略的物理约束。

μ_i 与 s_i 相互依赖（摩擦生热 → 闪温 → 润滑 → 摩擦），每个微凸体需解不动点：

$$\mu_i=(1-s_i(\mu_i))\mu_{dry}+s_i(\mu_i)\mu_{wet}$$

`μ − F(μ)` 在 [μ_wet, μ_dry] 上单调，根唯一；生产求解器用**对称二分**（区间宽度逐次减半，
无条件收敛），并保留带松弛 Picard 作对照（§6.3 记录其极限环现象）。

### 3.3 微凸体集合与数值实验（任务要求 (1)）

* 接触斑点半径：对数正态，中值 {{a_median}} mm，σ_log = {{a_sigma}}，按 E_ice(T) 赫兹缩放；
* 局部压力：对数正态，中值 {{ck}}·p_real，σ_log = {{p_sigma}}，按 H(θ) 截断（物理上限）；
* 集合规模 N = {{n_asperities}}，每个温度 {{n_reps}} 次独立重复（不同种子）；
* 面积加权给出 macroscale μ，同时输出 λ、闪温统计、迭代次数、收敛标志。

### 3.4 标定策略（防“对着答案调参”）

参数分层，避免把验证数据当训练数据的循环论证：

| 参数 | 角色 | 来源 |
|---|---|---|
| σ_c(θ)、τ_s/σ_c、E(θ)、H(θ) | 文献固定 | 冰力学经验关联式 |
| μ_wet、w、集合统计、v、p_nom | 文献/工况固定 | 文献与船体-冰工况 |
| **p_real（真实接触压力）** | **唯一标定量** | 由标定子集 {{calib_temps}} °C 辨识 |
| {{valid_temps}} °C | **留出验证点** | 不参与辨识，仅检验预测能力 |

识别结果：p_real = {{p_real}} MPa（真实接触面积分数 A_r/A_n = p_nom/p_real = {{area_frac}}），
目标函数（相对残差平方和）在最优值附近呈单峰，见报告 JSON `objective_curve`。
p_real 用 log 空间黄金分割最小化相对残差平方和（集合样本库预抽一次，标定迭代只改本构参数，
避免蒙特卡洛噪声掩盖目标函数梯度）。

### 3.5 验证数据（文献锚定基准）

`reference_data.py` 记录 5 个温度点的文献中值与跨文献离散度（相对 {{ref_unc_min_pct}}–{{ref_unc_max_pct}} %），
并显式声明“原始文献表格在离线沙箱不可获取、数值为报道区间中值”这一局限（§7）。
温度 {{valid_temps}} °C 为留出点。

---

## 4 计算结果

<!-- AUTO_TABLES -->

---

## 5 关键发现

### 5.1 温度依赖的主要载体是“干摩擦分支”，不是融水润滑

由 §3.2，ζ_ice = {{zeta_ice}} 使界面闪温仅为 {{flash_warm}} K（{{t_warm}} °C）～{{flash_cold}} K（{{t_cold}} °C）
量级（面积平均，p90 分别为 {{flash_warm_p90}} K / {{flash_cold_p90}} K），相对过冷度 {{theta_range}} K 而言偏小，
因此融膜覆盖度 λ 只有 {{lam_warm}}（{{t_warm}} °C, v={{velocity}} m/s）→ {{lam_cold}}（{{t_cold}} °C）。
结果是：**在低速（{{velocity}} m/s 量级）蠕变主导区间，μ(T) 的温度依赖主要来自冰的界面剪切强度
τ_s(θ) 上升，融水润滑只是暖端的小幅修正**（贡献量见表 4-2 的 `μ_dry → μ` 差值）。
这与“暖冰靠水膜润滑”的直觉相反，但与“低速时闪温不足以形成连续融膜、μ≈τ_s/p_real”的经典图像一致。

### 5.2 高滑移速度下润滑机制接管（机理自洽性检验）

速度敏感性扫描（v = {{v_sweep}} m/s）显示：同一温度下 λ 随 v 单调上升
（{{t_warm}} °C：{{lam_v_min}} → {{lam_v_max}}），μ 随 v 单调下降（{{mu_v_min}} → {{mu_v_max}}）。
这是模型对“高速冰摩擦进入融水润滑区、摩擦系数明显下降”这一实验趋势的独立复现，
说明热-力耦合项是有效的而非装饰项（图 4）。

### 5.3 参考数据的不确定度大于验收阈值

参考表跨文献离散度 {{ref_unc_min_pct}}–{{ref_unc_max_pct}} %，而验收阈值是 {{err_threshold_pct}}。
因此“逐点误差 < {{err_threshold_pct}}”检验的是**仿真曲线与文献中值的中心趋势一致性**，
不能替代与受控实验（同冰种、同粗糙度、同速度）的同条件对比。这一结论写进
`calibration_report.json` 的 `reference_data_provenance.limitations` 与 §7，不做粉饰。

---

## 6 收敛性与数值证据（任务要求 (c)）

### 6.1 一维瞬态导热有限差分 vs 解析解

独立数值解（隐式 Euler + 追赶法，无量纲化 θ_Fo = θ_ξξ，表面热流边界）对解析解
`θ(0,Fo)=2√(Fo/π)` 的收敛性见下表：最细网格相对误差 {{fd_finest}}，相邻网格观测收敛阶
{{fd_order_range}}（理论：空间二阶 + ΔFo ∝ Δξ² 的时间一阶 → 综合二阶）。
这验证了解析闪温公式（进而整个接触律的热学部分）的实现正确性。

<!-- AUTO_FD -->

### 6.2 集合规模与重复次数

* 集合规模：N = {{ens_sizes}}，生产规模 N={{n_asperities}} 相对 N={{ens_finest}} 的 μ 偏差 {{ens_dev_pct}}（阈值 {{ens_threshold_pct}}）；
* 重复次数：{{reps_temp}} °C 下累计均值随重复次数稳定，{{reps_max}} 次重复的标准误 {{stderr_final}}（≈{{stderr_final_pct}} % of μ），
  生产配置 {{n_reps}} 次重复的标准误为 {{stderr_prod_min_pct}}–{{stderr_prod_max_pct}} %，远小于 {{err_threshold_pct}} 判据。

<!-- AUTO_CONV -->

### 6.3 局部不动点求解器：二分 vs 松弛 Picard（真实数值现象，如实记录）

* 二分求解器：5 个温度点全部收敛，最大最终残差 {{fp_residual}}，迭代次数 {{fp_iters}}（区间逐次减半）；
* 松弛 Picard（ω={{omega}}）：{{picard_fail_temps}} 出现极限环，{{picard_iter}} 次迭代不收敛。
  定位到触发因素是“高闪温接触斑点”（a ≳ 1 mm 且 p ≳ 30 MPa）：此时 s(μ) 的强非线性反馈
  使 Picard 映射的谱半径 > 1。生产代码因此默认二分求解器，并保留 `solver="picard"` 供对照，
  该现象由 `tests/test_hull_ice_friction.py::TestSolverConvergence` 固化。

---

## 7 注意点、局限与适用范围

**踩过的坑（写进 KNOWN_TRAPS 供后来者）**

1. **热分配系数不能省**：忽略 Blok 分配（ζ_ice={{zeta_ice}}）会把界面闪温高估约 {{partition_error_factor}} 倍，
   导致几乎所有温度下都“融化润滑”，μ(T) 曲线被压平（早期原型中 λ 在 -25 °C 仍达 0.27–
   0.49，逐点误差最大 24 %–35 %，验证不通过）。加回分配后才落到 {{max_rel_err_pct}}。
2. **硬度上限的方向性**：把微凸体压力上限设为随冷端升高的 H(θ) 会部分抵消温度效应
   （冷端压力更高 → 闪温更高），必须与 τ_s(θ) 分支一起看，否则 μ(T) 会过于平坦
   （早期原型 λ 在 -25 °C 压不下去，误差 24 %+）。
3. **Picard 迭代的极限环**：见 §6.3。修法不是调 ω，而是换无条件收敛的区间方法。
4. **标定-验证必须分开**：单参数 p_real 只用 {{calib_temps}} °C 三个点辨识，另 {{valid_temps}} °C 留出，
   否则 5 点全用就是对着答案调参，验证信息量为零。

**局限**

* **参考数据**：文献锚定中值，非原始实验表格（离线沙箱不可获取原文数据）；跨文献离散度
  {{ref_unc_min_pct}}–{{ref_unc_max_pct}} %，故 {{err_threshold_pct}} 判据只检验中心趋势；
* **热学模型**：一维半无限体假设，忽略有限试样/夹具的整体升温（长时间滑动会抬高 λ，
  因此本报告的 μ 是偏“干”的估计）；未含接触压力导致的熔点下降（≈0.0074 K/MPa，量级 <1 K）；
* **力学模型**：干摩擦分支用常数 p_real（真实冰的蠕变会使 p_real 随应变率/温度变化）；
  未含磨损、破碎、犁沟与粘滑项；
* **工况**：仅 5 个温度点、1 个标定参数，参数冗余度低；温度区间 {{temperature_range}} °C、速度
  {{v_sweep}} m/s 内验证，区间外为外推；
* **盐度**：模型支持熔点降低（-0.054 °C/psu，35 psu → -1.89 °C），但本次验证用淡水冰，
  海冰情形未做定量验证。

**适用范围**：船体-冰低速滑移摩擦（{{v_sweep}} m/s、{{temperature_range}} °C、0.5–2 MPa 名义压力）的
温度依赖趋势估计与冰阻力摩擦项的 μ(T) 取值；不适用于冰破碎、粘滑振动、极低速静摩擦
（stiction）与高温融化层主导的极端高速工况。

---

## 8 误差判定（逐项，显式回答“是否 < 3 %”）

<!-- AUTO_ERRORS -->

---

## 9 复现与审计链

| 项目 | 路径 |
|---|---|
| 模块 | `src/tensorphysics/hull_ice_friction/{ice_materials,thermal_contact,contact_law,experiment,calibration,reference_data,validation}.py` |
| 端到端脚本 | `examples/run_hull_ice_friction_calibration.py` |
| 判据产物（含全部数字） | `artifacts/hull_ice_friction/calibration_report.json` |
| 图 | `artifacts/hull_ice_friction/figures/{mu_vs_temperature,mechanism_branches,convergence,velocity_sensitivity}.png` |
| 测试 | `tests/test_hull_ice_friction.py`（{{n_tests}} passed） |
| 文档生成 | `examples/render_hull_ice_friction_tables.py`（模板 + JSON → 本文件，数字零手抄） |

全程无外部数据下载、无硬编码结果；所有数字由 `run_validation()` 计算后写入 JSON，
本文表格与正文占位符由 `render_hull_ice_friction_tables.py` 从 JSON 自动填充。

## 4' 图表

![μ(T) 标定与验证](artifacts/hull_ice_friction/figures/mu_vs_temperature.png)

**图 1** 数值实验 μ(T)（红色，{{n_reps}} 次重复 ± 标准误）、幂律拟合曲线（蓝色，
μ={{mu0}}+{{alpha}}·θ^{{beta}}，R²={{r2}}）与文献锚定参考中值（灰色，误差棒为跨文献离散度）。
浅红色竖带为标定子集（{{calib_temps}} °C），其余为留出验证点。

![机理分解](artifacts/hull_ice_friction/figures/mechanism_branches.png)

**图 2** 左：干摩擦分支 τ_s(θ)/p_real、混合律结果与融水润滑折减量；右：面积加权融膜覆盖度 λ、
界面闪温（均值/p90）与过冷度 θ。可见 λ 在冷端趋于 0，温度依赖由干分支承担。

![收敛性](artifacts/hull_ice_friction/figures/convergence.png)

**图 3** 收敛性证据：(a) 瞬态导热 FD 对解析解误差随网格加密（观测阶≈2）；(b) μ 随集合规模；
(c) 局部不动点残差——生产二分求解器（实线）区间逐次减半，松弛 Picard（点线）在高闪温斑点处进入极限环；
(d) 重复次数累计均值与标准误。

![速度敏感性](artifacts/hull_ice_friction/figures/velocity_sensitivity.png)

**图 4** 滑移速度敏感性（v = {{v_sweep}} m/s）：摩擦生热抬高 λ、降低 μ，复现
“高速滑移把冰摩擦推入融水润滑区”的实验趋势（即使环境温度很冷）。