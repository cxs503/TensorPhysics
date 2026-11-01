# 冰-结构接触相变与融膜演化成熟化（W8-N01 / m1212）

**仓库**：TensorPhysics → 新领域模块 `src/tensorphysics/ice_contact/`
**分支**：`agent/m1212`  **任务**：W8-N01 冰-结构接触相变与融膜演化（压力融化 + 摩擦融化 + 融膜厚度 ODE + 冰粘附温度依赖 + Stribeck 扩展）
**产物**：`artifacts/ice_contact_m1212/results.json`、`pytest.log`、`figures/*.png`、`tests/test_ice_contact_*.py`、本报告

---

## 1. 案例介绍

### 1.1 物理背景

冰与结构（钢）滑动接触界面存在**三相结构**：冰 — 融膜（水） — 结构。界面热力学与摩擦学双向耦合，三种相变机制相互竞争：

1. **压力融化**：接触压力使冰 Ih 的熔点降低（Clausius-Clapeyron 反常斜率为负），高压下本"很冷"的冰相对当地熔点变成过热态，无需摩擦热即可融化；
2. **摩擦融化**：摩擦功率 tau*V 转为热流密度，一部分转为潜热使冰融化，另一部分经 Stefan 条件向冰体与结构导热；
3. **再冻结**：当导热损失超过摩擦热时，融膜向冷冰体/冷结构凝固（Stefan 移动边界）。

三种机制决定融膜厚度 h，融膜厚度经润滑状态（膜厚参数 Lambda = h/sigma_rms）决定摩擦系数，摩擦系数又反馈回摩擦热——构成闭环。工程意义覆盖冰区航行船体冰载荷、破冰结构接触摩擦、除冰/防冰表面设计。

### 1.2 模型几何与参数

单节点（0-D）接触控制体：接触长度 L，法向名义压力 p，滑移速度 V，冰温 T_i，结构表面温度 T_s。冰侧与结构侧各取半无限体热渗透深度 delta = sqrt(alpha*L/V)（一次接触驻留时间 L/V 内的导热尺度）。

热物性取值（`properties.py`，0 °C 参考值）：

| 参数 | 符号 | 取值 | 单位 |
|---|---|---|---|
| 冰密度 | ρ_i | 916.8 | kg/m³ |
| 水密度 | ρ_w | 999.8 | kg/m³ |
| 冰/水比热 | c_i / c_w | 2050 / 4186 | J/kg/K |
| 冰/水导热 | k_i / k_w | 2.22 / 0.561 | W/m/K |
| 融化潜热 | L_f | 3.3355e5 | J/kg |
| 冰熔点（常压） | T_m0 | 273.15 | K |
| 钢 (rho/c/k) | — | 7850 / 475 / 45 | — |

### 1.3 控制方程

- 压力融化： dT_m/dp = T_m*(v_w - v_i)/L_f，精确积分 T_m(p) = T_m0 * exp[dv*(p-p0)/L_f]；
- 融膜 ODE（质量 + 能量守恒）：

      dh/dt = (q_fric + q_ice + q_struct)/(rho_i*L_f) - c_d*h*V/L
      q_fric  = mu(h,p,V,T)*p*V
      q_ice   = k_i*(T_i - T_m(p))/sqrt(alpha_i*L/V)
      q_struct= (T_s - T_m(p))/(h/k_w + delta_s/k_s)

  正号表示向融膜供热（融冰），负号表示膜失热（再冻结）；末项为融膜被拖曳/挤压排出接触区的质量平衡；
- 摩擦闭合（扩展 Stribeck）：mu = A_r(Lambda)*mu_b + (1-A_r)*eta(T)*V/(p*h)，A_r = exp(-Lambda)，Lambda = h/sigma_rms，并设残余润滑摩擦下限 mu_floor；
- 冰粘附：sigma_adh(T) 由界面准液层厚度 h_QLL = A*(T_m - T)^(-1/3) 与经验强度律双分支给出（-10 °C 峰值 0.9 MPa）。

## 2. 方法 / 流程 / 工具链

1. 新建领域包 `src/tensorphysics/ice_contact/`（properties / pressure_melting / stefan / film / adhesion / tribology / competition / verification），纯 Python 数学核心，仅 1-D Stefan 参考解需要 numpy；
2. 每个物理关系都配一个**独立参考解**：
   - (a) Clausius-Clapeyron ODE 的 RK4 数值积分 vs 指数精确积分；
   - (b) 融膜 ODE 的**定常点二分求解** vs **手推二次型闭式解**（两者互不调用）；
   - (c) 1-D **焓法**（有限体积显式）vs **Neumann 相似解** X = 2*lambda*sqrt(alpha*t)；
   - (d) 参数空间扫描 + 5 条物理方向门；
3. `benchmarks/ice_contact/run_verification.py` 汇编全部判据 → `results.json` + 7 张图；
4. `benchmarks/ice_contact/render_report.py` 从 `results.json` **程序化注入**全部数字生成本报告（零手抄）；
5. pytest：53 例全绿（判据 f，阈值 18 例）。

## 3. 计算结果

### 3.1 判据总表

| 判据 | 内容 | 阈值 | 实测误差 | 结论 |
|---|---|---|---|---|
| (a) | 纯压力融化温度 vs CC 解析解 | <1 % | 1.5e-13 % | PASS |
| (b) | 纯摩擦融化稳态膜厚 vs 解析解 | <5 % | 0.0022 % | PASS |
| (c) | 再冻结速率 vs Stefan 解析解 | <5 % | 1.275 % | PASS |
| (d) | 三机制竞争 + 物理方向门 | 全门通过 | 1.774 % (onset) | PASS |
| (e) | 端到端 1 MPa/0.1 m/s/-10 °C 输出时历 | 四算例可用 | — | PASS |
| (f) | pytest 用例数 >= 18 | >=18 | 53 passed / 0 failed | PASS |

### 3.2 判据 (a) 压力融化（Clausius-Clapeyron）

斜率 dT_m/dp = **-0.0741533 K/MPa**，文献值 -0.0742 K/MPa，相对误差 0.063 %。

| p [MPa] | T_m 数值积分 [K] | T_m 解析 [K] | 相对误差 | 熔点降 [K] |
|---|---|---|---|---|
| 1 | 273.0759 | 273.0759 | 2.1e-14 % | 0.07414 |
| 10 | 272.4095 | 272.4095 | 8.3e-14 % | 0.7405 |
| 50 | 269.4674 | 269.4674 | 1.5e-13 % | 3.683 |
| 100 | 265.8344 | 265.8344 | 1.1e-13 % | 7.316 |

-10 °C 冰的**压力融化阈值压力** = **137.4 MPa**（1 MPa 名义接触仅降 0.0741 K，故压力融化只可能在**粗糙峰尺度**（1e2 MPa 量级）起作用）。

### 3.3 判据 (b) 纯摩擦融化稳态膜厚

两类比较：定摩擦系数闭式解，以及自洽（mu = mu(h)）耦合定常点 vs 二次型闭式解；另外从三个初值做 RK4 瞬态积分，验证收敛到同一解析定常态。

| 模式 | p [MPa] | V [m/s] | μ | 数值膜厚 [m] | 参考膜厚 [m] | 相对误差 |
|---|---|---|---|---|---|---|
| pure_friction_melting_closed_form | 1 | 1 | 0.05 | 1.635e-05 | 1.635e-05 | 0.0e+00 % |
| pure_friction_melting_closed_form | 1 | 4 | 0.05 | 1.635e-05 | 1.635e-05 | 0.0e+00 % |
| pure_friction_melting_closed_form | 2 | 2 | 0.05 | 3.27e-05 | 3.27e-05 | 0.0e+00 % |
| pure_friction_melting_closed_form | 4 | 8 | 0.05 | 6.54e-05 | 6.54e-05 | 0.0e+00 % |
| coupled | 1 | 1 | 0.0655 | 4.54e-07 | 4.54e-07 | 0.002 % |
| coupled | 1 | 4 | 0.03574 | 1.205e-06 | 1.205e-06 | 0.000 % |
| coupled | 2 | 2 | 0.01417 | 2.019e-06 | 2.019e-06 | 0.000 % |
| coupled | 4 | 8 | 0.013 | 2.295e-06 | 2.295e-06 | 0.000 % |

瞬态收敛（初值 → 最终膜厚 vs 闭式解）：

| 初始膜厚 [m] | 积分终值 [m] | 解析定常 [m] | 相对误差 |
|---|---|---|---|
| 5e-08 | 1.205e-06 | 1.205e-06 | 0.000 % |
| 2e-07 | 1.205e-06 | 1.205e-06 | 0.000 % |
| 1e-06 | 1.205e-06 | 1.205e-06 | 0.000 % |

### 3.4 判据 (c) 再冻结 vs Stefan 解析解

Delta T = 10 K，Ste = 0.1255，Neumann 常数 lambda = 0.2455。终时刻（1 ms）冻结层厚度与网格收敛：

| 网格数 | dx [m] | dt [s] | 步数 | 数值冻结厚 [µm] | Neumann [µm] | 最大相对误差 |
|---|---|---|---|---|---|---|
| 150 | 2.67e-07 | 2.12e-07 | 4715 | 5.814 | 5.685 | 4.91 % |
| 300 | 1.33e-07 | 5.31e-08 | 18855 | 5.749 | 5.685 | 2.47 % |
| 600 | 6.67e-08 | 1.33e-08 | 75405 | 5.718 | 5.685 | 1.28 % |

误差随网格加密单调下降 → 一阶收敛，最细网格误差 1.275 %（150 格 4.907 %）。

### 3.5 判据 (d) 三机制竞争与物理方向门

| 门 | 结论 | 判据内容 |
|---|---|---|
| G1_friction_melting_V | PASS | film thickness must increase with sliding speed above the melting onset |
| G2_load_p | PASS | film thickness must increase with contact pressure |
| G3_thermal_T | PASS | colder ice must thin the steady film (Range -5..-30 degC) and raise friction monotonically; the thickness plateau below -30 degC is the friction self-regulation |
| G4_pressure_melting | PASS | pressure melting must be negligible at 1 MPa and active above ~135 MPa at -10 degC |
| G5_mechanism_coverage | PASS | pressure melting, friction melting and refreezing must each dominate somewhere |

融化起始速度（等温滑块）数值二分 vs 闭式解 V_c = (-(Delta_i + Delta_s)/(mu*p))^2：

| p [MPa] | V_c 数值 [m/s] | V_c 闭式 [m/s] | 相对偏差 |
|---|---|---|---|
| 0.5 | 1.445 | 1.657 | 12.76 % |
| 1 | 0.4038 | 0.4111 | 1.77 % |
| 2 | 0.1012 | 0.1012 | 0.05 % |
| 4 | 0.0246 | 0.02455 | 0.17 % |

四压力最大偏差 12.76 %（p = 1 MPa 处 1.77 %）：闭式解取干摩擦系数 mu_b，而闭合在膜底还含一项随 V 增长的粘性份额，故 V_c 越大（p 越小）偏差越大，方向一致。

### 3.6 判据 (e) 端到端算例

**主算例（任务指定）：1 MPa / 0.1 m/s / -10 °C，冷滑块**

- 熔点降 0.07414 K，冰粘附强度 0.9 MPa；
- 稳态：**无持久融膜**（exists = False），摩擦落在边界润滑支 mu = 0.1001；
- 1 µm 初始融膜的**寿命 = 2.17 ms**，随后进入干接触/边界分支；
- 机制速率（m/s）：摩擦融化 3.27e-05、压力融化 4.95e-07、**再冻结 0.000487** —— 再冻结主导；
- 该工况等温滑块下的融化起始速度 V_c = 0.4038 m/s，指定速度 0.1 m/s **低于** V_c，故模型判定为边界摩擦支——与冰摩擦文献中"低速高摩擦、高速才进入融化润滑"的**速度弱化**规律一致。

四个算例汇总（含补充的摩擦融化 / 压力融化 / 再冻结分支）：

| 算例 | p [MPa] | V [m/s] | T [°C] | 热平衡滑块 | 有稳态膜 | h_ss [µm] | μ_ss | 润滑区 | 膜寿命 [ms] |
|---|---|---|---|---|---|---|---|---|---|
| specified_1MPa_0p1mps_253K | 1 | 0.1 | -10 | no | no | 0 | 0.1001 | boundary | 2.17 |
| friction_melting_1MPa_1mps_253K | 1 | 1 | -10 | yes | yes | 0.454 | 0.0655 | boundary | — |
| pressure_melting_20MPa_5mmps_272p5K | 20 | 0.005 | -0.65 | yes | yes | 57.5 | 0.005 | hydrodynamic | — |
| refreezing_1MPa_1mps_253K_cold_slider | 1 | 1 | -10 | no | no | 0 | 0.1024 | boundary | 0.8 |

机制速率分解（m/s）：

| 算例 | 摩擦融化 | 压力融化 | 再冻结 | 排出 | dh/dt |
|---|---|---|---|---|---|
| specified_1MPa_0p1mps_253K | 3.27e-05 | 4.95e-07 | 0.000487 | 1e-09 | -0.000454 |
| friction_melting_1MPa_1mps_253K | 0.000214 | 1.57e-06 | 0.000211 | 4.54e-06 | 9.77e-11 |
| pressure_melting_20MPa_5mmps_272p5K | 1.64e-06 | 2.21e-06 | 9.71e-07 | 2.87e-06 | 7.92e-15 |
| refreezing_1MPa_1mps_253K_cold_slider | 0.000335 | 1.57e-06 | 0.00154 | 1e-08 | -0.0012 |

### 3.7 结果图

### 图 1 Clausius-Clapeyron 压力融化线：精确积分 vs 切线近似，标出 -10 °C 冰的融化阈值压力

![fig1_pressure_melting](figures/fig1_pressure_melting.png)
### 图 2 再冻结 Stefan 移动边界验证：Neumann 相似解 vs 三种网格的焓法解

![fig2_stefan_refreezing](figures/fig2_stefan_refreezing.png)
### 图 3 稳态融膜厚度与摩擦系数随滑移速度的变化（-10 °C，滑块已热平衡）

![fig3_steady_film_map](figures/fig3_steady_film_map.png)
### 图 4 四个算例的融膜厚度与摩擦系数时历

![fig4_case_histories](figures/fig4_case_histories.png)
### 图 5 三机制竞争参数空间图：压-速平面上主导机制分布（左 -10 °C，右 -0.65 °C）

![fig5_mechanism_map](figures/fig5_mechanism_map.png)
### 图 6 冰粘附强度温度依赖与准液层厚度

![fig6_adhesion](figures/fig6_adhesion.png)
### 图 7 以融膜厚度为坐标的扩展 Stribeck 曲线与润滑区间划分

![fig7_stribeck](figures/fig7_stribeck.png)


## 4. 注意点（踩过的坑 / 局限 / 适用范围）

1. **焓法 + 等温边界对"无过热"融化失效**：把壁面温度钉在 T_m 作 Dirichlet 边界时，首格焓只会从下方渐近趋于融化焓，融化前沿永远进不到邻格（经典焓法病态）。因此 `solve_stefan_1d(mode="melting")` 强制要求过热壁（T_w > T_m），再冻结（判据 (c)）不受影响。这是**求解器施加的物理前提**，不是结果修饰。
2. **结构侧热状态决定融膜存亡**：结构侧 Stefan 汇以"膜热阻 + 半无限结构热阻"串联表示；冷滑块（-10 °C 钢）比热平衡滑块多吸收一个量级的摩擦热，使 1 MPa/1 m/s 工况反而无膜。连续滑行中滑块自身会升温，故报告同时给出等温滑块（equilibrated_structure=True）分支，两者都标注。
3. **排出系数 c_d 与接触长度 L**：融膜厚度正比于 1/c_d；c_d 取 O(1) 是量级假设，未做挤压流（Reynolds）求解，因此**膜厚的绝对值**只具量级意义，趋势与竞争关系才是主张。
4. **高压端外推**：p > 50 MPa 时塞状排出假设不再成立（扫描中最大膜厚达数百 µm），故压力融化分支只主张**机制判定与驱动量**（熔点降、过热度），不主张该处膜厚数值。同理 Clapeyron 在 >50 MPa 忽略 dv、L_f 变化，会低估熔点降（IAPWS 曲线弯曲）。
5. **粘附模型是工程标定**：峰值 0.9 MPa @ -10 °C、脆化段衰减长度 25 K 为冰-碳钢剪切试验量级的经验取值；应变率依赖未建模。准液层 h ~ (Delta T)^(-1/3) 与 vdW 桥接应力（~ Delta T）提供温变**机理**解释。
6. **摩擦闭合是混合润滑载荷分担模型**，不是 Reynolds 方程解；mu_floor 防止高压端给出非物理的无摩擦界面。膜厚参数 Lambda 对粗糙度 sigma_rms 敏感（本报告取 1 µm），故 mu 的绝对值依赖粗糙度设定。
7. **0-D 单节点**：无横向流动、无冰变形、无粗糙峰温度分布，不能替代专业接触/CFD 求解器；本模块定位是 TensorPhysics 编排层的**可审计物理闭合 + 验收夹具**。
8. **git/CI 环境**：本 worktree 中 `tensorphysics` 的可编辑安装指向其它 worktree，故所有命令显式带 `PYTHONPATH=src:.`；报告与图件的生成脚本已固化该路径。

## 5. 误差判定（逐项 vs 基准，显式回答是否 <3 %）

| 关键量 | 基准（独立来源） | 相对误差 | 是否 <3 % |
|---|---|---|---|
| 压力融化温度 T_m(p) (p<=100 MPa) | Clapeyron 精确积分（解析） | 1.5e-13 % | 是（<3 %） |
| Clapeyron 斜率 dT/dp | 文献 -0.0742 K/MPa | 0.063 % | 是（<3 %） |
| 纯摩擦融化稳态膜厚 | 手推二次型闭式解 | 0.0022 % | 是（<3 %） |
| 瞬态积分 → 定常膜厚 | 同一闭式解（自洽 μ） | 0.0002 % | 是（<3 %） |
| 再冻结前沿（最细网格 600 格） | Neumann 相似解 | 1.275 % | 是（<3 %） |
| 再冻结前沿（150 格） | Neumann 相似解 | 4.907 % | 否（≥3 %） |
| 融化起始速度 V_c (1 MPa) | 平方根平衡闭式解 | 1.774 % | 是（<3 %） |
| 融化起始速度 V_c (0.5-4 MPa 最差) | 同上 | 12.76 % | 否（≥3 %） |

结论：**主判据关键量全部 <3 %**（(a) 机器精度、(b) 0.0022 %、(c) 1.275 %、(d) 1.77 %）。两处超 3 % 的项目如实列出：① 150 格粗网格的 Stefan 前沿 4.907 %（网格离散误差，随加密收敛，见 3.4）；② 低压端融化起始速度 12.76 %（闭式解未含膜底粘性份额，方向一致）。二者均**不影响判据 (b)(c)(d) 的通过与结论方向**，不做粉饰。

### 5.1 审计链

- 主判据数字全部来自 `artifacts/ice_contact_m1212/results.json`（由 `benchmarks/ice_contact/run_verification.py` 生成）；
- 测试用例输出：`artifacts/ice_contact_m1212/pytest.log`（53 passed）；图表由脚本重放生成（7 张 PNG）；
- 复现命令：`PYTHONPATH=src:. python3 benchmarks/ice_contact/run_verification.py` → `PYTHONPATH=src:. python3 benchmarks/ice_contact/render_report.py`；
- 未重复使用任何已 verified 案例；本节点全部为全新实现。

### 5.2 未完成 / 明确不做

- 未与实验数据直接比对（本机无冰-钢摩擦实验数据）：判据全部对标解析/相似解，属**代码与物理正确性验证**，非实验标定；
- 未做二维/三维接触、粗糙峰分布、冰变形与破碎耦合；
- 未接入 TensorFEM/TensorLBM 求解器（本模块是闭合与夹具，非新求解器内核）。
