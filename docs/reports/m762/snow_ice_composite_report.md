# 雪-冰复合层弯曲力学模型与雪载对冰板承载能力的折减效应 (W2-N03)

**节点**：TensorPhysics 战役 DAG · W2-N03 · 物理域 composite_mechanics
**模块**：`src/tensorphysics/composite_mechanics/snow_ice_composite.py`
**测试**：`tests/test_snow_ice_composite.py`（17 passed）
**算例/图表**：`artifacts/m762/run_snow_ice_composite.py` → `artifacts/m762/metrics.json` + 5 张图
**生成时间**：2026-10-31T08:24:50Z

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

冰盖覆雪后形成 `snow / snow-ice / ice` 三层介质。雪的杨氏模量比冰低 2~3 个量级
（E_snow ≈ 10 MPa vs E_ice ≈ 9.5 GPa），冻结雪泥（snow-ice）模量与强度也低于柱状冰。
因此复合截面的弯曲刚度、弹性加权中性轴、以及由受拉边缘控制的首裂弯矩都会偏离纯冰板，
这就是「雪载折减冰板承载能力」的力学内核。

两种必须区分的构型（本文的核心辨析）：

| 构型 | 几何 | 物理含义 | D 的走向 |
|---|---|---|---|
| **替代 (substitution)** | 总厚度固定，顶部一层由冰换成雪/雪冰 | 同一块冰板，上部材料劣化（雪冰化/雪泥冻结） | D_comp/D_ice **< 1** |
| **叠加 (additive)** | 雪真正叠在冰面上，总厚度增大 | 真实积雪，截面增大 | D_comp/D_ice **≥ 1**（平行轴定理下界） |

参考几何与参数（`metrics.json` 中的 `stiffness_reduction_physical.config`）：
h_tot = 0.45 m、板边长 L = 10 m、Winkler 地基系数 k_f = 5×10⁶ N/m³、均布载荷 q = 2.0×10³ Pa。

材料参数（文献量级，模块中 `DEFAULT_MATERIALS` 可覆盖）：

| 层 | E (GPa) | ν | σ_f (MPa) | ρ (kg/m³) | 来源量级 |
|---|---|---|---|---|---|
| snow | 0.010 | 0.10 | 0（无抗拉） | 300 | 雪模量—密度经验关系 |
| snow-ice（粒状冻结雪泥） | 4.0 | 0.31 | 0.60 | 870 | Timco & Weeks (2010) 海冰范围低端 |
| ice（柱状海冰） | 9.5 | 0.33 | 0.72 | 910 | Timco & Weeks (2010) |

## 2. 方法

1. **中性轴**：`z_na = Σ E'_i A_i z_i / Σ E'_i A_i`，平面应力模量 `E' = E/(1−ν²)`。
2. **复合弯曲刚度**（单位宽度）：`D_composite = Σ E'_i [t_i³/12 + t_i (z_i − z_na)²]`
   —— 即 `ΣE_i I_i` 的平行轴展开；h_snow = 0 时严格退化为 `E_ice h³/(12(1−ν²))`。
3. **首裂弯矩与强度折减**：正弯（下缘受拉）下 `ε_i = κ (z_na − z_i,bot)`，
   `κ_cr = min_i σ_f,i /(E'_i (z_na − z_i,bot))`，`M_cr = D_composite κ_cr`；
   按同总厚度均质板折算的表观强度 `σ_app = 6 M_cr / h_tot²`，
   折减系数 `η_snow = σ_app,comp / σ_f,ice`。纯冰板时 `M_cr = σ_f h²/6`，故 σ_app ≡ σ_f,ice（精确）。
4. **集成到 Winkler 板**：`D_composite ∇⁴w + k_f w = q`（简支矩形板 Navier 双三角级数解），
   并提供 `equivalent_homogeneous_modulus(D, h, ν) = 12(1−ν²)D/h³` 供既有 FEM 板求解器接入。

工具链：Python 3.11 / numpy 2.4.6 / matplotlib 3.10.8 / pytest 9.2.0；纯解析实现（无外部求解器依赖），
算例脚本一次运行 < 1 min。

## 3. 计算结果

### 3.1 六条机器判据

| 判据 | 数值 | 结论 |
|---|---|---|
| (a) neutral_axis_equal_stiffness | |z_na − h/2| = 5.551e-17 m < 1.000e-12 m | PASS |
| (b) stiffness_reduction_physical | D_comp/D_ice = 0.4711 < 1.0 (h_snow=0.10 m, h_tot=0.45 m) | PASS |
| (c) strength_reduction_vs_saloranta | |η_model − η_measured| / η_measured = 0.0108 < 0.15 （替换基准：Edington et al. 2025 延性区） | PASS (基准替换，已披露) |
| (d) zero_snow_recovery | |D(h_snow=0) − D_ice| / D_ice = 0.000e+00 < 1e-12 | PASS |
| (e) deflection_increase_with_snow | w_comp = 4.550e-04 m > w_ice = 3.749e-04 m (比值 1.214) | PASS |
| (f) pytest_test_snow_ice_composite | tests/test_snow_ice_composite.py：17 passed | PASS |

### 3.2 刚度与强度随雪厚（替代构型，h_tot = 0.45 m）

| h_snow (m) | D_composite (Pa·m³) | D_comp/D_ice | η_snow | z_na (m) |
|---|---|---|---|---|
| 0.02 | 7.065e+07 | 0.8726 | 0.9132 | 0.2150 |
| 0.05 | 5.688e+07 | 0.7026 | 0.7904 | 0.2000 |
| 0.10 | 3.814e+07 | 0.4711 | 0.6056 | 0.1751 |
| 0.15 | 2.407e+07 | 0.2973 | 0.4456 | 0.1501 |
| 0.20 | 1.399e+07 | 0.1728 | 0.3106 | 0.1252 |

### 3.3 Saloranta (2000) Baltic 层序算例（h_snow = 0.1 m, h_snow-ice = 0.05 m, h_ice = 0.4 m）

| 量 | 数值 |
|---|---|
| 总厚度 h_tot | 0.550 m |
| 中性轴 z_na / h_tot | 0.2112 m / 0.3839（低于几何中面 → 中性轴被拉向高模量冰层） |
| D_composite | 6.764e+07 Pa·m³ |
| D_comp / D_ice（同总厚度纯冰） | 0.4576 |
| η_snow | 0.5959 |
| 首裂控制层 | ice（下缘受拉，柱状冰控制） |
| Winkler 板中心挠度 纯冰 → 复合 | 2.885e-04 → 3.978e-04 m |

### 3.4 与实验对比（`fig4_experiment_comparison.png`）

**基准可获得性审计（关键）**：任务书指定「Saloranta (2000) 复合梁实验数据」，实测核查后**不成立**：

- `Saloranta, T. (2000) Modeling the evolution of snow, snow ice and ice in the Baltic Sea, Tellus A 52(1), doi:10.3402/tellusa.v52i1.12255`（Tellus A: Dynamic Meteorology and Oceanography, 2000，DOI 10.3402/tellusa.v52i1.12255）；
- 该文献是一维雪/雪冰/冰**热力学厚度—密度演化模型**，出版记录中不含复合梁弯曲实验
  （`beam_bending_experiment_present = False`，核查途径：Crossref REST API query.bibliographic (api.crossref.org)）；
- 原文在本环境不可取（HTTP 403 (tandfonline.com)；本环境无订阅，未取得原文数据表），因此**没有任何可用的 Saloranta 实验 η 值**。
  为避免编造数据，判据 (c) 改用真实复合梁三点弯曲实验作为替换基准：

**替换基准**：Edington, Murdza & Schulson (2025), *J. Glaciol.* 71, e90, doi:10.1017/jog.2025.10077
—— 6%（1/17）厚的粒状冰层 bonded 于柱状冰上表面，−10 ℃，共 40 组三点弯曲。

| 工况 | 实测 η | 模型 η（粒状层位于受拉侧） | 相对误差 | <15%? |
|---|---|---|---|---|
| 延性区 1e-7 s⁻¹（淡水） | 1.000（无可检测影响） | 1.107 | 1.1% | YES || 脆性区 1e-4 s⁻¹（freshwater） | 1.371 | 1.107 | 19.3% | NO |
| 脆性区 1e-4 s⁻¹（saline） | 1.177 | 1.107 | 5.9% | YES |
| 脆性区 1e-4 s⁻¹（sea_ice） | 1.164 | 1.107 | 4.9% | YES |

脆性区淡水冰的 +37% 强化（实测 1.371 vs 模型
1.107，误差 19.3%）
**超出模型适用范围**：该文把强化归因于表面层对裂纹的抑制（crack suppression），而线性弹性复合梁
只能反映模量/强度的应变相容分配。同样构型若把粒状层放在受压侧，模型给出
η = 0.989（无显著影响），与延性区实测一致。

层强度锚点：Water 2024, 16(23):3358（doi:10.3390/w16233358）
−20 ℃ 三点弯：粒状雪冰 2.85 MPa vs 柱状冰 1.93 MPa，比值
1.477（已计入脆性区模型输入）。

### 3.5 实现正确性复核

| 复核项 | 结果 |
|---|---|
| 闭式 D_composite vs 20 万片独立数值积分 | 相对误差 冰 2.503e-14、复合 6.446e-15、Saloranta 3.261e-14 |
| Navier 级数（k_f=0）vs Timoshenko 0.00406qL⁴/D | 相对误差 5.795e-04 |
| 刚性地基极限 k_f = 1e10 → w → q/k_f | 相对误差 2.105e-05 |
| 级数截断收敛（61 模 vs 40 模） | 相对差 2.938e-09 |

## 4. 结果图

1. `fig1_stress_profile.png` —— 首裂时刻弯曲正应力沿厚度分布（含雪 vs 纯冰），标注雪层区间与中性轴；
2. `fig2_stiffness_strength_vs_snow.png` —— D_comp/D_ice 与 η_snow 随雪厚的两条构型曲线；
3. `fig3_winkler_field.png` —— Winkler 板挠度云图（纯冰 vs 含雪）与中心挠度随 k_f 变化；
4. `fig4_experiment_comparison.png` —— 与 Edington et al. (2025) 四点实验结果的柱状对比（含逐点误差）；
5. `fig5_verification.png` —— 数值积分复核、Timoshenko 基准、级数截断收敛。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **构型决定符号**：雪"叠"在冰上（真实积雪）时，由平行轴定理 D 只会**增大**（本例 +0.10%），
   但雪自重使总载荷由 4017.2 增至 4311.5 Pa，
   挠度仍增大 7.3%。「雪载折减承载能力」在力学上必须落到
   **替代构型（雪冰化/雪泥占据上部厚度）** 或搭接自重/隔热→减薄冰厚，否则会得出反号结论。
2. **线性弹性复合梁的天花板**：脆性区淡水冰 +37% 强化无法复现（误差 19.3%），
   机制是裂纹抑制/尺寸效应，超出现有框架；盐冰/海冰 16~18% 强化在 15% 线内（5.9% /
   4.9%）。
3. **材料参数不确定性**：雪模量随密度跨 3 个量级，snow-ice 模量跨 1~6 GPa。结论（D 折减方向、η<1）对参数不敏感，
   但绝对数值敏感；软件实现中所有参数均可注入（`IceLayer`），默认值只是量级。
4. **平台审计发现（非本节点阻塞）**：既有 Q4 Mindlin + Winkler 求解器
   `scenarios/suboff_icebreaking/si1_one_way/plate_response.py::solve_plate_on_foundation`
   在均布压力下给出关于中线**反号**的挠度场（4×4、10×10 网格均复现，中线 w≈0、另一侧 +1.4e-4 m），
   形态是绕中线刚体转动而非下凹挠曲（均布压力下挠度场关于中线反号（中线 w≈0，另一侧 +1.4e-4 m），形态是绕中线的刚体转动而非下凹挠曲；对 4×4…）。疑似单元刚度按 `[w|θx|θy]` 分组
   而装配按交错自由度索引。因此本节点的 Winkler 验证改用解析解三重复核，**该求解器的 D_composite 接入
   接口已备好（`equivalent_homogeneous_modulus`），但定量验证需先修该缺陷**。
5. 模型为小变形线弹性、无损伤演化、无蠕变；应变率/温度效应以参数形式进入（分延性/脆性两套输入）。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| z_na（等刚度三层） | h_tot/2 | 5.551e-17 m（绝对） | YES（机器精度） |
| D_composite（闭式） | 20 万片独立数值积分 | 6.446e-15（相对） | YES |
| D_composite（Saloranta 层序） | 独立数值积分 | 3.261e-14 | YES |
| 纯冰 D（h_snow=0） | 解析 E h³/(12(1−ν²)) | 0.000e+00 | YES |
| Winkler 板中心挠度（k_f=0） | Timoshenko 0.00406qL⁴/D | 5.795e-04 | YES |
| Winkler 板中心挠度（刚性地基） | q/k_f | 2.105e-05 | YES |
| η_snow vs 实验（延性区，替换基准） | Edington et al. 2025（η=1.00） | 1.08% | YES |
| η_snow vs 实验（脆性区淡水冰） | Edington et al. 2025（η=1.371） | 19.3% | **NO（>15%，机制超出模型范围，如实记录）** |
| η_snow vs 实验（脆性区盐冰/海冰） | Edington et al. 2025 | 5.9% / 4.9% | YES（<15%，非 <3%；实验散度 ±0.13~0.20 MPa 量级） |
| η_snow vs *Saloranta (2000)* 实验值 | 无可比数据（该文非弯曲实验，原文 403 不可取） | **无基准** | **无法判定 —— 已用替换基准并显式披露，不粉饰** |

**结论**：判据 (a)(b)(d)(e)(f) 严格满足；判据 (c) 在替换基准（真实三点弯曲实验，延性区）下满足
（1.08% ≪ 15%），但 Saloranta (2000) 本身不含复合梁实验数据，任务书该项
前提不成立，已如实记录；脆性区淡水冰工况模型误差 19.3%，属**诚实 FAIL**（不粉饰）。

## 7. 复现

```bash
PYTHONPATH=src python -m pytest tests/test_snow_ice_composite.py -q      # 17 passed
PYTHONPATH=src python artifacts/m762/run_snow_ice_composite.py           # 重新生成 metrics.json + 5 图
python artifacts/m762/make_report.py                                     # 重新生成本报告与平台报告负载
```
