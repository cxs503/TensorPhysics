# 冰压压力-面积关系参数库与 `ex` 标定（GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB）

> 状态：**交付 + 诚实保留项（partial）**。机器复核入口：
> `PYTHONPATH=src python scripts/run_gap_pp2_pressure_area.py`；
> 参考对照 8/8 行 PASS（`artifacts/ice_pressure_area/reference_table.md`）。

## 0. 结论摘要

**交付的三件事**

1. **参数库**（`data/ice_pressure_area/pressure_area_sources.json`，7 条记录 / 2 条树内可核 / 5 条待一手核验），每条强制声明 `area_definition`、单位、承压足迹几何、工况与证据等级；逐条 sha256 锚定（aggregate `44dd193c3e9bcebf…`）。
2. **面积口径代数**（`src/tensorphysics/ice_pressure_area/conventions.py`）：给出显式转换律 `ex_ref = 1 + (ex_src − 1)·r`，`r = d lnA_src/d lnA_ref`，并**证明**定比口径（线载荷/板格）不改变 `ex` —— 这条直接否决了「口径统一即可收缩带宽」的隐含假设。
3. **分层标定**（`calibrate.py`）：DerSimonian–Laird 随机效应合并（按尺度分层）+ 留一交叉 + 过程区宽度扫描 + 协变量可辨识性审计（不可辨识者显式 WITHHELD，不拟合）。

**可引用取值（本单结论）**

- 结构板格/局部接触（reference convention = `nominal_projected_contact`）：`ex = 0.450`，90 % 区间 `[0.408, 0.492]`（设计规则层，k=2，`tau=0.000`）。
- 由此推导的结构需求带宽（面积比 2.3593，由平台实测 1.32x–1.74x 反演得到）：`[1.420, 1.525]`，相对声明区间收窄 **5.04×**。
- 实测局部压力层（中尺度现场压入 + 现场数据汇编）：`ex = 0.618`，90 % 区间 `[0.495, 0.741]`，`tau=0.081`，I²=57.9 %（实质异质）→ **该层 90 % 区间不落在原声明带内**（带宽上限 1.889 > 声明上限 1.824），即原 1.74x 上限对实测局部压力**不保守**。
- 全局（结构尺度）层：`ex = 0.300`，90 % 区间 `[0.250, 0.350]`（k=1，区间即来源声明带，无跨源异质信息）。

**诚实保留项（未达判据的部分，第 2 轮直接输入）**

- 判据(1)「≥2 个一手数据源入库」：**记录齐、核验未达**。5 条 accuracy-reference 记录 `primary_source_verified=false` —— 本构建环境无外网出口，ISO 19906 / IACS UR I2 / Masterson & Frederking (1993) / Sanderson (1988) / Johnston et al. (1998) 原文均未打开。`fail_stage=primary_source_verification`，`failure_reason=no_network_egress`；第 2 轮需要一次可访问原文的环境（机构订阅/情报室）把 5 条记录的 `primary_source_verified` 升级。
- 判据(2)「`ex` 与环境参数（温度/…）关系」：**WITHHELD**。装配集中仅 1 条记录带温度区间，协变量审计机器判定 `ice_temperature_C` 不可辨识（`n_finite=1/2`，原因：`only 1/2 sources report this covariate`）→ 不拟合、不出数字。
- 面积口径「法向投影 vs 浸湿面积」的**统一**：库内 `wetted_surface_area` 计数为 0，即现有可用来源无一采用浸湿面积口径 → 只能给出解析转换律（含 `r = 1 + beta` 分支），不能给经验换算系数。（这正是原 gap 想解决但数据不支持的一点，如实记录。）

---

## 1. 问题设定与基准出处

冰压压力-面积律 `p(A) = p0·A^(−ex)` 中 `ex` 长期以声明扫描区间 `0.30–0.70` 使用，带来两个后果：`ex` 无可引用出处，结构需求带宽被放大为 `1.32x–1.74x` 且无法收缩。根因有二：(a) 来源侧无一手数据入库/无面积口径声明；(b) 口径（法向投影面积、含破碎损伤区面积、线载荷每米面积、板格名义面积、浸湿面积）混用。

本单把「基准」定义为三类可机器复核的锚：

- **树内 sha256 锚定数据**：`benchmarks/ice_nav_refs/assets/iacs_ur_i_pc6.json`（PC6 板格压力表，公式 `p = 2.5·A^−0.45`）与 `src/tensorphysics/ice_modules/local_pressure.py::FREDERKSON_REF`（5 点 p–A 表，声明 `ex=0.185`，被同一文件中的模型以 `ex=0.15` 使用）；
- **平台实测需求带宽** `[1.32, 1.74]`（gap 记录，`ex∈[0.3, 0.7]`）；
- **解析恒等式**：定比口径不变性 `ex_ref = ex_src`、损伤区对数导数闭式 `r = (1+t/2)/(1+t)`、等方差合并 = 算术平均。

## 2. 方法与实现

**(a) 参数库与证据分级**（`library.py`）：每条记录必须给出 `citation`（含年份+出处，占位符被拒）、`units`（逐值单位）、`area_definition`（受控词表 5 项）、`geometry`（足迹形状/长宽比/周长因子 λ/过程区宽度比）、`conditions`（冰型/厚度/温度/缩比/压入速率/面积窗）、`role`（`accuracy_reference` 可裁决计算 / `regression_anchor` 仅供漂移检测）。`detect_conflicts` 以 `z = |ex_i−ex_j|/√(σ_i²+σ_j²) > 2k` 判定同族冲突并分级。

**(b) 面积口径代数**（`conventions.py`）：设源口径面积 `A_src = g(A_ref)`，`r = d ln g/d ln A_ref`，则 `ex_ref = 1 + (ex_src − 1)·r`。

推导：`p_ref = F/A_ref`、`p_src = F/A_src`，力的口径无关、压力的口径有关；`d ln p_src/d ln A_ref = d ln p_ref/d ln A_ref + (1 − r)`，代回 `ex_ref = −d ln p_ref/d ln A_ref` 即得。

由此得两条可直接引用的结论：

1. `r ≡ 1` ⇒ `ex` 严格不变（线载荷/板格/定比面积）——**单位与定比口径的簿记不可能解释 `ex` 散布**；
2. 损伤区口径（材料过程区宽度 `delta` 与足迹无关，`t = λ·delta/√A_c`）：`r = (1 + t/2)/(1 + t) < 1`，故 `ex_nominal = 1 + (ex_damaged − 1)r`，偏移量上界 `|Δex| ≤ (1 − ex)·t/2`，且随面积增大消失 —— 口径只在**小面积端**有影响。

**(c) 标定**（`calibrate.py`）：`σ_i` 由 90 % 声明带换算（`σ = (hi−lo)/(2·1.6449)`）；口径归一到 `nominal_projected_contact`（`r` 同时放大区间）；按尺度分层做 DL 随机效应合并，报告 `Q/τ²/I²`；留一交叉；协变量用**加权设计矩阵秩 + 取值对比数**做可辨识性审计。

## 3. 参数表（无量纲量）

| 符号 | 含义 | 取值 | 来源 |
|---|---|---|---|
| `ex`（design_rule, nominal projected） | 压力-面积指数 | 0.4500 [0.4084, 0.4916] | 本单合并（2 条规则转写） |
| `ex`（measured_local） | 同上，实测层 | 0.6181 [0.4952, 0.7409] | 本单合并（2 条文献层） |
| `ex`（global） | 结构尺度 | 0.3000 [0.2500, 0.3500] | Sanderson (1988) 汇编 |
| `ex`（naive 不分层） | 错误示范：全源硬合并 | 0.4772 [0.3827, 0.5717] | 本单（对照用） |
| `A_2/A_1` | 需求带宽反演面积比 | 2.3593 | 由 1.32x/1.74x 反演 |
| `λ = P/√A` | 周长因子 | 4.25（4:1 矩形贴片）、4.00（方形） | 几何定义 |
| `delta/√A_c` | 过程区宽度比（假设，非实测） | 0, 0.05, 0.1, 0.2, 0.3 | 本单扫描 |
| `t = λ·delta/√A_c` | 无量纲环宽 | 0 – 1.275 | 由扫描导出 |
| `tau` | 层内异质（随机效应） | 0.0000（design）、0.0807（measured） | 本单 |
| `I²` | 异质占比 | 0.0 % / 57.9 % | 本单 |

## 4. 对照表（ref vs got，逐项打勾）

机器生成全文见 `artifacts/ice_pressure_area/reference_table.md`。最差的数值核对行为 `R7_demand_bandwidth_high_end`：ref=1.74 / got=1.82366，误差 4.81 %（容差 5 %）。解析类核对（容差 ≤1e-3 %）的最差误差 1.08e-09 %，即转换律闭式与差分、以及定比不变性在数值精度内成立。

| id | criterion | ref | got | error % | tol % | verdict |
|---|---|---|---|---|---|---|
| R8_transfer_derivative_coarse_step | same log-derivative at a coarse step h = 1e-3 (convergence of the difference scheme) | 0.850877 | 0.850877 | 2.07e-07 | 0.01 | PASS |
| R1_recover_rule_exponent_IACS | log-space fit of the sha256-anchored IACS PC6 table recovers its declared ex | 0.45 | 0.45 | 1.23e-14 | 1 | PASS |
| R2_recover_inrepo_table_exponent | fit of the in-repo FREDERKSON_REF table recovers the exponent recorded for it | 0.185 | 0.185648 | 0.35 | 1 | PASS |
| R3_ratio_invariance | constant area ratio (line-load, panel) leaves ex bit-identical | 0.45 | 0.45 | 1.23e-14 | 1e-06 | PASS |
| R4_ratio_invariance_panel | panel_nominal_area convention at A = 7.5 m^2 leaves ex bit-identical | 0.45 | 0.45 | 1.23e-14 | 1e-06 | PASS |
| R5_transfer_derivative_analytic_vs_fd | d ln A_d/d ln A_c: closed form vs central finite difference (h = 1e-5, A = 1 m^2) | 0.850877 | 0.850877 | 1.08e-09 | 0.001 | PASS |
| R6_demand_bandwidth_low_end | demand bandwidth at the low declared exponent end (1.32x measured by platform) | 1.32 | 1.2937 | 1.99 | 5 | PASS |
| R7_demand_bandwidth_high_end | demand bandwidth at the high declared exponent end (1.74x measured by platform) | 1.74 | 1.82366 | 4.81 | 5 | PASS |

## Exponent by stratum (90 % intervals)

| stratum | k | ex pooled | 90 % CI | tau | I2 % | use |
|---|---|---|---|---|---|---|
| anchor::local | 1 | 0.1500 | [0.1000, 0.2000] | 0.0000 | 0.0 | drift-anchor only, never pooled as evidence |
| anchor::model_scale | 1 | 0.1850 | [0.1700, 0.2000] | 0.0000 | 0.0 | drift-anchor only, never pooled as evidence |
| design_rule | 2 | 0.4500 | [0.4084, 0.4916] | 0.0000 | 0.0 | pooled |
| global | 1 | 0.3000 | [0.2500, 0.3500] | 0.0000 | 0.0 | pooled |
| measured_local | 2 | 0.6181 | [0.4952, 0.7409] | 0.0807 | 57.9 | pooled |

## 5. 收敛性与参数敏感性

- **差分步长两分辨率**：`h=1e-3` 误差 2.07e-7 %，`h=1e-5` 误差 1.08e-9 %（`R8` vs `R5`），中心差分二阶收敛，闭式导数可靠。
- **口径转换敏感性**（`delta/√A_c` 扫描，`A_c = 1 m²`，`λ = 4.25`）：

| `delta/√A_c` | `r` | `ex_nominal`（输入 `ex_damaged=0.45`） | 设计层合并 | 实测层合并 |
|---|---|---|---|---|
| 0 | 1.0000 | 0.4500 | 0.4500 | 0.6181 |
| 0.05 | 0.9124 | 0.4982 | 0.4500 | 0.6181 |
| 0.1 | 0.8509 | 0.5320 | 0.4500 | 0.6181 |
| 0.2 | 0.7703 | 0.5764 | 0.4500 | 0.6181 |
| 0.3 | 0.7198 | 0.6041 | 0.4500 | 0.6181 |

  观察：分层合并值对口径扫描**不变**，因为装配集内无一来源声明 `damaged_zone_area` 或 `wetted_surface_area` 口径（见 §0 保留项）。扫描给出的是**转换函数灵敏度**：若一条来源声明损伤区口径，`delta/√A=0.2` 时 `ex` 将从 0.45 上移约 0.126。
- **留一交叉（design_rule）**：k=2 < 3，留一退化；改用逐源读数做敏感性：
  - `ISO_19906_2010_A8_FIRST_YEAR`：`ex = 0.4500` [0.4000, 0.5500]（口径修正 -0.0000）
  - `IACS_UR_I2_PC6_PATCH_PRESSURE`：`ex = 0.4500` [0.4000, 0.5000]（口径修正 -0.0000）
- **留一交叉（measured_local）**：k=2 < 3，留一退化；改用逐源读数做敏感性：
  - `MASTERSON_FREDKING_1993_LOCAL_PRESSURE_AREA`：`ex = 0.7000` [0.5500, 0.8000]（口径修正 +0.0000）
  - `JOHNSTON_ET_AL_1998_MSI_INDENTATION`：`ex = 0.5500` [0.4500, 0.6500]（口径修正 +0.0000）
- **协变量可辨识性审计**：
  - `intercept`：distinct=1, n_finite=2 → identifiable
  - `ice_thickness_m`：distinct=2, n_finite=2 → identifiable
  - `ice_temperature_C`：distinct=1, n_finite=1 → WITHHELD（only 1/2 sources report this covariate）

## 6. 失败模式、工件与复现

| 项 | 内容 |
|---|---|
| 复现命令 | `PYTHONPATH=src python scripts/run_gap_pp2_pressure_area.py`（含图）；`--no-figures` 关图（约 1 s） |
| 单测 | `python -m pytest tests/test_ice_pressure_area.py -q`（37 项，含端到端） |
| 全量工件 | `artifacts/ice_pressure_area/calibration.json`（含全部机器读数） |
| 对照表 | `artifacts/ice_pressure_area/reference_table.md` |
| 图 | `artifacts/ice_pressure_area/figures/fig1…fig4` |
| 库锚 | `data/ice_pressure_area/MANIFEST.json`（逐条 + 聚合 sha256） |

**被这批工作揭出的真实缺陷（族内冲突，机器判定阻塞）**

- `[blocking]` family=`iacs_ur_i` ['IACS_UR_I2_PC6_PATCH_PRESSURE', 'TENSORPHYSICS_LOCAL_PRESSURE_IACS_EX015']：`ex=[0.45, 0.15]`，`z=6.98`
- `[blocking]` family=`iacs_ur_i` ['IACS_UR_I2_PC6_PATCH_PRESSURE', 'TENSORPHYSICS_FREDERKSON_REF_TABLE_EX015']：`ex=[0.45, 0.185]`，`z=8.35`

即同一仓内对 IACS 压力-面积律存在 `ex=0.15`（`ice_modules/local_pressure.py`）、`ex=0.185`（该文件自带的 p–A 参考表实测斜率）与 `ex=0.45`（`benchmarks/ice_nav_refs` 资产）三个读数。消费者对照（以模型自身 `p0=14 MPa` 复位，纯指数效应）：`A=50 m²` 处模型比标定律高 223.3 %；模型相对其自带参考表的误差随面积升至 14.49 %。

**失败模式**

1. `primary_source_verification`（阻塞判据(1) 的完全达成）：无外网，5 条 accuracy-reference 记录未对原文核验；本单未把 `primary_source_verified` 置 true，也未用二手值冒充一手。
2. 温度/环境依赖：`WITHHELD`，非「不支持」——需要第 2 轮补入 ≥2 条同族且带温区间的来源才能回归。
3. 浸湿面积口径：库内 0 条 → 只能给转换律分支，给不出经验换算系数。
4. **文献数值的出处等级**：5 条 accuracy-reference 记录的数值来自公开综述/规则形式的文献知识，而非本次对原文的抄录（`primary_source_verified=false` 即此意）。方法层面（口径代数、分层合并、可辨识性审计、冲突检出）不依赖这些具体数值：原文核验后只需更新 `reported` 字段并 `--seal` 重跑，标定链自动更新。引用单个数值前必须先完成这一步。
5. `design_rule` 层 `tau=0` 是**转写一致性**而非测量一致性（两条规则转写同值 0.45），不能当作「`ex` 已被独立测量确证 0.45」的证据；引用时须并列 `measured_local` 层。

