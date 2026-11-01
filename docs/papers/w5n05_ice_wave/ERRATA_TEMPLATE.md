# Errata — W5-N05 冰盖下弯曲重力波（mission m994，report #823）

> 本文件由 `examples/ice_wave_w5n05_errata.py` 从 `artifacts/w5n05/results.json` 程序化生成，**数字零手抄**（模板：`ERRATA_TEMPLATE.md`）。
> 缘起：独立评审 report #961 复算指出两处算术不一致。**未重跑物理管线**——只从存储数组（alpha 4×8 网格、EnKF 真值/背景均值/分析均值）复算拟合与统计量。

## 0. 结论摘要

{{SUMMARY_TABLE}}

**根因一句话**：`results.json` 的两处数值都没有算错；断点在于 ①派生指数的**计算口径未落盘**（只存标量，不存“固定哪条切片”）；②正文只印 8 点交叉校核表，而指数用 4×8 = 32 点网格算出，复算方从可见表格出发必然命中不了校准口径；③派生百分比的有效位数高于其依赖的原始量（4 位小数 RMSE）。{{ROOT_CAUSE_EXTRA}}

## 1. 案例介绍

- **物理**：MIZ（密集流冰）把薄板弯曲重力波的刚性 D 与质量 ρh 按浓度 n_c 做 Voigt 并联均匀化，
  `D_eff = n_c·D_floe(1 + i·tan δ)`；复波数虚部即空间衰减率 α，`A(x) = A0·exp(−αx)`。
  标定目标是幂律 `alpha ~ n_c^p h_i^q`。
- **参数与几何**：{{MIZ_PARAMS}}
- **EnKF**：{{ENKF_PARAMS}}
- **源工件**：{{SOURCE}}

## 2. 怎么做的

1. 以 `artifacts/w5n05/results.json`（报告 #823 的渲染源）为唯一真值，不触碰物理求解器；
2. 枚举 alpha 指数族的**全部口径**：每条固定冰厚切片的 p、每条固定浓度切片的 q、32 点联合双回归 OLS（含标准误 / R² / 对数残差）、单变量 OLS；
3. 用同一拟合器在**报告印出的 8 点子网格**上重算，以识别评审所用口径；
4. EnKF 降幅按 `1 − RMSE_a/RMSE_b` 从存储数组全精度复算，并另按“先舍入到 4 位小数再推算”复现评审链路；
5. 全部结果同时写入 `artifacts/w5n05/errata/errata_w5n05.json` 与本文件（同一脚本），回归门 `--check`。

## 3. 计算结果

### 3.1 MIZ 标定指数的生成口径（核定）

报告句（REPORT.md §3.5）：「标定关系（对数-对数最小二乘）：`alpha ~ n_c^p`，p = 0.3840；`alpha ~ h_i^q`，q = 1.7530。」

代码口径（`src/tensorphysics/ice_wave/miz.py::sweep_attenuation`）：

{{CONVENTION_LIST}}

复算命中（浮点级一致）：

{{REPRO_TABLE}}

→ **报告值就是这两条切片斜率的精确舍入**：`results.json` → 报告之间不存在数值断点，也不存在算术错误。缺陷是**口径缺席**——键名 `alpha_vs_concentration_exponent` 不含“固定冰厚多少”，报告正文也没有写。

### 3.2 全局指数的各种口径对照（32 点全网格）

{{GLOBAL_TABLE}}

{{NON_SEPARABILITY}}

### 3.3 评审复算值的来源（差异归因）

{{SUBGRID_NOTE}}

{{SUBGRID_TABLE}}

{{SUBGRID_CONCLUSION}}

### 3.4 EnKF 降幅

定义（`examples/ice_wave_w5n05_pipeline.py::enkf_stage`）：`RMSE_b = RMS(background_mean − truth)`，`RMSE_a = RMS(analysis_mean − truth)`，`reduction = 1 − RMSE_a/RMSE_b`。

{{ENKF_TABLE}}

{{ENKF_NOTE}}

### 3.5 断点与根因（按层归因）

{{BREAKPOINT_TABLE}}

## 4. 结果图

{{FIG_LIST}}

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

- **alpha 本身不是争议点**：{{CROSS_CHECK}}
- 「对数-对数最小二乘」在非可分模型上是**病态提问**：切片指数随另一变量单调漂移（{{SLICE_DRIFT}}），任何单值都会因切片选择而变动数十个百分点——报告与评审的差异正是这个自由度的体现。
- 未做 bootstrap：切片标准误取自对数线性回归的经典 OLS 公式，未把 alpha 自身 ~1e-4 量级的数值误差逐点传播（量级可忽略）。
- 文献腿仍为 WITHHELD（Meylan et al. 2018 具体数值本次环境仍不可读），故 p/q 无外部基准可对标，误差判定只对**同工件内的口径一致性**下结论。
- 未重跑物理管线：alpha 网格与 EnKF 集合沿用 `results.json` 存储值。

## 6. 误差判定（逐项对基准，显式回答是否 < 3%）

{{ERROR_TABLE}}

{{ERROR_CONCLUSION}}

## 7. 工件清单

{{ARTIFACTS}}