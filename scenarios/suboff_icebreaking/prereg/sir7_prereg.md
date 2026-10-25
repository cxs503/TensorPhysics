# SIR-7 预注册：G7 文献 / 实验对照（P-h 曲线等公开数据）

- **创建时间**：2026-10-25T05:10:21Z
- **缺口**：G7 —— 契约原有要求未兑现：物理正确性锚须对照文献/实验（冰板穿透载荷-挠度等公开数据）
- **依赖**：**SIR-5（断裂演化）+ SIR-6（规模产出）**；战役出口腿
- **算力/预算**：**CPU**；S（<1 班次）
- **append-only**：冻结后只允许 `## Addendum <UTC Z>` 追加

## 1. 科学目标

把 SIR-6 的规模产出与**公开发表的实验/解析数据**做定量对照：冰板穿透载荷-挠度（P-h）曲线的
峰值载荷与峰位挠度、以及船-冰阻力统计，给出误差表。所有参考数字必须有出处（DOI/报告号 +
页码 + 表号）并以三元组登记——无出处数字一律判 FAIL。

## 2. 物理锚

| 锚 | 内容 | 出处 |
|---|---|---|
| A7.1 | **冰板穿透载荷-挠度（P-h）公开数据**（冰池试验或承载力解析/半解析解） | **执行时冻结**：每条参考须写明 DOI/报告号 + 页码 + 表号/图号，转录值入 `sir7_reference.csv` 并附三元组。骨架期只冻结纪律（不得写入未经核对的具体数字） |
| A7.2 | 船-冰阻力实验对照（10 个 level-ice 工况 + 4 个公式基线） | Hu & Zhou, IJNAOE 8(2), 169-176 (2016), DOI `10.1016/j.ijnaoe.2016.01.004`；转录件 `TensorDEM/benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv`；口径冻结于 `TensorDEM/tests/test_dem3d_ship_ice_formula_baselines.py` |
| A7.3 | SUBOFF 阻力实验（流体侧） | Liu & Huang, CRDKNSWC/HD-1298-11, ADA359226, Table 14, p.23 |
| A7.4 | 趋势锚：峰值前切线刚度随挠度非增 | 浮冰板承载力物理（弹性→屈服→破坏）；作为趋势门 |

**为什么 A7.1 不在骨架期写数字**：契约红线「锚必须给文献出处、数字零手抄」。P-h 实验数据的
具体数值必须在 SIR-7 执行班次逐条核对原文后转录并三元组化；骨架期凭记忆写入数字就是手抄，
正是本战役要消灭的行为。此处只冻结「必须有出处 + 必须三元组化 + 必须有误差门」。

## 3. 机器门

| 门 | 判据 |
|---|---|
| M7.1 | P-h 关键点误差门：峰值载荷、峰位挠度相对误差 ≤ 15%（粗网格/模型误差容忍，执行时冻结最终口径） |
| M7.2 | 船-冰阻力门：MAPE ≤ 30%（同 MT Uikku 公式基线量级） |
| M7.3 | 出处门：每个参考数值带 DOI/报告号 + 页码 + 表号，且写入三元组；缺出处判 FAIL |
| M7.4 | 趋势门：峰值前切线刚度随挠度非增 |
| M7.5 | 复现门（L4）：同 case 两跑关键数 `|Δ| < ε`（ε 执行时冻结） |

## 4. 规模判据

- 对照点 ≥ 5（P-h 曲线采样点；或船-冰 10 工况全跑）。
- 重复数 ≥ 2（记录 Δ）。

## 5. 内容符合判据（语义门 → `semantic_criteria.json` → `SIR-7`）

- `sir7_provenance_reference_cited`（provenance）：参考数据文件含出处列且三元组可复算。
- `sir7_physics_peak_load_captured`（physics_switch / numeric_lower_bound）：P-h 峰值载荷被捕获（非单调吹爆、非缺失）。

## 6. 交付物

`sir7_reference/validation_report.json` + `gates.json` + `sir7_reference.csv`（含出处列：doi/report/page/table）
+ P-h 对照曲线（figure）+ 误差表。

## 7. 停机条件

出处门红（任一无出处数字）→ 停，该条参考作废重取；不得以公式基线冒充实验真值
（MT Uikku README 明确：公式是 baseline 不是 ground truth）。

## Addendum 记录

（暂无）
