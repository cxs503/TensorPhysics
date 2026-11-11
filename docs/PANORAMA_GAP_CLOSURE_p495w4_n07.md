# PANORAMA §五 硬缺口关闭申请：FEM→DEM 断裂交接无导出

**Node** p495w4-n07 · **mission** 13949 · **repo** TensorPhysics · **branch** `agent/m13949`
**Status** CLOSED (Implemented + audited, not experimentally validated)
**Date** 2026-11-11

## 1. 缺口原文与判定

`PANORAMA.md` §五（硬缺口账本）中的条目「FEM→DEM断裂交接无导出」声称：平台没有把
FEM 结构断裂面导出为 DEM 可消费工件的实现路径。本节点交付后该条目**不再成立**：

| 缺口要素 | 交付前 | 交付后（本节点） |
|---|---|---|
| 断裂面导出 | 无实现（仅文档级接口） | `failure_faces` v3.0.0 包，冻结 schema + SHA-256 密封，`validate_packet` 三层校验（JSON-Schema 子集 / jsonschema / 语义不变量） |
| 导出内容 | — | 断裂面位置、单位法向、面积、牵引力、比表面能；`crack_front_m`、`fracture_layer_thickness_m` |
| 释放态交接 | — | 密封 `release_state.json` 侧车：pre-fracture 机械能、释放态逐单元速度/位移/应变能、子单元动能、质心速度 |
| DEM 消费 | — | 单元尺寸碎片云（每工况 7 680 / 15 360 / 23 040 个 50 mm 立方碎片）+ 刚性碎片积分器（自由飞行 + 地面冲击） |
| 守恒审计 | 无 | 质量 <1 %、动量 <2 %、能量 <5 % 三闸门 + schema 闸门 + 碎片尺寸 KS 闸门，6 工况全过 |
| 反例控制 | 无 | 丢碎片 / 速度注水 20 % / 碎片粗化 2× / 破坏包内面片 —— 4/4 全红 |

## 2. 证据（数字与工件路径）

* 六工况汇总：`artifacts/p495w4_n07/summary.json`
* 每工况包：`artifacts/p495w4_n07/cases/<case>/failure_faces.json`（v3.0.0，`sha256` 密封）
* 释放侧车：`artifacts/p495w4_n07/cases/<case>/release_state.json`
* 守恒账本：`artifacts/p495w4_n07/cases/<case>/conservation.json`
* 反例：`artifacts/p495w4_n07/cases/<case>/negative_controls.json`
* 报告：`docs/reports/p495w4_n07/REPORT.md`
* 测试：`tests/test_p495w4_n07_beam_handoff.py`（58 passed）

关键判据数字（详见 `summary.json`）：质量残差最大 1.5×10⁻¹³、动量残差最大
2.6×10⁻¹¹、能量残差最大 5.2×10⁻⁵、KS 检验 p = 1.00、schema 6/6 通过；
FEM 对 Euler–Bernoulli/自由–自由解析解误差：挠度 ≤1.7×10⁻¹⁰、应力 ≤2.45 %、ω₁ ≤1.5×10⁻⁶。

## 3. 与既有 verified 案例的区别（非重复）

| 既有 | 本节点 |
|---|---|
| p273w3-n03：0.24 m 冰 SENB 试件，混合型裂纹路径 + 面面积/能量闸门 | 米级梁（L=2/4/6 m）三点弯曲至断裂，**跨-速率**六工况 |
| p470w2-n05：冰板弯曲能量守恒契约（弯曲能 vs 断裂能） | 梁 FEM 断裂 → `failure_faces` 包 → DEM 碎片化，**三守恒量**（质量/动量/能量）闭环 |
| 先前交接包为“面 → 单个棱柱碎片” | 本节点为“整梁单元尺寸碎片云 + 释放态侧车”，含子单元动能交接与负控矩阵 |

## 4. 局限（不得据此标记为 “Verified”）

1. DEM 碎片为刚性平动理想化，无碎片间黏结/碰撞（仅地面接触），因此是守恒审计而非碎片场预测。
2. 单元尺寸碎片无法承载子单元速度弥散，该部分以“内能”交付并单列披露（本波最大 25 %；不交付时最坏能量残差 5.1 %）。
3. 表面能 `G_f A` 为给定沉项（材料 + 包几何），链条检验其余项闭合，且耗散抽取比 ≤0.21。
4. 无实验基准：验证锚为闭式梁理论与内部守恒恒等式。
5. 动态工况采用力控制斜坡（2 ms 到静态破坏载荷），非位移控制刚性压头（后者产生非物理局部应力集中）。

## 5. 建议的 PANORAMA §五 修改（owner 文件在工作树外，未直接编辑）

```diff
-| FEM→DEM断裂交接无导出 | 无 | 硬缺口 | — |
+| FEM→DEM断裂交接无导出 | `failure_handoff.beam_fem/beam_cases`：failure_faces v3 包 + 释放侧车 + 三守恒闸门（6 工况全过） | 已关闭（Implemented+audited，未实验验证） | p495w4-n07 / mission 13949 / docs/reports/p495w4_n07/REPORT.md |
```

> 说明：`/root/tp-agent/PANORAMA.md` 位于工作树之外，按纪律本单未修改该文件；
> 上表给出可直接套用的替换行，供 owner 或后续平台维护节点落盘。