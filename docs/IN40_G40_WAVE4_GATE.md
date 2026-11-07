# IN40-G40 波4核销门 · Plan 40 收官报告

- **计划**：plan_id=84 · p39_wave2_aggregated  
- **核销门 mission**：5546 · 证据截止 `2026-11-07T13:25:26Z`  
- **机器裁决**：`NOT_ACCEPTED_NODES_OPEN` · 计划状态裁决 `retained_active` · 通过率 **0/10 = 0.0%**（阈值 80%）  
- **报告摘要 digest**：`bcceb9b843a8d33cf54b6840e4ef2b36560424d829b0569c9f405a23111b24db`

> 裁决规则（机器可算，先于结论）：节点通过 = 该节点全部机器判据 `accept`；计划 `completed` 需 **通过率 ≥ 80% 且无 critical failure**；缺席证据一律记 `withheld`，不得上抬为通过。

## 1. 案例介绍：Plan 40 的物理问题与节点构成

Plan 40 是「谱系继承波 2」：10 个节点把船舶总体性能的六大方向（阻力/推进/耐波/操纵/稳性/结构振动）与跨求解器界面统一压到同一张验收面上。每个节点自带机器判据，核销门的职责不是做物理，而是**判定这些物理结论是否已具备可继承的证据**。

**表 1 · 10 节点构成与验收状态（node_status / reproduction_verdict 取自 campaign_nodes）**

| ext_id | repo | mission | node_status | repro_verdict | 判据数 | 节点裁决 | 谱系 |
|---|---|---|---|---|---|---|---|
| IN40-N01 | TensorFVM | 5642 | in-flight | — | 5 | ⛔ withheld | iterate |
| IN40-N02 | ship-performance-platform | 5537 | in-flight | — | 6 | ⛔ withheld | iterate |
| IN40-N03 | ship-performance-platform | 5641 | in-flight | — | 5 | ⛔ withheld | iterate |
| IN40-N04 | TensorFVM | 5539 | in-flight | — | 5 | ⛔ withheld | iterate |
| IN40-N05 | TensorFEM | 5543 | in-flight | — | 5 | ⛔ withheld | iterate |
| IN40-N06 | ship-performance-platform | 5544 | in-flight | — | 6 | ⛔ withheld | iterate |
| IN40-N07 | ship-performance-platform | 5540 | in-flight | — | 5 | ⛔ withheld | iterate |
| IN40-N08 | ship-performance-platform | 5637 | in-flight | — | 5 | ⛔ withheld | iterate |
| IN40-N09 | ship-performance-platform | 5545 | in-flight | — | 6 | ⛔ withheld | iterate |
| IN40-N10 | TensorPhysics | 5542 | in-flight | — | 6 | ⛔ withheld | iterate |

**物理目标（各节点自述，逐节点引原文）**

- **IN40-N01**（TensorFVM）：在TensorFVM中实现双体船粘性CFD模拟，提取形式因子(1+k)并完成粘性压力阻力分解，以KVLCC2双体构型为基准与ITTC经验公式裁决。
- **IN40-N02**（ship-performance-platform）：建立冰阻力双轨验证体系：(a) Lindqvist(1989)水平冰层经验方法在5-15kn速度区间的全链对标与适用域边界；(b) 冰-水-气三场LES输运的谱-分辨率双收敛，清偿τ0>1高黏区-3.18% honest FAIL。
- **IN40-N03**（ship-performance-platform）：建立绿色航行能效闭环：(a) Flettner旋筒风帆Magnus效应解析锚+推力-功耗净收益量化；(b) EPL/ShaPoLi功率限值下的船-机-桨匹配与可达航速闭环。
- **IN40-N04**（TensorFVM）：在TensorFVM中实现KVLCC2纯横荡/纯首摇PMM（Planar Motion Mechanism）虚拟试验CFD基准，提取水动力导数并与SIMMAN 2021 workshop数据对标。
- **IN40-N05**（TensorFEM）：建立船体梁总纵弯曲与上层建筑振动噪声的耦合判级链：(a) WAVES2D载荷→FEM梁模型→极限强度CSR校核全链闭合；(b) 螺旋桨叶频/机械激励到上层建筑振动的ISO 6954加权RMS判级。
- **IN40-N06**（ship-performance-platform）：建立推进链物理不变量闸门与空化初生真值化：(a) η_O≤1/P_E≤P_D/跨端点空化数一致，把已交付的REAL敞水锚接到活端点；(b) 叶梢速度恒等式、σ_n/σ_0.7R定义、Keller叶面积判据与Burrill图裁决。
- **IN40-N07**（ship-performance-platform）：建立减摇装置（被动U型水舱/主动减摇鳍）与横摇耦合的首次可验证面：舱液一阶模态解析锚+调谐-减摇率曲线+能量账本。
- **IN40-N08**（ship-performance-platform）：实现耐波性2D剖面水动力系数（附加质量/阻尼系数）的Frank脉动源求解器，从经验频率因子走向积分方程解，以Lewis/Vugts双锚定标。
- **IN40-N09**（ship-performance-platform）：消除阻力链相关补贴双口径分裂（Holtrop Ca vs ITTC-78 ΔCf）并统一CII折减因子分船型表：(a) 建立ITTC-78 ΔCf权威基准，Holtrop Ca换算，KVLCC2 U_SN从7.03%降至≤3%；(b) MEPC.354(78) G4分船型dd值建仓，旧模块单一口径迁移，2027-2030 Z%外推区间。
- **IN40-N10**（TensorPhysics）：建立跨求解器几何保真契约与mortar界面算子统一：(a) CAD积分vs离散网格的Cb/L/B/∇/S前置闸门；(b) 2D/3D mortar界面算子补齐：多边形裁剪、局部守恒测度修正与patch test阶次。

## 2. 怎么做的：采集/评估分离的核销门

1. **证据快照（collect，只读）**：从平台库只读导出 `campaign_nodes ⊗ tensor_dev_missions`（plan_id=84）与 `tensor_guide_tracker`，连同每节点关联报告数，落盘为 `raw/*.json` 并记录 sha256；不打补丁、不改库。
2. **判据解析（机器可算）**：优先读 `campaign_nodes.metadata.acceptance_criteria`，回退解析 goal 内「机器可算验收判据」段的 `C1…` / `(1)…` 编号——判据**解析得到**而非人工摘抄，这是「矩阵覆盖」可被机器裁的前提。
3. **评估（evaluate，纯函数）**：由快照 JSON 重算节点裁决 → 谱系继承 → 计划裁决，期间无网络、无数据库、无时钟依赖，结果字节可复现。
4. **取证分级**：`REAL/MOCK/STUB/WITHHELD/NOT_SUPPORTED`；无提交 → 判据 `withheld`，节点级 `evidence_grade=WITHHELD`。
5. **裁决线**：节点全判据 accept → 节点 accept；通过率 ≥ 80% 且无 critical failure → plan `completed` 并触发下一波提案，否则 plan 保持 `active`。

**审计链**

| step | artifact | sha256/值 |
|---|---|---|
| 1-collect | raw/nodes_plan84.json | f28b6484915515d937c86b8b0847948538d3c9d6aff3fe881ddc4b7fb67d4653 |
| 1-collect | raw/plan.json | afabbe99b1d9c2fce8b6fcfe13a8d69c93b8f1b8d7a647b36627870c24c09e03 |
| 1-collect | raw/guide_tracker.json | 525c41c55804e14b26a4c8e11b572aefce2f001a7b00a133db623c09dfedafa6 |
| 1-collect | raw/provenance.json | 855d6ec04269875957c1279c9edeb3d8ec25cee84eab1a7c5db85851d2e83633 |
| 2-parse | criteria from campaign_nodes.metadata (plan 84) | 54 criteria |
| 3-decide | pass rule rate>=0.8 | pass rate 0/10 = 0.0% < 80%: plan stays active, nodes must iterate |
| 4-evaluate | module tensorphysics.campaign.wave4_gate | plan_id=84, gate_mission_id=5546 |

## 3. 计算结果：10 节点 × 判据验收矩阵

矩阵规模 **10 节点 × 54 条判据**（每条判据带显式裁决，缺席记 `withheld`）。

**表 2 · 节点级验收汇总（判据通过数 / 已测判据 / 证据等级 / 前置条件 / critical）**

| ext_id | 判据通过 | 已测判据 | 节点裁决 | 证据等级 | 前置 done+pass | critical failure |
|---|---|---|---|---|---|---|
| IN40-N01 | 0/5 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N02 | 0/6 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N03 | 0/5 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N04 | 0/5 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N05 | 0/5 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N06 | 0/6 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N07 | 0/5 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N08 | 0/5 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N09 | 0/6 | 0 | ⛔ withheld | WITHHELD | no | no |
| IN40-N10 | 0/6 | 0 | ⛔ withheld | WITHHELD | no | no |

**逐判据明细（每个节点）**

<details><summary>IN40-N01 · withheld · 0/5</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | 网格收敛C_T差≤2% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 形式因子与ITTC'78偏差≤5% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | 干扰因子曲线单调 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 粘性压力占比15-25% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N02 · withheld · 0/6</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | Lindqvist与原文≤1% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 15工况R_ice偏差≤10% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | LES谱-5/3斜率R²≥0.95 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 高黏区误差≤|2|% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | 双轨交叉表落盘 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C6 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N03 · withheld · 0/5</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | C_L与Swenson偏差≤8% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 净推力增益≥5% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | EPL可达航速唯一解 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | CII改善≥1级 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N04 · withheld · 0/5</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | 导数与SIMMAN中值偏差≤15% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 相位差≤5° | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | 网格收敛≤3% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 振幅一致性≤5% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N05 · withheld · 0/5</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | 极限弯矩与CSR偏差≤5% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 模态频率偏差≤8% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | ISO 6954判级一致 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 叶频响应偏差≤10% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N06 · withheld · 0/6</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | η_O≤1 100%通过 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | P_E≤P_D 0违反 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | 空化数定义差≤0.1% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | Keller与Burrill一致 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | 闸门has_verdict字段 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C6 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N07 · withheld · 0/5</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | 水舱频率偏差≤2% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 调谐减摇率≥40% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | 鳍减摇RMS降≥50% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 能耗收益比≤0.3 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N08 · withheld · 0/5</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | Green函数偏差≤1% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | Lewis附加质量≤3% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | Vugts阻尼≤8% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 频率曲线光滑 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N09 · withheld · 0/6</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | ΔCf偏差≤0.5% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | Ca换算≤1% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | U_SN≤3% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | CII分船型逐格一致 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | Z%无单点外推值 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C6 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

<details><summary>IN40-N10 · withheld · 0/6</summary>

| cid | 判据原文 | 裁决 | measured | reference | 相对误差% | 证据 |
|---|---|---|---|---|---|---|
| C1 | 几何偏差≤1% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C2 | 常值传递≤1e-12 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C3 | 线性阶≥0.9 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C4 | 守恒偏差≤0.1% | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C5 | patch test三阶通过 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |
| C6 | pytest全绿 | ⛔ withheld | — | — | — | no acceptance submission at gate cutoff |

</details>

**计划级裁决数字**

| 量 | 值 |
|---|---|
| 通过节点数 | 0/10 |
| 通过率 P | 0.0000 |
| 阈值 | 0.80 |
| critical failure | none |
| 前置条件满足 (done+pass) | 0/10 |
| 计划裁决 | `retained_active` |
| 下一波提案触发 | no |
| 裁决理由 | pass rate 0/10 = 0.0% < 80%: plan stays active, nodes must iterate |

**机器自检（6 项）**

| cid | 要求 | 通过 | 数字 | 备注 |
|---|---|---|---|---|
| G1 | every prerequisite node is status=done with reproduction_verdict=pass | ❌ | 0/10 | satisfied: none |
| G2 | acceptance matrix covers all 10 wave-4 nodes with >=1 criterion each | ✅ | 10/10 nodes, 54 criteria | every cell carries an explicit verdict; withheld is a verdict, not a pass |
| G3 | pass-rate rule applied: completed iff rate >= 80% and no critical failure | ✅ | 0/10 = 0.0%, critical=0 | pass rate 0/10 = 0.0% < 80%: plan stays active, nodes must iterate |
| G4 | closing report carries a limitations section naming every open node and its cause | ✅ | 10/10 open nodes attributed, 13 limitation entries | unattributed: none |
| G5 | guide_tracker gaps enumerated as next-wave proposal raw material | ✅ | 8/12 metrics below target, 18 next-wave seeds | each gap carries metric_key, value, target and progress |
| G6 | report body digest computed and (when reviewed) signed by a non-participant | ✅ | digest=bcceb9b843a8…, reviewer=none | peer review pending |

## 4. 结果图

**图 · fig_w4_matrix.png**  
![fig_w4_matrix.png](artifacts/in40_g40_wave4_gate/figures/fig_w4_matrix.png)  
10 节点 x 判据验收矩阵：全部格子为 withheld（无提交），故矩阵表达的是证据空缺而非物理失败。

**图 · fig_w4_passrate.png**  
![fig_w4_passrate.png](artifacts/in40_g40_wave4_gate/figures/fig_w4_passrate.png)  
计划级裁决：通过 0/10，门槛 80%（虚线），裁决 retained_active。

**图 · fig_w4_guide_gaps.png**  
![fig_w4_guide_gaps.png](artifacts/in40_g40_wave4_gate/figures/fig_w4_guide_gaps.png)  
guide_tracker 各指标 progress：红=progress<0.5，黄=未达标，绿=达标；虚线为目标 1.0。

## 5. 误差判定：关键量 vs 基准（逐项显式回答是否 <3%）

误差判定只在节点递交了 `measured/reference` 时才有数；本轮无节点提交验收 JSON，因此全部为 `withheld`——**不是「误差合格」，而是「尚无误差可判」**，按纪律不得含糊。

| ext_id | 关键量 | 基准来源 | 实测 | 基准 | 相对误差% | <3%？ | 依据 |
|---|---|---|---|---|---|---|---|
| IN40-N01 | 网格收敛C_T差≤2% | declared in node goal | — | — | — | — | withheld |
| IN40-N02 | Lindqvist与原文≤1% | declared in node goal | — | — | — | — | withheld |
| IN40-N03 | C_L与Swenson偏差≤8% | declared in node goal | — | — | — | — | withheld |
| IN40-N04 | 导数与SIMMAN中值偏差≤15% | declared in node goal | — | — | — | — | withheld |
| IN40-N05 | 极限弯矩与CSR偏差≤5% | declared in node goal | — | — | — | — | withheld |
| IN40-N06 | η_O≤1 100%通过 | declared in node goal | — | — | — | — | withheld |
| IN40-N07 | 水舱频率偏差≤2% | declared in node goal | — | — | — | — | withheld |
| IN40-N08 | Green函数偏差≤1% | declared in node goal | — | — | — | — | withheld |
| IN40-N09 | ΔCf偏差≤0.5% | declared in node goal | — | — | — | — | withheld |
| IN40-N10 | 几何偏差≤1% | declared in node goal | — | — | — | — | withheld |

**结论**：0/10 个节点具备可判误差；无基准可对的项须声明原因，本轮原因统一为「节点未递交验收工件（mission 仍在运行或无报告）」——honest FAIL，不粉饰。

## 6. 谱系继承裁决

继承是**推导**而非声称：`MATURE` 需节点 `accept` 且至少一条判据对外部基准实测；`ITERATE` 表示能力仍在门内；`SHELVE` 保留给「已废弃且无证据」的节点。

| 谱系 | 节点 | 含义 |
|---|---|---|
| mature | — | 可进入下一波基线 |
| iterate | IN40-N01, IN40-N02, IN40-N03, IN40-N04, IN40-N05, IN40-N06, IN40-N07, IN40-N08, IN40-N09, IN40-N10 | 未过门，须迭代后再裁 |
| shelve | — | 暂搁，需重建 |

## 7. 局限节：未通过节点清单与归因（封闭四分类）

四分类：模型形式误差 `model_form` / 输入不确定度 `input_uncertainty` / 实现缺陷 `implementation_bug` / 证据缺口 `evidence_gap`。未通过节点逐条列明，不允许「还需努力」这种类。

| ext_id | 节点裁决 | 归因类 | 陈述 | 证据 | 置信 | 下一步 |
|---|---|---|---|---|---|---|
| IN40-N01 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='queued'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N02 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N03 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='queued'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N04 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N05 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N06 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N07 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N08 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='queued'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N09 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |
| IN40-N10 | withheld | evidence_gap | node delivered no acceptance submission before the gate cutoff | campaign_nodes.status='in-flight'; mission status='running'; linked reports=0 | high | let the node finish and publish its acceptance artifact, then re-run the gate |

**报告级局限（自述，按纪律显式声明）**

- IN40-N01 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5642 status=`queued`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/5 通过（5 条 withheld）。
- IN40-N02 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5537 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/6 通过（6 条 withheld）。
- IN40-N03 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5641 status=`queued`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/5 通过（5 条 withheld）。
- IN40-N04 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5539 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/5 通过（5 条 withheld）。
- IN40-N05 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5543 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/5 通过（5 条 withheld）。
- IN40-N06 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5544 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/6 通过（6 条 withheld）。
- IN40-N07 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5540 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/5 通过（5 条 withheld）。
- IN40-N08 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5637 status=`queued`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/5 通过（5 条 withheld）。
- IN40-N09 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5545 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/6 通过（6 条 withheld）。
- IN40-N10 未过门：节点裁决 `withheld`，归因 `evidence_gap`；campaign_nodes.status=`in-flight`、mission 5542 status=`running`、关联报告 0 篇、reproduction_verdict=`None`；判据 0/6 通过（6 条 withheld）。
- 本轮核销门在节点仍 inflight 的证据截止时刻裁决：缺席证据记 withheld 而非通过，因此通过率 0% 表达的是「尚无证据」，不是「物理结论错误」。
- guide_tracker 的 progress 是代理口径（如首次相关性/无人值守用去重案例数代理），不能直接等同工程覆盖率，下一波提案须先校准口径再定目标。
- 跨求解器一致性（N10）与推进链不变量（N06）在本轮无独立探针复核，平台侧 API 只能作为旁证，不能替代节点自证。

## 8. guide_tracker 缺口（下一波提案原材料）

`tensor_guide_tracker` 共 12 项指标，8 项未达标；progress = metric_value / target_value（target=0 视为已达标）。

| metric_key | 当前值 | 目标值 | progress | 缺口 | 未达标 |
|---|---|---|---|---|---|
| guide.kb_samples | 0.000 | 10000.000 | 0.0000 | 10000.000 | yes |
| repro_rate | 0.000 | 0.950 | 0.0000 | 0.950 | yes |
| guide.unattended_cases | 65.000 | 1000.000 | 0.0650 | 935.000 | yes |
| guide.benchmark_cases | 1.000 | 10.000 | 0.1000 | 9.000 | yes |
| guide.ip_assets | 1.000 | 5.000 | 0.2000 | 4.000 | yes |
| guide.tool_categories | 4.000 | 5.000 | 0.8000 | 1.000 | yes |
| guide.lineage_coverage | 0.820 | 1.000 | 0.8199 | 0.180 | yes |
| guide.success_rate | 0.799 | 0.900 | 0.8882 | 0.101 | yes |
| blacklisted_models | 0.000 | — | 1.0000 | 0.000 | no |
| cheat_detected | 0.000 | — | 1.0000 | 0.000 | no |
| guide.qr_pass_rate | 1.000 | 1.000 | 1.0000 | 0.000 | no |
| total_strikes | 0.000 | — | 1.0000 | 0.000 | no |

**最主要的未覆盖能力（按缺口排序，取前 6）**

- `guide.kb_samples`：0.000 / 10000.000（progress 0.00%）— 知识库 ≥3学科/3型船/≥1万条 | ship_samples 行数（报告 ≠ 样本，不可混算） | src=db:SELECT count(*) AS samples FROM ship_samples | progress=0.0
- `repro_rate`：0.000 / 0.950（progress 0.00%）— 无备注
- `guide.unattended_cases`：65.000 / 1000.000（progress 6.50%）— 无人值守 ≥1000 工况 | 扫荡线成功案例去重数（案例级代理口径，每案例≈1-3 工况） | src=db:SELECT count(DISTINCT substring(goal from '案例 ([a-zA-Z0-9_]+)（')) AS cases FROM tensor_dev_missions WHERE goal LIKE '%案例 %（Tensor%' AND status='succeeded' | progress=0.065
- `guide.benchmark_cases`：1.000 / 10.000（progress 10.00%）— 对标算例 ≥10（SUBOFF/DTMB/KVLCC2） | 台账：已完成的独立对标算例数（KVLCC2@15kn 真算=1） | src=ledger:benchmark_cases | progress=0.1
- `guide.ip_assets`：1.000 / 5.000（progress 20.00%）— 标准≥3/专利≥5/论文≥5/软著≥5 | 台账：计数口径取「已立项/骨架」数（论文骨架 1） | src=ledger:ip_assets | progress=0.2
- `guide.tool_categories`：4.000 / 5.000（progress 80.00%）— 工具接入 ≥5 类 | 台账：已真实实现工具类计数（CFD/FEA/DEM/势流切片/经验公式/规范） | src=ledger:tool_categories | progress=0.8

## 9. 下一波提案触发与原材料

```json
{
  "triggered": false,
  "reason": "pass rate 0/10 = 0.0% < 80%: plan stays active, nodes must iterate",
  "blocked_by": [
    "IN40-N01",
    "IN40-N02",
    "IN40-N03",
    "IN40-N04",
    "IN40-N05",
    "IN40-N06",
    "IN40-N07",
    "IN40-N08",
    "IN40-N09",
    "IN40-N10"
  ],
  "carry_over_gaps": [
    "guide.kb_samples",
    "repro_rate",
    "guide.unattended_cases",
    "guide.benchmark_cases",
    "guide.ip_assets",
    "guide.tool_categories",
    "guide.lineage_coverage",
    "guide.success_rate"
  ],
  "seeds": []
}
```

**下一波提案种子（种子清单）**

- 补齐 `guide.kb_samples`（0/10000，progress 0.0%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `repro_rate`（0/0.95，progress 0.0%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `guide.unattended_cases`（65/1000，progress 6.5%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `guide.benchmark_cases`（1/10，progress 10.0%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `guide.ip_assets`（1/5，progress 20.0%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `guide.tool_categories`（4/5，progress 80.0%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `guide.lineage_coverage`（0.8199/1，progress 82.0%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- 补齐 `guide.success_rate`（0.7994/0.9，progress 88.8%）：由未过门节点 IN40-N01, IN40-N02, IN40-N03 的能力缺口反推提案
- IN40-N01（TensorFVM）复跑提案：目标为把 5 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N02（ship-performance-platform）复跑提案：目标为把 6 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N03（ship-performance-platform）复跑提案：目标为把 5 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N04（TensorFVM）复跑提案：目标为把 5 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N05（TensorFEM）复跑提案：目标为把 5 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N06（ship-performance-platform）复跑提案：目标为把 6 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N07（ship-performance-platform）复跑提案：目标为把 5 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N08（ship-performance-platform）复跑提案：目标为把 5 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N09（ship-performance-platform）复跑提案：目标为把 6 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）
- IN40-N10（TensorPhysics）复跑提案：目标为把 6 条判据全部测出并递交 acceptance JSON（当前 0 篇报告、status=in-flight）

**计划状态变更**：`未应用`（裁决 `retained_active`；`wave4_gate.plan_update_sql()` 给出对应语句，未达门槛时为空操作，计划保持 `active`）。

---

本报告由 `tensorphysics.campaign.wave4_gate` 程序化生成；node 裁决、通过率、谱系、缺口全部由证据快照重算。发布回执见文末（render 时刻 publish_status=`not_published`，回执行为权威记录，且不改动证据 digest）。任何把裁决改写得更乐观的版本都会被 `verify_report_integrity` 拒绝（honest FAIL 不可覆盖）。

> 发布回执：`tensor_reports.id=6518`（147662 bytes，含 3 张图内嵌），报告摘要 digest `bcceb9b843a8d33cf54b6840e4ef2b36560424d829b0569c9f405a23111b24db` 未变——发布不修改证据裁决。
