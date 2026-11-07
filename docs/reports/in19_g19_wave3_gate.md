# 波3（IN19）核销门报告：全节点验收汇总与谱系继承裁决

- 战役计划：`求解器验证与船舶能力深化·谱系继承波3`（plan_id=39）
- 本门节点：`IN19-G19`（mission_id=5208）
- 证据截止时刻（cutoff）：`2026-11-07T06:51:43Z`
- 报告正文摘要（sha256，**规范化 JSON 核心字段**，非 markdown 文件哈希）：`5022e51af33a2b5bb137434a6ae4ca8d1f55d821e686e9156cedde11c2c616e5`
- **门最终裁决：`NOT_ACCEPTED_NODES_OPEN`**

> 本报告的每一个数字都由 `tensorphysics.campaign.wave3_gate` 从原始证据快照（战役表 dump + 平台探针原始响应）程序化生成，报告正文摘要参与 peer review 签名，人工无法在不改变摘要的前提下修改结论。

## 1 案例介绍：波3 的九个任务节点

波3（IN19：求解器验证与船舶能力深化·谱系继承）把上一波已验证的求解器与船舶能力推进到「可判据验收」：每个节点自带机器可算验收判据（C1…Cn），而本门（IN19-G19）负责在证据层面把它们核销成一张可继承的谱系。门本身不生产物理结果，只做三件事：汇总验收、归因未过、裁决继承。

| 节点 | 仓库 | 名称 | 任务状态 | 成熟化项 | 验收结论 | 判据(已测/总) |
|---|---|---|---|---|---|---|
| IN19-T1 | TensorLBM | LBM曲边壁面边界格式力误差定标：Bouzidi线性/二次+动量交换+应力积分四因子正交扫描 | running | 否 | WITHHELD | 0/5 |
| IN19-T2 | TensorPhysics | 跨求解器界面算子通量守恒契约：FVM/LBM/DEM插值投影算子守恒残差门禁 | running | 否 | WITHHELD | 0/6 |
| IN19-T3 | TensorDEM | 冰接触键滞弹性与延性-脆性转变：Maxwell-Wiechert延迟弹性+率相关强度标定 | running | 否 | WITHHELD | 0/7 |
| IN19-T4 | ship-performance-platform | 湿表面积闭包三分裂消除：πBL/Holtrop-84/Denny-Mumford规范级裁决与阻力链闸门成熟化 | running | 是 | WITHHELD | 0/5 |
| IN19-T5 | ship-performance-platform | 螺旋桨敞水REAL锚+Wageningen B系列+ITTC-78尺度效应+船-桨-舵交互成熟化 | running | 是 | WITHHELD | 0/5 |
| IN19-T6 | ship-performance-platform | SUBOFF阻力精度修复与KVLCC2基准闭环成熟化：从67%误差到≤5%达标 | running | 是 | WITHHELD | 0/5 |
| IN19-T7 | ship-performance-platform | 波浪增阻与耐波性RAO验证：Holtrop波浪增阻+势流BEM纵摇/垂荡RAO对标ITTC推荐谱 | running | 否 | WITHHELD | 0/5 |
| IN19-T8 | ship-performance-platform | EEXI/CII数字验证成熟化：IMO MEPC.328(75)公式实现+典型船型达标评级+ESD效果量化 | running | 是 | WITHHELD | 0/5 |
| IN19-T9 | ship-performance-platform | 船舶性能预报不确定度量化框架：Sobol敏感度+蒙特卡洛传播+U_V对KPI闸门显式裁决 | running | 否 | WITHHELD | 0/6 |

## 2 怎么做的：证据链与判定流程

门的工作流是五步、全自动、可重放：

1. **证据快照**：只读导出 `campaign_nodes` ⊗ `tensor_dev_missions`（plan_id=39，10 行），记录每个节点的 goal 文本、mission 状态、交付物；快照文件与 sha256 落在审计链中。
2. **判据解析**：从 goal 的「机器可算验收判据」段用正则解析出 `C1…Cn`，矩阵行数因此是机器产物而非人工抄写（判据 C1 可机器校验）。
3. **验收判定**：每个判据只有四种结局——PASS / FAIL / PARTIAL / WITHHELD。**未交付证据一律 WITHHELD，不得升格为 PASS**；节点结论由判据汇总得到（任一 FAIL → FAIL；全 WITHHELD → WITHHELD；否则 PARTIAL）。
4. **归因与继承**：非 PASS 节点必须落进封闭四分类根因（模型形式误差 / 输入不确定度 / 实现 bug / 证据缺口）；继承结论由节点记录派生（MATURE 需要 PASS **且** 至少一条判据对过外部基准），禁止人工断言。
5. **门自检 + 独立探针**：对五项战役判据 C1–C5 逐项给出通过状态与数字；同时对可达的 ship-performance-platform 端点做**门外独立探针**，使误差表在节点未交付时仍有实测数字，也让「已成熟」的说法可被外部否证。

```
raw/nodes_plan39.json ──┐
raw/platform_probes.json ─┼─> wave3_gate.build_report ─> gate_report.json (digest)
references.json ────────┘                        └─> gate_report.md (本文)
```

## 3 验收矩阵（判据 C1）

| 节点 | 判据 | 判据摘要 | 结论 | 实测 | 基准 | 相对误差 |
|---|---|---|---|---|---|---|
| IN19-T1 | C1 | Bouzidi二次格式在N≥256时CD误差<3% | WITHHELD | — | — | — |
| IN19-T1 | C2 | 观测收敛阶p≥1.9(二次)或p≥0.9(线性) | WITHHELD | — | — | — |
| IN19-T1 | C3 | ME力提取与应力积分力提取在 finest grid 上差异<1% | WITHHELD | — | — | — |
| IN19-T1 | C4 | 32组扫描全部进pytest参数化门禁 | WITHHELD | — | — | — |
| IN19-T1 | C5 | 负控：Bouzidi线性在粗网格(N=64)必须差>10%，负控不红即判判据无效 | WITHHELD | — | — | — |
| IN19-T2 | C1 | 常数场守恒残差<1e-14 | WITHHELD | — | — | — |
| IN19-T2 | C2 | 线性场观测阶p≥1.9 | WITHHELD | — | — | — |
| IN19-T2 | C3 | 总通量守恒残差/Σ\|flux\|<1e-12 | WITHHELD | — | — | — |
| IN19-T2 | C4 | ME回写动量守恒<1e-10 | WITHHELD | — | — | — |
| IN19-T2 | C5 | FVM↔LBM与LBM↔DEM两条链守恒后场差<2% | WITHHELD | — | — | — |
| IN19-T2 | C6 | 每个算子有一个「故意破坏守恒」负控测试必须红 | WITHHELD | — | — | — |
| IN19-T3 | C1 | 延性-脆性转折落在1e-4…1e-3 s⁻¹文献区间，强度峰值5-15MPa | WITHHELD | — | — | — |
| IN19-T3 | C2 | 六档峰值强度vs文献误差<15%(转折区<20%) | WITHHELD | — | — | — |
| IN19-T3 | C3 | 瞬时弹性模量vs文献E=9-10GPa误差<10% | WITHHELD | — | — | — |
| IN19-T3 | C4 | 蠕变曲线幂律拟合R²>0.95 | WITHHELD | — | — | — |
| IN19-T3 | C5 | 键能量账本闭合残差<1e-10 | WITHHELD | — | — | — |
| IN19-T3 | C6 | 含负控(率无关模型在转折区外必须差>30%) | WITHHELD | — | — | — |
| IN19-T3 | C7 | 三种子离散度<10% | WITHHELD | — | — | — |
| IN19-T4 | C1 | 30船型上\|ΔS\|/S≤1.5%(对权威闭包) | WITHHELD | — | — | — |
| IN19-T4 | C2 | m4926的δS/S从+5.5767%降到\|δ\|<1%(同一复算脚本) | WITHHELD | — | — | — |
| IN19-T4 | C3 | 语料内算例C_T漂移<2%且现网600项pytest全绿 | WITHHELD | — | — | — |
| IN19-T4 | C4 | 禁改任何文献参考值 | WITHHELD | — | — | — |
| IN19-T4 | C5 | 偏差spread若>1.5%必须输出机理解释并降级为honest-WITHHELD | WITHHELD | — | — | — |
| IN19-T5 | C1 | 恒等式η0=J·KT/KQ相对残差<1e-12 | WITHHELD | — | — | — |
| IN19-T5 | C2 | 与公开B4-70/B4-55图谱表格点\|ΔKT\|<3%、\|ΔKQ\|<3%、\|Δη0\|<3% | WITHHELD | — | — | — |
| IN19-T5 | C3 | 全尺度换算后1-KT/J²线性段R²>0.98 | WITHHELD | — | — | — |
| IN19-T5 | C4 | 48条曲线全部进pytest参数化门禁，API响应里MOCK标记归零 | WITHHELD | — | — | — |
| IN19-T5 | C5 | 船-桨-舵交互t/w/η_R与实验对标误差<10% | WITHHELD | — | — | — |
| IN19-T6 | C1 | SUBOFF四工况\|C_T_pred-C_T_exp\|/C_T_exp≤5% | WITHHELD | — | — | — |
| IN19-T6 | C2 | KVLCC2在Fr=0.142的C_T误差≤5% | WITHHELD | — | — | — |
| IN19-T6 | C3 | 阻力链各环节贡献量化输出(百分比) | WITHHELD | — | — | — |
| IN19-T6 | C4 | 现网600项pytest全绿 | WITHHELD | — | — | — |
| IN19-T6 | C5 | 修复前后diff报告(程序化注入，零手抄) | WITHHELD | — | — | — |
| IN19-T7 | C1 | Holtrop波浪增阻系数与ITTC推荐值误差<10% | WITHHELD | — | — | — |
| IN19-T7 | C2 | Wigley III纵摇/垂荡RAO峰值频率与势流理论解误差<5% | WITHHELD | — | — | — |
| IN19-T7 | C3 | KCS RAO与SIMMAN 2008试验数据峰值误差<10% | WITHHELD | — | — | — |
| IN19-T7 | C4 | RAO积分复现波浪增阻量级误差<15% | WITHHELD | — | — | — |
| IN19-T7 | C5 | pytest≥8全绿 | WITHHELD | — | — | — |
| IN19-T8 | C1 | EEXI计算值与IMO参考值\|err\|≤3%(3艘典型船型) | WITHHELD | — | — | — |
| IN19-T8 | C2 | CII评级与参考评级完全匹配 | WITHHELD | — | — | — |
| IN19-T8 | C3 | ESD效果量级与制造商数据\|err\|≤15%(PBCF 2-4%, STS 1-3%, 空气润滑5-10%) | WITHHELD | — | — | — |
| IN19-T8 | C4 | pytest≥10全绿 | WITHHELD | — | — | — |
| IN19-T8 | C5 | API端点文档+OpenAPI schema更新 | WITHHELD | — | — | — |
| IN19-T9 | C1 | Sobol指数ΣS_i≈1(机器校验) | WITHHELD | — | — | — |
| IN19-T9 | C2 | 蒙特卡洛N=1000收敛：μ_R,σ_R相对变化<1%(N=500 vs N=1000) | WITHHELD | — | — | — |
| IN19-T9 | C3 | U_input量化输出且U_input<U_SN(若U_input>U_SN须说明) | WITHHELD | — | — | — |
| IN19-T9 | C4 | U_V量化输出，对SUBOFF 4例做显式裁决 | WITHHELD | — | — | — |
| IN19-T9 | C5 | 若裁决与KPI闸门(\|err\|≤5%)不一致须解释原因 | WITHHELD | — | — | — |
| IN19-T9 | C6 | pytest≥12全绿 | WITHHELD | — | — | — |

矩阵覆盖 **9/9 节点、49 条判据**；每行都带显式结论，没有任何一行因为「无证据」而被省略。

## 4 误差判定表（判据 C4）

**锚定规则（先验，非事后挑选）**：每个节点只锚定一条关键量——若该节点有门侧探针，取其 |相对误差| 最大者（最不利证据优先）；否则取节点自己已交付的第一条实测判据；两者皆无则记为 `withheld`。误差列由**未圆整原值**计算，表中展示值可能已按有效位圆整，全精度见 §5。

关键量 vs 基准逐项列出。`basis=node_criterion` 表示由节点自己的交付判定，`gate_probe` 表示门外独立实测（当前平台状态，非节点交付），`withheld` 表示截止时刻无任何实测——**声明为缺失，而不是留空**。

| 节点 | 关键量 | 基准来源 | 阈值 | 实测 | 基准 | 相对误差 | 达标 | 依据 |
|---|---|---|---|---|---|---|---|---|
| IN19-T1 | Bouzidi二次格式在N≥256时CD误差<3% | declared in node goal | — | — | — | — | — | withheld |
| IN19-T2 | 常数场守恒残差<1e-14 | declared in node goal | — | — | — | — | — | withheld |
| IN19-T3 | 延性-脆性转折落在1e-4…1e-3 s⁻¹文献区间，强度峰值5-15MPa | declared in node goal | — | — | — | — | — | withheld |
| IN19-T4 | KCS 湿表面积 S (L=230 m, T=10.8 m, Cb=0.651) | ship-performance-platform kcs_spec_data… | 1.50% | 9651.49 | 9424 | +2.41% | **否** | gate_probe |
| IN19-T5 | B5-55 扭矩系数 K_Q @ J=0.3727 | ship-performance-platform artifacts/ref… | 3.00% | 0.04771 | 0.029819 | +60.00% | **否** | gate_probe |
| IN19-T6 | SUBOFF 总阻力系数 C_T (V=5 m/s) | ship-performance-platform backend/data/… | 5.00% | 0.024675 | 0.0032 | +671.09% | **否** | gate_probe |
| IN19-T7 | Holtrop波浪增阻系数与ITTC推荐值误差<10% | declared in node goal | — | — | — | — | — | withheld |
| IN19-T8 | EEXI计算值与IMO参考值\|err\|≤3%(3艘典型船型) | declared in node goal | — | — | — | — | — | withheld |
| IN19-T9 | Sobol指数ΣS_i≈1(机器校验) | declared in node goal | — | — | — | — | — | withheld |

合计 9 行：3 行有实测、6 行 `withheld`（原因均为「节点未交付验收证据」，逐行已注明）。

## 5 门外独立探针（可外部否证的基线快照）

| 探针 | 节点 | 端点 | 物理量 | 实测 | 基准 | 基准来源 | 基准不确定度 | 相对误差 | 阈值 |
|---|---|---|---|---|---|---|---|---|---|
| P1-SUBOFF-S | 311 | `/calculations/resistance` | SUBOFF 模型湿表面积 S | 5.18 m2 | 6.328 m2 | ship-performance-platform backend/data/… | ±10% | **-18.14%** | 3% |
| P2-SUBOFF-CT | 311 | `/calculations/resistance` | SUBOFF 总阻力系数 C_T (V=5 m/s) | 0.024675 - | 0.0032 - | ship-performance-platform backend/data/… | ±10% | **+671.09%** | 5% |
| P3-KCS-S | 309 | `/calculations/resistance` | KCS 湿表面积 S (L=230 m, T=10.8 m… | 9651.49 m2 | 9424 m2 | ship-performance-platform kcs_spec_data… | ±0.5% | **+2.41%** | 1.5% |
| P4-KT | 310 | `/propulsion/predict` | B5-55 推力系数 K_T @ J=0.3727 | 0.3377 - | 0.241143 - | ship-performance-platform artifacts/ref… | ±5% | **+40.04%** | 3% |
| P5-KQ | 310 | `/propulsion/predict` | B5-55 扭矩系数 K_Q @ J=0.3727 | 0.04771 - | 0.029819 - | ship-performance-platform artifacts/ref… | ±5% | **+60.00%** | 3% |

探针只回答「此刻主分支上的平台在什么水平」，不替代节点验收；它的用途是让波4 的起点可被外部复算，并防止把探针数值悄悄写成节点成果。

## 6 根因分析（判据 C2）

共 9 个节点未通过，逐节点归因如下（封闭四分类）。

| 节点 | 根因类别 | 归因陈述 | 证据 | 置信 | 建议动作 |
|---|---|---|---|---|---|
| IN19-T1 | `evidence_gap`（证据缺口） | node delivered no acceptance evidence before the gate cutoff | mission 5201 status=running; no linked report, no acceptance JSON, worktree clean at cutoff | high | rerun the node, publish an acceptance artifact, then re-run the gate |
| IN19-T2 | `evidence_gap`（证据缺口） | node delivered no acceptance evidence before the gate cutoff | mission 5207 status=running; no linked report, no acceptance JSON, worktree clean at cutoff | high | rerun the node, publish an acceptance artifact, then re-run the gate |
| IN19-T3 | `evidence_gap`（证据缺口） | node delivered no acceptance evidence before the gate cutoff | mission 5199 status=running; no linked report, no acceptance JSON, worktree clean at cutoff | high | rerun the node, publish an acceptance artifact, then re-run the gate |
| IN19-T4 | `evidence_gap`（证据缺口） | no acceptance submission; gate probe on /calculations/resistance measures +2.41% on KCS 湿表面积 S (L=230 m, T=10.8 m, Cb=0… | mission 5200 status=running; probe P3-KCS-S measured 9651.49 vs reference 9424 m2 (ship-performance… | high | deliver the node's own acceptance run before the gate can adjudicate |
| IN19-T5 | `evidence_gap`（证据缺口） | no acceptance submission; gate probe on /propulsion/predict measures +60.00% on B5-55 扭矩系数 K_Q @ J=0.3727, so the node … | mission 5203 status=running; probe P5-KQ measured 0.04771 vs reference 0.029819 - (ship-performance… | high | deliver the node's own acceptance run before the gate can adjudicate |
| IN19-T6 | `evidence_gap`（证据缺口） | no acceptance submission; gate probe on /calculations/resistance measures +671.09% on SUBOFF 总阻力系数 C_T (V=5 m/s), so th… | mission 5204 status=running; probe P2-SUBOFF-CT measured 0.024675 vs reference 0.0032 - (ship-perfo… | high | deliver the node's own acceptance run before the gate can adjudicate |
| IN19-T7 | `evidence_gap`（证据缺口） | node delivered no acceptance evidence before the gate cutoff | mission 5205 status=running; no linked report, no acceptance JSON, worktree clean at cutoff | high | rerun the node, publish an acceptance artifact, then re-run the gate |
| IN19-T8 | `evidence_gap`（证据缺口） | EEXI/CII 规范实现接口在当前主分支不存在（GET /calculations/eexi 与 /calculations/cii 均 404），属能力缺席而非精度不足 | raw/platform_probes.json: eexi_endpoint status=404, cii_endpoint status=404 | high | 按 MEPC.328(75) 实现 EEXI/CII 数值链并暴露端点，再以 IMO 参考船复算 |
| IN19-T9 | `evidence_gap`（证据缺口） | node delivered no acceptance evidence before the gate cutoff | mission 5206 status=running; no linked report, no acceptance JSON, worktree clean at cutoff | high | rerun the node, publish an acceptance artifact, then re-run the gate |

按类别汇总：`evidence_gap` → IN19-T1, IN19-T2, IN19-T3, IN19-T4, IN19-T5, IN19-T6, IN19-T7, IN19-T8, IN19-T9。

**诚实边界**：截止时刻 9/9 个节点的 mission 仍为进行中，未通过的直接原因是「**证据缺口**」，而不是「物理不达标」——两者不可互相冒充：把证据缺口写成物理失败会低估能力，写成通过则高估。门只记录「此刻无法核销」，并保留门外探针给出的当前水平作为定量锚点。

## 7 谱系继承裁决（判据 C3）

| 节点 | 验收结论 | 继承裁决 | 理由 |
|---|---|---|---|
| IN19-T1 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running），无对过基准的实测证据可继承 |
| IN19-T2 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running），无对过基准的实测证据可继承 |
| IN19-T3 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running），无对过基准的实测证据可继承 |
| IN19-T4 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running）；门侧探针实测 KCS 湿表面积 S (L=230 m, T=10.8 m, Cb=0.651) 偏差 +2.41%（阈值 1.5%），交付后仍须先过该判据 |
| IN19-T5 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running）；门侧探针实测 B5-55 扭矩系数 K_Q @ J=0.3727 偏差 +60.00%（阈值 3%），交付后仍须先过该判据 |
| IN19-T6 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running）；门侧探针实测 SUBOFF 总阻力系数 C_T (V=5 m/s) 偏差 +671.09%（阈值 5%），交付后仍须先过该判据 |
| IN19-T7 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running），无对过基准的实测证据可继承 |
| IN19-T8 | WITHHELD | **迭代 → 保留在波4轨道** | 能力在主干缺席（EEXI/CII 端点 404），属未实现而非精度不足 → 先按规范实现，再由门复算后才能继承 |
| IN19-T9 | WITHHELD | **迭代 → 保留在波4轨道** | 验收未提交（mission running），无对过基准的实测证据可继承 |

汇总：成熟 0 个（无）；迭代 9 个（IN19-T1, IN19-T2, IN19-T3, IN19-T4, IN19-T5, IN19-T6, IN19-T7, IN19-T8, IN19-T9）；搁置 0 个（无）。

## 8 门的自检：五项战役判据

| 判据 | 要求 | 结果 | 数字 | 说明 |
|---|---|---|---|---|
| C1 | acceptance matrix covers all 9 wave-3 nodes | PASS | 9/9 | every node carries at least one parsed machine criterion with explicit verdict |
| C2 | every non-accepted node has a root-cause analysis in the closed taxon… | PASS | 9/9 | classes: model_form_error, input_uncertainty, implementation_bug, evidence_gap |
| C3 | lineage inheritance explicit for every node and derived (never assert… | PASS | 9/9 | MATURE/ITERATE/SHELVE assigned per node; no MATURE without ACCEPT |
| C4 | summary report carries an itemised error judgement table (key quantit… | PASS | 9/9 rows (3 measured, 6 withheld) | withheld rows are stated, not omitted |
| C5 | report passed peer review by at least one non-participant | PASS | 1/1 reviewer | reviewer=tp-ver-6 (verifier mission 5213, review report 5922), verdict=accept_with_minor,… |

门最终裁决 `NOT_ACCEPTED_NODES_OPEN`：C1–C5 全部通过，节点级 0/9 通过。判据通过不等于物理达标——它只证明「报告如实、可复算、可审计」。

## 9 图

![验收矩阵](artifacts/in19_g19_wave3_gate/figures/acceptance_matrix.png)

*图1 波3 验收矩阵：行=节点，列=该节点的机器判据；灰=未评（WITHHELD），色块=通过/部分/失败。整幅矩阵几乎全灰，即是本轮门的定量结论。*

![误差谱](artifacts/in19_g19_wave3_gate/figures/error_spectrum.png)

*图2 门外独立探针的相对误差谱（对数纵轴）与各自阈值；柱高超出阈线即为当前不达标，标注值为实测误差。*

![谱系](artifacts/in19_g19_wave3_gate/figures/lineage_summary.png)

*图3 谱系继承分布：按 repo 归并的 MATURE/ITERATE/SHELVE 计数，以及判据状态构成。*

## 10 局限与适用范围

1. 证据截止时刻九个任务节点（306–314）的 mission 均为 running：门判定的是『此刻能否核销』，不是『物理是否可达』。节点交付后必须重跑本门（evaluate 模式读同一套 raw 快照 + 新提交），旧结论不得沿用。
2. 门侧探针只覆盖可达的 4 个 ship-performance-platform 端点；TensorLBM/TensorDEM 节点（306/308）与 TensorPhysics 自身的跨求解器契约节点（307）在门侧无独立复算，其判据只能是 WITHHELD，不能由门代劳。
3. SUBOFF 基准（C_T=0.0032）在仓储中未标注对应的 Re 数，且探针工况 Fn=0.765 落在 Holtrop 适用域（Fn 0.10–0.45）之外：观测到的 +671% 是『越域外推 + 几何闭包错配』的合成量，不能单独归因于某一环节。
4. B 系列基准为 6 点数字化表格（±5%），探针在 P/D=0.95 取得 K_T/K_Q 却只能对 P/D=1.0 内插比较；K_T/K_Q 的绝对偏差因此有 ±5% 基准不确定度，但观测到 +41%/+61% 远超该量级。
5. EEXI/CII 能力缺席的判定基于端点 404（GET /calculations/eexi、/calculations/cii）；若实现存在于其他路径/网关，本门结论需按新证据修订——门不推测未暴露的实现。
6. peer review 的独立性由评审者自证（非本波参与者）；门只能校验『签名摘要与正文摘要一致』与『评审结论为接受』，无法在机制上证明评审者未参与本波。这是本门 C5 的已知弱点。
7. 误差判定以相对误差为单位，未做 GCI/网格收敛或不确定度传播（后者本是节点314的任务，尚未交付）；因此表中的误差是点估计，不含验证不确定度 U_V。

## 11 下一波提案方向

1. 波4 首要：把『节点交付 → 门自动重跑』做成闭环。本门 evaluate 模式已是纯函数（读 raw 快照出报告），只需在 306–314 任一节点 complete 时触发一次，避免核销门与节点交付脱节。
2. 副艇体（SUBOFF）阻力的模型形式修正：Holtrop 剩余阻力项在 Fn>0.45 外推即失效，应改为 ITTC-78 形状因子(1+k)三分法 + 副艇体专用剩余阻力闭包，并以 C_T=0.0032 作 5% 门禁；该项是节点311 的核心欠账。
3. 推进链：以 Oosterveld & van Oossanen (1975) 24 项多项式取代当前 BEMT/回归混合路径；门侧探针已显示 K_T/K_Q 偏差 +41%/+61%，先把敞水面做成 REAL 再谈船-桨-舵交互（节点310 的两段式交付）。
4. 湿表面积：三分裂（πBL/Holtrop-84/Denny-Mumford/回转体近似）必须收敛为单一权威闭包并记录适用域；门侧探针显示同一 KCS 在不同路径下相差 2.4%–4.4%，已与 1.5% 判据冲突（节点309）。
5. EEXI/CII：MEPC.328(75) 是纯规范计算，应最先在其上取得『规范级 REAL』证据（端点当前 404）；建议波4 把它作为独立可交付项，而不是挂在成熟化节点尾部。
6. 不确定度：节点314 的 U_V 框架应前置到阻力链修复之前——否则波4 会在没有 U_V 的情况下继续用 5% 点估计判达标，无法区分『模型形式误差』与『输入噪声』。
7. 跨求解器契约（节点307）应产出一个可被 FVM/LBM/DEM 三方共同调用的守恒残差检查器，并纳入 CI 门禁，避免每波重复人工比对。
8. 门机制本身：把『非本波参与者 peer review』接入平台的 repro/verdicts 通道（该通道当前 0 条记录），使 C5 从自证变为机制保证。

## 12 Peer review（判据 C5）

- 评审者：`tp-ver-6 (verifier mission 5213, review report 5922)`（TensorPlatform verifier role; did not own any plan_id=39 node (306-314)）
- 独立性声明：非本波参与者
- 评审对象摘要：`5022e51af33a2b5bb137434a6ae4ca8d1f55d821e686e9156cedde11c2c616e5`（与本文正文摘要一致：True）
- 评审结论：`accept_with_minor`，签署时间 `2026-11-07T06:59:59Z`
- 评审发现：
  - F1-F4 confirmed: independent endpoint recomputation reproduces ct=0.024675, S=5.18 m2, KCS S=9651.49 m2, EEXI/CII=404/404, and the 9 nodes / 49 criteria / 9 error rows (3+6) / 9 root causes / 0 mature lineage counts exactly
  - F5 clean: no exaggeration found - no WITHHELD upgraded to PASS, gate probes not passed off as node deliverables, no error rounded into compliance, C5 honestly recorded as FAIL
  - F6 minor open: the declared body digest (sha256 of the canonical JSON payload) is not the sha256 of the published markdown, and the payload itself was not published, so C5 anchoring was not third-party verifiable
  - F7 minor open: engine + raw evidence were not retrievable at the git revision cited in the audit chain (not yet committed at review time)
  - F8 minor open: error table displayed 4 significant digits while errors were computed from unrounded values (+670.94% vs +671.09% if recomputed from displayed value)
  - F9 note: the per-node anchor quantity was not declared in the report, so readers could not tell whether anchoring was selected after the fact (the change made the report more conservative, not more flattering)
  - F10 scope: P4/P5 (B5-55 K_T/K_Q) not independently recomputed because references.json was not retrievable at the cited revision; mission states were not re-read from the campaign DB - neither changes the BLOCKED verdict
- 要求修改：
  - R1 make the digest reproducible: publish the covered canonical payload (or its exact recipe) so the C5 signature is verifiable by a third party
  - R2 commit the gate engine and raw evidence at the cited git revision, and correct the audit-chain sha256 references
  - R3 raise the display precision of the error table, or state that the error column is computed from unrounded values
  - R4 declare the rule by which each node's key quantity is anchored, so anchoring cannot look post-hoc
  - R5 after this review, re-run the gate in evaluate mode so C5 goes 0/1 -> 1/1, but keep the overall verdict BLOCKED_CRITERIA (0/9 node acceptances) and relax no node threshold

### 既往评审（签署摘要已被后续版本取代，记录其发现与作者回应）

| 评审者 | 版本摘要 | 结论 | 主要发现 | 作者回应 |
|---|---|---|---|---|
| `tp-qa-1 (verifier mission 5212, review report 5921)` | `e1f9bee878b541d8…` | accept_with_minor | independent recomputation confirmed every published number (ct=0.024675, S=5.18/9651.49, EEXI/CII=404) and all C1-C5 co… | Accepted and superseded: the review signed the pre-revision digest e1f9bee8 (report 5915), so it is recorded as a prior review rather than … |

## 13 审计链

| # | 资料 | sha256 | 说明 |
|---|---|---|---|
| 1 | `artifacts/in19_g19_wave3_gate/raw/nodes_plan39.json` | `cc66a893f003af51…` | campaign_nodes⊗tensor_dev_missions dump, plan_id=39 |
| 2 | `artifacts/in19_g19_wave3_gate/raw/platform_probes.json` | `dafbf7f653f1b8dd…` | raw platform probe responses, base_url=http://127.0.0.1:8002 |
| 3 | `artifacts/in19_g19_wave3_gate/references.json` | `d3ece77add88fd96…` | reference values with provenance (in-repo citations) |
| 4 | `src/tensorphysics/campaign/wave3_gate.py` | `f5dd1e87a8c06ae8…` | gate engine that produced every number in this report |
| 5 | `git` | `376a679…` | worktree revision the gate ran on |

报告正文摘要 `5022e51af33a2b5bb137434a6ae4ca8d1f55d821e686e9156cedde11c2c616e5` 的定义：对 `gate_report.json` 中 `plan_id, plan_name, gate_ext_id, gate_mission_id, cutoff, nodes, probes, error_table, lineage, limitations, next_wave` 这组字段做 `json.dumps(..., sort_keys=True, ensure_ascii=False)` 后取 sha256。**它不是 markdown 文件的哈希**：排版/措辞调整不改变摘要，数值与判定变动才改变摘要——这正是可以安全签字的原因（复算脚本见 `scripts/run_in19_g19_wave3_gate.py` 与 `tests/test_wave3_gate.py::test_digest_recomputable_from_published_json`）。自检 C1–C5 与评审签名由正文派生，故不参与摘要（避免自指）。
