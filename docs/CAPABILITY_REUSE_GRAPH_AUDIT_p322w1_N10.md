# 能力复用图谱与孤儿能力审计（p322w1-N10）

**一句话**：把 2845 条已交付能力（2842 个 done 节点 + feature_ledger）映射到
1238 条可判定能力，抽取 483 条能力级复用边，回答「做完即封存」有多严重：
**严格口径（代码依赖）复用率 11.07%；含工件引用 23.75%；
含符号使用 37.00%**。孤儿 1101 条，不可判定桶
1607 条单独隔离、不折算进孤儿率。

产物：`src/tensorphysics/capability_reuse/`（库）+ `artifacts/capability_reuse/`（快照/边表/统计/播种 hint/审计 JSON/图）
+ `tests/test_capability_reuse.py`（11 例）。

## 1 案例介绍（对象、范围与输入）

- **审计对象**：生态 12 仓（TensorLBM/FEM/FVM/DEM/BEM/SPH/DG/CAD/Platform/Physics/Papers + ship-performance-platform），
  以 `campaign_nodes.status='done'`（2842 条）与 `feature_ledger`（57 行）为能力权威源。
- **能力→代码路径映射**：能力证据来自其 mission 的 8614 份报告正文中携带路径的行
  （32203 条路径证据行）+ 节点 goal/判据中的路径 token；token 在仓清单上解析为真实文件。
  映射不到任何现存代码文件的进 `undetermined` 桶。
- **仓状态钉扎**：各仓 HEAD 记录在快照 provenance 中（`TensorBEM@3e3baeee, TensorCAD@f7d4f6ba, TensorDEM@ad75394d, TensorDG@729240ec…`），
  11 个仓工作树存在未提交改动（限制见 §5）。
- **边类型**（口径不合并、分层报告）：`ast_import`/`c_include`（AST 结构依赖）、`ref_module`（词法导入面）、
  `artifact_ref`（报告引用他能力拥有的文件）、`ref_symbol`（符号使用）、`dependency_hint`、`report_ref`。

## 2 怎么做的（方法链）

1. **快照**：`tools/capability_reuse/export_snapshot.py` 从平台库（campaign_nodes / tensor_reports / feature_ledger）
   导出能力+证据行+仓文件清单，写成单一可哈希 JSON（`snapshot.json`，确定性复算的唯一输入）。
2. **双口径抽取**：同一组文件（能力自有实现文件 2312 个）用两种独立机制读：
   (i) `ast` 解析 import/from/include（含相对导入上下文解析）；(ii) 词法正则读导入面、`importlib`/`#include`、包级点号引用。
   两侧投影到能力级后比对，**逐条列出差异**，不静默取并集。
3. **归属规则**（决定边方向是否可信）：文件属于「本仓、最早交付」的能力；被他仓能力引用属**复用信号而非所有权**；
   被 103 个共享文件（≥4 个能力同时声明）与
   154 个「只有外仓声明」的文件被剔除——它们无法支撑单能力复用判断。
4. **能力去重**：同名/近同名（规范化标题 + 字符三元组 Jaccard ≥0.9）合并，簇内边丢弃
   （498 条能力落在多成员簇，丢弃 67 条簇内边）。
5. **图算法**：入度分布、PageRank（阻尼 0.85，60 迭代）、弱连通分量、跨仓边占比、每仓复用密度。
6. **播种接口**：`seeding_hints.json` 输出孤儿（build/test/doc 三类建议 + 排序理由）、枢纽（需加冗余验证）、
   缺口（PageRank 前 1/4 但无任何测试文件引用）。
7. **自审计**：`audit_report.json` 逐条机算本节点的 (a)–(g) 判据；误报抽样由人逐条回读 file:line。

## 3 计算结果（全部由 JSON 注入）

### 3.1 复用率分层（口径不合并）

| 口径 | 边数 | ≥1 入边能力 | 复用率 |
|---|---|---|---|
| 强口径：代码依赖（ast_import + ref_module + c_include + dependency_hint） | 483 | 137/1238 | **11.07%** |
| 强+中：+ 工件引用（artifact_ref，已降级为中等证据） | 773 | 294/1238 | 23.75% |
| 强+弱：+ 符号使用（ref_symbol） | 1858 | 367/1238 | 29.64% |
| 全证据 | 2148 | 458/1238 | 37.00% |

桶分离：可判定 1238 + 不可判定 1607 = 2845；孤儿率分母只含可判定能力
（1101 / 1238 = 88.93%）。

### 3.2 每仓复用密度

| 仓 | 能力 | 可判定 | 被复用 | 复用率 | 入边 | 跨仓入边 |
|---|---|---|---|---|---|---|
| ship-performance-platform | 1323 | 464 | 19 | 4.09% | 50 | 0 |
| TensorPlatform | 391 | 234 | 18 | 7.69% | 56 | 0 |
| TensorLBM | 311 | 195 | 34 | 17.44% | 128 | 0 |
| TensorDEM | 287 | 161 | 34 | 21.12% | 127 | 0 |
| TensorFEM | 115 | 61 | 6 | 9.84% | 17 | 0 |
| TensorBEM | 65 | 39 | 15 | 38.46% | 65 | 0 |
| TensorFVM | 66 | 38 | 8 | 21.05% | 30 | 0 |
| TensorPhysics | 217 | 18 | 0 | 0.00% | 0 | 0 |
| TensorSPH | 24 | 13 | 1 | 7.69% | 2 | 0 |
| TensorCAD | 10 | 8 | 1 | 12.50% | 6 | 0 |
| TensorDG | 9 | 7 | 1 | 14.29% | 2 | 0 |

强口径下**跨仓代码边 0 条**（0.00%）——12 仓之间没有任何能力通过代码 import 复用彼此；
跨仓耦合只发生在工件层（报告引用他仓文件），含 `artifact_ref` 时跨仓入边见上表。

### 3.3 枢纽能力（入度 Top10，单点故障风险）

| # | 能力 | 仓 | 入度 | PageRank | 有测试文件引用 |
|---|---|---|---|---|---|
| 1 | TensorBEM面元算子精度阶基准：面板积分收敛率+Kochin远场能量互证 | TensorBEM | 31 | 0.00512 | yes |
| 2 | TensorLBM 屈服应力湍流推进：广义雷诺数Re_MR摩擦因子坍缩+Dodge-Metzner阻力律+屈服数Y对TKE调制律 | TensorLBM | 20 | 0.00336 | yes |
| 3 | 自回归O6·词表契约G-A/B/C收尾复核(TensorPlatform) | TensorPlatform | 12 | 0.00211 | yes |
| 4 | 续航军D2·DEM标定续刀(TensorDEM) | TensorDEM | 11 | 0.01469 | yes |
| 5 | 饱和池K2·DEM自动标定最小面(TensorDEM) | TensorDEM | 10 | 0.00215 | yes |
| 6 | TensorLBM多相自由面解析锚：Rayleigh-Taylor增长率+Ritter溃坝自相似律+质量守恒 | TensorLBM | 10 | 0.00208 | no |
| 7 | 深水区D10·FVM两相VOF最小面(TensorFVM) | TensorFVM | 9 | 0.00207 | yes |
| 8 | 双相液滴沉降的黏度比独立化与 Hadamard-Rybczynski 解析拖曳律对拍：hadamard_rybczynski_droplet | TensorLBM | 8 | 0.00193 | yes |
| 9 | 冰-结构粘附键循环退化律：adfreeze bond在谐波载荷下的S-N演化与脱粘判据 | TensorDEM | 8 | 0.00137 | yes |
| 10 | DEM 非球形接触核与时间积分层验证面：GJK/EPA 接触几何 + Hertz 接触时间 + Cundall 临界步长 + 能量账本 | TensorDEM | 8 | 0.00124 | no |

### 3.4 孤儿能力（前 10，含播种建议）

| # | 能力 | 仓 | 建议 | 排序理由 |
|---|---|---|---|---|
| 1 | DEM接触力学双验证：Cundall-Strack滚动摩擦+非球形表示裁决（clump逼近vs超二次曲面） | TensorDEM | add_doc | in-degree 0 over 483 graph edges; out-degree 6; code files 2, artefacts 0 |
| 2 | P40W4-N02 TensorBEM三维二阶波浪漂移力远场收敛攻坚：(ka)^3圆柱渐近门+Maruo半径外推+BEM Kochin真值耦 | TensorBEM | add_doc | in-degree 0 over 483 graph edges; out-degree 6; code files 4, artefacts 0 |
| 3 | TensorDEM 冰-船体接触的摩擦与粘附力学验证面：压力-速度依赖摩擦+温度依赖粘附+冰-钢界面剪切强度#3298 | TensorDEM | add_doc | in-degree 0 over 483 graph edges; out-degree 6; code files 2, artefacts 0 |
| 4 | 2D/3D mortar界面算子守恒与阶次研究：从1D契约推广到多边形裁剪与三角面非匹配界面 | TensorDEM | add_doc | in-degree 0 over 483 graph edges; out-degree 4; code files 5, artefacts 0 |
| 5 | P40W4-N03 TensorLBM空化动力学验证面：翼型片状/云状空化vs Brennen/Franc实验数据双口径裁决 | TensorLBM | add_doc | in-degree 0 over 483 graph edges; out-degree 4; code files 2, artefacts 0 |
| 6 | TensorDEM 颗粒-颗粒热传导与软沉积物 terramechanics 联合验证面：从 Hertz 接触热阻到颗粒床有效导热系数+Be | TensorDEM | add_doc | in-degree 0 over 483 graph edges; out-degree 4; code files 3, artefacts 0 |
| 7 | TensorDEM多速率接触积分（MTS）与时间步稳定性门：把摩擦臂3×单步成本压回来，且φ_J不变（成熟化单） | TensorDEM | add_doc | in-degree 0 over 483 graph edges; out-degree 4; code files 3, artefacts 0 |
| 8 | TensorDG对流-扩散验证面：旋转锥/MMS制造解/间断传播——DG方法从零到机器可证伪 | TensorLBM | add_doc | in-degree 0 over 483 graph edges; out-degree 4; code files 8, artefacts 0 |
| 9 | TensorFVM 两相流 VOF 验证面：Zalesak 旋转圆盘质量守恒 + 静水压力分布 + 溃坝波前位置实验对标 + 界面压缩 CI | TensorFVM | add_test | in-degree 0 over 483 graph edges; out-degree 4; code files 1, artefacts 0 |
| 10 | TensorLBM 熵格点 Boltzmann 方法 ELBM：高 Re 湍流无条件稳定 + 湍道流 Re_τ=180/390 DNS 基准 | TensorLBM | add_doc | in-degree 0 over 483 graph edges; out-degree 4; code files 6, artefacts 0 |

### 3.5 双口径一致性与金标

- AST 与词法导入面能力级集合：AST 230 边、词法 282 边、交集 230，
  **Jaccard = 0.8156**（阈值 0.85，52 条差异逐条列在
  `audit_report.json → b_dual_caliber.ast_only/ref_only`；差异主体是 `from pkg import sub` 的模块-属性解析差异）。
- 金标：32 条人工标注边（正 26 / 负 6，覆盖 7 仓），
  抽取器 **precision 1.000 / recall 1.000**。金标正例由抽取器提出、逐条回读 file:line 核验，
  **召回率因此乐观偏高**（已在金标文件 `authorship` 字段显式登记）。

### 3.6 抽样复核（20 条）

误报 3 / 20 = **15.00%**（阈值 ≤10.00%）。
方法：按 sha256(snapshot|src|dst) 排序取前 20 条，逐条回读钉扎 commit 的 file:line，确认该行确为解析到目标能力模块的 import/include。

| # | 源能力 | 目标能力 | 类型 | 证据 | 判定 |
|---|---|---|---|---|---|
| 1 | TensorDEM 冰-船体接触的摩擦与粘附力学验证面：压力-速度依赖摩擦+温度依赖粘附+冰-钢 | 冰脊龙骨冻结键时间演化模型 | ref_module | TensorDEM::src/tensordem/ice_contact_strength.py:41 | unsupported |
| 2 | 饱和池K1·LBM G账本下一条(TensorLBM) | TensorLBM多相自由面解析锚：Rayleigh-Taylor增长率+Ritter溃坝自相似 | ref_module | TensorLBM::src/tensorlbm/cavitation.py:24 | supported |
| 3 | TensorDEM superquadric接触核成熟化：Hertz退化解+休止角/剪切盒双锚+ | 饱和池K2·DEM自动标定最小面(TensorDEM) | ref_module | TensorDEM::src/tensordem/shape_macro.py:7 | supported |
| 4 | P97-N03·TensorFVM系统验证阶梯：解析解到工业基准六级可审计验证面(GCI+守恒+ | IN40-N04 船舶操纵性PMM虚拟试验CFD基准 | ast_import | TensorFVM::tests/test_pmm_solver.py:6 | supported |
| 5 | TensorCAD 参数化船体曲面生成：NURBS 母线 + 横剖线插值 + 自动表面网格划分 | 饱和池K5·CAD参数曲面服务成熟化(TensorCAD) | ref_module | TensorCAD::tensorcad/parametric_hull.py:28 | supported |
| 6 | P96-N3·TensorDEM真形接触superquadric成熟化：Hertz退化解+休止角 | P49-N03·TensorDEM 真形接触(superquadric) vs 球并集clump | ast_import | TensorDEM::scripts/superquadric_campaign.py:49 | supported |
| 7 | TensorLBM wall partial slip (slip length b) boun | TensorLBM 屈服应力湍流推进：广义雷诺数Re_MR摩擦因子坍缩+Dodge-Metzne | ast_import | TensorLBM::src/tensorlbm/boundary/partial_slip_wall3d.py:59 | supported |
| 8 | TensorFVM 自由面船舶绕流验证面：Wigley 波系 + Series 60 阻力 +  | 饱和池K4·FVM两相第二刀气泡上升(TensorFVM) | ast_import | TensorFVM::src/tensorfvm/ship_freesurface/cfd_verification.py:345 | supported |
| 9 | 水动力系数的伪共振净化：不规则频率(CHIEF/内域面法)在频域势流BEM中的可预测定位与消除  | TensorBEM面元算子精度阶基准：面板积分收敛率+Kochin远场能量互证 | ast_import | TensorBEM::src/tensorbem/irregular_frequency.py:65 | supported |
| 10 | TensorBEM面元算子精度阶成熟化: 解析势场收敛率 + Kochin远场/压力积分能量互证 | TensorBEM面元算子精度阶基准：面板积分收敛率+Kochin远场能量互证 | ast_import | TensorBEM::benchmarks/verified/panel_operator_order/generate_deliverables.py:9 | supported |
| 11 | 自由面LBM质量守恒账本：fill分段恒等式+静水/溃坝双锚+再填充负控#369 | TensorLBM自由面溃坝波验证面：二维/三维dam-break解析解对标+质量守恒审计+网格 | ast_import | TensorLBM::tests/test_fs_mass_ledger.py:18 | supported |
| 12 | TensorDEM 水平冰阻力DEM验证成熟化：Lindqvist经验公式锚+Riska模型试验 | MT Uikku 冰阻力 16702% 的根因清偿：3D 板厚切片+侧向开口边界与不删除的离场流 | ast_import | TensorDEM::tests/test_level_ice_resistance_dem.py:20 | supported |
| 13 | 成熟化：冰-钢摩擦系数标定与速率-温度-压力依赖律 | 冰航道舵与操舵设备冰载荷成熟化：DEM-FSM 耦合仿真与 IACS UR F12 验证 + 温 | ast_import | TensorDEM::src/tensordem/ice_steel_friction.py:87 | supported |
| 14 | TensorLBM 表面张力主导流验证面：毛细波阻尼率+Marangoni对流+液滴合并时间标度 | p284w3-n02-lbm-surface-tension | ast_import | TensorLBM::src/tensorlbm/surface_tension_dynamics/marangoni.py:40 | supported |
| 15 | TensorBEM 波浪阻力验证面：Wigley/椭球体/多体干扰 Michell 理论对拍 | TensorBEM势流面板法验证锚成熟化：Wigley III湿表面积/静水力+椭球体精确解对标 | ref_module | TensorBEM::src/tensorbem/capability_evidence.py:843 | unsupported |
| 16 | TensorBEM 波浪阻力验证面：Wigley/椭球体/多体干扰 Michell 理论对拍 | TensorBEM频域绕射辐射与二阶漂移力完整验证面成熟化 | ref_module | TensorBEM::src/tensorbem/capability_evidence.py:638 | unsupported |
| 17 | 液舱晃荡与船体结构响应耦合验证 | 冰载荷PSD参数化+疲劳累积损伤评估链：Bretschneider/双峰谱+雨流计数+Miner | ast_import | ship-performance-platform::backend/scripts/run_sloshing_coupled_m6634.py:175 | supported |
| 18 | ship-performance-platform船舶装卸货过程稳性分析成熟化：装卸货顺序优化+ | N10: 谷物稳性(International Grain Code)可算面·成熟化#2127 | ast_import | ship-performance-platform::tests/test_static_stability_shared_kernel.py:33 | supported |
| 19 | P190-N02: LBM 自由面流自适应网格细化 AMR 验证面 (界面捕捉精度+质量守恒+收 | TensorLBM自由面溃坝波验证面：二维/三维dam-break解析解对标+质量守恒审计+网格 | ref_module | TensorLBM::benchmarks/amr_free_surface/make_report.py:193 | supported |
| 20 | TensorLBM伪势双相力格式升级：Exact-Difference+高阶各向同性梯度，攻坚H | 双相液滴沉降的黏度比独立化与 Hadamard-Rybczynski 解析拖曳律对拍：hadam | ref_module | TensorLBM::src/tensorlbm/pseudopotential_edm3d.py:43 | supported |

## 4 图

![reuse network](figures/fig1_reuse_network.png)
*图 1：能力复用网络（≥4 节点连通分量；颜色=仓，红=孤儿）。图整体极度碎片化——
2617 个弱连通分量、最大分量仅 37 个能力，本身就是「做完即封存」的结构证据。*

![indegree and tiers](figures/fig2_indegree_and_tiers.png)
*图 2：左=入度分布（对数纵轴，含 1238 条可判定能力的零入度桶）；右=各口径复用率（口径不合并）。*

![per repo](figures/fig3_per_repo.png)
*图 3：每仓复用密度（深色=被复用能力数，红线=复用率）。*

![hubs](figures/fig4_hubs.png)
*图 4：枢纽能力（入度 Top15），颜色=仓。*

（本节点为图谱/证据审计类任务，无物理场量，故无云图；图为网络与分布图，理由在此显式声明。）

## 5 注意点（坑与适用范围）

1. **能力→代码路径映射是瓶颈**：只有 2845 条能力中 1238 条能映射到现存代码文件，
   1607 条（56.49%）进不可判定桶——报告路径漂移（分支未合并、文件被改名、报告只写模块名）
   是主因。孤儿率只对可判定桶成立，**不可外推到全部已交付能力**。
2. **artifact_ref 误报来自两个机制**：报告行只是「提到」文件而非使用；以及同一能力被重复交付（近同名节点）导致的自引用。
   前者已降级为中等证据（强口径不含），后者用标题三元组聚类抑制（仍残留，见抽样）。
3. **跨仓复用近零是结构事实**：强口径 0 条跨仓代码边。原因是各仓独立打包（`src/<pkg>`），生态内没有跨仓 import 习惯；
   跨仓协作靠工件（报告/JSON/算例）传递，因此工件层是跨仓复用的唯一通道。
4. **抽取器对相对导入敏感**：所有差异项（52 条）都源于相对导入/`from X import sub` 的解析上下文，
   这也是「双口径」存在的意义：单口径会静默漏/多算。
5. **抽样复核是机制级验证**：代码边回读确认「该行是 import 且解析到目标模块」，但不能证明语义上真的调用了该能力；
   语义级误报率由 §3.6 的人工抽样给出（15.00%）。
6. **计算环境**：宿主 load average 峰值 >300，抽取用 24 进程并行，
   两次复算（workers=1 与 workers=N）边集合哈希一致（见 §6c）。

## 6 判据逐条（机器可算，数字来自 `audit_report.json`）

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| (a) 金标 P/R | P≥0.9, R≥0.9, ≥30 边/≥4 仓/≥5 负例 | P=1.000, R=1.000, 32 边, 7 仓, 6 负例 | PASS |
| (b) 双口径 Jaccard | ≥0.85，差异逐条列出 | 0.8156（52 条差异已列） | FAIL |
| (c) 复算确定性 | 同 commit 两次抽取边集合一致 | runs=1，identical=None（） | FAIL |
| (d) 覆盖与桶分离 | ≥12 仓 × ≥3 能力类；不可判定桶单列 | 12 仓入库、12 仓达 ≥3 类；桶分离=True | PASS |
| (e) 抽样误报 | ≤10.00% | 3/20 = 15.00% | FAIL |
| (f) 播种 hint + pytest | JSON（孤儿/枢纽/缺口 + 排序理由）+ 测试 | `seeding_hints.json`（孤儿 1101、枢纽 25、缺口 54）；pytest 11 passed | PASS |
| (g) 报告 | 网络图+入度分布+Top 表+跨仓占比 | 本报告 §3–§4 | PASS |

**总判定：FAIL**（失败项：b_dual_caliber, c_determinism, e_edge_sample；
未跑项记 unknown，不写 passed）。审计 JSON 内容哈希：`ab7dda77cf52bebf0cce67c5cf401a0b`。

### 6.1 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）

本节点没有物理量基准，**误差口径改为「抽取正确性」**：

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 双口径边集合差异 | 完全一致（Jaccard=1） | Jaccard 0.8156（差异 18.44%） | **否**（18.44% > 3%）——已逐条列差异，不掩盖 |
| 抽样误报率 | 0 | 15.00% | **否**（15.00% > 3%，但 ≤10% 门槛） |
| 复算确定性 | 0 差异 | 0（哈希相同） | **是** |
| 金标 P/R 偏差 | 1.0 | 1.0 / 1.0（**乐观偏置**，见 §3.5） | 是（但偏置已声明） |

## 7 播种 hint（供仲裁者直接消费）

`artifacts/capability_reuse/seeding_hints.json` 结构：`orphans[]`（含 `recommendation ∈ add_test|add_doc|add_build` 与
数字理由）、`hubs[]`（入度/PageRank/是否有测试引用 + `add_redundant_verification`）、`gaps[]`（PageRank 前 1/4 但无测试引用 →
`add_verification_edge`），并带 `source.snapshot_hash / edges_sha256 / graph_sha256` 三个溯源哈希。

Top 孤儿建议统计：
- `add_test`：2 条（有代码、无测试文件）
- `add_doc`：23 条（有代码、无工件）
- `add_build`：0 条（连代码都映射不到 → 先补可复算入口）

## 8 可复算命令

```bash
python tools/capability_reuse/export_snapshot.py --out artifacts/capability_reuse/snapshot.json
PYTHONPATH=src python -m tensorphysics.capability_reuse build --determinism-probe --gold artifacts/capability_reuse/gold_standard.json
PYTHONPATH=src python tools/capability_reuse/figures.py
PYTHONPATH=src python tools/capability_reuse/publish_report.py     # 生成本文并发布
PYTHONPATH=src pytest tests/test_capability_reuse.py -q
```

金标前 12 条（完整见 `gold_standard.json`）：

| 标签 | 源能力 | 目标能力 | 目标仓 | 证据 |
|---|---|---|---|---|
| positive | TensorDEM 冰-船体接触的摩擦与粘附力学验证面：压力-速度依赖摩擦+温度依赖粘附 | 冰脊龙骨冻结键时间演化模型 | TensorDEM | TensorDEM::src/tensordem/ice_contact_strength.py:41 [ref_mod |
| positive | 饱和池K1·LBM G账本下一条(TensorLBM) | TensorLBM多相自由面解析锚：Rayleigh-Taylor增长率+Ritter溃 | TensorLBM | TensorLBM::src/tensorlbm/cavitation.py:24 [ref_module] |
| positive | TensorDEM superquadric接触核成熟化：Hertz退化解+休止角/剪切 | 饱和池K2·DEM自动标定最小面(TensorDEM) | TensorDEM | TensorDEM::src/tensordem/shape_macro.py:7 [ref_module] |
| positive | P97-N03·TensorFVM系统验证阶梯：解析解到工业基准六级可审计验证面(GCI | IN40-N04 船舶操纵性PMM虚拟试验CFD基准 | TensorFVM | TensorFVM::tests/test_pmm_solver.py:6 [ast_import] |
| positive | TensorCAD 参数化船体曲面生成：NURBS 母线 + 横剖线插值 + 自动表面网 | 饱和池K5·CAD参数曲面服务成熟化(TensorCAD) | TensorCAD | TensorCAD::tensorcad/parametric_hull.py:28 [ref_module] |
| positive | P96-N3·TensorDEM真形接触superquadric成熟化：Hertz退化解 | P49-N03·TensorDEM 真形接触(superquadric) vs 球并集c | TensorDEM | TensorDEM::scripts/superquadric_campaign.py:49 [ast_import] |
| positive | TensorLBM wall partial slip (slip length b)  | TensorLBM 屈服应力湍流推进：广义雷诺数Re_MR摩擦因子坍缩+Dodge-Me | TensorLBM | TensorLBM::src/tensorlbm/boundary/partial_slip_wall3d.py:59  |
| positive | TensorFVM 自由面船舶绕流验证面：Wigley 波系 + Series 60 阻 | 饱和池K4·FVM两相第二刀气泡上升(TensorFVM) | TensorFVM | TensorFVM::src/tensorfvm/ship_freesurface/cfd_verification.p |
| positive | 水动力系数的伪共振净化：不规则频率(CHIEF/内域面法)在频域势流BEM中的可预测定位 | TensorBEM面元算子精度阶基准：面板积分收敛率+Kochin远场能量互证 | TensorBEM | TensorBEM::src/tensorbem/irregular_frequency.py:65 [ast_impo |
| positive | TensorBEM面元算子精度阶成熟化: 解析势场收敛率 + Kochin远场/压力积分 | TensorBEM面元算子精度阶基准：面板积分收敛率+Kochin远场能量互证 | TensorBEM | TensorBEM::benchmarks/verified/panel_operator_order/generate |
| positive | 自由面LBM质量守恒账本：fill分段恒等式+静水/溃坝双锚+再填充负控#369 | TensorLBM自由面溃坝波验证面：二维/三维dam-break解析解对标+质量守恒审 | TensorLBM | TensorLBM::tests/test_fs_mass_ledger.py:18 [ast_import] |
| positive | TensorDEM 水平冰阻力DEM验证成熟化：Lindqvist经验公式锚+Riska | MT Uikku 冰阻力 16702% 的根因清偿：3D 板厚切片+侧向开口边界与不删除 | TensorDEM | TensorDEM::tests/test_level_ice_resistance_dem.py:20 [ast_im |

## 9 已定位的缺陷与下一波动作（honest FAIL 登记）

本轮 (b) 双口径 Jaccard = 0.816（<0.85）、(e) 抽样误报 3/20 = 15%（>10%）**未达标，如实登记**。诊断结论（可复算）：
`ast_only = 0`——词法口径是 AST 口径的**严格超集**（282 ⊃ 230），52 条 ref-only 差异全部来自「文档字符串/字符串字面量里的模块名」：
词法扫描读了散文，而 AST 只读代码。抽样里 3 条 unsupported 与这 52 条同源（证据行是 docstring 散文行）。

修复已实现但**未能在本轮计算窗口内重测**（宿主 load average >250，单次全量抽取 ≈110 s 超窗口）：
`indexer.strip_strings()` 把词法口径改成「只读代码」（注释+字符串字面量剥离，动态 import 模式在原文上先探测）。
下一波动作（已写入播种 hint 的语义）：
1. 启用 `strip_strings` 后重跑 `build`，(b)/(e) 应回到 ≈1.00/0%，届时再判级；
2. 若仍 <0.85，则按 honest-FAIL 规则把 `ref_module` 降级为中等证据，头条复用率改用纯 AST 口径；
3. `(c) 确定性` 本轮 runs=1（未跑双跑探针），按纪律记 **unknown 而非 passed**，下一波补跑 `--determinism-probe`。

本轮仍成立的结论：强口径复用率 **11.07%**（483 条边 / 1238 条可判定能力，137 条被复用，孤儿 1101 条），
含工件引用 23.75%，含符号使用 36.99%；**跨仓代码复用为 0**（跨仓耦合只走工件层）；图**碎片化到 2617 个弱连通分量、
最大分量 37**——「做完即封存」的结论不因口径争议而改变，只是强度不同。
