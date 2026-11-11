# 验收判据注册表与跨战役判据冲突裁决（p495w4-n02）

## 1 案例介绍

平台的历史验收判据散落在任务 `goal` 文本里（自然语言 + 符号阈值），同一物理量在不同战役中被写成
不同单位、不同措辞、不同阈值，无法机器比对。本工作把判据变成**一等对象**并在 TensorPhysics 中实现
四段流水线：抽取（带 provenance）→ 规范化（单位/统计口径）→ 跨战役冲突检测（容忍带）→ 仲裁（可重放裁决链）。

- 输入语料：平台库终态任务（status ∈ ('succeeded', 'failed', 'revoked', 'stopped')）中 **N = 600** 个带数值判据的任务目标文本，
  按 `md5(id)` 确定性抽样；SQL 与其 sha256 记录在 `corpus.json`（`0a9ead4793a2f262…`）。
- 注入配置：`artifacts/criterion_conflict/injected_metric_config.json`（sha256 `95760bc945d4f413…`），含 17 个物理量、单位换算表、
  统计口径别名与逐量容忍带；代码中**零域词**（扫描见 §5）。
- 语料/注册表/冲突/仲裁/图 全部由 `scripts/run_criterion_conflict_audit.py` 程序化生成，报告本文件亦由脚本写出（零手抄）。

## 2 怎么做的

1. **抽取器**（`extract.py`）：按行 + 子句切分（`,` `;` `。` `|` `且`，数字千分位保护），定位比较符
   （`≤ ≥ < > 不超过 至少 at most at least ±`），绑定比较符**邻近**数值与单位 token（单位 token 由配置注入），
   物理量=子句窗口内**就近**匹配（ASCII 别名带词边界，避免 `ratio` 命中 `iteration`），
   统计口径=`逐点/平均/最坏/p95/rms` 别名；括号单位 `[%]`/`（pct）` 也可识别，
   配置 `exclusion_markers` 注入的策略性文案（如 honest-FAIL 说明）在抽取前剔除。每条判据带
   `mission_id + field + line + char span + verbatim snippet`。**fail-closed**：物理量不在词表→`unmapped_quantity` 残渣；
   单位不在换算表→`unmapped_unit`，一律不进入注册表/冲突检测。
2. **规范化**（`normalize.py`）：`threshold_normalized = threshold × unit_factor`；缺失单位回落到配置
   `default_unit`；比较符映射为方向 `upper/lower/equal/range`；统计口径标签归一。
3. **冲突检测**（`conflict.py`）：键 = `(物理量, 统计口径, 方向)`；同键下按相对容忍带做**单链聚类**，
   跨战役（`campaign` 不同）且相对差 `|a-b|/max(|a|,|b|) > tolerance` 报 `CONFLICT`，
   `severity = delta/tolerance`；输出确定性排序与 `conflict_id`（哈希）。
4. **注册表**（`registry.py`）：全量判据 + `registry_hash`（sha256，无时间戳，幂等）。
5. **仲裁**（`arbitration.py`）：`escalate` 保留开放，`adopt_left/adopt_right/harmonize/dismiss` 关闭；
   JSONL 哈希链 + `replay_digest`，篡改/重复/缺理由全部 fail-closed。

## 3 计算结果（数字由脚本写入）

| 指标 | 数值 | 判据 |
|---|---|---|
| 抽取判据总数 / 已映射 | 561 / 448 | — |
| 残渣（未映射物理量） | 2499 | fail-closed 不进注册表 |
| 抽取覆盖（实例级） | **93.3%**（512/549） | ≥90% |
| 抽取覆盖（任务级） | 93.6%（190/203） | 参考 |
| 注入冲突召回 | **100%**（10/10） | ≥90% |
| 注入误报 | **0** / 8 控制例 | ≤1 |
| 跨战役冲突 | 86 | — |
| 注册表幂等 | run1=f32db4b23fc11a3c… run2=f32db4b23fc11a3c… → **True** | 同 hash |
| 仲裁回放 | digest=ca3af0de33b24f08… 复算一致=**True**；链完整=**True** | 可重放 |
| 域词渗漏 | **0** 命中（扫描 74 个特征词） | 0 |

### Top-5 真实跨战役冲突

| # | 物理量 | 口径 | 方向 | 阈值（左 vs 右） | 相对差 | severity | 来源任务 |
|---|---|---|---|---|---|---|---|
| 1 | `metric.relative_error` | unspecified | upper | 0.1 (%) vs 30 (%) | 99.7% | 6.64 | `5437` / `13373` |
| 2 | `metric.relative_error` | unspecified | upper | 0.5 (%) vs 30 (%) | 98.3% | 6.56 | `6651` / `13373` |
| 3 | `metric.relative_error` | unspecified | upper | 1 (%) vs 30 (%) | 96.7% | 6.44 | `7609` / `13373` |
| 4 | `metric.relative_error` | pointwise | upper | 1 (%) vs 15 (%) | 93.3% | 6.22 | `11923` / `11327` |
| 5 | `metric.relative_error` | unspecified | upper | 2 (%) vs 30 (%) | 93.3% | 6.22 | `10981` / `13373` |

原文出处（两段原文，逐条审计）：

- **[metric.relative_error]**
  - `5437` L1: (4)闸门GATE_PASS：nabla误差≤0.1%、离散vsCAD≤0.5%、S≤1%、
  - `13373` L1: (d) 线性叠加误差≤30%、一阶修正≤15%、全耦合CFD≤10%
- **[metric.relative_error]**
  - `6651` L1: 【机器可算验收判据】(a)表列插值连续性L=90/120/180/250相对误差<0.5%
  - `13373` L1: (d) 线性叠加误差≤30%、一阶修正≤15%、全耦合CFD≤10%
- **[metric.relative_error]**
  - `7609` L1: (b) JONSWAP 谱积分 ∫S(ω)dω = H_s²/16 误差 <1%
  - `13373` L1: (d) 线性叠加误差≤30%、一阶修正≤15%、全耦合CFD≤10%
- **[metric.relative_error]**
  - `11923` L1: 机器可算验收判据：(a) GZ 曲线逐点误差 <1%（vs 商用软件或手算）
  - `11327` L1: C3 冲蚀速率vs DNV RP O501设计值相对误差<15%（逐点位列出
- **[metric.relative_error]**
  - `10981` L1: 机器可算验收判据：(a) Sod 激波管（1D）：激波位置、接触间断、膨胀波尾部与精确解误差 <2%（P2 阶，50 个单元）
  - `13373` L1: (d) 线性叠加误差≤30%、一阶修正≤15%、全耦合CFD≤10%

## 4 结果图

- `fig1_extraction.png`：各仓抽取量 + 覆盖实例回收（红=未回收）。
- `fig2_conflict_heatmap.png`：冲突在「物理量 × 统计口径」上的分布。
- `fig3_threshold_spectrum.png`：Top-3 冲突物理量的归一化阈值谱，按战役分色，虚线为冲突两侧阈值。
- `fig4_arbitration.png`：Top-10 冲突 severity 与仲裁状态（红=已关闭、橙=已升级、灰=开放）。

## 5 注意点与局限

1. **参考标注器的循环性**：覆盖率的“基准实例”由独立正则（比较符+数值，且同一行出现配置别名）给出，
   与抽取器共享词表——它度量的是解析器而非词表召回；未映射物理量以残渣单列（2499 条），不粉饰。
2. **别名匹配精度**：ASCII 别名带词边界（`(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])`），并限定在同一比较符
   子句窗口内、最多 `12` 字符的**后方**回退；这消除了 `ratio`→`iteration`、`ka≥10 … 误差≤1dB` 一类误配，
   但相邻子句中括号内的样本量（如 `N≥800`）会进入残渣而非注册表。
3. **单位语义**：`%` 与裸小数按配置因子统一（`%`→0.01），`‰`→0.001；中文单位（秒/毫秒/个/节）由配置 token 表支持，
   未声明单位回落 `default_unit`，可能把 `3 分钟` 误读为 `3 s`——本语料未触发，但属于配置责任。
4. **单链聚类**：阈值聚类用相对容忍带做单链，簇内链式差值可能略超容忍带；因此每对簇只报“极值代表对”，
   不会穷举所有阈值对（避免 O(n²) 与噪声），审计以代表对为准。
5. **方向语义**：仅 `upper/lower` 方向参与冲突；`equal/range` 判据不表达单侧带，明确排除。
6. **跨战役定义**：冲突只在不同 `campaign` 间报告；同战役多个任务的历史漂移由既有 `criterion_registry.drift` 负责，二者互补。
7. **词表粒度**：`metric.relative_error` 汇聚了不同物理量的误差阈值（nabla/阻力/RAO/GZ…），冲突清单
   因此反映的是"平台跨战役验收口径不一致"，不是同一物理量的重复实验；更细粒度需要注入更细的物理量词表。
8. **策略性文案**：`exclusion_markers` 由配置注入（当前含 honest-FAIL / 不可接受 / 粉饰 等否定式表述），
   避免把"误差>15% 的项不许粉饰"这类治理语句误读为下界判据；词表属数据工件，代码零域词。

## 6 误差判定

| 关键量 | 基准 | 误差/偏差 | 是否 <3% |
|---|---|---|---|
| 单位换算（`%`/`‰`/`kB`/`ms`… → 规范单位） | 配置换算因子（精确十进制） | 0（浮点除法，机器精度） | **是** |
| 数值阈值抽取 | 注入语料真值（synthetic） | 0（逐例精确匹配） | **是** |
| 注入冲突召回/误报 | 10 正例 + 8 控制例真值 | recall 100%，FP 0 | **是** |
| 抽取覆盖率 | 参考标注器（非真值，见 §5.1） | 93.3%（缺失 6.7%） | 不适用：无物理真值基准；覆盖率作为召回代理，缺失项见 `coverage.misses` |
| 冲突相对差 `delta` | 解析定义 `|a-b|/max(|a|,|b|)` | 0（闭式计算） | **是** |

结论：本任务无实验/解析物理基准，所有“数值”均为确定性字符串/单位运算与合成真值对照；
可量化的换算与抽取误差为 0（<3%），覆盖率指标的误差需按“参考标注器”口径解释，已在 §5.1 声明。

## 7 可复现、评审状态与最后验证

- 复现命令：`PYTHONPATH=src python3 scripts/run_criterion_conflict_audit.py --limit 600`
  （生成 `corpus.json` / `registry.json` / `conflicts.json` / `arbitration.jsonl` / 4 张图 / 本报告 / `summary.json`）。
- 注册表 hash：`f32db4b23fc11a3cab0e0daf76b9627f9abd85fcb244ee3ac24501db1f07ffd7`；配置 hash：`95760bc945d4f413fa8a67636e82c8f3d6f2dc5982e804064adf0d80f7b0c436`；语料 SQL hash：`0a9ead4793a2f2625973248f5df216503be40e81ea8be8d723e7c7bf00a25999`。
- 测试：`PYTHONPATH=src python3 -m pytest tests/test_criterion_conflict.py -q`
  （43 passed；模块覆盖率 90.89%，≥90% 判据）。
- 评审状态：本报告由审计脚本自检生成，**待独立席位复核**；所有判据数字与路径均写入
  `artifacts/criterion_conflict/summary.json`，可机器复判。
- 最后验证日期：2026-11-11（本轮运行生成时间，见 summary.json）。
