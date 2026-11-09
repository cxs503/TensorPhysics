# P92W3-N09 验收判据预注册与阈值漂移审计（成熟化）

**节点**：P92W3-N09 · **仓**：TensorPhysics · **模块**：
`src/tensorphysics/campaign/criterion_registry/`
（`manifest.py` / `drift.py` / `ledger.py` / `kpi.py` / `registration.py` / `bench.py`）
**语料指纹 sha256**：`01d420d1ad918cc3cbc5cdccdc51a4aa9942639b73be680e48fd3486ea180d26`

## ① 案例介绍

战役 DAG 节点的验收判据（“门”的阈值与口径）在派单前被提出，在收单时被引用。
若判据只存在于自然语言里，收单方可以在事后**悄悄调门**：把阈值从 3 % 放到 8 %、
把分母从 reference 换成 baseline、把排除集换掉，而验收记录看上去仍然“通过”。
本节点把判据变成**事前冻结、事后可审计、漂移可判定**的结构化工件：

1. 判据被抽取为六个**结构性字段**（`threshold / unit / denominator / exclusion /
   reference_path / comparator`），不含任何领域语义；
2. 冻结时序列化为 canonical JSON 并计算 sha256，写入 metadata（预注册锚点）；
3. 收单时从交付报告重新抽取，与冻结版**逐字段 diff**，分类为
   `unchanged / tightened / loosened / scope_changed / undefined`；
4. 每条漂移写入事件流（append-only ledger），并由 KPI 视图汇总
   `registered_rate / loosened_rate / unregistered_rate`；
5. 失效方向 **fail-closed**：不可解析 → `unregistered`，绝不落成 `pass`。

本节点为**机制/后端**任务，无物理场量、无几何，故不做 Cb/L/B 几何校验；
结果图为度量图（字段 F1 与分类混淆矩阵），其“基准”是同语料下旧抽取器的基线。

## ② 怎么做的

**抽取契约（先定后测）**：阈值取**比较符锚定**的数字（`on 24 cases recall >= 0.92`
取 0.92 而非 24）；区间（`2-3 pct`、`between 10 and 20 ms`）取上界并置
`comparator=in_range`；单位取阈值数字后的 token 并经结构停用词过滤；比较符覆盖
符号（`<= ≥ ±`）与自然语言（`at most / no less than / not exceeding …`）。

**冻结**：`freeze_manifest()` 生成 canonical JSON（键排序、无空白）→ SHA-256 →
`status=registered`；任何字段事后改动都会使 `verify_frozen_hash()` 失败（`tampered`）。

**漂移判定优先级**（`DriftClassifier`）：`scope_changed > loosened > tightened >
undefined > unchanged`；阈值方向由 `comparator` 决定（`le/lt/eq` 越小越紧，
`ge/gt` 越大越紧）；缺失字段 → `undefined`；冻结版未注册 → `unregistered`。

**基准协议**：同一份标注语料（50 例抽取 + 24 例漂移检测，指纹
`01d420d1ad918cc3…`）分别跑 **旧抽取器（git HEAD，基线）** 与
**加固抽取器（本单）**，两次测量语料指纹相同；另设 20 例**留出集**（抽取器冻结后
写定，仅测一次、不回调），用于防止“自编语料自证”。

**复现**：
```bash
PYTHONPATH=src python -m tensorphysics.campaign.criterion_registry.bench --label hardened_v2 \
    --out artifacts/criterion_drift_audit/bench_v2_hardened.json
PYTHONPATH=src python scripts/m8455_criterion_drift_report.py
python -m pytest tests/test_criterion_drift_hardening.py -q
```

## ③ 计算结果

### 3.1 机器验收判据（基线 vs 加固，同语料）

| 判据 | 加固版 | 旧版基线 | 结论 |
|---|---|---|---|
| (a) 检测 recall ≥ 0.92 | 1.0000 | 0.9167 | PASS |
| (a) 检测 precision ≥ 0.95 | 1.0000 | 1.0000 | PASS |
| (a) 24 例精确分类准确率 | 1.0000 | 0.9167 | — |
| (b) 字段级 micro-F1 ≥ 0.95（50 例） | 1.0000 | 0.8049 | PASS |
| (b') 留出集 20 例 micro-F1 ≥ 0.90（冻结后单次测量） | 0.9735 | 0.7523 | PASS |
| (c) fail-closed 负控（6 探针） | ok | ok | PASS |
| (d) 零领域语义扫描命中数 | 0 | 0 | PASS |

### 3.2 字段级抽取（tp/fp/fn 与 precision/recall）

| 字段 | 加固 tp/fp/fn | 加固 P | 加固 R | 旧版 P | 旧版 R |
|---|---|---|---|---|---|
| threshold | 42/0/0 | 1.0000 | 1.0000 | 0.8095 | 0.8095 |
| unit | 36/0/0 | 1.0000 | 1.0000 | 0.7692 | 0.8333 |
| denominator | 9/0/0 | 1.0000 | 1.0000 | 0.8333 | 0.5556 |
| exclusion | 2/0/0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| reference_path | 2/0/0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| comparator | 38/0/0 | 1.0000 | 1.0000 | 1.0000 | 0.7368 |

### 3.3 漂移分类混淆矩阵（24 例：12 未变 + 12 漂移）

| 期望 / 预测 | unchanged | tightened | loosened | scope_changed | undefined | unregistered |
|---|---|---|---|---|---|---|
| unchanged | 12 | 0 | 0 | 0 | 0 | 0 |
| tightened | 0 | 3 | 0 | 0 | 0 | 0 |
| loosened | 0 | 0 | 3 | 0 | 0 | 0 |
| scope_changed | 0 | 0 | 0 | 3 | 0 | 0 |
| undefined | 0 | 0 | 0 | 0 | 2 | 0 |
| unregistered | 0 | 0 | 0 | 0 | 0 | 1 |

### 3.4 fail-closed 负控

| 探针 | 判定 | 抽出条目数 |
|---|---|---|
| empty | unregistered | 0 |
| none | unregistered | 0 |
| not_a_string | unregistered | 0 |
| placeholder | unregistered | 0 |
| prose_no_numbers | unregistered | 0 |
| whitespace | unregistered | 0 |

### 3.5 留出集残差（诚实 FAIL 面，未回调）

| 文本 | 期望 | 实际 |
|---|---|---|
| peak value not greater than 3.5e2 Pa | comparator=le, threshold=350.0, unit=Pa | threshold=350.0, unit=Pa |
| error <= 2.0 % w.r.t. baseline | comparator=le, denominator=baseline, threshold=2.0, unit=% | comparator=le, threshold=2.0, unit=% |
| correlation >= 0.995 across 500 samples | comparator=ge, threshold=0.995 | comparator=ge, threshold=0.995, unit=across |

## ④ 结果图

- `artifacts/criterion_drift_audit/figures/fig1_field_f1.png`
  ——六字段 F1 与 micro-F1，基线 vs 加固（虚线为 0.95 门）。
- `artifacts/criterion_drift_audit/figures/fig2_drift_confusion.png`
  ——24 例漂移分类混淆矩阵（对角线即正确分类）。

![fig1](figures/fig1_field_f1.png)

![fig2](figures/fig2_drift_confusion.png)

## ⑤ 注意点与局限

1. **语料自编性**：主语料由本单作者编写，存在“按契约写语料→契约完美”的过拟合
   风险；因此补了 20 例冻结后一次性测量的留出集（micro-F1 0.9735），
   残差 3 例见 3.5。
2. **已知盲区**：`not greater than`（否定式）未覆盖；`w.r.t.` 分母标记因词边界
   未命中；`>= 0.995 across 500 samples` 把 `across` 误判为单位。三者均为可枚举
   的规则缺口，但**未回调**以保留留出集效度，留待下一节点补测。
3. **单语句式**：语料为英文/符号混排；中文判据（如「不超过」）未覆盖。
4. **`unregistered` 与 `undefined` 的边界**：冻结侧可解析、报告侧完全不可解析时
   记 `unregistered`（fail-closed 状态），字段级缺失记 `undefined`；二者都不构成
   通过，但语义不同，已在 KPI 中分开计数。
5. **`audit_delivery` 的位置回退匹配**：仅在两侧条目数相等时按位对齐，否则缺失项
   记 `unregistered`，避免“错位对比”产生假 `unchanged`。
6. 本节点不计算物理量，故无解析解/文献基准误差；判据余量见 ⑥。

## ⑥ 误差判定（关键量 vs 门槛，逐项）

| 关键量 | 实测（加固） | 门槛 | 相对门槛余量 | 结论 |
|---|---|---|---|---|
| 抽取 micro-F1（50 例） | 1.0000 | 0.95 | 5.26 % | 余量 >3 %，非擦线通过 |
| 检测 recall（24 例） | 1.0000 | 0.92 | 8.70 % | 余量 >3 % |
| 检测 precision（24 例） | 1.0000 | 0.95 | 5.26 % | 余量 >3 % |
| 留出集 micro-F1（20 例） | 0.9735 | 0.90 | 8.17 % | 余量 >3 % |
| 旧版基线抽取 F1 | 0.8049 | 0.95 | 不达标 | 基线未过门（Δ = +0.1951 绝对 / +24.2 % 相对） |
| 旧版基线检测 recall | 0.9167 | 0.92 | 不达标 | 基线未过门（Δ = +0.0833） |

**显式回答「是否 <3%」**：本任务无物理基准值，误差以「相对门槛的余量」计。
四项关键量相对门槛的余量分别为
5.26 % / 8.70 % /
5.26 % / 8.17 %，
**均 >3 %**，即全部判据以明确余量达成；同时留出集仍有 3 处字段级残差（F1
0.9735），如实登记为已知盲区，不粉饰。

## 审计链

- 语料指纹：`01d420d1ad918cc3cbc5cdccdc51a4aa9942639b73be680e48fd3486ea180d26`
- 产物 sha256：
| 文件 | sha256 |
|---|---|
| bench_v1_legacy.json | 913781cc56db7a651942600a400bc0ff4e121bd46dea9b14c8b6733cf2ce6d3b |
| bench_v2_hardened.json | fe95bbf1248d45634cdf5a0734832e949beb78b955efc3c1435f51c822031377 |
| fig1_field_f1.png | b0e7ef18b366ffd468b42523f02953879c0d0e908417d7b800066ee2b764d6bf |
| fig2_drift_confusion.png | ca6081c822cd1a6cae7c2c0d0a43efbf20a300a020c424098b422980067c5f7c |
- 测试：`tests/test_criterion_drift_hardening.py`（25 项）+
  `tests/test_criterion_preregistration.py`（24 项，回归）全部通过。
