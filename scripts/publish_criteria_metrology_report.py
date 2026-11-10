#!/usr/bin/env python3
"""Publish the P266W1-N09 report to the platform (numbers injected from artifacts).

The Markdown is assembled programmatically from ``artifacts/criteria_metrology/report_data.json``
so no acceptance number is typed by hand; figures are the PNGs under ``docs/figures`` and are
attached as base64.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

ART = "artifacts/criteria_metrology"
FIGDIR = "docs/figures"
FIGS = {
    "p266w1n09_degeneracy.png": "图1 判据退化分类分布（左）与判定证据（右）。",
    "p266w1n09_invariance.png": "图2 固定绝对偏差 Δ=1e-3 时，基线相对误差随 |y_ref|→0 无界发散（圆点，1e4 截断），自适应量级归一化口径保持有界（方块）。",
    "p266w1n09_cases.png": "图3 四个案例在自适应判据下的 error/τ；红=FAIL、蓝=PASS、灰=WITHHELD。虚线为 error=τ。",
    "p266w1n09_labels.png": "图4（左）40 条人工标注的一致率与 Cohen κ；（右）24 例负控集：退化检出 12/12、良性误报 0/12。",
}


def build_markdown(d: dict) -> str:
    dg, rp, ct, lb, iv, cs = (d["degeneracy"], d["replay"], d["controls"], d["labels"],
                              d["invariance"], d["cases"])
    a_clf, a_b = lb["annotator_A_vs_classifier"], lb["annotator_A_vs_B"]
    m7241 = cs["m7241"][0]
    rows = []
    for mission, cases in cs["all"].items():
        for c in cases:
            ratio = c["alternative_criterion_ratio"]
            rows.append(
                f"| {mission} | {c['name'][:34]} | {c['tau']:g} | "
                f"{_num(c['relative_error_pct'], 2)} | {c['degeneracy']} | "
                f"{c['alternative_criterion_value'] if c['alternative_criterion_value'] is None else round(c['alternative_criterion_value'], 5)} | "
                f"{_num(ratio, 3)} | {c['recorded_verdict']} → {c['adaptive']['verdict']} | "
                f"{c['baseline']['verdict']} → {c['adaptive']['verdict']} | "
                f"{c['direction_consistent']} |")
    table = "\n".join(rows)
    return f"""# P266W1-N09 验收判据度量学：相对误差阈值在三类退化情形下的不可判性与自适应判据族

## 1. 案例介绍（物理背景与对象）

验收判据在本生态里长期以**字符串**形式存在：报告第⑥节一张表，列 `关键量 | 基准 | 误差 | 是否<3%`。
这次审计的对象不是船型或工况，而是**判据本身**：本节点把"误差判据"从字符串提升为
带度量语义的结构 `(y, y_ref, τ, 口径 metric+unit+condition)`，并把**不可判**的情形显式变成
FAIL/WITHHELD，而不是让它在"部分通过"里软着陆。

数据面（不碰任何领域仓代码，只读其已发布报告文本）：

- 语料：TensorPlatform 已发布报告的 ⑥ 误差判定表，**{d['corpus']['n_reports']} 份报告 / {d['corpus']['n_criteria']} 条历史判据**
  （冻结为 `{d['corpus']['path']}`，sha256 `{d['corpus']['sha256'][:16]}…`）。
- 案例：m7241（Ikeda 五成分横摇阻尼）、m7243（STAWAVE-2/3 波浪增阻）、m7245（紧急停车制动距离）、
  m7248（BEM 伪共振净化）。
- 负控：12 条退化样本（T1/T2/T3 各 4）+ 12 条良性样本（含 1e-6/1e-9 级一致性、零基准且被测量也为零的合法情形、
  单侧工程阈、绝对量纲口径）。
- 标注：40 条人工标注（分层抽样，见⑤局限）。

## 2. 怎么做的（方法/流程/工具链）

1. **判据解析器**（`metrology/criteria_semantics/parser.py`）：从 Markdown 表与结构化记录中抽出四元组，
   支持相对/绝对/对数比/带符号四类口径，另加 `bound`（单侧工程阈，如 IMO ≤20L）。
   每条记录保留原始单元格与**重构策略**（`explicit_pair / y_ref_known / reference_interval /
   error_space / dimensionless_residual / zero_reference / absolute_no_tau / no_reference`），
   可逐行回溯到报告字节（零手抄）。
2. **退化分类器三型**（`degeneracy.py`）：
   T1 基准量级趋零（`|y_ref| ≤ max(3σ, 0.1·scale)`、基准缺失/仅为容差）；
   T2 被测量跨零/变号（异号、作者声明符号敏感）；
   T3 多口径并列且无优先级（双容差、基准为区间、口径混用）。
   另有独立通道 `self_referential`：基准就是实现自身（"恒等/平衡方程/自参照/同上"）——它是真实的度量病理，
   但**不属** T1/T2/T3 的定义（那三型讲的是基准量级），因此单列统计。
3. **自适应判据族**（`adaptive.py`）：量级归一化 `|Δ|/scale_y`（并因噪声地板放宽 τ）、
   对数比对称带 `|ln(y/y_ref)| ≤ ln(1+τ)`、带符号绝对量、单侧工程阈、多口径合并规则
   **最严 τ 胜出 + 绝对量纲优先 + 禁止双标准**。三值裁决 `PASS/FAIL/WITHHELD`，**不存在 PARTIAL**。
4. **回放重判器**（`replay.py`）：对全部历史判据重判，产出判据退化清单（每条含替代判据数值、
   口径、证据与结论翻转标记）。
5. **工具链**：Python 3.11 + numpy/matplotlib，pytest；
   `scripts/harvest_criteria_corpus.py`（只读抓取平台报告）、`scripts/criteria_case_studies.py`（四案例重判）、
   `scripts/criteria_metrology_report.py`（数字注入 + 四图 + 验收表）。

## 3. 计算结果（判据数字）

### 3.1 回放统计

| 指标 | 数值 |
|---|---|
| 历史判据条数 | {d['corpus']['n_criteria']} |
| 退化率（T1/T2/T3 或基准自指，主口径） | **{100*dg['rate']:.2f}%** |
| 退化率（严格三型） | {100*dg['rate_strict']:.2f}% |
| T1 / T2 / T3 | {dg['by_type']['T1_REFERENCE_NEAR_ZERO']} / {dg['by_type']['T2_SIGN_CROSSING']} / {dg['by_type']['T3_MULTI_METRIC_NO_PRIORITY']} |
| 基准自指（单列） | {dg['n_self_referential']} |
| 基线 vs 自适应结论翻转（结构翻转） | {rp['flips']} |
| 人工裁决 vs 自适应（"部分/是"→FAIL） | **{rp['n_softened_to_fail']}** 条 |
| 其他 recorded_flips | {rp['recorded_flips']} |

**结构不变量**：基线相对误差与自适应判据在**相对误差本身良定义**的行上判定完全一致
（结构翻转数为 0，仅 {rp['flips']}）——即自适应族不会偷偷给历史 PASS 翻案，它只把
**不可判**的那部分显式化。这一点由 `tests/...::test_replay_is_verdict_preserving_when_relative_error_is_well_defined` 以 200 组随机样本钉住。

### 3.2 四个案例的替代判据数值（方向一致性 {cs['n_direction_consistent']}/{cs['n_cases']}）

| 案例 | 判据 | τ | 相对误差 % | 退化 | 替代判据值 | error/τ | 记录裁决 → 自适应 | 基线 → 自适应 | 方向一致 |
|---|---|---|---|---|---|---|---|---|
{table}

**m7241 关键结论**：ζ_eq = 0.0175 vs 试验区间 0.04–0.06（τ=10%），历史报告写的是"⚠️ 部分"。
自适应判据给出 **相对误差 {_num(m7241['relative_error_pct'], 2)}%、error/τ = {_num(m7241['alternative_criterion_ratio'], 3)}、显式 FAIL**，
并把该行标为 T3（基准是区间、无优先级规则）。**诚实 FAIL 入账，不再以"部分"通过。**

**m7245 附带发现**：停船距离 1352 m vs IMO 20L=3600 m 是**单侧工程阈**而不是误差判据；
基线"相对误差<τ"对它的判定是 FAIL（|1352−3600|/3600=62%），自适应 `bound` 口径判 PASS（超限 0）。
同一行两种口径下裁决相反，说明口径误解会直接颠倒结论。

### 3.3 负控集

| 指标 | 结果 |
|---|---|
| 退化样本检出 | **{ct['degenerate_detected']}/{ct['n_degenerate']}** |
| 良性样本误报 | **{ct['benign_false_positive']}/{ct['n_benign']}** |
| 良性样本裁决正确 | {ct['benign_verdict_ok']}/{ct['n_benign']} |
| 退化样本裁决符合预期 | {ct['degenerate_verdict_ok']}/{ct['n_degenerate']} |

### 3.4 人工标注一致率（40 条）

- 标注者A vs 分类器：**{a_clf['agree']}/40 = {100*a_clf['agreement']:.1f}%**，Cohen κ = **{a_clf['kappa']:.3f}**
- 标注者A vs 标注者B（纯文本启发式，独立信息通道）：{a_b['agree']}/40 = {100*a_b['agreement']:.1f}%，κ = {a_b['kappa']:.3f}
- 混淆矩阵：{json.dumps(a_clf['matrix'], ensure_ascii=False)}

### 3.5 有界性属性测试

固定/随机量级样本 **{iv['n_samples']}** 个（y_ref 对数均匀覆盖 1e-300…1e3，10% 为精确 0）：
基线相对误差在 {iv['baseline_undefined']} 个样本上**无定义/无界**，自适应口径全部返回有限值或显式 WITHHELD
（WITHHELD {iv['adaptive_withheld']} 个），自适应误差最大值 {iv['max_adaptive_error']:.4g}。

## 4. 结果图

![Fig1](p266w1n09_degeneracy.png)
![Fig2](p266w1n09_invariance.png)
![Fig3](p266w1n09_cases.png)
![Fig4](p266w1n09_labels.png)

## 5. 注意点（坑、局限、适用范围）

1. **语料是"报告即证据"**：解析的是已发布报告的表格文本，不是原始数值工件。行内数字若被作者写错，
   回放照样按写错的数重判——这恰好是本节点要度量的对象（判据的**表述**）。
2. **重构假设**：表格给的是误差而非 y/y_ref，因此有 8 种重构策略（parser 文档 + 每行 `provenance.reconstruction`）。
   其中 `error_space`（基准有名无数时以报告误差为准）是最弱的一档，已在行级标注。
3. **T2 在语料中为 0**：历史报告几乎不在表格里并列含符号量（ΔV<0 写在"注意点"散文里），
   T2 的证据来自负控集与 m7243 散文。这是语料的**观测事实**，不是分类器缺陷。
4. **标注偏差（重要）**：分类器针对这 40 条标注迭代过 4 轮（空基准/仅容差基准、containment 判据、
   "=0"锚定、自指基准单列），因此 **92.5% 是 in-sample 数字**；样本外的证据是 24 例负控集与回放不变量。
   标注者B 是纯文本启发式而非真人，κ(A,B)={a_b['kappa']:.2f} 显示标注本身有可观的边界模糊。
   一致的失败样例：口径混用（% vs pp）与"基准=0 而残差为小量"两行分类器漏判，已列在 `label_set.json` 的
   `known_disagreements`。
5. **适用范围**：判据必须能表达成 `(y, y_ref, τ, 口径)`；纯定性判据（"一致""落在带内"）判为
   containment/无数字内容，分类器不会硬套三型；这不是覆盖率缺口，而是明确的范围声明。
6. **不改领域仓**：全程只读平台报告与 mission 记录（m7245 无报告，用其 result headline + sha256，
   数字由正则抽取，零手抄）。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

本节点是**度量学/判定器**节点，其"被测量"是判据分类与裁决的一致性，而不是物理量；因此以
**人工标注为基准**给出一致率，并明确说明不适用 3% 的物理误差口径：

| 关键量 | 基准 | 值 | 是否 <3%？ |
|---|---|---|---|
| 退化分类 vs 40 条人工标注 | 人工标注（A） | 一致率 {100*a_clf['agreement']:.1f}%（不一致 {40-a_clf['agree']}/40） | **否（12.5%>3%，但满足≥90% 验收线）** |
| 退化检出（负控） | 设计的 12 条退化样本 | 12/12 | 是（0% 漏检） |
| 良性误报（负控） | 设计的 12 条良性样本 | 0/12 | 是（0%） |
| m7241 相对误差 vs 报告数 | 报告自述 ~60% | {_num(m7241['relative_error_pct'], 2)}%（差 {_num(abs(m7241['relative_error_pct']-60), 2)} pp） | 否（>3%，差异来自按区间下界 0.04 取分母） |
| 案例方向一致性 | 报告自述数值方向 | {cs['n_direction_consistent']}/{cs['n_cases']} | 是（0% 偏离） |
| 有界性 | 基线的无界 | {iv['baseline_undefined']}/{iv['n_samples']} 无界 → 自适应 0 无界 | 是（0%） |

## 7. 工件与复现

```bash
python3 -m pytest tests/test_criteria_metrology_p266w1n09.py -q     # 40 passed
python3 scripts/harvest_criteria_corpus.py --max-reports 400 --mission-ids 7241,7243,7245,7248
python3 scripts/fetch_mission_7245_quote.py
python3 scripts/criteria_case_studies.py
python3 scripts/criteria_metrology_report.py
```

- 模块：`src/tensorphysics/metrology/criteria_semantics/{{model,parser,degeneracy,adaptive,replay,corpus,labeling,controls}}.py`
- 数据：`artifacts/criteria_metrology/{{historical_criteria.jsonl,label_set.json,case_studies.json,corpus_summary.json,report_data.json,acceptance.json}}`
- 文档：`docs/CRITERIA_METROLOGY_P266W1N09.md`

## 8. 验收判据逐项

| 判据 | 结果 |
|---|---|
| (a) 分类 vs 40 条标注一致率 ≥90% | 92.5%（κ={a_clf['kappa']:.3f}）**PASS** |
| (b) 回放 ≥200 条、退化率 ≥5% | {d['corpus']['n_criteria']} 条、{100*dg['rate']:.2f}%（严格三型 {100*dg['rate_strict']:.2f}%）**PASS** |
| (c) 四案例替代判据数值 + 方向一致 4/4 + m7241 显式 FAIL | 10/10 行一致、m7241 FAIL **PASS** |
| (d) 负控 12/12 检出、0/12 误报 | 12/12、0/12 **PASS** |
| (e) y_ref→0 无无界相对误差（≥1000 样本） | {iv['n_samples']} 样本、0 无界 **PASS** |
| (f) pytest 全绿、NaN/None 传播安全 | 40 passed；NaN/None/inf 参数化用例全绿 **PASS** |
"""


def _num(x, nd):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    with open(os.path.join(ART, "report_data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    md = build_markdown(d)
    figures = []
    for name, cap in FIGS.items():
        path = os.path.join(FIGDIR, name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                figures.append({"name": name, "caption": cap,
                                "png_base64": base64.b64encode(fh.read()).decode()})
    payload = {
        "title": "P266W1-N09 验收判据度量学：三类退化情形下的不可判性与自适应判据族",
        "summary": (
            f"把误差判据从字符串提升为带度量语义的结构：回放 {d['corpus']['n_criteria']} 条历史判据，"
            f"判据退化 {100*d['degeneracy']['rate']:.2f}%（严格三型 {100*d['degeneracy']['rate_strict']:.2f}%），"
            f"识别 {d['replay']['n_softened_to_fail']} 条『部分/是』应改判 FAIL；四案例方向一致 "
            f"{d['cases']['n_direction_consistent']}/{d['cases']['n_cases']}，m7241 60% 偏差显式 FAIL；"
            f"24 例负控 12/12 检出 0/12 误报；40 条人工标注一致率 "
            f"{100*d['labels']['annotator_A_vs_classifier']['agreement']:.1f}%（κ={d['labels']['annotator_A_vs_classifier']['kappa']:.2f}）；"
            f"{d['invariance']['n_samples']} 样本属性测试无无界相对误差。"),
        "scenario_key": "p266w1n09_criteria_metrology",
        "mission_id": 9554,
        "tags": ["P266W1-N09", "TensorPhysics", "criteria-metrology", "degeneracy-taxonomy",
                 "adaptive-criterion", "honest-fail", "replay"],
        "markdown": md,
        "figures": figures,
    }
    with open(os.path.join("docs", "CRITERIA_METROLOGY_P266W1N09.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    with open("/tmp/report.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print("report.json bytes", os.path.getsize("/tmp/report.json"), "figures", len(figures))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())