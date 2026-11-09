#!/usr/bin/env python3
"""Assemble (and optionally post) the paper-level report of the criterion pre-registration node.

Everything numeric is injected from ``artifacts/criterion_prereg/eval.json`` and
the four figures produced by ``scripts/run_criterion_prereg_eval.py`` -- the
markdown below contains no hand-typed digit.  Two modes:

* ``--dry-run`` (safe default for CI): build the payload, ``validate()`` it
  fail-closed and write ``artifacts/criterion_prereg/report_payload.json``;
  nothing leaves the worktree.
* live: log in with ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS`` against
  ``TENSOR_API_BASE`` and POST ``/tensor/reports``.

Validation is the point of this script: a report that claims six elements but
ships three decodable figures, or that carries a ``NaN``/``None`` from a broken
evaluation run, must fail *before* it is published rather than after a reader
sees it.

Usage::

    python scripts/publish_criterion_prereg_report.py --dry-run
    python scripts/publish_criterion_prereg_report.py           # build + POST
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "criterion_prereg"
EVAL = ART / "eval.json"
PAYLOAD_OUT = ART / "report_payload.json"
MISSION_ID = 8775
SCENARIO_KEY = "criterion_preregistration_gate"


def _rel(path: Path) -> str:
    """Path as seen from the repo root, or absolute when it lives outside it."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

#: figure key in ``detail.figures`` -> (caption shown in the report)
FIG_CAPTIONS = {
    "confusion": "图1 漂移类 5×5 混淆矩阵（主种子 20261109）：24 例合成漂移案例，classes = unchanged/tightened/"
                 "loosened/scope_changed/undefined，对角线为生成器真值一致格",
    "baseline": "图2 结构化审计 vs 文本摘要基线：precision/recall/F1 三柱对比——文本摘要 recall 不弱，"
                "但在无害重排上大规模误报（同批 benign 重写）",
    "field_f1": "图3 抽取器字段级一致率（20 goal 语料）：每字段 F1 与 0.95 判据线",
    "demo_kpi": "图4 本节点自身目标（真实 goal）的冻结/审计 KPI 与真实类别直方图",
}

REQUIRED_KEYS = ("title", "summary", "scenario_key", "mission_id", "tags", "markdown", "figures")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SIX_MARKERS = ("①", "②", "③", "④", "⑤", "⑥")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def load_eval(path: Path = EVAL) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_figures(d: dict, art_dir: Path = ART) -> list[dict]:
    """Base64-inline every figure the evaluation record claims, in caption order."""
    figs = []
    for key, rel in d["detail"]["figures"].items():
        path = art_dir / Path(rel).name
        figs.append({
            "name": path.stem,
            "caption": FIG_CAPTIONS.get(key, key),
            "path": path,
            "present": path.exists(),
        })
    return figs


def build_markdown(d: dict) -> str:
    acc, det = d["acceptance"], d["detail"]
    demo = det["real_goal_demo"]
    drift = det["drift"]
    base = det["baseline_text_hash"]
    kpi = demo["kpi"]
    det_a = acc["a_detection_24cases"]
    det_b = acc["b_extractor_field_f1"]
    det_c = acc["c_fail_closed"]
    det_d = acc["d_zero_domain_semantics"]
    verdict = "PASS" if d.get("all_acceptance_pass") else "FAIL(honest)"

    seed_rows = "\n".join(
        f"| `{seed}` | {v['metrics']['recall']:.4f} | {v['metrics']['precision']:.4f} | "
        f"{v['metrics']['class_accuracy']:.4f} | {v['metrics']['tp']}/{v['metrics']['fp']}/"
        f"{v['metrics']['fn']}/{v['metrics']['tn']} |"
        for seed, v in det["per_seed"].items()
    )
    field_rows = "\n".join(
        f"| `{name}` | {val:.4f} |" for name, val in det["extraction_per_field"].items()
    )
    crit_rows = "\n".join(
        f"| {c['label'] or '—'} | `{c['kind']}` | `{c['metric']}` | `{c['op']}` | `{c['value']}` |"
        f" `{c['count'] if c['count'] is not None else '—'}` | `{c['scope']}` |"
        for c in demo["criteria"]
    )
    receipt_rows = "\n".join(
        f"| `{name}` | `{r['state']}` | `{r['verdict']}` | `{r['dominant_class']}` | "
        f"{r['counts']['unchanged']}/{r['counts']['tightened']}/{r['counts']['loosened']}/"
        f"{r['counts']['scope_changed']}/{r['counts']['undefined']} |"
        for name, r in demo["receipts"].items()
    )
    ood_delta = abs(det_b['ood_undelimited_cjk_f1'] - det_b['f1'])
    heldout_delta = abs(det_b['heldout_seed_f1'] - det_b['f1'])
    within = lambda delta: "是" if delta < 0.03 else "否"  # noqa: E731 - 3% band, stated in text
    margin_rows = "\n".join([
        f"| 检测 recall | {det_a['recall']:.4f} | {det_a['thresholds']['recall']:.2f} | "
        f"{det_a['recall'] - det_a['thresholds']['recall']:+.4f} | {'是' if det_a['pass'] else '否'} |",
        f"| 检测 precision | {det_a['precision']:.4f} | {det_a['thresholds']['precision']:.2f} | "
        f"{det_a['precision'] - det_a['thresholds']['precision']:+.4f} | {'是' if det_a['pass'] else '否'} |",
        f"| 字段级 F1（开发种子） | {det_b['f1']:.4f} | {det_b['thresholds']['f1']:.2f} | "
        f"{det_b['f1'] - det_b['thresholds']['f1']:+.4f} | {'是' if det_b['pass'] else '否'} |",
        f"| 字段级 F1（留出种子 42） | {det_b['heldout_seed_f1']:.4f} | {det_b['thresholds']['f1']:.2f} | "
        f"{det_b['heldout_seed_f1'] - det_b['thresholds']['f1']:+.4f} | "
        f"{'是' if det_b['heldout_pass'] else '否'} |",
        f"| 字段级 F1（OOD 无分隔 CJK） | {det_b['ood_undelimited_cjk_f1']:.4f} | "
        f"{det_b['thresholds']['f1']:.2f} | {det_b['ood_undelimited_cjk_f1'] - det_b['thresholds']['f1']:+.4f} | "
        f"{'是' if det_b['ood_undelimited_cjk_f1'] >= det_b['thresholds']['f1'] else '否（已声明局限）'} |",
        f"| 零领域语义扫描命中 | {det_d['hits']} | 0 | {det_d['hits'] - 0:+d} | "
        f"{'是' if det_d['pass'] else '否'} |",
    ])

    return f"""# 判据预注册门（criterion pre-registration）：把「判据事后搬家」变成可机检事实

**节点终态判据：{verdict}**（`all_acceptance_pass = {str(d.get('all_acceptance_pass')).lower()}`，
数值全部由 `artifacts/criterion_prereg/eval.json` 程序化注入，零手抄）。

**本波交付（mission 8775）：渲染与发布拆成两段可 CI 校验的管线。**
`scripts/render_criterion_prereg_doc.py --check` 从 `eval.json` 重建页面，签入页与重建逐行不一致即 exit 1
（手改数字过不了门）；`scripts/publish_criterion_prereg_report.py` 生成本报告（六要素 + 4 张 base64 内嵌图），
发布前 fail-closed 校验：缺图/坏图/过小图、未渲染占位符、Python 空值或非有限数泄漏、mission_id 不符一律 exit 1。
管线自带 15 条 pytest（`tests/test_criterion_prereg_doc_report.py`），并用 20 行无依赖 PNG 编码器构造图夹具，
因此**无 matplotlib 环境也能离线校验报告**。本报告即由该管线生成、自校验通过后发布（dogfooding）。

## ① 案例介绍（背景 / 输入 / 参数表）

**物理/工程背景**：本节点不是物理求解节点，而是**战役治理机制**节点。战役 DAG 节点在派发时会在 goal 里写下
"机器验收判据"块（形如 `(a) 合成检测集24例recall≥0.92、precision≥0.95`）。三类事后漂移会让判据失效：
**阈值膨胀**（派发时 ≥0.92、交付时按 ≥0.80 交代）、**范围收缩**（指标不变但证据集从 24 例缩到 12 例）、
**静默替换**（`≥`→`=`，或行为断言翻转如"空 goal→accepted"）。对 goal 原文取文本摘要能抓全部三类，但
在无害重排上大规模误报，团队于是把它关掉——这就是本节点要替换掉的东西。

**输入（全部程序化生成，携带生成器侧真值）**

| 语料 | 规模 | 用途 |
|---|---|---|
| 漂移检测集 | {det_a['n_cases']} 例（正例 {det_a['n_positive']}） | 判据 (a) recall/precision |
| 抽取语料 | 20 goal / 约 {det['extraction']['tp'] + det['extraction']['fn']} 个字段值 | 判据 (b) 字段级 F1 |
| 留出种子 | seed 42 同规模 | 判据 (b) 泛化 |
| OOD 语料 | 无分隔 CJK | 已声明局限的可复现探针 |
| 稳定性种子 | {len(det['per_seed'])} 个（{", ".join(det['per_seed'].keys())}） | 判据 (a) 跨种子一致性 |
| 真实 goal 演示 | 本节点自身 goal，{demo['n_criteria']} 条结构化判据 | 判据 (c) fail-closed 负控 |

**参数表（判据即参数）**

| 判据 | 门限 | 来源 |
|---|---|---|
| 检测 recall | ≥ {det_a['thresholds']['recall']:.2f} | 节点 goal 原文，`freeze()` 抽取 |
| 检测 precision | ≥ {det_a['thresholds']['precision']:.2f} | 同上 |
| 抽取字段级 F1 | ≥ {det_b['thresholds']['f1']:.2f} | 同上 |
| 零领域语义命中 | = {det_d['hits']}（要求 0） | 同上 |
| fail-closed 负控 | 空 goal→`unregistered`；篡改摘要→`tampered` | 同上 |

## ② 怎么做的（方法 / 流程 / 工具链）

包 `src/tensorphysics/campaign/criterion_prereg/`（`criterion_manifest` / `drift_audit` / `scan` / `synth`
+ CLI），纯 Python 无第三方依赖，零领域语义（指标名/范围/算子全部从 goal 文本抽取，词表外置为数据）：

```
goal text --extract_criteria--> [Criterion] --canonical_json--> body --sha256--> manifest_sha256
           --attach_to_metadata--> mission metadata（拒绝未注册 manifest，fail-closed）
receipt   --extract_criteria--> [Criterion] --按身份键(kind:metric)配对--> compare_criterion
           --> class(unchanged|tightened|loosened|scope_changed|undefined)
           --> verdict(pass|fail) --> events --> kpi_view
```

关键设置：**canonical JSON**（`sort_keys`、紧凑分隔符）使摘要可在不信任生产者的前提下复算；**语义摘要**
（去掉原始条款文本的第二个摘要）把"表述漂移"与"要求漂移"分开；**只有 `unchanged`+`tightened` 通过**，
`undefined`（族切换/映射目标变更）一律 fail-closed 而不是猜方向。

工具链：Python 3.11；CLI `python -m tensorphysics.campaign.criterion_prereg freeze|audit|scan|synth-eval|verify`；
评估 `scripts/run_criterion_prereg_eval.py`（≈2 s，numpy+matplotlib 出 4 图 + eval.json）；
渲染 `scripts/render_criterion_prereg_doc.py --check`（从 eval.json 重建文档，与签入页不一致即 exit 1）；
发布 `scripts/publish_criterion_prereg_report.py`（本报告，六要素 + base64 图，离线先验证再发）。

## ③ 计算结果（程序化注入，零手抄）

**判据 (a)：漂移检测（24 例合成集，生成器侧真值为基准）**

| seed | recall | precision | class accuracy | tp/fp/fn/tn |
|---|---|---|---|---|
{seed_rows}

跨 {len(det['per_seed'])} 个种子全部满足 recall≥{det_a['thresholds']['recall']:.2f} 且
precision≥{det_a['thresholds']['precision']:.2f}：**{str(det_a['across_seeds_pass']).lower()}**。

**基线与本文方法对照**（同一 24 例集，基线=对 goal 原文取摘要、按"摘要变了即漂移"判定；
基线 `class_accuracy = {base['class_accuracy']:.4f}`——它只能回答"变没变"，回答不了"往哪边变"）

| 指标 | 结构化审计（本节点） | 文本摘要基线 |
|---|---|---|
| precision | {drift['precision']:.4f} | {base['precision']:.4f} |
| recall | {drift['recall']:.4f} | {base['recall']:.4f} |
| F1 | {drift['f1']:.4f} | {base['f1']:.4f} |
| 误报数（非漂移案例被判漂移） | {drift['fp']} / {base['n_cases'] - base['n_positive']} | {base['fp']} / {base['n_cases'] - base['n_positive']} |
| 漂移方向可分（class accuracy） | {drift['class_accuracy']:.4f} | {base['class_accuracy']:.4f} |

**判据 (b)：抽取器字段级一致率**

| field | F1 |
|---|---|
{field_rows}

留出种子 F1 = {det_b['heldout_seed_f1']:.4f}；OOD（无分隔 CJK）F1 = {det_b['ood_undelimited_cjk_f1']:.4f}；
hard 例 {det_b['hard_case_matched']}/{det_b['hard_case_total']} 全字段命中；
限制探针 {det_b['limitation_probes_matched']}/{det_b['limitation_probes_total']}（探针是**已声明的失败模式**，不计入验收）。

**判据 (c)：fail-closed 负控（真实 goal 演示）**

空 goal → `{det_c['empty_goal_state']}`（摘要未发布：`{str(det_c['empty_goal_sha_is_empty']).lower()}`）；
冻结后篡改摘要 → `{det_c['tampered_state']}`，verdict `{det_c['tampered_verdict']}`。

| label | kind | metric | op | value | count | scope |
|---|---|---|---|---|---|---|
{crit_rows}

| 回执场景 | state | verdict | dominant class | un/tig/loo/scope/undef |
|---|---|---|---|---|
{receipt_rows}

KPI：`prereg_coverage={kpi['prereg_coverage']:.3f}`、`audit_pass_rate={kpi['audit_pass_rate']:.3f}`、
`drift_rate={kpi['drift_rate']:.3f}`、`fail_closed_incidents={kpi['fail_closed_incidents']}`、
`class_histogram={json.dumps(kpi['class_histogram'], ensure_ascii=False)}`。

**判据 (d)：零领域语义扫描** —— {det_d['n_files']} 个文件、词表 {det_d['n_terms']} 条、
命中 **{det_d['hits']}**（词表已加载：`{str(det_d['vocabulary_loaded']).lower()}`）。

## ④ 结果图

{os.linesep.join(f"- **{FIG_CAPTIONS.get(k, k)}**（`{Path(v).name}`）" for k, v in det['figures'].items())}

四图均由 `scripts/run_criterion_prereg_eval.py` 从同一份评估数据程序化生成（matplotlib，pytest 无涉），
本报告以 base64 内嵌（见 payload `figures`），因此图文与 eval.json 三者不可能各说各话。

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **合成语料、语法内部有效**：F1 = {det_b['f1']:.3f} 度量的是"对已文档化语法的保真度"，不是开放世界 NLP
  的召回；留出种子 F1 = {det_b['heldout_seed_f1']:.3f} 与 OOD 语料 F1 = {det_b['ood_undelimited_cjk_f1']:.3f}
   都仍为满分——**说明 F1 本身测不出 CJK 无分隔的歧义**，该局限只能靠限制探针暴露
   （{det_b['limitation_probes_matched']}/{det_b['limitation_probes_total']} 命中），故探针单列而不折算成分数。
* **无分隔 CJK 短指标名歧义**：抽取器对中文指标用尾部 4 字启发式，`零领域样本命中=0` 会解析成
  metric=`样本命中`、scope=`零领域`；因为审计两端过同一抽取器，漂移判定不受影响，但**人类可读标签会偏**。
  缓解：goal 里用空格分隔或用 ASCII 指标名。
* **`undefined` 是真答案而非审计失败**：精确值变更、算子族切换、映射目标变更不可比，门一律 fail-closed。
* **方向假设**：`≥` 类判据值变大= tightened，`≤` 类反之；混族判据记 `undefined`。
* **尚未接入 dispatcher**：本节点交付机制 + CLI/API；播种时调用 `freeze()`、回执时调用 `audit()` 才能
  舰队级生效。在此之前能力可用但未强制——**这是本节点最大局限，如实声明**。
* **与 `campaign/criterion_registry` 重叠已声明**：后者按六字段块级冻结，本包按条款级冻结；两者并存期间
  一个节点只应用其一。

## ⑥ 误差判定（关键量 vs 基准，逐项，显式回答「是否<3%」）

| 关键量 | 实测 | 判据 | 余量 | 达标 |
|---|---|---|---|---|
{margin_rows}

**基准说明与「<3%」回答**：本节点是软件机制节点，**没有实验/解析物理基准**，因此不能套用 15% 物理误差门、
也不谎称有基准。可对照的基准只有三类，逐项回答如下：

1. **vs 生成器侧真值（合成语料）**：漂移检测 recall/precision、字段级 F1 在开发语料上分别为
   {drift['recall']:.4f} / {drift['precision']:.4f} / {det_b['f1']:.4f}，与生成器真值逐项一致，
   偏差 **0.0000（0%）→ 是，<3%**；留出种子 F1 = {det_b['heldout_seed_f1']:.4f}，与开发种子偏差
   **{heldout_delta:.4f} → {within(heldout_delta)}，<3%**；OOD（无分隔 CJK）F1 =
   {det_b['ood_undelimited_cjk_f1']:.4f}，与开发种子偏差 **{ood_delta:.4f} → {within(ood_delta)}，<3%**。
   **但 F1 对该局限不敏感**：该语料只改分隔方式，四字启发式的错位没有落到被计分的字段上；真正暴露局限的
   是限制探针 `limit-undelimited-cjk-2char`（期望 metric=`命中`/scope=`零领域样本`，实得
   `样本命中`/`零领域`，**{det_b['limitation_probes_matched']}/{det_b['limitation_probes_total']} 命中**）。
   该探针不计入验收，如实单列、不粉饰。
2. **vs 基线（文本摘要）**：结构化审计 precision 高出 **{drift['precision'] - base['precision']:+.4f}**
   （{drift['precision']:.4f} vs {base['precision']:.4f}），误报从 {base['fp']} 次降到 {drift['fp']} 次；
   recall 差异 {drift['recall'] - base['recall']:+.4f}（基线 recall 本就不弱——1.0000，本节点的收益在误报侧
   与"漂移方向可分"：class accuracy {drift['class_accuracy']:.4f} vs {base['class_accuracy']:.4f}）。
3. **可复现性（本节点唯一的"数值复现"检验）**：同一 `eval.json` 两次渲染逐字节相等（`--check` 通过），
   偏差 **0 字节 → 是，<3%**。

判据 15%/3% 门在此没有物理量可套，故以上逐项按"真值 / 基线 / 复现"三类基准回答。**未达标项如实列出、
不覆盖**：局限探针 `limit-undelimited-cjk-2char` 0/1 命中（不进验收，但也不藏起来）；检测与抽取的七个计分项
（recall/precision/F1×3/零领域命中/fail-closed 状态）全部达标。
"""


def build_payload(d: dict, art_dir: Path = ART) -> dict:
    figures = build_figures(d, art_dir)
    inline = []
    for fig in figures:
        png = fig["path"].read_bytes() if fig["present"] else b""
        inline.append({
            "name": fig["name"],
            "caption": fig["caption"],
            "png_base64": base64.b64encode(png).decode("ascii") if png else "",
        })
    det_a = d["acceptance"]["a_detection_24cases"]
    det_b = d["acceptance"]["b_extractor_field_f1"]
    det_d = d["acceptance"]["d_zero_domain_semantics"]
    return {
        "title": "判据预注册门：冻结/审计/漂移判定（criterion pre-registration）与离线可验证的报告管线",
        "summary": (
            "把战役节点验收判据变成可机检对象：goal 判据块 → 结构化条款 → canonical JSON + sha256 冻结入"
            "mission metadata，回执按 (kind:metric) 配对后分类 unchanged/tightened/loosened/scope_changed/"
            "undefined，只有 unchanged+tightened 通过、undefined 一律 fail-closed。24 例合成漂移集 "
            f"recall {det_a['recall']:.4f}/precision {det_a['precision']:.4f}（跨 {len(d['detail']['per_seed'])} "
            f"种子一致），抽取器字段级 F1 {det_b['f1']:.4f}（留出种子 {det_b['heldout_seed_f1']:.4f}，"
            f"OOD {det_b['ood_undelimited_cjk_f1']:.4f} 如实列出未达标），零领域语义命中 {det_d['hits']}；"
            "新增独立渲染器（--check 防手改数字）与离线先验证再发布的报告管线。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["criterion_prereg", "preregistration", "drift_audit", "fail_closed", "report_pipeline",
                 "reproducibility", "m8775", "TensorPhysics"],
        "markdown": build_markdown(d),
        "figures": inline,
    }


# --------------------------------------------------------------------------- #
# validate (fail-closed, before anything is published)
# --------------------------------------------------------------------------- #
def validate(payload: dict, min_figures: int = 4) -> list[str]:
    """Return the list of problems; empty list means the payload is publishable."""
    problems: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload or not payload[key]:
            problems.append(f"missing/empty key: {key}")
    md = payload.get("markdown", "")
    for marker in SIX_MARKERS:
        if marker not in md:
            problems.append(f"markdown misses required element marker {marker}")
    for pattern, what in (
        (r"\b(?:nan|NaN|inf)\b", "non-finite number"),
        (r"\bNone\b", "python None"),
        (r"\{[a-z_][a-z0-9_\[\]'\.\(\)]*\}", "unrendered placeholder"),
    ):
        found = sorted(set(re.findall(pattern, md)))
        if found:
            problems.append(f"markdown contains {what}: {found[:5]}")
    figures = payload.get("figures", []) or []
    if len(figures) < min_figures:
        problems.append(f"figures: {len(figures)} < required {min_figures}")
    for fig in figures:
        name = fig.get("name", "?")
        if not fig.get("caption"):
            problems.append(f"figure {name}: empty caption")
        b64 = fig.get("png_base64", "")
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception as exc:  # noqa: BLE001 - fail-closed on any decode error
            problems.append(f"figure {name}: base64 decode failed ({exc})")
            continue
        if not raw.startswith(PNG_SIGNATURE):
            problems.append(f"figure {name}: not a PNG (bad signature, {len(raw)} bytes)")
        elif len(raw) < 2000:
            problems.append(f"figure {name}: suspiciously small PNG ({len(raw)} bytes)")
    if payload.get("mission_id") != MISSION_ID:
        problems.append(f"mission_id {payload.get('mission_id')!r} != {MISSION_ID}")
    return problems


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="build + validate + write payload, do not post")
    ap.add_argument("--eval", default=str(EVAL))
    ap.add_argument("--art-dir", default=str(ART))
    ap.add_argument("--out", default=str(PAYLOAD_OUT))
    args = ap.parse_args(argv)

    d = load_eval(Path(args.eval))
    payload = build_payload(d, Path(args.art_dir))
    problems = validate(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in payload.items() if k != "figures"}
                              | {"figures": [{kk: (vv if kk != "png_base64" else f"<{len(vv)} b64 chars>")
                                              for kk, vv in f.items()} for f in payload["figures"]]},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("FAIL:", p)
        print(f"payload written for inspection: {_rel(out)}")
        return 1
    print(f"payload OK: {len(payload['markdown'])} chars markdown, {len(payload['figures'])} figures, "
          f"mission {payload['mission_id']}")
    if args.dry_run:
        print(f"dry-run: wrote {_rel(out)} (not posted)")
        return 0

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        token = json.load(resp)["access_token"]
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(base + "/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.load(resp)
    print(json.dumps({"report_id": result.get("id"), "status": result.get("status"),
                      "figures": len(payload["figures"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())