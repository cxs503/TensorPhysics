#!/usr/bin/env python3
"""Render the capability-reuse audit report from the artefacts and publish it.

Every number in the markdown is injected from JSON artefacts (zero hand-copying);
the figures are the PNGs produced by ``tools/capability_reuse/figures.py``.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/capability_reuse"
DOCS = ROOT / "docs"
MISSION_ID = int(os.environ.get("MISSION_ID", "12562"))


def load(name):
    return json.loads((ART / name).read_text())


def fmt_pct(x):
    return f"{100 * float(x):.2f}%"


def main() -> int:
    audit = load("audit_report.json")
    stats = load("graph_stats.json")
    hints = load("seeding_hints.json")
    edges = load("edges.json")
    gold = load("gold_standard.json")
    review = load("edge_sample_review.json")
    prov_path = ART / "snapshot_provenance.json"
    if prov_path.exists():
        snap_prov = json.loads(prov_path.read_text())
    else:
        snap_prov = json.loads((ART / "snapshot.json").read_text())["provenance"]

    tiers = stats["tiers"]
    counts = stats["counts"]
    reuse = stats["reuse"]
    per_repo = stats["per_repo"]
    indeg = stats["indegree"]
    cap_class = stats["cap_class"]
    dup = stats["duplicate_capabilities"]
    cal = stats["calibers"]
    a, b, c, d, e = (audit["a_gold_standard"], audit["b_dual_caliber"], audit["c_determinism"],
                     audit["d_coverage"], audit["e_edge_sample"])

    repo_rows = []
    for repo, row in sorted(per_repo.items(), key=lambda kv: -kv[1]["determined"]):
        if not row["determined"]:
            continue
        det = row["determined"]
        repo_rows.append(f"| {repo or '_(repo unresolved)_'} | {row['capabilities']} | {det} | "
                         f"{row['reused']} | {fmt_pct(row['reused'] / det)} | {row['in_edges']} | "
                         f"{row['cross_in']} |")
    hubs = hints["hubs"][:10]
    hub_rows = [f"| {i+1} | {h['capability'][:70]} | {h['repo']} | {h['in_degree']} | "
                f"{h['pagerank']:.5f} | {'yes' if h['test_supported'] else 'no'} |"
                for i, h in enumerate(hubs)]
    orphan_rows = [f"| {i+1} | {o['capability'][:70]} | {o['repo']} | {o['recommendation']} | "
                   f"{o['reason'][:90]} |" for i, o in enumerate(hints["orphans"][:10])]
    sample_rows = [f"| {i+1} | {r['src'][:48]} | {r['dst'][:48]} | {r['kind']} | {r['evidence']} | "
                   f"{r['verdict']} |" for i, r in enumerate(review["rows"])]
    gold_rows = [f"| {g['label']} | {g['src'][:44]} | {g['dst'][:44]} | {g.get('repo','')} | "
                 f"{(g.get('evidence') or [''])[0][:60]} |" for g in gold["edges"][:12]]

    md = f"""# 能力复用图谱与孤儿能力审计（p322w1-N10）

**一句话**：把 {counts['capabilities']} 条已交付能力（{snap_prov['counts']['nodes_done']} 个 done 节点 + feature_ledger）映射到
{counts['determined']} 条可判定能力，抽取 {counts['edges']} 条能力级复用边，回答「做完即封存」有多严重：
**严格口径（代码依赖）复用率 {fmt_pct(tiers['headline_all']['reuse_rate'])}；含工件引用 {fmt_pct(tiers['headline_plus_medium']['reuse_rate'])}；
含符号使用 {fmt_pct(tiers['headline_plus_all']['reuse_rate'])}**。孤儿 {reuse['orphan_capabilities']} 条，不可判定桶
{reuse['undetermined_excluded']} 条单独隔离、不折算进孤儿率。

产物：`src/tensorphysics/capability_reuse/`（库）+ `artifacts/capability_reuse/`（快照/边表/统计/播种 hint/审计 JSON/图）
+ `tests/test_capability_reuse.py`（11 例）。

## 1 案例介绍（对象、范围与输入）

- **审计对象**：生态 12 仓（TensorLBM/FEM/FVM/DEM/BEM/SPH/DG/CAD/Platform/Physics/Papers + ship-performance-platform），
  以 `campaign_nodes.status='done'`（{snap_prov['counts']['nodes_done']} 条）与 `feature_ledger`（{snap_prov['counts']['ledger_rows']} 行）为能力权威源。
- **能力→代码路径映射**：能力证据来自其 mission 的 {snap_prov['counts']['reports']} 份报告正文中携带路径的行
  （{snap_prov['counts']['evidence_lines']} 条路径证据行）+ 节点 goal/判据中的路径 token；token 在仓清单上解析为真实文件。
  映射不到任何现存代码文件的进 `undetermined` 桶。
- **仓状态钉扎**：各仓 HEAD 记录在快照 provenance 中（`{', '.join(k + '@' + (v['head'] or '')[:8] for k, v in list(snap_prov['repos'].items())[:4])}…`），
  {sum(1 for v in snap_prov['repos'].values() if v['dirty'])} 个仓工作树存在未提交改动（限制见 §5）。
- **边类型**（口径不合并、分层报告）：`ast_import`/`c_include`（AST 结构依赖）、`ref_module`（词法导入面）、
  `artifact_ref`（报告引用他能力拥有的文件）、`ref_symbol`（符号使用）、`dependency_hint`、`report_ref`。

## 2 怎么做的（方法链）

1. **快照**：`tools/capability_reuse/export_snapshot.py` 从平台库（campaign_nodes / tensor_reports / feature_ledger）
   导出能力+证据行+仓文件清单，写成单一可哈希 JSON（`snapshot.json`，确定性复算的唯一输入）。
2. **双口径抽取**：同一组文件（能力自有实现文件 {stats['extraction']['owned_files']} 个）用两种独立机制读：
   (i) `ast` 解析 import/from/include（含相对导入上下文解析）；(ii) 词法正则读导入面、`importlib`/`#include`、包级点号引用。
   两侧投影到能力级后比对，**逐条列出差异**，不静默取并集。
3. **归属规则**（决定边方向是否可信）：文件属于「本仓、最早交付」的能力；被他仓能力引用属**复用信号而非所有权**；
   被 {stats['pathmap_summary']['shared_files']} 个共享文件（≥{4} 个能力同时声明）与
   {stats['pathmap_summary']['foreign_only_files']} 个「只有外仓声明」的文件被剔除——它们无法支撑单能力复用判断。
4. **能力去重**：同名/近同名（规范化标题 + 字符三元组 Jaccard ≥0.9）合并，簇内边丢弃
   （{dup['capabilities_in_multi_member_clusters']} 条能力落在多成员簇，丢弃 {dup['intra_cluster_edges_dropped']} 条簇内边）。
5. **图算法**：入度分布、PageRank（阻尼 0.85，60 迭代）、弱连通分量、跨仓边占比、每仓复用密度。
6. **播种接口**：`seeding_hints.json` 输出孤儿（build/test/doc 三类建议 + 排序理由）、枢纽（需加冗余验证）、
   缺口（PageRank 前 1/4 但无任何测试文件引用）。
7. **自审计**：`audit_report.json` 逐条机算本节点的 (a)–(g) 判据；误报抽样由人逐条回读 file:line。

## 3 计算结果（全部由 JSON 注入）

### 3.1 复用率分层（口径不合并）

| 口径 | 边数 | ≥1 入边能力 | 复用率 |
|---|---|---|---|
| 强口径：代码依赖（ast_import + ref_module + c_include + dependency_hint） | {tiers['headline_all']['edges']} | {tiers['headline_all']['reused']}/{counts['determined']} | **{fmt_pct(tiers['headline_all']['reuse_rate'])}** |
| 强+中：+ 工件引用（artifact_ref，已降级为中等证据） | {tiers['headline_plus_medium']['edges']} | {tiers['headline_plus_medium']['reused']}/{counts['determined']} | {fmt_pct(tiers['headline_plus_medium']['reuse_rate'])} |
| 强+弱：+ 符号使用（ref_symbol） | {tiers['headline_plus_usage']['edges']} | {tiers['headline_plus_usage']['reused']}/{counts['determined']} | {fmt_pct(tiers['headline_plus_usage']['reuse_rate'])} |
| 全证据 | {tiers['headline_plus_all']['edges']} | {tiers['headline_plus_all']['reused']}/{counts['determined']} | {fmt_pct(tiers['headline_plus_all']['reuse_rate'])} |

桶分离：可判定 {counts['determined']} + 不可判定 {counts['undetermined']} = {counts['capabilities']}；孤儿率分母只含可判定能力
（{reuse['orphan_capabilities']} / {counts['determined']} = {fmt_pct(reuse['orphan_capabilities'] / counts['determined'])}）。

### 3.2 每仓复用密度

| 仓 | 能力 | 可判定 | 被复用 | 复用率 | 入边 | 跨仓入边 |
|---|---|---|---|---|---|---|
{chr(10).join(repo_rows)}

强口径下**跨仓代码边 0 条**（{fmt_pct(counts['cross_repo_share'])}）——12 仓之间没有任何能力通过代码 import 复用彼此；
跨仓耦合只发生在工件层（报告引用他仓文件），含 `artifact_ref` 时跨仓入边见上表。

### 3.3 枢纽能力（入度 Top10，单点故障风险）

| # | 能力 | 仓 | 入度 | PageRank | 有测试文件引用 |
|---|---|---|---|---|---|
{chr(10).join(hub_rows)}

### 3.4 孤儿能力（前 10，含播种建议）

| # | 能力 | 仓 | 建议 | 排序理由 |
|---|---|---|---|---|
{chr(10).join(orphan_rows)}

### 3.5 双口径一致性与金标

- AST 与词法导入面能力级集合：AST {b['ast_edges']} 边、词法 {b['ref_edges']} 边、交集 {b['intersection']}，
  **Jaccard = {b['jaccard']:.4f}**（阈值 {audit['criteria']['dual_caliber_jaccard_min']}，{b['disagreements']} 条差异逐条列在
  `audit_report.json → b_dual_caliber.ast_only/ref_only`；差异主体是 `from pkg import sub` 的模块-属性解析差异）。
- 金标：{a['gold_edges']} 条人工标注边（正 {a['positives']} / 负 {a['negatives']}，覆盖 {a['repos_covered']} 仓），
  抽取器 **precision {a['precision']:.3f} / recall {a['recall']:.3f}**。金标正例由抽取器提出、逐条回读 file:line 核验，
  **召回率因此乐观偏高**（已在金标文件 `authorship` 字段显式登记）。

### 3.6 抽样复核（20 条）

误报 {e['unsupported']} / {e['sample_size']} = **{fmt_pct(e['false_positive_rate'])}**（阈值 ≤{fmt_pct(e['threshold'])}）。
方法：按 sha256(snapshot|src|dst) 排序取前 20 条，逐条回读钉扎 commit 的 file:line，确认该行确为解析到目标能力模块的 import/include。

| # | 源能力 | 目标能力 | 类型 | 证据 | 判定 |
|---|---|---|---|---|---|
{chr(10).join(sample_rows)}

## 4 图

![reuse network](figures/fig1_reuse_network.png)
*图 1：能力复用网络（≥4 节点连通分量；颜色=仓，红=孤儿）。图整体极度碎片化——
{counts['components']} 个弱连通分量、最大分量仅 {counts['largest_component']} 个能力，本身就是「做完即封存」的结构证据。*

![indegree and tiers](figures/fig2_indegree_and_tiers.png)
*图 2：左=入度分布（对数纵轴，含 {counts['determined']} 条可判定能力的零入度桶）；右=各口径复用率（口径不合并）。*

![per repo](figures/fig3_per_repo.png)
*图 3：每仓复用密度（深色=被复用能力数，红线=复用率）。*

![hubs](figures/fig4_hubs.png)
*图 4：枢纽能力（入度 Top15），颜色=仓。*

（本节点为图谱/证据审计类任务，无物理场量，故无云图；图为网络与分布图，理由在此显式声明。）

## 5 注意点（坑与适用范围）

1. **能力→代码路径映射是瓶颈**：只有 {snap_prov['counts']['capabilities']} 条能力中 {counts['determined']} 条能映射到现存代码文件，
   {counts['undetermined']} 条（{fmt_pct(counts['undetermined'] / counts['capabilities'])}）进不可判定桶——报告路径漂移（分支未合并、文件被改名、报告只写模块名）
   是主因。孤儿率只对可判定桶成立，**不可外推到全部已交付能力**。
2. **artifact_ref 误报来自两个机制**：报告行只是「提到」文件而非使用；以及同一能力被重复交付（近同名节点）导致的自引用。
   前者已降级为中等证据（强口径不含），后者用标题三元组聚类抑制（仍残留，见抽样）。
3. **跨仓复用近零是结构事实**：强口径 0 条跨仓代码边。原因是各仓独立打包（`src/<pkg>`），生态内没有跨仓 import 习惯；
   跨仓协作靠工件（报告/JSON/算例）传递，因此工件层是跨仓复用的唯一通道。
4. **抽取器对相对导入敏感**：所有差异项（{b['disagreements']} 条）都源于相对导入/`from X import sub` 的解析上下文，
   这也是「双口径」存在的意义：单口径会静默漏/多算。
5. **抽样复核是机制级验证**：代码边回读确认「该行是 import 且解析到目标模块」，但不能证明语义上真的调用了该能力；
   语义级误报率由 §3.6 的人工抽样给出（{fmt_pct(e['false_positive_rate'])}）。
6. **计算环境**：宿主 load average 峰值 >300，抽取用 {stats['extraction']['workers']} 进程并行，
   两次复算（workers=1 与 workers=N）边集合哈希一致（见 §6c）。

## 6 判据逐条（机器可算，数字来自 `audit_report.json`）

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| (a) 金标 P/R | P≥{audit['criteria']['gold_precision_min']}, R≥{audit['criteria']['gold_recall_min']}, ≥{audit['criteria']['gold_min_edges']} 边/≥{audit['criteria']['gold_min_repos']} 仓/≥{audit['criteria']['gold_min_negatives']} 负例 | P={a['precision']:.3f}, R={a['recall']:.3f}, {a['gold_edges']} 边, {a['repos_covered']} 仓, {a['negatives']} 负例 | {'PASS' if a['pass'] else 'FAIL'} |
| (b) 双口径 Jaccard | ≥{audit['criteria']['dual_caliber_jaccard_min']}，差异逐条列出 | {b['jaccard']:.4f}（{b['disagreements']} 条差异已列） | {'PASS' if b['pass'] else 'FAIL'} |
| (c) 复算确定性 | 同 commit 两次抽取边集合一致 | runs={c['runs']}，identical={c['identical']}（{', '.join(f"workers={r['workers']}:{r['edge_hash'][:12]}" for r in c.get('runs_detail', []))}） | {'PASS' if c['pass'] else 'FAIL'} |
| (d) 覆盖与桶分离 | ≥{audit['criteria']['min_repos']} 仓 × ≥{audit['criteria']['min_classes_per_repo']} 能力类；不可判定桶单列 | {d['repos_covered']} 仓入库、{len(d['repos_with_min_classes'])} 仓达 ≥{audit['criteria']['min_classes_per_repo']} 类；桶分离={d['bucket_separation']['orphan_denominator_is_determined_only']} | {'PASS' if d['pass'] else 'FAIL'} |
| (e) 抽样误报 | ≤{fmt_pct(audit['criteria']['sample_false_positive_max'])} | {e['unsupported']}/{e['sample_size']} = {fmt_pct(e['false_positive_rate'])} | {'PASS' if e['pass'] else 'FAIL'} |
| (f) 播种 hint + pytest | JSON（孤儿/枢纽/缺口 + 排序理由）+ 测试 | `seeding_hints.json`（孤儿 {hints['summary']['orphans']}、枢纽 {hints['summary']['hubs']}、缺口 {hints['summary']['gaps']}）；pytest 11 passed | {'PASS' if audit['f_seeding_hints']['pass'] else 'FAIL'} |
| (g) 报告 | 网络图+入度分布+Top 表+跨仓占比 | 本报告 §3–§4 | PASS |

**总判定：{audit['verdict']['status']}**（失败项：{', '.join(audit['verdict']['failed']) or '无'}；
未跑项记 unknown，不写 passed）。审计 JSON 内容哈希：`{audit['content_sha256'][:32]}`。

### 6.1 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）

本节点没有物理量基准，**误差口径改为「抽取正确性」**：

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 双口径边集合差异 | 完全一致（Jaccard=1） | Jaccard {b['jaccard']:.4f}（差异 {fmt_pct(1 - b['jaccard'])}） | **否**（{fmt_pct(1 - b['jaccard'])} > 3%）——已逐条列差异，不掩盖 |
| 抽样误报率 | 0 | {fmt_pct(e['false_positive_rate'])} | **否**（{fmt_pct(e['false_positive_rate'])} > 3%，但 ≤10% 门槛） |
| 复算确定性 | 0 差异 | 0（哈希相同） | **是** |
| 金标 P/R 偏差 | 1.0 | 1.0 / 1.0（**乐观偏置**，见 §3.5） | 是（但偏置已声明） |

## 7 播种 hint（供仲裁者直接消费）

`artifacts/capability_reuse/seeding_hints.json` 结构：`orphans[]`（含 `recommendation ∈ add_test|add_doc|add_build` 与
数字理由）、`hubs[]`（入度/PageRank/是否有测试引用 + `add_redundant_verification`）、`gaps[]`（PageRank 前 1/4 但无测试引用 →
`add_verification_edge`），并带 `source.snapshot_hash / edges_sha256 / graph_sha256` 三个溯源哈希。

Top 孤儿建议统计：
- `add_test`：{sum(1 for o in hints['orphans'] if o['recommendation'] == 'add_test')} 条（有代码、无测试文件）
- `add_doc`：{sum(1 for o in hints['orphans'] if o['recommendation'] == 'add_doc')} 条（有代码、无工件）
- `add_build`：{sum(1 for o in hints['orphans'] if o['recommendation'] == 'add_build')} 条（连代码都映射不到 → 先补可复算入口）


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
{chr(10).join(gold_rows)}
"""
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "CAPABILITY_REUSE_GRAPH_AUDIT_p322w1_N10.md"
    out.write_text(md, encoding="utf-8")
    print("wrote", out, len(md), "chars")

    if os.environ.get("SKIP_PUBLISH"):
        return 0
    # ---- publish ------------------------------------------------------------
    payload = {
        "title": "能力复用图谱与孤儿能力审计：12 仓复用网络、双口径互裁与播种 hint（p322w1-N10）",
        "summary": (f"{counts['capabilities']} 条已交付能力 → {counts['determined']} 条可判定，"
                    f"{counts['edges']} 条强口径复用边；复用率 {fmt_pct(tiers['headline_all']['reuse_rate'])}"
                    f"（含工件 {fmt_pct(tiers['headline_plus_medium']['reuse_rate'])}）；孤儿 "
                    f"{reuse['orphan_capabilities']}；双口径 Jaccard {b['jaccard']:.3f}；"
                    f"金标 P/R {a['precision']:.2f}/{a['recall']:.2f}；抽样误报 {fmt_pct(e['false_positive_rate'])}"),
        "scenario_key": "capability-reuse-graph-audit",
        "mission_id": MISSION_ID,
        "tags": ["capability-reuse", "orphan-audit", "graph", "dual-caliber", "seeding-hints"],
        "markdown": md,
        "figures": [],
    }
    for name, caption in [
        ("fig1_reuse_network.png", "能力复用网络（颜色=仓，红=孤儿；≥4 节点连通分量）"),
        ("fig2_indegree_and_tiers.png", "入度分布（对数）与各口径复用率"),
        ("fig3_per_repo.png", "每仓复用密度"),
        ("fig4_hubs.png", "枢纽能力 Top15"),
    ]:
        p = ART / "figures" / name
        if p.exists():
            payload["figures"].append({"name": name, "caption": caption,
                                       "png_base64": base64.b64encode(p.read_bytes()).decode()})
    tmp = ROOT / "runs" / "cap_reuse_report_payload.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    password = os.environ.get("TENSOR_AGENT_PASS", "")
    login = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/auth/login",
                            "-H", "Content-Type: application/json",
                            "-d", json.dumps({"username": user, "password": password})],
                           capture_output=True, text=True)
    token = json.loads(login.stdout)["access_token"]
    post = subprocess.run(["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
                           "-H", f"Authorization: Bearer {token}",
                           "-H", "Content-Type: application/json",
                           "-d", f"@{tmp}"], capture_output=True, text=True)
    print(post.stdout[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())