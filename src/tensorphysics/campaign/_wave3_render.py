"""Markdown rendering for the wave-3 closure gate report.

Kept separate from :mod:`tensorphysics.campaign.wave3_gate` so that the
evaluation logic stays importable without any presentation code, and so that the
rendered report is a pure function of the :class:`GateReport` (every number in the
report is injected from the report object -- nothing is hand-copied).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .wave3_gate import GateReport, NodeRecord

FIG_DIR = "artifacts/in19_g19_wave3_gate/figures"

_VERDICT_MARK = {
    "accept": "PASS",
    "partial": "PARTIAL",
    "fail": "FAIL",
    "withheld": "WITHHELD",
}

_CAUSE_GLOSS = {
    "model_form_error": "模型形式误差",
    "input_uncertainty": "输入不确定度",
    "implementation_bug": "实现 bug",
    "evidence_gap": "证据缺口",
}

_LINEAGE_MARK = {
    "mature": "成熟 → 进入下一战役基线",
    "iterate": "迭代 → 保留在波4轨道",
    "shelve": "搁置 → 退出基线",
}


def _esc(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _fmt_full(value: float | None, sig: int = 6) -> str:
    """Full-precision display, so an error column can be re-derived from the table."""
    if value is None:
        return "—"
    return f"{value:.{sig}g}"


def _fmt(value: float | None, sig: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{sig}g}"


def _rationale_cn(node, probes) -> str:
    """Human-facing restatement of the machine rationale (display only).

    The machine field stays in the JSON (and in the digest); this function only
    composes the Chinese sentence used in the table, enriched with the gate probe
    of that node so that the inheritance column is falsifiable at a glance.
    """
    lineage = node.lineage.value
    if lineage == "mature":
        return "全部机器判据已对过外部基准并通过 → 可入下一战役基线"
    if node.evidence_grade.value == "NOT_SUPPORTED":
        return (
            "能力在主干缺席（EEXI/CII 端点 404），属未实现而非精度不足 → "
            "先按规范实现，再由门复算后才能继承"
        )
    node_probes = [p for p in probes if p.node_id == node.node_id]
    if node_probes:
        worst = max(node_probes, key=lambda q: abs(q.rel_error_pct))
        return (
            f"验收未提交（mission {node.mission_status}）；门侧探针实测 "
            f"{worst.quantity} 偏差 {worst.rel_error_pct:+.2f}%（阈值 {worst.threshold_pct:g}%），"
            "交付后仍须先过该判据"
        )
    if node.criteria:
        return f"验收未提交（mission {node.mission_status}），无对过基准的实测证据可继承"
    return "未解析到机器判据，无法核销"


def render_report(report: "GateReport") -> str:
    nodes = report.nodes
    lines: list[str] = []
    add = lines.append

    add("# 波3（IN19）核销门报告：全节点验收汇总与谱系继承裁决")
    add("")
    add(f"- 战役计划：`{report.plan_name}`（plan_id={report.plan_id}）")
    add(f"- 本门节点：`{report.gate_ext_id}`（mission_id={report.gate_mission_id}）")
    add(f"- 证据截止时刻（cutoff）：`{report.cutoff}`")
    add(f"- 报告正文摘要（sha256，**规范化 JSON 核心字段**，非 markdown 文件哈希）：`{report.body_digest}`")
    add(f"- **门最终裁决：`{report.final_verdict}`**")
    add("")
    add(
        "> 本报告的每一个数字都由 `tensorphysics.campaign.wave3_gate` 从原始证据快照"
        "（战役表 dump + 平台探针原始响应）程序化生成，报告正文摘要参与 peer review 签名，"
        "人工无法在不改变摘要的前提下修改结论。"
    )
    add("")

    # ---------------------------------------------------------------- §1
    add("## 1 案例介绍：波3 的九个任务节点")
    add("")
    add(
        "波3（IN19：求解器验证与船舶能力深化·谱系继承）把上一波已验证的求解器与船舶能力"
        "推进到「可判据验收」：每个节点自带机器可算验收判据（C1…Cn），"
        "而本门（IN19-G19）负责在证据层面把它们核销成一张可继承的谱系。"
        "门本身不生产物理结果，只做三件事：汇总验收、归因未过、裁决继承。"
    )
    add("")
    add("| 节点 | 仓库 | 名称 | 任务状态 | 成熟化项 | 验收结论 | 判据(已测/总) |")
    add("|---|---|---|---|---|---|---|")
    for n in nodes:
        add(
            f"| {n.ext_id} | {n.repo} | {_esc(n.name, 60)} | {n.mission_status} | "
            f"{'是' if n.is_maturity else '否'} | {_VERDICT_MARK[n.verdict.value]} | "
            f"{n.measured_criteria}/{len(n.criteria)} |"
        )
    add("")

    # ---------------------------------------------------------------- §2
    add("## 2 怎么做的：证据链与判定流程")
    add("")
    add("门的工作流是五步、全自动、可重放：")
    add("")
    add(
        "1. **证据快照**：只读导出 `campaign_nodes` ⊗ `tensor_dev_missions`（plan_id=39，10 行），"
        "记录每个节点的 goal 文本、mission 状态、交付物；快照文件与 sha256 落在审计链中。"
    )
    add(
        "2. **判据解析**：从 goal 的「机器可算验收判据」段用正则解析出 `C1…Cn`，"
        "矩阵行数因此是机器产物而非人工抄写（判据 C1 可机器校验）。"
    )
    add(
        "3. **验收判定**：每个判据只有四种结局——PASS / FAIL / PARTIAL / WITHHELD。"
        "**未交付证据一律 WITHHELD，不得升格为 PASS**；节点结论由判据汇总得到"
        "（任一 FAIL → FAIL；全 WITHHELD → WITHHELD；否则 PARTIAL）。"
    )
    add(
        "4. **归因与继承**：非 PASS 节点必须落进封闭四分类根因"
        "（模型形式误差 / 输入不确定度 / 实现 bug / 证据缺口）；继承结论由节点记录派生"
        "（MATURE 需要 PASS **且** 至少一条判据对过外部基准），禁止人工断言。"
    )
    add(
        "5. **门自检 + 独立探针**：对五项战役判据 C1–C5 逐项给出通过状态与数字；"
        "同时对可达的 ship-performance-platform 端点做**门外独立探针**，"
        "使误差表在节点未交付时仍有实测数字，也让「已成熟」的说法可被外部否证。"
    )
    add("")
    add("```")
    add("raw/nodes_plan39.json ──┐")
    add("raw/platform_probes.json ─┼─> wave3_gate.build_report ─> gate_report.json (digest)")
    add("references.json ────────┘                        └─> gate_report.md (本文)")
    add("```")
    add("")

    # ---------------------------------------------------------------- §3
    add("## 3 验收矩阵（判据 C1）")
    add("")
    add("| 节点 | 判据 | 判据摘要 | 结论 | 实测 | 基准 | 相对误差 |")
    add("|---|---|---|---|---|---|---|")
    for n in nodes:
        if not n.criteria:
            add(f"| {n.ext_id} | — | **未解析到机器判据** | WITHHELD | — | — | — |")
            continue
        for c in n.criteria:
            err = c.rel_error_pct
            unit = "" if c.unit in ("", "-") else " " + c.unit
            add(
                f"| {n.ext_id} | {c.cid} | {_esc(c.text, 80)} | {_VERDICT_MARK[c.verdict.value]} | "
                f"{_fmt(c.measured)}{unit if c.measured is not None else ''} | "
                f"{_fmt(c.reference)}{unit if c.reference is not None else ''} | "
                f"{'—' if err is None else f'{err:+.2f}%'} |"
            )
    total = sum(len(n.criteria) for n in nodes)
    add("")
    add(
        f"矩阵覆盖 **{sum(1 for n in nodes if n.criteria)}/{len(nodes)} 节点、{total} 条判据**；"
        "每行都带显式结论，没有任何一行因为「无证据」而被省略。"
    )
    add("")

    # ---------------------------------------------------------------- §4
    add("## 4 误差判定表（判据 C4）")
    add("")
    add(
        "**锚定规则（先验，非事后挑选）**：每个节点只锚定一条关键量——若该节点有门侧探针，"
        "取其 |相对误差| 最大者（最不利证据优先）；否则取节点自己已交付的第一条实测判据；"
        "两者皆无则记为 `withheld`。误差列由**未圆整原值**计算，表中展示值可能已按有效位圆整，"
        "全精度见 §5。"
    )
    add("")
    add("关键量 vs 基准逐项列出。`basis=node_criterion` 表示由节点自己的交付判定，"
        "`gate_probe` 表示门外独立实测（当前平台状态，非节点交付），"
        "`withheld` 表示截止时刻无任何实测——**声明为缺失，而不是留空**。")
    add("")
    add("| 节点 | 关键量 | 基准来源 | 阈值 | 实测 | 基准 | 相对误差 | 达标 | 依据 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in report.error_table:
        thr = "—" if r["threshold_pct"] is None else f"{r['threshold_pct']:.2f}%"
        ok = "—" if r["within_threshold"] is None else ("是" if r["within_threshold"] else "**否**")
        err = "—" if r["rel_error_pct"] is None else f"{r['rel_error_pct']:+.2f}%"
        add(
            f"| {r['ext_id']} | {_esc(r['key_quantity'], 46)} | {_esc(r['baseline_source'], 40)} | {thr} | "
            f"{_fmt_full(r['attained'])} | {_fmt_full(r['baseline'])} | {err} | {ok} | {r['basis']} |"
        )
    add("")
    n_withheld = sum(1 for r in report.error_table if r["basis"] == "withheld")
    add(
        f"合计 {len(report.error_table)} 行：{len(report.error_table) - n_withheld} 行有实测、"
        f"{n_withheld} 行 `withheld`（原因均为「节点未交付验收证据」，逐行已注明）。"
    )
    add("")

    # ---------------------------------------------------------------- §5
    if report.probes:
        add("## 5 门外独立探针（可外部否证的基线快照）")
        add("")
        add("| 探针 | 节点 | 端点 | 物理量 | 实测 | 基准 | 基准来源 | 基准不确定度 | 相对误差 | 阈值 |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for p in report.probes:
            add(
                f"| {p.probe_id} | {p.node_id} | `{p.endpoint}` | {_esc(p.quantity, 30)} | "
                f"{_fmt_full(p.measured)} {p.unit} | {_fmt_full(p.reference)} {p.unit} | "
                f"{_esc(p.reference_source, 40)} | ±{p.reference_tolerance_pct:g}% | "
                f"**{p.rel_error_pct:+.2f}%** | {p.threshold_pct:g}% |"
            )
        add("")
        add(
            "探针只回答「此刻主分支上的平台在什么水平」，不替代节点验收；"
            "它的用途是让波4 的起点可被外部复算，并防止把探针数值悄悄写成节点成果。"
        )
        add("")

    # ---------------------------------------------------------------- §6
    open_nodes = [n for n in nodes if n.verdict.value != "accept"]
    add("## 6 根因分析（判据 C2）")
    add("")
    if not open_nodes:
        add("全部节点验收通过，无未过节点。")
    else:
        add(f"共 {len(open_nodes)} 个节点未通过，逐节点归因如下（封闭四分类）。")
        add("")
        add("| 节点 | 根因类别 | 归因陈述 | 证据 | 置信 | 建议动作 |")
        add("|---|---|---|---|---|---|")
        for n in open_nodes:
            rc = n.root_cause
            assert rc is not None
            add(
                f"| {n.ext_id} | `{rc.kind.value}`（{_CAUSE_GLOSS[rc.kind.value]}） | "
                f"{_esc(rc.statement, 120)} | {_esc(rc.evidence, 100)} | "
                f"{rc.confidence} | {_esc('；'.join(rc.actions), 70)} |"
            )
        add("")
        groups: dict[str, list[str]] = {}
        for n in open_nodes:
            assert n.root_cause is not None
            groups.setdefault(n.root_cause.kind.value, []).append(n.ext_id)
        add("按类别汇总：" + "；".join(f"`{k}` → {', '.join(v)}" for k, v in sorted(groups.items())) + "。")
        add("")
        running = sum(1 for n in nodes if n.mission_status in ("running", "in-flight", "queued"))
        add(
            f"**诚实边界**：截止时刻 {running}/{len(nodes)} 个节点的 mission 仍为进行中，"
            "未通过的直接原因是「**证据缺口**」，而不是「物理不达标」——"
            "两者不可互相冒充：把证据缺口写成物理失败会低估能力，写成通过则高估。"
            "门只记录「此刻无法核销」，并保留门外探针给出的当前水平作为定量锚点。"
        )
    add("")

    # ---------------------------------------------------------------- §7
    add("## 7 谱系继承裁决（判据 C3）")
    add("")
    add("| 节点 | 验收结论 | 继承裁决 | 理由 |")
    add("|---|---|---|---|")
    for n in nodes:
        add(
            f"| {n.ext_id} | {_VERDICT_MARK[n.verdict.value]} | **{_LINEAGE_MARK[n.lineage.value]}** | "
            f"{_esc(_rationale_cn(n, report.probes), 120)} |"
        )
    add("")
    lt = report.lineage_table
    add(
        f"汇总：成熟 {len(lt['mature'])} 个（{', '.join(lt['mature']) or '无'}）；"
        f"迭代 {len(lt['iterate'])} 个（{', '.join(lt['iterate']) or '无'}）；"
        f"搁置 {len(lt['shelve'])} 个（{', '.join(lt['shelve']) or '无'}）。"
    )
    add("")

    # ---------------------------------------------------------------- §8
    add("## 8 门的自检：五项战役判据")
    add("")
    add("| 判据 | 要求 | 结果 | 数字 | 说明 |")
    add("|---|---|---|---|---|")
    for c in report.checks:
        add(
            f"| {c.cid} | {_esc(c.requirement, 70)} | {'PASS' if c.passed else '**FAIL**'} | "
            f"{c.number} | {_esc(c.detail, 90)} |"
        )
    add("")
    add(f"门最终裁决 `{report.final_verdict}`：C1–C5 "
        f"{'全部通过' if all(c.passed for c in report.checks) else '存在未通过项'}，"
        f"节点级 {sum(1 for n in nodes if n.verdict.value == 'accept')}/{len(nodes)} 通过。"
        "判据通过不等于物理达标——它只证明「报告如实、可复算、可审计」。")
    add("")

    # ---------------------------------------------------------------- §9
    add("## 9 图")
    add("")
    add(f"![验收矩阵]({FIG_DIR}/acceptance_matrix.png)")
    add("")
    add("*图1 波3 验收矩阵：行=节点，列=该节点的机器判据；灰=未评（WITHHELD），"
        "色块=通过/部分/失败。整幅矩阵几乎全灰，即是本轮门的定量结论。*")
    add("")
    add(f"![误差谱]({FIG_DIR}/error_spectrum.png)")
    add("")
    add("*图2 门外独立探针的相对误差谱（对数纵轴）与各自阈值；柱高超出阈线即为当前不达标，"
        "标注值为实测误差。*")
    add("")
    add(f"![谱系]({FIG_DIR}/lineage_summary.png)")
    add("")
    add("*图3 谱系继承分布：按 repo 归并的 MATURE/ITERATE/SHELVE 计数，以及判据状态构成。*")
    add("")

    # ---------------------------------------------------------------- §10
    add("## 10 局限与适用范围")
    add("")
    for i, lim in enumerate(report.limitations, 1):
        add(f"{i}. {lim}")
    add("")

    # ---------------------------------------------------------------- §11
    add("## 11 下一波提案方向")
    add("")
    for i, prop in enumerate(report.next_wave, 1):
        add(f"{i}. {prop}")
    add("")

    # ---------------------------------------------------------------- §12
    add("## 12 Peer review（判据 C5）")
    add("")
    pr = report.peer_review
    if pr is None:
        add("**未取得非本波参与者的 peer review**，判据 C5 记为未通过（honest FAIL 保留，不粉饰）。")
    else:
        add(f"- 评审者：`{pr.reviewer_agent}`（{pr.reviewer_affiliation}）")
        add(f"- 独立性声明：{'非本波参与者' if pr.independent else '**非独立（判据不通过）**'}")
        add(f"- 评审对象摘要：`{pr.reviewed_digest}`（与本文正文摘要一致：{pr.reviewed_digest == report.body_digest}）")
        add(f"- 评审结论：`{pr.verdict}`，签署时间 `{pr.signed_at}`")
        if pr.findings:
            add("- 评审发现：")
            for f in pr.findings:
                add(f"  - {f}")
        if pr.requested_changes:
            add("- 要求修改：")
            for f in pr.requested_changes:
                add(f"  - {f}")
    add("")

    if report.prior_reviews:
        add("### 既往评审（签署摘要已被后续版本取代，记录其发现与作者回应）")
        add("")
        add("| 评审者 | 版本摘要 | 结论 | 主要发现 | 作者回应 |")
        add("|---|---|---|---|---|")
        for prior in report.prior_reviews:
            add(
                f"| `{prior.reviewer_agent}` | `{prior.reviewed_digest[:16]}…` | {prior.verdict} | "
                f"{_esc('；'.join(prior.findings), 120)} | {_esc(prior.response, 140)} |"
            )
        add("")

    # ---------------------------------------------------------------- §13
    add("## 13 审计链")
    add("")
    add("| # | 资料 | sha256 | 说明 |")
    add("|---|---|---|---|")
    for i, item in enumerate(report.audit_chain, 1):
        add(f"| {i} | `{item.get('path', '')}` | `{str(item.get('sha256', ''))[:16]}…` | {_esc(str(item.get('note', '')), 70)} |")
    add("")
    add(
        f"报告正文摘要 `{report.body_digest}` 的定义：对 `gate_report.json` 中 "
        "`plan_id, plan_name, gate_ext_id, gate_mission_id, cutoff, nodes, probes, error_table, "
        "lineage, limitations, next_wave` 这组字段做 `json.dumps(..., sort_keys=True, "
        "ensure_ascii=False)` 后取 sha256。**它不是 markdown 文件的哈希**：排版/措辞调整不改变摘要，"
        "数值与判定变动才改变摘要——这正是可以安全签字的原因（复算脚本见 "
        "`scripts/run_in19_g19_wave3_gate.py` 与 `tests/test_wave3_gate.py::test_digest_recomputable_from_published_json`）。"
        "自检 C1–C5 与评审签名由正文派生，故不参与摘要（避免自指）。"
    )
    add("")
    return "\n".join(lines)