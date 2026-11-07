"""Markdown rendering of the wave-4 (IN40-G40) plan-closure report.

Six-section paper convention, all numbers injected from the machine report (no
hand-copied value), and every open node is named with its closed-taxonomy root
cause so the limitations section cannot degrade into prose.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .wave3_gate import Lineage, RootCauseKind, Verdict

if TYPE_CHECKING:  # pragma: no cover
    from .wave4_gate import Wave4Report

_VERDICT_MARK = {
    Verdict.ACCEPT.value: "✅ accept",
    Verdict.PARTIAL.value: "🟡 partial",
    Verdict.FAIL.value: "❌ fail",
    Verdict.WITHHELD.value: "⛔ withheld",
}


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _num(value: Any, digits: int = 3, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_closing_report(report: "Wave4Report") -> str:
    acc = report.acceptance
    nodes = report.nodes
    matrix = report.matrix
    lineage = report.lineage_table
    unmet = [g for g in report.guide_gaps if g.unmet]
    open_nodes = [n for n in nodes if n.verdict is not Verdict.ACCEPT]
    trigger = report.proposal_trigger()

    L: list[str] = []
    add = L.append

    add(f"# {report.gate_ext_id} 波4核销门 · Plan 40 收官报告")
    add("")
    add(
        f"- **计划**：plan_id={report.plan_id} · {report.plan_name}  \n"
        f"- **核销门 mission**：{report.gate_mission_id} · 证据截止 `{report.cutoff}`  \n"
        f"- **机器裁决**：`{report.final_verdict}` · 计划状态裁决 `{acc.decision.value}` · "
        f"通过率 **{acc.pass_count}/{acc.total} = {acc.pass_rate:.1%}**（阈值 {report.threshold:.0%}）  \n"
        f"- **报告摘要 digest**：`{report.body_digest}`"
    )
    add("")
    add("> 裁决规则（机器可算，先于结论）：节点通过 = 该节点全部机器判据 `accept`；"
        f"计划 `completed` 需 **通过率 ≥ {report.threshold:.0%} 且无 critical failure**；"
        "缺席证据一律记 `withheld`，不得上抬为通过。")
    add("")

    # ---------------------------------------------------------------- 1. 案例
    add("## 1. 案例介绍：Plan 40 的物理问题与节点构成")
    add("")
    add(
        "Plan 40 是「谱系继承波 2」：10 个节点把船舶总体性能的六大方向（阻力/推进/耐波/操纵/"
        "稳性/结构振动）与跨求解器界面统一压到同一张验收面上。每个节点自带机器判据，"
        "核销门的职责不是做物理，而是**判定这些物理结论是否已具备可继承的证据**。"
    )
    add("")
    rows = []
    for n in nodes:
        rows.append(
            [
                n.ext_id,
                n.repo,
                str(n.mission_id),
                n.node_status,
                _num(n.reproduction_verdict),
                f"{len(n.criteria)}",
                _VERDICT_MARK.get(n.verdict.value, n.verdict.value),
                n.lineage.value,
            ]
        )
    add("**表 1 · 10 节点构成与验收状态（node_status / reproduction_verdict 取自 campaign_nodes）**")
    add("")
    add(_table(["ext_id", "repo", "mission", "node_status", "repro_verdict", "判据数", "节点裁决", "谱系"], rows))
    add("")
    add("**物理目标（各节点自述，逐节点引原文）**")
    add("")
    for n in nodes:
        add(f"- **{n.ext_id}**（{n.repo}）：{n.goal or '未声明'}")
    add("")

    # ---------------------------------------------------------------- 2. 方法
    add("## 2. 怎么做的：采集/评估分离的核销门")
    add("")
    add(
        "1. **证据快照（collect，只读）**：从平台库只读导出 `campaign_nodes ⊗ tensor_dev_missions`"
        f"（plan_id={report.plan_id}）与 `tensor_guide_tracker`，连同每节点关联报告数，"
        "落盘为 `raw/*.json` 并记录 sha256；不打补丁、不改库。\n"
        "2. **判据解析（机器可算）**：优先读 `campaign_nodes.metadata.acceptance_criteria`，"
        "回退解析 goal 内「机器可算验收判据」段的 `C1…` / `(1)…` 编号——判据**解析得到**而非人工摘抄，"
        "这是「矩阵覆盖」可被机器裁的前提。\n"
        "3. **评估（evaluate，纯函数）**：由快照 JSON 重算节点裁决 → 谱系继承 → 计划裁决，"
        "期间无网络、无数据库、无时钟依赖，结果字节可复现。\n"
        "4. **取证分级**：`REAL/MOCK/STUB/WITHHELD/NOT_SUPPORTED`；无提交 → 判据 `withheld`，"
        "节点级 `evidence_grade=WITHHELD`。\n"
        f"5. **裁决线**：节点全判据 accept → 节点 accept；通过率 ≥ {report.threshold:.0%} 且无 critical "
        "failure → plan `completed` 并触发下一波提案，否则 plan 保持 `active`。"
    )
    add("")
    add("**审计链**")
    add("")
    add(_table(["step", "artifact", "sha256/值"], [[str(a.get("step")), str(a.get("artifact")), str(a.get("value"))] for a in report.audit_chain]) or "（无）")
    add("")

    # ---------------------------------------------------------------- 3. 结果
    add("## 3. 计算结果：10 节点 × 判据验收矩阵")
    add("")
    n_nodes, n_crit = report.matrix_shape
    add(f"矩阵规模 **{n_nodes} 节点 × {n_crit} 条判据**（每条判据带显式裁决，缺席记 `withheld`）。")
    add("")
    rows = []
    for m in matrix:
        rows.append(
            [
                m["ext_id"],
                f"{m['criteria_accepted']}/{m['criteria_total']}",
                f"{m['criteria_measured']}",
                _VERDICT_MARK.get(m["node_verdict"], m["node_verdict"]),
                m["evidence_grade"],
                "yes" if m["prerequisites_ok"] else "no",
                "yes" if m["is_critical_failure"] else "no",
            ]
        )
    add("**表 2 · 节点级验收汇总（判据通过数 / 已测判据 / 证据等级 / 前置条件 / critical）**")
    add("")
    add(_table(["ext_id", "判据通过", "已测判据", "节点裁决", "证据等级", "前置 done+pass", "critical failure"], rows))
    add("")
    add("**逐判据明细（每个节点）**")
    add("")
    for m in matrix:
        add(f"<details><summary>{m['ext_id']} · {m['node_verdict']} · {m['criteria_accepted']}/{m['criteria_total']}</summary>")
        add("")
        add(_table(
            ["cid", "判据原文", "裁决", "measured", "reference", "相对误差%", "证据"],
            [
                [
                    c["cid"],
                    c["text"],
                    _VERDICT_MARK.get(c["verdict"], c["verdict"]),
                    _num(c["measured"]),
                    _num(c["reference"]),
                    _num(c["rel_error_pct"], 3),
                    c["evidence"] or "—",
                ]
                for c in m["rows"]
            ],
        ))
        add("")
        add("</details>")
        add("")
    add("**计划级裁决数字**")
    add("")
    add(_table(
        ["量", "值"],
        [
            ["通过节点数", f"{acc.pass_count}/{acc.total}"],
            ["通过率 P", f"{acc.pass_rate:.4f}"],
            ["阈值", f"{report.threshold:.2f}"],
            ["critical failure", ", ".join(acc.critical) or "none"],
            ["前置条件满足 (done+pass)", str(report.prerequisites["count"])],
            ["计划裁决", f"`{acc.decision.value}`"],
            ["下一波提案触发", "yes" if acc.next_wave_triggered else "no"],
            ["裁决理由", acc.reason],
        ],
    ))
    add("")
    add("**机器自检（6 项）**")
    add("")
    add(_table(
        ["cid", "要求", "通过", "数字", "备注"],
        [[c.cid, c.requirement, "✅" if c.passed else "❌", c.number, c.detail] for c in report.checks],
    ))
    add("")

    # ---------------------------------------------------------------- 4. 图
    add("## 4. 结果图")
    add("")
    if report.figures:
        for fig in report.figures:
            name = fig.get("name", "figure")
            path = fig.get("path", "")
            caption = fig.get("caption", "")
            add(f"**图 · {name}**  \n![{name}]({path})  \n{caption}")
            add("")
    else:
        add("（本报告无图：该节点为聚合裁决类，物理场量由各任务节点自出；"
            "本节保留图谱位以便下一波补齐。）")
        add("")

    # ---------------------------------------------------------------- 5. 误差判定
    add("## 5. 误差判定：关键量 vs 基准（逐项显式回答是否 <3%）")
    add("")
    add(
        "误差判定只在节点递交了 `measured/reference` 时才有数；本轮无节点提交验收 JSON，"
        "因此全部为 `withheld`——**不是「误差合格」，而是「尚无误差可判」**，按纪律不得含糊。"
    )
    add("")
    rows = []
    for r in report.error_table:
        within = r["within_threshold"]
        verdict_3pct = "—" if within is None else ("是" if r["rel_error_pct"] is not None and abs(r["rel_error_pct"]) <= 3.0 else "否")
        rows.append(
            [
                r["ext_id"],
                r["key_quantity"][:40],
                r["baseline_source"][:28],
                _num(r["attained"]),
                _num(r["baseline"]),
                _num(r["rel_error_pct"], 3),
                verdict_3pct,
                r["basis"],
            ]
        )
    add(_table(["ext_id", "关键量", "基准来源", "实测", "基准", "相对误差%", "<3%？", "依据"], rows))
    add("")
    judged = [r for r in report.error_table if r["basis"] != "withheld" and r["rel_error_pct"] is not None]
    within3 = [r for r in judged if abs(r["rel_error_pct"]) <= 3.0]
    if judged:
        add(
            f"**结论**：{len(judged)}/{len(report.error_table)} 个节点具备可判误差，"
            f"其中 {len(within3)} 个 <3%（逐项见上表「<3%？」列）；"
            f"其余 {len(report.error_table) - len(judged)} 个为 `withheld`——不是「误差合格」，是「尚无误差可判」。"
        )
    else:
        add(
            f"**结论**：0/{len(report.error_table)} 个节点具备可判误差；无基准可对的项须声明原因，"
            "本轮原因统一为「节点未递交验收工件（mission 仍在运行或无报告）」——honest FAIL，不粉饰。"
        )
    add("")

    # ---------------------------------------------------------------- 6. 谱系
    add("## 6. 谱系继承裁决")
    add("")
    add(
        "继承是**推导**而非声称：`MATURE` 需节点 `accept` 且至少一条判据对外部基准实测；"
        "`ITERATE` 表示能力仍在门内；`SHELVE` 保留给「已废弃且无证据」的节点。"
    )
    add("")
    add(_table(
        ["谱系", "节点", "含义"],
        [
            [Lineage.MATURE.value, ", ".join(lineage.get("mature", [])) or "—", "可进入下一波基线"],
            [Lineage.ITERATE.value, ", ".join(lineage.get("iterate", [])) or "—", "未过门，须迭代后再裁"],
            [Lineage.SHELVE.value, ", ".join(lineage.get("shelve", [])) or "—", "暂搁，需重建"],
        ],
    ))
    add("")

    # ---------------------------------------------------------------- 7. 局限
    add("## 7. 局限节：未通过节点清单与归因（封闭四分类）")
    add("")
    add(
        "四分类：模型形式误差 `model_form` / 输入不确定度 `input_uncertainty` / 实现缺陷 "
        "`implementation_bug` / 证据缺口 `evidence_gap`。未通过节点逐条列明，不允许「还需努力」这种类。"
    )
    add("")
    if open_nodes:
        add(_table(
            ["ext_id", "节点裁决", "归因类", "陈述", "证据", "置信", "下一步"],
            [
                [
                    n.ext_id,
                    n.verdict.value,
                    n.root_cause.kind.value if n.root_cause else "—",
                    n.root_cause.statement if n.root_cause else "—",
                    n.root_cause.evidence if n.root_cause else "—",
                    n.root_cause.confidence if n.root_cause else "—",
                    "；".join(n.root_cause.actions) if n.root_cause else "—",
                ]
                for n in open_nodes
            ],
        ))
    else:
        add("无未通过节点。")
    add("")
    add("**报告级局限（自述，按纪律显式声明）**")
    add("")
    for line in report.limitations:
        add(f"- {line}")
    add("")

    # ---------------------------------------------------------------- 8. guide
    add("## 8. guide_tracker 缺口（下一波提案原材料）")
    add("")
    add(
        f"`tensor_guide_tracker` 共 {len(report.guide_gaps)} 项指标，{len(unmet)} 项未达标；"
        "progress = metric_value / target_value（target=0 视为已达标）。"
    )
    add("")
    add(_table(
        ["metric_key", "当前值", "目标值", "progress", "缺口", "未达标"],
        [
            [g.metric_key, _num(g.metric_value), _num(g.target_value), f"{g.progress:.4f}", _num(g.gap), "yes" if g.unmet else "no"]
            for g in report.guide_gaps
        ],
    ))
    add("")
    add("**最主要的未覆盖能力（按缺口排序，取前 6）**")
    add("")
    for g in unmet[:6]:
        add(f"- `{g.metric_key}`：{_num(g.metric_value)} / {_num(g.target_value)}（progress {g.progress:.2%}）— {g.notes or '无备注'}")
    add("")

    # ---------------------------------------------------------------- 9. 下一波
    add("## 9. 下一波提案触发与原材料")
    add("")
    add("```json")
    import json as _json

    add(_json.dumps(trigger, ensure_ascii=False, indent=2))
    add("```")
    add("")
    if report.next_wave:
        add("**下一波提案种子（种子清单）**")
        add("")
        for seed in report.next_wave:
            add(f"- {seed}")
        add("")
    add(
        f"**计划状态变更**：`{'已应用' if report.plan_update_applied else '未应用'}`"
        f"（裁决 `{acc.decision.value}`；`wave4_gate.plan_update_sql()` 给出对应语句，"
        "未达门槛时为空操作，计划保持 `active`）。"
    )
    add("")
    add("---")
    add("")
    add(
        f"本报告由 `tensorphysics.campaign.wave4_gate` 程序化生成；node 裁决、通过率、谱系、缺口全部由证据快照重算。"
        "发布回执见文末（render 时刻 publish_status="
        f"`{report.publish_status}`，回执行为权威记录，且不改动证据 digest）。任何把裁决改写得更乐观的版本都会被 "
        "`verify_report_integrity` 拒绝（honest FAIL 不可覆盖）。"
    )
    add("")
    return "\n".join(L)