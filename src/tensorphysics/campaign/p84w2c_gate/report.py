"""Markdown renderer for the P84-W2C campaign gate (six-element paper layout).

Every number in the report is interpolated from the verdict dict; there is no
hand-typed value anywhere, so the document cannot drift away from the JSON it
describes.  The six required elements are:

1. 案例介绍 (campaign identity, physics scope, why a gate)
2. 怎么做的 (criteria, bilateral contract method, toolchain, reproduce commands)
3. 计算结果 (per-node tables, campaign criteria numbers, contract checks)
4. 结果图 (figure reference + what each panel encodes)
5. 注意点 (pitfalls, limitations, honest-FAIL semantics)
6. 误差判定 (per-criterion error vs threshold with an explicit "<3%?" answer)
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

SECTIONS = (
    "## 1. 案例介绍",
    "## 2. 怎么做的",
    "## 3. 计算结果",
    "## 4. 结果图",
    "## 5. 注意点",
    "## 6. 误差判定",
)

_STATUS_MARK = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WITHHELD": "⚪ WITHHELD",
                "NOT_SUPPORTED": "⛔ NOT_SUPPORTED"}


def _mark(status: str) -> str:
    return _STATUS_MARK.get(status, status)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        magnitude = abs(float(value))
        if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
            return f"{value:.3e}"
        return f"{value:.{digits}g}"
    return str(value)


def _threshold_text(crit: Mapping[str, Any]) -> str:
    threshold = crit.get("threshold")
    if isinstance(threshold, list):
        return f"[{threshold[0]}, {threshold[1]}]"
    return _fmt(threshold)


def _error_line(crit: Mapping[str, Any]) -> tuple[str, str]:
    """Return (error text, explicit '<3%?' answer)."""
    value = crit.get("value")
    if value is None:
        return "无数字证据（判据 WITHHELD）", "无法判定（未交付数字）"
    comparator = crit.get("comparator")
    threshold = crit.get("threshold")
    if comparator == "rel_dev_le":
        ref = threshold if threshold else None
        # the reference is stored in the spec, re-derive from detail when absent
        detail = str(crit.get("detail", ""))
        rel = None
        if "=" in detail:
            try:
                rel = float(detail.split("=")[-1].strip().split()[0])
            except (ValueError, IndexError):
                rel = None
        if rel is None and ref:
            rel = abs(float(value) - float(ref)) / abs(float(ref))
        if rel is None:
            return "相对偏差不可解析", "无法判定"
        return f"相对偏差 {rel:.3e}（阈 {_threshold_text(crit)}）", ("是" if rel < 0.03 else "否")
    if comparator == "in_range":
        return f"值 {_fmt(value)} ∈ {_threshold_text(crit)}", "无独立基准（区间判据）"
    if comparator in ("le", "ge"):
        return f"|值| = {_fmt(abs(float(value)))} vs 阈 {_threshold_text(crit)}", \
            ("无独立基准（阈判据）" if abs(float(value)) >= 0.03 else "是（量级 <3%）")
    return "布尔负控", "无独立基准（布尔判据）"


def _node_table(node: Mapping[str, Any]) -> list[str]:
    lines = [
        f"#### {node['node_key']} · {node['title']}",
        "",
        f"- 仓库：`{node['repo']}` · 物理域：{node['physics_domain']} · mission={node['mission_id']}"
        f" · 平台状态：`{node.get('platform_status') or 'n/a'}`",
        f"- 节点裁决：**{_mark(node['verdict'])}** · 产物家族 {len(node['artifacts'])} 个"
        f"（可解析 {sum(1 for a in node['artifacts'] if a.get('parseable'))}）"
        f" · pytest `{node['pytest'].get('status')}`"
        f"（mode={node['pytest'].get('mode')}, collected={node['pytest'].get('collected', 0)}, "
        f"passed={node['pytest'].get('passed', 0)}, failed={node['pytest'].get('failed', 0)}, "
        f"errors={node['pytest'].get('errors', 0)}）",
        "",
        "| 判据 | 陈述 | 值 | 阈值 | 裁决 |",
        "|---|---|---|---|---|",
    ]
    for crit in node.get("criteria", []):
        lines.append(
            f"| {crit['id']} | {crit['statement']} | {_fmt(crit.get('value'))} | "
            f"{_threshold_text(crit)} | {_mark(crit['status'])} |")
    if node.get("artifacts"):
        lines += ["", "| 产物 | 类型 | 可解析 | sha256(前16) |", "|---|---|---|---|"]
        for artifact in node["artifacts"]:
            lines.append(f"| `{artifact['path']}` | {artifact['kind']} | "
                         f"{artifact['parseable']} | `{str(artifact.get('sha256', ''))[:16]}` |")
    if node.get("gaps"):
        lines += ["", "未通过项 / gap："] + [f"- {gap}" for gap in node["gaps"]]
    lines.append("")
    return lines


def render_markdown(report: Mapping[str, Any],
                    figure_rel: str | None = None,
                    reproduce: Sequence[str] = ()) -> str:
    nodes = report.get("nodes", [])
    contracts = report.get("contracts", [])
    criteria = report.get("criteria", [])
    lineage_summary = report.get("lineage_summary", {})
    gaps = report.get("gap_analysis", [])

    totals = {
        "nodes": len(nodes),
        "node_pass": sum(1 for n in nodes if n["verdict"] == "PASS"),
        "node_fail": sum(1 for n in nodes if n["verdict"] == "FAIL"),
        "node_withheld": sum(1 for n in nodes if n["verdict"] in ("WITHHELD", "NOT_SUPPORTED")),
        "crit_total": sum(len(n.get("criteria", [])) for n in nodes),
        "crit_pass": sum(1 for n in nodes for c in n.get("criteria", []) if c["status"] == "PASS"),
        "crit_fail": sum(1 for n in nodes for c in n.get("criteria", []) if c["status"] == "FAIL"),
        "crit_withheld": sum(1 for n in nodes for c in n.get("criteria", [])
                             if c["status"] == "WITHHELD"),
        "artifacts": sum(len(n.get("artifacts", [])) for n in nodes),
        "artifacts_ok": sum(1 for n in nodes for a in n.get("artifacts", []) if a.get("parseable")),
        "suite_executed": sum(1 for n in nodes if n.get("pytest", {}).get("mode") == "executed"),
        "suite_green": sum(1 for n in nodes if n.get("pytest", {}).get("status") == "PASS"),
        "contract_pass": sum(1 for c in contracts if c["status"] == "PASS"),
        "contract_total": len(contracts),
        "checks": sum(len(c.get("checks", [])) for c in contracts),
        "checks_pass": sum(1 for c in contracts for ch in c.get("checks", [])
                           if ch["status"] == "PASS"),
    }

    out: list[str] = []
    out.append(f"# P84-W2C 战役集成验收门（{report.get('gate_id', 'P84W2C-N11')}）——"
               f"全节点验收汇总与跨域接口互裁")
    out.append("")
    out.append(f"- 产物 schema：`{report['schema_name']}` v{report['schema_version']}"
               f" → fingerprint `{report['fingerprint']}`")
    out.append(f"- 战役计划：plan_id={report['plan_id']} · {report['plan_title']}")
    out.append(f"- 生成时间（UTC）：`{report['generated_utc']}`")
    out.append(f"- **总裁决：`{report['overall_status']}` · 阻塞项 {report['blocker_count']} 条**")
    out.append("")

    # ---------------- 1 ----------------
    out.append(SECTIONS[0])
    out.append("")
    out.append("P84-W2C 是谱系继承波（Wave 2C）：十个并行研究节点覆盖求解器验证前沿"
               "（TensorLBM 翼型气动、LES 湍流、悬沙冲刷；TensorBEM 阴极保护；TensorDEM 颗粒卸料）"
               "与船舶工程成熟化（岸电排放折减、ORC 余热回收 EEDI credit、货物系固、进坞墩木、"
               "谷物稳性）。节点之间不共享代码，只通过**产物**消费彼此结果，因此「战役是否可交付」"
               "不能由 PR 是否合并判断，只能由「产物是否落盘、pytest 是否可解析且绿、跨域接口"
               "是否真的对得上」判断。")
    out.append("")
    out.append(f"本门汇总 {totals['nodes']} 个节点的验收结果：节点裁决 PASS {totals['node_pass']} / "
               f"FAIL {totals['node_fail']} / WITHHELD {totals['node_withheld']}；"
               f"节点级判据 {totals['crit_total']} 条中 PASS {totals['crit_pass']}、"
               f"FAIL {totals['crit_fail']}、WITHHELD {totals['crit_withheld']}。"
               f"产物 {totals['artifacts']} 个、可解析 {totals['artifacts_ok']} 个；"
               f"pytest 套件执行 {totals['suite_executed']} 个、全绿 {totals['suite_green']} 个。")
    out.append("")
    out.append("| 节点 | 仓库 | 物理域 | 节点裁决 | 平台状态 | 判据通过 |")
    out.append("|---|---|---|---|---|---|")
    for node in nodes:
        passed = sum(1 for c in node.get("criteria", []) if c["status"] == "PASS")
        out.append(f"| {node['node_key']} | `{node['repo']}` | {node['physics_domain']} | "
                   f"{_mark(node['verdict'])} | `{node.get('platform_status') or 'n/a'}` | "
                   f"{passed}/{len(node.get('criteria', []))} |")
    out.append("")
    out.append("### 1.1 谱系继承裁决")
    out.append("")
    out.append(f"父计划 plan {lineage_summary.get('parent_plan_id')} 的同名节点被逐条对表："
               f"`inherited_verified` {lineage_summary.get('inherited_verified')} 个、"
               f"`inherited_unverified` {lineage_summary.get('inherited_unverified')} 个、"
               f"`seeded_new` {lineage_summary.get('seeded_new')} 个。"
               f"判定语义：{lineage_summary.get('note', '')}。")
    out.append("")
    out.append("| 节点 | 父节点 | 父 mission | 父状态 | 谱系裁决 |")
    out.append("|---|---|---|---|---|")
    for node in nodes:
        lin = node.get("lineage", {})
        out.append(f"| {node['node_key']} | {lin.get('parent_node') or '—'} | "
                   f"{lin.get('parent_mission_id') or '—'} | {lin.get('parent_status') or '—'} | "
                   f"`{lin.get('verdict')}` |")
    out.append("")

    # ---------------- 2 ----------------
    out.append(SECTIONS[1])
    out.append("")
    out.append("判据实现在 `src/tensorphysics/campaign/p84w2c_gate/`（stdlib-only，不 import 任何"
               "求解器栈，也不 import 消费方运行时）：`spec.py` 只声明节点/判据/契约，"
               "`collect.py` 采集磁盘证据并**亲自执行**节点的 pytest 套件（junit XML → 归一化 JSON），"
               "`contracts.py` 用 `ast` 逐侧检查消费方声明并做数值锚对比，`criteria.py` 判 C1–C4，"
               "`lineage.py` 做谱系继承裁决。")
    out.append("")
    out.append("| 战役判据 | 内容 | 判据数字 | 裁决 |")
    out.append("|---|---|---|---|")
    metric_text = {
        "C1": (f"pytest JSON 可解析 {criteria[0]['metrics'].get('pytest_json_parseable') if criteria else '—'}"
               f"/{criteria[0]['metrics'].get('nodes') if criteria else '—'}；"
               f"执行 {criteria[0]['metrics'].get('pytest_executed') if criteria else '—'}；"
               f"全绿 {criteria[0]['metrics'].get('pytest_green') if criteria else '—'}"),
        "C2": (f"必需契约 {next((c['metrics'].get('required_passed') for c in criteria if c['id'] == 'C2'), '—')}"
               f"/{next((c['metrics'].get('required_contracts') for c in criteria if c['id'] == 'C2'), '—')} 通过；"
               f"数据级一致性 PASS "
               f"{next((c['metrics'].get('data_conformance_pass') for c in criteria if c['id'] == 'C2'), '—')}"),
        # the character count is self-referential (writing it changes the length),
        # so the number lives in the JSON field ``markdown.chars`` only.
        "C3": (f"字符数见 JSON markdown.chars（下限 "
               f"{next((c['metrics'].get('min_chars') for c in criteria if c['id'] == 'C3'), '—')}）；"
               f"章节 "
               f"{next((c['metrics'].get('sections_found') for c in criteria if c['id'] == 'C3'), '—')}"
               f"/{next((c['metrics'].get('sections_required') for c in criteria if c['id'] == 'C3'), '—')}；"
               f"图件 {next((c['metrics'].get('figure_present') for c in criteria if c['id'] == 'C3'), '—')}"),
        "C4": (f"裁决 JSON schema 指纹 {next((c['metrics'].get('schema_fingerprint') for c in criteria if c['id'] == 'C4'), '—')}；"
               f"节点 {next((c['metrics'].get('nodes') for c in criteria if c['id'] == 'C4'), '—')}、"
               f"判据 {next((c['metrics'].get('criteria_total') for c in criteria if c['id'] == 'C4'), '—')}"),
    }
    for crit in criteria:
        out.append(f"| {crit['id']} | {crit['statement']} | {metric_text.get(crit['id'], '')} | "
                   f"{_mark(crit['status'])} |")
    out.append("")
    if reproduce:
        out.append("复现命令（证据全部落在 `artifacts/p84w2c_gate/`）：")
        out.append("")
        out.append("```bash")
        out.extend(reproduce)
        out.append("```")
        out.append("")

    # ---------------- 3 ----------------
    out.append(SECTIONS[2])
    out.append("")
    out.append("### 3.1 战役判据裁决与数字")
    out.append("")
    out.append("| 判据 | 陈述 | 裁决 | 判据数字 |")
    out.append("|---|---|---|---|")
    for crit in criteria:
        out.append(f"| {crit['id']} | {crit['statement']} | {_mark(crit['status'])} | "
                   f"{metric_text.get(crit['id'], '')} |")
    out.append("")
    out.append("### 3.2 跨域接口契约（双侧验证）")
    out.append("")
    out.append("| 契约 | 生产侧节点 | 生产 schema | 消费侧 | 契约裁决 | 数据级一致性 | 检查项通过 |")
    out.append("|---|---|---|---|---|---|---|")
    for contract in contracts:
        ok = sum(1 for ch in contract["checks"] if ch["status"] == "PASS")
        out.append(f"| `{contract['contract_id']}` | {contract['producer_node']} | "
                   f"`{contract['producer_schema']}` | `{contract['consumer']}` | "
                   f"{_mark(contract['status'])} | `{contract['data_conformance']}` | "
                   f"{ok}/{len(contract['checks'])} |")
    out.append("")
    for contract in contracts:
        out.append(f"#### 契约 `{contract['contract_id']}` 检查明细（指纹 "
                   f"`{contract['fingerprint'][:16]}`）")
        out.append("")
        out.append("| 检查 | 裁决 | 数值 | 阈值 | 说明 |")
        out.append("|---|---|---|---|---|")
        for check in contract["checks"]:
            out.append(f"| {check['name']} | {_mark(check['status'])} | "
                       f"{_fmt(check.get('value'))} | {_fmt(check.get('threshold'))} | "
                       f"{check.get('detail', '')} |")
        if contract.get("gaps"):
            out += ["", "缺口："] + [f"- {gap}" for gap in contract["gaps"]]
        out.append("")
    out.append("### 3.3 逐节点验收明细")
    out.append("")
    for node in nodes:
        out += _node_table(node)

    # ---------------- 4 ----------------
    out.append(SECTIONS[3])
    out.append("")
    if figure_rel:
        out.append(f"![P84-W2C gate matrix]({figure_rel})")
        out.append("")
        out.append("左图：十个节点各自声明的验收判据按 PASS/FAIL/WITHHELD 堆叠，直接暴露每个节点的"
                   "「黑箱程度」——灰色段越长说明该节点交付的数字越少。右图：战役级证据覆盖率，"
                   "四根柱分别是「产物家族可解析率」「pytest 套件执行且全绿率」「跨域契约通过率」"
                   "「战役判据 C1–C4 通过率」，全部由本 JSON 的计数现算，没有手抄数字。")
    else:
        out.append("本门图件未生成（matplotlib 不可用或门被要求不产图），原因见审计链 `audit.figure`；"
                   "这是纯证据汇总型门禁的已知局限：判据数字仍完整保存在 JSON 中，"
                   "任何图件缺失都不影响 C1–C4 的机器裁决。")
    out.append("")

    # ---------------- 5 ----------------
    out.append(SECTIONS[4])
    out.append("")
    out.append("- **honest-FAIL 语义**：任一节点未交付即计入 gap 分析，`WITHHELD` 永不等于 PASS；"
               "因此本门在节点在飞阶段必然给出 FAIL，这不是门坏了，而是战役尚未可交付。")
    out.append("- **只在磁盘上取证**：门不跑节点的物理算例、不重算流场，只解析产物并执行节点自己的"
               "pytest 套件；节点若在报告里自称达标却未落盘 JSON，判据一律 WITHHELD。")
    out.append("- **外部基准由节点自报**：Abbott 表值、Soulsby–Whitehouse 曲线、DNV 电网因子等"
               "外部表格无法在本仓复算，判据要求节点在产物中显式给出 deviation 字段（来源写进 "
               "`source`），门只复核该数字是否越过阈值——这是「自报 + 双口径」的折中，局限性已记录。")
    out.append("- **消费方静态检查而非运行**：消费侧（ship-platform）与生产侧不同运行时，门用 "
               "`ast` 检查类字段、模块常量与动压定律 AST 形状，并做数值锚对比；它证明两侧契约一致，"
               "但不证明消费方端到端 API 调用成功（该证据应由消费方自己的集成测试提供）。")
    out.append("- **pytest 时间上限**：节点套件在门内执行时有超时上限，超时记为 FAIL(mode=executed,"
               "exit_code=124) 而不是静默跳过；大算例应下沉到集群并把 JSON 结果落盘。")
    out.append("- **谱系裁决依赖平台证据**：父计划节点状态来自 `campaign_plans`；离线模式下继承一律"
               "降级为 `inherited_unverified`，不会凭标题同名给 `inherited_verified`。")
    out.append("")

    # ---------------- 6 ----------------
    out.append(SECTIONS[5])
    out.append("")
    out.append("逐项误差与阈值对照如下（键量为节点声明的「关键量 vs 基准」）；"
               "`是否<3%` 列显式回答每个关键量是否落在 3% 以内，"
               "无独立基准的项写明「无独立基准」及其判据类型，不含糊。")
    out.append("")
    out.append("| 节点 | 判据 | 关键量 | 值 | 误差 / 偏差 | 基准/阈值 | 是否<3% | 裁决 |")
    out.append("|---|---|---|---|---|---|---|---|")
    explicit_lt3 = 0
    explicit_ge3 = 0
    for node in nodes:
        for crit in node.get("criteria", []):
            err_text, lt3 = _error_line(crit)
            if lt3 == "是":
                explicit_lt3 += 1
            elif lt3 == "否":
                explicit_ge3 += 1
            out.append(f"| {node['node_key']} | {crit['id']} | {crit['statement']} | "
                       f"{_fmt(crit.get('value'))} | {err_text} | {crit.get('source', '')} | "
                       f"{lt3} | {_mark(crit['status'])} |")
    out.append("")
    out.append(f"汇总：显式落在 3% 以内的关键量 {explicit_lt3} 项，显式超过 3% 的 {explicit_ge3} 项，"
               f"其余 {totals['crit_total'] - explicit_lt3 - explicit_ge3} 项因判据为区间/量级/布尔型"
               f"或数字未交付而不适用 3% 门槛（逐行已注明原因）。"
               f"本门不把「未交付」粉饰成「达标」：{totals['crit_withheld']} 条判据以 WITHHELD 落表。")
    out.append("")

    # ---------------- 7 ----------------
    out.append("## 7. Gap 分析（未通过节点与阻塞项）")
    out.append("")
    out.append(f"阻塞项 {report['blocker_count']} 条，按节点归并如下；每条都给出可执行的补齐动作。")
    out.append("")
    out.append("| 节点 | 严重度 | 缺口 | 补齐动作 |")
    out.append("|---|---|---|---|")
    for gap in gaps:
        out.append(f"| {gap['node_key']} | {gap.get('severity', 'blocker')} | {gap['reason']} | "
                   f"{gap['remediation']} |")
    out.append("")

    # ---------------- 8 ----------------
    out.append("## 8. 审计链与可复现性")
    out.append("")
    out.append(f"- 裁决 JSON：`{report['artifacts_index']['report_json']}` · "
               f"schema 指纹 `{report['artifacts_index'].get('schema_fingerprint', '')}`")
    out.append(f"- 证据目录：`{report['artifacts_index']['evidence_dir']}`（逐节点 pytest 归一化记录）")
    out.append(f"- 契约指纹：" + "、".join(f"`{c['contract_id']}={c['fingerprint'][:16]}`"
                                          for c in contracts))
    audit = report.get("audit", {})
    if audit.get("platform"):
        platform = audit["platform"]
        out.append(f"- 平台证据：available={platform.get('available')} · "
                   f"plan_nodes={len(platform.get('nodes', {}))} · "
                   f"missions={len(platform.get('missions', {}))}")
    if audit.get("package_files"):
        out.append("- 本门实现文件 sha256：" + "；".join(
            f"`{entry['path']}={entry['sha256'][:16]}`" for entry in audit["package_files"]))
    out.append("")
    return "\n".join(out) + "\n"