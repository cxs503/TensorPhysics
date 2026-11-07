"""P20 campaign integration gate: assemble, verify, and report.

The P20 gate evaluates four criteria:

======================  =====================================================
criterion               how it is decided
======================  =====================================================
(a) DAG 收敛             拓扑排序成功 + 所有依赖节点状态为 done
(b) 跨仓接口契约         FSI / 静水力 / 耐波 三组接口 schema 验证
(c) 成熟化单汇总         ≥2 条成熟化单（N07/N08）报告完整
(d) 战役总结报告         六要素完整（案例/方法/结果/图表/注意点/误差）
======================  =====================================================

Overall status is ``PASS`` only when (a)-(d) all pass.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import contracts as C
from . import dag as D
from . import maturity as M

GATE_SCHEMA = "p20_campaign_gate_verdict.v1"
GATE_ID = "P20-G1"
REPORT_TITLE = "P20 全战役集成验收门 (P20-G1)"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


@dataclass
class GateCriterion:
    criterion_id: str
    label: str
    status: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _aggregate(statuses: list[str]) -> str:
    if not statuses:
        return "WITHHELD"
    if "FAIL" in statuses:
        return "FAIL"
    if all(s == "PASS" for s in statuses):
        return "PASS"
    return "WITHHELD"


def load_evidence(evidence_dir: str | Path) -> dict:
    """Load every snapshot the P20 gate needs; missing files stay ``None``."""
    d = Path(evidence_dir)
    return {
        "campaign": _read_json(d / "campaign_nodes.json"),
        "maturity_dir": d / "maturity" if (d / "maturity").is_dir() else None,
        "contracts_dir": d / "contracts" if (d / "contracts").is_dir() else None,
        "fsi_payload": _read_json(d / "contracts" / "fsi_handoff.json")
            if (d / "contracts" / "fsi_handoff.json").is_file() else None,
        "hydrostatic_payload": _read_json(d / "contracts" / "hydrostatic_handoff.json")
            if (d / "contracts" / "hydrostatic_handoff.json").is_file() else None,
        "seakeeping_payload": _read_json(d / "contracts" / "seakeeping_handoff.json")
            if (d / "contracts" / "seakeeping_handoff.json").is_file() else None,
        "report": _read_json(d / "campaign_report.json"),
        "paths": {
            "campaign": str(d / "campaign_nodes.json"),
            "fsi": str(d / "contracts" / "fsi_handoff.json"),
            "hydrostatic": str(d / "contracts" / "hydrostatic_handoff.json"),
            "seakeeping": str(d / "contracts" / "seakeeping_handoff.json"),
            "report": str(d / "campaign_report.json"),
        },
    }


def run_gate(evidence_dir: str | Path, *,
             generated_utc: str | None = None) -> dict:
    """Evaluate the four P20 gate criteria and return the full verdict."""
    ev = load_evidence(evidence_dir)
    stamp = generated_utc or datetime.now(timezone.utc).isoformat()

    campaign = ev["campaign"] or {"nodes": []}

    # ---------------------------------------------------------------- (a) DAG
    try:
        campaign_dag = D.build_dag_from_plan(campaign)
        convergence = campaign_dag.verify_convergence()
        dag_status = "PASS" if convergence.converged else "FAIL"
        crit_a = GateCriterion(
            "a", "DAG 无环验证 + 全节点依赖收敛", dag_status,
            convergence.detail,
            evidence={
                "acyclic": convergence.acyclic,
                "all_deps_done": convergence.all_deps_done,
                "converged": convergence.converged,
                "sorted_ids": convergence.sorted_ids,
                "blocked_nodes": convergence.blocked_nodes,
                "done_count": convergence.done_count,
                "total_count": convergence.total_count,
            },
        )
    except (ValueError, KeyError) as exc:
        crit_a = GateCriterion(
            "a", "DAG 无环验证 + 全节点依赖收敛", "FAIL",
            f"DAG construction failed: {exc}",
            evidence={"error": str(exc)},
        )
        convergence = None

    # ------------------------------------------------------------ (b) contracts
    contract_results: list[C.ContractResult] = []
    missing_payloads: list[str] = []

    if ev["fsi_payload"] is not None:
        contract_results.append(C.validate_fsi_handoff(ev["fsi_payload"]))
    else:
        missing_payloads.append("fsi_handoff.json")

    if ev["hydrostatic_payload"] is not None:
        contract_results.append(C.validate_hydrostatic_handoff(ev["hydrostatic_payload"]))
    else:
        missing_payloads.append("hydrostatic_handoff.json")

    if ev["seakeeping_payload"] is not None:
        contract_results.append(C.validate_seakeeping_handoff(ev["seakeeping_payload"]))
    else:
        missing_payloads.append("seakeeping_handoff.json")

    contract_statuses = [
        "PASS" if c.passed else "FAIL" for c in contract_results
    ]
    contract_statuses += ["WITHHELD"] * len(missing_payloads)
    crit_b_status = _aggregate(contract_statuses)
    crit_b = GateCriterion(
        "b", "跨仓接口契约验证（FSI/静水力/耐波）", crit_b_status,
        f"{len(contract_results)} contract checks run "
        f"({sum(c.passed for c in contract_results)} PASS / "
        f"{sum(not c.passed for c in contract_results)} FAIL); "
        f"missing payloads: {missing_payloads or 'none'}",
        evidence={
            "contracts": [c.as_dict() for c in contract_results],
            "missing_payloads": missing_payloads,
        },
    )

    # ------------------------------------------------------------ (c) maturity
    if ev["maturity_dir"] is not None:
        mat_summary = M.collect_maturity(evidence_dir)
    else:
        # Build from campaign nodes if maturity dir not available
        mat_summary = M.build_maturity_from_nodes(
            campaign.get("nodes", []), {})

    # C3: ≥2 maturity reports (N07/N08) complete
    n07 = next((n for n in mat_summary.nodes if n.node_id == "P20-N07"), None)
    n08 = next((n for n in mat_summary.nodes if n.node_id == "P20-N08"), None)
    n07_ok = n07 is not None and n07.delivered and n07.validation_status == "passed"
    n08_ok = n08 is not None and n08.delivered and n08.validation_status == "passed"
    maturity_count = sum(1 for ok in (n07_ok, n08_ok) if ok)
    crit_c_status = "PASS" if maturity_count >= 2 else "FAIL"
    crit_c = GateCriterion(
        "c", "成熟化单汇总（≥2条 N07/N08 报告完整）", crit_c_status,
        f"{mat_summary.delivered_count}/{mat_summary.total_count} maturity reports; "
        f"N07={'PASS' if n07_ok else 'FAIL'}, N08={'PASS' if n08_ok else 'FAIL'}; "
        f"overall TRL = {mat_summary.overall_trl:.1f}",
        evidence=mat_summary.as_dict(),
    )

    # ------------------------------------------------------------ (d) report
    report = ev["report"]
    six_elements = ["case_introduction", "methodology", "results",
                    "figures", "notes", "error_analysis"]
    if report is not None:
        present = [e for e in six_elements if e in report]
        missing = [e for e in six_elements if e not in report]
        crit_d_status = "PASS" if not missing else "FAIL"
        crit_d = GateCriterion(
            "d", "战役总结报告六要素完整", crit_d_status,
            f"{len(present)}/6 elements present; "
            f"missing: {missing or 'none'}",
            evidence={
                "present": present,
                "missing": missing,
                "report_path": ev["paths"]["report"],
            },
        )
    else:
        crit_d = GateCriterion(
            "d", "战役总结报告六要素完整", "FAIL",
            "campaign_report.json not found in evidence directory",
            evidence={"report_path": ev["paths"]["report"], "present": False},
        )

    criteria = [crit_a, crit_b, crit_c, crit_d]
    overall = "PASS" if all(c.status == "PASS" for c in criteria) else "FAIL"

    # --------------------------------------------------------------- provenance
    provenance = {}
    for key, path_str in ev["paths"].items():
        p = Path(path_str)
        provenance[key] = {
            "path": str(p), "present": p.is_file(),
            "sha256": sha256_file(p) if p.is_file() else None,
        }

    blocking = [f"{c.criterion_id}:{c.status}:{c.label}"
                for c in criteria if c.status != "PASS"]

    return {
        "schema": GATE_SCHEMA,
        "gate_id": GATE_ID,
        "title": REPORT_TITLE,
        "generated_utc": stamp,
        "campaign": {
            "plan_id": campaign.get("plan_id", 27),
            "title": campaign.get("title", "P20 Campaign"),
            "status": campaign.get("status", "unknown"),
            "n_nodes": len(campaign.get("nodes") or []),
        },
        "overall_status": overall,
        "blocking": blocking,
        "criteria": [c.as_dict() for c in criteria],
        "provenance": provenance,
        "limitations": [
            "DAG convergence depends on node status being accurately reported; "
            "stale status data may produce false PASS/FAIL.",
            "Contract validators check format/type/dimension consistency, not the "
            "physical accuracy of the delivered data.",
            "Maturity assessment requires delivered artifacts; absent artifacts "
            "are reported as not delivered, never inferred.",
            "Report element check is structural (key presence), not semantic "
            "(content quality).",
        ],
    }


def render_markdown(verdict: dict) -> str:
    """Render the P20 gate verdict as a markdown report."""
    camp = verdict.get("campaign") or {}
    lines = [
        f"# {verdict.get('title')} — 验收报告",
        "",
        f"- schema: `{verdict.get('schema')}`",
        f"- 生成时间 (UTC): `{verdict.get('generated_utc')}`",
        f"- 战役计划: `{camp.get('title')}` (plan_id={camp.get('plan_id')}, "
        f"status=`{camp.get('status')}`)",
        f"- 节点数: {camp.get('n_nodes')}",
        "",
        "## 1. 门裁决总览",
        "",
        "| 判据 | 内容 | 裁决 | 说明 |",
        "|---|---|---|---|",
    ]
    for c in verdict.get("criteria") or []:
        status_cell = {
            "PASS": "✅ PASS", "FAIL": "❌ FAIL", "WITHHELD": "⚪ WITHHELD",
        }.get(c["status"], c["status"])
        lines.append(
            f"| ({c['criterion_id']}) | {c['label']} | {status_cell} | "
            f"{c['detail']} |")

    lines += [
        "",
        f"**总裁决 (overall_status): `{verdict['overall_status']}`**",
        "",
    ]
    if verdict.get("blocking"):
        lines += ["阻塞项 (blocking):", ""]
        lines += [f"- `{b}`" for b in verdict["blocking"]]
        lines.append("")

    # --- (a) DAG detail
    crit_a = next((c for c in verdict["criteria"] if c["criterion_id"] == "a"), None)
    if crit_a:
        ev = crit_a.get("evidence", {})
        lines += [
            "## 2. DAG 收敛详情",
            "",
            f"- 无环: {'是' if ev.get('acyclic') else '否'}",
            f"- 依赖收敛: {'是' if ev.get('all_deps_done') else '否'}",
            f"- 已完成节点: {ev.get('done_count', 0)}/{ev.get('total_count', 0)}",
            f"- 拓扑序: `{ev.get('sorted_ids')}`",
            f"- 阻塞节点: `{ev.get('blocked_nodes')}`",
            "",
        ]

    # --- (b) contract detail
    crit_b = next((c for c in verdict["criteria"] if c["criterion_id"] == "b"), None)
    if crit_b:
        ev = crit_b.get("evidence", {})
        lines += ["## 3. 跨仓接口契约详情", ""]
        for cr in ev.get("contracts", []):
            lines.append(
                f"### {cr['contract_id']}: {cr['title']}")
            lines.append(f"- 裁决: `{cr['verdict']}` ({cr['n_checks']} checks, "
                         f"{cr['n_failed']} failed)")
            lines.append(f"- 生产者: {cr['producer']}")
            lines.append(f"- 消费者: {cr['consumer']}")
            lines.append("")
        if ev.get("missing_payloads"):
            lines.append(f"**缺失载荷**: {ev['missing_payloads']}")
            lines.append("")

    # --- (c) maturity detail
    crit_c = next((c for c in verdict["criteria"] if c["criterion_id"] == "c"), None)
    if crit_c:
        ev = crit_c.get("evidence", {})
        lines += [
            "## 4. 成熟化单汇总",
            "",
            f"- 已交付: {ev.get('delivered_count', 0)}/{ev.get('total_count', 0)}",
            f"- 验证通过: {ev.get('validation_passed', 0)}",
            f"- 整体 TRL: {ev.get('overall_trl', 0):.1f}",
            "",
            "| 节点 | TRL | 交付 | 验证 | 测试通过率 |",
            "|---|---|---|---|---|",
        ]
        for n in ev.get("nodes", []):
            lines.append(
                f"| {n['node_id']} | {n['trl']} | "
                f"{'✅' if n['delivered'] else '❌'} | "
                f"{n['validation_status']} | "
                f"{n['test_pass_rate']:.0%} |")
        lines.append("")

    # --- (d) report detail
    crit_d = next((c for c in verdict["criteria"] if c["criterion_id"] == "d"), None)
    if crit_d:
        ev = crit_d.get("evidence", {})
        lines += [
            "## 5. 战役总结报告六要素",
            "",
            f"- 已包含: {ev.get('present', [])}",
            f"- 缺失: {ev.get('missing', [])}",
            "",
        ]

    # --- limitations
    lines += [
        "## 6. 局限性与说明",
        "",
    ]
    for lim in verdict.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")

    # --- provenance
    lines += ["## 7. 溯源 (Provenance)", ""]
    for key, info in verdict.get("provenance", {}).items():
        sha = info.get("sha256", "N/A")
        lines.append(f"- `{key}`: present={info['present']}, "
                     f"sha256=`{sha[:16] if sha else 'N/A'}...`")
    lines.append("")

    return "\n".join(lines)


def write_outputs(verdict: dict, verdict_path: str | Path,
                  report_path: str | Path) -> None:
    """Write the gate verdict JSON and markdown report."""
    vp = Path(verdict_path)
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))

    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(render_markdown(verdict))
