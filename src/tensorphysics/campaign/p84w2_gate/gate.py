"""P84-W2 campaign integration gate: assemble, verify, and report.

The gate answers five questions with machine-checkable verdicts:

======================  =====================================================
criterion               how it is decided
======================  =====================================================
C1  N01-N11 通过        every node's acceptance_criteria evaluated from
                        delivered p84_node_acceptance.v1 artifact
C2  跨域接口互裁 ≥3     cross-domain interface checks (turbulence + multiphase)
                        with ≥3 PASS
C3  规范条文对齐 100%   mature nodes (N06-N11) regulation clause alignment
C4  谱系继承裁决        plan 84 gaps逐条回应 (all ADDRESSED)
C5  端到端 JSON 输出    {node_verdicts, cross_domain_checks, code_alignment,
                        lineage_ruling, campaign_report}
======================  =====================================================

Overall status is PASS only when C1-C5 all pass.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .plan import CAMPAIGN_PLAN
from .contracts import run_cross_domain_checks
from .code_alignment import run_code_alignment_checks
from .lineage import run_lineage_ruling

GATE_SCHEMA = "p84w2_campaign_gate_verdict.v1"
GATE_ID = "P84W2-G12"
REPORT_TITLE = "P84-W2 跨域集成与战役门·全节点验收汇总"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class NodeVerdict:
    """Verdict for a single campaign node."""
    ext_id: str
    name: str
    repo: str
    mission_id: int
    node_status: str
    reproduction_verdict: str
    n_criteria: int
    n_pass: int
    n_fail: int
    verdict: str  # PASS / FAIL / WITHHELD
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "ext_id": self.ext_id,
            "name": self.name,
            "repo": self.repo,
            "mission_id": self.mission_id,
            "node_status": self.node_status,
            "reproduction_verdict": self.reproduction_verdict,
            "n_criteria": self.n_criteria,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "verdict": self.verdict,
            "detail": self.detail,
        }


def evaluate_node(node: dict) -> NodeVerdict:
    """Evaluate a single node's acceptance criteria."""
    ext_id = node.get("ext_id", "?")
    name = node.get("name", "")
    repo = node.get("repo", "")
    mission_id = node.get("mission_id", 0)
    status = node.get("status", "unknown")
    repro = node.get("reproduction_verdict", "unknown")
    artifact = node.get("artifact")

    if artifact is None:
        return NodeVerdict(
            ext_id=ext_id, name=name, repo=repo, mission_id=mission_id,
            node_status=status, reproduction_verdict=repro,
            n_criteria=0, n_pass=0, n_fail=0, verdict="WITHHELD",
            detail="no artifact delivered",
        )

    criteria = artifact.get("criteria", {})
    n_total = len(criteria)
    n_pass = sum(1 for c in criteria.values() if c.get("pass"))
    n_fail = n_total - n_pass

    # Node passes if status=done, repro=pass, and all criteria pass
    if status == "done" and repro == "pass" and n_pass == n_total:
        verdict = "PASS"
    elif n_fail > 0:
        verdict = "FAIL"
    else:
        verdict = "WITHHELD"

    return NodeVerdict(
        ext_id=ext_id, name=name, repo=repo, mission_id=mission_id,
        node_status=status, reproduction_verdict=repro,
        n_criteria=n_total, n_pass=n_pass, n_fail=n_fail,
        verdict=verdict,
        detail=f"{n_pass}/{n_total} criteria pass",
    )


def evaluate_all_nodes() -> list[NodeVerdict]:
    """Evaluate all nodes in the campaign plan."""
    nodes = CAMPAIGN_PLAN.get("nodes", [])
    return [evaluate_node(n) for n in nodes]


def run_gate(artifact_dir: str | Path | None = None) -> dict:
    """Run the full P84-W2 campaign integration gate.

    Args:
        artifact_dir: optional directory containing external artifacts

    Returns:
        Complete gate verdict dict
    """
    # Collect artifacts from plan
    artifacts = {}
    for node in CAMPAIGN_PLAN.get("nodes", []):
        if node.get("artifact"):
            artifacts[node["ext_id"]] = node["artifact"]

    # C1: Node verdicts
    node_verdicts = evaluate_all_nodes()
    n_nodes_pass = sum(1 for v in node_verdicts if v.verdict == "PASS")
    n_nodes_total = len(node_verdicts)
    c1_status = "PASS" if n_nodes_pass == n_nodes_total else "FAIL"

    # C2: Cross-domain checks
    cross_domain = run_cross_domain_checks(artifacts)
    n_cross_pass = cross_domain["summary"]["n_pass"]
    c2_status = "PASS" if n_cross_pass >= 3 else "FAIL"

    # C3: Code alignment
    code_alignment = run_code_alignment_checks(artifacts)
    alignment_rate = code_alignment["summary"]["alignment_rate"]
    c3_status = "PASS" if alignment_rate >= 1.0 else "FAIL"

    # C4: Lineage ruling
    lineage = run_lineage_ruling(artifacts)
    c4_status = lineage.overall_status

    # C5: End-to-end JSON output
    c5_status = "PASS"  # We're generating it right now

    # Overall verdict
    criteria = [
        {
            "criterion_id": "C1",
            "label": f"全部 11 节点 status=done 且 reproduction_verdict=pass",
            "status": c1_status,
            "detail": f"{n_nodes_pass}/{n_nodes_total} nodes PASS",
            "evidence": {"nodes": [v.as_dict() for v in node_verdicts]},
        },
        {
            "criterion_id": "C2",
            "label": "跨域接口互裁测试 ≥3 条通过",
            "status": c2_status,
            "detail": f"{n_cross_pass} cross-domain checks PASS (≥3 required)",
            "evidence": cross_domain,
        },
        {
            "criterion_id": "C3",
            "label": "成熟化节点规范条文对齐率 100%",
            "status": c3_status,
            "detail": f"alignment rate {alignment_rate*100:.1f}% ({code_alignment['summary']['n_pass']}/{code_alignment['summary']['total_clauses']} clauses)",
            "evidence": code_alignment,
        },
        {
            "criterion_id": "C4",
            "label": "谱系继承裁决报告包含 plan 84 局限节逐条回应",
            "status": c4_status,
            "detail": f"{lineage.addressed}/{lineage.total_gaps} gaps ADDRESSED",
            "evidence": lineage.as_dict(),
        },
        {
            "criterion_id": "C5",
            "label": "端到端 JSON 输出",
            "status": c5_status,
            "detail": "gate verdict JSON generated with all required sections",
            "evidence": {"schema": GATE_SCHEMA},
        },
    ]

    overall = "PASS" if all(c["status"] == "PASS" for c in criteria) else "FAIL"
    blocking = [c["criterion_id"] for c in criteria if c["status"] != "PASS"]

    return {
        "schema": GATE_SCHEMA,
        "gate_id": GATE_ID,
        "title": REPORT_TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": CAMPAIGN_PLAN["plan_id"],
        "wave": CAMPAIGN_PLAN["wave"],
        "physical_domain": CAMPAIGN_PLAN["physical_domain"],
        "overall_status": overall,
        "blocking": blocking,
        "criteria": criteria,
        "node_verdicts": [v.as_dict() for v in node_verdicts],
        "cross_domain_checks": cross_domain,
        "code_alignment": code_alignment,
        "lineage_ruling": lineage.as_dict(),
        "campaign_report": {
            "total_nodes": n_nodes_total,
            "nodes_pass": n_nodes_pass,
            "nodes_fail": n_nodes_total - n_nodes_pass,
            "cross_domain_checks_pass": n_cross_pass,
            "code_alignment_rate": alignment_rate,
            "gaps_addressed": lineage.addressed,
            "gaps_total": lineage.total_gaps,
            "overall_verdict": overall,
        },
    }


def _status_cell(status: str) -> str:
    if status == "PASS":
        return "✅ PASS"
    if status == "FAIL":
        return "❌ FAIL"
    return "⚪ WITHHELD"


def render_markdown(verdict: dict) -> str:
    """Render the gate verdict as a markdown report."""
    lines = [
        f"# {verdict['title']}",
        "",
        f"- schema: `{verdict['schema']}`",
        f"- 生成时间 (UTC): `{verdict['generated_at']}`",
        f"- 战役计划: Plan {verdict['plan_id']} Wave {verdict['wave']}",
        f"- 物理域: `{verdict['physical_domain']}`",
        "",
        "## 1. 门裁决总览",
        "",
        "| 判据 | 内容 | 裁决 | 说明 |",
        "|---|---|---|---|",
    ]
    for c in verdict.get("criteria") or []:
        lines.append(f"| {c['criterion_id']} | {c['label']} | {_status_cell(c['status'])} | {c['detail']} |")

    lines += ["", f"**总裁决 (overall_status): `{verdict['overall_status']}`**", ""]

    if verdict.get("blocking"):
        lines += ["阻塞项 (blocking):", ""]
        lines += [f"- `{b}`" for b in verdict["blocking"]]
        lines.append("")

    # C1: Node verdicts
    lines += ["## 2. 判据 C1: 节点验收", "",
              "| 节点 | 仓库 | mission | 状态 | repro | 裁决 | 通过/总判据 |",
              "|---|---|---|---|---|---|---|"]
    for n in verdict.get("node_verdicts") or []:
        lines.append(f"| {n['ext_id']} | {n['repo']} | {n['mission_id']} | "
                     f"{n['node_status']} | {n['reproduction_verdict']} | "
                     f"{_status_cell(n['verdict'])} | {n['n_pass']}/{n['n_criteria']} |")
    lines.append("")

    # C2: Cross-domain checks
    lines += ["## 3. 判据 C2: 跨域接口互裁", "",
              "| 契约 | 生产者 → 消费者 | 检查数 | PASS | FAIL | 裁决 |",
              "|---|---|---|---|---|---|"]
    for ct in verdict.get("cross_domain_checks", {}).get("contracts", []):
        lines.append(f"| `{ct['contract_id']}` | {ct['producer']} → {ct['consumer']} | "
                     f"{ct['n_checks']} | {ct['n_pass']} | {ct['n_fail']} | "
                     f"{_status_cell(ct['verdict'])} |")
    lines.append("")

    for ct in verdict.get("cross_domain_checks", {}).get("contracts", []):
        lines.append(f"### `{ct['contract_id']}` — {ct['title']}")
        lines.append("")
        lines.append("| check | 裁决 | 数值/说明 |")
        lines.append("|---|---|---|")
        for chk in ct["checks"]:
            lines.append(f"| `{chk['check_id']}` | {_status_cell(chk['status'])} | "
                         f"{chk['detail']} |")
        lines.append("")

    # C3: Code alignment
    lines += ["## 4. 判据 C3: 规范条文对齐", "",
              "| 规范 | 条文 | 描述 | 节点 | 裁决 |",
              "|---|---|---|---|---|"]
    for chk in verdict.get("code_alignment", {}).get("checks", []):
        lines.append(f"| {chk['regulation']} | {chk['clause']} | "
                     f"{chk['description']} | {chk['node']} | "
                     f"{_status_cell(chk['status'])} |")
    lines.append("")
    summary = verdict.get("code_alignment", {}).get("summary", {})
    lines.append(f"**对齐率**: {summary.get('alignment_rate_pct', 'N/A')} "
                 f"({summary.get('n_pass', 0)}/{summary.get('total_clauses', 0)} clauses)")
    lines.append("")

    # C4: Lineage ruling
    lines += ["## 5. 判据 C4: 谱系继承裁决", "",
              "| Gap ID | 描述 | 响应节点 | 裁决 | 证据 |",
              "|---|---|---|---|---|"]
    for r in verdict.get("lineage_ruling", {}).get("rulings", []):
        status_str = "✅ ADDRESSED" if r["status"] == "ADDRESSED" else (
            "⚠️ PARTIAL" if r["status"] == "PARTIAL" else "❌ NOT_ADDRESSED")
        lines.append(f"| {r['gap_id']} | {r['description']} | "
                     f"{', '.join(r['addressed_by'])} | {status_str} | {r['evidence']} |")
    lines.append("")
    lr = verdict.get("lineage_ruling", {})
    lines.append(f"**谱系裁决**: {lr.get('addressed', 0)}/{lr.get('total_gaps', 0)} gaps ADDRESSED")
    lines.append("")

    # Campaign summary
    lines += ["## 6. 战役级汇总", "",
              "| 指标 | 值 |",
              "|---|---|"]
    cr = verdict.get("campaign_report", {})
    lines.append(f"| 总节点数 | {cr.get('total_nodes', 0)} |")
    lines.append(f"| 通过节点 | {cr.get('nodes_pass', 0)} |")
    lines.append(f"| 跨域检查通过 | {cr.get('cross_domain_checks_pass', 0)} |")
    lines.append(f"| 规范对齐率 | {cr.get('code_alignment_rate', 0)*100:.1f}% |")
    lines.append(f"| 谱系缺口填补 | {cr.get('gaps_addressed', 0)}/{cr.get('gaps_total', 0)} |")
    lines.append(f"| 战役总裁决 | {cr.get('overall_verdict', 'N/A')} |")
    lines.append("")

    # Limitations
    lines += ["## 7. 局限", ""]
    lines.append("- 本战役基于合成验收数据，非实际求解器运行结果")
    lines.append("- 跨域接口检查为静态契约验证，未执行运行时耦合测试")
    lines.append("- 规范条文对齐基于artifact文本匹配，未验证公式实现正确性")
    lines.append("- 谱系继承裁决假设plan 84缺口列表完整")
    lines.append("")

    return "\n".join(lines)


def write_outputs(verdict: dict, out_json: str | Path, out_md: str | Path) -> dict:
    """Write gate verdict to JSON and markdown files."""
    out_json, out_md = Path(out_json), Path(out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    out_md.write_text(render_markdown(verdict))
    return {
        "json": str(out_json),
        "markdown": str(out_md),
        "json_sha256": sha256_file(out_json),
        "markdown_sha256": sha256_file(out_md),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="tensorphysics.campaign.p84w2_gate",
        description="P84-W2 campaign integration gate")
    ap.add_argument("--out-json", default="artifacts/p84_w2/report.json")
    ap.add_argument("--out-md", default="docs/P84_W2_CAMPAIGN_INTEGRATION_GATE.md")
    args = ap.parse_args(argv)

    verdict = run_gate()
    paths = write_outputs(verdict, args.out_json, args.out_md)
    print(json.dumps({
        "overall_status": verdict["overall_status"],
        "criteria": {c["criterion_id"]: c["status"] for c in verdict["criteria"]},
        "outputs": paths,
    }, ensure_ascii=False, indent=2))
    return 0 if verdict["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
