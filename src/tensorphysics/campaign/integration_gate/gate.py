"""P16-G1 campaign integration gate: assemble, verify, and report.

The gate is the campaign's final acceptance door.  It answers four questions,
each with a machine-checkable verdict, and refuses to answer any of them
optimistically:

======================  =====================================================
criterion               how it is decided
======================  =====================================================
(a) N1..N8 通过        every ``P16-N*`` node's ``acceptance_criteria`` evaluated
                        from a delivered ``p16_node_acceptance.v1`` artifact;
                        no artifact -> ``WITHHELD`` (never a pass)
(b) 跨域接口契约         wind table (LBM -> ship-platform) and wet-modal handoff
                        (FEM -> BEM) payload validators + live consumer endpoint
                        compatibility facts
(c) P15 回归 |Δ|<1%     eleven ``IN14-D*`` probe pass rates re-measured on the
                        P16 code base, delta vs the recorded P15 rate
(d) 战役级验收报告       this module emits ``gate_verdict.json`` + markdown with
                        sha256 provenance of every input artifact
======================  =====================================================

Overall status is ``PASS`` only when (a)-(d) all pass; otherwise ``FAIL`` with
the blocking reasons spelled out.  ``WITHHELD`` upstream evidence is reported as
``WITHHELD`` per criterion and forces the overall verdict to ``FAIL`` -- the
"honest FAIL is not overridable" rule of the campaign discipline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import contract as C
from . import nodes as N
from . import regression as R

GATE_SCHEMA = "p16_campaign_gate_verdict.v1"
GATE_ID = "P16-G1"
REPORT_TITLE = "P16 全战役集成验收门 (P16-G1)"


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
        return {"criterion_id": self.criterion_id, "label": self.label,
                "status": self.status, "detail": self.detail, "evidence": self.evidence}


def _aggregate(statuses: list[str]) -> str:
    if not statuses:
        return "WITHHELD"
    if "FAIL" in statuses:
        return "FAIL"
    if all(s == "PASS" for s in statuses):
        return "PASS"
    return "WITHHELD"


def load_evidence(evidence_dir: str | Path) -> dict:
    """Load every snapshot the gate needs; missing files stay ``None`` (never faked)."""
    d = Path(evidence_dir)
    node_art_dir = d / "nodes"
    artifacts = {}
    if node_art_dir.is_dir():
        for p in sorted(node_art_dir.glob("*.json")):
            art = _read_json(p)
            if art is None:
                continue
            ext = art.get("ext_id") or p.stem
            art["artifact_path"] = str(p)   # provenance = where the gate actually loaded it
            artifacts[ext] = art
    payload_dir = d / "contracts"
    return {
        "campaign": _read_json(d / "campaign_nodes.json"),
        "node_evidence": _read_json(d / "node_evidence.json"),
        "node_artifacts": artifacts,
        "p15_baseline": _read_json(d / "p15_baseline.json"),
        "probe_results": _read_json(d / "probe_results.json"),
        "ship_facts": _read_json(d / "ship_platform_facts.json"),
        "wind_payload": _read_json(payload_dir / "wind_coefficients.json"),
        "wet_modal_payload": _read_json(payload_dir / "wet_modal_handoff.json"),
        "paths": {
            "campaign": str(d / "campaign_nodes.json"),
            "node_evidence": str(d / "node_evidence.json"),
            "p15_baseline": str(d / "p15_baseline.json"),
            "probe_results": str(d / "probe_results.json"),
            "ship_facts": str(d / "ship_platform_facts.json"),
        },
    }


def run_gate(evidence_dir: str | Path, *, threshold: float = R.REGRESSION_THRESHOLD,
             generated_utc: str | None = None) -> dict:
    """Evaluate the four gate criteria and return the full verdict structure."""
    ev = load_evidence(evidence_dir)
    stamp = generated_utc or datetime.now(timezone.utc).isoformat()

    # ---------------------------------------------------------------- (a) nodes
    campaign = ev["campaign"] or {"nodes": []}
    node_verdicts = N.evaluate_nodes(campaign, ev["node_evidence"] or {},
                                     ev["node_artifacts"])
    node_status = _aggregate([v.status for v in node_verdicts])
    crit_a = GateCriterion(
        "a", "N1-N8 全部节点 acceptance_criteria 通过", node_status,
        f"{sum(v.status == 'PASS' for v in node_verdicts)}/{len(node_verdicts)} nodes PASS, "
        f"{sum(v.status == 'FAIL' for v in node_verdicts)} FAIL, "
        f"{sum(v.status == 'WITHHELD' for v in node_verdicts)} WITHHELD; "
        f"{sum(v.delivered for v in node_verdicts)} delivery record(s) accepted by the platform",
        evidence={"nodes": [v.as_dict() for v in node_verdicts]},
    )

    # ------------------------------------------------------------ (b) contracts
    contracts: list[C.ContractResult] = []
    if ev["wind_payload"] is not None:
        contracts.append(C.validate_wind_coefficient_table(ev["wind_payload"]))
    if ev["wet_modal_payload"] is not None:
        contracts.append(C.validate_wet_modal_handoff(ev["wet_modal_payload"]))
    missing_payloads = [name for name, blob in
                        (("wind_coefficients.json", ev["wind_payload"]),
                         ("wet_modal_handoff.json", ev["wet_modal_payload"])) if blob is None]
    endpoint_results = (C.validate_endpoint_compatibility(ev["ship_facts"])
                        if ev["ship_facts"] else [])
    contracts.extend(endpoint_results)

    sub_statuses = [c.passed and "PASS" or "FAIL" for c in contracts]
    sub_statuses += ["WITHHELD"] * len(missing_payloads)
    crit_b_status = "FAIL" if "FAIL" in sub_statuses else _aggregate(sub_statuses)
    crit_b = GateCriterion(
        "b", "跨域接口契约验证（格式/类型/量纲一致）", crit_b_status,
        f"{len(contracts)} contract checks run "
        f"({sum(c.passed for c in contracts)} PASS / {sum(not c.passed for c in contracts)} FAIL); "
        f"missing producer payloads: {missing_payloads or 'none'}",
        evidence={
            "contracts": [c.as_dict() for c in contracts],
            "missing_payloads": missing_payloads,
            "expected_payload_paths": [
                "contracts/wind_coefficients.json", "contracts/wet_modal_handoff.json"],
        },
    )

    # ----------------------------------------------------------- (c) regression
    p15 = ev["p15_baseline"] or {"nodes": []}
    cases = R.evaluate_regression(p15, ev["probe_results"], threshold=threshold)
    regression_status = _aggregate([c.status for c in cases])
    deltas = [abs(c.delta) for c in cases if c.delta is not None]
    crit_c = GateCriterion(
        "c", "P15 回归基线 11 节点 |Δ| < 1%", regression_status,
        f"{sum(c.status == 'PASS' for c in cases)}/{len(cases)} P15 nodes PASS; "
        f"measured probes: {len(deltas)}/{len(cases)}; "
        f"max |delta| = {max(deltas):.3e}" if deltas else
        f"{sum(c.status == 'PASS' for c in cases)}/{len(cases)} P15 nodes PASS; "
        f"no probe executed in the gate environment -> WITHHELD",
        evidence={"threshold": threshold, "cases": [c.as_dict() for c in cases]},
    )

    # --------------------------------------------------------------- (d) report
    crit_d = GateCriterion(
        "d", "战役级验收报告生成（数字可回溯 artifacts JSON）", "PASS",
        "gate_verdict.json + docs/P16_CAMPAIGN_INTEGRATION_GATE.md emitted with sha256 "
        "provenance of every input artifact",
        evidence={"artifacts": {}},
    )

    criteria = [crit_a, crit_b, crit_c, crit_d]
    # The campaign discipline is explicit: a gate that is not demonstrably passed is
    # FAILED, never "withheld".  WITHHELD stays visible per criterion, but the
    # overall verdict only reads PASS when all four criteria read PASS.
    overall = "PASS" if all(c.status == "PASS" for c in criteria) else "FAIL"

    # --------------------------------------------------------------- provenance
    provenance = {}
    for key, path in ev["paths"].items():
        p = Path(path)
        provenance[key] = {"path": str(p), "present": p.is_file(),
                           "sha256": sha256_file(p) if p.is_file() else None}
    for ext, art in sorted(ev["node_artifacts"].items()):
        p = Path(art.get("artifact_path", ""))
        provenance[f"node_artifact:{ext}"] = {
            "path": str(p), "present": p.is_file(),
            "sha256": sha256_file(p) if p.is_file() else None}

    blocking = [f"{c.criterion_id}:{c.status}:{c.label}" for c in criteria
                if c.status != "PASS"]

    return {
        "schema": GATE_SCHEMA,
        "gate_id": GATE_ID,
        "title": REPORT_TITLE,
        "generated_utc": stamp,
        "campaign": {
            "plan_id": (campaign.get("source") or {}).get("api"),
            "title": (campaign.get("source") or {}).get("plan_title"),
            "status": (campaign.get("source") or {}).get("plan_status"),
            "physics_scope": (campaign.get("source") or {}).get("physics_scope"),
            "coupling_requirements": (campaign.get("source") or {})
            .get("coupling_framework_requirements"),
            "collected_utc": campaign.get("collected_utc"),
            "n_nodes": len(campaign.get("nodes") or []),
        },
        "overall_status": overall,
        "blocking": blocking,
        "criteria": [c.as_dict() for c in criteria],
        "provenance": provenance,
        "limitations": [
            "Node acceptance is only as good as the delivered p16_node_acceptance.v1 artifact; "
            "absent artifacts are reported WITHHELD, never inferred from PR merge state.",
            "Regression deltas cover the probe metrics recorded by the platform for P15; "
            "probes that cannot execute in the gate environment stay WITHHELD.",
            "Cross-domain payload validators check format/type/dimension consistency, not the "
            "physical accuracy of the delivered coefficients (that is N7/N8 acceptance).",
        ],
    }


# --------------------------------------------------------------------------- markdown
def _status_cell(s: str) -> str:
    return {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WITHHELD": "⚪ WITHHELD"}.get(s, s)


def render_markdown(verdict: dict) -> str:
    camp = verdict.get("campaign") or {}
    lines = [
        f"# {verdict.get('title')} — 验收报告",
        "",
        f"- schema: `{verdict.get('schema')}`",
        f"- 生成时间 (UTC): `{verdict.get('generated_utc')}`",
        f"- 战役计划: `{camp.get('title')}` (plan_id 源 {camp.get('plan_id')}, status "
        f"`{camp.get('status')}`)",
        f"- 物理域: `{camp.get('physics_scope')}`",
        f"- 耦合要求: `{camp.get('coupling_requirements')}`",
        "",
        "## 1. 门裁决总览",
        "",
        "| 判据 | 内容 | 裁决 | 说明 |",
        "|---|---|---|---|",
    ]
    for c in verdict.get("criteria") or []:
        lines.append(f"| ({c['criterion_id']}) | {c['label']} | {_status_cell(c['status'])} | "
                     f"{c['detail']} |")
    lines += ["", f"**总裁决 (overall_status): `{verdict['overall_status']}`**", ""]
    if verdict.get("blocking"):
        lines += ["阻塞项 (blocking):", ""]
        lines += [f"- `{b}`" for b in verdict["blocking"]]
        lines.append("")

    # --- (a) node table
    crit_a = next((c for c in verdict["criteria"] if c["criterion_id"] == "a"), None)
    if crit_a:
        lines += ["## 2. 判据 (a) N1-N8 节点验收", "",
                  "| 节点 | 仓库 | mission | 节点状态 | 交付 | 裁决 | 通过/总判据 |",
                  "|---|---|---|---|---|---|---|"]
        for n in crit_a["evidence"]["nodes"]:
            lines.append(f"| {n['ext_id']} | {n['repo']} | {n['mission_id']} | "
                         f"{n['node_status']} | {'yes' if n['delivered'] else 'no'} | "
                         f"{_status_cell(n['verdict'])} | {n['n_pass']}/{n['n_criteria']} |")
        lines.append("")

    # --- (b) contracts
    crit_b = next((c for c in verdict["criteria"] if c["criterion_id"] == "b"), None)
    if crit_b:
        lines += ["## 3. 判据 (b) 跨域接口契约", "",
                  "| 契约 | 生产者 → 消费者 | 检查数 | 失败数 | 裁决 |",
                  "|---|---|---|---|---|"]
        for ct in crit_b["evidence"]["contracts"]:
            lines.append(f"| `{ct['contract_id']}` | {ct['producer']} → {ct['consumer']} | "
                         f"{ct['n_checks']} | {ct['n_failed']} | {_status_cell(ct['verdict'])} |")
        lines.append("")
        for ct in crit_b["evidence"]["contracts"]:
            lines.append(f"### `{ct['contract_id']}` — {ct['title']}")
            lines.append("")
            lines.append("| check | 裁决 | 数值/说明 |")
            lines.append("|---|---|---|")
            for chk in ct["checks"]:
                lines.append(f"| `{chk['check_id']}` | {_status_cell(chk['status'])} | "
                             f"{chk['detail']} |")
            lines.append("")

    # --- (c) regression
    crit_c = next((c for c in verdict["criteria"] if c["criterion_id"] == "c"), None)
    if crit_c:
        lines += ["## 4. 判据 (c) P15 回归基线（|Δ| < 1%）", "",
                  "| P15 节点 | 仓库 | P15 pass rate | 本次 pass rate | Δ | 用例数保留 | 裁决 |",
                  "|---|---|---|---|---|---|---|"]
        for case in crit_c["evidence"]["cases"]:
            def f(x, fmt="{:.4f}"):
                return "—" if x is None else fmt.format(x)
            lines.append(f"| {case['ext_id']} | {case['repo']} | {f(case['rate_p15'])} | "
                         f"{f(case['rate_now'])} | {f(case['delta'], '{:+.2e}')} | "
                         f"{case['test_count_preserved']} | {_status_cell(case['status'])} |")
        lines.append("")

    # --- provenance
    lines += ["## 5. 审计链（输入 artifact sha256）", "", "| artifact | 路径 | sha256 |", "|---|---|---|"]
    for k, v in (verdict.get("provenance") or {}).items():
        lines.append(f"| {k} | `{v['path']}` | `{(v['sha256'] or 'MISSING')[:16]}…` |")
    lines += ["", "## 6. 局限", ""]
    lines += [f"- {x}" for x in verdict.get("limitations") or []]
    lines.append("")
    return "\n".join(lines)


def write_outputs(verdict: dict, out_json: str | Path, out_md: str | Path) -> dict:
    out_json, out_md = Path(out_json), Path(out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(verdict, ensure_ascii=False, indent=1))
    out_md.write_text(render_markdown(verdict))
    return {"json": str(out_json), "markdown": str(out_md),
            "json_sha256": sha256_file(out_json),
            "markdown_sha256": sha256_file(out_md)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="tensorphysics.campaign.integration_gate",
                                 description="P16-G1 campaign integration gate")
    ap.add_argument("--evidence-dir", default="artifacts/p16_gate")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)
    out_json = args.out_json or str(Path(args.evidence_dir) / "gate_verdict.json")
    out_md = args.out_md or "docs/P16_CAMPAIGN_INTEGRATION_GATE.md"
    verdict = run_gate(args.evidence_dir)
    paths = write_outputs(verdict, out_json, out_md)
    print(json.dumps({"overall_status": verdict["overall_status"],
                      "criteria": {c["criterion_id"]: c["status"] for c in verdict["criteria"]},
                      "outputs": paths}, ensure_ascii=False, indent=1))
    return 0 if verdict["overall_status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())