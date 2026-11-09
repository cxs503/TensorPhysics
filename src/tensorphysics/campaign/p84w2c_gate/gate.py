"""P84-W2C-N11 campaign gate: assemble evidence, judge, publish verdict.

The gate is the DAG integration node of the P84-W2C campaign (plan 385).  It
answers the four machine-checkable campaign criteria and refuses to answer any
of them optimistically:

======  ==========================================================  ==============
id      how it is decided                                           non-green
======  ==========================================================  ==============
C1      every node's pytest run re-executed by the gate, junit       FAIL
        XML normalised to JSON, suite green and non-empty
C2      every *required* cross-domain contract passes bilateral      FAIL
        checks (consumer AST + producer schema + numeric anchor),
        and any delivered producer artefact conforms to its schema
C3      campaign markdown: six paper elements present, figure       FAIL
        exists, >= MIN_REPORT_CHARS characters
C4      verdict JSON written, validated against the registered      FAIL
        closed schema, carries per-node numbers + fingerprint
======  ==========================================================  ==============

``overall_status`` is ``PASS`` only when C1..C4 are all ``PASS`` **and** every
node verdict is ``PASS``.  ``WITHHELD`` evidence never becomes green: an
undelivered node fails the campaign, and every failure is expanded into a
remediation line in ``gap_analysis``.
"""
from __future__ import annotations

import hashlib
import json
import platform as py_platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schema import canonical_json, payload_digest, validate
from ..schema_defs import schema_for, schema_fingerprint
from . import collect as C
from . import contracts as K
from . import criteria as J
from . import lineage as L
from .figures import write_gate_figure
from .report import SECTIONS, render_markdown
from .spec import (CONTRACTS, GATE_CRITERIA, MIN_REPORT_CHARS, NODES, PARENT_PLAN_ID)

GATE_ID = "P84W2C-N11"
SCHEMA_NAME = "campaign_gate_report.v1"
SCHEMA_VERSION = "1.0.0"
REPORT_JSON = "campaign_gate_report.json"
REPORT_MD = "P84W2C_CAMPAIGN_GATE.md"
FIGURE_PNG = "p84w2c_gate_matrix.png"
EVIDENCE_DIR = "node_evidence"

_NODE_FIELDS = ("node_key", "mission_id", "repo", "title", "physics_domain",
                "root_present", "platform_status", "verdict", "criteria",
                "artifacts", "pytest", "key_numbers", "gaps", "lineage")

_REMEDIATION = (
    ("节点工作树缺失", "创建节点工作树（检出 base_ref）并按交付物清单落盘产物"),
    ("未落盘任何 JSON 产物", "按节点交付物 glob 落盘 case/results JSON（含判据数字字段）"),
    ("JSON 产物全部不可解析", "修复产物 JSON 编码/截断后重新落盘"),
    ("pytest 套件未执行", "交付声明中的测试文件并确保 pytest 可执行（门内会亲自执行）"),
    ("pytest 未全绿", "修复失败/错误用例直至 pytest 全绿；honest FAIL 不得掩盖"),
    ("判据", "在产物 JSON 中写入该判据的数值字段（见 spec.py 的 aliases 列表）"),
    ("父计划", "补齐父计划节点状态证据或接受 inherited_unverified 裁决"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remediation(reason: str) -> str:
    for prefix, action in _REMEDIATION:
        if reason.startswith(prefix) or prefix in reason:
            return action
    return "补齐该缺口并把证据落盘到节点 worktree，然后重跑本门"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_audit(pkg_dir: Path) -> list[dict[str, str]]:
    entries = []
    for path in sorted(pkg_dir.glob("*.py")):
        entries.append({"path": path.name, "sha256": C.sha256_file(path)})
    return entries


def _node_record(collected: Mapping[str, Any], spec: Any,
                 lineage: Mapping[str, Any]) -> dict[str, Any]:
    criteria = [J.judge_criterion(
        crit,
        collected.get("candidates", {}).get(crit.cid) or collected["key_numbers"].get(crit.cid),
        collected.get("metric_sources", {}).get(crit.cid, ""))
        for crit in spec.criteria]
    node = {
        "node_key": collected["node_key"],
        "mission_id": collected["mission_id"],
        "repo": collected["repo"],
        "title": collected["title"],
        "physics_domain": collected["physics_domain"],
        "root_present": bool(collected.get("root_present")),
        "platform_status": str(collected.get("platform_status") or ""),
        "verdict": "WITHHELD",
        "criteria": criteria,
        "artifacts": collected.get("artifacts", []),
        "pytest": collected["pytest"],
        "key_numbers": {
            crit["id"]: (crit["value"] if crit["value"] is not None
                         else ("true" if crit["status"] == "PASS" and crit["comparator"] == "is_true"
                               else None))
            for crit in criteria
        },
        "gaps": [],
        "lineage": lineage,
    }
    node["verdict"] = J.node_verdict(node)
    node["gaps"] = J.node_evidence_gaps({**collected, "criteria": criteria}, spec)
    return {key: node[key] for key in _NODE_FIELDS}


def _gap_analysis(nodes: Sequence[Mapping[str, Any]],
                  criteria: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if node["verdict"] == "PASS":
            continue
        reasons = node.get("gaps") or [f"节点裁决 {node['verdict']} 但未给出原因"]
        out.append({
            "node_key": node["node_key"],
            "severity": "blocker",
            "reason": f"裁决 {node['verdict']}；" + "；".join(reasons),
            "remediation": _remediation(reasons[0]),
        })
    for crit in criteria:
        if crit["status"] == "PASS":
            continue
        for gap in crit.get("gaps", []):
            out.append({
                "node_key": f"CAMPAIGN-{crit['id']}",
                "severity": "blocker",
                "reason": gap,
                "remediation": _remediation(gap),
            })
    return out


def _platform_view(platform: Mapping[str, Any] | None,
                   mission_ids: Sequence[int]) -> dict[str, Any]:
    if not platform:
        return {"available": False, "reason": "platform evidence not requested"}
    rows = platform.get("missions", {}) if isinstance(platform, Mapping) else {}
    statuses = {str(mid): (rows.get(str(mid)) or {}).get("status") for mid in mission_ids}
    compact_nodes = {
        key: {"name": value.get("name"), "status": value.get("status"),
              "mission_id": value.get("mission_id")}
        for key, value in (platform.get("nodes", {}) or {}).items()
    }
    return {
        "available": bool(platform.get("available")),
        "reason": platform.get("reason", ""),
        "plan_id": platform.get("plan_id"),
        "nodes": compact_nodes,
        "missions": {str(mid): (rows.get(str(mid)) or {}) for mid in mission_ids},
        "mission_statuses": statuses,
    }


def run_gate(node_roots: Mapping[str, Path],
             out_dir: Path,
             consumer_roots: Mapping[str, Path] | None = None,
             plan_id: int = 385,
             plan_title: str = "P84-W2C seed (谱系继承波·Plan 84 Wave 2C)",
             platform: Mapping[str, Any] | None = None,
             parent_nodes: Mapping[str, Any] | None = None,
             doc_path: Path | None = None,
             run_pytest: bool = True,
             pytest_timeout: int = 900,
             make_figure: bool = True,
             reproduce: Sequence[str] = (),
             pkg_dir: Path | None = None) -> dict[str, Any]:
    """Run the whole gate end-to-end and write JSON + markdown + figure."""
    pkg_dir = Path(pkg_dir) if pkg_dir else Path(__file__).resolve().parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / EVIDENCE_DIR
    evidence_dir.mkdir(parents=True, exist_ok=True)
    consumer_roots = {k: Path(v) for k, v in (consumer_roots or {}).items() if v is not None}

    platform_view = _platform_view(platform, [spec.mission_id for spec in NODES])
    mission_status = platform_view.get("mission_statuses", {})

    collected = {
        spec.node_key: C.collect_node(
            Path(node_roots.get(spec.node_key, "")),
            spec,
            evidence_dir,
            run_suite=run_pytest,
            timeout=pytest_timeout,
            platform_status=str(mission_status.get(str(spec.mission_id)) or ""),
        )
        for spec in NODES
    }

    parent_for_lineage: Mapping[str, Any] = parent_nodes or {}
    if not parent_for_lineage and platform_view.get("nodes"):
        parent_for_lineage = {}
    lineage_results, lineage_summary = L.adjudicate_all(
        [{"node_key": spec.node_key, "title": spec.title} for spec in NODES],
        parent_for_lineage,
        {spec.node_key: spec.lineage_title for spec in NODES},
    )
    lineage_summary["parent_plan_id"] = int(PARENT_PLAN_ID)

    nodes = [
        _node_record(collected[spec.node_key], spec, lineage_results[index])
        for index, spec in enumerate(NODES)
    ]

    producer_payloads = {key: value.get("payloads", []) for key, value in collected.items()}
    contracts = K.evaluate_all_contracts(consumer_roots, producer_payloads)

    report_json_path = out_dir / REPORT_JSON
    figure_path = out_dir / FIGURE_PNG
    figure_rel = None
    figure_reason: str | None = "figure disabled"
    if make_figure:
        written, figure_reason = write_gate_figure({"nodes": nodes, "contracts": contracts,
                                                    "criteria": [], "overall_status": "PENDING"},
                                                   figure_path)
        figure_path = written or figure_path
        figure_rel = f"{FIGURE_PNG}" if written else None

    md_path = Path(doc_path) if doc_path else (out_dir / REPORT_MD)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_dir_rel = Path("..") / Path("..")  # resolved later against doc location
    figure_from_doc = None
    if figure_rel:
        try:
            figure_from_doc = str(figure_path.relative_to(md_path.parent))
        except ValueError:
            figure_from_doc = str(figure_path)
        figure_rel = figure_from_doc

    schema_fp = schema_fingerprint(SCHEMA_NAME)
    c1 = J.evaluate_c1(nodes, evidence_dir)
    c2 = J.evaluate_c2(contracts)

    criteria = [c1, c2, J.evaluate_c3(md_path, SECTIONS, figure_path if figure_rel else None)]
    criteria.append({"id": "C4", "statement": GATE_CRITERIA["C4"], "status": "FAIL",
                     "metrics": {}, "gaps": ["verdict JSON not sealed yet"]})
    audit = {
        "package_files": _package_audit(pkg_dir),
        "platform": platform_view,
        "tool": f"p84w2c_gate/{SCHEMA_VERSION}",
        "python": sys.version.split()[0] + f" ({py_platform.machine()})",
        "figure": figure_reason,
    }
    artifacts_index = {
        "report_json": str(report_json_path),
        "report_md": str(md_path),
        "figure_png": str(figure_path) if figure_rel else None,
        "evidence_dir": str(evidence_dir),
        "criterion_hits": sum(1 for n in nodes for c in n["criteria"] if c["status"] == "PASS"),
        "criterion_total": sum(len(n["criteria"]) for n in nodes),
        "schema_fingerprint": schema_fp,
    }

    base = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "plan_id": int(plan_id),
        "plan_title": plan_title,
        "gate_id": GATE_ID,
        "generated_utc": _now(),
        "nodes": nodes,
        "contracts": contracts,
        "lineage_summary": lineage_summary,
        "artifacts_index": artifacts_index,
        "audit": audit,
    }
    placeholder_c4 = {"id": "C4", "statement": GATE_CRITERIA["C4"], "status": "FAIL",
                      "metrics": {}, "gaps": ["verdict JSON not sealed yet"]}
    criteria = [c1, c2,
                J.evaluate_c3(md_path, SECTIONS, figure_path if figure_rel else None),
                placeholder_c4]

    # Fixed-point loop: the markdown contains the campaign criteria verdicts
    # (including C3's own character count), so render -> measure -> re-render.
    # It converges in two iterations; the loop cap is a safety net, not a fudge.
    report: dict[str, Any] = {}
    for _ in range(6):
        criteria = [c1, c2, criteria[2], placeholder_c4]
        gap_analysis = _gap_analysis(nodes, criteria)
        report = {
            **base,
            "criteria": criteria,
            "gap_analysis": gap_analysis,
            "blocker_count": len(gap_analysis),
            "overall_status": J.overall_status(criteria, nodes),
            "markdown": {"path": str(md_path), "chars": 0,
                         "min_chars": MIN_REPORT_CHARS, "sections": list(SECTIONS),
                         "figure": figure_rel},
            "fingerprint": "",
        }
        if make_figure:
            written, figure_reason = write_gate_figure(report, figure_path)
            audit["figure"] = figure_reason
            if written is None and figure_rel:
                report["markdown"]["figure"] = None
                figure_rel = None
        text = render_markdown(report, report["markdown"]["figure"], reproduce)
        md_path.write_text(text, encoding="utf-8")
        report["markdown"]["chars"] = len(text)
        report["fingerprint"] = payload_digest(
            {key: value for key, value in report.items() if key != "fingerprint"})
        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
        new_c3 = J.evaluate_c3(md_path, SECTIONS,
                               figure_path if report["markdown"]["figure"] else None)
        new_c4 = J.evaluate_c4(report_json_path, schema_fp)
        stable = ([new_c3["status"], new_c4["status"]]
                  == [criteria[2]["status"], criteria[3]["status"]]
                  and report["markdown"]["chars"] == len(text))
        criteria = [c1, c2, new_c3, new_c4]
        if stable:
            break

    # Final seal: write the settled verdict, then re-verify it from disk.
    report["criteria"] = criteria
    report["gap_analysis"] = _gap_analysis(nodes, criteria)
    report["blocker_count"] = len(report["gap_analysis"])
    report["overall_status"] = J.overall_status(criteria, nodes)
    report["generated_utc"] = _now()
    text = render_markdown(report, report["markdown"]["figure"], reproduce)
    md_path.write_text(text, encoding="utf-8")
    report["markdown"]["chars"] = len(text)
    report["fingerprint"] = payload_digest(
        {key: value for key, value in report.items() if key != "fingerprint"})
    validate(schema_for(SCHEMA_NAME), report)
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")

    # C3/C4 describe the files that actually exist now.  The markdown no longer
    # prints its own character count (self-referential), so the statuses settle
    # in at most two passes; the loop is a safety net with an explicit bound.
    for _ in range(4):
        final_c3 = J.evaluate_c3(md_path, SECTIONS,
                                 figure_path if report["markdown"]["figure"] else None)
        final_c4 = J.evaluate_c4(report_json_path, schema_fp)
        settled = (final_c3["status"] == report["criteria"][2]["status"]
                   and final_c4["status"] == report["criteria"][3]["status"]
                   and final_c3["metrics"]["chars"] == report["markdown"]["chars"])
        report["criteria"][2] = final_c3
        report["criteria"][3] = final_c4
        report["gap_analysis"] = _gap_analysis(nodes, report["criteria"])
        report["blocker_count"] = len(report["gap_analysis"])
        report["overall_status"] = J.overall_status(report["criteria"], nodes)
        if settled:
            break
        text = render_markdown(report, report["markdown"]["figure"], reproduce)
        md_path.write_text(text, encoding="utf-8")
        report["markdown"]["chars"] = len(text)
        report["fingerprint"] = payload_digest(
            {key: value for key, value in report.items() if key != "fingerprint"})
        validate(schema_for(SCHEMA_NAME), report)
        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")

    report["fingerprint"] = payload_digest(
        {key: value for key, value in report.items() if key != "fingerprint"})
    validate(schema_for(SCHEMA_NAME), report)
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    return report


def summarise(report: Mapping[str, Any]) -> str:
    """One-line human summary (used by the CLI)."""
    nodes = report["nodes"]
    crit = report["criteria"]
    return (f"{report['overall_status']} | nodes PASS "
            f"{sum(1 for n in nodes if n['verdict'] == 'PASS')}/{len(nodes)} | "
            + " ".join(f"{c['id']}={c['status']}" for c in crit)
            + f" | blockers={report['blocker_count']} | fp={report['fingerprint'][:16]}")