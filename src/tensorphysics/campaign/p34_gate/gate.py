"""P34-G1 campaign integration gate: assemble evidence, judge, publish verdict.

The gate answers the five machine-checkable criteria of the campaign
(``C1``..``C5``) and refuses to answer any of them optimistically:

======  ==========================================================  ==============
id      how it is decided                                           non-green
======  ==========================================================  ==============
C1      recorded pytest runs of the 7 node worktrees, counts        WITHHELD/FAIL
        re-parsed from junit XML, node minimum test count checked
C2      declared artifact families resolved on disk; JSON parsed,   WITHHELD/FAIL
        PNG magic + IEND trailer verified, sha256 recorded
C3      regression suites of pre-existing domain modules; every     WITHHELD/FAIL
        required family needs >=1 green suite
C4      campaign summary report: six elements, embedded figures     WITHHELD/FAIL
        exist as real PNG, explicit "<3%" error verdict
C5      zero domain-semantics scan: no forbidden kernel imports,    FAIL
        no unfilled markers, no domain vocabulary in logic
======  ==========================================================  ==============

``overall_status`` is ``PASS`` only when all five are ``PASS``.  A ``WITHHELD``
criterion (missing upstream evidence) still forces ``FAIL``: an undelivered node
is never accepted by silence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import criteria as K
from . import spec as S

GATE_SCHEMA = "p34_campaign_gate_verdict.v1"
VERDICT_JSON = "gate_verdict.json"
VERDICT_MD = "gate_verdict.md"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_sha(root: Path, paths) -> list[dict]:
    out = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        try:
            rel = str(p.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            rel = str(p)
        out.append({"path": rel, "sha256": sha256_file(p)})
    return out


def run_gate(collection: dict,
             regression: dict | None = None,
             report_path: Path | None = None,
             pkg_dir: Path | None = None,
             deliverables: list[Path] | None = None,
             repo_root: Path | None = None,
             out_dir: Path | None = None) -> dict:
    pkg_dir = Path(pkg_dir) if pkg_dir else Path(__file__).resolve().parent
    repo_root = Path(repo_root) if repo_root else pkg_dir.parents[3]
    deliverables = list(deliverables or [])
    nodes = collection.get("nodes") or []

    verdicts = [
        K.evaluate_c1(nodes),
        K.evaluate_c2(nodes),
        K.evaluate_c3(regression),
        K.evaluate_c4(report_path, repo_root),
        K.evaluate_c5(pkg_dir, deliverables),
    ]
    statuses = [v.status for v in verdicts]
    overall = "PASS" if all(s == K.PASS for s in statuses) else "FAIL"
    blocking = [v.criterion_id for v in verdicts if v.status != K.PASS]
    withheld = [v.criterion_id for v in verdicts if v.status == K.WITHHELD]

    verdict = {
        "schema": GATE_SCHEMA,
        "gate_id": S.GATE_ID,
        "campaign_id": S.CAMPAIGN_ID,
        "campaign_title": S.CAMPAIGN_TITLE,
        "overall_status": overall,
        "blocking_criteria": blocking,
        "withheld_criteria": withheld,
        "criteria": [v.as_dict() for v in verdicts],
        "criterion_summary": {v.criterion_id: v.status for v in verdicts},
        "node_table": [{"ext_id": n["ext_id"], "mission_id": n["mission_id"],
                        "repo": n["repo"], "title": n["title"],
                        "worktree": n.get("worktree"),
                        "delivered": bool(n.get("delivered")),
                        "n_gaps": len(n.get("gaps") or []),
                        "gaps": n.get("gaps") or [],
                        "git": n.get("git"),
                        "pytest_passed": (n.get("pytest") or {}).get("passed", 0),
                        "pytest_green": (n.get("pytest") or {}).get("green", False)}
                       for n in nodes],
        "collection": {
            "utc": collection.get("utc"),
            "arch": collection.get("arch"),
            "root_template": collection.get("root_template"),
            "summary": collection.get("summary"),
        },
        "regression": regression or None,
        "report_path": str(report_path) if report_path else None,
        "provenance": {
            "collection_index": _rel_sha(repo_root, [Path(out_dir or ".") / "collection_index.json"]),
            "node_evidence": _rel_sha(repo_root, sorted((Path(out_dir or ".") / "nodes").glob("*.json")))
            if out_dir else [],
            "regression": _rel_sha(repo_root, [Path(regression["path"])]) if regression and regression.get("path") else [],
            "report": _rel_sha(repo_root, [report_path]) if report_path else [],
        },
        "limitations": [
            "C1 evaluates the pytest runs recorded in this collection pass; a node that changes "
            "its branch afterwards is not covered until the gate is re-run.",
            "C2 proves format validity (JSON parses, PNG magic+IEND) and byte-level provenance, "
            "not physical correctness of the numbers inside.",
            "C3 covers the regression families listed in spec.REQUIRED_REGRESSION_FAMILIES; "
            "families without a recorded suite are WITHHELD, never assumed green.",
            "C4 is a structural check of the campaign report (sections, figures, explicit error "
            "verdict), not a peer review of its physics.",
            "C5 proves absence of kernel imports / unfilled markers / domain vocabulary in the "
            "gate logic; it cannot prove absence of semantic leakage in prose.",
        ],
    }

    md = render_markdown(verdict)
    verdict["markdown"] = md
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / VERDICT_JSON).write_text(
            json.dumps({k: v for k, v in verdict.items() if k != "markdown"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        (out / VERDICT_MD).write_text(md, encoding="utf-8")
    return verdict


def render_markdown(verdict: dict) -> str:
    icon = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WITHHELD": "⏸ WITHHELD"}
    lines = [
        f"# {S.GATE_ID} 全战役集成验收门裁决（自动生成）",
        "",
        f"- schema: `{verdict['schema']}`",
        f"- 战役: {verdict['campaign_id']} — {verdict['campaign_title']}",
        f"- 采集时间 (UTC): `{verdict['collection']['utc']}`  架构: `{verdict['collection']['arch']}`",
        f"- **总裁决 (overall_status): `{verdict['overall_status']}`**",
        f"- 阻塞判据: `{verdict['blocking_criteria']}`  证据不足判据: `{verdict['withheld_criteria']}`",
        "",
        "## 1. 判据总览",
        "",
        "| 判据 | 内容 | 裁决 | 说明 |",
        "|---|---|---|---|",
    ]
    for c in verdict["criteria"]:
        lines.append(f"| {c['criterion_id']} | {c['label']} | {icon.get(c['status'], c['status'])} | {c['detail']} |")

    lines += ["", "## 2. 节点交付状态", "",
              "| 节点 | mission | 仓库 | pytest 通过 | 交付 | 缺口数 |",
              "|---|---|---|---|---|---|"]
    for n in verdict["node_table"]:
        lines.append(f"| {n['ext_id']} | {n['mission_id']} | {n['repo']} | {n['pytest_passed']} | "
                     f"{'✅' if n['delivered'] else '—'} | {n['n_gaps']} |")
    lines += ["", "### 2.1 未闭合缺口", ""]
    any_gap = False
    for n in verdict["node_table"]:
        if n["gaps"]:
            any_gap = True
            lines.append(f"- **{n['ext_id']}**: " + "; ".join(n["gaps"]))
    if not any_gap:
        lines.append("- 无")

    lines += ["", "## 3. C3 回归（既有模块）", ""]
    reg = verdict.get("regression")
    if reg and reg.get("suites"):
        lines += ["| 家族 | 目标 | 仓库 | 结果 | 通过 | 失败 | 错误 | 架构 |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in verdict["criteria"][2]["evidence"]["suites"]:
            lines.append(f"| {s['family']} | {str(s['target'])[:60]} | {s['repo']} | "
                         f"{'✅' if s['green'] else '❌'} | {s['passed']} | {s['failed']} | "
                         f"{s['errors']} | {s['arch']} |")
    else:
        lines.append("- 尚未记录回归运行")

    lines += ["", "## 4. 审计链（provenance）", ""]
    for key, items in verdict["provenance"].items():
        if not items:
            lines.append(f"- `{key}`: —")
        for it in items:
            lines.append(f"- `{key}`: `{it['path']}` sha256=`{it['sha256'][:16]}…`")

    lines += ["", "## 5. 局限与说明", ""]
    lines += [f"- {lim}" for lim in verdict["limitations"]]
    lines.append("")
    return "\n".join(lines)