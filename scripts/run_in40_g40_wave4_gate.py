#!/usr/bin/env python3
"""Run the wave-4 (IN40-G40) plan-closure gate end to end.

Three deliberately separated modes:

``collect``
    Snapshot the evidence: dump plan 84 (nodes x missions), the per-mission report
    counts and ``tensor_guide_tracker`` from the platform database **read-only**.
    Nothing is interpreted here; every file gets a sha256.

``evaluate``
    Recompute the closing report from the committed snapshot.  Deterministic and
    offline: no network, no database, no clock-dependent behaviour.  Writes the
    machine report (JSON), the paper-grade closing report (Markdown) and figures.

``publish``
    Post the closing report to ``tensor_reports`` (figures embedded as base64)
    and stamp the publish status into the machine report.

Usage::

    python scripts/run_in40_g40_wave4_gate.py collect
    python scripts/run_in40_g40_wave4_gate.py evaluate
    python scripts/run_in40_g40_wave4_gate.py publish
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign import wave4_gate as w4  # noqa: E402

OUT = ROOT / "artifacts" / "in40_g40_wave4_gate"
RAW = OUT / "raw"
FIGS = OUT / "figures"
REPORT_JSON = OUT / "wave4_gate_report.json"
DOC_MD = ROOT / "docs" / "IN40_G40_WAVE4_GATE.md"
SUBMISSION_DIR = RAW / "acceptance"

PLAN_ID = w4.PLAN_ID
GATE_MISSION_ID = 5460  # overwritten by the ledger row of IN40-G40 during collect
GATE_EXT_ID = w4.GATE_EXT_ID

NODES_SQL = (
    "select json_agg(t) from ("
    "select cn.id as node_id, cn.name, cn.type_hint as node_type, cn.status, cn.mission_id, "
    "cn.dependencies, cn.reproduction_verdict, cn.consecutive_fails, cn.block_reason, "
    "cn.metadata, m.status as mission_status, m.finished_at, "
    "(select count(*) from tensor_reports r where r.mission_id = cn.mission_id) as report_count, "
    "(select coalesce(json_agg(r.id order by r.id), '[]'::json) from tensor_reports r "
    " where r.mission_id = cn.mission_id) as report_ids "
    "from campaign_nodes cn left join tensor_dev_missions m on m.id = cn.mission_id "
    f"where cn.plan_id = {PLAN_ID} order by cn.id) t"
)

PLAN_SQL = (
    "select json_build_object('id', id, 'name', name, 'title', title, 'status', status, "
    "'metadata', metadata, 'plan_version', plan_version) from campaign_plans "
    f"where id = {PLAN_ID}"
)

GUIDE_SQL = (
    "select json_agg(t) from (select metric_key, metric_value, target_value, notes "
    "from tensor_guide_tracker order by metric_key) t"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _psql_json(sql: str) -> object:
    cmd = [
        "psql",
        "-U",
        os.environ.get("TP_PG_USER", "postgres"),
        "-d",
        os.environ.get("TP_PG_DB", "tensorplatform"),
        "-At",
        "-c",
        sql,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return json.loads(out) if out else None


def _psql_columns(table: str) -> set[str]:
    """Column names of ``table`` (the ledger schema drifts; probe, do not assume)."""
    cmd = [
        "psql",
        "-U",
        os.environ.get("TP_PG_USER", "postgres"),
        "-d",
        os.environ.get("TP_PG_DB", "tensorplatform"),
        "-At",
        "-c",
        f"select column_name from information_schema.columns where table_name = '{table}'",
    ]
    return set(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split())


# --------------------------------------------------------------------------- collect
def collect() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cols = _psql_columns("campaign_nodes")
    nodes_sql = NODES_SQL
    if "type_hint" not in cols:
        nodes_sql = nodes_sql.replace("cn.type_hint as node_type", "cn.node_type")
    nodes = _psql_json(nodes_sql) or []
    plan = _psql_json(PLAN_SQL)
    guide = _psql_json(GUIDE_SQL) or []

    files = {
        "nodes_plan84.json": nodes,
        "plan.json": plan,
        "guide_tracker.json": guide,
    }
    provenance = {
        "collected_at": _now(),
        "plan_id": PLAN_ID,
        "read_only": True,
        "sql": {"nodes": nodes_sql, "plan": PLAN_SQL, "guide": GUIDE_SQL},
        "files": {},
    }
    for name, payload in files.items():
        path = RAW / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        provenance["files"][name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    (RAW / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    gate_row = next((n for n in nodes if str((n.get("metadata") or {}).get("ext_id")) == GATE_EXT_ID), None)
    print(
        json.dumps(
            {
                "nodes": len(nodes),
                "gate_mission_id": (gate_row or {}).get("mission_id"),
                "guide_metrics": len(guide),
                "files": provenance["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# --------------------------------------------------------------------------- evaluate
def _load_submissions() -> dict[int, dict]:
    """Optional per-node acceptance JSONs (``raw/acceptance/<ext_id>.json``)."""
    out: dict[int, dict] = {}
    if not SUBMISSION_DIR.exists():
        return out
    for path in sorted(SUBMISSION_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        node_id = int(payload["node_id"])
        payload.setdefault("artifacts", [str(path.relative_to(ROOT))])
        out[node_id] = payload
    return out


def _limitations(nodes) -> list[str]:
    """One limitation entry per open node (machine-checked by gate check G4)."""
    lines: list[str] = []
    for n in nodes:
        if n.verdict is w4.Verdict.ACCEPT:
            continue
        cause = n.root_cause.kind.value if n.root_cause else "unclassified"
        lines.append(
            f"{n.ext_id} 未过门：节点裁决 `{n.verdict.value}`，归因 `{cause}`；"
            f"campaign_nodes.status=`{n.node_status}`、mission {n.mission_id} "
            f"status=`{n.mission_status}`、关联报告 {n.report_count} 篇、"
            f"reproduction_verdict=`{n.reproduction_verdict}`；"
            f"判据 {sum(1 for c in n.criteria if c.verdict is w4.Verdict.ACCEPT)}/{len(n.criteria)} 通过"
            f"（{sum(1 for c in n.criteria if c.verdict is w4.Verdict.WITHHELD)} 条 withheld）。"
        )
    lines.append(
        "本轮核销门在节点仍 inflight 的证据截止时刻裁决：缺席证据记 withheld 而非通过，"
        "因此通过率 0% 表达的是「尚无证据」，不是「物理结论错误」。"
    )
    lines.append(
        "guide_tracker 的 progress 是代理口径（如首次相关性/无人值守用去重案例数代理），"
        "不能直接等同工程覆盖率，下一波提案须先校准口径再定目标。"
    )
    lines.append(
        "跨求解器一致性（N10）与推进链不变量（N06）在本轮无独立探针复核，"
        "平台侧 API 只能作为旁证，不能替代节点自证。"
    )
    return lines


def _next_wave_seeds(nodes, gaps, acceptance) -> list[str]:
    seeds: list[str] = []
    for g in gaps:
        if g.unmet:
            seeds.append(
                f"补齐 `{g.metric_key}`（{g.metric_value:g}/{g.target_value:g}，progress {g.progress:.1%}）："
                f"由未过门节点 {', '.join(acceptance.open[:3]) or '—'} 的能力缺口反推提案"
            )
    for n in nodes:
        if n.verdict is w4.Verdict.ACCEPT:
            continue
        seeds.append(
            f"{n.ext_id}（{n.repo}）复跑提案：目标为把 {len(n.criteria)} 条判据全部测出并递交 "
            f"acceptance JSON（当前 {n.report_count} 篇报告、status={n.node_status}）"
        )
    return seeds[:20]


def _figures(report) -> list[dict]:
    """Three gate figures; ASCII labels only (no CJK font dependency)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGS.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    cmap = {
        w4.Verdict.ACCEPT.value: 3.0,
        w4.Verdict.PARTIAL.value: 2.0,
        w4.Verdict.FAIL.value: 1.0,
        w4.Verdict.WITHHELD.value: 0.0,
    }

    # -- (1) acceptance matrix heatmap -------------------------------------
    matrix = report.matrix
    ncols = max(len(m["rows"]) for m in matrix)
    grid = np.full((len(matrix), ncols), np.nan)
    for i, m in enumerate(matrix):
        for j, row in enumerate(m["rows"]):
            grid[i, j] = cmap[row["verdict"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=120)
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f"C{j + 1}" for j in range(ncols)], fontsize=7)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([m["ext_id"].replace("IN40-", "") for m in matrix], fontsize=7)
    for i, m in enumerate(matrix):
        for j, row in enumerate(m["rows"]):
            ax.text(j, i, row["verdict"][0].upper(), ha="center", va="center", fontsize=6)
    ax.set_title(
        f"IN40 acceptance matrix: {report.matrix_shape[0]} nodes x {report.matrix_shape[1]} criteria\n"
        f"(A=accept P=partial F=fail W=withheld) pass rate {report.pass_rate:.0%}",
        fontsize=8,
    )
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 3)), ax=ax,
                 ticks=[0, 1, 2, 3], label="withheld / fail / partial / accept")
    fig.tight_layout()
    p1 = FIGS / "fig_w4_matrix.png"
    fig.savefig(p1)
    plt.close(fig)
    out.append({"name": "fig_w4_matrix.png", "path": str(p1.relative_to(ROOT)),
                "caption": "10 节点 x 判据验收矩阵：全部格子为 withheld（无提交），故矩阵表达的是证据空缺而非物理失败。"})

    # -- (2) pass rate vs threshold ----------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=120)
    bars = ax.bar(["passed", "open"], [report.acceptance.pass_count,
                                       report.acceptance.total - report.acceptance.pass_count],
                  color=["#2e7d32", "#b0bec5"])
    ax.axhline(report.threshold * report.acceptance.total, color="#c62828", ls="--",
               label=f"{report.threshold:.0%} threshold ({report.threshold * report.acceptance.total:.1f} nodes)")
    ax.set_ylabel("nodes")
    ax.set_title(f"Plan {report.plan_id}: {report.acceptance.decision.value}", fontsize=9)
    ax.legend(fontsize=7)
    for rect, val in zip(bars, [report.acceptance.pass_count,
                                report.acceptance.total - report.acceptance.pass_count]):
        ax.text(rect.get_x() + rect.get_width() / 2, val, str(val), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p2 = FIGS / "fig_w4_passrate.png"
    fig.savefig(p2)
    plt.close(fig)
    out.append({"name": "fig_w4_passrate.png", "path": str(p2.relative_to(ROOT)),
                "caption": f"计划级裁决：通过 {report.acceptance.pass_count}/{report.acceptance.total}，"
                           f"门槛 {report.threshold:.0%}（虚线），裁决 {report.acceptance.decision.value}。"})

    # -- (3) guide tracker progress ----------------------------------------
    gaps = report.guide_gaps
    if gaps:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
        keys = [g.metric_key.replace("guide.", "") for g in gaps]
        vals = [g.progress for g in gaps]
        ax.barh(range(len(gaps)), vals, color=["#c62828" if v < 0.5 else "#f9a825" if v < 1 else "#2e7d32" for v in vals])
        ax.set_yticks(range(len(gaps)))
        ax.set_yticklabels(keys, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.axvline(1.0, color="#37474f", ls="--", lw=1)
        ax.set_xlabel("progress = value / target")
        ax.set_title("tensor_guide_tracker gaps (next-wave proposal raw material)", fontsize=9)
        fig.tight_layout()
        p3 = FIGS / "fig_w4_guide_gaps.png"
        fig.savefig(p3)
        plt.close(fig)
        out.append({"name": "fig_w4_guide_gaps.png", "path": str(p3.relative_to(ROOT)),
                    "caption": "guide_tracker 各指标 progress：红=progress<0.5，黄=未达标，绿=达标；虚线为目标 1.0。"})
    return out


def evaluate() -> dict:
    if not (RAW / "nodes_plan84.json").exists():
        raise SystemExit("run `collect` first: raw snapshot missing")
    nodes_raw = json.loads((RAW / "nodes_plan84.json").read_text(encoding="utf-8"))
    plan = json.loads((RAW / "plan.json").read_text(encoding="utf-8"))
    guide_raw = json.loads((RAW / "guide_tracker.json").read_text(encoding="utf-8"))
    prov = json.loads((RAW / "provenance.json").read_text(encoding="utf-8"))

    submissions = _load_submissions()
    entries = [n for n in nodes_raw if str((n.get("metadata") or {}).get("ext_id")) != GATE_EXT_ID]
    gate_row = next((n for n in nodes_raw if str((n.get("metadata") or {}).get("ext_id")) == GATE_EXT_ID), None)
    nodes = [w4.node_from_ledger(e, submission=submissions.get(int(e["node_id"]))) for e in entries]
    nodes.sort(key=lambda n: n.ext_id)

    gaps = w4.guide_gaps_from_rows(guide_raw)
    pre_acceptance = w4.decide_plan(nodes)
    limitations = _limitations(nodes)
    seeds = _next_wave_seeds(nodes, gaps, pre_acceptance)

    audit_chain = [
        {"step": "1-collect", "artifact": f"raw/nodes_plan84.json", "value": prov["files"]["nodes_plan84.json"]["sha256"]},
        {"step": "1-collect", "artifact": "raw/plan.json", "value": prov["files"]["plan.json"]["sha256"]},
        {"step": "1-collect", "artifact": "raw/guide_tracker.json", "value": prov["files"]["guide_tracker.json"]["sha256"]},
        {"step": "1-collect", "artifact": "raw/provenance.json", "value": _sha256(RAW / "provenance.json")},
        {"step": "2-parse", "artifact": f"criteria from campaign_nodes.metadata (plan {PLAN_ID})", "value": f"{sum(len(n.criteria) for n in nodes)} criteria"},
        {"step": "3-decide", "artifact": f"pass rule rate>={w4.PASS_THRESHOLD}", "value": pre_acceptance.reason},
        {"step": "4-evaluate", "artifact": "module tensorphysics.campaign.wave4_gate", "value": f"plan_id={PLAN_ID}, gate_mission_id={(gate_row or {}).get('mission_id')}"},
    ]

    report = w4.build_report(
        plan_id=PLAN_ID,
        plan_name=str(plan.get("name") or ""),
        gate_ext_id=GATE_EXT_ID,
        gate_mission_id=int((gate_row or {}).get("mission_id") or GATE_MISSION_ID),
        cutoff=prov["collected_at"],
        nodes=nodes,
        guide_gaps=gaps,
        limitations=limitations,
        next_wave=seeds,
        audit_chain=audit_chain,
    )

    figures = _figures(report)
    artifacts = [
        {"kind": "machine_report", "path": str(REPORT_JSON.relative_to(ROOT))},
        {"kind": "closing_report", "path": str(DOC_MD.relative_to(ROOT))},
    ]
    for name, meta in prov["files"].items():
        artifacts.append({"kind": "raw_snapshot", "path": f"raw/{name}", "sha256": meta["sha256"], "bytes": meta["bytes"]})
    for fig in figures:
        artifacts.append({"kind": "figure", "path": fig["path"], "name": fig["name"]})
    report = dataclasses.replace(report, figures=tuple(figures), artifacts=tuple(artifacts))
    OUT.mkdir(parents=True, exist_ok=True)
    DOC_MD.parent.mkdir(parents=True, exist_ok=True)
    payload = w4.write_report(report, json_path=str(REPORT_JSON), md_path=str(DOC_MD))

    summary = {
        "final_verdict": payload["final_verdict"],
        "plan_decision": payload["plan_acceptance"]["plan_decision"],
        "pass_rate": payload["plan_acceptance"]["pass_rate"],
        "pass_count": payload["plan_acceptance"]["pass_count"],
        "threshold": payload["threshold"],
        "nodes": payload["matrix_shape"]["nodes"],
        "criteria": payload["matrix_shape"]["criteria"],
        "critical_failures": payload["plan_acceptance"]["critical_failures"],
        "checks": {c["cid"]: c["passed"] for c in payload["checks"]},
        "report_digest": payload["report_digest"],
        "plan_update_sql": w4.plan_update_sql(report),
        "artifacts": {
            "report_json": str(REPORT_JSON.relative_to(ROOT)),
            "closing_report_md": str(DOC_MD.relative_to(ROOT)),
            "figures": [f["path"] for f in figures],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# --------------------------------------------------------------------------- publish
def _login() -> str:
    import urllib.request

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    body = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(
        base + "/auth/login", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data.get("access_token") or data["token"]


def publish() -> None:
    import urllib.error
    import urllib.request

    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    md = DOC_MD.read_text(encoding="utf-8")
    figures = []
    for fig in payload.get("figures", []):
        path = ROOT / fig["path"]
        if not path.exists():
            continue
        png = path.read_bytes()
        figures.append(
            {
                "name": fig["name"],
                "caption": fig["caption"],
                "png_base64": base64.b64encode(png).decode(),
            }
        )
    body = {
        "title": f"IN40-G40 波4核销门 · Plan 40 收官报告（{payload['final_verdict']} / "
                 f"{payload['plan_acceptance']['plan_decision']} / 通过率 {payload['plan_acceptance']['pass_rate']:.0%}）",
        "summary": (
            f"Plan 40（plan_id={payload['plan_id']}，{payload['matrix_shape']['nodes']} 节点 / "
            f"{payload['matrix_shape']['criteria']} 条判据）核销门裁决："
            f"final_verdict={payload['final_verdict']}，计划裁决={payload['plan_acceptance']['plan_decision']}，"
            f"通过率 {payload['plan_acceptance']['pass_count']}/{payload['plan_acceptance']['total_nodes']}"
            f"（阈值 {payload['threshold']:.0%}），critical failure={payload['plan_acceptance']['critical_failures'] or 'none'}。"
            "缺席证据一律 withheld，不上抬为通过。"
        ),
        "scenario_key": "campaign_gate_in40_g40",
        "mission_id": int(os.environ.get("MISSION_ID", "5546")),
        "tags": ["campaign-gate", "IN40-G40", "plan-40", "lineage-inheritance", "acceptance-matrix"],
        "markdown": md,
        "figures": figures,
        "status": "published",
    }
    blob = json.dumps(body, ensure_ascii=False).encode()
    token = _login()
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=blob,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        if "uq_reports_mission_md5" in detail:  # idempotent re-publish of the same body
            existing = _find_published(body["mission_id"], md, body["title"], token, base)
            if existing:
                _stamp_published(existing, len(blob))
                print(json.dumps({"bytes": len(blob), "report_id": existing,
                                  "note": "already published (identical markdown digest)"}))
                return
        raise SystemExit(f"publish failed: HTTP {exc.code}: {detail[:1200]}")
    report_id = result.get("id") if isinstance(result, dict) else None
    _stamp_published(report_id, len(blob))
    print(json.dumps({"bytes": len(blob), "report_id": report_id, "response": result}, ensure_ascii=False)[:1200])


def _find_published(mission_id: int, md: str, title: str, token: str, base: str) -> int | None:
    """Existing report id whose markdown matches ours (idempotent publish).

    The list endpoint projects only summaries, so candidates are matched on title
    and then their full body is fetched and hash-compared: two runs that agree on
    verdict but differ in cutoff must not be conflated.
    """
    import hashlib
    import urllib.request

    want = hashlib.md5(md.encode("utf-8")).hexdigest()
    req = urllib.request.Request(
        f"{base}/tensor/reports?mission_id={mission_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        rows = json.loads(resp.read().decode())
    items = rows if isinstance(rows, list) else rows.get("items", [])
    for row in items:
        if row.get("mission_id") not in (None, mission_id):
            continue
        if title and row.get("title") and row["title"] != title:
            continue
        detail_req = urllib.request.Request(
            f"{base}/tensor/reports/{row['id']}", headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(detail_req, timeout=120) as resp:
            detail = json.loads(resp.read().decode())
        if hashlib.md5(str(detail.get("markdown", "")).encode("utf-8")).hexdigest() == want:
            return int(row["id"])
    return None


def _stamp_published(report_id: int | None, nbytes: int) -> None:
    """Record the publish receipt in the machine report and append it to the MD.

    Only non-digest fields are touched (``publish_status`` / ``publish_id``), and
    the Markdown gets an append-only receipt line, so the evidence digest and the
    integrity check stay valid after publication.
    """
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    payload["publish_status"] = "published"
    payload["publish_id"] = report_id
    payload["publish_bytes"] = nbytes
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    receipt = (
        f"\n> 发布回执：`tensor_reports.id={report_id}`（{nbytes} bytes，含 3 张图内嵌），"
        f"报告摘要 digest `{payload['report_digest']}` 未变——发布不修改证据裁决。\n"
    )
    with open(DOC_MD, "a", encoding="utf-8") as fh:
        fh.write(receipt)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("collect", "evaluate", "publish"):
        print(__doc__)
        return 2
    {"collect": collect, "evaluate": evaluate, "publish": publish}[argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))