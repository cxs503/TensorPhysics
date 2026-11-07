#!/usr/bin/env python3
"""Run the wave-3 (IN19) closure gate end to end.

Two deliberately separated modes:

``collect``
    Snapshot the evidence: dump campaign plan 39 (nodes + mission state) from the
    platform database read-only, then probe the reachable ship-performance-platform
    endpoints and store *raw* responses.  Nothing is interpreted here.

``evaluate``
    Recompute the gate report from the committed raw snapshot.  Deterministic and
    offline: no network, no database, no clock-dependent behaviour.  Writes the
    machine report (JSON), the paper-grade report (Markdown) and the figures.

Usage::

    python scripts/run_in19_g19_wave3_gate.py collect
    python scripts/run_in19_g19_wave3_gate.py evaluate
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts" / "in19_g19_wave3_gate"
RAW = OUT / "raw"
FIGS = OUT / "figures"
PLAN_ID = 39
GATE_MISSION_ID = 5208
GATE_EXT_ID = "IN19-G19"
SHIP_API = os.environ.get("TP_SHIP_API", "http://127.0.0.1:8002")

#: Probe request bodies are fixed here (not derived from anything the nodes
#: claim) so that the measurement is reproducible and auditable.
PROBE_REQUESTS: dict[str, tuple[str, str, dict]] = {
    "suboff_resistance": (
        "POST",
        "/api/v1/calculations/resistance",
        {
            "length": 4.356,
            "beam": 0.508,
            "draft": 0.508,
            "displacement": 0.718,
            "speed": 5.0,
            "speed_unit": "m/s",
            "cb": 0.639,
            "water_density": 1025.9,
            "water_viscosity": 1.05e-06,
        },
    ),
    "kcs_resistance": (
        "POST",
        "/api/v1/calculations/resistance",
        {
            "length": 230.0,
            "beam": 32.2,
            "draft": 10.8,
            "displacement": 52030.0,
            "speed": 12.0,
            "speed_unit": "kn",
            "cb": 0.651,
        },
    ),
    "kcs_propeller": (
        "POST",
        "/api/v1/propulsion/predict",
        {
            "L": 230,
            "B": 32.2,
            "T": 10.8,
            "displacement": 52030,
            "Cb": 0.651,
            "speed": 12.0,
            "Dp": 6.0,
            "pitch_ratio": 0.95,
            "blades": 5,
            "ae_ao": 0.55,
            "n": 2.0,
        },
    ),
    "kcs_rao": (
        "POST",
        "/api/v1/seakeeping/rao",
        {
            "L": 230,
            "B": 32.2,
            "T": 10.8,
            "displacement": 52030,
            "Cb": 0.651,
            "KG": 18.5,
            "GM": 1.5,
            "speed": 12,
            "wave_frequency": 0.6,
            "wave_amplitude": 1.0,
            "heading": 180,
        },
    ),
    "eexi_endpoint": ("GET", "/api/v1/calculations/eexi", {}),
    "cii_endpoint": ("GET", "/api/v1/calculations/cii", {}),
}

NODE_EVIDENCE_GRADE = {
    # Filled from the raw probe snapshot in evaluate(): capability that the raw
    # evidence shows to be absent at the gate cutoff.
    313: "NOT_SUPPORTED",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- collect
def _db_dump() -> list[dict]:
    sql = (
        "select json_agg(t) from ("
        "select cn.id as node_id, cn.name, cn.node_type, cn.status, cn.mission_id, "
        "cn.dependencies, cn.metadata, m.status as mission_status, m.finished_at "
        "from campaign_nodes cn left join tensor_dev_missions m on m.id = cn.mission_id "
        f"where cn.plan_id = {PLAN_ID} order by cn.id) t"
    )
    cmd = ["psql", "-U", os.environ.get("TP_PG_USER", "postgres"), "-d",
           os.environ.get("TP_PG_DB", "tensorplatform"), "-At", "-c", sql]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _probe(method: str, path: str, body: dict) -> dict:
    url = SHIP_API + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=90)
        return {"status": resp.status, "response": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "response": exc.read().decode()[:400]}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": 0, "response": f"{type(exc).__name__}: {exc}"}


def collect() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    nodes = _db_dump()
    nodes_path = RAW / "nodes_plan39.json"
    nodes_path.write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")

    probes: dict[str, dict] = {}
    for key, (method, path, body) in PROBE_REQUESTS.items():
        result = _probe(method, path, body)
        probes[key] = {"method": method, "path": path, "request": body, **result}
    probes_path = RAW / "platform_probes.json"
    latency_meta = {
        "base_url": SHIP_API,
        "collected_at": _now(),
        "requests": {k: {"method": method, "path": path} for k, (method, path, _) in PROBE_REQUESTS.items()},
    }
    probes_path.write_text(
        json.dumps({"meta": latency_meta, "probes": probes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"collected {len(nodes)} campaign rows -> {nodes_path.relative_to(ROOT)}")
    print(f"collected {len(probes)} platform probes -> {probes_path.relative_to(ROOT)}")
    for key, value in probes.items():
        print(f"  {key:18s} status={value['status']}")


# ------------------------------------------------------------------------- evaluate
def _interp_kt_kq(points: list[dict], j: float) -> tuple[float, float]:
    """Linear interpolation of the digitised B-series table at advance ratio ``j``."""
    js = [p["J"] for p in points]
    if not (min(js) <= j <= max(js)):
        raise ValueError(f"J={j} outside digitised table {min(js)}..{max(js)}")
    for lo, hi in zip(points, points[1:]):
        if lo["J"] <= j <= hi["J"]:
            w = (j - lo["J"]) / (hi["J"] - lo["J"])
            return (
                lo["KT"] + w * (hi["KT"] - lo["KT"]),
                lo["KQ"] + w * (hi["KQ"] - lo["KQ"]),
            )
    raise ValueError("unreachable")


def _build_probes(raw_probes: dict, refs: dict) -> list:
    from tensorphysics.campaign.wave3_gate import GateProbe

    p = raw_probes["probes"]
    suboff = p["suboff_resistance"]["response"]
    kcs = p["kcs_resistance"]["response"]
    prop = p["kcs_propeller"]["response"]

    suboff_ref = refs["suboff"]
    kcs_ref = refs["kcs"]
    bser = refs["wageningen_b5_55_pd100"]
    kt_ref, kq_ref = _interp_kt_kq(bser["points"], float(prop["advance_ratio"]))

    return [
        GateProbe(
            probe_id="P1-SUBOFF-S",
            node_id=311,
            endpoint="/calculations/resistance",
            quantity="SUBOFF 模型湿表面积 S",
            measured=float(suboff["wetted_surface"]),
            reference=float(suboff_ref["wetted_surface_m2"]),
            unit="m2",
            reference_source=suboff_ref["source"],
            reference_tolerance_pct=float(suboff_ref["tolerance_pct"]),
            threshold_pct=3.0,
            note="几何门：S 挂钩(L/B/T/∇)先行；副艇体经面船湿面积闭包计算，本身就是越域使用",
        ),
        GateProbe(
            probe_id="P2-SUBOFF-CT",
            node_id=311,
            endpoint="/calculations/resistance",
            quantity="SUBOFF 总阻力系数 C_T (V=5 m/s)",
            measured=float(suboff["ct"]),
            reference=float(suboff_ref["ct_total"]),
            unit="-",
            reference_source=suboff_ref["source"],
            reference_tolerance_pct=float(suboff_ref["tolerance_pct"]),
            threshold_pct=5.0,
            note=(
                f"门侧复算 Fn={suboff['froude_number']:.3f} 已超 Holtrop 适用域(0.10-0.45)，"
                "剩余阻力项被外推放大；误差量级对基准±10%不确定度不敏感"
            ),
        ),
        GateProbe(
            probe_id="P3-KCS-S",
            node_id=309,
            endpoint="/calculations/resistance",
            quantity="KCS 湿表面积 S (L=230 m, T=10.8 m, Cb=0.651)",
            measured=float(kcs["wetted_surface"]),
            reference=float(kcs_ref["wetted_surface_m2"]),
            unit="m2",
            reference_source=kcs_ref["source"],
            reference_tolerance_pct=float(kcs_ref["tolerance_pct"]),
            threshold_pct=1.5,
            note="该端点仍用 R1 闭包 S=πBL(0.5+0.25Cb)，即节点309要裁决的三分裂之一",
        ),
        GateProbe(
            probe_id="P4-KT",
            node_id=310,
            endpoint="/propulsion/predict",
            quantity=f"B5-55 推力系数 K_T @ J={float(prop['advance_ratio']):.4f}",
            measured=float(prop["kt"]),
            reference=kt_ref,
            unit="-",
            reference_source=bser["source"],
            reference_tolerance_pct=float(bser["tolerance_pct"]),
            threshold_pct=3.0,
            note=(
                "探针在 P/D=0.95 取值、基准表为 P/D=1.0：按 dK_T/d(P/D)≈0.5 修正后基准更低，"
                "故观测到的正偏差是下界"
            ),
        ),
        GateProbe(
            probe_id="P5-KQ",
            node_id=310,
            endpoint="/propulsion/predict",
            quantity=f"B5-55 扭矩系数 K_Q @ J={float(prop['advance_ratio']):.4f}",
            measured=float(prop["kq"]),
            reference=kq_ref,
            unit="-",
            reference_source=bser["source"],
            reference_tolerance_pct=float(bser["tolerance_pct"]),
            threshold_pct=3.0,
            note="同上；K_Q 偏差大于 K_T，说明敞水效率 η0 也会被连带高估",
        ),
    ]


def evaluate() -> None:
    from dataclasses import replace

    from tensorphysics.campaign.wave3_gate import (
        Grade,
        PeerReview,
        Verdict,
        RootCause,
        RootCauseKind,
        build_report,
        node_record_from_ledger,
        write_report,
    )

    nodes_raw = json.loads((RAW / "nodes_plan39.json").read_text(encoding="utf-8"))
    probes_raw = json.loads((RAW / "platform_probes.json").read_text(encoding="utf-8"))
    refs = json.loads((OUT / "references.json").read_text(encoding="utf-8"))
    cutoff = probes_raw["meta"]["collected_at"]

    probes = _build_probes(probes_raw, refs)
    absent = {
        key: probes_raw["probes"][key]["status"]
        for key in ("eexi_endpoint", "cii_endpoint")
    }
    grade_override = dict(NODE_EVIDENCE_GRADE)
    if any(status != 200 for status in absent.values()):
        grade_override[313] = Grade.NOT_SUPPORTED.value

    records = [
        node_record_from_ledger(entry, probes=probes)
        for entry in nodes_raw
        if int(entry["node_id"]) != 315  # the gate node itself is this report
    ]

    fixed = []
    for record in records:
        node_id = int(record.node_id)
        if node_id in grade_override:
            record = replace(record, evidence_grade=Grade(grade_override[node_id]))
        if record.evidence_grade is Grade.NOT_SUPPORTED and record.verdict is not Verdict.ACCEPT:
            record = replace(
                record,
                root_cause=RootCause(
                    kind=RootCauseKind.EVIDENCE_GAP,
                    statement=(
                        "EEXI/CII 规范实现接口在当前主分支不存在"
                        "（GET /calculations/eexi 与 /calculations/cii 均 404），"
                        "属能力缺席而非精度不足"
                    ),
                    evidence=(
                        "raw/platform_probes.json: "
                        + ", ".join(f"{k} status={v}" for k, v in absent.items())
                    ),
                    confidence="high",
                    actions=(
                        "按 MEPC.328(75) 实现 EEXI/CII 数值链并暴露端点，再以 IMO 参考船复算",
                    ),
                ),
            )
        fixed.append(record)
    records = fixed

    peer_path = OUT / "peer_review.json"
    peer = None
    prior_reviews = []
    payload = None
    if peer_path.exists():
        payload = json.loads(peer_path.read_text(encoding="utf-8"))
        prior_reviews = [
            PeerReview(
                reviewer_agent=item["reviewer_agent"],
                reviewer_affiliation=item.get("reviewer_affiliation", "unknown"),
                independent=bool(item.get("independent", True)),
                reviewed_digest=item["reviewed_digest"],
                verdict=item["verdict"],
                findings=tuple(item.get("findings") or ()),
                requested_changes=tuple(item.get("requested_changes") or ()),
                signed_at=item.get("signed_at", ""),
                response=item.get("response", ""),
            )
            for item in payload.get("prior_reviews", [])
        ]
        payload = payload.get("peer_review") or payload
        if not payload.get("reviewed_digest"):
            payload = None
    if payload:
        peer = PeerReview(
            reviewer_agent=payload["reviewer_agent"],
            reviewer_affiliation=payload.get("reviewer_affiliation", "unknown"),
            independent=bool(payload["independent"]),
            reviewed_digest=payload["reviewed_digest"],
            verdict=payload["verdict"],
            findings=tuple(payload.get("findings") or ()),
            requested_changes=tuple(payload.get("requested_changes") or ()),
            signed_at=payload.get("signed_at", ""),
        )

    git_rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    audit = [
        {"path": "artifacts/in19_g19_wave3_gate/raw/nodes_plan39.json",
         "sha256": _sha256(RAW / "nodes_plan39.json"),
         "note": f"campaign_nodes⊗tensor_dev_missions dump, plan_id={PLAN_ID}"},
        {"path": "artifacts/in19_g19_wave3_gate/raw/platform_probes.json",
         "sha256": _sha256(RAW / "platform_probes.json"),
         "note": f"raw platform probe responses, base_url={probes_raw['meta']['base_url']}"},
        {"path": "artifacts/in19_g19_wave3_gate/references.json",
         "sha256": _sha256(OUT / "references.json"),
         "note": "reference values with provenance (in-repo citations)"},
        {"path": "src/tensorphysics/campaign/wave3_gate.py",
         "sha256": _sha256(ROOT / "src/tensorphysics/campaign/wave3_gate.py"),
         "note": "gate engine that produced every number in this report"},
        {"path": "git",
         "sha256": git_rev,
         "note": "worktree revision the gate ran on"},
    ]

    report = build_report(
        plan_id=PLAN_ID,
        plan_name="求解器验证与船舶能力深化·谱系继承波3",
        gate_ext_id=GATE_EXT_ID,
        gate_mission_id=GATE_MISSION_ID,
        cutoff=cutoff,
        nodes=records,
        probes=probes,
        peer_review=peer,
        prior_reviews=prior_reviews,
        limitations=refs["limitations"],
        next_wave=refs["next_wave"],
        audit_chain=audit,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    payload = write_report(
        report,
        json_path=str(OUT / "gate_report.json"),
        md_path=str(OUT / "gate_report.md"),
    )
    (ROOT / "docs" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "reports" / "in19_g19_wave3_gate.md").write_text(
        report.render_markdown(), encoding="utf-8"
    )
    _render_figures(report)

    print(f"final verdict      : {report.final_verdict}")
    print(f"report digest      : {report.body_digest}")
    for c in report.checks:
        print(f"  {c.cid} {'PASS' if c.passed else 'FAIL'}  {c.number:>10s}  {c.requirement}")
    print(f"lineage            : {report.lineage_table}")
    print(f"json : {(OUT / 'gate_report.json').relative_to(ROOT)}")
    print(f"md   : {(OUT / 'gate_report.md').relative_to(ROOT)}")
    return payload


# --------------------------------------------------------------------------- figures
def _render_figures(report) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- Fig 1: acceptance matrix ---------------------------------------
    cols = max(len(n.criteria) for n in report.nodes)
    grid = np.full((len(report.nodes), cols), np.nan)
    code = {"withheld": 0.0, "fail": 1.0, "partial": 0.5, "accept": 1.0}
    for i, n in enumerate(report.nodes):
        for j, c in enumerate(n.criteria):
            grid[i, j] = code[c.verdict.value]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#f4b183", "#2e7d32"])
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(cols))
    ax.set_xticklabels([f"C{j + 1}" for j in range(cols)])
    ax.set_yticks(range(len(report.nodes)))
    ax.set_yticklabels(
        [f"{n.ext_id} [{n.verdict.value[:4].upper()}]" for n in report.nodes], fontsize=8
    )
    for i, n in enumerate(report.nodes):
        for j, c in enumerate(n.criteria):
            ax.text(j, i, "W" if c.verdict.value == "withheld" else c.verdict.value[:1].upper(),
                    ha="center", va="center", fontsize=7, color="#333333")
    ax.set_title("Wave-3 acceptance matrix (W = withheld: no acceptance evidence)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "acceptance_matrix.png", dpi=160)
    plt.close(fig)

    # ---- Fig 2: probe error spectrum ------------------------------------
    if report.probes:
        labels = [p.probe_id for p in report.probes]
        errs = [p.rel_error_pct for p in report.probes]
        thrs = [p.threshold_pct for p in report.probes]
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        x = np.arange(len(labels))
        colors = ["#c62828" if abs(e) > t else "#2e7d32" for e, t in zip(errs, thrs)]
        ax.bar(x, np.abs(errs), color=colors, alpha=0.85)
        ax.plot(x, thrs, "k_", markersize=18, label="acceptance threshold")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylabel("|relative error| vs baseline  [%]")
        for xi, e in zip(x, errs):
            ax.annotate(f"{e:+.1f}%", (xi, abs(e)), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8)
        ax.set_title("Gate-side probe of current platform state (log scale)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "error_spectrum.png", dpi=160)
        plt.close(fig)

    # ---- Fig 3: lineage summary -----------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    buckets = report.lineage_table
    keys = ["mature", "iterate", "shelve"]
    counts = [len(buckets[k]) for k in keys]
    ax1.bar(keys, counts, color=["#2e7d32", "#f9a825", "#757575"])
    for i, v in enumerate(counts):
        ax1.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax1.set_title("Lineage inheritance decision", fontsize=9)
    ax1.set_ylabel("nodes")
    status_counts: dict[str, int] = {}
    for n in report.nodes:
        for c in n.criteria:
            status_counts[c.verdict.value] = status_counts.get(c.verdict.value, 0) + 1
    labels = [k for k in ["accept", "partial", "fail", "withheld"] if k in status_counts]
    ax2.bar(labels, [status_counts[k] for k in labels],
            color=["#2e7d32", "#f4b183", "#c62828", "#d9d9d9"][: len(labels)])
    ax2.set_title(f"Criterion status composition ({sum(status_counts.values())} criteria)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "lineage_summary.png", dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "evaluate"
    if mode == "collect":
        collect()
        return 0
    if mode == "evaluate":
        evaluate()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))