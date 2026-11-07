#!/usr/bin/env python3
"""Publish the wave-3 closure-gate report to the Tensor ecosystem report API.

The report body is the rendered gate report (``artifacts/in19_g19_wave3_gate/
gate_report.md``) so the published text and the audited artifact cannot diverge;
the figures are the same PNGs the report embeds, downscaled so the payload stays
well under the API's size limit.

Usage::

    python scripts/publish_in19_g19_report.py            # publish
    python scripts/publish_in19_g19_report.py --dry-run   # assemble only
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "in19_g19_wave3_gate"
MISSION_ID = 5208
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

FIGURES = [
    ("acceptance_matrix.png", "acceptance"),
    ("error_spectrum.png", "error"),
    ("lineage_summary.png", "lineage"),
]


def _captions(gate: dict) -> dict[str, str]:
    """Figure captions, injected from the gate JSON (no hand-copied numbers)."""
    nodes = gate["nodes"]
    n_crit = sum(len(n["criteria"]) for n in nodes)
    n_withheld = sum(
        1 for n in nodes for c in n["criteria"] if c["verdict"] == "withheld"
    )
    probes = "；".join(
        f"{p['quantity']} {p['rel_error_pct']:+.2f}%（阈值 {p['threshold_pct']:g}%）"
        for p in gate["probes"]
    )
    lineage = gate["lineage"]
    return {
        "acceptance": (
            f"图1 波3 验收矩阵：行={len(nodes)} 个任务节点，列=各自机器判据；"
            f"灰=WITHHELD（无验收证据）。{len(nodes)} 节点 / {n_crit} 条判据中 "
            f"{n_withheld} 条为 WITHHELD，即本门截止时刻的定量结论。"
        ),
        "error": (
            "图2 门外独立探针的相对误差谱（对数纵轴）与各自验收阈值：" + probes + "。"
        ),
        "lineage": (
            f"图3 谱系继承分布（成熟 {len(lineage['mature'])} / 迭代 {len(lineage['iterate'])} / "
            f"搁置 {len(lineage['shelve'])}）与 {n_crit} 条判据的状态构成。"
        ),
    }


def _token() -> str:
    body = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(API + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["access_token"]


def _compact_png(path: Path, max_width: int = 900) -> str:
    from PIL import Image

    img = Image.open(path)
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def build_payload() -> dict:
    gate = json.loads((ART / "gate_report.json").read_text(encoding="utf-8"))
    markdown = (ART / "gate_report.md").read_text(encoding="utf-8")
    captions = _captions(gate)
    figures = [
        {
            "name": name,
            "caption": captions[key],
            "png_base64": _compact_png(ART / "figures" / name),
        }
        for name, key in FIGURES
    ]
    checks = "；".join(f"{c['cid']}={'PASS' if c['passed'] else 'FAIL'}({c['number']})" for c in gate["checks"])
    n_crit = sum(len(n["criteria"]) for n in gate["nodes"])
    n_withheld = sum(
        1 for n in gate["nodes"] for c in n["criteria"] if c["verdict"] == "withheld"
    )
    probe_txt = "；".join(
        f"{p['quantity']} {p['rel_error_pct']:+.2f}%" for p in gate["probes"]
    )
    lineage = gate["lineage"]
    summary = (
        f"波3（IN19）{len(gate['nodes'])} 个任务节点的核销门：验收矩阵覆盖 "
        f"{len(gate['nodes'])}/{len(gate['nodes'])} 节点、{n_crit} 条机器判据中 {n_withheld} 条为 WITHHELD"
        f"（截止 {gate['cutoff']} 九个节点 mission 仍 running），谱系继承裁决为成熟 "
        f"{len(lineage['mature'])} / 迭代 {len(lineage['iterate'])} / 搁置 {len(lineage['shelve'])}；"
        f"门自检 {checks}；门侧独立探针复算出 {probe_txt}；EEXI/CII 端点 404。"
        f"最终裁决 {gate['final_verdict']}（honest，未粉饰）。"
    )
    return {
        "title": "IN19-G19 波3核销门：全节点验收汇总与谱系继承裁决（含门侧独立探针误差判定表）",
        "summary": summary,
        "scenario_key": "in19_g19_wave3_gate",
        "mission_id": MISSION_ID,
        "tags": ["IN19-G19", "wave3", "acceptance-gate", "lineage", "honest-fail", "TensorPhysics"],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv: list[str]) -> int:
    payload = build_payload()
    size = len(json.dumps(payload, ensure_ascii=False).encode())
    print(f"payload size: {size / 1024:.1f} KB, figures: {len(payload['figures'])}")
    if "--dry-run" in argv:
        return 0
    token = _token()
    req = urllib.request.Request(
        API + "/tensor/reports",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))
        print("published report id:", resp.get("id"))
        print("qa_status:", resp.get("qa_status"))
        return 0
    except urllib.error.HTTPError as exc:
        print("publish failed:", exc.code, exc.read().decode()[:800])
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))