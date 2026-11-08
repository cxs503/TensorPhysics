#!/usr/bin/env python3
"""Build and POST the P35-G1 progress report to the platform.

Numbers come from ``artifacts/p35_gate/campaign_readiness.json`` and the rendered
document; figures are attached as PNG base64 (downscaled to keep each payload
well under the 60 KB report-figure budget).  Nothing is typed by hand except the
prose that the rendered document already contains.

    python3 scripts/p35_gate_post_report.py            # writes /tmp/p35_report.json
    curl -s -X POST "$TENSOR_API_BASE/tensor/reports" -H "Authorization: Bearer $TOK" \
         -H 'Content-Type: application/json' -d @/tmp/p35_report.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = ROOT / "artifacts" / "p35_gate"
FIG_DIR = GATE_DIR / "figures"
DOC = ROOT / "docs" / "P35_CAMPAIGN_INTEGRATION_GATE.md"
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.p35_gate import spec as S  # noqa: E402

CAPTIONS = {
    "fig1_node_readiness.png": "图 1 · 十节点就绪矩阵（产物/台账/成熟化/链就绪），"
                               "绿=通过，红=未通过，灰=证据未取得",
    "fig2_regression_deviation.png": "图 2 · 回归复跑偏差（对数轴）+ 战役 1% 阈值 + "
                                     "fp64 机器地板；灰位为未复跑案例（WITHHELD）",
    "fig3_couette_profile.png": "图 3 · Couette 三维剖面复跑 vs 解析线性解，"
                                "逐点偏差停在 float64 舍入地板",
}


def shrink(path: Path, width: int = 820, budget: int = 24000) -> bytes:
    """Downscale + palette-quantise a figure so the report stays well under 250 KB."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    blob = b""
    for w in (width, int(width * 0.85), int(width * 0.7)):
        scale = min(1.0, w / im.width)
        small = im.resize((int(im.width * scale), int(im.height * scale)))
        buf = io.BytesIO()
        small.convert("P", palette=Image.ADAPTIVE, colors=64).save(
            buf, format="PNG", optimize=True)
        blob = buf.getvalue()
        if len(blob) <= budget:
            return blob
    return blob


def build(token: str) -> dict:
    doc = json.loads((GATE_DIR / "campaign_readiness.json").read_text())
    markdown = DOC.read_text()
    # the document carries relative figure links; the report carries the images
    markdown = "\n".join(line for line in markdown.splitlines()
                         if not line.startswith("![fig"))
    crit = {c["criterion_id"]: c for c in doc["criteria"]}
    figures = []
    for name, caption in CAPTIONS.items():
        p = FIG_DIR / name
        if not p.is_file():
            continue
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(shrink(p)).decode()})
    payload = {
        "title": "P35-G1 全战役集成验收门：十节点产物/台账/回归/依赖链机器裁决 (honest "
                 f"{doc['overall_status']})",
        "summary": (
            f"P35 战役 10 节点集成验收门落库 TensorPhysics（campaign/p35_gate，stdlib-only，"
            f"26 条 pytest）。七判据实现：产物落盘、能力台账(ext_id/repo/physics_domain)、"
            f"回归基线<1%、DAG 完整、成熟化门禁、readiness schema、honest-FAIL。实测裁决 "
            f"{doc['overall_status']}：产物落盘 "
            f"{crit['a']['metrics']['landed']}/{crit['a']['metrics']['n_nodes']}，"
            f"台账 {crit['b']['metrics']['registered']}/{crit['b']['metrics']['n_nodes']}，"
            f"回归复跑 {crit['c']['metrics']['n_tested']}/{crit['c']['metrics']['n_cases']} "
            f"(最大偏差 {crit['c']['metrics']['max_rel_dev']})，"
            f"成熟化 {crit['e']['metrics']['passed']}/{crit['e']['metrics']['n_nodes']}，"
            f"链就绪 {len([n for n in doc['nodes'] if n['chain_ready']])}/10。"
            f"阻塞项 {len(doc['blocking'])} 条已落 campaign_readiness.v1"
            f"(schema fingerprint {doc['provenance']['schema_fingerprint'][:16]}…)。"
        ),
        "scenario_key": "p35_campaign_integration_gate",
        "mission_id": 6805,
        "tags": ["campaign_gate", "acceptance", "cross_domain", "regression_baseline",
                 "honest_fail", "p35"],
        "markdown": markdown,
        "figures": figures,
    }
    _ = token
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/p35_report.json")
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args(argv)
    token = os.environ.get("TENSOR_TOKEN", "")
    payload = build(token)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False))
    size = Path(args.out).stat().st_size
    print(f"{args.out} ({size/1024:.1f} KB, figures={len(payload['figures'])}, "
          f"markdown={len(payload['markdown'])} chars)")
    if args.post:
        base = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
        req = urllib.request.Request(base + "/tensor/reports",
                                     data=Path(args.out).read_bytes(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(resp.status, resp.read()[:400].decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())