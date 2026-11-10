"""Publish the P54-W3-N02 report to the platform report API (/tensor/reports).

Logs in with $TENSOR_AGENT_USER/$TENSOR_AGENT_PASS, embeds the campaign report
markdown plus the six figures (base64 PNG), and prints the API response.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m9110")
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
RES = ROOT / "artifacts" / "p54w3_pasternak" / "results.json"
MD = ROOT / "docs" / "reports" / "p54w3_pasternak_foundation_report.md"
FIGDIR = ROOT / "artifacts" / "p54w3_pasternak"


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        BASE + "/" + path, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as exc:  # surface the API error body
        return {"http_error": exc.code, "body": exc.read().decode()[:2000]}


def main():
    tok = post(
        "auth/login",
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]},
    )["access_token"]
    data = json.loads(RES.read_text())
    markdown = MD.read_text()
    figs = []
    for f in data["figures"]:
        p = FIGDIR / f["file"]
        figs.append(
            {
                "name": f["file"],
                "caption": f["caption"],
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            }
        )
    summary = (
        "Pasternak 双参数地基替代 Winkler 占位（关闭 GAP-FEM-P0）：新增闭式 Hetényi/Vlasov Green 函数"
        "（复幅角 Bessel 核，5e-9 vs mpmath）、环积分圆盘场、冰参数标定表；BFS 板 FEM 增 "
        "foundation_model/g_p。判据 a–f 全通过：Winkler 退化 0 差异；FEM vs 解析 r=0 5e-6 / RMS 1.93%"
        "（γ=1）与 4.3e-4 / 0.13%（β=1）；剪切层单调 ρ=-1（7 档）；首裂载荷 γ=1 +37.7%、β=1 +301%；"
        "自由板负控 9e-7（Navier）；32 pytest 全绿。"
    )
    payload = {
        "title": "P54-W3-N02 Pasternak 双参数弹性地基：Hetényi 解析锚 + 冰板弯曲破坏 FEM（关闭 GAP-FEM-P0）",
        "summary": summary,
        "scenario_key": "ice_bending_pasternak_foundation",
        "mission_id": 9110,
        "tags": [
            "P54-W3-N02",
            "GAP-FEM-P0",
            "pasternak",
            "elastic-foundation",
            "ice-sheet",
            "bending-failure",
            "hetenyi",
            "fem-verification",
            "TensorPhysics",
        ],
        "markdown": markdown,
        "figures": figs,
    }
    out = post("tensor/reports", payload, token=tok)
    print(json.dumps(out, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()