#!/usr/bin/env python3
"""Submit the P366-W2-N01 report to the platform API (figures + numbers from JSON)."""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
RESULTS = json.loads((ROOT / "runs" / "p366w2_n01" / "results.json").read_text())
FIGDIR = ROOT / "artifacts" / "p366w2_n01"
MD = ROOT / "docs" / "reports" / "p366w2_n01_fem_ice_plate.md"


def login() -> str:
    data = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/auth/login", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read())
    return body.get("access_token") or body.get("token", "")


def post(path: str, payload: dict, token: str) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> int:
    token = login()
    if not token:
        print("login failed", file=sys.stderr)
        return 1
    figures = []
    for f in RESULTS["figures"]:
        path = FIGDIR / f["name"]
        figures.append({
            "name": f["name"],
            "caption": f["caption"],
            "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })
    c1 = RESULTS["criteria"]["C1_hetenyi_point_load_5pct"]
    c2 = RESULTS["criteria"]["C2_patch_test_1e-10"]
    c3 = RESULTS["criteria"]["C3_pasternak_to_winkler_0p1pct"]
    c4 = RESULTS["criteria"]["C4_dependence_laws"]
    summary = (
        "冰板弯曲 FEM 升级到 Pasternak/Hetényi 双参数地基并以闭式解对标（r/l∈[0.1,5] 误差 "
        f"{100*c1['max_rel_error_used']:.2f}% < 5%）；patch test {c2['max_rel_error']:.2e} < 1e-10；"
        f"Pasternak→Winkler 退化 {c3['rel_diff_at_smallest_g_p']:.1e} < 0.1%；"
        "新增冰厚–温度–加载速率依赖律：∂σmax/∂h<0、∂σmax/∂|T|<0 通过，"
        "∂σmax/∂v>0 如实 FAIL（速率律在强度/承载力侧成立）。"
    )
    payload = {
        "title": "P366-W2-N01 | 冰板 Pasternak/Hetényi 地基 FEM 对标 + 冰厚–温度–加载速率依赖律",
        "summary": summary,
        "scenario_key": "ice_navigation",
        "mission_id": 10711,
        "tags": ["ice_navigation", "fem", "pasternak", "hetenyi", "ice_sheet",
                 "design_law", "verification", "m10711"],
        "markdown": MD.read_text(),
        "figures": figures,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())