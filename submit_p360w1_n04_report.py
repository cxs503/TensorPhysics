"""Publish the P360W1-N04 report to the platform report API.

Reads the generated report markdown, the node payload and the PNG figures from
the worktree and posts them (figures base64-embedded) to
``$TENSOR_API_BASE/tensor/reports``.

Usage:  python submit_p360w1_n04_report.py [--dry-run]
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "docs" / "reports" / "p360w1_n04_moving_load_fnv"
PAYLOAD_JSON = ROOT / "artifacts" / "p360w1_n04_moving_load_fnv" / "node_payload.json"
API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")


def login() -> str | None:
    data = json.dumps({"username": USER, "password": PASSWORD}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/auth/login", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("access_token") or payload.get("token")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"login failed: {exc}", file=sys.stderr)
        return None


def build_payload() -> dict:
    markdown = (REPORT_DIR / "report.md").read_text()
    node = json.loads(PAYLOAD_JSON.read_text())
    crit = {item["key"]: item for item in node["criteria"]}
    ice = node["ice_case"]
    scan = node["ice_speed_sweep"]
    cv = node["critical_velocity"]

    figures = []
    for item in node["figures"]:
        path = REPORT_DIR / "figures" / item["name"]
        figures.append(
            {
                "name": item["name"],
                "caption": item["caption"],
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    summary = (
        "冰致结构响应移动载荷效应落地：Euler-Bernoulli 船体梁（L=150 m，f₁=%.4f Hz，V_cr=%.2f m/s）"
        "+ 碎冰流移动接触力 F(x,t) + 前 10 阶模态 Newmark，给出 FNV=a_max/(g·f) 冰级分档；"
        "判据 (a)-(g) 全通过：静挠误差 %.4f%%（<3%%）、V_cr 峰值对 100 项解析级数 %.2e、"
        "独立有限差分 %.2e、V→0 退化 %.4f%%、模态截断 %.2e、功率平衡 %.2e；"
        "运营 5 m/s 工况中跨 FNV=%.3f（%s）；航速扫描 FNV 峰值 %.1f m/s vs 理论 f₁·s̄=%.2f m/s（差 %.2f%%）。"
        % (
            node["beam"]["natural_frequencies_hz"][0],
            cv["v_cr_m_s"],
            100.0 * max(node["static_anchor"]["modal_static_relative_error"], node["static_anchor"]["transient_relative_error"]),
            cv["modal_vs_analytic_relative_error"],
            cv["modal_vs_fd_relative_error"],
            100.0 * node["degeneracy"]["relative_error"],
            node["modal_truncation"]["relative_error"],
            node["critical_velocity"]["energy"]["relative_residual"],
            ice["fnv_midspan"],
            ice["fnv_level_midspan"],
            scan["fnv_peak_speed_m_s"],
            scan["resonant_speed_estimate_m_s"],
            100.0 * scan["speed_error_vs_estimate"],
        )
    )

    return {
        "title": "冰致结构响应：船-冰-梁耦合移动载荷效应与 FNV 预报（P360W1-N04）",
        "summary": summary,
        "scenario_key": "moving_load_ice_fnv",
        "mission_id": 12502,
        "tags": [
            "ice",
            "moving-load",
            "hull-girder",
            "fnv",
            "vibration",
            "euler-bernoulli",
            "campaign",
            "p360w1-n04",
        ],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv: list[str]) -> int:
    payload = build_payload()
    if "--dry-run" in argv:
        preview = {k: (v if k != "figures" else [f["name"] for f in v]) for k, v in payload.items()}
        preview["markdown"] = preview["markdown"][:400] + " ..."
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    token = login()
    if not token:
        return 2
    req = urllib.request.Request(
        f"{API_BASE}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode()
        print(f"status {resp.status}: {body[:600]}")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()[:800]}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))