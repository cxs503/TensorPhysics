"""Publish the P233W2-N07 report to the platform report API.

Reads the generated report markdown and figures from
``docs/reports/ice_vibration_spectral/`` and posts them, with the PNG figures
base64-embedded, to ``$TENSOR_API_BASE/tensor/reports``.

Usage:  python submit_p233w2_n07_report.py [--dry-run]
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
REPORT_DIR = ROOT / "docs" / "reports" / "ice_vibration_spectral"
API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")

CAPTIONS = {
    "fig1_excitation.png": "图 1 冰破碎激励时程与 Welch PSD（标注 f_ice 与结构半功率带）",
    "fig2_frf.png": "图 2 SDOF 位移频响/相位：闭式解 vs 状态空间独立解",
    "fig3_response_psd.png": "图 3 案例 A 灯塔响应 PSD 与文献频带",
    "fig4_mode_shapes.png": "图 4 案例 B 船体梁 5 自由度振型",
    "fig5_resonance_map.png": "图 5 共振风险图：f_ice(U) vs 各阶模态半功率带",
    "fig6_hull_chain.png": "图 6 案例 B 船首加速度响应 PSD",
    "fig7_grid_study.png": "图 7 Welch 网格分辨率相对结构带宽的响应 RMS 偏差",
}


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
    payload = json.loads((REPORT_DIR / "node_payload.json").read_text())
    crit = payload["criteria"]
    figures = []
    for path in sorted((REPORT_DIR / "figures").glob("*.png")):
        figures.append(
            {
                "name": path.stem,
                "caption": CAPTIONS.get(path.name, path.name),
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    summary = (
        "冰致振动频谱管线落地：间歇破碎激励（f_ice=U/s）+ SDOF/MDOF 频响 + 响应 PSD + "
        "半功率带共振风险判据；AC1-AC5 全通过（SDOF 固有频率 0.040%、MDOF 特征频率 2e-15、"
        f"频响幅值 {crit['AC2']['sdof']['amplitude_max_relative_error']:.1e}、"
        f"白噪声 RMS 0.060%、判据一致性 {crit['AC4']['result']['n_agree']}/{crit['AC4']['result']['n_points']}、"
        f"文献主频误差 {100*crit['AC5']['result']['relative_frequency_error']:.2f}%）。"
    )
    return {
        "title": "冰致振动频谱分析与共振风险判据（P233W2-N07）",
        "summary": summary,
        "scenario_key": "ice_vibration_spectral",
        "mission_id": 10678,
        "tags": ["ice", "vibration", "spectral", "psd", "resonance", "campaign", "P233W2-N07"],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv: list[str]) -> int:
    payload = build_payload()
    if "--dry-run" in argv:
        print(json.dumps({k: (v if k != "figures" else [f["name"] for f in v]) for k, v in payload.items()},
                         ensure_ascii=False, indent=2)[:2000])
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
        print(f"status {resp.status}: {body[:600]}")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()[:800]}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))