#!/usr/bin/env python3
"""Publish the P350W1-N04 report to the platform report API.

Reads the machine-verified numbers from
``artifacts/structural_acoustic_radiation/verification.json`` and the generated
markdown report, so nothing is hand-copied.  Credentials come from
``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / "artifacts" / "structural_acoustic_radiation" / "verification.json"
DOC = ROOT / "docs" / "P350W1_N04_STRUCTURAL_ACOUSTIC_RADIATION.md"
FIG_DIR = ROOT / "artifacts" / "structural_acoustic_radiation"


def post(url: str, payload: dict, token: str | None = None):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read().decode()


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]

    status, body = post(f"{base}/auth/login", {"username": user, "password": password})
    token = json.loads(body).get("access_token") or json.loads(body).get("token")
    print(f"login status={status} token={'ok' if token else 'missing'}")
    if not token:
        print(body[:400])
        return 1

    ver = json.loads(VER.read_text())
    markdown = DOC.read_text(encoding="utf-8")

    captions = {
        "fig_radiation_efficiency.png":
            "Fig 1 辐射效率 σ_rad(ka)：单极子 (ka)²/(1+(ka)²)、无限障板活塞 "
            "1−2J₁(2ka)/(2ka)、无障板板偶极子 (ka)⁴/(1+(ka)⁴) 及三条渐近线",
        "fig_far_field_decay.png":
            "Fig 2 远场声压 p_rms(r) 随距离衰减：解析球与通用链均严格 ∝1/r（log-log 斜率 −1.000）",
        "fig_piston_onaxis.png":
            "Fig 3 活塞轴上声压：Rayleigh 积分精确解 vs 远场近似 ρcka²v₀/2r vs 功率平均法（ka=2）",
        "fig_sdof_response.png":
            "Fig 4 SDOF 结构振动级速度频响：数值 F₀/|Z| 与共振闭式 F₀/c_d 重合（f_n=100 Hz, ζ=0.02）",
        "fig_chain_spl.png":
            "Fig 5 全链预报 SPL vs 距离（f=100 Hz 共振，S=1 m², a=0.3 m），三种 σ 模型",
    }
    figures = []
    for name, caption in captions.items():
        path = FIG_DIR / name
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    c = ver["criteria"]
    report = {
        "title": "水下结构声辐射：振动→声压的解析锚与半经验预报链（P350W1-N04）",
        "summary": (
            "建立并验证振动速度→辐射声功率→远场声压级全链：脉动球/活塞圆盘闭式锚 + "
            "偶极子半经验 σ，判据 (a)-(j) 全 PASS；"
            f"球声功率误差 {c['a']['max_relative_error']:.1e}、活塞 σ 独立求积误差 "
            f"{c['b']['max_independent_quadrature_error']:.1e}、1/r 衰减误差 "
            f"{c['c']['relative_error']:.1e}、活塞轴上 vs Rayleigh {c['g']['max_relative_error']:.1%}、"
            f"pytest {c['e']['n_passed']} passed，provenance sha256 "
            f"{ver['provenance']['payload_sha256'][:16]}…"
        ),
        "scenario_key": "P350W1-N04",
        "mission_id": 7793,
        "tags": ["acoustics", "structural-vibration", "underwater-radiation",
                 "analytic-anchor", "radiation-efficiency", "provenance"],
        "markdown": markdown,
        "figures": figures,
        "metadata": {
            "node": ver["node"],
            "repository": ver["repository"],
            "branch": ver["branch"],
            "module": ver["module"],
            "all_pass": ver["all_pass"],
            "verification_json": "artifacts/structural_acoustic_radiation/verification.json",
            "provenance_sha256": ver["provenance"]["payload_sha256"],
            "criteria": {k: v["passes"] for k, v in c.items()},
            "artifacts": ver["artifacts"],
        },
    }

    out = Path("/tmp/report_m7793.json")
    out.write_text(json.dumps(report, ensure_ascii=False))
    status, body = post(f"{base}/tensor/reports", report, token)
    print(f"report status={status}")
    print(body[:600])
    return 0 if status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
