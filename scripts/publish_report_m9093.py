"""Publish the p229w4-n05 report to the platform report API.

Credentials come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``) and are exchanged at
``POST /auth/login`` for a bearer token; nothing is written to disk and no
secret is echoed.  Figures are embedded as base64 from the artifact
directory, the markdown body is the repo report.

Usage: ``python scripts/publish_report_m9093.py [--dry-run]``
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

DOC = "docs/DESIGN_SEA_STATE_IFORM.md"
FIGURES = [
    (
        "fig1_contours.png",
        "IFORM environmental contours in the (H_s, T_z) plane for p_f = 1e-2/1e-3/1e-4, the "
        "convention-C contour, the moment-matched Gaussian surrogate contour and the sea-state MC cloud.",
        "artifacts/design_sea_state_ifrom/fig1_contours.png",
    ),
    (
        "fig2_rosenblatt_anchor.png",
        "Left: the radius-beta circle in standard-normal space (theta=0 is the extreme-H_s point). "
        "Right: model marginal return-value curve vs 2e5-sample empirical curve, with the contour "
        "extreme H_s falling exactly on the analytic anchor F^-1(Phi(beta)).",
        "artifacts/design_sea_state_ifrom/fig2_rosenblatt_anchor.png",
    ),
    (
        "fig3_adjudication.png",
        "Left: Rayleigh distribution of wave heights for H_s = 6 m, correct scale H_s/2 vs the "
        "H_s/4 trap, closed-form mode and the 4e6-wave MC MLE. Right: five design-wave-power routes "
        "at p_f = 1 %/yr with deviations against the exact long-term quadrature reference.",
        "artifacts/design_sea_state_ifrom/fig3_adjudication.png",
    ),
]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main(argv: list[str]) -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]

    login = _post(f"{base}/auth/login", {"username": user, "password": password})
    token = login.get("access_token") or login.get("token") or login.get("accessToken")
    if not token:
        print("login response keys:", sorted(login), file=sys.stderr)
        return 2

    with open(DOC, encoding="utf-8") as fh:
        markdown = fh.read()

    figures = []
    for name, caption, path in FIGURES:
        with open(path, "rb") as fh:
            figures.append(
                {
                    "name": name,
                    "caption": caption,
                    "png_base64": base64.b64encode(fh.read()).decode(),
                }
            )

    payload = {
        "title": "设计海况统计口径验证面：IFORM 环境等值线 × 目标可靠度 β × 超越概率 × Rayleigh 众值互裁",
        "summary": (
            "p229w4-n05: closed-form Rosenblatt transform of the joint (H_s,T_z) sea state, IFORM "
            "contours at both beta conventions, Rayleigh short-term conventions, and a three-way "
            "adjudication (IFORM / Rayleigh rule / exact long-term quadrature). 13/13 machine criteria "
            "PASS, 26 pytest green: contour vs analytic anchor 8.9e-16, extreme-H_s anchor exact, "
            "outside-contour probability calibrated to 0.4-1.0 % of exp(-beta^2/2), Rayleigh rule vs "
            "IFORM 0.9 %, IFORM vs exact long-term 0.3-1.2 %."
        ),
        "scenario_key": "design_sea_state_ifrom_convention_adjudication",
        "mission_id": 9093,
        "tags": [
            "sea-state",
            "IFORM",
            "environmental-contour",
            "Rosenblatt",
            "reliability-index",
            "Rayleigh",
            "verification",
            "TensorPhysics",
        ],
        "markdown": markdown,
        "figures": figures,
    }

    if "--dry-run" in argv:
        print("dry-run payload bytes:", len(json.dumps(payload)))
        return 0

    result = urllib.request.urlopen(
        urllib.request.Request(
            f"{base}/tensor/reports",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        ),
        timeout=180,
    )
    print("report status:", result.status)
    print(result.read().decode()[:600])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        print("HTTP error", exc.code, exc.read().decode()[:400], file=sys.stderr)
        sys.exit(3)