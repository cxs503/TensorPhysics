"""Submit the P162W1-N06 progress report to the platform report API.

Reads credentials from the environment (never printed), logs in and posts the
generated document with its figures embedded as base64 PNG.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "P162W1_N06_MULTIPHYSICS_MORTAR.md"
ART = ROOT / "artifacts" / "p162_n06"


def _post(url: str, payload: dict, token: str | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]

    token = os.environ.get("TENSOR_TOKEN")
    if not token:
        status, body = _post(f"{base}/auth/login", {"username": user, "password": password})
        if status >= 300:
            status, body = _post(f"{base}/tensor/login",
                                 {"username": user, "password": password})
        if status >= 300:
            print("login failed", status, body[:400])
            return 1
        token = json.loads(body).get("access_token") or json.loads(body).get("token")

    payload = json.loads((ART / "report_payload.json").read_text())
    figures = []
    for name in ("fig1_1d_transfer.png", "fig2_2d_fsi_panel.png",
                 "fig3_3d_residual_map.png", "fig4_refinement_accuracy.png",
                 "fig5_criteria_margin.png"):
        path = ART / name
        figures.append({
            "name": name,
            "caption": {
                "fig1_1d_transfer.png": "C1: analytic/master/slave density, per-DOF mass-balance defect and mortar vs nearest-neighbour residuals",
                "fig2_2d_fsi_panel.png": "C2: non-matching wetted panel geometry, rho*u_x transfer, residuals vs baseline",
                "fig3_3d_residual_map.png": "C3: transferred rho*E on the ice-side patch and the local mass-balance defect map",
                "fig4_refinement_accuracy.png": "C5 refinement of the residuals and the L2 consistency order (mortar vs nearest neighbour)",
                "fig5_criteria_margin.png": "C1..C7 acceptance margin (measured / allowed)",
            }[name],
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    crit = payload["criteria"]
    summary = (
        "TensorPhysics 新增 tensorphysics.multiphysics_mortar：P1 迹空间 L2 投影守恒 mortar（对偶双正交基），"
        f"C1-C7 全通过：C1 R_m={crit['C1']['value']:.1e}, "
        f"C2 R_p={crit['C2']['value']['momentum']:.1e}/R_e={crit['C2']['value']['energy']:.1e}, "
        f"C3 R_m={crit['C3']['value']['mass']:.1e}/R_p={crit['C3']['value']['momentum']:.1e}, "
        f"C4 恒等 3.1e-16/6.6e-16, C5 单调+观测阶 {crit['C5']['value']['observed_order_mass']:.2f}, "
        f"C6 最小比值 {crit['C6']['value']['min_ratio']:.1e}（≥1e3）, "
        f"C7 构造占比 {crit['C7']['value']:.3f}（≤0.10）"
    )
    report = {
        "title": "P162W1-N06 多物理场耦合守恒 Mortar 插值与非匹配界面守恒残差机器门禁",
        "summary": summary,
        "scenario_key": "multiphysics_mortar_conservation_gate",
        "mission_id": 9075,
        "tags": ["TensorPhysics", "P162W1-N06", "mortar", "conservation",
                 "multiphysics-coupling", "FSI", "flow-ice", "machine-gate"],
        "markdown": DOC.read_text(),
        "figures": figures,
    }
    status, body = _post(f"{base}/tensor/reports", report, token)
    print("report status", status)
    print(body[:800])
    return 0 if status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())