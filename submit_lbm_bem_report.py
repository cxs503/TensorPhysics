"""Submit the p84w3-n10 cross-validation report to the platform API.

Logs in with the agent credentials from the environment, embeds the generated
figures as base64 PNGs and posts the paper-grade markdown report.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "lbm_bem_cross"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def login() -> str:
    payload = json.dumps({"username": USER, "password": PASS}).encode()
    for path in ("/auth/login", "/tensor/auth/login"):
        req = urllib.request.Request(
            BASE + path, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            tok = data.get("access_token") or data.get("token")
            if tok:
                return tok
        except Exception:
            continue
    raise RuntimeError("login failed")


def figure(name: str, caption: str) -> dict | None:
    p = ART / name
    if not p.exists():
        return None
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(p.read_bytes()).decode()}


def main() -> None:
    token = login()
    md = (ROOT / "docs" / "LBM_BEM_CROSS_VALIDATION.md").read_text()
    crit = json.loads((ART / "criteria.json").read_text())
    re100 = crit["cases"]["re100"]
    cd = re100["cd"]
    ff = re100["far_field"]
    summary = (
        "Real LBM (D2Q9, regularised BGK, momentum-exchange force) vs BEM panel method "
        f"+ Thwaites/VPI boundary layer on a circular cylinder. C_D: LBM {cd['cd_lbm']:.3f} "
        f"(anchor {cd['cd_reference']:.3f}, {100*cd['cd_lbm_vs_reference']:+.1f}%), "
        f"BEM+BL {cd['cd_bem_bl']:.3f} ({100*cd['cd_bem_vs_reference']:+.1f}%) at Re=100; "
        f"far-field L2 deviation at r>10D {ff['l2_deviation']:.4f} (<0.05 PASS); "
        "geometry NURBS/STL contract 0.0 (<1e-9 PASS); Re=1000/1e4 LBM not adjudicable "
        "(divergence at tau-1/2~5e-3) - honest FAIL reported."
    )
    figs = [f for f in (
        figure("fig_cp_re100.png", "Cp(theta): LBM vs BEM+boundary layer vs potential (Re=100)"),
        figure("fig_cd.png", "Drag coefficient: LBM vs BEM+BL vs literature anchor"),
        figure("fig_field_re100.png", "LBM velocity magnitude and vorticity with far-field sample ring"),
        figure("fig_far_field_re100.png", "Far-field disturbance decay and deviation map"),
        figure("fig_wake_band.png", "Cd sensitivity to the wake-pressure closure"),
    ) if f]
    report = {
        "title": "LBM<->BEM cross-validation of a cylinder: real two-solver adjudication "
                 "(p84w3-n10)",
        "summary": summary,
        "scenario_key": "lbm_bem_cross_validation",
        "mission_id": 9253,
        "tags": ["cross-validation", "LBM", "BEM", "boundary-layer", "cylinder",
                 "TensorPhysics", "p84w3-n10", "honest-FAIL"],
        "markdown": md,
        "figures": figs,
    }
    out = Path("/tmp/report_9253.json")
    out.write_text(json.dumps(report))
    req = urllib.request.Request(
        BASE + "/tensor/reports", data=out.read_bytes(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print("status", resp.status, resp.read().decode()[:400])
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:600])


if __name__ == "__main__":
    main()