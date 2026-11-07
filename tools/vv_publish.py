"""Build and publish the mission report in one process (no shared temp file).

The report body and figures come from ``artifacts/vv_gate/**``; credentials from
``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``.  Building and
posting in the same interpreter avoids the shared ``/tmp/report.json`` path used
by other agents on this machine.

Usage::  python tools/vv_publish.py --mission 5389 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import vv_report  # noqa: E402

CAPTIONS = {
    "fig_gci_vs_r": "Manufactured control f(h)=3+0.5h^2: GCI of the fine result versus the refinement "
                    "ratio with the coarse grid fixed (0.05 -> 0.0227 -> 0.0129, monotone decrease).",
    "fig_order_comparison": "Observed order of accuracy of the cross-repository runs through the same gate "
                            "against the theoretical order p=2 and the +/-0.2 band.",
    "fig_convergence": "Log-log convergence of the measured error norms: TensorLBM channel case "
                       "(slopes 0.75/0.89/0.81) and the TensorFEM manufactured solution (slope 1.96).",
    "fig_vv20": "ASME V&V 20 acceptance for the ten table cases: |E|/U_val against the limit 1.",
    "fig_lbm_error_field": "TensorLBM channel Poiseuille: axial velocity profiles on three grids (left) "
                           "and the pointwise error against the analytic parabola (right).",
}


def build_payload(mission: int) -> dict:
    summary = json.loads((vv_report.ART / "summary.json").read_text(encoding="utf-8"))
    figures = []
    for name, caption in CAPTIONS.items():
        path = vv_report.FIG / f"{name}.png"
        if path.exists():
            figures.append({"name": f"{name}.png", "caption": caption,
                            "png_base64": vv_report._png_b64(path)})
    return {
        "title": "IN30-N07 成熟化：Richardson/GCI + ASME V&V 20 跨求解器收敛与不确定度机器门",
        "summary": (
            "交付与求解器无关的 V&V 机器门（tensorphysics.vv）：≥3 套网格 -> 观测阶 p、Richardson 外推 "
            "f_exact、GCI_fine、渐近性检验、ASME V&V 20 的 U_val 与 PASS/WARN/FAIL 裁决；含 "
            "TensorLBM/FVM/FEM/DEM/BEM 五种 benchmark JSON 方言的适配器与 CLI。制造解正控 p=2.000000、"
            "f_exact 误差 0；振荡/发散/停滞与 1% 噪声负控均非静默 PASS；V&V20 十行表与手算 10/10 一致；"
            "TensorFEM 制造解真跑 p=1.962（PASS）且本包的 Richardson/GCI 与其仓内独立实现逐位一致；"
            "TensorLBM 自带 Poiseuille 验证算例实测 p≈0.75-0.89，门判 FAIL，判据(c) 的 LBM/FVM 部分"
            "如实记为未满足。"
        ),
        "scenario_key": "vv_gci_asme_vv20_gate",
        "mission_id": mission,
        "tags": ["V&V", "GCI", "Richardson", "ASME-VV20", "convergence", "cross-repo",
                 "TensorLBM", "TensorFEM"],
        "markdown": vv_report.build_markdown(summary),
        "figures": figures,
    }


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, token: str):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=int, default=5389)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(args.mission)
    size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    print(f"payload {size} bytes, figures {len(payload['figures'])}, "
          f"markdown {len(payload['markdown'])} chars")
    if args.dry_run:
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    login = _post(f"{base}/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                        "password": os.environ["TENSOR_AGENT_PASS"]})
    token = login.get("access_token") or login.get("token")
    if not token:
        print(f"login failed: {json.dumps(login)[:300]}")
        return 2
    try:
        posted = _post(f"{base}/tensor/reports", payload, token)
        print(f"posted: {json.dumps(posted)[:300]}")
    except urllib.error.HTTPError as exc:
        print(f"post failed {exc.code}: {exc.read().decode('utf-8')[:400]}")
        return 3

    check = _get(f"{base}/tensor/reports?mission_id={args.mission}", token)
    items = check if isinstance(check, list) else (check.get("items") or check.get("reports") or [])
    print(f"reports for mission {args.mission}: {len(items)}")
    for report in items[:3]:
        print("  id", report.get("id"), "md_len", len(report.get("markdown") or ""),
              "figures", len(report.get("figures") or []),
              "title", str(report.get("title"))[:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())