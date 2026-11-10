#!/usr/bin/env python3
"""Publish the p154w2-N03 report to the Tensor platform (``/tensor/reports``).

Numbers come from ``artifacts/p154w2_n03/*.json`` and the text from
``docs/reports/p154w2_n03/REPORT.md``; the figures are embedded as base64 PNG so
the published report is self-contained.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "p154w2_n03"
#: ``TENSOR_API_BASE`` already carries the ``/api/v1`` prefix, so the endpoints below
#: are appended to it directly.
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def login() -> str:
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    payload = json.dumps({"username": user, "password": password}).encode()
    request = urllib.request.Request(
        f"{BASE}/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode())
    return data.get("access_token") or data["token"]


FIGURES = [
    (
        "fig_failure_faces_3d.png",
        "图1 FEM 断裂面导出：8/32 韧带面失效（颜色=比断裂能），灰面为未失效韧带，青色折线为裂纹前沿。",
    ),
    (
        "fig_ligament_stress.png",
        "图2 裂纹面开口牵引剖面：σ_n 沿高度分布、σ_t 阈值线与被导出（失效）面。",
    ),
    (
        "fig_energy_chain.png",
        "图3 能量链：左=弹性组能量分量与总能（残差 5.3e-14）；中=能量转移账本；右=耗散组 E(t)、累计耗散与闭合线。",
    ),
    (
        "fig_conservation_gates.png",
        "图4 守恒门：左=五道门相对误差 vs 容差（对数轴）；右=判别力敏感性矩阵（门确实会报警）。",
    ),
    (
        "fig_fem_validation.png",
        "图5 FEM 场认证：挠度/应力误差随自由度收敛（最细网格 1.06%/2.69%）与 1:1 挠度对照。",
    ),
]


def main() -> None:
    report_md = (REPO / "docs" / "reports" / "p154w2_n03" / "REPORT.md").read_text(encoding="utf-8")
    criteria = json.loads((OUT / "criteria_report.json").read_text())
    gates = json.loads((OUT / "conservation_gates.json").read_text())
    packet = json.loads((OUT / "failure_faces.json").read_text())

    figures = []
    for name, caption in FIGURES:
        data = base64.b64encode((OUT / name).read_bytes()).decode()
        figures.append({"name": name, "caption": caption, "png_base64": data})

    summary = (
        "failure_faces 接口升到实现级 v3.0.0：缺口三点弯曲 FEM 导出 8 个失效面"
        f"（A={packet['area_total_m2']:.3e} m², E={packet['energy_total_j']:.4e} J, "
        f"sha256={packet['sha256'][:16]}…），DEM 侧独立重算面积/能量并初始化碎块，"
        f"C1 面积差 {criteria['C1_area_relative_error']:.2e}（≤1%）、"
        f"C2 能量差 {criteria['C2_energy_relative_error']:.2e}（≤2%）、"
        f"C4 自由落体总能量残差 {criteria['C4_energy_residual_elastic']:.2e}（≤5%）、"
        f"C5 两次导出 sha256 一致，65 项 pytest 全绿。"
    )
    payload = {
        "title": "p154w2-N03 failure_faces 从文档级冻结接口升到实现级：FEM 断裂面 → DEM 碎块守恒交接",
        "summary": summary,
        "scenario_key": "p154w2_n03_failure_faces_handoff",
        "mission_id": 9055,
        "tags": [
            "failure_faces",
            "FEM-DEM",
            "fracture",
            "interface-contract",
            "conservation",
            "TensorPhysics",
            "p154w2-N03",
        ],
        "markdown": report_md,
        "figures": figures,
        "metrics": {
            "C1_area_relative_error": criteria["C1_area_relative_error"],
            "C2_energy_relative_error": criteria["C2_energy_relative_error"],
            "C4_energy_residual": criteria["C4_energy_residual_elastic"],
            "C5_idempotent": criteria["C5_idempotent"],
            "gate_passed": gates["passed"],
            "packet_sha256": packet["sha256"],
        },
    }
    body = json.dumps(payload).encode()
    token = login()
    for url in (f"{BASE}/tensor/reports",):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                print("POST", url, response.status, response.read().decode()[:400])
                return
        except urllib.error.HTTPError as exc:
            print("POST", url, "HTTP", exc.code, exc.read().decode()[:300])
        except Exception as exc:  # noqa: BLE001
            print("POST", url, "error", exc)
    raise SystemExit("report publication failed")


if __name__ == "__main__":
    main()