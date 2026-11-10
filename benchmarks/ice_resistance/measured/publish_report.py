"""Publish the paper-grade report for mission 9425 to the platform report API.

The markdown body is read from ``docs/ICE_RESISTANCE_MEASURED_ANCHOR.md`` (whose
numbers are injected from the adjudication artifact) and the five figures are
embedded as base64 PNG -- so the report is reproducible from the worktree, not
hand-assembled.

Run:  python benchmarks/ice_resistance/measured/publish_report.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DOC = REPO / "docs" / "ICE_RESISTANCE_MEASURED_ANCHOR.md"
FIG_DIR = REPO / "artifacts" / "ice_resistance" / "figures"
ARTIFACT = REPO / "artifacts" / "ice_resistance" / "measured_adjudication.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

FIGURES = [
    ("measured_vs_predicted.png", "实测 vs 预测（MT Uikku 10 点，1:1 线与 ±15% 判据带）"),
    ("per_point_error.png", "逐点相对误差（对数轴）：本仓 Lindqvist 支路 vs 源文发表 Lindqvist，15% 线"),
    ("physics_gates_and_measured.png", "C4 物理门：左 R(h) 单调 + 实测点；右 R(V) 单调与 V→0 有限截距"),
    ("component_attribution.png", "误差归因：R_crush / R_break / R_submersion 堆叠 vs 实测"),
    ("k_bend_calibration.png", "k_bend 标定曲线：单一闭合常数无法使中位误差降到 15%"),
]


def login() -> str:
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(
        f"{BASE}/auth/login", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    token = payload.get("access_token") or payload.get("token")
    if not token:
        raise RuntimeError(f"login returned no token: {list(payload)}")
    return token


def main() -> int:
    if not PASS:
        print("TENSOR_AGENT_PASS not set", file=sys.stderr)
        return 2
    markdown = DOC.read_text(encoding="utf-8")
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    crit = art["criteria"]
    c3 = crit["C3_15pct_adjudication"]

    figures = []
    for name, caption in FIGURES:
        p = FIG_DIR / name
        if not p.is_file():
            raise SystemExit(f"missing figure {p}")
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(p.read_bytes()).decode("ascii"),
            }
        )

    report = {
        "title": "冰阻力实测锚库与首次实测对标裁决：15% 判据由 WITHHELD 转为可判（诚实 FAIL，误差集中在 R_break 闭合项）",
        "summary": (
            f"建立冰阻力唯一真值来源：2 个公开可数字化实测系列（MT Uikku 冰池 10 点 + S.A. Agulhas II 实船 4 点），"
            f"逐点 DOI/表号/sha256，双录入 10% 抽检一致率 100%。首次执行 15% 对标："
            f"全谱 Lindqvist 支路中位相对误差 {100*c3['best_median_rel_error']:.1f}%（预注册四配置 198–383%），"
            f"honest-FAIL；误差 99.9% 来自速度无关分量，R_break 占预测 61.8%。"
            f"C1/C2/C4/C5 全绿，C3 如实判否并给出归因与标定假设（不采用）。"
        ),
        "scenario_key": "ice_resistance_measured_anchor",
        "mission_id": 9425,
        "tags": [
            "ice-resistance",
            "measured-anchor",
            "Lindqvist",
            "MT-Uikku",
            "Agulhas-II",
            "15pct-adjudication",
            "honest-FAIL",
            "TensorPhysics",
            "p273w3-n07",
        ],
        "markdown": markdown,
        "figures": figures,
    }
    body = json.dumps(report).encode()
    token = login()
    req = urllib.request.Request(
        f"{BASE}/tensor/reports",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = resp.read().decode()
            print(f"status={resp.status} response={out[:400]}")
            return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()[:600]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())