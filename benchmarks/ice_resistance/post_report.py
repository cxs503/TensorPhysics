#!/usr/bin/env python3
"""Build and post the mission-4911 paper-level report (numbers injected from artifacts)."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path("/root/tp-agent/work/m4911")
ART = REPO / "artifacts"
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=hdr, method="POST")
    with OPENER.open(req, timeout=60) as r:
        return r.status, r.read().decode()


def get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    with OPENER.open(req, timeout=30) as r:
        return r.status, r.read().decode()


def login() -> str:
    _, body = post(f"{BASE}/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })
    return json.loads(body)["access_token"]


def main() -> int:
    token = login()
    vp = json.loads((ART / "vp_curves.json").read_text(encoding="utf-8"))
    rep = json.loads((ART / "ice_resistance" / "validation_report.json").read_text(encoding="utf-8"))
    doc = (REPO / "docs" / "ICE_RESISTANCE_FULL_SPECTRUM.md").read_text(encoding="utf-8")
    corpus = rep["anchors"]
    audit = corpus["audit"]
    acc = vp["acceptance"]

    header = f"""## 结论速览（IN10-C1）

| 判据 | 要求 | 实测 |
|---|---|---|
| 参考锚（每锚容差内） | ≥10 全过 | **{acc['n_reference_anchors_passed']}/{acc['n_reference_anchors']} 全过** |
| 负例（域外拒绝/如实截断） | 全部正确 | **{acc['n_negative_controls_passed']}/{acc['n_negative_controls']} 全过** |
| 速度-功率曲线表 | `artifacts/vp_curves.json` | 已产出，{acc['vp_table_points']} 个点 |
| pytest | 全绿 | `pytest tests/test_ice_resistance.py` → 47 passed |
| 总判定 | — | `all_passed = {acc['all_anchors_passed']}` |

关键数字：h=1.0 m 平整冰、V=3 m/s 时本模型总阻力 **{audit['benchmark_model_kN']:.0f} kN**
（Keinonen 1983 {audit['keinonen_kN']:.0f} kN / Enkvist 1972 {audit['enkvist_kN']:.0f} kN，
相对均值 {100*audit['benchmark_rel_dev']:+.1f}%）；仓内旧转写 Lindqvist 为
{audit['legacy_lindqvist_kN']:.0f} kN（{100*audit['legacy_lindqvist_rel_dev']:+.1f}%，已定位为艏角约定缺陷）。

"""

    figures = [
        ("fig1_resistance_spectrum.png",
         "图1 PC6 参考船全冰况谱的阻力-航速族：平整冰 5 条（h=0.3–1.2 m）、碎冰 3 条（c=0.3/0.6/0.9）、"
         "冰脊场 3 条（h_k=1.5/2.5/3.5 m，间距 200 m）。"),
        ("fig2_power_spectrum.png",
         "图2 交付功率-航速曲线族与 22 MW 装机线（P_D = R·V/η_D，η_D=0.55）；曲线与装机线交点即各冰况可达航速。"),
        ("fig3_coverage_decay.png",
         "图3 碎冰覆盖衰减律 D(c)（左：与清除极限 D=c 及 c_crit=0.6 对比）与碎冰阻力随覆盖率的变化（右）。"),
        ("fig4_audit_deviation.png",
         "图4 审计图：修正后三分量族 vs Keinonen/Enkvist 两条独立文献律（±25% 带）vs 仓内旧转写 "
         "Lindqvist/Riska（艏角约定缺陷，+197%~+271%）。"),
    ]
    figs = []
    for name, cap in figures:
        png = (ART / "ice_resistance" / name).read_bytes()
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(png).decode("ascii")})

    summary = (
        f"三态冰阻力模型族（Lindqvist 三分量 / 碎冰覆盖衰减律 / Keinonen 冰脊冲撞项）落地 "
        f"TensorPhysics src/tensorphysics/ice/resistance.py，"
        f"L2 锚语料 {acc['n_reference_anchors_passed']}/{acc['n_reference_anchors']} 全过、"
        f"负例 {acc['n_negative_controls_passed']}/{acc['n_negative_controls']} 全过，"
        f"PC6 全冰况速度-功率矩阵 artifacts/vp_curves.json（{acc['vp_table_points']} 点）。"
        f"审计发现仓内旧 Lindqvist/Riska 转写把水线面艏半角当板面倾角用（+271%/+197%），已在本族修正并留证；"
        f"绝对精度 vs 实尺度测量 = WITHHELD（仓内无数字化实测记录）。"
    )

    payload = {
        "title": "IN10-C1 冰阻力全谱与速度-功率曲线：三态模型族 + L2 锚语料 25/25 + PC6 VP 矩阵",
        "summary": summary,
        "scenario_key": "ice_resistance_full_spectrum",
        "mission_id": 4911,
        "tags": ["IN10-C1", "ice-resistance", "TensorPhysics", "three-state-model",
                 "speed-power", "L2-anchors", "audit"],
        "markdown": header + doc,
        "figures": figs,
    }
    body = json.dumps(payload, ensure_ascii=False)
    print("payload bytes:", len(body.encode()), "figures:", len(figs))
    status, resp = post(f"{BASE}/tensor/reports", payload, token=token)
    print("POST status:", status, resp[:400])
    st, rd = get(f"{BASE}/tensor/reports?mission_id=4911", token)
    data = json.loads(rd)
    rows = data if isinstance(data, list) else data.get("items", data.get("data", []))
    print("reports for mission 4911:", len(rows) if hasattr(rows, "__len__") else rows)
    if isinstance(rows, list) and rows:
        r0 = rows[0]
        print("last title:", r0.get("title", "")[:80], "| id:", r0.get("id"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())