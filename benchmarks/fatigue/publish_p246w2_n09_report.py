#!/usr/bin/env python3
"""Publish the p246w2-n09 detail-fatigue report to TensorPlatform.

Reads the machine-generated acceptance artifact, the report markdown and the six
figures, embeds the figures as base64 PNG and posts the payload to
``/tensor/reports``.  Credentials are taken from the environment
(``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``); nothing is
written to disk.

Usage (repository root, after ``benchmarks/fatigue/detail_fatigue_maturity.py``)::

    PYTHONPATH=src python3 benchmarks/fatigue/publish_p246w2_n09_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
API = os.environ["TENSOR_API_BASE"].rstrip("/")
MISSION_ID = 9186
SCENARIO_KEY = "tensorphysics_hull_detail_fatigue_maturity"
ART = ROOT / "artifacts/p246w2_n09"

FIG_CAPTIONS = {
    "fig1_sn_curve.png": (
        "图1 IIW FAT90 双对数 S-N 设计曲线（m₁=3，Δσ_c=90 MPa @ 2×10⁶，拐点 10⁷ 后 m₂=5）、"
        "log-log 最小二乘拟合（m=3.000，R²=1.000）与声明散差模型（s(logN)=0.2）生成的模拟焊接试验散点"),
    "fig2_kt_validation.png": (
        "图2 真实 T6 平面应力 FEM 计算的 K_t 与解析锚对照：开孔（Kirsch 精确解 3.000，FEM 3.0635，"
        "+2.12%）、对接焊趾（Inglis 1+2√(h/ρ)，误差 +3.12%/+1.60%），柱上标注数值与相对误差"),
    "fig3_hotspot_extrapolation.png": (
        "图3 T 型角接接头板面 σ_xx 剖面（t/4 与 t/16 网格）与 0.4t/0.9t/1.4t 采样点；"
        "水平虚线为三点二次外推到焊趾的热点应力（1.1037 vs 1.0668，相差 3.46%）"),
    "fig4_mesh_convergence.png": (
        "图4 网格敏感性：热点应力随焊趾首层单元尺寸 t/4→t/8→t/16 的变化"
        "（1.1037→1.0627→1.0668，总变化 3.35%，判据 5%）"),
    "fig5_miner_two_block.png": (
        "图5 左：两级应力块谱 Miner 损伤随块数的累积与 D=1 失效交点（9.918 块）；"
        "右：顺序探针（高→低 与 低→高 损伤完全相同，Δ=0.0）——Miner 忽略载荷顺序效应，"
        "即 C7 诚实门要求声明的局限"),
    "fig6_t_joint_field.png": (
        "图6 T 型接头 σ_xx 应力场云图（真实 FEM 解）：主板受拉时载荷绕横向加强筋传递，"
        "焊趾处形成局部集中（锐角焊趾为奇异点，故热点应力以 0.4t/0.9t/1.4t 外推定义）"),
}


def _post(url: str, payload: dict, token: str | None = None, timeout: int = 240) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def login() -> str:
    data = _post(API + "/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })
    return data["access_token"]


def build_payload() -> dict:
    acc = json.loads((ART / "detail_fatigue_acceptance.json").read_text())
    markdown = (ROOT / "docs/P246W2_N09_HULL_DETAIL_FATIGUE_MATURITY.md").read_text()
    c = acc["criteria"]
    s = acc["summary"]

    figures = []
    for name, caption in FIG_CAPTIONS.items():
        path = ART / "figures" / name
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    summary = (
        f"真实 T6 平面应力 FEM → 热点外推 → IIW FAT90 → Miner 全链成熟化，判据 "
        f"{s['n_pass']}/{s['n_criteria']} 通过，patch test 精确复现均匀场。"
        f"拟合 m={c['C1']['design_curve_fit']['slope_m']:.3f}、R²=1.000、FAT 反推 90.000 MPa、"
        f"DNV D(air) 锚 0.019%；开孔 K_t {c['C2']['hole']['kt_fem']:.3f} vs Kirsch（+"
        f"{c['C2']['hole']['error_pct']:.2f}%）、对接焊趾 vs Inglis "
        f"{max(r['error_pct'] for r in c['C2']['butt_joint_toe']):.2f}%；"
        f"三点外推 vs t/16 参考 {c['C4']['error_pct']['quadratic_3point']:.2f}%；"
        f"网格 t/4→t/16 变化 {c['C6']['changes_pct']['t4_to_t16']:.2f}%；"
        f"两区块谱累积与独立代数差 0.0。诚实项：设计曲线寿命保守 2.79×，"
        f"单试件散差带 2.13×；热点外推仅恢复孔边峰值 71.8%；真实试验数据 WITHHELD。"
    )

    return {
        "title": "P246W2-N09 船体结构细节疲劳评估成熟化（热点应力 + IIW S-N + Miner，真实 FEM 验证）",
        "summary": summary,
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": [
            "P246W2-N09",
            "detail-fatigue",
            "hot-spot-stress",
            "IIW-FAT90",
            "DNV-RP-C203",
            "Miner",
            "rainflow",
            "plane-stress-FEM",
            "Kirsch",
            "Inglis",
        ],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> int:
    token = login()
    payload = build_payload()
    response = _post(API + "/tensor/reports", payload, token=token)
    print(json.dumps(response, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())