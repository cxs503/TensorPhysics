#!/usr/bin/env python3
"""Publish the p156w2-n02 SIMP density-filter report to TensorPlatform.

Reads the machine-generated acceptance record, the generated markdown report and
the figures, embeds the PNGs as base64 and posts to ``/tensor/reports``.
Credentials come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``).

Usage (repository root, after the acceptance + figure + doc steps)::

    PYTHONPATH=src python scripts/publish_p156w2_n02_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = os.environ["TENSOR_API_BASE"].rstrip("/")
MISSION_ID = 10869
SCENARIO_KEY = "tensorphysics_simp_density_filter_maturity"
ART = ROOT / "artifacts/p156w2_n02_simp_filter"

FIG_CAPTIONS = {
    "fig1_mbb_designs.png": (
        "图1 MBB 梁（L/H=3，Vf=0.4）三档网格：上排无滤波（棋盘格与单点铰接满布），"
        "下排 PDE(Helmholtz)+Heaviside 滤波（协议 B：固定物理 r_min）；标题内为棋盘格指数"),
    "fig2_cantilever_designs.png": (
        "图2 悬臂梁（L/H=2，Vf=0.5）三档网格的滤波设计（协议 B），棋盘格指数随网格加密保持低位"),
    "fig3_checkerboard_map.png": (
        "图3 逐单元棋盘格诊断场 |混合二阶差分|/2：无滤波设计满布交替（亮），滤波设计接近 0（暗）"),
    "fig4_compliance_levels.png": (
        "图4 各档网格收敛柔度（协议 B 参考线），显示细化网格带来的柔度变化幅度"),
    "fig5_mesh_independence.png": (
        "图5 左：三档网格最优拓扑的最小 Jaccard（协议 B 与经典 A 对比，红虚线 0.85 判据）；"
        "右：柔度随 h 的 Richardson 收敛（标注各自的观测阶）"),
    "fig6_sensitivity_consistency.png": (
        "图6 解析伴随灵敏度 vs 中心差分（δ=1e-6）的相对误差：l2、活跃集(≥1%峰值)均值与最大值，红虚线为 1% 判据"),
    "fig7_regularisers.png": (
        "图7 中网格四种正则化（无滤波／线性 hat／hat+Heaviside／PDE+Heaviside）的设计、"
        "棋盘格指数与灰度比例对比"),
    "fig8_radius_below_h.png": (
        "图8 负控：r_min/h=0.5（左，故意违反 r_min>h）与 r_min/h=1.5（右）的设计对比，"
        "说明滤波半径必须大于单元尺寸"),
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
    rec = json.loads((ART / "acceptance.json").read_text())
    markdown = (ROOT / "docs/P156W2_N02_SIMP_FILTER.md").read_text()
    c = rec["criteria"]
    v = rec["verdict"]

    figures = []
    for name, caption in FIG_CAPTIONS.items():
        path = ART / name
        if not path.exists():
            continue
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    jmin = min(c["C3"]["cases"][n]["B"]["jaccard_min"] for n in c["C3"]["cases"])
    summary = (
        f"SIMP 密度滤波从 WITHHELD 抬升为 REAL：新增 Helmholtz(PDE) 隐式滤波 + Heaviside 投影"
        f"（解析灵敏度链），在 MBB(L/H=3,Vf=0.4) 与悬臂(L/H=2,Vf=0.5) 三档网格上验收，"
        f"判据 {v['n_pass']}/{v['n_total']} 通过，未通过 {', '.join(v['failed']) if v['failed'] else '无'}。"
        f"棋盘格指数抑制比最小 {c['C1b']['value_min']:.2f}x、均值 {c['C1b']['value_mean']:.2f}x；"
        f"灵敏度 vs 中心差分最差 l2 "
        f"{max(c['C2']['cases'][n]['rows']['filtered_B']['l2_rel'] for n in c['C2']['cases']):.2e}；"
        f"三档网格最小 Jaccard {jmin:.3f}；FE 人造解柔度收敛阶 "
        f"{min(c['C3']['cases'][n]['B']['order_manufactured_probe']['order'] for n in c['C3']['cases']):.2f}。"
        f"诚实项：文字判据「无滤波/有滤波柔度比≥5」被变分原理证伪（实测 "
        f"{c['C1_literal']['value_mean']:.3f}），如实判 FAIL 并由棋盘格指数抑制比承载物理意图。"
    )

    return {
        "title": "P156W2-N02 TensorFEM SIMP 密度滤波器验证面（棋盘格抑制 / 灵敏度一致性 / 网格无关性）",
        "summary": summary,
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": [
            "P156W2-N02",
            "SIMP",
            "topology-optimization",
            "density-filter",
            "Helmholtz-PDE-filter",
            "Heaviside-projection",
            "adjoint-sensitivity",
            "checkerboard",
            "mesh-independence",
            "MBB-beam",
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