#!/usr/bin/env python3
"""Publish the IN101-N09 report to TensorPlatform (``/tensor/reports``).

Reads the machine-generated results JSON, the report markdown and the six
figures, embeds the figures as base64 PNG and posts the payload.  Credentials
come from the environment (``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``,
``TENSOR_AGENT_PASS``); nothing is written to disk.

Usage (from the repository root, after ``run_in101_n09.py``)::

    PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/publish_in101_n09_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
API = os.environ["TENSOR_API_BASE"].rstrip("/")
MISSION_ID = 8350
SCENARIO_KEY = "ice_partial_factors_target_inverse"

FIG_CAPTIONS = {
    "fig1_pa_joint.png": "图1 联合 p–A 状态：两种相关假设共享同一对边缘分布，只差 "
    "$\\rho_{\\ln p,\\ln A}$；右图为年极值冰力密度（散度膨胀即负控的物理内容）",
    "fig2_beta_vs_kappa.png": "图2 可靠度指标随分项系数乘积的曲线、目标线、Newton 解、"
    "单变量二分解，以及 FORM/IS/粗 MC/Rao–Blackwell MC 四个独立估计",
    "fig3_newton.png": "图3 左：三组分散初值的残差收敛（二次收敛、同一根）；"
    "右：设计值劈分约定 $\\gamma_f(\\kappa)$、$\\psi(\\kappa)$ 与解点",
    "fig4_iso_deviations.png": "图4 与 ISO 19906 A.2×A.3 表格分项系数逐项偏差图（±15% 验收带），"
    "下图为缺口行的实际达到 $\\beta$",
    "fig5_negative_control.png": "图5 负控：计入/忽略 p–A 负相关时的标定分项系数与载荷散度",
    "fig6_sensitivity.png": "图6 左：包络定理伴随导数 vs 中心差分；右：标定分项系数对 $\\beta_T$ 的敏感性",
}


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())


def login() -> str:
    data = _post(
        API + "/auth/login",
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
    )
    return data["access_token"]


def build_payload() -> dict:
    data = json.loads(
        (ROOT / "artifacts/iso19906_target_inverse/in101_n09_results.json").read_text()
    )
    markdown = (ROOT / "docs/reports/in101_n09/iso19906_target_inverse.md").read_text()
    sol = data["inverse"]["solution"]
    nc = data["negative_control"]
    summary = data["criteria_summary"]

    figures = []
    for name, caption in FIG_CAPTIONS.items():
        path = ROOT / "artifacts/iso19906_target_inverse" / name
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    return {
        "title": "IN101-N09 ISO 19906 冰压设计分项系数目标可靠度反标定（成熟化）",
        "summary": (
            f"两方程两未知量反问题解 (γ_f, ψ) = ({sol['gamma_f']:.4f}, {sol['psi']:.4f})，"
            f"κ*={sol['kappa_star']:.4f}，达到 β={sol['beta_achieved']:.9f}；"
            f"Jacobian 条件数 {sol['jacobian_condition_number']:.2f}，三初值极差 "
            f"{data['inverse']['gamma_f_spread_rel']:.1e}；FORM 与闭式解析值、重要性抽样一致到 0.1% 量级；"
            f"忽略 p–A 负相关使 γ_f 偏大 {100 * nc['delta_gamma_f_rel']:.1f}%（检出）。"
            f"机器判据 {summary['passed']}/{summary['total']} 通过。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": [
            "IN101-N09",
            "ISO19906",
            "partial-factors",
            "FORM",
            "importance-sampling",
            "negative-control",
            "ice",
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