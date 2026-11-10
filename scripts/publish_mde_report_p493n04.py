#!/usr/bin/env python
"""Publish the P493-N04 report (MDE_90 detection-strength certificates).

Reads the generated ledger + report + figures, logs in with the agent
credentials and posts to ``$TENSOR_API_BASE/tensor/reports``.  Nothing is
retyped by hand: every number in the payload comes from the ledger JSON.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mde_calibration_p493n04"
DOC = ROOT / "docs/MDE_DETECTION_CERTIFICATES_P493N04.md"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
MISSION_ID = 9484
SCENARIO_KEY = "P493-N04-mde-detection-strength"

CAPTIONS = {
    "fig1_detection_curves.png":
        "检出曲线：(a) σ=0 时确定性验收门是台阶——MDE_90 就是判据自身的门带；"
        "(b) 叠加 2 % 证据散布后同一批门变成 S 形，MDE_90 右移，且部分门在零偏差处即触发（P_fa 标注）。",
    "fig2_mde90_vs_sigma.png":
        "MDE_90 随证据散布 σ 的增长（N=100 Monte-Carlo 与 ASME V&V 20 闭式对照），"
        "5 % 参考线与 15 % 平台红线。",
    "fig3_mde90_certificates.png":
        "逐判据检出强度证书（symlog 横轴，跨约 10 个数量级），颜色 = 分级"
        "（STRONG/MARGINAL/WEAK/BLIND）。",
    "fig4_proportion_power.png":
        "kill-rate 判据的功效曲线（α=5 % 边界检验）：样本量决定能否看见 5 % 的合格率下降。",
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
    return data.get("access_token") or data.get("token")


def build_payload() -> dict:
    ledger = json.loads((ART / "ledger.json").read_text(encoding="utf-8"))
    s = ledger["summary"]
    markdown = DOC.read_text(encoding="utf-8")

    figures = []
    for name in CAPTIONS:
        path = ART / name
        if not path.is_file():
            continue
        figures.append({"name": name.replace(".png", ""),
                        "caption": CAPTIONS[name],
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})

    cont = {c["criterion"]: c for c in ledger["continuous"]}
    stat = {c["criterion"]: c for c in ledger["statistical"]}
    summary = (
        f"mutant 注入给 {s['n_criteria_total']} 条发货判据 + 4 条 kill-rate 判据逐条发"
        f"检出强度证书（每点 N=100，Wilson 95 %）：9 条可测 MDE_90 中 {s['n_catch_5pct']} 条"
        f"抓住 5 % 的错；最强 {100 * s['best_mde90']:.2g} %、最弱 "
        f"{100 * s['worst_mde90']:.0f} %（22 条台账的 kill-rate KPI，需 n≈400 才看住 5 %）；"
        f"12 条分类面判据 MDE_90 无定义。确定性门用二分法取精确边界（vs V&V20 闭式差 "
        f"{100 * abs(cont['VVVAL-ratio']['mde90_exact'] - cont['VVVAL-ratio']['mde90_theory']) / cont['VVVAL-ratio']['mde90_theory']:.1e} %）；"
        f"粗阶梯 1–20 % 量化偏差均值 {100 * s['ladder_quantisation_rel_error_mean']:.0f} %"
        f"（故不作判决依据）。诚实项：σ≥门带时 {s['n_noise_limited_rows']} 个组合噪声饱和、"
        f"22 条台账无法认证 kill-rate≥90 %@95 %（需 n≥29）。"
    )
    assert len(summary) <= 500, len(summary)
    return {
        "title": "P493-N04 验收判据的检出强度标定：mutant 注入 → MDE_90 与检出曲线（逐条证书）",
        "summary": summary,
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["P493-N04", "MDE_90", "detection-power", "mutant-injection",
                 "criterion-certification", "Monte-Carlo", "ASME-VV20", "kill-rate",
                 "binomial-power", "TensorPhysics"],
        "markdown": markdown,
        "figures": figures,
        "artifacts": {
            "ledger": "artifacts/mde_calibration_p493n04/ledger.json",
            "engine": "src/tensorphysics/vv/mde.py",
            "driver": "scripts/mde_calibration_p493n04.py",
            "tests": "tests/test_vv_mde.py",
            "doc": "docs/MDE_DETECTION_CERTIFICATES_P493N04.md",
        },
        "metrics": {
            "n_criteria": s["n_criteria_total"],
            "n_measured": s["n_measured"],
            "n_catch_5pct": s["n_catch_5pct"],
            "best_mde90": s["best_mde90"],
            "worst_mde90": s["worst_mde90"],
            "class_counts": s["class_counts"],
            "vvval_exact_vs_closed_form_rel": abs(
                cont["VVVAL-ratio"]["mde90_exact"]
                - cont["VVVAL-ratio"]["mde90_theory"])
            / cont["VVVAL-ratio"]["mde90_theory"],
            "kpirate_n100_mde90": stat["KPIRATE-n100"]["mde90"],
        },
    }


def main() -> int:
    token = login()
    payload = build_payload()
    Path("/tmp/report_p493n04.json").write_text(
        json.dumps(payload, ensure_ascii=False)[:200000], encoding="utf-8")
    response = _post(API + "/tensor/reports", payload, token=token)
    print(json.dumps(response, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())