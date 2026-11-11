"""Publish the paper-grade maturation report (mission 11210 / node AG29W1-N02).

Markdown is read from ``docs/ICE_RESISTANCE_MEASURED_ANCHOR_V2.md`` (numbers injected
from the artifacts) and the figures are embedded as base64 PNG, so the report is
reproducible from the worktree instead of hand-assembled.

Run::

    python benchmarks/ice_resistance/measured/publish_report_v2.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DOC = REPO / "docs" / "ICE_RESISTANCE_MEASURED_ANCHOR_V2.md"
FIG_DIR = REPO / "artifacts" / "ice_resistance" / "figures"
ARTIFACT = REPO / "artifacts" / "ice_resistance" / "measured_adjudication_v2.json"
GAPS = REPO / "data" / "measured" / "ice_resistance" / "evidence_gaps.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

MISSION_ID = 11210

FIGURES = [
    ("uncertainty_vs_criterion.png",
     "逐点测量相对不确定度 u/R（柱）vs 15% 判据线（虚线）vs 模型误差（三角）：判据比噪声还细"),
    ("external_validity_probe.png",
     "域外探针（SALMI2014-UBIB）：模型约定带 vs 实测 vs 源文上折理论，两族夹住实测"),
    ("library_capability_summary.png",
     "真值库能力清单：判决集点数 / 有不确定度点 / 可证伪点 / 可证真点 / 证据缺口 / 带时戳尝试"),
    ("measured_vs_predicted.png",
     "基线判决集：实测 vs 预测（MT Uikku 10 点，1:1 与 ±15% 带）"),
    ("component_attribution.png",
     "基线归因：R_crush / R_break / R_submersion 分量堆叠 vs 实测"),
]


def login() -> str:
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(
        f"{BASE}/auth/login", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
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
    gaps = json.loads(GAPS.read_text(encoding="utf-8"))
    crit = art["criteria"]
    c7 = art["C7_criterion_metrology"]
    c8 = art["C8_external_validity"]
    c9 = art["C9_evidence_gaps"]

    figures = []
    for name, caption in FIGURES:
        path = FIG_DIR / name
        if not path.is_file():
            print(f"missing figure: {path}", file=sys.stderr)
            return 3
        figures.append(
            {
                "name": name.replace(".png", ""),
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )

    summary = (
        "冰阻力实测锚库成熟化：3 系列/18 点全部治理完备（台架·尺度·不确定度·数字化方法·sha256）；"
        f"新增判据度量学证明 15% 判据在现有真值上可证伪({c7['n_fail_resolvable_at_kmax']}/{c7['n_points_with_uncertainty']} 点 3σ)但不可证真"
        f"({c7['n_pass_resolvable']}/{c7['n_points_with_uncertainty']} 点可确认 15%，因 u/R 中位数 {100 * c7['median_u_over_R']:.1f}% > 15%)；"
        f"新增第二独立战役 UBIB 作域外探针：可采纳约定带内模型高估 {c8['best_admissible_pred_over_meas']:.2f}×，"
        f"源文上折理论低估 {c8['published_ubib_theory']['leiviska_ubib_sigma500']['median_rel_error_pct']:.0f}%，两族夹住实测；"
        f"证据缺口 {c9['n_gaps']} 个、带时戳尝试 {c9['n_attempts']} 次；15% 判决仍为 honest FAIL"
        f"（中位误差 {art['baseline']['best_median_rel_error_pct']:.1f}%），未调参。"
    )

    payload = {
        "title": "冰阻力实测锚库成熟化：判据度量学、外部效度与证据缺口机制（AG29W1-N02）",
        "summary": summary,
        "scenario_key": "ice_resistance_measured_anchor_v2",
        "mission_id": MISSION_ID,
        "tags": [
            "ice-resistance", "measured-anchor", "criterion-metrology", "external-validity",
            "evidence-gap", "TensorPhysics", "AG29W1-N02",
        ],
        "markdown": markdown,
        "figures": figures,
        "criteria": crit,
        "evidence": {
            "artifact": "artifacts/ice_resistance/measured_adjudication_v2.json",
            "baseline_artifact": art["baseline"]["source_artifact"],
            "baseline_sha256": art["baseline"]["sha256"],
            "evidence_gaps": {
                "path": "data/measured/ice_resistance/evidence_gaps.json",
                "n_gaps": c9["n_gaps"],
                "n_attempts": c9["n_attempts"],
                "attempt_log_sha256": c9["attempt_log_sha256"],
                "gap_ids": [g["gap_id"] for g in gaps["gaps"]],
            },
            "tests": "python3 -m pytest tests/test_measured_criterion.py tests/test_measured_anchor.py -q -> 49 passed",
            "verdict_15pct": art["verdict_15pct"],
        },
    }

    token = login()
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"{BASE}/tensor/reports",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read().decode())
    print(json.dumps({"posted": True, "report_id": out.get("id") or out.get("report_id"),
                      "summary_chars": len(summary), "figures": [f["name"] for f in figures]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())