#!/usr/bin/env python3
"""Publish the P155W1-N09 (joint applicability envelope) report to the platform.

Numbers come from ``artifacts/applicability_interaction/evidence.json`` and the
markdown body is the node report itself, so nothing is re-typed by hand.

Usage::

    PYTHONPATH=src python tools/publish_applicability_interaction_report.py --publish
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "applicability_interaction" / "evidence.json"
DOC = ROOT / "docs" / "reports" / "p155w1_n09" / "APPLICABILITY_INTERACTION.md"
FIGDIR = ROOT / "artifacts" / "applicability_interaction" / "figures"
MISSION_ID = int(os.environ.get("MISSION_ID", "13891"))

CAPTIONS = [
    ("joint_plane_before_after.png",
     "图 1｜同一能力的二维参数面：左=单轴包络（离线门全绿，把真实失效区也判为 REAL），"
     "右=联合包络（深蓝框为保守胞，边界带被判 OUT_OF_DOMAIN）。黑点=闭式真值失效集。"),
    ("convergence_and_cost.png",
     "图 2｜fail-closed 的代价与收敛：左=过度拒绝随联合格点加密下降（3 个合成能力），"
     "右=对应评估次数（代价随格点平方增长）。"),
    ("gate_transcript.png",
     "图 3｜离线门话语（抽样 1/20 真值格点）：真值失效点上基线写 REAL 的次数 vs 联合包络（0）。"),
]


def _rates(evidence: dict, label: str) -> dict:
    return {case["capability_id"]: case["rates"][label] for case in evidence["cases"]}


def build_payload(evidence: dict, markdown: str) -> dict:
    figures = []
    for name, caption in CAPTIONS:
        path = FIGDIR / name
        if not path.exists():
            raise SystemExit(f"missing figure {path}")
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})
    matured = _rates(evidence, "matured")
    baseline = _rates(evidence, "baseline")
    fine = _rates(evidence, "fine")
    c6 = evidence["criterion_c6"]
    baseline_total = sum(int(row["false_in"]) for row in baseline.values())
    matured_total = sum(int(row["false_in"]) for row in matured.values())
    worst = max(row["false_out_fraction_of_box"] for row in matured.values())
    best = max(row["false_out_fraction_of_box"] for row in fine.values())
    return {
        "title": "适用域机器包络：联合（多轴）探测堵住离线门在联合失效角点写 REAL 的漏洞"
                 "（P155W1-N09 / m13891）",
        "summary": (
            "TensorPhysics 的适用域包络机制原先逐轴扫描、其余轴取标称，隐含假设「域=各轴区间之积」；"
            "当能力只在两个参数同时变坏时失效，一维区间全满，离线门（不重跑评估算子）把越界点写成 REAL。"
            "新增 interaction 模块（阶2b 联合格点+见证点、阶3b 见证射线二分、保守胞判定接线）："
            f"三个合成能力在 61×61 闭式真值格点上，基线误认证 {baseline_total} 次（100% 真值失效点），"
            f"成熟态 {matured_total} 次；过度拒绝最大 {100*worst:.2f}% 搜索面（上限 10%），"
            f"加密到 25 格点降至 {100*best:.2f}%。判据 C6 "
            f"{'PASS' if c6['passed'] else 'FAIL'}；三个真实能力的一维包络逐字段不变。"
            "残留风险：保守胞只保证含已采样失效点，薄于一个格胞的失效区仍可能别名（honest limitation）。"
        ),
        "scenario_key": "p155w1-n09-applicability-joint-envelope",
        "mission_id": MISSION_ID,
        "tags": ["applicability-envelope", "fail-closed", "out-of-domain", "joint-probe",
                 "acceptance-criteria", "tensorphysics", "p155w1-n09"],
        "markdown": markdown,
        "figures": figures,
        "metrics": {
            "baseline_false_certifications": baseline_total,
            "matured_false_certifications": matured_total,
            "criterion_c6_passed": bool(c6["passed"]),
            "worst_over_refusal_fraction_of_box": worst,
            "fine_over_refusal_fraction_of_box": best,
            "joint_boxes_per_case": {k: row["boxes"] for k, row in matured.items()},
            "joint_evaluations_per_case": {k: row["interaction_evaluations"]
                                           for k, row in matured.items()},
            "real_capability_axes_unchanged": all(
                all(row["axes_unchanged"].values())
                for row in evidence["real_capability_invariance"]),
            "config_sha256": evidence["config_sha256"],
            "evidence_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
        },
    }


def publish(payload: dict) -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    pwd = os.environ["TENSOR_AGENT_PASS"]
    login = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": user, "password": pwd}).encode(),
        headers={"Content-Type": "application/json"},
    )
    token = json.load(urllib.request.urlopen(login, timeout=30)).get("access_token")
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    try:
        resp = urllib.request.urlopen(req, timeout=240)
        print("published:", resp.status, resp.read()[:400])
    except urllib.error.HTTPError as exc:  # pragma: no cover
        print("publish failed:", exc.code, exc.read()[:800])
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    markdown = DOC.read_text(encoding="utf-8")
    payload = build_payload(evidence, markdown)
    out = EVIDENCE.parent / "report_payload.json"
    out.write_text(json.dumps({k: v for k, v in payload.items() if k != "figures"},
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"payload: {out} ({len(payload['figures'])} figures)")
    if args.publish:
        publish(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())