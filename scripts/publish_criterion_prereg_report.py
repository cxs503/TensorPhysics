#!/usr/bin/env python3
"""Publish the p92w2b-n07 criterion pre-registration report (paper-level).

Reads ``artifacts/criterion_prereg/eval.json`` + ``docs/CRITERION_PREREGISTRATION.md``
(six-element report, numbers rendered from the artifact) and the four PNG
figures, then POSTs it to the platform report API.  Credentials come from the
environment (TENSOR_AGENT_USER / TENSOR_AGENT_PASS, optional TENSOR_API_BASE);
nothing secret is written to disk or printed.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 7803
ART = ROOT / "artifacts" / "criterion_prereg"

CAPTIONS = {
    "fig_confusion_matrix.png": "Fig. 1 - drift classification confusion (24-case synthetic set, main seed): rows = generator truth, columns = audit output.",
    "fig_baseline_vs_audit.png": "Fig. 2 - structured audit vs the naive goal-text-digest baseline: identical recall, precision 1.000 vs 0.542 (reformatting and legitimate tightening are not drift).",
    "fig_extractor_field_agreement.png": "Fig. 3 - extractor field-level agreement per field against generator-side ground truth, 0.95 acceptance line.",
    "fig_demo_kpi.png": "Fig. 4 - true-class histogram of the detection set (left) and the real-goal demo KPI over five receipt scenarios (right).",
}


def post(path: str, body: dict, token: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def build_payload() -> dict:
    ev = json.loads((ART / "eval.json").read_text(encoding="utf-8"))
    acc = ev["acceptance"]
    markdown = (ROOT / "docs" / "CRITERION_PREREGISTRATION.md").read_text(encoding="utf-8")
    figures = []
    for name, caption in CAPTIONS.items():
        blob = base64.b64encode((ART / name).read_bytes()).decode("ascii")
        figures.append({"name": name, "caption": caption, "png_base64": blob})
    summary = (
        "Campaign-node acceptance criteria are now pre-frozen, post-hoc auditable and drift-decidable: "
        f"structured extraction -> canonical JSON -> sha256 into mission metadata; drift classes "
        f"unchanged/tightened/loosened/scope_changed/undefined with a fail-closed verdict. "
        f"Measured: 24-case detection recall {acc['a_detection_24cases']['recall']:.4f} / precision "
        f"{acc['a_detection_24cases']['precision']:.4f} (>=0.92/0.95), extractor field F1 "
        f"{acc['b_extractor_field_f1']['f1']:.4f} (>=0.95), empty goal -> {acc['c_fail_closed']['empty_goal_state']}, "
        f"tampered digest -> {acc['c_fail_closed']['tampered_state']}, zero-domain hits "
        f"{acc['d_zero_domain_semantics']['hits']} over {acc['d_zero_domain_semantics']['n_files']} files. "
        f"Node verdict: {'ALL PASS' if ev['all_acceptance_pass'] else 'NOT ALL PASS'}."
    )
    return {
        "title": "Criterion pre-registration: freeze, audit and drift detection for campaign-node acceptance criteria (p92w2b-n07)",
        "summary": summary,
        "scenario_key": "p92w2b-n07-criterion-preregistration",
        "mission_id": MISSION_ID,
        "tags": ["campaign", "criterion-preregistration", "drift-audit", "fail-closed", "reproducibility", "p92w2b"],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> int:
    payload = build_payload()
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(payload), encoding="utf-8")
    print("payload bytes", out.stat().st_size, "figures", len(payload["figures"]), "markdown chars", len(payload["markdown"]))
    status, body = 0, ""
    for path in ("/login", "/auth/login"):
        status, body = post(path, {"username": USER, "password": PASS})
        if status == 200:
            break
    if status != 200:
        print("login failed", status, body[:300])
        return 1
    token = json.loads(body).get("access_token") or json.loads(body).get("token")
    print("login ok, token len", len(token or ""))
    status, body = post("/tensor/reports", payload, token)
    print("report post ->", status)
    print(body[:400])
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    sys.exit(main())