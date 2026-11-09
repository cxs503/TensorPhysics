#!/usr/bin/env python3
"""Build and publish the P92W3-N09 report payload (figures inlined as base64).

The payload is always written to ``/tmp/report_m8455.json`` and to
``artifacts/criterion_drift_audit/report_payload.json`` so that a delayed
publisher (watchdog) can post it verbatim if the API is down at finish time.

Usage:
    python scripts/m8455_publish_report.py [--post]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "criterion_drift_audit"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
MISSION_ID = 8455


def _post(url: str, body: dict, token: str | None = None, timeout: float = 20.0) -> dict:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def login(base: str) -> str:
    body = {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }
    for path in ("/auth/login", "/login"):
        try:
            resp = _post(base + path, body)
        except Exception:
            continue
        token = resp.get("access_token") or resp.get("token")
        if token:
            return token
    raise RuntimeError("login failed on all known endpoints")


def build_payload() -> dict:
    metrics = json.loads((OUT / "report_metrics.json").read_text())
    md = (ROOT / "docs" / "reports" / "m8455_criterion_drift_audit.md").read_text()
    figs = []
    for name, caption in (
        ("fig1_field_f1",
         "Per-field extraction F1, legacy vs hardened extractor on the same 50-case synthetic corpus "
         "(sha256 quoted in title); dashed line = 0.95 gate; held-out 20-case micro-F1 0.9735."),
        ("fig2_drift_confusion",
         "Drift classification confusion matrix over the 24 labelled synthetic cases "
         "(12 unchanged, 3 tightened, 3 loosened, 3 scope_changed, 2 undefined, 1 unregistered)."),
    ):
        png = OUT / "figures" / f"{name}.png"
        figs.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(png.read_bytes()).decode(),
        })
    h = metrics["hardened"]
    summary = (
        "Criterion pre-registration matured: comparator-anchored six-field extraction (micro-F1 "
        f"{h['extraction_f1']:.4f} on 50 synthetic cases vs legacy {metrics['legacy_baseline']['extraction_f1']:.4f}), "
        f"drift detection on 24 labelled cases precision {h['detection_precision']:.4f} / recall "
        f"{h['detection_recall']:.4f}, fail-closed controls all unregistered, held-out 20-case F1 "
        f"{h['held_out_f1']:.4f} (measured once, 3 known residuals), zero domain-semantics scan hits = 0."
    )
    return {
        "title": "P92W3-N09 验收判据预注册与阈值漂移审计（成熟化）：比较符锚定抽取 + fail-closed 审计 + 可复现漂移基准",
        "summary": summary,
        "scenario_key": "campaign-criterion-preregistration-drift-audit",
        "mission_id": MISSION_ID,
        "tags": ["campaign", "acceptance-criteria", "pre-registration", "sha256", "drift-audit",
                 "fail-closed", "kpi", "benchmark", "tensorphysics"],
        "markdown": md,
        "figures": figs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args()

    payload = build_payload()
    blob = json.dumps(payload, ensure_ascii=False)
    for path in (Path("/tmp/report_m8455.json"), OUT / "report_payload.json"):
        path.write_text(blob, encoding="utf-8")
    print(f"payload {len(blob)} bytes -> /tmp/report_m8455.json, {OUT/'report_payload.json'}")

    if not args.post:
        return 0

    last = None
    for attempt in range(1, 4):
        try:
            token = login(API)
            resp = _post(API + "/tensor/reports", payload, token)
            print(f"published on attempt {attempt}: {json.dumps(resp)[:300]}")
            return 0
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"attempt {attempt} failed: {type(exc).__name__}: {exc}")
            time.sleep(2)
    print(f"PUBLISH FAILED (payload preserved on disk): {last}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())