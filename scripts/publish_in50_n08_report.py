#!/usr/bin/env python
"""Publish the IN50-N08 report to TensorPlatform ``POST /tensor/reports``.

The payload is ``artifacts/in50_n08/report_payload.json``, produced by
``scripts/run_in50_n08_acceptance.py`` (figures already palette-compressed to
keep the body ~76 kB, far below the 250 kB endpoint limit).

Usage::

    PYTHONPATH=src python scripts/publish_in50_n08_report.py [--dry-run]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MISSION_ID = 5829
PAYLOAD = ROOT / "artifacts" / "in50_n08" / "report_payload.json"


def main() -> int:
    dry_run = "--dry-run" in sys.argv[1:]
    if not PAYLOAD.exists():
        print(f"missing payload {PAYLOAD}; run scripts/run_in50_n08_acceptance.py first")
        return 2
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    payload["mission_id"] = MISSION_ID
    body = json.dumps(payload)
    n_bytes = len(body.encode("utf-8"))
    print(f"payload bytes: {n_bytes} (figures: {len(payload.get('figures', []))})")
    if n_bytes > 250_000:
        print("refusing to post: body above the 250 kB endpoint limit")
        return 3
    if dry_run:
        pathlib.Path("/tmp/report_in50_n08.json").write_text(body, encoding="utf-8")
        print("dry run: wrote /tmp/report_in50_n08.json")
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    login = requests.post(f"{base}/auth/login",
                          json={"username": user, "password": password}, timeout=30)
    login.raise_for_status()
    data = login.json()
    token = data.get("access_token") or data.get("token")
    if not token:
        print("login response without token:", json.dumps(data)[:300])
        return 2
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(f"{base}/tensor/reports", headers=headers, data=body, timeout=180)
    print("POST /tensor/reports ->", resp.status_code)
    print(resp.text[:400])
    if resp.status_code not in (200, 201):
        return 1
    check = requests.get(f"{base}/tensor/reports", headers=headers,
                         params={"mission_id": MISSION_ID}, timeout=60)
    print("verify GET ->", check.status_code)
    try:
        rows = check.json()
        items = rows.get("items", rows) if isinstance(rows, dict) else rows
        print("reports for mission:", len(items))
    except Exception:                                        # pragma: no cover
        print(check.text[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())