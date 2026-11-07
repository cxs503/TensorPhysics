#!/usr/bin/env python3
"""Publish the P76-N09 report to the platform report API.

Credentials are read from the environment (TENSOR_AGENT_USER / TENSOR_AGENT_PASS
/ TENSOR_API_BASE); nothing secret goes on the command line.  The payload is the
JSON produced by ``scripts/spectral_fatigue_report.py``.
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
PAYLOAD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/report.json"


def post(path, body, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


status, body = 0, ""
for path in ("/login", "/auth/login"):
    status, body = post(path, {"username": USER, "password": PASS})
    if status == 200:
        break
if status != 200:
    print("login failed", status, body[:300])
    raise SystemExit(1)
token = json.loads(body).get("access_token") or json.loads(body).get("token")
print("login ok, token len", len(token or ""))

payload = json.load(open(PAYLOAD))
print("payload bytes", os.path.getsize(PAYLOAD),
      "figures", len(payload.get("figures", [])),
      "markdown chars", len(payload.get("markdown", "")))
status, body = post("/tensor/reports", payload, token)
print("report post ->", status)
print(body[:600])