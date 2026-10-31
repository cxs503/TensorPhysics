"""Publish the W4-N03 report to the platform reports endpoint."""

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, json.loads(response.read().decode())


status, body = post("/auth/login", {"username": USER, "password": PASS})
token = body.get("access_token") or body.get("token")
print("login", status, "token" if token else body)

report = json.loads(open("/tmp/report.json").read())
status, body = post("/tensor/reports", report, token)
print("report", status, json.dumps(body)[:400])