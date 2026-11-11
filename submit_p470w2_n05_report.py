#!/usr/bin/env python
"""Submit the p470w2-n05 FEM→DEM energy conservation report to the platform API."""

from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPORT_PATH = HERE / "artifacts" / "p470w2_n05" / "report.json"
MISSION_ID = 13874
SCENARIO_KEY = "p470w2-n05-fem-dem-energy"
TITLE = "FEM→DEM Ice Plate Fracture Energy Conservation Contract"


def _login(base: str) -> str:
    """Login and return access token."""
    req = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main():
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

    # Load report
    with open(REPORT_PATH) as f:
        report = json.load(f)

    # Login
    token = _login(base)

    # Submit
    url = base + "/tensor/reports"
    req = urllib.request.Request(
        url,
        data=json.dumps(report).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = resp.read().decode()
            print(f"Status: {resp.status}")
            print(f"Response: {result[:500]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(f"Response: {e.read().decode()[:500]}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
