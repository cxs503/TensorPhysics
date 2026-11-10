#!/usr/bin/env python3
"""Fetch mission 7245's result headline and extract the case-study numbers by regex.

m7245 published no report (the mission was closed by PR merge), so its numbers live in the
mission record's ``result.headline``.  We do **not** re-type them: the headline is stored here
verbatim with its SHA-256 and the numbers are extracted by regex at run time (zero hand-copy).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request

OUT = "artifacts/criteria_metrology/mission_7245_headline.json"


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - local service
        tok = json.loads(r.read().decode())["access_token"]
    req = urllib.request.Request(base + "/tensor/missions/7245",
                                 headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        mission = json.loads(r.read().decode())
    result = mission.get("result") or {}
    headline = (result.get("headline") or (result.get("evidence") or {})
                .get("summary", {}).get("headline") or "")
    if not headline:
        raise SystemExit("no headline in mission 7245")

    stop = re.search(r"停船距离\s*([\d.]+)\s*m", headline)
    length = re.search(r"L\s*=\s*([\d.]+)\s*m", headline)
    imo = re.search(r"≤\s*(\d+)\s*L", headline)
    kt = re.search(r"K_T\s*误差\s*([\d.]+)\s*%", headline)
    if not (stop and length and imo and kt):
        raise SystemExit(f"cannot extract all numbers from headline: {headline}")
    imo_20L = float(length.group(1)) * float(imo.group(1))
    art = {
        "mission_id": 7245,
        "headline": headline,
        "sha256": hashlib.sha256(headline.encode()).hexdigest(),
        "reference_hint": "IMO 紧急停船衡准 ≤20L；K_T 对标 van Lammeren 四象限多项式",
        "recorded_verdict": "pass (mission result 'accepted': true; no report published)",
        "extracted": {"s_stop_m": float(stop.group(1)), "L_m": float(length.group(1)),
                      "imo_multiples": float(imo.group(1)), "imo_20L_m": imo_20L,
                      "kT_err": float(kt.group(1))},
        "extraction_rules": {"s_stop_m": r"停船距离\s*([\d.]+)\s*m",
                             "L_m": r"L\s*=\s*([\d.]+)\s*m",
                             "imo_multiples": r"≤\s*(\d+)\s*L",
                             "kT_err": r"K_T\s*误差\s*([\d.]+)\s*%"},
        "consumed_by": "scripts/criteria_case_studies.py",
    }
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(art, fh, ensure_ascii=False, indent=1)
    print("wrote", OUT, art["extracted"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())