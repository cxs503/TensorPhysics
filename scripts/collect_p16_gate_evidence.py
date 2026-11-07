#!/usr/bin/env python3
"""Collect the raw evidence that the P16 campaign integration gate evaluates.

The gate itself (:mod:`tensorphysics.campaign.integration_gate`) is *offline and
deterministic*: it consumes JSON snapshots so that every number in the campaign
acceptance report is traceable to a recorded artifact.  This script produces
those snapshots from authoritative sources and never invents a value:

1. ``campaign_nodes.json``   -- plan 33 node table + acceptance criteria (platform API)
2. ``node_evidence.json``    -- per-node delivery state N1..N8 (platform API mission records)
3. ``p15_baseline.json``     -- the 11 P15 validation probes + recorded mission checks
4. ``ship_platform_facts.json`` -- ship-platform OpenAPI facts for the two cross-domain
   endpoints (``/calculations/wind_resistance``, ``/fem/modal``)

Usage::

    python3 scripts/collect_p16_gate_evidence.py [--out artifacts/p16_gate]

Credentials come from ``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS`` (same convention as the platform report tooling).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone

P16_PLAN_ID = 33
P15_PLAN_ID = 15
SHIP_API = os.environ.get("SHIP_API_BASE", "http://127.0.0.1:8002")


def _http(url: str, token: str | None = None, body: dict | None = None, timeout: float = 30.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def login() -> tuple[str, str]:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    status, raw = _http(
        f"{base}/auth/login",
        body={
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
    )
    payload = json.loads(raw)
    return payload.get("access_token") or payload["token"], base


def api_get(base: str, token: str, path: str):
    return json.loads(_http(f"{base}{path}", token=token)[1])


TEST_TARGET_RE = re.compile(r"(tests/[\w/]+\.py)")
COUNT_RES = [
    re.compile(r"(\d+)\s*/\s*(\d+)\s*passed"),          # "19/19 passed"
    re.compile(r"(\d+)\s+passed(?:,\s*(\d+)\s+failed)?"),  # "16 passed"
    re.compile(r"(\d+)\s+passed"),                      # fallback
]


def parse_checks(checks: str) -> dict:
    """Extract machine-usable probe facts from a P15 mission ``checks`` string."""
    targets = sorted(set(TEST_TARGET_RE.findall(checks or "")))
    passed = total = failed = None
    for pat in COUNT_RES[:2]:
        m = pat.search(checks or "")
        if m:
            g = [x for x in m.groups() if x is not None]
            if len(g) == 2:
                passed, total = int(g[0]), int(g[1])
                failed = total - passed
            else:
                passed = int(g[0])
            break
    return {
        "test_targets": targets,
        "n_passed_recorded": passed,
        "n_total_recorded": total,
        "n_failed_recorded": failed,
        "checks_text": (checks or "")[:600],
    }


def collect(out: pathlib.Path, *, plan_p16: int = P16_PLAN_ID, plan_p15: int = P15_PLAN_ID,
            ship_api: str = SHIP_API) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    token, base = login()
    stamp = datetime.now(timezone.utc).isoformat()

    # ---- 1. P16 campaign node table -------------------------------------------
    p16 = api_get(base, token, f"/tensor/campaigns/plans/{plan_p16}/nodes")["items"]
    plan16 = api_get(base, token, f"/tensor/campaigns/plans/{plan_p16}")
    nodes = []
    for n in p16:
        md = n.get("metadata") or {}
        nodes.append(
            {
                "plan_id": n["plan_id"],
                "node_id": n["id"],
                "ext_id": md.get("ext_id"),
                "name": n["name"],
                "repo": md.get("repo"),
                "mission_id": n.get("mission_id"),
                "status": n.get("status"),
                "reproduction_verdict": n.get("reproduction_verdict"),
                "consecutive_fails": n.get("consecutive_fails"),
                "acceptance_criteria": md.get("acceptance_criteria") or [],
                "goal": md.get("goal"),
                "depends_on": n.get("metadata", {}).get("depends_on"),
            }
        )
    campaign = {
        "schema": "p16_campaign_nodes.v1",
        "collected_utc": stamp,
        "source": {
            "api": f"{base}/tensor/campaigns/plans/{plan_p16}",
            "plan_title": plan16.get("title"),
            "plan_status": plan16.get("status"),
            "physics_scope": plan16.get("physics_scope"),
            "coupling_framework_requirements": plan16.get("coupling_framework_requirements"),
            "validation_standards": plan16.get("validation_standards"),
            "planned_start": plan16.get("planned_start"),
            "planned_end": plan16.get("planned_end"),
        },
        "nodes": nodes,
    }
    (out / "campaign_nodes.json").write_text(json.dumps(campaign, ensure_ascii=False, indent=1))

    # ---- 2. per-node delivery evidence (N1..N8 + gate) ------------------------
    evidence = {
        "schema": "p16_node_evidence.v1",
        "collected_utc": stamp,
        "nodes": {},
    }
    for n in nodes:
        ext = n["ext_id"] or ""
        if not ext.startswith("P16-N"):
            continue
        mid = n["mission_id"]
        rec = {"mission_id": mid, "node_status": n["status"], "delivered": False,
               "received_utc": stamp}
        if mid:
            try:
                m = api_get(base, token, f"/tensor/missions/{mid}")
                res = m.get("result") or {}
                rec.update(
                    {
                        "mission_status": m.get("status"),
                        "accepted": bool(res.get("accepted")),
                        "checks": (res.get("checks") or "")[:1200],
                        "evidence": (res.get("evidence") or None),
                        "headline": res.get("headline"),
                        "result_keys": sorted(res.keys()),
                    }
                )
                # A node counts as delivered only when the platform recorded an
                # accepted result AND the campaign node itself is closed as done.
                rec["delivered"] = bool(res.get("accepted")) and n["status"] in (
                    "done", "completed", "succeeded", "passed",
                )
            except urllib.error.HTTPError as exc:  # pragma: no cover - network
                rec["error"] = f"HTTP {exc.code}"
        rec["reason_if_not_delivered"] = (
            None if rec["delivered"] else
            f"campaign node status={n['status']!r}, mission accepted="
            f"{rec.get('accepted')!r}: no machine-readable acceptance artifact delivered yet"
        )
        evidence["nodes"][ext] = rec
    (out / "node_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1))

    # ---- 3. P15 baseline probes (11 delivered nodes) -------------------------
    p15 = api_get(base, token, f"/tensor/campaigns/plans/{plan_p15}/nodes")["items"]
    p15_nodes = []
    for n in p15:
        md = n.get("metadata") or {}
        ext = md.get("ext_id") or ""
        if not ext.startswith("IN14-D"):
            continue
        mid = n.get("mission_id")
        probe = {"test_targets": [], "n_passed_recorded": None, "n_total_recorded": None,
                 "n_failed_recorded": None, "checks_text": ""}
        mission_status = None
        if mid:
            m = api_get(base, token, f"/tensor/missions/{mid}")
            res = m.get("result") or {}
            mission_status = m.get("status")
            probe = parse_checks(res.get("checks") or "")
            probe["accepted"] = bool(res.get("accepted"))
            probe["headline"] = res.get("headline")
        p15_nodes.append(
            {
                "ext_id": ext,
                "node_id": n["id"],
                "name": n["name"],
                "repo": md.get("repo"),
                "mission_id": mid,
                "node_status": n.get("status"),
                "mission_status": mission_status,
                "probe": probe,
            }
        )
    plan15 = api_get(base, token, f"/tensor/campaigns/plans/{plan_p15}")
    baseline = {
        "schema": "p15_regression_baseline.v1",
        "collected_utc": stamp,
        "source": {"api": f"{base}/tensor/campaigns/plans/{plan_p15}", "plan_id": plan_p15,
                   "plan_status": plan15.get("status"),
                   "plan_updated_at": plan15.get("updated_at"),
                   "plan_planned_end": plan15.get("planned_end")},
        "nodes": p15_nodes,
    }
    (out / "p15_baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=1))

    # ---- 4. ship-platform cross-domain endpoint facts -------------------------
    facts = {"schema": "ship_platform_facts.v1", "collected_utc": stamp, "base": ship_api,
             "endpoints": {}}
    try:
        _, raw = _http(f"{ship_api}/openapi.json")
        spec = json.loads(raw)
        paths = spec.get("paths", {})
        schemas = spec.get("components", {}).get("schemas", {})
    except Exception as exc:  # pragma: no cover - network
        spec, paths, schemas = None, {}, {}
        facts["error"] = repr(exc)
    for ep, keys, related in (
        ("/api/v1/calculations/wind_resistance", ("Wind",), "wind"),
        ("/api/v1/fem/modal", ("Modal",), "modal"),
    ):
        entry = {
            "present": ep in paths,
            "methods": sorted(paths.get(ep, {}).keys()),
            "related_paths": sorted(p for p in paths if related in p.lower()),
        }
        sch = {}
        for name, blob in schemas.items():
            if any(k in name for k in keys):
                sch[name] = blob
        entry["schemas"] = sch
        facts["endpoints"][ep] = entry
    facts["n_paths"] = len(paths)
    (out / "ship_platform_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1))

    return {"campaign_nodes": len(nodes), "node_evidence": len(evidence["nodes"]),
            "p15_nodes": len(p15_nodes), "ship_paths": facts["n_paths"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/p16_gate")
    ap.add_argument("--ship-api", default=SHIP_API)
    args = ap.parse_args()
    summary = collect(pathlib.Path(args.out), ship_api=args.ship_api)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()