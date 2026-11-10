#!/usr/bin/env python3
"""Snapshot campaign node acceptance artifacts into ``tensor_reports/campaign_nodes/``.

The revival registry is not allowed to invent its evidence: it reads the very
artifacts a campaign node produced.  This tool is the *only* writer of that
directory -- it copies, verbatim, the mission record (goal + campaign metadata)
and the linked acceptance report (markdown) from the platform API, and stamps
every file with a sha256 so downstream extraction can be refuted.

Usage::

    python tools/sync_campaign_nodes.py --plan 337 --missions 7827

Credentials come from ``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS``; nothing is logged (no token, no password).  The tool is
read-only against the platform (GET only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

OUT_DIR = Path(__file__).resolve().parents[1] / "tensor_reports" / "campaign_nodes"


def _login() -> str:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    body = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    req = urllib.request.Request(
        base + "/auth/login", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as fh:
        return json.load(fh)["access_token"]


def _get(token: str, path: str) -> dict:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    req = urllib.request.Request(
        base + path, headers={"Authorization": "Bearer " + token}
    )
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def _unwrap(payload):
    if isinstance(payload, dict) and "data" in payload and (
        "items" not in payload or payload.get("data") is None
    ):
        return payload["data"]
    return payload


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot(token: str, mission_id: int, plan_id: int | None) -> dict:
    mission = _unwrap(_get(token, "/tensor/missions/%d" % mission_id))
    reports = _unwrap(_get(token, "/tensor/reports?mission_id=%d" % mission_id))
    items = reports.get("items", reports) if isinstance(reports, dict) else reports
    report = None
    if items:
        rid = items[0]["id"]
        report = _unwrap(_get(token, "/tensor/reports/%d" % rid))
    node = None
    nodes_all: list[dict] = []
    if plan_id is not None:
        nodes = _unwrap(_get(token, "/tensor/campaigns/plans/%d/nodes" % plan_id))
        nodes_all = nodes.get("items", [])
        for cand in nodes_all:
            if cand.get("mission_id") == mission_id:
                node = cand
                break
    meta = (node or {}).get("metadata") or {}
    artifact = {
        "schema": "campaign_node_artifact.v1",
        "mission_id": mission_id,
        "plan_id": plan_id,
        "node_id": (node or {}).get("id"),
        "node_ext_id": meta.get("ext_id"),
        "node_status": (node or {}).get("status"),
        "node_dependencies": (node or {}).get("dependencies") or [],
        "depends_on_missions": sorted(
            {
                m
                for dep in ((node or {}).get("dependencies") or [])
                for m in [next((x.get("mission_id") for x in nodes_all if x.get("id") == dep), None)]
                if m
            }
        ),
        "reproduction_verdict": (node or {}).get("reproduction_verdict"),
        "mission_status": mission.get("status"),
        "goal": mission.get("goal"),
        "result_checks": (mission.get("result") or {}).get("checks")
        if isinstance(mission.get("result"), dict)
        else None,
        "report": None,
    }
    if report:
        md = report.get("markdown") or ""
        artifact["report"] = {
            "id": report.get("id"),
            "title": report.get("title"),
            "summary": report.get("summary"),
            "scenario_key": report.get("scenario_key"),
            "status": report.get("status"),
            "markdown": md,
            "sha256": sha256_text(md),
            "title_sha256": sha256_text(report.get("title") or ""),
        }
    return artifact


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=int, default=337)
    ap.add_argument("--missions", type=int, nargs="*", default=[7827])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    token = _login()
    args.out.mkdir(parents=True, exist_ok=True)
    plan_nodes = _unwrap(_get(token, "/tensor/campaigns/plans/%d/nodes" % args.plan))
    plan_missions = [n["mission_id"] for n in plan_nodes.get("items", []) if n.get("mission_id")]
    targets = [(m, args.plan) for m in plan_missions] + [(m, None) for m in args.missions]
    written = []
    for mission_id, plan_id in targets:
        art = snapshot(token, mission_id, plan_id)
        path = args.out / ("m%d.json" % mission_id)
        path.write_text(json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        written.append(str(path))
        print("wrote %s (report=%s)" % (path, (art.get("report") or {}).get("id")), file=sys.stderr)
    print(json.dumps({"written": written, "count": len(written)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())