#!/usr/bin/env python3
"""Assemble the lineage-debt replay bundle from a raw campaign snapshot.

The raw snapshot under ``artifacts/lineage_debt/inputs/raw`` was pulled read-only
from the campaign API (``/tensor/campaigns/plans/<id>``,
``/tensor/campaigns/plans/<id>/nodes``, ``/tensor/reports?mission_id=<id>`` and
``/tensor/reports/<id>``) at the timestamp recorded in ``PROVENANCE`` below.
This script only *re-packs* those files into one deterministic bundle and
records a sha256 for every input, so the replay is reproducible offline and the
report can point at an exact byte image of the evidence it judged.

Usage::

    python scripts/gen_lineage_debt_bundle.py            # writes inputs/lineage_debt_bundle.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "lineage_debt" / "inputs" / "raw"
OUT = ROOT / "artifacts" / "lineage_debt" / "inputs" / "lineage_debt_bundle.json"

PLAN_IDS = (26, 173, 388)
DOWNSTREAM_PLAN = 388

PROVENANCE = {
    "api_base": "http://127.0.0.1:8010/api/v1",
    "fetched_utc": "2026-11-09T03:40:00Z",
    "endpoints": [
        "GET /tensor/campaigns/plans/{26,173,388}",
        "GET /tensor/campaigns/plans/{26,173,388}/nodes",
        "GET /tensor/reports?mission_id=<mission>&limit=10",
        "GET /tensor/reports/<report_id>",
    ],
    "note": ("read-only snapshot; reports are the published campaign reports of the plan's "
             "own nodes, plans 26/173/388 are the parent plan, the owner plan and the "
             "successor wave named by metadata.autonomy.new_plan"),
    "secrets": "none (token used at fetch time was not persisted)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    plans_raw = json.loads((RAW / "plans.json").read_text(encoding="utf-8"))
    plans = {str(pid): plans_raw[str(pid)] for pid in PLAN_IDS}
    nodes = {str(pid): json.loads((RAW / f"nodes_{pid}.json").read_text(encoding="utf-8"))["items"]
             for pid in PLAN_IDS}

    reports = {}
    for path in sorted((RAW / "reports").glob("*.json")):
        rep = json.loads(path.read_text(encoding="utf-8"))
        if rep.get("markdown") is None:
            continue
        reports[str(rep["id"])] = rep

    inputs = {str(p.relative_to(RAW)): sha256(p) for p in sorted(RAW.rglob("*.json"))}

    bundle = {
        "schema": "lineage_debt_input.v1",
        "fetched_utc": PROVENANCE["fetched_utc"],
        "source": PROVENANCE,
        "provenance": {"input_sha256": inputs},
        "plans": plans,
        "nodes": nodes,
        "reports": reports,
        "downstream_plan": DOWNSTREAM_PLAN,
    }
    OUT.write_text(json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "bundle": str(OUT.relative_to(ROOT)),
        "sha256": sha256(OUT),
        "plans": sorted(plans),
        "nodes": {k: len(v) for k, v in sorted(nodes.items())},
        "reports": len(reports),
        "input_files": len(inputs),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())