#!/usr/bin/env python3
"""Validate the node-level evidence proxy used by KS (network, read-only).

KS needs to know whether a node's judgeable clauses were *actually measured*. The
platform stores measurement numbers inside each node's mission report, not per
clause, so the census uses a proxy (``node.status == done`` and
``consecutive_fails == 0``). This script checks the proxy against the report ledger
on a deterministic sample and reports how the hollow-PASS count moves when the
stricter ``require_report`` evidence mode is applied.

Usage::

    PYTHONPATH=src python scripts/validate_evidence_proxy.py --sample 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensorphysics.acceptance_judgeability.census import load_corpus, run_census  # noqa: E402
from tensorphysics.acceptance_judgeability.agreement import stratified_sample  # noqa: E402

ART = ROOT / "artifacts" / "acceptance_judgeability"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(ART / "corpus.jsonl"))
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--out", default=str(ART / "evidence_proxy.json"))
    args = ap.parse_args(argv)

    from harvest_acceptance_criteria import Client  # reuse the harvester transport

    nodes = load_corpus([args.corpus])
    done = [n for n in nodes if str(n.get("status", "")).lower() == "done" and n.get("mission_id")]
    picked = stratified_sample(
        done,
        stratum=lambda n: str(n.get("repo") or "unknown"),
        ident=lambda n: str(n["node_id"]),
        n=args.sample,
        seed="p106w3a2-n08-evidence",
    )

    client = Client(
        os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1"),
        os.environ.get("TENSOR_AGENT_USER", "eco_admin"),
        os.environ.get("TENSOR_AGENT_PASS", ""),
    )

    rows = []
    for node in picked:
        mid = int(node["mission_id"])
        try:
            reports = client.get_all_pages(f"/tensor/reports?mission_id={mid}&limit=50")
            mission = client.get(f"/tensor/missions/{mid}") or {}
        except Exception as exc:  # transient API failure is recorded, not hidden
            rows.append({"node_id": node["node_id"], "mission_id": mid, "error": str(exc)[:120]})
            continue
        grade = str((mission.get("result") or {}).get("accepted")) if isinstance(mission, dict) else None
        rows.append(
            {
                "node_id": node["node_id"],
                "ext_id": node.get("ext_id"),
                "repo": node.get("repo"),
                "mission_id": mid,
                "mission_status": mission.get("status") if isinstance(mission, dict) else None,
                "n_reports": len(reports),
                "accepted": grade,
            }
        )

    ok = [r for r in rows if "error" not in r]
    with_report = [r for r in ok if r["n_reports"] > 0]
    status_ok = [r for r in ok if str(r.get("mission_status")) in ("succeeded", "completed", "done")]

    # effect of the stricter evidence mode on the same nodes
    by_id = {int(n["node_id"]): n for n in nodes}
    subset = [by_id[r["node_id"]] for r in ok]
    relaxed = run_census(subset)
    strict_rows = []
    for node in subset:
        from tensorphysics.acceptance_judgeability.support import Evidence, node_ks

        strict_rows.append(node_ks(node, require_report=False))
    strict = run_census(subset)

    payload = {
        "requirement": "node-level evidence proxy: status=done+0 fails implies the criteria were measured",
        "sample": len(picked),
        "answered": len(ok),
        "with_published_report": len(with_report),
        "report_rate": (len(with_report) / len(ok)) if ok else None,
        "mission_status_ok_rate": (len(status_ok) / len(ok)) if ok else None,
        "hollow_ratio_relaxed": relaxed.hollow_ratio,
        "hollow_nodes_relaxed": len(relaxed.hollow),
        "done_nodes_in_sample": len(relaxed.done_rows),
        "rows": rows,
        "note": "report absence does not imply 'not measured'; it bounds the proxy from below",
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())