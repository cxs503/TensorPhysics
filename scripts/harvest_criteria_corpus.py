#!/usr/bin/env python3
"""Harvest the historical acceptance-criterion corpus from published platform reports.

Read-only: pulls ``GET /tensor/reports`` (+ per-report detail) from TensorPlatform, extracts
every criterion row of the ⑥ error-verdict tables, and freezes them to JSONL with the
SHA-256 of the source Markdown.  No domain repository is touched; only report text is read.

Usage::

    python3 scripts/harvest_criteria_corpus.py --out artifacts/criteria_metrology/historical_criteria.jsonl \
        --max-reports 160

Credentials come from ``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorphysics.metrology.criteria_semantics import corpus as C  # noqa: E402
from tensorphysics.metrology.criteria_semantics import replay as R  # noqa: E402


def login(base: str) -> str:
    import urllib.request
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - local service
        d = json.loads(r.read().decode())
    return d.get("access_token") or d.get("token")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/criteria_metrology/historical_criteria.jsonl")
    ap.add_argument("--max-reports", type=int, default=160)
    ap.add_argument("--cache-dir", default="/tmp/tp_report_cache")
    ap.add_argument("--summary", default="artifacts/criteria_metrology/corpus_summary.json")
    ap.add_argument("--mission-ids", default="",
                    help="comma separated mission ids to fetch explicitly (case studies)")
    args = ap.parse_args()

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    token = login(base)
    docs = C.harvest_report_docs(base, token, max_reports=args.max_reports,
                                 cache_dir=args.cache_dir)
    mids = [int(m) for m in args.mission_ids.split(",") if m.strip()]
    if mids:
        extra = C.harvest_report_docs(base, token, max_reports=len(mids),
                                      cache_dir=args.cache_dir, mission_ids=mids)
        have = {d.report_id for d in docs}
        docs += [d for d in extra if d.report_id not in have]
    probe = C.score_docs(docs)
    rows = C.corpus_from_docs(docs)
    meta = {
        "source": "TensorPlatform /tensor/reports (published reports, Markdown ⑥ tables)",
        "n_reports_scanned": probe["n_reports"],
        "n_criteria": len(rows),
        "reports_with_criteria": sum(1 for r in probe["reports"] if r["n_criteria"]),
        "explicit_mission_ids": mids,
        "extractor": "tensorphysics.metrology.criteria_semantics.parser",
    }
    man = C.write_corpus(rows, args.out, meta=meta)
    ledger = R.replay_corpus(rows)
    summary = {k: v for k, v in ledger.items() if k != "entries"} | {
        "manifest": man, "corpus_meta": meta,
        "per_report": probe["reports"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ("n_criteria", "n_degenerate", "degeneracy_rate", "by_type", "flips",
                       "recorded_flips", "n_reports")}, ensure_ascii=False, indent=1))
    print("wrote", args.out, man["sha256"][:16])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())