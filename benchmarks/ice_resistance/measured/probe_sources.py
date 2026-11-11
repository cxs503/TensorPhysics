#!/usr/bin/env python
"""Probe the public sources that the measured-anchor library still lacks.

The result is an *attempt log*: every row records what was tried, when, from
which worktree, and what came back (HTTP status / transport error).  The log is
the evidence behind ``data/measured/ice_resistance/evidence_gaps.json`` -- a gap
that cannot be closed must at least be *documented as closed-attempted*, with
the exact failure mode, otherwise the next node repeats the same dead ends.

Nothing here is allowed to invent data: a probe that fails simply records
``status: null`` and the transport error.

Usage::

    python benchmarks/ice_resistance/measured/probe_sources.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"
DEFAULT_OUT = DATA_ROOT / "evidence_gaps" / "attempts_latest.json"

UA = "Mozilla/5.0 (compatible; TensorPhysics measured-anchor probe; research use)"

# (gap_id, label, url) -- one row per *target*, several rows per gap allowed.
TARGETS: List[Dict[str, str]] = [
    # -- GAP-LINDQVIST-1989: the original POAC'89 paper and its tables ------
    {
        "gap_id": "GAP-LINDQVIST-1989",
        "label": "Lindqvist (1989) POAC'89 paper table (level-ice resistance)",
        "url": "https://www.poac.com/PapersOnline.html",
    },
    {
        "gap_id": "GAP-LINDQVIST-1989",
        "label": "Lindqvist 1989 via publisher landing page",
        "url": "https://doi.org/10.1016/0165-232X(89)90030-5",
    },
    # -- GAP-RISKA-1997: Riska (1997) / Kämäräinen tables -------------------
    {
        "gap_id": "GAP-RISKA-1997",
        "label": "Riska (1997) ice resistance model tables",
        "url": "https://www.atomicklin.com/riska",
    },
    {
        "gap_id": "GAP-RISKA-1997",
        "label": "Riska 1997 via NRC/Transport Canada landing page",
        "url": "https://nrc-publications.canada.ca/eng/search/?q=riska+ice+resistance",
    },
    # -- GAP-INDEPENDENT-DOWNWARD-SERIES: second downward-bending campaign --
    {
        "gap_id": "GAP-INDEPENDENT-DOWNWARD-SERIES",
        "label": "Aalto DA (DSpace 7 REST) discover query: level ice resistance model tests",
        "url": (
            "https://aaltodoc.aalto.fi/server/api/discover/search/objects?query="
            + urllib.parse.quote("level ice resistance model tests")
        ),
    },
    {
        "gap_id": "GAP-INDEPENDENT-DOWNWARD-SERIES",
        "label": "Core.ac.uk aggregate search",
        "url": "https://core.ac.uk/search?q=level%20ice%20resistance%20model%20tests",
    },
    {
        "gap_id": "GAP-INDEPENDENT-DOWNWARD-SERIES",
        "label": "DiVA portal (Nordic theses)",
        "url": "https://www.diva-portal.org/smash/resultList.jsf?query=ice+resistance+model+test",
    },
    # -- GAP-MTUIKKU-PRIMARY-PDF: primary PDF of series A -------------------
    {
        "gap_id": "GAP-MTUIKKU-PRIMARY-PDF",
        "label": "Hu & Zhou (2016) DOI landing page (series A primary source)",
        "url": "https://doi.org/10.1016/j.ijnaoe.2016.01.004",
    },
    {
        "gap_id": "GAP-MTUIKKU-PRIMARY-PDF",
        "label": "OpenAlex OA-status record for the series A DOI",
        "url": "https://api.openalex.org/works/doi:10.1016/j.ijnaoe.2016.01.004",
    },
    {
        "gap_id": "GAP-MTUIKKU-PRIMARY-PDF",
        "label": "Gold-OA PDF endpoint reported by OpenAlex (Elsevier ScienceDirect)",
        "url": "https://www.sciencedirect.com/science/article/pii/S2092678215300133/pdf",
    },
    {
        "gap_id": "GAP-MTUIKKU-PRIMARY-PDF",
        "label": "DOAJ record mirror (IJNAOE is DOAJ-listed)",
        "url": "https://doaj.org/article/a394245d3fac4a638ddb77eb95ebfb67",
    },
]


def probe(url: str, timeout: float = 20.0) -> Dict[str, Any]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    rec: Dict[str, Any] = {
        "url": url,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": None,
        "content_type": None,
        "bytes_read": 0,
        "first_bytes_sha256": None,
        "error": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as f:
            blob = f.read(4096)
            rec["http_status"] = int(getattr(f, "status", 0) or 0)
            rec["content_type"] = f.headers.get("content-type")
            rec["bytes_read"] = len(blob)
            import hashlib

            rec["first_bytes_sha256"] = hashlib.sha256(blob).hexdigest()
    except urllib.error.HTTPError as exc:  # type: ignore[union-attr]
        rec["http_status"] = int(exc.code)
        rec["error"] = f"HTTPError: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - transport failures are data here
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args(argv)

    rows = []
    for t in TARGETS:
        rec = probe(t["url"], timeout=args.timeout)
        rec.update({"gap_id": t["gap_id"], "label": t["label"]})
        rows.append(rec)
        print(
            f"{t['gap_id']:<32} {rec['http_status']!s:<5} "
            f"{(rec['error'] or rec['content_type'] or '')[:60]}  {t['url'][:70]}",
            file=sys.stderr,
        )

    doc = {
        "schema_version": "1.0",
        "kind": "retrieval_attempt_log",
        "worktree": str(REPO),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network_policy": "worktree egress is partially filtered; 403/timeout is a normal outcome",
        "n_attempts": len(rows),
        "n_reachable": sum(1 for r in rows if r["http_status"] == 200),
        "attempts": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(REPO)}  ({doc['n_reachable']}/{doc['n_attempts']} reachable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())