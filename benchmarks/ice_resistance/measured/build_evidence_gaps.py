#!/usr/bin/env python
"""Assemble ``data/measured/ice_resistance/evidence_gaps.json`` from the attempt log.

Gap entries are *not* free-text apologies: each one names the missing source, why
the 15 % criterion cannot do without it, the dated attempts that failed, and what
would unblock it.  The registry is generated from
``evidence_gaps/attempts_latest.json`` (written by ``probe_sources.py``) so that
the attempts cannot drift away from the claim.

Run::

    python benchmarks/ice_resistance/measured/probe_sources.py     # refresh attempts
    python benchmarks/ice_resistance/measured/build_evidence_gaps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorphysics.ice.measured_anchor import sha256_file  # noqa: E402
from tensorphysics.ice.measured_criterion import (  # noqa: E402
    evidence_gap_summary,
    validate_evidence_gaps,
)

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"
ATTEMPT_LOG = DATA_ROOT / "evidence_gaps" / "attempts_latest.json"
OUT = DATA_ROOT / "evidence_gaps.json"

GAP_TEXT: Dict[str, Dict[str, Any]] = {
    "GAP-LINDQVIST-1989": {
        "target": "Lindqvist (1989), 'A straightforward method for calculation of ice resistance of "
        "ships', POAC'89 -- the original resistance table and its constant set",
        "why_needed": "the 15 % verdict is currently taken against a *second-hand* transcription of the "
        "Lindqvist column (Hu & Zhou 2016, Table 8). Without the original the report cannot say whether "
        "the platform's Lindqvist branch or the source's transcription is the deviating one",
        "unblock": "POAC proceedings volume (paper-level PDF) or a library scan of the 1989 paper; "
        "table rows can be appended as a new series under measured_dataset/v1 without code changes",
        "status": "OPEN",
        "affects": ["C3 15pct verdict attribution", "parameter provenance of the bending closure"],
    },
    "GAP-RISKA-1997": {
        "target": "Riska (1997) level-ice resistance formulation and the Kämäräinen/Riska test tables",
        "why_needed": "the second published law family used as a cross-check; without the original its "
        "predictions stay second-hand as well, so a systematic transcription error could hide in both "
        "columns at once",
        "unblock": "Transport Canada / NRC report scan (TP 13110 series) or a university library copy",
        "status": "OPEN",
        "affects": ["cross-law comparison table", "closure re-derivation target"],
    },
    "GAP-MTUIKKU-PRIMARY-PDF": {
        "target": "Publisher PDF of Hu & Zhou (2016), Int. J. Naval Arch. Ocean Eng. 8(2) 169-176 "
        "(DOI 10.1016/j.ijnaoe.2016.01.004)",
        "why_needed": "series MTUIKKU-HZ2016 is the only series in the 15 % verdict set with measured "
        "resistance, and its digits currently come from an archived in-ecosystem transcription. "
        "Verification against the publisher PDF is the difference between 'MEASURED-PRIMARY' and "
        "'MEASURED-PRIMARY via a secondary source'",
        "unblock": "the article is gold OA (OpenAlex is_oa=true): the single blocking step is fetching "
        "https://www.sciencedirect.com/science/article/pii/S2092678215300133/pdf from an unfiltered host. "
        "One download closes the gap permanently because the tables are already transcribed and dual-entered",
        "status": "OPEN",
        "affects": ["C1 provenance grade", "C2 double entry (currently cannot check against the original)"],
    },
    "GAP-INDEPENDENT-DOWNWARD-SERIES": {
        "target": "A second test campaign with a *downward-bending* hull and measured resistance at "
        ">= 4 ice thicknesses (a different ship and basin than MT Uikku)",
        "why_needed": "with one campaign the FAIL verdict cannot be separated from a campaign-specific "
        "bias; the closure constant k_bend has no out-of-sample series to be validated on, which is "
        "exactly why the k_bend calibration of m9425 stays applied_to_validated_model=False",
        "unblock": "an OA thesis/paper with digitised level-ice resistance tables for a second hull; "
        "the upward-bending campaign ingested in this node (SALMI2014-UBIB) is independent but outside "
        "the downward-bending envelope, so it sharpens external validity without closing this gap",
        "status": "PARTIAL",
        "affects": ["out-of-sample validation of any closure re-calibration", "C3 verdict generality"],
    },
}


def main() -> int:
    log = json.loads(ATTEMPT_LOG.read_text(encoding="utf-8"))
    by_gap: Dict[str, List[Dict[str, Any]]] = {}
    for a in log["attempts"]:
        by_gap.setdefault(a["gap_id"], []).append(
            {
                "utc": a["utc"],
                "url": a["url"],
                "http_status": a["http_status"],
                "content_type": a.get("content_type"),
                "error": a.get("error"),
                "what": a["label"],
            }
        )

    gaps = []
    for gap_id, meta in GAP_TEXT.items():
        rows = by_gap.get(gap_id, [])
        entry = {"gap_id": gap_id, **meta, "attempts": rows}
        if gap_id == "GAP-INDEPENDENT-DOWNWARD-SERIES":
            entry["partial_progress"] = {
                "series_added": "SALMI2014-UBIB",
                "path": "series/salmi2014_ubib/series.json",
                "what_it_closes": "a second, independent measured campaign (different hull, different "
                "breaking mode, different decade of the same basin)",
                "what_it_does_not_close": "downward-bending coverage; the concept hull bends ice upwards",
                "sha256": sha256_file(DATA_ROOT / "series" / "salmi2014_ubib" / "series.json"),
            }
        gaps.append(entry)

    doc = {
        "schema_version": "1.0",
        "kind": "measured_evidence_gap_registry",
        "library": "data/measured/ice_resistance",
        "asset_mode": "measured_dataset/v1",
        "generated_by": "benchmarks/ice_resistance/measured/build_evidence_gaps.py",
        "statement": "A gap is registered only with dated retrieval attempts. No data point in this "
        "library was ever filled in from a source that is not archived under series/*/source/.",
        "attempt_log": {
            "path": "evidence_gaps/attempts_latest.json",
            "sha256": sha256_file(ATTEMPT_LOG),
            "n_attempts": log["n_attempts"],
            "n_reachable": log["n_reachable"],
            "generated_utc": log["generated_utc"],
        },
        "gaps": gaps,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    errors = validate_evidence_gaps(doc, root=DATA_ROOT)
    print(json.dumps({"out": str(OUT.relative_to(REPO)), "errors": errors,
                      "summary": evidence_gap_summary(doc)}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())