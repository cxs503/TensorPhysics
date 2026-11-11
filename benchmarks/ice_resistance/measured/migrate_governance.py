#!/usr/bin/env python
"""Add the ``measured_dataset/v1`` governance block to every archived series.

The first measured-anchor node (m9425) shipped a schema that carried *provenance*
(citation, sources, digests) but left four of the fields the campaign contract
asks for implicit:

=================  ==========================================================
contract field     where it lives before this migration
=================  ==========================================================
source             provenance.citation / provenance.sources            (ok)
bench (facility)   implicit, sometimes in prose only                    (gap)
scale              provenance.scale (one series only)                  (gap)
ice thickness      points[].h_m                                        (ok)
speed              points[].V_ms                                       (ok)
ship type          ship.name / ship.source                             (ok)
uncertainty        points[].R_unc_kN, but never declared                (gap)
digitisation       provenance.sources[].kind, implicit                 (gap)
=================  ==========================================================

This migration makes the four missing ones *declared and machine-checked*
(``tensorphysics.ice.measured_criterion.validate_governance``).  It rewrites
``series.json`` files in place and refreshes the registry digests, so the asset
version stays ``measured_dataset/v1`` and every existing byte-level check keeps
working -- the digests simply move with the new content.

Run::

    python benchmarks/ice_resistance/measured/migrate_governance.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorphysics.ice.measured_anchor import sha256_file  # noqa: E402

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"

GOVERNANCE: Dict[str, Dict[str, Any]] = {
    "MTUIKKU-HZ2016": {
        "facility": {
            "name": None,
            "kind": "ice_basin",
            "model_scale": 31.56,
            "reported_frame": "full_scale",
            "model_ice": "not carried by the archived transcription",
            "known_gaps": [
                "basin name/licence: the archived in-ecosystem transcription did not carry it",
                "model-ice preparation procedure and target strengths",
            ],
        },
        "digitisation": {
            "method": "two independent transcriptions of the archived Table 3/8 transcription",
            "passes": ["pass_a", "pass_b"],
            "independent_paths": [
                "pass A: column-major CSV, SI-derived units (kPa/MPa) kept",
                "pass B: row-major text, engineering units (MPa/GPa) converted inline",
            ],
            "checks": ["double_entry agreement (sample + full population)", "digest re-verification"],
            "tool": "benchmarks/ice_resistance/measured/build_assets.py",
        },
        "uncertainty": {
            "reported": True,
            "definition": "per-point standard deviation of the measured towing force, tabulated next to "
            "the mean force by the source",
            "value_kN": "points[].R_unc_kN",
        },
        "applicability": {
            "ice_breaking_mode": "downward_bending",
            "lindqvist_family_applicable": True,
            "in_15pct_verdict_set": True,
            "role": "primary adjudication truth for the 15 % criterion",
            "limitations": [
                "one ship, one campaign: cannot separate ice-strength from geometry effects",
                "digitised summary forces only, not the raw force time series",
                "secondary transcription (publisher PDF not reachable from the worktree)",
            ],
        },
    },
    "AGULHAS2-FS2012": {
        "facility": {
            "name": "S.A. Agulhas II, Baltic Sea ice trials 2012-03-21/22",
            "kind": "full_scale_trial",
            "model_scale": None,
            "reported_frame": "full_scale",
            "model_ice": None,
            "known_gaps": ["no measured thrust or resistance: only speed and an engine-power band"],
        },
        "digitisation": {
            "method": "two independent transcriptions of the reported trial periods",
            "passes": ["pass_a", "pass_b"],
            "independent_paths": [
                "pass A: column-major CSV of the trial period table",
                "pass B: row-major text of the same periods",
            ],
            "checks": ["double_entry agreement", "digest re-verification"],
            "tool": "benchmarks/ice_resistance/measured/build_assets.py",
        },
        "uncertainty": {
            "reported": False,
            "definition": "the source reports an engine-power band, not a per-point uncertainty",
            "value_kN": None,
        },
        "applicability": {
            "ice_breaking_mode": "downward_bending",
            "lindqvist_family_applicable": True,
            "in_15pct_verdict_set": False,
            "role": "power-band screening only; cannot support a 15 % claim",
            "limitations": ["no resistance measurement", "bow angles assumed, not measured"],
        },
    },
    "SALMI2014-UBIB": {
        # governance block is written by build_salmi2014_series.py; listed here so
        # that --check reports a complete picture of the library.
    },
}

SERIES_FILES = {
    "MTUIKKU-HZ2016": DATA_ROOT / "series" / "mtuikku_hz2016" / "series.json",
    "AGULHAS2-FS2012": DATA_ROOT / "series" / "agulhas2_fs2012" / "series.json",
    "SALMI2014-UBIB": DATA_ROOT / "series" / "salmi2014_ubib" / "series.json",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report missing governance blocks, write nothing")
    args = ap.parse_args(argv)

    registry_path = DATA_ROOT / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    by_id = {e["series_id"]: e for e in registry["series"]}
    changed = []

    for sid, path in SERIES_FILES.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        block = GOVERNANCE.get(sid) or {}
        missing = [k for k in ("facility", "digitisation", "uncertainty", "applicability") if k not in doc]
        if args.check:
            print(f"{sid:<18} missing governance keys: {missing}")
            continue
        if missing and block:
            for key, value in block.items():
                doc[key] = value
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changed.append(sid)

    if args.check:
        return 0

    for sid, path in SERIES_FILES.items():
        digest = sha256_file(path)
        entry = by_id.get(sid)
        if entry is None:
            entry = {
                "series_id": sid,
                "path": str(path.relative_to(DATA_ROOT)),
                "grade": json.loads(path.read_text(encoding="utf-8"))["provenance"]["grade"],
                "measured_quantity": json.loads(path.read_text(encoding="utf-8"))["measured_quantity"],
                "n_points": len(json.loads(path.read_text(encoding="utf-8"))["points"]),
                "doi": json.loads(path.read_text(encoding="utf-8"))["provenance"].get("doi"),
            }
            registry["series"].append(entry)
            by_id[sid] = entry
        entry["sha256"] = digest
        entry["path"] = str(path.relative_to(DATA_ROOT))

    registry["series"].sort(key=lambda e: e["series_id"])
    registry["governance_version"] = "measured_dataset/v1+governance-2026-11-10"
    registry["n_series"] = len(registry["series"])
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"rewritten_series": changed,
                      "registry_series": [e["series_id"] for e in registry["series"]],
                      "registry_updated": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())