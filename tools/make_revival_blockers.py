#!/usr/bin/env python3
"""Build ``revival_blockers.json``: one declared blocker per honest FAIL.

Two provenances are possible and both are recorded verbatim:

``curated``
    a human/agent read the report and cited the section that states the
    obstruction (``basis`` field);
``rule``
    nobody declared one, so the blocker is derived from the record's
    attribution label by :data:`ATTRIBUTION_TO_BLOCKER`; the derivation is
    written into ``basis`` so a reader can audit -- and override -- it.

Undeclared FAILs stay visible with kind ``undeclared`` (never auto-resolvable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.fail_revival.blockers import DECLARATION_SCHEMA  # noqa: E402
from tensorphysics.fail_revival.extract import extract_directory  # noqa: E402
from tensorphysics.fail_revival.model import canonical_json  # noqa: E402

#: report section that states the obstruction, per real FAIL (mission:line)
CURATED = {
    "7813:180": {
        "blocker_id": "caliber:response-bandwidth",
        "kind": "numerical_caliber_change",
        "note": "5% exceedance band is only attainable for epsilon <= 0.05; realistic seas sit at "
                "0.09-0.27 because the platform heave RAO carries an excitation notch, so the "
                "criterion needs a bandwidth-aware caliber revision, not a code fix.",
        "basis": "curated:report 9211 summary + section 3.5 bandwidth sweep",
    },
    "7815:141": {
        "blocker_id": "capability:maneuvering-derivative-uncertainty",
        "kind": "capability_slug",
        "note": "turn diameter stays -61.2% off the anchor; sensitivity table shows the "
                "derivative/coefficient bandwidth reaches +-40.1%, so the missing capability is a "
                "derivative-source uncertainty band, not a tuning constant.",
        "basis": "curated:report 9178 sections 5-6 (sensitivity + reachable D/L band)",
    },
    "7815:143": {
        "blocker_id": "mission:6008",
        "kind": "upstream_mission",
        "note": "advance A_D/L is 73.6% off while the anchor set is not self-consistent; the "
                "upstream baseline implementation must be corrected before this criterion can be "
                "re-measured.",
        "basis": "curated:report 9178 error table note '(anchor not self-consistent)'",
    },
    "7815:144": {
        "blocker_id": "mission:6008",
        "kind": "upstream_mission",
        "note": "transfer A_T/L is -53.6% off under the same inconsistent anchor set.",
        "basis": "curated:report 9178 error table note '(anchor not self-consistent)'",
    },
    "7815:147": {
        "blocker_id": "capability:maneuvering-derivative-uncertainty",
        "kind": "capability_slug",
        "note": "uncertainty band +-40.1% exceeds the +-8% gate by 5x; blocked on the same missing "
                "derivative uncertainty capability.",
        "basis": "curated:report 9178 section 6 uncertainty band",
    },
    "7827:65": {
        "blocker_id": "caliber:fs-discretisation-sign",
        "kind": "numerical_caliber_change",
        "note": "truncated-domain free-surface solve does not converge: the 3-point backward "
                "difference has Im k > 0 and damps the physical k = k0 wave; needs an upwind/panel "
                "caliber revision.",
        "basis": "curated:report 9181 summary (discrete sign analysis) + section on f_fs_study",
    },
    "7827:68": {
        "blocker_id": "caliber:fs-discretisation-sign",
        "kind": "numerical_caliber_change",
        "note": "Fr sweep would report numerical noise until the free-surface discretisation sign "
                "convention is revised.",
        "basis": "curated:report 9181 summary",
    },
    "7827:70": {
        "blocker_id": "caliber:fs-discretisation-sign",
        "kind": "numerical_caliber_change",
        "note": "energy bookkeeping between pressure integration and far-field flux disagrees by "
                ">10% because the wave field is numerically damped.",
        "basis": "curated:report 9181 verdict table",
    },
    "7827:83": {
        "blocker_id": "caliber:fs-discretisation-sign",
        "kind": "numerical_caliber_change",
        "note": "free-surface R_w error 2803% at Fr = 0.3 follows from the same damped wave field.",
        "basis": "curated:report 9181 error table",
    },
}

ATTRIBUTION_TO_BLOCKER = {
    "numerical_caliber": ("numerical_caliber_change", "caliber:derived-{mission}"),
    "missing_reference": ("capability_slug", "capability:reference-anchor"),
    "capability_gap": ("capability_slug", "capability:derived-gap-{mission}"),
    "scope_limit": ("numerical_caliber_change", "caliber:scope-{mission}"),
    "model_form": ("capability_slug", "capability:derived-model-form-{mission}"),
    "unclassified": ("undeclared", "undeclared:{fail_id}"),
}


def build(corpus: Path) -> dict:
    extraction = extract_directory(corpus)
    grouped: dict[str, dict] = {}
    for rec in extraction.records:
        tag = "%d:%d" % (rec.mission_id, rec.line_no)
        if tag in CURATED:
            spec = dict(CURATED[tag])
            provenance = "curated"
            basis = spec.pop("basis")
        else:
            kind, template = ATTRIBUTION_TO_BLOCKER.get(
                rec.attribution, ("undeclared", "undeclared:{fail_id}")
            )
            spec = {
                "blocker_id": template.format(mission=rec.mission_id, fail_id=rec.fail_id),
                "kind": kind,
                "note": "rule-derived from attribution=%s (basis=%s)"
                        % (rec.attribution, rec.attribution_basis or "none"),
            }
            provenance = "rule"
            basis = "rule:attribution=%s;%s" % (rec.attribution, rec.attribution_basis or "none")
        entry = grouped.setdefault(
            spec["blocker_id"],
            {
                "blocker_id": spec["blocker_id"],
                "kind": spec["kind"],
                "fail_keys": [],
                "note": spec["note"],
                "provenance": {},
            },
        )
        entry["fail_keys"].append(rec.fail_id)
        entry["provenance"][rec.fail_id] = {"mode": provenance, "basis": basis,
                                            "mission_id": rec.mission_id, "line_no": rec.line_no}
    blockers = sorted(grouped.values(), key=lambda e: e["blocker_id"])
    return {
        "schema": DECLARATION_SCHEMA,
        "note": "One declared blocker per honest FAIL. Curated declarations cite the report section; "
                "rule-derived ones record the attribution they came from. Data only -- no code "
                "semantics are attached to these names.",
        "corpus": str(corpus),
        "record_count": len(extraction.records),
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=Path(__file__).resolve().parents[1] / "tensor_reports" / "campaign_nodes")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or args.corpus / "revival_blockers.json"
    payload = build(args.corpus)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(canonical_json({"out": str(out), "records": payload["record_count"],
                          "blockers": payload["blocker_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())