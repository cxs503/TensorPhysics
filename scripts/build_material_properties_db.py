#!/usr/bin/env python3
"""One-shot reproducer for the material/fluid property database (MPDB v1).

    python scripts/build_material_properties_db.py            # rebuild + write artefacts
    python scripts/build_material_properties_db.py --check     # verify committed artefacts
    python scripts/build_material_properties_db.py --validate  # schema + anchor audit only

Writes (src/tensorphysics/material_properties/data/):
    mpdb_v1.jsonl, mpdb_v1.substances.json, mpdb_v1.anchors.json,
    mpdb_v1.meta.json, mpdb_v1.schema.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorphysics.material_properties import build, check_committed  # noqa: E402
from tensorphysics.material_properties.build import anchor_audit, build_records  # noqa: E402
from tensorphysics.material_properties.sources import SOURCE_REGISTRY  # noqa: E402
from tensorphysics.material_properties.schema import validate_records  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        res = check_committed()
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if "--validate" in argv:
        recs = build_records()
        errs = validate_records(recs, SOURCE_REGISTRY)
        anchors = anchor_audit(recs)
        failed = [a for a in anchors if not a["passed"]]
        print(json.dumps({
            "rows": len(recs),
            "schema_errors": errs,
            "anchors_total": len(anchors),
            "anchors_passed": len(anchors) - len(failed),
            "anchors_failed": [a["id"] for a in failed],
            "max_rel_error": max((a["rel_error"] or 0.0) for a in anchors),
        }, indent=2))
        return 0 if not errs and not failed else 1
    out = build()
    meta = out["meta"]
    print(json.dumps({k: meta[k] for k in (
        "db", "schema_version", "rows", "substances", "substance_classes", "sources",
        "rows_with_provenance", "rows_formula_primary", "rows_table_primary", "rows_derived",
        "anchors_total", "anchors_passed", "anchor_max_rel_error", "jsonl_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
