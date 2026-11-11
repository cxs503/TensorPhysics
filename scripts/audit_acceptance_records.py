"""Inventory the mission acceptance records committed under ``docs/``.

Why this exists
---------------
A mission's acceptance record (``docs/<mission>/outputs/acceptance_criteria.json``)
is a *claim*: "these were this node's criteria, these were its measurements,
they passed".  For mission 8598 that claim was, for a while, false in the
cheapest possible way -- the committed file was a byte-for-byte copy of another
mission's record (shaft-torsion metrics plus a test flag this node never runs).
Nothing in the file itself said so; it took a full re-derivation to notice.

Records emitted by ``scripts/run_criterion_prereg_eval.py`` close that hole from
the other side: they are sealed with ``payload_sha256`` over their own canonical
body, so a stale, foreign or hand-edited copy refutes itself **offline** -- no
artefact cache, no corpora re-run.  This script is the census of that property:

* every record found under ``docs/**/acceptance_criteria.json`` is reported with
  ``sealed`` (does the seal verify?), ``mission_id`` and a body digest;
* records whose body digest equals another record's are flagged as
  ``copies`` -- a shared body across two missions is either a copy or a shared
  generator, and both deserve a look;
* ``--fail-on-unsealed`` turns the census into a gate (exit 1) for trees that
  are supposed to be fully sealed; the default is a report (exit 0), because
  older missions' records predate the seal and are fixed by re-emitting them,
  not by hiding them.

Design notes
------------
The digest is *not* reimplemented here: it is imported from
``run_criterion_prereg_eval`` (``deliverable_digest`` / ``verify_deliverable``),
so the seal has exactly one definition and the census cannot drift from the
emitter.  The scan itself touches only JSON files; it never imports numpy,
matplotlib or the solver packages *to decide* anything.

Usage::

    python scripts/audit_acceptance_records.py                    # census
    python scripts/audit_acceptance_records.py --json-out census.json
    python scripts/audit_acceptance_records.py --fail-on-unsealed  # gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

DIGEST_KEY = "payload_sha256"  # noqa: F841  (documented next to the imports below)


def _digest_helpers():
    """The emitter's seal functions (single definition of the digest)."""
    import run_criterion_prereg_eval as ev  # noqa: PLC0415  (heavy import, on demand)

    return ev.deliverable_digest, ev.verify_deliverable


def body_digest(record: dict) -> str:
    """Canonical digest of the record body, independent of the seal helper."""
    return hashlib.sha256(
        json.dumps({k: v for k, v in record.items() if k != DIGEST_KEY},
                   sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify(record: dict) -> dict:
    """Describe one record: is its seal valid, and if not, why not."""
    _, verify = _digest_helpers()
    ok, reason = verify(record)
    return {
        "sealed": bool(ok),
        "reason": reason,
        "mission_id": record.get("mission_id"),
        "node_id": record.get("node_id"),
        "schema": record.get("schema"),
        "all_acceptance_pass": record.get("all_acceptance_pass"),
        "n_acceptance_blocks": len(record.get("acceptance") or {}) if isinstance(record.get("acceptance"), dict) else 0,
        "body_sha256": body_digest(record),
    }


def _blank(reason: str) -> dict:
    """A row for a file that carries no usable record."""
    return {"sealed": False, "reason": reason, "mission_id": None, "node_id": None,
            "schema": None, "all_acceptance_pass": None, "n_acceptance_blocks": 0,
            "body_sha256": None}


def scan(root: Path) -> list[dict]:
    """Every ``**/outputs/acceptance_criteria.json`` (and the ``docs/`` digests)."""
    root = Path(root)
    paths = sorted(set(root.glob("docs/**/outputs/acceptance_criteria.json"))
                   | set(root.glob("docs/**/acceptance_criteria.json")))
    rows: list[dict] = []
    for path in paths:
        rel = str(path.relative_to(root))
        reason = None
        try:
            record = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, ValueError) as exc:  # unreadable or not JSON
            record, reason = None, f"unreadable: {exc.__class__.__name__}"
        if isinstance(record, dict):
            rows.append({"path": rel, **classify(record)})
        else:
            rows.append({"path": rel, **_blank(reason or "not a JSON object")})
    return rows


def group_copies(rows: list[dict]) -> list[dict]:
    """Records sharing a body digest, i.e. the same file under several names."""
    seen: dict[str, list[str]] = {}
    for row in rows:
        if row.get("body_sha256"):
            seen.setdefault(row["body_sha256"], []).append(row["path"])
    return [{"body_sha256": digest, "paths": paths}
            for digest, paths in sorted(seen.items()) if len(paths) > 1]


def summarise(root: Path) -> dict:
    rows = scan(root)
    copies = group_copies(rows)
    copied_paths = {p for group in copies for p in group["paths"]}
    return {
        "root": str(root),
        "n_records": len(rows),
        "n_sealed": sum(1 for r in rows if r["sealed"]),
        "n_unsealed": sum(1 for r in rows if not r["sealed"]),
        "n_duplicated_bodies": len(copied_paths),
        "all_sealed": all(r["sealed"] for r in rows) if rows else True,
        "records": rows,
        "copies": copies,
        "verdict": "ok" if rows and all(r["sealed"] for r in rows) else
                   ("no records" if not rows else "unsealed records present"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT), help="tree to scan (default: repo root)")
    ap.add_argument("--json-out", default=None, help="write the census here")
    ap.add_argument("--md-out", default=None, help="write a markdown table here")
    ap.add_argument("--fail-on-unsealed", action="store_true",
                    help="exit 1 unless every record's seal verifies")
    ap.add_argument("--quiet", action="store_true", help="print nothing on stdout")
    args = ap.parse_args(argv)

    report = summarise(Path(args.root))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    if args.md_out:
        lines = ["| record | mission | sealed | reason |", "|---|---|---|---|"]
        for row in report["records"]:
            lines.append(f"| `{row['path']}` | {row['mission_id']} | "
                         f"{'yes' if row['sealed'] else '**no**'} | {row['reason']} |")
        Path(args.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps({k: v for k, v in report.items() if k not in ("records", "copies")},
                         ensure_ascii=False, indent=2))
        for row in report["records"]:
            if not row["sealed"]:
                print(f"unsealed: {row['path']} ({row['reason']})")
    if args.fail_on_unsealed and not report["all_sealed"]:
        print("acceptance-record audit: FAIL (unsealed records present)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())