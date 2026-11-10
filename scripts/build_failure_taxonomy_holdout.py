"""Build the document-disjoint held-out FAIL corpus (v2) from a curation sheet.

The sheet (:mod:`tensorphysics.failure_taxonomy.data.heldout_curation.json`) holds
one row per record: a source document, a marker substring occurring exactly once
in it, and the analyst-assigned class.  This builder

1. rejects any source document that the frozen v1 corpus already uses
   (document-disjointness is the whole point of a held-out set);
2. extracts the *verbatim* line containing the marker (windowed to
   ``MAX_BODY`` characters, explicitly flagged when windowed);
3. rejects near-duplicate bodies (any 40-character shard shared with a frozen
   body would make the held-out evaluation optimistic);
4. writes ``heldout_fail_set.jsonl`` with per-record provenance
   (``source_path``, ``source_line``, ``source_sha256``, ``segment_sha256``,
   ``windowed``) and ``heldout_manifest.json`` with the pinned payload digest,
   per-class counts and the disjointness statement.

Run from the repository root::

    PYTHONPATH=src python scripts/build_failure_taxonomy_holdout.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.failure_taxonomy import (  # noqa: E402
    CLASS_CODES,
    load_domain_lexicon,
    load_frozen_set,
)
from tensorphysics.failure_taxonomy.dataset import DATA_DIR, digest_of  # noqa: E402

MAX_BODY = 300
SHARD = 40
SHEET = DATA_DIR / "heldout_curation.json"
OUT = DATA_DIR / "heldout_fail_set.jsonl"
MANIFEST = DATA_DIR / "heldout_manifest.json"
ANNOTATOR = "agent:m9812"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _window(line: str, marker: str) -> tuple[str, bool]:
    """Return the segment to store, and whether it had to be windowed."""
    if len(line) <= MAX_BODY:
        return line, False
    start = max(0, line.find(marker) - MAX_BODY // 3)
    start = min(start, max(0, len(line) - MAX_BODY))
    seg = line[start : start + MAX_BODY]
    return ("…" + seg + "…") if len(line) > MAX_BODY else seg, True


def _shards(text: str) -> set[str]:
    return {text[i : i + SHARD] for i in range(max(1, len(text) - SHARD + 1))}


def main() -> int:
    sheet = json.loads(SHEET.read_text(encoding="utf-8"))
    rows = sheet["rows"]
    frozen = load_frozen_set()
    frozen_docs = {r.provenance["source_path"] for r in frozen}
    frozen_shards: set[str] = set()
    for rec in frozen:
        frozen_shards |= _shards(rec.body)

    forbidden = load_domain_lexicon()
    terms = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) if t.isascii() else re.compile(re.escape(t), re.IGNORECASE) for t in forbidden]

    out: list[dict] = []
    docs: dict[str, int] = {}
    seen_markers: set[tuple[str, str]] = set()
    problems: list[str] = []

    for row in rows:
        rid, rel, marker, cls = row["record_id"], row["source_path"], row["marker"], row["gold_class"]
        if cls not in CLASS_CODES:
            problems.append(f"{rid}: unknown class {cls!r}")
            continue
        if rel in frozen_docs:
            problems.append(f"{rid}: source document already used by the frozen v1 corpus")
            continue
        path = Path(rel)
        if not path.is_file():
            problems.append(f"{rid}: missing source document {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [i for i, line in enumerate(text.splitlines(), start=1) if marker in line]
        if len(hits) != 1:
            problems.append(f"{rid}: marker occurs on {len(hits)} lines (need exactly 1) in {rel}")
            continue
        if (rel, marker) in seen_markers:
            problems.append(f"{rid}: duplicate (document, marker) pair")
            continue
        seen_markers.add((rel, marker))
        lineno = hits[0]
        line = text.splitlines()[lineno - 1].strip()
        body, windowed = _window(line, marker)
        if _shards(body) & frozen_shards:
            problems.append(f"{rid}: body shares a {SHARD}-character shard with the frozen corpus")
            continue
        leaked = [t.pattern for t in terms if t.search(body)]
        out.append(
            {
                "record_id": rid,
                "body": body,
                "gold_class": cls,
                "annotator": ANNOTATOR,
                "provenance": {
                    "source_path": rel,
                    "source_line": lineno,
                    "source_sha256": _sha(text),
                    "segment_sha256": _sha(body),
                    "windowed": windowed,
                    "sheet_marker": marker,
                    "cue_leak_note": "marker is a document-specific string, not a classifier cue lexeme"
                    + ("" if not leaked else f"; NOTE body contains external-vocabulary terms: {leaked}"),
                },
            }
        )
        docs[rel] = docs.get(rel, 0) + 1

    if problems:
        for p in problems:
            print("ERROR", p, file=sys.stderr)
        return 2

    unknown = sorted({r["gold_class"] for r in out} ^ set(CLASS_CODES))
    if unknown:
        print(f"ERROR class coverage mismatch: {unknown}", file=sys.stderr)
        return 2

    per_class = {code: sum(1 for r in out if r["gold_class"] == code) for code in CLASS_CODES}
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n", encoding="utf-8")

    manifest = {
        "n_records": len(out),
        "sha256": digest_of(out),
        "per_class": per_class,
        "n_documents": len(docs),
        "max_records_per_document": max(docs.values()),
        "records_per_document": dict(sorted(docs.items())),
        "annotator": ANNOTATOR,
        "protocol": sheet["protocol"],
        "document_disjoint_from_frozen": True,
        "frozen_documents": len(frozen_docs),
        "windowed_records": [r["record_id"] for r in out if r["provenance"]["windowed"]],
        "source_scope": ["docs/**/*.md"],
        "builder": "scripts/build_failure_taxonomy_holdout.py",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_records", "sha256", "per_class", "n_documents")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())