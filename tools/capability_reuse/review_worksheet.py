"""Emit candidate edges (with file:line) for manual verification and a 20-edge sample.

The output is a *worksheet*: the reviewer opens each cited line in the pinned
checkout, marks the verdict, and the reviewed file becomes
``artifacts/capability_reuse/{gold_standard,edge_sample_review}.json``.  Nothing
here decides a verdict.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.capability_reuse.audit import sample_edges  # noqa: E402


def load_edges(path: Path):
    payload = json.loads(path.read_text())
    return payload["edges"], payload.get("snapshot_hash", "")


def main() -> int:
    edges, snap_hash = load_edges(ROOT / "artifacts/capability_reuse/edges.json")
    by_kind = defaultdict(list)
    for e in edges:
        by_kind[e["kind"]].append(e)

    print(f"# edges={len(edges)} snapshot={snap_hash[:12]}")
    for k, rows in sorted(by_kind.items()):
        print(f"  {k}: {len(rows)}")

    # ---- sample of 20 edges for the false-positive audit (deterministic) ----
    pairs = sorted({(e["src"], e["dst"]) for e in edges})
    sample = sample_edges(pairs, n=20, salt=snap_hash)
    idx = defaultdict(list)
    for e in edges:
        idx[(e["src"], e["dst"])].append(e)
    out = {"schema": "capability-reuse-edge-review/1", "sample_size": len(sample),
           "review_file": "artifacts/capability_reuse/edge_sample_review.json",
           "unsupported": 0, "rows": []}
    print("\n## SAMPLE 20 (verdicts to be filled by hand)")
    for s, d in sample:
        evs = sorted(idx[(s, d)], key=lambda e: (e["file"], e["line"]))
        row = {"src": s, "dst": d, "verdict": "unreviewed",
               "evidence": [f"{e['file']}:{e['line']} [{e['kind']}] {e['detail']}" for e in evs[:2]]}
        out["rows"].append(row)
        print(f"- {s} -> {d}\n    {row['evidence']}")
    (ROOT / "artifacts/capability_reuse/edge_sample_worksheet.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())