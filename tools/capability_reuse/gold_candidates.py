"""Emit N extra candidate edges (different salt) with raw context, for gold labelling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.capability_reuse.audit import sample_edges  # noqa: E402
from tensorphysics.capability_reuse.indexer import load_snapshot  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    salt = sys.argv[2] if len(sys.argv) > 2 else "gold"
    edges = json.loads((ROOT / "artifacts/capability_reuse/edges.json").read_text())["edges"]
    snap = json.loads((ROOT / "artifacts/capability_reuse/snapshot.json").read_text())
    review = json.loads((ROOT / "artifacts/capability_reuse/edge_sample_worksheet.json").read_text())
    used = {(r["src"], r["dst"]) for r in review["rows"]}
    by = {}
    for e in edges:
        if (e["src"], e["dst"]) in used:
            continue
        by.setdefault((e["src"], e["dst"]), []).append(e)
    pairs = sorted(by)
    out = []
    for s, d in sample_edges(pairs, n=n, salt=salt):
        evs = sorted(by[(s, d)], key=lambda e: (e["file"], e["line"]))
        repo = next((c["repo"] for c in snap["capabilities"] if c["cid"] == d), "")
        out.append({"src": s, "dst": d, "dst_repo": repo,
                    "evidence": [f"{e['file']}:{e['line']} [{e['kind']}] {e['detail']}" for e in evs[:2]]})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())