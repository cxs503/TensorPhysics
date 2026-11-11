"""Seeding hints: what the arbitrator should seed next, with explicit reasons."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from .model import HINT_SCHEMA

RECOMMENDATIONS = {
    "no_test": "add_test",
    "no_doc": "add_doc",
    "no_code": "add_build",
    "reuse_candidate": "advertise_reuse",
}


def _classify(pathmap_row: Dict[str, List]) -> str:
    if not pathmap_row.get("code") and not pathmap_row.get("header"):
        return "no_code"
    tests = [p for p in pathmap_row.get("code", []) if "/tests/" in p or "test_" in p]
    if not tests:
        return "no_test"
    if not pathmap_row.get("artifact"):
        return "no_doc"
    return "reuse_candidate"


def build_hints(stats: Dict[str, object], top_k: int = 25) -> Dict[str, object]:
    indeg: Dict[str, int] = stats["indegree"]                       # type: ignore[assignment]
    outdeg: Dict[str, int] = stats["outdegree"]                     # type: ignore[assignment]
    ranks: Dict[str, float] = stats["pagerank"]                     # type: ignore[assignment]
    pathmap: Dict[str, Dict[str, List]] = stats["pathmap"]           # type: ignore[assignment]
    per_repo = stats["per_repo"]
    test_reused = set(stats.get("test_reused") or [])
    cap_repo: Dict[str, str] = stats.get("cap_repo") or {}      # type: ignore[assignment]

    def repo_of(cid: str) -> str:
        return cap_repo.get(cid, "")

    undetermined = set(stats.get("undetermined") or [])
    determined = set(indeg) - undetermined
    orphans = sorted((cid for cid in determined if indeg.get(cid, 0) == 0),
                     key=lambda c: (-outdeg.get(c, 0), c))
    orphan_rows = []
    for cid in orphans[:top_k]:
        row = pathmap.get(cid, {})
        kind = _classify(row)
        orphan_rows.append({
            "capability": cid,
            "repo": repo_of(cid),
            "recommendation": RECOMMENDATIONS[kind],
            "reason": (f"in-degree 0 over {int(stats['counts']['edges'])} graph edges; "
                       f"out-degree {outdeg.get(cid, 0)}; "
                       f"code files {len(row.get('code', []))}, artefacts {len(row.get('artifact', []))}"),
            "evidence": (row.get("code") or row.get("artifact") or [])[:3],
        })

    hubs = []
    for cid in sorted(indeg, key=lambda c: (-indeg.get(c, 0), -ranks.get(c, 0.0), c))[:top_k]:
        if indeg.get(cid, 0) == 0:
            continue
        hubs.append({
            "capability": cid,
            "repo": repo_of(cid),
            "in_degree": indeg.get(cid, 0),
            "pagerank": ranks.get(cid, 0.0),
            "test_supported": cid in test_reused,
            "recommendation": "add_redundant_verification",
            "reason": (f"in-degree {indeg.get(cid, 0)} (top-{top_k} hub), "
                       f"pagerank {ranks.get(cid, 0.0):.4f}; a defect here propagates to "
                       f"{indeg.get(cid, 0)} downstream capabilities; "
                       f"test-file support: {'yes' if cid in test_reused else 'no'}"),
        })

    order = sorted(ranks, key=lambda c: (-ranks.get(c, 0.0), c))
    q = max(1, len(order) // 4)
    gaps = []
    for cid in order[:q]:
        if ranks.get(cid, 0.0) <= 0:
            continue
        if cid in test_reused or indeg.get(cid, 0) == 0:
            continue
        row = pathmap.get(cid, {})
        gaps.append({
            "capability": cid,
            "repo": repo_of(cid),
            "pagerank": ranks.get(cid, 0.0),
            "in_degree": indeg.get(cid, 0),
            "recommendation": "add_verification_edge",
            "reason": (f"pagerank top quartile ({ranks.get(cid, 0.0):.4f}) with in-degree "
                       f"{indeg.get(cid, 0)} but no reference from any test file"),
            "evidence": (row.get("code") or [])[:2],
        })

    return {
        "schema": HINT_SCHEMA,
        "summary": {
            "orphans": len(orphans),
            "orphans_listed": len(orphan_rows),
            "hubs": len(hubs),
            "gaps": len(gaps),
            "reuse_rate": stats["reuse"]["reuse_rate"],
        },
        "orphans": orphan_rows,
        "hubs": hubs,
        "gaps": gaps,
    }