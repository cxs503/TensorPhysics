"""Graph construction, dual-caliber cross-validation and reuse metrics."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .extract import ExtractionResult, evidence_edges, map_capabilities
from .indexer import RepoIndex
from .model import (CAP_EDGE_SCHEMA, Capability, Edge, canonical_hash,
                    cluster_capabilities, dedupe_edges, normalize_capability_name)

TEST_MARKERS = ("/tests/", "/test/", "test_", "_test.py", "/testing/")


def _is_test_path(path: str) -> bool:
    p = "/" + path.replace("::", "/")
    return any(m in p for m in TEST_MARKERS)


@dataclass
class Graph:
    capabilities: Dict[str, Capability]
    determined: Set[str]
    undetermined: Set[str]
    edges: List[Edge]
    stats: Dict[str, object] = field(default_factory=dict)
    cluster: Dict[str, str] = field(default_factory=dict)

    def edge_pairs(self) -> Set[Tuple[str, str]]:
        return {(e.src, e.dst) for e in self.edges}


def capability_graph(caps: Sequence[Capability], edges: Iterable[Edge],
                     pathmap: Dict[str, Dict[str, List]]) -> Graph:
    by_id = {c.cid: c for c in caps}
    determined = {c.cid for c in caps if pathmap.get(c.cid, {}).get("code") or pathmap.get(c.cid, {}).get("header")}
    undetermined = set(by_id) - determined
    cluster = cluster_capabilities({c.cid: (normalize_capability_name(c.name), c.repo) for c in caps})
    keep = []
    for e in edges:
        if e.src not in by_id or e.dst not in by_id or e.src == e.dst:
            continue
        if cluster.get(e.src) == cluster.get(e.dst):
            continue          # same capability delivered twice: not a reuse edge
        keep.append(e)
    keep = dedupe_edges(keep)
    return Graph(capabilities=by_id, determined=determined, undetermined=undetermined,
                 edges=keep, cluster=cluster)


def degrees(g: Graph) -> Tuple[Counter, Counter, Dict[str, Set[str]]]:
    indeg: Counter = Counter()
    outdeg: Counter = Counter()
    srcs: Dict[str, Set[str]] = defaultdict(set)
    for e in g.edges:
        indeg[e.dst] += 1
        outdeg[e.src] += 1
        srcs[e.dst].add(e.src)
    return indeg, outdeg, srcs


def pagerank(g: Graph, damping: float = 0.85, iters: int = 60) -> Dict[str, float]:
    nodes = sorted(g.capabilities)
    n = len(nodes)
    if n == 0:
        return {}
    out: Dict[str, List[str]] = defaultdict(list)
    for e in g.edges:
        out[e.src].append(e.dst)
    rank = {v: 1.0 / n for v in nodes}
    for _ in range(iters):
        nxt = {v: (1.0 - damping) / n for v in nodes}
        dangling = sum(rank[v] for v in nodes if not out[v])
        for v in nodes:
            if not out[v]:
                continue
            share = damping * rank[v] / len(out[v])
            for w in out[v]:
                nxt[w] += share
        for v in nodes:
            nxt[v] += damping * dangling / n
        rank = nxt
    return rank


def components(g: Graph) -> List[List[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in g.edges:
        adj[e.src].add(e.dst)
        adj[e.dst].add(e.src)
    seen: Set[str] = set()
    comps: List[List[str]] = []
    for v in sorted(g.capabilities):
        if v in seen:
            continue
        q = deque([v])
        seen.add(v)
        comp: List[str] = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c))
    return comps


def indegree_distribution(indeg: Counter, universe: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """In-degree histogram over the *determined* capability set (zeros included)."""
    ids = list(universe) if universe is not None else list(indeg)
    dist: Counter = Counter(indeg.get(c, 0) for c in ids)
    top = int(max(dist)) if dist else 0
    out = {str(k): dist.get(k, 0) for k in range(0, top + 1)}
    out[">max"] = 0
    return out


def _jaccard(a: Set[Tuple[str, str]], b: Set[Tuple[str, str]]) -> Dict[str, object]:
    inter, union = a & b, a | b
    return {
        "ast_edges": len(a), "ref_edges": len(b), "intersection": len(inter),
        "union": len(union), "jaccard": round(len(inter) / len(union), 6) if union else 1.0,
        "agreement_ratio": round(len(inter) / len(a), 6) if a else 1.0,
        "coverage_ref_of_ast": round(len(inter) / len(b), 6) if b else 1.0,
        "ast_only": [list(x) for x in sorted(a - b)],
        "ref_only": [list(x) for x in sorted(b - a)],
    }


def dual_caliber_compare(ast_edges: Sequence[Edge], ref_edges: Sequence[Edge]) -> Dict[str, object]:
    """Cross-validate the two mechanisms that read the same files.

    Two comparisons are reported and never merged:

    * ``capability_level`` -- AST vs lexical **import surface** (both read the
      same relation: "module B imports module A"); this is the criterion-(b)
      number.
    * ``usage_level`` -- AST vs the **symbol/usage** reading, which by
      construction sees references that are not dependencies (``usage ⊃
      dependency``); reported informationally, expected to be far lower.
    """
    a = {(e.src, e.dst) for e in ast_edges}
    imp = {(e.src, e.dst) for e in ref_edges if e.kind != "ref_symbol"}
    all_ref = {(e.src, e.dst) for e in ref_edges}
    return {
        "capability_level": _jaccard(a, imp),
        "usage_level": _jaccard(a, all_ref),
    }


def build_graph(snapshot: dict, repos: Dict[str, RepoIndex],
                extraction: ExtractionResult,
                strength: Optional[Dict[str, List[str]]] = None,
                params: Optional[Dict[str, object]] = None) -> Tuple[Graph, Dict[str, object]]:
    """Full pipeline: map capabilities, project both calibers, add evidence edges."""
    strength = strength or {"strong": ["ast_import", "c_include", "ref_module", "artifact_ref",
                                       "dependency_hint"],
                            "knowledge": ["report_ref"], "usage": ["ref_symbol"]}
    caps, ownership, pathmap = map_capabilities(snapshot, repos, params)
    code = extract_code_edges_wrapper(snapshot, repos, ownership, params)
    ev = evidence_edges(snapshot, repos, ownership, pathmap)
    all_edges = code.ast_edges + code.ref_edges + ev
    strong_kinds = set(strength.get("strong") or ())
    headline = [e for e in all_edges if e.kind in strong_kinds]
    graph = capability_graph(caps, headline, pathmap)

    indeg, outdeg, srcs = degrees(graph)
    ranks = pagerank(graph)
    comps = components(graph)
    by_repo = {c.cid: c.repo for c in caps}
    cross = [e for e in graph.edges if by_repo.get(e.src) != by_repo.get(e.dst)]
    per_repo: Dict[str, Dict[str, object]] = {}
    for cid, c in sorted(((c.cid, c) for c in caps), key=lambda kv: kv[0]):
        r = per_repo.setdefault(c.repo, {"capabilities": 0, "determined": 0, "reused": 0,
                                         "out_edges": 0, "in_edges": 0, "cross_out": 0, "cross_in": 0})
        r["capabilities"] += 1
        if cid in graph.determined:
            r["determined"] += 1
        if indeg.get(cid):
            r["reused"] += 1
    for e in graph.edges:
        rs, rd = by_repo.get(e.src), by_repo.get(e.dst)
        if rs:
            per_repo.setdefault(rs, {})["out_edges"] = per_repo.get(rs, {}).get("out_edges", 0) + 1
            if rs != rd:
                per_repo[rs]["cross_out"] = per_repo[rs].get("cross_out", 0) + 1
        if rd:
            per_repo.setdefault(rd, {})["in_edges"] = per_repo.get(rd, {}).get("in_edges", 0) + 1
            if rs != rd:
                per_repo[rd]["cross_in"] = per_repo[rd].get("cross_in", 0) + 1

    kinds = Counter(e.kind for e in all_edges)
    dupe_dropped = {e for e in all_edges
                    if graph.cluster.get(e.src) == graph.cluster.get(e.dst)}
    # sensitivity: what each evidence tier would add on its own
    def _tier_rate(kinds_sel) -> Dict[str, object]:
        g = capability_graph(caps, [e for e in all_edges if e.kind in kinds_sel], pathmap)
        ind, _o, _s = degrees(g)
        reused_n = sum(1 for c in g.determined if ind.get(c))
        return {"edges": len(g.edges), "reused": reused_n,
                "reuse_rate": round(reused_n / len(g.determined), 6) if g.determined else 0.0}

    tiers = {name: _tier_rate(set(kinds_sel)) for name, kinds_sel in sorted(strength.items())
             if isinstance(kinds_sel, list)}
    tiers["headline_all"] = _tier_rate(set(strong_kinds))
    tiers["headline_plus_medium"] = _tier_rate(set(strong_kinds) | set(strength.get("medium") or ()))
    tiers["headline_plus_usage"] = _tier_rate(set(strong_kinds) | set(strength.get("usage") or ()))
    tiers["headline_plus_all"] = _tier_rate(set(strong_kinds) | set(strength.get("medium") or ())
                                            | set(strength.get("usage") or ())
                                            | set(strength.get("knowledge") or ()))
    reused = {cid for cid in graph.determined if indeg.get(cid)}
    test_reuse = {e.dst for e in code.ast_edges + code.ref_edges if _is_test_path(e.file)}
    clusters = Counter(graph.cluster.values())
    stats = {
        "schema": CAP_EDGE_SCHEMA,
        "duplicate_capabilities": {
            "clusters": len(clusters),
            "capabilities_in_multi_member_clusters": sum(n for n in clusters.values() if n > 1),
            "largest_cluster": max(clusters.values()) if clusters else 0,
            "intra_cluster_edges_dropped": int(len(dupe_dropped)),
        },
        "counts": {
            "capabilities": len(caps),
            "determined": len(graph.determined),
            "undetermined": len(graph.undetermined),
            "edges": len(graph.edges),
            "distinct_pairs": len(graph.edge_pairs()),
            "cross_repo_edges": len(cross),
            "cross_repo_share": round(len(cross) / len(graph.edges), 6) if graph.edges else 0.0,
            "components": len(comps),
            "largest_component": len(comps[0]) if comps else 0,
        },
        "calibers": dual_caliber_compare(code.ast_edges, code.ref_edges),
        "extraction": dict(code.stats),
        "kinds": dict(sorted(kinds.items())),
        "edge_strength": strength,
        "tiers": tiers,
        "headline_edge_kinds": sorted(strong_kinds),
        "reuse": {
            "reuse_rate": round(len(reused) / len(graph.determined), 6) if graph.determined else 0.0,
            "reused_capabilities": len(reused),
            "orphan_capabilities": len(graph.determined - reused),
            "undetermined_excluded": len(graph.undetermined),
            "verified_by_test": len(test_reuse),
            "reuse_rate_including_undetermined_denominator": round(
                len(reused) / len(caps), 6) if caps else 0.0,
        },
        "indegree_distribution": indegree_distribution(indeg, graph.determined),
        "per_repo": {k: dict(sorted(v.items())) for k, v in sorted(per_repo.items())},
        "pagerank": {k: round(v, 9) for k, v in sorted(ranks.items(), key=lambda kv: (-kv[1], kv[0]))},
        "cap_repo": {c.cid: c.repo for c in caps},
        "cluster": dict(sorted(graph.cluster.items())),
        "cap_class": {c.cid: {"cap_class": c.cap_class or "unclassified", "repo": c.repo,
                              "name": c.name} for c in caps},
        "undetermined": sorted(graph.undetermined),
        "indegree": {k: indeg.get(k, 0) for k in sorted(graph.capabilities)},
        "outdegree": {k: outdeg.get(k, 0) for k in sorted(graph.capabilities)},
        "components": comps[:200],
        "pathmap": pathmap,
        "ownership": {f"{r}::{p}": c for (r, p), c in sorted(ownership.items())},
        "test_reused": sorted(test_reuse),
    }
    return graph, stats


def extract_code_edges_wrapper(snapshot: dict, repos: Dict[str, RepoIndex], ownership,
                               params: Optional[Dict[str, object]] = None) -> ExtractionResult:
    from .extract import extract_code_edges
    return extract_code_edges(snapshot, repos, ownership, params=params)