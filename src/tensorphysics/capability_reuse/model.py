"""Graph primitives: canonical serialisation, capabilities, evidence edges."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CAP_EDGE_SCHEMA = "capability-reuse-edges/1"
SNAPSHOT_SCHEMA = "capability-reuse-snapshot/1"
HINT_SCHEMA = "capability-reuse-seeding-hints/1"
AUDIT_SCHEMA = "capability-reuse-audit/1"

#: calibers of code evidence (both target the same relation: "module B depends
#: on module A"); ``ref`` is the mechanical lexical reading of the same files.
CALIBERS = ("ast", "ref")


def canonical_json(obj: Any) -> str:
    """Stable serialisation: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


@dataclass(frozen=True)
class Capability:
    """One delivered capability (a ``done`` campaign node or feature-ledger row)."""

    cid: str
    name: str
    repo: str
    domain: Optional[str] = None
    cap_class: Optional[str] = None
    mission_id: Optional[int] = None
    status: str = "done"
    origin: str = "node"  # node | feature_ledger

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "name": self.name,
            "repo": self.repo,
            "domain": self.domain,
            "cap_class": self.cap_class,
            "mission_id": self.mission_id,
            "status": self.status,
            "origin": self.origin,
        }


_CODE_PREFIX_RE = __import__("re").compile(r"(?i)^\s*(?:[a-z]{1,4}\d{2,5}[-_][a-z]?\d{1,3}|in\d{1,3}[-_][a-z]\d{1,3}|p\d+[-_][a-z]\d+)\s*[:：\-]?\s*")


def normalize_capability_name(name: str) -> str:
    """Normalise a capability title for duplicate detection (no domain knowledge).

    Strips leading node codes (``P190-N01:``, ``IN27-S6`` ...), brick/maturity
    prefixes, ASCII parentheticals, punctuation and whitespace, so that the same
    capability delivered twice collapses to the same key.
    """
    import re as _re
    s = name.strip()
    s = _CODE_PREFIX_RE.sub("", s)
    s = _re.sub(r"^\s*(?:成熟化|maturity|成熟化[:：])\s*", "", s, flags=_re.I)
    s = _re.sub(r"[(（][^)）]*[A-Za-z][^)）]*[)）]", "", s)
    s = _re.sub(r"[\s:：,，.。;；!！?？、\-—_/|（）()\[\]【】'\"“”]+", "", s)
    return s.lower()


def name_trigrams(text: str) -> set:
    if len(text) < 3:
        return {text}
    return {text[i:i + 3] for i in range(len(text) - 2)}


def cluster_capabilities(names: Dict[str, Tuple[str, str]], threshold: float = 0.9) -> Dict[str, str]:
    """Union capabilities that are the same deliverable (name equality or trigram sim)."""
    parent: Dict[str, str] = {cid: cid for cid in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_norm: Dict[Tuple[str, str], List[str]] = {}
    for cid, (norm, repo) in names.items():
        by_norm.setdefault((repo, norm), []).append(cid)
    for ids in by_norm.values():
        for other in ids[1:]:
            union(ids[0], other)
    # near-duplicates inside a repository: trigram Jaccard
    by_repo: Dict[str, List[Tuple[str, set]]] = {}
    for cid, (norm, repo) in names.items():
        by_repo.setdefault(repo, []).append((cid, name_trigrams(norm)))
    for repo, rows in by_repo.items():
        rows.sort()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, ta = rows[i]
                b, tb = rows[j]
                if find(a) == find(b) or not ta or not tb:
                    continue
                inter = len(ta & tb)
                union_j = inter / len(ta | tb)
                if union_j >= threshold:
                    union(a, b)
    return {cid: find(cid) for cid in names}


@dataclass(frozen=True)
class Edge:
    """A directed reuse edge ``src`` reuses ``dst`` with typed provenance.

    ``file``/``line`` always point at the raw evidence (source file line for
    code edges, report line for report edges) so any edge can be re-checked by
    hand.
    """

    src: str
    dst: str
    kind: str          # ast_import | ref_symbol | ref_module | c_include | artifact_ref | report_ref | dependency_hint
    caliber: str       # ast | ref | evidence
    file: str = ""
    line: int = 0
    detail: str = ""

    def key(self) -> Tuple[str, str, str]:
        return (self.src, self.dst, self.kind)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "caliber": self.caliber,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
        }


def deserialize_edges(rows: Sequence[Dict[str, Any]]) -> List[Edge]:
    return [Edge(**{k: r.get(k, "") if k not in ("line",) else int(r.get(k, 0) or 0)
                    for k in ("src", "dst", "kind", "caliber", "file", "line", "detail")})
            for r in rows]


def edge_rows(edges: Iterable[Edge]) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in sorted(edges, key=lambda e: (e.src, e.dst, e.kind, e.file, e.line))]


def dedupe_edges(edges: Iterable[Edge]) -> List[Edge]:
    """One edge per (src, dst, kind); keep the smallest file:line as evidence."""
    best: Dict[Tuple[str, str, str], Edge] = {}
    for e in edges:
        k = e.key()
        cur = best.get(k)
        if cur is None or (e.file, e.line) < (cur.file, cur.line):
            best[k] = e
    return [best[k] for k in sorted(best)]