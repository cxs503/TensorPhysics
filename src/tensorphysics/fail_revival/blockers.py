"""Blocker graph of honest FAILs: who is waiting on what, and can it be lifted?

Every FAIL declares one blocker -- a capability slug, an upstream mission, or a
numerical-caliber change (``BLOCKER_KINDS``).  Declarations are data
(``revival_blockers.json``) and carry their own provenance: a curated
declaration cites the report section that states the obstruction; a
rule-derived declaration records the attribution label it came from.  Nothing is
guessed silently.

The graph adds the campaign DAG on top: a FAIL whose mission depends on an
upstream mission inherits that dependency as an edge, so an upstream mission
landing is *visible* to the scheduler even when nobody edited the declaration.

Resolvability is an annotation, not a guess:

``capability_slug``          lifts when the capability is ``done`` in the state
``upstream_mission``         lifts when the upstream mission is ``done``
``numerical_caliber_change`` lifts only when a caliber revision is adopted
``undeclared``               never lifts automatically (stays visible as debt)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Iterable, Mapping, Sequence

from .model import BLOCKER_KINDS, Blocker, FailRecord

DECLARATION_SCHEMA = "fail_blocker_declarations.v1"
RESOLUTION_RULES = {
    "capability_slug": "capability_landing",
    "upstream_mission": "upstream_mission_done",
    "numerical_caliber_change": "caliber_revision",
    "undeclared": "manual_declaration_required",
}


@dataclass(frozen=True)
class BlockerState:
    """What the world currently knows about blockers and campaign progress."""

    capabilities_done: frozenset[str] = frozenset()
    missions_done: frozenset[int] = frozenset()
    caliber_revisions: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities_done": sorted(self.capabilities_done),
            "missions_done": sorted(self.missions_done),
            "caliber_revisions": sorted(self.caliber_revisions),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlockerState":
        return cls(
            capabilities_done=frozenset(d.get("capabilities_done") or ()),
            missions_done=frozenset(int(m) for m in (d.get("missions_done") or ())),
            caliber_revisions=frozenset(d.get("caliber_revisions") or ()),
        )


def load_declarations(path: Path | str) -> tuple[dict[str, Blocker], dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != DECLARATION_SCHEMA:
        raise ValueError("unexpected blocker declaration schema: %r" % (data.get("schema"),))
    blockers: dict[str, Blocker] = {}
    for entry in data.get("blockers", []):
        b = Blocker.from_dict(entry)
        blockers[b.blocker_id] = b
    return blockers, data


def blocker_of(fail_id: str, blockers: Mapping[str, Blocker]) -> Blocker | None:
    for b in blockers.values():
        if fail_id in b.fail_keys:
            return b
    return None


def blocker_done(blocker: Blocker, state: BlockerState) -> bool:
    if blocker.kind == "capability_slug":
        slug = blocker.blocker_id.split(":", 1)[-1]
        return slug in state.capabilities_done or blocker.blocker_id in state.capabilities_done
    if blocker.kind == "upstream_mission":
        raw = blocker.blocker_id.split(":", 1)[-1]
        try:
            return int(raw) in state.missions_done
        except ValueError:
            return False
    if blocker.kind == "numerical_caliber_change":
        return blocker.blocker_id in state.caliber_revisions
    return False


@dataclass(frozen=True)
class BlockerGraph:
    records: tuple[FailRecord, ...]
    blockers: Mapping[str, Blocker]
    edges: tuple[tuple[str, str, str], ...]  # (src, dst, kind)
    provenance: Mapping[str, Any]

    # ------------------------------------------------------------------ views
    def fail_nodes(self) -> tuple[str, ...]:
        return tuple(r.fail_id for r in self.records)

    def blocker_nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self.blockers))

    def edges_of(self, node: str) -> tuple[tuple[str, str, str], ...]:
        return tuple(e for e in self.edges if e[0] == node or e[1] == node)

    def dangling(self) -> tuple[str, ...]:
        """FAILs with no declared blocker (they stay in the debt column)."""
        return tuple(r.fail_id for r in self.records if blocker_of(r.fail_id, self.blockers) is None)

    def topological_order(self) -> tuple[str, ...] | None:
        """Kahn order over fail/blocker/mission nodes; ``None`` if cyclic."""
        nodes = set(self.fail_nodes()) | set(self.blocker_nodes())
        nodes |= {n for e in self.edges for n in (e[0], e[1])}
        indeg = {n: 0 for n in nodes}
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for src, dst, _ in self.edges:
            if src not in adj or dst not in adj:
                continue
            adj[src].append(dst)
            indeg[dst] += 1
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in sorted(adj[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
            ready.sort()
        return tuple(order) if len(order) == len(nodes) else None

    def resolvability(self, state: BlockerState) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for bid, b in sorted(self.blockers.items()):
            out[bid] = {
                "kind": b.kind,
                "resolution_rule": RESOLUTION_RULES[b.kind],
                "done": blocker_done(b, state),
                "fail_count": len(b.fail_keys),
                "note": b.note,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fail_blocker_graph.v1",
            "fail_nodes": list(self.fail_nodes()),
            "blocker_nodes": list(self.blocker_nodes()),
            "edges": [{"src": s, "dst": d, "kind": k} for s, d, k in self.edges],
            "dangling": list(self.dangling()),
            "acyclic": self.topological_order() is not None,
            "blocker_kinds": {b: sorted(set(x.kind for x in self.blockers.values())) for b in [""]}[""],
            "declaration_provenance": dict(self.provenance),
        }


def build_graph(records: Sequence[FailRecord], blockers: Mapping[str, Blocker],
                *, mission_deps: Mapping[int, Iterable[int]] | None = None,
                provenance: Mapping[str, Any] | None = None) -> BlockerGraph:
    edges: list[tuple[str, str, str]] = []
    for rec in records:
        b = blocker_of(rec.fail_id, blockers)
        if b is None:
            continue
        edges.append((rec.fail_id, b.blocker_id, "blocked_by"))
    for mission, deps in sorted((mission_deps or {}).items()):
        for dep in sorted(set(int(d) for d in deps)):
            edges.append(("mission:%d" % dep, "mission:%d" % mission, "upstream_of"))
            for rec in records:
                if rec.mission_id == mission:
                    edges.append(("mission:%d" % dep, rec.fail_id, "upstream_of"))
    return BlockerGraph(
        records=tuple(sorted(records, key=lambda r: (r.mission_id, r.line_no))),
        blockers=blockers,
        edges=tuple(edges),
        provenance=dict(provenance or {}),
    )


def undeclared_blocker(fail_id: str, note: str) -> Blocker:
    """Explicit 'nobody declared a blocker' marker (never auto-resolvable)."""
    return Blocker(blocker_id="undeclared:" + fail_id, kind="undeclared",
                   fail_keys=(fail_id,), note=note)