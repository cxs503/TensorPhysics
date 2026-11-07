"""DAG construction, topological sort, and cycle detection for P20 campaign plans.

Plan 27 defines a directed acyclic graph of campaign nodes.  This module
provides:

* ``CampaignDAG`` -- immutable DAG built from a node list with explicit
  ``depends_on`` edges.
* ``topological_sort`` -- Kahn's algorithm; returns ordered node IDs or raises
  ``CycleError`` when a cycle is detected.
* ``verify_convergence`` -- checks that every node's dependencies are in
  ``done`` state and the graph is acyclic.
* ``inject_cycle`` -- negative-control helper that deliberately creates a
  cycle so that ``topological_sort`` must raise ``CycleError``.

All functions are pure (no I/O) so they can be unit-tested without fixtures.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class CycleError(ValueError):
    """The campaign graph contains a cycle; topological sort is impossible."""


@dataclass(frozen=True)
class DAGNode:
    """One node in the campaign DAG."""

    node_id: str
    name: str
    status: str  # "done", "running", "queued", "failed", "withheld"
    depends_on: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")


@dataclass(frozen=True)
class ConvergenceResult:
    """Result of DAG convergence verification."""

    acyclic: bool
    all_deps_done: bool
    sorted_ids: list[str] | None
    blocked_nodes: list[str]
    done_count: int
    total_count: int
    detail: str

    @property
    def converged(self) -> bool:
        return self.acyclic and self.all_deps_done

    def as_dict(self) -> dict:
        return {
            "acyclic": self.acyclic,
            "all_deps_done": self.all_deps_done,
            "converged": self.converged,
            "sorted_ids": self.sorted_ids,
            "blocked_nodes": self.blocked_nodes,
            "done_count": self.done_count,
            "total_count": self.total_count,
            "detail": self.detail,
        }


class CampaignDAG:
    """Immutable directed acyclic graph of campaign nodes.

    Construction validates that every ``depends_on`` reference points to a
    known node ID.  Self-loops are rejected at construction time.
    """

    def __init__(self, nodes: Iterable[DAGNode]) -> None:
        self._nodes: dict[str, DAGNode] = {}
        self._adj: dict[str, list[str]] = defaultdict(list)  # parent -> children
        self._in_degree: dict[str, int] = {}

        for node in nodes:
            if node.node_id in self._nodes:
                raise ValueError(f"duplicate node_id: {node.node_id!r}")
            self._nodes[node.node_id] = node
            self._in_degree[node.node_id] = 0

        for node in nodes:
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise ValueError(
                        f"node {node.node_id!r} depends on unknown node {dep!r}")
                if dep == node.node_id:
                    raise ValueError(
                        f"self-loop detected on node {node.node_id!r}")
                self._adj[dep].append(node.node_id)
                self._in_degree[node.node_id] = self._in_degree.get(node.node_id, 0) + 1

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def get_node(self, node_id: str) -> DAGNode:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id!r}")
        return self._nodes[node_id]

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm.  Returns ordered node IDs or raises ``CycleError``."""
        in_deg = dict(self._in_degree)
        queue: deque[str] = deque(n for n, d in in_deg.items() if d == 0)
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self._adj.get(node, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            remaining = [n for n in self._nodes if n not in set(order)]
            raise CycleError(
                f"cycle detected among nodes: {remaining}")
        return order

    def verify_convergence(self) -> ConvergenceResult:
        """Check acyclicity and that all dependencies are in 'done' state."""
        try:
            sorted_ids = self.topological_sort()
            acyclic = True
        except CycleError:
            return ConvergenceResult(
                acyclic=False, all_deps_done=False, sorted_ids=None,
                blocked_nodes=self.node_ids,
                done_count=sum(1 for n in self._nodes.values() if n.status == "done"),
                total_count=len(self._nodes),
                detail="graph contains a cycle; convergence impossible",
            )

        done_ids = {nid for nid, n in self._nodes.items() if n.status == "done"}
        blocked: list[str] = []
        for nid in sorted_ids:
            node = self._nodes[nid]
            if node.status != "done" and node.depends_on:
                missing = [d for d in node.depends_on if d not in done_ids]
                if missing:
                    blocked.append(nid)

        all_deps_done = len(blocked) == 0
        done_count = sum(1 for n in self._nodes.values() if n.status == "done")

        if all_deps_done:
            detail = (f"all {len(self._nodes)} nodes acyclic and dependencies satisfied; "
                      f"{done_count}/{len(self._nodes)} in 'done' state")
        else:
            detail = (f"{len(blocked)} node(s) blocked by unfinished dependencies: "
                      f"{blocked}")

        return ConvergenceResult(
            acyclic=True, all_deps_done=all_deps_done, sorted_ids=sorted_ids,
            blocked_nodes=blocked, done_count=done_count,
            total_count=len(self._nodes), detail=detail,
        )


def build_dag_from_plan(plan: dict) -> CampaignDAG:
    """Build a ``CampaignDAG`` from a plan dict (e.g. from campaign_nodes.json).

    Expected plan structure::

        {
          "plan_id": 27,
          "nodes": [
            {"ext_id": "P20-N01", "name": "...", "status": "done",
             "depends_on": []},
            ...
          ]
        }
    """
    nodes = []
    for n in plan.get("nodes") or []:
        ext_id = n.get("ext_id") or n.get("node_id") or ""
        name = n.get("name") or ext_id
        status = n.get("status") or "unknown"
        deps = tuple(n.get("depends_on") or n.get("dependencies") or [])
        nodes.append(DAGNode(node_id=ext_id, name=name, status=status,
                             depends_on=deps, metadata=n))
    return CampaignDAG(nodes)


def inject_cycle(nodes: list[DAGNode], cycle_chain: list[str] | None = None) -> list[DAGNode]:
    """Negative-control helper: mutate a copy of nodes to create a cycle.

    Creates a circular dependency chain: chain[0] → chain[1] → ... → chain[-1] → chain[0].
    Each node in the chain gets the previous node (wrapping around) as a dependency,
    guaranteeing a cycle regardless of the original dependency structure.

    If ``cycle_chain`` is given, it must be a list of existing node IDs.
    Otherwise the first two nodes are used.
    """
    node_ids = {n.node_id for n in nodes}
    if cycle_chain is None:
        if len(nodes) < 2:
            raise ValueError("need at least 2 nodes to inject a cycle")
        cycle_chain = [nodes[0].node_id, nodes[1].node_id]

    for cid in cycle_chain:
        if cid not in node_ids:
            raise ValueError(f"cycle_chain references unknown node: {cid!r}")

    # Build the set of cycle dependencies: each node depends on its predecessor
    cycle_deps: dict[str, str] = {}
    for i, cid in enumerate(cycle_chain):
        prev = cycle_chain[i - 1]  # wraps around: chain[0] depends on chain[-1]
        cycle_deps[cid] = prev

    mutated = []
    for n in nodes:
        deps = list(n.depends_on)
        if n.node_id in cycle_deps:
            # Add the cycle predecessor as a dependency
            pred = cycle_deps[n.node_id]
            if pred not in deps:
                deps.append(pred)
        mutated.append(DAGNode(
            node_id=n.node_id, name=n.name, status=n.status,
            depends_on=tuple(deps), metadata=n.metadata))
    return mutated
