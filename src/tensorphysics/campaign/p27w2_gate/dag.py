"""DAG convergence verification for the P27-W2 campaign (plan 153).

The P27-W2 campaign is a 10-node directed acyclic graph:

* N01 (961) BEM radiation/diffraction  -> N02 (962) seakeeping RAO
  -> N03 (963) roll damping three-channel cross-adjudication
* N06 (966) DEM sphere-sphere rolling friction -> N07 (967) DEM
  non-spherical representation
* N04 (964) spectral fatigue, N05 (965) buoyancy-coupled plate,
  N08 (968) turbulence u+ decomposition, N09 (969) AMR backflow
  correction are independent roots
* N10 (970) integration acceptance depends on all nine predecessors

This module provides pure (no I/O) verification:

* ``CampaignDAG`` -- immutable DAG built from a node list; rejects
  unknown dependency references and self-loops at construction time.
* ``topological_sort`` -- Kahn's algorithm; raises ``CycleError`` on a
  cycle.
* ``verify_convergence`` -- acyclicity plus terminal-state check for the
  integration node's dependency closure.  A node is *terminal* when its
  status normalizes to ``done`` or ``honest_fail`` (an honest FAIL is a
  valid terminal state and must never be masked).
* ``inject_cycle`` -- negative-control helper that forces a cycle so the
  cycle detector must fire.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Statuses that count as "finished" for convergence purposes.
TERMINAL_DONE = "done"
TERMINAL_HONEST_FAIL = "honest_fail"
TERMINAL_STATES = frozenset({TERMINAL_DONE, TERMINAL_HONEST_FAIL})


def normalize_status(status: str) -> str:
    """Normalize a campaign status string to its canonical lowercase form."""
    if not isinstance(status, str):
        raise ValueError("status must be a string")
    return status.strip().lower().replace("-", "_").replace(" ", "_")


def is_terminal(status: str) -> bool:
    """True when the status is a valid terminal state (done / honest_fail)."""
    return normalize_status(status) in TERMINAL_STATES


class CycleError(ValueError):
    """The campaign graph contains a cycle; topological sort is impossible."""


@dataclass(frozen=True)
class DAGNode:
    """One node in the campaign DAG."""

    node_id: int
    name: str
    status: str
    depends_on: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, int) or self.node_id <= 0:
            raise ValueError("node_id must be a positive integer")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")


@dataclass(frozen=True)
class ConvergenceResult:
    """Result of DAG convergence verification."""

    acyclic: bool
    all_deps_done: bool
    sorted_ids: list[int] | None
    blocked_nodes: list[int]
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
    """Immutable directed acyclic graph of campaign nodes."""

    def __init__(self, nodes: list[DAGNode]) -> None:
        if not nodes:
            raise ValueError("at least one node is required")
        self._nodes: dict[int, DAGNode] = {}
        for node in nodes:
            if node.node_id in self._nodes:
                raise ValueError(f"duplicate node id {node.node_id}")
            self._nodes[node.node_id] = node
        for node in nodes:
            for dep in node.depends_on:
                if dep == node.node_id:
                    raise ValueError(f"node {node.node_id} depends on itself")
                if dep not in self._nodes:
                    raise ValueError(
                        f"node {node.node_id} depends on unknown node {dep}"
                    )

    @property
    def nodes(self) -> dict[int, DAGNode]:
        return dict(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def topological_sort(self) -> list[int]:
        """Kahn's algorithm; raises :class:`CycleError` when a cycle exists."""
        indegree = {nid: 0 for nid in self._nodes}
        children: dict[int, list[int]] = defaultdict(list)
        for node in self._nodes.values():
            for dep in node.depends_on:
                indegree[node.node_id] += 1
                children[dep].append(node.node_id)
        queue = deque(sorted(nid for nid, deg in indegree.items() if deg == 0))
        order: list[int] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in children[nid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._nodes):
            cyclic = sorted(nid for nid, deg in indegree.items() if deg > 0)
            raise CycleError(f"cycle involves nodes {cyclic}")
        return order

    def dependency_closure(self, node_id: int) -> set[int]:
        """All transitive predecessors of ``node_id`` (exclusive)."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        seen: set[int] = set()
        stack = list(self._nodes[node_id].depends_on)
        while stack:
            dep = stack.pop()
            if dep in seen:
                continue
            seen.add(dep)
            stack.extend(self._nodes[dep].depends_on)
        return seen


def build_dag_from_plan(plan_nodes: list[dict]) -> CampaignDAG:
    """Build a :class:`CampaignDAG` from raw campaign-plan node dicts.

    Each dict needs ``id``, ``name``, ``status`` and optional
    ``dependencies`` (list of node ids).
    """
    nodes = [
        DAGNode(
            node_id=int(n["id"]),
            name=str(n["name"]),
            status=str(n.get("status", "queued")),
            depends_on=tuple(int(d) for d in (n.get("dependencies") or ())),
            metadata=n.get("metadata") or {},
        )
        for n in plan_nodes
    ]
    return CampaignDAG(nodes)


def verify_convergence(
    dag: CampaignDAG,
    integration_node_id: int,
    terminal_states: frozenset[str] = TERMINAL_STATES,
) -> ConvergenceResult:
    """Check acyclicity and that every transitive predecessor of the
    integration node is in a terminal state (done or honest_fail)."""
    total = len(dag)
    try:
        order = dag.topological_sort()
    except CycleError as exc:
        return ConvergenceResult(
            acyclic=False,
            all_deps_done=False,
            sorted_ids=None,
            blocked_nodes=[],
            done_count=0,
            total_count=total,
            detail=f"cycle detected: {exc}",
        )
    closure = dag.dependency_closure(integration_node_id)
    blocked = sorted(
        nid
        for nid in closure
        if normalize_status(dag.nodes[nid].status) not in terminal_states
    )
    done = sum(
        1
        for nid, node in dag.nodes.items()
        if normalize_status(node.status) in terminal_states
    )
    detail = (
        "converged: acyclic and all "
        f"{len(closure)} predecessors of node {integration_node_id} terminal"
        if not blocked
        else (
            f"not converged: {len(blocked)} predecessor(s) of node "
            f"{integration_node_id} not terminal: {blocked}"
        )
    )
    return ConvergenceResult(
        acyclic=True,
        all_deps_done=not blocked,
        sorted_ids=order,
        blocked_nodes=blocked,
        done_count=done,
        total_count=total,
        detail=detail,
    )


def inject_cycle(dag: CampaignDAG, node_id: int) -> CampaignDAG:
    """Return a copy of ``dag`` with a back-edge that closes a cycle.

    A direct child of ``node_id`` (one that already depends on it) is made
    a new dependency of ``node_id``, closing a two-node cycle.  Negative
    control: the resulting graph must fail topological sort with
    :class:`CycleError`.
    """
    target = dag.nodes[node_id]
    children = [
        n.node_id
        for n in dag.nodes.values()
        if node_id in n.depends_on
    ]
    if not children:
        raise ValueError(f"node {node_id} has no children to close a cycle")
    back = children[0]
    mutated = DAGNode(
        node_id=target.node_id,
        name=target.name,
        status=target.status,
        depends_on=target.depends_on + (back,),
        metadata=target.metadata,
    )
    return CampaignDAG([mutated if n.node_id == node_id else n
                       for n in dag.nodes.values()])
