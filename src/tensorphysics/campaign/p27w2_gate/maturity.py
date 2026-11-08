"""Maturity aggregation for the P27-W2 campaign (criterion c).

The campaign designates two maturation nodes:

* **N04 (964)** spectral fatigue -- Dirlik moment approximation vs
  Rayleigh narrow-band limit, analytic self-consistency ladder.
* **N09 (969)** AMR backflow correction -- flux-conserving LBM adaptive
  mesh refinement, cross-level flux balance.

The maturity report must contain an output summary for *every* maturation
node: verdict, key numbers, artifact paths.  Aggregation is pure so the
gate can be tested without I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: node_id -> (label, kind) for the P27-W2 maturation nodes
P27W2_MATURITY_NODES: dict[int, tuple[str, str]] = {
    964: ("N04 spectral fatigue (Dirlik vs Rayleigh)", "spectral_fatigue"),
    969: ("N09 AMR backflow correction (flux-conserving LBM AMR)", "amr"),
}


@dataclass
class NodeMaturity:
    """Maturity record of one maturation node."""

    node_id: int
    label: str
    kind: str
    verdict: str  # "done" | "honest_fail" | "in_flight" | "missing"
    summary: str
    key_numbers: dict[str, float] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)

    @property
    def has_summary(self) -> bool:
        return bool(self.summary.strip())

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "kind": self.kind,
            "verdict": self.verdict,
            "summary": self.summary,
            "key_numbers": self.key_numbers,
            "artifact_paths": list(self.artifact_paths),
            "has_summary": self.has_summary,
        }


@dataclass
class MaturityReport:
    """Campaign-level maturity report."""

    nodes: list[NodeMaturity] = field(default_factory=list)

    @property
    def covered_node_ids(self) -> set[int]:
        return {n.node_id for n in self.nodes}

    def missing_nodes(self, expected: dict[int, tuple[str, str]]) -> list[int]:
        """Expected nodes without a non-empty output summary."""
        summarized = {n.node_id for n in self.nodes if n.has_summary}
        return sorted(set(expected) - summarized)

    @property
    def complete(self) -> bool:
        return bool(self.nodes) and all(n.has_summary for n in self.nodes)

    def as_dict(self) -> dict:
        return {
            "maturation_nodes": [n.as_dict() for n in self.nodes],
            "node_count": len(self.nodes),
            "all_summarized": self.complete,
        }


def parse_maturity_artifact(node_id: int, artifact: dict | None) -> NodeMaturity:
    """Build a :class:`NodeMaturity` from a node's result artifact.

    ``artifact`` is the parsed ``result.json`` (or equivalent) of the
    node.  A missing artifact yields an ``in_flight``/``missing`` record
    so the gate can report the gap honestly.
    """
    label, kind = P27W2_MATURITY_NODES.get(
        node_id, (f"node {node_id}", "unknown")
    )
    if not artifact:
        return NodeMaturity(
            node_id=node_id, label=label, kind=kind,
            verdict="missing", summary="",
        )
    if "verdict" in artifact:
        verdict = str(artifact["verdict"])
    elif "status" in artifact:
        verdict = str(artifact["status"])
    elif artifact.get("accepted") is False:
        verdict = "honest_fail"
    else:
        verdict = "done"
    summary = str(artifact.get("summary", "")).strip()
    if not summary:
        summary = str(artifact.get("headline", "")).strip()
    if not summary:
        numbers = list((artifact.get("key_numbers") or {}).items())[:5]
        if numbers:
            summary = (
                f"verdict={verdict}; "
                + "; ".join(f"{k}={v}" for k, v in numbers)
            )
        else:
            summary = f"verdict={verdict}"
    return NodeMaturity(
        node_id=node_id,
        label=label,
        kind=kind,
        verdict=verdict,
        summary=summary,
        key_numbers={
            k: float(v)
            for k, v in (artifact.get("key_numbers") or {}).items()
            if isinstance(v, (int, float))
        },
        artifact_paths=list(artifact.get("artifacts") or []),
    )


def build_maturity_report(
    artifacts: dict[int, dict | None],
    expected: dict[int, tuple[str, str]] = P27W2_MATURITY_NODES,
) -> MaturityReport:
    """Aggregate maturation-node artifacts into a campaign report.

    ``artifacts`` maps node id to its parsed result artifact (or ``None``
    when the node has not produced one yet).  Every expected node gets a
    record, so coverage gaps are visible instead of silent.
    """
    nodes = [
        parse_maturity_artifact(node_id, artifacts.get(node_id))
        for node_id in sorted(expected)
    ]
    return MaturityReport(nodes=nodes)


def maturity_coverage(
    report: MaturityReport,
    expected: dict[int, tuple[str, str]] = P27W2_MATURITY_NODES,
) -> dict:
    """Coverage statistics used by the gate for criterion (c)."""
    missing = report.missing_nodes(expected)
    summarized = [n for n in report.nodes if n.has_summary]
    return {
        "expected_nodes": sorted(expected),
        "covered_nodes": sorted(report.covered_node_ids),
        "missing_nodes": missing,
        "summarized_count": len(summarized),
        "coverage": (
            len(summarized) / len(expected) if expected else 0.0
        ),
        "complete": not missing and report.complete,
    }
