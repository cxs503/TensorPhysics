"""Maturity report collection and summary for P20 campaign nodes.

Collects maturity artifacts from P20 nodes (N01/N02/N03/N07/N08) and
produces a campaign-level maturity summary with:

* Per-node maturity level (TRL 1-9 scale)
* Delivered artifact inventory
* Cross-node dependency satisfaction
* Campaign-level maturity score
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


MATURITY_SCHEMA = "p20_maturity_summary.v1"

# Expected P20 nodes that contribute maturity reports
P20_MATURITY_NODES = ["P20-N01", "P20-N02", "P20-N03", "P20-N07", "P20-N08"]


@dataclass
class NodeMaturity:
    """Maturity record for one campaign node."""

    node_id: str
    name: str
    trl: int  # Technology Readiness Level 1-9
    delivered: bool
    artifact_count: int
    tests_passed: int
    tests_total: int
    validation_status: str  # "passed", "failed", "withheld"
    notes: str = ""
    artifacts: list[str] = field(default_factory=list)

    @property
    def test_pass_rate(self) -> float:
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "trl": self.trl,
            "delivered": self.delivered,
            "artifact_count": self.artifact_count,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "test_pass_rate": round(self.test_pass_rate, 4),
            "validation_status": self.validation_status,
            "notes": self.notes,
            "artifacts": self.artifacts,
        }


@dataclass
class MaturitySummary:
    """Campaign-level maturity summary."""

    schema: str = MATURITY_SCHEMA
    campaign_id: str = "P20"
    plan_id: int = 27
    nodes: list[NodeMaturity] = field(default_factory=list)
    overall_trl: float = 0.0
    delivered_count: int = 0
    total_count: int = 0
    validation_passed: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "plan_id": self.plan_id,
            "overall_trl": round(self.overall_trl, 2),
            "delivered_count": self.delivered_count,
            "total_count": self.total_count,
            "validation_passed": self.validation_passed,
            "detail": self.detail,
            "nodes": [n.as_dict() for n in self.nodes],
        }


def parse_maturity_artifact(artifact: dict, node_id: str) -> NodeMaturity:
    """Parse a maturity artifact dict into a NodeMaturity record.

    Expected artifact structure::

        {
          "schema": "p20_node_maturity.v1",
          "node_id": "P20-N01",
          "name": "Resistance prediction baseline",
          "trl": 7,
          "delivered": true,
          "artifacts": ["artifacts/n01/resistance.json", ...],
          "tests": {"passed": 12, "total": 12},
          "validation_status": "passed",
          "notes": "Holtrop-Mennen validated against KCS E779"
        }
    """
    return NodeMaturity(
        node_id=artifact.get("node_id") or node_id,
        name=artifact.get("name") or node_id,
        trl=int(artifact.get("trl", 0)),
        delivered=bool(artifact.get("delivered", False)),
        artifact_count=len(artifact.get("artifacts") or []),
        tests_passed=int((artifact.get("tests") or {}).get("passed", 0)),
        tests_total=int((artifact.get("tests") or {}).get("total", 0)),
        validation_status=artifact.get("validation_status", "withheld"),
        notes=artifact.get("notes", ""),
        artifacts=artifact.get("artifacts") or [],
    )


def collect_maturity(evidence_dir: str | Path) -> MaturitySummary:
    """Collect maturity artifacts from the evidence directory.

    Looks for ``maturity/<node_id>.json`` files and parses them.
    Missing nodes are recorded as not delivered.
    """
    d = Path(evidence_dir)
    maturity_dir = d / "maturity"

    nodes: list[NodeMaturity] = []
    for nid in P20_MATURITY_NODES:
        art_path = maturity_dir / f"{nid.lower().replace('-', '_')}.json"
        if not art_path.exists():
            # Try alternate naming
            art_path = maturity_dir / f"{nid}.json"
        if art_path.exists():
            try:
                artifact = json.loads(art_path.read_text())
                nodes.append(parse_maturity_artifact(artifact, nid))
            except (json.JSONDecodeError, ValueError) as exc:
                nodes.append(NodeMaturity(
                    node_id=nid, name=nid, trl=0, delivered=False,
                    artifact_count=0, tests_passed=0, tests_total=0,
                    validation_status="withheld",
                    notes=f"artifact parse error: {exc}"))
        else:
            nodes.append(NodeMaturity(
                node_id=nid, name=nid, trl=0, delivered=False,
                artifact_count=0, tests_passed=0, tests_total=0,
                validation_status="withheld",
                notes="no maturity artifact found"))

    delivered = sum(1 for n in nodes if n.delivered)
    val_passed = sum(1 for n in nodes if n.validation_status == "passed")
    trls = [n.trl for n in nodes if n.trl > 0]
    overall_trl = sum(trls) / len(trls) if trls else 0.0

    detail = (f"{delivered}/{len(nodes)} maturity reports delivered; "
              f"{val_passed}/{len(nodes)} validation passed; "
              f"overall TRL = {overall_trl:.1f}")

    return MaturitySummary(
        nodes=nodes, overall_trl=overall_trl,
        delivered_count=delivered, total_count=len(nodes),
        validation_passed=val_passed, detail=detail,
    )


def build_maturity_from_nodes(campaign_nodes: list[dict],
                              artifacts: dict[str, dict]) -> MaturitySummary:
    """Build maturity summary from campaign node list and artifact dicts.

    Used when evidence directory is not available but node data is.
    """
    nodes: list[NodeMaturity] = []
    for nid in P20_MATURITY_NODES:
        matching = [n for n in campaign_nodes if n.get("ext_id") == nid]
        node_info = matching[0] if matching else {}
        art = artifacts.get(nid)

        if art is not None:
            nodes.append(parse_maturity_artifact(art, nid))
        else:
            nodes.append(NodeMaturity(
                node_id=nid,
                name=node_info.get("name", nid),
                trl=0, delivered=False,
                artifact_count=0, tests_passed=0, tests_total=0,
                validation_status="withheld",
                notes="no maturity artifact in evidence"))

    delivered = sum(1 for n in nodes if n.delivered)
    val_passed = sum(1 for n in nodes if n.validation_status == "passed")
    trls = [n.trl for n in nodes if n.trl > 0]
    overall_trl = sum(trls) / len(trls) if trls else 0.0

    detail = (f"{delivered}/{len(nodes)} maturity reports delivered; "
              f"{val_passed}/{len(nodes)} validation passed; "
              f"overall TRL = {overall_trl:.1f}")

    return MaturitySummary(
        nodes=nodes, overall_trl=overall_trl,
        delivered_count=delivered, total_count=len(nodes),
        validation_passed=val_passed, detail=detail,
    )
