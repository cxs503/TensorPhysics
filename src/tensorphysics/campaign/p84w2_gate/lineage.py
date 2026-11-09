"""Lineage inheritance ruling for P84-W2.

Compares plan 84's identified gaps against the delivered nodes to determine
whether this wave has filled the gaps. Each gap is evaluated as:
- ADDRESSED: at least one node explicitly addresses it
- PARTIAL: some aspects addressed but not fully
- NOT_ADDRESSED: no node claims to address it

The ruling produces a per-gap verdict and an overall lineage assessment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan import PLAN_84_GAPS, CAMPAIGN_PLAN


@dataclass
class GapRuling:
    """Ruling on a single plan 84 gap."""
    gap_id: str
    description: str
    addressed_by: list[str]
    status: str  # ADDRESSED / PARTIAL / NOT_ADDRESSED
    evidence: str = ""
    resolution: str = ""

    def as_dict(self) -> dict:
        return {
            "gap_id": self.gap_id,
            "description": self.description,
            "addressed_by": self.addressed_by,
            "status": self.status,
            "evidence": self.evidence,
            "resolution": self.resolution,
        }


@dataclass
class LineageRuling:
    """Overall lineage inheritance ruling."""
    plan_id: int
    wave: int
    total_gaps: int
    addressed: int
    partial: int
    not_addressed: int
    rulings: list[GapRuling] = field(default_factory=list)
    overall_status: str = "FAIL"  # PASS only if all gaps ADDRESSED

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "wave": self.wave,
            "total_gaps": self.total_gaps,
            "addressed": self.addressed,
            "partial": self.partial,
            "not_addressed": self.not_addressed,
            "overall_status": self.overall_status,
            "rulings": [r.as_dict() for r in self.rulings],
        }


def evaluate_gap(gap: dict, artifacts: dict[str, dict], nodes: list[dict]) -> GapRuling:
    """Evaluate whether a single gap is addressed by the delivered nodes.

    Args:
        gap: gap definition from PLAN_84_GAPS
        artifacts: mapping ext_id -> artifact dict
        nodes: list of node definitions from CAMPAIGN_PLAN

    Returns:
        GapRuling with status and evidence
    """
    gap_id = gap["gap_id"]
    addressed_by = gap.get("addressed_by", [])
    resolution = gap.get("resolution", "")

    # Check if all addressing nodes are done with pass verdict
    node_map = {n["ext_id"]: n for n in nodes}
    all_done = True
    all_pass = True
    evidence_parts = []

    for node_id in addressed_by:
        node = node_map.get(node_id)
        if node is None:
            all_done = False
            evidence_parts.append(f"{node_id}: not in campaign plan")
            continue

        if node.get("status") != "done":
            all_done = False
            evidence_parts.append(f"{node_id}: status={node.get('status')}")
            continue

        if node.get("reproduction_verdict") != "pass":
            all_pass = False
            evidence_parts.append(f"{node_id}: verdict={node.get('reproduction_verdict')}")
            continue

        # Check artifact exists
        artifact = artifacts.get(node_id)
        if artifact is None:
            all_pass = False
            evidence_parts.append(f"{node_id}: no artifact delivered")
            continue

        # Check artifact criteria pass
        criteria = artifact.get("criteria", {})
        n_pass = sum(1 for c in criteria.values() if c.get("pass"))
        n_total = len(criteria)
        evidence_parts.append(f"{node_id}: {n_pass}/{n_total} criteria pass")

        if n_pass < n_total:
            all_pass = False

    evidence = "; ".join(evidence_parts) if evidence_parts else "no addressing nodes"

    if all_done and all_pass and len(addressed_by) > 0:
        status = "ADDRESSED"
    elif all_done and len(addressed_by) > 0:
        status = "PARTIAL"
    else:
        status = "NOT_ADDRESSED"

    return GapRuling(
        gap_id=gap_id,
        description=gap["description"],
        addressed_by=addressed_by,
        status=status,
        evidence=evidence,
        resolution=resolution,
    )


def run_lineage_ruling(artifacts: dict[str, dict]) -> LineageRuling:
    """Run lineage inheritance ruling against plan 84 gaps.

    Args:
        artifacts: mapping ext_id -> artifact dict

    Returns:
        LineageRuling with per-gap verdicts and overall status
    """
    nodes = CAMPAIGN_PLAN.get("nodes", [])
    rulings: list[GapRuling] = []

    for gap in PLAN_84_GAPS:
        ruling = evaluate_gap(gap, artifacts, nodes)
        rulings.append(ruling)

    n_addressed = sum(1 for r in rulings if r.status == "ADDRESSED")
    n_partial = sum(1 for r in rulings if r.status == "PARTIAL")
    n_not = sum(1 for r in rulings if r.status == "NOT_ADDRESSED")

    overall = "PASS" if n_addressed == len(rulings) else "FAIL"

    return LineageRuling(
        plan_id=CAMPAIGN_PLAN["plan_id"],
        wave=CAMPAIGN_PLAN["wave"],
        total_gaps=len(rulings),
        addressed=n_addressed,
        partial=n_partial,
        not_addressed=n_not,
        rulings=rulings,
        overall_status=overall,
    )
