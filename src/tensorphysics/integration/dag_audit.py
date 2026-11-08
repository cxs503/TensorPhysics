"""Campaign DAG convergence audit (P27W3 nodes N01..N08).

The audit answers one question with machine evidence: *are all upstream nodes
of this campaign node done, and what did they claim?*  It reads the platform's
own tables (read-only ``psql``; no ORM dependency, no writes), so the verdict
is reproducible outside the agent session::

    psql -t -A -c "select id,name,status,... from campaign_nodes where plan_id=165"

Anything that cannot be read (database down, table missing) is reported as
``WITHHELD`` with the exact command and error, never as a pass.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "CampaignNode",
    "audit_plan",
    "NODE_CRITERIA",
]

#: Acceptance criteria of the P27W3 nodes (mirrors campaign_nodes.metadata;
#: duplicated here only so a missing database is still visible as WITHHELD
#: rather than as an empty criterion list).
NODE_CRITERIA = {
    "P27W3-N01": ["Haskind relation", "pulsating Green function convergence",
                  "hemisphere added mass/damping analytic anchor"],
    "P27W3-N02": ["long-wave heave RAO -> 1 (<5%)", "Wigley resonance (<5%)",
                  "Ikeda five-component self-consistency (<3%)",
                  "B44 vs Ikeda high-frequency trend (rho>0.9)",
                  "free-decay zeta_eq vs Ikeda (<15%)",
                  "m5292 three fields upgraded to REAL", "pytest>=20 green"],
    "P27W3-N03": ["SUBOFF C_T (<8%)", "KCS C_T (<5%)", "Cb/L/B self-consistency",
                  "reference quadruple fail-closed", "guide_tracker verified"],
    "P27W3-N04": ["Rayleigh limit (<1%)", "Dirlik K-S (<=0.05)",
                  "narrow-band degeneracy K-S (<=0.01)", "Wirsching-Light (<5%)",
                  "S-N integration (<0.1%)", "two-slope S-N (<1%)",
                  "Miner accumulation (<=1e-6)", "zero-stress control",
                  "Hs monotonicity", "pytest>=20 green"],
    "P27W3-N05": ["cantilever deflection (<1%)", "plate deflection (<2%)",
                  "cantilever f1 (<3%)", "plate f11 (<3%)", "K_t = 3.0 +- 0.1",
                  "h-refinement order >= 1.9"],
    "P27W3-N06": ["tangential spring (<2%)", "rolling resistance (<3%)",
                  "spin damping R^2>0.98", "superellipsoid stiffness (<=2%)",
                  "clump convergence (<=1 deg)",
                  "static-pile dissipation (<=1e-6/step)", "verdict.json"],
    "P27W3-N07": ["u+ 95% CI width <2%", "domain-length <1%", "SGS negative control",
                  "AMR mass conservation <1e-12", "AMR momentum <1e-10",
                  "AMR vs uniform L2 <1%", "interface flux <1e-14",
                  "u+_error_decomposition.json"],
    "P27W3-N08": ["7 upstream nodes done", "BEM <-> seakeeping RAO transfer <1%",
                  "structural <-> fatigue transfer <1%",
                  "DEM <-> ice-plate bond mapping <2%",
                  "maturity evidence records complete",
                  "integration report >= 3000 chars", "pytest>=10 green"],
}


@dataclass(frozen=True)
class CampaignNode:
    node_id: int
    name: str
    status: str
    mission_id: int | None
    dependencies: tuple[int, ...]
    ext_id: str
    is_maturity: bool

    @property
    def done(self) -> bool:
        return self.status == "done"

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "ext_id": self.ext_id,
            "status": self.status,
            "done": self.done,
            "mission_id": self.mission_id,
            "dependencies": list(self.dependencies),
            "is_maturity": self.is_maturity,
            "acceptance_criteria": NODE_CRITERIA.get(self.ext_id, []),
        }


_QUERY = (
    "select coalesce(json_agg(json_build_object("
    "'id', id, 'name', name, 'status', status, 'mission_id', mission_id, "
    "'dependencies', dependencies, 'ext_id', metadata->>'ext_id', "
    "'is_maturity', coalesce((metadata->>'is_maturity')::boolean, false))"
    " order by id), '[]'::json) "
    "from campaign_nodes where plan_id = {plan_id}"
)


def audit_plan(plan_id: int, dsn_args: Sequence[str] = ("psql", "-U", "postgres",
                                                        "-d", "tensorplatform",
                                                        "-t", "-A", "-c"),
               timeout: float = 30.0) -> dict:
    """Read the node table of ``plan_id`` and classify DAG convergence.

    Returns a dict with ``available`` (bool), ``nodes`` (list), ``all_done``,
    ``blocking`` (nodes not done) and the provenance of the query.
    """
    cmd = list(dsn_args) + [_QUERY.format(plan_id=int(plan_id))]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:      # pragma: no cover
        return {"available": False, "error": str(exc), "command": " ".join(cmd),
                "checked_at_utc": stamp, "nodes": [], "all_done": False,
                "blocking": [], "plan_id": plan_id}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip()[:400],
                "command": " ".join(cmd), "checked_at_utc": stamp, "nodes": [],
                "all_done": False, "blocking": [], "plan_id": plan_id}

    rows = json.loads(proc.stdout or "[]")
    nodes = [CampaignNode(
        node_id=int(r["id"]), name=str(r["name"]), status=str(r["status"]),
        mission_id=None if r["mission_id"] is None else int(r["mission_id"]),
        dependencies=tuple(int(v) for v in (r.get("dependencies") or [])),
        ext_id=str(r.get("ext_id") or ""), is_maturity=bool(r.get("is_maturity")),
    ) for r in rows]
    blocking = [n for n in nodes if not n.done]
    return {
        "available": True,
        "plan_id": int(plan_id),
        "checked_at_utc": stamp,
        "query": _QUERY.format(plan_id=int(plan_id)),
        "nodes": [n.as_dict() for n in nodes],
        "all_done": bool(nodes) and not blocking,
        "n_nodes": len(nodes),
        "n_done": sum(1 for n in nodes if n.done),
        "blocking": [{"node_id": n.node_id, "ext_id": n.ext_id,
                      "status": n.status, "mission_id": n.mission_id}
                     for n in blocking],
    }