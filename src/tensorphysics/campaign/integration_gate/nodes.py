"""Acceptance evaluation for the eight P16 maturity / new-domain nodes (N1..N8).

The campaign node table (``artifacts/p16_gate/campaign_nodes.json``) fixes the
machine-checkable acceptance criteria of every node as *text*.  A gate can only
be mechanical when the node hands back numbers, so the contract with the nodes
is a small JSON artifact::

    {
      "schema": "p16_node_acceptance.v1",
      "ext_id": "P16-N1",
      "mission_id": 5184,
      "delivered_utc": "2026-11-07T07:00:00+00:00",
      "criteria": {
        "C1": {"value": 0.0021, "limit": 0.003, "op": "le", "pass": true,
               "note": "imposed-field closure R=4/6/8"},
        ...
      },
      "artifacts": [{"path": "artifacts/...json", "sha256": "..."}]
    }

The evaluator never trusts ``pass`` on its own: whenever ``value``, ``limit`` and
``op`` are present it recomputes the comparison and flags an *attestation
mismatch* as a failure.  That keeps "honest FAIL" honest in both directions --
a node cannot self-declare a pass that its own numbers contradict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NODE_ARTIFACT_SCHEMA = "p16_node_acceptance.v1"

_CRIT_ID_RE = re.compile(r"^\s*([A-Za-z]\d+|[a-z])\s*[:：]\s*(.+)$", re.S)
_OP_PATTERNS = (
    ("ge", ("≥", ">=")),
    ("le", ("≤", "<=")),
    ("lt", ("<",)),
    ("gt", (">",)),
)
_REL_TOL = 1e-9


@dataclass
class CriterionVerdict:
    criterion_id: str
    text: str
    status: str            # PASS / FAIL / WITHHELD
    op: str | None = None
    value: float | None = None
    limit: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "text": self.text,
            "status": self.status,
            "op": self.op,
            "value": self.value,
            "limit": self.limit,
            "note": self.note,
        }


@dataclass
class NodeVerdict:
    ext_id: str
    name: str
    repo: str | None
    mission_id: int | None
    node_status: str
    criteria: list[CriterionVerdict] = field(default_factory=list)
    delivered: bool = False
    reason: str | None = None
    evidence_path: str | None = None

    @property
    def status(self) -> str:
        statuses = {c.status for c in self.criteria}
        if "FAIL" in statuses:
            return "FAIL"
        if statuses and statuses == {"PASS"}:
            return "PASS"
        return "WITHHELD"

    def as_dict(self) -> dict:
        return {
            "ext_id": self.ext_id,
            "name": self.name,
            "repo": self.repo,
            "mission_id": self.mission_id,
            "node_status": self.node_status,
            "delivered": self.delivered,
            "verdict": self.status,
            "n_criteria": len(self.criteria),
            "n_pass": sum(c.status == "PASS" for c in self.criteria),
            "n_fail": sum(c.status == "FAIL" for c in self.criteria),
            "n_withheld": sum(c.status == "WITHHELD" for c in self.criteria),
            "reason": self.reason,
            "evidence_path": self.evidence_path,
            "criteria": [c.as_dict() for c in self.criteria],
        }


def parse_criteria(texts: list[str]) -> list[tuple[str, str]]:
    """Split ``"C1: text"`` / ``"a: text"`` criterion strings into (id, text)."""
    out: list[tuple[str, str]] = []
    for i, t in enumerate(texts or []):
        m = _CRIT_ID_RE.match(t or "")
        if m:
            out.append((m.group(1), m.group(2).strip()))
        else:
            out.append((f"#{i + 1}", (t or "").strip()))
    return out


def infer_op(text: str) -> str | None:
    for op, tokens in _OP_PATTERNS:
        if any(tok in text for tok in tokens):
            return op
    return None


def compare(value: float, limit: float, op: str) -> bool:
    if op == "le":
        return value <= limit * (1.0 + _REL_TOL) if limit >= 0 else value <= limit
    if op == "ge":
        return value >= limit * (1.0 - _REL_TOL) if limit >= 0 else value >= limit
    if op == "lt":
        return value < limit
    if op == "gt":
        return value > limit
    raise ValueError(f"unknown comparison operator {op!r}")


def evaluate_node(node: dict, evidence_record: dict | None,
                  artifact: dict | None) -> NodeVerdict:
    """Evaluate one campaign node against its delivered acceptance artifact."""
    pairs = parse_criteria(node.get("acceptance_criteria") or [])
    verdict = NodeVerdict(
        ext_id=node.get("ext_id") or "?",
        name=node.get("name") or "",
        repo=node.get("repo"),
        mission_id=node.get("mission_id"),
        node_status=node.get("status") or "unknown",
        delivered=bool((evidence_record or {}).get("delivered")),
    )

    if artifact is None:
        reason = (
            "no p16_node_acceptance.v1 artifact delivered (node still "
            f"{verdict.node_status!r}, mission accepted="
            f"{(evidence_record or {}).get('accepted')!r}); no criterion is evaluable"
        )
        verdict.reason = reason
        verdict.criteria = [
            CriterionVerdict(cid, text, "WITHHELD", op=infer_op(text), note=reason)
            for cid, text in pairs
        ]
        return verdict

    problems: list[str] = []
    if artifact.get("schema") != NODE_ARTIFACT_SCHEMA:
        problems.append(f"schema tag {artifact.get('schema')!r} != {NODE_ARTIFACT_SCHEMA!r}")
    if artifact.get("ext_id") not in (None, node.get("ext_id")):
        problems.append(f"ext_id {artifact.get('ext_id')!r} != {node.get('ext_id')!r}")
    if node.get("mission_id") and artifact.get("mission_id") not in (None, node.get("mission_id")):
        problems.append(f"mission_id {artifact.get('mission_id')!r} != {node.get('mission_id')!r}")
    if problems:
        verdict.reason = "rejected artifacts: " + "; ".join(problems)
        verdict.criteria = [
            CriterionVerdict(cid, text, "FAIL", op=infer_op(text), note=verdict.reason)
            for cid, text in pairs
        ]
        return verdict

    crit_map = artifact.get("criteria") or {}
    for cid, text in pairs:
        rec = crit_map.get(cid)
        if not isinstance(rec, dict):
            verdict.criteria.append(CriterionVerdict(
                cid, text, "WITHHELD", op=infer_op(text),
                note="criterion not present in delivered artifact"))
            continue
        value = rec.get("value")
        limit = rec.get("limit")
        op = rec.get("op") or infer_op(text)
        declared = rec.get("pass")
        numeric = (isinstance(value, (int, float)) and not isinstance(value, bool)
                   and isinstance(limit, (int, float)) and not isinstance(limit, bool)
                   and op is not None)
        if numeric:
            recomputed = compare(float(value), float(limit), op)
            status = "PASS" if recomputed else "FAIL"
            note = rec.get("note") or ""
            if declared is not None and bool(declared) != recomputed:
                status = "FAIL"
                note = (note + " | attestation mismatch: node declared pass="
                        f"{declared} but {value} {op} {limit} -> {recomputed}").strip(" |")
            verdict.criteria.append(CriterionVerdict(
                cid, text, status, op=op, value=float(value), limit=float(limit), note=note))
        elif declared is not None:
            verdict.criteria.append(CriterionVerdict(
                cid, text, "PASS" if declared else "FAIL", op=op,
                value=float(value) if isinstance(value, (int, float)) else None,
                limit=float(limit) if isinstance(limit, (int, float)) else None,
                note=(rec.get("note") or "") + " [attested, no numeric limit to recompute]"))
        else:
            verdict.criteria.append(CriterionVerdict(
                cid, text, "WITHHELD", op=op,
                note="artifact entry has neither numeric value/limit nor a pass flag"))
    verdict.evidence_path = (artifact.get("artifact_path")
                             or (artifact.get("artifacts") or [{}])[0].get("path"))
    return verdict


def evaluate_nodes(campaign: dict, evidence: dict,
                   artifacts: dict[str, dict] | None = None) -> list[NodeVerdict]:
    """Evaluate every ``P16-N*`` node of the campaign snapshot."""
    artifacts = artifacts or {}
    ev_nodes = (evidence or {}).get("nodes") or {}
    out: list[NodeVerdict] = []
    for node in campaign.get("nodes") or []:
        ext = node.get("ext_id") or ""
        if not ext.startswith("P16-N"):
            continue
        out.append(evaluate_node(node, ev_nodes.get(ext), artifacts.get(ext)))
    return out