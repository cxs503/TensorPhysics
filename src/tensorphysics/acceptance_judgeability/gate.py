"""Acceptance-gate hook: judgeability as a *machine rule*, not a review comment.

Two integration surfaces, both thin on purpose:

1. :func:`clauses_from_node_metadata` -- reads a campaign node's ``metadata`` and
   returns its criteria using the platform's single existing parser
   (``tensorphysics.campaign.wave4_gate.parse_acceptance_criteria``); this package
   never re-implements criterion extraction, so "which clauses exist" stays one
   source of truth.
2. :func:`judgeability_flag` -- the plain ``dict`` a gate record carries
   (``NodeRecord.judgeability``); ``campaign.wave3_gate.NodeRecord.verdict``
   consults it and downgrades a PASS whose criteria cannot carry one.

The flag is a dict rather than an object so the shared gate record does not have
to import this package (no import cycle, and older records without the field keep
working unchanged).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .classify import classify_all
from .support import (
    DEFAULT_POLICY,
    UNIFORM_WEIGHTS,
    Adjudication,
    Clause,
    Evidence,
    PassPolicy,
    WeightPolicy,
    adjudicate,
    enforce_verdict,
    gate_flag,
    node_ks,
)


def clauses_from_node_metadata(metadata: Mapping[str, Any]) -> list[Clause]:
    """Extract a node's acceptance criteria with the platform's existing parser."""
    from tensorphysics.campaign.wave4_gate import parse_acceptance_criteria

    return [Clause(cid, text) for cid, text in parse_acceptance_criteria(metadata)]


def judgeability_flag(
    clauses: Iterable[Any],
    evidence: Evidence,
    *,
    policy: PassPolicy = DEFAULT_POLICY,
    weights: WeightPolicy = UNIFORM_WEIGHTS,
) -> dict[str, Any]:
    """Adjudicate and return the record-level flag."""
    return gate_flag(adjudicate(clauses, evidence, policy=policy, weights=weights))


def judgeability_flag_for_node(
    metadata: Mapping[str, Any],
    *,
    status: str,
    consecutive_fails: int = 0,
    n_reports: int = 0,
    policy: PassPolicy = DEFAULT_POLICY,
    weights: WeightPolicy = UNIFORM_WEIGHTS,
    require_report: bool = False,
) -> dict[str, Any]:
    """Adjudicate a campaign node straight from its ``metadata`` + status fields."""
    node = {
        "status": status,
        "consecutive_fails": consecutive_fails,
        "evidence": {"n_reports": n_reports},
    }
    evidence = Evidence.from_node(node, require_report=require_report)
    return judgeability_flag(
        clauses_from_node_metadata(metadata), evidence, policy=policy, weights=weights
    )


def enforce(verdict: str, flag: Mapping[str, Any] | Adjudication) -> str:
    """Downgrade a pass-like verdict when the flag says its support is hollow."""
    if isinstance(flag, Adjudication):
        return enforce_verdict(verdict, flag)
    if bool(flag.get("supports_pass", True)):
        return str(verdict)
    return "WITHHELD" if str(verdict).upper() in ("ACCEPT", "PASS") else str(verdict)


def scoreboard(nodes: Iterable[Mapping[str, Any]], *, policy: PassPolicy = DEFAULT_POLICY) -> list[dict[str, Any]]:
    """Per-node scorecard rows (KS + class counts + severity) for a node list."""
    out: list[dict[str, Any]] = []
    for node in nodes:
        row = node_ks(node, policy=policy)
        out.append(row.as_dict())
    return out


def blocking_clauses(clauses: Iterable[Any]) -> list[dict[str, Any]]:
    """The clauses that stop a PASS, with the reason each one blocks."""
    classified = classify_all(clauses)
    out: list[dict[str, Any]] = []
    for clause in classified:
        if clause.contributes_to_support:
            continue
        out.append(
            {
                "cid": clause.cid,
                "judgeability": clause.judgeability.value,
                "subtype": clause.subtype.value,
                "text": clause.text,
            }
        )
    return out