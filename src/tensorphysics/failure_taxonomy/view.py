"""Blocked-node view: the consumer used by the campaign DAG (criterion C6).

A campaign gate knows *which* nodes are blocked; it does not know *why* they are
blocked in method terms, and it certainly does not know what to do next.  This
module glues the two together:

    blocked node ids + FAIL records  ->  [{node_id, failure_class, evidence, reroute_suggestion}]

Design notes
------------
* The view never invents a record.  A blocked node without a frozen/attached
  FAIL record is reported as ``UNCLASSIFIED`` with an explicit evidence-gap
  entry and an ``ESCALATE`` suggestion, which keeps the coverage statistic
  honest instead of silently dropping the node.
* Each view item carries the *matched* trigger evidence fields, so the reroute
  move is auditable back to the record.
* :func:`to_ledger_entry` renders a view item as a capability-delta shaped
  entry with ``category="falsified_hypothesis"``: a failed attempt is a
  negative result and is booked, not deleted.  The campaign ledger consumes
  these entries (the ledger mechanism delivered by the upstream node).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .decide import DEFAULT_TABLE, ESCALATE, RerouteDecision, decide
from .evidence import Evidence, extract
from .labelers import SignalRuleLabeler
from .selective import LOW_MARGIN, margin_of, selective_decision
from .taxonomy import CLASS_CODES

UNCLASSIFIED = "UNCLASSIFIED"


@dataclass(frozen=True)
class FailRecordView:
    """Minimal record shape accepted by the view (node-aware FAIL record)."""

    node_id: str
    body: str
    record_id: str = ""
    provenance: Mapping[str, object] | None = None


def _as_evidence(record) -> Evidence:
    if isinstance(record, Evidence):
        return record
    if isinstance(record, FailRecordView):
        raise TypeError("FailRecordView must be labelled, not passed to _as_evidence")
    return record.evidence  # type: ignore[attr-defined]


def blocked_node_view(
    blocked_nodes: Iterable[str],
    records: Iterable[object],
    labeler: SignalRuleLabeler | None = None,
    table=DEFAULT_TABLE,
    margin_floor: float | None = None,
) -> list[dict]:
    """Emit ``{failure_class, evidence, reroute_suggestion}`` per blocked node.

    ``records`` may hold :class:`FailRecordView` objects or any object exposing
    ``node_id``, ``body`` and optionally ``record_id``/``provenance``.
    """
    by_node: dict[str, object] = {}
    for rec in records:
        by_node[str(getattr(rec, "node_id"))] = rec

    assign = labeler.label if labeler is not None else None
    if assign is None:
        raise ValueError("a labeler is required for the blocked-node view")

    items: list[dict] = []
    for node_id in blocked_nodes:
        node_id = str(node_id)
        rec = by_node.get(node_id)
        if rec is None:
            items.append(
                {
                    "node_id": node_id,
                    "failure_class": UNCLASSIFIED,
                    "evidence": {"reason": "no FAIL record attached to this node", "signals": {}},
                    "reroute_suggestion": {
                        "action": ESCALATE,
                        "secondary_action": "",
                        "matched_triggers": [],
                        "evidence_gap": True,
                        "rationale_code": "missing_fail_record",
                        "expected_effect": "manual triage: attach the node's FAIL record first",
                    },
                }
            )
            continue

        body = str(getattr(rec, "body"))
        labelled = assign(body)
        evidence = extract(body, labeler.lexicon) if hasattr(labeler, "lexicon") else None
        present = evidence.present() if evidence is not None else ()
        decision: RerouteDecision = decide(labelled.code, present, table=table)
        evidence_block: dict[str, object] = {
            "record_id": str(getattr(rec, "record_id", "")),
            "signals": list(present),
            "confidence_margin": labelled.confidence,
            "provenance": dict(getattr(rec, "provenance", {}) or {}),
        }
        if margin_floor is not None and evidence is not None:
            score_margin = margin_of(labeler.scores(evidence))
            evidence_block["score_margin"] = round(score_margin, 6)
            decision = selective_decision(decision, score_margin, margin_floor)
            evidence_block["abstained"] = decision.rationale_code == LOW_MARGIN
        items.append(
            {
                "node_id": node_id,
                "failure_class": labelled.code,
                "evidence": evidence_block,
                "reroute_suggestion": decision.as_dict(),
            }
        )
    return items


def summarise_view(items: Sequence[Mapping[str, object]]) -> dict:
    """Coverage / routing statistics of a view (used by gates and the report)."""
    by_class: dict[str, int] = {c: 0 for c in CLASS_CODES}
    by_action: dict[str, int] = {}
    unclassified = 0
    for item in items:
        code = str(item.get("failure_class"))
        if code == UNCLASSIFIED:
            unclassified += 1
        else:
            by_class[code] = by_class.get(code, 0) + 1
        action = str((item.get("reroute_suggestion") or {}).get("action"))
        by_action[action] = by_action.get(action, 0) + 1
    n = len(items)
    return {
        "n_blocked": n,
        "classified": n - unclassified,
        "unclassified": unclassified,
        "coverage": (n - unclassified) / n if n else 0.0,
        "by_class": by_class,
        "by_action": by_action,
    }


def to_ledger_entry(item: Mapping[str, object], mission_id: str | int | None = None) -> dict:
    """Render one view item as a capability-delta entry (``falsified_hypothesis``)."""
    code = str(item.get("failure_class"))
    suggestion = dict(item.get("reroute_suggestion") or {})
    node_id = str(item.get("node_id"))
    action = str(suggestion.get("action"))
    return {
        "category": "falsified_hypothesis",
        "summary": f"{node_id}: {code} -> {action}",
        "reason": (
            f"node blocked with failure class {code}; "
            f"routing rationale {suggestion.get('rationale_code', '')}; "
            f"trigger evidence {suggestion.get('matched_triggers', [])}"
        ),
        "evidence": {
            "node_id": node_id,
            "failure_class": code,
            "evidence": item.get("evidence", {}),
            "reroute": suggestion,
            "mission_id": mission_id,
        },
        "artifacts": [f"failure_class={code}", f"reroute={action}"],
    }


__all__ = [
    "FailRecordView",
    "LOW_MARGIN",
    "UNCLASSIFIED",
    "blocked_node_view",
    "summarise_view",
    "to_ledger_entry",
]