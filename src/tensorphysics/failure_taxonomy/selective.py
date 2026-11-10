"""Selective classification: abstain from a reroute when the annotator is unsure.

Why this exists
---------------
The document-disjoint audit (:mod:`tensorphysics.failure_taxonomy.audit`) shows the
in-corpus agreement of the rule annotator does not survive unseen documents.  A
scheduler consuming the taxonomy has a second line of defence: do not route what the
annotator cannot route confidently.  This module turns the annotator's own score
margin into an abstention gate, and measures the risk/coverage trade-off that the
gate buys.

Two quantities are kept separate on purpose:

``margin``
    ``top1_score - top2_score`` of the class scores.  It is available at decision
    time (no labels needed), so a gate built on it can run in production.
``risk_coverage``
    for a threshold sweep, the fraction of records still routed (coverage) and the
    error rate among them (risk).  A gate is useful only if risk falls as coverage
    falls; a flat curve means the margin carries no information about errors and the
    gate is theatre.  ``aurc`` summarises the whole curve (lower is better, 0 is
    perfect ranking of errors by margin).

The abstention itself is *not* a seventh class: an abstained record is reported with
the ``ESCALATE`` directive, the reason code ``low_margin`` and the margin that
triggered it, which keeps the audit trail of *why* the scheduler did not route.

Domain-neutral: nothing here knows what the labels mean.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from .decide import ESCALATE, RerouteDecision
from .evidence import Evidence, extract
from .labelers import SignalRuleLabeler
from .taxonomy import CLASS_CODES

LOW_MARGIN = "low_margin"


@dataclass(frozen=True)
class Margined:
    """A class assignment together with the score margin that supports it."""

    code: str
    margin: float
    scores: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "margin": round(self.margin, 6),
            "scores": {k: round(v, 6) for k, v in sorted(self.scores.items())},
        }


def margin_of(scores: dict[str, float]) -> float:
    """``top1 - top2`` over the class scores; 0.0 when nothing is scored."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ordered) < 2 or ordered[0][1] <= 0.0:
        return 0.0
    return float(ordered[0][1] - ordered[1][1])


def label_with_margin(labeler: SignalRuleLabeler, body: str, lexicon=None) -> Margined:
    """Label a body and attach the margin of the winning class."""
    evidence = extract(body, lexicon) if lexicon is not None else None
    if evidence is None:
        labelled = labeler.label(body)
        evidence = labelled.evidence if hasattr(labelled, "evidence") else None
    if evidence is None:  # pragma: no cover - labelers always carry evidence
        raise TypeError("labeler did not expose the evidence it used")
    scores = labeler.scores(evidence)
    return Margined(code=labeler.label_from_evidence(evidence).code, margin=margin_of(scores), scores=dict(scores))


def margins(labeler: SignalRuleLabeler, bodies: Iterable[str], lexicon=None) -> list[Margined]:
    return [label_with_margin(labeler, body, lexicon) for body in bodies]


def risk_coverage(
    codes: Sequence[str],
    margins_: Sequence[float],
    gold: Sequence[str],
    thresholds: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0),
) -> list[dict]:
    """Risk/coverage at each margin threshold (records kept iff ``margin >= t``)."""
    if not (len(codes) == len(margins_) == len(gold)):
        raise ValueError("codes, margins and gold must have the same length")
    n = len(gold)
    rows = []
    for t in sorted(thresholds):
        kept = [i for i in range(n) if margins_[i] >= t]
        if kept:
            correct = sum(1 for i in kept if codes[i] == gold[i])
            acc = correct / len(kept)
        else:
            acc = 0.0
        rows.append(
            {
                "threshold": round(float(t), 6),
                "n_kept": len(kept),
                "coverage": round(len(kept) / n, 6) if n else 0.0,
                "accuracy": round(acc, 6),
                "risk": round(1.0 - acc, 6) if kept else 1.0,
            }
        )
    return rows


def aurc(curve: Sequence[dict]) -> float:
    """Area under the risk-coverage curve, integrated in coverage order."""
    pts = sorted(curve, key=lambda r: r["coverage"])
    area, prev_x = 0.0, 0.0
    for row in pts:
        x, y = row["coverage"], row["risk"]
        area += (x - prev_x) * y
        prev_x = x
    if pts:
        area += (1.0 - prev_x) * pts[-1]["risk"]
    return round(area, 6)


def selective_decision(
    decision: RerouteDecision,
    margin: float,
    floor: float,
) -> RerouteDecision:
    """Downgrade a reroute to ``ESCALATE`` when the margin is below ``floor``."""
    if margin >= floor:
        return decision
    return replace(
        decision,
        action=ESCALATE,
        secondary_action="",
        evidence_gap=True,
        rationale_code=LOW_MARGIN,
        expected_effect=(
            "abstain: score margin {:.3f} below floor {:.3f}; collect more evidence "
            "or route by hand".format(margin, floor)
        ),
    )


__all__ = [
    "LOW_MARGIN",
    "Margined",
    "aurc",
    "label_with_margin",
    "margin_of",
    "margins",
    "risk_coverage",
    "selective_decision",
    "CLASS_CODES",
]