"""Coverage judgement: inherited debt x descendant criterion -> coverage matrix.

Four outcomes (names come from the injected vocabulary, defaults in brackets):

``covered`` [covered]
    a descendant criterion names the debt's quantity (optionally conditioned)
    or cites the debt's report/mission/node/plan id explicitly.
``covered_by_domain`` [covered_by_domain]
    only the *domain* is acknowledged -- a weak promise, kept distinct on
    purpose so that a reviewer can see the difference between "this quantity is
    re-measured" and "somebody works in this area".
``deferred`` [deferred]
    no criterion covers it, but a deferral registry entry states why (a human
    decision, auditable and dated).
``dropped`` [dropped]
    nobody covers it and nobody said why.  This is the only outcome the gate
    refuses to accept.

Rule priority is fixed and tested: ``explicit_ref`` > ``quantity`` >
``domain`` > ``deferral`` > ``dropped``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import VocabConfig
from .criteria import Criterion
from .model import DebtItem

__all__ = ["CoverageEntry", "CoverageDecision", "match_debt", "build_matrix", "resolve_deferral"]


@dataclass(frozen=True)
class CoverageEntry:
    debt_id: str
    key: str
    coverage: str
    rule: str
    matched: tuple[dict, ...] = ()
    deferred_reason: str | None = None
    requires_reason: bool = False

    def as_dict(self) -> dict:
        return {
            "debt_id": self.debt_id,
            "key": self.key,
            "coverage": self.coverage,
            "rule": self.rule,
            "matched": [dict(m) for m in self.matched],
            "deferred_reason": self.deferred_reason,
            "requires_reason": self.requires_reason,
        }


@dataclass(frozen=True)
class CoverageDecision:
    entry: CoverageEntry
    matched_criterion_ids: tuple[str, ...] = ()


def resolve_deferral(debt: DebtItem, deferrals: Mapping[str, str] | None) -> str | None:
    """Look up a deferral reason for a debt.

    Accepted keys (checked in this order): the debt id, the debt key,
    ``node:<id>``, ``mission:<id>``, ``report:<id>``, ``domain:<domain>``.
    Values must be non-empty strings; an empty value is *not* a reason.
    """
    if not deferrals:
        return None
    candidates = [debt.id, debt.key]
    if debt.node_id is not None:
        candidates.append(f"node:{debt.node_id}")
    if debt.mission_id is not None:
        candidates.append(f"mission:{debt.mission_id}")
    if debt.report_id is not None:
        candidates.append(f"report:{debt.report_id}")
    if debt.domain:
        candidates.append(f"domain:{debt.domain}")
    for cand in candidates:
        reason = deferrals.get(cand)
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return None


def _condition_compatible(debt: DebtItem, criterion: Criterion, cfg: VocabConfig) -> bool:
    """Does the criterion mention the debt's case/condition tag?

    Records granularity, it does not veto coverage: a criterion that names the
    quantity without naming the case still promises to re-measure it.  Numeric
    conditions are matched as standalone tokens, so a case labelled ``2`` is not
    "found" inside a threshold such as ``20%``.
    """
    wildcard = cfg.states.get("wildcard_condition", "ALL")
    if debt.condition is None or str(debt.condition).strip().upper() == str(wildcard).upper():
        return True
    tag = str(debt.condition).split()[0]
    if not tag:
        return True
    text = criterion.text or ""
    if not text:
        return True
    if re.fullmatch(r"[0-9.]+", tag):
        return re.search(r"(?<![0-9.])" + re.escape(tag) + r"(?![0-9.%])", text) is not None
    return tag.lower() in text.lower()


def match_debt(debt: DebtItem, criteria: Iterable[Criterion], cfg: VocabConfig,
               deferrals: Mapping[str, str] | None = None) -> CoverageEntry:
    cov = cfg.coverage
    covered = cov.get("covered", "covered")
    by_domain = cov.get("covered_by_domain", "covered_by_domain")
    deferred = cov.get("deferred", "deferred")
    dropped = cov.get("dropped", "dropped")

    crit_list = list(criteria)

    # 1) explicit identifier reference (strongest: the descendant names the evidence)
    ref_hits: list[dict] = []
    for c in crit_list:
        for kind, ids in debt.refs.items():
            shared = set(c.refs.get(kind, ())) & set(ids)
            if shared:
                ref_hits.append({"criterion_id": c.criterion_id, "kind": "ref", "ref_kind": kind,
                                 "ids": sorted(shared), "node_id": c.node_id, "source": c.source})
    if ref_hits:
        return CoverageEntry(debt.id, debt.key, covered, cov.get("rule_explicit_ref", "explicit_ref"),
                             tuple(sorted(ref_hits, key=lambda h: (str(h["criterion_id"]), h["ref_kind"]))))

    # 2) quantity identity (plus condition compatibility)
    if debt.quantity is not None:
        q_hits: list[dict] = []
        for c in crit_list:
            if c.quantity == debt.quantity:
                # the criterion promises to re-measure this quantity; whether it also
                # names the debt's case/condition is recorded, not required
                q_hits.append({"criterion_id": c.criterion_id, "kind": "quantity",
                               "quantity": c.quantity, "threshold_pct": c.threshold_pct,
                               "condition_match": _condition_compatible(debt, c, cfg),
                               "node_id": c.node_id, "source": c.source})
        if q_hits:
            return CoverageEntry(debt.id, debt.key, covered, cov.get("rule_quantity", "quantity"),
                                 tuple(sorted(q_hits, key=lambda h: str(h["criterion_id"]))))

    # 3) domain-only acknowledgement
    d_hits = [{"criterion_id": c.criterion_id, "kind": "domain", "domain": c.domain,
               "node_id": c.node_id, "source": c.source}
              for c in crit_list if debt.domain and c.domain == debt.domain]
    if d_hits:
        return CoverageEntry(debt.id, debt.key, by_domain, cov.get("rule_domain", "domain"),
                             tuple(sorted(d_hits, key=lambda h: str(h["criterion_id"]))))

    # 4) deferral with an explicit reason
    reason = resolve_deferral(debt, deferrals)
    if reason:
        return CoverageEntry(debt.id, debt.key, deferred, cov.get("rule_deferral", "deferral"), (),
                             deferred_reason=reason)

    # 5) nobody covers it, nobody said why -> dropped (fail-closed input)
    return CoverageEntry(debt.id, debt.key, dropped, "none", (), deferred_reason=None, requires_reason=True)


def build_matrix(debts: list[DebtItem], criteria: Iterable[Criterion], cfg: VocabConfig,
                 deferrals: Mapping[str, str] | None = None) -> dict:
    """Coverage matrix + counts, with the decision written back onto each debt."""
    crit_list = list(criteria)
    entries: list[CoverageEntry] = []
    for debt in debts:
        entry = match_debt(debt, crit_list, cfg, deferrals)
        debt.coverage = entry.coverage
        debt.covered_by = [dict(m) for m in entry.matched]
        debt.deferred_reason = entry.deferred_reason
        entries.append(entry)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.coverage] = counts.get(e.coverage, 0) + 1
    rules: dict[str, int] = {}
    for e in entries:
        rules[e.rule] = rules.get(e.rule, 0) + 1
    return {
        "criteria_total": len(crit_list),
        "criteria_nodes": sorted({c.node_id for c in crit_list if c.node_id is not None}),
        "counts": dict(sorted(counts.items())),
        "rules": dict(sorted(rules.items())),
        "entries": [e.as_dict() for e in sorted(entries, key=lambda x: (x.key, x.debt_id))],
        "matrix": [
            {
                "debt_key": e.key,
                "debt_id": e.debt_id,
                "coverage": e.coverage,
                "matched_criteria": sorted(m["criterion_id"] for m in e.matched),
            }
            for e in sorted(entries, key=lambda x: (x.key, x.debt_id))
        ],
    }