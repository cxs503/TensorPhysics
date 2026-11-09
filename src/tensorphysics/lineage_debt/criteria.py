"""Child-node acceptance criteria: structured metadata first, regex fallback.

A *criterion* is what a descendant node promises to check.  The ledger needs it
in machine form to decide whether an inherited debt is genuinely covered.

Resolution order (documented contract, tested):

1. **structured metadata** -- ``node["metadata"]["acceptance_criteria"]`` (or
   ``node["metadata"]["criteria"]``) may hold a list whose entries are dicts::

       {"quantity": "q1", "domain": "<domain>", "threshold_pct": 10.0,
        "refs": {"report": [9001]}, "text": "..."}

   Dict entries are trusted verbatim; unknown keys are ignored.
2. **string entries** (the campaign's de-facto format, a JSON-encoded list of
   sentences) are parsed with the configured threshold patterns and reference
   patterns -- quantity aliases are canonicalised through the injected
   vocabulary, never hard-coded.
3. **free text** (``metadata["goal"]``) is parsed with the same regexes so a
   node that never filled ``acceptance_criteria`` still yields criteria (the
   source is recorded as ``goal_regex`` so the coverage matrix shows which
   promises came from prose).

Criteria without a parseable threshold are still useful: they carry quantity
and domain identity, which is what coverage matching consumes.  A criterion
that yields neither quantity nor domain is recorded as ``unparsed`` -- never
silently dropped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .config import VocabConfig

__all__ = ["Criterion", "parse_node_criteria", "parse_criteria_list"]


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    plan_id: int | None
    node_id: int | None
    mission_id: int | None
    text: str
    source: str                     # "metadata" | "metadata_string" | "goal_regex" | "metadata_unparsed"
    domain: str | None = None
    quantity: str | None = None
    threshold_pct: float | None = None
    refs: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "mission_id": self.mission_id,
            "text": self.text,
            "source": self.source,
            "domain": self.domain,
            "quantity": self.quantity,
            "threshold_pct": None if self.threshold_pct is None else round(float(self.threshold_pct), 6),
            "refs": {k: sorted(int(v) for v in vals) for k, vals in sorted(self.refs.items())},
        }


def _int_list(values: Any) -> tuple[int, ...]:
    out: list[int] = []
    if values is None:
        return ()
    if isinstance(values, (int, float, str)):
        values = [values]
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(out)))


def _refs_from_text(text: str, cfg: VocabConfig) -> dict[str, tuple[int, ...]]:
    refs: dict[str, set[int]] = {}
    patterns = (cfg.criteria_contract.get("ref_patterns") or {})
    for kind, pats in patterns.items():
        for pat in pats or ():
            for m in re.finditer(pat, text):
                try:
                    refs.setdefault(kind, set()).add(int(m.group("v")))
                except (IndexError, TypeError, ValueError):
                    continue
    return {k: tuple(sorted(v)) for k, v in sorted(refs.items())}


def _threshold_from_text(text: str, cfg: VocabConfig) -> tuple[str | None, float | None]:
    """Return ``(raw_quantity_label, threshold_pct)`` from a criterion sentence."""
    for pat in (cfg.criteria_contract.get("threshold_patterns") or ()):
        for m in re.finditer(pat, text):
            raw = (m.group("qty") or "").strip(" ，,;；:：()（）")
            if not raw:
                continue
            # strip leading prose such as "判据(a) " / "- " / "G1 "
            raw = re.sub(r"^[\(\（]?[a-zA-Z]?\d?[\)\）]?\s*", "", raw) if re.match(r"^[\(\（]", raw) else raw
            try:
                val = float(m.group("val"))
            except (TypeError, ValueError):
                continue
            return raw, val
    return None, None


def _criterion_from_text(text: str, cfg: VocabConfig, *, plan_id: int | None, node_id: int | None,
                         mission_id: int | None, source: str, ordinal: int) -> Criterion:
    raw_qty, threshold = _threshold_from_text(text, cfg)
    domain: str | None = None
    quantity: str | None = None
    if raw_qty:
        domain, quantity = cfg.canonical_quantity(raw_qty)
        if domain == cfg.states.get("unclassified_domain", "unclassified"):
            domain = None
    if domain is None:
        domain = cfg.domain_by_alias(text)
    refs = _refs_from_text(text, cfg)
    cid = f"n{node_id}:{source}:{ordinal}"
    return Criterion(criterion_id=cid, plan_id=plan_id, node_id=node_id, mission_id=mission_id,
                     text=text.strip(), source=source, domain=domain, quantity=quantity,
                     threshold_pct=threshold, refs=refs)


def parse_criteria_list(entries: Iterable[Any], cfg: VocabConfig, *, plan_id: int | None = None,
                        node_id: int | None = None, mission_id: int | None = None,
                        source: str = "metadata_string") -> list[Criterion]:
    out: list[Criterion] = []
    for i, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            text = str(entry.get("text") or entry.get("criterion") or "").strip()
            raw_qty = entry.get("quantity")
            domain = entry.get("domain")
            quantity: str | None = None
            if raw_qty is not None:
                d2, quantity = cfg.canonical_quantity(str(raw_qty))
                domain = domain or (None if d2 == cfg.states.get("unclassified_domain", "unclassified") else d2)
            refs: dict[str, tuple[int, ...]] = {}
            for kind, vals in (entry.get("refs") or {}).items():
                got = _int_list(vals)
                if got:
                    refs[str(kind)] = got
            for kind, vals in _refs_from_text(text, cfg).items():
                refs[kind] = tuple(sorted(set(refs.get(kind, ())) | set(vals)))
            thr = entry.get("threshold_pct", entry.get("threshold"))
            try:
                thr_f = None if thr is None else float(thr)
            except (TypeError, ValueError):
                thr_f = None
            out.append(Criterion(
                criterion_id=str(entry.get("id") or f"n{node_id}:{source}:{i}"),
                plan_id=plan_id, node_id=node_id, mission_id=mission_id,
                text=text or json.dumps(entry, ensure_ascii=False, sort_keys=True),
                source="metadata", domain=domain if domain else cfg.domain_by_alias(text),
                quantity=quantity, threshold_pct=thr_f, refs=refs,
            ))
        else:
            text = str(entry)
            out.append(_criterion_from_text(text, cfg, plan_id=plan_id, node_id=node_id,
                                            mission_id=mission_id, source=source, ordinal=i))
    return out


def _coerce_entries(raw: Any) -> list[Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [line.strip(" -*\t") for line in text.splitlines() if line.strip(" -*\t")]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, Mapping):
            return [parsed]
        return [parsed]
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, Iterable):
        return list(raw)
    return None


def parse_node_criteria(node: Mapping[str, Any], cfg: VocabConfig) -> list[Criterion]:
    """All machine-readable criteria promised by one campaign node."""
    meta = node.get("metadata") or {}
    plan_id = int(node["plan_id"]) if node.get("plan_id") is not None else None
    node_id = int(node["id"]) if node.get("id") is not None else None
    mission_id = int(node["mission_id"]) if node.get("mission_id") is not None else None

    out: list[Criterion] = []
    for field_name in ("acceptance_criteria", "criteria"):
        entries = _coerce_entries(meta.get(field_name))
        if entries:
            out.extend(parse_criteria_list(entries, cfg, plan_id=plan_id, node_id=node_id,
                                           mission_id=mission_id, source="metadata_string"))
    goal = meta.get("goal")
    if goal:
        # only machine-judgeable fragments of the prose: those carrying a '<=x%' bound
        for i, sentence in enumerate(re.split(r"[；;。\n]", str(goal))):
            if _threshold_from_text(sentence, cfg)[1] is None:
                continue
            out.append(_criterion_from_text(sentence.strip(), cfg, plan_id=plan_id, node_id=node_id,
                                            mission_id=mission_id, source="goal_regex", ordinal=i))
    name = node.get("name")
    if not out and name:
        out.append(Criterion(criterion_id=f"n{node_id}:title", plan_id=plan_id, node_id=node_id,
                             mission_id=mission_id, text=str(name), source="metadata_unparsed",
                             domain=cfg.domain_by_alias(str(name)), quantity=None,
                             threshold_pct=None, refs={}))
    # stable ordering + de-duplication on (text, source, threshold)
    uniq: dict[tuple[str, str, float | None], Criterion] = {}
    for c in out:
        uniq.setdefault((c.text, c.source, c.threshold_pct), c)
    return [uniq[k] for k in sorted(uniq, key=lambda k: (k[1], k[0], k[2] if k[2] is not None else -1))]