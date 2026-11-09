"""Injectable configuration for the lineage-inheritance debt ledger.

The ledger must be **domain-free logic + domain-bearing data**: every domain
name, quantity alias, table-header token, verdict marker and deny-list term is
read from a JSON vocabulary file (``configs/lineage_debt.vocab.json`` by
default, or any file/dict passed by the caller).  Swapping the vocabulary is
therefore enough to move the ledger to a different engineering domain, and the
``selfcheck`` module can prove that no domain word leaked into the logic.

Nothing in this module interprets the vocabulary; it only validates it and
exposes lookups (``canonical_quantity``, ``domain_of``, ``threshold_for``,
``as_dict``) so that the rest of the package never has to guess.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "VocabConfig",
    "load_config",
]

CONFIG_SCHEMA = "lineage_debt_vocab.v1"

#: Shipped vocabulary (package data; override with ``load_config(path)``).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "lineage_debt.vocab.json"


def _norm(text: Any) -> str:
    """Lower-case, strip markdown emphasis and squeeze whitespace.

    Underscores are **kept**: they are part of the symbol vocabulary
    (``C_T`` vs ``C_A`` vs ``T_opt``), so turning them into spaces would let a
    one-letter alias fire inside an unrelated symbol.
    """
    s = "" if text is None else str(text)
    s = s.replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass(frozen=True)
class QuantitySpec:
    key: str
    aliases: tuple[str, ...]
    unit: str = "-"


@dataclass(frozen=True)
class DomainSpec:
    key: str
    aliases: tuple[str, ...]
    quantities: Mapping[str, QuantitySpec]

    def match_quantity(self, text: str) -> str | None:
        """Canonical quantity whose alias occurs in ``text`` (longest alias wins).

        Aliases are matched with ASCII word boundaries so that one-letter
        symbols (e.g. a single-letter factor) do not fire inside other words.
        """
        best: tuple[int, str] | None = None
        hay = _norm(text)
        for qkey, spec in self.quantities.items():
            for alias in spec.aliases:
                a = _norm(alias)
                if not a:
                    continue
                if re.match(r"^[a-z0-9_+\-/ɵηφσθωμ²°·. ]+$", a):
                    # short symbol aliases must not fire inside ratios/products
                    # such as "a_t/l" or "c_t/c_f"
                    fence = r"[a-z0-9_/.\-]" if len(a) <= 2 else r"[a-z0-9_]"
                    pat = "(?<!" + fence + ")" + re.escape(a) + "(?!" + fence + ")"
                    hit = re.search(pat, hay) is not None
                else:
                    hit = a in hay
                if hit and (best is None or len(a) > best[0]):
                    best = (len(a), qkey)
        return None if best is None else best[1]


@dataclass(frozen=True)
class VocabConfig:
    """Parsed vocabulary + contracts + thresholds (all caller-injectable)."""

    schema: str
    raw: Mapping[str, Any]
    domains: Mapping[str, DomainSpec]
    thresholds: Mapping[str, Any]
    table_contract: Mapping[str, Any]
    criteria_contract: Mapping[str, Any]
    states: Mapping[str, str]
    coverage: Mapping[str, str]
    deny_list: tuple[str, ...] = field(default=())

    # -- lookups ---------------------------------------------------------
    def canonical_quantity(self, raw_label: str) -> tuple[str, str]:
        """Map a raw label to ``(domain_key, quantity_key)``.

        Falls back to ``(unclassified, cleaned_raw_label)`` so that an
        unknown label is *reported*, never silently dropped.
        """
        label = _norm(raw_label)
        best: tuple[int, str, str] | None = None
        for dkey, spec in self.domains.items():
            q = spec.match_quantity(label)
            if q is None:
                continue
            # prefer the longest matched alias -> most specific quantity
            for alias in spec.quantities[q].aliases:
                a = _norm(alias)
                if a and a in label and (best is None or len(a) > best[0]):
                    best = (len(a), dkey, q)
        if best is not None:
            return best[1], best[2]
        return self.states.get("unclassified_domain", "unclassified"), re.sub(r"\s+", " ", str(raw_label)).strip()

    def domain_by_alias(self, text: str) -> str | None:
        hay = _norm(text)
        best: tuple[int, str] | None = None
        for dkey, spec in self.domains.items():
            for alias in spec.aliases:
                a = _norm(alias)
                if not a:
                    continue
                if re.fullmatch(r"[a-z0-9_+\-/ ]+", a):
                    hit = re.search(r"(?<![a-z0-9_])" + re.escape(a) + r"(?![a-z0-9_])", hay) is not None
                else:
                    hit = a in hay
                if hit and (best is None or len(a) > best[0]):
                    best = (len(a), dkey)
        return None if best is None else best[1]

    def threshold_for(self, domain: str | None, quantity: str | None) -> float:
        """Resolution order: per-quantity > per-domain > default."""
        by_q = {str(k): float(v) for k, v in (self.thresholds.get("by_quantity") or {}).items()}
        by_d = {str(k): float(v) for k, v in (self.thresholds.get("by_domain") or {}).items()}
        if quantity is not None and quantity in by_q:
            return by_q[quantity]
        if domain is not None and domain in by_d:
            return by_d[domain]
        return float(self.thresholds.get("default_pct", 15.0))

    # -- vocabulary accessors -------------------------------------------
    @property
    def error_header_patterns(self) -> tuple[str, ...]:
        return tuple(self.table_contract.get("error_header_patterns") or ())

    @property
    def quantity_headers(self) -> tuple[str, ...]:
        return tuple(_norm(h) for h in (self.table_contract.get("quantity_headers") or ()))

    @property
    def condition_headers(self) -> tuple[str, ...]:
        return tuple(_norm(h) for h in (self.table_contract.get("condition_headers") or ()))

    @property
    def empty_markers(self) -> tuple[str, ...]:
        return tuple(_norm(m) for m in (self.table_contract.get("empty_markers") or ()))

    @property
    def fail_markers(self) -> tuple[str, ...]:
        return tuple(_norm(m) for m in ((self.table_contract.get("verdict_markers") or {}).get("fail") or ()))

    def is_error_header(self, header: str) -> bool:
        h = _norm(header)
        return any(_norm(p) in h for p in self.error_header_patterns)

    def is_empty(self, cell: str) -> bool:
        return _norm(cell) in self.empty_markers

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "domains": sorted(self.domains),
            "quantities": {d: sorted(s.quantities) for d, s in sorted(self.domains.items())},
            "thresholds": dict(self.thresholds),
            "deny_list_size": len(self.deny_list),
        }


def _build(spec: Mapping[str, Any]) -> VocabConfig:
    schema = str(spec.get("schema") or CONFIG_SCHEMA)
    domains: dict[str, DomainSpec] = {}
    for dkey, dval in (spec.get("domains") or {}).items():
        qs: dict[str, QuantitySpec] = {}
        for qkey, qval in (dval.get("quantities") or {}).items():
            qs[qkey] = QuantitySpec(
                key=qkey,
                aliases=tuple(qval.get("aliases") or (qkey,)),
                unit=str(qval.get("unit") or "-"),
            )
        domains[dkey] = DomainSpec(
            key=dkey,
            aliases=tuple(dval.get("aliases") or (dkey,)) + (dkey,),
            quantities=qs,
        )
    deny = tuple(str(w) for w in (spec.get("deny_list") or ()))
    return VocabConfig(
        schema=schema,
        raw=spec,
        domains=domains,
        thresholds=dict(spec.get("thresholds") or {"default_pct": 15.0}),
        table_contract=dict(spec.get("table_contract") or {}),
        criteria_contract=dict(spec.get("criteria_contract") or {}),
        states=dict(spec.get("states") or {"open": "debt_open", "closed": "debt_closed"}),
        coverage=dict(spec.get("coverage") or {}),
        deny_list=deny,
    )


def load_config(source: Mapping[str, Any] | str | Path | None = None) -> VocabConfig:
    """Load a vocabulary from a dict, a JSON path, or the shipped default."""
    if source is None:
        source = DEFAULT_CONFIG_PATH
    if isinstance(source, Mapping):
        return _build(source)
    path = Path(source)
    with path.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)
    if str(spec.get("schema") or "") != CONFIG_SCHEMA:
        raise ValueError(f"unexpected vocabulary schema: {spec.get('schema')!r}")
    return _build(spec)


def iter_state_names(cfg: VocabConfig) -> Iterable[str]:
    yield from (cfg.states.get("open", "debt_open"), cfg.states.get("closed", "debt_closed"))