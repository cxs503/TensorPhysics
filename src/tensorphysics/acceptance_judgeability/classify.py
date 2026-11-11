"""Clause -> judgeability classification (metrology only, no domain vocabulary).

A criterion is *judgeable* when three things exist at once: a **quantity**, a
**reference** it can miss, and a **tolerance** that decides the miss. Every other
combination is degraded in a specific, named way, because "degraded in some way"
is not actionable -- the census needs to say *which* piece is missing so the
upgrade path can name the piece to add.

The classifier is deliberately shallow (feature detection + an ordered rule set)
instead of a statistical model: an acceptance gate must be explainable clause by
clause to the author of the clause, and it must not depend on training data that
this repository does not own.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_DATA = Path(__file__).with_name("data") / "lexicon_default.json"


class Judgeability(str, Enum):
    """Closed four-class taxonomy of an acceptance clause."""

    JUDGEABLE = "JUDGEABLE"              # quantity + reference + tolerance
    THRESHOLD_ONLY = "THRESHOLD_ONLY"    # quantity + bound, no reference
    RECORD_ONLY = "RECORD_ONLY"          # record / monotonic / non-crash / prose
    WITHHELD_OK = "WITHHELD_OK"          # explicit "data missing" declaration


class Subtype(str, Enum):
    """Why a clause landed in its class -- drives the upgrade path."""

    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"   # JUDGEABLE
    SELF_REFERENCE = "SELF_REFERENCE"           # JUDGEABLE (closure/residual/balance)
    NO_BASELINE = "NO_BASELINE"                 # THRESHOLD_ONLY
    NO_TOLERANCE = "NO_TOLERANCE"               # THRESHOLD_ONLY
    MONOTONIC = "MONOTONIC"                     # RECORD_ONLY
    QUALITATIVE = "QUALITATIVE"                 # RECORD_ONLY
    UNCLASSIFIED = "UNCLASSIFIED"               # RECORD_ONLY, needs review
    DECLARED_MISSING = "DECLARED_MISSING"       # WITHHELD_OK


@dataclass(frozen=True)
class ClauseFeatures:
    """Everything the rule set is allowed to look at; kept for audit."""

    text: str
    n_numbers: int = 0
    has_comparator: bool = False
    has_percent: bool = False
    has_unit: bool = False
    has_tolerance: bool = False
    has_reference: bool = False
    has_self: bool = False
    has_withheld: bool = False
    has_record: bool = False
    has_monotonic: bool = False
    has_arrow: bool = False
    has_interval: bool = False
    has_assertion: bool = False
    has_operator_word: bool = False
    has_rate_word: bool = False
    has_verdict_context: bool = False
    has_count_fraction: bool = False
    hits: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_numbers": self.n_numbers,
            "comparator": self.has_comparator,
            "percent": self.has_percent,
            "unit": self.has_unit,
            "tolerance": self.has_tolerance,
            "reference": self.has_reference,
            "self_reference": self.has_self,
            "withheld": self.has_withheld,
            "record": self.has_record,
            "monotonic": self.has_monotonic,
            "arrow": self.has_arrow,
            "interval": self.has_interval,
            "assertion": self.has_assertion,
            "operator_word": self.has_operator_word,
            "rate_word": self.has_rate_word,
            "verdict_context": self.has_verdict_context,
            "count_fraction": self.has_count_fraction,
            "hits": list(self.hits),
        }


@dataclass(frozen=True)
class ClassifiedClause:
    """One clause with its class, subtype and the evidence that produced them."""

    cid: str
    text: str
    judgeability: Judgeability
    subtype: Subtype
    features: ClauseFeatures
    needs_review: bool = False

    @property
    def contributes_to_support(self) -> bool:
        """Only ``JUDGEABLE`` can ever support a PASS."""
        return self.judgeability is Judgeability.JUDGEABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "text": self.text,
            "judgeability": self.judgeability.value,
            "subtype": self.subtype.value,
            "needs_review": self.needs_review,
            "features": self.features.as_dict(),
        }


# --------------------------------------------------------------------------- lexicon
def _compile(markers: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Split markers into CJK substrings and boundary-anchored Latin ones."""
    patterns: list[re.Pattern[str]] = []
    for m in markers:
        if not m:
            continue
        if re.search(r"[A-Za-z]", m):
            # Latin markers are word-anchored: "rate" must not match inside
            # "generate", and "benchmark" must not match inside "benchmarkcase".
            head = r"(?<![A-Za-z])" if m[0].isalpha() else ""
            tail = r"(?![A-Za-z])" if m[-1].isalpha() else ""
            patterns.append(re.compile(head + re.escape(m) + tail, re.IGNORECASE))
        else:
            patterns.append(re.compile(re.escape(m)))
    return tuple(patterns)


def _compile_regex(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """Compile *raw regular expressions* (unlike :func:`_compile`, no escaping)."""
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns if p)


class Lexicon:
    """Compiled marker sets; replaceable at runtime for a different vocabulary."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = dict(raw)
        self.version = str(raw.get("version", "unversioned"))
        self._comparators = _compile(raw["comparators"])
        self._bounds = _compile(raw["bound_words"])
        self._tolerances = _compile(raw["tolerance_words"])
        self._references = _compile(raw["reference_words"])
        self._withheld = _compile(raw["withheld_words"])
        self._records = _compile(raw["record_words"])
        self._selfs = _compile(raw["self_words"])
        self._monotonic = _compile(raw["monotonic_words"])
        self._arrows = _compile(raw["arrow_markers"])
        self._intervals = _compile(raw.get("interval_markers", ()))
        self._asserts = _compile(raw.get("assert_words", ()))
        self._ops = _compile(raw.get("op_words", ()))
        self._refpats = _compile_regex(raw.get("reference_patterns", ()))
        self._rates = _compile(raw.get("rate_words", ()))
        self._verdicts = _compile(raw.get("verdict_words", ()))
        self._id_prefix = re.compile(raw.get("id_prefix_regex", r"^(?!)"))
        self._count_fraction = re.compile(r"\b\d+\s*/\s*\d+\b")
        self._units = _compile(raw["unit_tokens"])
        self._number = re.compile(raw["number_regex"])
        # "relative error" must not be read as the reference word "relative"
        self._tolerance_phrase = re.compile(
            r"(相对误差|相对偏差|relative\s+error|rel\.?\s*error)", re.IGNORECASE
        )

    def _hit(self, text: str, *groups: Sequence[re.Pattern[str]]) -> list[str]:
        out: list[str] = []
        for group in groups:
            for pat in group:
                if pat.search(text):
                    out.append(pat.pattern)
        return out

    def strip_id_prefix(self, text: str) -> str:
        """Drop a leading criterion id (``C1:``, ``AC6``, ``(a)``, ``3)``).

        The id is bookkeeping, not a quantity: leaving it in makes ``AC6`` contribute
        the number 6 and turns a prose clause into a numeric one.
        """
        return self._id_prefix.sub("", str(text or ""), count=1).strip()

    def features(self, text: str) -> ClauseFeatures:
        t = self.strip_id_prefix(text)
        probe = self._tolerance_phrase.sub(" ", t)  # strip "relative error" phrasing
        numbers = [m.group(0) for m in self._number.finditer(t)]
        hits: list[str] = []
        comparator = bool(self._hit(t, self._comparators))
        if comparator:
            hits.append("comparator")
        bound = bool(self._hit(t, self._bounds))
        if bound:
            hits.append("bound_word")
        tolerance = bool(self._hit(t, self._tolerances))
        if tolerance:
            hits.append("tolerance_word")
        reference = bool(self._hit(probe, self._references)) or bool(
            self._hit(probe, self._refpats)
        )
        if reference:
            hits.append("reference_word")
        selfref = bool(self._hit(t, self._selfs))
        if selfref:
            hits.append("self_word")
        rate_word = bool(self._hit(t, self._rates))
        if rate_word:
            hits.append("rate_word")
        verdict_ctx = bool(self._hit(t, self._verdicts))
        if verdict_ctx:
            hits.append("verdict_context")
        count_fraction = bool(self._count_fraction.search(t))
        if count_fraction:
            hits.append("count_fraction")
        withheld = bool(self._hit(t, self._withheld))
        if withheld:
            hits.append("withheld_word")
        record = bool(self._hit(t, self._records))
        if record:
            hits.append("record_word")
        monotonic = bool(self._hit(t, self._monotonic))
        if monotonic:
            hits.append("monotonic_word")
        interval = bool(self._hit(t, self._intervals)) and bool(
            re.search(r"[\[\u3010(]\s*[-+\u2212]?\d", t) or "\u2208" in t or "\u533a\u95f4" in t
        )
        if interval:
            hits.append("interval")
        op_word = bool(self._hit(t.lower(), self._ops))
        if op_word:
            hits.append("op_word")
        assert_word = bool(self._hit(t, self._asserts))
        if assert_word:
            hits.append("assert_word")
        arrow = bool(self._hit(t, self._arrows))
        if arrow:
            hits.append("arrow")
        unit = bool(self._hit(t, self._units))
        if unit:
            hits.append("unit")
        return ClauseFeatures(
            text=t,
            n_numbers=len(numbers),
            has_comparator=comparator or bound,
            has_percent="%" in t or "pct" in t.lower(),
            has_unit=unit,
            has_tolerance=tolerance,
            has_reference=reference,
            has_self=selfref,
            has_withheld=withheld,
            has_record=record,
            has_monotonic=monotonic,
            has_arrow=arrow,
            has_interval=interval,
            has_assertion=assert_word,
            has_operator_word=op_word,
            has_rate_word=rate_word,
            has_verdict_context=verdict_ctx,
            has_count_fraction=count_fraction,
            hits=tuple(hits),
        )


@lru_cache(maxsize=1)
def _default_lexicon() -> Lexicon:
    return Lexicon(json.loads(_DATA.read_text(encoding="utf-8")))


DEFAULT_LEXICON: Lexicon = _default_lexicon()
_ACTIVE: list[Lexicon] = [DEFAULT_LEXICON]


def get_lexicon() -> Lexicon:
    """The lexicon currently in force (defaults to the packaged metrology one)."""
    return _ACTIVE[0]


def set_lexicon(lexicon: Lexicon | None) -> None:
    """Install/replace the active lexicon (``None`` restores the packaged default)."""
    _ACTIVE[0] = lexicon if lexicon is not None else DEFAULT_LEXICON


def lexicon_from_dict(raw: Mapping[str, Any]) -> Lexicon:
    return Lexicon(raw)


# --------------------------------------------------------------------------- rules
_CLASSIFY_CACHE: dict[tuple[str, str], tuple[Judgeability, Subtype, ClauseFeatures, bool]] = {}


def classify(text: str, *, cid: str = "", lexicon: Lexicon | None = None) -> ClassifiedClause:
    """Classify one acceptance clause.

    Rule order (first match wins, and it is part of the contract):

    R1 explicit missing-data declaration, without a numeric bound -> ``WITHHELD_OK``
    R2 numeric bound + reference (external or self-closure)       -> ``JUDGEABLE``
    R3 numeric bound, no reference                                -> ``THRESHOLD_ONLY``
    R4 arrow/limit relation with a number, no tolerance           -> ``THRESHOLD_ONLY``
    R5 monotonicity requirement                                   -> ``RECORD_ONLY``
    R6 record/declaration/qualitative wording                     -> ``RECORD_ONLY``
    R7 anything left                                              -> ``RECORD_ONLY`` + review
    """
    lx = lexicon or get_lexicon()
    body = " ".join(str(text or "").split())
    key = (f"{id(lx)}:{lx.version}", body)
    cached = _CLASSIFY_CACHE.get(key)
    if cached is not None:
        judge, sub, feats, review = cached
        return ClassifiedClause(cid, body, judge, sub, feats, review)
    out = _classify_uncached(body, cid, lx)
    if len(_CLASSIFY_CACHE) > 100_000:  # bound the cache; corpora are far smaller
        _CLASSIFY_CACHE.clear()
    _CLASSIFY_CACHE[key] = (out.judgeability, out.subtype, out.features, out.needs_review)
    return out


def _classify_uncached(body: str, cid: str, lx: Lexicon) -> ClassifiedClause:
    f = lx.features(body)

    if (
        f.has_withheld
        and not f.has_verdict_context
        and not (f.has_comparator and f.n_numbers)
    ):
        return ClassifiedClause(cid, body, Judgeability.WITHHELD_OK, Subtype.DECLARED_MISSING, f)

    # a numeric bound may be written as a comparator (<= 3 %), an interval (in [a, b])
    # or an assertion with a number (coverage 100 %, 4/4 detected)
    numeric_bound = (
        f.has_comparator
        or f.has_operator_word
        or f.has_interval
        or (f.has_assertion and f.n_numbers > 0)
    ) and f.n_numbers > 0
    tolerance_available = numeric_bound or (f.has_tolerance and f.has_percent)

    self_is_baseline = f.has_self and not (f.has_rate_word or f.has_count_fraction)
    if tolerance_available and (f.has_reference or self_is_baseline):
        subtype = Subtype.EXTERNAL_REFERENCE if f.has_reference else Subtype.SELF_REFERENCE
        return ClassifiedClause(cid, body, Judgeability.JUDGEABLE, subtype, f)

    if tolerance_available:
        return ClassifiedClause(cid, body, Judgeability.THRESHOLD_ONLY, Subtype.NO_BASELINE, f)

    if f.has_arrow and f.n_numbers > 0:
        return ClassifiedClause(cid, body, Judgeability.THRESHOLD_ONLY, Subtype.NO_TOLERANCE, f)

    if f.has_monotonic:
        return ClassifiedClause(cid, body, Judgeability.RECORD_ONLY, Subtype.MONOTONIC, f)

    if f.has_record or f.has_reference or f.has_self or f.has_tolerance:
        return ClassifiedClause(cid, body, Judgeability.RECORD_ONLY, Subtype.QUALITATIVE, f)

    return ClassifiedClause(
        cid, body, Judgeability.RECORD_ONLY, Subtype.UNCLASSIFIED, f, needs_review=True
    )


def classify_all(
    clauses: Iterable[Mapping[str, str]] | Iterable[tuple[str, str]],
    *,
    lexicon: Lexicon | None = None,
) -> list[ClassifiedClause]:
    """Classify ``[{cid, text}]`` / ``[(cid, text)]`` in order."""
    out: list[ClassifiedClause] = []
    for item in clauses:
        if isinstance(item, Mapping):
            cid = str(item.get("cid", ""))
            text = str(item.get("text", ""))
        elif hasattr(item, "cid") and hasattr(item, "text"):
            cid, text = str(item.cid), str(item.text)
        else:
            cid, text = (str(x) for x in item)
        out.append(classify(text, cid=cid, lexicon=lexicon))
    return out