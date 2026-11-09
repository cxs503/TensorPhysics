"""Structured criterion manifests with pre-registration (freeze) and hashing.

A campaign node's acceptance criterion is parsed into six *structural* fields
(no domain semantics), serialised to canonical JSON and hashed.  The hash is
the pre-registration anchor: an auditor re-extracts the criterion from the
delivered report, and any field-level change is classified as drift
(see ``drift.py``) instead of silently passing.

Extraction contract (fixed, structural only)
-------------------------------------------
``threshold``      the numeric level attached to a comparator.  When several
                   numbers appear (e.g. "on 24 cases recall >= 0.92") the level
                   is the number *anchored on the comparator*, not the first
                   number of the sentence.  For an interval ("2-3 pct",
                   "between 10 and 20 ms") the upper bound is used together
                   with comparator ``in_range``.
``unit``           the token immediately after the level ("%" / "pct" / "mm"),
                   filtered by a structural stop-word list.
``comparator``     ``le, lt, ge, gt, eq, ne, in_range`` (symbols, unicode
                   symbols and natural-language phrasings are all mapped).
``denominator``    text after ``of`` / ``relative to`` / ``compared with`` / ``vs``.
``exclusion``      text after ``excluding`` / ``except`` / ``minus``.
``reference_path`` a ``/path/like`` token or the value after ``ref:`` / ``source:``.

Fail-closed: a block without a numeric level yields **no** criterion, so it can
never be frozen as "registered".
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


# The six canonical fields (structural only, zero domain semantics)
CRITERION_FIELDS = frozenset({
    "threshold",
    "unit",
    "denominator",
    "exclusion",
    "reference_path",
    "comparator",
})


@dataclass(frozen=True)
class CriterionField:
    """A single field in a criterion manifest."""
    name: str
    value: Any
    present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "present": self.present}


@dataclass
class CriterionManifest:
    """Structured representation of one acceptance criterion.

    Attributes:
        criterion_id: unique identifier for this criterion
        fields: dict mapping field name → CriterionField
        status: 'registered' | 'unregistered' | 'tampered'
        frozen_at: ISO timestamp of freeze moment
        frozen_by: identifier of who/what froze this
        sha256: hash of canonical JSON (for tamper detection)
    """
    criterion_id: str
    fields: dict[str, CriterionField] = field(default_factory=dict)
    status: str = "unregistered"
    frozen_at: Optional[str] = None
    frozen_by: Optional[str] = None
    sha256: Optional[str] = None

    def to_canonical_json(self) -> str:
        """Produce canonical JSON for hashing (sorted keys, no whitespace)."""
        canonical = {}
        for fname in sorted(CRITERION_FIELDS):
            if fname in self.fields and self.fields[fname].present:
                canonical[fname] = self.fields[fname].value
            else:
                canonical[fname] = None
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Full dict representation including metadata."""
        return {
            "criterion_id": self.criterion_id,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "status": self.status,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionManifest":
        """Reconstruct from dict."""
        fields = {}
        for fname, fdata in data.get("fields", {}).items():
            fields[fname] = CriterionField(
                name=fdata["name"],
                value=fdata["value"],
                present=fdata.get("present", True),
            )
        return cls(
            criterion_id=data["criterion_id"],
            fields=fields,
            status=data.get("status", "unregistered"),
            frozen_at=data.get("frozen_at"),
            frozen_by=data.get("frozen_by"),
            sha256=data.get("sha256"),
        )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# A number: optional sign, thousands separators, decimals, scientific notation.
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)

# Range: "2-3", "10 to 20", "10–20" (en dash).
_RANGE_RE = re.compile(
    r"(?P<lo>" + _NUMBER_RE.pattern + r")\s*(?:-|–|—|to|and)\s*(?P<hi>"
    + _NUMBER_RE.pattern + r")"
)

# Comparators, ordered from most specific to least specific.  Each rule is
# tried at every position; the earliest match wins (ties -> earlier rule).
_COMPARATOR_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"<=|≤|⩽"), "le"),
    (re.compile(r">=|≥|⩾"), "ge"),
    (re.compile(r"!=|≠"), "ne"),
    (re.compile(
        r"\b(?:at\s+most|no\s+more\s+than|no\s+greater\s+than|not\s+exceeding"
        r"|no\s+larger\s+than|at\s+or\s+below|not\s+more\s+than)\b", re.I), "le"),
    (re.compile(
        r"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than|at\s+or\s+above"
        r"|no\s+fewer\s+than|not\s+fewer\s+than)\b", re.I), "ge"),
    (re.compile(r"\b(?:within|in\s+the\s+range|between)\b|±", re.I), "in_range"),
    (re.compile(r"<(?![=<])"), "lt"),
    (re.compile(r">(?![=>])"), "gt"),
    (re.compile(r"=="), "eq"),
    (re.compile(r"\b(?:equals?|equal\s+to|exactly)\b", re.I), "eq"),
)

# Structural stop words that must never be treated as a unit token.
_UNIT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "between", "by",
    "compared", "during", "each", "except", "excluding", "for", "from",
    "in", "including", "into", "is", "it", "must", "no", "not", "of", "on",
    "or", "over", "per", "relative", "should", "than", "that", "the", "then",
    "to", "under", "vs", "was", "were", "with", "within", "without",
})

_UNIT_MAX_LEN = 12


def _clean_value(value: str) -> str:
    """Clean extracted value: strip whitespace and trailing punctuation."""
    if not value:
        return value
    # Strip trailing punctuation that might be from sentence structure
    value = value.strip()
    value = value.rstrip(".,;:)]}")
    value = value.lstrip("([({")
    return value.strip()


def _to_float(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_numeric(text: str) -> Optional[float]:
    """Extract first numeric value from text (kept for backwards compatibility)."""
    match = _NUMBER_RE.search(text or "")
    if match:
        return _to_float(match.group())
    return None


def _find_comparator(text: str) -> tuple[Optional[str], int, int]:
    """Return (name, start, end) of the earliest comparator occurrence."""
    best: Optional[tuple[int, int, str, int]] = None
    for order, (pattern, name) in enumerate(_COMPARATOR_RULES):
        m = pattern.search(text)
        if not m:
            continue
        candidate = (m.start(), order, name, m.end())
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None, -1, -1
    return best[2], best[0], best[3]


def _extract_comparator(text: str) -> Optional[str]:
    """Extract comparison operator from text (symbols or natural language)."""
    name, _, _ = _find_comparator(text or "")
    return name


def _unit_at(text: str, end: int) -> Optional[str]:
    """Unit token immediately following a level, or None."""
    rest = text[end:]
    m = re.match(r"\s*(%|‰|°[A-Za-z]?|[A-Za-zµμ][A-Za-z0-9µμ_%°/\.\-]*)", rest)
    if not m:
        return None
    token = m.group(1).rstrip("./-")
    if not token or len(token) > _UNIT_MAX_LEN:
        return None
    if token.lower() in _UNIT_STOPWORDS:
        return None
    return token


def _extract_unit(text: str) -> Optional[str]:
    """Unit token associated with the first number (compatibility helper)."""
    m = _NUMBER_RE.search(text or "")
    if not m:
        return None
    return _unit_at(text, m.end())


def _extract_level(text: str) -> tuple[Optional[float], Optional[str], int]:
    """Return (level, comparator, end_offset) with comparator anchoring.

    Range expressions ("2-3 pct", "between 10 and 20 ms") use the upper bound
    and comparator ``in_range``.
    """
    comparator, c_start, c_end = _find_comparator(text)
    has_explicit = comparator is not None

    # Range first (only when no explicit inequality phrasing precedes it).
    if not has_explicit:
        rng = _RANGE_RE.search(text)
        if rng:
            return _to_float(rng.group("hi")), "in_range", rng.end("hi")

    if comparator == "in_range":
        # ±X / within X
        plus_minus = re.search(r"±\s*(" + _NUMBER_RE.pattern + r")", text)
        if plus_minus:
            return _to_float(plus_minus.group(1)), comparator, plus_minus.end(1)
        between = re.search(
            r"\bbetween\s+(" + _NUMBER_RE.pattern + r")\s*(?:and|-|–|to)\s*("
            + _NUMBER_RE.pattern + r")", text, re.I)
        if between:
            return _to_float(between.group(2)), comparator, between.end(2)

    search_from = c_end if has_explicit else 0
    m = _NUMBER_RE.search(text, search_from)
    if m is None:
        m = _NUMBER_RE.search(text)
    if m is None:
        return None, comparator, -1
    return _to_float(m.group()), comparator, m.end()


_DENOM_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:relative\s+to|compared\s+with|compared\s+to|vs\.?|w\.r\.t\.?)\b\s+(.+)", re.I),
    re.compile(r"\bof\b\s+(?![\d.+-])(.+)", re.I),
)


def _extract_denominator(text: str) -> Optional[str]:
    tail = None
    for pattern in _DENOM_MARKERS:
        m = pattern.search(text)
        if m:
            tail = m.group(1)
            break
    if tail is None:
        return None
    tail = re.split(r"[,;(:]|\b(?:excluding|except|minus)\b", tail, maxsplit=1)[0]
    tail = _clean_value(tail)
    return tail or None


def _extract_exclusion(text: str) -> Optional[str]:
    m = re.search(r"(?:exclud(?:e|ing)|except|not\s+including|minus)\s+([^,;.(]+)",
                  text, re.I)
    if not m:
        return None
    return _clean_value(m.group(1)) or None


def _extract_reference_path(text: str) -> Optional[str]:
    m = re.search(r"(?:ref(?:erence)?|baseline|source)\s*[:=]\s*([^\s,;)]+)", text, re.I)
    if m:
        return _clean_value(m.group(1)) or None
    m = re.search(r"(?<![\w:/])/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*", text)
    if m:
        return m.group(0)
    return None


def extract_criteria_from_goal(goal_text: str) -> list[CriterionManifest]:
    """Extract structured criteria from a mission goal text.

    Each numbered / bulleted block becomes at most one criterion.  A block is
    only materialised when a numeric level is found; otherwise it stays
    unregistered (fail-closed).

    Args:
        goal_text: The mission goal text (may contain multiple criteria)

    Returns:
        List of CriterionManifest objects. Empty list if no criteria found.
        Each criterion has status='unregistered' until freeze_manifest() is called.
    """
    if not goal_text or not isinstance(goal_text, str):
        return []

    criteria: list[CriterionManifest] = []

    # Split into criterion blocks by looking for numbered items or bullet points
    # Patterns: "1.", "- ", "* ", "C1:", "criterion:", etc.
    block_pattern = r"(?:^|\n)\s*(?:\d{1,2}[\.\)](?=[ \t])|[-*•](?=[ \t])|[Cc]\d+[:\.](?=[ \t]))\s*"
    blocks = re.split(block_pattern, goal_text)

    # If no blocks found, treat entire text as one criterion
    if len(blocks) <= 1:
        blocks = [goal_text]

    for idx, block in enumerate(blocks):
        block = block.strip()
        if not block or len(block) < 3:
            continue

        threshold, comparator, level_end = _extract_level(block)
        # Fail-closed: no numeric level -> this block is not a criterion.
        if threshold is None:
            continue

        fields: dict[str, CriterionField] = {
            "threshold": CriterionField("threshold", threshold, True),
        }

        unit = _unit_at(block, level_end) if level_end >= 0 else None
        if unit:
            fields["unit"] = CriterionField("unit", unit, True)

        if comparator:
            fields["comparator"] = CriterionField("comparator", comparator, True)

        denominator = _extract_denominator(block)
        if denominator:
            fields["denominator"] = CriterionField("denominator", denominator, True)

        exclusion = _extract_exclusion(block)
        if exclusion:
            fields["exclusion"] = CriterionField("exclusion", exclusion, True)

        reference_path = _extract_reference_path(block)
        if reference_path:
            fields["reference_path"] = CriterionField("reference_path", reference_path, True)

        criteria.append(CriterionManifest(
            criterion_id=f"criterion_{idx:03d}",
            fields=fields,
            status="unregistered",
        ))

    return criteria


def freeze_manifest(
    manifest: CriterionManifest,
    frozen_by: str,
    frozen_at: Optional[Any] = None,
) -> CriterionManifest:
    """Freeze a criterion manifest with sha256 hash.

    Args:
        manifest: The criterion to freeze
        frozen_by: Identifier of who/what is freezing (e.g., agent id, timestamp)
        frozen_at: Optional timestamp (datetime, or an ISO-8601 string kept
            as-is so that a freeze can be reproduced across processes;
            defaults to now UTC)

    Returns:
        New CriterionManifest with status='registered', sha256 set, frozen_at/by set.
    """
    if frozen_at is None:
        frozen_at = datetime.now(timezone.utc)

    if isinstance(frozen_at, datetime):
        frozen_at_str = frozen_at.isoformat()
    else:
        frozen_at_str = str(frozen_at)

    # Compute sha256 of canonical JSON
    canonical = manifest.to_canonical_json()
    sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # Create frozen copy
    frozen = CriterionManifest(
        criterion_id=manifest.criterion_id,
        fields=dict(manifest.fields),
        status="registered",
        frozen_at=frozen_at_str,
        frozen_by=frozen_by,
        sha256=sha256,
    )
    return frozen


def verify_frozen_hash(manifest: CriterionManifest) -> bool:
    """Verify that a frozen manifest's hash matches its content.

    Args:
        manifest: A frozen CriterionManifest

    Returns:
        True if hash matches, False if tampered or not frozen.
    """
    if manifest.status != "registered" or manifest.sha256 is None:
        return False

    canonical = manifest.to_canonical_json()
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return computed == manifest.sha256