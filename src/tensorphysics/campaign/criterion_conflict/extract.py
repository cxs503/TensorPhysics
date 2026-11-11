"""Criterion extractor: text -> :class:`Criterion` objects with provenance.

Every emitted criterion carries ``mission_id + field + line + char span +
verbatim snippet``.  Fail-closed: a clause whose quantity is not in the
injected vocabulary is kept as ``unmapped_quantity`` evidence but never enters
the registry or conflict detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .config import ConflictConfig, load_config
from .model import Criterion, SourceRef, make_criterion_id
from .normalize import normalize_comparator, normalize_statistic, normalize_threshold, normalize_unit

_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
_NUM_RE = re.compile(_NUM)
_UNIT_RE = re.compile(r"\s*[\[\(（]?\s*(%|‰|个百分点|[A-Za-zµμ][A-Za-z0-9µμ_/\.\-]*)")
MAX_ALIAS_DISTANCE = 12  # chars: a quantity must be named near its comparator
_UNIT_STOP = frozenset({
    "a", "an", "and", "as", "at", "be", "by", "cases", "each", "for", "from",
    "in", "is", "items", "must", "not", "of", "on", "or", "per", "should",
    "than", "the", "to", "vs", "with", "within",
})

# Structural comparator tokens (symbols + natural-language forms).  This table
# carries no domain semantics.
_COMPARATORS: tuple[tuple[str, str], ...] = (
    ("<=", "le"), ("≥", "ge"), ("≤", "le"), (">=", "ge"), ("≦", "le"), ("≧", "ge"),
    ("==", "eq"), ("＜=", "le"), ("＞=", "ge"), ("＜", "lt"), ("＞", "gt"),
    ("不超过", "le"), ("不大于", "le"), ("不高于", "le"), ("至多", "le"),
    ("不少于", "ge"), ("不低于", "ge"), ("不小于", "ge"), ("至少", "ge"),
    ("at most", "le"), ("at least", "ge"), ("no more than", "le"),
    ("no less than", "ge"), ("not more than", "le"), ("not less than", "ge"),
    ("not exceeding", "le"), ("±", "in_range"),
    ("小于", "lt"), ("大于", "gt"), ("<", "lt"), (">", "gt"), ("=", "eq"),
)
_CMP_RE = re.compile("|".join(re.escape(tok) for tok, _ in sorted(_COMPARATORS, key=lambda p: -len(p[0]))))
_CMP_MAP = dict(_COMPARATORS)
_SEG_SPLIT = re.compile(r"(?<!\d)[,，](?!\d)|[;；。|+]|(?<!\d)(?:且|并且|以及)(?!\d)")


@dataclass
class ExtractionResult:
    criteria: list[Criterion] = field(default_factory=list)
    residue: list[dict[str, Any]] = field(default_factory=list)
    segments_scanned: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": [c.to_dict() for c in self.criteria],
            "residue": list(self.residue),
            "segments_scanned": self.segments_scanned,
        }


def _iter_segments(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield (segment_text, 1-based line, char offset inside the line)."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        start = 0
        for m in _SEG_SPLIT.finditer(line):
            yield line[start:m.start()], line_no, start
            start = m.end()
        yield line[start:], line_no, start


def _compile_alias_table(table: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str, str]]:
    """Compile aliases with word boundaries for ASCII tokens.

    Boundary guards prevent sub-string mismatches such as an alias like
    ``ratio`` matching inside ``iteration`` — a real precision defect found
    on the historical corpus.
    """
    compiled: list[tuple[re.Pattern[str], str, str]] = []
    for alias, value in table:
        pattern = re.escape(alias)
        if re.match(r"[A-Za-z0-9_]", alias):
            pattern = r"(?<![A-Za-z0-9_])" + pattern
        if re.search(r"[A-Za-z0-9_]$", alias):
            pattern = pattern + r"(?![A-Za-z0-9_])"
        compiled.append((re.compile(pattern, re.IGNORECASE), alias, value))
    return compiled


def _distance(match: re.Match[str], anchor: int) -> int:
    """Distance from a match to the comparator anchor (0 if it straddles it)."""
    if match.end() <= anchor:
        return anchor - match.end()
    if match.start() >= anchor:
        return match.start() - anchor
    return 0


def _best_alias(segment: str, table: list[tuple[re.Pattern[str], str, str]],
                anchor: int, lo: int, hi: int) -> tuple[Optional[str], Optional[str], int, int]:
    """Closest alias **before** the comparator, else shortly after it.

    Quantity names in acceptance clauses precede the operator ("error < 5%");
    an alias after the operator is only accepted when it is adjacent, which
    keeps unrelated prose ("ka >= 10 ... error <= 1 dB") from claiming the
    threshold.
    """
    before: Optional[tuple[int, int, int, str, str]] = None
    after: Optional[tuple[int, int, int, str, str]] = None
    for regex, alias, value in table:
        last = None
        for match in regex.finditer(segment, lo, anchor):
            last = match
        if last is not None:
            candidate = (anchor - last.end(), -len(alias), last.start(), alias, value)
            if before is None or candidate < before:
                before = candidate
        else:
            match = regex.search(segment, anchor, hi)
            if match is not None and _distance(match, anchor) <= MAX_ALIAS_DISTANCE:
                candidate = (_distance(match, anchor), -len(alias), match.start(), alias, value)
                if after is None or candidate < after:
                    after = candidate
    best = before if before is not None else after
    if best is None:
        return None, None, -1, -1
    _, _, idx, alias, value = best
    return value, alias, idx, idx + len(alias)


def _find_metric(segment: str, table: list[tuple[re.Pattern[str], str, str]],
                 anchor: int, lo: int, hi: int) -> tuple[Optional[str], Optional[str], int, int]:
    return _best_alias(segment, table, anchor, lo, hi)


def _find_statistic(segment: str, table: list[tuple[re.Pattern[str], str, str]],
                    anchor: int, lo: int, hi: int) -> tuple[Optional[str], Optional[str]]:
    value, alias, _, _ = _best_alias(segment, table, anchor, lo, hi)
    return value, alias


def _number_near(segment: str, start: int, end: int) -> tuple[Optional[float], int, int]:
    """Number bound to a comparator: prefer the token right after it."""
    tail = segment[end:]
    m = re.match(r"\s*(?:为|达到|达|是|:|：)?\s*(" + _NUM + ")", tail)
    if m:
        try:
            return float(m.group(1).replace(",", "")), end + m.start(1), end + m.end(1)
        except ValueError:
            pass
    head = segment[:start]
    matches = list(_NUM_RE.finditer(head))
    if matches and (start - matches[-1].end()) <= 20:
        m = matches[-1]
        try:
            return float(m.group(0).replace(",", "")), m.start(), m.end()
        except ValueError:
            return None, -1, -1
    return None, -1, -1


def _unit_after(segment: str, number_end: int, tokens: tuple[str, ...] = ()) -> Optional[str]:
    for token in tokens:
        if not token:
            continue
        if segment.startswith(token, number_end):
            return token
        for opener in ("[", "(", "（"):
            if segment.startswith(opener + token, number_end) or \
                    segment.startswith(opener + " " + token, number_end):
                return token
    m = _UNIT_RE.match(segment, number_end)
    if not m:
        return None
    token = m.group(1)
    if not token or len(token) > 12 or token.lower() in _UNIT_STOP:
        return None
    return token


def extract_criteria(text: str, source: SourceRef, config: "ConflictConfig | dict | str") -> ExtractionResult:
    """Extract all criteria from one text field (goal or report)."""
    cfg = load_config(config)
    metric_table = _compile_alias_table(cfg.alias_table())
    stat_table = _compile_alias_table(cfg.statistic_table())
    unit_tokens = cfg.unit_tokens()
    markers = tuple(m.lower() for m in cfg.exclusion_markers)
    result = ExtractionResult()
    if not text:
        return result

    for segment, line_no, offset in _iter_segments(text):
        if not segment.strip():
            continue
        if markers and any(marker in segment.lower() for marker in markers):
            continue
        result.segments_scanned += 1
        cmps = list(_CMP_RE.finditer(segment))
        if not cmps:
            continue
        seen_spans: set[tuple[int, int]] = set()
        for i, cm in enumerate(cmps):
            # clause window: a comparator only claims a quantity named in its own
            # clause (bounded by neighbouring comparators), never a distant alias.
            lo = cmps[i - 1].end() if i > 0 else 0
            hi = cmps[i + 1].start() if i + 1 < len(cmps) else len(segment)
            metric_id, metric_raw, q_start, q_end = _find_metric(segment, metric_table, cm.start(), lo, hi)
            stat_name, stat_raw = _find_statistic(segment, stat_table, cm.start(), lo, hi)
            comparator = _CMP_MAP.get(cm.group(0))
            threshold, n_start, n_end = _number_near(segment, cm.start(), cm.end())
            if threshold is None or (n_start, n_end) in seen_spans:
                continue
            seen_spans.add((n_start, n_end))
            unit_raw = _unit_after(segment, n_end, unit_tokens)
            criterion_source = SourceRef.make(
                mission_id=source.mission_id, campaign=source.campaign,
                repo=source.repo, field=source.field, line=line_no,
                char_start=offset + min(q_start if q_start >= 0 else cm.start(), cm.start()),
                char_end=offset + max(cm.end(), n_end),
                snippet=segment.strip(),
            )
            if metric_id is None:
                result.residue.append({
                    "reason": "unmapped_quantity",
                    "mission_id": source.mission_id,
                    "field": source.field,
                    "line": line_no,
                    "snippet": segment.strip()[:160],
                    "threshold": threshold,
                    "comparator": comparator,
                })
                continue
            metric = cfg.metric(metric_id)
            unit, factor, effective_unit = normalize_unit(unit_raw, metric)
            status = "mapped" if unit is not None else "unmapped_unit"
            normalized = normalize_threshold(threshold, factor)
            if normalized is None:
                status = "unmapped_unit"
            criterion = Criterion(
                criterion_id=make_criterion_id(criterion_source, metric_id, stat_name or cfg.default_statistic,
                                               comparator, threshold, unit),
                quantity=metric_id,
                quantity_raw=metric_raw,
                unit=unit,
                unit_raw=effective_unit,
                statistic=(stat_name if stat_name and cfg.canonical_statistic(stat_name)
                           else cfg.default_statistic),
                statistic_raw=stat_raw,
                threshold=threshold,
                threshold_normalized=normalized,
                comparator=comparator,
                direction=normalize_comparator(comparator),
                status=status,
                source=criterion_source,
            )
            result.criteria.append(criterion)
    return result


def extract_from_entries(entries: Iterable[dict[str, Any]], config: "ConflictConfig") -> list[Criterion]:
    """Extract from corpus entries ``{mission_id, campaign, repo, goal, report?}``."""
    out: list[Criterion] = []
    for entry in entries:
        mission_id = str(entry.get("mission_id", entry.get("id", "")))
        campaign = str(entry.get("campaign") or mission_id)
        repo = str(entry.get("repo", ""))
        for field_name in ("goal", "report"):
            text = entry.get(field_name)
            if not text:
                continue
            source = SourceRef.make(mission_id, campaign, repo, field_name, 0, 0, 0, "")
            out.extend(extract_criteria(text, source, config).criteria)
    return out
