"""Criterion parser: free-form criterion strings -> metric-semantic quadruples.

Three entry points:

``parse_criterion(text)``
    One criterion string -> :class:`~.model.ParsedCriterion`
    (``metric`` / ``tolerance`` / ``tolerance_kind`` / ``unit`` / ``condition``).

``extract_quadruple(text, y, y_ref)``
    Full ``(y, y_ref, tau, metric+unit+condition)`` quadruple used by the replay
    re-judge (the numeric side is supplied by the artifact, the semantic side by the
    parser).

``parse_goal_criteria(text)``
    Mission/acceptance text -> list of :class:`GoalCriterion`, i.e. the ``>=90%`` /
    ``>=200 条`` / ``12/12`` shapes written in campaign goals.

The parser never raises on malformed input: unrecognised text yields the module level
``UNPARSED`` sentinel and a reason, so a missing tolerance becomes an explicit
``WITHHELD`` downstream rather than a silent pass.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .model import (
    ABSOLUTE,
    LOG_RATIO,
    RELATIVE,
    SIGNED,
    UNPARSED,
    CriterionRecord,
    ParsedCriterion,
    finite,
)

__all__ = [
    "parse_criterion",
    "extract_quadruple",
    "parse_goal_criteria",
    "GoalCriterion",
    "parse_markdown_criteria_table",
    "PATTERNS",
]

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_UNIT = r"(?:[A-Za-z°µ%‰/²³^·]+(?:\s*/\s*[A-Za-z°µ²³^]+)?)?"


def _to_float(text: str) -> float | None:
    try:
        return finite(float(text))
    except (TypeError, ValueError):
        return None


def _pct_to_fraction(text: str) -> float | None:
    value = _to_float(text)
    return None if value is None else value / 100.0


#: Ordered pattern table.  First match wins, so the more specific semantics (log ratio,
#: signed deviation, dimensionless ratio) are tested before generic absolute forms.
PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    # name, regex, metric, tolerance_kind
    (
        "log_ratio_band",
        r"\|\s*ln\s*\([^)]*\)\s*\|\s*(?:<=|≤|<)\s*ln\s*\(\s*1\s*\+\s*(" + _NUM + r")\s*\)",
        LOG_RATIO,
        "fraction",
    ),
    (
        "signed_keyword",
        r"(带符号|符号一致|sign\s*(?:must\s*)?(?:match|preserved)|signed\s*(?:error|deviation))",
        SIGNED,
        "unknown",
    ),
    (
        "quantity_ratio",
        r"\|\s*[^|]{1,40}\|\s*/\s*(?:\|[^|]{1,24}\||[^<≤\s|]{1,24})[^0-9<≤]{0,12}(?:<=|≤|<)\s*(" + _NUM + r")(?![0-9])\s*(" + _UNIT + r")?",
        RELATIVE,
        "fraction",
    ),
    (
        "dimensionless_ratio",
        r"\|\s*Δ[^|]{0,24}\|\s*/\s*[A-Za-zΣσ][^<≤]{0,24}?(?:<=|≤|<)\s*(" + _NUM + r")",
        RELATIVE,
        "fraction",
    ),
    (
        "relative_percent",
        r"(?:相对误差|相对偏差|relative\s*(?:error|l2|L2|difference)|rel[-_. ]?\\s*(?:error|err|l2|L2|difference)"
        r"|rel-L2|max\s*rel|残差|residual|误差|error|偏差|deviation|drift)"
        r"[^0-9<≤]{0,32}?(?:<=|≤|<)\s*(" + _NUM + r")\s*%",
        RELATIVE,
        "fraction",
    ),
    (
        "relative_fraction",
        r"(?:相对误差|相对偏差|relative\s*error|rel[-_. ]?\\s*(?:error|err|l2|L2|difference)|rel-L2|max\s*rel|残差|residual|相对)"
        r"[^0-9<≤]{0,32}?(?:<=|≤|<)\s*(" + _NUM + r")(?![0-9%])",
        RELATIVE,
        "fraction",
    ),
    (
        "absolute_with_unit",
        r"(?:max\s*)?\|[^|]{1,40}\|[^0-9<≤]{0,24}?(?:<=|≤|<)\s*(" + _NUM + r")\s*(" + _UNIT + r")",
        ABSOLUTE,
        "absolute",
    ),
    (
        "absolute_generic",
        r"(?:eps|eps_max|ε|Δ|diff|偏差|偏差量)[^0-9<≤]{0,24}?(?:<=|≤|<)\s*(" + _NUM + r")\s*(" + _UNIT + r")",
        ABSOLUTE,
        "absolute",
    ),
)

_COMPILED = tuple((name, re.compile(rx, re.IGNORECASE), metric, kind) for name, rx, metric, kind in PATTERNS)

_CONDITION_RX = re.compile(r"(?:在|under|at|when|条件)[:：]?\s*([^,，;；]{2,40})", re.IGNORECASE)


def parse_criterion(text: str | None) -> ParsedCriterion:
    """Parse one criterion string into metric semantics.

    >>> parse_criterion("相对误差 < 2%").tolerance
    0.02
    >>> parse_criterion("max|A-B| <= 1e-12").tolerance_kind
    'absolute'
    >>> parse_criterion("|ln(y/y_ref)| <= ln(1+0.03)").metric
    'log_ratio'
    >>> parse_criterion("???") is UNPARSED or parse_criterion("???").metric
    'unparsed'
    """
    raw = "" if text is None else str(text).strip()
    if not raw:
        return UNPARSED
    for name, rx, metric, kind in _COMPILED:
        m = rx.search(raw)
        if not m:
            continue
        tol_text = m.group(1) if m.groups() else None
        if metric == SIGNED:
            tol = None
            kind = "unknown"
        elif kind == "fraction" and "%" in m.group(0):
            tol = _pct_to_fraction(tol_text or "")
        else:
            tol = _to_float(tol_text or "")
        unit = None
        if m.lastindex and m.lastindex >= 2:
            unit = (m.group(2) or "").strip() or None
        cond = None
        mc = _CONDITION_RX.search(raw)
        if mc:
            cond = mc.group(1).strip() or None
        return ParsedCriterion(
            raw=raw,
            metric=metric,
            tolerance=tol,
            tolerance_kind=kind,
            unit=unit,
            condition=cond,
            pattern=name,
        )
    return UNPARSED


def extract_quadruple(
    text: str | None,
    y: Any = None,
    y_ref: Any = None,
    *,
    scale: Any = None,
    noise: Any = None,
    measurand: str = "",
    source: str = "",
    rid: str = "",
    reported_error: Any = None,
    family: str | None = None,
) -> CriterionRecord:
    """Assemble the full quadruple ``(y, y_ref, tau, metric+unit+condition)``."""
    parsed = parse_criterion(text)
    return CriterionRecord(
        rid=rid or f"{source}::{measurand}::{parsed.metric}",
        source=source,
        measurand=measurand or (text or "").strip()[:60],
        criterion=(text or "").strip(),
        parsed=parsed,
        y=finite(y),
        y_ref=finite(y_ref),
        reported_error=finite(reported_error),
        unit=parsed.unit,
        scale=finite(scale),
        noise=finite(noise),
        family=family,
    )


# --------------------------------------------------------------------------------------
# goal / acceptance text
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GoalCriterion:
    """A single acceptance sentence reduced to (comparand, comparator, value, unit)."""

    text: str
    subject: str
    comparator: str  # ">=" | "<=" | "==" | "range"
    value: float | None
    unit: str | None
    kind: str  # "rate" | "count" | "ratio_case" | "bool" | "samples" | "fraction"
    num: float | None = None
    den: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "subject": self.subject,
            "comparator": self.comparator,
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind,
            "num": self.num,
            "den": self.den,
        }


_BULLET_RX = re.compile(r"^\s*(?:[-*•]|\([a-f]\)|（[a-f]）)\s*(.+)$")
_RATIO_CASE_RX = re.compile(r"(\d+)\s*/\s*(\d+)")
_COUNT_RX = re.compile(r"(?:>=|≥|>)\s*(\d+)\s*(条|个|例|项|samples?|cases?|records?)")
_RATE_RX = re.compile(r"(?:>=|≥|>)\s*(" + _NUM + r")\s*%")
_NUM_CMP_RX = re.compile(r"(<=|≤|<|>=|≥|>)\s*(" + _NUM + r")")
_ERROR_RX = re.compile(r"(误差|error|偏差|deviation|一致率|命中|agreement|accuracy|recall)")


def parse_goal_criteria(text: str | None) -> list[GoalCriterion]:
    """Extract acceptance criteria from mission/goal prose.

    Handles the shapes actually used in campaign goals: percentages (``≥90%``), corpus
    sizes (``≥200 条``), hit ratios (``12/12``), bounded errors (``误差 <3%``) and
    sample counts (``≥1000 个随机样本``).
    """
    if not text:
        return []
    out: list[GoalCriterion] = []
    for line in str(text).splitlines():
        m = _BULLET_RX.match(line) or re.match(r"^\s*(\([a-f]\)|[（(][a-f][)）])?\s*(.+)$", line)
        body = (m.group(m.lastindex) if m else line).strip()
        if not body:
            continue
        rate = _RATE_RX.search(body)
        count = _COUNT_RX.search(body)
        ratio = _RATIO_CASE_RX.search(body)
        numcmp = _NUM_CMP_RX.search(body)
        err = _ERROR_RX.search(body)
        if ratio and ("命中" in body or "良" in body or "样本" in body or "case" in body.lower() or "退化" in body):
            out.append(
                GoalCriterion(body, body[:40], "==", float(ratio.group(1)) / max(1, int(ratio.group(2))),
                              None, "ratio_case", float(ratio.group(1)), float(ratio.group(2)))
            )
            continue
        if count and ("样本" in body or "sample" in body.lower()):
            out.append(GoalCriterion(body, body[:40], ">=", float(count.group(1)), count.group(2), "samples"))
            continue
        if rate:
            out.append(GoalCriterion(body, body[:40], ">=", _pct_to_fraction(rate.group(1)), "%", "rate"))
            continue
        if count:
            out.append(GoalCriterion(body, body[:40], ">=", float(count.group(1)), count.group(2), "count"))
            continue
        if numcmp and err:
            cmp_txt = numcmp.group(1)
            unit = "%" if body[numcmp.end():numcmp.end() + 1] == "%" or body[max(0, numcmp.start() - 12):numcmp.start()].rstrip().endswith("%") else None
            val = _pct_to_fraction(numcmp.group(2)) if unit == "%" else _to_float(numcmp.group(2))
            out.append(
                GoalCriterion(body, body[:40], "<=" if cmp_txt in ("<=", "≤", "<") else ">=", val, unit, "fraction")
            )
            continue
        if numcmp:
            out.append(
                GoalCriterion(body, body[:40], numcmp.group(1), _to_float(numcmp.group(2)), None, "fraction")
            )
    return out


# --------------------------------------------------------------------------------------
# markdown criterion tables (six-element report section (6))
# --------------------------------------------------------------------------------------
_TABLE_SEP_RX = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_TABLE_ROW_RX = re.compile(r"^\s*\|(.+)\|\s*$")


def parse_markdown_criteria_table(markdown: str) -> list[dict[str, str]]:
    """Extract rows from markdown criterion tables.

    A row is accepted when the line starts and ends with ``|`` and the header contains a
    criterion-ish word (判据/阈值/标准/criterion/threshold) or the section title mentions
    误差判定.  Returns the raw cell tuples plus the header, so the caller keeps the
    provenance and decides which column is the measured value.
    """
    rows: list[dict[str, str]] = []
    lines = (markdown or "").splitlines()
    header: list[str] | None = None
    for i, line in enumerate(lines):
        m = _TABLE_ROW_RX.match(line)
        if not m:
            if header is not None and line.strip() and not line.strip().startswith("|"):
                header = None
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if header is None:
            joined = " ".join(cells).lower()
            if any(k in joined for k in ("判据", "阈值", "标准", "criterion", "threshold", "验收")):
                header = cells
            continue
        if _TABLE_SEP_RX.match(line):
            continue
        if len(cells) == len(header):
            rows.append({"header": "|".join(header), "row": "|".join(cells), "cells": cells, "line": str(i)})
    return rows


def iter_bullets(text: str) -> Iterable[str]:
    for line in (text or "").splitlines():
        m = _BULLET_RX.match(line)
        if m:
            yield m.group(1).strip()