"""Structural evidence extraction for FAIL records.

A FAIL record carries free prose plus whatever structured fields the gate
happened to persist.  Rather than trusting either channel alone, this module
reduces a record to a fixed, domain-neutral vector of **signal flags**:

``reference_lexeme``      the text mentions an independent yardstick
``absence_lexeme``        the text asserts something is missing
``artifact_lexeme``       the text mentions deliverables / artefacts
``withheld_lexeme``       the text marks the verdict as undischargeable
``threshold_lexeme``      the text mentions a tolerance / criterion / band
``quantified_gap``        a numeric margin or ratio is reported
``residual_magnitude``    a scientific-notation residual / iteration count
``convergence_lexeme``    iteration, refinement, residual, order vocabulary
``assumption_lexeme``     assumption, idealisation, intrinsic-limit vocabulary
``envelope_lexeme``       extrapolation / validity-range vocabulary
``defect_lexeme``         defect, first-version, patch vocabulary

Lexemes live in the external cue file (``data/cue_lexicon.json``) so that no
domain noun is ever embedded in this code.  All numbers are matched by generic
regular expressions; nothing here knows what the numbers physically mean.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

SIGNAL_KEYS: tuple[str, ...] = (
    "reference_lexeme",
    "absence_lexeme",
    "artifact_lexeme",
    "withheld_lexeme",
    "threshold_lexeme",
    "quantified_gap",
    "residual_magnitude",
    "convergence_lexeme",
    "assumption_lexeme",
    "envelope_lexeme",
    "defect_lexeme",
)

_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%|±|\d+(?:\.\d+)?\s*倍")
_SCI = re.compile(r"\d+(?:\.\d+)?\s*[eE][-+]?\d+|\d+(?:\.\d+)?\s*×\s*10")
_ITER = re.compile(r"\b\d+\s*(?:次迭代|iterations?|iter)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Evidence:
    """Structured evidence of a single FAIL record."""

    signals: Mapping[str, bool]
    numeric_tokens: tuple[str, ...] = ()

    def present(self) -> tuple[str, ...]:
        """Signal keys that fired, in canonical order."""
        return tuple(k for k in SIGNAL_KEYS if self.signals.get(k))

    def as_dict(self) -> dict[str, Any]:
        return {
            "signals": {k: bool(self.signals.get(k)) for k in SIGNAL_KEYS},
            "numeric_tokens": list(self.numeric_tokens),
        }

    def has(self, key: str) -> bool:
        return bool(self.signals.get(key))


def _match_any(text: str, lexemes: list[str]) -> bool:
    low = text.lower()
    return any(lex in low for lex in lexemes)


def extract(body: str, lexicons: Mapping[str, list[str]]) -> Evidence:
    """Reduce *body* to an :class:`Evidence` vector.

    ``lexicons`` maps signal key -> list of lowercase cue lexemes; it is loaded
    from external data so the code stays domain-neutral.
    """
    text = body or ""
    signals: dict[str, bool] = {}
    for key in SIGNAL_KEYS:
        if key == "quantified_gap":
            signals[key] = bool(_PERCENT.search(text))
        elif key == "residual_magnitude":
            signals[key] = bool(_SCI.search(text) or _ITER.search(text))
        else:
            signals[key] = _match_any(text, lexicons.get(key, []))

    tokens = tuple(
        dict.fromkeys(
            m.group(0) for m in (_PERCENT.finditer(text) or [])
        )
    ) + tuple(dict.fromkeys(m.group(0) for m in _SCI.finditer(text)))
    return Evidence(signals=signals, numeric_tokens=tokens)


__all__ = ["Evidence", "SIGNAL_KEYS", "extract"]