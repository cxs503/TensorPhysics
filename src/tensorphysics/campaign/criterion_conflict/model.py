"""First-class criterion objects (quantity / unit / statistic / threshold / direction / provenance).

The dataclasses are deliberately domain-free: every domain term enters through
``ConflictConfig`` (see ``config.py``), never through code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _short(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class SourceRef:
    """Where a criterion came from: mission, campaign, repo, field, line, span."""

    mission_id: str
    campaign: str
    repo: str
    field: str  # "goal" | "report"
    line: int  # 1-based line number inside the field text
    char_start: int  # character offset of the clause inside the line
    char_end: int
    snippet: str

    @classmethod
    def make(cls, mission_id: str, campaign: str, repo: str, field: str,
             line: int, char_start: int, char_end: int, snippet: str) -> "SourceRef":
        return cls(mission_id=str(mission_id), campaign=str(campaign or ""),
                   repo=str(repo or ""), field=str(field), line=int(line),
                   char_start=int(char_start), char_end=int(char_end),
                   snippet=_short(snippet))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRef":
        return cls(**data)


@dataclass(frozen=True)
class Criterion:
    """A machine-checkable acceptance criterion.

    ``quantity`` / ``unit`` / ``statistic`` / ``direction`` are *canonical*
    values produced by the normalisation engine; ``*_raw`` keep the verbatim
    tokens for audit.  ``status`` is fail-closed: only ``mapped`` criteria are
    eligible for conflict detection.
    """

    criterion_id: str
    quantity: Optional[str]
    quantity_raw: Optional[str]
    unit: Optional[str]
    unit_raw: Optional[str]
    statistic: Optional[str]
    statistic_raw: Optional[str]
    threshold: Optional[float]
    threshold_normalized: Optional[float]
    comparator: Optional[str]
    direction: Optional[str]  # "upper" | "lower" | "equal" | "range" | None
    status: str  # "mapped" | "unmapped_quantity" | "unmapped_unit"
    source: SourceRef

    def canonical(self) -> dict[str, Any]:
        """Stable, timestamp-free projection used for hashing and comparisons."""
        def _round(x: Optional[float]) -> Optional[float]:
            return None if x is None else round(float(x), 12)

        return {
            "criterion_id": self.criterion_id,
            "quantity": self.quantity,
            "statistic": self.statistic,
            "unit": self.unit,
            "comparator": self.comparator,
            "direction": self.direction,
            "threshold": _round(self.threshold),
            "threshold_normalized": _round(self.threshold_normalized),
            "status": self.status,
            "source": self.source.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Criterion":
        payload = dict(data)
        payload["source"] = SourceRef.from_dict(payload["source"])
        return cls(**payload)


def make_criterion_id(source: SourceRef, quantity: Optional[str],
                      statistic: Optional[str], comparator: Optional[str],
                      threshold: Optional[float], unit: Optional[str]) -> str:
    """Deterministic criterion id: same inputs -> same id, across processes."""
    payload = "|".join([
        source.mission_id, source.field, str(source.line),
        str(source.char_start), str(source.char_end),
        quantity or "", statistic or "", comparator or "",
        "" if threshold is None else f"{float(threshold):.12g}", unit or "",
    ])
    return "crit_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
