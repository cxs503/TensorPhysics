"""Cross-campaign conflict detection over normalised criteria.

Rule (structural): two criteria *conflict* when they share
``(canonical quantity, statistic, comparison direction)`` and their normalised
thresholds differ by more than the configured relative tolerance band.  Only
criteria backed by different campaigns are compared (same-campaign variation is
not a cross-campaign conflict).

Determinism: conflicts are ordered by (quantity, statistic, direction,
conflict_id); the registry hash never depends on dict iteration order.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .config import ConflictConfig
from .model import Criterion, canonical_json


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    quantity: str
    statistic: str
    direction: str
    tolerance_relative: float
    delta_relative: float
    left: Criterion
    right: Criterion

    @property
    def severity(self) -> float:
        """How far beyond the tolerance band (1.0 = exactly at the band edge)."""
        if self.tolerance_relative <= 0:
            return math.inf
        return self.delta_relative / self.tolerance_relative

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "quantity": self.quantity,
            "statistic": self.statistic,
            "direction": self.direction,
            "tolerance_relative": round(self.tolerance_relative, 12),
            "delta_relative": round(self.delta_relative, 12),
            "severity": round(self.severity, 6) if math.isfinite(self.severity) else None,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


def _campaign_of(criterion: Criterion) -> str:
    return criterion.source.campaign or criterion.source.mission_id


def _relative_gap(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0 else abs(a - b) / denom


def _conflict_id(quantity: str, statistic: str, direction: str,
                 left: Criterion, right: Criterion) -> str:
    ids = sorted([left.criterion_id, right.criterion_id])
    payload = "|".join([quantity, statistic, direction, ids[0], ids[1]])
    return "cf_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cluster(values: list[tuple[float, Criterion]], tol: float) -> list[list[tuple[float, Criterion]]]:
    """Single-linkage clustering of thresholds using the relative tolerance band."""
    clusters: list[list[tuple[float, Criterion]]] = []
    for value, crit in values:
        if clusters:
            prev = clusters[-1][-1][0]
            if _relative_gap(prev, value) <= tol:
                clusters[-1].append((value, crit))
                continue
        clusters.append([(value, crit)])
    return clusters


def detect_conflicts(criteria: Iterable[Criterion], config: ConflictConfig,
                     same_campaign_ok: bool = False) -> list[Conflict]:
    """Detect normalised threshold conflicts (deterministic, cluster-based)."""
    groups: dict[tuple[str, str, str], list[Criterion]] = {}
    for crit in criteria:
        if crit.status != "mapped":
            continue
        if crit.threshold_normalized is None or crit.quantity is None:
            continue
        if crit.direction not in ("upper", "lower"):
            continue  # equality / range criteria do not express a one-sided band
        key = (crit.quantity, crit.statistic or config.default_statistic, crit.direction)
        groups.setdefault(key, []).append(crit)

    conflicts: list[Conflict] = []
    for (quantity, statistic, direction), items in groups.items():
        tol = config.tolerance(quantity)
        # collapse duplicates (same campaign + mission + threshold) to keep the
        # pairwise search linear in the number of distinct claims.
        unique: dict[tuple[str, str, float], Criterion] = {}
        for c in items:
            key = (_campaign_of(c), c.source.mission_id, round(float(c.threshold_normalized), 12))
            unique.setdefault(key, c)
        ordered = sorted(((float(c.threshold_normalized), c) for c in unique.values()),
                         key=lambda p: (p[0], p[1].criterion_id))
        clusters = _cluster(ordered, tol)
        if len(clusters) < 2:
            continue
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # representative pair: extreme values of the two clusters
                lo_value, lo_crit = clusters[i][0]
                hi_value, hi_crit = clusters[j][-1]
                delta = _relative_gap(lo_value, hi_value)
                if delta <= tol:
                    continue
                candidates = [(lo_crit, hi_crit), (lo_crit, clusters[j][0][1]),
                              (clusters[i][-1][1], hi_crit)]
                pair = next(
                    ((a, b) for a, b in candidates if _campaign_of(a) != _campaign_of(b)),
                    None,
                )
                if pair is None and not same_campaign_ok:
                    continue
                left, right = pair if pair is not None else (lo_crit, hi_crit)
                if _campaign_of(left) == _campaign_of(right) and not same_campaign_ok:
                    continue
                conflicts.append(Conflict(
                    conflict_id=_conflict_id(quantity, statistic, direction, left, right),
                    quantity=quantity, statistic=statistic, direction=direction,
                    tolerance_relative=tol,
                    delta_relative=delta,
                    left=left, right=right,
                ))
    conflicts.sort(key=lambda c: (c.quantity, c.statistic, c.direction,
                                  -c.delta_relative, c.conflict_id))
    return conflicts


def conflicts_digest(conflicts: Iterable[Conflict]) -> str:
    payload = [c.to_dict() for c in conflicts]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
