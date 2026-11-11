"""Normalisation engine: unit conversion + statistic labels + comparator direction.

All tables come from the injected :class:`ConflictConfig`; this module owns
only structural logic (no domain vocabulary).
"""
from __future__ import annotations

import math
from typing import Optional

from .config import ConflictConfig, MetricSpec

DIRECTION = {
    "le": "upper", "lt": "upper", "ge": "lower", "gt": "lower",
    "eq": "equal", "ne": "equal", "in_range": "range",
}

COMPARATOR_SYMBOL = {
    "le": "≤", "lt": "<", "ge": "≥", "gt": ">", "eq": "=", "in_range": "∈",
}


def normalize_comparator(comparator: Optional[str]) -> Optional[str]:
    """Map structural comparator token to a comparison direction."""
    if comparator is None:
        return None
    return DIRECTION.get(str(comparator).strip().lower())


def normalize_unit(unit_raw: Optional[str], metric: MetricSpec) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Return (canonical_unit_token, factor_to_canonical, effective_raw_unit).

    A missing unit falls back to the metric's ``default_unit`` (declared in the
    injected config) so that bare numbers are interpreted consistently.  An
    unknown unit is *not* guessed: the caller marks the criterion
    ``unmapped_unit`` and it never reaches conflict detection.
    """
    effective = unit_raw if unit_raw not in (None, "") else metric.default_unit
    factor = metric.factor(effective)
    if factor is None:
        return None, None, unit_raw
    return str(effective), factor, effective


def normalize_threshold(threshold: Optional[float], factor: Optional[float]) -> Optional[float]:
    if threshold is None or factor is None:
        return None
    value = float(threshold) * float(factor)
    return value if math.isfinite(value) else None


def normalize_statistic(statistic_raw: Optional[str], config: ConflictConfig,
                        alias_table: Optional[list[tuple[str, str]]] = None) -> str:
    """Resolve a statistic alias; unresolved -> config.default_statistic."""
    if statistic_raw:
        for alias, name in (alias_table if alias_table is not None else config.statistic_table()):
            if alias.lower() == str(statistic_raw).strip().lower():
                return name
    return config.default_statistic


def tolerance_for(metric_id: Optional[str], config: ConflictConfig) -> float:
    if metric_id is None:
        return config.default_tolerance_relative
    return config.tolerance(metric_id)
