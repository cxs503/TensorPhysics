"""Injected vocabulary + tolerance bands for criterion normalisation.

Zero domain semantics live in this module: metric ids, aliases, units and
tolerance bands all come from a JSON config supplied by the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .model import canonical_json

CONFIG_SCHEMA = "criterion-conflict-config/1.0"


class ConfigError(ValueError):
    """Raised when the injected configuration is inconsistent (fail-closed)."""


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    aliases: tuple[str, ...]
    canonical_unit: str
    units: dict[str, float]
    default_unit: Optional[str] = None
    tolerance_relative: float = 0.10
    dimension: str = "dimensionless"

    def factor(self, unit: Optional[str]) -> Optional[float]:
        if unit is None:
            return None
        for token, factor in self.units.items():
            if token.lower() == str(unit).strip().lower():
                return float(factor)
        return None


@dataclass(frozen=True)
class StatisticSpec:
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ConflictConfig:
    schema_version: str
    metrics: tuple[MetricSpec, ...]
    statistics: tuple[StatisticSpec, ...]
    default_statistic: str = "unspecified"
    default_tolerance_relative: float = 0.10
    exclusion_markers: tuple[str, ...] = ()

    # ---- lookups -------------------------------------------------------
    def metric(self, metric_id: str) -> MetricSpec:
        for m in self.metrics:
            if m.metric_id == metric_id:
                return m
        raise ConfigError(f"unknown metric id: {metric_id}")

    def tolerance(self, metric_id: str) -> float:
        for m in self.metrics:
            if m.metric_id == metric_id:
                return m.tolerance_relative
        return self.default_tolerance_relative

    def alias_table(self) -> list[tuple[str, str]]:
        """(alias, metric_id) pairs, longest alias first for greedy matching."""
        pairs: list[tuple[str, str]] = []
        for m in self.metrics:
            for alias in m.aliases:
                pairs.append((alias, m.metric_id))
        pairs.sort(key=lambda p: (-len(p[0]), p[0]))
        return pairs

    def statistic_table(self) -> list[tuple[str, str]]:
        pairs = [(alias, s.name) for s in self.statistics for alias in s.aliases]
        pairs.sort(key=lambda p: (-len(p[0]), p[0]))
        return pairs

    def unit_tokens(self) -> tuple[str, ...]:
        tokens: set[str] = set()
        for m in self.metrics:
            tokens.update(m.units)
        return tuple(sorted(tokens, key=len, reverse=True))

    def canonical_statistic(self, name: str) -> Optional[str]:
        for s in self.statistics:
            if s.name == name:
                return s.name
        return None

    # ---- hashing / IO ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "default_statistic": self.default_statistic,
            "default_tolerance_relative": self.default_tolerance_relative,
            "exclusion_markers": list(self.exclusion_markers),
            "statistics": [{"name": s.name, "aliases": list(s.aliases)} for s in self.statistics],
            "metrics": [
                {
                    "id": m.metric_id,
                    "aliases": list(m.aliases),
                    "canonical_unit": m.canonical_unit,
                    "units": dict(sorted(m.units.items())),
                    "default_unit": m.default_unit,
                    "tolerance_relative": m.tolerance_relative,
                    "dimension": m.dimension,
                }
                for m in self.metrics
            ],
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictConfig":
        if not isinstance(data, dict):
            raise ConfigError("config must be a JSON object")
        version = data.get("schema_version", CONFIG_SCHEMA)
        if version != CONFIG_SCHEMA:
            raise ConfigError(f"unsupported schema_version: {version!r}")
        stats_raw = data.get("statistics", [])
        stats = tuple(
            StatisticSpec(name=str(s["name"]), aliases=tuple(str(a) for a in s.get("aliases", [])))
            for s in stats_raw
        )
        metrics_raw = data.get("metrics", [])
        if not metrics_raw:
            raise ConfigError("config must define at least one metric")
        metrics: list[MetricSpec] = []
        for m in metrics_raw:
            mid = str(m.get("id") or m.get("metric_id") or "").strip()
            if not mid:
                raise ConfigError("metric entry without id")
            aliases = tuple(str(a) for a in m.get("aliases", []) if str(a).strip())
            if not aliases:
                raise ConfigError(f"metric {mid!r} has no aliases")
            units = {str(k): float(v) for k, v in dict(m.get("units", {})).items()}
            if not units:
                raise ConfigError(f"metric {mid!r} has no unit table")
            canonical_unit = str(m.get("canonical_unit") or "")
            if not canonical_unit:
                raise ConfigError(f"metric {mid!r} has no canonical_unit")
            tol = float(m.get("tolerance_relative", data.get("default_tolerance_relative", 0.10)))
            if not math.isfinite(tol) or tol < 0:
                raise ConfigError(f"metric {mid!r} has invalid tolerance_relative")
            default_unit = m.get("default_unit")
            if default_unit is not None and default_unit not in units:
                raise ConfigError(f"metric {mid!r} default_unit not in unit table")
            metrics.append(MetricSpec(
                metric_id=mid, aliases=aliases, canonical_unit=canonical_unit,
                units=units, default_unit=default_unit, tolerance_relative=tol,
                dimension=str(m.get("dimension", "dimensionless")),
            ))
        # cross-metric alias collisions make extraction ambiguous -> fail closed
        seen: dict[str, str] = {}
        for m in metrics:
            for alias in m.aliases:
                key = alias.lower()
                if key in seen and seen[key] != m.metric_id:
                    raise ConfigError(f"alias {alias!r} shared by {seen[key]!r} and {m.metric_id!r}")
                seen[key] = m.metric_id
        default_stat = str(data.get("default_statistic", "unspecified"))
        if stats and default_stat not in {s.name for s in stats}:
            raise ConfigError(f"default_statistic {default_stat!r} not declared")
        markers = tuple(str(m) for m in data.get("exclusion_markers", []) if str(m).strip())
        return cls(
            schema_version=version, metrics=tuple(metrics), statistics=stats,
            default_statistic=default_stat,
            default_tolerance_relative=float(data.get("default_tolerance_relative", 0.10)),
            exclusion_markers=markers,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ConflictConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def load_config(source: "ConflictConfig | dict[str, Any] | str | Path") -> ConflictConfig:
    if isinstance(source, ConflictConfig):
        return source
    if isinstance(source, dict):
        return ConflictConfig.from_dict(source)
    return ConflictConfig.from_path(source)


def leak_terms(config: ConflictConfig) -> list[str]:
    """All textual aliases (used by the domain-leak scan)."""
    terms: list[str] = []
    for m in config.metrics:
        terms.extend(m.aliases)
    for s in config.statistics:
        terms.extend(s.aliases)
    return terms
