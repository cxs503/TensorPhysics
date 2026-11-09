"""Parameter axes of a capability's applicability envelope.

An axis is a *single scalar parameter* of a capability plus the span the
probing operator is allowed to explore.  Nothing in this module knows what the
axis means physically: names, spans, scales and the expected monotonicity law
are supplied by configuration (``capability.py``), never hard-coded here.

Two sampling lattices are offered:

``geometric``
    Chebyshev-Lobatto spacing in scale coordinates,
    ``u_i = (1 - cos(pi i / (N-1))) / 2``.  The step grows geometrically away
    from both ends of the span, so the *geometric gradient scan* of the probing
    operator puts most samples where a boundary is most likely to sit (the ends
    of the declared span) while still covering the interior.

``uniform``
    ``u_i = i / (N-1)`` in scale coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["AxisSpec", "AxisConfigError", "SCALES", "DISTRIBUTIONS", "MONOTONE_LAWS"]

SCALES = ("linear", "log")
DISTRIBUTIONS = ("geometric", "uniform")
MONOTONE_LAWS = ("none", "increasing", "decreasing")


class AxisConfigError(ValueError):
    """An axis declaration is malformed (fail-closed: never silently repaired)."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AxisConfigError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise AxisConfigError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class AxisSpec:
    """One parameter axis: searched span, sampling lattice and expected laws.

    ``monotone``/``monotone_slack``/``sign``/``floor`` describe the *physically
    expected* behaviour of the measured scalar along the axis (criterion (b) of
    the probing operator: monotonicity / sign-law breaks).  ``floor`` marks a
    round-off floor: two values below it are treated as equal, so that
    estimator noise on the order of round-off cannot fabricate a law break.
    """

    name: str
    low: float
    high: float
    scale: str = "linear"
    samples: int = 17
    distribution: str = "geometric"
    monotone: str = "none"
    monotone_slack: float = 0.3
    floor: float | None = None
    nominal: float | None = None
    quantum: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AxisConfigError("axis name must be a non-empty string")
        lo, hi = _finite(self.low, f"{self.name}.low"), _finite(self.high, f"{self.name}.high")
        if not lo < hi:
            raise AxisConfigError(f"{self.name}: low must be < high, got [{lo}, {hi}]")
        if self.scale not in SCALES:
            raise AxisConfigError(f"{self.name}: scale must be one of {SCALES}")
        if self.distribution not in DISTRIBUTIONS:
            raise AxisConfigError(f"{self.name}: distribution must be one of {DISTRIBUTIONS}")
        if self.scale == "log" and lo <= 0.0:
            raise AxisConfigError(f"{self.name}: log scale needs low > 0, got {lo}")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 5:
            raise AxisConfigError(f"{self.name}: samples must be an int >= 5")
        if self.monotone not in MONOTONE_LAWS:
            raise AxisConfigError(f"{self.name}: monotone must be one of {MONOTONE_LAWS}")
        _finite(self.monotone_slack, f"{self.name}.monotone_slack")
        if self.monotone_slack < 0.0:
            raise AxisConfigError(f"{self.name}: monotone_slack must be >= 0")
        if self.floor is not None:
            _finite(self.floor, f"{self.name}.floor")
            if self.floor < 0.0:
                raise AxisConfigError(f"{self.name}: floor must be >= 0")
        if self.quantum is not None:
            q = _finite(self.quantum, f"{self.name}.quantum")
            if q <= 0.0:
                raise AxisConfigError(f"{self.name}: quantum must be > 0")

    # -- scale map -------------------------------------------------------
    def to_scale(self, value: float) -> float:
        value = _finite(value, f"{self.name}.value")
        return math.log(value) if self.scale == "log" else value

    def from_scale(self, coordinate: float) -> float:
        coordinate = _finite(coordinate, f"{self.name}.coordinate")
        return math.exp(coordinate) if self.scale == "log" else coordinate

    @property
    def width(self) -> float:
        """Width of the searched span in *value* units."""
        return self.high - self.low

    @property
    def scale_width(self) -> float:
        return self.to_scale(self.high) - self.to_scale(self.low)

    def nominal_value(self) -> float:
        """Reference value used for the axes not currently being probed."""
        if self.nominal is not None:
            return _finite(self.nominal, f"{self.name}.nominal")
        return self.from_scale(0.5 * (self.to_scale(self.low) + self.to_scale(self.high)))

    def snap(self, value: float) -> float:
        """Nearest representable value on a quantised axis (identity otherwise)."""
        if self.quantum is None:
            return value
        q = self.quantum
        return min(self.high, max(self.low, round(value / q) * q))

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def fraction(self, value: float) -> float:
        """Position of ``value`` inside the searched span, in scale coordinates."""
        return (self.to_scale(value) - self.to_scale(self.low)) / self.scale_width

    def coordinates(self) -> tuple[float, ...]:
        """The sample lattice of the geometric gradient scan, ascending."""
        n = self.samples
        u0, u1 = self.to_scale(self.low), self.to_scale(self.high)
        if self.distribution == "uniform":
            fractions = [i / (n - 1) for i in range(n)]
        else:  # Chebyshev-Lobatto: geometric gradient, dense at both ends
            fractions = [0.5 * (1.0 - math.cos(math.pi * i / (n - 1))) for i in range(n)]
        values: list[float] = []
        for f in fractions:
            value = self.from_scale(u0 + f * (u1 - u0))
            if self.scale == "linear":
                # snap the ends exactly onto the declared span
                if f == 0.0:
                    value = self.low
                elif f == 1.0:
                    value = self.high
            values.append(value)
        if self.quantum is not None:
            # a quantised axis (e.g. an integer grid size) can only be sampled at
            # multiples of the quantum: snap and drop duplicates, keeping the ends
            q = self.quantum
            snapped = [round(v / q) * q for v in values]
            kept: list[float] = []
            for v in snapped:
                if not kept or abs(v - kept[-1]) > 1e-12 * max(1.0, abs(v)):
                    kept.append(v)
            if self.low not in kept:
                kept.insert(0, self.low)
            if self.high not in kept:
                kept.append(self.high)
            return tuple(kept)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "low": self.low,
            "high": self.high,
            "scale": self.scale,
            "samples": self.samples,
            "distribution": self.distribution,
            "monotone": self.monotone,
            "monotone_slack": self.monotone_slack,
            "floor": self.floor,
            "nominal": self.nominal_value(),
            "quantum": self.quantum,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AxisSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise AxisConfigError(f"unknown axis keys: {unknown}")
        return cls(**dict(mapping))