"""Constraint model: inequality / equality / discrete feasibility.

A *window* is the subset of a declared parameter span on which every constraint
of a case is satisfied.  Three constraint kinds are supported, and all of them
are reduced to one scalar *margin* with a single sign convention:

``inequality``  ``margin = side * value``  (``side = +1`` means ``value >= 0``
                is feasible, ``side = -1`` means ``value <= 0`` is feasible)
``equality``    ``margin = tolerance - |value - target|``
``discrete``    ``margin = tolerance - distance(value, allowed set)``

Because every kind ends up as ``margin >= 0 <=> feasible``, the scanner, the
root finder, the active-set logic and the boundary marcher are kind-agnostic and
contain no branching on physical meaning.  Discrete constraints are excluded
from continuous boundary detection (a lattice does not have a boundary to
march); they are applied afterwards as a snapping step
(:mod:`tensorphysics.envelope.marching`), which keeps the two semantics from
ever being mixed silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Tuple

__all__ = [
    "Constraint",
    "ConstraintContract",
    "ConstraintKind",
    "ConstraintSet",
    "EvalCounter",
    "Evaluation",
    "Lattice",
    "lattice_distance",
]


class ConstraintKind(str, Enum):
    INEQUALITY = "inequality"
    EQUALITY = "equality"
    DISCRETE = "discrete"


class EvalCounter:
    """Counts individual constraint evaluations (the unit of the cost budget)."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def tick(self, n: int = 1) -> None:
        self.count += n


@dataclass(frozen=True)
class ConstraintContract:
    """Declared expectations used by the contract checker (never by the solver)."""

    slope: Mapping[str, int] = field(default_factory=dict)
    convex: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"slope": dict(self.slope), "convex": self.convex}


@dataclass(frozen=True)
class Lattice:
    """Discrete feasible set: ``offset + k * step`` or an explicit value list."""

    step: float | None = None
    offset: float = 0.0
    values: Tuple[float, ...] = ()
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.values and (self.step is not None):
            raise ValueError("lattice: give either step or values, not both")
        if not self.values and (self.step is None or self.step <= 0.0):
            raise ValueError("lattice: step must be positive (or provide values)")
        if self.tolerance < 0.0:
            raise ValueError("lattice: tolerance must be >= 0")

    def nearest(self, value: float) -> tuple[float, float]:
        """Return ``(nearest_allowed, |value - nearest_allowed|)``."""

        if self.values:
            best = min(self.values, key=lambda v: abs(v - value))
            return best, abs(value - best)
        k = round((value - self.offset) / float(self.step))
        best = self.offset + k * float(self.step)
        return best, abs(value - best)

    def candidates(self, lo: float, hi: float) -> tuple[float, ...]:
        """Allowed values inside ``[lo, hi]`` (ascending)."""

        if self.values:
            return tuple(sorted(v for v in self.values if lo <= v <= hi))
        step = float(self.step)
        k_lo = math.ceil((lo - self.offset) / step)
        k_hi = math.floor((hi - self.offset) / step)
        if k_hi < k_lo:
            return ()
        return tuple(self.offset + k * step for k in range(int(k_lo), int(k_hi) + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "offset": self.offset,
            "values": list(self.values),
            "tolerance": self.tolerance,
        }


def lattice_distance(value: float, lattice: Lattice) -> float:
    return lattice.nearest(value)[1]


@dataclass(frozen=True)
class Constraint:
    """One feasibility condition, reduced to a scalar margin."""

    id: str
    kind: ConstraintKind
    evaluator: Callable[[Mapping[str, Any]], float]
    side: int = 1
    target: float = 0.0
    tolerance: float = 0.0
    lattice: Lattice | None = None
    active_tolerance: float = 0.0
    contract: ConstraintContract = field(default_factory=ConstraintContract)
    description: str = ""
    expression: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind is ConstraintKind.INEQUALITY and self.side not in (-1, 1):
            raise ValueError(f"constraint {self.id!r}: side must be +1 or -1")
        if self.kind is ConstraintKind.DISCRETE and self.lattice is None:
            raise ValueError(f"constraint {self.id!r}: discrete constraint needs a lattice")
        if self.tolerance < 0.0:
            raise ValueError(f"constraint {self.id!r}: tolerance must be >= 0")

    # -- evaluation ---------------------------------------------------------
    def value(self, params: Mapping[str, Any], counter: EvalCounter | None = None) -> float:
        raw = self.evaluator(params)
        if counter is not None:
            counter.tick()
        return float(raw)

    def margin(self, params: Mapping[str, Any], counter: EvalCounter | None = None) -> float:
        value = self.value(params, counter)
        if self.kind is ConstraintKind.INEQUALITY:
            return float(self.side) * value
        if self.kind is ConstraintKind.EQUALITY:
            return self.tolerance - abs(value - self.target)
        assert self.lattice is not None
        return self.tolerance - lattice_distance(value, self.lattice)

    def is_active(self, margin: float) -> bool:
        return abs(margin) <= self.active_tolerance

    @property
    def continuous(self) -> bool:
        return self.kind is not ConstraintKind.DISCRETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "side": self.side,
            "target": self.target,
            "tolerance": self.tolerance,
            "lattice": self.lattice.to_dict() if self.lattice else None,
            "active_tolerance": self.active_tolerance,
            "contract": self.contract.to_dict(),
            "description": self.description,
        }


@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating a whole constraint set at one parameter point."""

    margins: Mapping[str, float]
    feasible: bool
    min_margin: float
    driver: str
    violated: Tuple[str, ...]
    active: Tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "margins": dict(self.margins),
            "feasible": self.feasible,
            "min_margin": self.min_margin,
            "driver": self.driver,
            "violated": list(self.violated),
            "active": list(self.active),
        }


@dataclass(frozen=True)
class ConstraintSet:
    """Ordered constraint collection; evaluation is deterministic in ``id`` order."""

    constraints: Tuple[Constraint, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("constraint set is empty")
        seen = set()
        for constraint in self.constraints:
            if constraint.id in seen:
                raise ValueError(f"duplicate constraint id {constraint.id!r}")
            seen.add(constraint.id)

    def __iter__(self) -> Iterable[Constraint]:
        return iter(self.constraints)

    def __len__(self) -> int:
        return len(self.constraints)

    def by_id(self, cid: str) -> Constraint:
        for constraint in self.constraints:
            if constraint.id == cid:
                return constraint
        raise KeyError(cid)

    @property
    def continuous_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.continuous)

    @property
    def discrete_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if not c.continuous)

    def margins(
        self, params: Mapping[str, Any], counter: EvalCounter | None = None
    ) -> dict[str, float]:
        return {c.id: c.margin(params, counter) for c in self.constraints}

    def evaluate(
        self, params: Mapping[str, Any], counter: EvalCounter | None = None
    ) -> Evaluation:
        margins = self.margins(params, counter)
        driver = min(margins, key=lambda k: (margins[k], k))
        violated = tuple(cid for cid, m in margins.items() if m < 0.0)
        active = tuple(c.id for c in self.constraints if c.is_active(margins[c.id]))
        return Evaluation(
            margins=margins,
            feasible=not violated,
            min_margin=margins[driver],
            driver=driver,
            violated=violated,
            active=active,
        )

    def continuous_feasible(
        self, params: Mapping[str, Any], counter: EvalCounter | None = None
    ) -> bool:
        """Feasibility ignoring discrete constraints (continuous window cell)."""

        return all(
            c.margin(params, counter) >= 0.0 for c in self.continuous_constraints
        )

    def describe(self) -> dict[str, Any]:
        return {"constraints": [c.to_dict() for c in self.constraints]}