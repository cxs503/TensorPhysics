"""Runtime four-term energy book for a coupled multi-solver run.

The contract the book exists to test is

    E_total(t) = E_kinetic(t) + E_potential(t) + E_internal(t) + E_dissipated(t)

    residual(t) = |E_total(t) - E_total(0)| / E_scale

with ``E_scale = sum_c |E_c(0)|`` so that the residual is relative to the
energy actually present at the start of the run (the campaign criterion).  The
book is filled from *solver states* (``DEM2D.kinetic_energy``,
``Q4PlaneStressFEM.strain_energy``, ``Q4ThermalFEM.energy()``, the LBM scalar
mass, the measured dashpot dissipation), never from a bookkeeping sum of the
transfers.  That is the point: a book-keeping identity would close by
construction and would not detect a coupling that quietly creates energy.

``E_dissipated`` enters the total with a **plus** sign.  It is energy that has
left the resolvable fields into an *explicitly modelled* sink (the contact
dashpot); keeping it in the total is what makes the dissipative interface
visible: an unbooked sink shows up as a residual exactly like an unbooked
source.

Decomposition used by the loop (stated here because any other decomposition
would give a different residual):

* ``kinetic``    — DEM particle translation + plate translational kinetic energy
* ``potential``  — contact spring energy ``sum 0.5 k overlap^2`` (recoverable,
  concentrated at the interface; gravity is switched off, the impact is
  horizontal in the plate frame)
* ``internal``   — fluid thermal energy (LBM scalar), plate thermal energy
  (lumped capacity) and plate elastic strain energy (recoverable continuum
  deformation)
* ``dissipated`` — measured dashpot work, accumulated with a positive sign
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

__all__ = [
    "EnergyBookError",
    "Component",
    "EnergySample",
    "EnergyBook",
]


class EnergyBookError(RuntimeError):
    """The book was used in a way that would make its residual meaningless."""


class Component(str, Enum):
    """The four terms of the contract."""

    KINETIC = "kinetic"
    POTENTIAL = "potential"
    INTERNAL = "internal"
    DISSIPATED = "dissipated"


COMPONENTS: tuple[Component, ...] = (
    Component.KINETIC,
    Component.POTENTIAL,
    Component.INTERNAL,
    Component.DISSIPATED,
)


@dataclass(frozen=True)
class EnergySample:
    """One recorded book state."""

    t: float
    values: Mapping[str, Mapping[str, float]]
    total: float

    def component_total(self, component: Component | str) -> float:
        key = component.value if isinstance(component, Component) else str(component)
        if key not in self.values:
            raise EnergyBookError(f"unknown energy component {key!r}")
        return float(sum(self.values[key].values()))

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "components": {k: dict(v) for k, v in self.values.items()},
            "total": self.total,
        }


class EnergyBook:
    """Accumulate the four-term balance from solver measurements.

    Parameters
    ----------
    sides:
        Account names inside each component (``fluid``, ``solid``,
        ``particles``).  The side split is diagnostic: the contract is on the
        *component* totals.
    """

    def __init__(self, sides: Sequence[str] = ("fluid", "solid", "particles")) -> None:
        sides = tuple(sides)
        if not sides:
            raise EnergyBookError("at least one side account is required")
        self.sides = sides
        self._state: dict[Component, dict[str, float]] = {
            c: {s: 0.0 for s in sides} for c in COMPONENTS
        }
        self._history: list[EnergySample] = []

    # -- accounts ---------------------------------------------------------
    def _check(self, component: Component | str, side: str) -> Component:
        key = Component(component)
        if side not in self._state[key]:
            raise EnergyBookError(f"unknown side {side!r}; known: {self.sides}")
        return key

    def set(self, component: Component | str, side: str, value: float) -> None:
        key = self._check(component, side)
        if not math.isfinite(value):
            raise EnergyBookError(f"non-finite {key.value} energy for {side!r}: {value!r}")
        self._state[key][side] = float(value)

    def add(self, component: Component | str, side: str, delta: float) -> None:
        key = self._check(component, side)
        if not math.isfinite(delta):
            raise EnergyBookError(
                f"non-finite {key.value} increment for {side!r}: {delta!r}"
            )
        self._state[key][side] += float(delta)

    def get(self, component: Component | str, side: str) -> float:
        return self._state[self._check(component, side)][side]

    def component_total(self, component: Component | str) -> float:
        key = Component(component)
        return float(sum(self._state[key].values()))

    def total(self) -> float:
        return float(sum(self.component_total(c) for c in COMPONENTS))

    # -- sampling ---------------------------------------------------------
    def sample(self, t: float) -> EnergySample:
        if self._history and t <= self._history[-1].t:
            raise EnergyBookError(
                f"energy samples must be strictly increasing in time "
                f"({t!r} after {self._history[-1].t!r})"
            )
        snap = EnergySample(
            t=float(t),
            values={c.value: dict(self._state[c]) for c in COMPONENTS},
            total=self.total(),
        )
        self._history.append(snap)
        return snap

    @property
    def history(self) -> tuple[EnergySample, ...]:
        return tuple(self._history)

    def initial_scale(self) -> float:
        """``E_scale = sum_c |E_c(0)|``; raises if the run starts with no energy."""
        if not self._history:
            raise EnergyBookError("no sample recorded yet")
        first = self._history[0]
        scale = sum(abs(first.component_total(c)) for c in COMPONENTS)
        if scale <= 0.0:
            raise EnergyBookError(
                "initial energy is zero: a relative residual would be undefined"
            )
        return scale

    def residual(self, scale: float | None = None,
                 components: Iterable[Component | str] | None = None) -> float:
        """Relative non-conservation of the (optionally restricted) book."""
        if len(self._history) < 2:
            return 0.0
        first, last = self._history[0], self._history[-1]
        keys = COMPONENTS if components is None else tuple(Component(c) for c in components)
        delta = sum(last.component_total(c) - first.component_total(c) for c in keys)
        if scale is None:
            scale = sum(abs(first.component_total(c)) for c in keys) or self.initial_scale()
        return abs(delta) / scale

    def absolute_imbalance(self, components: Iterable[Component | str] | None = None) -> float:
        """``|E_total(t_end) - E_total(0)|`` in joules (no normalisation)."""
        if len(self._history) < 2:
            return 0.0
        first, last = self._history[0], self._history[-1]
        keys = COMPONENTS if components is None else tuple(Component(c) for c in components)
        return abs(sum(last.component_total(c) - first.component_total(c) for c in keys))

    def series(self) -> dict[str, list[float]]:
        """Column-oriented time series of the book (component totals + total)."""
        out: dict[str, list[float]] = {"t": [s.t for s in self._history],
                                       "total": [s.total for s in self._history]}
        for c in COMPONENTS:
            out[c.value] = [s.component_total(c) for s in self._history]
        return out