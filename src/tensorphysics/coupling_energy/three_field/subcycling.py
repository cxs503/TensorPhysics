"""Multi-rate scheduling and degree-``p`` reconstruction of interface data.

Three solvers with three different stable time steps share one physical clock::

    dt_solid   (explicit plate, stability limited)
    dt_particles (DEM contact, stability limited)
    dt_fluid   (lattice step derived from the lattice diffusivity)

The exchange between any two of them is *reconstructed*: a solver that needs a
counterpart quantity at one of its own instants only has the counterpart's
samples, and evaluates a degree-``p`` polynomial through the last ``p+1``
samples (Lagrange form).  ``p = 1`` is linear extrapolation from two samples,
``p = 2`` quadratic from three, ``p = 3`` cubic from four.  This is what a
partitioned code does when it does not sub-step the counterpart, and the
reconstruction error is what the order gate measures.

The reconstruction is deliberately *causal*: ``value_at(t)`` only uses samples
with ``t_sample <= t``, which is the only information a partitioned solver would
have at that instant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "SubcyclingError",
    "RateSchedule",
    "History",
    "lagrange_value",
    "integrate_history",
    "observed_order",
]


class SubcyclingError(RuntimeError):
    """The schedule or the reconstruction was used outside its contract."""


# ---------------------------------------------------------------------------
# rates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RateSchedule:
    """The three solver steps plus the exchange (``macro``) interval.

    ``macro`` is the largest of the participating steps; the sub-cycling ratio
    of a pair is the ratio of the two steps and is reported so that a residual
    can be quoted *at a ratio* instead of "in general".
    """

    dt_fluid: float
    dt_solid: float
    dt_particles: float

    def __post_init__(self) -> None:
        for name in ("dt_fluid", "dt_solid", "dt_particles"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise SubcyclingError(f"{name} must be positive and finite, got {value!r}")

    @property
    def dt_macro(self) -> float:
        return max(self.dt_fluid, self.dt_solid, self.dt_particles)

    def ratio(self, coarse: str, fine: str) -> float:
        steps = {"fluid": self.dt_fluid, "solid": self.dt_solid,
                 "particles": self.dt_particles}
        for name in (coarse, fine):
            if name not in steps:
                raise SubcyclingError(f"unknown solver {name!r}")
        if steps[fine] > steps[coarse]:
            raise SubcyclingError(
                f"{fine!r} is not finer than {coarse!r} "
                f"({steps[fine]!r} > {steps[coarse]!r})"
            )
        return steps[coarse] / steps[fine]

    def as_dict(self) -> dict:
        return {
            "dt_fluid": self.dt_fluid,
            "dt_solid": self.dt_solid,
            "dt_particles": self.dt_particles,
            "dt_macro": self.dt_macro,
            "ratio_solid_over_particles": self.dt_solid / self.dt_particles,
            "ratio_fluid_over_solid": self.dt_fluid / self.dt_solid,
            "ratio_fluid_over_particles": self.dt_fluid / self.dt_particles,
        }


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------
def lagrange_value(ts: Sequence[float], ys: Sequence[np.ndarray | float],
                   t: float) -> np.ndarray | float:
    """Lagrange polynomial through ``(ts, ys)`` evaluated at ``t``.

    ``ys`` may hold vectors; the interpolation is then componentwise, which is
    what an interface with several nodes needs.
    """
    n = len(ts)
    if n == 0 or len(ys) != n:
        raise SubcyclingError("ts and ys must be non-empty and of equal length")
    if n == 1:
        return ys[0]
    value = None
    for i in range(n):
        weight = 1.0
        for j in range(n):
            if i == j:
                continue
            den = ts[i] - ts[j]
            if den == 0.0:
                raise SubcyclingError("duplicate sample instants in the reconstruction window")
            weight *= (t - ts[j]) / den
        term = weight * np.asarray(ys[i], dtype=float)
        value = term if value is None else value + term
    return value


@dataclass
class History:
    """Causal sample history of one exchanged quantity.

    ``degree`` is the reconstruction order ``p``; samples are kept in
    increasing time and only the last ``degree + 1`` are ever used.
    """

    name: str
    degree: int = 2
    times: list[float] = field(default_factory=list)
    values: list[np.ndarray | float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise SubcyclingError("reconstruction degree must be >= 0")

    def push(self, t: float, value) -> None:
        if self.times and t <= self.times[-1]:
            raise SubcyclingError(
                f"history {self.name!r} must be pushed in increasing time "
                f"({t!r} after {self.times[-1]!r})"
            )
        self.times.append(float(t))
        self.values.append(np.asarray(value, dtype=float) if np.ndim(value) else float(value))

    def window(self, t: float) -> tuple[list[float], list]:
        """Last ``p+1`` samples at or before ``t`` (causal)."""
        usable = [(ts, v) for ts, v in zip(self.times, self.values) if ts <= t + 1e-15]
        if not usable:
            raise SubcyclingError(
                f"history {self.name!r} has no sample at or before t={t!r}"
            )
        take = usable[-(self.degree + 1):]
        return [u[0] for u in take], [u[1] for u in take]

    def value_at(self, t: float):
        ts, ys = self.window(t)
        return lagrange_value(ts, ys, t)

    @property
    def latest(self) -> float:
        if not self.times:
            raise SubcyclingError(f"history {self.name!r} is empty")
        return self.times[-1]


def integrate_history(history: History, t0: float, t1: float, n_quad: int = 8) -> float:
    """``int_{t0}^{t1} reconstruction dt`` with a degree-``2 n_quad - 1`` rule.

    The receiver must integrate the reconstructed history over its own step;
    for a positive quantity (power) the sign of the integral is the transfer
    direction.
    """
    if t1 <= t0:
        raise SubcyclingError("integration interval must be increasing")
    if n_quad < 1:
        raise SubcyclingError("n_quad must be >= 1")
    # composite Simpson when the node count allows it, trapezoid otherwise
    if n_quad % 2 == 1:
        xs = np.linspace(t0, t1, n_quad)
        w = np.ones(n_quad)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w *= (t1 - t0) / (3.0 * (n_quad - 1))
    else:
        n_quad += 1
        xs = np.linspace(t0, t1, n_quad)
        w = np.ones(n_quad)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w *= (t1 - t0) / (3.0 * (n_quad - 1))
    total = 0.0
    for x, wi in zip(xs, w):
        total += wi * float(np.asarray(history.value_at(float(x))).sum())
    return float(total)


def observed_order(steps: Sequence[float], errors: Sequence[float]) -> float | None:
    """Fit ``error ~ C h^q`` through the given (step, error) pairs."""
    st = [s for s in steps if s > 0.0]
    er = [e for e in errors if e > 0.0]
    if len(st) < 2 or len(st) != len(er):
        return None
    qs = [
        math.log(er[i] / er[i + 1]) / math.log(st[i] / st[i + 1])
        for i in range(len(st) - 1)
    ]
    return float(sum(qs) / len(qs))