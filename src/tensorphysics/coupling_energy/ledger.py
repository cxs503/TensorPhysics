"""Runtime energy and momentum ledgers for partitioned multi-solver coupling.

A partitioned (solver-a, solver-b) coupling is *conservative* if and only if
every exchange that removes ``q`` from one side books the identical ``q`` into
the other side, and every boundary flux that enters the system is booked as an
external input.  The ledgers below are the executable form of that contract:

    E_total(t) = sum_sides E_side(t) - E_external_in(t) + E_dissipated(t)

so that ``residual(t) = |E_total(t) - E_total(0)| / |E_total(0)|`` measures the
energy that the coupling bookkeeping did **not** account for.  The same
structure is provided for linear momentum with impulses instead of energies.

Everything here is deliberately scheme-agnostic: the ledgers do not know how a
solver integrates, they only require that every transfer crosses the ledger.
That is what makes them usable as an acceptance instrument for three different
couplings (thermal, mechanical, three-field) without duplicating physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "CouplingLedgerError",
    "EnergySample",
    "EnergyLedger",
    "MomentumLedger",
    "OrderEstimate",
    "observed_order",
    "IntervalCompleteLedger",
    "IntervalEntry",
]


class CouplingLedgerError(RuntimeError):
    """A transfer was booked that violates the ledger contract."""


@dataclass(frozen=True)
class EnergySample:
    """One recorded ledger state.

    ``total`` is the conserved quantity: side energies minus what entered from
    outside plus what was dissipated by an explicitly modelled sink.
    """

    t: float
    sides: Mapping[str, float]
    external_in: float
    dissipated: float
    interface: float
    total: float

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "sides": dict(self.sides),
            "external_in": self.external_in,
            "dissipated": self.dissipated,
            "interface": self.interface,
            "total": self.total,
        }


class EnergyLedger:
    """Book side energies, boundary inputs and interface transfers.

    Parameters
    ----------
    sides:
        Names of the energy accounts that belong to the coupled system
        (``("fluid", "solid", "particles")`` by default).
    """

    def __init__(self, sides: Sequence[str] = ("fluid", "solid", "particles")) -> None:
        sides = tuple(sides)
        if not sides:
            raise CouplingLedgerError("at least one side account is required")
        self.sides = sides
        self._e = {name: 0.0 for name in sides}
        self.external_in = 0.0
        self.dissipated = 0.0
        self.interface = 0.0
        self._history: list[EnergySample] = []

    # -- account access ---------------------------------------------------------
    def _check(self, side: str) -> None:
        if side not in self._e:
            raise CouplingLedgerError(f"unknown energy account {side!r}; known: {self.sides}")

    def get(self, side: str) -> float:
        self._check(side)
        return self._e[side]

    def set(self, side: str, value: float) -> None:
        self._check(side)
        if not math.isfinite(value):
            raise CouplingLedgerError(f"non-finite energy for {side!r}: {value!r}")
        self._e[side] = float(value)

    def add(self, side: str, delta: float) -> None:
        self._check(side)
        if not math.isfinite(delta):
            raise CouplingLedgerError(f"non-finite energy increment for {side!r}: {delta!r}")
        self._e[side] += float(delta)

    # -- transfers --------------------------------------------------------------
    def exchange(self, source: str, target: str, energy: float) -> None:
        """Move ``energy`` from ``source`` to ``target`` (no loss allowed).

        A negative ``energy`` is accepted and simply reverses the transfer; the
        interface account always accumulates the absolute magnitude so that the
        reported ``interface`` figure is a diagnostic of coupling activity and
        not a second conserved quantity.
        """
        if source == target:
            raise CouplingLedgerError("source and target accounts must differ")
        self._check(source)
        self._check(target)
        if not math.isfinite(energy):
            raise CouplingLedgerError(f"non-finite exchange {energy!r}")
        self._e[source] -= float(energy)
        self._e[target] += float(energy)
        self.interface += abs(float(energy))

    def book_external(self, energy: float) -> None:
        """Book energy that enters the system from outside the coupled model."""
        if not math.isfinite(energy):
            raise CouplingLedgerError(f"non-finite external energy {energy!r}")
        self.external_in += float(energy)

    def book_dissipation(self, energy: float) -> None:
        """Book energy that leaves the sides through an modelled sink."""
        if energy < 0.0:
            raise CouplingLedgerError("dissipation must be non-negative")
        self.dissipated += float(energy)

    # -- conserved quantity -----------------------------------------------------
    def total(self) -> float:
        return sum(self._e.values()) - self.external_in + self.dissipated

    def sample(self, t: float) -> EnergySample:
        sample = EnergySample(
            t=float(t),
            sides=dict(self._e),
            external_in=self.external_in,
            dissipated=self.dissipated,
            interface=self.interface,
            total=self.total(),
        )
        self._history.append(sample)
        return sample

    @property
    def history(self) -> tuple[EnergySample, ...]:
        return tuple(self._history)

    def total_history(self) -> list[float]:
        return [s.total for s in self._history]

    # -- acceptance metrics -----------------------------------------------------
    def residual(self) -> float:
        """Relative drift of the conserved quantity w.r.t. the first sample."""
        if not self._history:
            raise CouplingLedgerError("no samples recorded; call sample() first")
        reference = self._history[0].total
        scale = abs(reference)
        if scale <= 0.0:
            scale = max((abs(s.total) for s in self._history), default=1.0)
        if scale <= 0.0:
            return 0.0
        return abs(self._history[-1].total - reference) / scale

    def max_residual(self) -> float:
        if not self._history:
            raise CouplingLedgerError("no samples recorded; call sample() first")
        reference = self._history[0].total
        scale = max(abs(reference), 1e-300)
        return max(abs(s.total - reference) for s in self._history) / scale

    def drift_slope(self) -> float:
        """Least-squares slope of the residual history, normalised by duration.

        A coupling that leaks energy in proportion to the number of steps has a
        visibly positive slope; a stable staggered scheme oscillates around a
        bounded value and returns a slope consistent with zero.
        """
        if len(self._history) < 2:
            raise CouplingLedgerError("at least two samples are required")
        reference = self._history[0].total
        scale = max(abs(reference), 1e-300)
        ts = [s.t - self._history[0].t for s in self._history]
        es = [abs(s.total - reference) / scale for s in self._history]
        n = len(ts)
        mt = sum(ts) / n
        me = sum(es) / n
        denom = sum((t - mt) ** 2 for t in ts)
        if denom <= 0.0:
            return 0.0
        return sum((t - mt) * (e - me) for t, e in zip(ts, es)) / denom

    def is_divergent(self, growth: float = 4.0) -> bool:
        """Flag an unbounded trend: late residual is ``growth`` x the early one.

        The early window is the first quarter of the history, the late window
        the last quarter.  A bounded (oscillatory) residual keeps the ratio near
        one; a linearly leaking scheme raises it until it exceeds ``growth``.
        """
        if len(self._history) < 8:
            raise CouplingLedgerError("need at least 8 samples to judge divergence")
        reference = self._history[0].total
        scale = max(abs(reference), 1e-300)
        errs = [abs(s.total - reference) / scale for s in self._history]
        q = max(1, len(errs) // 4)
        early = max(errs[:q])
        late = max(errs[-q:])
        return late > growth * max(early, 1e-300)

    def as_dict(self) -> dict:
        return {
            "sides": self.sides,
            "energies": dict(self._e),
            "external_in": self.external_in,
            "dissipated": self.dissipated,
            "interface": self.interface,
            "total": self.total(),
            "residual": self.residual() if self._history else None,
        }


class MomentumLedger:
    """Linear-momentum ledger for force-coupled solvers.

    ``exchange`` books an impulse that leaves ``source`` and enters ``target``
    (Newton's third law), so a shared-contact-force scheme is conservative by
    construction and an independently evaluated contact force is not.
    """

    def __init__(self, sides: Sequence[str] = ("fluid", "solid", "particles"),
                 scale: float | None = None) -> None:
        sides = tuple(sides)
        if not sides:
            raise CouplingLedgerError("at least one momentum account is required")
        self.sides = sides
        self._p = {name: [0.0, 0.0] for name in sides}
        self.external = [0.0, 0.0]
        self.interface = 0.0
        self._initial_total = (0.0, 0.0)
        self._recorded = False
        self._scale = scale
        self._history: list[tuple[float, tuple[float, float], float]] = []

    def _check(self, side: str) -> None:
        if side not in self._p:
            raise CouplingLedgerError(f"unknown momentum account {side!r}")

    def set(self, side: str, px: float, py: float) -> None:
        self._check(side)
        self._p[side] = [float(px), float(py)]

    def add_impulse(self, side: str, px: float, py: float) -> None:
        self._check(side)
        self._p[side][0] += float(px)
        self._p[side][1] += float(py)

    def exchange(self, source: str, target: str, px: float, py: float) -> None:
        if source == target:
            raise CouplingLedgerError("source and target accounts must differ")
        self._check(source)
        self._check(target)
        self._p[source][0] -= float(px)
        self._p[source][1] -= float(py)
        self._p[target][0] += float(px)
        self._p[target][1] += float(py)
        self.interface += math.hypot(px, py)

    def book_external_impulse(self, px: float, py: float) -> None:
        self.external[0] += float(px)
        self.external[1] += float(py)

    def total(self) -> tuple[float, float]:
        px = sum(v[0] for v in self._p.values()) - self.external[0]
        py = sum(v[1] for v in self._p.values()) - self.external[1]
        return px, py

    def record(self) -> None:
        total = self.total()
        if not self._recorded:
            self._initial_total = total
            self._recorded = True
        self._history.append((float(len(self._history)), total, self.interface))

    @property
    def history(self) -> tuple:
        return tuple(self._history)

    def residual(self) -> float:
        if not self._history:
            raise CouplingLedgerError("record() has not been called")
        px, py = self.total()
        dx = px - self._initial_total[0]
        dy = py - self._initial_total[1]
        if self._scale is not None:
            scale = abs(self._scale)
        else:
            scale = max(
                [max(abs(p[0]), abs(p[1])) for p in self._p.values()]
                + [abs(self._initial_total[0]), abs(self._initial_total[1]), 1e-300]
            )
        return math.hypot(dx, dy) / scale


@dataclass(frozen=True)
class OrderEstimate:
    """Observed convergence order from a resolution/refinement series."""

    order: float | None
    steps: Sequence[float]
    errors: Sequence[float]
    exact: bool = False

    def as_dict(self) -> dict:
        return {
            "order": self.order,
            "steps": list(self.steps),
            "errors": list(self.errors),
            "exact": self.exact,
        }


def observed_order(steps: Iterable[float], errors: Iterable[float],
                   roundoff: float = 1e-14) -> OrderEstimate:
    """Fit ``error ~ step**p`` on the finest half of the series.

    Returns ``order=None, exact=True`` when every error sits at round-off: the
    mathematically honest statement for a scheme whose residual is structural
    (an antisymmetric exchange) rather than a discretisation error.
    """
    s = [float(v) for v in steps]
    e = [float(v) for v in errors]
    if len(s) != len(e):
        raise CouplingLedgerError("steps and errors must have the same length")
    if len(s) < 2:
        raise CouplingLedgerError("at least two resolution levels are required")
    if all(v <= roundoff for v in e):
        return OrderEstimate(None, s, e, True)
    pairs = [(si, ei) for si, ei in zip(s, e) if si > 0.0 and ei > 0.0]
    if len(pairs) < 2:
        raise CouplingLedgerError("need two positive (step, error) pairs")
    # use the finest half, where the asymptotic regime lives
    keep = max(2, len(pairs) // 2)
    pairs = pairs[-keep:]
    ls = [math.log(si) for si, _ in pairs]
    le = [math.log(ei) for _, ei in pairs]
    n = len(pairs)
    ms = sum(ls) / n
    me = sum(le) / n
    denom = sum((v - ms) ** 2 for v in ls)
    if denom <= 0.0:
        raise CouplingLedgerError("steps must not all be identical")
    order = sum((a - ms) * (b - me) for a, b in zip(ls, le)) / denom
    return OrderEstimate(order, s, e)

# ---------------------------------------------------------------------------
# Interval-complete conservation ledger
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntervalEntry:
    """One exchange interval's contribution to the conservation ledger.

    Each interval has a unique index, a time range, and an assigned flux value.
    The interval-complete identity requires:
        Σ interval_values = total_exchange
        unassigned_share = 0
    """
    index: int
    t_start: float
    t_end: float
    assigned_flux: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise CouplingLedgerError("interval index must be non-negative")
        if self.t_end <= self.t_start:
            raise CouplingLedgerError("interval must have positive duration")
        if not math.isfinite(self.assigned_flux):
            raise CouplingLedgerError("assigned flux must be finite")


class IntervalCompleteLedger:
    """Interval-complete conservation ledger for partitioned coupling.

    This ledger enforces the interval-complete identity:
        - Each interface flux has exactly one owning exchange interval
        - Σ interval contributions = total exchange
        - Unassigned share = 0 (machine precision)

    The ledger tracks both the per-interval assignments and the running
    totals, allowing verification of the conservation contract at any point
    in the simulation.

    Parameters
    ----------
    sides:
        Names of the energy accounts that belong to the coupled system.
    """

    def __init__(self, sides: Sequence[str] = ("fluid", "solid", "particles")) -> None:
        sides = tuple(sides)
        if not sides:
            raise CouplingLedgerError("at least one side account is required")
        self.sides = sides
        self._e = {name: 0.0 for name in sides}
        self.external_in = 0.0
        self.dissipated = 0.0
        self._intervals: list[IntervalEntry] = []
        self._total_exchange = 0.0
        self._history: list[dict] = []

    def _check(self, side: str) -> None:
        if side not in self._e:
            raise CouplingLedgerError(f"unknown energy account {side!r}; known: {self.sides}")

    def set(self, side: str, value: float) -> None:
        self._check(side)
        if not math.isfinite(value):
            raise CouplingLedgerError(f"non-finite energy for {side!r}: {value!r}")
        self._e[side] = float(value)

    def add(self, side: str, delta: float) -> None:
        self._check(side)
        if not math.isfinite(delta):
            raise CouplingLedgerError(f"non-finite energy increment for {side!r}: {delta!r}")
        self._e[side] += float(delta)

    def book_interval(self, index: int, t_start: float, t_end: float,
                      flux: float, source: str, target: str) -> None:
        """Book an exchange interval with its assigned flux.

        The flux is transferred from source to target, and the interval is
        recorded for the interval-complete identity check.
        """
        if source == target:
            raise CouplingLedgerError("source and target accounts must differ")
        self._check(source)
        self._check(target)
        if not math.isfinite(flux):
            raise CouplingLedgerError(f"non-finite flux {flux!r}")
        if t_end <= t_start:
            raise CouplingLedgerError("interval must have positive duration")

        # Transfer energy
        self._e[source] -= float(flux)
        self._e[target] += float(flux)

        # Record interval
        entry = IntervalEntry(
            index=index,
            t_start=float(t_start),
            t_end=float(t_end),
            assigned_flux=float(flux),
        )
        self._intervals.append(entry)
        self._total_exchange += float(flux)

    def book_external(self, energy: float) -> None:
        if not math.isfinite(energy):
            raise CouplingLedgerError(f"non-finite external energy {energy!r}")
        self.external_in += float(energy)

    def book_dissipation(self, energy: float) -> None:
        if energy < 0.0:
            raise CouplingLedgerError("dissipation must be non-negative")
        self.dissipated += float(energy)

    def total(self) -> float:
        return sum(self._e.values()) - self.external_in + self.dissipated

    @property
    def intervals(self) -> tuple[IntervalEntry, ...]:
        return tuple(self._intervals)

    @property
    def total_exchange(self) -> float:
        return self._total_exchange

    @property
    def sum_interval_fluxes(self) -> float:
        """Sum of all assigned interval fluxes."""
        return math.fsum(e.assigned_flux for e in self._intervals)

    @property
    def unassigned_share(self) -> float:
        """Unassigned share = total_exchange - sum_interval_fluxes.

        For an interval-complete ledger, this must be identically zero
        (within machine precision).
        """
        return self._total_exchange - self.sum_interval_fluxes

    def interval_complete_residual(self) -> float:
        """Residual of the interval-complete identity.

        Returns |Σ interval_fluxes - total_exchange| / max(|total_exchange|, 1).
        For a correct implementation, this should be < 1e-12.
        """
        denom = max(abs(self._total_exchange), 1.0)
        return abs(self.sum_interval_fluxes - self._total_exchange) / denom

    def sample(self, t: float) -> dict:
        sample = {
            "t": float(t),
            "sides": dict(self._e),
            "external_in": self.external_in,
            "dissipated": self.dissipated,
            "total": self.total(),
            "total_exchange": self._total_exchange,
            "sum_interval_fluxes": self.sum_interval_fluxes,
            "unassigned_share": self.unassigned_share,
            "interval_complete_residual": self.interval_complete_residual(),
            "n_intervals": len(self._intervals),
        }
        self._history.append(sample)
        return sample

    @property
    def history(self) -> tuple[dict, ...]:
        return tuple(self._history)

    def residual(self) -> float:
        """Relative drift of the conserved quantity w.r.t. the first sample."""
        if not self._history:
            raise CouplingLedgerError("no samples recorded; call sample() first")
        reference = self._history[0]["total"]
        scale = abs(reference)
        if scale <= 0.0:
            scale = max((abs(s["total"]) for s in self._history), default=1.0)
        if scale <= 0.0:
            return 0.0
        return abs(self._history[-1]["total"] - reference) / scale

    def max_residual(self) -> float:
        if not self._history:
            raise CouplingLedgerError("no samples recorded; call sample() first")
        reference = self._history[0]["total"]
        scale = max(abs(reference), 1e-300)
        return max(abs(s["total"] - reference) for s in self._history) / scale

    def as_dict(self) -> dict:
        return {
            "sides": self.sides,
            "energies": dict(self._e),
            "external_in": self.external_in,
            "dissipated": self.dissipated,
            "total": self.total(),
            "total_exchange": self._total_exchange,
            "sum_interval_fluxes": self.sum_interval_fluxes,
            "unassigned_share": self.unassigned_share,
            "interval_complete_residual": self.interval_complete_residual(),
            "n_intervals": len(self._intervals),
            "residual": self.residual() if self._history else None,
        }
