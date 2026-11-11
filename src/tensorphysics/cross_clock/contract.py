"""Coupling-interface contract for heterogeneous solver clocks (FEM / LBM / DEM).

The contract is deliberately *quantitative*: a widget-free description of what a
solver exposes to a coupling partner, so that a clock mismatch can be audited
mechanically instead of being described in prose.

Vocabulary
----------
clock
    A solver advances with a fixed fine step ``dt``.  A coupling window ``h``
    must be an integer multiple of ``dt`` (:meth:`SolverClock.ratio`); the
    integer ratio is the *clock ratio* ``r`` of the pair, the quantity the
    adaptive controller of :mod:`tensorphysics.cross_clock.clock` moves.

conserved quantity
    One of ``mass``, ``momentum``, ``energy``.  Solvers expose *interval
    integrals* of these over their own fine steps (an interval-complete ledger),
    never single point values: a point value cannot be interpolated
    conservatively, an integral can.

window
    ``[t_n, t_{n+1}]`` of the *coarse* (coupling) clock.  For every window the
    producer's interval-complete aggregate and the amount the consumer actually
    applied are both recorded in a :class:`WindowLedger`, so a mixed-up,
    duplicated or dropped share shows up as a non-zero residual.

The audit quantities of a window are kept separate on purpose:

``ledger_drift``
    ``|produced - applied| / scale``.  This is the *bookkeeping* defect (lost or
    duplicated share).  A conforming exchange drives it to round-off; it is a
    design invariant, not a physics result, and the reports say so.

``reference_drift``
    ``|applied - reference| / scale`` where ``reference`` is the same window
    integral evaluated on a monolithic single-clock run.  This one carries the
    physics content: it is what interpolation and staggering actually cost.

``stagger_residual``
    relative difference between the partner state the consumer *predicted* for
    the window and the state it ended up with after the window, i.e. the
    explicit-coupling (lag) defect.  It is not a conservation defect but it is
    the dominant term of the solution error and is therefore always recorded
    next to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

QUANTITIES: tuple[str, str, str] = ("mass", "momentum", "energy")

__all__ = [
    "QUANTITIES",
    "ContractError",
    "SolverClock",
    "IntervalSample",
    "WindowLedger",
    "ConservationPort",
    "CouplingContract",
    "check_tiling",
    "drift_report",
    "audit_record",
]


class ContractError(ValueError):
    """Raised when a coupling contract or a sample series violates its terms."""


# --------------------------------------------------------------------------- #
# clock
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SolverClock:
    """Fixed-step clock of one solver.

    ``dt`` is the solver's own fine step; ``t0`` its start time.  A coupling
    window ``h`` is admissible only if ``h / dt`` is an integer within
    ``rtol``; that integer is the clock ratio ``r``.
    """

    name: str
    dt: float
    t0: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ContractError(f"clock {self.name!r}: dt must be positive, got {self.dt!r}")

    def ratio(self, h: float, rtol: float = 1e-9) -> int:
        """Integer number of own fine steps inside the coupling window ``h``."""
        q = h / self.dt
        r = int(round(q))
        if r < 1 or abs(q - r) > rtol * max(1.0, abs(q)):
            raise ContractError(
                f"clock {self.name!r}: window {h!r} is not an integer multiple of dt={self.dt!r} "
                f"(window/dt = {q!r})"
            )
        return r

    def steps(self, h: float) -> int:
        return self.ratio(h)

    def edges(self, t0: float, h: float) -> np.ndarray:
        n = self.ratio(h)
        return t0 + self.dt * np.arange(n + 1, dtype=float)


# --------------------------------------------------------------------------- #
# samples and windows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntervalSample:
    """Interval-complete aggregate of one conserved quantity over ``[t0, t1]``.

    ``mean_rate[q]`` is the *mean rate* of that quantity over the interval, i.e.
    ``total(q) == mean_rate[q] * duration``.  Rates are the primitive because
    they are what a solver can produce from its own sub-steps without a
    quadrature assumption; totals (the conserved content) are derived.
    """

    t0: float
    t1: float
    mean_rate: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.t1 > self.t0:
            raise ContractError(f"interval must have t1 > t0, got [{self.t0!r}, {self.t1!r}]")
        for q, v in self.mean_rate.items():
            if q not in QUANTITIES:
                raise ContractError(f"unknown conserved quantity {q!r}")
            if not np.isfinite(v):
                raise ContractError(f"non-finite mean rate for {q!r}")

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def total(self, q: str) -> float:
        return float(self.mean_rate.get(q, 0.0)) * self.duration

    def centroid(self) -> float:
        """Time of the interval midpoint (the allocation point of a held rate)."""
        return 0.5 * (self.t0 + self.t1)

    def split(self, n: int) -> list["IntervalSample"]:
        """Subdivide into ``n`` equal sub-intervals with the same mean rate."""
        if n < 1:
            raise ContractError("split needs n >= 1")
        edges = np.linspace(self.t0, self.t1, n + 1)
        return [
            IntervalSample(float(edges[i]), float(edges[i + 1]), dict(self.mean_rate))
            for i in range(n)
        ]


@dataclass
class WindowLedger:
    """One coupling window: produced / applied / reference aggregates."""

    index: int
    t0: float
    t1: float
    produced: dict[str, float] = field(default_factory=dict)
    applied: dict[str, float] = field(default_factory=dict)
    reference: dict[str, float] | None = None
    stagger_residual: float | None = None
    ratio: int | None = None
    bound: float | None = None

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def residual(self, q: str) -> float:
        return float(self.applied.get(q, 0.0)) - float(self.produced.get(q, 0.0))

    def reference_residual(self, q: str) -> float | None:
        if self.reference is None:
            return None
        return float(self.applied.get(q, 0.0)) - float(self.reference.get(q, 0.0))

    @staticmethod
    def scale(produced: float, applied: float, reference: float | None, floor: float) -> float:
        """Normaliser of a drift: the largest magnitude in play, floored.

        The floor keeps the ratio meaningful when a quantity legitimately
        integrates to (near) zero over a window, e.g. the impulse of a
        half-cycle.  It must be supplied by the caller: a drift normalised by
        ``|q|`` alone can be inflated to 1.0 by a single sign change.
        """
        cands = [abs(produced), abs(applied)]
        if reference is not None:
            cands.append(abs(reference))
        return max(float(floor), *cands)

    def ledger_drift(self, q: str, floor: float) -> float:
        ref = None if self.reference is None else self.reference.get(q)
        denom = self.scale(float(self.produced.get(q, 0.0)), float(self.applied.get(q, 0.0)), ref, floor)
        return abs(self.residual(q)) / denom

    def reference_drift(self, q: str, floor: float) -> float | None:
        if self.reference is None:
            return None
        prov = float(self.produced.get(q, 0.0))
        appl = float(self.applied.get(q, 0.0))
        ref = self.reference.get(q)
        denom = self.scale(prov, appl, ref, floor)
        return abs(float(self.reference_residual(q))) / denom

    def as_dict(self) -> dict:
        out = {
            "index": self.index,
            "t0": self.t0,
            "t1": self.t1,
            "duration": self.duration,
            "ratio": self.ratio,
            "produced": dict(self.produced),
            "applied": dict(self.applied),
            "bound": self.bound,
            "stagger_residual": self.stagger_residual,
        }
        if self.reference is not None:
            out["reference"] = dict(self.reference)
        return out


# --------------------------------------------------------------------------- #
# ports
# --------------------------------------------------------------------------- #
class ConservationPort(Protocol):  # pragma: no cover - structural typing only
    """What a solver must expose to take part in a clock-mismatched exchange."""

    name: str
    clock: SolverClock

    def drain(self, t0: float, t1: float) -> IntervalSample:
        """Interval-complete aggregate of every exposed quantity over the window."""

    def apply(self, quantity: str, total: float, t0: float, t1: float) -> None:
        """Receive ``total`` of ``quantity`` distributed over ``[t0, t1]``."""


@dataclass(frozen=True)
class CouplingContract:
    """Declared terms of one solver pair.

    ``quantities`` are the exchanged conserved quantities.  ``mode`` names the
    exchange pattern, which fixes which error bound is admissible:

    ``impulse_displacement``
        work-conjugate pair (impulse -> structure, displacement -> fluid).  The
        impulse integral and the work integral are both transferred exactly per
        window; only the *distribution inside the window* is interpolated.
    ``rate_hold``
        the consumer reconstructs the partner rate as a piecewise constant per
        window (causal zeroth-order hold, second-order in the window integral).
    ``rate_linear``
        causal linear extrapolation of the partner rate from the last two
        windows (second order pointwise, but conditionally unstable: diagnostic).
    """

    producer: str
    consumer: str
    quantities: tuple[str, ...]
    tolerance: float
    mode: str = "rate_hold"
    min_ratio: int = 1
    max_ratio: int = 4096
    order: int = 2

    def __post_init__(self) -> None:
        if tuple(self.quantities) != tuple(q for q in QUANTITIES if q in self.quantities):
            raise ContractError(
                f"quantities must be a subset of {QUANTITIES} in canonical order, got {self.quantities!r}"
            )
        if not self.quantities:
            raise ContractError("a coupling contract must exchange at least one quantity")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ContractError(f"tolerance must be positive, got {self.tolerance!r}")
        if self.mode not in ("impulse_displacement", "rate_hold", "rate_linear"):
            raise ContractError(f"unknown exchange mode {self.mode!r}")
        if not 1 <= self.min_ratio <= self.max_ratio:
            raise ContractError(f"bad ratio bounds [{self.min_ratio}, {self.max_ratio}]")

    def admits(self, ratio: int) -> bool:
        return self.min_ratio <= int(ratio) <= self.max_ratio

    def as_dict(self) -> dict:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "quantities": list(self.quantities),
            "tolerance": self.tolerance,
            "mode": self.mode,
            "min_ratio": self.min_ratio,
            "max_ratio": self.max_ratio,
            "order": self.order,
        }


# --------------------------------------------------------------------------- #
# structural checks
# --------------------------------------------------------------------------- #
def check_tiling(samples: Sequence[IntervalSample], rtol: float = 1e-12) -> list[str]:
    """Return the list of structural violations of an interval-complete series.

    A conforming series tiles its time axis exactly once, in increasing order:
    no gap, no overlap, no zero/negative interval, finite rates.
    """
    problems: list[str] = []
    prev: IntervalSample | None = None
    for i, s in enumerate(samples):
        if s.t1 <= s.t0:
            problems.append(f"sample {i}: non-positive duration [{s.t0}, {s.t1}]")
        for q, v in s.mean_rate.items():
            if not np.isfinite(v):
                problems.append(f"sample {i}: non-finite rate for {q}")
        if prev is not None:
            if s.t0 < prev.t1 - rtol * max(1.0, abs(prev.t1)):
                problems.append(f"sample {i}: overlaps previous interval ({s.t0} < {prev.t1})")
            elif s.t0 > prev.t1 + rtol * max(1.0, abs(prev.t1)):
                problems.append(f"sample {i}: gap since previous interval ({s.t0} > {prev.t1})")
        prev = s
    return problems


def drift_report(windows: Iterable[WindowLedger], scale: float | None = None) -> dict:
    """Aggregate the per-window drifts of a run.

    ``scale`` (physical magnitude of the exchanged quantity per window) is
    mandatory for a meaningful number; when omitted it is taken from the data
    (max |produced|) and the report says so.
    """
    windows = list(windows)
    if not windows:
        raise ContractError("no windows to report")
    qs = sorted({q for w in windows for q in list(w.produced) + list(w.applied)})
    out: dict[str, dict] = {}
    for q in qs:
        floor = scale if scale is not None else max(
            max(abs(float(w.produced.get(q, 0.0))) for w in windows), 1e-300
        )
        led = [w.ledger_drift(q, floor) for w in windows]
        ref = [w.reference_drift(q, floor) for w in windows]
        ref = [float(v) for v in ref if v is not None]
        stag = [float(w.stagger_residual) for w in windows if w.stagger_residual is not None]
        out[q] = {
            "scale": floor,
            "ledger_drift_max": max(led),
            "ledger_drift_mean": float(np.mean(led)),
            "reference_drift_max": max(ref) if ref else None,
            "reference_drift_mean": float(np.mean(ref)) if ref else None,
            "stagger_residual_max": max(stag) if stag else None,
            "n_windows": len(windows),
        }
    return out


def audit_record(contract: CouplingContract, windows: Sequence[WindowLedger], scale: float | None = None) -> dict:
    """Machine-readable acceptance fragment for one coupled run."""
    problems = check_tiling([
        IntervalSample(w.t0, w.t1, {"mass": 0.0}) for w in windows
    ])
    rep = drift_report(windows, scale)
    return {
        "contract": contract.as_dict(),
        "windows": len(windows),
        "tiling_violations": problems,
        "drift": rep,
        "ledger_ok": all(v["ledger_drift_max"] < 1e-12 for v in rep.values()),
        "reference_ok": all(
            v["reference_drift_max"] is None or v["reference_drift_max"] < 1e-3 for v in rep.values()
        ),
        "ratios": sorted({int(w.ratio) for w in windows if w.ratio is not None}),
    }