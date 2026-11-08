"""Multirate exchange operators: sub-cycling and temporal reconstruction.

Between two solvers with different stable time steps the partner data are only
available at *exchange instants* ``t_k = k h``.  The receiving solver needs an
interface history over ``[t_k, t_k+h]`` and the reconstruction rule decides the
order of the resulting exchange error:

===============  ===========================================================
scheme           interface history used by the receiver
===============  ===========================================================
``independent``  no shared history at all: each side evaluates the interface
                 flux/force from its own field.  This is the classic
                 Dirichlet-Dirichlet partitioned coupling and its conservation
                 defect is a *spatial* consistency error, i.e. O(1) in ``dt``.
``zoh``          shared flux held over the exchange interval (causal, first
                 order: the held value is stale by up to ``h``).
``linear``       shared flux continued with the slope of the last two samples
                 (causal *linear extrapolation*; second order for smooth
                 histories and it needs no future sample, unlike trapezoidal
                 time interpolation).
===============  ===========================================================

The module provides the three reconstruction rules plus a deterministic
quadrature harness (:func:`quadrature_error`) that measures the order of each
rule on an analytic signal, independently of any solver.  The coupled thermal
run in :mod:`tensorphysics.coupling_energy.scenarios` re-measures the same
orders on the real interface flux, so a scheme that only looks second order on
a smooth signal is caught.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from .ledger import CouplingLedgerError, OrderEstimate, observed_order

__all__ = [
    "TemporalScheme",
    "SCHEME_ORDER",
    "ExchangeSample",
    "reconstruct_interval",
    "quadrature_error",
    "MultiRateSchedule",
    "reconstruction_order_scan",
]


class TemporalScheme(str, Enum):
    """Interface-history reconstruction used by the receiving solver."""

    INDEPENDENT = "independent"
    ZOH = "zoh"
    LINEAR = "linear"


#: Declared nominal order of each scheme (``independent`` has no rate).
SCHEME_ORDER: dict[TemporalScheme, int] = {
    TemporalScheme.INDEPENDENT: 0,
    TemporalScheme.ZOH: 1,
    TemporalScheme.LINEAR: 2,
}


@dataclass(frozen=True)
class ExchangeSample:
    """One interface sample at an exchange instant."""

    t: float
    value: float


def reconstruct_interval(scheme: TemporalScheme, previous: ExchangeSample | None,
                         current: ExchangeSample, t: float) -> float:
    """Reconstruct the interface value at time ``t`` inside the current interval.

    ``previous`` may be ``None`` for the very first interval: both ``zoh`` and
    ``linear`` then degrade to the sample value (a documented start-up
    transient, not a silent fallback).
    """
    if scheme is TemporalScheme.INDEPENDENT:
        raise CouplingLedgerError(
            "the independent scheme has no shared history to reconstruct; "
            "each side evaluates its own interface value"
        )
    if t < current.t:
        raise CouplingLedgerError("reconstruction is evaluated forward in time only")
    if scheme is TemporalScheme.ZOH or previous is None:
        return current.value
    dt = current.t - previous.t
    if dt <= 0.0:
        raise CouplingLedgerError("exchange instants must be strictly increasing")
    slope = (current.value - previous.value) / dt
    return current.value + slope * (t - current.t)


def _interval_integrals(scheme: TemporalScheme, previous: ExchangeSample | None,
                        current: ExchangeSample, h: float) -> float:
    """Exact integral of the reconstructed history over one exchange interval."""
    if scheme is TemporalScheme.INDEPENDENT:
        raise CouplingLedgerError("independent scheme integrates its own history")
    q0 = reconstruct_interval(scheme, previous, current, current.t)
    q1 = reconstruct_interval(scheme, previous, current, current.t + h)
    return 0.5 * (q0 + q1) * h


def quadrature_error(signal: Callable[[float], float], h: float, t_end: float,
                     scheme: TemporalScheme) -> tuple[float, list[float], list[float]]:
    """Exchange-quadrature error of ``scheme`` for a smooth analytic signal.

    The receiver only ever sees the reconstructed history, so the energy it
    accepts per interval is ``int q~ dt``.  The returned error is

        eps = sum_k |int_{I_k} q~ - int_{I_k} q_exact| / sum_k |int_{I_k} q_exact|

    which is the *relative non-conservation* of the exchange itself, evaluated
    without any solver in the loop (deterministic and round-off clean).
    """
    if h <= 0.0 or t_end <= 0.0:
        raise CouplingLedgerError("h and t_end must be positive")
    if scheme is TemporalScheme.INDEPENDENT:
        raise CouplingLedgerError("independent scheme has no shared history")
    n = int(round(t_end / h))
    if n < 1:
        raise CouplingLedgerError("at least one exchange interval is required")
    previous: ExchangeSample | None = None
    errors: list[float] = []
    exacts: list[float] = []
    for k in range(n):
        t0 = k * h
        current = ExchangeSample(t0, signal(t0))
        received = _interval_integrals(scheme, previous, current, h)
        # analytic interval integral of the signal (Simpson, exact enough)
        exact = (h / 6.0) * (signal(t0) + 4.0 * signal(t0 + 0.5 * h) + signal(t0 + h))
        errors.append(abs(received - exact))
        exacts.append(abs(exact))
        previous = current
    denom = sum(exacts)
    if denom <= 0.0:
        raise CouplingLedgerError("degenerate signal: zero total exchange")
    return sum(errors) / denom, errors, exacts


@dataclass(frozen=True)
class MultiRateSchedule:
    """Exchange schedule for ratio-``r`` sub-cycling.

    ``dt_fine`` is the stable step of the fast solver (fluid / DEM), the slow
    solver uses ``h = ratio * dt_fine`` and the two sides exchange once per
    coarse step.
    """

    dt_fine: float
    ratio: int

    def __post_init__(self) -> None:
        if self.ratio < 1:
            raise CouplingLedgerError("sub-cycling ratio must be >= 1")
        if self.dt_fine <= 0.0:
            raise CouplingLedgerError("dt_fine must be positive")

    @property
    def dt_coarse(self) -> float:
        return self.dt_fine * self.ratio

    def exchange_instants(self, n_coarse_steps: int) -> list[float]:
        return [k * self.dt_coarse for k in range(n_coarse_steps + 1)]


def reconstruction_order_scan(
    signal: Callable[[float], float],
    h: float,
    t_end: float,
    schemes: Sequence[TemporalScheme] = (TemporalScheme.ZOH, TemporalScheme.LINEAR),
    refinements: Sequence[float] = (1.0, 0.5, 0.25),
) -> dict[str, OrderEstimate]:
    """Measure the observed exchange order of each scheme under ``h`` refinement."""
    out: dict[str, OrderEstimate] = {}
    for scheme in schemes:
        steps = [h * f for f in refinements]
        errs = [quadrature_error(signal, s, t_end, scheme)[0] for s in steps]
        out[scheme.value] = observed_order(steps, errs)
    return out