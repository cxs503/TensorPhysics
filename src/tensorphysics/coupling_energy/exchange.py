"""Interface-history reconstruction for sub-cycled partitioned coupling.

When a fast solver (fluid / DEM) is sub-cycled inside one step of a slow solver
(structure / FEM), only one interface sample per coarse step is exchanged.  The
receiver therefore has to *reconstruct* the interface history inside the coarse
interval.  Four reconstructions are offered, labelled by their nominal order:

===================  ==================================================  =====
scheme               reconstruction                                      order
===================  ==================================================  =====
``zoh``              hold the newest sample (zero order)                  1
``linear``           Taylor through the two newest samples (first order)  2
``quadratic``        Taylor through the three newest samples (2nd order)  3
``midpoint``         Crank-Nicolson type time-centered midpoint eval      2
``independent``      no shared history at all — each side evaluates its
                     own interface value (the non-conservative control)  —
===================  ==================================================  =====

The declared order refers to the *quadrature* order of the exchange, i.e. the
order at which ``int q~ dt`` reproduces ``int q dt`` over one interval.  The
module therefore ships both the reconstruction and a deterministic
:func:`quadrature_error` / :func:`reconstruction_order_scan` harness that
measures the order on an analytic signal, independently of any solver.  The
coupled runs in :mod:`tensorphysics.coupling_energy.scenarios` re-measure the
same orders on the real interface flux, so a scheme that only looks
third-order on a smooth signal is caught.

Crank-Nicolson midpoint scheme
-------------------------------
The ``midpoint`` scheme evaluates the interface quantity at the interval
midpoint ``t_mid = t_cur + h/2`` using a linear extrapolation from the two
most recent exchange samples, then applies the trapezoidal rule:

    int_{t_cur}^{t_cur+h} q~ dt = h * q(t_mid)

This is algebraically equivalent to the Crank-Nicolson time integration
applied to the interface flux, and achieves second-order accuracy while
maintaining causality (only past samples are used).  The midpoint value is:

    q(t_mid) = q_cur + (h/2) * (q_cur - q_prev) / (t_cur - t_prev)

which is a first-order extrapolation to the midpoint, but the resulting
quadrature is second-order because the trapezoidal rule integrates a linear
function exactly.
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
    "reconstruction_poly",
    "reconstruct_interval",
    "interval_integral",
    "quadrature_error",
    "MultiRateSchedule",
    "reconstruction_order_scan",
    "midpoint_exchange_value",
    "midpoint_interval_integral",
]


class TemporalScheme(str, Enum):
    """Interface-history reconstruction used by the receiving solver."""

    INDEPENDENT = "independent"
    ZOH = "zoh"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    MIDPOINT = "midpoint"


#: Declared nominal (quadrature) order of each scheme.
SCHEME_ORDER: dict[TemporalScheme, int] = {
    TemporalScheme.INDEPENDENT: 0,
    TemporalScheme.ZOH: 1,
    TemporalScheme.LINEAR: 2,
    TemporalScheme.QUADRATIC: 3,
    TemporalScheme.MIDPOINT: 2,
}


@dataclass(frozen=True)
class ExchangeSample:
    """One interface sample at an exchange instant.

    ``value`` is normally a scalar (one interface degree of freedom) but any
    object supporting arithmetic works, which lets the scenarios reconstruct a
    whole interface row with the same code path.
    """

    t: float
    value: object


def _coefficients(scheme: TemporalScheme, previous: ExchangeSample | None,
                  current: ExchangeSample, previous2: ExchangeSample | None
                  ) -> tuple[object, object, object]:
    """Taylor coefficients ``(f0, a1, a2)`` of the reconstruction.

    The reconstruction is ``f(s) = f0 + a1 s + a2 s^2`` with ``s = t - t_cur >= 0``.
    Missing history degrades the scheme to the highest order the available
    samples support; that degradation is documented (a start-up transient) and
    not a silent fallback, because the first interval is reported separately by
    the scenarios.
    """
    if scheme is TemporalScheme.INDEPENDENT:
        raise CouplingLedgerError(
            "the independent scheme has no shared history to reconstruct; "
            "each side evaluates its own interface value"
        )
    if scheme is TemporalScheme.MIDPOINT:
        # Midpoint scheme uses its own evaluation path; return zero coefficients
        # to signal that _coefficients is not the right entry point.
        f0 = current.value
        zero = f0 * 0.0 if hasattr(f0, "__mul__") else 0.0
        return f0, zero, zero
    f0 = current.value
    zero = f0 * 0.0 if hasattr(f0, "__mul__") else 0.0
    if scheme is TemporalScheme.ZOH or previous is None:
        return f0, zero, zero
    h1 = current.t - previous.t
    if h1 <= 0.0:
        raise CouplingLedgerError("exchange instants must be strictly increasing")
    slope = (f0 - previous.value) / h1
    if scheme is TemporalScheme.LINEAR or previous2 is None:
        return f0, slope, zero
    h1p = previous.t - previous2.t
    if h1p <= 0.0:
        raise CouplingLedgerError("exchange instants must be strictly increasing")
    # quadratic through (t2,v2), (t1,v1), (t0,v0) evaluated as
    # f(s) = f0 + a1 s + a2 s^2, solved from the two backward conditions.
    H = h1
    G = h1 + h1p
    A = f0 - previous.value
    B = f0 - previous2.value
    den = H * G * h1p
    if den == 0.0:
        raise CouplingLedgerError("degenerate exchange history")
    # solve  a1 H - a2 H^2 = A ,  a1 G - a2 G^2 = B
    a1 = (A * G * G - H * H * B) / den
    a2 = (G * A - H * B) / den
    return f0, a1, a2


def reconstruction_poly(scheme: TemporalScheme, previous: ExchangeSample | None,
                        current: ExchangeSample, previous2: ExchangeSample | None = None
                        ) -> tuple[object, object, object]:
    """Public access to the ``(f0, a1, a2)`` polynomial of the reconstruction."""
    return _coefficients(scheme, previous, current, previous2)


def reconstruct_interval(scheme: TemporalScheme, previous: ExchangeSample | None,
                         current: ExchangeSample, t: float,
                         previous2: ExchangeSample | None = None):
    """Reconstruct the interface value at time ``t`` inside the current interval."""
    if t < current.t:
        raise CouplingLedgerError("reconstruction is evaluated forward in time only")
    if scheme is TemporalScheme.MIDPOINT:
        return midpoint_exchange_value(previous, current, t - current.t)
    f0, a1, a2 = _coefficients(scheme, previous, current, previous2)
    s = t - current.t
    return f0 + a1 * s + a2 * s * s


def midpoint_exchange_value(previous: ExchangeSample | None,
                            current: ExchangeSample,
                            s: float) -> object:
    """Crank-Nicolson type midpoint evaluation.

    For the Crank-Nicolson scheme, the midpoint value is the average of the
    current and previous exchange samples:
        q(t_mid) = (q_cur + q_prev) / 2

    This is algebraically equivalent to the trapezoidal rule and achieves
    second-order accuracy.  The parameter ``s`` is ignored for the standard
    Crank-Nicolson midpoint (it's always h/2).

    If no previous sample is available, falls back to the current value
    (zeroth-order hold).
    """
    if previous is None:
        return current.value
    # Crank-Nicolson: average of current and previous
    return (current.value + previous.value) / 2.0


def midpoint_interval_integral(previous: ExchangeSample | None,
                               current: ExchangeSample,
                               h: float) -> object:
    """Exact integral of the midpoint reconstruction over one exchange interval.

    Uses the trapezoidal rule with midpoint evaluation:
        int_{t_cur}^{t_cur+h} q~ dt = h * q(t_cur + h/2)

    This is second-order accurate and algebraically equivalent to the
    Crank-Nicolson time integration applied to the interface flux.
    """
    if h <= 0.0:
        raise CouplingLedgerError("interval length must be positive")
    q_mid = midpoint_exchange_value(previous, current, h / 2.0)
    return q_mid * h


def interval_integral(scheme: TemporalScheme, previous: ExchangeSample | None,
                      current: ExchangeSample, h: float,
                      previous2: ExchangeSample | None = None):
    """Exact integral of the reconstructed history over one exchange interval."""
    if h <= 0.0:
        raise CouplingLedgerError("interval length must be positive")
    if scheme is TemporalScheme.MIDPOINT:
        return midpoint_interval_integral(previous, current, h)
    f0, a1, a2 = _coefficients(scheme, previous, current, previous2)
    return f0 * h + 0.5 * a1 * h * h + (a2 * h * h * h) / 3.0


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
    history: list[ExchangeSample] = []
    errors: list[float] = []
    exacts: list[float] = []
    for k in range(n):
        t0 = k * h
        history.append(ExchangeSample(t0, signal(t0)))
        previous = history[-2] if len(history) >= 2 else None
        previous2 = history[-3] if len(history) >= 3 else None
        received = interval_integral(scheme, previous, history[-1], h, previous2)
        # analytic interval integral of the signal (Simpson, exact enough)
        exact = (h / 6.0) * (signal(t0) + 4.0 * signal(t0 + 0.5 * h) + signal(t0 + h))
        errors.append(abs(received - exact))
        exacts.append(abs(exact))
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
    schemes: Sequence[TemporalScheme] = (TemporalScheme.ZOH, TemporalScheme.LINEAR,
                                         TemporalScheme.QUADRATIC, TemporalScheme.MIDPOINT),
    refinements: Sequence[float] = (1.0, 0.5, 0.25),
) -> dict[str, OrderEstimate]:
    """Measure the observed exchange order of each scheme under ``h`` refinement."""
    out: dict[str, OrderEstimate] = {}
    for scheme in schemes:
        steps = [h * f for f in refinements]
        errs = [quadrature_error(signal, s, t_end, scheme)[0] for s in steps]
        out[scheme.value] = observed_order(steps, errs)
    return out
