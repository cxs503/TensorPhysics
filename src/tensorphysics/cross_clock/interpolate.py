"""Causal in-window reconstruction of a partner rate, with computable error bounds.

The consumer of a clock-mismatched exchange must evaluate the partner's rate
*inside* a window it did not resolve.  Only past data may be used (a partitioned
code cannot wait for the future), so the admissible family is a hold or an
extrapolation, not an interpolation.

Schemes
-------
``hold``
    The window mean rate is held constant across the window.  The window
    *integral* is exact (that is the conservation property), the *pointwise*
    error is ``O(h)``.
``linear``
    The rate is extrapolated from the last two window means.  Pointwise
    ``O(h^2)`` but conditionally unstable for oscillatory partners (it can
    amplify a sign flip window to window), so it is exposed as a diagnostic
    only; the coupling manager refuses it unless explicitly asked.

Computable bounds
-----------------
The bound uses only data the coupling manager already holds: the producer's own
fine samples inside the window.  With ``M1 = max|Δr_fine| / dt_fine`` the
classical mean-value estimate for holding an interval mean gives

* pointwise  ``|r - r̄| <= M1 * h / 2``
* window integral of the absolute error ``<= M1 * h^2 / 2``

Both are *sound* (never below the true value) and, for smooth signals, within a
small factor of it; ``tests/test_cross_clock_interpolate.py`` measures the ratio
over a manufactured solution so that a bound that is merely a curve fit would be
visible.  For ``linear`` the analogous curvature-based bound uses the second
difference of the fine samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contract import ContractError, IntervalSample

__all__ = [
    "HoldRate",
    "LinearRate",
    "reconstruct",
    "fine_rate_variation",
    "fine_rate_curvature",
    "interpolation_bounds",
    "measured_error",
    "bound_report",
]


def _window(h: float) -> float:
    if not np.isfinite(h) or h <= 0.0:
        raise ContractError(f"window must be positive, got {h!r}")
    return float(h)


@dataclass(frozen=True)
class HoldRate:
    """Piecewise-constant reconstruction of a partner rate over one window."""

    mean_rate: float
    h: float

    def __post_init__(self) -> None:
        _window(self.h)

    def value(self, t: float, t0: float = 0.0) -> float:
        return float(self.mean_rate)

    def integral(self, a: float, b: float) -> float:
        return float(self.mean_rate) * (b - a)

    @property
    def order(self) -> int:
        return 1


@dataclass(frozen=True)
class LinearRate:
    """Causal linear extrapolation of the partner rate from the last two windows."""

    mean_rate: float
    previous_rate: float
    h: float

    def __post_init__(self) -> None:
        _window(self.h)

    @property
    def slope(self) -> float:
        return (self.mean_rate - self.previous_rate) / self.h

    def value(self, t: float, t0: float = 0.0) -> float:
        return float(self.mean_rate) + self.slope * (t - t0)

    def integral(self, a: float, b: float, t0: float = 0.0) -> float:
        mid = 0.5 * (a + b)
        return float(self.value(mid, t0)) * (b - a)

    @property
    def order(self) -> int:
        return 2


def reconstruct(history: Sequence[IntervalSample], quantity: str, scheme: str = "hold"):
    """Build the causal reconstruction for the *next* window from history."""
    if not history:
        raise ContractError("reconstruction needs at least one finished window")
    last = history[-1]
    h = last.duration
    if scheme == "hold":
        return HoldRate(float(last.mean_rate.get(quantity, 0.0)), h)
    if scheme == "linear":
        if len(history) < 2:
            return HoldRate(float(last.mean_rate.get(quantity, 0.0)), h)
        prev = history[-2]
        return LinearRate(
            float(last.mean_rate.get(quantity, 0.0)),
            float(prev.mean_rate.get(quantity, 0.0)),
            h,
        )
    raise ContractError(f"unknown reconstruction scheme {scheme!r}")


# --------------------------------------------------------------------------- #
# bounds from the producer's fine samples
# --------------------------------------------------------------------------- #
def fine_rate_variation(times: Sequence[float], rates: Sequence[float]) -> float:
    """``max |Δr| / Δt`` over the fine samples (units of rate per time)."""
    t = np.asarray(times, dtype=float).ravel()
    r = np.asarray(rates, dtype=float).ravel()
    if t.size != r.size or t.size < 2:
        return 0.0
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ContractError("fine sample times must increase")
    return float(np.max(np.abs(np.diff(r)) / dt))


def fine_rate_curvature(times: Sequence[float], rates: Sequence[float]) -> float:
    """``max |Δ²r| / Δt²`` over the fine samples (units of rate per time^2)."""
    t = np.asarray(times, dtype=float).ravel()
    r = np.asarray(rates, dtype=float).ravel()
    if t.size < 3:
        return 0.0
    h = np.diff(t)
    hc = 0.5 * (h[1:] + h[:-1])
    d2 = (r[2:] - 2.0 * r[1:-1] + r[:-2]) / (hc * hc)
    return float(np.max(np.abs(d2)))


def interpolation_bounds(
    times: Sequence[float],
    rates: Sequence[float],
    h: float,
    scheme: str = "hold",
    estimator: str = "fine",
    previous_mean: float | None = None,
) -> dict:
    """Sound bounds on the reconstruction error over a window of width ``h``.

    Two estimators, both *computable*, with different information requirements:

    ``fine``
        uses the producer's own fine samples inside the window
        (``M1 = max|dr/dt|``).  Sound and tight, but it needs the sub-window
        samples, i.e. it can only be applied a posteriori by the coupling
        manager.  It also inherits any lattice-scale noise in the producer,
        which is why it is not the default.
    ``aggregate``
        uses only two consecutive *window means* -- the information a consumer
        that receives nothing but aggregates actually has.  The in-window
        variation is estimated as ``|r̄_n - r̄_{n-1}| / h``; the resulting
        bound is ``|Δr̄|/2`` pointwise and ``|Δr̄| h / 2`` on the window
        integral, i.e. the relative bound is ``|Δr̄| / (2 |r̄|)``.

    Returns the bounds plus the raw quantities they were built from, so the
    bound can be audited independently.
    """
    h = _window(h)
    if estimator == "aggregate":
        if previous_mean is None:
            raise ContractError("the aggregate estimator needs the previous window mean")
        delta = abs(float(rates[0]) - float(previous_mean))
        m1 = delta / h
        m2 = delta / (h * h)
    elif estimator == "fine":
        m1 = fine_rate_variation(times, rates)
        m2 = fine_rate_curvature(times, rates)
    else:
        raise ContractError(f"unknown estimator {estimator!r}")
    if scheme == "hold":
        # |r - r̄| <= M1 h/2 pointwise, hence ∫|r - r̄| <= M1 h^2 / 2.  The sharp
        # constant 1/4 only holds for a *linear* profile; using it would make the
        # bound unsound on convex windows (measured: 1 % violations).
        pointwise = 0.5 * m1 * h
        integral = 0.5 * m1 * h * h
    elif scheme == "linear":
        pointwise = 0.5 * m2 * h * h
        integral = 0.5 * m2 * h * h * h
    else:
        raise ContractError(f"unknown reconstruction scheme {scheme!r}")
    return {
        "scheme": scheme,
        "estimator": estimator,
        "h": h,
        "variation_M1": m1,
        "curvature_M2": m2,
        "pointwise_bound": float(pointwise),
        "integral_bound": float(integral),
    }


def measured_error(
    times: Sequence[float],
    rates: Sequence[float],
    t0: float,
    h: float,
    scheme: str = "hold",
    previous_rate: float | None = None,
) -> dict:
    """True reconstruction error for a signal given by *fine samples*.

    The reference is the piecewise-linear interpolant of the fine samples, i.e.
    the finest information the producer has; measuring against anything coarser
    would hide the reconstruction error we are bounding.
    """
    t = np.asarray(times, dtype=float).ravel()
    r = np.asarray(rates, dtype=float).ravel()
    h = _window(h)
    # the window mean must be taken over the *closed* window and normalised by the
    # span actually sampled: with a half-open mask the trapezoid misses one
    # interval and under-reports the mean by dt/h (1 % at 100 samples).
    inner = (t >= t0) & (t <= t0 + h)
    if np.sum(inner) >= 2:
        ts, rs = t[inner], r[inner]
        # trapezoid over the span actually sampled: rescaling by h/span would be a
        # bias *injection* (it shrinks the mean of a positive signal by dt/h)
        mean = float(np.trapezoid(rs, ts) / (ts[-1] - ts[0]))
    else:
        mean = float(np.interp(t0 + 0.5 * h, t, r))
    if scheme == "hold":
        rec = np.full_like(t, mean)
    elif scheme == "linear":
        prev = mean if previous_rate is None else float(previous_rate)
        slope = (mean - prev) / h
        rec = mean + slope * (t - t0)
    else:
        raise ContractError(f"unknown scheme {scheme!r}")
    mask = (t >= t0) & (t <= t0 + h)
    if not np.any(mask):
        raise ContractError("no fine samples inside the window")
    err = rec[mask] - r[mask]
    return {
        "window": [t0, t0 + h],
        "pointwise_error": float(np.max(np.abs(err))),
        "integral_error": float(np.trapezoid(np.abs(err), t[mask])),
        "mean_rate": mean,
    }


def bound_report(
    times: Sequence[float],
    rates: Sequence[float],
    h: float,
    scheme: str = "hold",
    tolerance: float | None = None,
) -> dict:
    """Bound + soundness metadata for one window, ready for an acceptance record."""
    b = interpolation_bounds(times, rates, h, scheme)
    out = dict(b)
    if tolerance is not None:
        out["tolerance"] = float(tolerance)
        out["within_tolerance"] = bool(b["integral_bound"] <= tolerance)
    return out