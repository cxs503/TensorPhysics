"""Pressure-area (p-A) extreme statistics — the load-side input of ISO 19906.

Ice pressure on a structural panel decreases with the loaded area according to
the ISO 19906 power law

    p(A) = p_ref * (A / A_ref) ** (-ex)                                   (1)

with an exponent ``ex`` in the 0.4 - 0.6 band for first-year ice (the in-repo
corpus asset ``benchmarks/ice_nav_refs/assets/riska_kujala_pressure_area.json``
uses ``ex = 0.5`` for the global pressure). Equation (1) describes the *central
tendency* of the pressure; the design action is an **annual maximum** and must
therefore be described by an extreme-value distribution.

This module supplies the statistically defensible route from p-A data to that
distribution:

* :class:`PressureAreaTrend` — the deterministic trend (1);
* :func:`storm_gumbel` — exact annual-maximum distribution of ``n`` independent
  storm peaks (the maximum of ``n`` Gumbel variates is Gumbel with the same
  scale and a location shifted by ``scale * ln(n)`` — the classical Gumbel
  stability property, used here as an analytic reference);
* :func:`simulate_annual_maxima` — a reproducible synthetic p-A observation
  ensemble (documented generator plant, seeded), because no field pressure
  record ships with the repository; the recovery of the known plant is the
  verification target;
* :func:`fit_gumbel_moments`, :func:`fit_gumbel_mle`,
  :func:`fit_gev_lmoments` — three independent parameter estimators;
* :func:`fit_pressure_area_trend` — log-log least squares for ``p_ref``/``ex``
  with standard errors.

Everything is pure Python (no numpy) so the fits are usable inside the
dependency-free reliability core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, gamma as _gamma, log, pi, sqrt
from random import Random
from typing import Iterable, Sequence

__all__ = [
    "EULER_GAMMA",
    "PressureAreaTrend",
    "GumbelParams",
    "GumbelFit",
    "GevFit",
    "TrendFit",
    "storm_gumbel",
    "gumbel_quantile",
    "simulate_annual_maxima",
    "fit_gumbel_moments",
    "fit_gumbel_mle",
    "fit_gev_lmoments",
    "fit_gev_mle",
    "fit_pressure_area_trend",
    "ks_statistic_gumbel",
    "plant_annual_params",
    "relative_error",
]

#: Euler-Mascheroni constant, the mean of a standard Gumbel.
EULER_GAMMA = 0.5772156649015329

_SQRT6_OVER_PI = sqrt(6.0) / pi  # Gumbel: scale = sigma * sqrt(6)/pi


# --------------------------------------------------------------------------- #
# Trend p(A)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PressureAreaTrend:
    """ISO 19906 pressure-area trend ``p(A) = p_ref (A/A_ref)**(-ex)``."""

    p_ref: float
    exponent: float
    a_ref: float = 1.0

    def __post_init__(self) -> None:
        if self.p_ref <= 0.0:
            raise ValueError("p_ref must be positive")
        if self.a_ref <= 0.0:
            raise ValueError("a_ref must be positive")
        if not -0.95 < self.exponent < 0.95:
            raise ValueError("pressure-area exponent outside the physical band")

    def mean_pressure(self, area: float) -> float:
        if area <= 0.0:
            raise ValueError("area must be positive")
        return self.p_ref * (area / self.a_ref) ** (-self.exponent)


@dataclass(frozen=True)
class TrendFit:
    """Log-log least-squares fit of the p-A trend with standard errors."""

    p_ref: float
    exponent: float
    a_ref: float
    se_log_p_ref: float
    se_exponent: float
    r_squared: float
    samples: int

    def to_dict(self) -> dict:
        return {
            "p_ref": self.p_ref,
            "exponent": self.exponent,
            "a_ref": self.a_ref,
            "se_log_p_ref": self.se_log_p_ref,
            "se_exponent": self.se_exponent,
            "r_squared": self.r_squared,
            "samples": self.samples,
        }


def fit_pressure_area_trend(
    areas: Sequence[float], pressures: Sequence[float], a_ref: float = 1.0
) -> TrendFit:
    """Least squares on ``ln p = ln p_ref - ex * ln(A/A_ref)``.

    Returns the exponent/among-point scatter statistics used to separate the
    deterministic area effect from the year-to-year extreme-value scatter.
    """
    if len(areas) != len(pressures) or len(areas) < 3:
        raise ValueError("need at least three (area, pressure) pairs")
    xs = [log(a / a_ref) for a in areas]
    ys = [log(p) for p in pressures]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        raise ValueError("areas must span a range")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    s2 = sum(r * r for r in resid) / dof
    se_slope = sqrt(s2 / sxx)
    se_intercept = sqrt(s2 * (1.0 / n + mx * mx / sxx))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - sum(r * r for r in resid) / ss_tot if ss_tot > 0 else 0.0
    return TrendFit(
        p_ref=exp(intercept),
        exponent=-slope,
        a_ref=a_ref,
        se_log_p_ref=se_intercept,
        se_exponent=se_slope,
        r_squared=r2,
        samples=n,
    )


# --------------------------------------------------------------------------- #
# Gumbel bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GumbelParams:
    """Gumbel (maxima) location/scale, i.e. ``F(x)=exp(-exp(-(x-loc)/scale))``."""

    loc: float
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")

    def mean(self) -> float:
        return self.loc + EULER_GAMMA * self.scale

    def std(self) -> float:
        return pi * self.scale / sqrt(6.0)

    def cov(self) -> float:
        return self.std() / self.mean()

    def cdf(self, x: float) -> float:
        return exp(-exp(-(x - self.loc) / self.scale))

    def quantile(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("quantile requires 0 < p < 1")
        return self.loc - self.scale * log(-log(p))

    def return_level(self, return_period: float) -> float:
        """Value exceeded with annual probability ``1/return_period``."""
        if return_period <= 1.0:
            raise ValueError("return period must exceed one year")
        return self.quantile(1.0 - 1.0 / return_period)

    def to_dict(self) -> dict:
        return {
            "loc": self.loc,
            "scale": self.scale,
            "mean": self.mean(),
            "std": self.std(),
            "cov": self.cov(),
        }


def storm_gumbel(storm: GumbelParams, storms_per_year: int) -> GumbelParams:
    """Annual maximum of ``storms_per_year`` independent storm peaks.

    ``max`` of ``n`` i.i.d. Gumbel(loc, scale) variates is Gumbel with the same
    scale and location ``loc + scale * ln(n)`` (Gumbel's stability property).
    """
    if storms_per_year < 1:
        raise ValueError("storms_per_year must be at least one")
    return GumbelParams(loc=storm.loc + storm.scale * log(storms_per_year), scale=storm.scale)


def gumbel_quantile(params: GumbelParams, p: float) -> float:
    return params.quantile(p)


@dataclass(frozen=True)
class GumbelFit:
    """A fitted Gumbel with the diagnostics used in the calibration report."""

    loc: float
    scale: float
    method: str
    samples: int
    log_likelihood: float
    ks_statistic: float
    extra: dict = field(default_factory=dict)

    @property
    def params(self) -> GumbelParams:
        return GumbelParams(self.loc, self.scale)

    def mean(self) -> float:
        return self.params.mean()

    def std(self) -> float:
        return self.params.std()

    def cov(self) -> float:
        return self.params.cov()

    def cdf(self, x: float) -> float:
        return self.params.cdf(x)

    def quantile(self, p: float) -> float:
        return self.params.quantile(p)

    def return_level(self, return_period: float) -> float:
        return self.params.return_level(return_period)

    def to_dict(self) -> dict:
        out = {
            "loc": self.loc,
            "scale": self.scale,
            "method": self.method,
            "samples": self.samples,
            "mean": self.params.mean(),
            "std": self.params.std(),
            "cov": self.params.cov(),
            "log_likelihood": self.log_likelihood,
            "ks_statistic": self.ks_statistic,
        }
        out.update(self.extra)
        return out


def _gumbel_loglik(xs: Sequence[float], loc: float, scale: float) -> float:
    total = 0.0
    for x in xs:
        y = (x - loc) / scale
        total += -(y + exp(-y))
    return total - len(xs) * log(scale)


def ks_statistic_gumbel(xs: Sequence[float], params: GumbelParams) -> float:
    """Kolmogorov-Smirnov distance of the sorted sample against Gumbel."""
    data = sorted(xs)
    n = len(data)
    d = 0.0
    for i, x in enumerate(data, start=1):
        f = params.cdf(x)
        d = max(d, abs(f - (i - 1) / n), abs(f - i / n))
    return d


def fit_gumbel_moments(xs: Sequence[float]) -> GumbelFit:
    """Method of moments: ``scale = sd sqrt(6)/pi``, ``loc = mean - gamma*scale``."""
    n = len(xs)
    if n < 3:
        raise ValueError("at least three maxima are required")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0.0:
        raise ValueError("degenerate sample (zero variance)")
    scale = sqrt(var) * _SQRT6_OVER_PI
    loc = mean - EULER_GAMMA * scale
    params = GumbelParams(loc, scale)
    return GumbelFit(
        loc=loc,
        scale=scale,
        method="moments",
        samples=n,
        log_likelihood=_gumbel_loglik(xs, loc, scale),
        ks_statistic=ks_statistic_gumbel(xs, params),
        extra={"sample_mean": mean, "sample_std": sqrt(var)},
    )


def fit_gumbel_mle(xs: Sequence[float], max_iterations: int = 200) -> GumbelFit:
    """Maximum likelihood for the Gumbel distribution.

    The location is eliminated analytically,
    ``mu(scale) = -scale ln(mean(exp(-x/scale)))``, leaving the one-dimensional
    score equation ``scale = mean((x - mu)(1 - exp(-(x-mu)/scale)))`` which is
    solved by bisection on a bracketed interval.
    """
    n = len(xs)
    if n < 3:
        raise ValueError("at least three maxima are required")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0.0:
        raise ValueError("degenerate sample (zero variance)")
    seed = sqrt(var) * _SQRT6_OVER_PI

    def mu_of(scale: float) -> float:
        acc = 0.0
        for x in xs:
            z = -x / scale
            acc += exp(z) if z > -700.0 else 0.0
        return -scale * log(max(acc / n, 1e-300))

    def score(scale: float) -> float:
        mu = mu_of(scale)
        acc = 0.0
        for x in xs:
            z = -(x - mu) / scale
            r = exp(z) if z > -700.0 else 0.0
            acc += (x - mu) * (1.0 - r)
        return scale - acc / n

    lo, hi = seed * 0.05, seed * 20.0
    f_lo, f_hi = score(lo), score(hi)
    expand = 0
    while f_lo * f_hi > 0.0 and expand < 60:
        lo *= 0.5
        hi *= 2.0
        f_lo, f_hi = score(lo), score(hi)
        expand += 1
    if f_lo * f_hi > 0.0:  # pragma: no cover - defensive
        raise RuntimeError("Gumbel ML score equation could not be bracketed")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = score(mid)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    scale = 0.5 * (lo + hi)
    loc = mu_of(scale)
    params = GumbelParams(loc, scale)
    return GumbelFit(
        loc=loc,
        scale=scale,
        method="mle",
        samples=n,
        log_likelihood=_gumbel_loglik(xs, loc, scale),
        ks_statistic=ks_statistic_gumbel(xs, params),
        extra={"sample_mean": mean, "sample_std": sqrt(var)},
    )


# --------------------------------------------------------------------------- #
# GEV (L-moments) — used to check that the Gumbel shape restriction is tenable
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GevFit:
    """GEV fit by the method of L-moments (Hosking 1985/1990)."""

    loc: float
    scale: float
    shape: float
    l_moments: tuple[float, float, float]
    samples: int
    log_likelihood: float

    def cdf(self, x: float) -> float:
        k = self.shape
        y = (x - self.loc) / self.scale
        if abs(k) < 1e-12:
            return exp(-exp(-y))
        inner = 1.0 - k * y
        if inner <= 0.0:
            return 0.0 if k > 0.0 else 1.0
        return exp(-inner ** (1.0 / k))

    def quantile(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("quantile requires 0 < p < 1")
        k = self.shape
        y = -log(p)
        if abs(k) < 1e-12:
            return self.loc - self.scale * log(y)
        return self.loc + self.scale * (1.0 - y**k) / k

    def to_dict(self) -> dict:
        return {
            "loc": self.loc,
            "scale": self.scale,
            "shape": self.shape,
            "log_likelihood": self.log_likelihood,
            "l_moments": list(self.l_moments),
        }


def _gev_loglik(xs: Sequence[float], loc: float, scale: float, shape: float) -> float:
    """GEV log-likelihood; ``-inf`` outside the support."""
    k = shape
    total = 0.0
    for x in xs:
        y = (x - loc) / scale
        if abs(k) < 1e-12:
            z = exp(-y)
            total += -(z + exp(-z))
            continue
        inner = 1.0 - k * y
        if inner <= 0.0:
            return float("-inf")
        total += ((1.0 - k) / k) * log(inner) - inner ** (1.0 / k)
    return total - len(xs) * log(scale)


def fit_gev_lmoments(xs: Sequence[float]) -> GevFit:
    """GEV by L-moments; ``shape ~ 0`` recovers the Gumbel case."""
    n = len(xs)
    if n < 10:
        raise ValueError("GEV L-moment fit needs at least ten maxima")
    data = sorted(xs)
    b0 = sum(data) / n
    b1 = sum(((i - 1) / (n - 1)) * x for i, x in enumerate(data, start=1)) / n
    b2 = (
        sum(((i - 1) * (i - 2) / ((n - 1) * (n - 2))) * x for i, x in enumerate(data, start=1))
        / n
    )
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    if l2 <= 0.0:
        raise ValueError("degenerate L-moments (l2 <= 0)")
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - log(2.0) / log(3.0)
    k = 7.8590 * c + 2.9554 * c * c
    if abs(k) < 1e-9:
        k = 1e-9
    alpha = l2 * k / ((1.0 - 2.0 ** (-k)) * _gamma(1.0 + k))
    xi = l1 - alpha * (1.0 - _gamma(1.0 + k)) / k
    return GevFit(
        loc=xi,
        scale=alpha,
        shape=k,
        l_moments=(l1, l2, l3),
        samples=n,
        log_likelihood=_gev_loglik(xs, xi, alpha, k),
    )


def fit_gev_mle(
    xs: Sequence[float], start: "GevFit | None" = None, iterations: int = 600
) -> GevFit:
    """GEV maximum likelihood by compass/pattern search from the L-moment start.

    The GEV likelihood is not needed for the calibration itself; it is used to
    make the Gumbel-vs-GEV likelihood-ratio test legitimate (a likelihood-ratio
    statistic requires both models at their own MLE).
    """
    base = start if start is not None else fit_gev_lmoments(xs)
    params = [base.loc, log(base.scale), base.shape]
    steps = [0.05 * max(base.scale, 1e-6) + 0.01 * abs(base.loc), 0.05, 0.05]
    best = _gev_loglik(xs, params[0], exp(params[1]), params[2])
    for _ in range(iterations):
        improved = False
        for i in range(3):
            for sign in (1.0, -1.0):
                trial = list(params)
                trial[i] += sign * steps[i]
                value = _gev_loglik(xs, trial[0], exp(trial[1]), trial[2])
                if value > best:
                    best, params, improved = value, trial, True
        if not improved:
            steps = [step * 0.5 for step in steps]
            if max(steps) < 1e-9:
                break
    return GevFit(
        loc=params[0],
        scale=exp(params[1]),
        shape=params[2],
        l_moments=base.l_moments,
        samples=base.samples,
        log_likelihood=best,
    )


# --------------------------------------------------------------------------- #
# Synthetic p-A observation ensemble (documented generator plant)
# --------------------------------------------------------------------------- #
def simulate_annual_maxima(
    trend: PressureAreaTrend,
    areas: Iterable[float],
    years: int,
    storm_scale_cov: float = 0.22,
    storms_per_year: int = 12,
    area_scatter_cov: float = 0.06,
    seed: int = 20261031,
) -> dict[float, list[float]]:
    """Simulate one annual-maximum pressure record per area.

    Generator (plant) — fully documented so that recovery is testable:

    1. storm peak at area ``A``: ``Gumbel(loc = m(A), scale = cov * m(A) *
       sqrt(6)/pi)`` with ``m(A) = trend.mean_pressure(A)`` (moment-matched to a
       coefficient of variation ``storm_scale_cov``);
    2. per-event area/contact scatter ``exp(N(0, area_scatter_cov))`` applied to
       the pressure (same area, different indentation geometry);
    3. the annual maximum is the largest of ``storms_per_year`` independent
       events, which is exactly ``Gumbel(loc + scale ln n, scale)`` in the
       limit of no area scatter.

    The record is *model-based* (no field pressure measurements ship with the
    repository); the plant parameters are the verification target of the fits.
    """
    if years < 3:
        raise ValueError("at least three years are required")
    if storms_per_year < 1:
        raise ValueError("storms_per_year must be at least one")
    if storm_scale_cov <= 0.0:
        raise ValueError("storm_scale_cov must be positive")
    rng = Random(seed)
    out: dict[float, list[float]] = {}
    for area in areas:
        mean = trend.mean_pressure(float(area))
        scale = mean * storm_scale_cov * _SQRT6_OVER_PI
        loc = mean - EULER_GAMMA * scale
        record: list[float] = []
        for _ in range(years):
            # uniform -> Gumbel by inverse CDF (clipped away from 0/1)
            best = -1.0
            for _ in range(storms_per_year):
                u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
                g = loc - scale * log(-log(u))
                if area_scatter_cov > 0.0:
                    z = rng.gauss(0.0, 1.0)
                    g *= exp(area_scatter_cov * z)
                if g > best:
                    best = g
            record.append(best)
        out[float(area)] = record
    return out


def plant_annual_params(
    trend: PressureAreaTrend,
    area: float,
    storm_scale_cov: float = 0.22,
    storms_per_year: int = 12,
) -> GumbelParams:
    """Analytic annual-maximum Gumbel of the generator plant (no scatter term)."""
    mean = trend.mean_pressure(area)
    scale = mean * storm_scale_cov * _SQRT6_OVER_PI
    loc = mean - EULER_GAMMA * scale
    return storm_gumbel(GumbelParams(loc, scale), storms_per_year)


def relative_error(value: float, reference: float) -> float:
    """Signed relative error ``(value - reference) / reference``."""
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return (value - reference) / reference