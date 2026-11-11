"""Error / cost distributions and the conformal order statistics behind them.

Everything in this module is distribution-shaped but domain-free: a *prior* is a
two-parameter log-normal law for a non-negative quantity (a relative error
magnitude or a CPU cost), and a *calibrated* view is the empirical law of a
residual sample.

Two conventions matter for the rest of the package:

* relative errors are handled as **magnitudes** ``r = |y_model - y_ref| / y_ref``
  (always positive), so the quantities of interest are one-sided risk bounds
  ``P(r <= eps)``;
* an interval on ``r`` is reported as a **one-sided upper bound** at a level
  ``p`` obtained from the conformal order statistic
  ``k = ceil((n + 1) p)`` of the residual sample.  Under exchangeability of the
  residual and the future observation this bound has finite-sample coverage
  ``>= p`` -- no distributional assumption, which is exactly what makes the
  recalibration in :mod:`tensorphysics.multifidelity.calibration` able to repair
  a mis-registered prior.
"""
from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Sequence

_NORMAL = NormalDist()


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return _NORMAL.cdf(x)


def normal_ppf(p: float) -> float:
    """Standard normal quantile; raises for ``p`` outside (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p!r}")
    return _NORMAL.inv_cdf(p)


def _check_level(level: float, name: str = "level") -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {level!r}")
    return level


@dataclass(frozen=True)
class LognormalPrior:
    """Log-normal prior for a positive quantity, parameterised by its median.

    ``median`` is the 50 % quantile (interpretable: "typical value"), ``sigma``
    is the shape parameter of ``ln X``.  Storing the median rather than the mean
    keeps the model registry readable and avoids the ``sigma**2/2`` drift that
    makes a "5 % error model" silently become a 5.5 % one.
    """

    median: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.median) or self.median <= 0.0:
            raise ValueError(f"median must be finite and > 0, got {self.median!r}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {self.sigma!r}")

    @property
    def mean(self) -> float:
        return self.median * math.exp(0.5 * self.sigma ** 2)

    def prob_le(self, x: float) -> float:
        """``P(X <= x)``."""
        if x <= 0.0:
            return 0.0
        if self.sigma == 0.0:
            return 1.0 if x >= self.median else 0.0
        return normal_cdf(math.log(x / self.median) / self.sigma)

    def quantile(self, p: float) -> float:
        _check_level(p, "p")
        return self.median * math.exp(self.sigma * normal_ppf(p))

    def upper_bound(self, level: float) -> float:
        """One-sided upper bound with coverage ``level``."""
        return self.quantile(level)

    def interval(self, level: float) -> tuple[float, float]:
        _check_level(level)
        tail = 0.5 * (1.0 - level)
        return self.quantile(tail), self.quantile(1.0 - tail)

    def sample(self, rng, size: int = 1):
        """Draw ``size`` samples (numpy array); ``rng`` is a ``numpy.random.Generator``."""
        if self.sigma == 0.0:
            return [self.median] * size
        return list(rng.lognormal(mean=math.log(self.median), sigma=self.sigma, size=size))

    def as_dict(self) -> Dict[str, float]:
        return {"median": self.median, "sigma": self.sigma, "mean": self.mean}


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Empirical law of a residual sample with conformal order statistics.

    ``conformal_upper(level)`` returns the ``ceil((n+1) level)``-th smallest
    value, i.e. the split-conformal one-sided upper prediction bound at
    ``level`` (Vovk et al. 2005; Lei et al. 2018 for the regression form).
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("EmpiricalDistribution needs at least one value")
        vals = tuple(sorted(float(v) for v in self.values))
        for v in vals:
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"residuals must be finite and >= 0, got {v!r}")
        object.__setattr__(self, "values", vals)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "EmpiricalDistribution":
        return cls(tuple(float(v) for v in values))

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    @property
    def mean(self) -> float:
        return sum(self.values) / self.n

    def prob_le(self, x: float) -> float:
        """Plain empirical CDF ``#{v <= x} / n`` (no smoothing, deliberately
        conservative when combined with the conformal upper bound)."""
        if x <= 0.0:
            return 0.0
        return bisect.bisect_right(self.values, float(x)) / self.n

    def quantile(self, p: float) -> float:
        """Linearly interpolated empirical quantile."""
        _check_level(p, "p")
        pos = p * (self.n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, self.n - 1)
        frac = pos - lo
        return self.values[lo] * (1.0 - frac) + self.values[hi] * frac

    def conformal_upper(self, level: float) -> float:
        """Split-conformal one-sided upper bound (finite-sample, no assumption)."""
        _check_level(level)
        k = math.ceil((self.n + 1) * level)
        k = min(max(k, 1), self.n)
        return self.values[k - 1]

    def interval(self, level: float) -> tuple[float, float]:
        _check_level(level)
        tail = 0.5 * (1.0 - level)
        return self.quantile(tail), self.conformal_upper(1.0 - tail)

    def upper_bound(self, level: float) -> float:
        return self.conformal_upper(level)

    def digest(self) -> str:
        h = hashlib.sha256()
        for v in self.values:
            h.update(f"{v:.12e};".encode())
        return h.hexdigest()

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "min": self.values[0],
            "median": self.median,
            "max": self.values[-1],
            "mean": self.mean,
            "digest": self.digest(),
        }


def coverage_rate(realised: Sequence[float], bounds: Sequence[float]) -> float:
    """Fraction of realisations below their (per-query) bound."""
    if len(realised) != len(bounds):
        raise ValueError("realised and bounds must have the same length")
    n = len(realised)
    if n == 0:
        raise ValueError("coverage_rate needs at least one realisation")
    hits = sum(1 for r, b in zip(realised, bounds) if r <= b)
    return hits / n


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (used for reporting only)."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = normal_ppf(0.5 + 0.5 * level)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)