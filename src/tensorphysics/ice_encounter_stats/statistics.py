"""Record statistics: exponentiality tests, correlation time, Campbell anchors.

The three tools used by the acceptance criteria:

``ks_exponential``
    Kolmogorov--Smirnov distance and asymptotic p-value for the composite
    hypothesis "the intervals are exponential with unknown rate" (rate estimated
    by maximum likelihood, i.e. ``1/mean``).  Estimating the parameter from the
    same sample makes the test conservative (the true p-value is *larger* than
    the reported one), so a rejection is strong evidence; a non-rejection is
    reported with that caveat, never as proof of Poisson-ness.

``correlation_time``
    Integral time scale ``tau_c = int_0^{tau0} rho(tau) dtau`` with ``tau0`` the
    first zero crossing of the autocorrelation -- the standard truncation that
    keeps the estimator from integrating noise.  The effective number of
    independent samples of a record of length ``T`` is ``N_eff = T / (2 tau_c)``.

``campbell_moments``
    Campbell's theorem for a Poisson shot noise: ``mu = lambda E[int h dt]`` and
    ``sigma^2 = lambda E[int h^2 dt]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Kolmogorov--Smirnov against the exponential
# --------------------------------------------------------------------------- #
def kolmogorov_sf(x: float) -> float:
    """Survival function Q(x) = P(sqrt(n) D_n > x) of the KS distribution."""
    if x <= 0.0:
        return 1.0
    if x > 20.0:
        return 0.0
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * x * x)
        total += term
        if abs(term) < 1e-14:
            break
    return float(min(max(total, 0.0), 1.0))


@dataclass(frozen=True)
class KSResult:
    n: int
    rate: float
    statistic: float
    p_value: float
    d_plus: float
    d_minus: float
    critical_05: float

    def to_dict(self) -> dict:
        return {
            "n": int(self.n),
            "rate": float(self.rate),
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "d_plus": float(self.d_plus),
            "d_minus": float(self.d_minus),
            "critical_05": float(self.critical_05),
            "reject_05": bool(self.p_value < 0.05),
        }


def ks_exponential(samples: Sequence[float], rate: Optional[float] = None) -> KSResult:
    """KS test of ``samples`` against Exponential(rate); rate defaults to 1/mean."""
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[x >= 0.0]
    n = int(x.size)
    if n < 5:
        raise ValueError("KS test needs at least 5 non-negative samples")
    if rate is None:
        mean = float(x.mean())
        if mean <= 0.0:
            raise ValueError("samples must have a positive mean")
        rate = 1.0 / mean
    rate = float(rate)
    cdf = -np.expm1(-rate * x)
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - cdf))
    d_minus = float(np.max(cdf - (i - 1.0) / n))
    d = max(d_plus, d_minus)
    # small-sample corrected argument (Stephens) -- the plain sqrt(n) D form is
    # anti-conservative for n of a few hundred.
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = kolmogorov_sf(lam)
    critical_05 = 1.358 / (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))
    return KSResult(n=n, rate=rate, statistic=d, p_value=p,
                    d_plus=d_plus, d_minus=d_minus, critical_05=critical_05)


def bonferroni_adjust(p_values: Sequence[float]) -> tuple:
    """(min adjusted p-value, family size) for a Bonferroni-corrected family."""
    p = np.asarray([float(v) for v in p_values], dtype=float)
    if p.size == 0:
        raise ValueError("empty p-value family")
    return float(min(1.0, p.size * np.min(p))), int(p.size)


# --------------------------------------------------------------------------- #
# autocorrelation / integral time scale
# --------------------------------------------------------------------------- #
def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalised autocorrelation rho[0..L-1] via FFT (biased, rho[0] = 1)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        raise ValueError("autocorrelation needs at least 8 samples")
    centred = x - x.mean()
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.rfft(centred, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    acf = acf / acf[0]
    if max_lag is not None:
        acf = acf[:max(2, int(max_lag))]
    return acf


def integral_time_scale(rho: np.ndarray, dt: float) -> tuple:
    """Integral time scale (first-zero-crossing truncation) and the crossing lag."""
    rho = np.asarray(rho, dtype=float)
    if rho.size < 3:
        raise ValueError("need at least 3 correlation lags")
    negative = np.flatnonzero(rho < 0.0)
    i0 = int(negative[0]) if negative.size else rho.size - 1
    if i0 < 2:
        return float(max(rho[1], 0.0) * dt), float(dt)
    segment = rho[: i0 + 1]
    # trapezoid on the positive, monotonically-decaying part
    tau_c = float(np.trapezoid(segment, dx=dt)) if hasattr(np, "trapezoid") \
        else float(np.trapz(segment, dx=dt))
    return tau_c, float(i0 * dt)


def correlation_time(t: np.ndarray, x: np.ndarray,
                     max_lag_s: Optional[float] = None) -> tuple:
    """Return (lags, rho, tau_c, zero_crossing_lag) for a uniformly sampled record."""
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    max_lag = None if max_lag_s is None else int(round(max_lag_s / dt))
    rho = autocorrelation(x, max_lag=max_lag)
    tau_c, tau0 = integral_time_scale(rho, dt)
    lags = np.arange(rho.size) * dt
    return lags, rho, tau_c, tau0


def effective_sample_size(duration_s: float, tau_c: float) -> float:
    """N_eff = T / (2 tau_c) -- number of independent integral-scale blocks."""
    if tau_c <= 0.0:
        raise ValueError("tau_c must be positive")
    return float(duration_s / (2.0 * tau_c))


def standard_error_of_mean(sigma: float, tau_c: float, duration_s: float) -> float:
    """Var(mean over T) = 2 sigma^2 tau_c / T for T >> tau_c."""
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    return float(sigma * math.sqrt(2.0 * tau_c / duration_s))


# --------------------------------------------------------------------------- #
# Campbell's theorem
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CampbellPrediction:
    rate: float
    mean: float
    variance: float

    @property
    def std(self) -> float:
        return float(math.sqrt(self.variance))

    @property
    def cv(self) -> float:
        return float(self.std / self.mean) if self.mean > 0.0 else float("nan")

    def to_dict(self) -> dict:
        return {"rate_per_s": float(self.rate), "mean_n": float(self.mean),
                "variance_n2": float(self.variance), "std_n": float(self.std),
                "cv": float(self.cv)}


def campbell_moments(rate: float, areas: np.ndarray,
                     sq_areas: np.ndarray) -> CampbellPrediction:
    """mu = lambda E[A], sigma^2 = lambda E[B] from the injected pulse marks."""
    a = np.asarray(areas, dtype=float)
    b = np.asarray(sq_areas, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("Campbell moments need at least one pulse mark")
    if a.size != b.size:
        raise ValueError("area and squared-area marks must have equal length")
    return CampbellPrediction(rate=float(rate), mean=float(rate * a.mean()),
                              variance=float(rate * b.mean()))


def relative_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return float("nan")
    return float(abs(value - reference) / abs(reference))


def mean_record_var(x: np.ndarray) -> float:
    return float(np.var(np.asarray(x, dtype=float), ddof=0))


__all__ = [
    "CampbellPrediction",
    "KSResult",
    "autocorrelation",
    "bonferroni_adjust",
    "campbell_moments",
    "correlation_time",
    "effective_sample_size",
    "integral_time_scale",
    "kolmogorov_sf",
    "ks_exponential",
    "mean_record_var",
    "relative_error",
    "standard_error_of_mean",
]