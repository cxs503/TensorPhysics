"""Load extrapolation: Weibull/Rayleigh fitting and Gumbel extreme value.

Implements:
- Weibull distribution fitting (2-parameter)
- Rayleigh distribution fitting (special case of Weibull with k=2)
- Gumbel extreme value extrapolation for long-term prediction
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import stats


@dataclass
class WeibullFitResult:
    """Result of Weibull distribution fitting."""
    shape: float       # k (shape parameter)
    scale: float       # lambda (scale parameter)
    ks_statistic: float
    ks_pvalue: float
    is_rayleigh: bool  # True if shape ≈ 2


@dataclass
class GumbelResult:
    """Result of Gumbel extreme value extrapolation."""
    location: float    # mu (location parameter)
    scale: float       # beta (scale parameter)
    return_period: float  # years
    extreme_value: float  # predicted extreme for return period


def weibull_fit(data: np.ndarray) -> WeibullFitResult:
    """Fit a 2-parameter Weibull distribution to data.

    Parameters
    ----------
    data : np.ndarray
        Positive-valued data (e.g., stress ranges, load amplitudes).

    Returns
    -------
    WeibullFitResult
    """
    data = data[data > 0]
    if len(data) < 3:
        raise ValueError("Need at least 3 positive data points for Weibull fit")

    # scipy.stats.weibull_min uses: f(x, c, loc, scale)
    # We fix loc=0 for 2-parameter Weibull
    shape, loc, scale = stats.weibull_min.fit(data, floc=0)

    # KS test
    ks_stat, ks_pval = stats.kstest(data, 'weibull_min', args=(shape, loc, scale))

    return WeibullFitResult(
        shape=shape,
        scale=scale,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        is_rayleigh=abs(shape - 2.0) < 0.1,
    )


def rayleigh_fit(data: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit a Rayleigh distribution to data.

    Rayleigh is a special case of Weibull with shape parameter k=2.

    Parameters
    ----------
    data : np.ndarray
        Positive-valued data.

    Returns
    -------
    scale, ks_statistic, ks_pvalue, mean
    """
    data = data[data > 0]
    if len(data) < 3:
        raise ValueError("Need at least 3 positive data points for Rayleigh fit")

    # Rayleigh scale parameter: sigma = sqrt(mean(x^2)/2)
    scale = np.sqrt(np.mean(data ** 2) / 2.0)

    # KS test
    ks_stat, ks_pval = stats.kstest(data, 'rayleigh', args=(0, scale))

    mean = scale * np.sqrt(np.pi / 2.0)

    return scale, ks_stat, ks_pval, mean


def gumbel_extreme(
    data: np.ndarray,
    return_period: float = 20.0,
    events_per_year: float = 1000.0,
) -> GumbelResult:
    """Fit Gumbel distribution and extrapolate to a return period.

    Parameters
    ----------
    data : np.ndarray
        Block maxima (e.g., annual maximum stress ranges).
    return_period : float
        Target return period in years.
    events_per_year : float
        Number of independent events per year.

    Returns
    -------
    GumbelResult
    """
    if len(data) < 3:
        raise ValueError("Need at least 3 data points for Gumbel fit")

    # Fit Gumbel (extreme value type I) to block maxima
    # scipy.stats.gumbel_r: f(x, loc, scale)
    loc, scale = stats.gumbel_r.fit(data)

    # Extrapolate to return period
    # N = return_period * events_per_year
    # F(x_N) = 1 - 1/N
    # x_N = loc - scale * ln(-ln(1 - 1/N))
    N = return_period * events_per_year
    extreme_value = loc - scale * np.log(-np.log(1.0 - 1.0 / N))

    return GumbelResult(
        location=loc,
        scale=scale,
        return_period=return_period,
        extreme_value=extreme_value,
    )


def generate_rayleigh_samples(n: int, scale: float, seed: int = None) -> np.ndarray:
    """Generate random samples from a Rayleigh distribution.

    Parameters
    ----------
    n : int
        Number of samples.
    scale : float
        Rayleigh scale parameter.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
    """
    rng = np.random.default_rng(seed)
    return rng.rayleigh(scale=scale, size=n)
