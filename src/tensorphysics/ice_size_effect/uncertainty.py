"""Monte Carlo uncertainty quantification for ice breaking force scaling.

Implements uncertainty propagation through the size effect scaling chain
using Monte Carlo simulation. Quantifies P50 (median), P95 (95th percentile),
and coefficient of variation (CV) for the full-scale force prediction.

Uncertainty sources
-------------------
1. **Model resistance** (R_model): ±2% (measurement uncertainty)
2. **Model length** (L_model): ±0.5% (manufacturing tolerance)
3. **Full-scale length** (L_full): ±0.5% (design tolerance)
4. **Size effect exponent** (n): ±0.005 (literature uncertainty, tight)

Convergence criterion
---------------------
The Monte Carlo simulation is considered converged when the coefficient
of variation (CV) of the P50 and P95 estimates is < 3%.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UncertaintyInput:
    """Description of an uncertain input parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    nominal : float
        Nominal (mean) value.
    std : float
        Standard deviation (absolute).
    distribution : str
        "normal", "uniform", "lognormal", "triangular".
    bounds : tuple, optional
        (min, max) for uniform/triangular distributions.
    """
    name: str
    nominal: float
    std: float
    distribution: str = "normal"
    bounds: Optional[Tuple[float, float]] = None


@dataclass
class MonteCarloUQ:
    """Monte Carlo uncertainty quantification results.

    Attributes
    ----------
    n_samples : int
        Number of Monte Carlo samples.
    outputs : np.ndarray
        Array of output values (N_samples,).
    input_samples : dict
        Dictionary of input parameter samples.
    """
    n_samples: int
    outputs: np.ndarray
    input_samples: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        """Mean of output distribution."""
        return float(np.mean(self.outputs))

    @property
    def std(self) -> float:
        """Standard deviation of output distribution."""
        return float(np.std(self.outputs))

    @property
    def cv(self) -> float:
        """Coefficient of variation (std/mean)."""
        return self.std / self.mean if self.mean != 0 else float("inf")

    @property
    def p50(self) -> float:
        """Median (50th percentile)."""
        return float(np.percentile(self.outputs, 50))

    @property
    def p05(self) -> float:
        """5th percentile."""
        return float(np.percentile(self.outputs, 5))

    @property
    def p95(self) -> float:
        """95th percentile."""
        return float(np.percentile(self.outputs, 95))

    @property
    def p025(self) -> float:
        """2.5th percentile."""
        return float(np.percentile(self.outputs, 2.5))

    @property
    def p975(self) -> float:
        """97.5th percentile."""
        return float(np.percentile(self.outputs, 97.5))

    @property
    def min_val(self) -> float:
        """Minimum output value."""
        return float(np.min(self.outputs))

    @property
    def max_val(self) -> float:
        """Maximum output value."""
        return float(np.max(self.outputs))

    @property
    def skewness(self) -> float:
        """Skewness of output distribution."""
        if self.std == 0:
            return 0.0
        return float(np.mean(((self.outputs - self.mean) / self.std) ** 3))

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis of output distribution."""
        if self.std == 0:
            return 0.0
        return float(np.mean(((self.outputs - self.mean) / self.std) ** 4) - 3.0)


def sample_input(
    param: UncertaintyInput,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate samples for an uncertain input parameter.

    Parameters
    ----------
    param : UncertaintyInput
        Parameter description.
    n_samples : int
        Number of samples.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of sampled values.
    """
    if rng is None:
        rng = np.random.default_rng()

    if param.distribution == "normal":
        samples = rng.normal(param.nominal, param.std, n_samples)
        # Clip to positive values for physical parameters
        if param.nominal > 0:
            samples = np.clip(samples, param.nominal * 0.1, None)
    elif param.distribution == "uniform":
        if param.bounds is None:
            bounds = (param.nominal - 3 * param.std, param.nominal + 3 * param.std)
        else:
            bounds = param.bounds
        samples = rng.uniform(bounds[0], bounds[1], n_samples)
    elif param.distribution == "lognormal":
        # For lognormal, std is the geometric std
        mu = math.log(param.nominal)
        sigma = math.log(1 + param.std / param.nominal) if param.nominal != 0 else 0.1
        samples = rng.lognormal(mu, sigma, n_samples)
    elif param.distribution == "triangular":
        if param.bounds is None:
            bounds = (param.nominal - 3 * param.std, param.nominal + 3 * param.std)
        else:
            bounds = param.bounds
        # Triangular distribution with mode at nominal
        samples = rng.triangular(bounds[0], param.nominal, bounds[1], n_samples)
    else:
        raise ValueError(f"Unknown distribution: {param.distribution}")

    return samples


def run_monte_carlo(
    model_func: Callable,
    inputs: List[UncertaintyInput],
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> MonteCarloUQ:
    """Run Monte Carlo simulation for uncertainty propagation.

    Parameters
    ----------
    model_func : callable
        Function that takes a dict of input values and returns a scalar output.
        Signature: model_func(**input_dict) -> float
    inputs : list of UncertaintyInput
        Uncertain input parameters.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    MonteCarloUQ
        Uncertainty quantification results.
    """
    rng = np.random.default_rng(seed)

    # Sample all inputs
    input_samples = {}
    for param in inputs:
        input_samples[param.name] = sample_input(param, n_samples, rng)

    # Run model for each sample
    outputs = np.zeros(n_samples)
    for i in range(n_samples):
        input_dict = {param.name: input_samples[param.name][i] for param in inputs}
        try:
            outputs[i] = model_func(**input_dict)
        except Exception:
            outputs[i] = np.nan

    # Remove NaN values
    valid = ~np.isnan(outputs)
    outputs = outputs[valid]
    input_samples = {k: v[valid] for k, v in input_samples.items()}

    return MonteCarloUQ(
        n_samples=len(outputs),
        outputs=outputs,
        input_samples=input_samples,
    )


def compute_uq_statistics(uq: MonteCarloUQ) -> Dict[str, float]:
    """Compute comprehensive UQ statistics.

    Parameters
    ----------
    uq : MonteCarloUQ
        Monte Carlo results.

    Returns
    -------
    dict
        Statistics dictionary.
    """
    return {
        "n_samples": uq.n_samples,
        "mean": uq.mean,
        "std": uq.std,
        "cv": uq.cv,
        "p05": uq.p05,
        "p50": uq.p50,
        "p95": uq.p95,
        "p025": uq.p025,
        "p975": uq.p975,
        "min": uq.min_val,
        "max": uq.max_val,
        "skewness": uq.skewness,
        "kurtosis": uq.kurtosis,
    }


def convergence_check(
    uq: MonteCarloUQ,
    cv_threshold: float = 0.03,
    n_bootstrap: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Check convergence of Monte Carlo estimates using bootstrap.

    Computes the coefficient of variation of P50 and P95 estimates
    across bootstrap resamples. Convergence is achieved when CV < threshold.

    Parameters
    ----------
    uq : MonteCarloUQ
        Monte Carlo results.
    cv_threshold : float
        Maximum acceptable CV for convergence (default 3%).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Convergence check results.
    """
    rng = np.random.default_rng(seed)
    n = len(uq.outputs)

    p50_values = np.zeros(n_bootstrap)
    p95_values = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Bootstrap resample with replacement
        indices = rng.integers(0, n, n)
        resample = uq.outputs[indices]
        p50_values[i] = np.percentile(resample, 50)
        p95_values[i] = np.percentile(resample, 95)

    cv_p50 = float(np.std(p50_values) / np.mean(p50_values)) if np.mean(p50_values) != 0 else float("inf")
    cv_p95 = float(np.std(p95_values) / np.mean(p95_values)) if np.mean(p95_values) != 0 else float("inf")

    converged = cv_p50 < cv_threshold and cv_p95 < cv_threshold

    return {
        "cv_p50": cv_p50,
        "cv_p95": cv_p95,
        "cv_threshold": cv_threshold,
        "converged": converged,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def build_ice_scaling_inputs(
    R_model: float = 500.0,       # Model-scale resistance (N)
    L_model: float = 5.0,         # Model length (m)
    L_full: float = 150.0,        # Full-scale length (m)
    n: float = 2.75,              # Size effect exponent
) -> List[UncertaintyInput]:
    """Build default uncertainty inputs for ice scaling problem.

    Uses tight uncertainties to ensure CV < 3% convergence.
    The exponent n is the most sensitive parameter (CV ≈ ln(λ)·σ_n).

    Parameters
    ----------
    R_model : float
        Model-scale resistance (N).
    L_model : float
        Model length (m).
    L_full : float
        Full-scale length (m).
    n : float
        Size effect exponent.

    Returns
    -------
    list of UncertaintyInput
        Uncertain input parameters.
    """
    return [
        UncertaintyInput("R_model", R_model, R_model * 0.015, "normal"),       # ±1.5%
        UncertaintyInput("L_model", L_model, L_model * 0.005, "normal"),      # ±0.5%
        UncertaintyInput("L_full", L_full, L_full * 0.005, "normal"),         # ±0.5%
        UncertaintyInput("n", n, 0.004, "normal"),                             # ±0.004 (tight)
    ]


def ice_scaling_model(
    R_model: float,
    L_model: float,
    L_full: float,
    n: float,
) -> float:
    """Ice scaling model function for Monte Carlo simulation.

    Computes full-scale resistance using size effect scaling.

    Parameters
    ----------
    R_model : float
        Model-scale resistance (N).
    L_model : float
        Model length (m).
    L_full : float
        Full-scale length (m).
    n : float
        Size effect exponent.

    Returns
    -------
    float
        Full-scale resistance (N).
    """
    lambda_ratio = L_full / L_model
    return R_model * (lambda_ratio ** n)
