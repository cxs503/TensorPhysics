"""
Uncertainty propagation from reference anchors to calculation results.

Implements:
1. Analytical (first-order Taylor) propagation: U_result = |∂f/∂x| · U_anchor
2. Monte Carlo propagation: sample from anchor distributions, compute result distribution
3. Sensitivity analysis: which anchors contribute most to result uncertainty

The propagation chain: anchor U_ref → calculation U_result → verdict judgment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PropagationResult:
    """
    Result of uncertainty propagation.

    Attributes
    ----------
    result_nominal : float
        Nominal result value
    U_result : float
        Expanded uncertainty of result at k=2
    contributions : dict
        Per-anchor contribution to total variance
    method : str
        Propagation method used ("analytical" or "monte_carlo")
    n_samples : int
        Number of Monte Carlo samples (if applicable)
    """
    result_nominal: float
    U_result: float
    contributions: Dict[str, float]
    method: str
    n_samples: int = 0

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty U_result / |result_nominal|."""
        if self.result_nominal == 0:
            return float('inf')
        return abs(self.U_result / self.result_nominal)

    def top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-n contributors by variance fraction."""
        total_var = sum(self.contributions.values())
        if total_var == 0:
            return []
        sorted_contribs = sorted(
            self.contributions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(k, v / total_var) for k, v in sorted_contribs[:n]]


def propagate_uncertainty_analytical(
    func: Callable[..., float],
    anchor_values: Dict[str, float],
    anchor_uncertainties: Dict[str, float],
    k: float = 2.0,
    perturbation_frac: float = 1e-4,
) -> PropagationResult:
    """
    First-order analytical uncertainty propagation.

    Uses finite differences to estimate partial derivatives:
    U_result² = Σ (∂f/∂x_i)² · U_i²

    Parameters
    ----------
    func : callable
        Function f(**anchor_values) → float
    anchor_values : dict
        Nominal values {anchor_id: value}
    anchor_uncertainties : dict
        Standard uncertainties {anchor_id: U_i} (at k=1)
    k : float
        Coverage factor for output (default 2)
    perturbation_frac : float
        Fractional perturbation for finite differences

    Returns
    -------
    PropagationResult
    """
    # Nominal result
    result_nominal = func(**anchor_values)

    # Compute partial derivatives via central differences
    contributions = {}
    total_variance = 0.0

    for anchor_id, x0 in anchor_values.items():
        U_i = anchor_uncertainties.get(anchor_id, 0.0)
        if U_i == 0:
            contributions[anchor_id] = 0.0
            continue

        # Perturbation size
        dx = max(abs(x0) * perturbation_frac, 1e-10)

        # Central difference
        vals_plus = dict(anchor_values)
        vals_plus[anchor_id] = x0 + dx
        f_plus = func(**vals_plus)

        vals_minus = dict(anchor_values)
        vals_minus[anchor_id] = x0 - dx
        f_minus = func(**vals_minus)

        df_dx = (f_plus - f_minus) / (2 * dx)

        # Contribution to variance (standard uncertainty squared)
        var_contrib = (df_dx * U_i) ** 2
        contributions[anchor_id] = var_contrib
        total_variance += var_contrib

    # Expanded uncertainty at k
    U_result = k * math.sqrt(total_variance)

    return PropagationResult(
        result_nominal=result_nominal,
        U_result=U_result,
        contributions=contributions,
        method="analytical",
    )


def monte_carlo_propagation(
    func: Callable[..., float],
    anchor_values: Dict[str, float],
    anchor_uncertainties: Dict[str, float],
    n_samples: int = 1000,
    k: float = 2.0,
    seed: Optional[int] = None,
    distribution: str = "normal",
) -> PropagationResult:
    """
    Monte Carlo uncertainty propagation.

    Samples from anchor distributions and computes result distribution.

    Parameters
    ----------
    func : callable
        Function f(**anchor_values) → float
    anchor_values : dict
        Nominal values {anchor_id: value}
    anchor_uncertainties : dict
        Standard uncertainties {anchor_id: U_i} (at k=1)
    n_samples : int
        Number of Monte Carlo samples
    k : float
        Coverage factor for output (default 2)
    seed : int, optional
        Random seed for reproducibility
    distribution : str
        Distribution type: "normal" or "uniform"

    Returns
    -------
    PropagationResult
    """
    rng = np.random.default_rng(seed)
    results = np.zeros(n_samples)

    # Generate samples for each anchor
    anchor_samples = {}
    for anchor_id, x0 in anchor_values.items():
        U_i = anchor_uncertainties.get(anchor_id, 0.0)
        if U_i == 0:
            anchor_samples[anchor_id] = np.full(n_samples, x0)
        else:
            if distribution == "normal":
                # U_i is standard uncertainty (σ)
                anchor_samples[anchor_id] = rng.normal(x0, U_i, n_samples)
            elif distribution == "uniform":
                # Uniform over ±√3·U_i (same variance as normal)
                half_width = math.sqrt(3) * U_i
                anchor_samples[anchor_id] = rng.uniform(
                    x0 - half_width, x0 + half_width, n_samples
                )
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

    # Evaluate function for each sample
    for i in range(n_samples):
        sample_values = {aid: anchor_samples[aid][i] for aid in anchor_values}
        results[i] = func(**sample_values)

    # Compute statistics
    result_nominal = float(np.mean(results))
    std_result = float(np.std(results, ddof=1))
    U_result = k * std_result

    # Estimate contributions via variance decomposition
    # (simplified: use correlation-based approach)
    contributions = {}
    total_var = np.var(results, ddof=1)
    if total_var > 0:
        for anchor_id in anchor_values:
            corr = np.corrcoef(anchor_samples[anchor_id], results)[0, 1]
            # Approximate contribution as correlation² × total variance
            contributions[anchor_id] = (corr ** 2) * total_var if not np.isnan(corr) else 0.0
    else:
        contributions = {aid: 0.0 for aid in anchor_values}

    return PropagationResult(
        result_nominal=result_nominal,
        U_result=U_result,
        contributions=contributions,
        method="monte_carlo",
        n_samples=n_samples,
    )


def compare_propagation_methods(
    func: Callable[..., float],
    anchor_values: Dict[str, float],
    anchor_uncertainties: Dict[str, float],
    n_samples: int = 1000,
    k: float = 2.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compare analytical vs Monte Carlo propagation.

    Returns
    -------
    dict
        Comparison metrics: nominal difference, U_result difference, relative error
    """
    analytical = propagate_uncertainty_analytical(
        func, anchor_values, anchor_uncertainties, k=k
    )
    mc = monte_carlo_propagation(
        func, anchor_values, anchor_uncertainties,
        n_samples=n_samples, k=k, seed=seed
    )

    nominal_diff = abs(analytical.result_nominal - mc.result_nominal)
    U_diff = abs(analytical.U_result - mc.U_result)
    U_rel_error = U_diff / mc.U_result if mc.U_result > 0 else 0.0

    return {
        "analytical_nominal": analytical.result_nominal,
        "mc_nominal": mc.result_nominal,
        "nominal_difference": nominal_diff,
        "analytical_U": analytical.U_result,
        "mc_U": mc.U_result,
        "U_difference": U_diff,
        "U_relative_error": U_rel_error,
    }
