#!/usr/bin/env python3
"""Case studies: m7272 (Weibull m) and m7269 (σ divergence) coverage audit.

Uses fast estimators suitable for Monte-Carlo coverage audit.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.stats.coverage_audit import CoverageAudit


# ============================================================================
# Case 1: Weibull shape parameter m (m7272)
# ============================================================================

def weibull_truth_gen(rng, n_samples, error_level):
    """Generate Weibull data with known shape parameter m=2.0."""
    true_m = 2.0
    lam = 100.0
    
    if error_level == 0:
        data = rng.weibull(true_m, size=n_samples) * lam
    elif error_level == 1:
        n_clean = int(0.9 * n_samples)
        n_dirty = n_samples - n_clean
        clean = rng.weibull(true_m, size=n_clean) * lam
        dirty = rng.weibull(0.5, size=n_dirty) * lam * 3
        data = np.concatenate([clean, dirty])
        rng.shuffle(data)
    elif error_level == 2:
        n1 = int(0.7 * n_samples)
        n2 = n_samples - n1
        d1 = rng.weibull(2.0, size=n1) * lam
        d2 = rng.weibull(1.0, size=n2) * lam * 1.5
        data = np.concatenate([d1, d2])
        rng.shuffle(data)
    else:
        raise ValueError(f"Unknown error_level {error_level}")
    
    return data, true_m


def weibull_m_loglinear(data, B=500, rng_seed=12345):
    """Weibull shape parameter via log-linear regression + bootstrap CI.
    
    Fast alternative to MLE: uses the linearised Weibull CDF.
    log(-log(1-F(x))) = m*log(x) - m*log(λ)
    
    This is the standard Weibull probability plot method.
    """
    n = len(data)
    
    def _fit_m(x):
        x_sorted = np.sort(x)
        # Median rank approximation for empirical CDF
        F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        # Clamp to avoid log(0)
        F = np.clip(F, 1e-10, 1 - 1e-10)
        y = np.log(-np.log(1 - F))
        x_log = np.log(x_sorted)
        # Linear regression: y = m * x_log + c
        A = np.column_stack([x_log, np.ones(n)])
        sol = np.linalg.lstsq(A, y, rcond=None)
        m_hat = sol[0][0]
        return max(m_hat, 0.01)  # ensure positive
    
    m_hat = _fit_m(data)
    
    # Nonparametric bootstrap
    rng_boot = np.random.default_rng(rng_seed)
    boot_indices = rng_boot.integers(0, n, size=(B, n))
    m_boots = np.zeros(B)
    for b in range(B):
        m_boots[b] = _fit_m(data[boot_indices[b]])
    
    ci_lo = float(np.percentile(m_boots, 2.5))
    ci_hi = float(np.percentile(m_boots, 97.5))
    return float(m_hat), ci_lo, ci_hi


# ============================================================================
# Case 2: σ divergence (m7269)
# ============================================================================

def sigma_truth_gen(rng, n_samples, error_level):
    """Generate peak stress data with known σ=10.0 MPa."""
    true_sigma = 10.0
    mu = 50.0
    
    if error_level == 0:
        data = rng.normal(mu, true_sigma, size=n_samples)
    elif error_level == 1:
        data = mu + true_sigma / math.sqrt(2.0) * rng.standard_t(df=4, size=n_samples)
    elif error_level == 2:
        n1 = n_samples // 2
        n2 = n_samples - n1
        d1 = rng.normal(mu, 8.0, size=n1)
        d2 = rng.normal(mu, 14.0, size=n2)
        data = np.concatenate([d1, d2])
        rng.shuffle(data)
    else:
        raise ValueError(f"Unknown error_level {error_level}")
    
    return data, true_sigma


def sigma_estimator(data):
    """Sample std + chi-squared CI for σ."""
    n = len(data)
    s = float(np.std(data, ddof=1))
    chi2_lo = stats.chi2.ppf(0.025, df=n - 1)
    chi2_hi = stats.chi2.ppf(0.975, df=n - 1)
    var_lo = (n - 1) * s**2 / chi2_hi
    var_hi = (n - 1) * s**2 / chi2_lo
    ci_lo = math.sqrt(var_lo)
    ci_hi = math.sqrt(var_hi)
    return s, ci_lo, ci_hi


# ============================================================================
# Run both case studies
# ============================================================================

def run_case_studies(n_replications: int = 500, seed: int = 42):
    """Run both case studies and return results."""
    results = {}
    
    # Case 1: Weibull m (m7272) — n=8, bootstrap B=500
    print("=" * 60)
    print("Case 1: Weibull shape parameter m (m7272)")
    print("  n=8, bootstrap B=500, target 95% CI")
    print("=" * 60)
    
    audit1 = CoverageAudit(
        truth_generator=weibull_truth_gen,
        estimator=weibull_m_loglinear,
        n_replications=n_replications,
        n_samples=8,
        target_level=0.95,
        seed=seed,
        estimator_name="weibull_loglinear_bootstrap_B500",
    )
    result1 = audit1.run(error_levels=[0, 1, 2])
    results["m7272_weibull_m"] = result1.as_dict()
    
    for lr in result1.levels:
        print(f"  Level {lr.error_level}: coverage={lr.coverage:.3f} "
              f"verdict={lr.verdict} Wilson_HW={lr.wilson_hw:.4f} "
              f"CI_width={lr.mean_ci_width:.3f}")
    print(f"  Overall: {result1.overall_verdict}")
    print(f"  n_min table: {result1.n_min_table}")
    print()
    
    # Case 2: σ divergence (m7269) — N_peak≈60
    print("=" * 60)
    print("Case 2: σ divergence (m7269)")
    print("  N_peak≈60, chi-squared CI, target 95% CI")
    print("=" * 60)
    
    audit2 = CoverageAudit(
        truth_generator=sigma_truth_gen,
        estimator=sigma_estimator,
        n_replications=n_replications,
        n_samples=60,
        target_level=0.95,
        seed=seed,
        estimator_name="sigma_chi2_N60",
    )
    result2 = audit2.run(error_levels=[0, 1, 2])
    results["m7269_sigma"] = result2.as_dict()
    
    for lr in result2.levels:
        print(f"  Level {lr.error_level}: coverage={lr.coverage:.3f} "
              f"verdict={lr.verdict} Wilson_HW={lr.wilson_hw:.4f} "
              f"CI_width={lr.mean_ci_width:.3f}")
    print(f"  Overall: {result2.overall_verdict}")
    print(f"  n_min table: {result2.n_min_table}")
    
    return results


if __name__ == "__main__":
    results = run_case_studies()
    
    out_dir = ROOT / "artifacts" / "coverage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "case_studies.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
