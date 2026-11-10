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


# ============================================================================
# Wave-2 additions (node p270w2-n03): faithful replay at the declared settings,
# vectorised so the n=8 / B=1000 combination is affordable inside N=500 audits.
# ============================================================================

def _plot_slope_multi(samples: np.ndarray) -> np.ndarray:
    """Weibull probability-plot slope for a stack of samples, shape (B, n).

    Vectorised form of the median-rank log-linear fit used by
    :func:`weibull_m_loglinear`:  y = m * log(x) + c  with
    y = log(-log(1-F)), F = (i-0.3)/(n+0.4).
    """
    samples = np.atleast_2d(samples)
    b, n = samples.shape
    x_sorted = np.sort(samples, axis=1)
    F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    y = np.log(-np.log(1.0 - F))
    x_log = np.log(x_sorted)
    xc = x_log - x_log.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    denom = (xc * xc).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (xc * yc).sum(axis=1) / denom
    # a degenerate resample (all draws identical -> zero spread) has no slope;
    # mark it NaN so the caller can redraw instead of poisoning the interval
    slope = np.where(denom > 0, slope, np.nan)
    return np.where(np.isfinite(slope), np.maximum(slope, 1e-2), np.nan)


def _bootstrap_m(samples: np.ndarray, B: int, rng: np.random.Generator,
                 max_redraws: int = 10) -> np.ndarray:
    """Bootstrap slopes; degenerate resamples are redrawn (never NaN-poisoned)."""
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    idx = rng.integers(0, n, size=(B, n))
    boot = _plot_slope_multi(samples[idx])
    bad = ~np.isfinite(boot)
    for _ in range(max_redraws):
        if not bad.any():
            break
        k = int(bad.sum())
        boot[bad] = _plot_slope_multi(samples[rng.integers(0, n, size=(k, n))])
        bad = ~np.isfinite(boot)
    return boot[np.isfinite(boot)]


def make_weibull_estimator(B: int = 1000, seed: int = 2024):
    """Percentile-bootstrap CI for the plot slope m, replaying m7272 settings.

    Deterministic: every call advances a private counter, so the k-th dataset
    seen by the audit gets its own reproducible bootstrap stream.
    """
    state = {"k": 0}

    def estimator(data):
        rng = np.random.default_rng(seed + state["k"])
        state["k"] += 1
        m_hat = float(_plot_slope_multi(np.asarray(data)[None, :])[0])
        boot = _bootstrap_m(np.asarray(data), B, rng)
        return m_hat, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    return estimator


def make_weibull_bca_estimator(B: int = 1000, seed: int = 2024):
    """Same point estimate, BCa interval (remediation probe for the replay)."""
    state = {"k": 0}

    def estimator(data):
        rng = np.random.default_rng(seed + state["k"])
        state["k"] += 1
        data = np.asarray(data)
        n = len(data)
        m_hat = float(_plot_slope_multi(data[None, :])[0])
        boot = _bootstrap_m(data, B, rng)
        # bias-correction z0
        frac = float(np.clip((boot < m_hat).mean(), 1.0 / (B + 1), 1 - 1.0 / (B + 1)))
        z0 = stats.norm.ppf(frac)
        # acceleration from leave-one-out jackknife
        jack = np.array([float(_plot_slope_multi(np.delete(data, i)[None, :])[0])
                         for i in range(n)])
        jm = jack.mean()
        num = ((jm - jack) ** 3).sum()
        den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
        a = num / den if den > 0 else 0.0
        alpha = 0.05
        zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
        qs = [stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in zs]
        lo, hi = np.percentile(boot, np.clip(np.array(qs) * 100, 0.5, 99.5))
        return m_hat, float(lo), float(hi)

    return estimator


def replay_m7272(n_replications: int = 500, B: int = 1000, seed: int = 42):
    """Replay the m7272 CI claim: n=8 Weibull samples, bootstrap B=1000."""
    out = {}
    for label, factory in (("percentile_bootstrap_B1000", make_weibull_estimator),
                           ("bca_bootstrap_B1000", make_weibull_bca_estimator)):
        audit = CoverageAudit(
            truth_generator=weibull_truth_gen,
            estimator=factory(B=B, seed=2024),
            n_replications=n_replications,
            n_samples=8,
            target_level=0.95,
            seed=seed,
            estimator_name=f"weibull_m_{label}",
        )
        out[label] = audit.run(error_levels=[0, 1, 2]).as_dict()
    return out


def replay_m7269(n_replications: int = 500, seed: int = 42):
    """Replay the m7269 CI claim: N_peak≈60, sample std + chi-square interval."""
    audit = CoverageAudit(
        truth_generator=sigma_truth_gen,
        estimator=sigma_estimator,
        n_replications=n_replications,
        n_samples=60,
        target_level=0.95,
        seed=seed,
        estimator_name="sigma_chi2_N60",
    )
    return {"chi2_N60": audit.run(error_levels=[0, 1, 2]).as_dict()}


def _normalise_scale(x: np.ndarray) -> np.ndarray:
    """Divide by the geometric mean (the Weibull shape MLE is scale-invariant)."""
    return x / np.exp(np.log(x).mean())


def _weibull_mle_m(x: np.ndarray) -> float:
    """Maximum-likelihood Weibull shape parameter (1-D score equation in log space).

    Data are re-scaled by their geometric mean so that no power overflows; the
    score is then  1/m - sum_i w_i log u_i  with w_i = u_i^m / sum_j u_j^m.
    """
    from scipy.optimize import brentq
    from scipy.special import logsumexp

    u = _normalise_scale(np.asarray(x, dtype=float))
    lu = np.log(u)

    def score(m: float) -> float:
        logS = logsumexp(m * lu)
        w = np.exp(m * lu - logS)
        return 1.0 / m - float((w * lu).sum())

    return float(brentq(score, 1e-3, 1e3, xtol=1e-12, rtol=1e-13))


def _weibull_loglik(m: float, x: np.ndarray) -> float:
    """Profile log-likelihood of the shape parameter with the scale profiled out."""
    from scipy.special import logsumexp

    u = _normalise_scale(np.asarray(x, dtype=float))
    lu = np.log(u)
    n = len(u)
    logS = logsumexp(m * lu)
    m_log_lam = logS - math.log(n)
    # sum_i (u_i/lam)^m == n exactly at the profiled scale lambda(m)
    return float(n * math.log(m) - n * m_log_lam + (m - 1.0) * lu.sum() - n)


def make_weibull_mle_profile_estimator(level: float = 0.95):
    """MLE + profile-likelihood CI (bias-aware remediation for the m7272 replay)."""
    from scipy.optimize import brentq
    from scipy.stats import chi2

    crit = chi2.ppf(level, df=1)

    def estimator(data):
        x = np.asarray(data, dtype=float)
        m_hat = _weibull_mle_m(x)
        l_max = _weibull_loglik(m_hat, x)

        def gap(m):
            return 2.0 * (l_max - _weibull_loglik(m, x)) - crit

        def edge(bracket, gap_at_a):
            # gap = 2(lmax - l(m)) - crit is < 0 inside the interval and > 0
            # outside; the bracket must straddle the root with the sign we expect
            a, b = bracket
            if (gap(a) - gap_at_a) * 1.0 == 0.0:
                return None
            if gap_at_a > 0 and not (gap(a) > 0 > gap(b)):
                return None
            if gap_at_a < 0 and not (gap(a) < 0 < gap(b)):
                return None
            return float(brentq(gap, a, b, xtol=1e-12))

        lo = edge((m_hat * 0.05, m_hat), gap_at_a=1.0)     # root sought with gap>0 at a
        hi = edge((m_hat, m_hat * 20.0 + 1.0), gap_at_a=-1.0)
        lo = lo if lo is not None else max(m_hat * 0.05, 1e-3)
        hi = hi if hi is not None else m_hat * 20.0 + 1.0
        return float(m_hat), float(lo), float(hi)

    return estimator


def replay_m7272_mle(n_replications: int = 500, seed: int = 42):
    audit = CoverageAudit(
        truth_generator=weibull_truth_gen,
        estimator=make_weibull_mle_profile_estimator(),
        n_replications=n_replications,
        n_samples=8,
        target_level=0.95,
        seed=seed,
        estimator_name="weibull_m_mle_profile",
    )
    return {"mle_profile": audit.run(error_levels=[0, 1, 2]).as_dict()}
