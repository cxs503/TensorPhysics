#!/usr/bin/env python3
"""Statistical Coverage Audit (SBC) — empirical coverage calibration for CIs.

Purpose
-------
Given *any* estimation pipeline f(data) → (θ̂, CI_lo, CI_hi), this tool
quantifies whether the claimed 95 % confidence interval actually covers the
true parameter θ* at the stated rate.  It does so by Monte-Carlo repetition:
generate N independent datasets with known truth, compute the CI each time,
and measure the empirical coverage fraction.

Three error-injection levels are supported:
  - Level 0: pure measurement noise (idealised)
  - Level 1: model structure error (bias in the estimator)
  - Level 2: model error + heteroscedastic noise

Three-state verdict:
  - WELL_CALIBRATED: coverage ∈ [0.93, 0.97]
  - CONSERVATIVE:    coverage > 0.97
  - DISTORTED:       coverage < 0.93  OR  (coverage > 0.99 AND CI width anomalous)

Stopping rule:
  From the coverage-vs-n curve, solve for the minimum n that achieves
  ±0.02 resolution (i.e. Wilson half-width ≤ 0.02 at the observed coverage).

Interface (zero domain semantics)
----------------------------------
The audit engine only requires two callables:
  - truth_generator(rng, n_samples, error_level) -> (data, true_theta)
  - estimator(data) -> (theta_hat, ci_lo, ci_hi)

All domain content (Weibull, σ-divergence, etc.) is injected via these
interfaces.  The engine itself knows nothing about ships, ice, or waves.

Usage
-----
    from tools.stats.coverage_audit import CoverageAudit
    audit = CoverageAudit(truth_generator, estimator, n_replications=500)
    result = audit.run(error_levels=[0, 1, 2])
    print(result.verdict)
    print(result.n_min_table)
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Protocols (structural typing — zero domain semantics)
# ---------------------------------------------------------------------------

class TruthGenerator(Protocol):
    """Generate a synthetic dataset with known ground truth.
    
    Parameters
    ----------
    rng : np.random.Generator
    n_samples : int
        Number of data points in the synthetic dataset.
    error_level : int
        0 = pure noise, 1 = model structure error, 2 = model error + heteroscedastic
    
    Returns
    -------
    data : Any
        The synthetic dataset (type is opaque to the audit engine).
    true_theta : float
        The known true parameter value.
    """
    def __call__(self, rng: np.random.Generator, n_samples: int,
                 error_level: int) -> tuple[Any, float]: ...


class Estimator(Protocol):
    """Estimate parameter and confidence interval from data.
    
    Parameters
    ----------
    data : Any
        The dataset (same type as returned by TruthGenerator).
    
    Returns
    -------
    theta_hat : float
        Point estimate.
    ci_lo : float
        Lower bound of confidence interval.
    ci_hi : float
        Upper bound of confidence interval.
    """
    def __call__(self, data: Any) -> tuple[float, float, float]: ...


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

WELL_CALIBRATED_LO = 0.93
WELL_CALIBRATED_HI = 0.97
CONSERVATIVE_LO = 0.97
DISTORTED_HI = 0.99


def judge_coverage(coverage: float, ci_width_ratio: float = 1.0) -> str:
    """Three-state verdict from empirical coverage.
    
    Parameters
    ----------
    coverage : float
        Empirical coverage fraction ∈ [0, 1].
    ci_width_ratio : float
        Ratio of mean CI width to expected width under ideal calibration.
        Used to detect anomalously wide CIs (ratio > 2.0).
    
    Returns
    -------
    str : 'WELL_CALIBRATED', 'CONSERVATIVE', or 'DISTORTED'
    """
    if coverage < WELL_CALIBRATED_LO:
        return "DISTORTED"
    if coverage > DISTORTED_HI and ci_width_ratio > 2.0:
        return "DISTORTED"
    if coverage > CONSERVATIVE_LO:
        return "CONSERVATIVE"
    return "WELL_CALIBRATED"


# ---------------------------------------------------------------------------
# Wilson interval for stopping rule
# ---------------------------------------------------------------------------

def wilson_half_width(k: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval for binomial proportion."""
    if n == 0:
        return 1.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    hi = (centre + spread) / denom
    lo = (centre - spread) / denom
    return 0.5 * (hi - lo)


def solve_n_min(target_coverage: float, target_hw: float = 0.02,
                n_max: int = 5000) -> int:
    """Minimum replications n for Wilson half-width ≤ target_hw.
    
    Uses the conservative approximation p=0.5 (worst case).
    """
    z = 1.96
    # Wilson half-width ≈ z * sqrt(p*(1-p)/n) for large n
    # Worst case p=0.5: hw ≈ z/(2*sqrt(n))
    # n ≈ (z/(2*hw))^2
    n_approx = int(math.ceil((z / (2 * target_hw))**2))
    return min(n_approx, n_max)


# ---------------------------------------------------------------------------
# Per-level result
# ---------------------------------------------------------------------------

@dataclass
class LevelResult:
    error_level: int
    n_replications: int
    n_covered: int
    coverage: float
    mean_theta_hat: float
    mean_ci_lo: float
    mean_ci_hi: float
    mean_ci_width: float
    ci_width_ratio: float  # relative to level-0 width
    wilson_hw: float
    verdict: str
    coverage_curve: list[float] = field(default_factory=list)  # running coverage

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class AuditResult:
    estimator_name: str
    true_theta: float
    target_level: float
    levels: list[LevelResult]
    n_min_table: dict[int, int]  # error_level -> n_min
    overall_verdict: str

    def as_dict(self) -> dict:
        return {
            "estimator_name": self.estimator_name,
            "true_theta": self.true_theta,
            "target_level": self.target_level,
            "levels": [lv.as_dict() for lv in self.levels],
            "n_min_table": self.n_min_table,
            "overall_verdict": self.overall_verdict,
        }


# ---------------------------------------------------------------------------
# Main audit engine
# ---------------------------------------------------------------------------

class CoverageAudit:
    """Empirical coverage calibration for confidence intervals.
    
    Parameters
    ----------
    truth_generator : callable
        (rng, n_samples, error_level) -> (data, true_theta)
    estimator : callable
        (data) -> (theta_hat, ci_lo, ci_hi)
    n_replications : int
        Number of Monte-Carlo replications (default 500).
    n_samples : int
        Number of data points per synthetic dataset (default 50).
    target_level : float
        Nominal confidence level (default 0.95).
    seed : int
        Base random seed.
    estimator_name : str
        Human-readable name for reporting.
    """

    def __init__(
        self,
        truth_generator: TruthGenerator,
        estimator: Estimator,
        n_replications: int = 500,
        n_samples: int = 50,
        target_level: float = 0.95,
        seed: int = 42,
        estimator_name: str = "unnamed",
    ):
        self.truth_generator = truth_generator
        self.estimator = estimator
        self.n_replications = n_replications
        self.n_samples = n_samples
        self.target_level = target_level
        self.seed = seed
        self.estimator_name = estimator_name

    def run(self, error_levels: Sequence[int] = (0, 1, 2)) -> AuditResult:
        """Run the coverage audit for each error level."""
        levels_results = []
        base_width = None

        for elvl in error_levels:
            rng = np.random.default_rng(self.seed + elvl * 1000)
            n_covered = 0
            theta_hats = []
            ci_los = []
            ci_his = []
            coverage_curve = []

            for rep in range(self.n_replications):
                data, true_theta = self.truth_generator(rng, self.n_samples, elvl)
                theta_hat, ci_lo, ci_hi = self.estimator(data)
                theta_hats.append(theta_hat)
                ci_los.append(ci_lo)
                ci_his.append(ci_hi)
                if ci_lo <= true_theta <= ci_hi:
                    n_covered += 1
                coverage_curve.append(n_covered / (rep + 1))

            coverage = n_covered / self.n_replications
            mean_width = np.mean([h - l for l, h in zip(ci_los, ci_his)])
            if base_width is None:
                base_width = mean_width
            width_ratio = mean_width / base_width if base_width > 0 else 1.0
            wilson_hw = wilson_half_width(n_covered, self.n_replications)
            verdict = judge_coverage(coverage, width_ratio)

            levels_results.append(LevelResult(
                error_level=elvl,
                n_replications=self.n_replications,
                n_covered=n_covered,
                coverage=coverage,
                mean_theta_hat=float(np.mean(theta_hats)),
                mean_ci_lo=float(np.mean(ci_los)),
                mean_ci_hi=float(np.mean(ci_his)),
                mean_ci_width=float(mean_width),
                ci_width_ratio=float(width_ratio),
                wilson_hw=float(wilson_hw),
                verdict=verdict,
                coverage_curve=coverage_curve,
            ))

        # Stopping rule: n_min for ±0.02 resolution
        n_min_table = {}
        for lr in levels_results:
            n_min_table[lr.error_level] = solve_n_min(lr.coverage, 0.02)

        # Overall verdict: worst across levels
        verdicts = [lr.verdict for lr in levels_results]
        if "DISTORTED" in verdicts:
            overall = "DISTORTED"
        elif "CONSERVATIVE" in verdicts:
            overall = "CONSERVATIVE"
        else:
            overall = "WELL_CALIBRATED"

        # Use the true_theta from level 0 (they should all share the same truth)
        rng0 = np.random.default_rng(self.seed)
        _, true_theta = self.truth_generator(rng0, self.n_samples, 0)

        return AuditResult(
            estimator_name=self.estimator_name,
            true_theta=float(true_theta),
            target_level=self.target_level,
            levels=levels_results,
            n_min_table=n_min_table,
            overall_verdict=overall,
        )


# ---------------------------------------------------------------------------
# Convenience: run and export
# ---------------------------------------------------------------------------

def run_audit_and_export(
    truth_generator: TruthGenerator,
    estimator: Estimator,
    output_path: str | Path,
    **kwargs,
) -> AuditResult:
    """Run audit and write JSON result to output_path."""
    audit = CoverageAudit(truth_generator, estimator, **kwargs)
    result = audit.run()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result.as_dict(), f, indent=2)
    return result


# ---------------------------------------------------------------------------
# CLI entry point (for cluster submission)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Coverage Audit — import and use CoverageAudit class.")
    print("See tests/test_coverage_audit.py for usage examples.")
