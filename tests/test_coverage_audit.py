#!/usr/bin/env python3
"""Tests for tools/stats/coverage_audit.py — synthetic estimators only.

Zero domain semantics: all test estimators and truth generators are
synthetic (Gaussian mean estimation, biased estimator, heteroscedastic).
No ship/ice/wave content.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.stats.coverage_audit import (
    CoverageAudit,
    judge_coverage,
    wilson_half_width,
    solve_n_min,
    WELL_CALIBRATED_LO,
    WELL_CALIBRATED_HI,
)


# ---------------------------------------------------------------------------
# Synthetic truth generators & estimators
# ---------------------------------------------------------------------------

def gaussian_truth_gen(rng, n_samples, error_level):
    """Generate n_samples from N(θ*, σ²) with optional bias/heteroscedasticity.
    
    θ* = 3.0 (fixed), σ = 1.0
    Level 0: pure noise
    Level 1: estimator will have +0.3 bias (injected at estimator level)
    Level 2: heteroscedastic noise (σ depends on sample index)
    """
    true_theta = 3.0
    if error_level == 0:
        data = rng.normal(true_theta, 1.0, size=n_samples)
    elif error_level == 1:
        # Same data, but the estimator will be biased
        data = rng.normal(true_theta, 1.0, size=n_samples)
    elif error_level == 2:
        # Heteroscedastic: variance grows with index
        sigmas = 0.5 + 0.05 * np.arange(n_samples)
        data = rng.normal(true_theta, sigmas)
    else:
        raise ValueError(f"Unknown error_level {error_level}")
    return data, true_theta


def well_calibrated_estimator(data):
    """Standard t-interval for the mean — well-calibrated for Gaussian data."""
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / math.sqrt(n)
    from scipy import stats
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return float(mean), float(mean - t_crit * se), float(mean + t_crit * se)


def biased_estimator(data):
    """Estimator with +0.3 systematic bias — should show under-coverage."""
    n = len(data)
    mean = np.mean(data) + 0.3  # systematic bias
    se = np.std(data, ddof=1) / math.sqrt(n)
    from scipy import stats
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return float(mean), float(mean - t_crit * se), float(mean + t_crit * se)


def conservative_estimator(data):
    """Estimator with artificially wide CI — should show over-coverage."""
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / math.sqrt(n)
    from scipy import stats
    t_crit = stats.t.ppf(0.995, df=n - 1)  # 99% CI reported as 95%
    return float(mean), float(mean - t_crit * se), float(mean + t_crit * se)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJudgeCoverage:
    def test_well_calibrated(self):
        assert judge_coverage(0.95) == "WELL_CALIBRATED"
        assert judge_coverage(0.93) == "WELL_CALIBRATED"
        assert judge_coverage(0.97) == "WELL_CALIBRATED"

    def test_conservative(self):
        assert judge_coverage(0.98) == "CONSERVATIVE"
        assert judge_coverage(0.99) == "CONSERVATIVE"

    def test_distorted_low(self):
        assert judge_coverage(0.80) == "DISTORTED"
        assert judge_coverage(0.92) == "DISTORTED"

    def test_distorted_high_with_wide_ci(self):
        assert judge_coverage(0.995, ci_width_ratio=3.0) == "DISTORTED"

    def test_conservative_high_but_normal_width(self):
        # coverage > 0.99 but CI width is normal → CONSERVATIVE not DISTORTED
        assert judge_coverage(0.995, ci_width_ratio=1.0) == "CONSERVATIVE"


class TestWilsonInterval:
    def test_known_values(self):
        # 95/100 → p=0.95, Wilson HW should be ~0.043
        hw = wilson_half_width(95, 100)
        assert 0.03 < hw < 0.06

    def test_zero_n(self):
        assert wilson_half_width(0, 0) == 1.0

    def test_monotone_decreasing(self):
        hw_100 = wilson_half_width(95, 100)
        hw_500 = wilson_half_width(475, 500)
        assert hw_500 < hw_100


class TestSolveNMin:
    def test_reasonable_range(self):
        n = solve_n_min(0.95, 0.02)
        # For ±0.02 at p=0.5: n ≈ (1.96/(2*0.02))^2 ≈ 2401
        assert 2000 < n < 3000

    def test_tighter_resolution_needs_more(self):
        n_coarse = solve_n_min(0.95, 0.05)
        n_fine = solve_n_min(0.95, 0.01)
        assert n_fine > n_coarse


class TestCoverageAuditWellCalibrated:
    """Well-calibrated estimator should give coverage ≈ 0.95 at level 0."""

    def test_level0_coverage(self):
        audit = CoverageAudit(
            truth_generator=gaussian_truth_gen,
            estimator=well_calibrated_estimator,
            n_replications=200,
            n_samples=30,
            target_level=0.95,
            seed=42,
            estimator_name="t_interval",
        )
        result = audit.run(error_levels=[0])
        lr = result.levels[0]
        assert lr.verdict == "WELL_CALIBRATED", (
            f"Expected WELL_CALIBRATED, got {lr.verdict} "
            f"(coverage={lr.coverage:.3f})"
        )
        assert 0.88 < lr.coverage < 1.0

    def test_all_levels(self):
        audit = CoverageAudit(
            truth_generator=gaussian_truth_gen,
            estimator=well_calibrated_estimator,
            n_replications=200,
            n_samples=30,
            seed=42,
            estimator_name="t_interval",
        )
        result = audit.run(error_levels=[0, 1, 2])
        assert len(result.levels) == 3
        assert result.overall_verdict in ("WELL_CALIBRATED", "CONSERVATIVE")
        # n_min_table should have entries for all levels
        assert set(result.n_min_table.keys()) == {0, 1, 2}


class TestCoverageAuditBiased:
    """Biased estimator should show under-coverage (DISTORTED)."""

    def test_level0_distorted(self):
        audit = CoverageAudit(
            truth_generator=gaussian_truth_gen,
            estimator=biased_estimator,
            n_replications=200,
            n_samples=30,
            seed=42,
            estimator_name="biased",
        )
        result = audit.run(error_levels=[0])
        lr = result.levels[0]
        # With bias=0.3 and SE≈1/sqrt(30)≈0.183, the CI won't cover θ* often
        assert lr.coverage < 0.93, (
            f"Expected coverage < 0.93 for biased estimator, got {lr.coverage:.3f}"
        )
        assert lr.verdict == "DISTORTED"


class TestCoverageAuditConservative:
    """Conservative estimator should show over-coverage."""

    def test_level0_conservative(self):
        audit = CoverageAudit(
            truth_generator=gaussian_truth_gen,
            estimator=conservative_estimator,
            n_replications=200,
            n_samples=30,
            seed=42,
            estimator_name="conservative",
        )
        result = audit.run(error_levels=[0])
        lr = result.levels[0]
        assert lr.coverage > 0.97, (
            f"Expected coverage > 0.97 for conservative estimator, "
            f"got {lr.coverage:.3f}"
        )
        assert lr.verdict == "CONSERVATIVE"


class TestCoverageAuditExport:
    """Test JSON export."""

    def test_as_dict_roundtrip(self, tmp_path):
        audit = CoverageAudit(
            truth_generator=gaussian_truth_gen,
            estimator=well_calibrated_estimator,
            n_replications=50,
            n_samples=20,
            seed=42,
            estimator_name="test_export",
        )
        result = audit.run(error_levels=[0])
        d = result.as_dict()
        # Verify JSON-serializable
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        assert d2["estimator_name"] == "test_export"
        assert len(d2["levels"]) == 1
        assert d2["levels"][0]["n_replications"] == 50


class TestCoverageCurve:
    """Coverage curve should converge."""

    def test_curve_length(self):
        audit = CoverageAudit(
            truth_generator=gaussian_truth_gen,
            estimator=well_calibrated_estimator,
            n_replications=100,
            n_samples=20,
            seed=42,
            estimator_name="curve_test",
        )
        result = audit.run(error_levels=[0])
        curve = result.levels[0].coverage_curve
        assert len(curve) == 100
        # First value is 0 or 1
        assert curve[0] in (0.0, 1.0)
        # Last value equals the final coverage
        assert abs(curve[-1] - result.levels[0].coverage) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
