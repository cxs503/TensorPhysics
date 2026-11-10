"""Tests for P210W2-N02: Ice Resistance Maturity (unified pipeline).

Acceptance criteria:
1. Sensitivity matrix condition number < 100 (identifiable)
2. Posterior 95% CI width < 0.1 for μ
3. Deviation from Lindqvist default registered
4. Idempotent reproduction (SHA-256 match)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorphysics.ice.ice_resistance_maturity import (
    MaturityResult,
    compute_sensitivity_condition_number,
    run_maturity_pipeline,
    write_maturity_artifacts,
)
from tensorphysics.ice.cross_impl_attribution import RefShip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ship():
    return RefShip()


@pytest.fixture
def fast_result():
    """Fast maturity run for basic tests (low MCMC budget)."""
    return run_maturity_pipeline(
        h=1.0, V=3.0,
        n_mcmc_steps=5000,
        n_mcmc_chains=2,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Criterion 1: Sensitivity matrix condition number < 100
# ---------------------------------------------------------------------------


class TestConditionNumber:
    """Sensitivity matrix condition number must be < 100 for identifiability."""

    def test_condition_number_computed(self, ship):
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        assert "condition_number" in result
        assert isinstance(result["condition_number"], float)

    def test_condition_number_finite(self, ship):
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        assert np.isfinite(result["condition_number"])

    def test_condition_number_lt_100(self, ship):
        """Criterion 1: κ < 100."""
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        assert result["condition_number"] < 100.0, (
            f"Condition number {result['condition_number']:.2f} >= 100"
        )

    def test_is_identifiable(self, ship):
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        assert result["is_identifiable"] is True

    def test_singular_values_positive(self, ship):
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        for sv in result["singular_values"]:
            assert sv > 0

    def test_rank_equals_n_components(self, ship):
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        assert result["rank"] == result["n_components"]

    def test_recommendation_present(self, ship):
        result = compute_sensitivity_condition_number(1.0, 3.0, ship)
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0

    def test_condition_number_stable_across_speeds(self, ship):
        """Condition number should be stable across a range of speeds."""
        for V in [1.0, 2.0, 3.0, 5.0]:
            result = compute_sensitivity_condition_number(1.0, V, ship)
            assert result["condition_number"] < 100.0, (
                f"κ={result['condition_number']:.2f} at V={V} m/s"
            )

    def test_condition_number_stable_across_thickness(self, ship):
        """Condition number should be stable across a range of thicknesses."""
        for h in [0.5, 1.0, 1.5, 2.0]:
            result = compute_sensitivity_condition_number(h, 3.0, ship)
            assert result["condition_number"] < 100.0, (
                f"κ={result['condition_number']:.2f} at h={h} m"
            )


# ---------------------------------------------------------------------------
# Criterion 2: Posterior 95% CI width < 0.1
# ---------------------------------------------------------------------------


class TestCIWidth:
    """Posterior 95% CI width for μ must be < 0.1."""

    def test_ci_width_computed(self, fast_result):
        cal = fast_result.calibration_result
        ci_width = cal.mu_95_ci_upper - cal.mu_95_ci_lower
        assert ci_width > 0

    def test_ci_width_lt_0_1(self, fast_result):
        """Criterion 2: CI width < 0.1."""
        cal = fast_result.calibration_result
        ci_width = cal.mu_95_ci_upper - cal.mu_95_ci_lower
        assert ci_width < 0.1, f"CI width {ci_width:.4f} >= 0.1"

    def test_ci_contains_true_value(self, fast_result):
        """95% CI should contain the true μ (synthetic data)."""
        cal = fast_result.calibration_result
        if not np.isnan(cal.mu_true):
            assert cal.mu_95_ci_lower <= cal.mu_true <= cal.mu_95_ci_upper, (
                f"95% CI [{cal.mu_95_ci_lower:.4f}, {cal.mu_95_ci_upper:.4f}] "
                f"does not contain μ_true={cal.mu_true:.4f}"
            )


# ---------------------------------------------------------------------------
# Criterion 3: Deviation from Lindqvist default registered
# ---------------------------------------------------------------------------


class TestDeviationRegistered:
    """Deviation from Lindqvist default μ=0.15 must be computed and registered."""

    def test_deviation_computed(self, fast_result):
        cal = fast_result.calibration_result
        assert hasattr(cal, "deviation_from_default")
        assert isinstance(cal.deviation_from_default, float)

    def test_mu_default_is_lindqvist(self, fast_result):
        cal = fast_result.calibration_result
        assert cal.mu_default == 0.15

    def test_deviation_equals_mean_minus_default(self, fast_result):
        cal = fast_result.calibration_result
        expected = cal.mu_posterior_mean - cal.mu_default
        assert abs(cal.deviation_from_default - expected) < 1e-10

    def test_criterion_3_passes(self, fast_result):
        assert fast_result.criterion_3_deviation_registered is True


# ---------------------------------------------------------------------------
# Criterion 4: Idempotent reproduction
# ---------------------------------------------------------------------------


class TestIdempotent:
    """Same input → same SHA-256 (deterministic pipeline)."""

    def test_sha256_non_empty(self, fast_result):
        assert len(fast_result.determinism_sha256) == 64

    def test_sha256_hex(self, fast_result):
        # Must be valid hex
        int(fast_result.determinism_sha256, 16)

    def test_reproducible(self):
        """Run pipeline twice with same seed → same SHA-256."""
        r1 = run_maturity_pipeline(h=1.0, V=3.0, n_mcmc_steps=3000, seed=42)
        r2 = run_maturity_pipeline(h=1.0, V=3.0, n_mcmc_steps=3000, seed=42)
        assert r1.determinism_sha256 == r2.determinism_sha256

    def test_criterion_4_passes(self, fast_result):
        assert fast_result.criterion_4_idempotent is True


# ---------------------------------------------------------------------------
# Cross-implementation divergence
# ---------------------------------------------------------------------------


class TestDivergence:
    """Cross-implementation divergence attribution."""

    def test_four_implementations_decomposed(self, fast_result):
        assert len(fast_result.decompositions) == 4
        for impl in ["TP_level", "IA_lindqvist", "Lit_original", "Lit_corrected"]:
            assert impl in fast_result.decompositions

    def test_five_components_each(self, fast_result):
        for impl, decomp in fast_result.decompositions.items():
            for comp in ["R_crush_N", "R_bend_N", "R_friction_N", "R_momentum_N", "R_buoyancy_N"]:
                assert comp in decomp, f"Missing {comp} in {impl}"

    def test_ia_vs_tp_ratio_reasonable(self, fast_result):
        """IA vs TP ratio should be in the range 2-5× (known defect)."""
        ratio = fast_result.divergence_ia_vs_tp["ratio"]
        assert 1.5 < ratio < 10.0, f"IA/TP ratio {ratio:.2f} out of expected range"

    def test_lit_corrected_equals_tp(self, fast_result):
        """Lit_corrected should be identical to TP_level."""
        tp = fast_result.decompositions["TP_level"]
        lit = fast_result.decompositions["Lit_corrected"]
        assert abs(tp["R_total_N"] - lit["R_total_N"]) < 1e-10

    def test_divergence_components_sum_to_total(self, fast_result):
        """Component diffs should sum to total diff."""
        div = fast_result.divergence_ia_vs_tp
        assert div["relative_residual"] < 1e-9


# ---------------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------------


class TestMCMCDiagnostics:
    """MCMC convergence diagnostics."""

    def test_gelman_rubin_computed(self, fast_result):
        cal = fast_result.calibration_result
        assert np.isfinite(cal.gelman_rubin_rhat)

    def test_ess_positive(self, fast_result):
        cal = fast_result.calibration_result
        assert cal.ess > 0

    def test_acceptance_rate_reasonable(self, fast_result):
        cal = fast_result.calibration_result
        assert 0.05 < cal.acceptance_rate < 0.9


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_all_criteria_passed(self, fast_result):
        d = fast_result.as_dict()
        assert d["all_criteria_passed"] is True

    def test_as_dict_complete(self, fast_result):
        d = fast_result.as_dict()
        assert "mission_id" in d
        assert "node" in d
        assert "generated_utc" in d
        assert "divergence" in d
        assert "decompositions" in d
        assert "sensitivity_condition" in d
        assert "calibration" in d
        assert "acceptance_criteria" in d

    def test_artifacts_written(self, tmp_path):
        result = run_maturity_pipeline(h=1.0, V=3.0, n_mcmc_steps=3000, seed=42)
        paths = write_maturity_artifacts(result, outdir=tmp_path)
        assert Path(paths["report_json"]).exists()
        assert Path(paths["summary_txt"]).exists()

    def test_report_json_valid(self, tmp_path):
        result = run_maturity_pipeline(h=1.0, V=3.0, n_mcmc_steps=3000, seed=42)
        paths = write_maturity_artifacts(result, outdir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text())
        assert "acceptance_criteria" in data
        assert "all_criteria_passed" in data

    def test_mission_id(self, fast_result):
        assert fast_result.mission_id == 9966

    def test_node_id(self, fast_result):
        assert fast_result.node == "P210W2-N02"
