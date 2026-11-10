"""Pytest suite for Bayesian MCMC ice–ship friction coefficient calibration.

Tests cover:
1. Forward model correctness (Lindqvist wrapper)
2. Likelihood and prior functions
3. MCMC sampler convergence (synthetic negative control μ=0.15)
4. Gelman–Rubin R̂ < 1.1
5. Effective sample size ESS > 1000
6. Posterior normalization error < 0.1%
7. Prior sensitivity < 5%
8. Reproducibility (same seed → same sha256)
9. 95% CI containment of true value
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from tensorphysics.ice.mcmc_calibration import (
    MU_HULL_ICE,
    CalibrationResult,
    MHChain,
    effective_sample_size,
    gelman_rubin_rhat,
    generate_synthetic_data,
    lindqvist_resistance_wrapper,
    log_likelihood,
    log_prior_normal,
    log_prior_uniform,
    posterior_normalization_error,
    prior_sensitivity_analysis,
    run_full_calibration,
    run_metropolis_hastings,
    run_multiple_chains,
    serialize_result,
)
from tensorphysics.ice.resistance import ShipGeometry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ship():
    return ShipGeometry(
        name="PC6 reference",
        L_pp=150.0,
        B=24.0,
        T=8.5,
        bow_angle_deg=30.0,
        flare_deg=25.0,
    )


@pytest.fixture
def synthetic_data(ship):
    """Synthetic data with μ_true = 0.15."""
    return generate_synthetic_data(
        mu_true=0.15,
        V_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        h_values=np.array([0.5, 1.0, 1.5]),
        sigma_noise=5000.0,
        seed=123,
        ship=ship,
    )


# ---------------------------------------------------------------------------
# Forward model tests
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_lindqvist_wrapper_positive(self, ship):
        R = lindqvist_resistance_wrapper(0.15, 2.0, 1.0, ship=ship)
        assert R > 0.0
        assert math.isfinite(R)

    def test_lindqvist_wrapper_mu_dependence(self, ship):
        R1 = lindqvist_resistance_wrapper(0.10, 2.0, 1.0, ship=ship)
        R2 = lindqvist_resistance_wrapper(0.20, 2.0, 1.0, ship=ship)
        # Higher friction → higher resistance
        assert R2 > R1

    def test_lindqvist_wrapper_invalid_mu(self, ship):
        with pytest.raises(ValueError):
            lindqvist_resistance_wrapper(-0.1, 2.0, 1.0, ship=ship)
        with pytest.raises(ValueError):
            lindqvist_resistance_wrapper(1.5, 2.0, 1.0, ship=ship)

    def test_lindqvist_wrapper_default_mu(self, ship):
        """Default μ=0.15 should give finite positive resistance."""
        R = lindqvist_resistance_wrapper(MU_HULL_ICE, 2.0, 1.0, ship=ship)
        assert R > 0.0


# ---------------------------------------------------------------------------
# Likelihood and prior tests
# ---------------------------------------------------------------------------


class TestLikelihoodAndPrior:
    def test_log_likelihood_finite(self, synthetic_data, ship):
        ll = log_likelihood(
            0.15,
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            ship=ship,
        )
        assert math.isfinite(ll)

    def test_log_likelihood_invalid_mu(self, synthetic_data, ship):
        ll = log_likelihood(
            -0.1,
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            ship=ship,
        )
        assert ll == -np.inf

    def test_log_prior_uniform_in_bounds(self):
        lp = log_prior_uniform(0.15, mu_lo=0.01, mu_hi=0.50)
        assert math.isfinite(lp)

    def test_log_prior_uniform_out_of_bounds(self):
        lp = log_prior_uniform(0.60, mu_lo=0.01, mu_hi=0.50)
        assert lp == -np.inf

    def test_log_prior_normal_finite(self):
        lp = log_prior_normal(0.15, mu_0=0.15, sigma_0=0.05)
        assert math.isfinite(lp)

    def test_log_prior_normal_out_of_domain(self):
        lp = log_prior_normal(-0.1, mu_0=0.15, sigma_0=0.05)
        assert lp == -np.inf


# ---------------------------------------------------------------------------
# MCMC sampler tests
# ---------------------------------------------------------------------------


class TestMHChain:
    def test_chain_runs(self, synthetic_data, ship):
        chain = run_metropolis_hastings(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_init=0.15,
            n_steps=5000,
            burn_in=1000,
            thin=1,
            seed=42,
            ship=ship,
        )
        assert chain.n_samples == 5000
        assert 0.0 < chain.acceptance_rate < 1.0
        assert chain.n_accepted > 0

    def test_chain_posterior_samples(self, synthetic_data, ship):
        chain = run_metropolis_hastings(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_init=0.15,
            n_steps=5000,
            burn_in=1000,
            thin=2,
            seed=42,
            ship=ship,
        )
        samples = chain.posterior_samples()
        assert len(samples) == (5000 - 1000) // 2
        assert np.all(samples > 0.0)
        assert np.all(samples < 1.0)


# ---------------------------------------------------------------------------
# Convergence diagnostics tests
# ---------------------------------------------------------------------------


class TestConvergenceDiagnostics:
    def test_gelman_rubin_rhat(self, synthetic_data, ship):
        chains = run_multiple_chains(
            n_chains=4,
            data_V=synthetic_data["V"],
            data_h=synthetic_data["h"],
            data_R=synthetic_data["R_exp"],
            data_sigma=synthetic_data["sigma"],
            mu_inits=[0.05, 0.10, 0.20, 0.25],
            seeds=[42, 1042, 2042, 3042],
            n_steps=10000,
            burn_in=2000,
            thin=2,
            ship=ship,
        )
        rhat = gelman_rubin_rhat(chains)
        assert rhat < 1.1, f"R̂ = {rhat:.4f} >= 1.1 (not converged)"

    def test_effective_sample_size(self, synthetic_data, ship):
        chain = run_metropolis_hastings(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_init=0.15,
            n_steps=30000,
            burn_in=5000,
            thin=1,
            seed=42,
            ship=ship,
        )
        samples = chain.posterior_samples()
        ess = effective_sample_size(samples)
        assert ess > 1000, f"ESS = {ess} < 1000"


# ---------------------------------------------------------------------------
# Posterior normalization test
# ---------------------------------------------------------------------------


class TestPosteriorNormalization:
    def test_normalization_error(self, synthetic_data, ship):
        chain = run_metropolis_hastings(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_init=0.15,
            n_steps=20000,
            burn_in=5000,
            thin=1,
            seed=42,
            ship=ship,
        )
        samples = chain.posterior_samples()

        def ll_fn(mu):
            return log_likelihood(
                mu,
                synthetic_data["V"],
                synthetic_data["h"],
                synthetic_data["R_exp"],
                synthetic_data["sigma"],
                ship=ship,
            )

        err = posterior_normalization_error(samples, log_prior_uniform, ll_fn, grid_points=10000)
        assert err < 0.001, f"Normalization error = {err:.6f} >= 0.1%"


# ---------------------------------------------------------------------------
# Prior sensitivity test
# ---------------------------------------------------------------------------


class TestPriorSensitivity:
    def test_prior_sensitivity_less_than_5pct(self, synthetic_data, ship):
        result = prior_sensitivity_analysis(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_0_uniform=(0.01, 0.50),
            mu_0_normal=(0.15, 0.05),
            seed=42,
            n_steps=15000,
            burn_in=3000,
            thin=2,
            ship=ship,
        )
        assert result["passed_5pct"], (
            f"Prior sensitivity deviation = {result['relative_deviation']:.4f} >= 5%"
        )


# ---------------------------------------------------------------------------
# Full calibration pipeline tests (negative control)
# ---------------------------------------------------------------------------


class TestFullCalibration:
    """Negative control: μ_gen = 0.15 ⇒ μ_est ∈ [0.14, 0.16] at 95% CI."""

    def test_negative_control_mu_recovery(self, synthetic_data, ship):
        result = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=30000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )

        # 95% CI must contain 0.15
        assert result.mu_95_ci_lower <= 0.15 <= result.mu_95_ci_upper, (
            f"95% CI [{result.mu_95_ci_lower:.4f}, {result.mu_95_ci_upper:.4f}] "
            f"does not contain μ_true=0.15"
        )

        # Posterior mean within [0.14, 0.16]
        assert 0.14 <= result.mu_posterior_mean <= 0.16, (
            f"Posterior mean {result.mu_posterior_mean:.4f} outside [0.14, 0.16]"
        )

    def test_gelman_rubin_in_full_calibration(self, synthetic_data, ship):
        result = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=30000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )
        assert result.gelman_rubin_rhat < 1.1, (
            f"R̂ = {result.gelman_rubin_rhat:.4f} >= 1.1"
        )

    def test_ess_in_full_calibration(self, synthetic_data, ship):
        result = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=30000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )
        assert result.ess > 1000, f"ESS = {result.ess} < 1000"

    def test_normalization_in_full_calibration(self, synthetic_data, ship):
        result = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=30000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )
        assert result.normalization_error < 0.001, (
            f"Normalization error = {result.normalization_error:.6f} >= 0.1%"
        )

    def test_prior_sensitivity_in_full_calibration(self, synthetic_data, ship):
        result = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=30000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )
        assert result.prior_sensitivity["passed_5pct"], (
            f"Prior sensitivity deviation = "
            f"{result.prior_sensitivity['relative_deviation']:.4f} >= 5%"
        )


# ---------------------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_sha256(self, synthetic_data, ship):
        """Same input + same seed → identical artifact sha256."""
        result1 = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=20000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )
        result2 = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=20000,
            burn_in=5000,
            thin=1,
            ship=ship,
        )

        json1 = serialize_result(result1).encode("utf-8")
        json2 = serialize_result(result2).encode("utf-8")
        sha1 = hashlib.sha256(json1).hexdigest()
        sha2 = hashlib.sha256(json2).hexdigest()
        assert sha1 == sha2, f"SHA256 mismatch: {sha1} != {sha2}"

    def test_different_seed_different_chain(self, synthetic_data, ship):
        """Different seed → different chain (probabilistic)."""
        chain1 = run_metropolis_hastings(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_init=0.15,
            n_steps=5000,
            burn_in=1000,
            seed=42,
            ship=ship,
        )
        chain2 = run_metropolis_hastings(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_init=0.15,
            n_steps=5000,
            burn_in=1000,
            seed=999,
            ship=ship,
        )
        # Chains should differ (with overwhelming probability)
        assert not np.allclose(chain1.chain, chain2.chain)


# ---------------------------------------------------------------------------
# Serialization test
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_serialize_result(self, synthetic_data, ship):
        result = run_full_calibration(
            synthetic_data["V"],
            synthetic_data["h"],
            synthetic_data["R_exp"],
            synthetic_data["sigma"],
            mu_true=0.15,
            n_chains=4,
            seed=42,
            n_steps=10000,
            burn_in=2000,
            thin=2,
            ship=ship,
        )
        json_str = serialize_result(result)
        d = json.loads(json_str)
        assert "mu_posterior_mean" in d
        assert "mu_95_ci_lower" in d
        assert "mu_95_ci_upper" in d
        assert "gelman_rubin_rhat" in d
        assert "ess" in d
        assert "normalization_error" in d
        assert "prior_sensitivity" in d
