"""Tests for non-Gaussian spectral fatigue admission domain.

Covers:
1. Winterstein transform: Gaussian negative control (γ3=0, γ4=3 → identity)
2. Moment matching: transform produces correct (γ3, γ4)
3. Error map: reproducibility (sha256)
4. Admission law: false allow ≤ 5%, false reject ≤ 15%
5. Geometry contract: monotonicity + convexity + no self-intersection
"""

import json
import hashlib
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.non_gaussian_spectral.winterstein import (
    WintersteinTransform,
    winterstein_hermite_transform,
    non_gaussianize_signal,
    gaussianize_signal,
    _compute_hermite_moments,
)
from tensorphysics.non_gaussian_spectral.non_gaussian_spectral import (
    NonGaussianSpectralMoments,
    _winterstein_range_correction_factor,
    spectral_damage_non_gaussian,
)
from tensorphysics.non_gaussian_spectral.admission import (
    AdmissionGrid,
    compute_error_map,
    check_admission,
    validate_geometry_contract,
)
from tensorphysics.fatigue.spectral import SpectralMoments


# ============================================================================
# Test 1: Gaussian negative control — Winterstein identity
# ============================================================================

class TestWintersteinGaussianControl:
    """γ3=0, γ4=3 → Winterstein transform is identity: max|W(u)−u| < 1e-12."""

    def test_identity_at_gaussian(self):
        """For γ3=0, γ4=3, the transform must be the identity."""
        tfm = winterstein_hermite_transform(gamma3=0.0, gamma4=3.0)
        assert abs(tfm.c3) < 1e-12, f"c3 should be 0, got {tfm.c3}"
        assert abs(tfm.c4) < 1e-12, f"c4 should be 0, got {tfm.c4}"

    def test_forward_identity(self):
        """Forward transform at Gaussian point is identity."""
        tfm = winterstein_hermite_transform(gamma3=0.0, gamma4=3.0)
        u = np.linspace(-3, 3, 1000)
        x = tfm.forward(u)
        max_dev = np.max(np.abs(x - u))
        assert max_dev < 1e-12, f"max|W(u)-u| = {max_dev} > 1e-12"

    def test_inverse_identity(self):
        """Inverse transform at Gaussian point is identity."""
        tfm = winterstein_hermite_transform(gamma3=0.0, gamma4=3.0)
        u = np.linspace(-3, 3, 1000)
        u_back = tfm.inverse(u)
        max_dev = np.max(np.abs(u_back - u))
        assert max_dev < 1e-10, f"max|W_inv(x)-x| = {max_dev} > 1e-10"

    def test_roundtrip(self):
        """Forward then inverse returns original."""
        tfm = winterstein_hermite_transform(gamma3=0.5, gamma4=4.0)
        u = np.linspace(-2, 2, 500)
        x = tfm.forward(u)
        u_back = tfm.inverse(x)
        max_dev = np.max(np.abs(u_back - u))
        assert max_dev < 1e-6, f"Roundtrip error = {max_dev}"


# ============================================================================
# Test 2: Moment matching — transform produces correct (γ3, γ4)
# ============================================================================

class TestWintersteinMomentMatching:
    """Verify that the transform produces the prescribed moments."""

    @pytest.mark.parametrize("gamma3,gamma4", [
        (0.0, 3.0),
        (0.5, 3.5),
        (0.8, 4.0),
        (1.0, 5.0),
        (0.3, 4.5),
    ])
    def test_moment_matching_quadrature(self, gamma3, gamma4):
        """Check moments via Gauss-Hermite quadrature (exact for polynomials)."""
        tfm = winterstein_hermite_transform(gamma3=gamma3, gamma4=gamma4)

        # Compute moments via quadrature
        m1, m2, m3_raw, m4_raw = _compute_hermite_moments(tfm.c3, tfm.c4)

        var = m2
        skew = m3_raw / var**1.5
        kurt = m4_raw / var**2

        # Check skewness
        if abs(gamma3) > 0.1:
            rel_err_g3 = abs(skew - gamma3) / abs(gamma3)
            assert rel_err_g3 < 0.10, (
                f"Skewness mismatch: target={gamma3}, computed={skew:.4f}, "
                f"rel_err={rel_err_g3:.3f}"
            )
        else:
            assert abs(skew) < 0.1, f"Skewness should be ~0, got {skew}"

        # Check kurtosis
        rel_err_g4 = abs(kurt - gamma4) / gamma4
        assert rel_err_g4 < 0.05, (
            f"Kurtosis mismatch: target={gamma4}, computed={kurt:.4f}, "
            f"rel_err={rel_err_g4:.3f}"
        )

    @pytest.mark.parametrize("gamma3,gamma4", [
        (0.5, 3.5),
        (0.8, 4.0),
    ])
    def test_moment_matching_samples(self, gamma3, gamma4):
        """Generate samples and check empirical moments match target."""
        tfm = winterstein_hermite_transform(gamma3=gamma3, gamma4=gamma4)
        rng = np.random.RandomState(42)
        u = rng.randn(200000)
        x = tfm.forward(u)

        # Standardize
        x = (x - x.mean()) / x.std()

        emp_g3 = float(np.mean(x**3))
        emp_g4 = float(np.mean(x**4))

        # Allow 20% tolerance for skewness (sampling noise), 10% for kurtosis
        if abs(gamma3) > 0.1:
            rel_err_g3 = abs(emp_g3 - gamma3) / abs(gamma3)
            assert rel_err_g3 < 0.20, (
                f"Skewness mismatch: target={gamma3}, empirical={emp_g3:.4f}, "
                f"rel_err={rel_err_g3:.3f}"
            )

        rel_err_g4 = abs(emp_g4 - gamma4) / gamma4
        assert rel_err_g4 < 0.10, (
            f"Kurtosis mismatch: target={gamma4}, empirical={emp_g4:.4f}, "
            f"rel_err={rel_err_g4:.3f}"
        )


# ============================================================================
# Test 3: Correction factor — Gaussian gives factor 1
# ============================================================================

class TestCorrectionFactor:
    """Winterstein correction factor = 1 for Gaussian."""

    def test_gaussian_correction_is_one(self):
        """For γ3=0, γ4=3, correction factor must be 1.0."""
        factor = _winterstein_range_correction_factor(0.0, 3.0, m=3.0)
        assert abs(factor - 1.0) < 1e-10, f"Correction factor = {factor}, expected 1.0"

    def test_correction_increases_with_kurtosis(self):
        """For positive excess kurtosis, correction factor > 1."""
        factor_3 = _winterstein_range_correction_factor(0.0, 3.0, m=3.0)
        factor_5 = _winterstein_range_correction_factor(0.0, 5.0, m=3.0)
        assert factor_5 > factor_3, (
            f"Correction should increase with kurtosis: "
            f"factor(γ4=5)={factor_5} vs factor(γ4=3)={factor_3}"
        )


# ============================================================================
# Test 4: Error map reproducibility
# ============================================================================

class TestErrorMapReproducibility:
    """Error map must be reproducible: same seed → same sha256."""

    def test_sha256_reproducibility(self):
        """Two runs with same parameters produce identical sha256."""
        grid1 = compute_error_map(
            gamma3_values=np.array([0.0, 0.5, 1.0]),
            gamma4_values=np.array([3.0, 4.0, 5.0]),
            alpha2_values=np.array([0.3, 0.6, 0.9]),
            n_realizations=2,
            duration_s=60.0,  # Short for speed
            seed_base=42,
        )
        grid2 = compute_error_map(
            gamma3_values=np.array([0.0, 0.5, 1.0]),
            gamma4_values=np.array([3.0, 4.0, 5.0]),
            alpha2_values=np.array([0.3, 0.6, 0.9]),
            n_realizations=2,
            duration_s=60.0,
            seed_base=42,
        )
        sha1 = grid1.sha256()
        sha2 = grid2.sha256()
        assert sha1 == sha2, f"SHA256 mismatch: {sha1} vs {sha2}"


# ============================================================================
# Test 5: Admission law accuracy
# ============================================================================

class TestAdmissionLawAccuracy:
    """Admission law: false allow ≤ 5%, false reject ≤ 15%."""

    def test_admission_accuracy(self):
        """Check admission law on hold-out points."""
        # Compute grid (small for speed)
        grid = compute_error_map(
            gamma3_values=np.array([0.0, 0.5, 1.0, 1.5]),
            gamma4_values=np.array([3.0, 4.0, 5.0, 7.0]),
            alpha2_values=np.array([0.3, 0.6, 0.9]),
            n_realizations=2,
            duration_s=60.0,
            seed_base=99,
        )

        # Hold-out points (not on the grid)
        hold_out = [
            (0.25, 3.5, 0.45),
            (0.75, 4.5, 0.75),
            (1.25, 6.0, 0.5),
        ]

        false_allows = 0
        false_rejects = 0
        total = len(hold_out)

        for g3, g4, a2 in hold_out:
            result = check_admission(g3, g4, a2, grid, method="dirlik_ng")

            # Compute actual error at this point by running a fresh simulation
            from tensorphysics.non_gaussian_spectral.admission import (
                _make_psd_callable,
                _compute_rainflow_damage,
                _compute_spectral_damage_all_methods,
            )
            from tensorphysics.fatigue.spectral import (
                synthesize_gaussian_history,
                moments_from_dict,
                spectral_moments,
            )

            psd_func = _make_psd_callable(a2)
            f_grid = np.linspace(0.01, 10.0, 256)
            psd_grid = psd_func(f_grid)
            mom_dict = spectral_moments(f_grid, psd_grid)
            moments = moments_from_dict(mom_dict, f_lo=float(f_grid[0]), f_hi=float(f_grid[-1]))

            n_samples = 1200
            gauss = synthesize_gaussian_history(psd_func, n_samples, 20.0, seed=77777)
            gauss_norm = (gauss - gauss.mean()) / max(gauss.std(), 1e-10)
            tfm = winterstein_hermite_transform(g3, g4, bounded=True, bound_sigma=4.0)
            ng_signal = tfm.forward(gauss_norm) * moments.sigma

            D_rf = _compute_rainflow_damage(ng_signal, 20.0, 1e12, 3.0)
            spec_d = _compute_spectral_damage_all_methods(moments, g3, g4, 1e12, 3.0)
            D_spec = spec_d["dirlik_ng"]

            actual_error = abs(D_spec - D_rf) / max(D_rf, 1e-300)
            actually_admitted = actual_error <= 0.10

            if result.admitted and not actually_admitted:
                false_allows += 1
            if not result.admitted and actually_admitted:
                false_rejects += 1

        false_allow_rate = false_allows / total
        false_reject_rate = false_rejects / total

        assert false_allow_rate <= 0.05, (
            f"False allow rate = {false_allow_rate:.2%} > 5%"
        )
        assert false_reject_rate <= 0.15, (
            f"False reject rate = {false_reject_rate:.2%} > 15%"
        )


# ============================================================================
# Test 6: Geometry contract
# ============================================================================

class TestGeometryContract:
    """Monotonicity + convexity + no self-intersection."""

    def test_geometry_contract(self):
        """All three geometry checks must pass."""
        grid = compute_error_map(
            gamma3_values=np.array([0.0, 0.5, 1.0, 1.5]),
            gamma4_values=np.array([3.0, 4.0, 5.0, 7.0]),
            alpha2_values=np.array([0.3, 0.6, 0.9]),
            n_realizations=2,
            duration_s=60.0,
            seed_base=55,
        )

        contract = validate_geometry_contract(grid, method="dirlik_ng")
        assert contract["monotonicity_pass"], (
            f"Monotonicity failed: g3={contract['monotonicity_gamma3']}, "
            f"g4={contract['monotonicity_gamma4']}"
        )
        assert contract["convexity_pass"], "Convexity failed"
        assert contract["no_self_intersection_pass"], "No self-intersection failed"
        assert contract["all_pass"], "Overall geometry contract failed"


# ============================================================================
# Test 7: Gaussian negative control — four spectral methods agree
# ============================================================================

class TestGaussianFourMethodAgreement:
    """At γ3=0, γ4=3, all four spectral methods should give <5% relative diff
    for a narrow-band process (α₂ > 0.95)."""

    def test_four_methods_agree_at_gaussian_narrowband(self):
        """All methods should agree within 5% for narrow-band Gaussian process."""
        # Create a very narrow-band PSD (α₂ close to 1)
        f = np.linspace(0.01, 10.0, 1024)
        fp = 1.0
        sigma_f = 0.05  # Very narrow
        psd = np.exp(-0.5 * ((f - fp) / sigma_f)**2)

        result = spectral_damage_non_gaussian(
            f, psd, gamma3=0.0, gamma4=3.0, C=1e12, m=3.0
        )

        damages = result["damage_rate_per_s"]
        D_rayleigh = damages["rayleigh_gaussian"]
        D_dirlik = damages["dirlik_gaussian"]
        D_bt = damages["bt_gaussian"]

        # For narrow-band, all should be close
        rel_diff_dirlik = abs(D_dirlik - D_rayleigh) / max(D_rayleigh, 1e-300)
        rel_diff_bt = abs(D_bt - D_rayleigh) / max(D_rayleigh, 1e-300)

        assert rel_diff_dirlik < 0.05, (
            f"Dirlik vs Rayleigh diff = {rel_diff_dirlik:.3%} > 5%"
        )
        assert rel_diff_bt < 0.05, (
            f"BT vs Rayleigh diff = {rel_diff_bt:.3%} > 5%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
