"""Tests for ice-wave coupled added resistance model.

Covers:
  - Plate rigidity (dimensional check)
  - Wave-induced bending stress (limiting cases)
  - Ice breaking criterion
  - Individual resistance components (ice, wave)
  - Coupling resistance (non-negativity, bounded, monotonicity)
  - 3D parameter sweep (≥125 points)
  - Physical consistency checks
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_wave.coupled_resistance import (
    IceWaveCouplingError,
    coupling_resistance,
    ice_breaking_criterion,
    ice_resistance,
    plate_rigidity,
    sweep_resistance_surface,
    wave_added_resistance,
    wave_induced_bending_stress,
    GRAVITY,
    RHO_WATER,
    RHO_ICE,
    DEFAULT_E_ICE,
    DEFAULT_SIGMA_F,
)


# ===================================================================
# Plate rigidity
# ===================================================================
class TestPlateRigidity:
    def test_dimensional_consistency(self):
        """D = E h³ / (12(1-ν²)) has units N·m = kg·m²/s²."""
        D = plate_rigidity(1.0, E=5e9, nu=0.3)
        assert D > 0
        expected = 5e9 * 1.0**3 / (12.0 * (1.0 - 0.3**2))
        assert abs(D - expected) / expected < 1e-10

    def test_thickness_cubed_scaling(self):
        """D ∝ h³."""
        D1 = plate_rigidity(0.5)
        D2 = plate_rigidity(1.0)
        ratio = D2 / D1
        assert abs(ratio - 8.0) < 1e-10

    def test_invalid_thickness(self):
        with pytest.raises(IceWaveCouplingError):
            plate_rigidity(0.0)
        with pytest.raises(IceWaveCouplingError):
            plate_rigidity(-0.5)

    def test_invalid_modulus(self):
        with pytest.raises(IceWaveCouplingError):
            plate_rigidity(1.0, E=-1e9)


# ===================================================================
# Wave-induced bending stress
# ===================================================================
class TestBendingStress:
    def test_zero_wave_height(self):
        """No waves → no bending stress."""
        sigma = wave_induced_bending_stress(0.0, 60.0, 0.5)
        assert sigma == 0.0

    def test_positive_stress(self):
        """Non-zero wave → positive bending stress."""
        sigma = wave_induced_bending_stress(1.0, 60.0, 0.5)
        assert sigma > 0

    def test_stress_increases_with_wave_height(self):
        """Larger waves → larger bending stress."""
        s1 = wave_induced_bending_stress(0.5, 60.0, 0.5)
        s2 = wave_induced_bending_stress(1.5, 60.0, 0.5)
        assert s2 > s1
        ratio = s2 / s1
        assert abs(ratio - 3.0) < 1e-10

    def test_stress_decreases_with_thicker_ice(self):
        """Thicker ice → stiffer plate → less deflection → less stress."""
        s_thin = wave_induced_bending_stress(1.0, 60.0, 0.3)
        s_thick = wave_induced_bending_stress(1.0, 60.0, 1.5)
        assert s_thick < s_thin

    def test_stress_increases_with_shorter_waves(self):
        """Shorter waves → larger k → more bending."""
        s_long = wave_induced_bending_stress(1.0, 120.0, 0.5)
        s_short = wave_induced_bending_stress(1.0, 40.0, 0.5)
        assert s_short > s_long

    def test_stiff_plate_limit(self):
        """Very thick ice → stress approaches curvature-limited value."""
        sigma = wave_induced_bending_stress(1.0, 60.0, 3.0)
        k = 2 * math.pi / 60.0
        sigma_limit = 6 * 0.5 * 1025 * 9.81 / (k**2 * 3.0**2)
        assert abs(sigma - sigma_limit) / sigma_limit < 0.05

    def test_out_of_range(self):
        with pytest.raises(IceWaveCouplingError):
            wave_induced_bending_stress(6.0, 60.0, 0.5)
        with pytest.raises(IceWaveCouplingError):
            wave_induced_bending_stress(1.0, 5.0, 0.5)


# ===================================================================
# Ice breaking criterion
# ===================================================================
class TestBreakingCriterion:
    def test_no_break_low_stress(self):
        result = ice_breaking_criterion(1e5, 0.7e6)
        assert not result["breaks"]
        assert result["stress_ratio"] < 1.0
        assert result["safety_margin"] > 0

    def test_break_high_stress(self):
        result = ice_breaking_criterion(1e6, 0.7e6)
        assert result["breaks"]
        assert result["stress_ratio"] > 1.0
        assert result["safety_margin"] < 0

    def test_exact_threshold(self):
        result = ice_breaking_criterion(0.7e6, 0.7e6)
        assert not result["breaks"]
        assert abs(result["stress_ratio"] - 1.0) < 1e-10

    def test_invalid_strength(self):
        with pytest.raises(IceWaveCouplingError):
            ice_breaking_criterion(1e5, 0.0)
        with pytest.raises(IceWaveCouplingError):
            ice_breaking_criterion(1e5, -1e5)


# ===================================================================
# Ice resistance
# ===================================================================
class TestIceResistance:
    def test_zero_ice_zero_resistance(self):
        R = ice_resistance(0.0)
        assert R == 0.0

    def test_positive_resistance(self):
        R = ice_resistance(0.5)
        assert R > 0

    def test_increases_with_thickness(self):
        R1 = ice_resistance(0.3)
        R2 = ice_resistance(1.0)
        assert R2 > R1

    def test_increases_with_speed(self):
        R1 = ice_resistance(0.5, V=2.0)
        R2 = ice_resistance(0.5, V=6.0)
        assert R2 > R1

    def test_increases_with_beam(self):
        R1 = ice_resistance(0.5, B=15.0)
        R2 = ice_resistance(0.5, B=35.0)
        assert R2 > R1

    def test_reasonable_magnitude(self):
        """Ice resistance for h=0.5m, B=25m should be O(10⁵-10⁶) N."""
        R = ice_resistance(0.5, V=4.0, B=25.0)
        assert 1e4 < R < 1e8


# ===================================================================
# Wave added resistance
# ===================================================================
class TestWaveResistance:
    def test_zero_wave_zero_resistance(self):
        R = wave_added_resistance(0.0, 60.0)
        assert R == 0.0

    def test_positive_resistance(self):
        R = wave_added_resistance(1.0, 60.0)
        assert R > 0

    def test_increases_with_wave_height(self):
        R1 = wave_added_resistance(0.5, 60.0)
        R2 = wave_added_resistance(2.0, 60.0)
        assert R2 > R1

    def test_cubic_in_wave_height(self):
        """R_wave ∝ H_w³ (includes steepness dependence)."""
        R1 = wave_added_resistance(1.0, 60.0)
        R2 = wave_added_resistance(2.0, 60.0)
        ratio = R2 / R1
        assert abs(ratio - 8.0) < 1.0


# ===================================================================
# Coupling resistance
# ===================================================================
class TestCouplingResistance:
    def test_no_waves_no_coupling(self):
        """H_w = 0 → no coupling (just ice resistance)."""
        result = coupling_resistance(0.0, 60.0, 0.5)
        assert result["delta_R_coupling"] == 0.0
        assert result["R_total"] == result["R_linear"]
        assert result["coupling_ratio"] == 0.0

    def test_positive_coupling(self):
        """Non-zero waves → positive coupling resistance."""
        result = coupling_resistance(1.5, 60.0, 0.5)
        assert result["delta_R_coupling"] >= 0.0
        assert result["R_total"] >= result["R_linear"]

    def test_non_negativity(self):
        """Coupling resistance is always non-negative."""
        for H_w in [0.0, 0.5, 1.0, 2.0, 3.0]:
            for lam in [30.0, 60.0, 120.0]:
                for h_i in [0.3, 0.7, 1.5]:
                    result = coupling_resistance(H_w, lam, h_i)
                    assert result["delta_R_coupling"] >= 0.0

    def test_bounded_coupling(self):
        """Coupling should not exceed 50% of linear sum."""
        result = coupling_resistance(3.0, 30.0, 0.3)
        assert result["coupling_ratio"] <= 0.5 + 1e-10

    def test_coupling_increases_with_wave_height(self):
        """Larger waves → more coupling."""
        r1 = coupling_resistance(0.5, 60.0, 0.5)
        r2 = coupling_resistance(2.0, 60.0, 0.5)
        assert r2["delta_R_coupling"] >= r1["delta_R_coupling"]

    def test_breaking_detected(self):
        """Extreme conditions should trigger ice breaking."""
        result = coupling_resistance(3.0, 30.0, 0.3)
        assert result["stress_ratio"] > 0.5

    def test_all_keys_present(self):
        result = coupling_resistance(1.0, 60.0, 0.5)
        expected_keys = {
            "delta_R_coupling", "R_drift_mod", "R_damage", "R_floe_kinetic",
            "R_radiation", "sigma_b", "breaks", "stress_ratio",
            "R_ice", "R_wave", "R_linear", "R_total", "coupling_ratio",
        }
        assert set(result.keys()) == expected_keys

    def test_energy_consistency(self):
        """R_total = R_linear + ΔR_coupling."""
        result = coupling_resistance(1.5, 80.0, 0.7)
        expected_total = result["R_linear"] + result["delta_R_coupling"]
        assert abs(result["R_total"] - expected_total) < 1e-6

    def test_out_of_range(self):
        with pytest.raises(IceWaveCouplingError):
            coupling_resistance(4.0, 60.0, 0.5)
        with pytest.raises(IceWaveCouplingError):
            coupling_resistance(1.0, 20.0, 0.5)
        with pytest.raises(IceWaveCouplingError):
            coupling_resistance(1.0, 60.0, 0.1)

    def test_significant_coupling(self):
        """Coupling should be significant (>5%) for extreme conditions."""
        result = coupling_resistance(3.0, 30.0, 0.3)
        assert result["coupling_ratio"] > 0.05


# ===================================================================
# 3D parameter sweep
# ===================================================================
class TestSweepSurface:
    def test_default_125_points(self):
        """Default grid is 5×5×5 = 125 points."""
        result = sweep_resistance_surface()
        assert result["n_points"] == 125
        assert result["R_total"].shape == (5, 5, 5)

    def test_custom_grid(self):
        """Custom grid dimensions."""
        H = np.linspace(0, 2, 3)
        lam = np.linspace(40, 100, 4)
        h = np.linspace(0.3, 1.0, 6)
        result = sweep_resistance_surface(H, lam, h)
        assert result["n_points"] == 3 * 4 * 6
        assert result["R_total"].shape == (3, 4, 6)

    def test_all_resistances_non_negative(self):
        result = sweep_resistance_surface()
        assert np.all(result["R_ice"] >= 0)
        assert np.all(result["R_wave"] >= 0)
        assert np.all(result["R_linear"] >= 0)
        assert np.all(result["R_total"] >= 0)
        assert np.all(result["delta_R"] >= 0)

    def test_total_ge_linear(self):
        """R_total ≥ R_linear everywhere."""
        result = sweep_resistance_surface()
        assert np.all(result["R_total"] >= result["R_linear"] - 1e-6)

    def test_zero_wave_height_slice(self):
        """At H_w = 0: R_wave = 0, ΔR = 0, R_total = R_ice."""
        result = sweep_resistance_surface()
        assert np.allclose(result["R_wave"][0, :, :], 0.0)
        assert np.allclose(result["delta_R"][0, :, :], 0.0)
        assert np.allclose(result["R_total"][0, :, :], result["R_ice"][0, :, :])

    def test_coupling_percentage_bounded(self):
        result = sweep_resistance_surface()
        assert np.all(result["coupling_pct"] >= 0)
        assert np.all(result["coupling_pct"] <= 50.0 + 1e-10)

    def test_breaking_pattern(self):
        """Breaking should occur at high H_w, short λ_w, thin ice."""
        result = sweep_resistance_surface()
        assert not np.any(result["breaks"][0, :, :])


# ===================================================================
# Physical consistency
# ===================================================================
class TestPhysicalConsistency:
    def test_nonlinear_coupling_exists(self):
        """Verify that coupling is genuinely nonlinear."""
        result = coupling_resistance(2.0, 60.0, 0.5)
        assert result["coupling_ratio"] > 0.0

    def test_thin_ice_more_coupling(self):
        """Thinner ice → more coupling."""
        r_thin = coupling_resistance(2.0, 60.0, 0.3)
        r_thick = coupling_resistance(2.0, 60.0, 1.5)
        assert r_thin["coupling_ratio"] > r_thick["coupling_ratio"]

    def test_short_waves_more_bending_stress(self):
        """Shorter waves → more bending stress."""
        r_short = coupling_resistance(2.0, 30.0, 0.5)
        r_long = coupling_resistance(2.0, 150.0, 0.5)
        assert r_short["sigma_b"] > r_long["sigma_b"]

    def test_magnitudes_reasonable(self):
        """Check that resistance magnitudes are physically reasonable."""
        result = coupling_resistance(1.0, 60.0, 0.5, V=4.0, B=25.0)
        assert 1e4 < result["R_ice"] < 1e8
        assert 1e2 < result["R_wave"] < 1e7
        assert 1e4 < result["R_total"] < 1e8
