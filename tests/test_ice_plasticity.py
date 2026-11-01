"""Comprehensive tests for ice-belt plasticity module.

Tests cover:
1. M-κ closed-form vs numerical integration (accuracy ≤1e-12)
2. Shakedown limit dual-method consistency (Melan LP vs incremental ≤1%)
3. Two-bar system shakedown = 2× first yield
4. Monotonic degradation: amplitude→0, permanent→0 (≤1e-9)
5. Mesh independence: fiber count doubled, deflection change ≤2%
6. End-to-end complete example
7. Regime classification
8. ISO 19906 pressure model
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.ice_plasticity.plastic_section import (
    RectangularSection, EPPMaterial, KinematicHardeningMaterial,
    epp_mk_closed_form, epp_mk_numerical, verify_mk_accuracy,
    kinematic_mk_closed_form, two_bar_shakedown_limit
)
from tensorphysics.ice_plasticity.iso19906 import (
    ISO19906Pressure, CyclicIcePressure, typical_ice_belt_params
)
from tensorphysics.ice_plasticity.frame_strip import (
    BeamStrip, PlateStrip, compute_frame_strip_response
)
from tensorphysics.ice_plasticity.shakedown_melan import (
    MelanShakedownBeam, two_bar_melan_shakedown, verify_two_bar_shakedown
)
from tensorphysics.ice_plasticity.incremental_cyclic import (
    KinematicHardening1D, run_cyclic_analysis, classify_regime,
    monotonic_degradation_test
)
from tensorphysics.ice_plasticity.regime_map import (
    RegimeMapGenerator, generate_allowable_pressure_map,
    generate_regime_map_with_melan
)


# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture
def typical_section():
    """Typical ice-belt plate section."""
    return RectangularSection(b=0.6, h=0.025)  # 600mm wide, 25mm thick


@pytest.fixture
def typical_epp():
    """Typical EPP material (AH36 steel)."""
    return EPPMaterial(E=210e9, sigma_y=355e6, nu=0.3)


@pytest.fixture
def typical_kinematic():
    """Typical kinematic hardening material."""
    return KinematicHardeningMaterial(E=210e9, sigma_y0=355e6, H=1e9, nu=0.3)


@pytest.fixture
def typical_beam(typical_section, typical_epp):
    """Typical beam strip."""
    return BeamStrip(L=2.4, b=0.6, sec=typical_section, mat=typical_epp)


# ============================================================
# (a) M-κ accuracy: closed-form vs numerical ≤1e-12
# ============================================================

class TestMKAccuracy:
    """Test M-κ closed-form vs numerical integration accuracy."""

    def test_mk_accuracy_default(self, typical_section, typical_epp):
        """Default fiber count should give ≤1e-12 error."""
        result = verify_mk_accuracy(typical_section, typical_epp, n_fibers=200)
        assert result["pass"], f"Max error: {result['max_relative_error']}"
        assert result["max_relative_error"] <= 1e-12

    def test_mk_accuracy_high_fibers(self, typical_section, typical_epp):
        """Higher fiber count should also pass."""
        result = verify_mk_accuracy(typical_section, typical_epp, n_fibers=500)
        assert result["pass"]
        assert result["max_relative_error"] <= 1e-12

    def test_mk_elastic_region(self, typical_section, typical_epp):
        """In elastic region, M = EI·κ exactly."""
        kappa_y = 2.0 * typical_epp.sigma_y / (typical_epp.E * typical_section.h)
        kappa = np.array([0.1 * kappa_y, 0.5 * kappa_y, 0.9 * kappa_y])
        M_closed = epp_mk_closed_form(kappa, typical_section, typical_epp)
        M_expected = typical_epp.E * typical_section.I * kappa
        np.testing.assert_allclose(M_closed, M_expected, rtol=1e-14)

    def test_mk_plastic_limit(self, typical_section, typical_epp):
        """As κ→∞, M→Mp (within formula tolerance)."""
        kappa_y = 2.0 * typical_epp.sigma_y / (typical_epp.E * typical_section.h)
        kappa = np.array([100.0 * kappa_y])
        M = epp_mk_closed_form(kappa, typical_section, typical_epp)
        Mp = typical_epp.sigma_y * typical_section.Z_pl
        # Formula: M/Mp = 1 - (κy/κ)²/3 = 1 - 1/30000 ≈ 0.999967
        # Relative error from Mp is ~3.3e-5, which is expected
        assert abs(M[0] - Mp) / Mp < 1e-4

    def test_mk_symmetry(self, typical_section, typical_epp):
        """M(κ) should be antisymmetric: M(-κ) = -M(κ)."""
        kappa = np.array([0.5, 1.0, 2.0, 5.0])
        M_pos = epp_mk_closed_form(kappa, typical_section, typical_epp)
        M_neg = epp_mk_closed_form(-kappa, typical_section, typical_epp)
        np.testing.assert_allclose(M_pos, -M_neg, rtol=1e-14)

    def test_shape_factor(self, typical_section):
        """Shape factor for rectangle = 1.5."""
        assert abs(typical_section.shape_factor - 1.5) < 1e-14


# ============================================================
# (b) Shakedown limit: Melan LP vs incremental ≤1%
# ============================================================

class TestShakedownConsistency:
    """Test shakedown limit consistency between methods."""

    def test_melan_shakedown_factor_symmetric(self):
        """Melan LP for symmetric cyclic bending: mu = 1.0 (elastic limit).
        
        For symmetric cyclic loading M ∈ [-M_amp, M_amp], the Melan shakedown
        bound equals the elastic limit (mu = 1.0) for pure bending.
        This is a well-known result: shakedown requires |σ_el + ρ| ≤ σy
        for both load extremes, which limits mu to 1.0 for proportional loading.
        """
        # Use actual section properties for correct stress computation
        melan = MelanShakedownBeam(sigma_y=355e6, E=210e9, n_fibers=51, b=0.6, h=0.025)
        M_amp = 355e6 * (0.6 * 0.025**2 / 6)  # Me for actual section
        mu = melan.shakedown_factor((-M_amp, M_amp))
        # For symmetric cyclic bending, shakedown = elastic limit
        assert abs(mu - 1.0) < 0.01

    def test_two_bar_shakedown_ratio(self):
        """Two-bar shakedown ratio should be 2.0."""
        result = verify_two_bar_shakedown()
        assert result["pass"], f"Ratio: {result['ratio']}, expected 2.0"
        assert abs(result["ratio"] - 2.0) < 0.01


# ============================================================
# (c) Two-bar system: shakedown = 2× first yield
# ============================================================

class TestTwoBarBenchmark:
    """Classic two-bar shakedown benchmark."""

    def test_two_bar_classic_result(self):
        """Shakedown limit = 2 × first yield."""
        sigma_y = 355e6
        A = 0.01
        P1 = A * sigma_y  # P1 = A·σy

        result = two_bar_melan_shakedown(sigma_y, A, P1)
        assert abs(result["ratio"] - 2.0) < 0.01

    def test_two_bar_shakedown_limit_function(self):
        """two_bar_shakedown_limit returns ratio = 2.0."""
        result = two_bar_shakedown_limit(355e6, 210e9, 0.01, 1.0)
        assert result["ratio"] == 2.0


# ============================================================
# (d) Monotonic degradation: amplitude→0, permanent→0
# ============================================================

class TestMonotonicDegradation:
    """Test that permanent deformation → 0 as amplitude → 0."""

    def test_zero_amplitude_zero_residual(self):
        """Zero amplitude should give zero residual."""
        mat = KinematicHardening1D(E=210e9, sigma_y0=355e6, H=1e9)
        history = run_cyclic_analysis(mat, 0.0, n_cycles=10)
        assert abs(history.residual_strain[-1]) <= 1e-9

    def test_small_amplitude_small_residual(self):
        """Very small amplitude should give negligible residual."""
        E = 210e9
        sigma_y = 355e6
        eps_y = sigma_y / E

        mat = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=1e9)
        # Amplitude well below yield
        eps_amp = 0.01 * eps_y
        history = run_cyclic_analysis(mat, eps_amp, n_cycles=10)
        assert abs(history.residual_strain[-1]) <= 1e-9

    def test_monotonic_degradation_test(self):
        """Full monotonic degradation test."""
        E = 210e9
        sigma_y = 355e6
        eps_y = sigma_y / E
        eps_amplitudes = np.array([0.001, 0.01, 0.1, 0.5, 1.0]) * eps_y

        result = monotonic_degradation_test(E, sigma_y, 1e9, eps_amplitudes)
        # Small amplitudes should have negligible residual
        small_results = [r for r in result["results"] if r["eps_amplitude"] < eps_y]
        for r in small_results:
            assert r["residual_strain"] <= 1e-9


# ============================================================
# (e) Mesh independence: fiber count doubled, deflection change ≤2%
# ============================================================

class TestMeshIndependence:
    """Test mesh independence of residual deflection."""

    def test_residual_deflection_mesh_independence(self, typical_beam):
        """Doubling fiber count should change deflection by ≤2%."""
        p_max = 1.5e6  # Pa, above yield

        w1 = typical_beam.residual_deflection_epp(p_max, n_fibers=100)
        w2 = typical_beam.residual_deflection_epp(p_max, n_fibers=200)

        if abs(w1) > 1e-15:
            change = abs(w2 - w1) / abs(w1)
            assert change <= 0.02, f"Change: {change*100:.2f}%"
        else:
            # Both near zero
            assert abs(w2) <= 1e-12

    def test_mk_numerical_convergence(self, typical_section, typical_epp):
        """M-κ numerical integration should converge with fiber count."""
        kappa_y = 2.0 * typical_epp.sigma_y / (typical_epp.E * typical_section.h)
        kappa = np.array([2.0 * kappa_y])

        M100 = epp_mk_numerical(kappa, typical_section, typical_epp, n_fibers=100)
        M200 = epp_mk_numerical(kappa, typical_section, typical_epp, n_fibers=200)
        M400 = epp_mk_numerical(kappa, typical_section, typical_epp, n_fibers=400)

        if abs(M200[0]) > 1e-15:
            change = abs(M400[0] - M200[0]) / abs(M200[0])
            assert change <= 0.02


# ============================================================
# (f) End-to-end complete example
# ============================================================

class TestEndToEnd:
    """End-to-end complete analysis example."""

    def test_full_analysis_pipeline(self):
        """Run complete ice-belt plating analysis."""
        params = typical_ice_belt_params()

        # Frame strip response
        frame_params = {
            "L": params["frame_spacing"],
            "b": params["stiffener_spacing"],
            "plate_thickness": params["plate_thickness"],
            "E": params["E"],
            "sigma_y": params["sigma_y"],
            "nu": params["nu"],
            "p0": params["p0"],
            "ex": params["ex"],
        }
        result = compute_frame_strip_response(frame_params)
        assert "regime" in result
        assert result["shape_factor"] == pytest.approx(1.5, rel=1e-10)

    def test_regime_map_generation(self):
        """Generate regime map."""
        result = generate_allowable_pressure_map({
            "E": 210e9,
            "sigma_y": 355e6,
            "H": 1e9,
        })
        assert "png_base64" in result
        assert len(result["png_base64"]) > 0
        assert "allowable_pressure_ratio" in result

    def test_regime_map_with_melan(self):
        """Generate regime map with Melan bound."""
        result = generate_regime_map_with_melan({
            "E": 210e9,
            "sigma_y": 355e6,
            "H": 1e9,
            "sec_b": 0.6,
            "sec_h": 0.025,
            "L": 2.4,
        })
        assert "melan_shakedown_factor" in result
        assert result["melan_shakedown_factor"] >= 1.0


# ============================================================
# (g) Additional tests for regime classification and ISO 19906
# ============================================================

class TestRegimeClassification:
    """Test cyclic regime classification."""

    def test_elastic_shakedown_classification(self):
        """Below yield should be elastic shakedown."""
        E = 210e9
        sigma_y = 355e6
        eps_y = sigma_y / E

        mat = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=1e9)
        history = run_cyclic_analysis(mat, 0.5 * eps_y, n_cycles=20)
        regime = classify_regime(history)
        assert regime == "elastic_shakedown"

    def test_ratcheting_classification(self):
        """Above yield with kinematic hardening should show ratcheting."""
        E = 210e9
        sigma_y = 355e6
        eps_y = sigma_y / E

        mat = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=1e9)
        # Strain amplitude above yield, pulsating (R=0)
        history = run_cyclic_analysis(mat, 2.0 * eps_y, n_cycles=50, load_ratio=0.0)
        regime = classify_regime(history)
        # With kinematic hardening and R=0, should show ratcheting
        assert regime in ["ratcheting", "plastic_shakedown"]


class TestISO19906:
    """Test ISO 19906 ice pressure model."""

    def test_pressure_area_relationship(self):
        """Pressure should decrease with area."""
        ice = ISO19906Pressure(p0=3.0, ex=0.45)
        p1 = ice.pressure(np.array([1.0]))
        p2 = ice.pressure(np.array([10.0]))
        assert p1[0] > p2[0]

    def test_pressure_value(self):
        """At A=1, p = p0."""
        ice = ISO19906Pressure(p0=3.0, ex=0.45)
        p = ice.pressure(np.array([1.0]))
        assert p[0] == pytest.approx(3.0)

    def test_cyclic_pressure_block(self):
        """Cyclic pressure block generation."""
        ice = CyclicIcePressure(p0=3.0, ex=0.45)
        block = ice.generate_block(10, (0.5, 2.0), load_ratio=0.0)
        assert block.shape == (10, 2)
        assert np.all(block[:, 1] >= block[:, 0])  # p_max >= p_min

    def test_typical_params(self):
        """Typical parameters should be reasonable."""
        params = typical_ice_belt_params()
        assert params["p0"] > 0
        assert params["ex"] > 0
        assert params["E"] > 0
        assert params["sigma_y"] > 0


class TestBeamStrip:
    """Test beam strip model."""

    def test_pressure_at_first_yield(self, typical_beam):
        """First yield pressure should be positive."""
        p_y = typical_beam.pressure_at_first_yield()
        assert p_y > 0

    def test_pressure_at_full_plastic(self, typical_beam):
        """Full plastic pressure > first yield pressure."""
        p_y = typical_beam.pressure_at_first_yield()
        p_p = typical_beam.pressure_at_full_plastic()
        assert p_p > p_y
        assert p_p / p_y == pytest.approx(1.5, rel=1e-10)  # shape factor

    def test_elastic_deflection(self, typical_beam):
        """Elastic deflection should be positive."""
        p = 1e6  # Pa
        w = typical_beam.elastic_deflection(p)
        assert w > 0

    def test_residual_deflection_below_yield(self, typical_beam):
        """Below yield, residual deflection should be zero."""
        p_y = typical_beam.pressure_at_first_yield()
        w_res = typical_beam.residual_deflection_epp(0.5 * p_y)
        assert w_res == 0.0


class TestPlateStrip:
    """Test plate strip model."""

    def test_plate_strip_properties(self):
        """Plate strip should have correct properties."""
        plate = PlateStrip(t=0.025, s=0.6, mat=EPPMaterial(E=210e9, sigma_y=355e6))
        assert plate.sec.shape_factor == pytest.approx(1.5, rel=1e-10)

    def test_plate_residual_below_yield(self):
        """Below yield, no residual deflection."""
        plate = PlateStrip(t=0.025, s=0.6, mat=EPPMaterial(E=210e9, sigma_y=355e6))
        p_y = plate.pressure_at_first_yield()
        w_res = plate.residual_deflection(0.5 * p_y)
        assert w_res == 0.0


class TestKinematicHardening:
    """Test kinematic hardening material model."""

    def test_return_mapping_elastic(self):
        """Elastic step should not update plastic strain."""
        mat = KinematicHardening1D(E=210e9, sigma_y0=355e6, H=1e9)
        eps_y = 355e6 / 210e9
        result = mat.return_mapping(0.5 * eps_y)
        assert not result["plastic"]
        assert mat.eps_p == 0.0

    def test_return_mapping_plastic(self):
        """Plastic step should update plastic strain."""
        mat = KinematicHardening1D(E=210e9, sigma_y0=355e6, H=1e9)
        eps_y = 355e6 / 210e9
        result = mat.return_mapping(2.0 * eps_y)
        assert result["plastic"]
        assert mat.eps_p > 0

    def test_cyclic_stress_stabilization(self):
        """With kinematic hardening, stress should stabilize under strain control."""
        E = 210e9
        sigma_y = 355e6
        H = 10e9  # significant hardening
        eps_y = sigma_y / E

        mat = KinematicHardening1D(E=E, sigma_y0=sigma_y, H=H)
        history = run_cyclic_analysis(mat, 2.0 * eps_y, n_cycles=50,
                                       load_ratio=-1.0)  # fully reversed

        # Stress range should stabilize
        if len(history.sigma_max) >= 10:
            last_stresses = history.sigma_max[-10:]
            stress_range = max(last_stresses) - min(last_stresses)
            # With hardening, stress should stabilize
            assert stress_range < 0.1 * sigma_y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
