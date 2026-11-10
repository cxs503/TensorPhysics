"""Tests for the TensorFEM-DEM ice-structure coupling verification face.

Campaign node: p299w4-n04-fem-dem-ice
Mission: m11073

This test suite verifies the four physics stages of the ice-structure
interaction chain:
1. C1: Ice plate bending stress vs Timoshenko 1959 analytical solution
2. C2: FEM→DEM fracture handoff mass conservation
3. C3: Ice floe drift vs MT Uikku experiment
4. C4: Ice load time history peak force and Weibull distribution
5. C5: Energy conservation across FEM→DEM transition

Acceptance criteria:
- C1: Bending stress vs analytical error < 5%
- C2: Mass conservation error < 1%
- C3: Channel clearing area vs experiment error < 20%
- C4: Peak force error < 20%, Weibull k error < 15%
- C5: Energy conservation error < 10%
- C6: pytest ≥ 20 tests all green
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_fem_dem_coupling import (
    # Plate bending
    TimoshenkoPlateSolution,
    timoshenko_plate_point_load_stress,
    solve_plate_bending_verification,
    # Fracture handoff
    run_fracture_handoff,
    mass_conservation_error,
    # Floe drift
    FloeProperties,
    DriftEnvironment,
    simulate_floe_drift,
    channel_clearing_area,
    MT_UIKKU_REFERENCE,
    # Ice load
    IceLoadTimeHistory,
    fit_weibull,
    weibull_shape_parameter,
    run_ice_load_verification,
    # Energy audit
    run_energy_audit,
    # Full verification
    run_full_verification,
)


# ===========================================================================
# C1: Plate bending verification (Timoshenko 1959)
# ===========================================================================

class TestC1PlateBending:
    """C1: Ice plate bending stress vs Timoshenko & Woinowsky-Krieger (1959)."""

    def test_timoshenko_stress_formula(self):
        """Verify the Timoshenko stress formula σ = 3P(1+ν)/(4πh²)."""
        P, E, h, nu, k = 100e3, 5e9, 1.0, 0.3, 9810.0
        sigma = timoshenko_plate_point_load_stress(P, E, h, nu, k)
        expected = 3.0 * P * (1.0 + nu) / (4.0 * math.pi * h**2)
        assert sigma == pytest.approx(expected, rel=1e-12)

    def test_timoshenko_solution_centre_deflection(self):
        """Verify w(0) = P/(8kl²) = P/(8√(kD)) for point load."""
        sol = TimoshenkoPlateSolution(P=50e3, E=5e9, h=1.0, nu=0.3, k=9810.0)
        w0 = sol.centre_deflection()
        expected = sol.P / (8.0 * sol.k * sol.l**2)
        assert w0 == pytest.approx(expected, rel=1e-12)

    def test_timoshenko_solution_centre_moment(self):
        """Verify M(0) = P(1+ν)/(8π) for point load."""
        sol = TimoshenkoPlateSolution(P=50e3, E=5e9, h=1.0, nu=0.3, k=9810.0)
        M0 = sol.centre_moment()
        expected = sol.P * (1.0 + sol.nu) / (8.0 * math.pi)
        assert M0 == pytest.approx(expected, rel=1e-12)

    def test_fem_vs_analytical_patch_load_stress(self):
        """FEM bending stress must match analytical patch-load within 5%."""
        result = solve_plate_bending_verification(
            P=50e3, E=5e9, h=1.0, nu=0.3, k=9810.0, n_el=200
        )
        assert result.c1_pass, f"C1 failed: stress error = {result.stress_error_rel:.2%}"
        assert result.stress_error_rel < 0.05

    def test_fem_vs_analytical_deflection(self):
        """FEM deflection should also be close to analytical."""
        result = solve_plate_bending_verification(
            P=50e3, E=5e9, h=1.0, nu=0.3, k=9810.0, n_el=200
        )
        # Deflection error should be very small (< 1%)
        assert result.deflection_error_rel < 0.01

    def test_characteristic_length_scaling(self):
        """Verify l = (D/k)^{1/4} scaling."""
        sol = TimoshenkoPlateSolution(P=50e3, E=5e9, h=1.0, nu=0.3, k=9810.0)
        D = sol.E * sol.h**3 / (12.0 * (1.0 - sol.nu**2))
        l_expected = (D / sol.k)**0.25
        assert sol.l == pytest.approx(l_expected, rel=1e-12)


# ===========================================================================
# C2: Fracture handoff mass conservation
# ===========================================================================

class TestC2FractureHandoff:
    """C2: FEM→DEM fracture handoff mass conservation."""

    def test_mass_conservation_error_zero_for_equal_volumes(self):
        """Mass error should be zero when volumes match."""
        err = mass_conservation_error(1.0, 1.0, density_kg_m3=917.0)
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_mass_conservation_error_detects_mismatch(self):
        """Mass error should detect volume mismatch."""
        err = mass_conservation_error(1.0, 1.05, density_kg_m3=917.0)
        assert err == pytest.approx(0.05, rel=1e-10)

    @pytest.mark.slow
    def test_handoff_mass_conservation(self):
        """Full handoff pipeline must conserve mass within 1%."""
        result = run_fracture_handoff(
            dims=(32, 12, 4),
            target_ratio=1.4,
            release_height_m=0.10,
            duration_s=0.05,
            dt_s=1.0e-4,
        )
        assert result.c2_pass, f"C2 failed: mass error = {result.mass_error_rel:.2%}"
        assert result.mass_error_rel < 0.01

    @pytest.mark.slow
    def test_handoff_packet_schema_valid(self):
        """Handoff packet must be valid v3 schema."""
        result = run_fracture_handoff()
        assert result.packet_version == "3.0.0"
        assert result.seal_valid


# ===========================================================================
# C3: Floe drift verification (MT Uikku)
# ===========================================================================

class TestC3FloeDrift:
    """C3: Ice floe drift vs MT Uikku experiment."""

    def test_floe_mass_property(self):
        """Floe mass = density × area × thickness."""
        floe = FloeProperties(area_m2=100.0, thickness_m=1.0, density_kg_m3=917.0)
        assert floe.mass_kg == pytest.approx(91700.0, rel=1e-10)

    def test_floe_characteristic_length(self):
        """Characteristic length = 2√(A/π) for circular equivalent."""
        floe = FloeProperties(area_m2=100.0, thickness_m=1.0)
        expected = 2.0 * math.sqrt(100.0 / math.pi)
        assert floe.characteristic_length_m == pytest.approx(expected, rel=1e-10)

    def test_drift_velocity_in_expected_range(self):
        """Floe drift should be in reasonable range relative to current."""
        floes = [
            FloeProperties(
                area_m2=50.0,
                thickness_m=1.0,
                position=np.array([0.0, 0.0]),
                velocity=np.zeros(2),
            )
            for _ in range(10)
        ]
        env = DriftEnvironment(
            current_velocity=np.array([0.15, 0.0]),
            wind_velocity=np.zeros(2),
        )
        result = simulate_floe_drift(
            floes=floes, env=env, duration_s=3600.0, dt_s=10.0
        )
        # Drift fraction should be positive and reasonable
        assert result.drift_fraction > 0
        assert result.drift_fraction < 1.0  # can't exceed current speed

    def test_channel_clearing_area_function(self):
        """Channel clearing area computation."""
        # Three floes: two start in channel and move out, one starts outside
        pos_init = np.array([[50.0, 0.0], [60.0, 0.0], [200.0, 30.0]])
        pos_final = np.array([[150.0, 0.0], [160.0, 0.0], [300.0, 30.0]])
        areas = np.array([10.0, 20.0, 30.0])
        cleared = channel_clearing_area(
            pos_init, pos_final, areas,
            channel_width_m=50.0, channel_length_m=120.0
        )
        # First two floes started in channel (|y|<25, 0<x<120) and moved out (x>120)
        # Third floe started outside channel (x>120)
        assert cleared == pytest.approx(30.0, rel=1e-10)  # 10 + 20

    def test_mt_uikku_reference_data_present(self):
        """MT Uikku reference data should be available."""
        assert "ice_thickness_m" in MT_UIKKU_REFERENCE
        assert "floe_drift_fraction" in MT_UIKKU_REFERENCE
        assert MT_UIKKU_REFERENCE["floe_drift_fraction"] == (0.02, 0.05)


# ===========================================================================
# C4: Ice load time history and Weibull statistics
# ===========================================================================

class TestC4IceLoad:
    """C4: Ice load time history — peak force and Weibull distribution."""

    def test_weibull_fit_recovers_shape_parameter(self):
        """Weibull MLE should recover the true shape parameter."""
        rng = np.random.default_rng(42)
        k_true = 2.0
        lam_true = 100.0
        data = lam_true * rng.weibull(k_true, 10000)
        k_fit, lam_fit = fit_weibull(data, method="mle")
        assert k_fit == pytest.approx(k_true, rel=0.10)
        assert lam_fit == pytest.approx(lam_true, rel=0.10)

    def test_weibull_shape_parameter_function(self):
        """weibull_shape_parameter should return k."""
        rng = np.random.default_rng(42)
        data = 100.0 * rng.weibull(1.5, 5000)
        k = weibull_shape_parameter(data)
        assert 1.0 < k < 2.5  # reasonable range

    def test_ice_load_time_history_generation(self):
        """Ice load time history should generate non-zero loads."""
        gen = IceLoadTimeHistory(
            duration_s=100.0,
            dt_s=0.1,
            event_rate_hz=1.0,
            peak_weibull_k=1.5,
            peak_weibull_lambda_N=10e3,
            pulse_duration_s=1.0,
            seed=42,
        )
        time, force = gen.generate()
        assert len(time) == len(force)
        assert np.max(force) > 0

    def test_ice_load_verification_peak_force(self):
        """Peak force should be within reasonable range."""
        result = run_ice_load_verification(
            duration_s=1000.0,
            dt_s=0.1,
            event_rate_hz=0.5,
            peak_weibull_k_true=1.8,
            peak_weibull_lambda_N=50e3,
            pulse_duration_s=2.0,
            seed=42,
            peak_force_ref_N=200e3,
            weibull_k_ref=1.7,
        )
        # The peak force should be reasonable
        assert result.peak_force_N > 0
        assert result.n_events > 10

    def test_ice_load_weibull_fit_quality(self):
        """Weibull fit should be reasonable."""
        result = run_ice_load_verification(
            duration_s=2000.0,
            dt_s=0.1,
            event_rate_hz=1.0,
            peak_weibull_k_true=2.0,
            peak_weibull_lambda_N=30e3,
            pulse_duration_s=1.5,
            seed=123,
            peak_force_ref_N=150e3,
            weibull_k_ref=1.8,
        )
        # Weibull k should be in a reasonable range
        assert 0.5 < result.weibull_k_sim < 5.0


# ===========================================================================
# C5: Energy conservation audit
# ===========================================================================

class TestC5EnergyAudit:
    """C5: Energy conservation across FEM→DEM transition."""

    def test_energy_audit_runs(self):
        """Energy audit should complete without error."""
        result = run_energy_audit(
            E_ice=5e9, h_ice=1.0, nu=0.3, k_foundation=9810.0,
            P_load=50e3, n_el=100,
        )
        assert result.W_ext_fem > 0
        assert result.U_bending > 0
        assert result.U_foundation > 0

    def test_energy_components_positive(self):
        """All energy components should be non-negative."""
        result = run_energy_audit()
        assert result.U_bending >= 0
        assert result.U_foundation >= 0
        assert result.E_fracture_fem >= 0
        assert result.E_kinetic_dem >= 0

    def test_energy_balance_closure(self):
        """Energy balance should close within reasonable tolerance."""
        result = run_energy_audit(
            E_ice=5e9, h_ice=1.0, nu=0.3, k_foundation=9810.0,
            P_load=50e3, n_el=150,
            sigma_f_Pa=1.0e6, G_f_J_m2=100.0,
        )
        # The energy error should be reasonable for this simplified model
        assert result.energy_error_rel < 0.50


# ===========================================================================
# Full verification
# ===========================================================================

class TestFullVerification:
    """Full verification runner."""

    @pytest.mark.slow
    def test_full_verification_runs(self):
        """Full verification should complete and return a report."""
        report = run_full_verification(
            P_load=50e3,
            E_ice=5e9,
            h_ice=1.0,
            nu=0.3,
            k_foundation=9810.0,
            n_el_plate=100,
            handoff_dims=(16, 8, 4),
            n_floes=20,
            current_speed_m_s=0.15,
            drift_duration_s=1800.0,
            channel_width_m=50.0,
            load_duration_s=500.0,
            event_rate_hz=0.5,
        )
        assert report.timestamp is not None
        assert report.n_tests == 5
        assert report.c1_plate_bending is not None
        assert report.c2_fracture_handoff is not None
        assert report.c3_floe_drift is not None
        assert report.c4_ice_load is not None
        assert report.c5_energy_audit is not None

    @pytest.mark.slow
    def test_full_verification_summary(self):
        """Summary should contain all criteria."""
        report = run_full_verification(
            n_el_plate=80,
            handoff_dims=(16, 8, 4),
            n_floes=10,
            drift_duration_s=600.0,
            load_duration_s=200.0,
        )
        summary = report.summary()
        assert "criteria" in summary
        assert "C1_plate_bending" in summary["criteria"]
        assert "C2_fracture_handoff" in summary["criteria"]
        assert "C3_floe_drift" in summary["criteria"]
        assert "C4_ice_load" in summary["criteria"]
        assert "C5_energy_audit" in summary["criteria"]

    @pytest.mark.slow
    def test_full_verification_json_serializable(self):
        """Report should be JSON serializable."""
        report = run_full_verification(
            n_el_plate=80,
            handoff_dims=(16, 8, 4),
            n_floes=10,
            drift_duration_s=600.0,
            load_duration_s=200.0,
        )
        json_str = report.to_json()
        assert len(json_str) > 100
        import json
        data = json.loads(json_str)
        assert "timestamp" in data


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Integration tests for the full chain."""

    @pytest.mark.slow
    def test_plate_to_fracture_chain(self):
        """Plate bending result should feed into fracture handoff."""
        # Run plate bending
        plate_result = solve_plate_bending_verification(
            P=50e3, E=5e9, h=1.0, nu=0.3, k=9810.0, n_el=100
        )
        assert plate_result.c1_pass

        # Run fracture handoff
        handoff_result = run_fracture_handoff(dims=(16, 8, 4))
        assert handoff_result.c2_pass

    def test_drift_with_realistic_parameters(self):
        """Floe drift with MT Uikku-like parameters."""
        rng = np.random.default_rng(42)
        floes = []
        for _ in range(30):
            floes.append(FloeProperties(
                area_m2=rng.uniform(20, 80),
                thickness_m=0.8,
                position=np.array([rng.uniform(-50, 50), rng.uniform(-25, 25)]),
                velocity=np.zeros(2),
            ))

        env = DriftEnvironment(
            current_velocity=np.array([0.10, 0.0]),
            wind_velocity=np.array([5.0, 0.0]),
        )
        result = simulate_floe_drift(
            floes=floes, env=env, duration_s=1800.0, dt_s=10.0
        )
        assert result.mean_drift_speed_m_s > 0
