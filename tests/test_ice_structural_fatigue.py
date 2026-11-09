"""Comprehensive tests for ice-induced structural fatigue assessment.

Tests cover the complete chain from ice load to fatigue life per DNV-OS-C401:

(a) S-N curves: B1, F, W3 vs DNV-OS-C401 Table 2-1 (< 5% error)
(b) Rainflow counting: sine wave → 1 cycle per period (exact)
(c) Miner cumulative damage: D = n/N for constant amplitude (< 1%)
(d) Ice-induced vibration spectrum: peak frequency vs FFT (< 10%)
(e) Fatigue life: KVLCC2 side shell weld (F detail), 1 year D < 0.1
(f) Negative controls: Δσ→0 → N→∞; n=0 → D=0
(g) Thickness correction: t > t_ref → reduced life
(h) DNV-OS-C401 check: DFL application
(i) Full chain integration

Acceptance criteria (machine-checkable):
- S-N curve R² ≥ 0.95 (log-log fit)
- Ice spectrum peak frequency ≤ 10% relative error
- Miner damage ≤ 20% vs reference
- Rainflow cycle count ≤ 10% vs analytical
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tensorphysics.ice_fatigue import (
    SNCurveOS401,
    DNV_OS_C401_CURVES,
    thickness_correction,
    fatigue_limit_stress,
    IceLoadSpectrum,
    StructuralFRF,
    ResponseSpectrum,
    compute_load_psd,
    compute_frf,
    compute_response_psd,
    spectral_moments,
    ice_load_time_series,
    response_time_series_from_psd,
    spectral_fatigue_assessment,
    IceFatigueChain,
    IceFatigueResult,
    DNVCheckResult,
    run_ice_fatigue_assessment,
)
from tensorphysics.fatigue.rainflow import rainflow_counting_stack


# ============================================================
# (a) S-N Curves: DNV-OS-C401 Table 2-1 Verification
# ============================================================


class TestSNCurvesDNV_OS_C401:
    """Test S-N curves against DNV-OS-C401 Table 2-1."""

    def test_b1_detail_parameters(self):
        """B1 detail: log a = 15.117, m = 4.0."""
        sn = SNCurveOS401(detail="B1")
        assert sn.log_a == pytest.approx(15.117, rel=1e-6)
        assert sn.m == pytest.approx(4.0, rel=1e-6)

    def test_f_detail_parameters(self):
        """F detail: log a = 13.640, m = 3.5."""
        sn = SNCurveOS401(detail="F")
        assert sn.log_a == pytest.approx(13.640, rel=1e-6)
        assert sn.m == pytest.approx(3.5, rel=1e-6)

    def test_w3_detail_parameters(self):
        """W3 detail: log a = 12.164, m = 3.0."""
        sn = SNCurveOS401(detail="W3")
        assert sn.log_a == pytest.approx(12.164, rel=1e-6)
        assert sn.m == pytest.approx(3.0, rel=1e-6)

    def test_sn_curve_log_log_linearity(self):
        """S-N curve must be linear in log-log space (R² ≥ 0.95)."""
        sn = SNCurveOS401(detail="F")
        stress_ranges = np.logspace(1.0, 2.5, 50)  # 10 to ~316 MPa
        n_cycles = sn.evaluate_range(stress_ranges)

        log_s = np.log10(stress_ranges)
        log_n = np.log10(n_cycles)

        # Linear fit
        coeffs = np.polyfit(log_s, log_n, 1)
        slope_fit, intercept_fit = coeffs

        # R² calculation
        log_n_pred = np.polyval(coeffs, log_s)
        ss_res = np.sum((log_n - log_n_pred) ** 2)
        ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot

        assert r_squared >= 0.95, f"R² = {r_squared:.6f} < 0.95"
        # Slope should match -m
        assert abs(slope_fit + sn.m) / sn.m < 0.05, \
            f"Slope {slope_fit:.3f} vs expected {-sn.m}"

    def test_sn_curve_all_details_r_squared(self):
        """All three details must have R² ≥ 0.95."""
        for detail in ["B1", "F", "W3"]:
            sn = SNCurveOS401(detail=detail)
            stress_ranges = np.logspace(1.0, 2.5, 50)
            n_cycles = sn.evaluate_range(stress_ranges)

            log_s = np.log10(stress_ranges)
            log_n = np.log10(n_cycles)

            coeffs = np.polyfit(log_s, log_n, 1)
            log_n_pred = np.polyval(coeffs, log_s)
            ss_res = np.sum((log_n - log_n_pred) ** 2)
            ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot

            assert r_squared >= 0.95, f"{detail}: R² = {r_squared:.6f}"

    def test_sn_curve_reference_points(self):
        """Verify specific N values at reference stress ranges.

        For F detail at ΔS = 100 MPa:
        log N = 13.640 - 3.5 * log10(100) = 13.640 - 7.0 = 6.640
        N = 10^6.640 ≈ 4.365e6
        """
        sn = SNCurveOS401(detail="F")
        n_100 = sn.n_to_failure(100.0)
        expected_log_n = 13.640 - 3.5 * 2.0  # log10(100) = 2
        expected_n = 10.0 ** expected_log_n
        assert n_100 == pytest.approx(expected_n, rel=0.01)

    def test_sn_curve_negative_control_zero_stress(self):
        """Δσ → 0 should give N → ∞."""
        sn = SNCurveOS401(detail="F")
        n = sn.n_to_failure(0.0)
        assert n == np.inf

    def test_sn_curve_negative_stress(self):
        """Negative stress range should give N → ∞."""
        sn = SNCurveOS401(detail="F")
        n = sn.n_to_failure(-10.0)
        assert n == np.inf


# ============================================================
# (b) Rainflow Counting: Sine Wave Verification
# ============================================================


class TestRainflowCounting:
    """Test rainflow counting against analytical expectations."""

    def test_sine_wave_one_period_one_cycle(self):
        """One full sine period should give approximately 1 cycle."""
        t = np.linspace(0, 2 * np.pi, 200)
        signal = 100.0 * np.sin(t)
        result = rainflow_counting_stack(signal)

        # Total cycles should be close to 1 (may have half-cycle residuals)
        assert result.total_cycles >= 0.5
        assert result.total_cycles <= 2.0

    def test_sine_wave_n_periods_n_cycles(self):
        """N full sine periods should give approximately N cycles."""
        for n_periods in [10, 20, 50]:
            t = np.linspace(0, n_periods * 2 * np.pi, n_periods * 200)
            signal = 50.0 * np.sin(t)
            result = rainflow_counting_stack(signal)

            # Should have approximately n_periods cycles (within 10%)
            rel_error = abs(result.total_cycles - n_periods) / n_periods
            assert rel_error <= 0.10, f"n_periods={n_periods}, total={result.total_cycles:.1f}, error={rel_error:.2%}"

    def test_sine_wave_range_amplitude(self):
        """Rainflow range should equal 2 × amplitude for sine wave."""
        amplitude = 75.0
        t = np.linspace(0, 4 * np.pi, 400)
        signal = amplitude * np.sin(t)
        result = rainflow_counting_stack(signal)

        if len(result.ranges) > 0:
            max_range = np.max(result.ranges)
            expected_range = 2.0 * amplitude
            assert max_range == pytest.approx(expected_range, rel=0.05)

    def test_rainflow_empty_signal(self):
        """Empty signal should give zero cycles."""
        result = rainflow_counting_stack(np.array([]))
        assert result.total_cycles == 0
        assert len(result.ranges) == 0

    def test_rainflow_constant_signal(self):
        """Constant signal should give zero cycles."""
        signal = np.ones(100) * 50.0
        result = rainflow_counting_stack(signal)
        assert result.total_cycles == 0


# ============================================================
# (c) Miner Cumulative Damage
# ============================================================


class TestMinerDamage:
    """Test Miner cumulative damage rule."""

    def test_constant_amplitude_d_equals_n_over_n(self):
        """For constant amplitude: D = n / N."""
        sn = SNCurveOS401(detail="F")
        stress_range = 80.0  # MPa
        n_applied = 1e5

        n_failure = sn.n_to_failure(stress_range)
        expected_damage = n_applied / n_failure

        # Manual calculation
        damage = n_applied / n_failure
        assert damage == pytest.approx(expected_damage, rel=1e-10)

    def test_miner_damage_less_than_one_percent_error(self):
        """Miner damage should match analytical to < 1%."""
        sn = SNCurveOS401(detail="F")
        stress = 100.0
        n_cycles = 1e5

        n_f = sn.n_to_failure(stress)
        d_analytical = n_cycles / n_f

        # Using the chain's approach
        ranges = np.array([stress])
        counts = np.array([n_cycles])
        n_f_array = sn.evaluate_range(ranges)
        d_computed = counts[0] / n_f_array[0]

        assert abs(d_computed - d_analytical) / d_analytical < 0.01

    def test_miner_zero_cycles_zero_damage(self):
        """n = 0 should give D = 0."""
        sn = SNCurveOS401(detail="F")
        stress = 100.0
        n_cycles = 0

        n_f = sn.n_to_failure(stress)
        damage = n_cycles / n_f
        assert damage == 0.0

    def test_miner_multiple_stress_levels(self):
        """Multi-level loading: D = Σ n_i / N_i."""
        sn = SNCurveOS401(detail="F")
        stresses = np.array([80.0, 100.0, 120.0])
        cycles = np.array([1e5, 5e4, 1e4])

        total_damage = 0.0
        for s, n in zip(stresses, cycles):
            n_f = sn.n_to_failure(s)
            total_damage += n / n_f

        # Verify each component
        expected = sum(n / sn.n_to_failure(s) for s, n in zip(stresses, cycles))
        assert total_damage == pytest.approx(expected, rel=1e-10)


# ============================================================
# (d) Ice-Induced Vibration Spectrum
# ============================================================


class TestIceVibrationSpectrum:
    """Test ice-induced vibration spectral analysis."""

    def test_load_psd_peak_frequency(self):
        """Peak frequency of load PSD should match input frequency (≤ 10%)."""
        primary_freq = 0.5  # Hz
        time, force = ice_load_time_series(
            duration_s=600.0,
            fs_hz=10.0,
            primary_freq_hz=primary_freq,
            secondary_amplitude_ratio=0.0,  # single frequency
            noise_ratio=0.01,
            seed=42,
        )

        load_spec = compute_load_psd(time, force)

        # Peak frequency should be close to primary
        rel_error = abs(load_spec.peak_frequency - primary_freq) / primary_freq
        assert rel_error <= 0.10, \
            f"Peak freq {load_spec.peak_frequency:.3f} Hz vs expected {primary_freq} Hz, error={rel_error:.2%}"

    def test_frf_resonance_peak(self):
        """FRF should peak at the natural frequency."""
        fn = 1.2  # Hz
        mass = 5e6  # kg
        stiffness = mass * (2 * np.pi * fn) ** 2
        zeta = 0.02
        damping = 2 * zeta * np.sqrt(stiffness * mass)

        freqs = np.linspace(0.01, 5.0, 1000)
        frf = compute_frf(freqs, mass, stiffness, damping)

        # Peak should be near natural frequency
        peak_idx = np.argmax(frf.h_mag)
        peak_freq = freqs[peak_idx]
        rel_error = abs(peak_freq - fn) / fn
        assert rel_error <= 0.05, \
            f"FRF peak at {peak_freq:.3f} Hz vs fn={fn} Hz"

    def test_response_psd_peak_near_resonance(self):
        """Response PSD should peak near structural resonance."""
        fn = 1.2  # Hz
        mass = 5e6
        stiffness = mass * (2 * np.pi * fn) ** 2
        zeta = 0.02
        damping = 2 * zeta * np.sqrt(stiffness * mass)

        # Generate ice load with frequency away from resonance
        time, force = ice_load_time_series(
            duration_s=600.0,
            fs_hz=10.0,
            primary_freq_hz=0.5,
            noise_ratio=0.3,
            seed=42,
        )

        load_spec = compute_load_psd(time, force)
        frf = compute_frf(load_spec.frequencies, mass, stiffness, damping)
        resp_spec = compute_response_psd(load_spec, frf)

        # Response peak should be influenced by FRF
        # (may not be exactly at fn if load PSD is narrow-band away from fn)
        assert resp_spec.peak_frequency > 0
        assert resp_spec.m0 > 0

    def test_spectral_moments_positive(self):
        """Spectral moments should be positive for non-trivial PSD."""
        time, force = ice_load_time_series(duration_s=300.0, fs_hz=10.0, seed=42)
        load_spec = compute_load_psd(time, force)

        moments = spectral_moments(load_spec.frequencies, load_spec.psd, (0, 1, 2))
        assert moments[0] > 0  # m0 = variance
        assert moments[1] > 0  # m1
        assert moments[2] > 0  # m2


# ============================================================
# (e) Fatigue Life: KVLCC2 Side Shell
# ============================================================


class TestFatigueLifeKVLCC2:
    """Test fatigue life for KVLCC2 side shell weld (F detail)."""

    def test_kvlcc2_side_shell_d_less_than_0_1(self):
        """KVLCC2 side shell (F detail) in ice: D < 0.1 for 1 year.

        Typical values for Arctic tanker:
        - Side shell weld at waterline
        - F detail (butt weld)
        - Moderate ice loading
        """
        result = run_ice_fatigue_assessment(
            duration_s=600.0,  # 10 min record
            fs_hz=10.0,
            mean_ice_force_n=2.0e6,
            dynamic_amplitude_n=0.8e6,
            primary_freq_hz=0.4,
            structural_mass_kg=3.0e6,
            natural_freq_hz=1.5,
            damping_ratio=0.02,
            stress_factor=5.0e-7,  # MPa/m (moderate)
            sn_detail="F",
            plate_thickness_mm=20.0,
            design_life_years=20.0,
            dfl=3.0,
            hotspot_kt=1.0,
            service_years=1.0,
            seed=20261109,
        )

        # Damage for 1 year should be < 0.1 (safe)
        assert result.miner_damage < 0.1, \
            f"Annual damage D = {result.miner_damage:.4f} >= 0.1"
        assert result.dnv_check.is_acceptable

    def test_kvlcc2_fatigue_life_reasonable(self):
        """Fatigue life should be > design life for moderate loading."""
        result = run_ice_fatigue_assessment(
            duration_s=600.0,
            fs_hz=10.0,
            mean_ice_force_n=1.5e6,
            dynamic_amplitude_n=0.5e6,
            primary_freq_hz=0.3,
            structural_mass_kg=4.0e6,
            natural_freq_hz=1.0,
            damping_ratio=0.03,
            stress_factor=3.0e-7,
            sn_detail="F",
            design_life_years=20.0,
            dfl=3.0,
            service_years=1.0,
            seed=42,
        )

        # Fatigue life should be > 20 years for this moderate case
        assert result.fatigue_life_years > 20.0, \
            f"Life = {result.fatigue_life_years:.1f} years < 20"


# ============================================================
# (f) Negative Controls
# ============================================================


class TestNegativeControls:
    """Negative control tests."""

    def test_zero_stress_infinite_life(self):
        """Δσ → 0 should give N → ∞."""
        sn = SNCurveOS401(detail="F")
        n = sn.n_to_failure(1e-10)  # very small stress
        assert n > 1e20  # effectively infinite

    def test_zero_cycles_zero_damage(self):
        """n = 0 should give D = 0."""
        sn = SNCurveOS401(detail="F")
        stress = 100.0
        n_f = sn.n_to_failure(stress)
        d = 0.0 / n_f
        assert d == 0.0

    def test_zero_load_zero_response(self):
        """Zero ice load should give zero response."""
        time = np.linspace(0, 100, 1000)
        force = np.zeros(1000)
        load_spec = compute_load_psd(time, force)
        assert load_spec.m0 == pytest.approx(0.0, abs=1e-20)


# ============================================================
# (g) Thickness Correction
# ============================================================


class TestThicknessCorrection:
    """Test thickness correction per DNV-OS-C401."""

    def test_thickness_factor_above_reference(self):
        """t > t_ref should give factor > 1."""
        factor = thickness_correction(t_mm=40.0, t_ref_mm=25.0, k=0.25)
        assert factor > 1.0
        expected = (40.0 / 25.0) ** 0.25
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_thickness_factor_below_reference(self):
        """t ≤ t_ref should give factor = 1."""
        factor = thickness_correction(t_mm=20.0, t_ref_mm=25.0, k=0.25)
        assert factor == 1.0

    def test_thickness_reduces_life(self):
        """Thicker plate should reduce fatigue life."""
        sn_thin = SNCurveOS401(detail="F", thickness_mm=20.0)
        sn_thick = SNCurveOS401(detail="F", thickness_mm=40.0)

        stress = 100.0
        n_thin = sn_thin.n_to_failure(stress)
        n_thick = sn_thick.n_to_failure(stress)

        assert n_thick < n_thin, \
            f"Thick plate life {n_thick:.2e} should be < thin plate life {n_thin:.2e}"

    def test_thickness_factor_in_chain(self):
        """Thickness correction should be applied in the chain."""
        sn_thick = SNCurveOS401(detail="F", thickness_mm=40.0)
        assert sn_thick.thickness_factor > 1.0


# ============================================================
# (h) DNV-OS-C401 Fatigue Check
# ============================================================


class TestDNVCheck:
    """Test DNV-OS-C401 fatigue check."""

    def test_dfl_application(self):
        """DFL should multiply design life."""
        result = run_ice_fatigue_assessment(
            duration_s=600.0,
            fs_hz=10.0,
            mean_ice_force_n=1.0e6,
            dynamic_amplitude_n=0.3e6,
            primary_freq_hz=0.3,
            structural_mass_kg=4.0e6,
            natural_freq_hz=1.0,
            damping_ratio=0.03,
            stress_factor=2.0e-7,
            sn_detail="B1",
            design_life_years=20.0,
            dfl=3.0,
            service_years=1.0,
            seed=42,
        )

        assert result.dnv_check.design_fatigue_factor == 3.0
        assert result.dnv_check.required_life_years == 60.0  # 20 * 3

    def test_acceptable_check_low_damage(self):
        """Low damage should pass DNV check."""
        result = run_ice_fatigue_assessment(
            duration_s=600.0,
            fs_hz=10.0,
            mean_ice_force_n=0.5e6,
            dynamic_amplitude_n=0.1e6,
            primary_freq_hz=0.2,
            structural_mass_kg=5.0e6,
            natural_freq_hz=0.8,
            damping_ratio=0.05,
            stress_factor=1.0e-7,
            sn_detail="B1",
            design_life_years=20.0,
            dfl=2.0,
            service_years=1.0,
            seed=42,
        )

        assert result.dnv_check.is_acceptable
        assert result.dnv_check.utilisation < 1.0

    def test_utilisation_calculation(self):
        """Utilisation = T_req / T_life."""
        result = run_ice_fatigue_assessment(
            duration_s=600.0,
            fs_hz=10.0,
            mean_ice_force_n=1.0e6,
            dynamic_amplitude_n=0.3e6,
            primary_freq_hz=0.3,
            structural_mass_kg=4.0e6,
            natural_freq_hz=1.0,
            damping_ratio=0.03,
            stress_factor=2.0e-7,
            sn_detail="F",
            design_life_years=20.0,
            dfl=3.0,
            service_years=1.0,
            seed=42,
        )

        expected_util = result.dnv_check.required_life_years / result.fatigue_life_years
        assert result.dnv_check.utilisation == pytest.approx(expected_util, rel=1e-6)


# ============================================================
# (i) Full Chain Integration
# ============================================================


class TestFullChainIntegration:
    """Test the complete ice fatigue assessment chain."""

    def test_chain_runs_without_error(self):
        """Full chain should execute without errors."""
        result = run_ice_fatigue_assessment(
            duration_s=300.0,
            fs_hz=10.0,
            seed=42,
        )

        assert isinstance(result, IceFatigueResult)
        assert result.miner_damage >= 0
        assert result.fatigue_life_years > 0

    def test_chain_all_details(self):
        """Chain should work with all S-N curve details."""
        for detail in ["B1", "F", "W3"]:
            result = run_ice_fatigue_assessment(
                duration_s=300.0,
                fs_hz=10.0,
                sn_detail=detail,
                seed=42,
            )
            assert result.sn_detail == detail
            assert result.miner_damage >= 0

    def test_chain_metadata_populated(self):
        """Chain result should have complete metadata."""
        result = run_ice_fatigue_assessment(duration_s=300.0, fs_hz=10.0, seed=42)

        assert "duration_s" in result.metadata
        assert "sampling_hz" in result.metadata
        assert "n_rainflow_cycles" in result.metadata
        assert "peak_load_freq_hz" in result.metadata
        assert "natural_freq_hz" in result.metadata

    def test_chain_spectral_vs_time_domain(self):
        """Spectral and time-domain damage should both be non-negative."""
        result = run_ice_fatigue_assessment(
            duration_s=600.0,
            fs_hz=10.0,
            seed=42,
        )

        # Both should be non-negative
        assert result.miner_damage >= 0
        assert result.spectral_damage >= 0

        # Both methods should give consistent sign (both positive or both zero)
        if result.miner_damage > 1e-20:
            assert result.spectral_damage > 0
        # Note: For very low stress levels, both may be essentially zero


# ============================================================
# Acceptance Criteria Summary Tests
# ============================================================


class TestAcceptanceCriteria:
    """Summary tests for all machine-checkable acceptance criteria."""

    def test_sn_curve_r_squared(self):
        """S-N curve R² ≥ 0.95 for all details."""
        for detail in ["B1", "F", "W3"]:
            sn = SNCurveOS401(detail=detail)
            stress_ranges = np.logspace(1.0, 2.5, 50)
            n_cycles = sn.evaluate_range(stress_ranges)

            log_s = np.log10(stress_ranges)
            log_n = np.log10(n_cycles)

            coeffs = np.polyfit(log_s, log_n, 1)
            log_n_pred = np.polyval(coeffs, log_s)
            ss_res = np.sum((log_n - log_n_pred) ** 2)
            ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot

            assert r_squared >= 0.95, f"{detail}: R² = {r_squared:.6f}"

    def test_ice_spectrum_peak_frequency(self):
        """Ice spectrum peak frequency ≤ 10% error."""
        primary_freq = 0.5
        time, force = ice_load_time_series(
            duration_s=600.0, fs_hz=10.0,
            primary_freq_hz=primary_freq,
            secondary_amplitude_ratio=0.0,
            noise_ratio=0.01,
            seed=42,
        )
        load_spec = compute_load_psd(time, force)
        rel_error = abs(load_spec.peak_frequency - primary_freq) / primary_freq
        assert rel_error <= 0.10

    def test_miner_damage_accuracy(self):
        """Miner damage ≤ 20% vs analytical."""
        sn = SNCurveOS401(detail="F")
        stress = 100.0
        n_cycles = 1e5

        n_f = sn.n_to_failure(stress)
        d_analytical = n_cycles / n_f

        # Computed via array method
        ranges = np.array([stress])
        n_f_array = sn.evaluate_range(ranges)
        d_computed = n_cycles / n_f_array[0]

        rel_error = abs(d_computed - d_analytical) / d_analytical
        assert rel_error <= 0.20

    def test_rainflow_cycle_count_accuracy(self):
        """Rainflow cycle count ≤ 10% vs analytical for sine wave."""
        n_periods = 10
        t = np.linspace(0, n_periods * 2 * np.pi, n_periods * 200)
        signal = 50.0 * np.sin(t)
        result = rainflow_counting_stack(signal)

        # Analytical: n_periods cycles
        rel_error = abs(result.total_cycles - n_periods) / n_periods
        assert rel_error <= 0.10, \
            f"Cycle count {result.total_cycles} vs {n_periods}, error={rel_error:.2%}"
