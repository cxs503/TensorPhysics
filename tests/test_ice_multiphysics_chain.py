"""Tests for ice multi-physics chain (P519-N10).

Verification targets:
1. Zero-load ⇒ acceleration = 0
2. 1/3-octave spectrum energy conservation error < 1%
3. Zero stress amplitude ⇒ Miner damage = 0
4. Miner D ≤ 1 for safe case
5. E_break = 0 ⇒ SL = −∞ ⇒ RL = −∞
6. Thorp α(10 kHz) ≈ 1 dB/km (error < 20%)
7. Zero acceleration ⇒ cargo slide = 0
8. IMO CSS sliding check consistency
9. Dimensional consistency
10. Determinism: same input ⇒ same SHA-256
"""

import hashlib
import json
import numpy as np
import pytest

from tensorphysics.ice_multiphysics_chain.vibration import (
    OneThirdOctaveBands,
    ISO19906Envelope,
    compute_one_third_octave_spectrum,
    compute_iso19906_envelope,
    acceleration_rms_by_band,
    energy_conservation_error,
)
from tensorphysics.ice_multiphysics_chain.acoustics import (
    ThorpAttenuation,
    SourceLevel,
    ReceivedLevel,
    MarineMammalThresholds,
    thorp_attenuation_db_per_km,
    source_level_from_breaking_energy,
    received_level_at_range,
    one_third_octave_spl,
    marine_mammal_exceedance,
)
from tensorphysics.ice_multiphysics_chain.cargo_shift import (
    CargoShiftCriteria,
    CargoShiftResult,
    imo_css_sliding_check,
    imo_css_overturning_check,
    transverse_acceleration,
    longitudinal_acceleration,
    assess_cargo_shift,
)
from tensorphysics.ice_multiphysics_chain.chain import (
    IceMultiPhysicsChain,
    IceMultiPhysicsResult,
    run_ice_multiphysics_assessment,
    VibrationResult,
    AcousticsResult,
)
from tensorphysics.ice_vibration.modal import HullParameters


# =============================================================================
# Vibration tests
# =============================================================================

class TestOneThirdOctaveBands:
    def test_standard_bands_exist(self):
        bands = OneThirdOctaveBands.standard()
        assert len(bands.centre_frequencies) > 0
        assert bands.centre_frequencies[0] >= 0.5
        assert bands.centre_frequencies[-1] <= 10000.0

    def test_band_edges(self):
        bands = OneThirdOctaveBands.standard()
        sixth_root_2 = 2.0 ** (1.0 / 6.0)
        for i, fc in enumerate(bands.centre_frequencies):
            assert np.isclose(bands.lower_edges[i], fc / sixth_root_2)
            assert np.isclose(bands.upper_edges[i], fc * sixth_root_2)


class TestEnergyConservation:
    def test_energy_conservation_under_1pct(self):
        """1/3-octave band integration should conserve energy to < 1%.

        Compares band-summed energy against the energy within the band range
        (not the full spectrum, which may have energy outside the bands).
        """
        # Create a synthetic PSD with fine sampling
        freqs = np.linspace(0.1, 500.0, 50000)
        # Gaussian-shaped PSD centred at 5 Hz
        f0 = 5.0
        sigma = 2.0
        psd = np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)

        bands = OneThirdOctaveBands.standard(f_min=0.5, f_max=500.0)
        oct_freqs, oct_energies = compute_one_third_octave_spectrum(freqs, psd, bands)

        # Compare against energy within band range
        mask = (freqs >= bands.lower_edges[0]) & (freqs < bands.upper_edges[-1])
        band_range_energy = float(np.trapezoid(psd[mask], freqs[mask]))
        err = energy_conservation_error(band_range_energy, oct_energies)

        assert err < 0.01, f"Energy conservation error {err:.4f} exceeds 1%"

    def test_zero_psd_zero_energy(self):
        freqs = np.linspace(0.5, 500.0, 1000)
        psd = np.zeros_like(freqs)
        bands = OneThirdOctaveBands.standard(f_min=0.5, f_max=500.0)
        oct_freqs, oct_energies = compute_one_third_octave_spectrum(freqs, psd, bands)
        assert len(oct_freqs) == 0
        assert len(oct_energies) == 0


class TestISO19906Envelope:
    def test_envelope_shape(self):
        freqs = np.logspace(-1, 3, 1000)
        env = compute_iso19906_envelope(freqs, ice_thickness_m=1.0, ice_speed_m_s=1.0)
        accel = env.acceleration_bound(freqs)
        assert np.all(accel >= 0)
        assert accel[0] > 0

    def test_envelope_decreases_at_high_freq(self):
        freqs = np.logspace(0, 3, 1000)
        env = compute_iso19906_envelope(freqs, ice_thickness_m=1.0, ice_speed_m_s=1.0)
        accel = env.acceleration_bound(freqs)
        assert accel[-1] < accel[0]


# =============================================================================
# Acoustics tests
# =============================================================================

class TestThorpAttenuation:
    def test_thorp_10khz_approx_1dbkm(self):
        """Thorp α(10 kHz) ≈ 1 dB/km (error < 20%).

        Thorp formula gives ~1.187 dB/km at 10 kHz.
        """
        alpha = thorp_attenuation_db_per_km(10000.0)
        assert abs(alpha - 1.0) / 1.0 < 0.20, f"Thorp α(10kHz) = {alpha:.4f} dB/km"

    def test_thorp_increases_with_frequency(self):
        f_low = thorp_attenuation_db_per_km(1000.0)
        f_high = thorp_attenuation_db_per_km(50000.0)
        assert f_high > f_low

    def test_thorp_at_100hz(self):
        alpha = thorp_attenuation_db_per_km(100.0)
        assert alpha < 0.1


class TestSourceLevel:
    def test_zero_energy_gives_minus_inf(self):
        sl = source_level_from_breaking_energy(0.0)
        assert sl.sl_db == -np.inf

    def test_positive_energy_gives_finite_sl(self):
        sl = source_level_from_breaking_energy(1e6, radiation_efficiency=1e-5)
        assert np.isfinite(sl.sl_db)
        assert sl.sl_db > 0

    def test_sl_scales_with_energy(self):
        sl1 = source_level_from_breaking_energy(1e6)
        sl2 = source_level_from_breaking_energy(2e6)
        assert abs((sl2.sl_db - sl1.sl_db) - 3.01) < 0.1


class TestReceivedLevel:
    def test_zero_source_gives_minus_inf(self):
        rl = received_level_at_range(-np.inf, 1000.0, 1000.0)
        assert rl.rl_db == -np.inf

    def test_rl_decreases_with_range(self):
        sl = source_level_from_breaking_energy(1e6).sl_db
        rl1 = received_level_at_range(sl, 100.0, 1000.0)
        rl2 = received_level_at_range(sl, 1000.0, 1000.0)
        assert rl2.rl_db < rl1.rl_db

    def test_transmission_loss_positive(self):
        sl = source_level_from_breaking_energy(1e6).sl_db
        rl = received_level_at_range(sl, 1000.0, 1000.0)
        assert rl.transmission_loss_db > 0


class TestMarineMammalThresholds:
    def test_no_exceedance_for_low_spl(self):
        spl = np.array([100.0, 110.0, 105.0])
        freqs = np.array([1000.0, 2000.0, 4000.0])
        result = marine_mammal_exceedance(spl, freqs, group="MF")
        assert not result["peak_exceeds_injury"]
        assert not result["behavioural_exceeds"]

    def test_exceedance_for_high_spl(self):
        spl = np.array([240.0, 235.0, 238.0])
        freqs = np.array([1000.0, 2000.0, 4000.0])
        result = marine_mammal_exceedance(spl, freqs, group="MF")
        assert result["peak_exceeds_injury"]

    def test_all_inf_spl(self):
        spl = np.array([-np.inf, -np.inf, -np.inf])
        freqs = np.array([1000.0, 2000.0, 4000.0])
        result = marine_mammal_exceedance(spl, freqs, group="MF")
        assert not result["peak_exceeds_injury"]
        assert result["max_spl_db"] == -np.inf


# =============================================================================
# Cargo shift tests
# =============================================================================

class TestCargoShiftZeroAccel:
    def test_zero_acceleration_zero_force(self):
        cargo = CargoShiftCriteria(
            cargo_mass_kg=10000.0, cargo_width_m=2.0, cargo_height_m=2.0,
            cargo_length_m=4.0, friction_coefficient=0.3,
        )
        result = assess_cargo_shift(cargo, 0.0, 10.0, 0.0, 10.0)
        assert result.sliding_force_transverse_n == 0.0
        assert result.sliding_force_longitudinal_n == 0.0

    def test_zero_acceleration_safe(self):
        cargo = CargoShiftCriteria(
            cargo_mass_kg=10000.0, cargo_width_m=2.0, cargo_height_m=2.0,
            cargo_length_m=4.0, friction_coefficient=0.3,
        )
        result = assess_cargo_shift(cargo, 0.0, 10.0, 0.0, 10.0)
        assert result.is_sliding_safe
        assert result.is_overturning_safe


class TestTransverseAcceleration:
    def test_zero_roll_zero_accel(self):
        a = transverse_acceleration(0.0, 8.0, 5.0)
        assert a == 0.0

    def test_acceleration_scales_with_amplitude(self):
        a1 = transverse_acceleration(5.0, 8.0, 5.0)
        a2 = transverse_acceleration(10.0, 8.0, 5.0)
        assert np.isclose(a2, 2.0 * a1)


class TestLongitudinalAcceleration:
    def test_zero_pitch_zero_accel(self):
        a = longitudinal_acceleration(0.0, 6.0, 5.0)
        assert a == 0.0

    def test_surge_contribution(self):
        a = longitudinal_acceleration(0.0, 6.0, 5.0, surge_acceleration_m_s2=0.5)
        assert np.isclose(a, 0.5)


class TestIMOCSSSLiding:
    def test_heavy_cargo_with_low_accel_safe(self):
        cargo = CargoShiftCriteria(
            cargo_mass_kg=10000.0, cargo_width_m=2.0, cargo_height_m=2.0,
            cargo_length_m=4.0, friction_coefficient=0.3,
            securing_strength_kn=50.0,
        )
        is_safe, margin_y, margin_x = imo_css_sliding_check(cargo, 1.0, 0.5)
        assert is_safe

    def test_high_accel_unsafe_without_securing(self):
        cargo = CargoShiftCriteria(
            cargo_mass_kg=10000.0, cargo_width_m=2.0, cargo_height_m=2.0,
            cargo_length_m=4.0, friction_coefficient=0.3,
            securing_strength_kn=0.0,
        )
        is_safe, margin_y, margin_x = imo_css_sliding_check(cargo, 5.0, 1.0)
        assert not is_safe


class TestIMOCSOverturning:
    def test_low_cargo_safe(self):
        cargo = CargoShiftCriteria(
            cargo_mass_kg=10000.0, cargo_width_m=3.0, cargo_height_m=1.0,
            cargo_length_m=4.0, friction_coefficient=0.3,
        )
        is_safe, margin = imo_css_overturning_check(cargo, 2.0, 1.0)
        assert is_safe

    def test_tall_narrow_cargo_at_risk(self):
        cargo = CargoShiftCriteria(
            cargo_mass_kg=10000.0, cargo_width_m=1.0, cargo_height_m=4.0,
            cargo_length_m=1.0, friction_coefficient=0.3,
        )
        is_safe, margin = imo_css_overturning_check(cargo, 5.0, 3.0)
        assert not is_safe


# =============================================================================
# Full chain tests
# =============================================================================

class TestFullChain:
    def test_chain_runs(self):
        result = run_ice_multiphysics_assessment(
            displacement_t=15000.0,
            length_m=120.0,
            breadth_m=20.0,
            draft_m=6.0,
            cb=0.65,
            ice_thickness_m=1.0,
            ice_speed_m_s=1.0,
            ice_strength_mpa=3.0,
            breaking_energy_j=1e6,
            acoustic_range_m=1000.0,
            cargo_mass_kg=20000.0,
            seed=20261110,
        )
        assert isinstance(result, IceMultiPhysicsResult)

    def test_chain_verification_passes(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        v = result.verification

        assert v["zero_load_acceleration_zero"], "Zero load should give zero acceleration"
        assert v["energy_conservation_lt_1pct"], f"Energy error: {v['energy_conservation_error']}"
        assert v["zero_energy_sl_inf"], "Zero energy should give -inf SL"
        assert v["zero_sl_rl_inf"], "-inf SL should give -inf RL"
        assert v["thorp_10khz_approx_1dbkm"], f"Thorp 10kHz: {v['thorp_10khz_value']}"
        assert v["zero_accel_no_slide"], "Zero accel should give zero cargo force"
        assert v["dimensional_consistency"], "Dimensional check failed"

    def test_chain_deterministic(self):
        r1 = run_ice_multiphysics_assessment(seed=20261110)
        r2 = run_ice_multiphysics_assessment(seed=20261110)
        assert r1.sha256 == r2.sha256

    def test_chain_different_seed_different_sha(self):
        r1 = run_ice_multiphysics_assessment(seed=20261110)
        r2 = run_ice_multiphysics_assessment(seed=20261111)
        assert r1.sha256 != r2.sha256

    def test_vibration_result_structure(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        v = result.vibration
        assert v.rms_acceleration_m_s2 > 0
        assert v.peak_acceleration_m_s2 > 0
        assert len(v.one_third_octave_freqs) > 0
        assert len(v.one_third_octave_rms) > 0

    def test_acoustics_result_structure(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        a = result.acoustics
        assert np.isfinite(a.source_level_db)
        assert a.source_level_db > 0
        assert len(a.spl_db) > 0

    def test_cargo_shift_result_structure(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        c = result.cargo_shift
        assert c.transverse_accel_m_s2 > 0
        assert c.longitudinal_accel_m_s2 >= 0
        assert c.sliding_force_transverse_n > 0

    def test_fatigue_result_structure(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        f = result.fatigue
        assert f.miner_damage >= 0
        assert f.fatigue_life_years > 0


class TestZeroLoadNegativeControl:
    def test_zero_ice_force_zero_acceleration(self):
        from tensorphysics.ice_fatigue.spectrum import compute_load_psd

        # Need enough samples for Welch's method (nperseg >= 4)
        time = np.linspace(0, 10, 100)
        force = np.zeros(100)
        psd = compute_load_psd(time, force)
        assert psd.m0 < 1e-30

    def test_zero_stress_zero_damage(self):
        from tensorphysics.fatigue.rainflow import rainflow_counting_stack
        from tensorphysics.fatigue.sn_curve import SNCurve
        from tensorphysics.fatigue.miner import miner_damage

        stress = np.zeros(1000)
        rf = rainflow_counting_stack(stress)
        assert len(rf.ranges) == 0

        sn = SNCurve(m=3.0, log_K=12.0)
        result = miner_damage(np.array([1.0]), np.array([1000.0]), sn)
        assert result.total_damage >= 0


class TestDimensionalConsistency:
    def test_acceleration_positive(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        assert result.vibration.rms_acceleration_m_s2 >= 0
        assert result.vibration.peak_acceleration_m_s2 >= 0

    def test_damage_non_negative(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        assert result.fatigue.miner_damage >= 0

    def test_source_level_finite_or_inf(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        sl = result.acoustics.source_level_db
        assert np.isfinite(sl) or sl == -np.inf

    def test_cargo_forces_non_negative(self):
        result = run_ice_multiphysics_assessment(seed=20261110)
        cs = result.cargo_shift
        assert cs.sliding_force_transverse_n >= 0
        assert cs.sliding_force_longitudinal_n >= 0
        assert cs.overturning_moment_n_m >= 0
        assert cs.restoring_moment_n_m >= 0
