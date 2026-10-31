"""Tests for the lock-in criteria, magnification factor and Duffing harmonic balance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_vibration.lockin import (
    LOCK_IN_AMPLIFICATION_THRESHOLD,
    classify_lock_in,
    damping_limit_for_amplification,
    detuning_ratio,
    duffing_backbone_amplitude,
    duffing_harmonic_balance_amplitudes,
    harmonic_projection_amplitude,
    is_locked_in,
    lock_in_band_hz,
    magnification_factor,
    steady_state_amplitude,
)
from tensorphysics.ice_vibration.lockin import amplification_band_half_width


class TestMagnificationFactor:
    def test_quasi_static_limit_is_unity(self):
        assert magnification_factor(0.01, 0.02) == pytest.approx(1.0, rel=1e-3)

    def test_resonance_equals_one_over_two_zeta(self):
        for zeta in (0.01, 0.02, 0.05):
            assert magnification_factor(1.0, zeta) == pytest.approx(1.0 / (2.0 * zeta), rel=1e-12)

    def test_five_percent_detuning_amplification(self):
        # the number behind the acceptance criterion: D(1.05, 0.02) > 3
        assert magnification_factor(1.05, 0.02) == pytest.approx(9.0284, rel=1e-4)
        assert magnification_factor(1.05, 0.02) > LOCK_IN_AMPLIFICATION_THRESHOLD

    def test_damping_limit_for_amplification_three(self):
        limit = damping_limit_for_amplification(1.05, LOCK_IN_AMPLIFICATION_THRESHOLD)
        assert limit == pytest.approx(0.1516, rel=1e-2)
        assert magnification_factor(1.05, limit) == pytest.approx(LOCK_IN_AMPLIFICATION_THRESHOLD, rel=1e-9)
        assert magnification_factor(1.05, limit * 1.1) < LOCK_IN_AMPLIFICATION_THRESHOLD

    def test_amplification_band_half_width_inverts_the_magnification_factor(self):
        for zeta in (0.02, 0.05):
            half_width = amplification_band_half_width(zeta, 3.0)
            assert magnification_factor(1.0 + half_width, zeta) == pytest.approx(3.0, rel=1e-9)
            assert half_width > 0.05

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            magnification_factor(0.0, 0.02)
        with pytest.raises(ValueError):
            magnification_factor(1.0, 1.5)


class TestLockInBand:
    def test_band_and_detuning(self):
        low, high = lock_in_band_hz(1.2)
        assert low == pytest.approx(1.14) and high == pytest.approx(1.26)
        assert detuning_ratio(1.26, 1.2) == pytest.approx(0.05)

    def test_band_membership_is_strict(self):
        assert is_locked_in(1.24, 1.2) is True
        # exactly 5 % is the boundary and is not inside the open band
        assert is_locked_in(1.26, 1.2) is False
        assert is_locked_in(1.40, 1.2) is False

    def test_classification_states(self):
        locked = classify_lock_in(1.20, 1.2, 20.0)
        assert locked.state == "LOCKED" and locked.lock_in_confirmed
        weak = classify_lock_in(1.20, 1.2, 1.5)
        assert weak.state == "IN_BAND_WEAK" and not weak.lock_in_confirmed
        out = classify_lock_in(1.40, 1.2, 8.0)
        assert out.state == "OUT_OF_BAND_AMPLIFIED"
        free = classify_lock_in(1.40, 1.2, 1.2)
        assert free.state == "FREE" and not free.lock_in_confirmed
        assert free.to_json_dict()["in_lock_in_band"] is False


class TestDuffingHarmonicBalance:
    def test_reduces_to_linear_amplitude_when_cubic_term_vanishes(self):
        freq, mass, damping, stiffness, force = 1.21, 4.858e6, 1.465e6, 2.762e8, 6.0e6
        w = 2.0 * math.pi * freq
        expected = force / math.sqrt((stiffness - mass * w**2) ** 2 + (damping * w) ** 2)
        amplitudes = duffing_harmonic_balance_amplitudes(freq, mass, damping, stiffness, force, 0.0)
        assert len(amplitudes) == 1
        assert amplitudes[0] == pytest.approx(expected, rel=1e-9)

    def test_hardening_oscillator_shifts_amplitude_downwards_at_resonance(self):
        freq, mass, damping, stiffness, force = 1.2001, 4.858e6, 1.465e6, 2.762e8, 6.0e6
        linear = duffing_harmonic_balance_amplitudes(freq, mass, damping, stiffness, force, 0.0)[0]
        hardening = duffing_harmonic_balance_amplitudes(freq, mass, damping, stiffness, force, 3.0e9)[0]
        assert hardening < linear

    def test_backbone_frequency_increases_with_amplitude_for_hardening(self):
        low = duffing_backbone_amplitude(0.01, 4.858e6, 2.762e8, 3.0e9)
        high = duffing_backbone_amplitude(0.20, 4.858e6, 2.762e8, 3.0e9)
        assert high > low
        assert duffing_backbone_amplitude(0.0, 4.858e6, 2.762e8, 3.0e9) == pytest.approx(
            math.sqrt(2.762e8 / 4.858e6) / (2.0 * math.pi), rel=1e-12
        )

    def test_rejects_bad_force(self):
        with pytest.raises(ValueError):
            duffing_harmonic_balance_amplitudes(1.0, 1.0, 0.1, 1.0, 0.0, 0.0)


class TestTimeDomainPostProcessing:
    def test_steady_state_amplitude_discards_transient(self):
        time = np.linspace(0.0, 10.0, 10001)
        signal = np.where(time < 5.0, 100.0, np.sin(2.0 * math.pi * time))
        steady = steady_state_amplitude(time, signal, discard_fraction=0.6)
        assert steady["peak_amplitude_m"] == pytest.approx(1.0, rel=1e-3)

    def test_harmonic_projection_recovers_amplitude_and_offset(self):
        time = np.linspace(0.0, 20.0, 20001)
        freq = 1.25
        signal = 3.0 + 2.5 * np.sin(2.0 * math.pi * freq * time + 0.7)
        projection = harmonic_projection_amplitude(time, signal, freq)
        assert projection["amplitude"] == pytest.approx(2.5, rel=1e-6)
        assert projection["offset"] == pytest.approx(3.0, rel=1e-6)
        assert projection["residual_rms"] < 1e-6