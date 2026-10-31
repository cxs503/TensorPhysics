"""Tests for the transient whiplash (modal-superposition) response."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_vibration.analysis import build_reference_case
from tensorphysics.ice_vibration.whiplash import (
    quasi_static_midship_moment,
    sdof_half_sine_response,
    sdof_half_sine_state,
    whiplash_response,
)


@pytest.fixture(scope="module")
def case():
    return build_reference_case()


class TestClosedFormHalfSine:
    def test_state_at_pulse_end_matches_the_response_array(self):
        omega, zeta, amp_over_mass, pulse = 2.0 * math.pi * 1.2, 0.02, 6.0e6 / 4.858e6, 0.3
        state = sdof_half_sine_state(omega, zeta, amp_over_mass, pulse)
        response = sdof_half_sine_response(
            np.array([pulse]), omega, zeta, amp_over_mass, pulse
        )
        assert response[0] == pytest.approx(state["displacement_m"], rel=1e-12)

    def test_impulse_limit_matches_impulse_response(self):
        # a very short pulse must reproduce x_peak ~ J / (m omega) with J = 2 F0 T_p / pi
        omega, zeta, pulse, amp_over_mass = 2.0 * math.pi * 1.2, 0.005, 1.0e-3, 1.0
        time = np.linspace(0.0, 4.0, 200001)
        response = sdof_half_sine_response(time, omega, zeta, amp_over_mass, pulse)
        impulse = 2.0 * amp_over_mass * pulse / math.pi
        # the peak of a damped impulse response is J/omega_d * exp(-zeta/sqrt(1-zeta^2)*atan(...)),
        # i.e. slightly below J/omega
        assert np.max(np.abs(response)) == pytest.approx(impulse / omega, rel=0.02)

    def test_zero_before_excitation_and_decays_afterwards(self):
        omega, zeta, pulse = 2.0 * math.pi * 1.2, 0.02, 0.3
        response = sdof_half_sine_response(np.array([0.0]), omega, zeta, 1.0, pulse)
        assert response[0] == pytest.approx(0.0)
        time = np.linspace(0.0, 60.0, 60001)
        long_response = sdof_half_sine_response(time, omega, zeta, 1.0, pulse)
        assert abs(long_response[-1]) < 1e-3 * np.max(np.abs(long_response))


class TestWhiplashResponse:
    def test_peak_midship_moment_matches_analytic_modal_superposition(self, case):
        result = whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, 0.6)
        assert result.peak_midship_moment_nm > 0.0
        assert result.moment_error_relative() < 0.08
        assert result.displacement_error_relative() < 0.08
        assert result.quarter_moment_error_relative() < 0.08

    def test_second_mode_carries_no_midship_moment_but_drives_the_bow(self, case):
        shares = whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, 0.6).modal_peak_shares()
        assert shares["per_mode_share"][1] < 1e-12
        two_mode = whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, 0.6)
        one_mode = whiplash_response(case.modal.truncated(1), case.bending_stiffness_nm2, 6.0e6, 0.6)
        # the antisymmetric mode reduces the bow displacement peak, so the two sets differ
        assert two_mode.peak_bow_displacement_m != pytest.approx(one_mode.peak_bow_displacement_m, rel=1e-6)

    def test_second_mode_share_at_the_quarter_point(self, case):
        result = whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, 0.6)
        shares = result.quarter_mode_shares()
        assert len(shares) == 2
        assert sum(shares) == pytest.approx(1.0, rel=1e-9)
        assert shares[1] > 0.3  # antisymmetric mode contributes materially off the node

    def test_dynamic_amplification_peaks_for_pulses_near_the_mode_period(self, case):
        f1 = case.first_frequency_hz
        amplification = {}
        for ratio in (0.2, 0.6, 1.0, 2.0):
            result = whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, ratio / f1)
            amplification[ratio] = result.diagnostics["whiplash_dynamic_amplification"]
        # short pulses carry little impulse, long pulses are quasi-static: the peak is intermediate
        assert amplification[0.2] < amplification[0.6]
        assert amplification[2.0] < amplification[0.6]
        assert 1.4 < amplification[0.6] < 1.9

    def test_quasi_static_limit_amplification_tends_to_one(self, case):
        f1 = case.first_frequency_hz
        long_pulse = whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, 6.0 / f1)
        assert long_pulse.diagnostics["whiplash_dynamic_amplification"] == pytest.approx(1.0, abs=0.06)

    def test_quasi_static_moment_reference_is_positive_and_linear_in_force(self, case):
        single = quasi_static_midship_moment(case.modal, case.bending_stiffness_nm2, 6.0e6)
        double = quasi_static_midship_moment(case.modal, case.bending_stiffness_nm2, 12.0e6)
        assert single > 0.0
        assert double == pytest.approx(2.0 * single, rel=1e-12)

    def test_invalid_inputs_are_rejected(self, case):
        with pytest.raises(ValueError):
            whiplash_response(case.modal, case.bending_stiffness_nm2, 0.0, 0.3)
        with pytest.raises(ValueError):
            whiplash_response(case.modal, case.bending_stiffness_nm2, 6.0e6, -0.3)