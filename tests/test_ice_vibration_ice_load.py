"""Tests for the ice load models and the time integrators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_vibration.ice_load import (
    IceLoadSeries,
    crushing_frequency,
    half_sine_pulse,
    harmonic_load,
    poisson_pulse_train,
    quasi_periodic_pulse_train,
    self_excitation_velocity_window,
    velocity_weakening_ice_force,
    velocity_weakening_slope,
)
from tensorphysics.ice_vibration.integrators import (
    assemble_modal_matrices,
    newmark_linear,
    newmark_mdof,
    rk4_sdof,
)
from tensorphysics.ice_vibration.lockin import magnification_factor
from tensorphysics.ice_vibration.whiplash import sdof_half_sine_response


class TestIceLoadSeries:
    def test_harmonic_load_amplitude_and_peak_frequency(self):
        series = harmonic_load(1.25, 5.0e6, duration_s=40.0, dt_s=1.0 / 500.0)
        assert series.peak_force_n == pytest.approx(5.0e6, rel=2e-3)
        peak = series.dominant_frequency(0.5, 3.0)
        assert peak["frequency_hz"] == pytest.approx(1.25, rel=2e-3)
        assert series.harmonic_amplitude(1.25) == pytest.approx(5.0e6, rel=2e-3)

    def test_quasi_periodic_train_is_narrowband(self):
        train = quasi_periodic_pulse_train(
            pulse_frequency_hz=1.25, amplitude_n=6.0e6, duration_s=60.0, dt_s=1.0 / 500.0
        )
        concentration = train.spectral_concentration(0.3, 4.0)
        assert concentration["concentration"] > 0.6
        assert concentration["peak_frequency_hz"] == pytest.approx(1.25, rel=0.02)

    def test_poisson_train_is_broadband(self):
        train = poisson_pulse_train(
            mean_rate_hz=1.25, amplitude_n=6.0e6, duration_s=60.0, dt_s=1.0 / 500.0, pulse_duration_s=0.44
        )
        assert train.spectral_concentration(0.3, 4.0)["concentration"] < 0.2

    def test_series_rejects_non_monotonic_time(self):
        with pytest.raises(ValueError):
            IceLoadSeries(time_s=np.array([0.0, 0.2, 0.1]), force_n=np.zeros(3), dt_s=0.1)

    def test_windowed_subrecord(self):
        series = harmonic_load(1.0, 1.0e6, duration_s=10.0, dt_s=0.01)
        window = series.windowed(2.0, 8.0)
        assert window.time_s[0] >= 2.0 and window.time_s[-1] <= 8.0
        assert window.duration_s == pytest.approx(6.0, abs=0.02)


class TestIcePhysicsRelations:
    def test_crushing_frequency_scales_with_speed_and_flaking_length(self):
        assert crushing_frequency(1.0, 0.8) == pytest.approx(1.25)
        assert crushing_frequency(2.0, 0.8) == pytest.approx(2.5)
        with pytest.raises(ValueError):
            crushing_frequency(0.0, 0.8)

    def test_velocity_weakening_force_decreases_with_indentation_rate(self):
        f_max, v_c = 6.0e6, 0.25
        slow = velocity_weakening_ice_force(0.01, f_max, v_c)
        fast = velocity_weakening_ice_force(0.50, f_max, v_c)
        assert slow > fast
        assert velocity_weakening_ice_force(0.0, f_max, v_c) == pytest.approx(f_max)

    def test_velocity_weakening_slope_is_negative_and_maximal_near_vc(self):
        f_max, v_c = 6.0e6, 0.25
        slope_star = velocity_weakening_slope(v_c / math.sqrt(3.0), f_max, v_c)
        assert slope_star < 0.0
        assert slope_star < velocity_weakening_slope(0.02, f_max, v_c)
        assert abs(slope_star) > abs(velocity_weakening_slope(2.0, f_max, v_c))

    def test_self_excitation_window_reports_negative_effective_damping(self):
        window = self_excitation_velocity_window(1.465e6, 6.0e6, 0.25)
        assert window["self_excitation_possible"] is True
        assert window["c_eff_min_n_s_m"] < 0.0
        assert window["velocity_at_max_negative_slope_m_s"] == pytest.approx(0.25 / math.sqrt(3.0))

    def test_half_sine_pulse_shape(self):
        t = np.linspace(0.0, 1.0, 1001)
        pulse = half_sine_pulse(t, 1.0, 0.5)
        assert pulse.max() == pytest.approx(1.0, rel=1e-3)
        assert np.all(pulse[t > 0.5] == 0.0)


class TestIntegrators:
    def test_newmark_reproduces_analytic_steady_state_amplification(self):
        mass, stiffness, zeta = 1.0e6, 2.0e8, 0.02
        omega = math.sqrt(stiffness / mass)
        damping = 2.0 * zeta * mass * omega
        freq = omega / (2.0 * math.pi) * 1.03
        amplitude = 1.0e6
        duration, dt = 120.0, 1.0 / (200.0 * freq)
        result = newmark_linear(mass, damping, stiffness, lambda t: amplitude * math.sin(2 * math.pi * freq * t),
                                duration, dt)
        tail = result.time_s >= 0.75 * duration
        design = np.column_stack([
            np.ones(np.count_nonzero(tail)),
            np.sin(2 * math.pi * freq * result.time_s[tail]),
            np.cos(2 * math.pi * freq * result.time_s[tail]),
        ])
        coeffs, *_ = np.linalg.lstsq(design, result.displacement[tail], rcond=None)
        measured = math.hypot(coeffs[1], coeffs[2]) / (amplitude / stiffness)
        analytic = magnification_factor(freq / (omega / (2 * math.pi)), zeta)
        assert measured == pytest.approx(analytic, rel=0.02)

    def test_rk4_matches_closed_form_half_sine_response(self):
        mass, stiffness, zeta, amplitude, pulse = 1.0e6, 2.0e8, 0.02, 4.0e6, 0.3
        omega = math.sqrt(stiffness / mass)
        damping = 2.0 * zeta * mass * omega
        marched = rk4_sdof(
            mass, damping, stiffness,
            lambda t: amplitude * math.sin(math.pi * t / pulse) if t <= pulse else 0.0,
            duration_s=8.0, dt_s=1.0e-4, sample_stride=20,
        )
        analytic = sdof_half_sine_response(
            marched.time_s, omega, zeta, amplitude / mass, pulse
        )
        error = np.max(np.abs(marched.displacement - analytic))
        assert error / np.max(np.abs(analytic)) < 1e-4

    def test_rk4_accepts_sampled_force_array(self):
        mass, stiffness, zeta = 1.0e6, 2.0e8, 0.02
        omega = math.sqrt(stiffness / mass)
        damping = 2.0 * zeta * mass * omega
        freq = omega / (2.0 * math.pi)
        duration, dt = 40.0, 1.0 / (200.0 * freq)
        n = int(round(duration / dt)) + 1
        time = np.arange(n) * dt
        force = 1.0e6 * np.sin(2.0 * math.pi * freq * time)
        from_array = rk4_sdof(mass, damping, stiffness, force, duration, dt)
        from_callable = rk4_sdof(
            mass, damping, stiffness, lambda t: 1.0e6 * math.sin(2.0 * math.pi * freq * t), duration, dt
        )
        # the sampled path interpolates linearly between samples, which differs from the
        # analytic forcing only at the RK4 sub-steps (interpolation error ~ (pi/200)^2 / 8)
        assert np.allclose(from_array.displacement, from_callable.displacement, rtol=2e-4, atol=1e-9)

    def test_newmark_mdof_two_mode_superposition_is_linear_superposition(self):
        masses = [1.0e6, 1.1e6]
        zetas = [0.02, 0.03]
        freqs = [1.2, 3.3]
        m_matrix, c_matrix, k_matrix = assemble_modal_matrices(masses, zetas, freqs)
        duration, dt = 20.0, 1.0 / 400.0
        n = int(round(duration / dt)) + 1
        ones = np.ones(n)
        force_matrix = np.column_stack([ones * 1.0e6, ones * 0.5e6])
        both = newmark_mdof(k_matrix, m_matrix, c_matrix, force_matrix, duration, dt)
        only_first = newmark_mdof(k_matrix, m_matrix, c_matrix, np.column_stack([ones * 1.0e6, np.zeros(n)]),
                                  duration, dt)
        only_second = newmark_mdof(k_matrix, m_matrix, c_matrix, np.column_stack([np.zeros(n), ones * 0.5e6]),
                                   duration, dt)
        assert np.allclose(both.displacement, only_first.displacement + only_second.displacement, atol=1e-12)

    def test_rk4_rejects_duffing_request_on_linear_newmark(self):
        with pytest.raises(ValueError):
            newmark_linear(1.0, 0.1, 1.0, lambda t: 0.0, 1.0, 0.01, strain_stiffness=1.0)