"""Tests for the moving ice load / FNV node (p360w1-n05).

Coverage map
------------
* structural side: geometry consistency, analytic frequencies vs Hermite beam
  FE, frequency ratios, modal mass/stiffness consistency;
* static reference (A1) and its modal convergence;
* classic moving-load reference (A2): critical-speed relation, numeric vs
  closed form, resonant limit, quasi-static limit;
* broken-ice excitation: floe-spacing / repetition-frequency relation,
  brittle-ductile sawtooth properties, dominant-frequency estimator;
* FNV: definition, tiers, narrowband amplitude recovery, energy balance (A4),
  and the qualitative resonance ordering of the speed scan (A3);
* independent verification: time-domain vs frequency-domain steady-state
  amplitude of a stationary pulse train.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_moving_load import (
    FREE_FREE,
    SIMPLY_SUPPORTED,
    IceFloeField,
    GirderBeam,
    analytic_point_load_response,
    band_amplitude,
    beam_fem_frequencies,
    build_contact_history,
    build_reference_case,
    classic_critical_speed,
    classify_fnv,
    dominant_frequency_hz,
    fnv_value,
    fourier_amplitude,
    geometry_audit,
    modal_newmark,
    run_contact_history,
    run_moving_point_load,
    sawtooth_fraction,
    static_midspan_deflection_analytic,
    static_midspan_deflection_modal,
    steady_state_fundamental_amplitude,
)

G = 9.80665


@pytest.fixture(scope="module")
def case():
    return build_reference_case()


@pytest.fixture(scope="module")
def beam(case):
    return case.beam


# ---------------------------------------------------------------------------
# Structural side
# ---------------------------------------------------------------------------


def test_hull_geometry_is_consistent(case):
    audit = geometry_audit(case)
    assert audit["block_coefficient_relative_error"] < 0.02
    assert audit["consistent"] is True
    assert audit["euler_bernoulli_applicable"] is True


def test_free_free_frequencies_match_beam_fem(beam):
    analytic = beam.free_free_frequencies_hz(3)
    fem = beam_fem_frequencies(beam, n_elem=60, support=FREE_FREE, n_modes=3)
    assert np.max(np.abs(fem - analytic) / analytic) < 0.01


def test_free_free_frequency_ratios_match_theory(beam):
    f = beam.free_free_frequencies_hz(3)
    # parenthesised: (beta_2/beta_1)^2 and (beta_3/beta_1)^2
    assert f[1] / f[0] == pytest.approx((7.8532046 / 4.7300407) ** 2, rel=1e-6)
    assert f[2] / f[0] == pytest.approx((10.9956078 / 4.7300407) ** 2, rel=1e-6)


def test_simply_supported_frequency_formula(beam):
    f = beam.simply_supported_frequencies_hz(2)
    expected = (math.pi / beam.length_m) ** 2 * beam.bending_wave_speed_m_s / (2.0 * math.pi)
    assert f[0] == pytest.approx(expected, rel=1e-12)
    assert f[1] / f[0] == pytest.approx(4.0, rel=1e-12)


def test_modal_stiffness_and_mass_are_consistent(beam):
    basis = beam.modal_basis(support=FREE_FREE, n_modes=5)
    k_n = basis.modal_stiffness_nm
    m_n = basis.modal_mass_kg
    omega = basis.omega_rad_s
    assert np.allclose(k_n, m_n * omega**2, rtol=1e-10)
    # simply supported reference: m_n = rho A L / 2, phi = sin(n pi x / L)
    ss = beam.modal_basis(support=SIMPLY_SUPPORTED, n_modes=3)
    expected = beam.effective_mass_per_length_kg_m * beam.length_m / 2.0
    assert np.allclose(ss.modal_mass_kg, expected, rtol=1e-8)
    # curvature of the simply supported shape is -(n pi / L)^2 phi
    x = np.array([0.25 * beam.length_m])
    phi = ss.shapes(x)[:, 0]
    phi_pp = ss.curvatures(x)[:, 0]
    assert np.allclose(phi_pp, -ss.mode_numbers**2 * (math.pi / beam.length_m) ** 2 * phi, rtol=1e-12)


def test_added_mass_factor_shifts_frequencies(beam):
    heavier = GirderBeam(
        length_m=beam.length_m,
        bending_stiffness_nm2=beam.bending_stiffness_nm2,
        mass_per_length_kg_m=beam.mass_per_length_kg_m,
        added_mass_factor=4.0,
    )
    f0 = beam.free_free_frequencies_hz(1)[0]
    f1 = heavier.free_free_frequencies_hz(1)[0]
    assert f1 == pytest.approx(0.5 * f0, rel=1e-12)


# ---------------------------------------------------------------------------
# A1: static reference
# ---------------------------------------------------------------------------


def test_a1_static_midspan_matches_closed_form(case):
    load = case.static_midspan_load_n()
    analytic = static_midspan_deflection_analytic(load, case.beam)
    modal = static_midspan_deflection_modal(load, case.beam, n_modes=10, support=SIMPLY_SUPPORTED)
    assert abs(modal - analytic) / analytic < 0.03


def test_a1_converges_with_mode_count(case):
    load = case.static_midspan_load_n()
    analytic = static_midspan_deflection_analytic(load, case.beam)
    coarse = static_midspan_deflection_modal(load, case.beam, n_modes=5, support=SIMPLY_SUPPORTED)
    fine = static_midspan_deflection_modal(load, case.beam, n_modes=200, support=SIMPLY_SUPPORTED)
    assert abs(fine - analytic) / analytic < abs(coarse - analytic) / analytic
    assert abs(fine - analytic) / analytic < 1e-4


# ---------------------------------------------------------------------------
# A2: classic moving load
# ---------------------------------------------------------------------------


def test_classic_critical_speed_relation(beam):
    v_cr = classic_critical_speed(beam, 1, SIMPLY_SUPPORTED)
    omega_1 = 2.0 * math.pi * beam.simply_supported_frequencies_hz(1)[0]
    assert v_cr == pytest.approx(omega_1 * beam.length_m / math.pi, rel=1e-12)


def test_a2_numeric_matches_closed_form_at_critical_speed(case):
    load = case.static_midspan_load_n()
    v_cr = classic_critical_speed(case.beam, 1, SIMPLY_SUPPORTED)
    resp = run_moving_point_load(
        case.beam, load, v_cr, n_modes=8, zeta=0.0, support=SIMPLY_SUPPORTED,
        monitor_x_m=case.monitor_x, samples_per_transit=3000,
    )
    ref = analytic_point_load_response(
        case.beam, load, v_cr, resp.time_s, n_modes=300, monitor_x_m=case.monitor_x
    )["displacement_m"]
    peak_num = resp.peak_displacement_m
    peak_ref = float(np.max(np.abs(ref)))
    assert abs(peak_num - peak_ref) / peak_ref < 0.10
    # the dynamic response must exceed the static one (amplification > 1)
    static_u = static_midspan_deflection_analytic(load, case.beam)
    assert peak_num > static_u


def test_analytic_resonant_limit_matches_near_resonant_solution(beam):
    """L'Hopital branch at Omega = omega vs a tiny detuning solved numerically."""
    basis = beam.modal_basis(support=SIMPLY_SUPPORTED, n_modes=1)
    omega = basis.omega_rad_s[0]
    force = 1.0e6
    v_exact = classic_critical_speed(beam, 1, SIMPLY_SUPPORTED)
    # the closed form is only valid while the force is on the span
    t = np.linspace(0.0, beam.length_m / v_exact, 4000)
    # analytic at exact first-mode resonance (analytic_point_load_response picks
    # the resonant-limit branch because Omega_1 = omega_1 at the critical speed)
    exact = analytic_point_load_response(beam, force, v_exact, t, n_modes=1)["displacement_m"]
    # numeric integration of the same single mode at the same speed
    basis1 = beam.modal_basis(support=SIMPLY_SUPPORTED, n_modes=1)
    resp = modal_newmark(
        basis1, t, np.full(t.size, force), v_exact * t, monitor_x_m=0.5 * beam.length_m, zeta=0.0
    )
    scale = float(np.max(np.abs(exact)))
    assert np.max(np.abs(resp.displacement_m - exact)) / scale < 0.01


def test_quasi_static_limit_of_moving_load(case):
    load = case.static_midspan_load_n()
    v_cr = classic_critical_speed(case.beam, 1, SIMPLY_SUPPORTED)
    resp = run_moving_point_load(
        case.beam, load, 2.0e-3 * v_cr, n_modes=8, zeta=0.02, support=SIMPLY_SUPPORTED,
        monitor_x_m=case.monitor_x, samples_per_transit=4000,
    )
    static_u = static_midspan_deflection_analytic(load, case.beam)
    assert resp.peak_displacement_m / static_u == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Broken-ice excitation
# ---------------------------------------------------------------------------


def test_floe_field_repetition_frequency_relation():
    field = IceFloeField(mean_floe_length_m=4.0)
    assert field.repetition_frequency_hz(8.0) == pytest.approx(2.0)
    assert field.critical_speed_m_s(1.5) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        field.repetition_frequency_hz(0.0)


def test_contact_train_spacing_reproduces_ice_frequency():
    field = IceFloeField(mean_floe_length_m=4.0, floe_length_cv=0.0, strength_cv=0.0)
    speed = 5.0
    t0, lengths, peaks = field.contact_train(speed, 20.0)
    assert np.allclose(lengths, 4.0)
    periods = np.diff(t0)
    assert np.allclose(periods, 4.0 / speed)
    assert np.allclose(peaks, field.peak_force_n)
    assert 1.0 / np.mean(periods) == pytest.approx(field.repetition_frequency_hz(speed), rel=1e-12)


def test_sawtooth_shape_bounds_and_mean():
    u = np.linspace(0.0, 1.0, 1001)
    residual = 0.4
    s = sawtooth_fraction(u, cycles=1, residual=residual)
    assert s.min() >= residual
    assert s.max() <= 1.0
    assert np.mean(s) == pytest.approx((1.0 + residual) / 2.0, abs=5e-3)


def test_contact_history_is_one_sided_and_moves_aft():
    beam = GirderBeam(length_m=150.0, bending_stiffness_nm2=7.0e12, mass_per_length_kg_m=1.2e5)
    field = IceFloeField(mean_floe_length_m=20.0, floe_length_cv=0.0, sawtooth_cycles=1)
    hist = build_contact_history(field, 5.0, 20.0, 1.0e-3, beam.length_m, sweep=True)
    assert np.all(hist.force_n >= 0.0)
    active = hist.force_n > 0.0
    # contact points start at the stem and travel aft
    assert np.max(hist.position_m[active]) <= beam.length_m + 1e-9
    assert np.min(hist.position_m[active]) >= 0.0
    starts = hist.position_m[active][:50]
    assert starts[0] > starts[-1]
    pinned = build_contact_history(field, 5.0, 20.0, 1.0e-3, beam.length_m, sweep=False)
    assert np.allclose(pinned.position_m[pinned.force_n > 0.0], beam.length_m)


def test_dominant_frequency_recovers_sine_and_band_amplitude():
    t = np.arange(0.0, 40.0, 1.0e-3)
    signal = 3.0 * np.sin(2.0 * math.pi * 1.25 * t + 0.4)
    f = dominant_frequency_hz(t, signal)
    assert f == pytest.approx(1.25, rel=2e-3)
    a = band_amplitude(t, signal, f)
    assert a == pytest.approx(3.0, rel=1e-2)


def test_fourier_amplitude_of_pure_sine():
    t = np.arange(0.0, 10.0, 1.0e-3)
    f0 = 2.0
    y = 5.0 * np.cos(2.0 * math.pi * f0 * t)
    assert abs(fourier_amplitude(t, y, f0)) == pytest.approx(5.0, rel=1e-6)
    assert abs(fourier_amplitude(t, y, 1.3 * f0)) < 0.5


# ---------------------------------------------------------------------------
# FNV
# ---------------------------------------------------------------------------


def test_fnv_value_and_tiers():
    assert fnv_value(0.0, 1.0) == 0.0
    assert fnv_value(G * 0.5, 1.0) == pytest.approx(0.5)
    assert classify_fnv(0.49) == "habitable"
    assert classify_fnv(0.5) == "tolerable"
    assert classify_fnv(0.99) == "tolerable"
    assert classify_fnv(1.0) == "unacceptable"
    with pytest.raises(ValueError):
        fnv_value(1.0, 0.0)


# ---------------------------------------------------------------------------
# Energy ledger (A4) and steady-state cross-check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ice_run(case):
    field = case.field
    speed = field.critical_speed_m_s(case.first_natural_frequency_hz)
    hist = build_contact_history(field, speed, 16.0, 1.0e-3, case.beam.length_m, sweep=True)
    resp = run_contact_history(
        case.beam, hist, n_modes=case.n_modes, zeta=case.zeta, support=FREE_FREE,
        monitor_x_m=case.monitor_x,
    )
    return speed, hist, resp


def test_a4_energy_balance_residual(ice_run):
    _, _, resp = ice_run
    assert resp.energy_residual_relative < 0.05
    # the ledger must be physically populated, not identically zero
    assert resp.work_input_j[-1] > 0.0
    assert np.max(resp.strain_energy_j) > 0.0
    assert np.max(resp.kinetic_energy_j) > 0.0
    assert resp.damping_dissipated_j[-1] > 0.0


def test_a4_damping_monotone(ice_run):
    _, _, resp = ice_run
    assert np.all(np.diff(resp.damping_dissipated_j) >= -1e-6 * np.max(resp.damping_dissipated_j))


def test_steady_state_frequency_domain_matches_time_domain(case):
    """Stationary pulse train: the FFT harmonic must match the FRF prediction."""
    field = case.field
    speed = case.critical_speed_m_s
    zeta = 0.02  # short build-up (1/(zeta omega_1) = 6.6 s) so the steady state is reached
    hist = build_contact_history(field, speed, 40.0, 1.0e-3, case.beam.length_m, sweep=False)
    resp = run_contact_history(
        case.beam, hist, n_modes=case.n_modes, zeta=zeta, support=FREE_FREE,
        monitor_x_m=case.monitor_x,
    )
    mask = resp.time_s >= 25.0
    t, a = resp.time_s[mask], resp.acceleration_m_s2[mask]
    f_dom = dominant_frequency_hz(t, a)
    a_time = band_amplitude(t, a, f_dom)
    basis = case.beam.modal_basis(support=FREE_FREE, n_modes=case.n_modes)
    prediction = steady_state_fundamental_amplitude(
        basis,
        t,
        hist.force_n[mask],
        load_x_m=case.beam.length_m,
        monitor_x_m=case.monitor_x,
        frequency_hz=f_dom,
        zeta=zeta,
    )
    a_freq = prediction["acceleration_amplitude_m_s2"]
    assert a_freq > 0.0
    assert abs(a_time - a_freq) / a_freq < 0.15


def test_a3_resonance_ordering_of_speed_scan(case):
    """FNV at V_cr must exceed FNV well above V_cr (resonance, not monotone drift)."""
    from tensorphysics.ice_moving_load import fnv_speed_scan

    results, _ = fnv_speed_scan(
        case, ratios=(1.0, 1.2), durations=24.0, dt_s=1.0e-3, keep_responses=False
    )
    by_ratio = {r.speed_ratio: r for r in results}
    assert by_ratio[1.0].fnv > 3.0 * by_ratio[1.2].fnv
    assert by_ratio[1.0].f_dominant_hz == pytest.approx(case.first_natural_frequency_hz, rel=0.05)
    assert by_ratio[1.0].fnv > by_ratio[1.0].fnv_raw * 0.5  # same order, not a degeneracy