"""Regression tests for the moving-load / FNV module (battle-plan criterion g).

Each test pins one acceptance criterion of the P360W1-N04 node, plus the unit
level guards (geometry validation, force law monotonicity, FNV thresholds) that
would catch a silent regression in the physics.

References used here are independent of the code under test:

* ``P L^3 / 48 EI``            -- closed-form static deflection,
* Fryba modal series           -- closed-form moving-load response,
* explicit finite differences  -- independent space/time discretisation,
* library ``newmark_mdof``     -- independent integrator implementation.

Run: ``python -m pytest tests/test_moving_load_fnv.py -q``
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.moving_load import (
    LoadPath,
    analytic_undamped_response,
    assess_fnv,
    classify_fnv,
    coupling_strength,
    critical_velocity_check,
    degeneracy_check,
    dominant_frequency,
    encounter_frequency_hz,
    fd_beam_response,
    floe_stream_paths,
    fnv_velocity_sweep,
    hull_girder_beam_from_particulars,
    ice_field_case,
    korzhavin_contact_force,
    library_integrator_cross_check,
    modal_truncation_check,
    moving_point_load,
    reference_floe_field,
    reference_hull_girder,
    sample_floe_field,
    solve_modal_response,
    static_anchor,
)

BEAM = reference_hull_girder()
LOAD = 5.0e6


# ---------------------------------------------------------------------------
# unit-level guards
# ---------------------------------------------------------------------------


def test_geometry_is_validated_before_dynamics():
    with pytest.raises(ValueError):
        hull_girder_beam_from_particulars(block_coefficient=1.4)
    with pytest.raises(ValueError):
        hull_girder_beam_from_particulars(length_m=150.0, breadth_m=5.0)
    with pytest.raises(ValueError):
        hull_girder_beam_from_particulars(midship_inertia_m4=0.0)
    beam = reference_hull_girder()
    assert beam.rho_a_kg_m == pytest.approx(1025.0 * 0.70 * 22.0 * 8.0 * 1.20, rel=1e-12)
    assert beam.ei_nm2 == pytest.approx(2.10e11 * 18.0, rel=1e-12)


def test_natural_frequencies_and_critical_velocity_relations():
    beam = BEAM
    f = beam.natural_frequencies(4)
    # Euler-Bernoulli: f_n = n^2 f_1
    for n in range(1, 5):
        assert f[n - 1] == pytest.approx(n ** 2 * f[0], rel=1e-12)
    # V_cr = omega_1 L / pi
    assert beam.critical_velocity() == pytest.approx(beam.omega(1) * beam.length_m / math.pi, rel=1e-12)
    # published-order magnitude for a 150 m hull girder: first mode of order 0.3-0.5 Hz
    assert 0.2 < f[0] < 0.6


def test_mode_shape_orthogonality():
    beam = BEAM
    x = np.linspace(0.0, beam.length_m, 20001)
    phi = beam.mode_shapes(3, x)
    mass = np.trapezoid(phi[0] ** 2, x) * beam.rho_a_kg_m
    assert mass == pytest.approx(beam.modal_mass(1), rel=1e-6)
    cross = np.trapezoid(phi[0] * phi[1], x)
    assert abs(cross) < 1e-6 * beam.length_m


def test_korzhavin_force_decreases_with_area_and_scales_with_thickness():
    f_small = korzhavin_contact_force(5.0, 1.0)
    f_large = korzhavin_contact_force(50.0, 1.0)
    assert 0.0 < f_small < f_large
    # pressure-area law: F proportional to A^(1+e)
    area_small, area_large = 5.0 * 1.0, 50.0 * 1.0
    assert f_large / f_small == pytest.approx((area_large / area_small) ** 0.9, rel=1e-3)
    assert korzhavin_contact_force(10.0, 2.0) > korzhavin_contact_force(10.0, 1.0)


def test_fnv_thresholds_and_dominant_frequency(df_tol=0.01):
    freq = 0.5
    t = np.arange(0.0, 60.0, 0.005)
    for amplitude, expected_level in ((2.0, "habitable"), (3.5, "tolerable"), (6.0, "unacceptable")):
        a = amplitude * np.sin(2.0 * math.pi * freq * t)
        result = assess_fnv(t, a)
        assert result.dominant_frequency_hz == pytest.approx(freq, rel=df_tol)
        assert result.peak_acceleration_m_s2 == pytest.approx(amplitude, rel=1e-3)
        expected_fnv = amplitude / (9.81 * freq)
        assert result.fnv == pytest.approx(expected_fnv, rel=2e-2)
        assert result.level == expected_level
    assert classify_fnv(0.49) == "habitable"
    assert classify_fnv(0.50) == "tolerable"
    assert classify_fnv(1.00) == "tolerable"
    assert classify_fnv(1.01) == "unacceptable"
    with pytest.raises(ValueError):
        classify_fnv(-1.0)


def test_dominant_frequency_separates_two_tones():
    t = np.arange(0.0, 40.0, 0.002)
    a = 0.4 * np.sin(2.0 * math.pi * 1.2 * t) + 2.0 * np.sin(2.0 * math.pi * 6.0 * t + 0.3)
    assert dominant_frequency(t, a) == pytest.approx(6.0, rel=0.01)


# ---------------------------------------------------------------------------
# (a) stationary load
# ---------------------------------------------------------------------------


def test_a_stationary_load_matches_plate_formula():
    result = static_anchor(BEAM, LOAD)
    assert result["modal_static_relative_error"] < 0.03
    assert result["transient_relative_error"] < 0.03
    assert result["criterion"]["passed"] is True
    exact = LOAD * BEAM.length_m ** 3 / (48.0 * BEAM.ei_nm2)
    assert result["exact_static_midspan_m"] == pytest.approx(exact, rel=1e-12)


# ---------------------------------------------------------------------------
# (b) critical velocity versus the closed-form series and the FD solver
# ---------------------------------------------------------------------------


def test_b_critical_velocity_matches_closed_form_series():
    result = critical_velocity_check(BEAM, LOAD, n_modes=10, n_modes_analytic=100, dt_s=2.0e-4)
    assert result["modal_vs_analytic_relative_error"] < 0.10
    assert result["criteria"][0]["passed"] is True
    # the dynamic amplification at V_cr is of order 1.5 (Fryba)
    assert 1.3 < result["daf_analytic"] < 1.8


def test_b_critical_velocity_matches_finite_difference_solver():
    result = critical_velocity_check(BEAM, LOAD, n_elements=120, dt_s=2.0e-4)
    assert result["modal_vs_fd_relative_error"] < 0.02


def test_fd_solver_converges_with_mesh_refinement():
    v = 0.7 * BEAM.critical_velocity()
    path = moving_point_load(BEAM, LOAD, v, x_entry_m=0.0)
    duration = BEAM.length_m / v
    coarse = fd_beam_response(BEAM, [path], duration, n_elements=100)
    fine = fd_beam_response(BEAM, [path], duration, n_elements=200)
    assert abs(coarse["midspan_peak_m"] - fine["midspan_peak_m"]) / abs(fine["midspan_peak_m"]) < 0.02


def test_analytic_resonance_limit_is_continuous():
    """The L'Hopital branch must join the regular expression smoothly."""
    v_cr = BEAM.critical_velocity()
    t = 0.4 * BEAM.length_m / v_cr
    q_lo, _, _ = __import__(
        "tensorphysics.moving_load.analytic", fromlist=["analytic_undamped_modal_state"]
    ).analytic_undamped_modal_state(BEAM, LOAD, v_cr * (1.0 - 1e-4), 0.0, t, 3)
    q_cr, _, _ = __import__(
        "tensorphysics.moving_load.analytic", fromlist=["analytic_undamped_modal_state"]
    ).analytic_undamped_modal_state(BEAM, LOAD, v_cr, 0.0, t, 3)
    assert np.allclose(q_lo, q_cr, rtol=2e-3, atol=1e-12)


# ---------------------------------------------------------------------------
# (c) FNV versus V/V_cr
# ---------------------------------------------------------------------------


def test_c_fnv_amplifies_towards_the_resonant_band():
    sweep = fnv_velocity_sweep(
        BEAM, LOAD, ratios=(0.5, 0.8, 1.0, 1.2), fine_ratios=(0.2, 0.5, 0.8, 1.0, 1.2),
        dt_s=2.0e-3, tail_s=4.0,
    )
    assert sweep["fnv_amplification_min_over_specified"] >= 3.0
    assert 0.4 <= sweep["daf_peak_ratio"] <= 1.0
    assert sweep["daf_peak"] >= 1.40
    assert sweep["criterion"]["passed"] is True


# ---------------------------------------------------------------------------
# (d) energy / power balance
# ---------------------------------------------------------------------------


def test_d_power_balance_closes_for_damped_and_undamped_cases():
    v = 0.8 * BEAM.critical_velocity()
    path = moving_point_load(BEAM, LOAD, v, x_entry_m=0.0)
    duration = BEAM.length_m / v + 1.0
    for zeta in (0.0, 0.015):
        response = solve_modal_response(BEAM, [path], duration, dt_s=5.0e-4, n_modes=10, damping_ratio=zeta)
        residual = response.energy_residual()
        assert residual["relative_residual"] < 0.05
    # undamped: nothing is dissipated
    undamped = solve_modal_response(BEAM, [path], duration, dt_s=5.0e-4, n_modes=10, damping_ratio=0.0)
    assert abs(undamped.dissipated_j[-1]) < 1e-6 * max(undamped.work_j[-1], 1.0)
    damped = solve_modal_response(BEAM, [path], duration, dt_s=5.0e-4, n_modes=10, damping_ratio=0.015)
    assert damped.dissipated_j[-1] > 0.0


# ---------------------------------------------------------------------------
# (e) modal truncation
# ---------------------------------------------------------------------------


def test_e_modal_truncation_error_below_five_percent():
    result = modal_truncation_check(BEAM, LOAD, n_modes_low=10, n_modes_high=40, dt_s=1.0e-3)
    assert result["relative_error"] < 0.05
    assert result["criterion"]["passed"] is True


# ---------------------------------------------------------------------------
# (f) degeneracy V -> 0
# ---------------------------------------------------------------------------


def test_f_slow_traverse_degenerates_to_static_solution():
    result = degeneracy_check(BEAM, LOAD, ratio=0.01, dt_s=4.0e-3)
    assert result["relative_error"] < 0.03
    assert result["daf"] < 1.05
    assert result["criterion"]["passed"] is True


# ---------------------------------------------------------------------------
# cross-implementation check
# ---------------------------------------------------------------------------


def test_modal_solver_agrees_with_library_newmark():
    result = library_integrator_cross_check(BEAM, LOAD, speed_ratio=0.6, dt_s=5.0e-4)
    assert result["relative_difference"] < 1e-8
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# ice field -> moving load -> FNV
# ---------------------------------------------------------------------------


def test_floe_stream_produces_ordered_moving_paths():
    field = sample_floe_field(n_floes=6, speed_m_s=5.0, seed=7)
    paths = floe_stream_paths(BEAM, field)
    assert len(paths) == 6
    times = [p.t_entry_s for p in paths]
    assert times == sorted(times)
    for path, diameter in zip(paths, field.diameters_m):
        assert path.duration_s == pytest.approx(diameter / 5.0, rel=1e-12)
        assert path.x_exit_m < path.x_entry_m
        assert path.magnitude_n > 0.0
        assert path.shape == "half_sine"
    assert encounter_frequency_hz(5.0, 16.0) == pytest.approx(0.3125, rel=1e-12)


def test_ice_case_fnv_is_finite_and_classified():
    case = ice_field_case(
        BEAM, speed_m_s=5.0, thickness_m=1.0, dt_s=4.0e-3, tail_s=3.0, n_floes=4
    )
    assert case["peak_acceleration_midspan_m_s2"] > 0.0
    assert case["fnv_midspan"] > 0.0
    assert case["fnv_level_midspan"] in {"habitable", "tolerable", "unacceptable"}
    assert case["energy_residual"]["relative_residual"] < 0.05
    assert set(case["stations"]) == {"0.25", "0.50", "0.92"}


def test_load_path_rejects_bad_input():
    with pytest.raises(ValueError):
        LoadPath(magnitude_n=-1.0, x_entry_m=0.0, x_exit_m=10.0, t_entry_s=0.0, duration_s=1.0)
    with pytest.raises(ValueError):
        LoadPath(magnitude_n=1.0, x_entry_m=0.0, x_exit_m=10.0, t_entry_s=0.0, duration_s=1.0, shape="ramp")
    with pytest.raises(ValueError):
        moving_point_load(BEAM, LOAD, speed_m_s=0.0)


def test_coupling_strength_is_negligible_at_ship_speed():
    """One-way coupling is quantitatively sufficient at 5 m/s (see report 5-7)."""
    result = coupling_strength(BEAM)
    assert result["speed_m_s"] == pytest.approx(5.0, rel=1e-12)
    assert result["feedback_ratio"] < 0.01
    assert result["one_way_coupling_adequate"] is True
    assert 1e-6 < result["crossover_indentation_rate_m_s"] < 1e-2
    assert result["ice_induced_damping_n_s_m"] > 0.0


def test_reference_field_is_reproducible():
    a = sample_floe_field(n_floes=8, seed=123)
    b = sample_floe_field(n_floes=8, seed=123)
    assert np.allclose(a.diameters_m, b.diameters_m)
    assert np.allclose(a.gaps_m, b.gaps_m)
    c = reference_floe_field()
    assert c.n_floes == 14 and c.thickness_m == 1.0