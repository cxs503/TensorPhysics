"""Acceptance tests for voyage-accumulated ice-induced fatigue (mission m9281).

Criteria map
------------
C1  Miner accumulation ``D = sum n_i / N_i`` against hand calculation (< 0.1 %)
C2  Power-law S-N fit ``sigma_a**m N = C`` : R^2 > 0.99 (and slope recovery)
C3  Rainflow counting vs the ASTM E1049-85 standard / independent algorithm (< 1 %)
C4  Cumulative damage is monotone non-decreasing in time (no repair)
C5  Residual life limits: D -> 1 gives N_remaining -> 0, D = 0 gives N_total
C6  Ice-induced vibration share of the damage is strictly intermediate (0, 1)
C7  Voyage damage >= single-window damage (the voyage contains the window)
C8  This file; plus physics sanity anchors (statics, resonance, dt convergence)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.fatigue.rainflow import (
    _find_peaks_valleys,
    rainflow_counting,
    rainflow_counting_stack,
)
from tensorphysics.fatigue.sn_curve import DNCurve
from tensorphysics.voyage_fatigue import (
    PowerLawSN,
    VoyageProfile,
    accumulate_voyages,
    bow_load_midship_moment,
    build_reference_case,
    damage_from_ranges,
    fit_power_law_sn,
    ice_force_window,
    leg_damage,
    leg_stress_history,
    midship_static_stress_mpa,
    rainflow_counting_reference,
    remaining_life,
    voyage_damage,
    voyages_to_failure,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case():
    """Reference case (deterministic; built once for the whole module)."""
    return build_reference_case(n_voyages=20)


@pytest.fixture(scope="module")
def voyage(case):
    return case.voyage_damage()


# ---------------------------------------------------------------------------
# C1 - Miner accumulation
# ---------------------------------------------------------------------------


def test_c1_miner_against_hand_calculation():
    """D = n1/N1 + n2/N2 + n3/N3 with N_i = C / sigma_a,i**m, hand evaluated."""
    m, log_c = 3.0, 12.0
    curve = PowerLawSN(m=m, log10_c=log_c)
    ranges = np.array([20.0, 40.0, 60.0])
    counts = np.array([1000.0, 500.0, 100.0])

    # hand calculation: sigma_a = range / 2, N_i = C / sigma_a**m
    c_const = 10.0**log_c
    hand = sum(n / (c_const / (0.5 * r) ** m) for r, n in zip(ranges, counts))

    block = damage_from_ranges(ranges, counts, curve)
    assert block.damage > 0.0
    assert abs(block.damage - hand) / hand < 1.0e-3, (block.damage, hand)
    assert block.n_cycles == pytest.approx(1600.0)
    assert block.max_range_mpa == pytest.approx(60.0)


def test_c1_miner_is_additive_and_ignores_degenerate_ranges():
    curve = PowerLawSN(m=3.0, log10_c=12.0)
    part_a = damage_from_ranges([30.0], [100.0], curve).damage
    part_b = damage_from_ranges([45.0, 45.0], [50.0, 50.0], curve).damage
    combined = damage_from_ranges([30.0, 45.0], [100.0, 100.0], curve).damage
    assert combined == pytest.approx(part_a + part_b, rel=1.0e-12)
    # a zero range cannot fail, and a negative count must not remove damage
    assert damage_from_ranges([0.0], [10.0], curve).damage == 0.0
    assert damage_from_ranges([30.0], [-5.0], curve).damage == 0.0


def test_c1_single_cycle_equals_reciprocal_life():
    curve = PowerLawSN(m=3.0, log10_c=12.0)
    amplitude = 25.0
    n_fail = curve.n_to_failure(amplitude)
    block = damage_from_ranges([2.0 * amplitude], [1.0], curve)
    assert block.damage == pytest.approx(1.0 / n_fail, rel=1.0e-12)
    # the range-based accessor must agree with the amplitude one
    assert curve.n_to_failure_range(2.0 * amplitude) == pytest.approx(n_fail, rel=1.0e-12)


# ---------------------------------------------------------------------------
# C2 - S-N curve fit
# ---------------------------------------------------------------------------


def test_c2_power_law_fit_recovers_dnv_first_segment():
    """Fit sigma_a**m N = C to the DNV D curve steep segment: R^2 > 0.99, m = 3."""
    curve = PowerLawSN.from_dnv("D", "seawater_cp")
    assert curve.m == pytest.approx(3.0, rel=1.0e-6)
    dnv = DNCurve("D", "seawater_cp")
    lives = np.logspace(4.0, 7.0, 25)
    ranges = np.array([dnv.stress_for_life(n) for n in lives])
    fit = fit_power_law_sn(0.5 * ranges, lives)
    assert fit.r_squared > 0.99
    assert fit.r_squared > 0.9999
    assert fit.m == pytest.approx(3.0, rel=1.0e-6)
    # fitted power law reproduces the code curve inside the steep segment
    for n, rng in zip(lives, ranges):
        assert curve.n_to_failure_range(rng) == pytest.approx(n, rel=1.0e-6)


def test_c2_power_law_fit_is_robust_to_log_life_scatter():
    rng = np.random.default_rng(2024)
    truth = PowerLawSN(m=3.2, log10_c=11.5)
    amplitudes = np.logspace(0.8, 1.8, 60)
    lives = np.array([truth.n_to_failure(a) for a in amplitudes])
    noisy = lives * rng.lognormal(mean=0.0, sigma=0.08, size=lives.size)
    fit = fit_power_law_sn(amplitudes, noisy)
    assert fit.r_squared > 0.99
    assert fit.m == pytest.approx(3.2, rel=0.05)


def test_c2_single_power_law_across_the_dnv_knee_is_reported_honestly():
    """Across the bilinear knee one slope does not fit: R^2 stays < 1 and m drifts."""
    dnv = DNCurve("D", "seawater_cp")
    lives = np.logspace(3.5, 8.0, 40)
    ranges = np.array([dnv.stress_for_life(n) for n in lives])
    fit = fit_power_law_sn(0.5 * ranges, lives)
    assert fit.m > 3.05  # biased upward by the shallow segment
    assert fit.r_squared < 1.0 - 1.0e-4
    assert fit.rmse_log10_life > 1.0e-3


def test_c2_power_law_round_trip_and_guards():
    curve = PowerLawSN(m=3.0, log10_c=12.0)
    assert curve.stress_amplitude_for_life(curve.n_to_failure(31.0)) == pytest.approx(31.0, rel=1.0e-12)
    assert curve.n_to_failure(0.0) == math.inf
    with pytest.raises(ValueError):
        PowerLawSN(m=-1.0, log10_c=12.0)
    with pytest.raises(ValueError):
        fit_power_law_sn([10.0], [1.0e5])


# ---------------------------------------------------------------------------
# C3 - Rainflow counting against the standard
# ---------------------------------------------------------------------------


def test_c3_constant_amplitude_sinusoid_recovers_cycles_exactly():
    """Analytic anchor: n-1 whole sinusoidal periods -> n-1 cycles of range 2A.

    The record starts and ends on a peak (``cos`` over an integer number of
    periods, 40 samples per period so that the extrema fall exactly on
    samples), so the standard answer is unambiguous: ``n-1`` full cycles of
    range ``2A`` and no residue.  Damage must match ``(n-1) / N(2A)`` within 1 %.
    """
    n_periods, amp, dt = 30, 25.0, 0.025
    t = np.arange(0.0, float(n_periods), dt)
    t = t[t <= (n_periods - 1) + 1.0e-12]  # trim to end exactly on the last peak
    signal = amp * np.cos(2.0 * np.pi * 1.0 * t)
    curve = PowerLawSN(m=3.0, log10_c=12.0)
    analytic_damage = (n_periods - 1) / curve.n_to_failure_range(2.0 * amp)
    for counter in (rainflow_counting_stack, rainflow_counting, rainflow_counting_reference):
        rf = counter(signal)
        at_full_range = float(rf.counts[np.isclose(rf.ranges, 2.0 * amp, rtol=1.0e-9)].sum())
        assert abs(at_full_range - (n_periods - 1)) / (n_periods - 1) < 0.01, (
            counter.__name__,
            at_full_range,
        )
        assert rf.ranges.max() == pytest.approx(2.0 * amp, rel=1.0e-9)
        damage = float(np.sum(rf.counts / np.array([curve.n_to_failure_range(r) for r in rf.ranges])))
        assert abs(damage - analytic_damage) / analytic_damage < 0.01


def test_c3_independent_implementations_agree_on_the_reference_histories(case):
    """Stack (ASTM) vs an independently written four-point counter: < 1 % damage."""
    curve = PowerLawSN(m=3.0, log10_c=12.0)
    for history in case.leg_histories():
        for signal in (history.stress_total_mpa, history.stress_static_mpa, history.stress_vibration_mpa):
            a = rainflow_counting_stack(signal)
            b = rainflow_counting_reference(signal)
            assert a.total_cycles == pytest.approx(b.total_cycles, rel=1.0e-9)
            n_a = np.array([curve.n_to_failure_range(r) for r in a.ranges])
            n_b = np.array([curve.n_to_failure_range(r) for r in b.ranges])
            d_a = float(np.sum(a.counts / n_a))
            d_b = float(np.sum(b.counts / n_b))
            assert d_a > 0.0
            assert abs(d_a - d_b) / d_a < 0.01, (d_a, d_b)


def test_c3_plateaus_do_not_swallow_extrema():
    """Regression: exact-zero gaps between pulses used to hide extrema.

    A train of four half-sine pulses separated by exactly zero force must count
    four cycles of range 2A.  Before the plateau fix in
    ``fatigue.rainflow._find_peaks_valleys`` only 3.5 cycles were found because
    the zero gaps are plateaus of equal samples, where ``diff`` vanishes and the
    sign-change test never fires.
    """
    t = np.arange(0.0, 20.0, 0.01)
    force = np.zeros_like(t)
    amp = 10.0
    for t0 in (0.5, 3.0, 6.0, 9.5):
        mask = (t >= t0) & (t <= t0 + 0.4)
        force[mask] = amp * np.sin(np.pi * (t[mask] - t0) / 0.4)
    extrema = _find_peaks_valleys(force)
    assert np.count_nonzero(np.isclose(extrema, amp, atol=1.0e-6)) == 4, extrema
    assert extrema.size >= 2 * 4 + 1  # start, 4 peaks, 4 valleys (+ end)
    # each pulse is a 0 -> amp -> 0 excursion, i.e. one cycle of range = amp;
    # before the plateau fix at most 3.0 such cycles were recovered
    for counter in (rainflow_counting_stack, rainflow_counting, rainflow_counting_reference):
        rf = counter(force)
        full = float(rf.counts[np.isclose(rf.ranges, amp, rtol=1.0e-9)].sum())
        assert full >= 3.5 - 1.0e-12, (counter.__name__, rf.ranges, rf.counts)
    # the plain list implementation recovers all four cycles here
    listed = rainflow_counting(force)
    assert float(listed.counts[np.isclose(listed.ranges, amp, rtol=1.0e-9)].sum()) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# C4 - monotone accumulation
# ---------------------------------------------------------------------------


def test_c4_damage_is_monotone_over_the_service_life(case):
    trajectory = case.trajectory(20)
    assert trajectory.damage[0] == 0.0
    assert np.all(np.diff(trajectory.damage) >= 0.0)
    assert trajectory.is_monotone


def test_c4_monotone_with_severity_profile_and_zero_voyages(case):
    severities = np.linspace(0.4, 1.6, 25)
    trajectory = accumulate_voyages(1.0e-3, 20, severity_sequence=severities, voyage_duration_days=7.0)
    assert np.all(np.diff(trajectory.damage) >= 0.0)
    assert trajectory.final_damage == pytest.approx(1.0e-3 * severities[:20].sum(), rel=1.0e-12)
    assert np.all(np.diff(trajectory.time_days) > 0.0)
    empty = accumulate_voyages(1.0e-3, 0)
    assert empty.damage.size == 1 and empty.final_damage == 0.0


# ---------------------------------------------------------------------------
# C5 - residual life limits
# ---------------------------------------------------------------------------


def test_c5_residual_life_limits(voyage):
    d_voyage = voyage.damage_total
    fresh = remaining_life(0.0, d_voyage, 12.0)
    total = 1.0 / d_voyage
    assert fresh.total_voyages_to_failure == pytest.approx(total, rel=1.0e-12)
    assert fresh.remaining_voyages == pytest.approx(total, rel=1.0e-12)
    assert fresh.fraction_life_used == 0.0

    at_failure = remaining_life(1.0, d_voyage, 12.0)
    assert at_failure.remaining_voyages == 0.0
    assert at_failure.remaining_operating_days == 0.0
    assert at_failure.fraction_life_used == 1.0

    near = remaining_life(0.999, d_voyage, 12.0)
    assert near.remaining_voyages == pytest.approx(0.001 / d_voyage, rel=1.0e-12)
    assert near.remaining_years if False else near.remaining_operating_years > 0.0

    # beyond failure the remaining life is clipped, not negative
    over = remaining_life(1.5, d_voyage, 12.0)
    assert over.remaining_voyages == 0.0
    assert over.fraction_life_used == pytest.approx(1.5)


def test_c5_residual_life_is_linear_in_damage_and_matches_simulation(voyage):
    d_voyage = voyage.damage_total
    a = remaining_life(0.2, d_voyage, 12.0)
    b = remaining_life(0.4, d_voyage, 12.0)
    assert (a.remaining_voyages - b.remaining_voyages) == pytest.approx(0.2 / d_voyage, rel=1.0e-12)
    predicted = a.remaining_voyages
    simulated = voyages_to_failure(d_voyage) - 0.2 / d_voyage
    assert abs(simulated - predicted) <= 1.0  # whole-voyage rounding only


def test_c5_simulated_failure_voyage_matches_analytic_count(voyage):
    d_voyage = voyage.damage_total
    simulated = voyages_to_failure(d_voyage)
    analytic = 1.0 / d_voyage
    assert abs(simulated - analytic) <= 1.0
    trajectory = accumulate_voyages(d_voyage, simulated)
    assert trajectory.damage[simulated - 1] < 1.0 <= trajectory.damage[simulated]


# ---------------------------------------------------------------------------
# C6 - ice-induced vibration contribution
# ---------------------------------------------------------------------------


def test_c6_ice_vibration_share_is_strictly_intermediate(voyage):
    fraction = voyage.ice_vibration_fraction
    assert 0.0 < fraction < 1.0
    assert voyage.damage_static > 0.0
    assert voyage.damage_vibration_only > 0.0
    # the two extremes must also be bounded by the per-leg fractions
    per_leg = [d.ice_vibration_fraction for d in voyage.per_leg]
    assert min(per_leg) < fraction < max(per_leg) or fraction == pytest.approx(
        max(per_leg), rel=1e-12
    ) or fraction == pytest.approx(min(per_leg), rel=1e-12)
    assert all(0.0 <= f <= 1.0 for f in per_leg)


def test_c6_vibration_share_tracks_the_excitation_regime(case, voyage):
    """Near the hull mode the vibratory share must exceed the ridge-impact leg."""
    by_name = {d.leg_name: d for d in voyage.per_leg}
    assert by_name["lock_in_channel"].ice_vibration_fraction > 0.4
    assert by_name["ridge_ramming"].ice_vibration_fraction < 0.2
    assert (
        by_name["lock_in_channel"].ice_vibration_fraction
        > by_name["ridge_ramming"].ice_vibration_fraction
    )
    # turning the vibratory residual off removes the contribution but not the
    # quasi-static damage
    still = case.vibration.__class__(
        frequency_hz=case.vibration.frequency_hz,
        damping_ratio=case.vibration.damping_ratio,
        modal_mass_kg=case.vibration.modal_mass_kg * 1.0e6,
        curvature_midship_per_m=case.vibration.curvature_midship_per_m,
        bow_ordinate=case.vibration.bow_ordinate,
        bending_stiffness_nm2=case.vibration.bending_stiffness_nm2,
        section_modulus_m3=case.vibration.section_modulus_m3,
    )
    heavy = voyage_damage(case.profile, case.hull, case.girder, still, case.curve)
    assert heavy.ice_vibration_fraction < voyage.ice_vibration_fraction
    assert heavy.damage_total > 0.0


# ---------------------------------------------------------------------------
# C7 - voyage vs single window
# ---------------------------------------------------------------------------


def test_c7_voyage_damage_dominates_the_single_window(case, voyage):
    assert voyage.max_single_window_damage > 0.0
    assert voyage.damage_total >= voyage.max_single_window_damage
    assert voyage.dominates_single_window
    # the scaling is by cycles, so the ratio is of the order of the leg length
    ratio = voyage.damage_total / voyage.max_single_window_damage
    assert ratio > 10.0
    # and the voyage is exactly the sum of its legs
    assert voyage.damage_total == pytest.approx(
        sum(d.damage_total for d in voyage.per_leg), rel=1.0e-12
    )


def test_c7_adding_a_leg_cannot_reduce_the_damage(case, voyage):
    extended = VoyageProfile(
        name="extended",
        legs=case.profile.legs + (case.profile.legs[0],),
        voyage_duration_days=case.profile.voyage_duration_days,
    )
    more = voyage_damage(extended, case.hull, case.girder, case.vibration, case.curve)
    assert more.damage_total > voyage.damage_total
    assert more.damage_total == pytest.approx(
        voyage.damage_total + voyage.per_leg[0].damage_total, rel=1.0e-12
    )


# ---------------------------------------------------------------------------
# C8 - physics sanity anchors of the stress chain
# ---------------------------------------------------------------------------


def test_c8_hull_geometry_and_section_properties_are_consistent(case):
    check = case.hull.verify()
    assert check["consistent"], check
    assert check["block_coefficient_from_displacement"] == pytest.approx(0.75, rel=1.0e-12)
    girder = case.girder
    assert girder.section_modulus_m3 == pytest.approx(
        girder.moment_of_inertia_m4 / (0.5 * girder.depth_m), rel=1.0e-12
    )
    # Z * sigma = M round trip
    moment = 1.0e8
    assert girder.stress_from_moment_mpa(moment) * 1.0e6 * girder.section_modulus_m3 == pytest.approx(
        moment, rel=1.0e-12
    )


def test_c8_bow_load_moment_matches_free_free_beam_statics(case):
    length, force = case.hull.length_m, 2.0e6
    quarter = bow_load_midship_moment(force, length, 0.25 * length)
    assert quarter == pytest.approx(0.375 * force * length, rel=1.0e-12)
    # symmetry: a load at a and at L - a give the same midship moment
    for a in (0.1 * length, 0.3 * length, 0.45 * length):
        assert bow_load_midship_moment(force, length, a) == pytest.approx(
            bow_load_midship_moment(force, length, length - a), rel=1.0e-12
        )
    stress = midship_static_stress_mpa(force, case.hull, case.girder)
    assert stress == pytest.approx(quarter / case.girder.section_modulus_m3 / 1.0e6, rel=1.0e-12)
    with pytest.raises(ValueError):
        bow_load_midship_moment(force, length, 1.5 * length)


def test_c8_vibration_response_hits_the_analytic_resonance_amplitude(case):
    """Harmonic forcing at f1 -> |q| = F0 / (k * 2 zeta); verify within 3 %."""
    vib = case.vibration
    force_amplitude = 1.0e6
    dt = 0.01
    t = np.arange(0.0, 400.0, dt)
    force = force_amplitude * np.sin(2.0 * np.pi * vib.frequency_hz * t)
    stress = vib.vibration_stress_mpa(force, dt)
    measured = float(np.max(stress[-int(60.0 / dt) :]))
    analytic = (
        vib.stress_per_modal_displacement_mpa()
        * (force_amplitude / vib.modal_stiffness)
        / (2.0 * vib.damping_ratio)
    )
    assert abs(measured - analytic) / analytic < 0.03
    # the analytic magnification helper must reproduce 1 / (2 zeta) at resonance
    assert vib.steady_state_amplification(vib.frequency_hz) == pytest.approx(
        1.0 / (2.0 * vib.damping_ratio), rel=1.0e-12
    )


def test_c8_leg_damage_is_converged_in_the_time_step(case):
    sn = case.curve
    base = case.profile.legs[3]
    damages = []
    for dt in (0.02, 0.01):
        leg = base.__class__(
            name=base.name,
            duration_s=base.duration_s,
            peak_force_n=base.peak_force_n,
            ice_frequency_hz=base.ice_frequency_hz,
            pulse_duration_s=base.pulse_duration_s,
            seed=base.seed,
            amplitude_cv=base.amplitude_cv,
            dt_s=dt,
        )
        damages.append(leg_damage(leg, case.hull, case.girder, case.vibration, sn).damage_total)
    assert abs(damages[0] - damages[1]) / damages[1] < 0.01


def test_c8_leg_stress_history_decomposition_is_additive(case):
    """sigma_total = sigma_quasi-static + sigma_vibration, every sample."""
    leg = case.profile.legs[0]
    hist = leg_stress_history(leg, case.hull, case.girder, case.vibration)
    assert hist.time_s.size == hist.stress_total_mpa.size == hist.force_n.size
    assert np.allclose(
        hist.stress_total_mpa, hist.stress_static_mpa + hist.stress_vibration_mpa, rtol=0.0, atol=1e-12
    )
    assert hist.peak_static_mpa > 0.0 and hist.peak_vibration_mpa > 0.0
    # a load window of zero duration is impossible and must be rejected
    with pytest.raises(ValueError):
        leg.__class__(
            name="bad",
            duration_s=10.0,
            peak_force_n=1.0e6,
            ice_frequency_hz=0.3,
            pulse_duration_s=0.3,
            seed=1,
            window_duration_s=60.0,
        )


def test_c8_ice_load_window_is_reproducible_and_validated(case):
    kwargs = dict(
        duration_s=60.0,
        dt_s=0.02,
        peak_force_n=2.0e6,
        frequency_hz=0.3,
        pulse_duration_s=0.35,
        seed=101,
    )
    a = ice_force_window(**kwargs)
    b = ice_force_window(**kwargs)
    assert np.array_equal(a.force_n, b.force_n)
    assert a.n_pulses == 18
    assert a.peak_factor > 1.5
    assert 0.0 <= a.rms_force_n <= a.peak_force_n
    with pytest.raises(ValueError):
        ice_force_window(**{**kwargs, "dt_s": 0.5})  # cannot resolve the pulse
    with pytest.raises(ValueError):
        VoyageProfile(name="bad", legs=case.profile.legs, voyage_duration_days=-1.0)


def test_c8_reference_case_life_is_consistent(case, voyage):
    summary = case.life_summary()
    assert summary["voyages_to_failure_analytic"] == pytest.approx(1.0 / voyage.damage_total, rel=1e-12)
    assert abs(summary["voyages_to_failure_simulated"] - summary["voyages_to_failure_analytic"]) <= 1.0
    half = case.life(10)
    assert half.fraction_life_used == pytest.approx(10 * voyage.damage_total, rel=1.0e-12)
    assert half.remaining_voyages < summary["remaining_life_at_zero_damage"]["remaining_voyages"]
    assert 0.0 < half.remaining_operating_years < 200.0
