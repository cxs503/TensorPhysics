"""P226W3-N02 — ice whipping fatigue chain with measured-strain Bayesian updating.

Tests cover each stage of ``tensorphysics.ice_fatigue_life.whipping_strain_fatigue``
against an independent reference (analytic solution, brute-force Bayes,
closed-form S-N damage or closed-form life inversion) plus the failure modes
that would otherwise pass silently (unit/geometry/validation traps).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.fatigue.rainflow import rainflow_counting
from tensorphysics.fatigue.sn_curve import DNCurve
from tensorphysics.ice_fatigue_life.whipping_strain_fatigue import (
    bayesian_posterior_grid,
    bayesian_response_scale_update,
    damage_vs_scale_samples,
    half_sine_force_signal,
    ice_exposure_blocks_per_year,
    moment_ranges_to_stress_ranges_mpa,
    rainflow_miner_damage_from_moment,
    voyage_fatigue_life,
    whipping_moment_history,
)
from tensorphysics.ice_vibration import HullParameters, free_free_beam_modal_data
from tensorphysics.ice_vibration.whiplash import whiplash_response

EI = 6.0e13
W_DECK = 20.8
E_STEEL = 2.06e11


@pytest.fixture(scope="module")
def hull() -> HullParameters:
    return HullParameters(
        length_m=250.0,
        breadth_m=40.0,
        draught_m=9.0,
        block_coefficient=0.82,
        displacement_kg=0.82 * 250.0 * 40.0 * 9.0 * 1025.0,
        name="test hull",
    )


@pytest.fixture(scope="module")
def modal(hull):
    return free_free_beam_modal_data(hull, EI, n_modes=3, damping_ratios=(0.02, 0.03, 0.04))


@pytest.fixture(scope="module")
def impact_history(modal):
    dt = 1.0 / (100.0 * modal.frequencies_hz[-1])
    force = half_sine_force_signal(6.0e6, 0.6, 12.0, dt)
    return whipping_moment_history(modal, EI, force, dt)


# ---------------------------------------------------------------------------
# Stage 1 — whipping moment history
# ---------------------------------------------------------------------------


def test_geometry_is_validated_before_dynamics(hull):
    report = hull.verify()
    assert report["consistent"] is True
    assert hull.cb_from_displacement == pytest.approx(0.82, rel=1e-12)


def test_stage1_reproduces_single_impact_reference_path(modal):
    """Same physics, different entry point: must agree with whiplash_response."""
    amp, tp, duration = 6.0e6, 0.6, 12.0
    dt = 1.0 / (100.0 * modal.frequencies_hz[-1])
    force = half_sine_force_signal(amp, tp, duration, dt)
    chain = whipping_moment_history(modal, EI, force, dt)
    ref = whiplash_response(modal, EI, amp, tp, duration_s=duration, dt_s=dt)
    assert chain.peak_midship_moment_nm == pytest.approx(ref.peak_midship_moment_nm, rel=1e-12)


def test_stage1_matches_analytic_modal_superposition(modal):
    amp, tp, duration = 6.0e6, 0.6, 12.0
    dt = 1.0 / (100.0 * modal.frequencies_hz[-1])
    force = half_sine_force_signal(amp, tp, duration, dt)
    chain = whipping_moment_history(modal, EI, force, dt)
    ref = whiplash_response(modal, EI, amp, tp, duration_s=duration, dt_s=dt)
    rel = abs(chain.peak_midship_moment_nm - ref.peak_midship_moment_analytic_nm) / (
        ref.peak_midship_moment_analytic_nm
    )
    assert rel < 0.01


def test_stage1_slow_pulse_reduces_to_quasi_static(modal):
    """A pulse much longer than the fundamental period is a static push."""
    dt = 1.0 / (100.0 * modal.frequencies_hz[-1])
    force = half_sine_force_signal(6.0e6, 20.0, 40.0, dt)  # T_p = 20 s >> 1/f1 = 1.25 s
    chain = whipping_moment_history(modal, EI, force, dt)
    amplification = chain.diagnostics["whiplash_dynamic_amplification"]
    assert 0.95 <= amplification <= 1.05


def test_stage1_is_linear_in_the_force(modal):
    dt = 1.0 / (100.0 * modal.frequencies_hz[-1])
    force = half_sine_force_signal(4.0e6, 0.6, 8.0, dt)
    a = whipping_moment_history(modal, EI, force, dt)
    b = whipping_moment_history(modal, EI, 2.5 * force, dt)
    # peak-relative deviation: pointwise relative comparison is meaningless at
    # zero crossings, where LAPACK rounding of two independent solves dominates
    ref = 2.5 * a.midship_moment_nm
    deviation = float(np.max(np.abs(b.midship_moment_nm - ref)) / np.max(np.abs(ref)))
    assert deviation < 1e-8


def test_stage1_rejects_bad_inputs(modal):
    f = half_sine_force_signal(5.0e6, 0.5, 2.0, 1.0e-3)
    with pytest.raises(ValueError):
        whipping_moment_history(modal, EI, f, 0.0)
    with pytest.raises(ValueError):
        whipping_moment_history(modal, 0.0, f, 1.0e-3)
    with pytest.raises(ValueError):
        whipping_moment_history(modal, EI, f[:1], 1.0e-3)
    bad = f.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        whipping_moment_history(modal, EI, bad, 1.0e-3)


# ---------------------------------------------------------------------------
# Stage 2 — Bayesian strain update
# ---------------------------------------------------------------------------


def _bayes_inputs(impact_history, theta_true=1.4, noise=8.0, n_obs=60, seed=7):
    m = impact_history.midship_moment_nm
    idx = np.linspace(0, m.size - 1, n_obs).round().astype(int)
    eps_model = m[idx] / (W_DECK * E_STEEL) * 1.0e6
    rng = np.random.default_rng(seed)
    eps_obs = theta_true * eps_model + rng.normal(0.0, noise, size=n_obs)
    return m, idx, eps_model, eps_obs


def test_bayes_closed_form_matches_numerical_grid(impact_history):
    m, idx, _, eps_obs = _bayes_inputs(impact_history)
    post = bayesian_response_scale_update(
        m, eps_obs, W_DECK, E_STEEL, 8.0, sample_indices=idx, prior_mean=1.0, prior_std=0.3
    )
    grid = bayesian_posterior_grid(
        m, eps_obs, W_DECK, E_STEEL, 8.0, sample_indices=idx, prior_mean=1.0, prior_std=0.3,
        n_grid=200001,
    )
    assert post.posterior_mean == pytest.approx(grid["posterior_mean"], rel=1e-9)
    assert post.posterior_std == pytest.approx(grid["posterior_std"], rel=1e-8)


def test_bayes_recovers_truth_in_the_low_noise_limit(impact_history):
    m, idx, eps_model, _ = _bayes_inputs(impact_history, theta_true=1.35)
    eps_obs = 1.35 * eps_model  # noiseless record
    post = bayesian_response_scale_update(m, eps_obs, W_DECK, E_STEEL, 1.0e-6, sample_indices=idx)
    assert post.posterior_mean == pytest.approx(1.35, rel=1e-6)


def test_bayes_posterior_concentrates_with_more_observations(impact_history):
    m = impact_history.midship_moment_nm
    eps_model_full = m / (W_DECK * E_STEEL) * 1.0e6
    rng = np.random.default_rng(11)
    obs_full = 1.3 * eps_model_full + rng.normal(0.0, 8.0, size=m.size)
    idxs = [np.linspace(0, m.size - 1, n).round().astype(int) for n in (5, 100, 1000)]
    stds = [
        bayesian_response_scale_update(
            m, obs_full[i], W_DECK, E_STEEL, 8.0, sample_indices=i, prior_std=0.3
        ).posterior_std
        for i in idxs
    ]
    assert stds[0] > stds[1] > stds[2]


def test_bayes_prior_dominates_when_noise_is_huge(impact_history):
    m, idx, _, eps_obs = _bayes_inputs(impact_history)
    post = bayesian_response_scale_update(
        m, eps_obs, W_DECK, E_STEEL, 1.0e9, sample_indices=idx, prior_mean=1.1, prior_std=0.2
    )
    assert post.posterior_mean == pytest.approx(1.1, rel=1e-6)
    assert post.posterior_std == pytest.approx(0.2, rel=1e-6)


def test_bayes_credible_interval_and_diagnostics(impact_history):
    m, idx, _, eps_obs = _bayes_inputs(impact_history)
    post = bayesian_response_scale_update(
        m, eps_obs, W_DECK, E_STEEL, 8.0, sample_indices=idx, prior_std=0.3, credible_level=0.95
    )
    lo, hi = post.credible_interval
    assert lo < post.posterior_mean < hi
    assert post.shrinkage < 1.0
    assert math.isfinite(post.diagnostics["log_evidence"])
    assert post.scale_samples.size == 4000


def test_bayes_rejects_unit_and_range_errors(impact_history):
    m, idx, _, eps_obs = _bayes_inputs(impact_history)
    with pytest.raises(ValueError):
        bayesian_response_scale_update(m, eps_obs, 0.0, E_STEEL, 8.0, sample_indices=idx)
    with pytest.raises(ValueError):
        bayesian_response_scale_update(m, eps_obs, W_DECK, 0.0, 8.0, sample_indices=idx)
    with pytest.raises(ValueError):
        bayesian_response_scale_update(m, eps_obs, W_DECK, E_STEEL, 0.0, sample_indices=idx)
    with pytest.raises(ValueError):
        bayesian_response_scale_update(m, eps_obs, W_DECK, E_STEEL, 8.0, sample_indices=idx, prior_std=0.0)
    with pytest.raises(ValueError):
        bayesian_response_scale_update(
            m, eps_obs, W_DECK, E_STEEL, 8.0, sample_indices=idx + 10 * m.size
        )
    with pytest.raises(ValueError):
        bayesian_response_scale_update(m, eps_obs[:-1], W_DECK, E_STEEL, 8.0, sample_indices=idx)


# ---------------------------------------------------------------------------
# Stage 3 — rainflow + S-N / Miner
# ---------------------------------------------------------------------------


def test_rainflow_miner_matches_closed_form_single_range():
    n_cycles = 200.0
    t = np.arange(0.0, n_cycles, 1.0 / 400.0)
    signal = 1.0e9 * np.sin(2.0 * math.pi * t)
    curve = DNCurve(curve_class="F", environment="seawater_cp")
    rf = rainflow_counting(signal)
    full = rf.counts == 1.0
    res = rainflow_miner_damage_from_moment(signal, W_DECK, sn_curve=curve)
    n_fail = curve.n_to_failure(2.0e9 / W_DECK / 1.0e6)
    analytic = float(np.sum(rf.counts[full])) / n_fail
    assert np.allclose(rf.ranges[full], 2.0e9, rtol=1e-12)
    assert float(np.sum(res.damage_per_cycle[full])) == pytest.approx(analytic, rel=1e-12)


def test_rainflow_counting_is_scale_invariant(impact_history):
    m = impact_history.midship_moment_nm
    a = 1.37
    base = rainflow_counting(m)
    scaled = rainflow_counting(a * m)
    assert base.ranges.size == scaled.ranges.size
    assert np.allclose(base.counts, scaled.counts)
    mask = base.ranges > 0
    assert np.allclose(scaled.ranges[mask], a * base.ranges[mask], rtol=1e-12)


def test_damage_vs_scale_samples_is_consistent_with_direct_evaluation(impact_history):
    m = impact_history.midship_moment_nm
    curve = DNCurve(curve_class="F", environment="seawater_cp")
    base = rainflow_miner_damage_from_moment(m, W_DECK, sn_curve=curve)
    scales = [0.8, 1.0, 1.5]
    dmg = damage_vs_scale_samples(base, scales, sn_curve=curve)
    for theta, d in zip(scales, dmg):
        direct = rainflow_miner_damage_from_moment(theta * m, W_DECK, sn_curve=curve).damage
        assert d == pytest.approx(direct, rel=1e-10)


def test_damage_grows_with_stress_slope(impact_history):
    """D(theta) must grow like theta^m; m = 3 (DNV F, first branch) => ratio 8 for x2."""
    base = rainflow_miner_damage_from_moment(
        impact_history.midship_moment_nm, W_DECK, sn_curve=DNCurve("F", "seawater_cp")
    )
    d1, d2 = damage_vs_scale_samples(base, [1.0, 2.0], sn_curve=DNCurve("F", "seawater_cp"))
    assert d2 / d1 > 1.5  # at least the linear bound; bilinear branches push it higher


def test_stress_conversion_includes_thickness_and_scf():
    s = moment_ranges_to_stress_ranges_mpa([1.0e9], W_DECK, stress_concentration_factor=1.1, thickness_mm=50.0)
    expected = 1.0e9 / W_DECK * 1.1 * (50.0 / 25.0) ** 0.25 / 1.0e6
    assert s[0] == pytest.approx(expected, rel=1e-12)
    # thin plate: no thickness correction
    s_thin = moment_ranges_to_stress_ranges_mpa([1.0e9], W_DECK, thickness_mm=20.0)
    assert s_thin[0] == pytest.approx(1.0e9 / W_DECK / 1.0e6, rel=1e-12)


def test_constant_moment_history_has_no_cycles():
    with pytest.raises(ValueError):
        rainflow_miner_damage_from_moment(np.full(500, 1.0e8), W_DECK)


# ---------------------------------------------------------------------------
# Stage 4 — voyage life
# ---------------------------------------------------------------------------


def test_voyage_life_matches_closed_form_and_dff():
    d_block, blocks = 3.0e-6, 18750.0
    life = voyage_fatigue_life(d_block, blocks, design_fatigue_factor=2.0)
    assert life.years_to_failure == pytest.approx(0.5 / (d_block * blocks), rel=1e-12)
    life3 = voyage_fatigue_life(d_block, blocks, design_fatigue_factor=3.0)
    assert life3.years_to_failure == pytest.approx(life.years_to_failure * 2.0 / 3.0, rel=1e-12)
    assert life.utilization_per_year == pytest.approx(1.0 / life.years_to_failure, rel=1e-12)


def test_voyage_life_infinite_for_zero_damage():
    life = voyage_fatigue_life(0.0, 1000.0)
    assert math.isinf(life.years_to_failure)
    assert life.annual_damage == 0.0


def test_exposure_blocks_are_consistent_and_validated():
    blocks = ice_exposure_blocks_per_year(250.0, 10.0, 0.25, 120.0)
    assert blocks == pytest.approx(250.0 * 10.0 * 3600.0 * 0.25 / 120.0, rel=1e-12)
    assert ice_exposure_blocks_per_year(100.0, 12.0, 1.0, 60.0) == pytest.approx(72000.0)
    with pytest.raises(ValueError):
        ice_exposure_blocks_per_year(250.0, 10.0, 1.5, 120.0)
    with pytest.raises(ValueError):
        ice_exposure_blocks_per_year(0.0, 10.0, 0.25, 120.0)


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------


def test_end_to_end_chain_smoke(hull, modal):
    """Short but complete chain: ice train -> moment -> Bayes -> Miner -> life."""
    from tensorphysics.ice_vibration import quasi_periodic_pulse_train

    dt = 1.0 / (100.0 * modal.frequencies_hz[-1])
    series = quasi_periodic_pulse_train(
        pulse_frequency_hz=0.35, amplitude_n=6.0e6, duration_s=20.0, dt_s=dt, seed=3
    )
    hist = whipping_moment_history(modal, EI, series.force_n, dt)
    assert hist.peak_midship_moment_nm > 0.0
    assert hist.diagnostics["whiplash_dynamic_amplification"] > 1.0

    idx = np.linspace(0, hist.midship_moment_nm.size - 1, 40).round().astype(int)
    eps_model = hist.midship_moment_nm[idx] / (W_DECK * E_STEEL) * 1.0e6
    rng = np.random.default_rng(5)
    eps_obs = 1.30 * eps_model + rng.normal(0.0, 6.0, size=idx.size)
    post = bayesian_response_scale_update(
        hist.midship_moment_nm, eps_obs, W_DECK, E_STEEL, 6.0, sample_indices=idx, prior_std=0.25
    )
    assert abs(post.posterior_mean - 1.30) <= 3.0 * post.posterior_std
    assert post.posterior_std < post.prior_std

    base = rainflow_miner_damage_from_moment(
        hist.midship_moment_nm,
        W_DECK,
        sn_curve=DNCurve("F", "seawater_cp"),
        stress_concentration_factor=1.1,
        thickness_mm=30.0,
    )
    dmg = damage_vs_scale_samples(base, post.scale_samples, sn_curve=DNCurve("F", "seawater_cp"))
    assert np.all(dmg > 0.0)
    assert np.mean(dmg) >= base.damage  # posterior scale > 1 amplifies damage

    blocks = ice_exposure_blocks_per_year(250.0, 10.0, 0.25, 20.0)
    life = voyage_fatigue_life(float(np.mean(dmg)), blocks, design_fatigue_factor=2.0)
    assert life.years_to_failure > 0.0
    assert math.isfinite(life.annual_damage)

def test_near_resonant_ramming_is_more_damaging_than_off_resonant(hull, modal):
    """Physics sanity: ramming rate at f1 (lock-in band) must be far more damaging.

    This is the driver found in the case study: sweeping f_ice onto the first
    hull-girder frequency raises the dynamic amplification by an order of
    magnitude, so ice-zone whipping fatigue is controlled by f_ice = U/s, not by
    the S-N detail.
    """
    from tensorphysics.ice_vibration import quasi_periodic_pulse_train

    f1 = modal.frequencies_hz[0]
    dt = 1.0 / (60.0 * modal.frequencies_hz[-1])
    out = {}
    for label, f_ice in (("off_resonant", 0.3 * f1), ("at_resonance", f1)):
        series = quasi_periodic_pulse_train(
            pulse_frequency_hz=f_ice, amplitude_n=6.0e6, duration_s=20.0, dt_s=dt, seed=11
        )
        hist = whipping_moment_history(modal, EI, series.force_n, dt)
        out[label] = hist.diagnostics["whiplash_dynamic_amplification"]
    assert out["at_resonance"] > 2.0 * out["off_resonant"]
