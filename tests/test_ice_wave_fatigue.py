"""Tests for the ice + wave combined fatigue module (node P82W4-N06, mission m7361).

Every group anchors one link of the chain to an independent reference:

* beam frequencies -> analytic free-free ``beta_L`` roots and the closed-form
  ``f_n = (beta_n L)^2 sqrt(EI/(m L^4)) / 2 pi``;
* modal superposition -> single-mode dynamic amplification factor
  ``1/sqrt((1-r^2)^2 + (2 zeta r)^2)``;
* rainflow -> ASTM E1049-85 counting on a sine (M periods -> M cycles) and
  agreement with the pre-existing ``tensorphysics.fatigue.rainflow``
  implementation;
* S-N -> DNV-RP-C203 tabulated ``Delta sigma(N=1e7)`` anchors;
* wave spectra -> ``m0 = Hs^2/16``;
* DEM ice load -> the m5528 IN32-N07 published statistics (negative control);
* combination law -> the exact limits (kappa(0) = 1, beta in [0, 1]).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorphysics.fatigue.rainflow import rainflow_counting_stack as reference_rainflow  # noqa: E402
from tensorphysics.fatigue.spectral import (  # noqa: E402
    dirlik_damage_rate,
    moments_from_dict,
    rayleigh_damage_rate,
    spectral_moments,
)
from tensorphysics.ice_wave_fatigue import (  # noqa: E402
    CALIBRATED_M5528,
    BeamModes,
    HullGirderCase,
    SeaState,
    StreamingRainflow,
    band_separation_scan,
    bimodal_f,
    calibrate_combination_law,
    combination_case,
    combination_law,
    damage_from_cycles,
    evaluate_combination_law,
    free_free_beta_roots,
    generate_ice_force,
    harmonic_ice_stress,
    ice_stress_history,
    jonswap_f,
    kappa_enlargement,
    load_statistics,
    m5528_reproduction,
    mechanism_diagnostic,
    modal_stress_history,
    overlap_weight,
    physical_wave_component,
    quasi_static_moment_free_free,
    rainflow_cycles,
    segmentation_invariance,
    spectral_width_measures,
    turning_points,
    wave_stress_psd,
    whipping_accounting,
    whipping_share,
)
from tensorphysics.ice_wave_fatigue import sn as snmod  # noqa: E402


# ---------------------------------------------------------------------------
# beam / modal superposition
# ---------------------------------------------------------------------------
def test_free_free_frequencies_match_closed_form():
    L, EI, mpl = 250.0, 5.0e13, 5.2e5
    modes = BeamModes(L, EI, mpl, n_modes=4, damping_ratio=0.015)
    beta_l = np.array([4.730040744862704, 7.853204624095838, 10.995607838001671,
                       14.137165491257464])
    exact = (beta_l / L) ** 2 * math.sqrt(EI / mpl) / (2.0 * math.pi)
    np.testing.assert_allclose(modes.freq[:4], exact, rtol=1e-12)
    # m5528 reference first frequency
    assert modes.freq[0] == pytest.approx(0.5586672438138549, rel=1e-9)
    # beta roots increase monotonically
    assert np.all(np.diff(free_free_beta_roots(6, L)) > 0)


def test_modal_superposition_reproduces_harmonic_daf():
    """Single-mode harmonic load -> deflection amplification 1/sqrt((1-r^2)^2+(2 z r)^2)."""
    L, EI, mpl = 250.0, 5.0e13, 5.2e5
    modes = BeamModes(L, EI, mpl, n_modes=1, damping_ratio=0.015)
    zeta = 0.015
    dt = 0.005
    errors = []
    for ratio in (0.4, 0.8, 1.0, 1.5):
        n = int(round(400.0 / dt))
        t = np.arange(n) * dt
        f = ratio * modes.freq[0]
        F = 1.0e6 * np.sin(2.0 * math.pi * f * t)
        res = modal_stress_history(modes, F, dt, 0.05 * L, 0.30 * L, 10.5, 2.06e11)
        # start-up transient decays as exp(-zeta*omega*t); 300 s = 15 decay times
        skip = int(300.0 / dt)
        amp = np.abs(res["sigma"][skip:]).max()
        q_static = 1.0e6 * float(modes.shape(0.05 * L, 0)) / modes.modal_stiffness[0]
        kap = float(modes.curvature_shape(0.30 * L, 0))
        amp_static = abs(-2.06e11 * 10.5 * q_static * kap)
        daf_num = amp / amp_static
        daf_ref = 1.0 / math.sqrt((1.0 - ratio ** 2) ** 2 + (2.0 * zeta * ratio) ** 2)
        errors.append(abs(daf_num / daf_ref - 1.0))
    assert max(errors) < 0.02, errors


def test_free_free_static_moment_boundary_and_equilibrium():
    """M(0)=M(L)=0, the shear jumps by -F at the load, and the reactions balance."""
    L, a, F = 250.0, 12.5, 19.19e6
    x = np.linspace(0.0, L, 20001)
    M = quasi_static_moment_free_free(x, F, a, L)
    assert abs(M[0]) < 1e-9 * F * L and abs(M[-1]) < 1e-9 * F * L
    # distributed rigid-body reactions r(s) = F/L + c (s - L/2), c = 12 F (a-L/2)/L^3
    c = 12.0 * F * (a - 0.5 * L) / L ** 3
    r = F / L + c * (x - 0.5 * L)
    dMdx = np.gradient(M, x)
    # dM/dx = -R(x) + F H(x-a)  with R(x) = int_0^x r ds
    R = F * x / L + c * x * (x - L) / 2.0
    expected = -R + F * (x > a)
    # exclude the two samples straddling the shear jump at x = a, where the
    # finite-difference derivative smears a step of size F
    mask = np.abs(x - a) > 2.0 * (x[1] - x[0])
    mask[:2] = False
    mask[-2:] = False
    assert np.max(np.abs(dMdx[mask] - expected[mask])) < 5e-3 * F
    # the inertial reactions balance the applied load: net force F, and the
    # net moment about mid-ship equals F (a - L/2)
    assert float(np.trapezoid(r, x)) == pytest.approx(F, rel=1e-9)
    assert float(np.trapezoid(r * (x - 0.5 * L), x)) == pytest.approx(F * (a - 0.5 * L), rel=1e-6)
    # moment magnitude is of the physically expected order
    assert 0.02 * F * L < np.abs(M).max() < 0.30 * F * L


# ---------------------------------------------------------------------------
# rainflow
# ---------------------------------------------------------------------------
def test_rainflow_sine_counts_and_ranges():
    f, fs, T = 1.0, 100.0, 25.0
    t = np.arange(int(T * fs)) / fs
    x = 100.0 * np.sin(2.0 * math.pi * f * t)
    r, m, c = rainflow_cycles(turning_points(x))
    assert c.sum() == pytest.approx(25.0, abs=1.0)
    assert r.max() == pytest.approx(200.0, rel=1e-6)


def test_rainflow_agrees_with_existing_reference_implementation():
    """The in-repo ASTM stack implementation must give the identical cycle set.

    ``tensorphysics.fatigue.rainflow.rainflow_counting_stack`` is the O(N) ASTM
    E1049-85 stack implementation of the platform, so agreement here pins the new
    streaming counter to an independent implementation (the list-based
    ``rainflow_counting`` in the same module does *not* agree on noisy signals --
    see ``docs/ICE_WAVE_COMBINED_FATIGUE.md`` section 5).
    """
    rng = np.random.default_rng(7)
    t = np.arange(4000) / 40.0
    x = (np.sin(2 * math.pi * 0.5 * t) + 0.4 * np.sin(2 * math.pi * 3.1 * t + 1.0)
         + 0.05 * rng.standard_normal(t.size))
    r, m, c = rainflow_cycles(turning_points(x))
    ref = reference_rainflow(x)
    assert float(c.sum()) == pytest.approx(float(np.sum(ref.counts)), rel=1e-9)
    d_new = damage_from_cycles(r, c, 3.0)
    d_ref = float(np.sum(ref.counts * ref.ranges ** 3))
    assert d_new == pytest.approx(d_ref, rel=1e-9)


def test_streaming_rainflow_is_segmentation_invariant():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(20000))
    whole = StreamingRainflow().feed(x).finalize()
    r_w, _, c_w = whole.cycles()
    d_whole = damage_from_cycles(r_w, c_w)
    for n_chunks in (3, 17, 101):
        rf = StreamingRainflow()
        bounds = np.linspace(0, x.size, n_chunks + 1).astype(int)
        for a, b in zip(bounds[:-1], bounds[1:]):
            rf.feed(x[a:b])
        r_s, _, c_s = rf.finalize().cycles()
        assert damage_from_cycles(r_s, c_s) == pytest.approx(d_whole, rel=1e-12)


def test_closed_block_permutation_invariance_and_naive_trap():
    res = segmentation_invariance(n_blocks=8, seeds=tuple(range(6)))
    assert res["max_seg_rel_dev"] <= 1e-9
    assert res["permutation_max_rel_dev"] <= 1e-9
    # the naive whole-record count on permuted concatenations does depend on order
    assert res["naive_permutation_spread_rel"] > 1e-6


# ---------------------------------------------------------------------------
# S-N / Miner
# ---------------------------------------------------------------------------
def test_dnv_sn_curve_anchors():
    d_air = snmod.get_curve("DNV-D-air")
    f3_air = snmod.get_curve("DNV-F3-air")
    assert d_air.dsigma(1.0e7) == pytest.approx(52.63, rel=1e-3)
    assert f3_air.dsigma(1.0e7) == pytest.approx(31.86, rel=1e-3)
    # seawater with CP is a 2.5x life reduction at fixed range
    sw = snmod.get_curve("DNV-F3-sw-cp")
    n_air = f3_air.life(50.0)
    n_sw = sw.life(50.0)
    assert n_air / n_sw == pytest.approx(2.5, rel=1e-6)
    # thickness correction (t/25)^0.2
    assert sw.thickness_factor(28.0) == pytest.approx((28.0 / 25.0) ** 0.2, rel=1e-12)


def test_miner_two_block_is_additive():
    curve = snmod.get_curve("DNV-F3-sw-cp")
    s1, n1, s2, n2 = 80.0, 1.0e4, 40.0, 5.0e5
    d = snmod.miner_damage(np.array([s1, s2]), np.array([0.0, 0.0]),
                           np.array([n1, n2]), curve)
    expected = n1 / curve.life(s1) + n2 / curve.life(s2)
    assert d["damage"] == pytest.approx(expected, rel=1e-12)


def test_goodman_limits():
    assert snmod.goodman_amplitude(20.0, 0.0, 490.0) == pytest.approx(20.0)
    assert np.isinf(snmod.goodman_amplitude(20.0, 490.0, 490.0))
    assert snmod.goodman_amplitude(20.0, -245.0, 490.0) == pytest.approx(20.0 / 1.5)


# ---------------------------------------------------------------------------
# spectra and wave transfer
# ---------------------------------------------------------------------------
def test_jonswap_and_bimodal_normalisation():
    f = np.linspace(1e-4, 2.0, 40001)
    hs, tp = 4.0, 9.0
    s = jonswap_f(f, hs, tp, 3.3)
    m0 = float(np.trapezoid(s, f))
    assert math.sqrt(m0) == pytest.approx(hs / 4.0, rel=0.03)
    sb = bimodal_f(f, hs, tp, 3.3, swell_energy_fraction=0.5, peak_separation=3.0)
    m0b = float(np.trapezoid(sb, f))
    # same total energy by construction; the two peaks are separated
    assert math.sqrt(m0b) == pytest.approx(hs / 4.0, rel=0.05)
    wide = spectral_width_measures(f, sb)
    narrow = spectral_width_measures(f, jonswap_f(f, hs, tp, 6.0))
    assert wide["eps"] > narrow["eps"]


def test_wave_transfer_vanishes_for_short_waves():
    case = HullGirderCase()
    sea = SeaState()
    freqs = np.array([0.02, 0.05, 0.1, 0.5, 1.0])
    tr = wave_stress_psd(freqs, case.modes, sea.elevation(), case.x_detail,
                         case.z_deck, case.E, case.length, case.breadth, case.draught)
    mag = np.abs(tr["transfer_pa_per_m"])
    assert mag[0] > mag[3] and mag[0] > mag[4]
    # short waves (wavelength << L) cannot bend the girder: cancellation of the
    # phase term makes the transfer at least 20x smaller than the long-wave one
    assert mag[0] / mag[4] > 20.0


def test_overlap_weight_limits():
    f = np.linspace(0.0, 2.0, 2001)
    a = np.exp(-((f - 0.3) ** 2) / (2 * 0.02 ** 2))
    b = np.exp(-((f - 1.2) ** 2) / (2 * 0.02 ** 2))
    assert overlap_weight(f, a, b) == pytest.approx(0.0, abs=1e-6)
    assert overlap_weight(f, a, a) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# ice leg / negative control
# ---------------------------------------------------------------------------
def test_dem_load_statistics_match_m5528():
    t, F = generate_ice_force(CALIBRATED_M5528, duration=300.0, dt=2.0e-3)
    st = load_statistics(F[int(60.0 / 2.0e-3):], 2.0e-3, 0.27933362190692745)
    assert st["peak"] / 1e6 == pytest.approx(19.19124139841869, rel=1e-9)
    assert st["mean_peak"] == pytest.approx(0.5515846715114354, rel=1e-6)
    assert st["rise_median"] == pytest.approx(0.219, rel=1e-9)
    assert st["qs_energy_frac"] == pytest.approx(0.932543769465116, rel=1e-6)


def test_negative_control_reproduces_m5528():
    rep = m5528_reproduction(HullGirderCase())
    assert rep["pass"], rep["worst_rel_err_pct"]
    assert rep["worst_rel_err_pct"] < 1e-5
    for key in ("damage_annual_goodman", "life_years", "whipping_share_peak",
                "dsigma_eq_MPa", "f1_Hz"):
        assert rep["comparisons"][key]["within_tol"]


def test_whipping_share_collapses_for_slow_harmonic_load():
    case = HullGirderCase()
    s = harmonic_ice_stress(case, f_hz=0.05, duration=400.0, dt=2.0e-3)
    assert s["whipping"]["share_peak"] < 0.08


def test_whipping_bookkeeping_reports_both_calibers():
    wh = whipping_accounting(HullGirderCase(), m=3.0)
    rows = {r["regime"]: r for r in wh["rows"]}
    crushing = rows["crushing_m5528"]
    assert crushing["amplification_A"] > 1.5
    assert crushing["damage_share_whip"] > 0.9
    assert crushing["share_peak"] < 0.5
    # the peak-basis caliber understates the damage amplification
    assert crushing["A_predicted_peak_caliber"] < crushing["amplification_A"]
    assert set(wh["subchecks"]) >= {"c1_harmonic_slow_share_below_8pct",
                                    "c3_peak_caliber_within_2pp"}


# ---------------------------------------------------------------------------
# combination law
# ---------------------------------------------------------------------------
def test_kappa_limits_and_monotonicity():
    assert kappa_enlargement(0.0, 10.0) == pytest.approx(1.0)
    assert kappa_enlargement(0.5, 10.0) > kappa_enlargement(0.2, 10.0)
    assert kappa_enlargement(0.5, 100.0) > kappa_enlargement(0.5, 10.0)


def test_combination_law_reduces_to_sum_rule_without_interaction():
    row = {
        "ratio": 10.0, "f_ice_hz": 0.43, "f_wave_hz": 0.043,
        "sigma_ice_mpa": 1.0, "sigma_wave_mpa": 1.0,
        "d_ice_rate": 2.0, "d_wave_rate": 1.0, "d_comb_rate": 3.0,
        "d_tb_rate": 3.0, "overlap_beta": 0.0,
    }
    law = combination_law(row, {"a": 0.0, "b": 1.0, "c": 0.0})
    assert law["kappa_carrier_amplification"] == pytest.approx(1.0)
    assert law["rho_law"] == pytest.approx(1.0)


def test_combination_law_uses_merged_limit_when_bands_overlap():
    row = {
        "ratio": 1.0, "f_ice_hz": 0.43, "f_wave_hz": 0.43,
        "sigma_ice_mpa": 1.0, "sigma_wave_mpa": 1.0,
        "d_ice_rate": 1.0, "d_wave_rate": 1.0, "d_comb_rate": 3.0,
        "d_tb_rate": 4.0, "overlap_beta": 1.0,
    }
    law = combination_law(row, {"a": 0.0, "b": 1.0, "c": 0.0})
    assert law["rho_law"] == pytest.approx(2.0)


def test_category_a_scan_and_law_calibration_smoke():
    case = HullGirderCase()
    scan = band_separation_scan(case, ratios=(10.0, 100.0), seeds=(0, 1),
                                amplitude_modes=("equal_rms",),
                                sigma_ratios=(0.3, 0.6, 1.0, 2.0),
                                duration=400.0, fs=20.0)
    rows = scan["rows"]
    assert len(rows) == 8
    # summing the two damages underestimates the combined rainflow damage
    assert max(r["rho_mean"] for r in rows) > 1.0
    for r in rows:
        assert r["rho_mean"] >= 0.99
    calib = calibrate_combination_law(rows, training_ratios=(10.0, 100.0))
    assert calib["n_points"] >= 3
    ev = evaluate_combination_law(rows, calib, holdout_ratios=())
    assert ev["law_stats_all"]["max_abs_pct"] < 20.0
    # the law is much better than the plain sum rule
    assert ev["law_stats_all"]["mean_abs_pct"] < ev["sum_rule_stats_all"]["mean_abs_pct"]


def test_mechanism_diagnostic_matches_predicted_range_enlargement():
    diag = mechanism_diagnostic(HullGirderCase(), ratio=10.0, seed=0,
                               duration=600.0, fs=20.0)
    assert diag["delta_measured_mpa"] > 0.0
    # the measured carrier-range enlargement is of the same order as
    # 2 sqrt(pi/2) sigma_rider (the prediction behind kappa(r, N))
    assert 0.2 < diag["delta_predicted_mpa"] / diag["delta_measured_mpa"] < 5.0


def test_combination_case_reports_finite_damage_ratio():
    case = HullGirderCase()
    r = combination_case(case, 1.0, seed=0, duration=300.0, fs=20.0)
    assert 0.9 < r["rho"] < 3.0
    assert r["d_ice_rate"] > 0 and r["d_wave_rate"] > 0
    assert 0.0 <= r["overlap_beta"] <= 1.0


# ---------------------------------------------------------------------------
# physical (Level B) leg
# ---------------------------------------------------------------------------
def test_physical_wave_component_matches_spectral_rms():
    case = HullGirderCase()
    sea = SeaState()
    w = physical_wave_component(case, sea, n_samples=18000, fs=100.0, seed=0)
    assert w["sigma_rms_mpa"] == pytest.approx(w["psd_sigma_rms_mpa"], rel=0.10)
    assert 1.0 < w["sigma_rms_mpa"] < 40.0
    assert w["width"]["eps"] > 0.0


# ---------------------------------------------------------------------------
# spectral methods (criterion b)
# ---------------------------------------------------------------------------
def test_dirlik_reduces_to_rayleigh_in_narrowband_limit():
    f = np.linspace(0.0, 4.0, 4001)
    s = np.exp(-((f - 1.0) ** 2) / (2 * 0.002 ** 2))
    mom = moments_from_dict(spectral_moments(f, s))
    assert dirlik_damage_rate(mom, 1.0, 3.0) == pytest.approx(
        rayleigh_damage_rate(mom, 1.0, 3.0), rel=1e-4)


def test_pipeline_quick_smoke(tmp_path):
    from tensorphysics.ice_wave_fatigue.pipeline import run_study
    res = run_study(quick=True)
    crit = res["criteria"]
    assert set(crit) == {"a_combination_law", "b_spectral_methods",
                         "c_whipping_bookkeeping", "d_rainflow_commutability",
                         "e_negative_control_m5528"}
    assert crit["e_negative_control_m5528"]["pass"]
    assert crit["d_rainflow_commutability"]["segmentation_pass"]
    assert res["criteria_summary"]["all_pass"] in (True, False)
    assert res["case"]["f1_hz"] == pytest.approx(0.5586672438138549, rel=1e-9)
    # JSON serialisable
    import json
    json.dumps(res)