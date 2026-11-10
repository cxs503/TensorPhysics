"""Tests for node ``P82W5-N09``: bimodal spectra, sum-rule adjudication, whipping.

The suite pins (a) the *implementation* invariants of the band split and of the
two-band closure, (b) the *declared* thresholds, and (c) the *measured* verdicts
that the report quotes, on short records where that is cheap enough for CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_wave_fatigue.bimodal import (
    N_SEPARATED,
    KappaClosure,
    band_case,
    band_split,
    bimodal_benchmark,
    equal_damage_amplitude_ratio,
    riding_kappa_convolution,
    sum_rule_envelope,
    two_band_riding_damage,
)
from tensorphysics.ice_wave_fatigue.combination import HullGirderCase, WIDTH_LEVELS
from tensorphysics.ice_wave_fatigue.whipping_regime import (
    JUMP_FRACTION_MAX,
    THETA_MIN,
    regime_plane,
    regime_consistency,
    whipping_regime_scan,
)
from tensorphysics.ice_wave_fatigue.w5n09 import (
    commutation_check,
    run_w5n09_study,
    summarise_criteria,
)

#: Closure fitted on the published Level-A grid (``artifacts/m8866/results.json``
#: -> ``calibration.closure``).  Pinned so that the closure is a fixed object in
#: the tests instead of being refitted on every run.
CLOSURE = KappaClosure(A=1.1971297933595106, b=0.3946966159032267, p=1.3565214734957316,
                       C=0.001, c=1.2182296884200563, q=0.05)


@pytest.fixture(scope="module")
def case() -> HullGirderCase:
    return HullGirderCase()


def test_equal_damage_amplitude_ratio_is_n_to_the_minus_one_over_m():
    assert equal_damage_amplitude_ratio(10.0, 3.0) == pytest.approx(10.0 ** (-1.0 / 3.0))
    assert equal_damage_amplitude_ratio(1.0, 3.0) == pytest.approx(1.0)
    # a band ratio below one cannot hide a rider: N is clamped at 1
    assert equal_damage_amplitude_ratio(0.3, 3.0) == pytest.approx(1.0)


def test_band_split_conserves_variance_and_places_the_valley(case):
    out = band_case(case, n_ratio=10.0, seed=0, duration=240.0, sigma_carrier=10.0)
    split = band_split(out["freqs"], out["s_comb"], out["f_ice_hz"], out["f_wave_hz"])
    assert out["f_wave_hz"] < split.f_valley_hz < out["f_ice_hz"]
    m0_total = split.mom_comb.m0
    assert split.mom_carrier.m0 + split.mom_rider.m0 == pytest.approx(m0_total, rel=1e-10)
    assert split.n_ratio == pytest.approx(10.0, rel=1e-9)
    # rider = higher frequency band -> its spectral centroid must be above the carrier's
    assert split.mom_rider.nu0 > split.mom_carrier.nu0


def test_band_split_merged_limit_uses_geometric_mean(case):
    out = band_case(case, n_ratio=1.0, seed=0, duration=120.0, sigma_carrier=10.0)
    split = band_split(out["freqs"], out["s_comb"], out["f_ice_hz"], out["f_wave_hz"])
    assert split.f_valley_hz == pytest.approx(out["f_ice_hz"], rel=1e-9)
    assert split.n_ratio == pytest.approx(1.0, rel=1e-9)


def test_riding_convolution_matches_the_analytic_m1_limit():
    """For ``m = 1`` the amplification is ``1 + sqrt(2) r`` exactly.

    ``E[R] = sqrt(2) sigma_c sqrt(pi/2)`` and ``E|delta| = 2 sigma_r sqrt(pi/2)``,
    hence ``E[R + |delta|] / E[R] = 1 + sqrt(2) r``.
    """
    for r in (0.1, 0.3, 0.6, 1.0):
        kappa = riding_kappa_convolution(1.0, r, m=1.0)
        assert kappa == pytest.approx(1.0 + math.sqrt(2.0) * r, rel=2e-3)


def test_riding_convolution_grows_with_r_and_m():
    ks = [riding_kappa_convolution(1.0, r, m=3.0) for r in (0.02, 0.05, 0.2, 0.5, 1.0)]
    assert ks[0] == pytest.approx(1.0, abs=0.1)
    assert ks == sorted(ks)
    assert riding_kappa_convolution(1.0, 0.50, m=5.0) > riding_kappa_convolution(1.0, 0.50, m=3.0)


def test_closure_is_bounded_and_increasing_in_r_and_N():
    assert CLOSURE.kappa(0.0, 10.0) == pytest.approx(1.0)
    assert CLOSURE.kappa(0.3, 10.0) > 1.0
    assert CLOSURE.kappa(0.3, 100.0) > CLOSURE.kappa(0.3, 10.0)
    assert CLOSURE.kappa(0.5, 10.0) > CLOSURE.kappa(0.3, 10.0)
    assert np.isfinite(CLOSURE.kappa(5.0, 1.0))


def test_two_band_closure_reduces_to_dirlik_in_the_merged_regime(case):
    out = band_case(case, n_ratio=1.0, seed=0, duration=240.0, sigma_carrier=10.0)
    res = two_band_riding_damage(out, CLOSURE, 3.0)
    assert res["regime"] == "merged_band_dirlik"
    assert res["kappa"] == 1.0
    assert res["damage"] == pytest.approx(out["d_comb_dirlik"])


def test_two_band_closure_uses_the_riding_branch_above_five(case):
    out = band_case(case, n_ratio=10.0, seed=0, duration=240.0, sigma_carrier=10.0)
    res = two_band_riding_damage(out, CLOSURE, 3.0)
    assert res["regime"] == "two_band_riding"
    assert res["n_ratio"] >= N_SEPARATED
    assert res["kappa"] > 1.0
    expected = out["d_rider_dirlik"] + res["kappa"] * out["d_carrier_dirlik"]
    assert res["damage"] == pytest.approx(expected)


def test_band_case_is_seed_reproducible(case):
    a = band_case(case, n_ratio=10.0, seed=7, duration=240.0, sigma_carrier=10.0)
    b = band_case(case, n_ratio=10.0, seed=7, duration=240.0, sigma_carrier=10.0)
    assert a["d_truth"] == b["d_truth"]
    assert a["rho_components"] == b["rho_components"]


def test_sum_rule_envelope_shrinks_with_band_separation_and_tolerance():
    env3 = sum_rule_envelope(CLOSURE, tol=0.03)
    env1 = sum_rule_envelope(CLOSURE, tol=0.01)
    r3 = [r["r_max_sum_rule"] for r in env3["rows"]]
    r1 = [r["r_max_sum_rule"] for r in env1["rows"]]
    assert all(a <= b + 1e-12 for a, b in zip(r1, r3))
    assert r3 == sorted(r3, reverse=True)
    assert env3["rows"][0]["sum_rule_at_r_tiny"] == pytest.approx(1.0, abs=1e-4)


def test_sum_rule_envelope_uses_the_component_damage_ratio():
    env_eq = sum_rule_envelope(CLOSURE, tol=0.03, d_carrier_over_rider=1.0)
    env_big = sum_rule_envelope(CLOSURE, tol=0.03, d_carrier_over_rider=10.0)
    assert env_big["rows"][-1]["r_max_sum_rule"] > env_eq["rows"][-1]["r_max_sum_rule"]


def test_benchmark_two_band_closure_respects_the_declared_threshold(case):
    """Criterion (b) on the two declared *separated* points (cheap subset)."""
    bench = bimodal_benchmark(case, levels=["medium_gamma3p3"], n_ratios=(10.0,),
                              seeds=(0,), closure=CLOSURE, duration=1200.0)
    row = bench["rows"][0]
    assert abs(row["err_two_band_pct"]) <= 10.0
    # single-peak Dirlik is the known loser in this regime -> pinned as a finding
    assert row["err_dirlik_pct"] < -5.0


def test_commutation_streaming_is_bitwise(case):
    out = commutation_check(case, n_ratio=10.0, seed=3, duration=240.0,
                            block_counts=(2, 7, 31))
    assert out["max_rel_dev"] <= 1e-9
    assert all(r["n_ranges_equal"] for r in out["rows"])


def test_regime_plane_declares_the_dimensionless_parameter(case):
    plane = regime_plane(case)
    T1 = 1.0 / case.f1_hz
    for row in plane["regimes"]:
        assert row["theta_transition_over_T1"] == pytest.approx(row["tau_transition_s"] / T1)
    assert plane["excluded"]  # the resonance exclusion is declared, not implicit


def test_whipping_regime_boundary_separates_smooth_from_impulsive(case):
    """The two-parameter regime definition must place the two ends correctly."""
    plane = regime_plane(case)
    keep = {"harmonic_0p05Hz", "harmonic_0p2Hz", "dem_crush"}
    plane["regimes"] = [r for r in plane["regimes"] if r["name"] in keep]
    scan = whipping_regime_scan(case, plane=plane)
    rows = {r["regime"]: r for r in scan["rows"]}
    smooth = rows["harmonic_0p05Hz"]
    impulsive = rows["dem_crush"]
    assert smooth["force_jump_fraction"] <= JUMP_FRACTION_MAX
    assert smooth["theta_transition_over_T1"] >= THETA_MIN
    assert smooth["quasi_static"] is True
    assert smooth["share_peak"] < 0.08
    assert impulsive["force_jump_fraction"] > JUMP_FRACTION_MAX
    assert impulsive["quasi_static"] is False
    assert impulsive["share_peak"] > 0.08
    cons = regime_consistency(scan)
    assert cons["c1_quasi_static_share_and_peak_caliber"] is True
    assert cons["c3_damage_caliber_identity_ok"] is True


def test_study_quick_mode_returns_every_criterion_and_the_verdicts():
    res = run_w5n09_study(quick=True)
    for key in ("criterion_a", "criterion_b", "criterion_c", "criterion_d",
                "criterion_e", "calibration", "envelope", "criteria", "environment"):
        assert key in res
    crit = res["criteria"]
    assert crit["e_negative_control"]["pass"] is True
    assert crit["d_commutation"]["pass"] is True
    assert crit["c_whipping_caliber"]["pass"] is True
    # the sum rule must be rejected (the node's headline adjudication)
    assert crit["a_sum_rule_rejected"]["verdict"] == "sum rule rejected"
    assert res["criterion_a"]["separated_equal_damage_inside_band"] == 0


def test_summarise_criteria_is_consistent_with_the_sections():
    res = run_w5n09_study(quick=True)
    crit = summarise_criteria(res)
    assert crit["b_spectral_methods"]["pass"] == res["criterion_b"]["pass_two_band"]
    assert crit["a_sum_rule_rejected"]["pass_as_law"] == res["criterion_a"]["pass_two_band_law"]
    assert crit["e_negative_control"]["worst_rel_err_pct"] == \
        res["criterion_e"]["worst_rel_err_pct"]