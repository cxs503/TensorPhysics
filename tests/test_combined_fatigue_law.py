"""P82W2-N08 -- ice + wave combined fatigue: combination law, closures, whipping.

Acceptance criteria of the campaign node are encoded as tests, together with the
analytic anchors they rest on.  The numbers quoted in the docstrings are the ones
produced by ``scripts/run_p82w2_n08_combination_law.py`` on the full records
(longer than the test records used here for speed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.fatigue.combined import combination as CB
from tensorphysics.fatigue.combined.ice_anchor import (
    ANCHOR_REFERENCE_RESULT,
    FreeFreeBeamModes,
    IceContactParams,
    generate_ice_force,
    modal_stress_history,
    run_anchor_case,
    whipping_share,
)
from tensorphysics.fatigue.combined.processes import (
    TwoBandCase,
    band_psd,
    rainflow_damage_rate,
    realised_process,
    spectral_damage_rates,
)
from tensorphysics.fatigue.combined.sn import (
    BASELINE_SN,
    CP_LIFE_FACTOR,
    SNCurveF3,
    SNParams,
    goodman_effective_range,
)
from tensorphysics.fatigue.rainflow import rainflow_counting_stack


# ---------------------------------------------------------------------------
# (0) S-N curve anchors
# ---------------------------------------------------------------------------
class TestSNCurve:
    def test_cp_life_is_2_5_times_the_free_corrosion_life(self):
        """DNV-RP-C203: seawater+CP = the in-air curve shifted by log10(2.5).

        The shift makes the CP life 2.5x longer than the *free-corrosion* curve
        and 2.5x shorter than the in-air curve (the in-air curve is the upper
        bound); the shift applies to both slopes.
        """
        air = SNCurveF3(SNParams(log_a1=11.510, log_a2=14.516,
                                 thickness_rule="dnv"))
        cp = SNCurveF3(SNParams(thickness_rule="dnv"))
        for ds in (5.0, 15.0, 60.0, 200.0):
            assert cp.life(ds) * CP_LIFE_FACTOR == pytest.approx(air.life(ds), rel=1e-12)

    def test_knee_is_continuous_at_31_86_MPa(self):
        curve = SNCurveF3()
        p = curve.params
        n_knee = p.n_knee
        # N at the knee from both branches
        n1 = 10.0 ** (p.log_a1 - p.m1 * math.log10(p.range_knee))
        n2 = 10.0 ** (p.log_a2 - p.m2 * math.log10(p.range_knee))
        assert n1 == pytest.approx(n_knee, rel=1e-12)
        # log a2 is quoted to 3 decimals, so continuity holds to ~1e-3
        assert n2 == pytest.approx(n_knee, rel=2e-3)
        # 1e7 cycles in air, /2.5 with cathodic protection
        assert n_knee == pytest.approx(1.0e7 / CP_LIFE_FACTOR, rel=1e-3)

    def test_legacy_knee_rule_matches_reference_implementation(self):
        legacy = SNCurveF3(SNParams(knee_rule="legacy_m5528"))
        assert legacy.params.range_knee_effective == pytest.approx(23.479, abs=1e-3)
        standard = SNCurveF3()
        # legacy knee -> shorter life (more damage) in the 23.5...31.9 MPa band
        assert legacy.life(30.0, 28.0) < standard.life(30.0, 28.0)

    def test_thickness_effect_direction(self):
        """DNV: thicker plate -> shorter life; the m5528 legacy rule is reversed."""
        dnv = SNCurveF3(SNParams(thickness_rule="dnv"))
        assert dnv.life(50.0, 25.0) > dnv.life(50.0, 28.0) > dnv.life(50.0, 40.0)
        legacy = SNCurveF3()
        assert legacy.life(50.0, 25.0) < legacy.life(50.0, 28.0) < legacy.life(50.0, 40.0)
        # the legacy direction is non-conservative by (t/25)^(0.2 m) on damage
        ratio = legacy.life(50.0, 40.0) / dnv.life(50.0, 40.0)
        assert ratio == pytest.approx(1.0986 ** 6, rel=1e-3)
        assert BASELINE_SN.thickness_factor(25.0) == 1.0

    def test_goodman_reduces_life_for_tension_mean(self):
        ramp = goodman_effective_range(100.0, 200.0, 490.0)
        assert ramp > 100.0
        assert goodman_effective_range(100.0, -100.0, 490.0) < 100.0
        assert goodman_effective_range(100.0, 0.0, 490.0) == 100.0
        assert math.isinf(goodman_effective_range(100.0, 490.0, 490.0))


# ---------------------------------------------------------------------------
# (e) negative control: waves off -> m5528 reference
# ---------------------------------------------------------------------------
class TestNegativeControl:
    def test_anchor_reproduces_m5528(self):
        res = run_anchor_case()
        ref = ANCHOR_REFERENCE_RESULT
        assert res["fatigue"]["damage_annual_goodman"] == pytest.approx(
            ref["damage_annual_goodman"], rel=2e-2)
        assert res["fatigue"]["life_years"] == pytest.approx(ref["life_years"], rel=2e-2)
        # the port is bit-exact, not merely within 2 %
        assert res["fatigue"]["damage_annual_goodman"] == pytest.approx(
            ref["damage_annual_goodman"], rel=1e-9)

    def test_stress_and_share_reproduced(self):
        res = run_anchor_case()
        assert res["whipping"]["share_peak"] == pytest.approx(
            ANCHOR_REFERENCE_RESULT["whipping_share_peak"], rel=1e-9)
        assert res["stress"]["nominal_rms_MPa"] == pytest.approx(
            ANCHOR_REFERENCE_RESULT["nominal_rms_MPa"], rel=1e-9)

    def test_first_mode_and_modal_mass(self):
        modes = FreeFreeBeamModes(250.0, 5.0e13, 5.2e5, n_modes=3, zeta=0.015)
        # f1 = beta1^2 sqrt(EI/m') / 2pi for beta1 L = 4.7300407
        f1 = (4.730040744862704 / 250.0) ** 2 * math.sqrt(5.0e13 / 5.2e5) / (2 * math.pi)
        assert modes.freq[0] == pytest.approx(f1, rel=1e-12)
        assert modes.freq[1] / modes.freq[0] == pytest.approx(
            (7.853204624095838 / 4.730040744862704) ** 2, rel=1e-12)
        assert np.all(modes.modal_mass > 0.0)


# ---------------------------------------------------------------------------
# (a) designed experiment: linearity of the combination law
# ---------------------------------------------------------------------------
class TestCombinationLaw:
    def test_fused_limit_at_equal_frequency(self):
        """Overlapping bands fuse: kappa -> (1+rho^2)^(m/2)/(1+rho^m)."""
        case = TwoBandCase(f_ice=0.5, f_wave=0.5, sigma_ice=5.0, sigma_wave=5.0,
                           bw_ice=0.02, bw_wave=0.02, duration=3600.0, fs=20.0, seed=11)
        out = CB._one_seed(case)
        assert out["ratio_linear"] == pytest.approx(2.0 ** 0.5, rel=0.12)

    def test_linear_law_fails_at_one_decade_separation(self):
        """Finding: at f_ice/f_w=10 with equal RMS the ratio is ~1.21 (>1.03)."""
        case = TwoBandCase(f_ice=0.5, f_wave=0.05, sigma_ice=5.0, sigma_wave=5.0,
                           duration=3600.0, fs=20.0, seed=11)
        out = CB._one_seed(case)
        assert out["ratio_linear"] > 1.10      # unsafe (>3 %) and well above noise
        assert out["ratio_linear"] < 1.35

    def test_damage_dominated_fast_band_is_almost_additive(self):
        """Additivity is recovered when one band dominates the damage."""
        case = TwoBandCase(f_ice=0.5, f_wave=0.005, sigma_ice=10.0, sigma_wave=5.0,
                           duration=3600.0, fs=20.0, seed=11)
        out = CB._one_seed(case)
        assert out["ratio_linear"] < 1.05

    def test_analytic_bounds(self):
        b = CB.analytic_bounds(5.0, 5.0)
        assert b["ratio_separated"] == 1.0
        assert b["ratio_coincident"] == pytest.approx(2.0 ** 0.5, rel=1e-12)
        b2 = CB.analytic_bounds(2.5, 5.0)
        assert b2["ratio_coincident"] == pytest.approx(
            (2.5 ** 2 + 25.0) ** 1.5 / (2.5 ** 3 + 125.0), rel=1e-12)

    def test_linearity_scan_rows_are_self_consistent(self):
        res = CB.linearity_scan(ratios=(1.0, 10.0), splits=(1.0,), f_fast=0.5,
                                duration=600.0, fs=20.0, seeds=(11,))
        rows = res["rows"]
        assert len(rows) == 2
        for row in rows:
            assert row["D_comb_mean"] > 0.0
            assert row["ratio_mean"] == pytest.approx(
                row["D_comb_mean"] / row["D_ice_plus_wave"], rel=1e-9)
            assert row["f_ratio"] == pytest.approx(
                row["f_ice_Hz"] / row["f_wave_Hz"], rel=1e-12)

    def test_admissible_region_reports_boundary(self):
        res = CB.linearity_scan(ratios=(1.0, 10.0, 100.0), splits=(1.0,), f_fast=0.5,
                                duration=600.0, fs=20.0, seeds=(11,))
        region = CB.admissible_region(res["rows"])
        assert region["tolerance"] == CB.LINEARITY_TOL
        assert len(region["rows"]) == 1
        assert "kappa_by_ratio" in region["rows"][0]


# ---------------------------------------------------------------------------
# (b) spectral closures vs rainflow truth
# ---------------------------------------------------------------------------
class TestSpectralClosures:
    def test_narrowband_closures_match_rainflow(self):
        psd = band_psd(0.5, 5.0, 0.02)
        real = realised_process(psd, 3600 * 20, 20.0, seed=5, f_centre=0.5, sigma_f=0.01)
        truth = rainflow_damage_rate(real["history"], 1.0 / 20.0)["damage_rate"]
        spec = spectral_damage_rates(real["moments"])
        assert spec["alpha2"] > 0.99
        assert abs(spec["dirlik"] / truth - 1.0) < 0.05
        assert abs(spec["tovo_benasciutti"] / truth - 1.0) < 0.05
        assert abs(spec["rayleigh"] / truth - 1.0) < 0.05

    def test_three_widths_within_ten_percent(self):
        res = CB.spectral_width_study(duration=1800.0, fs=20.0, seeds=(5, 17, 29))
        alphas = [r["alpha2"] for r in res["rows"]]
        assert min(alphas) < 0.7 < max(alphas)          # genuinely different widths
        for row in res["rows"]:
            assert abs(row["err_dirlik_pct"]) < 10.0
            assert abs(row["err_tb_pct"]) < 10.0

    def test_rayleigh_is_the_narrowband_limit(self):
        psd = band_psd(1.0, 4.0, 0.005)
        real = realised_process(psd, 3600 * 40, 40.0, seed=9, f_centre=1.0, sigma_f=0.005)
        truth = rainflow_damage_rate(real["history"], 1.0 / 40.0)["damage_rate"]
        spec = spectral_damage_rates(real["moments"])
        assert spec["alpha2"] > 0.999
        assert abs(spec["rayleigh"] / truth - 1.0) < 0.03


# ---------------------------------------------------------------------------
# (c) whipping accounting
# ---------------------------------------------------------------------------
class TestWhippingAccounting:
    def test_quasi_static_operating_point_below_8pct(self):
        res = CB.whipping_accounting(
            operating_points={"quasi_static": CB.WHIPPING_OPERATING_POINTS["quasi_static"]},
            duration=900.0, dt=4.0e-3,
            synthetic_seeds=(7,), synthetic_duration=1800.0)
        row = res["rows"][0]
        assert row["share_peak"] < 0.08
        assert row["slow_varying_lt_8pct"] is True

    def test_share_law_only_covers_the_small_share_corner(self):
        """The additive-cycle share law is valid only for small shares."""
        res = CB._synthetic_share_law(seeds=(7, 19), duration=3600.0, fs=20.0,
                                      ratios=(1.0, 2.0), share_values=(0.05,))
        assert res["deviation_pp"] <= CB.WHIPPING_TOL_PP
        wide = CB._synthetic_share_law(seeds=(7,), duration=3600.0, fs=20.0,
                                       ratios=(10.0,), share_values=(0.5,))
        assert wide["deviation_pp"] > 50.0

    def test_absorption_mechanism_brackets_the_measurement(self):
        """The measured amplification lies between the cycle-addition lower bound
        and the full peak-absorption upper bound, with an efficiency that grows
        with the band separation."""
        res = CB._synthetic_share_law(seeds=(7, 19), duration=3600.0, fs=20.0,
                                      ratios=(1.0, 10.0), share_values=(0.3,))
        eff = res["absorption_efficiency_by_ratio"]
        assert eff[1.0] < 0.15 < 0.35 < eff[10.0] < 0.65
        assert res["all_within_bounds"] is True
        assert res["deviation_pp"] > CB.WHIPPING_TOL_PP

    def test_share_law_fails_on_physical_ice_chain(self):
        """Finding: the RMS share does not predict the damage amplification."""
        res = CB.whipping_accounting(
            operating_points={"lock_in": dict(v_indent=0.12)},
            duration=300.0, dt=2.0e-3,
            synthetic_seeds=(7,), synthetic_duration=1800.0)
        row = res["rows"][0]
        assert row["deviation_pp"] > 2.0 * 10
        assert row["damage_share_of_whipping"] > 0.8

    def test_amplitude_and_variance_conventions_are_totally_different(self):
        conv = CB._convention_amplifications(0.759, 0.351, 3.0)
        assert conv["amplitude_additive_pct"] > 200.0
        assert conv["variance_additive_pct"] > 200.0
        conv_small = CB._convention_amplifications(0.08, 0.05, 3.0)
        assert conv_small["amplitude_additive_pct"] < 20.0


# ---------------------------------------------------------------------------
# (d) commutativity / order invariance
# ---------------------------------------------------------------------------
class TestCommutativity:
    def test_periodic_record_rotation_and_reversal(self):
        fs, T = 20.0, 400.0
        t = np.arange(int(T * fs)) / fs
        x = np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 0.05 * t)
        res = CB.rainflow_commutativity(x, 1.0 / fs, f_period_s=20.0)
        assert res["pass_1e9"]["segment_accumulation"]
        assert res["pass_1e9"]["time_reversal"]
        assert res["pass_1e9"]["cyclic_rotation"]

    def test_stationary_record_reversal_and_accumulation(self):
        x = np.random.default_rng(3).normal(size=20000)
        res = CB.rainflow_commutativity(x, 0.02)
        assert res["pass_1e9"]["segment_accumulation"]
        assert res["pass_1e9"]["time_reversal"]
        # open-segment permutation is NOT an invariance: report, do not assert 1e-9
        assert res["raw_open_segment_permutation_max"] > 1e-9

    def test_rainflow_counter_is_the_validated_astm_one(self):
        # hand-worked ASTM E1049-85 four-point example
        res = rainflow_counting_stack(np.array([-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0]))
        pairs = sorted(zip(np.round(res.ranges, 6), np.round(res.means, 6), res.counts))
        assert pairs == [(3.0, -0.5, 0.5), (4.0, -1.0, 0.5), (4.0, 1.0, 1.0),
                         (8.0, 0.0, 0.5), (8.0, 1.0, 0.5), (9.0, 0.5, 0.5)]


# ---------------------------------------------------------------------------
# mechanism: slow cycles absorb the fast peak-to-trough amplitude
# ---------------------------------------------------------------------------
class TestMechanism:
    def test_two_tone_range_absorption(self):
        fs, T = 20.0, 2000.0
        t = np.arange(int(T * fs)) / fs
        A, ff, fw = 5.0, 0.5, 0.05
        x = A * np.sin(2 * np.pi * ff * t) + A * np.sin(2 * np.pi * fw * t)
        rf = rainflow_counting_stack(x)
        slow = rf.counts[rf.ranges > 3 * A]
        # the slow cycles exist and their range has absorbed ~2A of fast amplitude
        assert slow.sum() > 0.5 * fw * T
        slow_ranges = rf.ranges[rf.ranges > 3 * A]
        assert np.all(slow_ranges > 2 * A)          # pure slow range would be 2A
        # hence the combined damage exceeds the sum of the separate damages
        d_f = rainflow_damage_rate(A * np.sin(2 * np.pi * ff * t), 1 / fs)["damage_rate"]
        d_w = rainflow_damage_rate(A * np.sin(2 * np.pi * fw * t), 1 / fs)["damage_rate"]
        d_c = rainflow_damage_rate(x, 1 / fs)["damage_rate"]
        assert d_c / (d_f + d_w) > 1.1

    def test_variance_of_sum_is_sum_of_variances(self):
        case = TwoBandCase(f_ice=0.5, f_wave=0.05, sigma_ice=5.0, sigma_wave=3.0,
                           duration=1800.0, fs=20.0, seed=11)
        ice = realised_process(band_psd(case.f_ice, case.sigma_ice, 0.05), case.n_samples,
                               case.fs, seed=11, f_centre=0.5, sigma_f=0.025)
        wave = realised_process(band_psd(case.f_wave, case.sigma_wave, 0.05), case.n_samples,
                                case.fs, seed=7712, f_centre=0.05, sigma_f=0.0025)
        combo = np.asarray(ice["history"]) + np.asarray(wave["history"])
        assert np.var(combo) == pytest.approx(
            np.var(ice["history"]) + np.var(wave["history"]), rel=2e-3)


# ---------------------------------------------------------------------------
# physical case
# ---------------------------------------------------------------------------
class TestPhysicalCase:
    def test_dem_ice_chain_load_statistics(self):
        params = IceContactParams.calibrated()
        t, F = generate_ice_force(params, duration=120.0, dt=2.0e-3)
        assert F.size == t.size
        assert F.max() > 10.0e6            # MN-scale ice force
        stats = __import__(
            "tensorphysics.fatigue.combined.ice_anchor", fromlist=["load_statistics"]
        ).load_statistics(F, 2.0e-3)
        assert 0.2 < stats["mean_peak"] < 0.7

    def test_realistic_case_is_not_additive(self):
        res = CB.realistic_case(duration=900.0, dt=1.0e-2, wave_seeds=(101,))
        row = res["rows"][0]
        assert row["ratio_linear"] > 1.05
        assert row["ignoring_whipping_underprediction_factor"] > 10.0
        assert res["whipping"]["share_peak"] > 0.3

    def test_quasi_static_part_is_softer_than_total(self):
        modes = FreeFreeBeamModes(250.0, 5.0e13, 5.2e5, n_modes=5, zeta=0.015)
        _, F = generate_ice_force(IceContactParams.calibrated(), duration=120.0, dt=2.0e-3)
        resp = modal_stress_history(modes, F, 2.0e-3, 12.5, 75.0, 10.5, 2.06e11)
        ws = whipping_share(resp["sigma"][10000:], resp["sigma_qs"][10000:])
        assert 0.0 < ws["share_peak"] < 1.0
        assert ws["share_rms"] > ws["share_peak"]