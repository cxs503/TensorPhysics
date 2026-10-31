"""End-to-end tests of the unified ice-induced vibration analysis driver.

A reduced-cost analysis configuration is used so the suite stays fast; the
production reference run (docs/ICE_VIBRATION_LOCKIN.md, scenarios/
ice_vibration_lockin/run_analysis.py) uses 80 steady-state cycles and a 120 s
record.
"""

from __future__ import annotations

import json
import math

import pytest

from tensorphysics.ice_vibration.analysis import (
    DUFFING_RELATIVE_TOLERANCE,
    WHIPLASH_RELATIVE_TOLERANCE,
    IceCondition,
    IceVibrationAnalysis,
    build_reference_case,
    spectrum_peak_deviation,
)


@pytest.fixture(scope="module")
def case():
    return build_reference_case()


@pytest.fixture(scope="module")
def fast_analysis(case):
    return IceVibrationAnalysis(
        case=case,
        steady_cycles=60,
        samples_per_cycle=120,
        discard_fraction=0.7,
        record_s=40.0,
        samples_per_second=500.0,
        self_excited_duration_s=30.0,
    )


@pytest.fixture(scope="module")
def report(fast_analysis):
    return fast_analysis.run(include_time_series=False)


class TestReferenceCase:
    def test_reference_case_frequency_and_geometry(self, case):
        assert case.first_frequency_hz == pytest.approx(1.20, rel=0.01)
        assert case.modal.n_modes >= 2
        assert case.hull.verify()["consistent"] is True
        assert case.ice.crushing_frequency_hz == pytest.approx(1.25, rel=1e-12)

    def test_ice_condition_validation(self):
        with pytest.raises(ValueError):
            IceCondition(peak_ice_force_n=0.0)
        with pytest.raises(ValueError):
            IceCondition(drift_speed_m_s=-1.0)

    def test_spectrum_peak_deviation(self):
        assert spectrum_peak_deviation(1.243, 1.20) == pytest.approx(0.035833, rel=1e-4)
        with pytest.raises(ValueError):
            spectrum_peak_deviation(1.0, 0.0)


class TestCriteria:
    def test_all_criteria_pass_on_the_reference_case(self, report):
        criteria = report["criteria"]
        failed = [entry for entry in criteria["entries"] if not entry["passed"]]
        assert not failed, failed
        assert criteria["n_criteria"] >= 8
        assert criteria["all_passed"] is True

    def test_duffing_criterion_within_tolerance(self, report):
        duffing = report["duffing_lock_in"]
        assert duffing["max_relative_error"] < DUFFING_RELATIVE_TOLERANCE
        assert duffing["min_amplification_in_band"] > 3.0
        assert len(duffing["rows"]) == 9

    def test_whiplash_criterion_within_tolerance(self, report):
        assert report["whiplash"]["midship_moment_relative_error"] < WHIPLASH_RELATIVE_TOLERANCE
        assert report["whiplash"]["passed"] is True

    def test_lock_in_amplification_and_band(self, report):
        lock_in = report["lock_in"]
        assert lock_in["min_edge_amplification_time_domain"] > 3.0
        assert lock_in["relative_tolerance"] == pytest.approx(0.05)
        assert lock_in["band_hz"] == pytest.approx([1.14, 1.26], rel=0.01)
        assert lock_in["fraction_of_band_points_locked"] == pytest.approx(1.0)
        assert lock_in["lock_in_band_inside_amplification_band"] is True

    def test_out_of_band_response_is_not_amplified(self, report):
        rows = {row["frequency_ratio"]: row for row in report["forced_response_sweep"]["rows"]}
        assert rows[1.40]["amplification_time_domain"] < 1.5
        assert rows[1.40]["state"] == "FREE"

    def test_reported_json_is_serialisable(self, report):
        text = json.dumps(report)
        assert len(text) > 1000
        restored = json.loads(text)
        assert restored["criteria"]["n_passed"] == report["criteria"]["n_passed"]


class TestLoadCases:
    def test_quasi_periodic_crushing_locks_in(self, report):
        pulse = report["pulse_train"]
        assert pulse["load_is_narrowband"] is True
        assert pulse["lock_in_diagnosis"]["lock_in"] is True
        # the excitation sits at U / s and the response follows it
        assert pulse["load_spectrum_peak_hz"]["frequency_hz"] == pytest.approx(1.25, rel=0.02)
        assert pulse["response_peak_relative_deviation_from_excitation"] < 0.05
        # intermittent crushing feeds the resonance through all harmonics and the arrival
        # jitter, so the response cannot be smaller than the single-harmonic estimate
        assert pulse["amplification_vs_fundamental_static"] > 3.0
        assert pulse["amplification_vs_fundamental_static"] >= pulse["analytic_magnification_at_load_peak"]

    def test_poisson_impacts_are_broadband(self, report):
        control = report["poisson_control"]
        assert control["load_is_narrowband"] is False
        assert control["lock_in_diagnosis"]["state"] == "BROADBAND_RANDOM_RESPONSE"
        assert control["load_spectral_concentration"]["concentration"] < 0.30

    def test_self_excited_lock_in_frequency_and_regimes(self, report):
        sweep = report["self_excited"]
        assert sweep["n_locked_cases"] >= 4
        assert sweep["all_locked_within_frequency_tolerance"] is True
        regimes = {row["drift_speed_m_s"]: row["regime"] for row in sweep["rows"]}
        assert regimes[0.30] == "SELF_EXCITED_LIMIT_CYCLE"
        assert regimes[2.00] == "STABLE_DAMPED"
        fast = [row for row in sweep["rows"] if row["drift_speed_m_s"] >= 0.9]
        assert all(row["peak_amplitude_m"] < 5e-3 for row in fast)

    def test_self_excited_amplitude_is_physical(self, report):
        locked = [row for row in report["self_excited"]["rows"] if row["regime"] == "SELF_EXCITED_LIMIT_CYCLE"]
        for row in locked:
            assert 0.005 < row["peak_amplitude_m"] < 0.30

    def test_damping_sensitivity_monotonic_amplification(self, report):
        rows = report["damping_sensitivity"]["rows"]
        amplifications = [row["amplification_at_upper_band_edge"] for row in rows]
        assert amplifications == sorted(amplifications, reverse=True)
        assert amplifications[0] > 3.0

    def test_shock_spectrum_peak(self, report):
        spectrum = report["whiplash_shock_spectrum"]
        assert 1.5 < spectrum["max_whiplash_dynamic_amplification"] < 1.9
        assert 0.4 < spectrum["peak_row"]["pulse_ratio_Tp_f1"] < 1.0
        assert spectrum["max_rel_error_over_sweep"] < WHIPLASH_RELATIVE_TOLERANCE

    def test_duffing_fold_region_reported(self, report):
        hysteresis = report["duffing_hysteresis"]
        assert hysteresis["n_frequencies_with_three_roots"] > 0
        low, high = hysteresis["fold_band_ratio"]
        assert 0.8 < low <= high < 1.4