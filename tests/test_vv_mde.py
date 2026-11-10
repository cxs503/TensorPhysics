"""Tests for the MDE_90 detection-strength engine (:mod:`tensorphysics.vv.mde`).

Every Monte-Carlo number the engine reports is checked against a closed form or
a known limit, so the calibration ledger cannot drift into "looks plausible".
"""
from __future__ import annotations

import math
import random

import pytest

from tensorphysics.vv import mde


# ---------------------------------------------------------------------------
# distribution / smoothing helpers
# ---------------------------------------------------------------------------
def test_wilson_interval_matches_textbook_values():
    lo, hi = mde.wilson_interval(90, 100)
    assert 0.824 < lo < 0.829
    assert 0.943 < hi < 0.947
    lo0, hi0 = mde.wilson_interval(0, 100)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.04
    lo1, hi1 = mde.wilson_interval(100, 100)
    assert hi1 == 1.0 and 0.96 < lo1 < 1.0


def test_pava_is_monotone_and_preserves_weighted_mean():
    raw = [0.1, 0.4, 0.3, 0.2, 0.9, 0.85]
    weights = [3, 1, 1, 1, 2, 2]
    fitted = mde.pava(raw, weights)
    assert all(b >= a for a, b in zip(fitted, fitted[1:]))
    assert math.isclose(sum(w * f for w, f in zip(weights, fitted)),
                        sum(w * r for w, r in zip(weights, raw)), rel_tol=1e-12)


def test_norm_cdf_ppf_roundtrip():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975):
        assert math.isclose(mde.norm_cdf(mde.norm_ppf(p)), p, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# closed-form statistics
# ---------------------------------------------------------------------------
def test_point_detector_mde90_matches_closed_form():
    threshold, sigma = 0.03, 0.01
    expected = threshold + mde.Z90 * sigma
    got = mde.mde90_analytic("point", threshold, sigma=sigma)
    assert math.isclose(got, expected, rel_tol=1e-6)
    # detection probability at the analytic MDE_90 is 90 % by construction
    assert math.isclose(mde.power_point(got, threshold, sigma), 0.90, abs_tol=1e-6)


def test_mean_of_n_and_bound_statistics():
    # averaging n observations shrinks the noise by sqrt(n)
    got = mde.mde90_analytic("mean", 0.03, sigma=0.01, n_obs=4)
    assert math.isclose(got, 0.03 + mde.Z90 * 0.01 / 2.0, rel_tol=1e-6)
    # an uncertainty-inflated boundary (|E| + k*sigma <= tau) is less sensitive
    bound = mde.mde90_analytic("bound", 0.04, sigma=0.01, k=1.0)
    point = mde.mde90_analytic("point", 0.03, sigma=0.01)
    assert math.isclose(bound, point, rel_tol=1e-6)
    assert bound > 0.04


def test_deterministic_mde90_is_the_step_location():
    detector = lambda y: abs(y) > 0.03          # noqa: E731
    got = mde.mde90_analytic("point", 0.03, sigma=0.0)
    assert math.isclose(got, 0.03, rel_tol=1e-9)
    assert mde.mde90_bisect(detector, lo=1e-12, hi=1.0) == pytest.approx(0.03, rel=1e-6)
    # a detector that never fires has no MDE_90 rather than a fake one
    assert mde.mde90_bisect(lambda y: False, lo=1e-12, hi=1.0) is None


# ---------------------------------------------------------------------------
# Monte-Carlo agrees with theory (the ledger's own audit path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.01, 0.02])
def test_monte_carlo_curve_matches_closed_form(sigma):
    threshold = 0.03
    detector = lambda y: abs(y) > threshold     # noqa: E731
    ladder = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08]
    curve = mde.detection_curve(detector, ladder, sigma=sigma, n_trials=2000, seed=11)
    for delta, p, lo, hi in zip(curve.deltas, curve.p_hat, curve.p_lo, curve.p_hi):
        theory = mde.power_point(delta, threshold, sigma)
        assert lo - 1e-12 <= theory <= hi + 1e-12 or abs(theory - p) < 0.06
    got = mde.mde90_interpolate(curve)
    assert got == pytest.approx(mde.mde90_analytic("point", threshold, sigma=sigma),
                                rel=0.08)


def test_detection_curve_is_monotone_after_pava_and_seeded():
    detector = lambda y: abs(y) > 0.03          # noqa: E731
    deltas = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    a = mde.detection_curve(detector, deltas, sigma=0.02, n_trials=100, seed=5)
    b = mde.detection_curve(detector, deltas, sigma=0.02, n_trials=100, seed=5)
    assert a.hits == b.hits                     # reproducible
    assert all(x <= y for x, y in zip(a.p_smooth, a.p_smooth[1:]))
    assert a.p_smooth[-1] >= a.p_smooth[0]


def test_censored_curve_reports_none_not_a_clamped_number():
    curve = mde.detection_curve(lambda y: False, [0.1, 0.2], n_trials=50, seed=1)
    assert mde.mde90_interpolate(curve) is None


# ---------------------------------------------------------------------------
# logistic fit
# ---------------------------------------------------------------------------
def test_logistic_fit_recovers_a_known_mde90():
    threshold, sigma = 0.03, 0.01
    detector = lambda y: abs(y) > threshold     # noqa: E731
    deltas = [0.005 * i for i in range(1, 13)]
    curve = mde.detection_curve(detector, deltas, sigma=sigma, n_trials=1000, seed=3)
    fit = mde.fit_logistic(curve.deltas, curve.hits, curve.trials)
    truth = mde.mde90_analytic("point", threshold, sigma=sigma)
    assert fit.separation is False
    assert fit.mde90 == pytest.approx(truth, rel=0.10)
    assert fit.mde50 == pytest.approx(threshold, rel=0.10)
    assert fit.b1 > 0.0


def test_logistic_fit_declares_separation_on_a_step():
    curve = mde.detection_curve(lambda y: abs(y) > 0.03,
                                [0.01, 0.02, 0.05, 0.1], n_trials=100, seed=2)
    fit = mde.fit_logistic(curve.deltas, curve.hits, curve.trials)
    assert fit.mde90 is None
    assert fit.separation is True


# ---------------------------------------------------------------------------
# proportion (kill-rate) family
# ---------------------------------------------------------------------------
def test_proportion_sample_size_arithmetic():
    assert mde.proportion_samples_required(0.90, 0.05) == 29
    assert mde.proportion_k_required(22, 0.90, 0.05) is None      # vacuous
    assert mde.proportion_k_required(29, 0.90, 0.05) == 29
    assert mde.proportion_k_required(100, 0.90, 0.05) == 96
    assert mde.proportion_k_crit(22, 0.90, 0.05) == 16


def test_proportion_power_is_level_alpha_at_the_boundary():
    for n in (22, 29, 100, 400):
        assert mde.power_proportion(0.0, n, 0.90, 0.05) <= 0.05 + 1e-12
        assert mde.power_proportion(0.90, n, 0.90, 0.05) == pytest.approx(1.0)


def test_proportion_mde90_shrinks_with_sample_size():
    small = mde.mde90_analytic("proportion", 0.90, n_obs=22, p_min=0.90)
    medium = mde.mde90_analytic("proportion", 0.90, n_obs=100, p_min=0.90)
    large = mde.mde90_analytic("proportion", 0.90, n_obs=400, p_min=0.90)
    assert small > medium > large
    assert small > 0.2            # 22 mutants cannot see a 20 % degradation
    assert large < 0.06           # 400 mutants resolve a 5 % one


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------
def test_certificate_schema_and_verdicts():
    det = lambda y: abs(y) > 0.06               # noqa: E731
    cert = mde.certificate("synthetic-tight", det, deltas=[0.01, 0.02, 0.05, 0.1, 0.2],
                           statistic="point", threshold=0.06, n_trials=100, seed=7,
                           theory=0.06)
    for key in ("criterion", "mde90", "mde90_exact", "mde90_mc_ladder", "class",
                "catches_reference_bias", "power_curve", "logistic",
                "false_alarm_at_zero_bias", "valid_detector"):
        assert key in cert
    assert cert["mde90"] == pytest.approx(0.06, rel=1e-6)
    assert cert["catches_reference_bias"] is False        # 6 % > 5 % reference
    assert cert["class"] == "MARGINAL"
    assert cert["false_alarm_at_zero_bias"] is False

    loose = mde.certificate("synthetic-loose", lambda y: abs(y) > 0.02,
                            deltas=[0.01, 0.02, 0.05, 0.1, 0.2], statistic="point",
                            n_trials=100, seed=7)
    assert loose["catches_reference_bias"] is True
    assert loose["class"] == "STRONG"


def test_certificate_flags_a_detector_that_fires_on_clean_evidence():
    cert = mde.certificate("broken", lambda y: True, deltas=[0.01, 0.05, 0.1],
                           statistic="point", n_trials=50, seed=1)
    assert cert["false_alarm_at_zero_bias"] is True
    assert cert["valid_detector"] is False
    assert cert["class"] == "FALSE_ALARM"


def test_classify_mde_boundaries():
    assert mde.classify_mde(0.01) == "STRONG"
    assert mde.classify_mde(0.05) == "STRONG"
    assert mde.classify_mde(0.08) == "MARGINAL"
    assert mde.classify_mde(0.15) == "WEAK"
    assert mde.classify_mde(0.5) == "BLIND"
    assert mde.classify_mde(None) == "CENSORED"


# ---------------------------------------------------------------------------
# end-to-end: the driver consumes the shipped ledger
# ---------------------------------------------------------------------------
def test_driver_end_to_end_on_shipped_ledger(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mde_calibration_p493n04", root / "scripts/mde_calibration_p493n04.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ship = module.load_shipped_ledger()
    criteria = module.shipped_criteria(ship)
    assert len(criteria) == 17
    # the same criterion objects the shipped mutation audit governs
    assert {c.id for c in criteria} >= {"VVVAL-ratio", "P20H-H6", "VVGCI-residual"}

    criterion = next(c for c in criteria if c.id == "VVVAL-ratio")
    detector = module.gate_detector(ship, criterion)
    assert detector(0.0) is False               # clean evidence is accepted
    assert detector(0.10) is True               # a 10 % bias is caught
    cert = module.calibrate_continuous(ship, criterion, n_trials=20, seed=1)
    assert cert["mde90_exact"] == pytest.approx(0.0119, rel=1e-3)
    assert cert["mde90_theory"] == pytest.approx(0.0119, rel=1e-3)
    assert cert["class"] == "STRONG"

    # structural criteria are honestly reported as un-calibratable by bias
    structural = module.calibrate_structural(
        [c for c in criteria if not c.value_fields])
    assert all(row["class"] == "NOT_APPLICABLE" for row in structural)
    assert all(row["mde90"] is None for row in structural)
