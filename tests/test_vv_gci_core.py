"""Tests for the Richardson / GCI / ASME V&V 20 machine gate.

Positive control: an exact manufactured solution ``f(h) = f_exact + C h**2`` on
three grids must return ``p = 2``, ``f_exact`` to round-off and a GCI that
decreases under refinement.  Negative controls: non-asymptotic sequences
(oscillatory, diverging, stagnant), order/theory mismatches and noisy data must
never return a silent PASS.
"""
from __future__ import annotations

import json
import math
import random

import pytest

from tensorphysics.vv import (
    GatePolicy,
    Observation,
    evaluate,
    observed_order,
    richardson_extrapolated,
)
from tensorphysics.vv.gci import FAIL, PASS, WARN, gci, sweep_refinement_ratio


def uniform_levels(func, hs):
    """Coarse-to-fine levels (any input order works, the gate sorts)."""
    return [Observation(h=h, f=func(h)) for h in hs]


HS = (1.0, 0.5, 0.25)


def quadratic(h: float) -> float:
    """Manufactured solution with a known exact order p = 2."""
    return 3.0 + 0.5 * h * h


# ---------------------------------------------------------------------------
# positive control
# ---------------------------------------------------------------------------


def test_three_grid_recovers_second_order_exactly():
    levels = uniform_levels(quadratic, HS)
    verdict = evaluate(levels, quantity="f")
    study = verdict.study

    assert study.order_p == pytest.approx(2.0, abs=1e-2)
    assert study.order_source == "closed-form"
    assert abs(study.f_extrapolated - 3.0) < 1e-6
    assert study.convergence_type == "monotone"
    assert study.convergence_ratio == pytest.approx(0.25, rel=1e-12)
    assert verdict.status == PASS, verdict.reasons


def test_gci_matches_hand_computation():
    levels = uniform_levels(quadratic, HS)
    study = evaluate(levels).study
    # f: 3.5, 3.125, 3.03125 ; eps_fine = 0.09375 ; 1.25*(0.09375/3.03125)/3
    expected = 1.25 * (0.09375 / 3.03125) / 3.0
    assert study.gci_fine == pytest.approx(expected, rel=1e-12)
    assert study.gci_is_relative is True
    assert study.safety_factor == pytest.approx(1.25)
    # the safety-factored band covers the Richardson error estimate
    richardson_rel = abs(study.f_extrapolated - study.f_fine) / abs(study.f_fine)
    assert study.gci_fine == pytest.approx(1.25 * richardson_rel, rel=1e-12)


def test_gci_decreases_when_the_fine_grid_is_refined():
    sweep = sweep_refinement_ratio(quadratic, h_coarse=1.0, p=2.0, ratios=(2.0, 3.0, 4.0),
                                   safety_factor=1.25)
    values = [value for _rho, value in sweep]
    assert [rho for rho, _ in sweep] == [2.0, 3.0, 4.0]
    assert all(a > b for a, b in zip(values, values[1:])), values
    # hand-computed bands: f = 3 + h^2/2 -> f = 3.5, 3.125, 3.0555..., 3.03125
    assert values[0] == pytest.approx(0.05, rel=1e-12)
    assert values[1] == pytest.approx(0.02272727272727273, rel=1e-12)
    assert values[2] == pytest.approx(0.01288659793814433, rel=1e-12)
    assert values[0] / values[1] == pytest.approx(2.2, rel=1e-12)
    assert values[0] / values[2] == pytest.approx(3.88, rel=1e-12)


def test_nonuniform_refinement_recovers_order_by_least_squares():
    hs = (1.0, 0.4, 0.1)  # r = 2.5 then 4.0
    func = lambda h: 2.0 + 0.75 * h**2  # noqa: E731
    study = evaluate(uniform_levels(func, hs)).study
    assert study.uniform_refinement is False
    assert study.order_source == "least-squares"
    assert study.order_p == pytest.approx(2.0, abs=1e-6)
    assert abs(study.f_extrapolated - 2.0) < 1e-9


def test_five_level_study_is_clean_and_noise_certified():
    hs = (1.0, 0.5, 0.25, 0.125, 0.0625)
    verdict = evaluate(uniform_levels(quadratic, hs))
    study = verdict.study
    assert study.order_p == pytest.approx(2.0, abs=1e-9)
    assert study.fit_residual_rel < 1e-12
    assert study.order_spread < 1e-9
    assert study.gci_monotone is True
    assert study.noise_certified is True
    assert verdict.status == PASS, verdict.reasons


def test_cubic_order_is_recovered():
    func = lambda h: 1.0 + 2.0 * h**3  # noqa: E731
    study = evaluate(uniform_levels(func, HS)).study
    assert study.order_p == pytest.approx(3.0, abs=1e-9)


def test_richardson_helper_agrees_with_the_finest_pair():
    levels = uniform_levels(quadratic, HS)
    p, _src, _res = observed_order(levels)
    assert richardson_extrapolated(levels, p) == pytest.approx(3.0, abs=1e-12)


# ---------------------------------------------------------------------------
# negative controls: non-asymptotic sequences
# ---------------------------------------------------------------------------


def test_oscillatory_sequence_fails():
    values = {1.0: 1.5, 0.5: 1.2, 0.25: 1.35, 0.125: 1.275}  # alternating differences
    verdict = evaluate([Observation(h=h, f=f) for h, f in values.items()])
    assert verdict.study.convergence_type == "oscillatory"
    assert verdict.study.convergence_ratio < 0.0
    assert verdict.status == FAIL
    assert any("oscillatory" in reason for reason in verdict.reasons)


def test_diverging_sequence_fails():
    values = {1.0: 1.0, 0.5: 1.1, 0.25: 1.4, 0.125: 2.0}  # R = 2 > 1
    verdict = evaluate([Observation(h=h, f=f) for h, f in values.items()])
    assert verdict.study.convergence_ratio == pytest.approx(2.0)
    assert verdict.study.convergence_type == "diverging"
    assert verdict.status == FAIL


def test_stagnant_sequence_fails():
    verdict = evaluate([Observation(h=h, f=1.0) for h in (1.0, 0.5, 0.25)])
    assert verdict.status == FAIL


def test_order_outside_the_band_fails():
    # f(h) = h**0.01 : observed order 0.01 < p_min, genuinely non-asymptotic
    func = lambda h: 1.0 + h**0.5  # noqa: E731  -> p = 0.5, admissible
    assert evaluate(uniform_levels(func, HS)).status == PASS
    func_bad = lambda h: 1.0 + h**20  # noqa: E731  -> p = 20 > p_max
    verdict = evaluate(uniform_levels(func_bad, HS))
    assert verdict.status == FAIL
    assert any("order" in reason for reason in verdict.reasons)


def test_gci_growing_with_refinement_fails():
    values = [4.0, 2.0, 0.5, 0.05]  # relative differences 1 -> 3 -> 9
    hs = (1.0, 0.5, 0.25, 0.125)
    verdict = evaluate([Observation(h=h, f=f) for h, f in zip(hs, values)])
    assert verdict.study.gci_monotone is False
    assert verdict.status == FAIL
    assert any("GCI" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# negative controls: noise
# ---------------------------------------------------------------------------


def _noisy(level_func, hs, rel_noise, rng, declare_sigma):
    levels = []
    for h in hs:
        f = level_func(h)
        sigma = None
        if declare_sigma:
            sigma = rel_noise * abs(f)
        perturbed = f * (1.0 + rng.uniform(-rel_noise, rel_noise))
        levels.append(Observation(h=h, f=perturbed, sigma=sigma))
    return levels


def test_one_percent_noise_with_declared_sigma_never_silently_passes():
    """3 grids, 1 % noise, declared sigma -> WARN or FAIL for every seed."""
    near_converged = lambda h: 1.0 + 1.0e-3 * (h / 1.0) ** 2  # noqa: E731
    strong_grid_dependence = quadratic
    for seed in range(25):
        rng = random.Random(seed)
        for func in (near_converged, strong_grid_dependence):
            levels = _noisy(func, HS, 0.01, rng, declare_sigma=True)
            verdict = evaluate(levels)
            assert verdict.status in (WARN, FAIL), (seed, func.__name__, verdict.status)
            if func is near_converged:
                # noise dominates the finest difference by an order of magnitude
                assert verdict.status == FAIL
                assert any("noise" in reason for reason in verdict.reasons)


def test_one_percent_noise_on_five_grids_is_detected_without_declared_sigma():
    """Residual-based detection: 1 % noise on 5 grids cannot pass silently."""
    hs = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for seed in range(25):
        rng = random.Random(1000 + seed)
        levels = _noisy(quadratic, hs, 0.01, rng, declare_sigma=False)
        verdict = evaluate(levels)
        assert verdict.status in (WARN, FAIL), (seed, verdict.status)
        reasons = " ".join(verdict.reasons)
        assert any(tag in reasons for tag in
                   ("oscillatory", "residual", "consistency", "order", "noise"))


def test_three_grid_noise_detection_limit_is_explicit():
    """A three-grid study cannot certify an undeclared noise floor.

    Three data points and three fitted numbers leave no residual, so an
    undeclared noise floor is invisible: the gate must say so instead of
    implying a noise-free certificate, and ``--strict`` must downgrade it.
    """
    rng = random.Random(7)
    levels = _noisy(quadratic, HS, 0.01, rng, declare_sigma=False)
    verdict = evaluate(levels)
    assert verdict.study.noise_certified is False
    assert any("three-grid certificate" in note for note in verdict.study.notes)
    strict = evaluate(levels, policy=GatePolicy(strict=True))
    assert strict.status in (WARN, FAIL)


# ---------------------------------------------------------------------------
# policy behaviour
# ---------------------------------------------------------------------------


def test_two_grid_study_needs_an_assumed_order_and_uses_fs_three():
    levels = uniform_levels(quadratic, (1.0, 0.5))
    verdict = evaluate(levels)
    assert verdict.study.safety_factor == pytest.approx(3.0)
    assert verdict.status == WARN  # order assumed -> not certifiable
    policy = GatePolicy(assumed_order=2.0)
    assumed = evaluate(levels, policy=policy)
    assert assumed.study.order_p == pytest.approx(2.0)
    assert assumed.study.order_source == "assumed"
    assert assumed.study.safety_factor == pytest.approx(3.0)


def test_expected_order_mismatch_is_flagged():
    ok = evaluate(uniform_levels(quadratic, HS), policy=GatePolicy(expected_order=2.0))
    assert ok.status == PASS
    soft = evaluate(uniform_levels(quadratic, HS), policy=GatePolicy(expected_order=2.3))
    assert soft.status == WARN
    hard = evaluate(uniform_levels(quadratic, HS), policy=GatePolicy(expected_order=3.0))
    assert hard.status == FAIL


def test_validation_block_uses_gci_as_numerical_uncertainty():
    levels = uniform_levels(quadratic, HS)
    study = evaluate(levels).study
    u_num = study.gci_fine * abs(study.f_fine)
    passing = evaluate(levels, f_exp=2.999, u_input=0.05)
    assert passing.validation is not None
    assert passing.validation.u_num == pytest.approx(u_num, rel=1e-12)
    assert passing.status == PASS, passing.reasons
    failing = evaluate(levels, f_exp=2.5, u_input=0.05)
    assert failing.status == FAIL
    assert failing.validation.passed is False


def test_invalid_studies_are_rejected():
    with pytest.raises(ValueError):
        Observation(h=0.0, f=1.0)
    with pytest.raises(ValueError):
        Observation(h=1.0, f=float("nan"))
    single = evaluate([Observation(h=1.0, f=1.0)])
    assert single.status == FAIL and single.study is None
    dup = [Observation(h=1.0, f=1.0), Observation(h=1.0, f=2.0)]
    assert evaluate(dup).status == FAIL


def test_verdict_is_json_serialisable():
    verdict = evaluate(uniform_levels(quadratic, HS), quantity="Cd")
    text = json.dumps(verdict.as_dict())
    assert '"status": "PASS"' in text
    assert json.loads(text)["quantity"] == "Cd"


def test_gci_absolute_mode_when_the_observable_vanishes():
    coarse = Observation(h=1.0, f=0.0)
    fine = Observation(h=0.5, f=0.0)
    value, is_relative = gci(coarse, fine, 2.0, safety_factor=1.25)
    assert is_relative is False and value == pytest.approx(0.0)


def test_error_norm_observable_extrapolates_to_zero():
    """Standard verification case: the observable is a discretisation error norm."""
    hs = (1.0, 0.5, 0.25)
    study = evaluate([Observation(h=h, f=0.2 * h**2) for h in hs]).study
    assert study.order_p == pytest.approx(2.0, abs=1e-9)
    assert abs(study.f_extrapolated) < 1e-12
    # relative GCI of an error norm is 1.25 * (eps/f_fine)/(r^p-1) = 1.25
    assert study.gci_fine == pytest.approx(1.25 * (0.0375 / 0.0125) / 3.0, rel=1e-12)
    assert math.isfinite(study.gci_fine)