"""Interpolation, error-bound and adaptive-clock tests for the cross-clock package."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.cross_clock import AdaptiveClock, ContractError
from tensorphysics.cross_clock.interpolate import (
    HoldRate,
    LinearRate,
    fine_rate_curvature,
    fine_rate_variation,
    interpolation_bounds,
    measured_error,
    reconstruct,
)
from tensorphysics.cross_clock.clock import ratio_scan
from tensorphysics.cross_clock.contract import IntervalSample


# --------------------------------------------------------------------------- #
# reconstructions
# --------------------------------------------------------------------------- #
def test_hold_reconstructs_the_window_mean_exactly():
    hold = HoldRate(2.5, 0.5)
    assert hold.value(0.1) == pytest.approx(2.5)
    assert hold.integral(0.0, 0.5) == pytest.approx(1.25)
    assert hold.order == 1
    with pytest.raises(ContractError):
        HoldRate(1.0, 0.0)


def test_linear_rate_extrapolates_causally_from_history():
    hist = [
        IntervalSample(0.0, 1.0, {"momentum": 1.0}),
        IntervalSample(1.0, 2.0, {"momentum": 3.0}),
    ]
    rec = reconstruct(hist, "momentum", "linear")
    assert isinstance(rec, LinearRate)
    assert rec.slope == pytest.approx(2.0)
    assert rec.value(2.5, t0=2.0) == pytest.approx(4.0)
    assert rec.order == 2
    # a single window cannot support an extrapolation, so it degrades to a hold
    assert isinstance(reconstruct(hist[:1], "momentum", "linear"), HoldRate)


# --------------------------------------------------------------------------- #
# computable bounds
# --------------------------------------------------------------------------- #
def test_fine_variation_and_curvature_of_a_linear_signal():
    t = np.linspace(0.0, 1.0, 101)
    r = 3.0 * t
    assert fine_rate_variation(t, r) == pytest.approx(3.0, rel=1e-9)
    assert fine_rate_curvature(t, r) == pytest.approx(0.0, abs=1e-8)
    r2 = t ** 2
    assert fine_rate_curvature(t, r2) == pytest.approx(2.0, rel=1e-6)


def test_fine_bound_is_sound_and_scales_with_the_window():
    dt = 2.0e-4
    t = np.arange(0.0, 2.0 + dt / 2, dt)
    r = np.sin(2.0 * np.pi * t)
    ratios = []
    for h in (0.02, 0.01, 0.005):
        bounds, trues = [], []
        for k in range(1, int(2.0 / h) - 1):
            t0 = k * h
            m = (t >= t0) & (t < t0 + h)
            b = interpolation_bounds(t[m], r[m], h, "hold", estimator="fine")
            e = measured_error(t, r, t0, h, "hold")
            bounds.append(b["integral_bound"])
            trues.append(e["integral_error"])
        bounds, trues = np.asarray(bounds), np.asarray(trues)
        assert np.all(bounds >= trues), "the bound must never be below the measured error"
        assert np.median(bounds / trues) < 10.0, "the bound must be tight enough to be usable"
        ratios.append(np.mean(bounds))
    # halving the window must reduce the bound by roughly 4 (integral of O(h) error)
    assert 3.0 < ratios[0] / ratios[1] < 5.0
    assert 3.0 < ratios[1] / ratios[2] < 5.0


def test_aggregate_bound_needs_only_two_window_means():
    """A consumer that sees nothing but aggregates can still bound the error."""
    b = interpolation_bounds(
        [0.0], [2.2], 0.5, "hold", estimator="aggregate", previous_mean=2.0
    )
    assert b["integral_bound"] == pytest.approx(0.2 * 0.5 / 2.0)
    with pytest.raises(ContractError):
        interpolation_bounds([0.0], [2.2], 0.5, "hold", estimator="aggregate")


def test_measured_error_reports_integral_and_pointwise():
    t = np.linspace(0.0, 1.0, 1001)
    r = np.cos(2.0 * np.pi * t)
    e = measured_error(t, r, 0.0, 0.25, "hold")
    assert e["pointwise_error"] > 0.0
    assert e["integral_error"] > 0.0
    assert e["pointwise_error"] >= e["integral_error"] / 0.25


# --------------------------------------------------------------------------- #
# adaptive clock
# --------------------------------------------------------------------------- #
def test_adaptive_clock_refines_on_violation_and_records_it():
    ctrl = AdaptiveClock(ratio=100, tolerance=1e-2, order=1)
    new = ctrl.update(bound=4e-2)          # four times the tolerance
    assert new < 100
    assert ctrl.history[-1]["action"] == "refine"
    assert ctrl.history[-1]["violated"] is True
    # repeated violation must keep refining, never stall above the minimum
    for _ in range(10):
        ctrl.update(bound=1e3)
    assert ctrl.ratio == ctrl.min_ratio


def test_adaptive_clock_does_not_churn_inside_the_hysteresis_band():
    ctrl = AdaptiveClock(ratio=8, tolerance=1e-2, order=1, hysteresis=1.5, growth=2.0)
    before = ctrl.ratio
    for _ in range(5):
        ctrl.update(bound=0.8e-2)          # between tolerance/1.5 and tolerance
    assert ctrl.ratio == before
    assert all(h["action"] == "hold" for h in ctrl.history)


def test_adaptive_clock_coarsens_when_the_margin_is_comfortable():
    ctrl = AdaptiveClock(ratio=4, tolerance=1e-2, order=1, hysteresis=1.5, growth=2.0)
    new = ctrl.update(bound=1e-4)
    assert new > 4
    assert ctrl.history[-1]["action"] == "coarsen"


def test_adaptive_clock_settle_reaches_tolerance_against_measured_bounds():
    """``settle`` must use measured bounds, so a wrong model cannot fake success."""
    truth = lambda r: 1e-3 * r ** 2  # noqa: E731  measured bound grows with the ratio
    ctrl = AdaptiveClock(ratio=64, tolerance=1e-3, order=2)
    res = ctrl.settle(truth, max_iter=20)
    assert res["converged"] is True
    assert truth(res["final_ratio"]) <= 1e-3
    assert res["final_ratio"] < 64


def test_ratio_scan_reports_the_coarsest_admissible_ratio():
    truth = lambda r: 1e-3 * r  # noqa: E731
    rep = ratio_scan(truth, (1, 2, 4, 8, 16), tolerance=1e-2)
    assert rep["coarsest_admissible"] == 8
    assert rep["ok_ratios"] == [1, 2, 4, 8]


def test_adaptive_clock_rejects_bad_configuration():
    with pytest.raises(ContractError):
        AdaptiveClock(ratio=1, tolerance=0.0)
    with pytest.raises(ContractError):
        AdaptiveClock(ratio=0, tolerance=1e-2)
    with pytest.raises(ContractError):
        AdaptiveClock(ratio=4, tolerance=1e-2, safety=1.5)
    with pytest.raises(ContractError):
        AdaptiveClock(ratio=4, tolerance=1e-2, growth=1.0)