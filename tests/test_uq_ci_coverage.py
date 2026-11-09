"""Tests of the interval-calibration study (IN145-N08 criterion C7).

The point of these tests is not "coverage is 0.95" -- with a handful of
replications that is not measurable -- but that the study *detects* the regime
where the intervals are wrong.  A starved budget (N = 128 for a degree-8 chaos
in three dimensions) is the positive control of the detector.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.uq import ishigami, ishigami_analytic, ishigami_spec
from tensorphysics.uq.ci_coverage import coverage_study, wilson_interval


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = wilson_interval(18, 20)
    assert 0.0 <= lo < 0.9 < hi <= 1.0
    lo_edge, hi_edge = wilson_interval(20, 20)
    assert hi_edge == pytest.approx(1.0)
    assert 0.75 < lo_edge < 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_coverage_study_is_calibrated_at_a_workable_budget() -> None:
    report = coverage_study(
        ishigami_spec(),
        ishigami,
        ishigami_analytic(),
        n_samples=256,
        degree=8,
        n_boot=80,
        replications=6,
    )
    payload = report.as_dict()
    assert payload["n_samples"] == 256
    assert len(payload["entries"]) == 6  # 3 inputs x (first, total)
    assert report.pooled_coverage >= 0.7
    assert not report.under_covering
    # null indices are scored by resolution, material ones by containment
    null_entries = [e for e in report.entries if e.is_null]
    assert null_entries and all(e.criterion == "resolution" for e in null_entries)
    material = [e for e in report.entries if not e.is_null]
    assert material and all(e.criterion == "interval" for e in material)
    for entry in material:
        assert 0.0 <= entry.under_coverage_pvalue <= 1.0
        lo, hi = entry.interval
        assert 0.0 <= lo <= entry.coverage <= hi <= 1.0


def test_starved_budget_is_flagged_as_under_covering() -> None:
    report = coverage_study(
        ishigami_spec(),
        ishigami,
        ishigami_analytic(),
        n_samples=128,
        degree=8,
        n_boot=60,
        replications=8,
    )
    assert report.pooled_coverage < report.coverage_floor or report.under_covering
    assert not report.calibrated


def test_coverage_study_validates_its_inputs() -> None:
    with pytest.raises(ValueError):
        coverage_study(
            ishigami_spec(), ishigami, ishigami_analytic(), replications=3
        )
    with pytest.raises(ValueError):
        coverage_study(
            ishigami_spec(),
            ishigami,
            {"first": [0.5], "total": [0.5]},
            replications=6,
        )
    with pytest.raises(TypeError):
        coverage_study("spec", ishigami, ishigami_analytic(), replications=6)  # type: ignore[arg-type]


def test_wilson_bounds_contain_the_nominal_level_for_a_large_sample() -> None:
    lo, hi = wilson_interval(950, 1000)
    assert lo < 0.95 < hi
    assert math.isclose(hi - lo, 0.0268, abs_tol=0.005)