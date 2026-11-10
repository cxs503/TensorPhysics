"""Tests for the IN157-N05 SIMP density-filter acceptance surface.

Fast coverage of: the literal checkerboard detector, the Spearman helper, the
filter mechanism that backs the negative control, the negative control itself
(r -> 0 reproduces the unfiltered run), the radius-scan bookkeeping and the
honest C1 bookkeeping (ratio < 5 must be recorded as FAIL, never redefined).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.topology_optimization.simp.acceptance_in157 import (
    RADIUS_SCAN,
    alternating_cell_fraction,
    count_test_cases,
    run_acceptance,
    spearman_rho,
)
from tensorphysics.topology_optimization.simp.density_filter import (
    DensityFilter,
    checkerboard_index,
)
from tensorphysics.topology_optimization.simp.problems import build_mbb_beam, run_simp


def _checkerboard(nely: int = 12, nelx: int = 16) -> np.ndarray:
    iy, ix = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    return ((iy + ix) % 2).astype(float)


# --------------------------------------------------------------- detector
def test_alternating_fraction_perfect_checkerboard_is_one():
    assert alternating_cell_fraction(_checkerboard(12, 16)) == pytest.approx(1.0)


def test_alternating_fraction_uniform_field_is_zero():
    assert alternating_cell_fraction(np.full((12, 16), 0.5)) == 0.0
    assert alternating_cell_fraction(np.zeros((10, 20))) == 0.0
    assert alternating_cell_fraction(np.ones((10, 20))) == 0.0


def test_alternating_fraction_stripes_are_not_checkerboard():
    stripe = np.zeros((12, 16))
    stripe[:, ::2] = 1.0  # alternating columns: diagonals disagree
    assert alternating_cell_fraction(stripe) == pytest.approx(0.0, abs=1e-12)


def test_alternating_fraction_rejects_bad_input():
    with pytest.raises(ValueError):
        alternating_cell_fraction(np.zeros(20))  # 1-D
    with pytest.raises(ValueError):
        alternating_cell_fraction(np.zeros((2, 20)))  # too small
    with pytest.raises(ValueError):
        alternating_cell_fraction(np.zeros((5, 5)), band=0.0)


def test_alternating_fraction_localised_patch_scales_with_patch_size():
    field = np.zeros((20, 20))
    field[4:6, 4:6] = [[0.0, 1.0], [1.0, 0.0]]
    idx = alternating_cell_fraction(field)
    assert 0.0 < idx < 0.05
    assert idx == pytest.approx(4.0 / 400.0)


# --------------------------------------------------------------- spearman
def test_spearman_monotone_increasing_is_one():
    assert spearman_rho([1.0, 1.5, 2.0, 2.5, 3.0], [10.0, 11.0, 30.0, 31.0, 99.0]) == pytest.approx(1.0)


def test_spearman_monotone_decreasing_is_minus_one():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_with_ties_and_constant_sample():
    rho = spearman_rho([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert -1.0 <= rho <= 1.0 and np.isfinite(rho)
    assert spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


# --------------------------------------------------------------- filter
def test_density_filter_keeps_the_box_and_unit_row_sums():
    """The argument behind C1: filtered fields stay inside the design box."""
    rng = np.random.default_rng(3)
    rho = rng.random(9 * 11)
    dens = DensityFilter(11, 9, 1.5)
    phys = dens.apply(rho)
    assert np.all(phys <= rho.max() + 1e-12) and np.all(phys >= rho.min() - 1e-12)
    row_sums = np.asarray(dens.matrix.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12)
    # a convex combination cannot leave [0, 1] even when rho does
    out_of_box = dens.apply(np.where(rng.random(rho.size) > 0.5, 1.0, 0.0))
    assert out_of_box.min() >= 0.0 and out_of_box.max() <= 1.0


def test_density_filter_suppresses_synthetic_checkerboard():
    field = _checkerboard(24, 32)
    dens = DensityFilter(32, 24, 1.5)
    assert alternating_cell_fraction(field) > 0.9
    filtered = dens.apply(field).reshape(24, 32)
    assert alternating_cell_fraction(filtered) < 0.05
    assert checkerboard_index(filtered) < 0.05


# ------------------------------------------------- negative control (r -> 0)
def test_negative_control_radius_zero_equals_unfiltered():
    prob = build_mbb_beam(12, 6, 0.5)
    seed_design = 0.5 + 0.1 * (np.random.default_rng(7).random(prob.n_elems) - 0.5)
    ref = run_simp(prob, radius=None, penal=3.0, iters=6, tol=5e-3, rho0=seed_design)
    neg = run_simp(prob, radius=0.0, penal=3.0, iters=6, tol=5e-3, rho0=seed_design)
    assert neg["compliance"] == pytest.approx(ref["compliance"], rel=1e-12)
    assert np.allclose(neg["rho"], ref["rho"], atol=1e-12)


# ------------------------------------------------------- acceptance record
@pytest.fixture(scope="module")
def quick_record(tmp_path_factory):
    out = tmp_path_factory.mktemp("simp_in157")
    return run_acceptance(
        out,
        workers=1,
        iters=8,
        tol=5e-3,
        seeds=(0,),
        radii=(1.0, 1.5, 2.0),
        quick=True,
    ), out


def test_record_covers_all_five_criteria(quick_record):
    rec, out = quick_record
    assert set(rec["criteria"]) == {"C1", "C2", "C3", "C4", "C5"}
    assert len(rec["criteria"]["C3"]["compliance"]) == len(rec["radius_scan"])
    assert (out / "verification.json").exists() and (out / "designs.npz").exists()
    payload = json.loads((out / "verification.json").read_text())
    assert set(payload["criteria"]) == {"C1", "C2", "C3", "C4", "C5"}


def test_c1_status_is_honest_bookkeeping(quick_record):
    rec, _ = quick_record
    c1 = rec["criteria"]["C1"]
    assert c1["status"] == ("PASS" if c1["ratio_mean"] >= 5.0 else "FAIL")
    assert "theoretical_bound" in c1 and c1["checkerboard_suppression"] > 0.0


def test_negative_control_criterion_uses_measured_numbers(quick_record):
    rec, _ = quick_record
    c4 = rec["criteria"]["C4"]
    assert c4["relative_deviation"] == pytest.approx(
        abs(c4["compliance"] - c4["compliance_reference"]) / c4["compliance_reference"]
    )
    assert c4["status"] == ("PASS" if c4["relative_deviation"] <= 0.01 else "FAIL")


def test_c3_spearman_matches_computed_ranking(quick_record):
    rec, _ = quick_record
    c3 = rec["criteria"]["C3"]
    assert c3["spearman_rho"] == pytest.approx(spearman_rho(c3["radii"], c3["compliance"]))
    assert c3["status"] == ("PASS" if c3["spearman_rho"] > 0.9 else "FAIL")


def test_verdict_lists_failed_criteria(quick_record):
    rec, _ = quick_record
    statuses = rec["verdict"]["criteria_status"]
    failed = sorted(k for k, v in statuses.items() if v == "FAIL")
    assert rec["verdict"]["failed"] == failed
    assert rec["verdict"]["n_pass"] + len(failed) == len(statuses)


def test_c5_counts_real_test_functions():
    assert count_test_cases("tests/test_simp_in157*.py") >= 8
    assert len(RADIUS_SCAN) >= 5