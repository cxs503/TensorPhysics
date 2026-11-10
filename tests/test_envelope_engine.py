"""Black-box tests of the operating-window engine.

Two groups:

* **synthetic windows with closed-form answers** -- the marched boundary, the
  component count and the area are compared with arithmetic the engine cannot
  have produced itself, so a pass means "right", not "self-consistent";
* **semantics** -- an empty window, a lattice, a two-component window and the
  reproducibility of the detection under refinement.

Every axis name, span, coefficient and tolerance in this file is synthetic
(``alpha`` / ``beta``); the shipped configurations live outside the package and
are exercised by the acceptance script instead.
"""

from __future__ import annotations

import math

import pytest

from tensorphysics.envelope import EnvelopeEngine, case_from_dict
from tensorphysics.envelope import benchmarks as synth


def run(case, condition=None, **kwargs):
    return EnvelopeEngine(case).run(condition=condition, **kwargs)


# ---------------------------------------------------------------------------
# 1-D windows: boundaries against closed-form answers
# ---------------------------------------------------------------------------

def test_strip_window_bounds_drivers_and_verdicts():
    result = run(synth.strip_case())
    assert result.status == "feasible"
    assert len(result.intervals) == 1
    interval = result.intervals[0]
    assert interval.lo == pytest.approx(0.2, abs=1e-13)
    assert interval.hi == pytest.approx(0.9, abs=1e-13)
    assert interval.lo_drivers == ("c_low",)
    assert interval.hi_drivers == ("c_high",)
    assert result.contracts.ok, result.contracts.to_dict()


def test_band_window_matches_closed_form_root():
    result = run(synth.band_case())
    half_width = math.sqrt(0.06)
    assert len(result.intervals) == 1
    assert result.intervals[0].lo == pytest.approx(0.4 - half_width, abs=1e-13)
    assert result.intervals[0].hi == pytest.approx(0.4 + half_width, abs=1e-13)
    # the margin is concave, so the feasible set is convex and the contract holds
    assert result.contracts.ok, result.contracts.to_dict()


def test_equality_constraint_gives_a_tolerance_band():
    result = run(synth.equality_band_case())
    assert len(result.intervals) == 1
    assert result.intervals[0].lo == pytest.approx(0.35, abs=1e-13)
    assert result.intervals[0].hi == pytest.approx(0.45, abs=1e-13)


def test_table_constraint_boundaries_are_exact_inside_segments():
    result = run(synth.table_case())
    assert len(result.intervals) == 1
    # crossings of the piecewise-linear margin with zero inside two segments
    assert result.intervals[0].lo == pytest.approx(1.0 / 6.0, abs=1e-13)
    assert result.intervals[0].hi == pytest.approx(5.0 / 6.0, abs=1e-13)


def test_two_intervals_are_one_window_object_with_two_branches():
    result = run(synth.offset_bands_case())
    assert result.status == "feasible"
    assert len(result.intervals) == 2
    assert result.intervals[0].hi == pytest.approx(0.3, abs=1e-13)
    assert result.intervals[1].lo == pytest.approx(0.7, abs=1e-13)
    # declared as a single window: the structural check must say so
    assert "structure:window" in result.contracts.failed


def test_window_is_reproducible_under_refinement():
    case = synth.band_case()
    coarse = run(case, n_scan=24)
    fine = run(case, n_scan=384)
    assert coarse.intervals[0].lo == pytest.approx(fine.intervals[0].lo, abs=1e-13)
    assert coarse.intervals[0].hi == pytest.approx(fine.intervals[0].hi, abs=1e-13)
    assert fine.active_sets["alpha"]["verified"] is True


def test_reported_cost_grows_with_the_scan_and_covers_both_grids():
    case = synth.strip_case()
    coarse = EnvelopeEngine(case).run(n_scan=64)
    fine = EnvelopeEngine(case).run(n_scan=192)
    constraints = len(case.constraints)
    # the window is detected twice (primary grid + verification grid), each pass
    # sampling every continuous constraint once per node
    assert coarse.n_evals >= 2 * 64 * constraints
    assert fine.n_evals > coarse.n_evals


# ---------------------------------------------------------------------------
# discrete feasibility
# ---------------------------------------------------------------------------

def test_lattice_snapping_is_reported_separately_from_the_continuous_window():
    result = run(synth.lattice_case())
    interval = result.intervals[0]
    assert interval.lo == pytest.approx(0.2, abs=1e-13)
    assert interval.hi == pytest.approx(0.9, abs=1e-13)
    assert result.discrete is not None
    assert result.discrete.n_settings == 8
    assert len(result.discrete.runs) == 1
    run_info = result.discrete.runs[0]
    assert run_info.lo == pytest.approx(0.2, abs=1e-12)
    assert run_info.hi == pytest.approx(0.9, abs=1e-12)


def test_lattice_points_outside_the_window_are_not_attainable():
    payload = {
        "case_id": "synth_narrow_lattice",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_low",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "var", "name": "alpha"},
                    "b": {"op": "const", "value": 0.35},
                },
            },
            {
                "id": "c_high",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "const", "value": 0.9},
                    "b": {"op": "var", "name": "alpha"},
                },
            },
            {
                "id": "c_setting",
                "kind": "discrete",
                "value": {"op": "var", "name": "alpha"},
                "lattice": {"step": 0.1, "tolerance": 1e-9},
            },
        ],
    }
    result = EnvelopeEngine(case_from_dict(payload)).run(n_scan=128)
    assert result.discrete is not None
    assert result.discrete.n_settings == 6
    assert result.discrete.runs[0].lo == pytest.approx(0.4, abs=1e-12)
    assert result.discrete.runs[0].hi == pytest.approx(0.9, abs=1e-12)


def test_lattice_only_case_is_refused():

    payload = {
        "case_id": "synth_only_lattice",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_setting",
                "kind": "discrete",
                "value": {"op": "var", "name": "alpha"},
                "lattice": {"step": 0.1, "tolerance": 1e-9},
            }
        ],
    }
    case = case_from_dict(payload)
    with pytest.raises(ValueError):
        EnvelopeEngine(case).run()


# ---------------------------------------------------------------------------
# empty windows
# ---------------------------------------------------------------------------

def test_empty_window_is_infeasible_without_boundary_or_area():
    result = run(synth.empty_case())
    assert result.status == "infeasible"
    assert result.intervals == ()
    assert result.components == ()
    assert result.area is None
    assert result.nearest is not None
    assert result.nearest.found is False
    assert result.nearest.residual > 0.0
    assert result.contracts.ok, result.contracts.to_dict()


def test_nearest_feasible_point_stays_inside_the_declared_span():
    result = run(synth.empty_case())
    point = result.nearest
    assert 0.0 <= point.params["alpha"] <= 1.0
    assert point.searched > 0
    assert point.n_evals > 0


def test_infeasible_semantics_are_also_checked_on_the_result_object():
    result = run(synth.empty_case())
    semantics = [o for o in result.contracts.outcomes if o.kind == "infeasible_semantics"]
    assert semantics and semantics[0].passed
    assert semantics[0].metrics["nearest_residual"] == pytest.approx(result.nearest.residual)


# ---------------------------------------------------------------------------
# 2-D marching
# ---------------------------------------------------------------------------

def test_disc_area_and_component_count_against_closed_form():
    case = synth.disc_case()
    result = run(case, n_lines=120, n_scan=120, n_grid_check=40)
    exact = math.pi * case.constants["radius"] ** 2
    assert result.status == "feasible"
    assert len(result.components) == 1
    assert result.components[0].closed is True
    assert result.components[0].clipped is False
    assert abs(result.area - exact) / exact <= 1e-3
    assert result.contracts.ok, result.contracts.to_dict()


def test_two_disjoint_regions_stay_two_components():
    case = synth.two_disc_case()
    result = run(case, n_lines=160, n_scan=120, n_grid_check=40)
    exact = 2.0 * math.pi * 0.3 ** 2
    assert len(result.components) == 2
    assert abs(result.area - exact) / exact <= 1e-3
    assert result.grid_check is not None
    assert result.grid_check.feasible_components_estimate() == 2
    assert result.contracts.ok, result.contracts.to_dict()


def test_parabolic_band_boundary_order_is_second_order():
    from tensorphysics.envelope import order_study

    study = order_study(
        synth.parabolic_band_case(),
        resolutions=(20, 40, 80),
        n_scan=96,
        analytic=synth.analytic_band(),
    )
    assert len(study.orders) == 2
    assert min(study.orders) >= 1.9, study.to_dict()
    assert study.outcome.passed


def test_parabolic_band_area_matches_closed_form():
    # integral of (upper - lower) over alpha in [0, 1] = 1.2 - 1.3 / 12
    expected = 1.2 - 1.3 / 12.0
    result = run(synth.parabolic_band_case(), n_lines=120, n_scan=96)
    assert len(result.components) == 1
    assert result.components[0].clipped is True  # the band runs off both span ends
    assert result.components[0].area_method == "trapezoid"
    assert abs(result.area - expected) / expected <= 1e-4


def test_plane_window_reports_the_binding_constraint_per_branch():
    result = run(synth.parabolic_band_case(), n_lines=80, n_scan=96)
    component = result.components[0]
    assert set(component.lower_drivers[1]) == {"c_low"}
    assert set(component.upper_drivers[1]) == {"c_high"}
    counts = result.active_sets["driver_counts"]
    assert counts["c_low"] > 0 and counts["c_high"] > 0


def test_plane_scan_grid_cross_check_agrees_in_topology():
    result = run(synth.disc_case(), n_lines=120, n_scan=120, n_grid_check=48)
    assert result.grid_check is not None
    assert result.grid_check.feasible_components_estimate() == len(result.components)
    # the cell count is a first-order estimate on a deliberately coarse lattice,
    # quoted only as an order-of-magnitude cross-check of the marched area
    assert result.grid_check.cells.area == pytest.approx(result.area, rel=5e-2)