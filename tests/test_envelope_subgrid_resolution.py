"""Sub-grid features: probes, the guarantee, and the failure it prevents.

The negative control in this file is the point of the whole exercise.  With a
single midpoint probe per segment -- the v1 mechanism -- a feasible pocket
narrower than the node spacing is either inflated to the entire span or lost
completely, and both verdicts pass every contract because nothing recorded the
scale that was probed.  Every test below either demonstrates the fixed behaviour
or demonstrates that the failure mode is still reachable when the probing is
deliberately coarsened, which is what makes the fix a measurement rather than a
claim.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorphysics.envelope import (
    EnvelopeConfigError,
    EnvelopeEngine,
    ResolutionPolicy,
    case_from_dict,
    load_case,
)

CASES = pathlib.Path(__file__).resolve().parents[1] / "scenarios" / "envelope_cases"

SCHEMA = "tensorphysics.envelope.case.v1"
NODE_SPACING = 1.0 / 95.0  # nodes of the default 96-point scan on [0, 1]
DEFAULT_SAMPLES = 8  # probes per scan cell (ResolutionPolicy default)
PROBE_SPACING = NODE_SPACING / DEFAULT_SAMPLES  # the guaranteed feature width


# ---------------------------------------------------------------------------
# case builders
# ---------------------------------------------------------------------------

def _square(axis: str, centre: float) -> dict:
    return {
        "op": "square",
        "arg": {
            "op": "sub",
            "a": {"op": "var", "name": axis},
            "b": {"op": "const", "value": centre},
        },
    }


def _pocket_payload(centre: float, half_width: float, resolution: dict | None = None) -> dict:
    """Feasible only where |alpha - centre| <= half_width."""

    payload = {
        "schema": SCHEMA,
        "case_id": f"pocket_{centre}_{half_width}",
        "title": "a pocket narrower than the node spacing",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_pocket",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "const", "value": half_width * half_width},
                    "b": _square("alpha", centre),
                },
            }
        ],
    }
    if resolution is not None:
        payload["contract"] = {"resolution": resolution}
    return payload


def _pocket_case(centre: float, half_width: float, resolution: dict | None = None):
    return case_from_dict(_pocket_payload(centre, half_width, resolution))


def _gap_case(centre: float, half_width: float):
    """Feasible everywhere except |alpha - centre| <= half_width."""

    return case_from_dict(
        {
            "schema": SCHEMA,
            "case_id": f"gap_{centre}_{half_width}",
            "title": "an infeasible gap narrower than the node spacing",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_gap",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "sub",
                        "a": _square("alpha", centre),
                        "b": {"op": "const", "value": half_width * half_width},
                    },
                }
            ],
        }
    )


def _channel_payload(eps: float, resolution: dict | None = None) -> dict:
    """Two-dimensional channel |beta - alpha| <= eps on the unit square."""

    payload = {
        "schema": SCHEMA,
        "case_id": f"channel_{eps}",
        "title": "diagonal channel thinner than the line spacing",
        "mode": "plane",
        "axes": [
            {"name": "alpha", "lo": 0.0, "hi": 1.0},
            {"name": "beta", "lo": 0.0, "hi": 1.0},
        ],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_hi",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "const", "value": eps},
                    "b": {
                        "op": "sub",
                        "a": {"op": "var", "name": "beta"},
                        "b": {"op": "var", "name": "alpha"},
                    },
                },
            },
            {
                "id": "c_lo",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "const", "value": eps},
                    "b": {
                        "op": "sub",
                        "a": {"op": "var", "name": "alpha"},
                        "b": {"op": "var", "name": "beta"},
                    },
                },
            },
        ],
    }
    if resolution is not None:
        payload["contract"] = {"resolution": resolution}
    return payload


def _channel_case(eps: float, resolution: dict | None = None):
    return case_from_dict(_channel_payload(eps, resolution))


def _run(case, **kwargs):
    return EnvelopeEngine(case).run(condition="one", **kwargs)


# ---------------------------------------------------------------------------
# 1-D: the two silent failure modes of a single midpoint probe
# ---------------------------------------------------------------------------

def test_pocket_at_the_midpoint_is_no_longer_inflated_to_the_whole_span():
    """v1 reported the whole span feasible because the midpoint fell inside."""

    result = _run(_pocket_case(0.5, 1e-3))
    assert result.status == "feasible"
    assert len(result.intervals) == 1
    interval = result.intervals[0]
    assert interval.lo == pytest.approx(0.499, rel=1e-9, abs=1e-12)
    assert interval.hi == pytest.approx(0.501, rel=1e-9, abs=1e-12)
    assert result.width == pytest.approx(0.002, rel=1e-9)


def test_pocket_off_the_midpoint_is_no_longer_lost():
    """v1 reported 'infeasible' and the window vanished without a trace."""

    result = _run(_pocket_case(0.3, 1e-3))
    assert result.status == "feasible"
    assert result.width == pytest.approx(0.002, rel=1e-9)
    interval = result.intervals[0]
    assert interval.lo == pytest.approx(0.299, rel=1e-9, abs=1e-12)
    assert interval.hi == pytest.approx(0.301, rel=1e-9, abs=1e-12)


def test_narrow_infeasible_gap_inside_a_feasible_span_is_found():
    result = _run(_gap_case(0.37, 8e-4))
    assert result.status == "feasible"
    assert len(result.intervals) == 2
    left, right = result.intervals
    assert left.lo == pytest.approx(0.0, abs=1e-15)
    assert left.hi == pytest.approx(0.37 - 8e-4, rel=1e-8, abs=1e-12)
    assert right.lo == pytest.approx(0.37 + 8e-4, rel=1e-8, abs=1e-12)
    assert right.hi == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("half_width", [4e-3, 1e-3, 7e-4])
def test_pocket_width_is_recovered_below_the_node_spacing(half_width):
    assert 2 * half_width < NODE_SPACING  # genuinely sub-grid
    assert 2 * half_width >= PROBE_SPACING  # ... and inside the guarantee
    result = _run(_pocket_case(0.417, half_width))
    assert result.width == pytest.approx(2 * half_width, rel=1e-9)


# ---------------------------------------------------------------------------
# the negative control: coarsen the probing and the v1 failure returns
# ---------------------------------------------------------------------------

def test_coarse_probing_loses_the_window_and_the_ledger_says_so():
    """Same code, same case; only the probe count differs.

    With one probe per scan cell (the v1 scale) a sub-cell pocket is invisible:
    the window is lost and the ledger reports the spacing it actually probed.
    """

    assert _run(_pocket_case(0.5, 1e-3)).width == pytest.approx(0.002, rel=1e-9)
    assert _run(_pocket_case(0.3, 1e-3)).width == pytest.approx(0.002, rel=1e-9)

    coarse_policy = {"samples_per_segment": 1, "max_depth": 0, "sub_cells": 2}
    for centre in (0.5, 0.3):
        lost = _run(_pocket_case(centre, 1e-3, coarse_policy))
        assert lost.status == "infeasible"  # v1: the window vanishes silently
        assert lost.resolution["guaranteed_width"] == pytest.approx(
            1.0 / 96.0, rel=1e-12  # one probe of the endpoints + 95 cells
        )
        assert not lost.ok  # the doubt is reported, never silently averaged away


def test_a_probe_inside_the_pocket_is_never_inflated_to_the_whole_span():
    """The v1 false *positive* is structurally impossible: disagreement is loud.

    One feasible probe among infeasible ones either gets refined or is reported
    as an unresolved span -- it can never be averaged into "the span is
    feasible".
    """

    for centre in (0.5, 47.0 / 95.0, 0.3):
        coarse_policy = {"samples_per_segment": 1, "max_depth": 0, "sub_cells": 2}
        result = _run(_pocket_case(centre, 1e-3, coarse_policy))
        silently_inflated = (
            result.status == "feasible" and (result.width or 0.0) > 2 * NODE_SPACING
        )
        assert not silently_inflated
        if result.status == "feasible":
            # a genuine pocket-scale window, never the span at large
            assert result.width <= 4 * NODE_SPACING


def test_a_declared_minimum_feature_width_catches_an_under_probed_scan():
    """The same coarse scan, but now the case says what it expects to resolve."""

    coarse = _pocket_case(
        0.5,
        1e-3,
        {
            "samples_per_segment": 1,
            "max_depth": 0,
            "sub_cells": 2,
            "min_feature_width": 1e-3,
        },
    )
    result = _run(coarse)
    assert not result.ok
    assert "resolution:window" in result.contracts.failed
    outcome = next(o for o in result.contracts.outcomes if o.id == "resolution:window")
    assert outcome.metrics["declared_min_feature_width"] == pytest.approx(1e-3)
    assert outcome.metrics["min_probed_width"] > 1e-3


def test_a_declared_minimum_feature_width_passes_when_the_probing_is_fine_enough():
    result = _run(_pocket_case(0.5, 1e-3, {"min_feature_width": 2e-3}))
    assert result.ok
    outcome = next(o for o in result.contracts.outcomes if o.id == "resolution:window")
    assert outcome.metrics["min_probed_width"] <= 2e-3


# ---------------------------------------------------------------------------
# the guarantee, and what happens when it cannot be met
# ---------------------------------------------------------------------------

def test_probed_width_is_reported_and_matches_the_policy():
    policy = ResolutionPolicy(samples_per_segment=7, max_depth=4, sub_cells=4)
    spec = policy.to_dict()
    spec["min_feature_width"] = None
    result = _run(_pocket_case(0.62, 1e-3, spec))
    ledger = result.resolution
    assert ledger["policy"]["samples_per_segment"] == 7
    assert ledger["guaranteed_width"] <= NODE_SPACING / 7
    assert ledger["refined_spans"] >= 1
    assert ledger["resolved"] is True


def test_unresolved_spans_are_reported_and_fail_the_contract():
    """Depth zero: a disagreement that cannot be subdivided is not hidden."""

    result = _run(_pocket_case(0.5, 1e-3, {"max_depth": 0}))
    assert result.resolution["resolved"] is False
    assert result.resolution["unresolved_spans"]
    assert not result.ok
    failed = [o.id for o in result.contracts.outcomes if not o.passed]
    assert "resolution:window" in failed


def test_a_margin_that_flips_faster_than_the_refinement_is_reported():
    """A zig-zag table margin: 200 sign flips inside one scan cell."""

    knots = [i / 2000.0 for i in range(2001)]
    values = [1.0 if i % 2 == 0 else -1.0 for i in range(2001)]
    case = case_from_dict(
        {
            "schema": SCHEMA,
            "case_id": "zigzag",
            "title": "margin flipping faster than the refinement follows",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_zig",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "table",
                        "arg": {"op": "var", "name": "alpha"},
                        "x": knots,
                        "y": values,
                    },
                }
            ],
            "contract": {"resolution": {"max_depth": 1, "sub_cells": 4}},
        }
    )
    result = _run(case)
    ledger = result.resolution
    assert ledger["suspect_spans"] >= 1
    assert ledger["unresolved_spans"], "a span that cannot be settled must be listed"
    assert not result.ok


def test_cost_of_the_sub_grid_pass_stays_local():
    """Only the spans that disagree are subdivided: the bill stays small."""

    nailed = _run(_pocket_case(0.5, 1e-3))
    assert nailed.resolution["suspect_spans"] >= 1
    assert nailed.n_evals < 20000

    smooth = case_from_dict(
        {
            "schema": SCHEMA,
            "case_id": "smooth",
            "title": "one affine boundary",
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
                        "b": {"op": "const", "value": 0.2},
                    },
                }
            ],
        }
    )
    smooth_result = _run(smooth)
    assert smooth_result.resolution["suspect_spans"] == 0
    assert smooth_result.n_evals < nailed.n_evals


# ---------------------------------------------------------------------------
# 2-D: a feature thinner than the line spacing
# ---------------------------------------------------------------------------

def test_diagonal_channel_is_feasible_where_the_line_marcher_used_to_say_no():
    eps = 2e-3
    true_area = 2 * eps - eps * eps
    coarse = _run(_channel_case(eps), n_lines=100, n_scan=100)
    assert coarse.status == "feasible"  # v1 answered 'infeasible'
    assert coarse.resolution["single_line_components"] > 0
    assert coarse.resolution["area_reliable"] is False
    assert coarse.resolution["line_spacing"] > 2 * eps

    fine = _run(_channel_case(eps), n_lines=400, n_scan=120)
    assert fine.status == "feasible"
    assert fine.resolution["area_reliable"] is True
    assert fine.area == pytest.approx(true_area, rel=1e-3)


def test_two_dimensional_area_reliability_is_not_a_false_alarm_on_a_smooth_disc():
    from tensorphysics.envelope import benchmarks as synth

    result = _run(synth.disc_case(), n_lines=120, n_scan=120)
    assert result.resolution["area_reliable"] is True
    assert result.resolution["single_line_components"] == 0
    assert result.ok


def test_plane_guarantee_accounts_for_the_line_spacing():
    eps = 2e-3
    case = _channel_case(eps, {"min_feature_width": 3e-3})
    coarse = _run(case, n_lines=100, n_scan=100)
    assert not coarse.ok  # line spacing 1/99 is coarser than the declared feature
    fine = _run(case, n_lines=400, n_scan=120)
    assert fine.ok


# ---------------------------------------------------------------------------
# reproducibility and the shipped cases
# ---------------------------------------------------------------------------

def test_resolution_ledger_is_reproducible_and_serialised():
    case = _pocket_case(0.5, 1e-3)
    first = _run(case).to_dict()
    second = _run(case).to_dict()
    assert first.pop("elapsed_s") >= 0.0
    assert second.pop("elapsed_s") >= 0.0
    assert first == second
    payload = first
    assert payload["resolution"]["policy"]["samples_per_segment"] == DEFAULT_SAMPLES
    assert payload["resolution"]["guaranteed_width"] > 0.0


@pytest.mark.parametrize("case_path", sorted(CASES.glob("*.json")))
def test_shipped_cases_are_unchanged_by_the_sub_grid_pass(case_path):
    """Migration must stay bit-identical: the new probes may not move a number."""

    payload = json.loads(case_path.read_text())
    coarse_payload = json.loads(json.dumps(payload))
    coarse_payload.setdefault("contract", {})
    coarse_payload["contract"]["resolution"] = {
        "samples_per_segment": 1,
        "max_depth": 0,
        "sub_cells": 2,
        "min_feature_width": None,
    }
    strict_case = load_case(case_path)
    coarse_case = case_from_dict(coarse_payload)
    for condition in strict_case.conditions:
        strict = EnvelopeEngine(strict_case).run(condition=condition.label)
        coarse = EnvelopeEngine(coarse_case).run(condition=condition.label)
        assert strict.status == coarse.status
        assert len(strict.intervals) == len(coarse.intervals)
        for a, b in zip(strict.intervals, coarse.intervals):
            assert a.lo == b.lo and a.hi == b.hi
        assert strict.resolution["suspect_spans"] == 0


def test_bad_resolution_configuration_is_refused():
    payload = _pocket_payload(0.5, 1e-3, {"samples_per_segment": 0})
    with pytest.raises(EnvelopeConfigError):
        case_from_dict(payload)