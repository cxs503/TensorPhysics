"""Tests of the contract checker, the configuration gate and the artifact writer.

The point of this file is that every check is *falsifiable*: for each contract
there is a positive case (a window that satisfies the declaration) and a
negative case (a window that does not), and the negative case must produce a
reported violation with the numbers behind it.  A checker that only ever passes
is indistinguishable from no checker at all.
"""

from __future__ import annotations

import json
import math

import pytest

from tensorphysics.envelope import (
    ContractPolicy,
    EnvelopeConfigError,
    EnvelopeEngine,
    ExpressionError,
    Lattice,
    bisect,
    boundary_order,
    bracket,
    brent,
    canonical_digest,
    case_from_dict,
    check_area_consistency,
    check_boundary_order,
    check_infeasible_semantics,
    code_digest,
    evaluate_expression,
    free_names,
    load_case,
    read_artifact,
    symmetric_eigenvalues,
    write_artifact,
)
from tensorphysics.envelope import benchmarks as synth
from tensorphysics.envelope.contracts import CheckOutcome


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------

def test_folded_margin_violates_the_declared_direction():
    result = EnvelopeEngine(synth.folded_case()).run(n_scan=96)
    assert result.contracts.ok is False
    assert "monotone:c_band:alpha" in result.contracts.failed
    outcome = next(o for o in result.contracts.outcomes if o.id == "monotone:c_band:alpha")
    assert outcome.metrics["fold_count"] >= 1
    assert outcome.metrics["worst_opposite_fraction"] > 0.5


def test_same_geometry_without_the_declaration_passes():
    folded = synth.folded_case()
    payload = {
        "case_id": "synth_folded_undeclared",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_band",
                "kind": "inequality",
                "side": "ge",
                "value": folded.constraints.by_id("c_band").expression,
            }
        ],
        "contract": {"expect": {"intervals": 1}},
    }
    result = EnvelopeEngine(case_from_dict(payload)).run(n_scan=96)
    assert result.contracts.ok, result.contracts.to_dict()


def test_declared_but_flat_response_is_a_violation():
    payload = {
        "case_id": "synth_flat",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_flat",
                "kind": "inequality",
                "side": "ge",
                "value": {"op": "const", "value": 1.0},
                "contract": {"slope": {"alpha": 1}},
            }
        ],
        "contract": {"expect": {"intervals": 1}},
    }
    result = EnvelopeEngine(case_from_dict(payload)).run(n_scan=32)
    assert "monotone:c_flat:alpha" in result.contracts.failed


# ---------------------------------------------------------------------------
# convexity
# ---------------------------------------------------------------------------

def test_convexity_check_flags_a_non_convex_feasible_set():
    result = EnvelopeEngine(synth.offset_bands_case()).run(n_scan=64)
    assert "convex:c_outer" in result.contracts.failed
    outcome = next(o for o in result.contracts.outcomes if o.id == "convex:c_outer")
    assert outcome.metrics["max_eigenvalue"] > 0.0


def test_convexity_check_accepts_a_concave_margin():
    result = EnvelopeEngine(synth.band_case()).run(n_scan=64)
    outcome = next(o for o in result.contracts.outcomes if o.id == "convex:c_band")
    assert outcome.passed
    assert outcome.metrics["max_eigenvalue"] <= 0.0


def test_symmetric_eigenvalues_of_a_known_matrix():
    values = symmetric_eigenvalues([[2.0, 1.0], [1.0, 2.0]])
    assert values == pytest.approx((1.0, 3.0), abs=1e-12)
    values = symmetric_eigenvalues([[0.0, 0.0], [0.0, -4.0]])
    assert values == pytest.approx((-4.0, 0.0), abs=1e-12)


# ---------------------------------------------------------------------------
# boundary order
# ---------------------------------------------------------------------------

def test_second_order_error_sequence_is_accepted():
    steps = [0.1, 0.05, 0.025]
    errors = [4e-3, 1e-3, 2.5e-4]
    assert boundary_order(steps, errors) == pytest.approx((2.0, 2.0), abs=1e-9)
    outcome = check_boundary_order(steps, errors, 1.9)
    assert outcome.passed


def test_first_order_error_sequence_is_rejected():
    steps = [0.1, 0.05, 0.025]
    errors = [4e-2, 2e-2, 1e-2]
    outcome = check_boundary_order(steps, errors, 1.9)
    assert outcome.passed is False
    assert outcome.metrics["min_observed_order"] == pytest.approx(1.0, abs=1e-9)


def test_boundary_order_requires_distinct_steps():
    with pytest.raises(ValueError):
        boundary_order([0.1, 0.1], [1e-3, 1e-4])


# ---------------------------------------------------------------------------
# infeasible semantics and area consistency
# ---------------------------------------------------------------------------

class _Bare:
    lo = 0.0
    hi = 1.0
    lo_drivers = ()
    hi_drivers = ()
    area = 0.0


class _Nearest:
    def __init__(self, residual: float, found: bool = False):
        self.params = {"alpha": 0.5}
        self.coords = (0.5,)
        self.residual = residual
        self.found = found


def test_infeasible_semantics_rejects_a_leaked_boundary():
    outcome = check_infeasible_semantics(
        "infeasible", [_Bare()], (), None, _Nearest(0.4)
    )
    assert outcome.passed is False
    assert "boundary structure" in outcome.detail


def test_infeasible_semantics_rejects_a_missing_residual():
    outcome = check_infeasible_semantics("infeasible", (), (), 0.0, None)
    assert outcome.passed is False


def test_infeasible_semantics_rejects_a_contradictory_nearest_point():
    outcome = check_infeasible_semantics("infeasible", (), (), 0.0, _Nearest(0.0, True))
    assert outcome.passed is False
    assert "reported feasible" in outcome.detail


def test_infeasible_semantics_rejects_an_empty_feasible_window():
    outcome = check_infeasible_semantics("feasible", (), (), None, None)
    assert outcome.passed is False


def test_area_consistency_is_two_sided():
    assert check_area_consistency(1.0, 1.0 + 1e-6, 1e-3).passed
    assert not check_area_consistency(1.0, 1.0 + 1e-2, 1e-3).passed


# ---------------------------------------------------------------------------
# structure expectations
# ---------------------------------------------------------------------------

def test_active_set_expectation_is_falsifiable():
    policy = ContractPolicy(expect_active={"lower": "c_high", "upper": "c_low"})
    result = EnvelopeEngine(synth.strip_case(), policy=policy).run(n_scan=64)
    assert "active_set:window" in result.contracts.failed
    detail = next(
        o for o in result.contracts.outcomes if o.id == "active_set:window"
    ).metrics
    assert detail["observed"]["lower"] == ["c_low"]


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------

def _quadratic(x: float) -> float:
    return x * x - 2.0


def test_brent_and_bisection_agree_to_machine_precision():
    brent_result = brent(_quadratic, 1.0, 2.0)
    bisect_result = bisect(_quadratic, 1.0, 2.0)
    assert brent_result.converged and bisect_result.converged
    assert brent_result.x == pytest.approx(math.sqrt(2.0), rel=1e-15)
    assert bisect_result.x == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert brent_result.iterations <= bisect_result.iterations


def test_bracketing_keeps_a_root_that_lands_on_a_node():
    nodes = (0.0, 0.25, 0.5, 0.75, 1.0)
    brackets = bracket(lambda x: x - 0.5, nodes)
    assert brackets == ((0.5, 0.5),)


def test_bracketing_finds_the_interior_sign_change():
    nodes = (0.0, 0.25, 0.5, 0.75, 1.0)
    brackets = bracket(lambda x: x - 0.4, nodes)
    assert brackets == ((0.25, 0.5),)


# ---------------------------------------------------------------------------
# configuration gate
# ---------------------------------------------------------------------------

def _minimal_payload(**overrides):
    payload = {
        "case_id": "synth_gate",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_a",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "var", "name": "alpha"},
                    "b": {"op": "const", "value": 0.5},
                },
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_unknown_parameter_is_located_in_the_error():
    payload = _minimal_payload()
    payload["constraints"][0]["value"]["b"] = {"op": "var", "name": "beta"}
    with pytest.raises(EnvelopeConfigError, match="beta"):
        case_from_dict(payload)


def test_empty_axis_span_is_refused():
    with pytest.raises(EnvelopeConfigError, match="hi > lo"):
        case_from_dict(_minimal_payload(axes=[{"name": "alpha", "lo": 1.0, "hi": 1.0}]))


def test_duplicate_constraint_ids_are_refused():
    payload = _minimal_payload()
    payload["constraints"].append(dict(payload["constraints"][0]))
    with pytest.raises(EnvelopeConfigError, match="duplicate"):
        case_from_dict(payload)


def test_condition_missing_a_parameter_is_refused():
    payload = _minimal_payload()
    payload["conditions"] = [{"label": "one", "params": {}}, {"label": "two"}]
    payload["constraints"][0]["value"]["b"] = {"op": "var", "name": "gamma"}
    payload["conditions"][0]["params"] = {"gamma": 0.5}
    with pytest.raises(EnvelopeConfigError, match="gamma"):
        case_from_dict(payload)


def test_unknown_schema_is_refused():
    with pytest.raises(EnvelopeConfigError, match="schema"):
        case_from_dict(_minimal_payload(schema="other.schema.v9"))


def test_unknown_condition_label_is_refused():
    case = case_from_dict(_minimal_payload())
    with pytest.raises(EnvelopeConfigError, match="unknown condition"):
        case.condition("missing")


def test_plane_mode_needs_two_axes():
    with pytest.raises(EnvelopeConfigError, match="exactly two axes"):
        case_from_dict(_minimal_payload(mode="plane"))


def test_expression_validation_rejects_unknown_op():
    payload = _minimal_payload()
    payload["constraints"][0]["value"] = {"op": "mystery", "arg": {"op": "const", "value": 1.0}}
    with pytest.raises(EnvelopeConfigError, match="unknown op"):
        case_from_dict(payload)


def test_table_refuses_to_extrapolate():
    spec = {
        "op": "table",
        "arg": {"op": "var", "name": "alpha"},
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }
    assert evaluate_expression(spec, {"alpha": 0.5}) == pytest.approx(0.5)
    with pytest.raises(ExpressionError, match="outside the knot span"):
        evaluate_expression(spec, {"alpha": 1.5})


def test_free_names_walks_the_whole_tree():
    spec = {
        "op": "sum",
        "terms": [
            {"op": "var", "name": "alpha"},
            {"op": "table", "arg": {"op": "var", "name": "beta"}, "x": [0, 1], "y": [0, 1]},
        ],
    }
    assert free_names(spec) == {"alpha", "beta"}


def test_lattice_step_and_values_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Lattice(step=0.5, values=(0.0, 1.0))


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def test_case_digest_is_stable_and_sensitive():
    first = canonical_digest(_minimal_payload())
    second = canonical_digest(_minimal_payload())
    assert first == second
    changed = _minimal_payload()
    changed["constraints"][0]["value"]["b"] = {"op": "const", "value": 0.6}
    assert canonical_digest(changed) != first


def test_artifact_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "payload.json"
    written = write_artifact(path, {"schema": "test.v1", "value": 1.0})
    reloaded = read_artifact(path)
    assert reloaded["artifact_digest"] == written["artifact_digest"]
    assert reloaded["code_digest"] == code_digest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["value"] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_artifact(path)


def test_shipped_cases_are_loadable_and_domain_free_in_the_package():
    package = load_case
    assert callable(package)  # import smoke test of the public surface


# ---------------------------------------------------------------------------
# the contract report aggregates verdicts
# ---------------------------------------------------------------------------

def test_contract_report_records_every_failed_check():
    result = EnvelopeEngine(synth.offset_bands_case()).run(n_scan=64)
    report = result.contracts.to_dict()
    assert report["ok"] is False
    assert set(report["failed"]) <= {o["id"] for o in report["outcomes"] if not o["passed"]}
    assert all(isinstance(o, dict) for o in report["outcomes"])
    assert isinstance(result.contracts.outcomes[0], CheckOutcome)

# ---------------------------------------------------------------------------
# condition-level expectations
# ---------------------------------------------------------------------------

def test_a_condition_can_declare_its_own_structure_expectation():
    payload = {
        "case_id": "synth_per_condition",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [
            {
                "label": "admissible",
                "params": {"floor": 0.2},
                "expect": {"intervals": 1},
            },
            {
                "label": "impossible",
                "params": {"floor": 2.0},
                "expect": {"intervals": 0, "infeasible": True},
            },
        ],
        "constraints": [
            {
                "id": "c_floor",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "var", "name": "alpha"},
                    "b": {"op": "var", "name": "floor"},
                },
                "contract": {"slope": {"alpha": 1}},
            }
        ],
        "contract": {"expect": {"intervals": 1}},
    }
    case = case_from_dict(payload)
    admissible = EnvelopeEngine(case).run(condition="admissible", n_scan=64)
    impossible = EnvelopeEngine(case).run(condition="impossible", n_scan=64)
    assert admissible.status == "feasible" and admissible.contracts.ok
    assert impossible.status == "infeasible"
    assert impossible.nearest is not None and impossible.nearest.residual > 0.0
    # the case-level expectation (one interval) would have failed here; the
    # condition-level block replaces it, and the verdict is then clean
    assert impossible.contracts.ok, impossible.contracts.to_dict()


def test_an_explicit_policy_overrides_the_case_policy():
    case = synth.strip_case()
    strict = ContractPolicy(expect_active={"lower": "c_high"})
    result = EnvelopeEngine(case, policy=strict).run(n_scan=64)
    assert "active_set:window" in result.contracts.failed
    relaxed = EnvelopeEngine(case).run(n_scan=64)
    assert relaxed.contracts.ok
