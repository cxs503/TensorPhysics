"""Black-box tests of the applicability-envelope mechanism.

Two groups:

* **synthetic capabilities** (:mod:`tensorphysics.applicability.benchmarks`) --
  pure algebra with closed-form boundaries, so the probe is graded against the
  truth rather than against itself.  Thirteen cases cover the three failure
  signals, both criterion forms and the edge semantics.
* **real capabilities** (the shipped configuration) -- the mechanism boundary of
  each registered capability must not be wider than the human-written
  declaration, and an out-of-domain query must be graded ``OUT_OF_DOMAIN``.

Axis names, spans, tolerances and capability ids are always injected as data:
nothing in this file is a physics statement.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tensorphysics.applicability import (ApplicabilityConfigError, AxisConfigError,
                                         CapabilityRegistry, DomainStatus, GradeWriteError,
                                         assign_grade, evaluate_point, probe_capability,
                                         probe_registry)
from tensorphysics.applicability.benchmarks import (SYNTHETIC_CASES, run_selfcheck,
                                                   synthetic_registry)

CONFIG_PATH = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorphysics"
               / "applicability" / "configs" / "default_capabilities.v1.json")


# ---------------------------------------------------------------------------
# synthetic capabilities: closed-form truth
# ---------------------------------------------------------------------------

def test_at_least_ten_synthetic_capabilities_are_exercised():
    assert len(SYNTHETIC_CASES) >= 10
    registry = synthetic_registry()
    assert len(registry) == len(SYNTHETIC_CASES)
    for case in SYNTHETIC_CASES:
        registry.get(case.capability_id)      # every case is registered


def test_synthetic_boundaries_match_closed_form_truth():
    result = run_selfcheck(rel_tol=1e-3)
    failed = [row["capability_id"] for row in result["cases"] if not row["passed"]]
    assert not failed, [(row["capability_id"], row["checks"]) for row in result["cases"]
                        if not row["passed"]]
    assert result["passed"] == result["total"]
    assert result["max_relative_error"] is not None
    assert result["max_relative_error"] < 0.03      # boundary error < 3 % of the truth


def test_boundary_refinement_reaches_a_tenth_of_a_percent_of_the_span():
    registry = synthetic_registry()
    report = probe_capability(registry.get("syn-square-window"))
    boundary = report.envelope.axes["x"].boundaries[0]
    assert boundary.converged
    assert boundary.span_fraction <= 1e-3
    assert boundary.confidence >= 0.9
    assert boundary.source == "criterion"


def test_quantised_axis_cannot_resolve_below_one_quantum():
    registry = synthetic_registry()
    report = probe_capability(registry.get("syn-quantised-axis"))
    axis_envelope = report.envelope.axes["n"]
    assert all(float(v).is_integer() for v in report.scans["n"].values)
    assert any("quantum" in note for note in axis_envelope.notes)
    assert axis_envelope.boundaries[0].width == pytest.approx(1.0)


def test_no_failure_in_span_never_extrapolates():
    registry = synthetic_registry()
    report = probe_capability(registry.get("syn-no-failure"))
    envelope = report.envelope
    verdict = envelope.classify({"x": 0.5})
    assert verdict.domain_status is DomainStatus.IN
    outside = envelope.classify({"x": 5.0})
    assert outside.domain_status is DomainStatus.UNKNOWN
    record = assign_grade(outside, "REAL")
    assert record.effective == "WITHHELD" and record.accepted is False


def test_empty_domain_forces_out_of_domain_grade():
    registry = synthetic_registry()
    spec = registry.get("syn-empty-domain")
    report = probe_capability(spec)
    assert report.envelope.axes["x"].intervals == ()
    verdict = report.envelope.classify({"x": 1.0}, spec=spec)
    assert verdict.domain_status is DomainStatus.OUT
    assert verdict.grade == "OUT_OF_DOMAIN"
    with pytest.raises(GradeWriteError):
        assign_grade(verdict, "REAL")
    relaxed = assign_grade(verdict, "REAL", strict=False)
    assert relaxed.effective == "OUT_OF_DOMAIN" and relaxed.accepted is False


def test_monotonicity_law_break_is_sampling_limited():
    registry = synthetic_registry()
    report = probe_capability(registry.get("syn-monotone-break"))
    axis_envelope = report.envelope.axes["x"]
    assert axis_envelope.samples_in == 1
    assert any("monotonicity law" in note for note in axis_envelope.notes)
    assert any(b.source == "law" for b in axis_envelope.boundaries)
    assert axis_envelope.confidence <= 0.5
    assert report.envelope.classify({"x": 1.5}).domain_status is DomainStatus.OUT


def test_sign_law_is_enforced_on_every_point_evaluation():
    registry = synthetic_registry()
    spec = registry.get("syn-sign-law")
    evaluation = evaluate_point(spec, {"x": 1.0})
    assert evaluation.status == "law"
    assert "sign law" in evaluation.reasons[0]


def test_two_axes_are_probed_independently():
    registry = synthetic_registry()
    report = probe_capability(registry.get("syn-two-axis-product"))
    assert report.envelope.classify({"x": 3.0, "y": 1.0}).domain_status is DomainStatus.OUT
    assert report.envelope.classify({"x": 1.0, "y": 1.0}).domain_status is DomainStatus.IN


def test_raising_evaluator_is_a_pathology():
    registry = synthetic_registry()
    report = probe_capability(registry.get("syn-singular-exception"))
    statuses = set(report.scans["x"].statuses)
    assert "pathology" in statuses
    assert "pass" in statuses


def test_probe_is_deterministic():
    registry = synthetic_registry()
    first = probe_capability(registry.get("syn-square-window")).envelope
    second = probe_capability(registry.get("syn-square-window")).envelope
    assert first.digest == second.digest
    assert first.axes["x"].to_dict() == second.axes["x"].to_dict()


# ---------------------------------------------------------------------------
# fail-closed configuration handling
# ---------------------------------------------------------------------------

def _base_capability() -> dict:
    return {
        "capability_id": "cfg-probe",
        "evaluator": {"target": "tensorphysics.applicability.metrics:power_law",
                      "kwargs": {"amplitude": 1.0, "exponent": 2.0, "offset": 0.0},
                      "axis_kwargs": {"x": "x"}},
        "axes": [{"name": "x", "low": 0.1, "high": 5.0, "samples": 9}],
        "policy": {"criterion": {"kind": "abs", "reference": 0.0, "tolerance": 1.0}},
    }


def _registry_of(*capabilities):
    return CapabilityRegistry.from_config({"schema_version": 1,
                                           "capabilities": list(capabilities)})


def test_unknown_keys_are_rejected():
    bad = _base_capability()
    bad["policy"]["criterion"]["toleranse"] = 1.0
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)
    bad = _base_capability()
    bad["surprise"] = True
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)
    bad = _base_capability()
    bad["evaluator"]["if_missing"] = "REAL"
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)


def test_schema_and_target_validation():
    good = _base_capability()
    with pytest.raises(ApplicabilityConfigError):
        CapabilityRegistry.from_config({"schema_version": 2, "capabilities": [good]})
    with pytest.raises(ApplicabilityConfigError):
        CapabilityRegistry.from_config({"schema_version": 1, "capabilities": []})
    bad = json.loads(json.dumps(good))
    bad["evaluator"]["target"] = "no.such.module:function"
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(good, json.loads(json.dumps(good)))


def test_every_axis_must_be_driven_by_the_evaluator():
    bad = _base_capability()
    bad["axes"].append({"name": "y", "low": 0.1, "high": 2.0})
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)
    bad = _base_capability()
    bad["declared"] = {"ghost": {"low": 0.0, "high": 1.0}}
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)
    bad = _base_capability()
    del bad["policy"]["criterion"]
    with pytest.raises(ApplicabilityConfigError):
        _registry_of(bad)


def test_axis_validation():
    bad = _base_capability()
    bad["axes"] = [{"name": "x", "low": 2.0, "high": 1.0}]
    with pytest.raises(AxisConfigError):
        _registry_of(bad)
    bad = _base_capability()
    bad["axes"] = [{"name": "x", "low": 0.0, "high": 1.0, "scale": "log"}]
    with pytest.raises(AxisConfigError):
        _registry_of(bad)
    bad = _base_capability()
    bad["axes"] = [{"name": "x", "low": 0.1, "high": 1.0, "samples": 3}]
    with pytest.raises(AxisConfigError):
        _registry_of(bad)


def test_unknown_capability_id_raises():
    registry = _registry_of(_base_capability())
    with pytest.raises(KeyError):
        registry.get("no-such-capability")


# ---------------------------------------------------------------------------
# real capabilities: mechanism boundary vs human declaration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def suite():
    registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
    return registry, probe_registry(registry)


def test_real_capabilities_have_an_envelope(suite):
    registry, probed = suite
    assert len(registry) >= 3
    for capability_id in registry.ids():
        report = probed.reports[capability_id]
        for name, axis_envelope in report.envelope.axes.items():
            assert axis_envelope.samples_total >= 5
            assert axis_envelope.samples_in >= 1, (capability_id, name)
            assert axis_envelope.intervals
            assert 0.0 <= axis_envelope.confidence <= 1.0


def test_real_capability_boundaries_are_not_wider_than_declarations(suite):
    registry, probed = suite
    checked = 0
    for capability_id in registry.ids():
        report = probed.reports[capability_id]
        assert report.compat is not None, capability_id
        assert report.compat.ok, (capability_id, report.compat.to_dict())
        assert report.compat.verdict in ("NOT_WIDER", "CONSERVATIVE")
        for compat_axis in report.compat.axes:
            assert compat_axis.wider_low_by == 0.0
            assert compat_axis.wider_high_by == 0.0
            if compat_axis.declared_low is not None:
                assert compat_axis.mechanism_low >= compat_axis.declared_low - compat_axis.slack
            if compat_axis.declared_high is not None:
                assert compat_axis.mechanism_high <= compat_axis.declared_high + compat_axis.slack
            checked += 1
    assert checked >= 3


def test_real_capability_out_of_domain_is_graded_out_of_domain(suite):
    registry, probed = suite
    checked = 0
    for capability_id in registry.ids():
        spec = registry.get(capability_id)
        envelope = probed.reports[capability_id].envelope
        axis_name = spec.axes[0].name
        verdict = envelope.classify({axis_name: spec.axes[0].high}, spec=spec)
        assert verdict.domain_status is DomainStatus.OUT
        assert verdict.grade == "OUT_OF_DOMAIN"
        with pytest.raises(GradeWriteError):
            assign_grade(verdict, "REAL")
        inside = envelope.axes[axis_name].intervals[0]
        midpoint = 0.5 * (inside.low + inside.high)
        assert envelope.classify({axis_name: midpoint}, spec=spec).domain_status is DomainStatus.IN
        checked += 1
    assert checked >= 3


def test_artifact_round_trip(suite):
    """The JSON artifact keeps every number the report quotes."""
    from tensorphysics.applicability.__main__ import build_artifact

    artifact, _suite, registry = build_artifact(
        CONFIG_PATH, queries={registry_id: {"damping": 1.5}
                              for registry_id in ("multigrid-contraction-vs-damping",)})
    assert artifact["schema"] == "tensorphysics.applicability.envelopes.v1"
    assert artifact["suite_digest"] and artifact["config_digest"]
    assert artifact["query_summary"]["total"] == 1
    assert artifact["query_summary"]["out_of_domain"] == 1
    grade = artifact["queries"][0]["grade"]
    assert grade["effective"] == "OUT_OF_DOMAIN" and grade["accepted"] is False
    json.dumps(artifact)          # JSON-serialisable end to end


def test_shipped_configuration_is_the_only_place_capability_names_live():
    """Zero-domain-semantics discipline: ids and axis names are data, not code."""
    registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
    package = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorphysics"
               / "applicability")
    code = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py")))
    for spec in registry:
        assert spec.capability_id not in code, spec.capability_id
        for axis_spec in spec.axes:
            assert f'"{axis_spec.name}"' not in code, (spec.capability_id, axis_spec.name)