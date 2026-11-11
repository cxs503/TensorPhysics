"""Acceptance tests of the joint (multi-axis) applicability probe.

The behaviour under test is a fail-closed requirement, not a numerical nicety:
an offline gate built from a one-axis-only envelope writes ``REAL`` for points
the capability fails as soon as two parameters are unfavourable *together*.
These tests pin the defect (baseline), the fix (matured), its soundness (every
box contains a witnessed failure), the conservation of the per-axis results
(bit-identical envelopes) and the fail-closed configuration handling.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from tensorphysics.applicability.artifact import gate, load_envelopes, suite_digest
from tensorphysics.applicability.capability import ApplicabilityConfigError, \
    CapabilityRegistry, InteractionConfig
from tensorphysics.applicability.envelope import DomainStatus, GradeWriteError
from tensorphysics.applicability.evaluation import evaluate_point
from tensorphysics.applicability.interaction import INTERACTION_CASES, \
    synthetic_interaction_registry
from tensorphysics.applicability.probe import probe_capability, probe_registry
from tensorphysics.applicability.verification import verify_c6_interaction_coverage

PAIR_CASES = list(INTERACTION_CASES)


def _probe(case, *, samples=None, interaction=None):
    registry = synthetic_interaction_registry()
    spec = registry.get(case.capability_id)
    if interaction is not None:
        spec = replace(spec, interaction=interaction)
    return spec, probe_capability(spec, interaction_samples=samples).envelope


def _single_axis_config() -> dict:
    return {"schema_version": 1, "capabilities": [{
        "capability_id": "invariance-probe",
        "evaluator": {"target": "tensorphysics.applicability.metrics:power_law",
                      "kwargs": {"amplitude": 1.0, "exponent": 2.0, "offset": 0.0},
                      "axis_kwargs": {"x": "x"}},
        "axes": [{"name": "x", "low": 0.1, "high": 4.0, "samples": 11}],
        "policy": {"criterion": {"kind": "abs", "reference": 0.0, "tolerance": 4.0}},
    }]}


# ---------------------------------------------------------------------------
# the defect (baseline) and the fix (matured)
# ---------------------------------------------------------------------------

def test_one_axis_only_envelope_certifies_a_failing_corner_as_real():
    """Baseline: the shipped one-axis-only envelope writes REAL at a failing point."""
    case = PAIR_CASES[0]
    spec, envelope = _probe(case, interaction=InteractionConfig(enabled=False))
    first, second = case.truth["axes"]
    point = {first: spec.axis(first).high, second: spec.axis(second).high}
    assert not case.truth_ok(point)                      # closed-form: outside
    assert envelope.interactions == ()                   # nothing joint was probed
    verdict = envelope.classify(point, verify=False)     # the offline gate path
    assert verdict.domain_status is DomainStatus.IN
    record = gate({case.capability_id: envelope}, case.capability_id, point, "REAL",
                  strict=False)
    assert record.effective == "REAL" and record.accepted is True


def test_matured_envelope_refuses_the_same_point_fail_closed():
    case = PAIR_CASES[0]
    spec, envelope = _probe(case)
    first, second = case.truth["axes"]
    point = {first: spec.axis(first).high, second: spec.axis(second).high}
    verdict = envelope.classify(point, verify=False)
    assert verdict.domain_status is DomainStatus.OUT
    assert verdict.grade == "OUT_OF_DOMAIN"
    assert "interaction" in verdict.axis_status
    assert verdict.axis_status["interaction"]["failure_type"] == "criterion"
    record = gate({case.capability_id: envelope}, case.capability_id, point, "REAL",
                  strict=False)
    assert record.effective == "OUT_OF_DOMAIN" and record.accepted is False
    with pytest.raises(GradeWriteError):
        gate({case.capability_id: envelope}, case.capability_id, point, "REAL", strict=True)


@pytest.mark.parametrize("case", PAIR_CASES, ids=[c.capability_id for c in PAIR_CASES])
def test_joint_probe_certifies_no_failing_lattice_point(case):
    """The hardened gate never returns IN for a point the closed form rejects."""
    spec, envelope = _probe(case)
    first, second = case.truth["axes"]
    axis_a, axis_b = spec.axis(first), spec.axis(second)
    n = 41
    certified = 0
    for i in range(n):
        for j in range(n):
            point = {first: axis_a.low + (axis_a.high - axis_a.low) * i / (n - 1),
                     second: axis_b.low + (axis_b.high - axis_b.low) * j / (n - 1)}
            inside = envelope.classify(point, verify=False).domain_status is DomainStatus.IN
            if inside and not case.truth_ok(point):
                certified += 1
    assert certified == 0


@pytest.mark.parametrize("case", PAIR_CASES, ids=[c.capability_id for c in PAIR_CASES])
def test_boxes_are_sound_every_box_contains_a_witnessed_failure(case):
    """A box may only claim OUT because a sampled failure really lives inside it."""
    spec, envelope = _probe(case)
    assert envelope.interactions, "a joint case must produce joint regions"
    for region in envelope.interactions:
        assert region.witnesses > 0 and region.boxes
        for box in region.boxes:
            point = dict(box.witness)
            evaluation = evaluate_point(spec, point)
            assert not evaluation.ok, "box witness must fail"
            assert box.contains(point)
            assert evaluation.status == box.status


def test_conservativeness_shrinks_with_the_joint_lattice():
    """Over-refusal is the price of fail-closedness; it must decay with the lattice."""
    case = PAIR_CASES[0]
    spec, coarse = _probe(case, samples=9)
    _, fine = _probe(case, samples=25)
    first, second = case.truth["axes"]
    axis_a, axis_b = spec.axis(first), spec.axis(second)

    def over_refusal(envelope):
        refused = total = 0
        n = 41
        for i in range(n):
            for j in range(n):
                point = {first: axis_a.low + (axis_a.high - axis_a.low) * i / (n - 1),
                         second: axis_b.low + (axis_b.high - axis_b.low) * j / (n - 1)}
                if not case.truth_ok(point):
                    continue
                total += 1
                refused += (envelope.classify(point, verify=False).domain_status
                            is not DomainStatus.IN)
        return refused / max(1, total)

    assert over_refusal(fine) < over_refusal(coarse)
    assert sum(r.nodes_evaluated for r in fine.interactions) > \
        sum(r.nodes_evaluated for r in coarse.interactions)


# ---------------------------------------------------------------------------
# consumption: persisted artifact, offline gate, digest
# ---------------------------------------------------------------------------

def test_joint_regions_survive_the_artifact_round_trip():
    case = PAIR_CASES[2]
    spec, envelope = _probe(case)
    artifact = {"schema": "tensorphysics.applicability.envelopes.v1",
                "suite_digest": suite_digest({case.capability_id: envelope}),
                "capabilities": [{"capability_id": case.capability_id,
                                  "envelope": envelope.to_dict()}]}
    loaded = load_envelopes(artifact)          # verifies the digest
    assert loaded[case.capability_id].to_dict() == envelope.to_dict()
    first, second = case.truth["axes"]
    point = {first: spec.axis(first).high, second: spec.axis(second).high}
    assert not case.truth_ok(point)
    assert gate(loaded, case.capability_id, point, strict=False).effective == "OUT_OF_DOMAIN"


def test_artifact_without_joint_regions_still_loads():
    """An artifact written before the joint probe existed keeps working."""
    case = PAIR_CASES[1]
    _, envelope = _probe(case)
    payload = envelope.to_dict()
    payload.pop("interactions")
    rebuilt = type(envelope).from_dict(payload)
    assert rebuilt.interactions == ()
    first, second = case.truth["axes"]
    # without joint regions the per-axis rule alone decides (the legacy behaviour)
    assert rebuilt.classify({first: 1.0, second: 1.0},
                            verify=False).domain_status in (DomainStatus.IN, DomainStatus.OUT)


def test_joint_probe_is_deterministic_and_feeds_the_suite_digest():
    case = PAIR_CASES[0]
    _, first = _probe(case)
    _, second = _probe(case)
    assert [r.to_dict() for r in first.interactions] == \
        [r.to_dict() for r in second.interactions]
    registry = synthetic_interaction_registry()
    suite_a = probe_registry(registry)
    suite_b = probe_registry(registry)
    assert suite_a.digest == suite_b.digest


# ---------------------------------------------------------------------------
# the per-axis results must not move
# ---------------------------------------------------------------------------

def test_single_axis_capabilities_get_no_joint_regions_and_keep_their_envelope():
    registry = CapabilityRegistry.from_config(_single_axis_config())
    spec = registry.get("invariance-probe")
    joint = probe_capability(spec).envelope
    plain = probe_capability(replace(spec, interaction=InteractionConfig(enabled=False))).envelope
    assert joint.interactions == ()
    assert joint.axes["x"].to_dict() == plain.axes["x"].to_dict()


# ---------------------------------------------------------------------------
# fail-closed configuration handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"samples": 4},
    {"samples": True},
    {"max_pairs": 0},
    {"max_rays": -1},
    {"enabled": "yes"},
    {"unknown_knob": 1},
])
def test_malformed_interaction_config_is_rejected(bad):
    entry = copy.deepcopy(_single_axis_config()["capabilities"][0])
    entry["interaction"] = bad
    with pytest.raises(ApplicabilityConfigError):
        CapabilityRegistry.from_config({"schema_version": 1, "capabilities": [entry]})


def test_interaction_defaults_are_configuration_data():
    assert InteractionConfig().to_dict() == {"enabled": True, "samples": 13,
                                             "max_pairs": 3, "max_rays": 24}
    entry = copy.deepcopy(_single_axis_config()["capabilities"][0])
    entry["interaction"] = {"samples": 7, "max_rays": 0}
    registry = CapabilityRegistry.from_config({"schema_version": 1, "capabilities": [entry]})
    spec = registry.get("invariance-probe")
    assert spec.interaction.samples == 7 and spec.interaction.max_rays == 0
    assert spec.digest_material()["interaction"]["samples"] == 7


# ---------------------------------------------------------------------------
# the machine criterion of the node
# ---------------------------------------------------------------------------

def test_c6_reports_the_defect_and_the_zero_false_certification():
    result = verify_c6_interaction_coverage()
    assert result.passed
    assert result.metric_value == 0
    baseline = sum(int(d["rates"]["baseline"]["false_in"])
                   for d in result.details if "rates" in d)
    matured = sum(int(d["rates"]["matured"]["false_in"])
                  for d in result.details if "rates" in d)
    assert baseline > 1000 and matured == 0
    case_rows = [d for d in result.details if "rates" in d]
    assert all(row["passed"] for row in case_rows)
    assert len(case_rows) >= 3
    assert result.details[-1]["real_capability_invariance"]
    assert all(item["passed"] for item in result.details[-1]["real_capability_invariance"])