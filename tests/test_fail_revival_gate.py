"""Zero-rewrite gate tests: ten rewrite attempts, one control, five reason codes."""
from __future__ import annotations

from pathlib import Path

import pytest

from tensorphysics.fail_revival import gate, synth
from tensorphysics.fail_revival.extract import extract_artifact, load_layout
from tensorphysics.fail_revival.scheduler import baseline_input_digest

LAYOUT = load_layout()


@pytest.fixture()
def registry():
    art = synth.corpus()[0]
    result = extract_artifact(art, layout=LAYOUT)
    return {r.fail_id: r for r in result.records}


def test_ten_rewrite_attempts_are_refused_with_the_right_reason(registry):
    rec = next(iter(registry.values()))
    cases = synth.tamper_cases(rec.fail_id, rec.criterion_sha256, baseline_input_digest(rec))
    assert len(cases) == 10
    outcomes = [
        (case.name, gate.zero_rewrite_gate(gate.request_from_payload(case.payload), registry))
        for case in cases
    ]
    assert all(not d.accepted for _, d in outcomes)
    mismatched = [(name, case.expected_reason, d.reason)
                  for (name, d), case in zip(outcomes, cases) if d.reason != case.expected_reason]
    assert mismatched == []
    assert len({d.reason for _, d in outcomes}) >= 5


def test_admissible_request_is_accepted(registry):
    rec = next(iter(registry.values()))
    case = synth.admissible_case(rec.fail_id, rec.criterion_sha256, baseline_input_digest(rec))
    decision = gate.zero_rewrite_gate(gate.request_from_payload(case.payload), registry)
    assert decision.accepted
    assert decision.reason == gate.ACCEPTED
    assert decision.criterion_sha256 == rec.criterion_sha256


def test_verdict_injection_gets_its_own_code(registry):
    rec = next(iter(registry.values()))
    payload = {
        "fail_key": rec.fail_id,
        "criterion_sha256": rec.criterion_sha256,
        "input_sha256": baseline_input_digest(rec),
        "verdict": "PASS",
    }
    decision = gate.zero_rewrite_gate(gate.request_from_payload(payload), registry)
    assert not decision.accepted
    assert decision.reason == gate.VERDICT_INJECTION_FORBIDDEN


def test_relaxed_tolerance_is_refused_even_with_correct_digest(registry):
    rec = next(iter(registry.values()))
    payload = {
        "fail_key": rec.fail_id,
        "criterion_sha256": rec.criterion_sha256,
        "input_sha256": baseline_input_digest(rec),
        "tolerance": 0.30,
    }
    decision = gate.zero_rewrite_gate(gate.request_from_payload(payload), registry)
    assert not decision.accepted
    assert decision.reason == gate.EDITABLE_FIELD_FORBIDDEN


def test_unregistered_fail_key_is_refused(registry):
    rec = next(iter(registry.values()))
    payload = {
        "fail_key": "FR-unknown",
        "criterion_sha256": rec.criterion_sha256,
        "input_sha256": baseline_input_digest(rec),
    }
    decision = gate.zero_rewrite_gate(gate.request_from_payload(payload), registry)
    assert decision.reason == gate.UNREGISTERED_FAIL


def test_decision_is_serialisable(registry):
    rec = next(iter(registry.values()))
    payload = {"fail_key": rec.fail_id, "criterion_sha256": rec.criterion_sha256,
               "input_sha256": baseline_input_digest(rec)}
    d = gate.zero_rewrite_gate(gate.request_from_payload(payload), registry).to_dict()
    assert d["schema"] == gate.GATE_SCHEMA
    assert d["accepted"] is True