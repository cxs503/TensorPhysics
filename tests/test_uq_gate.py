"""Tests of the gate --- pre-evidence record, error gate, ASME chain, injections.

Two properties are under test:

1. **the falsification matrix**: no injected defect may produce ``PASS``, and
   each injection must be refused for its *own* reason (a case that is withheld
   for the wrong reason is a silent gap in the gate);
2. **the pre-evidence contract**: a claim cannot be certified without an
   admissible UQ record, and the record is digest-addressed so it can be cited.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tensorphysics.uq import (
    FAIL,
    PASS,
    WITHHELD,
    UQEvidenceError,
    chain_asme_validation,
    engineering_error_gate,
    holtrop_type_benchmark,
    injection_suite,
    ishigami,
    ishigami_spec,
    pre_evidence,
    require_pass,
    run_uq_study,
)
from tensorphysics.uq.gate import canonical_digest


def _narrow_study(**overrides):
    model = lambda points: 1000.0 + 5.0 * ishigami(points)  # noqa: E731
    kwargs = dict(
        n_samples=256,
        degree=8,
        n_boot=0,
        n_base_mc=1 << 13,
        bootstrap_mc=0,
        budget_n_mc=4000,
    )
    kwargs.update(overrides)
    return run_uq_study(model, ishigami_spec(), **kwargs)


# ---------------------------------------------------------------------------
# falsification matrix
# ---------------------------------------------------------------------------
def test_injection_suite_passes_and_its_control_is_a_clean_study() -> None:
    report = injection_suite()
    assert report.control_passes, report.control_codes
    assert report.all_safe, [c.as_dict() for c in report.cases if not c.safe]
    assert report.all_named, [c.as_dict() for c in report.cases if not c.named]
    assert report.passed
    codes = report.code_union()
    for required in ("UQ-W01-NONFINITE", "UQ-W02-TOOFEW", "UQ-W03-LOO",
                     "UQ-W04-CROSSCHECK", "UQ-W05-DEGENERATE", "UQ-W06-HASH",
                     "UQ-W07-NOMC", "UQ-W08-NOPCE"):
        assert required in codes


def test_no_injection_produces_a_pass() -> None:
    report = injection_suite()
    passes = [c.case_id for c in report.cases if c.observed_status == PASS]
    # only the threshold knife-edge control may resolve to PASS
    assert passes in ([], ["V5a-threshold-exact"]), passes

    defected = [
        c
        for c in report.cases
        if not c.case_id.startswith("V5") and c.case_id != "V6-wide-band"
    ]
    for case in defected:
        assert case.observed_status == WITHHELD, case.as_dict()
    # a band above 15 % is a *decision*, not missing evidence: it must read FAIL
    wide_band = next(c for c in report.cases if c.case_id == "V6-wide-band")
    assert wide_band.observed_status == FAIL
    knife = next(c for c in report.cases if c.case_id == "V5b-threshold-epsilon-above")
    assert knife.observed_status == FAIL


# ---------------------------------------------------------------------------
# pre-evidence record
# ---------------------------------------------------------------------------
def test_pre_evidence_is_json_serialisable_and_digest_stable() -> None:
    study = _narrow_study()
    record = pre_evidence(study, label="narrow")
    payload = record.as_dict()
    assert json.loads(json.dumps(payload))["status"] == study.status
    assert payload["sample_hash"] == study.samples.hash
    assert payload["spec_fingerprint"] == study.spec.fingerprint()
    assert payload["n_samples"] == 256
    assert payload["solver_evaluations"] >= 256
    assert payload["digest"] == canonical_digest(record.payload())
    # the same study rebuilds the same digest (deterministic payload)
    assert pre_evidence(study, label="narrow").digest == record.digest
    # a different label is a different record
    assert pre_evidence(study, label="other").digest != record.digest


def test_gate_passes_only_with_admissible_evidence() -> None:
    study = _narrow_study()
    assert study.status == PASS
    decision = engineering_error_gate(study=study, relative_error=0.02)
    assert decision.passed
    assert decision.codes == ()
    assert decision.u_input > 0.0
    assert decision.evidence_digest == pre_evidence(study).digest


def test_gate_withholds_without_evidence() -> None:
    decision = engineering_error_gate(study=None, relative_error=0.001)
    assert decision.status == WITHHELD
    assert decision.codes == ("UQ-G01-NO-EVIDENCE",)
    assert decision.evidence_digest == ""


def test_gate_propagates_a_withheld_study() -> None:
    withdrawn = _narrow_study(run_mc=False)  # missing second opinion
    assert withdrawn.status == WITHHELD
    decision = engineering_error_gate(study=withdrawn, relative_error=0.001)
    assert decision.status == WITHHELD
    assert "UQ-G02-UQ-WITHHELD" in decision.codes


def test_gate_fails_on_the_engineering_threshold() -> None:
    study = _narrow_study()
    decision = engineering_error_gate(study=study, relative_error=0.20)
    assert decision.status == FAIL
    assert decision.codes == ("UQ-G03-ERROR",)
    # exactly at the threshold the claim is still admissible (">15 %" is the rule)
    boundary = engineering_error_gate(study=study, relative_error=0.15)
    assert boundary.status == PASS


def test_gate_adjudicates_a_band_wider_than_the_threshold() -> None:
    wide = holtrop_type_benchmark(rel_uncertainty=0.04)
    study = run_uq_study(
        wide.resistance_kn,
        wide.spec,
        n_samples=256,
        degree=4,
        n_boot=0,
        run_mc=False,
        require_cross_check=False,
        budget_n_mc=4000,
    )
    assert study.status == FAIL
    decision = engineering_error_gate(study=study)
    assert decision.status == FAIL
    assert decision.codes == ("UQ-G04-BAND",)
    assert decision.band_relative_width > 0.15


def test_require_pass_raises_with_the_reason() -> None:
    decision = engineering_error_gate(study=None)
    with pytest.raises(UQEvidenceError) as excinfo:
        require_pass(decision)
    assert "UQ-G01-NO-EVIDENCE" in str(excinfo.value)
    assert require_pass(engineering_error_gate(study=_narrow_study())).passed


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError):
        engineering_error_gate(study=None, threshold=0.0)
    with pytest.raises(ValueError):
        engineering_error_gate(study=None, relative_error=float("nan"))


# ---------------------------------------------------------------------------
# ASME chain
# ---------------------------------------------------------------------------
def test_chain_asme_validation_exports_the_uq_half_width() -> None:
    bench = holtrop_type_benchmark()
    study = run_uq_study(
        bench.resistance_kn,
        bench.spec,
        n_samples=256,
        degree=4,
        n_boot=0,
        n_base_mc=1 << 13,
        bootstrap_mc=100,
        budget_n_mc=8000,
    )
    assert study.status == PASS, study.verdict.reason_codes
    anchor = float(bench.reference_case["RT_kN"])
    chained = chain_asme_validation(
        study=study,
        f_sim=float(study.budget.mean),
        f_exp=anchor,
        u_num=float(study.surrogate.eps_loo_abs),
        label="holtrop",
    )
    assert chained["asme"]["u_input"] == pytest.approx(0.5 * study.budget.width)
    assert chained["asme"]["u_val"] == pytest.approx(
        np.hypot(chained["asme"]["u_num"], chained["asme"]["u_input"])
    )
    assert chained["combined_status"] == PASS
    assert chained["gate"]["status"] == PASS
    assert chained["evidence"]["digest"] == pre_evidence(study, label="holtrop").digest


def test_chain_is_withheld_when_the_uq_side_is_withheld() -> None:
    bench = holtrop_type_benchmark()
    study = run_uq_study(
        bench.resistance_kn,
        bench.spec,
        n_samples=128,
        degree=4,
        n_boot=0,
        run_mc=False,
        require_cross_check=False,
        budget_n_mc=4000,
    )
    assert study.status in (WITHHELD, FAIL)
    chained = chain_asme_validation(
        study=study,
        f_sim=float(study.budget.mean),
        f_exp=float(bench.reference_case["RT_kN"]),
        u_num=1.0,
    )
    assert chained["combined_status"] in (WITHHELD, FAIL)