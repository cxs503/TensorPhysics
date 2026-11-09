"""Unit tests of the three-state UQ gate (PASS / FAIL / WITHHELD).

The acceptance criteria are exercised end-to-end in ``test_uq_acceptance.py``;
this file tests the gate's decision logic directly with hand-built diagnostics so
that every branch is covered, including the ones a well-behaved model will never
trigger (algebra violations, contradicting estimators).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq import (
    FAIL,
    PASS,
    WITHHELD,
    CallableAdapter,
    InputModel,
    Normal,
    SobolIndices,
    UQDiagnostics,
    UQOptions,
    UQPolicy,
    UQStudy,
    Uniform,
    evaluate_gate,
)
from tensorphysics.uq.benchmarks import ishigami_function, ishigami_input_model
from tensorphysics.uq.gate import IndexComparison, _disputed_entries


def _indices(s1, st, low=None, high=None, lo_st=None, hi_st=None, names=("x1", "x2")) -> SobolIndices:
    return SobolIndices(
        names=list(names),
        S1=np.array(s1, dtype=float),
        ST=np.array(st, dtype=float),
        variance=1.0,
        estimator="pce",
        S1_low=None if low is None else np.array(low, dtype=float),
        S1_high=None if high is None else np.array(high, dtype=float),
        ST_low=None if lo_st is None else np.array(lo_st, dtype=float),
        ST_high=None if hi_st is None else np.array(hi_st, dtype=float),
    )


def _cross(s1, st, low, high, lo_st, hi_st) -> SobolIndices:
    idx = _indices(s1, st, low, high, lo_st, hi_st)
    idx.estimator = "saltelli-jansen"
    return idx


def _diag(indices, cross=None, comparison=None, **kwargs) -> UQDiagnostics:
    base = dict(
        n_samples=2048,
        dim=2,
        basis_size=15,
        n_terms=8,
        degree=4,
        loo_error=0.01,
        output_variance=1.0,
        indices=indices,
        cross_check=cross,
        comparison=comparison,
    )
    base.update(kwargs)
    return UQDiagnostics(**base)


def _comparison(ref, cand) -> IndexComparison:
    from tensorphysics.uq.indices import compare_indices

    return compare_indices(ref, cand, abs_floor=1e-3)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def test_policy_defaults_match_the_acceptance_thresholds() -> None:
    policy = UQPolicy()
    assert policy.max_loo_error == 0.1
    assert policy.max_cross_check_rel_dev == 0.05
    assert policy.min_oversampling == 2.0
    assert policy.required_samples(500) == 1000
    assert policy.required_samples(4) == 64  # the absolute floor wins below it


# ---------------------------------------------------------------------------
# PASS
# ---------------------------------------------------------------------------


def test_gate_passes_when_estimators_agree() -> None:
    ref = _indices([0.4, 0.2], [0.6, 0.4], [0.38, 0.18], [0.42, 0.22], [0.58, 0.38], [0.62, 0.42])
    cand = _cross([0.41, 0.195], [0.61, 0.39], [0.35, 0.13], [0.47, 0.25], [0.55, 0.33], [0.67, 0.45])
    verdict = evaluate_gate(_diag(ref, cand, _comparison(ref, cand)))
    assert verdict.status == PASS, verdict.messages
    assert verdict.usable


def test_gate_withholds_when_cross_check_was_required_but_missing() -> None:
    ref = _indices([0.4, 0.2], [0.6, 0.4])
    verdict = evaluate_gate(_diag(ref))
    assert verdict.status == WITHHELD
    assert verdict.reasons == ["CROSS_CHECK_MISSING"]


# ---------------------------------------------------------------------------
# WITHHELD branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, code",
    [
        (dict(nonfinite_output_count=1), "NONFINITE_MODEL_OUTPUT"),
        (dict(nonfinite_input_count=2), "NONFINITE_INPUT_SAMPLES"),
        (dict(adapter_failures=3), "ADAPTER_FAILURES"),
        (dict(n_samples=32), "INSUFFICIENT_SAMPLES"),
        (dict(basis_size=2000, n_samples=2048), "INSUFFICIENT_SAMPLES"),
        (dict(loo_error=0.25), "LOO_ERROR_ABOVE_THRESHOLD"),
        (dict(loo_error=float("nan")), "LOO_ERROR_UNAVAILABLE"),
        (dict(output_variance=0.0), "DEGENERATE_OUTPUT_VARIANCE"),
        (dict(correlated_inputs=True), "CORRELATED_INPUTS_UNSUPPORTED"),
    ],
)
def test_gate_withholds_on_inadmissible_evidence(kwargs, code) -> None:
    ref = _indices([0.4, 0.2], [0.6, 0.4])
    cand = _cross([0.4, 0.2], [0.6, 0.4], [0.3, 0.1], [0.5, 0.3], [0.5, 0.3], [0.7, 0.5])
    verdict = evaluate_gate(_diag(ref, cand, _comparison(ref, cand), **kwargs))
    assert verdict.status == WITHHELD
    assert code in verdict.reasons
    assert not verdict.usable


def test_gate_marks_an_imprecise_cross_check_inconclusive() -> None:
    ref = _indices([0.40, 0.20], [0.60, 0.40])
    # Monte-Carlo estimate far off, but its bootstrap interval covers the PCE value
    cand = _cross([0.20, 0.05], [0.60, 0.40], [0.05, -0.05], [0.45, 0.25], [0.40, 0.25], [0.80, 0.55])
    verdict = evaluate_gate(_diag(ref, cand, _comparison(ref, cand)))
    assert verdict.status == WITHHELD
    assert "CROSS_CHECK_INCONCLUSIVE" in verdict.reasons


# ---------------------------------------------------------------------------
# FAIL branches
# ---------------------------------------------------------------------------


def test_gate_fails_when_estimators_contradict_each_other() -> None:
    ref = _indices([0.40, 0.20], [0.60, 0.40])
    # Monte-Carlo estimate far off *and* its interval excludes the PCE value
    cand = _cross([0.10, 0.05], [0.60, 0.40], [0.06, 0.01], [0.14, 0.09], [0.50, 0.30], [0.70, 0.50])
    verdict = evaluate_gate(_diag(ref, cand, _comparison(ref, cand)))
    assert verdict.status == FAIL
    assert "PCE_MC_ESTIMATORS_DISAGREE" in verdict.reasons
    assert not verdict.usable


def test_gate_fails_on_variance_algebra_violation() -> None:
    ref = _indices([0.40, 0.20], [0.30, 0.10])  # ST < S1 for both parameters
    cand = _cross([0.40, 0.20], [0.30, 0.10], [0.30, 0.10], [0.50, 0.30], [0.25, 0.05], [0.35, 0.15])
    cmp_ = IndexComparison(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, {"per_parameter": []})
    verdict = evaluate_gate(_diag(ref, cand, cmp_))
    assert verdict.status == FAIL
    assert "INDEX_ALGEBRA_VIOLATION" in verdict.reasons


def test_gate_fails_on_sum_of_first_order_above_one() -> None:
    ref = _indices([0.7, 0.6], [0.8, 0.7])
    cmp_ = IndexComparison(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, {"per_parameter": []})
    cand = _cross([0.7, 0.6], [0.8, 0.7], [0.6, 0.5], [0.8, 0.7], [0.7, 0.6], [0.9, 0.8])
    verdict = evaluate_gate(_diag(ref, cand, cmp_))
    assert verdict.status == FAIL
    assert "INDEX_ALGEBRA_VIOLATION" in verdict.reasons


def test_gate_fails_on_negative_index() -> None:
    ref = _indices([-0.10, 0.90], [0.10, 0.90])
    cand = _cross([-0.10, 0.90], [0.10, 0.90], [-0.2, 0.8], [0.0, 1.0], [0.0, 0.8], [0.2, 1.0])
    cmp_ = IndexComparison(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, {"per_parameter": []})
    verdict = evaluate_gate(_diag(ref, cand, cmp_))
    assert verdict.status == FAIL
    assert "SIGNIFICANTLY_NEGATIVE_INDEX" in verdict.reasons


def test_disputed_entries_skip_negligible_indices() -> None:
    ref = _indices([0.0, 0.5], [0.5, 0.5])
    cand = _cross([0.004, 0.5], [0.5, 0.5], [0.0, 0.4], [0.01, 0.6], [0.4, 0.4], [0.6, 0.6])
    assert _disputed_entries(ref, cand, UQPolicy()) == []


# ---------------------------------------------------------------------------
# the redundant-engine checks: the same policy through the study driver
# ---------------------------------------------------------------------------


def test_study_withholds_on_correlated_inputs() -> None:
    model = InputModel(
        ["a", "b"],
        [Normal(0.0, 1.0), Normal(0.0, 1.0)],
        correlation=np.array([[1.0, 0.7], [0.7, 1.0]]),
    )
    adapter = CallableAdapter(lambda X: X[:, 0] + X[:, 1], model.names)
    result = UQStudy(model, adapter, UQOptions(n_samples=256, degrees=(2, 3), cross_check=False)).run()
    assert result.verdict.status == WITHHELD
    assert "CORRELATED_INPUTS_UNSUPPORTED" in result.verdict.reasons


def test_study_records_the_sample_digest_in_the_verdict() -> None:
    model = ishigami_input_model()
    adapter = CallableAdapter(ishigami_function, model.names)
    result = UQStudy(model, adapter, UQOptions(n_samples=256, degrees=(4,), cross_check=False)).run()
    extra = result.verdict.diagnostics["extra"]
    assert extra["sample_digest_sha256"] == result.samples.digest()
    assert extra["physical_digest_sha256"] == result.samples.physical_digest()
    assert len(extra["sample_digest_sha256"]) == 64


def test_gate_str_is_readable() -> None:
    ref = _indices([0.4, 0.2], [0.6, 0.4])
    verdict = evaluate_gate(_diag(ref))
    assert str(verdict).startswith("WITHHELD")