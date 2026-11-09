"""Tests for the V&V core gate (m8768, re-open of PR #199).

The core gate composes the existing functional sub-gates (gci / validation /
cross_solver); these tests pin the composition semantics -- stage fusion,
provenance, JSON serialisability and error paths -- with analytic cases
whose expected numbers are computed by hand:

* ``f(h) = 1 + 0.01 h^2`` on ``h = 1, 1/2, 1/4`` gives an exact observed
  order ``p = 2``, Richardson value ``f_ext = 1.0`` and a finest-pair GCI of
  ``1.25 * (0.001875/1.000625) / 3 ~= 7.807e-4`` (relative).
"""

from __future__ import annotations

import json
import math

import pytest

from tensorphysics.vv.core import CoreVerdict, VVCore, evaluate_core
from tensorphysics.vv.cross_solver import SolverSample
from tensorphysics.vv.gci import FAIL, PASS, WARN, GatePolicy, Observation


def quad_levels(sigma: float | None = None) -> list[Observation]:
    """f(h) = 1 + 0.01 h^2 on h = 1, 1/2, 1/4 (exact p = 2, f_exact = 1)."""
    return [
        Observation(h=h, f=1.0 + 0.01 * h * h, sigma=sigma)
        for h in (1.0, 0.5, 0.25)
    ]


def oscillatory_levels() -> list[Observation]:
    return [Observation(h=1.0, f=1.0), Observation(h=0.5, f=1.1), Observation(h=0.25, f=1.05)]


def outlier_samples() -> list[SolverSample]:
    return [
        SolverSample(solver="LBM", f=1.000625, u=0.001),
        SolverSample(solver="FVM", f=1.000625, u=0.001),
        SolverSample(solver="FEM", f=1.05, u=0.001),
    ]


def warn_spread_samples() -> list[SolverSample]:
    # spread_rel = 0.024375 / 1.0128125 ~= 0.0241 in (0.02, 0.05) -> WARN
    return [
        SolverSample(solver="LBM", f=1.000625),
        SolverSample(solver="FVM", f=1.025),
    ]


# ---------------------------------------------------------------------------
# stage 1+2: convergence + validation
# ---------------------------------------------------------------------------


def test_pass_with_noise_certificate():
    verdict = evaluate_core(quad_levels(sigma=1e-6), f_exp=1.0)
    assert isinstance(verdict, CoreVerdict)
    assert verdict.status == PASS
    study = verdict.convergence.study
    assert study is not None
    assert study.order_p == pytest.approx(2.0, abs=1e-9)
    assert study.f_extrapolated == pytest.approx(1.0, abs=1e-12)
    assert study.gci_fine == pytest.approx(1.25 * 0.001875 / 1.000625 / 3.0, rel=1e-9)
    # validation stage: E = 0.000625, U_val = GCI * |f_fine|
    assert verdict.validation is not None
    assert verdict.validation.passed
    assert verdict.validation.ratio == pytest.approx(0.000625 / (study.gci_fine * abs(study.f_fine)), rel=1e-9)
    assert verdict.agreement is None


def test_oscillatory_sequence_fails():
    verdict = evaluate_core(oscillatory_levels())
    assert verdict.status == FAIL
    assert verdict.convergence.study is not None
    assert verdict.convergence.study.convergence_type == "oscillatory"
    assert any("oscillatory" in r for r in verdict.reasons)


def test_validation_fail_when_reference_far():
    verdict = evaluate_core(quad_levels(sigma=1e-6), f_exp=1.5)
    assert verdict.status == FAIL
    assert verdict.validation is not None
    assert not verdict.validation.passed
    assert verdict.validation.ratio > 100.0


def test_no_reference_no_validation_stage():
    verdict = evaluate_core(quad_levels(sigma=1e-6))
    assert verdict.validation is None
    assert verdict.status == PASS


def test_policy_passthrough_expected_order():
    policy = GatePolicy(expected_order=2.0)
    verdict = evaluate_core(quad_levels(sigma=1e-6), gate_policy=policy)
    names = [c["check"] for c in verdict.checks]
    assert "order-band/expected" in names
    check = next(c for c in verdict.checks if c["check"] == "order-band/expected")
    assert check["status"] == PASS
    assert verdict.status == PASS


# ---------------------------------------------------------------------------
# stage 3: cross-solver fusion
# ---------------------------------------------------------------------------


def test_cross_solver_fail_fuses_to_fail():
    verdict = evaluate_core(quad_levels(sigma=1e-6), f_exp=1.0, samples=outlier_samples())
    assert verdict.convergence.status == PASS
    assert verdict.agreement is not None
    assert verdict.agreement.status == FAIL
    assert verdict.status == FAIL
    # cross-solver checks are appended under the prefixed name
    assert "cross-solver/outlier" in [c["check"] for c in verdict.checks]
    assert "FEM" in verdict.agreement.outliers


def test_cross_solver_warn_fuses_to_warn():
    verdict = evaluate_core(quad_levels(sigma=1e-6), f_exp=1.0, samples=warn_spread_samples())
    assert verdict.convergence.status == PASS
    assert verdict.agreement is not None
    assert verdict.agreement.status == WARN
    assert verdict.status == WARN
    assert verdict.agreement.spread_rel == pytest.approx(0.024375 / 1.0128125, rel=1e-9)


def test_no_samples_no_agreement_stage():
    verdict = evaluate_core(quad_levels(sigma=1e-6))
    assert verdict.agreement is None
    assert verdict.status == verdict.convergence.status
    assert all(not c["check"].startswith("cross-solver/") for c in verdict.checks)


# ---------------------------------------------------------------------------
# standalone stages
# ---------------------------------------------------------------------------


def test_standalone_validation_stage():
    core = VVCore()
    ok = core.validation(1.000625, 1.0, u_num=0.001)
    assert ok.passed
    tight = core.validation(1.000625, 1.0, u_num=1e-6)
    assert not tight.passed
    exact = core.validation(1.0, 1.0, u_num=0.0)
    assert exact.passed


def test_standalone_validation_rejects_non_finite():
    core = VVCore()
    with pytest.raises(ValueError):
        core.validation(float("nan"), 1.0, u_num=0.001)


def test_cross_solver_stage_needs_two_solvers():
    core = VVCore()
    with pytest.raises(ValueError):
        core.cross_solver([SolverSample(solver="LBM", f=1.0)])


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_evaluate_core_matches_class_api():
    levels = quad_levels(sigma=1e-6)
    core = VVCore()
    assert core.run(levels, f_exp=1.0) == evaluate_core(levels, f_exp=1.0)


def test_as_dict_is_json_serialisable():
    verdict = evaluate_core(
        quad_levels(sigma=1e-6), f_exp=1.0, samples=outlier_samples()
    )
    payload = json.dumps(verdict.as_dict())
    doc = json.loads(payload)
    assert doc["schema"] == "tensorphysics.vv.core/1"
    assert doc["status"] == FAIL
    assert doc["convergence"]["status"] == PASS
    assert doc["agreement"]["status"] == FAIL
    assert doc["validation"]["status"] == "PASS"
    assert doc["reasons"] == list(verdict.reasons)
