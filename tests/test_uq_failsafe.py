"""Fail-safe policy: predeclared PCE quality floor and MC degradation.

Acceptance item F4: when the sample budget is too small for the chosen chaos
truncation, the study must switch to the quasi-Monte-Carlo route automatically
and still return indices within 10 % of the analytic ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.benchmarks import ishigami, ishigami_analytic, ishigami_spec
from tensorphysics.uq.failsafe import (
    DEFAULT_Q2_MIN,
    MODE_MONTECARLO,
    MODE_PCE,
    assess,
    direct_budget,
    indices_from_mc,
    loo_q2,
    run_with_failsafe,
)
from tensorphysics.uq.pce import PCESurrogate
from tensorphysics.uq.sampler import sample
from tensorphysics.uq.sobol_mc import saltelli_indices
from tensorphysics.uq.workflow import run_uq_study


def _fake_surrogate(eps_loo: float, n_terms: int = 3, n_samples: int = 128) -> PCESurrogate:
    return PCESurrogate(
        degree=2,
        families=("legendre", "legendre", "legendre"),
        multi_index=np.zeros((1, 3), dtype=int),
        coef=np.zeros(1),
        active=(0,),
        eps_loo=float(eps_loo),
        eps_loo_abs=float(eps_loo),
        rmse=float(eps_loo),
        r2=0.0,
        n_samples=int(n_samples),
        n_terms=int(n_terms),
        method="lars",
    )


def test_loo_q2_is_one_for_perfect_surrogate():
    assert loo_q2(_fake_surrogate(0.0)) == pytest.approx(1.0)
    assert loo_q2(None) is None


def test_loo_q2_maps_eps_loo_to_coefficient_of_determination():
    assert loo_q2(_fake_surrogate(0.5)) == pytest.approx(1.0 - 0.25)


def test_assess_accepts_good_surrogate():
    decision = assess(_fake_surrogate(0.05), n_valid=512, n_samples=512, dim=3)
    assert decision.mode == MODE_PCE
    assert decision.triggered is False
    assert decision.q2_threshold_met is True
    assert decision.reasons == ()


def test_assess_triggers_when_fit_failed():
    decision = assess(
        None,
        n_valid=8,
        n_samples=8,
        dim=3,
        fit_error="ValueError: ols needs at least P=286 samples",
    )
    assert decision.mode == MODE_MONTECARLO
    assert "PCE-FIT-FAILED" in decision.reasons
    assert decision.as_dict()["fit_error"].startswith("ValueError")


def test_assess_triggers_below_quality_floor():
    decision = assess(_fake_surrogate(0.95), n_valid=512, n_samples=512, dim=3, q2_min=0.5)
    assert decision.mode == MODE_MONTECARLO
    assert "PCE-Q2-BELOW-FLOOR" in decision.reasons
    assert decision.q2_threshold_met is False


def test_assess_triggers_on_sample_shortage_even_when_loo_is_optimistic():
    """Ten samples for a degree-8 3-D truncation cannot support the surrogate."""
    decision = assess(_fake_surrogate(0.001), n_valid=10, n_samples=10, dim=3)
    assert decision.mode == MODE_MONTECARLO
    assert "PCE-TOOFEW-SAMPLES" in decision.reasons


def test_assess_rejects_invalid_floor():
    with pytest.raises(ValueError):
        assess(_fake_surrogate(0.1), n_valid=100, n_samples=100, q2_min=0.0)
    with pytest.raises(ValueError):
        assess(_fake_surrogate(0.1), n_valid=100, n_samples=100, q2_min=1.0)


def test_indices_from_mc_is_a_promotion_not_a_reinterpretation():
    mc = saltelli_indices(lambda u: u[:, 0] + 0.5 * u[:, 1], 2, n_base=1024, seed=3)
    ind = indices_from_mc(mc)
    np.testing.assert_allclose(ind.first, mc.first)
    np.testing.assert_allclose(ind.total, mc.total)
    assert ind.mean == pytest.approx(mc.mean)


def test_direct_budget_brackets_analytic_interval():
    # y = 10 x with x ~ U(1, 2): the analytic 95 % interval is 10*[1.025, 1.975]
    spec = ishigami_spec()
    seen = {}

    class _Solver:
        def evaluate(self, points):
            seen["n"] = points.shape[0]
            return points[:, 0] + points[:, 1]

    budget = direct_budget(_Solver(), spec, n_mc=4096, seed=7)
    assert seen["n"] == 4096
    assert budget.lower95 < budget.upper95
    assert budget.definition.endswith("failsafe")


def test_run_with_failsafe_degrades_and_stays_within_ten_percent():
    study = run_with_failsafe(
        ishigami,
        ishigami_spec(),
        n_samples=10,
        degree=8,
        seed=7,
        failsafe_n_base=2048,
        failsafe_budget_n_mc=512,
    )
    assert study.mode == MODE_MONTECARLO
    assert study.decision.triggered is True
    assert study.status in {"PASS", "FAIL"}  # the band is adjudicated, not withheld
    analytic = ishigami_analytic()
    first_err = float(np.max(np.abs(study.indices.first - analytic["first"])))
    total_err = float(np.max(np.abs(study.indices.total - analytic["total"])))
    assert max(first_err, total_err) <= 0.10, (first_err, total_err)
    codes = [r.code for r in study.verdict.reasons]
    assert "UQ-N05-MC-FAILSAFE" in codes
    payload = study.as_dict()
    assert payload["mode"] == MODE_MONTECARLO
    assert payload["metadata"]["n_mc_evaluations"] > 0
    assert payload["study"]["status"] == study.study.status


def test_run_with_failsafe_keeps_pce_when_quality_is_good():
    study = run_with_failsafe(
        ishigami,
        ishigami_spec(),
        n_samples=256,
        degree=8,
        seed=11,
        n_base_mc=512,
        run_mc=False,
    )
    assert study.mode == MODE_PCE
    assert study.decision.q2 >= DEFAULT_Q2_MIN
    assert study.indices is not None
    assert study.budget is not None


def test_plain_workflow_is_unchanged_by_the_failsafe_wrapper():
    """The wrapper must not alter the plain workflow's own verdict."""
    plain = run_uq_study(
        ishigami, ishigami_spec(), n_samples=64, degree=8, seed=3, run_mc=False
    )
    wrapped = run_with_failsafe(
        ishigami, ishigami_spec(), n_samples=64, degree=8, seed=3,
        n_base_mc=512, run_mc=False,
    )
    assert wrapped.study.status == plain.status
    assert wrapped.study.verdict.as_dict()["reason_codes"] == plain.verdict.as_dict()["reason_codes"]


def test_sample_design_is_reused_not_redrawn():
    design = sample(ishigami_spec(), 32, rule="sobol", seed=99)
    study = run_with_failsafe(
        ishigami, ishigami_spec(), n_samples=32, degree=4, seed=99,
        failsafe_n_base=1024, failsafe_budget_n_mc=256,
    )
    assert study.study.samples.hash == design.hash