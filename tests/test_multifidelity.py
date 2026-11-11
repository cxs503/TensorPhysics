"""Tests for the multi-fidelity model selection engine.

The registry axes here are synthetic (``x1``, ``x2``) on purpose: the engine must
carry no domain vocabulary, and the numeric expectations below are readable
without a physics context.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.multifidelity import (
    Budget,
    EmpiricalDistribution,
    LognormalPrior,
    ModelRegistry,
    StratifiedEmpiricalError,
    StratumSpec,
    TruthGenerator,
    TruthSpec,
    calibrate_registry,
    coverage_rate,
    frontier_curve,
    pareto_front,
    run_backtest,
    run_verification,
    sample_queries,
    select_model,
    wilson_interval,
)
from tensorphysics.multifidelity.backtest import AxisSpec


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _registry_dict():
    return {
        "schema_version": 1,
        "models": [
            {"model_id": "cheap", "fidelity_level": 1, "label": "cheap",
             "error_prior": {"median": 0.10, "sigma": 0.40},
             "cost": {"median_s": 1.0, "sigma": 0.20},
             "domain": {"x1": [0.0, 10.0]}},
            {"model_id": "mid", "fidelity_level": 2, "label": "mid",
             "error_prior": {"median": 0.05, "sigma": 0.30},
             "cost": {"median_s": 100.0, "sigma": 0.30},
             "domain": {"x1": [0.0, 10.0]}},
            {"model_id": "fine", "fidelity_level": 3, "label": "fine",
             "error_prior": {"median": 0.01, "sigma": 0.20},
             "cost": {"median_s": 10000.0, "sigma": 0.10},
             "domain": {"x1": [0.0, 10.0], "x2": [0.0, 1.0]}},
        ],
    }


@pytest.fixture()
def registry() -> ModelRegistry:
    return ModelRegistry.from_dict(_registry_dict())


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------
def test_lognormal_prior_quantiles_and_probability():
    prior = LognormalPrior(median=0.05, sigma=0.3)
    assert prior.prob_le(0.05) == pytest.approx(0.5, abs=1e-12)
    assert prior.quantile(0.5) == pytest.approx(0.05, rel=1e-12)
    q90 = prior.quantile(0.90)
    assert prior.prob_le(q90) == pytest.approx(0.90, abs=1e-9)
    assert prior.mean > prior.median  # right-skewed by construction
    lo, hi = prior.interval(0.90)
    assert lo < prior.median < hi
    with pytest.raises(ValueError):
        LognormalPrior(median=-1.0, sigma=0.2)


def test_empirical_conformal_order_statistic():
    values = [0.01 * (i + 1) for i in range(10)]  # 0.01 ... 0.10
    dist = EmpiricalDistribution.from_values(values)
    assert dist.conformal_upper(0.90) == pytest.approx(0.10)  # ceil(11*0.9)=10 -> max
    assert dist.conformal_upper(0.50) == pytest.approx(0.06)  # ceil(11*0.5)=6
    assert dist.prob_le(0.055) == pytest.approx(0.5)


def test_conformal_bound_finite_sample_coverage():
    """Exchangeable holdout: the 90 % conformal bound must cover ~90 % on average."""
    coverages = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        calibration = rng.lognormal(mean=math.log(0.05), sigma=0.4, size=500)
        holdout = rng.lognormal(mean=math.log(0.05), sigma=0.4, size=2000)
        bound = EmpiricalDistribution.from_values(calibration).conformal_upper(0.90)
        cov = coverage_rate(holdout, [bound] * holdout.size)
        assert 0.83 <= cov <= 0.96  # single-sample spread
        coverages.append(cov)
    # mean coverage is the finite-sample guarantee: 90 % +- MC noise
    assert 0.885 <= float(np.mean(coverages)) <= 0.915
    # the conformal statistic is never below the naive empirical quantile
    rng = np.random.default_rng(2024)
    sample = rng.lognormal(mean=math.log(0.05), sigma=0.4, size=500)
    assert (EmpiricalDistribution.from_values(sample).conformal_upper(0.90)
            >= np.quantile(sample, 0.90) - 1e-12)


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson_interval(950, 1000)
    assert lo < 0.95 < hi
    assert 0.0 <= lo and hi <= 1.0
    with pytest.raises(ValueError):
        coverage_rate([], [])


# ---------------------------------------------------------------------------
# registry and applicability
# ---------------------------------------------------------------------------
def test_registry_rejects_duplicates_and_missing_fields():
    bad = _registry_dict()
    bad["models"].append(dict(bad["models"][0]))
    with pytest.raises(ValueError):
        ModelRegistry.from_dict(bad)
    no_cost = _registry_dict()
    del no_cost["models"][0]["cost"]
    with pytest.raises(ValueError):
        ModelRegistry.from_dict(no_cost)


def test_domain_admissibility_reports_axis(registry):
    assert [m.model_id for m in registry.admissible({"x1": 5.0, "x2": 0.5})] == ["cheap", "mid", "fine"]
    assert [m.model_id for m in registry.admissible({"x1": 5.0})] == ["cheap", "mid"]
    violations = dict(registry.excluded({"x1": 20.0, "x2": 0.5}))
    assert violations["cheap"]["reason"] == "AXIS_ABOVE_DOMAIN"
    assert violations["cheap"]["axis"] == "x1"


def test_cost_prior_probability_consistency():
    model = ModelRegistry.from_dict(_registry_dict()).get("mid")
    assert model.cost.prob_le(100.0) == pytest.approx(0.5, abs=1e-12)
    assert model.cost.prob_le(model.cost.quantile(0.9)) == pytest.approx(0.90, abs=1e-9)
    assert model.cost.mean_s > model.cost.median_s


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def test_min_cost_picks_cheapest_feasible(registry):
    sel = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(epsilon=0.5, cost=1.0e6))
    assert sel.ok and sel.model_id == "cheap"
    assert sel.uncertainty["prob_error_le_epsilon"] >= 0.95
    assert sel.uncertainty["prob_cost_le_cost"] >= 0.95


def test_min_error_respects_cost_budget(registry):
    loose = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(epsilon=0.2, cost=1.0e6),
                         objective="min_error")
    assert loose.model_id == "fine"
    tight = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(epsilon=0.2, cost=50.0),
                         objective="min_error")
    assert tight.model_id == "cheap"  # mid cannot promise P(cost<=50) >= 0.95


def test_negative_control_epsilon_below_floor(registry):
    """eps < 1 % must be refused: no admissible model can promise it."""
    for eps in (0.005, 0.008):
        sel = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(epsilon=eps, cost=1.0e7))
        assert not sel.ok
        assert sel.reason_code == "EPSILON_BELOW_MODEL_FLOOR"
        assert sel.model_id is None
        assert sel.uncertainty["advisory_only"] is True
        assert sel.uncertainty["advisory_fallback_model_id"] == "fine"


def test_refusal_when_cost_budget_below_floor(registry):
    sel = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(cost=0.5))
    assert not sel.ok and sel.reason_code == "COST_BELOW_MODEL_FLOOR"


def test_refusal_when_no_model_admissible(registry):
    sel = select_model(registry, {"x1": 99.0, "x2": 0.5}, Budget(epsilon=0.1, cost=1.0e6))
    assert not sel.ok and sel.reason_code == "NO_ADMISSIBLE_MODEL"
    assert sel.uncertainty["n_admissible"] == 0


def test_refusal_when_no_model_meets_error_budget(registry):
    # above the floor (0.01) but unreachable at 95 % confidence by any model
    sel = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(epsilon=0.012, cost=1.0e7))
    assert not sel.ok and sel.reason_code == "NO_MODEL_MEETS_ERROR_BUDGET"


def test_uncertainty_block_is_informative(registry):
    sel = select_model(registry, {"x1": 5.0, "x2": 0.5}, Budget(epsilon=0.05, cost=1.0e6))
    assert sel.ok
    u = sel.uncertainty
    for key in ("error_upper", "error_median", "prob_error_le_epsilon", "prob_cost_le_cost",
                "expected_cost", "cost_upper", "risk_error_above_epsilon", "slack_error"):
        assert key in u
    assert u["error_upper"] >= u["error_median"]
    assert 0.0 <= u["prob_error_le_epsilon"] <= 1.0
    assert u["risk_error_above_epsilon"] == pytest.approx(1.0 - u["prob_error_le_epsilon"])
    # epsilon minus the chosen model's median error is the accuracy slack
    assert u["slack_error"] == pytest.approx(0.05 - u["error_median"])


def test_pareto_front_and_staircase(registry):
    params = {"x1": 5.0, "x2": 0.5}
    front = pareto_front(registry, params)
    assert front == ["cheap", "mid", "fine"]  # all three trade error for cost
    curve = frontier_curve(registry, params)
    costs = [row["cost_median_s"] for row in curve]
    errors = [row["error_median"] for row in curve]
    assert costs == sorted(costs) and errors == sorted(errors, reverse=True)
    # a dominated model must not appear
    dominated = ModelRegistry.from_dict({
        "schema_version": 1,
        "models": [
            {"model_id": "good", "error_prior": {"median": 0.02, "sigma": 0.2},
             "cost": {"median_s": 10.0}, "domain": {"x1": [0.0, 10.0]}},
            {"model_id": "bad", "error_prior": {"median": 0.05, "sigma": 0.2},
             "cost": {"median_s": 100.0}, "domain": {"x1": [0.0, 10.0]}},
        ],
    })
    assert pareto_front(dominated, params) == ["good"]


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def test_calibration_replaces_view_without_mutating_registry(registry):
    residuals = {"cheap": [0.12] * 50, "mid": [0.06] * 50, "fine": [0.011] * 50}
    calibrated = calibrate_registry(registry, residuals, level=0.90)
    assert registry.get("cheap").error.kind == "prior"
    assert calibrated.get("cheap").error.kind == "empirical"
    # a degenerate residual sample puts the bound exactly at the sample value
    assert calibrated.get("cheap").error.upper_bound(0.90, {"x1": 5.0, "x2": 0.5}) == pytest.approx(0.12)
    assert calibrated.get("cheap").calibration["kind"] == "empirical"


def test_stratified_view_prefers_populated_stratum_and_falls_back(registry):
    stratum = StratumSpec(name="high", bounds={"x1": (5.0, 10.0)})
    pooled = [0.05] * 100
    view = StratifiedEmpiricalError(
        pooled=EmpiricalDistribution.from_values(pooled),
        strata=[stratum],
        samples={"high": [0.30] * 40},
        min_n=30,
    )
    assert view.upper_bound(0.90, {"x1": 7.0, "x2": 0.5}) == pytest.approx(0.30)
    assert view.upper_bound(0.90, {"x1": 1.0, "x2": 0.5}) == pytest.approx(0.05)
    # too few samples in the stratum -> pooled fallback, never a silent 1-sample bound
    sparse = StratifiedEmpiricalError(
        pooled=EmpiricalDistribution.from_values(pooled),
        strata=[stratum],
        samples={"high": [0.30] * 5},
        min_n=30,
    )
    assert sparse.upper_bound(0.90, {"x1": 7.0, "x2": 0.5}) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        StratifiedEmpiricalError(pooled=EmpiricalDistribution.from_values(pooled),
                                 strata=[stratum], samples={"high": [0.3] * 40}, min_n=1)


def test_calibrated_selection_is_more_conservative(registry):
    """Widening the residuals must not make the engine *less* careful."""
    loose = {"cheap": [0.10] * 100, "mid": [0.05] * 100, "fine": [0.01] * 100}
    tight_res = {"cheap": [0.30] * 100, "mid": [0.20] * 100,
                 "fine": [0.02] * 80 + [0.06] * 20}
    params = {"x1": 5.0, "x2": 0.5}
    budget = Budget(epsilon=0.04, cost=1.0e6)
    a = select_model(calibrate_registry(registry, loose), params, budget)
    b = select_model(calibrate_registry(registry, tight_res), params, budget)
    assert a.ok and a.model_id == "fine"
    # 4 % is now only carried by 80 % of the finest model's residuals, i.e. below alpha
    assert not b.ok
    assert b.reason_code == "NO_MODEL_MEETS_ERROR_BUDGET"


# ---------------------------------------------------------------------------
# truth generator / query sampler
# ---------------------------------------------------------------------------
def test_prior_consistent_truth_reproduces_prior(registry):
    model = registry.get("mid")
    truth = TruthGenerator(TruthSpec(label="control"))
    latents = truth.latents(4000, 1, seed=5)
    draws = np.array([truth.error(model, {"x1": 5.0, "x2": 0.5}, latents[i, 0]) for i in range(4000)])
    for level in (0.5, 0.9):
        empirical = float((draws <= model.prior.quantile(level)).mean())
        assert abs(empirical - level) < 0.025


def test_jump_indicator_is_uniform_and_rate_is_honoured(registry):
    model = registry.get("mid")
    latents = TruthGenerator(TruthSpec(label="h")).latents(5000, 1, seed=11)
    calm = TruthGenerator(TruthSpec(label="calm", jump_prob=0.0))
    tail = TruthGenerator(TruthSpec(label="tail", jump_prob=0.20, jump_median=3.0))
    base = np.array([calm.error(model, {"x1": 5.0, "x2": 0.5}, latents[i, 0]) for i in range(5000)])
    jumped = np.array([tail.error(model, {"x1": 5.0, "x2": 0.5}, latents[i, 0]) for i in range(5000)])
    hit_rate = float((jumped > base * 1.5).mean())
    assert 0.17 <= hit_rate <= 0.23  # the indicator is uniform, not normal


def test_sample_queries_is_deterministic_and_in_range():
    axes = {"x1": AxisSpec(0.0, 10.0), "x2": AxisSpec(0.1, 100.0, scale="log")}
    a = sample_queries(axes, n=50, seed=3, epsilon_range=(0.02, 0.2), cost_range=(10.0, 1e4))
    b = sample_queries(axes, n=50, seed=3, epsilon_range=(0.02, 0.2), cost_range=(10.0, 1e4))
    assert [q["params"] for q in a] == [q["params"] for q in b]
    for q in a:
        assert 0.0 <= q["params"]["x1"] <= 10.0
        assert 0.1 <= q["params"]["x2"] <= 100.0
        assert 0.02 <= q["budget"].epsilon <= 0.2
        assert 10.0 <= q["budget"].cost <= 1e4


def test_backtest_records_oracle_and_realisations(registry):
    truth = TruthGenerator(TruthSpec(label="control"))
    axes = {"x1": AxisSpec(0.0, 10.0)}
    queries = sample_queries(axes, n=200, seed=17, epsilon_range=(0.05, 0.2), cost_range=(10.0, 1e6))
    queries = [{"index": q["index"], "params": dict(q["params"], x2=0.5), "budget": q["budget"]}
               for q in queries]
    bt = run_backtest(registry, truth, queries, latent_seed=3)
    metrics = bt.metrics()
    assert metrics["n_queries"] == 200
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    rec = bt.recommended
    assert rec, "the control truth must produce recommendations"
    assert all(r.realised_error is not None and r.realised_cost is not None for r in rec)
    assert metrics["oracle_optimal_rate"] is None or 0.0 <= metrics["oracle_optimal_rate"] <= 1.0


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def verification_result():
    """One 400-query verification run shared by the end-to-end assertions."""
    return run_verification(n_queries=400)


def test_end_to_end_verification_acceptance(verification_result):
    result = verification_result
    artifact = result.artifact
    assert artifact["node"] == "p9003w1_n05"
    checks = artifact["acceptance"]
    assert checks["all_pass"] is True
    assert checks["a_recommendation_accuracy"]["value"] >= 0.90
    assert checks["b_cost_efficiency"]["value"] >= 0.90
    assert checks["c_interval_coverage"]["value"] >= 0.85
    assert checks["d_negative_control"]["pass"] is True
    # the registered-prior baseline must honestly fail the coverage criterion
    baseline = artifact["backtests"]["deployment"]["prior_only"]
    assert baseline["coverage"] < 0.85
    # the mechanism control must reproduce the nominal levels
    control = artifact["backtests"]["prior_consistent"]["prior_only"]
    assert control["accuracy"] >= 0.95
    assert 0.86 <= control["coverage"] <= 0.93
    assert artifact["physical_anchor"]["rows"]
    assert artifact["pareto"]["front_models"]


def test_confidence_sweep_is_monotone_in_alpha(verification_result):
    result = verification_result
    rows = [r for r in result.artifact["confidence_sweep"]
            if r["variant"] == "conformal_stratified"]
    assert len(rows) == 3
    # a stronger guarantee cannot buy a higher recommendation rate
    assert rows[0]["recommendation_rate"] >= rows[-1]["recommendation_rate"]
    assert all(0.0 <= r["coverage"] <= 1.0 for r in rows)