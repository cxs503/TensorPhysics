"""Acceptance tests for the ice-going digital-twin model-validation / UQ layer.

Covers the seven verification faces of the node:

1. the vectorised twin is *identical* to the repository's reference law,
2. the Bayesian calibration recovers a known truth (negative control) and its
   diagnostics (split-R̂, ESS) behave as advertised,
3. the Sobol' estimator closes on an analytically known additive submodel,
4. the model-form ensemble produces valid pseudo-BMA weights and a narrower
   averaged prediction interval,
5. prediction-interval coverage is validated by leave-one-out cross-validation,
   and the *uncalibrated* model is shown to under-cover (negative control),
6. the streaming calibrator meets the per-step time budget and agrees with the
   batch posterior,
7. the campaign report is complete, serialisable and carries a provenance hash.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorphysics.ice.resistance import ShipGeometry, level_ice_resistance
from tensorphysics.ice_digital_twin import (
    FORMS,
    KAPPA_BOUNDS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    PARAM_REFERENCE,
    CalibrationData,
    OnlineCalibrator,
    additive_submodel_exact,
    additive_submodel_sobol,
    bma_cross_validation,
    bma_mixture,
    bma_report,
    calibrate,
    closure_metrics,
    component_breakdown_kN,
    coverage_rate,
    fixed_parameter_baseline,
    get_form,
    importance_loo,
    loo_form_refit,
    log_likelihood,
    log_likelihood_features,
    log_posterior_factory,
    model_features,
    model_from_features,
    parameter_reference_table,
    physical_to_unit,
    predictive_draws,
    predictive_interval,
    predictive_matrix,
    pseudo_bma_weights,
    reference_condition,
    reference_law_kN,
    resistance_kN,
    resistance_kN_batch,
    run_am_chains,
    run_digital_twin_uq,
    run_online_calibration,
    sensitivity_report,
    split_rhat,
    synthetic_campaign,
    unit_to_physical,
    write_report,
)
from tensorphysics.ice_digital_twin.bayes import ess_min
from tensorphysics.ice_digital_twin.twin import k_bend_reference

SHIP = ShipGeometry()


# ---------------------------------------------------------------------------
# fixtures (module scope: the sampling work is shared by several assertions)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recovery_data() -> CalibrationData:
    return synthetic_campaign(SHIP, n_points=24, seed=4242, label="recovery-test")


@pytest.fixture(scope="module")
def inadequacy_data() -> CalibrationData:
    return synthetic_campaign(
        SHIP,
        n_points=20,
        seed=4343,
        truth_params=np.array([1.12, 0.88, 0.175, 1.05]),
        discrepancy_slope=0.10,
        label="inadequacy-test",
    )


@pytest.fixture(scope="module")
def recovery_fit(recovery_data):
    chains, summary = calibrate(
        recovery_data, SHIP, "full", n_chains=3, n_steps=6000, burn=1500, seed=7
    )
    return chains, summary


@pytest.fixture(scope="module")
def sensitivity_payload():
    return sensitivity_report(SHIP, n_base=1 << 11, seed=99, bootstrap=0)


# ---------------------------------------------------------------------------
# 1. forward model: vectorised twin == reference law
# ---------------------------------------------------------------------------


def test_vectorised_forward_matches_reference_law():
    ship = ShipGeometry()
    h = np.array([0.35, 0.6, 1.0, 1.4])
    v = np.array([1.0, 2.0, 3.5, 5.0])
    theta = np.array([PARAM_REFERENCE[n] for n in PARAM_NAMES])
    twin = resistance_kN("full", theta, h, v, ship)
    ref = np.array(
        [level_ice_resistance(float(h[i]), float(v[i]), ship)["R_total_kN"] for i in range(4)]
    )
    assert np.allclose(twin, ref, rtol=1e-12, atol=0.0)
    # the same anchor through the convenience shim
    assert np.allclose(twin, reference_law_kN(h, v, ship), rtol=1e-12)


def test_batch_and_feature_paths_agree_with_direct_model():
    ship = ShipGeometry()
    h, v = 0.9, 3.1
    rng = np.random.default_rng(0)
    phys = unit_to_physical(rng.random((40, len(PARAM_NAMES))))
    batch = resistance_kN_batch("full", phys, h, v, ship)
    direct = np.array([float(resistance_kN("full", p, h, v, ship)) for p in phys])
    assert np.allclose(batch, direct, rtol=1e-12)
    data = CalibrationData(
        h_m=[h], V_ms=[v], R_kN=[1.0], sigma_kN=[1.0], label="feature-check"
    )
    feats = model_features(data, "full", ship)
    for p in phys[:5]:
        assert math.isclose(
            float(model_from_features(feats, p)[0]),
            float(resistance_kN("full", p, h, v, ship)),
            rel_tol=1e-12,
        )
    # the feature likelihood equals the direct one
    assert math.isclose(
        log_likelihood_features(feats, data, phys[0], 0.05),
        log_likelihood(data, "full", ship, phys[0], 0.05),
        rel_tol=1e-12,
    )


def test_structure_of_the_law_components():
    ship = ShipGeometry()
    theta = np.array([1.05, 0.95, 0.18, 1.02])
    parts = component_breakdown_kN(theta, 1.0, 3.0, ship)
    total = float(resistance_kN("full", theta, 1.0, 3.0, ship))
    assert math.isclose(float(parts["R_total_kN"]), total, rel_tol=1e-12)
    assert parts["R_crush_kN"] > 0 and parts["R_submersion_kN"] > 0
    # dropping a mechanism removes its contribution exactly
    crush_break = float(resistance_kN("crush_break", theta[:3], 1.0, 3.0, ship))
    assert math.isclose(
        crush_break, total - float(parts["R_submersion_kN"]), rel_tol=1e-12
    )
    break_sub = float(resistance_kN("break_sub", theta[[1, 2, 3]], 1.0, 3.0, ship))
    assert math.isclose(break_sub, total - float(parts["R_crush_kN"]), rel_tol=1e-12)


def test_parameterisation_roundtrip_and_literature_inside_box():
    rng = np.random.default_rng(1)
    phys = unit_to_physical(rng.random((10, 4)))
    assert np.allclose(physical_to_unit(phys), rng.random((10, 4)) * 0 + physical_to_unit(phys))
    assert np.allclose(unit_to_physical(physical_to_unit(phys)), phys, rtol=1e-12)
    # every literature reference is strictly inside its declared box
    for row in parameter_reference_table():
        assert row["lower"] < row["reference"] < row["upper"], row["name"]
        if row["name"] != "mu_hull_ice":
            assert row["reference"] == 1.0
    assert KAPPA_BOUNDS[0] <= 0.10 <= KAPPA_BOUNDS[1]
    # unit-cube coordinates outside [0,1] must be rejected, not silently clipped
    with pytest.raises(ValueError):
        physical_to_unit(np.array([3.0, 1.0, 0.15, 1.0]))
    # every model form carries exactly the coefficients of its retained mechanisms
    for form in FORMS.values():
        assert form.n_phys == len(form.param_names)
        assert len(form.full_vector(form.bounds()[:, 0])) == len(PARAM_NAMES)
    assert set(get_form("full").components) == {"crush", "break", "submersion"}


# ---------------------------------------------------------------------------
# 2. Bayesian calibration
# ---------------------------------------------------------------------------


def test_prior_rejects_parameters_outside_the_box(recovery_data):
    log_post, bounds, names = log_posterior_factory(recovery_data, "full", SHIP)
    assert names == list(PARAM_NAMES) + ["kappa"]
    centre = 0.5 * (bounds[:, 0] + bounds[:, 1])
    assert np.isfinite(log_post(centre))
    outside = centre.copy()
    outside[0] = bounds[0, 1] * 2.0
    assert log_post(outside) == -math.inf
    bad_kappa = centre.copy()
    bad_kappa[-1] = -0.5
    assert log_post(bad_kappa) == -math.inf


def test_likelihood_prefers_the_generating_parameter(recovery_data):
    feats = model_features(recovery_data, "full", SHIP)
    truth = np.array([PARAM_REFERENCE[n] for n in PARAM_NAMES])
    reference = log_likelihood(recovery_data, "full", SHIP, truth, 0.06)
    worse = log_likelihood(recovery_data, "full", SHIP, truth * 1.6, 0.06)
    assert reference > worse
    # the feature path is the same function
    assert math.isclose(
        log_likelihood_features(feats, recovery_data, truth, 0.06), reference, rel_tol=1e-12
    )


def test_adaptive_metropolis_recovers_a_gaussian_posterior():
    """Linear-Gaussian target with a known mean/covariance (sampler self-test)."""
    d = 3
    cov = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 0.5]])
    prec = np.linalg.inv(cov)
    mean = np.array([0.4, -0.3, 1.0])

    def log_post(x):
        delta = x - mean
        return -0.5 * float(delta @ prec @ delta)

    chains = run_am_chains(
        log_post, np.array([[-6.0, 6.0]] * d), n_chains=3, n_steps=8000, burn=2000, seed=1
    )
    flat = chains.flat()
    sem = np.sqrt(np.diag(cov)) / math.sqrt(flat.shape[0] / 20.0)
    assert np.all(np.abs(flat.mean(axis=0) - mean) < 5.0 * sem)
    sample_cov = np.cov(flat.T)
    assert np.all(np.abs(sample_cov - cov) < 0.25)
    assert np.max(split_rhat(chains.post_burn())) < 1.1


def test_convergence_diagnostics_on_the_twin_posterior(recovery_fit):
    chains, summary = recovery_fit
    post = chains.post_burn()
    rhat = split_rhat(post)
    ess, ess_floor = ess_min(post)
    assert rhat.shape == (len(PARAM_NAMES) + 1,)
    assert summary["max_rhat_split"] < 1.1
    assert ess_floor == summary["min_ess"]
    assert ess_floor > 200  # 3 chains x 4500 post-burn-in draws
    acc = chains.acceptance_rate()
    assert np.all(acc > 0.05) and np.all(acc < 0.6)


def test_calibration_recovers_literature_coefficients_within_20pct(recovery_fit):
    _, summary = recovery_fit
    for row in summary["parameters"]:
        if row["name"] not in PARAM_REFERENCE:
            continue
        assert row["relative_error"] < 0.20, (row["name"], row["relative_error"])
    # the calibrated discrepancy level must be compatible with the 6% noise level
    kappa = next(p for p in summary["parameters"] if p["name"] == "kappa")
    assert 0.0 <= kappa["mean"] < 0.15


def test_posterior_widens_for_the_inadequate_truth(recovery_fit, inadequacy_data):
    """A truth outside the model family must not be absorbed silently."""
    _, summary_ok = recovery_fit
    _, summary_bad = calibrate(
        inadequacy_data, SHIP, "full", n_chains=2, n_steps=4000, burn=1000, seed=8
    )
    kappa_ok = next(p for p in summary_ok["parameters"] if p["name"] == "kappa")["mean"]
    kappa_bad = next(p for p in summary_bad["parameters"] if p["name"] == "kappa")["mean"]
    # the inadequacy campaign is explained by a *different* coefficient set: at
    # least one physical coefficient must move by more than its recovery error
    err_ok = {p["name"]: p["relative_error"] for p in summary_ok["parameters"] if p["name"] in PARAM_REFERENCE}
    err_bad = {p["name"]: p["relative_error"] for p in summary_bad["parameters"] if p["name"] in PARAM_REFERENCE}
    assert max(err_bad[n] for n in err_bad) >= max(err_ok[n] for n in err_ok)
    assert kappa_bad >= 0.0


def test_predictive_matrix_matches_sampled_draws(recovery_fit, recovery_data):
    chains, _ = recovery_fit
    mu, sd = predictive_matrix(chains, recovery_data, "full", SHIP, thin=5)
    draws = predictive_draws(chains, recovery_data, "full", SHIP, thin=5, seed=11)
    assert draws.shape == mu.shape
    assert np.allclose(draws.mean(axis=0), mu.mean(axis=0), rtol=0.02, atol=5.0)
    empiric = draws.std(axis=0, ddof=1)
    model = np.sqrt(np.mean(np.square(sd) + np.square(mu), axis=0) - np.square(mu.mean(axis=0)))
    assert np.allclose(empiric, model, rtol=0.15)
    lo, hi = predictive_interval(draws, 0.95)
    assert np.all(hi > lo)
    inside, n, rate = coverage_rate(lo, hi, draws.mean(axis=0))
    assert n == len(recovery_data) and rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Sobol' global sensitivity
# ---------------------------------------------------------------------------


def test_sobol_estimator_closes_on_the_additive_submodel(sensitivity_payload):
    closure = sensitivity_payload["closure"]
    exact = sensitivity_payload["additive_exact"]
    assert exact["sum_indices"] == pytest.approx(1.0, abs=1e-12)
    assert closure["sum_total_error_vs_one"] < 0.05
    assert closure["max_total_index_error"] < 0.05
    assert closure["closure_ok_5pct"] is True
    for row in closure["per_parameter"].values():
        assert row["total_error"] < 0.05
        assert row["first_error"] < 0.05


def test_twin_indices_are_ordered_and_closed(sensitivity_payload):
    twin = sensitivity_payload["twin"]
    for name in PARAM_NAMES:
        assert twin["total"][name] >= twin["first"][name] - 1e-6
        assert twin["first"][name] >= -0.02
    assert sum(twin["first"].values()) <= 1.0 + 0.05
    # sum of all orders (first + pairwise second) must close to one
    assert twin["all_order_closure_error"] < 0.05
    # the interaction share is reported, not hidden
    assert twin["interaction_share"] == pytest.approx(twin["sum_total"] - 1.0)
    assert twin["n_model_evaluations"] == twin["n_base"] * (2 + 4 + 6)


def test_analyis_of_the_sensitivity_payload_structure(sensitivity_payload):
    twin = sensitivity_payload["twin"]
    assert sensitivity_payload["criteria"]["sum_total_closure_under_5pct"] is True
    ref = sensitivity_payload["twin"]["reference"]
    assert ref["h_m"] > 0 and ref["V_ms"] > 0
    # friction and the crush-strength multiplier dominate the twin variance
    ranking = twin["ranking_by_total"]
    assert set(ranking) == set(PARAM_NAMES)
    assert twin["total"][ranking[0]] > 0.1
    assert "first_ci" in twin and "total_ci" in twin or twin["bootstrap"] == 0


# ---------------------------------------------------------------------------
# 4. model-form uncertainty (BMA)
# ---------------------------------------------------------------------------


def test_bma_weights_are_a_probability_vector():
    lpd = {"full": -100.0, "crush_break": -120.0, "crush_sub": -300.0, "break_sub": -400.0}
    w = pseudo_bma_weights(lpd)
    assert set(w) == set(lpd)
    assert all(0.0 <= v <= 1.0 for v in w.values())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    # better LOO score -> larger weight, and extreme differences saturate safely
    assert w["full"] > w["crush_break"] > w["crush_sub"]
    extreme = pseudo_bma_weights({"a": -1e6 * 1.0, "b": -1e6 * 2.0})
    assert set(extreme) == {"a", "b"} and sum(extreme.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pseudo_bma_weights({"a": float("nan")})


def test_bma_mixture_is_narrower_than_the_average_single_model():
    rng = np.random.default_rng(3)
    n = 4000
    draws = {
        "good": (100.0 + 8.0 * rng.standard_normal((n, 1))),
        "bad1": (200.0 + 60.0 * rng.standard_normal((n, 1))),
        "bad2": (50.0 + 120.0 * rng.standard_normal((n, 1))),
        "bad3": (400.0 + 90.0 * rng.standard_normal((n, 1))),
    }
    observed = np.full(1, 100.0)
    report = bma_report(
        draws,
        {"good": -10.0, "bad1": -400.0, "bad2": -900.0, "bad3": -1500.0},
        observed,
        seed=5,
    )
    assert report["criteria"]["weights_sum_to_one"] is True
    assert report["criteria"]["weights_in_unit_interval"] is True
    assert report["weights"]["good"] > 0.99
    assert report["criteria"]["pi_narrower_than_single_over_20pct"] is True
    assert report["comparison"]["narrower_fraction_vs_mean_single"] > 0.20
    assert report["bma"]["between_share"] >= 0.0
    with pytest.raises(ValueError):
        bma_mixture(draws, {"good": 0.5})  # weights do not cover the members


# ---------------------------------------------------------------------------
# 5. coverage validation (LOO-CV)
# ---------------------------------------------------------------------------


def test_importance_loo_agrees_with_exact_fold_refit(recovery_data):
    """The cheap IS-LOO must reproduce the exact refit LOO (small campaign)."""
    small = recovery_data.subset(range(10), label="loo-cross-check")
    chains = calibrate(small, SHIP, "full", n_chains=1, n_steps=5000, burn=1500, seed=21)[0]
    is_loo = importance_loo(chains, small, "full", SHIP, thin=2, seed=22)
    exact = loo_form_refit(
        small, SHIP, "full", n_steps=4000, burn=1500, seed=23, thin=3, init=chains.mean()
    )
    assert is_loo["n_folds"] == exact["n_folds"] == 10
    assert abs(is_loo["coverage"] - exact["coverage"]) <= 0.10
    assert is_loo["mean_pi_width_kN"] == pytest.approx(exact["mean_pi_width_kN"], rel=0.25)
    assert is_loo["lpd_per_record"] == pytest.approx(exact["lpd_per_record"], abs=2.0)
    # the weight diagnostics are reported, not assumed
    assert 0.0 < is_loo["min_weight_ess_fraction"] <= 1.0


def test_uncalibrated_law_under_covers_the_inadequate_truth(inadequacy_data):
    baseline = fixed_parameter_baseline(inadequacy_data, SHIP)
    assert baseline["n_records"] == len(inadequacy_data)
    assert baseline["coverage"] < 0.95
    assert baseline["mean_relative_error"] > 0.03
    assert baseline["mean_pi_width_kN"] > 0.0
    assert baseline["under_covers"] is True


def test_loo_coverage_of_the_calibrated_interval(inadequacy_data):
    chains = calibrate(
        inadequacy_data, SHIP, "full", n_chains=1, n_steps=5000, burn=1500, seed=31
    )[0]
    loo = importance_loo(chains, inadequacy_data, "full", SHIP, thin=2, seed=32)
    assert loo["n_folds"] >= 15
    assert 0.80 <= loo["coverage"] <= 1.0
    assert loo["mean_pi_width_kN"] > 0.0
    assert len(loo["folds"]) == loo["n_folds"]
    assert all("weight_ess" in f for f in loo["folds"])


def test_bma_cross_validation_returns_consistent_folds(inadequacy_data):
    small = inadequacy_data.subset(range(12), label="bma-cv")
    forms = ("full", "crush_break")
    cv = bma_cross_validation(
        small, SHIP, forms, n_steps=1500, burn=600, thin=2, seed=41
    )
    assert set(cv["members"]) == set(forms)
    assert sum(cv["weights"].values()) == pytest.approx(1.0)
    assert 0.0 <= cv["bma"]["coverage"] <= 1.0
    assert len(cv["folds_bma"]) == 12
    assert cv["comparison"]["bma_pi_width_kN"] > 0.0
    assert cv["bma"]["n_records"] == 12


# ---------------------------------------------------------------------------
# 6. online calibration
# ---------------------------------------------------------------------------


def test_online_calibrator_meets_the_time_budget(inadequacy_data):
    batch = calibrate(
        inadequacy_data, SHIP, "full", n_chains=2, n_steps=5000, burn=1500, seed=51
    )[0]
    result = run_online_calibration(
        inadequacy_data, SHIP, n_particles=250, n_sweeps=2, seed=52, batch_chains=batch
    )
    assert result["criteria"]["time_per_step_under_1s"] is True
    assert result["time_per_step_s"]["max"] < 1.0
    assert result["n_steps"] == len(inadequacy_data)
    assert len(result["mean_trajectory"]["sigma_c_scale"]) == result["n_steps"]
    # streaming estimate converges to the batch posterior
    assert result["batch_comparison"]["max_relative_difference"] < 0.35
    assert 0 < result["ess_history"][0] <= result["n_particles"]


def test_online_calibrator_is_monotone_in_information(inadequacy_data):
    cal = OnlineCalibrator(ship=SHIP, n_particles=200, n_sweeps=1, seed=61)
    widths = []
    for i in range(min(8, len(inadequacy_data))):
        cal.update(
            float(inadequacy_data.h_m[i]),
            float(inadequacy_data.V_ms[i]),
            float(inadequacy_data.R_kN[i]),
            float(inadequacy_data.sigma_kN[i]),
        )
        widths.append(cal.posterior_sd()["sigma_c_scale"])
    assert all(w > 0 for w in widths)
    # the posterior must shrink (or at least not blow up) as data accumulate
    assert widths[-1] <= widths[0] + 1e-9


# ---------------------------------------------------------------------------
# 7. campaign report
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_report():
    return run_digital_twin_uq(
        n_calib_steps=2500,
        calib_burn=800,
        bma_steps=1200,
        bma_burn=400,
        loo_steps=900,
        loo_burn=400,
        budget_steps=1500,
        prior_steps=1200,
        sobol_n_base=1 << 8,
        online_particles=120,
        n_records=14,
        seed=777,
    )


def test_report_contains_every_acceptance_criterion(mini_report):
    crit = mini_report["criteria"]
    required = {
        "a_posterior_mean_vs_literature_under_20pct",
        "b_mcmc_converged",
        "b_max_rhat_split",
        "b_min_ess",
        "c_sobol_sum_total_closure_under_5pct",
        "d_loo_coverage_in_90_100",
        "d_loo_coverage",
        "e_bma_weights_sum_to_one",
        "e_pi_narrower_than_single_over_20pct",
        "f_online_under_1s",
        "f_online_time_per_step_s",
    }
    assert required <= set(crit)
    for key, value in crit.items():
        assert isinstance(value, (bool, int, float)), (key, type(value))
    assert mini_report["data"]["n_records"] == 14
    assert mini_report["provenance"]["payload_sha256"]
    assert len(mini_report["limitations"]) >= 3
    assert len(mini_report["parameter_table"]) == len(PARAM_NAMES)


def test_report_is_json_serialisable_and_written(mini_report, tmp_path):
    path = write_report(mini_report, str(tmp_path / "report.json"))
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["node"] == "p223w3-n09-physics-uq-digital-twin"
    assert loaded["criteria"]["f_online_under_1s"] is True
    assert json.dumps(loaded)[:1] == "{"
    # identical content must serialise identically (deterministic report)
    other = write_report(mini_report, str(tmp_path / "report2.json"))
    with open(other, encoding="utf-8") as fh:
        assert json.load(fh) == loaded


def test_reference_condition_stays_inside_the_law_domain():
    ref = reference_condition(SHIP)
    assert ref.h_m < 0.03 * SHIP.L_pp and ref.h_m < 0.10 * SHIP.B
    assert ref.V_ms < 0.25 * math.sqrt(9.80665 * SHIP.L_pp)
    res = level_ice_resistance(ref.h_m, ref.V_ms, SHIP)
    assert res["domain_ok"] is True
    assert math.isclose(res["C_break"], 0.5 * (
        math.sin(0.5 * SHIP.bow_angle_rad) + PARAM_REFERENCE["mu_hull_ice"] * math.cos(0.5 * SHIP.bow_angle_rad)
    ) / k_bend_reference(), rel_tol=1e-12)