"""Tests for the IN101-N09 target-reliability inverse (ISO 19906 ice force).

The suite is deliberately built around falsifiable statements:

* analytic identities that the implementation must reproduce exactly
  (correlation mapping, variance decomposition, closed-form ln-space index);
* the mission's own acceptance criteria, both as *passing* checks on the
  delivered study and as *failing* checks on a mutated copy of it (a criterion
  that cannot fail is not a criterion);
* the negative control, including a case where the control must NOT fire
  (``ex = 0`` removes the p-A correlation, so nothing should be detected).
"""
from __future__ import annotations

import copy
import math

import pytest

from tensorphysics.ice_partial_factors.p_area_joint import (
    PAreaForceModel,
    analytic_beta_ln,
    build_problem,
    characteristic_force,
    design_resistance_median,
    force_moments,
    log_cov_to_sigma,
)
from tensorphysics.ice_partial_factors.target_inverse_checks import (
    conditional_mc_pf,
    design_table_rows,
    iso_deviation_map,
    negative_control,
    reliability_crosscheck,
)
from tensorphysics.ice_partial_factors.target_inverse_study import (
    evaluate_criteria,
    run_study,
)
from tensorphysics.ice_partial_factors.target_reliability_inverse import (
    INITIAL_GUESSES,
    adjoint_beta_sensitivity,
    beta_derivative_check,
    beta_of_kappa,
    calibrate_kappa,
    condition_number,
    design_split,
    solve_factors,
    solve_factors_multistart,
)


@pytest.fixture(scope="module")
def model() -> PAreaForceModel:
    return PAreaForceModel()


@pytest.fixture(scope="module")
def kappa_star(model: PAreaForceModel) -> float:
    return calibrate_kappa(model)["kappa_star"]


@pytest.fixture(scope="module")
def inverse(model: PAreaForceModel) -> dict:
    return solve_factors_multistart(model)


# --------------------------------------------------------------------------- #
# Joint p-A state: analytic identities
# --------------------------------------------------------------------------- #
def test_log_cov_to_sigma_is_the_exact_lognormal_relation() -> None:
    for cov in (0.05, 0.15, 0.35):
        sigma = log_cov_to_sigma(cov)
        assert math.sqrt(math.exp(sigma * sigma) - 1.0) == pytest.approx(cov, rel=1e-12)


def test_pressure_area_correlation_is_negative_and_matches_the_power_law(
    model: PAreaForceModel,
) -> None:
    rho = model.pressure_area_correlation()
    expected = -model.exponent * model.sigma_a / model.sigma_p
    assert rho == pytest.approx(expected, rel=1e-12)
    assert rho < 0.0


def test_nataf_reproduces_the_log_space_correlation(model: PAreaForceModel) -> None:
    """The assembled problem must have exactly the ISO power-law log-correlation."""
    problem = build_problem(model, 1.4)
    assert problem.transform.rho0[1][2] == pytest.approx(
        model.pressure_area_correlation(), abs=1e-6
    )
    # and the physical (Pearson) correlation is the one the transform was given
    assert problem.transform.correlation[1][2] == pytest.approx(
        model.pressure_area_pearson(), rel=1e-12
    )


def test_variance_decomposition_of_the_force(model: PAreaForceModel) -> None:
    """``Var_indep - Var_corr = 2 ex sigma_lnA**2`` (the physics of the control)."""
    corr = force_moments(model, False)
    indep = force_moments(model, True)
    delta = indep["var_ln_force"] - corr["var_ln_force"]
    assert delta == pytest.approx(2.0 * model.exponent * model.sigma_a**2, rel=1e-12)
    assert indep["sigma_ln_force"] > corr["sigma_ln_force"] > 0.0
    assert corr["median_force_mn"] == pytest.approx(indep["median_force_mn"], rel=1e-12)


def test_characteristic_force_and_resistance_conventions(model: PAreaForceModel) -> None:
    f_k = characteristic_force(model)
    mom = force_moments(model)
    # 100-year level = 99 % quantile of the annual maximum
    z = 2.3263478740408408
    assert f_k == pytest.approx(
        math.exp(mom["mu_ln_force"] + mom["sigma_ln_force"] * z), rel=1e-6
    )
    r_median = design_resistance_median(model, 1.5)
    # 5 % fractile of the resistance equals kappa * F_k by construction
    r_k = r_median * math.exp(model.sigma_r * (-1.6448536269514722))
    assert r_k == pytest.approx(1.5 * f_k, rel=1e-9)


# --------------------------------------------------------------------------- #
# Reliability solver on this problem
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa", [1.0, 1.4, 2.0])
def test_form_reproduces_the_closed_form_index(model: PAreaForceModel, kappa: float) -> None:
    """The failure domain is a half-space in u, hence FORM is exact here."""
    beta_form = beta_of_kappa(model, kappa)[0]
    beta_ln = analytic_beta_ln(model, kappa)
    assert beta_form == pytest.approx(beta_ln, rel=2e-6)


def test_beta_increases_with_the_factor_product(model: PAreaForceModel) -> None:
    betas = [beta_of_kappa(model, k)[0] for k in (0.8, 1.0, 1.2, 1.5, 2.0)]
    assert all(b2 > b1 for b1, b2 in zip(betas, betas[1:]))


def test_bisection_hits_the_target(model: PAreaForceModel, kappa_star: float) -> None:
    result = calibrate_kappa(model)
    assert abs(result["beta_achieved"] - 4.0) <= 1e-6
    assert result["converged"] is True
    assert result["kappa_star"] == pytest.approx(kappa_star, rel=1e-9)


# --------------------------------------------------------------------------- #
# Mission criterion (b): the inverse is solved and unique
# --------------------------------------------------------------------------- #
def test_two_equation_inverse_hits_target(model: PAreaForceModel, inverse: dict) -> None:
    sol = inverse["solution"]
    assert abs(sol["beta_achieved"] - 4.0) <= 1e-3
    assert sol["converged"] is True
    # the split equation is satisfied to solver accuracy
    assert sol["gamma_f"] == pytest.approx(sol["split"]["F_star_mn"] / sol["split"]["F_k_mn"], rel=1e-9)
    assert sol["gamma_f"] * sol["psi"] == pytest.approx(sol["kappa_star"], rel=1e-12)


def test_inverse_agrees_with_the_one_unknown_bisection(
    model: PAreaForceModel, inverse: dict, kappa_star: float
) -> None:
    assert inverse["solution"]["kappa_star"] == pytest.approx(kappa_star, rel=1e-6)


def test_inverse_is_unique_from_three_starts(inverse: dict) -> None:
    assert inverse["all_converged"] is True
    assert len(INITIAL_GUESSES) == 3
    assert inverse["gamma_f_spread_rel"] <= 0.01
    assert inverse["solution"]["jacobian_condition_number"] < 1e3


def test_jacobian_is_non_degenerate(inverse: dict) -> None:
    jac = inverse["solution"]["jacobian"]
    assert abs(jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]) > 0.0
    assert condition_number(jac) == pytest.approx(
        inverse["solution"]["jacobian_condition_number"], rel=1e-12
    )
    # a rank-one matrix has an infinite condition number -> the indicator can fail
    assert condition_number([[1.0, 1.0], [2.0, 2.0]]) == float("inf")


# --------------------------------------------------------------------------- #
# Mission criterion (c): adjoint sensitivities
# --------------------------------------------------------------------------- #
def test_adjoint_sensitivity_matches_finite_differences(
    model: PAreaForceModel, inverse: dict
) -> None:
    check = beta_derivative_check(model, inverse["solution"])
    assert check["max_relative_error"] <= 0.05
    for key in ("d_beta_d_kappa", "d_beta_d_gamma_f", "d_beta_d_psi"):
        assert check[key]["adjoint"] > 0.0
    # chain rule: d beta / d gamma_f = (d beta / d kappa) * psi
    sol = inverse["solution"]
    assert check["d_beta_d_gamma_f"]["adjoint"] == pytest.approx(
        check["d_beta_d_kappa"]["adjoint"] * sol["psi"], rel=1e-12
    )


def test_adjoint_sensitivity_uses_the_envelope_formula(
    model: PAreaForceModel, inverse: dict
) -> None:
    sol = inverse["solution"]
    sens = adjoint_beta_sensitivity(model, sol["kappa_star"])
    # d beta / d kappa = R* / (kappa * ||grad g||)  -> recompute independently
    problem = build_problem(model, sol["kappa_star"])
    beta, form, _r = beta_of_kappa(model, sol["kappa_star"])
    assert form is not None
    r_star = form.design_point_x[3]
    grad = problem.gradient_u(list(form.design_point_u))
    norm = math.sqrt(sum(g * g for g in grad))
    assert sens["d_beta_d_kappa"] == pytest.approx(r_star / (sol["kappa_star"] * norm), rel=1e-6)
    assert beta == pytest.approx(4.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Mission criterion (a): simulation cross-checks
# --------------------------------------------------------------------------- #
def test_importance_sampling_and_conditional_mc_agree_with_form(
    model: PAreaForceModel, kappa_star: float
) -> None:
    cross = reliability_crosscheck(
        model, kappa_star, samples_is=60_000, samples_mc=200_000, seed=7
    )
    assert cross["form_vs_is_relative_error"] <= 0.03
    assert abs(cross["conditional_monte_carlo"]["relative_error_vs_form"]) <= 0.03
    # the closed-form index is a fourth, simulation-free reference
    assert cross["form_vs_ln_relative_error"] <= 0.01
    assert cross["conditional_monte_carlo"]["covers_form"] is True


def test_conditional_mc_is_unbiased_for_the_exact_failure_probability(
    model: PAreaForceModel, kappa_star: float
) -> None:
    """Rao-Blackwell MC must converge to the closed-form Pf, not merely to FORM."""
    exact_pf = 0.5 * math.erfc(analytic_beta_ln(model, kappa_star) / math.sqrt(2.0))
    out = conditional_mc_pf(model, kappa_star, samples=2_000_000, seed=11)
    assert out["cov_pf"] < 0.10, out["cov_pf"]
    assert out["pf"] == pytest.approx(exact_pf, rel=3.0 * out["cov_pf"] + 0.02)
    assert out["beta"] == pytest.approx(analytic_beta_ln(model, kappa_star), rel=0.03)


def test_crude_monte_carlo_is_not_censored(model: PAreaForceModel, kappa_star: float) -> None:
    cross = reliability_crosscheck(
        model, kappa_star, samples_is=20_000, samples_mc=400_000, seed=3
    )
    assert cross["monte_carlo"]["failures"] >= 10
    assert cross["monte_carlo"]["censored"] is False
    # honest reporting: whatever the estimated band is, both end points are finite
    lo, hi = cross["monte_carlo"]["beta_one_sigma_band"]
    assert hi > lo and math.isfinite(lo) and math.isfinite(hi)


# --------------------------------------------------------------------------- #
# Mission criterion (d): ISO table deviation map
# --------------------------------------------------------------------------- #
def test_deviation_map_covers_every_table_pair(inverse: dict, model: PAreaForceModel) -> None:
    sol = inverse["solution"]
    iso = iso_deviation_map(model, sol["gamma_f"], sol["psi"])
    assert iso["rows_total"] == len(design_table_rows())
    assert iso["rows_total"] >= 10
    for row in iso["rows"]:
        assert row["kappa_table"] == pytest.approx(
            row["table_gamma_f"] * row["table_psi"], rel=1e-12
        )
        assert row["deviation_kappa_rel"] == pytest.approx(
            (row["kappa_table"] - sol["kappa_star"]) / sol["kappa_star"], rel=1e-12
        )
        if row["exceeds_tolerance"] and row["verdict"] == "under_provides_kappa":
            example = row["non_conservative_example"]
            assert example is not None
            assert example["beta_achieved"] < 4.0
            assert example["pf_ratio_achieved_over_target"] > 1.0
            assert example["R_k_required_mn_by_calibration"] > example["R_k_required_mn_by_table"]


def test_deviation_map_marks_the_in_repo_anchor_as_verifiable(
    inverse: dict, model: PAreaForceModel
) -> None:
    sol = inverse["solution"]
    iso = iso_deviation_map(model, sol["gamma_f"], sol["psi"])
    anchors = [r for r in iso["rows"] if r["source_tier"] == "in_repo"]
    assert anchors and all(r["primary_source_verified"] for r in anchors)
    secondary = [r for r in iso["rows"] if r["source_tier"] == "secondary"]
    assert secondary and not any(r["primary_source_verified"] for r in secondary)


# --------------------------------------------------------------------------- #
# Mission criterion (e): the negative control fires ... and can fail
# --------------------------------------------------------------------------- #
def test_negative_control_is_detected(model: PAreaForceModel) -> None:
    control = negative_control(model)
    assert control["rho_p_a"] < 0.0
    assert control["direction_as_expected"] is True
    assert control["delta_gamma_f_rel"] >= 0.05
    assert control["detected"] is True
    # the physical model is over-designed when the naive factors are applied to it
    assert control["beta_physical_model_at_naive_kappa"] > 4.0


def test_negative_control_does_not_fire_without_a_correlation() -> None:
    """With ``ex = 0`` there is no p-A correlation, so nothing may be detected."""
    flat = PAreaForceModel(exponent=0.0)
    assert flat.pressure_area_correlation() == 0.0
    control = negative_control(flat)
    assert abs(control["delta_gamma_f_rel"]) < 1e-6
    assert control["detected"] is False


# --------------------------------------------------------------------------- #
# The criteria evaluator itself must be able to fail
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quick_study() -> dict:
    return run_study({"samples_mc": 120_000, "samples_is": 30_000})


def test_quick_study_criteria_all_pass(quick_study: dict) -> None:
    rows = {row["criterion"]: row for row in quick_study["machine_criteria"]}
    assert len(rows) == 5
    for name in ("b_inverse_solved_unique", "c_beta_sensitivity_vs_finite_difference",
                 "d_iso_table_deviation_map", "e_negative_control_pa_correlation"):
        assert rows[name]["passed"] is True, (name, rows[name]["measured"])
    # criterion (a): the deterministic parts must hold even with few samples
    measured = rows["a_form_vs_simulation"]["measured"]
    assert measured["form_vs_is_relative_error"] <= 0.03
    assert measured["form_vs_exact_relative_error"] <= 0.03


def test_quick_study_records_numbers_not_prose(quick_study: dict) -> None:
    for row in quick_study["machine_criteria"]:
        assert row["measured"], row["criterion"]
        assert isinstance(row["passed"], bool)
    sol = quick_study["inverse"]["solution"]
    for key in ("gamma_f", "psi", "kappa_star", "beta_achieved", "jacobian_condition_number"):
        assert isinstance(sol[key], float)


def test_criteria_reject_a_corrupted_solution(quick_study: dict) -> None:
    """Mutant: shift the calibrated pair -> criteria (b) and (d) must go red."""
    mutant = copy.deepcopy(quick_study)
    sol = mutant["inverse"]["solution"]
    sol["gamma_f"] *= 1.20
    sol["beta_achieved"] = 4.5
    mutant["negative_control"]["delta_gamma_f_rel"] = 0.001
    mutant["sensitivity"]["max_relative_error"] = 0.10
    rows = {row["criterion"]: row for row in evaluate_criteria(mutant, mutant["tolerances"])}
    assert rows["b_inverse_solved_unique"]["passed"] is False
    assert rows["c_beta_sensitivity_vs_finite_difference"]["passed"] is False
    assert rows["e_negative_control_pa_correlation"]["passed"] is False


def test_solve_factors_reports_its_starting_point(model: PAreaForceModel) -> None:
    one = solve_factors(model, start=(2.0, 1.2))
    assert one["start"] == [2.0, 1.2]
    assert one["converged"] is True
    split = design_split(model, one["kappa_star"])
    assert split["gamma_f"] == pytest.approx(one["gamma_f"], rel=1e-9)