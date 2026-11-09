"""Independent checks of the target-reliability inverse (IN101-N09).

Three verification faces, each with its own failure mode:

* :func:`reliability_crosscheck` -- FORM against design-point importance
  sampling, crude Monte Carlo (with one-sigma bands in ``beta``) and the
  closed-form ln-space index of the lognormal problem.  Criterion (a).
* :func:`iso_deviation_map` -- row-by-row deviation of the calibrated pair from
  the ISO 19906 Annex A table pairs, with a quantitative non-conservative
  working example for every row outside the 15 % band.  Criterion (d).
* :func:`negative_control` -- the same calibration on the same marginals with
  ``rho(p, A) = 0``; the calibrated load factor must rise by at least 5 %.
  Criterion (e).
"""
from __future__ import annotations

import dataclasses
from math import inf, log
from typing import Sequence

from ..structural.distributions import norm_cdf, norm_ppf
from .iso_tables import DESIGN_CASES, LOAD_FACTOR_ROWS, REPO_ANCHOR_ROWS, RESISTANCE_FACTOR_ROWS
from .p_area_joint import (
    PAreaForceModel,
    analytic_beta_ln,
    build_problem,
    characteristic_force,
    force_moments,
)
from .target_reliability_inverse import (
    INITIAL_GUESSES,
    TARGET_BETA,
    beta_of_kappa,
    rel_error,
    solve_factors,
    solve_factors_multistart,
)

__all__ = [
    "reliability_crosscheck",
    "conditional_mc_pf",
    "iso_deviation_map",
    "negative_control",
    "beta_target_sweep_inverse",
    "model_parameter_sensitivity",
    "design_table_rows",
]


def _beta_from_pf(pf: float) -> float:
    if pf <= 0.0:
        return inf
    if pf >= 1.0:
        return -inf
    return -norm_ppf(pf)


def reliability_crosscheck(
    model: PAreaForceModel,
    kappa: float,
    samples_is: int = 200_000,
    samples_mc: int = 4_000_000,
    seed: int = 20261109,
    ignore_correlation: bool = False,
) -> dict:
    """Criterion (a): FORM against importance sampling and crude Monte Carlo.

    Per method: the reliability index, the one-sigma interval in ``beta`` (from
    the estimated Pf standard error) and whether that interval covers FORM.  The
    closed-form ln-space index is added as a third, simulation-free reference.
    """
    problem = build_problem(model, kappa, ignore_correlation=ignore_correlation)
    beta_form, form, _r = beta_of_kappa(model, kappa, ignore_correlation)
    mc = problem.monte_carlo_pf(samples=samples_mc, seed=seed)
    is_ = problem.importance_sampling_pf(samples=samples_is, seed=seed + 1, form_result=form)
    beta_is = _beta_from_pf(is_.pf)
    beta_mc = _beta_from_pf(mc.pf) if mc.pf > 0.0 else inf
    is_band = (
        _beta_from_pf(is_.pf + is_.std_error),
        _beta_from_pf(max(is_.pf - is_.std_error, 1e-300)),
    )
    mc_band = (
        _beta_from_pf(mc.pf + mc.std_error) if mc.pf > 0.0 else inf,
        _beta_from_pf(max(mc.pf - mc.std_error, 1e-300)) if mc.pf > 0.0 else inf,
    )
    beta_ln = analytic_beta_ln(model, kappa, ignore_correlation)
    return {
        "kappa": kappa,
        "beta_form": beta_form,
        "pf_form": form.pf,
        "beta_ln_space": beta_ln,
        "beta_ln_vs_form_relative_error": rel_error(beta_ln, beta_form),
        "importance_sampling": {
            "samples": is_.samples,
            "pf": is_.pf,
            "std_error_pf": is_.std_error,
            "cov_pf": is_.std_error / is_.pf if is_.pf > 0 else inf,
            "beta": beta_is,
            "beta_one_sigma_band": list(is_band),
            "relative_error_vs_form": rel_error(beta_is, beta_form),
            "covers_form": bool(is_band[0] <= beta_form <= is_band[1]),
        },
        "monte_carlo": {
            "samples": mc.samples,
            "failures": mc.failures,
            "pf": mc.pf,
            "std_error_pf": mc.std_error,
            "beta": beta_mc,
            "beta_one_sigma_band": list(mc_band),
            "relative_error_vs_form": rel_error(beta_mc, beta_form) if beta_mc != inf else inf,
            "covers_form": bool(mc_band[0] <= beta_form <= mc_band[1]),
            "beta_95_band": [
                _beta_from_pf(mc.pf + 1.96 * mc.std_error) if mc.pf > 0.0 else inf,
                _beta_from_pf(max(mc.pf - 1.96 * mc.std_error, 1e-300)) if mc.pf > 0.0 else inf,
            ],
            "covers_form_95": bool(
                mc.pf > 0.0
                and _beta_from_pf(mc.pf + 1.96 * mc.std_error)
                <= beta_form
                <= _beta_from_pf(max(mc.pf - 1.96 * mc.std_error, 1e-300))
            ),
            "censored": not mc.censor_ok,
        },
        "conditional_monte_carlo": _conditional_entry(
            model, kappa, samples_mc, seed, beta_form, ignore_correlation
        ),
        "form_vs_is_relative_error": abs(rel_error(beta_is, beta_form)),
        "form_vs_ln_relative_error": abs(rel_error(beta_ln, beta_form)),
    }


def conditional_mc_pf(
    model: PAreaForceModel,
    kappa: float,
    samples: int = 4_000_000,
    seed: int = 20261109,
    ignore_correlation: bool = False,
) -> dict:
    """Crude Monte Carlo with the resistance integrated analytically (Rao-Blackwell).

    ``R`` is lognormal and independent of the ice force ``F = k_F p A``, so

        P(g < 0 | F) = P(R < F | F) = Phi((ln F - mu_lnR) / sigma_lnR)

    exactly.  Averaging that conditional probability over crude (i.i.d.) samples
    of the *load* variables is an unbiased estimator of the same ``Pf`` with a
    much smaller variance than counting failures, i.e. the crude Monte Carlo of
    the mission criterion with one variance-reduction step.  It also exercises
    the same Nataf sampling path as :meth:`monte_carlo_pf`.
    """
    import numpy as np

    from ..structural._batch import batch_u_to_x, batch_norm_cdf

    problem = build_problem(model, kappa, ignore_correlation=ignore_correlation)
    from .target_reliability_inverse import design_resistance_median  # local import

    r_median = design_resistance_median(model, kappa, ignore_correlation)
    mu_r = log(r_median)
    sigma_r = model.sigma_r
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((samples, problem.transform.dimension))
    x = batch_u_to_x(problem.transform, w)
    ln_f = np.log(x[:, 0]) + np.log(x[:, 1]) + np.log(x[:, 2])
    z = (ln_f - mu_r) / sigma_r
    pf_i = batch_norm_cdf(z)
    pf = float(np.mean(pf_i))
    std_error = float(np.std(pf_i, ddof=1) / np.sqrt(samples))
    beta = _beta_from_pf(pf)
    return {
        "samples": int(samples),
        "pf": pf,
        "std_error_pf": std_error,
        "cov_pf": std_error / pf if pf > 0.0 else inf,
        "beta": beta,
        "beta_one_sigma_band": [
            _beta_from_pf(pf + std_error),
            _beta_from_pf(max(pf - std_error, 1e-300)),
        ],
        "method": "crude MC, resistance integrated analytically (Rao-Blackwell)",
    }


def _conditional_entry(
    model: PAreaForceModel,
    kappa: float,
    samples: int,
    seed: int,
    beta_form: float,
    ignore_correlation: bool,
) -> dict:
    """Wrap :func:`conditional_mc_pf` with the FORM comparison flags."""
    entry = conditional_mc_pf(
        model, kappa, samples=samples, seed=seed, ignore_correlation=ignore_correlation
    )
    band = entry["beta_one_sigma_band"]
    entry["relative_error_vs_form"] = rel_error(entry["beta"], beta_form)
    entry["covers_form"] = bool(band[0] <= beta_form <= band[1])
    return entry


def design_table_rows() -> list[dict]:
    """``(gamma_F, gamma_M)`` pairs adjudicated: A.2 x A.3 plus the in-repo anchor."""
    rows: list[dict] = []
    for load in LOAD_FACTOR_ROWS:
        for res in RESISTANCE_FACTOR_ROWS:
            rows.append(
                {
                    "id": f"{load.row_key} x {res.row_key}",
                    "table_gamma_f": load.value,
                    "table_psi": res.value,
                    "source_tier": "secondary",
                    "primary_source_verified": False,
                    "load_description": load.description,
                    "resistance_description": res.description,
                }
            )
    anchor_load = next((r for r in REPO_ANCHOR_ROWS if r.quantity == "gamma_F"), None)
    anchor_res = next((r for r in REPO_ANCHOR_ROWS if r.quantity in ("gamma_M", "psi")), None)
    if anchor_load is not None and anchor_res is not None:
        rows.append(
            {
                "id": f"repo_anchor {anchor_load.row_key} x {anchor_res.row_key}",
                "table_gamma_f": anchor_load.value,
                "table_psi": anchor_res.value,
                "source_tier": "in_repo",
                "primary_source_verified": True,
                "load_description": anchor_load.description,
                "resistance_description": anchor_res.description,
            }
        )
    return rows


def _non_conservative_example(
    model: PAreaForceModel,
    kappa_table: float,
    kappa_star: float,
    beta_target: float,
    ignore_correlation: bool = False,
) -> dict:
    """Quantify what a table pair that under-provides ``kappa`` does to safety."""
    beta_ach, form, _r = beta_of_kappa(model, kappa_table, ignore_correlation)
    f_k = characteristic_force(model, ignore_correlation)
    pf_ach = float(norm_cdf(-beta_ach)) if beta_ach > 0.0 else 0.5
    pf_target = float(norm_cdf(-beta_target))
    return {
        "kappa_table": kappa_table,
        "kappa_star": kappa_star,
        "kappa_shortfall_rel": (kappa_star - kappa_table) / kappa_star,
        "beta_achieved": beta_ach,
        "beta_deficit": beta_target - beta_ach,
        "pf_achieved": pf_ach,
        "pf_target": pf_target,
        "pf_ratio_achieved_over_target": pf_ach / pf_target if pf_target > 0.0 else inf,
        "F_k_mn": f_k,
        "R_k_required_mn_by_table": kappa_table * f_k,
        "R_k_required_mn_by_calibration": kappa_star * f_k,
        "resistance_under_design_rel": (kappa_star - kappa_table) / kappa_star,
        "converged": form is not None and form.converged,
    }


def iso_deviation_map(
    model: PAreaForceModel,
    gamma_f: float,
    psi: float,
    beta_target: float = TARGET_BETA,
    tolerance_report: float = 0.15,
    ignore_correlation: bool = False,
) -> dict:
    """Criterion (d): row-by-row deviation map against the ISO 19906 tables."""
    kappa_star = gamma_f * psi
    rows: list[dict] = []
    for row in design_table_rows():
        kappa_table = row["table_gamma_f"] * row["table_psi"]
        dev_gamma = (row["table_gamma_f"] - gamma_f) / gamma_f
        dev_psi = (row["table_psi"] - psi) / psi
        dev_kappa = (kappa_table - kappa_star) / kappa_star
        exceeds = max(abs(dev_gamma), abs(dev_psi), abs(dev_kappa)) > tolerance_report
        beta_tab, _form_tab, _r_tab = beta_of_kappa(model, kappa_table, ignore_correlation)
        entry = dict(row)
        entry.update(
            {
                "calibrated_gamma_f": gamma_f,
                "calibrated_psi": psi,
                "calibrated_kappa": kappa_star,
                "kappa_table": kappa_table,
                "deviation_gamma_f_rel": dev_gamma,
                "deviation_psi_rel": dev_psi,
                "deviation_kappa_rel": dev_kappa,
                "exceeds_tolerance": exceeds,
                "tolerance": tolerance_report,
                "verdict": (
                    "under_provides_kappa" if kappa_table < kappa_star else "over_provides_kappa"
                ),
                "beta_achieved_table": beta_tab,
                "pf_achieved_table": float(norm_cdf(-beta_tab)) if beta_tab > 0.0 else 0.5,
            }
        )
        if exceeds and kappa_table < kappa_star:
            entry["non_conservative_example"] = _non_conservative_example(
                model, kappa_table, kappa_star, beta_target, ignore_correlation
            )
            entry["example_note"] = "quantified below (achieved beta and Pf ratio)"
        else:
            entry["non_conservative_example"] = None
            entry["example_note"] = (
                "no non-conservative case: deviation within the 15 % band"
                if not exceeds
                else "kappa_table exceeds the calibrated value (conservative side)"
            )
        rows.append(entry)
    return {
        "rows": rows,
        "tolerance": tolerance_report,
        "rows_total": len(rows),
        "rows_exceeding_tolerance": sum(1 for r in rows if r["exceeds_tolerance"]),
        "non_conservative_examples": sum(1 for r in rows if r["non_conservative_example"]),
        "provenance_note": (
            "A.2/A.3 values are secondary transcriptions; every verdict derived from them "
            "is conditional (primary_source_verified=False). The repo_anchor pair is "
            "implemented in this repository and fully verifiable."
        ),
        "design_cases_in_repo": [dataclasses.asdict(c) for c in DESIGN_CASES],
    }


def negative_control(
    model: PAreaForceModel,
    beta_target: float = TARGET_BETA,
    threshold: float = 0.05,
    starts: Sequence[tuple[float, float]] = INITIAL_GUESSES,
) -> dict:
    """Criterion (e): ignoring the p-A correlation must inflate ``gamma*`` by >= 5 %.

    Two calibrations on the *same marginals*: the physical model (negative p-A
    correlation implied by the ISO exponent) and the negative control
    (``rho = 0``, both marginals kept).  Dropping the correlation inflates the
    load scatter and therefore the calibrated load factor; the check is that the
    inflation is detected rather than silently absorbed.
    """
    corr = solve_factors_multistart(model, beta_target=beta_target, starts=starts)
    indep = solve_factors_multistart(
        model, beta_target=beta_target, starts=starts, ignore_correlation=True
    )
    sol_c = corr["solution"]
    sol_i = indep["solution"]
    delta_gamma = (sol_i["gamma_f"] - sol_c["gamma_f"]) / sol_c["gamma_f"]
    delta_kappa = (sol_i["kappa_star"] - sol_c["kappa_star"]) / sol_c["kappa_star"]
    mom_c = force_moments(model, False)
    mom_i = force_moments(model, True)
    beta_physical_at_naive = beta_of_kappa(model, sol_i["kappa_star"], False)[0]
    return {
        "rho_p_a": model.pressure_area_correlation(),
        "correlated": {
            "gamma_f": sol_c["gamma_f"],
            "psi": sol_c["psi"],
            "kappa_star": sol_c["kappa_star"],
            "sigma_ln_force": mom_c["sigma_ln_force"],
            "cov_force": mom_c["cov_force"],
            "converged": sol_c["converged"],
        },
        "independent": {
            "gamma_f": sol_i["gamma_f"],
            "psi": sol_i["psi"],
            "kappa_star": sol_i["kappa_star"],
            "sigma_ln_force": mom_i["sigma_ln_force"],
            "cov_force": mom_i["cov_force"],
            "converged": sol_i["converged"],
        },
        "delta_gamma_f_rel": delta_gamma,
        "delta_kappa_rel": delta_kappa,
        "sigma_ln_force_inflation_rel": (mom_i["sigma_ln_force"] - mom_c["sigma_ln_force"])
        / mom_c["sigma_ln_force"],
        "beta_physical_model_at_naive_kappa": beta_physical_at_naive,
        "beta_overdesign": beta_physical_at_naive - beta_target,
        "threshold": threshold,
        "detected": bool(delta_gamma >= threshold),
        "direction_as_expected": bool(delta_gamma > 0.0),
    }


def beta_target_sweep_inverse(
    model: PAreaForceModel,
    beta_targets: Sequence[float] = (3.3, 3.8, 4.0, 4.3),
    ignore_correlation: bool = False,
) -> list[dict]:
    """Calibrated pair versus the target reliability (the ``beta`` sensitivity)."""
    rows = []
    for beta_t in beta_targets:
        res = solve_factors(model, beta_target=beta_t, ignore_correlation=ignore_correlation)
        rows.append(
            {
                "beta_target": beta_t,
                "pf_target": float(norm_cdf(-beta_t)),
                "gamma_f": res["gamma_f"],
                "psi": res["psi"],
                "kappa_star": res["kappa_star"],
                "beta_achieved": res["beta_achieved"],
                "iterations": res["iterations"],
                "converged": res["converged"],
            }
        )
    for i, row in enumerate(rows):
        if 0 < i < len(rows) - 1:
            lo, hi = rows[i - 1], rows[i + 1]
            row["d_gamma_f_d_beta_central"] = (hi["gamma_f"] - lo["gamma_f"]) / (
                hi["beta_target"] - lo["beta_target"]
            )
    return rows


def model_parameter_sensitivity(
    model: PAreaForceModel,
    relative_step: float = 0.05,
    ignore_correlation: bool = False,
) -> list[dict]:
    """``d beta / d(theta)`` for the physical inputs (central differences on FORM)."""
    base = solve_factors(model, ignore_correlation=ignore_correlation)
    kappa = base["kappa_star"]
    rows = []
    for name in ("exponent", "a_cov", "pressure_model_cov", "load_model_cov", "resistance_cov"):
        nominal = getattr(model, name)
        h = relative_step * max(abs(nominal), 1e-6)
        values = []
        for sign in (1.0, -1.0):
            perturbed = model.with_(**{name: nominal + sign * h})
            values.append(beta_of_kappa(perturbed, kappa, ignore_correlation)[0])
        rows.append(
            {
                "parameter": name,
                "nominal": nominal,
                "step": h,
                "beta_plus": values[0],
                "beta_minus": values[1],
                "d_beta_d_parameter": (values[0] - values[1]) / (2.0 * h),
            }
        )
    return rows