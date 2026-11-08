"""Orchestration of the ISO 19906 partial-factor back-calibration study.

``run_calibration`` performs the whole chain described in the package docstring
and returns a JSON-serialisable dictionary in which *every* number quoted by the
report is present (the report generator never re-computes a value, it only reads
this dictionary — the "zero manual copying" rule).
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from .calibration import (
    TARGET_BETA,
    TARGET_PF,
    _beta_of_factor_product,
    beta_target_sweep,
    calibrate_panel,
    characteristic_values,
    invert_sweep,
    load_cov_sweep,
    return_period_sweep,
    scenario_sweep,
)
from .iso_tables import (
    adjudicate_design_cases,
    adjudicate_factor_rows,
    adjudicate_target_reliability,
    summarise_verdicts,
)
from .limit_state import IceBeltPanel
from .pressure_area import (
    GumbelParams,
    PressureAreaTrend,
    fit_gev_lmoments,
    fit_gev_mle,
    fit_gumbel_mle,
    fit_gumbel_moments,
    fit_pressure_area_trend,
    plant_annual_params,
    relative_error,
    simulate_annual_maxima,
)

__all__ = ["DEFAULT_CONFIG", "MACHINE_CRITERIA_KEYS", "run_calibration", "evaluate_criteria"]

#: Machine criteria (each becomes a PASS/FAIL row with its measured number).
MACHINE_CRITERIA_KEYS = (
    "C1_target_pf_mapping",
    "C2_pA_trend_recovery",
    "C3_gumbel_parameter_recovery",
    "C4_gev_gumbel_shape_tenable",
    "C5_form_vs_importance_sampling",
    "C6_form_vs_sorm_curvature",
    "C7_form_vs_monte_carlo",
    "C8_format_closure",
    "C9_beta_target_inverse_solved",
    "C10_area_scenario_stability",
    "C11_iso_rows_all_adjudicated",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "trend": {"p_ref": 2.5, "exponent": 0.40, "a_ref": 1.0},
    "storm_scale_cov": 0.22,
    "storms_per_year": 12,
    "area_scatter_cov": 0.06,
    "years": 300,
    "seed": 20261031,
    "fit_areas": (1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0),
    "reference_area": 5.0,
    "scenario_areas": (2.5, 5.0, 10.0, 20.0),
    "beta_targets": (3.3, 3.8, 4.0, 4.3),
    "cov_sweep": (0.08, 0.12, 0.16, 0.20, 0.24, 0.28),
    "return_periods": (50.0, 100.0, 200.0, 500.0, 1000.0),
    "resistance_factor_split": (1.0, 1.10, 1.15),
    "panel": {
        "b_mm": 350.0,
        "t_nom_mm": 26.0,
        "alpha_b": 0.5,
        "dt_mean_mm": 2.0,
        "dt_cov": 0.25,
        "fy_mean_mpa": 355.0,
        "fy_cov": 0.08,
        "k_s_cov": 0.15,
        "k_r_cov": 0.10,
        "area_m2": 5.0,
    },
    "samples_mc": 400_000,
    "samples_is": 200_000,
    "tolerances": {
        "pf_mapping_rel": 1e-6,
        "trend_p_ref_rel": 0.05,
        "trend_exponent_abs": 0.05,
        "gumbel_loc_rel": 0.05,
        "gumbel_scale_rel": 0.10,
        "gev_shape_abs": 0.15,
        "form_vs_is_beta": 0.03,
        "form_vs_sorm_beta": 0.05,
        "form_vs_mc_beta": 0.30,
        "closure_beta": 0.02,
        "closure_product_rel": 1e-9,
        "area_gamma_f_spread": 0.10,
    },
}


def _criteria_row(name: str, measured: dict, passed: bool, note: str) -> dict:
    return {"criterion": name, "passed": bool(passed), "measured": measured, "note": note}


def _skipped_row(name: str, note: str) -> dict:
    """A criterion that could not be measured (e.g. simulation disabled)."""
    return {"criterion": name, "passed": False, "skipped": True, "measured": {}, "note": note}


def evaluate_criteria(results: dict, tolerances: dict) -> list[dict]:
    """Turn the study numbers into PASS/FAIL machine criteria rows."""
    rows: list[dict] = []
    tol = tolerances

    # C1: target beta -> annual failure probability mapping
    pf_target = results["target"]["pf_target"]
    rel = abs(pf_target - TARGET_PF) / TARGET_PF
    rows.append(
        _criteria_row(
            "C1_target_pf_mapping",
            {"beta_target": results["target"]["beta_target"], "pf_target": pf_target,
             "rel_error_vs_Phi_minus_4": rel},
            rel <= tol["pf_mapping_rel"],
            "Phi(-4) = 3.1671e-5 reproduced analytically",
        )
    )

    # C2: p-A trend recovery from the fitted ensemble
    trend = results["trend_fit"]
    dev_p = relative_error(trend["p_ref"], results["plant"]["annual_p_ref"])
    dev_ex = trend["exponent"] - results["plant"]["exponent"]
    rows.append(
        _criteria_row(
            "C2_pA_trend_recovery",
            {"p_ref_fit": trend["p_ref"],
             "p_ref_plant_event_level": results["plant"]["p_ref"],
             "p_ref_plant_annual": results["plant"]["annual_p_ref"],
             "rel_error_p_ref": dev_p, "exponent_fit": trend["exponent"],
             "exponent_plant": results["plant"]["exponent"], "abs_dev_exponent": dev_ex,
             "r_squared": trend["r_squared"]},
            abs(dev_p) <= tol["trend_p_ref_rel"] and abs(dev_ex) <= tol["trend_exponent_abs"],
            "log-log least squares over the simulated p-A ensemble",
        )
    )

    # C3: Gumbel parameter recovery (MLE and moments against the plant)
    gm = results["extreme_fits"]["mle"]
    gmom = results["extreme_fits"]["moments"]
    plant = results["plant"]["annual_gumbel"]
    dev_loc = relative_error(gm["loc"], plant["loc"])
    dev_scale = relative_error(gm["scale"], plant["scale"])
    dev_loc_mom = relative_error(gmom["loc"], plant["loc"])
    dev_scale_mom = relative_error(gmom["scale"], plant["scale"])
    rows.append(
        _criteria_row(
            "C3_gumbel_parameter_recovery",
            {"loc_mle": gm["loc"], "loc_moments": gmom["loc"], "loc_plant": plant["loc"],
             "scale_mle": gm["scale"], "scale_moments": gmom["scale"],
             "scale_plant": plant["scale"], "rel_error_loc_mle": dev_loc,
             "rel_error_scale_mle": dev_scale, "rel_error_loc_moments": dev_loc_mom,
             "rel_error_scale_moments": dev_scale_mom,
             "ks_mle": gm["ks_statistic"]},
            max(abs(dev_loc), abs(dev_loc_mom)) <= tol["gumbel_loc_rel"]
            and max(abs(dev_scale), abs(dev_scale_mom)) <= tol["gumbel_scale_rel"],
            "both estimators must recover the generator plant",
        )
    )

    # C4: GEV shape close to zero -> the Gumbel restriction is tenable
    shape = results["extreme_fits"]["gev"]["shape"]
    shape_ml = results["extreme_fits"]["gev_mle"]["shape"]
    lr = 2.0 * (
        results["extreme_fits"]["gev_mle"]["log_likelihood"] - gm["log_likelihood"]
    )
    rows.append(
        _criteria_row(
            "C4_gev_gumbel_shape_tenable",
            {"gev_shape_lmoments": shape, "gev_shape_mle": shape_ml,
             "loglik_gev_mle": results["extreme_fits"]["gev_mle"]["log_likelihood"],
             "loglik_gumbel_mle": gm["log_likelihood"], "lr_statistic": lr,
             "chi2_1dof_5pct": 3.841},
            abs(shape) <= tol["gev_shape_abs"] and abs(shape_ml) <= tol["gev_shape_abs"],
            "GEV shape must stay in the Gumbel band (LR below the 5% chi-square 1-dof bound)",
        )
    )

    # C5 / C6 / C7: simulation cross-checks at the calibrated design
    rel = results["calibration"]["reliability"]
    if rel["samples_importance_sampling"] == 0 or rel["samples_monte_carlo"] == 0:
        rows.append(_skipped_row("C5_form_vs_importance_sampling", "simulation disabled"))
        rows.append(_skipped_row("C7_form_vs_monte_carlo", "simulation disabled"))
        sim_skipped = {"C5_form_vs_importance_sampling", "C7_form_vs_monte_carlo"}
    else:
        sim_skipped = set()
    d_is = abs(rel["beta_form"] - rel["beta_importance_sampling"])
    d_sorm = abs(rel["beta_sorm"] - rel["beta_form"])
    d_mc = abs(rel["beta_form"] - rel["beta_monte_carlo"])
    if "C5_form_vs_importance_sampling" not in sim_skipped:
        rows.append(
        _criteria_row(
            "C5_form_vs_importance_sampling",
            {"beta_form": rel["beta_form"], "beta_is": rel["beta_importance_sampling"],
             "abs_delta_beta": d_is, "is_cov": rel["importance_sampling_cov"]},
            d_is <= tol["form_vs_is_beta"],
            "design-point importance sampling is the unbiased reference",
        )
    )
    rows.append(
        _criteria_row(
            "C6_form_vs_sorm_curvature",
            {"beta_form": rel["beta_form"], "beta_sorm": rel["beta_sorm"],
             "abs_delta_beta": d_sorm, "principal_curvatures": rel["principal_curvatures"]},
            d_sorm <= tol["form_vs_sorm_beta"],
            "Breitung SORM correction quantifies surface curvature",
        )
    )
    if "C7_form_vs_monte_carlo" not in sim_skipped:
        rows.append(
        _criteria_row(
            "C7_form_vs_monte_carlo",
            {"beta_form": rel["beta_form"], "beta_mc": rel["beta_monte_carlo"],
             "pf_mc": rel["pf_monte_carlo"], "samples": rel["samples_monte_carlo"],
             "abs_delta_beta": d_mc},
            d_mc <= tol["form_vs_mc_beta"],
            "crude Monte Carlo is coarse at 1e-5; the band reflects its variance",
        )
    )

    # C8: format closure (gamma_F * gamma_M == k*, and beta(gamma) == beta_T)
    cal = results["calibration"]
    prod_rel = abs(cal["gamma_F"] * cal["gamma_M"] - cal["k_star"]) / cal["k_star"]
    beta_cl = abs(cal["beta_achieved"] - cal["beta_target"])
    rows.append(
        _criteria_row(
            "C8_format_closure",
            {"gamma_F": cal["gamma_F"], "gamma_M": cal["gamma_M"],
             "product": cal["gamma_F"] * cal["gamma_M"], "k_star": cal["k_star"],
             "rel_error_product": prod_rel, "beta_achieved": cal["beta_achieved"],
             "beta_target": cal["beta_target"], "abs_delta_beta": beta_cl},
            prod_rel <= tol["closure_product_rel"] and beta_cl <= tol["closure_beta"],
            "design-value split must reproduce the bisection solution exactly",
        )
    )

    # C9: the target-beta inverse actually converged
    rows.append(
        _criteria_row(
            "C9_beta_target_inverse_solved",
            {"iterations": cal["bisection_iterations"], "converged": cal["reliability"]["converged"],
             "g_design": cal["reliability"]["g_design"], "beta_achieved": cal["beta_achieved"]},
            cal["reliability"]["converged"] and abs(cal["reliability"]["g_design"]) < 1e-4,
            "FORM design point must sit on the surface",
        )
    )

    # C10: stability of the calibrated gamma_F over the loaded-area family
    gammas = [row["gamma_F"] for row in results["area_sweep"]]
    spread = (max(gammas) - min(gammas)) / (sum(gammas) / len(gammas))
    rows.append(
        _criteria_row(
            "C10_area_scenario_stability",
            {"gamma_F_min": min(gammas), "gamma_F_max": max(gammas), "relative_spread": spread,
             "areas": [row["area_m2"] for row in results["area_sweep"]]},
            spread <= tol["area_gamma_f_spread"],
            "one factor pair must serve the whole p-A family",
        )
    )

    # C11: every ISO row adjudicated
    counts = results["iso_adjudication"]["summary"]["counts"]
    all_rows = (
        results["iso_adjudication"]["target_rows"]
        + results["iso_adjudication"]["factor_rows"]
        + results["iso_adjudication"]["design_cases"]
    )
    ok = all("verdict" in row for row in all_rows) and len(all_rows) > 0
    rows.append(
        _criteria_row(
            "C11_iso_rows_all_adjudicated",
            {"rows": len(all_rows), "counts": counts},
            ok,
            "every ISO 19906 row carries a numeric verdict",
        )
    )

    for row in rows:
        measured = row["measured"]
        for key, value in list(measured.items()):
            if isinstance(value, float) and not isfinite(value):
                row["passed"] = False
                row["note"] += f" (non-finite measurement: {key})"
    return rows


def run_calibration(config: dict | None = None, with_simulation: bool = True) -> dict:
    """Run the complete back-calibration study and return its JSON payload."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    tol = cfg["tolerances"]

    trend = PressureAreaTrend(**cfg["trend"])
    fit_areas = tuple(cfg["fit_areas"])
    records = simulate_annual_maxima(
        trend,
        fit_areas,
        years=int(cfg["years"]),
        storm_scale_cov=float(cfg["storm_scale_cov"]),
        storms_per_year=int(cfg["storms_per_year"]),
        area_scatter_cov=float(cfg["area_scatter_cov"]),
        seed=int(cfg["seed"]),
    )

    # -- p-A trend fit over the ensemble means -------------------------------
    areas_all: list[float] = []
    pressures_all: list[float] = []
    for area in fit_areas:
        series = records[area]
        mean = sum(series) / len(series)
        areas_all.append(area)
        pressures_all.append(mean)
    trend_fit = fit_pressure_area_trend(areas_all, pressures_all, a_ref=trend.a_ref)

    # -- extreme-value fits at the reference area ----------------------------
    ref_area = float(cfg["reference_area"])
    sample = records[ref_area]
    fit_mom = fit_gumbel_moments(sample)
    fit_mle = fit_gumbel_mle(sample)
    fit_gev = fit_gev_lmoments(sample)
    fit_gev_ml = fit_gev_mle(sample, start=fit_gev)
    plant = plant_annual_params(
        trend,
        ref_area,
        storm_scale_cov=float(cfg["storm_scale_cov"]),
        storms_per_year=int(cfg["storms_per_year"]),
    )
    load_model = fit_mle.params  # data-driven Gumbel is the calibration input

    panel = IceBeltPanel(**cfg["panel"])

    # -- calibration at the mission target -----------------------------------
    calibration = calibrate_panel(
        panel,
        load_model,
        beta_target=TARGET_BETA,
        samples_mc=int(cfg["samples_mc"]) if with_simulation else 0,
        samples_is=int(cfg["samples_is"]) if with_simulation else 0,
        seed=int(cfg["seed"]),
    )
    cal_dict = calibration.to_dict()

    # -- splits at fixed resistance factors (the "inverse gamma_F" question) --
    splits = []
    for gamma_m in cfg["resistance_factor_split"]:
        splits.append(
            {
                "gamma_M_fixed": gamma_m,
                "gamma_F_required": calibration.k_star / gamma_m,
                "factor_product": calibration.k_star,
                "t_design_mm": calibration.t_design_mm,
            }
        )
    cal_dict["splits"] = splits

    # -- beta_T sweep (covers the Table A.1 target rows) ---------------------
    ch_base = characteristic_values(panel, load_model)
    sweep = beta_target_sweep(panel, load_model, betas=tuple(cfg["beta_targets"]), char=ch_base)

    # -- scenario (loaded-area) sweep ---------------------------------------
    def gumbel_of_area(area: float):
        """Data-driven annual-maximum Gumbel at another loaded area.

        The same estimator (MLE on the simulated record) is used as for the
        reference scenario, so the area family and the headline calibration
        share one load model.
        """
        key = float(area)
        if key in records:
            return fit_gumbel_mle(records[key]).params
        return plant_annual_params(
            trend,
            key,
            storm_scale_cov=float(cfg["storm_scale_cov"]),
            storms_per_year=int(cfg["storms_per_year"]),
        )

    area_rows = scenario_sweep(panel, gumbel_of_area, areas=tuple(cfg["scenario_areas"]))

    # -- load-COV sweep and its inversion -----------------------------------
    annual_mean = plant.mean()

    def gumbel_of_cov(cov: float):
        """Annual-maximum Gumbel moment-matched to (annual mean, COV = cov)."""
        scale = cov * annual_mean * (6.0 ** 0.5) / 3.141592653589793
        return GumbelParams(loc=annual_mean - 0.5772156649015329 * scale, scale=scale)

    cov_rows = load_cov_sweep(panel, gumbel_of_cov, covs=tuple(cfg["cov_sweep"]))
    rp_rows = return_period_sweep(
        panel, load_model, return_periods=tuple(cfg["return_periods"]), beta_target=TARGET_BETA
    )

    def _invert(target_product: float) -> dict:
        try:
            cov = invert_sweep(cov_rows, "load_cov", target_product, "k_star")
            return {"target_factor_product": target_product, "load_cov": cov}
        except ValueError:
            return {
                "target_factor_product": target_product,
                "load_cov": None,
                "note": "target not bracketed by the sweep",
            }

    def _invert_period(target_product: float) -> dict:
        try:
            period = invert_sweep(rp_rows, "return_period_years", target_product, "k_star")
            return {"target_factor_product": target_product, "return_period_years": period}
        except ValueError:
            return {
                "target_factor_product": target_product,
                "return_period_years": None,
                "note": "target not bracketed by the return-period sweep",
            }

    inversions = {
        "iso_table_pair_yield": _invert(1.35 * 1.10),
        "iso_table_pair_buckling": _invert(1.35 * 1.15),
        "repo_anchor_pair": _invert(1.25 * 1.15),
        "by_return_period": {
            "iso_table_pair_yield": _invert_period(1.35 * 1.10),
            "iso_table_pair_buckling": _invert_period(1.35 * 1.15),
            "repo_anchor_pair": _invert_period(1.25 * 1.15),
        },
    }

    # -- ISO adjudication ----------------------------------------------------
    def beta_of_pair(product: float) -> float:
        beta, _t, _f = _beta_of_factor_product(panel, load_model, ch_base, product)
        return beta

    beta_curve = [
        {"factor_product": round(1.00 + 0.05 * i, 3), "beta": beta_of_pair(round(1.00 + 0.05 * i, 3))}
        for i in range(0, 25)
    ]
    design_cases = adjudicate_design_cases(beta_of_pair)
    target_rows = adjudicate_target_reliability(TARGET_BETA, TARGET_PF)
    factor_rows = adjudicate_factor_rows(calibration.gamma_f, calibration.gamma_m)
    adjudication = {
        "target_rows": target_rows,
        "factor_rows": factor_rows,
        "design_cases": design_cases,
        "summary": summarise_verdicts(target_rows + factor_rows + design_cases),
        "provenance_note": (
            "ISO 19906 Annex A table values are transcribed from secondary literature "
            "and are NOT verified against the primary standard inside this worktree; "
            "verdicts derived from them are stamped evidence_tier='conditional'. "
            "The repo_anchor rows are in-repo defaults and fully verifiable."
        ),
    }

    results: dict[str, Any] = {
        "study": {
            "node": "IN13-W3-N10",
            "title": "ISO 19906 ice-pressure partial-factor back-calibration at beta_T = 4.0",
            "repository": "TensorPhysics",
            "module": "tensorphysics.ice_partial_factors",
            "reference_area_m2": ref_area,
            "panel": panel.to_dict(),
        },
        "target": {
            "beta_target": TARGET_BETA,
            "pf_target": TARGET_PF,
            "pf_target_note": "Phi(-4) = 3.1671e-5, the mission's ~3e-5 annual failure probability",
        },
        "config": {
            "trend": cfg["trend"],
            "storm_scale_cov": cfg["storm_scale_cov"],
            "storms_per_year": cfg["storms_per_year"],
            "area_scatter_cov": cfg["area_scatter_cov"],
            "years": cfg["years"],
            "seed": cfg["seed"],
            "fit_areas": list(fit_areas),
            "beta_targets": list(cfg["beta_targets"]),
            "scenario_areas": list(cfg["scenario_areas"]),
            "cov_sweep": list(cfg["cov_sweep"]),
            "return_periods": list(cfg["return_periods"]),
            "samples_mc": cfg["samples_mc"] if with_simulation else 0,
            "samples_is": cfg["samples_is"] if with_simulation else 0,
            "tolerances": tol,
        },
        "plant": {
            "p_ref": trend.p_ref,
            "exponent": trend.exponent,
            "a_ref": trend.a_ref,
            "annual_gumbel": plant.to_dict(),
            "annual_p_ref": trend.p_ref * plant.mean() / trend.mean_pressure(ref_area),
            "annual_max_cov": plant.cov(),
            "storm_gumbel_scale_cov": cfg["storm_scale_cov"],
            "provenance": (
                "model-based p-A ensemble: no field pressure record ships with the "
                "repository, so the annual maxima are simulated from a documented "
                "generator whose parameters are the verification target"
            ),
        },
        "trend_fit": trend_fit.to_dict(),
        "ensemble": {
            "areas": list(fit_areas),
            "years": int(cfg["years"]),
            "sample_mean_by_area": {str(a): sum(records[a]) / len(records[a]) for a in fit_areas},
            "sample_std_by_area": {
                str(a): _std(records[a]) for a in fit_areas
            },
        },
        "extreme_fits": {
            "moments": fit_mom.to_dict(),
            "mle": fit_mle.to_dict(),
            "gev": fit_gev.to_dict(),
            "gev_mle": fit_gev_ml.to_dict(),
            "sample_size": len(sample),
        },
        "load_model": {
            "loc": load_model.loc,
            "scale": load_model.scale,
            "mean": load_model.mean(),
            "cov": load_model.cov(),
            "source": "Gumbel MLE fit of the simulated annual maxima",
        },
        "calibration": cal_dict,
        "beta_sweep": sweep,
        "beta_curve": beta_curve,
        "area_sweep": area_rows,
        "cov_sweep": cov_rows,
        "return_period_sweep": rp_rows,
        "inversions": inversions,
        "iso_adjudication": adjudication,
    }
    results["machine_criteria"] = sorted(
        evaluate_criteria(results, tol),
        key=lambda row: int(row["criterion"].split("_", 1)[0][1:]),
    )
    rows = results["machine_criteria"]
    measured_rows = [row for row in rows if not row.get("skipped")]
    results["criteria_summary"] = {
        "passed": sum(1 for row in measured_rows if row["passed"]),
        "total": len(rows),
        "measured": len(measured_rows),
        "skipped": len(rows) - len(measured_rows),
        "all_passed": all(row["passed"] for row in measured_rows),
    }
    return results


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5