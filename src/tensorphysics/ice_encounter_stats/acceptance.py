"""Acceptance driver: AC1..AC6 of the ice-encounter statistics node.

Each criterion is a pure function of the injected manifest so that the whole
node is reproducible from ``configs/prereg_p233w2_n05.json`` plus a seed offset.
Nothing here estimates a quantity twice: the record synthesis, event extraction
and statistics all come from the sibling modules, and every number that reaches
the report is written into ``acceptance.json`` first.

Anchors
-------
All comparisons use *population* anchors (Monte-Carlo over the injected mark
spectra, :func:`~tensorphysics.ice_encounter_stats.signal.population_marks`) so
that a single record's fluctuation is measured against the true population and
not against another realization.  The record-conditional Campbell value (built
from that record's own marks) is reported alongside as a decomposition of the
error.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .config import CaseSpec, ConfigError, Manifest, manifest_hash
from .events import auto_level, extract_events
from .signal import Record, population_marks, simulate_record
from .statistics import (CampbellPrediction, bonferroni_adjust, campbell_moments,
                         correlation_time, effective_sample_size,
                         ks_exponential, relative_error,
                         standard_error_of_mean)
from .stopping import (PilotStatistics, StoppingRule,
                       tail_factor_from_quantile)

NODE_ID = "P233W2-N05"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if not np.isfinite(v) else round(v, digits)
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return _round(value.tolist(), digits)
    if isinstance(value, Mapping):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


def _block(manifest: Manifest, key: str) -> Mapping[str, Any]:
    execution = manifest.execution
    if key not in execution:
        raise ConfigError(f"manifest.execution is missing the {key!r} block")
    return execution[key]


def _spp(manifest: Manifest) -> int:
    return int(manifest.execution.get("numerics", {}).get("samples_per_pulse", 40))


def _campbell(case: CaseSpec, record: Record) -> CampbellPrediction:
    """Conditional Campbell prediction from the marks that generated ``record``."""
    return campbell_moments(case.encounter_rate, record.marks["area_ns"],
                            record.marks["sq_area_n2s"])


def _population(case: CaseSpec, n_marks: int, seed: int) -> CampbellPrediction:
    areas, sq = population_marks(case, n_marks, seed)
    return campbell_moments(case.encounter_rate, areas, sq)


def _recover(case: CaseSpec, record: Record, cfg: Mapping[str, Any]) -> tuple:
    level = auto_level(record.force, float(cfg.get("level_quantile", 0.95)),
                       float(cfg.get("level_fraction", 0.30)))
    events = extract_events(record.t, record.force, level,
                            min_gap_s=float(cfg.get("min_gap_s", 0.0)))
    return events, level


# --------------------------------------------------------------------------- #
# AC1 -- Poisson-ness of uniform-field encounters
# --------------------------------------------------------------------------- #
def ac1_poisson_uniform(manifest: Manifest, case: CaseSpec) -> dict:
    cfg = _block(manifest, "ac1")
    spp = _spp(manifest)
    per_seed = []
    for seed in cfg["seeds"]:
        record = simulate_record(case, float(cfg["duration_s"]), int(seed), spp)
        events, level = _recover(case, record, cfg)
        ks_recovered = ks_exponential(events.intervals)
        truth = np.diff(record.marks["encounter_times_s"])
        ks_injected = ks_exponential(truth) if truth.size >= 5 else None
        lags, rho, tau_c, tau0 = correlation_time(record.t, record.force,
                                                  float(cfg.get("max_lag_s", 5.0)))
        per_seed.append({
            "seed": int(seed),
            "n_injected": int(record.marks["encounter_times_s"].size),
            "n_recovered": events.n_events,
            "recovery_fraction": float(events.n_events
                                       / max(record.marks["encounter_times_s"].size, 1)),
            "detection_level_n": level,
            "recovered_rate_per_s": events.rate,
            "injected_rate_per_s": float(case.encounter_rate),
            "ks_recovered": ks_recovered.to_dict(),
            "ks_injected": None if ks_injected is None else ks_injected.to_dict(),
            "interval_cv": float(np.std(events.intervals, ddof=1) / np.mean(events.intervals)),
            "intervals_s": _round(events.intervals),
            "tau_c_s": float(tau_c),
            "zero_crossing_s": float(tau0),
            "acf": {"lags_s": _round(lags[: int(cfg.get("acf_points", 120))]),
                    "rho": _round(rho[: int(cfg.get("acf_points", 120))])},
        })
    p_recovered = [s["ks_recovered"]["p_value"] for s in per_seed]
    p_injected = [s["ks_injected"]["p_value"] for s in per_seed
                  if s["ks_injected"] is not None]
    min_adj, family = bonferroni_adjust(p_recovered)
    min_adj_truth, family_truth = (bonferroni_adjust(p_injected) if p_injected
                                   else (float("nan"), 0))
    return {
        "min_p_value": float(min(p_recovered)),
        "min_adjusted_p_value": float(min_adj),
        "family_size": int(family),
        "min_adjusted_p_value_truth": float(min_adj_truth),
        "family_size_truth": int(family_truth),
        "detection_rate_ratio": float(np.mean(
            [s["recovered_rate_per_s"] / s["injected_rate_per_s"] for s in per_seed])),
        "mean_recovery_fraction": float(np.mean([s["recovery_fraction"] for s in per_seed])),
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------- #
# AC2 -- negative control: a clustered field must be rejected
# --------------------------------------------------------------------------- #
def ac2_clustered_control(manifest: Manifest, clustered: CaseSpec) -> dict:
    cfg = _block(manifest, "ac2")
    spp = _spp(manifest)
    per_seed = []
    for seed in cfg["seeds"]:
        record = simulate_record(clustered, float(cfg["duration_s"]), int(seed), spp)
        events, level = _recover(clustered, record, cfg)
        ks_recovered = ks_exponential(events.intervals)
        truth = np.diff(record.marks["encounter_times_s"])
        ks_injected = ks_exponential(truth) if truth.size >= 5 else None
        per_seed.append({
            "seed": int(seed),
            "n_injected": int(record.marks["encounter_times_s"].size),
            "n_recovered": events.n_events,
            "detection_level_n": level,
            "ks_recovered": ks_recovered.to_dict(),
            "ks_injected": None if ks_injected is None else ks_injected.to_dict(),
            "interval_cv": float(np.std(events.intervals, ddof=1) / np.mean(events.intervals)),
            "interval_cv_truth": float(np.std(truth, ddof=1) / np.mean(truth))
            if truth.size > 1 else float("nan"),
            "intervals_s": _round(events.intervals),
        })
    p_recovered = [s["ks_recovered"]["p_value"] for s in per_seed]
    p_injected = [s["ks_injected"]["p_value"] for s in per_seed
                  if s["ks_injected"] is not None]
    max_p = float(max(p_recovered))
    return {
        "max_p_value": max_p,
        "max_adjusted_p_value": float(min(1.0, max_p * len(p_recovered))),
        "min_p_value_truth": float(min(p_injected)) if p_injected else float("nan"),
        "mean_interval_cv": float(np.mean([s["interval_cv"] for s in per_seed])),
        "mean_interval_cv_truth": float(np.mean([s["interval_cv_truth"] for s in per_seed])),
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------- #
# AC3 -- Campbell shot-noise variance
# --------------------------------------------------------------------------- #
def ac3_campbell_variance(manifest: Manifest) -> dict:
    cfg = _block(manifest, "ac3")
    spp = _spp(manifest)
    n_marks = int(cfg.get("population_marks", 200_000))
    rows = []
    for index, name in enumerate(cfg["cases"]):
        case = manifest.case(name)
        pop = _population(case, n_marks, int(cfg.get("mark_seed", 1301)) + index)
        variances, means, cond_var, cond_mean = [], [], [], []
        for seed in cfg["seeds"]:
            record = simulate_record(case, float(cfg["duration_s"]), int(seed), spp)
            cond = _campbell(case, record)
            variances.append(record.var)
            means.append(record.mean)
            cond_var.append(cond.variance)
            cond_mean.append(cond.mean)
        var_mean = float(np.mean(variances))
        probe = simulate_record(case, float(cfg["duration_s"]),
                                int(cfg["seeds"][0]) + 9173, spp)
        _, _, tau_c, _ = correlation_time(probe.t, probe.force,
                                          float(cfg.get("max_lag_s", 5.0)))
        rows.append({
            "case": name,
            "encounter_rate_per_s": float(case.encounter_rate),
            "variance_measured_n2": var_mean,
            "variance_campbell_population_n2": pop.variance,
            "variance_relative_error": relative_error(var_mean, pop.variance),
            "variance_campbell_conditional_n2": float(np.mean(cond_var)),
            "variance_relative_error_conditional": relative_error(var_mean, float(np.mean(cond_var))),
            "mean_measured_n": float(np.mean(means)),
            "mean_campbell_population_n": pop.mean,
            "mean_relative_error": relative_error(float(np.mean(means)), pop.mean),
            "cv_measured": float(np.sqrt(var_mean) / np.mean(means)),
            "cv_campbell": pop.cv,
            "n_seeds": len(cfg["seeds"]),
            "duration_s": float(cfg["duration_s"]),
            "tau_c_s": float(tau_c),
        })
    worst = max(row["variance_relative_error"] for row in rows)
    return {
        "max_variance_relative_error": float(worst),
        "max_variance_relative_error_conditional": float(
            max(row["variance_relative_error_conditional"] for row in rows)),
        "max_mean_relative_error": float(max(row["mean_relative_error"] for row in rows)),
        "cases": rows,
        "n_cases": len(rows),
    }


# --------------------------------------------------------------------------- #
# AC4 -- T_req scaling of the ensemble spread
# --------------------------------------------------------------------------- #
def ac4_treq_scaling(manifest: Manifest, case: CaseSpec) -> dict:
    cfg = _block(manifest, "ac4")
    spp = _spp(manifest)
    rule = StoppingRule(eps=float(cfg["eps"]),
                        tail_factor=float(cfg.get("tail_factor", 1.0)))

    reference = simulate_record(case, float(cfg["reference_duration_s"]),
                                int(cfg["seeds"][0]), spp)
    pop = _population(case, int(cfg.get("population_marks", 200_000)),
                      int(cfg.get("mark_seed", 4401)))
    _, _, tau_c, _ = correlation_time(reference.t, reference.force,
                                      float(cfg.get("max_lag_s", 5.0)))
    t_req = rule.required_time(pop.cv, tau_c)

    rows = []
    for mult in cfg["multipliers"]:
        duration = float(mult) * t_req
        n_ens = int(cfg["ensembles"])
        means = np.empty(n_ens)
        for k in range(n_ens):
            seed = int(cfg["seeds"][0]) + 1000 * (k + 1) + int(round(100 * float(mult)))
            means[k] = simulate_record(case, duration, seed, spp).mean
        std_obs = float(np.std(means, ddof=1))
        std_pred = standard_error_of_mean(pop.std, tau_c, duration)
        rows.append({
            "multiplier": float(mult),
            "duration_s": duration,
            "n_ensembles": n_ens,
            "n_eff": effective_sample_size(duration, tau_c),
            "std_of_means_n": std_obs,
            "std_predicted_n": std_pred,
            "relative_error": relative_error(std_obs, std_pred),
            "mean_of_means_n": float(np.mean(means)),
            "relative_sem": float(std_obs / abs(np.mean(means))),
            "monte_carlo_std_error": float(1.0 / np.sqrt(2.0 * (n_ens - 1))),
            "means_n": _round(means),
        })
    worst = max(row["relative_error"] for row in rows)
    trend = [row["std_of_means_n"] * np.sqrt(row["duration_s"]) for row in rows]
    return {
        "max_relative_error": float(worst),
        "eps": rule.eps,
        "tail_factor": rule.tail_factor,
        "t_req_s": t_req,
        "tau_c_s": float(tau_c),
        "cv_population": pop.cv,
        "sigma_population_n": pop.std,
        "mean_population_n": pop.mean,
        "reference_duration_s": float(cfg["reference_duration_s"]),
        "n_eff_required": rule.required_effective_samples(pop.cv),
        "sqrt_t_flatness": float(np.max(trend) / max(np.min(trend), 1e-30)),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# AC5 -- coverage of the stopping rule (calibrate, then evaluate)
# --------------------------------------------------------------------------- #
def _random_case(manifest: Manifest, rng: np.random.Generator, index: int) -> CaseSpec:
    ranges = manifest.ac5_ranges()
    base = manifest.base_case()

    def u(key):
        lo, hi = ranges[key]
        return float(rng.uniform(float(lo), float(hi)))

    overrides = {
        "name": f"ac5_{index:02d}",
        "beam_m": u("beam_m"),
        "speed_mps": u("speed_mps"),
        "number_density_per_m2": u("number_density_per_m2"),
        "floe_size": {"distribution": "loguniform",
                      "params": {"min": u("floe_min_m"), "max": u("floe_max_m")}},
        "thickness": {"distribution": "constant", "params": {"value": u("thickness_m")}},
        "force": {"coefficient": u("force_coefficient"),
                  "size_exponent": 1.0,
                  "thickness_exponent": 1.0,
                  "log_sigma": u("force_log_sigma")},
    }
    return base.with_overrides(overrides)


def _pilot(case: CaseSpec, rule: StoppingRule, pilot_duration_s: float, seed: int,
           spp: int, max_lag_s: float) -> tuple:
    pilot = simulate_record(case, pilot_duration_s, seed, spp)
    _, _, tau_c, _ = correlation_time(pilot.t, pilot.force, max_lag_s)
    stats = PilotStatistics.from_record(pilot, tau_c)
    return rule.assess(stats), stats, tau_c


def _window_mean(record: Record, duration_s: float) -> float:
    i1 = int(np.searchsorted(record.t, duration_s))
    i1 = max(2, min(i1, record.t.size))
    return float(np.mean(record.force[:i1]))


def ac5_stopping_coverage(manifest: Manifest) -> dict:
    cfg = _block(manifest, "ac5")
    spp = _spp(manifest)
    max_lag = float(cfg.get("max_lag_s", 5.0))
    n_marks = int(cfg.get("population_marks", 200_000))
    eps = float(cfg["eps"])
    max_duration = cfg.get("max_duration_s")

    # ---- stage 1: calibrate the tail factor at the one-sigma duration ------ #
    cal = cfg["calibration"]
    one_sigma = StoppingRule(eps=eps, tail_factor=1.0,
                             max_duration_s=cal.get("max_duration_s", max_duration))
    rng = np.random.default_rng(int(cal["case_seed"]))
    cal_rows, cal_errors = [], []
    for i in range(int(cal["n_cases"])):
        case = _random_case(manifest, rng, i)
        pop = _population(case, n_marks, int(cal["mark_seed"]) + i)
        decision, stats, tau_c = _pilot(case, one_sigma, float(cal["pilot_duration_s"]),
                                        int(cal["pilot_seed"]) + i, spp, max_lag)
        record = simulate_record(case, decision.required_time_s,
                                 int(cal["production_seed"]) + i, spp)
        error = relative_error(record.mean, pop.mean)
        cal_errors.append(error)
        cal_rows.append({"case": case.name, "pilot_cv": stats.cv,
                         "pilot_tau_c_s": tau_c, "t_required_s": decision.required_time_s,
                         "relative_error": error})
    zeta_cal = tail_factor_from_quantile(cal_errors, eps, float(cal["alpha"]))
    zeta_reg = float(cfg["tail_factor"])
    tolerance = float(cfg.get("tail_factor_tolerance", 0.15))
    zeta_rel_error = relative_error(zeta_cal, zeta_reg)
    cal_errors_arr = np.asarray(cal_errors)

    # ---- stage 2: evaluate coverage on fresh, independent cases ----------- #
    rule = StoppingRule(eps=eps, tail_factor=zeta_reg, max_duration_s=max_duration)
    rng2 = np.random.default_rng(int(cfg["case_seed"]))
    rows = []
    for i in range(int(cfg["n_cases"])):
        case = _random_case(manifest, rng2, i)
        pop = _population(case, n_marks, int(cfg.get("mark_seed", 77001)) + i)
        decision, stats, tau_c = _pilot(case, rule, float(cfg["pilot_duration_s"]),
                                        int(cfg["pilot_seed"]) + i, spp, max_lag)
        record = simulate_record(case, decision.required_time_s,
                                 int(cfg["production_seed"]) + i, spp)
        error = relative_error(record.mean, pop.mean)
        naive_duration = 2.0 * tau_c * (stats.cv / eps) ** 2
        naive_error = relative_error(_window_mean(record, naive_duration), pop.mean)
        rows.append({
            "case": case.name,
            "encounter_rate_per_s": float(case.encounter_rate),
            "pilot_cv": stats.cv,
            "pilot_tau_c_s": tau_c,
            "t_required_s": decision.required_time_s,
            "t_required_one_sigma_s": naive_duration,
            "n_eff_required": decision.required_effective_samples,
            "required_events": float(decision.required_time_s * case.encounter_rate),
            "mean_estimate_n": record.mean,
            "mean_true_n": pop.mean,
            "relative_error": error,
            "naive_relative_error": naive_error,
            "band": eps,
            "covered": bool(error <= eps),
            "naive_covered": bool(naive_error <= eps),
        })
    coverage = float(np.mean([row["covered"] for row in rows])) if rows else float("nan")
    naive_coverage = float(np.mean([row["naive_covered"] for row in rows])) if rows else float("nan")
    return {
        "coverage": coverage,
        "naive_coverage": naive_coverage,
        "eps": eps,
        "tail_factor": zeta_reg,
        "tail_factor_calibrated": zeta_cal,
        "calibration_relative_error": zeta_rel_error,
        "calibration_within_tolerance": bool(zeta_rel_error <= tolerance),
        "calibration_tolerance": tolerance,
        "calibration_alpha": float(cal["alpha"]),
        "calibration_cases": int(cal["n_cases"]),
        "calibration_covered_at_one_sigma": float(np.mean(cal_errors_arr <= eps)),
        "calibration_q50": float(np.quantile(cal_errors_arr, 0.5)),
        "calibration_q90": float(np.quantile(cal_errors_arr, 0.90)),
        "n_cases": len(rows),
        "n_covered": int(sum(row["covered"] for row in rows)),
        "mean_relative_error": float(np.mean([r["relative_error"] for r in rows])),
        "median_t_required_s": float(np.median([r["t_required_s"] for r in rows])),
        "median_t_required_one_sigma_s": float(np.median([r["t_required_one_sigma_s"] for r in rows])),
        "cases": rows,
        "calibration_rows": cal_rows,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass
class AcceptanceResult:
    results: dict
    verdict: dict
    outdir: pathlib.Path
    elapsed_s: float


def run_acceptance(manifest: Manifest, outdir: Optional[str] = None,
                   make_figures: bool = True, write: bool = True) -> AcceptanceResult:
    """Run AC1..AC6 and (optionally) persist artifacts + verdict."""
    started = time.time()
    out = pathlib.Path(outdir) if outdir is not None else pathlib.Path("artifacts") / "p233w2_n05"
    if write:
        out.mkdir(parents=True, exist_ok=True)

    base = manifest.base_case()
    clustered = manifest.case("clustered")
    results: dict = {
        "node": NODE_ID,
        "manifest_sha256": manifest_hash(manifest.payload),
        "manifest_source": manifest.source,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "case_base": _round(base.to_dict()),
        "criterion_ids": [entry["id"] for entry in manifest.criteria],
    }

    results["ac1"] = _round(ac1_poisson_uniform(manifest, base))
    results["ac2"] = _round(ac2_clustered_control(manifest, clustered))
    results["ac3"] = _round(ac3_campbell_variance(manifest))
    results["ac4"] = _round(ac4_treq_scaling(manifest, base))
    results["ac5"] = _round(ac5_stopping_coverage(manifest))

    from .prereg import validate
    from .report import build_report

    results["ac6"] = {"complete_score": 0.0, "sections": [], "figures": [],
                      "report_path": None}
    report = build_report(manifest, results, out, make_figures=make_figures, write=write)
    results["ac6"] = _round(report.documentation_check)
    results["report"] = {"markdown_path": str(report.markdown_path),
                         "figures": [str(p) for p in report.figure_paths]}

    verdict = validate(manifest, results)
    results["verdict"] = verdict["verdict"]
    elapsed = time.time() - started
    results["elapsed_s"] = round(elapsed, 3)

    if write:
        # re-render now that the verdict is known, so the report header is final
        from .report import render_markdown

        final_markdown = render_markdown(manifest, results)
        report.markdown_path.write_text(final_markdown, encoding="utf-8")
        (out / "acceptance.json").write_text(
            json.dumps(_round(results, 8), indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8")
        (out / "verdict.json").write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return AcceptanceResult(results=results, verdict=verdict, outdir=out, elapsed_s=elapsed)


__all__ = ["AcceptanceResult", "NODE_ID", "ac1_poisson_uniform",
           "ac2_clustered_control", "ac3_campbell_variance", "ac4_treq_scaling",
           "ac5_stopping_coverage", "run_acceptance"]