#!/usr/bin/env python
"""P226W3-N02 — ice-induced hull-girder whipping fatigue with measured-strain Bayesian updating.

Executed chain (all four stages are REAL numerics, no mock data):

1. continuous ice-load spectrum (quasi-periodic crushing train)
   -> Newmark modal superposition -> midship/quarter whipping moment history
2. synthetic-but-justified measured deck strain (known truth scale theta_true,
   Gaussian gauge noise) -> conjugate Gaussian Bayesian update of the response scale
3. rainflow (ASTM E1049-85) on the moment history -> stress ranges at the detail
   -> DNV-RP-C203 S-N curve -> Palmgren-Miner cumulative damage
4. damage per 120 s exposure block x annual ice-exposure blocks -> voyage life

Verification battery (each item is machine-checkable and reported with a number):

V1 hull geometry consistency (Cb from L/B/T/displacement)
V2 stage-1 driver vs analytic modal superposition (single half-sine impact)
V3 Bayesian posterior: closed-form conjugate vs brute-force numerical grid
V4 Bayesian recovery: known truth recovered + 95 % credible-interval coverage
   over 200 independent noise realisations (nominal 95 %, binomial band reported)
V5 rainflow + Miner against the closed-form single-range analytic damage
V6 rainflow scale invariance (used to reuse one count for all posterior draws)
V7 voyage life against the closed-form inversion

Outputs: ``runs/P226W3_N02/p226w3_n02_summary.json`` and four PNG figures.

Usage::

    python scripts/p226w3_n02_whipping_strain_fatigue_chain.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.fatigue.sn_curve import DNCurve  # noqa: E402
from tensorphysics.ice_fatigue_life.whipping_strain_fatigue import (  # noqa: E402
    bayesian_posterior_grid,
    bayesian_response_scale_update,
    damage_vs_scale_samples,
    half_sine_force_signal,
    ice_exposure_blocks_per_year,
    rainflow_miner_damage_from_moment,
    voyage_fatigue_life,
    whipping_moment_history,
)
from tensorphysics.ice_vibration import (  # noqa: E402
    HullParameters,
    free_free_beam_modal_data,
    quasi_periodic_pulse_train,
)
from tensorphysics.ice_vibration.whiplash import whiplash_response  # noqa: E402

OUT_DIR = ROOT / "runs" / "P226W3_N02"
FIG_DIR = OUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Case definition (single source of truth for the report numbers)
# ---------------------------------------------------------------------------
CASE = {
    "hull": {
        "name": "P226W3-N02 reference 250 m ice-class bulker",
        "length_m": 250.0,
        "breadth_m": 40.0,
        "draught_m": 9.0,
        "block_coefficient": 0.82,
        "displacement_kg": 0.82 * 250.0 * 40.0 * 9.0 * 1025.0,
        "bending_stiffness_nm2": 6.0e13,
        "deck_section_modulus_m3": 20.8,
        "youngs_modulus_pa": 2.06e11,
    },
    "modal": {"n_modes": 3, "damping_ratios": [0.02, 0.03, 0.04]},
    "ice": {
        "model": "quasi_periodic_pulse_train (intermittent crushing)",
        "pulse_frequency_hz": 0.35,
        "amplitude_n": 6.0e6,
        "pulse_duration_s": None,          # default 0.55/f_ice
        "arrival_jitter": 0.05,
        "amplitude_cv": 0.10,
        "block_duration_s": 120.0,
        "seed": 20260331,
    },
    "detail": {
        "sn_curve_class": "F",
        "sn_environment": "seawater_cp",
        "stress_concentration_factor": 1.10,
        "thickness_mm": 30.0,
        "thickness_ref_mm": 25.0,
        "thickness_exponent": 0.25,
    },
    "bayes": {
        "theta_true": 1.35,                # true model under-prediction of the response
        "noise_std_microstrain": 10.0,
        "n_observations": 120,
        "prior_mean": 1.0,
        "prior_std": 0.25,
        "credible_level": 0.95,
        "n_posterior_samples": 4000,
        "seed": 20260901,
    },
    "exposure": {
        "operating_days_per_year": 250.0,
        "transit_hours_per_day": 10.0,
        "ice_fraction_of_transit": 0.25,
        "design_fatigue_factor": 2.0,
    },
}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else float("nan")


def build_case() -> dict:
    """Build hull, modal set and continuous ice-load block."""
    hc = CASE["hull"]
    hull = HullParameters(
        length_m=hc["length_m"],
        breadth_m=hc["breadth_m"],
        draught_m=hc["draught_m"],
        block_coefficient=hc["block_coefficient"],
        displacement_kg=hc["displacement_kg"],
        name=hc["name"],
    )
    geometry = hull.verify()
    modal = free_free_beam_modal_data(
        hull,
        hc["bending_stiffness_nm2"],
        n_modes=CASE["modal"]["n_modes"],
        damping_ratios=CASE["modal"]["damping_ratios"],
    )
    ic = CASE["ice"]
    dt = 1.0 / (200.0 * modal.frequencies_hz[-1])
    series = quasi_periodic_pulse_train(
        pulse_frequency_hz=ic["pulse_frequency_hz"],
        amplitude_n=ic["amplitude_n"],
        duration_s=ic["block_duration_s"],
        dt_s=dt,
        pulse_duration_s=ic["pulse_duration_s"],
        arrival_jitter=ic["arrival_jitter"],
        amplitude_cv=ic["amplitude_cv"],
        seed=ic["seed"],
    )
    return {"hull": hull, "geometry": geometry, "modal": modal, "series": series, "dt_s": dt}


# ---------------------------------------------------------------------------
# Verification items
# ---------------------------------------------------------------------------


def verify_stage1_against_analytic(modal, ei: float) -> dict:
    """V2: chain driver vs the analytic modal-superposition half-sine response."""
    amp, tp, duration = 8.0e6, 0.6, 24.0
    dt = 1.0 / (200.0 * modal.frequencies_hz[-1])
    force = half_sine_force_signal(amp, tp, duration, dt)
    chain = whipping_moment_history(modal, ei, force, dt)
    ref = whiplash_response(modal, ei, amp, tp, duration_s=duration, dt_s=dt)
    analytic_peak = ref.peak_midship_moment_analytic_nm
    return {
        "chain_peak_midship_moment_nm": chain.peak_midship_moment_nm,
        "reference_path_peak_midship_moment_nm": ref.peak_midship_moment_nm,
        "analytic_half_sine_peak_midship_moment_nm": analytic_peak,
        "relative_error_chain_vs_reference_path": _rel(chain.peak_midship_moment_nm, ref.peak_midship_moment_nm),
        "relative_error_chain_vs_analytic": _rel(chain.peak_midship_moment_nm, analytic_peak),
        "whiplash_dynamic_amplification": chain.diagnostics["whiplash_dynamic_amplification"],
        "tolerance": 0.01,
    }


def verify_bayes_grid(post, model_moment, bcfg: dict, indices) -> dict:
    """V3: closed-form posterior vs brute-force numerical integration."""
    grid = bayesian_posterior_grid(
        model_moment,
        post.observed_strain_microstrain,
        section_modulus_m3=post.section_modulus_m3,
        youngs_modulus_pa=post.youngs_modulus_pa,
        noise_std_microstrain=post.noise_std_microstrain,
        sample_indices=indices,
        prior_mean=post.prior_mean,
        prior_std=post.prior_std,
        n_grid=400001,
    )
    return {
        "closed_form_mean": post.posterior_mean,
        "grid_mean": grid["posterior_mean"],
        "closed_form_std": post.posterior_std,
        "grid_std": grid["posterior_std"],
        "mean_relative_error": _rel(post.posterior_mean, grid["posterior_mean"]),
        "std_relative_error": _rel(post.posterior_std, grid["posterior_std"]),
        "tolerance": 1e-6,
    }


def verify_bayes_recovery(
    model_moment, section_modulus, youngs_modulus, noise_std, indices, bcfg: dict, n_trials: int = 200
) -> dict:
    """V4: truth recovery and credible-interval coverage over independent noise draws."""
    rng = np.random.default_rng(bcfg["seed"] + 7)
    eps_model = model_moment[indices] / (section_modulus * youngs_modulus) * 1.0e6
    theta_true = float(bcfg["theta_true"])
    hits = 0
    means = []
    stds = []
    for _ in range(n_trials):
        obs = theta_true * eps_model + rng.normal(0.0, noise_std, size=eps_model.size)
        post = bayesian_response_scale_update(
            model_moment,
            obs,
            section_modulus_m3=section_modulus,
            youngs_modulus_pa=youngs_modulus,
            noise_std_microstrain=noise_std,
            sample_indices=indices,
            prior_mean=bcfg["prior_mean"],
            prior_std=bcfg["prior_std"],
            n_samples=200,
            credible_level=bcfg["credible_level"],
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        lo, hi = post.credible_interval
        hits += int(lo <= theta_true <= hi)
        means.append(post.posterior_mean)
        stds.append(post.posterior_std)
    coverage = hits / n_trials
    nominal = bcfg["credible_level"]
    binomial_sigma = math.sqrt(nominal * (1.0 - nominal) / n_trials)
    return {
        "theta_true": theta_true,
        "n_trials": n_trials,
        "coverage": coverage,
        "nominal_coverage": nominal,
        "coverage_deviation_sigma": (coverage - nominal) / binomial_sigma,
        "posterior_mean_of_means": float(np.mean(means)),
        "posterior_mean_bias_relative": _rel(float(np.mean(means)), theta_true),
        "posterior_std_of_means": float(np.std(means, ddof=1)),
    }


def verify_miner_closed_form() -> dict:
    """V5: single-range sinusoid -> every full cycle has range 2*amplitude and
    ``D = n_full / N(2*amplitude/W)`` exactly (closed-form S-N identity).

    The trailing rainflow residue half-cycle is excluded from the analytic
    comparison because its range depends on where the record is truncated
    (ASTM E1049-85 boundary rule), not on the physics.
    """
    from tensorphysics.fatigue.rainflow import rainflow_counting

    n_cycles = 500.0
    t = np.arange(0.0, n_cycles, 1.0 / 400.0)  # 400 samples per 1 Hz period
    signal = 1.0e9 * np.sin(2.0 * math.pi * t)  # +/- 1 GN*m -> range 2 GN*m
    W = 20.8
    curve = DNCurve(curve_class="F", environment="seawater_cp")
    rf = rainflow_counting(signal)
    full = rf.counts == 1.0
    expected_range = 2.0e9
    range_dev = float(np.max(np.abs(rf.ranges[full] / expected_range - 1.0))) if full.any() else float("inf")

    res = rainflow_miner_damage_from_moment(signal, W, sn_curve=curve)  # no thickness / SCF
    stress_range = expected_range / W / 1.0e6  # MPa
    n_fail = curve.n_to_failure(stress_range)
    analytic_damage = float(np.sum(rf.counts[full])) / n_fail
    computed_full_damage = float(np.sum(res.damage_per_cycle[full]))
    return {
        "stress_range_mpa": stress_range,
        "n_full_cycles": int(rf.n_full),
        "n_half_cycle_residue": int(rf.n_half),
        "max_full_cycle_range_deviation": range_dev,
        "n_cycles_to_failure": n_fail,
        "analytic_full_cycle_damage": analytic_damage,
        "computed_full_cycle_damage": computed_full_damage,
        "total_damage_including_residue": res.total_cycles and float(res.damage),
        "relative_error": _rel(computed_full_damage, analytic_damage),
        "tolerance": 1e-12,
    }


def verify_rainflow_scale_invariance(moment_history) -> dict:
    """V6: rainflow(a*x) ranges == a * rainflow(x) ranges (exact reuse property)."""
    from tensorphysics.fatigue.rainflow import rainflow_counting

    a = 1.37
    from tensorphysics.fatigue.rainflow import RainflowResult  # noqa: F401

    base = rainflow_counting(moment_history)
    scaled = rainflow_counting(a * moment_history)
    if base.ranges.size != scaled.ranges.size:
        return {"ok": False, "reason": "different bin counts", "n_base": int(base.ranges.size)}
    mask = base.ranges > 0
    rel = float(np.max(np.abs(scaled.ranges[mask] / (a * base.ranges[mask]) - 1.0))) if mask.any() else 0.0
    counts_equal = bool(np.allclose(base.counts, scaled.counts))
    return {
        "ok": rel < 1e-12 and counts_equal,
        "max_relative_range_deviation": rel,
        "counts_identical": counts_equal,
        "n_bins": int(base.ranges.size),
    }


def verify_voyage_life(damage_per_block: float, blocks_per_year: float, dff: float) -> dict:
    """V7: years = (1/DFF) / (D_block * blocks_per_year)."""
    life = voyage_fatigue_life(damage_per_block, blocks_per_year, design_fatigue_factor=dff)
    analytic = (1.0 / dff) / (damage_per_block * blocks_per_year)
    return {
        "years_computed": life.years_to_failure,
        "years_closed_form": analytic,
        "relative_error": _rel(life.years_to_failure, analytic),
        "tolerance": 1e-12,
    }


# ---------------------------------------------------------------------------
# Main chain
# ---------------------------------------------------------------------------


def main() -> dict:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    built = build_case()
    hull, modal, series, dt = built["hull"], built["modal"], built["series"], built["dt_s"]
    ei = CASE["hull"]["bending_stiffness_nm2"]
    W_deck = CASE["hull"]["deck_section_modulus_m3"]
    E_steel = CASE["hull"]["youngs_modulus_pa"]
    det = CASE["detail"]
    bcfg = CASE["bayes"]
    expo = CASE["exposure"]

    # ---- stage 1: whipping moment history from the continuous ice spectrum ----
    hist = whipping_moment_history(modal, ei, series.force_n, dt)
    modal_rms_share = (
        np.sqrt(np.mean(hist.modal_coordinates**2, axis=0))
        / float(np.sqrt(np.sum(np.mean(hist.modal_coordinates**2, axis=0))))
    ).tolist()

    # ---- stage 2: synthetic measured strain (known truth) + Bayesian update ----
    theta_true = float(bcfg["theta_true"])
    noise = float(bcfg["noise_std_microstrain"])
    n_obs = int(bcfg["n_observations"])
    indices = np.linspace(0, hist.midship_moment_nm.size - 1, n_obs).round().astype(int)
    rng = np.random.default_rng(bcfg["seed"])
    eps_model = hist.midship_moment_nm[indices] / (W_deck * E_steel) * 1.0e6
    eps_obs = theta_true * eps_model + rng.normal(0.0, noise, size=n_obs)
    post = bayesian_response_scale_update(
        hist.midship_moment_nm,
        eps_obs,
        section_modulus_m3=W_deck,
        youngs_modulus_pa=E_steel,
        noise_std_microstrain=noise,
        sample_indices=indices,
        prior_mean=bcfg["prior_mean"],
        prior_std=bcfg["prior_std"],
        n_samples=bcfg["n_posterior_samples"],
        credible_level=bcfg["credible_level"],
        seed=bcfg["seed"],
    )
    post_lo, post_hi = post.credible_interval

    # posterior-predictive standardised residual (sanity check on the synthetic
    # record: it must be ~1 by construction, it is NOT a model-validation test)
    resid = (post.posterior_mean * eps_model - eps_obs) / noise
    predictive_rms_residual = float(np.sqrt(np.mean(resid**2)))

    # ---- stage 3: rainflow + Miner (prior scale = 1.0, posterior ensemble) ----
    base = rainflow_miner_damage_from_moment(
        hist.midship_moment_nm,
        W_deck,
        sn_curve=DNCurve(curve_class=det["sn_curve_class"], environment=det["sn_environment"]),
        stress_concentration_factor=det["stress_concentration_factor"],
        thickness_mm=det["thickness_mm"],
        thickness_ref_mm=det["thickness_ref_mm"],
        thickness_exponent=det["thickness_exponent"],
    )
    curve = DNCurve(curve_class=det["sn_curve_class"], environment=det["sn_environment"])
    damage_samples = damage_vs_scale_samples(base, post.scale_samples, sn_curve=curve)
    damage_prior = float(base.damage)  # theta = 1.0 (prior mean, no update)
    damage_post_mean = float(np.mean(damage_samples))
    damage_p05, damage_p95 = np.percentile(damage_samples, [5.0, 95.0])

    # ---- stage 4: voyage life ----
    blocks_per_year = ice_exposure_blocks_per_year(
        operating_days_per_year=expo["operating_days_per_year"],
        transit_hours_per_day=expo["transit_hours_per_day"],
        ice_fraction_of_transit=expo["ice_fraction_of_transit"],
        block_duration_s=CASE["ice"]["block_duration_s"],
    )
    life_prior = voyage_fatigue_life(damage_prior, blocks_per_year, expo["design_fatigue_factor"])
    life_post = voyage_fatigue_life(damage_post_mean, blocks_per_year, expo["design_fatigue_factor"])
    years_samples = (1.0 / expo["design_fatigue_factor"]) / (
        damage_samples * blocks_per_year
    )
    life_p05, life_p50, life_p95 = np.percentile(years_samples, [5.0, 50.0, 95.0])

    # ---- sensitivity: same pulse amplitude, ramming rate swept towards f1 ----
    sensitivity = []
    for f_ice in (0.20, 0.35, 0.50, 0.60, 0.80):
        if abs(f_ice - CASE["ice"]["pulse_frequency_hz"]) < 1e-12:
            sens_hist = hist
        else:
            sens_series = quasi_periodic_pulse_train(
                pulse_frequency_hz=f_ice,
                amplitude_n=CASE["ice"]["amplitude_n"],
                duration_s=CASE["ice"]["block_duration_s"],
                dt_s=dt,
                pulse_duration_s=CASE["ice"]["pulse_duration_s"],
                arrival_jitter=CASE["ice"]["arrival_jitter"],
                amplitude_cv=CASE["ice"]["amplitude_cv"],
                seed=CASE["ice"]["seed"],
            )
            sens_hist = whipping_moment_history(modal, ei, sens_series.force_n, dt)
        sens_detail = rainflow_miner_damage_from_moment(
            sens_hist.midship_moment_nm,
            W_deck,
            sn_curve=curve,
            stress_concentration_factor=det["stress_concentration_factor"],
            thickness_mm=det["thickness_mm"],
            thickness_ref_mm=det["thickness_ref_mm"],
            thickness_exponent=det["thickness_exponent"],
        )
        sensitivity.append(
            {
                "pulse_frequency_hz": f_ice,
                "pulse_frequency_over_f1": f_ice / modal.frequencies_hz[0],
                "peak_midship_moment_nm": sens_hist.peak_midship_moment_nm,
                "dynamic_amplification": sens_hist.diagnostics["whiplash_dynamic_amplification"],
                "max_stress_range_mpa": float(np.max(sens_detail.stress_ranges_mpa)),
                "damage_per_block": sens_detail.damage,
                "years_to_failure_theta_1": (
                    voyage_fatigue_life(
                        sens_detail.damage, blocks_per_year, expo["design_fatigue_factor"]
                    ).years_to_failure
                ),
            }
        )

    # ---- verification battery ----
    verification = {
        "V1_geometry_consistency": {
            "consistent": bool(built["geometry"]["consistent"]),
            "cb_input": hull.block_coefficient,
            "cb_from_displacement": hull.cb_from_displacement,
            "relative_deviation": _rel(hull.cb_from_displacement, hull.block_coefficient),
            "displacement_kg": hull.displacement_kg,
            "marginal": float(hull.cb_from_displacement * hull.length_m * hull.breadth_m * hull.draught_m * 1025.0),
        },
        "V2_stage1_vs_analytic": verify_stage1_against_analytic(modal, ei),
        "V3_bayes_closed_form_vs_grid": verify_bayes_grid(post, hist.midship_moment_nm, bcfg, indices),
        "V4_bayes_recovery_and_coverage": verify_bayes_recovery(
            hist.midship_moment_nm, W_deck, E_steel, noise, indices, bcfg
        ),
        "V5_miner_closed_form": verify_miner_closed_form(),
        "V6_rainflow_scale_invariance": verify_rainflow_scale_invariance(hist.midship_moment_nm),
        "V7_voyage_life_closed_form": verify_voyage_life(
            damage_post_mean, blocks_per_year, expo["design_fatigue_factor"]
        ),
    }

    criteria = {
        "stage1_whipping_moment_vs_analytic_lt_1pct": verification["V2_stage1_vs_analytic"][
            "relative_error_chain_vs_analytic"
        ] < 0.01,
        "bayes_analytic_vs_grid_lt_1e-6": (
            verification["V3_bayes_closed_form_vs_grid"]["mean_relative_error"] < 1e-6
            and verification["V3_bayes_closed_form_vs_grid"]["std_relative_error"] < 1e-6
        ),
        "bayes_truth_within_2sigma": abs(post.posterior_mean - theta_true) <= 2.0 * post.posterior_std,
        "bayes_coverage_within_3binomial_sigma": abs(
            verification["V4_bayes_recovery_and_coverage"]["coverage_deviation_sigma"]
        ) <= 3.0,
        "miner_closed_form_lt_1e-12": verification["V5_miner_closed_form"]["relative_error"] < 1e-12
        and verification["V5_miner_closed_form"]["max_full_cycle_range_deviation"] < 1e-12,
        "rainflow_scale_invariance_exact": bool(verification["V6_rainflow_scale_invariance"]["ok"]),
        "voyage_life_closed_form_lt_1e-12": verification["V7_voyage_life_closed_form"]["relative_error"] < 1e-12,
        "posterior_predictive_residual_near_unity": abs(predictive_rms_residual - 1.0) < 0.25,
        "geometry_consistent": bool(built["geometry"]["consistent"]),
    }
    criteria = {key: bool(value) for key, value in criteria.items()}

    summary = {
        "node": "P226W3-N02",
        "title": "Ice-induced whipping fatigue with measured-strain Bayesian updating",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": CASE,
        "computed": {
            "modal_frequencies_hz": list(modal.frequencies_hz),
            "modal_bow_ordinates": list(modal.mode_shape_bow),
            "modal_masses_kg": list(modal.modal_masses_kg),
            "modal_rms_share": modal_rms_share,
            "ice_block": {
                "n_samples": int(series.n_samples),
                "dt_s": dt,
                "n_pulses": series.metadata.get("n_pulses"),
                "peak_force_n": series.peak_force_n,
                "rms_force_n": series.rms_force_n,
            },
            "whipping": hist.to_json_dict(),
            "strain_microstrain": {
                "model_peak": float(np.max(np.abs(eps_model))),
                "observed_peak": float(np.max(np.abs(eps_obs))),
                "noise_std": noise,
                "n_observations": n_obs,
            },
            "bayes": post.to_json_dict(),
            "bayes_interval": [float(post_lo), float(post_hi)],
            "posterior_predictive_standardised_rms_residual": predictive_rms_residual,
            "prior_information_share": post.diagnostics["prior_information"]
            / (post.diagnostics["prior_information"] + post.diagnostics["data_information"]),
            "detail_fatigue": base.to_json_dict(),
            "damage_per_block": {
                "prior_theta_1p0": damage_prior,
                "posterior_mean": damage_post_mean,
                "posterior_p05": float(damage_p05),
                "posterior_p95": float(damage_p95),
                "ratio_posterior_over_prior": damage_post_mean / damage_prior,
            },
            "voyage_life": {
                "blocks_per_year": blocks_per_year,
                "prior": life_prior.to_json_dict(),
                "posterior_mean": life_post.to_json_dict(),
                "years_p05": float(life_p05),
                "years_p50": float(life_p50),
                "years_p95": float(life_p95),
                "design_target_years": 20.0,
                "below_design_target": bool(float(life_p50) < 20.0),
            },
        },
        "sensitivity_pulse_frequency": sensitivity,
        "verification": verification,
        "criteria": criteria,
        "criteria_all_passed": bool(all(criteria.values())),
        "runtime_s": time.time() - t_start,
    }

    # ---- figures ----
    figures = make_figures(hist, post, eps_obs, eps_model, indices, base, curve, damage_samples, years_samples, life_prior)
    summary["figures"] = figures

    with open(OUT_DIR / "p226w3_n02_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            summary,
            fh,
            indent=2,
            ensure_ascii=False,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )

    print(json.dumps(
        {
            "criteria": criteria,
            "criteria_all_passed": summary["criteria_all_passed"],
            "posterior_mean": post.posterior_mean,
            "posterior_std": post.posterior_std,
            "damage_prior": damage_prior,
            "damage_posterior_mean": damage_post_mean,
            "years_p50": float(life_p50),
            "runtime_s": summary["runtime_s"],
        },
        indent=2,
        default=lambda o: o.item() if hasattr(o, "item") else str(o),
    ))
    return summary


def make_figures(hist, post, eps_obs, eps_model, indices, base, curve, damage_samples, years_samples, life_prior):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    t = hist.time_s
    m = hist.midship_moment_nm

    # -- fig 1: whipping moment history + posterior envelope -------------------
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=False)
    ax = axes[0]
    ax.plot(t, hist.force_n / 1e6, lw=0.6, color="#555555")
    ax.set_ylabel("bow ice force [MN]")
    ax.set_title(
        "P226W3-N02 continuous ice crushing train "
        f"(f_ice = {CASE['ice']['pulse_frequency_hz']:g} Hz, {CASE['ice']['amplitude_n'] / 1e6:g} MN pulses)"
    )
    ax.grid(alpha=0.3)

    ax = axes[1]
    ens = post.posterior_moment_ensemble(m) / 1e6
    lo, hi = np.percentile(ens, [5.0, 95.0], axis=0)
    ax.fill_between(t, lo, hi, color="#1f77b4", alpha=0.25, label="posterior 5-95 % envelope")
    ax.plot(t, m / 1e6, color="#1f77b4", lw=0.8, label="model (theta = 1)")
    ax.plot(t, post.posterior_mean * m / 1e6, color="#d62728", lw=0.9, ls="--",
            label=f"posterior mean (theta = {post.posterior_mean:.3f})")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("midship moment [MN m]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, hist.midship_moment_nm / 1e6, lw=0.7, color="#1f77b4", label="midship")
    ax.plot(t, hist.quarter_moment_nm / 1e6, lw=0.7, color="#2ca02c", alpha=0.8, label="quarter (L/4)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("bending moment [MN m]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "p226w3_n02_fig1_whipping_moment_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["fig1"] = str(p.relative_to(ROOT))

    # -- fig 2: Bayesian update: prior/posterior pdf + strain fit --------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    from statistics import NormalDist

    nd = NormalDist()
    grid = np.linspace(post.prior_mean - 4 * post.prior_std, post.posterior_mean + 4 * post.posterior_std, 600)
    prior_pdf = [nd.pdf((g - post.prior_mean) / post.prior_std) / post.prior_std for g in grid]
    post_pdf = [nd.pdf((g - post.posterior_mean) / post.posterior_std) / post.posterior_std for g in grid]
    ax = axes[0]
    ax.plot(grid, prior_pdf, lw=1.6, label=f"prior N({post.prior_mean}, {post.prior_std}$^2$)")
    ax.plot(grid, post_pdf, lw=1.8, color="#d62728", label=f"posterior N({post.posterior_mean:.3f}, {post.posterior_std:.3f}$^2$)")
    ax.axvline(CASE["bayes"]["theta_true"], color="k", ls=":", lw=1.2, label="truth")
    ax.set_xlabel(r"response scale $\theta$")
    ax.set_ylabel("density")
    ax.set_title("Bayesian update of the whipping response scale from deck strain")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(eps_model, eps_obs, s=12, alpha=0.75, label="measured vs model strain")
    lim = [min(eps_model.min(), eps_obs.min()) - 5, max(eps_model.max(), eps_obs.max()) + 5]
    ax.plot(lim, lim, "k--", lw=1.0, label=r"$\theta=1$")
    ax.plot(lim, [post.posterior_mean * v for v in lim], color="#d62728", lw=1.4,
            label=rf"posterior mean $\theta$={post.posterior_mean:.3f}")
    ax.set_xlabel(r"model strain $M_{model}/W$ [$\mu\varepsilon$]")
    ax.set_ylabel(r"measured strain [$\mu\varepsilon$]")
    ax.set_title(f"strain record ({eps_obs.size} samples, noise {CASE['bayes']['noise_std_microstrain']:g} $\\mu\\varepsilon$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "p226w3_n02_fig2_bayesian_update.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["fig2"] = str(p.relative_to(ROOT))

    # -- fig 3: rainflow damage spectrum vs S-N curve -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    order = np.argsort(-base.stress_ranges_mpa)
    ax.bar(np.arange(min(25, order.size)), base.damage_per_cycle[order][:25], color="#1f77b4")
    ax.set_xlabel("rainflow bin (sorted by descending stress range)")
    ax.set_ylabel("Miner damage contribution D$_i$")
    ax.set_title(f"damage concentration, D$_{{block}}$ = {base.damage:.3e} (theta = 1)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    s_plot = np.logspace(math.log10(5.0), math.log10(200.0), 200)
    n_plot = np.array([curve.n_to_failure(s) for s in s_plot])
    ax.loglog(s_plot, n_plot, "k-", lw=1.6, label=f"DNV {curve.curve_class} seawater+CP")
    ax.loglog(base.stress_ranges_mpa, np.maximum(base.counts, 1e-1), "o", ms=4, color="#d62728",
              label="rainflow bins (counts)")
    ax.set_xlabel("stress range [MPa]")
    ax.set_ylabel("N cycles to failure  /  counted cycles")
    ax.set_title("rainflow spectrum vs S-N curve at the detail")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG_DIR / "p226w3_n02_fig3_rainflow_sn.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["fig3"] = str(p.relative_to(ROOT))

    # -- fig 4: life distribution + sensitivity ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    ax.hist(years_samples, bins=60, color="#1f77b4", alpha=0.8)
    ax.axvline(life_prior.years_to_failure, color="k", ls="--", lw=1.5,
               label=f"prior $\\theta$=1: {life_prior.years_to_failure:.1f} yr")
    ax.axvline(np.median(years_samples), color="#d62728", lw=1.6,
               label=f"posterior median: {np.median(years_samples):.1f} yr")
    ax.set_xlabel("ice-zone fatigue life to D$_{crit}$=1/DFF [years]")
    ax.set_ylabel("posterior draws")
    ax.set_title("posterior-predictive voyage life (DFF = 2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    thetas = np.linspace(0.6, 2.0, 140)
    d_curve = damage_vs_scale_samples(base, thetas, sn_curve=curve)
    years_curve = (1.0 / CASE["exposure"]["design_fatigue_factor"]) / (d_curve * life_prior.blocks_per_year)
    ax.plot(thetas, years_curve, lw=1.8, color="#1f77b4")
    lo, hi = post.credible_interval
    ax.axvspan(lo, hi, color="#d62728", alpha=0.18, label="posterior 95 % CI of $\\theta$")
    ax.axvline(CASE["bayes"]["theta_true"], color="k", ls=":", lw=1.2, label="truth")
    ax.set_yscale("log")
    ax.set_xlabel(r"response scale $\theta$")
    ax.set_ylabel("life [years]")
    ax.set_title("life sensitivity: damage scales ~ $\\theta^{m}$ (S-N slope m = 3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG_DIR / "p226w3_n02_fig4_voyage_life.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["fig4"] = str(p.relative_to(ROOT))

    return paths


if __name__ == "__main__":
    main()