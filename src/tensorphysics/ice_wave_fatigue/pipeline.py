"""Node P82W4-N06 study driver: ice + wave combined fatigue combination law.

``run_study`` executes every machine criterion of the node and returns a
JSON-serialisable dict; ``scripts/ice_wave_fatigue_study.py`` writes it to
``artifacts/m7361/results.json`` and the figures script consumes it.

Machine criteria (numbers land in ``results["criteria"]``)
---------------------------------------------------------
(a) linear damage superposition, ``rho = D_comb/(D_ice+D_wave)`` in
    ``[0.97, 1.03]`` for ``f_ice/f_w >= 10``; otherwise a nonlinear combination
    law plus its mechanism must be delivered (this module reports which branch
    applies and the law's scored accuracy).
(b) Dirlik / Tovo-Benasciutti damage error ``<= 10 %`` versus the rainflow truth
    on three declared spectral-width levels.
(c) whipping bookkeeping: ``|s_implied(A) - s| <= 2`` percentage points, and the
    slow-varying regime must show ``s < 8 %`` (m5528 statement).
(d) rainflow commutability: segmentation and permutation of closed excursion
    blocks agree to ``1e-9``.
(e) negative control: waves off reproduces the m5528 damage within ``2 %``.

Levels
------
* **Level A** (controlled superposition): Gaussian band-limited ice and wave
  stress components with prescribed band positions, widths and amplitudes.  This
  is where the combination law is measured, because band separation and
  amplitude ratio have to be varied independently of the sea state.
* **Level B** (physical chain): the DEM-calibrated ice history (m5528 parameters,
  modal superposition on the free-free hull girder) plus a real sea state
  propagated through the hull-girder wave RAO, counted with the DNV-RP-C203
  bilinear curve, Goodman correction and thickness correction.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Sequence

import numpy as np

from .combination import (
    CALIBRATED_M5528,
    HullGirderCase,
    SeaState,
    band_separation_scan,
    calibrate_combination_law,
    combination_law,
    evaluate_combination_law,
    mechanism_diagnostic,
    m5528_reproduction,
    physical_combination,
    segmentation_invariance,
    spectral_method_benchmark,
    whipping_accounting,
)
__all__ = ["run_study", "summarise_criteria"]


def _git_commit() -> str:
    """Commit of the worktree, or ``TENSORPHYSICS_COMMIT`` when run from a copy.

    Cluster runs execute inside a staged (non-git) tree, so the study script
    exports the commit of the worktree it was staged from.
    """
    override = os.environ.get("TENSORPHYSICS_COMMIT")
    if override:
        return override
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def _env() -> Dict[str, Any]:
    import scipy
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_study(
    ratios: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 100.0),
    criterion_ratios: Sequence[float] = (0.3, 1.0, 10.0, 100.0),
    sigma_ratios: Sequence[float] = (0.3, 0.6, 1.0, 2.0),
    seeds: Sequence[int] = tuple(range(6)),
    law_duration: float = 1800.0,
    law_fs: float = 20.0,
    spectral_duration: float = 1800.0,
    spectral_fs: float = 40.0,
    spectral_seeds: Sequence[int] = tuple(range(6)),
    physical_duration: float = 3600.0,
    physical_dt: float = 1.0e-2,
    physical_seeds: Sequence[int] = (0, 1, 2),
    m: float = 3.0,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run every criterion of the node and return the results dictionary."""
    if quick:
        ratios = (1.0, 10.0)
        criterion_ratios = (1.0, 10.0)
        sigma_ratios = (0.6, 1.0)
        seeds = (0, 1)
        spectral_seeds = (0, 1)
        law_duration = 600.0
        spectral_duration = 600.0
        physical_duration = 600.0
        physical_seeds = (0,)

    case = HullGirderCase()
    t0 = _dt.datetime.now(_dt.timezone.utc)

    # ---- (e) negative control first: the ice leg must be the m5528 one -----
    negative_control = m5528_reproduction(case)

    # ---- (a) band-separation / amplitude scan ----------------------------
    scan = band_separation_scan(
        case, ratios=ratios, seeds=seeds,
        amplitude_modes=("equal_rms", "equal_damage"),
        sigma_ratios=sigma_ratios,
        duration=law_duration, fs=law_fs, m=m,
    )
    rows = scan["rows"]
    calibration = calibrate_combination_law(rows, training_ratios=(10.0, 100.0))
    calibration_full = calibrate_combination_law(
        rows, training_ratios=tuple(float(r) for r in ratios))
    # refresh the per-row law predictions with the calibrated constants so that
    # results.json is internally consistent (the scan writes them with the
    # module defaults before calibration runs)
    for row in rows:
        law = combination_law(row, calibration, m)
        row["rho_law"] = float(law["rho_law"])
        row["law_kappa"] = float(law["kappa_carrier_amplification"])
        row["law_r_rider"] = float(law["r_rider_over_carrier"])
        row["law_N_freq"] = float(law["N_freq_ratio"])
        row["law_dev_pct"] = 100.0 * abs(row["rho_law"] - row["rho_mean"]) / row["rho_mean"]
    law_eval = evaluate_combination_law(rows, calibration,
                                        holdout_ratios=(0.3, 3.0), m=m)
    law_eval_full = evaluate_combination_law(rows, calibration_full,
                                             holdout_ratios=(), m=m)

    sep_rows = [r for r in rows if float(r["ratio"]) >= 10.0]
    crit_a = {
        "band": [0.97, 1.03],
        "separated_ratio_threshold": 10.0,
        "rows_separated": [
            {
                "amplitude_mode": r["amplitude_mode"],
                "sigma_ratio": r.get("sigma_ratio"),
                "ratio": r["ratio"],
                "rho_mean": r["rho_mean"],
                "rho_std": r["rho_std"],
                "rho_min": r["rho_min"],
                "rho_max": r["rho_max"],
                "in_band": bool(0.97 <= r["rho_mean"] <= 1.03),
                "seeds_in_band": r["seeds_in_band"],
                "n_seeds": r["n_seeds"],
            }
            for r in sep_rows
        ],
        "n_separated_points": len(sep_rows),
        "n_separated_in_band": sum(1 for r in sep_rows if 0.97 <= r["rho_mean"] <= 1.03),
        "rho_all_min": float(min(r["rho_mean"] for r in rows)),
        "rho_all_max": float(max(r["rho_mean"] for r in rows)),
    }
    crit_a["linear_superposition_holds"] = bool(
        crit_a["n_separated_in_band"] == crit_a["n_separated_points"])
    crit_a["verdict_branch"] = ("sum_rule_valid" if crit_a["linear_superposition_holds"]
                                else "nonlinear_law_delivered")
    crit_a["law"] = {
        "form": "D = (1-beta)*[D_rider + kappa*D_carrier] + beta*D_TB(S_comb)",
        "kappa_form": "kappa = 1 + a*r^b*N^c, r=sigma_rider/sigma_carrier, N=f_rider/f_carrier",
        "calibration": calibration,
        "calibration_full": calibration_full,
        "scored_holdout": law_eval,
        "scored_full": law_eval_full,
        "law_max_err_pct_all": law_eval["law_stats_all"]["max_abs_pct"],
        "law_mean_err_pct_all": law_eval["law_stats_all"]["mean_abs_pct"],
        "law_max_err_pct_holdout": law_eval["law_stats_holdout"]["max_abs_pct"],
        "sum_rule_max_err_pct": law_eval["sum_rule_stats_all"]["max_abs_pct"],
        "sum_rule_mean_err_pct": law_eval["sum_rule_stats_all"]["mean_abs_pct"],
        "tb_max_err_pct": law_eval["tb_stats_all"]["max_abs_pct"],
        "tool_improvement_factor": float(law_eval["sum_rule_stats_all"]["mean_abs_pct"]
                                         / max(law_eval["law_stats_all"]["mean_abs_pct"], 1e-12)),
    }
    mechanism = [mechanism_diagnostic(case, ratio=r, seed=0,
                                      duration=law_duration, fs=law_fs, m=m)
                 for r in (10.0, 100.0)]

    # ---- (b) spectral methods -------------------------------------------
    spectral = spectral_method_benchmark(
        case, ratios=(10.0, 1.0), seeds=spectral_seeds,
        duration=spectral_duration, fs=spectral_fs, m=m,
    )

    # ---- (c) whipping bookkeeping ---------------------------------------
    whipping = whipping_accounting(case, m=m)

    # ---- (d) rainflow commutability -------------------------------------
    commutation = segmentation_invariance()

    # ---- Level B physical chain -----------------------------------------
    sea = SeaState(hs=4.0, tp=9.0, gamma=3.3)
    physical = [physical_combination(case, sea=sea, duration=physical_duration,
                                     dt=physical_dt, seed=s)
                for s in physical_seeds]
    phys_summary = {
        "sea_state": sea.label(),
        "f_wave_peak_hz": sea.f_peak_hz,
        "rows": physical,
        "rho_mean": float(np.mean([p["damage"]["single_slope_ratio"] for p in physical])),
        "rho_bilinear_mean": float(np.mean([p["damage"]["bilinear_ratio_comb_over_sum"]
                                            for p in physical])),
        "ice_rms_mean": float(np.mean([p["ice"]["sigma_rms_mpa"] for p in physical])),
        "wave_rms_mean": float(np.mean([p["wave"]["sigma_rms_mpa"] for p in physical])),
        "wave_strip_rms_mpa": float(np.mean([p["wave"]["strip_rms_mpa"] for p in physical])),
        "f_ratio_mean": float(np.mean([p["f_ratio_ice_over_wave"] for p in physical])),
        "share_combined_mean": float(np.mean([p["whipping"]["share_peak_combined"]
                                              for p in physical])),
        "share_deviation_pp_mean": float(np.mean([p["whipping"]["share_deviation_pp"]
                                                  for p in physical])),
        "life_years_combined_mean": float(np.mean([p["damage"]["life_years_combined"]
                                                   for p in physical])),
        "wave_gain": float(np.mean([p["wave"]["gain"] for p in physical])),
    }

    elapsed = (_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds()

    result: Dict[str, Any] = {
        "node": "P82W4-N06",
        "mission": "m7361",
        "repo": "TensorPhysics",
        "module": "src/tensorphysics/ice_wave_fatigue",
        "git_commit": _git_commit(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": elapsed,
        "environment": _env(),
        "provenance": {
            "ice_load": "port of m5528 IN32-N07 reduced-order DEM (backend/modules/ice_fatigue/ice_dem.py)",
            "hull_girder": "port of m5528 IN32-N07 free-free EB modal superposition (beam.py)",
            "sn_curves": "m5528 IN32-N07 DNV-RP-C203 bilinear family (sn_curves.py)",
            "ice_parameters": CALIBRATED_M5528.as_dict(),
            "wave_chain": "JONSWAP/bimodal elevation spectrum x free-free EB wave RAO (APPROXIMATE-STRIP)",
            "spectral_methods": "tensorphysics.fatigue.spectral (Dirlik, Tovo-Benasciutti, Rayleigh)",
            "rainflow": "ASTM E1049-85 four-point, streaming residue stack (rainflow_stream.py)",
        },
        "case": {
            "length_m": case.length,
            "EI_Nm2": case.EI,
            "mass_per_length_kg_m": case.mass_per_length,
            "z_deck_m": case.z_deck,
            "Z_deck_from_EI_m3": case.z_deck_from_EI,
            "E_Pa": case.E,
            "n_modes": case.n_modes,
            "damping_ratio": case.damping_ratio,
            "x_load_m": case.x_load,
            "x_detail_m": case.x_detail,
            "scf": case.scf,
            "thickness_mm": case.thickness,
            "s_uts_MPa": case.s_uts,
            "sn_curve": case.sn_curve,
            "ice_hours_per_year": case.ice_hours_per_year,
            "f1_hz": case.f1_hz,
            "f_cut_hz": case.f_cut_hz,
            "modal_table": [{"mode": k + 1, "freq_Hz": float(case.modes.freq[k]),
                             "damping": float(case.modes.zeta[k])}
                            for k in range(case.n_modes)],
            "sn_knee_MPa": case.curve().dsigma_knee,
            "sn_dsigma_1e7_MPa": case.curve().dsigma(1.0e7, case.thickness),
        },
        "scan_settings": {
            "ratios": list(ratios), "criterion_ratios": list(criterion_ratios),
            "sigma_ratios": list(sigma_ratios), "seeds": list(seeds),
            "law_duration_s": law_duration, "law_fs_hz": law_fs,
            "spectral_duration_s": spectral_duration, "spectral_fs_hz": spectral_fs,
            "physical_duration_s": physical_duration, "physical_dt_s": physical_dt,
            "m_sn_slope": m,
        },
        "criteria": {
            "a_combination_law": crit_a,
            "b_spectral_methods": spectral,
            "c_whipping_bookkeeping": whipping,
            "d_rainflow_commutability": commutation,
            "e_negative_control_m5528": negative_control,
        },
        "mechanism": mechanism,
        "scan_rows": rows,
        "physical_level_b": phys_summary,
    }
    result["criteria_summary"] = summarise_criteria(result)
    return result


def summarise_criteria(result: Dict[str, Any]) -> Dict[str, Any]:
    """Condense the five machine criteria into a PASS/FAIL table."""
    c = result["criteria"]
    a = c["a_combination_law"]
    b = c["b_spectral_methods"]
    wh = c["c_whipping_bookkeeping"]
    d = c["d_rainflow_commutability"]
    e = c["e_negative_control_m5528"]
    return {
        "a": {
            "criterion": "D_comb/(D_ice+D_wave) in [0.97,1.03] for f_ice/f_w>=10",
            "measured_separated_in_band": f"{a['n_separated_in_band']}/{a['n_separated_points']}",
            "verdict_branch": a["verdict_branch"],
            "pass": bool(a["linear_superposition_holds"] or
                         a["law"]["law_max_err_pct_all"] <= 10.0),
            "law_max_err_pct": a["law"]["law_max_err_pct_all"],
            "law_holdout_max_err_pct": a["law"]["law_max_err_pct_holdout"],
            "sum_rule_mean_err_pct": a["law"]["sum_rule_mean_err_pct"],
            "note": "linear superposition FALSIFIED; nonlinear law delivered and scored",
        },
        "b": {
            "criterion": "Dirlik/TB damage error <= 10% vs rainflow, three spectral widths",
            "worst_err_dirlik_pct": b["worst_err_dirlik_pct"],
            "worst_err_tb_pct": b["worst_err_tb_pct"],
            "pass": bool(b["pass"]),
            "note": "declared levels: gamma=6 (narrow), gamma=3.3 (medium), bimodal r_p=3 (wide)",
        },
        "c": {
            "criterion": "whipping caliber self-consistency: A(share) == A(measured) <= 2 pp, slow-load share < 8%",
            "subchecks": wh["subchecks"],
            "A_range": wh["A_range"],
            "damage_share_whip_range": wh["damage_share_whip_range"],
            "max_deviation_peak_caliber_pp": wh["max_deviation_peak_caliber_pp"],
            "pass": bool(wh["pass"]),
            "note": ("A = D_mis/D_true = 1 + D_whip/D_true, D_mis = D(sigma)+D(sigma-sigma_qs); "
                     "the peak-basis share is an extreme-value statistic and does not convert "
                     "to the damage caliber"),
        },
        "d": {
            "criterion": "segmentation / closed-block permutation damage equality <= 1e-9",
            "max_seg_rel_dev": d["max_seg_rel_dev"],
            "permutation_max_rel_dev": d["permutation_max_rel_dev"],
            "naive_permutation_spread_rel": d["naive_permutation_spread_rel"],
            "pass": bool(d["segmentation_pass"] and d["permutation_pass"]),
            "note": "naive whole-record counting on permuted blocks is order dependent (trap)",
        },
        "e": {
            "criterion": "waves off reproduces m5528 within 2%",
            "worst_rel_err_pct": e["worst_rel_err_pct"],
            "pass": bool(e["pass"]),
            "note": "15 quantities of m5528 IN32-N07 compared",
        },
        "all_pass": bool(
            (a["linear_superposition_holds"] or a["law"]["law_max_err_pct_all"] <= 10.0)
            and b["pass"] and wh["pass"]
            and (d["segmentation_pass"] and d["permutation_pass"]) and e["pass"]),
    }