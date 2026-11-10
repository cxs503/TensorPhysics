#!/usr/bin/env python3
"""P82W2-N08 -- ice+wave combined fatigue: combination-law adjudication.

Runs the whole designed experiment and writes

    artifacts/p82w2_n08/combination_law.json   (all numbers, programmatic)
    artifacts/p82w2_n08/fig1..fig6*.png        (figures)

Criteria (pre-declared, see ``docs/P82W2_N08_ICE_WAVE_COMBINATION_LAW.md``):
a) linear superposition: D_comb/(D_ice+D_wave) in [0.97,1.03] for f_ice/f_w >= 10
b) spectral closures (Dirlik / Tovo-Benasciutti) vs rainflow truth <= 10 %
   over three spectral widths
c) whipping accounting: damage amplification explained by the whipping share
   within 2 percentage points (slow-varying whipping share < 8 %)
d) rainflow commutativity: order changes give bit-identical damage (1e-9)
e) negative control: waves off -> m5528 reference within 2 %

Usage:  PYTHONPATH=src python3 scripts/run_p82w2_n08_combination_law.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.fatigue.combined import combination as CB  # noqa: E402
from tensorphysics.fatigue.combined.processes import (  # noqa: E402
    TwoBandCase, band_psd, realised_process, rainflow_damage_rate,
)
from tensorphysics.fatigue.rainflow import rainflow_counting_stack  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "p82w2_n08"
OUT.mkdir(parents=True, exist_ok=True)


def _jsonable(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
def run():
    print("[1/6] anchor negative control (criterion e)")
    from tensorphysics.fatigue.combined.ice_anchor import (
        ANCHOR_REFERENCE_RESULT, run_anchor_case,
    )
    anchor = run_anchor_case()
    ref = ANCHOR_REFERENCE_RESULT
    e_ctrl = {
        "damage_annual_goodman": anchor["fatigue"]["damage_annual_goodman"],
        "reference": ref["damage_annual_goodman"],
        "rel_dev_pct": 100.0 * (anchor["fatigue"]["damage_annual_goodman"] / ref["damage_annual_goodman"] - 1.0),
        "life_years": anchor["fatigue"]["life_years"],
        "reference_life_years": ref["life_years"],
        "life_rel_dev_pct": 100.0 * (anchor["fatigue"]["life_years"] / ref["life_years"] - 1.0),
        "whipping_share_peak": anchor["whipping"]["share_peak"],
        "reference_whipping_share_peak": ref["whipping_share_peak"],
        "tolerance_pct": 2.0,
        "damage_annual_standard_knee": anchor["fatigue"]["damage_annual_goodman_standard_knee"],
    }
    e_ctrl["passes"] = bool(abs(e_ctrl["rel_dev_pct"]) <= 2.0)

    print("[2/6] frequency-separation design (criterion a)")
    scan = CB.linearity_scan(
        ratios=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        splits=(0.5, 1.0, 2.0),
        f_fast=0.5, sigma_wave=5.0, bw=0.05,
        duration=10800.0, fs=20.0, seeds=(11, 23, 37, 41),
    )
    region = CB.admissible_region(scan["rows"])
    sep_rows = [r for r in scan["rows"]
                if max(float(r["f_ratio"]), 1.0 / float(r["f_ratio"])) >= 10.0]
    a_ctrl = {
        "tolerance_abs": CB.LINEARITY_TOL,
        "cells_at_separation_ge_10": [
            {"f_ratio": r["f_ratio"], "sigma_split": r["sigma_ice_MPa"] / r["sigma_wave_MPa"],
             "ratio_mean": r["ratio_mean"], "ratio_std": r["ratio_std"],
             "in_tolerance": bool(abs(r["ratio_mean"] - 1.0) <= CB.LINEARITY_TOL)}
            for r in sep_rows
        ],
        "all_in_tolerance": bool(all(abs(r["ratio_mean"] - 1.0) <= CB.LINEARITY_TOL for r in sep_rows)),
        "worst_overall": max(scan["rows"], key=lambda r: abs(float(r["ratio_mean"]) - 1.0))["ratio_mean"],
        "admissible_region": region,
    }
    a_ctrl["passes"] = a_ctrl["all_in_tolerance"]

    print("[3/6] spectral closures vs three spectral widths (criterion b)")
    widths = CB.spectral_width_study(duration=3600.0, fs=20.0, seeds=(5, 17, 29, 43, 61))
    b_ctrl = {
        "tolerance_abs_pct": 100.0 * CB.SPECTRAL_ERROR_TOL,
        "rows": widths["rows"],
        "passes": bool(all(abs(r["err_dirlik_abs_pct_mean"]) <= 100.0 * CB.SPECTRAL_ERROR_TOL
                           and abs(r["err_tb_abs_pct_mean"]) <= 100.0 * CB.SPECTRAL_ERROR_TOL
                           for r in widths["rows"])),
    }

    print("[4/6] whipping accounting (criterion c)")
    whip = CB.whipping_accounting(duration=600.0, dt=2.0e-3)
    slow = [r for r in whip["rows"] if r["operating_point"] == "quasi_static"][0]
    c_ctrl = {
        "tolerance_pp": CB.WHIPPING_TOL_PP,
        "slow_varying_share_peak": slow["share_peak"],
        "slow_varying_below_8pct": bool(slow["share_peak"] < 0.08),
        "physical_deviations_pp": [r["deviation_pp"] for r in whip["rows"]],
        "synthetic_deviation_pp": whip["synthetic_share_law"]["deviation_pp"],
        "synthetic_passes": whip["synthetic_share_law"]["passes"],
        "verdict": whip["verdict"],
    }
    c_ctrl["passes"] = bool(c_ctrl["synthetic_passes"] and c_ctrl["verdict"]["law_holds_on_physical_ice"])

    print("[5/6] rainflow commutativity (criterion d)")
    fs_c, T_c = 20.0, 600.0
    t = np.arange(int(T_c * fs_c)) / fs_c
    periodic = np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 0.05 * t)
    comm_periodic = CB.rainflow_commutativity(periodic, 1.0 / fs_c, f_period_s=20.0)
    case = TwoBandCase(f_ice=0.5, f_wave=0.05, sigma_ice=5.0, sigma_wave=5.0,
                       duration=3600.0, fs=20.0, seed=11)
    ice_r = realised_process(band_psd(case.f_ice, case.sigma_ice, case.bw_ice), case.n_samples,
                             case.fs, seed=11, f_centre=case.f_ice,
                             sigma_f=case.bw_ice * case.f_ice)
    wave_r = realised_process(band_psd(case.f_wave, case.sigma_wave, case.bw_wave), case.n_samples,
                              case.fs, seed=7712, f_centre=case.f_wave,
                              sigma_f=case.bw_wave * case.f_wave)
    comb = np.asarray(ice_r["history"]) + np.asarray(wave_r["history"])
    comm_random = CB.rainflow_commutativity(comb, 1.0 / case.fs, f_period_s=None)
    d_ctrl = {
        "tolerance_rel": CB.COMMUTATIVITY_TOL,
        "periodic_record": comm_periodic["checks_rel_diff"],
        "stationary_composite": comm_random["checks_rel_diff"],
        "stationary_raw_open_segment_permutation_max": comm_random["raw_open_segment_permutation_max"],
        "segmentation_vs_whole_rel_diff": comm_random["segmentation_vs_whole_rel_diff"],
        "passes": bool(all(comm_periodic["pass_1e9"].values())
                       and all(comm_random["pass_1e9"].values())
                       and comm_periodic["pass_1e9"] and comm_random["pass_1e9"]),
    }

    print("[6/6] physically grounded case (DEM ice chain + JONSWAP wave)")
    phys = CB.realistic_case(duration=3600.0, dt=5.0e-3)

    criteria = {
        "a_linear_superposition": a_ctrl,
        "b_spectral_closures": b_ctrl,
        "c_whipping_accounting": c_ctrl,
        "d_rainflow_commutativity": d_ctrl,
        "e_negative_control_m5528": e_ctrl,
    }
    result = {
        "node": "P82W2-N08",
        "mission_id": 7763,
        "repo": "TensorPhysics",
        "sn": {"C_F3_CP": CB.SN_C_F3_CP, "m": CB.SN_M, "gamma_range": CB.GAMMA_RANGE},
        "criteria": criteria,
        "scan": scan,
        "widths": widths,
        "whipping": whip,
        "physical_case": phys,
        "anchor": anchor,
        "analytic_bounds_equal_rms": CB.analytic_bounds(5.0, 5.0),
        "verdict": {
            "a": "PASS" if a_ctrl["passes"] else "FAIL",
            "b": "PASS" if b_ctrl["passes"] else "FAIL",
            "c": "PASS" if c_ctrl["passes"] else "FAIL",
            "d": "PASS" if d_ctrl["passes"] else "FAIL",
            "e": "PASS" if e_ctrl["passes"] else "FAIL",
        },
    }
    (OUT / "combination_law.json").write_text(json.dumps(_jsonable(result), indent=1))
    print("wrote", OUT / "combination_law.json")

    _figures(result, scan, widths, whip, phys, anchor, ice_r, wave_r, comb, t, periodic)
    return result


# ---------------------------------------------------------------------------
def _figures(result, scan, widths, whip, phys, anchor, ice_r, wave_r, comb, t, periodic):
    rows = scan["rows"]
    splits = sorted({round(r["sigma_ice_MPa"] / r["sigma_wave_MPa"], 6) for r in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(splits)))

    # fig1: kappa(r) with tolerance band and the fused limit
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for c, sp in zip(colors, splits):
        g = sorted([r for r in rows if round(r["sigma_ice_MPa"] / r["sigma_wave_MPa"], 6) == sp],
                   key=lambda r: r["f_ratio"])
        x = [max(r["f_ratio"], 1.0 / r["f_ratio"]) for r in g]
        y = [r["ratio_mean"] for r in g]
        e = [r["ratio_std"] for r in g]
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, lw=1.6, capsize=2.5, color=c,
                    label=rf"$\sigma_{{ice}}/\sigma_{{wave}}={sp:g}$")
    ax.axhspan(1.0 - CB.LINEARITY_TOL, 1.0 + CB.LINEARITY_TOL, color="0.85", zorder=0,
               label="±3 % tolerance")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    for r in (10.0,):
        ax.axvline(r, color="tab:red", ls="--", lw=0.9)
        ax.text(r * 1.05, 1.42, r"$f_{ice}/f_w=10$", color="tab:red", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("fast/slow band separation  $\\max(r,1/r)$")
    ax.set_ylabel("$\\kappa = D_{comb}/(D_{ice}+D_{wave})$")
    ax.set_title("Combination law: linear additivity fails well past a decade of separation")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_combination_law.png", dpi=150)
    plt.close(fig)

    # fig2: mechanism (range histograms, r=10, equal RMS)
    case = TwoBandCase(f_ice=0.5, f_wave=0.05, sigma_ice=5.0, sigma_wave=5.0,
                       duration=10800.0, fs=20.0, seed=11)
    ice = realised_process(band_psd(case.f_ice, 5.0, 0.05), case.n_samples, case.fs,
                           seed=11, f_centre=case.f_ice, sigma_f=0.025)
    wav = realised_process(band_psd(case.f_wave, 5.0, 0.05), case.n_samples, case.fs,
                           seed=7712, f_centre=case.f_wave, sigma_f=0.0025)
    h = np.asarray(ice["history"]) + np.asarray(wav["history"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, (name, hist, col) in zip(axes, [("ice band + wave band (sum)", h, "tab:blue"),
                                            ("wave band alone", wav["history"], "tab:orange")]):
        rf = rainflow_counting_stack(np.asarray(hist))
        ax.hist(rf.ranges, bins=60, weights=rf.counts, color=col, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("rainflow stress range [MPa]")
        ax.set_ylabel("count")
        ax.set_title(f"{name}\n$D$={rainflow_damage_rate(hist, 1.0 / case.fs)['damage_rate']:.3e} /s")
        ax.grid(alpha=0.3)
    fig.suptitle("Mechanism at $f_{ice}/f_w=10$: the slow cycles absorb the fast peak-to-trough "
                 "amplitude (range grows), while the fast cycles keep their ranges")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_mechanism_ranges.png", dpi=150)
    plt.close(fig)

    # fig3: spectral closure error vs spectral width
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    a2 = [r["alpha2"] for r in widths["rows"]]
    ax.bar(np.arange(len(a2)) - 0.18, [r["err_dirlik_abs_pct_mean"] for r in widths["rows"]],
           width=0.36, label="Dirlik", color="tab:blue")
    ax.bar(np.arange(len(a2)) + 0.18, [r["err_tb_abs_pct_mean"] for r in widths["rows"]],
           width=0.36, label="Tovo-Benasciutti", color="tab:green")
    for i, r in enumerate(widths["rows"]):
        ax.text(i, 1.0, f"{r['shape']}\n$\\alpha_2$={r['alpha2']:.3f}", ha="center", fontsize=7.5)
    ax.axhline(10.0, color="tab:red", ls="--", lw=1.0, label="10 % acceptance line")
    ax.set_xticks([])
    ax.set_ylabel("|D_spectral / D_rainflow − 1|  [%]")
    ax.set_title("Spectral closures vs rainflow truth (5 realisations each)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_spectral_closures.png", dpi=150)
    plt.close(fig)

    # fig4: whipping accounting
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    names = [r["operating_point"] for r in whip["rows"]]
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [100 * r["share_peak"] for r in whip["rows"]], width=0.4,
                label="share_peak (m5528 convention)", color="tab:blue")
    axes[0].bar(x + 0.2, [100 * r["share_rms"] for r in whip["rows"]], width=0.4,
                label="share_rms", color="tab:cyan")
    axes[0].axhline(8.0, color="tab:red", ls="--", lw=1.0, label="8 % (slow-varying)")
    axes[0].set_xticks(x, names, fontsize=8)
    axes[0].set_ylabel("whipping stress share [%]")
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("Whipping stress share by operating point")
    axes[0].grid(alpha=0.3, axis="y")
    w = np.array([100 * r["share_rms"] for r in whip["rows"]])
    meas = np.array([r["A_measured_pct"] for r in whip["rows"]])
    pred = np.array([r["A_predicted_pct"] for r in whip["rows"]])
    axes[1].bar(x - 0.2, meas, width=0.4, label="$A_{measured}$ (rainflow)", color="tab:purple")
    axes[1].bar(x + 0.2, pred, width=0.4, label="$A_{predicted}$ (share law)", color="tab:gray")
    axes[1].set_yscale("symlog")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("damage amplification of whipping [%]")
    axes[1].set_title("The stress share does not predict the damage amplification")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_whipping_accounting.png", dpi=150)
    plt.close(fig)

    # fig5: physical case
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.0))
    n_show = int(120.0 * 200)
    tt = np.arange(n_show) * 0.005
    from tensorphysics.fatigue.combined.processes import combined_spectrum  # noqa: F401
    from tensorphysics.fatigue.combined.ice_anchor import (
        FreeFreeBeamModes, IceContactParams, generate_ice_force, modal_stress_history,
    )
    from tensorphysics.fatigue.combined.ice_anchor import ANCHOR_CASE
    cfg = dict(ANCHOR_CASE)
    modes = FreeFreeBeamModes(cfg["length_m"], cfg["EI_Nm2"], cfg["mass_per_length_kg_m"],
                              n_modes=int(cfg["n_modes"]), zeta=cfg["zeta"])
    _, F = generate_ice_force(IceContactParams.calibrated(), duration=120.0, dt=0.005)
    resp = modal_stress_history(modes, F, 0.005, cfg["x_load_m"], cfg["x_detail_m"],
                                cfg["z_deck_m"], cfg["E_Pa"])
    ice_short = resp["sigma"] / 1.0e6
    axes[0].plot(tt, ice_short, lw=0.8, label="ice (DEM chain, whipping included)")
    axes[0].plot(tt, resp["sigma_qs"] / 1.0e6, lw=1.0, ls="--", label="quasi-static part")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("nominal stress [MPa]")
    axes[0].set_title("Ice-induced hot-spot stress: resonant ringing (whipping) dominates the ranges")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].bar(np.arange(3) - 0.2, [r["ratio_linear"] for r in phys["rows"]], width=0.4,
                label=r"$\kappa$ measured (physical case)", color="tab:blue")
    axes[1].axhline(1.03, color="tab:red", ls="--", lw=1.0)
    axes[1].set_xticks(range(3), [f"wave seed {r['wave_seed']}" for r in phys["rows"]], fontsize=8)
    axes[1].set_ylabel("$D_{comb}/(D_{ice}+D_{wave})$")
    axes[1].set_title(
        rf"Physical case: separation {phys['design']['separation_ratio']:.2f}, "
        rf"$D_{{ice}}/D_{{wave}}$={phys['summary']['D_ice_over_D_wave']:.1f} "
        rf"-> $\kappa$={phys['summary']['ratio_linear_mean']:.3f} (linear law unsafe)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_physical_case.png", dpi=150)
    plt.close(fig)

    # fig6: negative control
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    e = result["criteria"]["e_negative_control_m5528"]
    vals = [e["damage_annual_goodman"], e["reference"], e["damage_annual_standard_knee"]]
    labels = ["TensorPhysics\nport (m5528 knee)", "m5528\npublished",
              "same cycles,\nDNV-standard knee"]
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:gray", "tab:orange"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("annual damage (DNV F3 CP, Goodman)")
    ax.set_title("Negative control: waves off reproduces m5528\n"
                 f"rel. deviation {e['rel_dev_pct']:+.2e} %")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_negative_control.png", dpi=150)
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    res = run()
    print(json.dumps(res["verdict"], indent=1))