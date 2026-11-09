#!/usr/bin/env python3
"""Figures for node P82W4-N06 (ice + wave combined fatigue).

Reads ``artifacts/m7361/results.json`` (produced by
``scripts/ice_wave_fatigue_study.py``) and writes ``artifacts/m7361/fig*.png``.
Every plotted number comes from that JSON or is recomputed by the module's own
functions -- nothing is hand-typed.

Usage::

    python3 scripts/ice_wave_fatigue_figures.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorphysics.ice_wave_fatigue import (  # noqa: E402
    CALIBRATED_M5528,
    HullGirderCase,
    SeaState,
    ice_stress_history,
    physical_wave_component,
)
from tensorphysics.ice_wave_fatigue.combination import (  # noqa: E402
    WIDTH_LEVELS,
    component_grid,
    gaussian_band_shape,
    kappa_enlargement,
    stress_spectrum_shape,
)

COLORS = {
    "ice": "#c0392b",
    "wave": "#1f6fb4",
    "comb": "#2c3e50",
    "law": "#16a085",
    "sum": "#7f8c8d",
    "tb": "#8e44ad",
    "dirlik": "#d35400",
    "nb": "#95a5a6",
}


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fig1_time_histories(res: dict, out: str) -> str:
    case = HullGirderCase()
    ice = ice_stress_history(case, CALIBRATED_M5528, duration=180.0, dt=2.0e-3)
    sea = SeaState(hs=4.0, tp=9.0, gamma=3.3)
    wave = physical_wave_component(case, sea, n_samples=int(180.0 * 100.0), fs=100.0, seed=3)
    t_ice = ice["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10.5))

    ax = axes[0]
    m = (t_ice >= 60.0) & (t_ice <= 120.0)
    load = res["criteria"]["e_negative_control_m5528"]["computed"]
    ax.set_ylabel("ice hotspot stress [MPa]")
    ax.plot(t_ice[m], ice["sigma_mpa"][m], color=COLORS["ice"], lw=0.8, label="total (modal superposition)")
    ax.plot(t_ice[m], ice["sigma_qs_mpa"][m], color=COLORS["ice"], lw=1.6, ls="--", alpha=0.8,
            label="quasi-static limit")
    ax.set_title(
        f"(a) DEM-calibrated ice load (m5528 parameters): peak {load['force_peak_MN']:.2f} MN, "
        f"mean {load['force_mean_MN']:.2f} MN, whipping share {load['whipping_share_peak'] * 100:.1f} %",
        fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    t_w = np.arange(wave["sigma_mpa"].size) / 100.0
    mw = t_w <= 180.0
    ax.plot(t_w[mw], wave["sigma_mpa"][mw], color=COLORS["wave"], lw=0.8)
    ax.set_ylabel("wave hotspot stress [MPa]")
    ax.set_title(f"(b) wave leg, {wave['sea_state']}: rms {wave['sigma_rms_mpa']:.2f} MPa, "
                 f"eps = {wave['width']['eps']:.3f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    n = min(ice["sigma_mpa"].size, wave["sigma_mpa"].size)
    comb = ice["sigma_mpa"][:n] + wave["sigma_mpa"][:n]
    tc = t_ice[:n]
    mc = (tc >= 60.0) & (tc <= 120.0)
    ax.plot(tc[mc], comb[mc], color=COLORS["comb"], lw=0.8)
    ax.set_ylabel("ice + wave [MPa]")
    ax.set_title("(c) superposed stress (linear superposition of the two linear responses)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[3]
    mz = (tc[:n] >= 90.0) & (tc[:n] <= 100.0)
    ax.plot(tc[:n][mz], ice["sigma_mpa"][:n][mz], color=COLORS["ice"], lw=0.9, label="ice")
    ax.plot(tc[:n][mz], wave["sigma_mpa"][:n][mz], color=COLORS["wave"], lw=0.9, label="wave")
    ax.plot(tc[:n][mz], comb[:n][mz], color=COLORS["comb"], lw=1.2, label="combined")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("stress [MPa]")
    ax.set_title("(d) zoomed 10 s: the high-frequency rider enlarges the low-frequency carrier range",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "fig1_time_histories.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig2_spectra(res: dict, out: str) -> str:
    case = HullGirderCase()
    n, fs = 36000, 20.0
    f_ice = case.ice_band_hz
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    grid, s_ice, _ = component_grid(gaussian_band_shape(f_ice, 0.15), 5.0, n, fs)
    ax.semilogy(grid[1:], s_ice[1:], color=COLORS["ice"], lw=1.6, label=f"ice band @ {f_ice:g} Hz")
    for name, cfg in WIDTH_LEVELS.items():
        shape = stress_spectrum_shape(cfg["wave_kind"], f_ice / 10.0, **cfg["wave_kwargs"])
        _, s_w, _ = component_grid(shape, 5.0, n, fs)
        ax.semilogy(grid[1:], s_w[1:], lw=1.0, ls="--", alpha=0.85,
                    label=f"wave {name}")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("stress PSD [MPa$^2$/Hz]")
    ax.set_title("(a) stress-space spectra at f_ice/f_w = 10 (unit rms)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    sea = SeaState()
    wave = physical_wave_component(case, sea, n_samples=n, fs=20.0, seed=0)
    ax.semilogy(wave["freqs_hz"][1:], wave["psd_mpa2_hz"][1:], color=COLORS["wave"], lw=1.4,
                label=f"wave (physical RAO), eps={wave['width']['eps']:.2f}")
    widths = res["criteria"]["b_spectral_methods"]["rows"]
    row = next(r for r in widths if r["level"] == "wide_bimodal_rp3" and r["ratio"] == 10.0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("stress PSD [MPa$^2$/Hz]")
    ax.set_title(f"(b) physical wave-stress PSD: bimodal case alpha2={row['alpha2']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out, "fig2_spectra.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig3_combination_law(res: dict, out: str) -> str:
    a = res["criteria"]["a_combination_law"]
    rows = res["scan_rows"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    modes = {"equal_rms": ("o", "equal rms"), "equal_damage": ("s", "equal per-component damage")}
    for mode, (marker, label) in modes.items():
        sel = [r for r in rows if r["amplitude_mode"] == mode]
        if not sel:
            continue
        x = [r["ratio"] for r in sel]
        y = [r["rho_mean"] for r in sel]
        ax.errorbar(x, y, yerr=[r["rho_std"] for r in sel], marker=marker, ls="none",
                    capsize=3, color=COLORS["comb"], label=f"rainflow truth ({label})")
        ax.plot(x, [r["rho_tb_model_mean"] for r in sel], marker=marker, ls=":",
                color=COLORS["tb"], alpha=0.8, label=f"Tovo-Benasciutti only ({label})")
    law_sel = [r for r in a["law"]["scored_holdout"]["rows"]
               if r["amplitude_mode"] == "equal_rms"]
    ax.plot([r["ratio"] for r in law_sel], [r["law_rho"] for r in law_sel], marker="^",
            color=COLORS["law"], lw=1.6, label="combination law (eq. in report)")
    ax.axhline(1.0, color=COLORS["sum"], lw=1.4, ls="--", label="linear superposition D1 + D2")
    ax.axhspan(a["band"][0], a["band"][1], color="green", alpha=0.15,
               label=f"acceptance band {a['band']}")
    ax.axvline(a["separated_ratio_threshold"], color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("band separation ratio  $f_{ice}/f_w$")
    ax.set_ylabel(r"$\rho = D_{comb}/(D_{ice}+D_{wave})$")
    ax.set_title("(a) combined / summed damage versus band separation\n"
                 f"sum rule mean error {a['law']['sum_rule_mean_err_pct']:.1f} %, "
                 f"law max error {a['law']['law_max_err_pct_all']:.1f} %", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out, "fig3_combination_law.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig4_mechanism(res: dict, out: str) -> str:
    mech = res["mechanism"]
    calib = res["criteria"]["a_combination_law"]["law"]["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    x = np.arange(len(mech))
    ax.bar(x - 0.18, [m["delta_measured_mpa"] for m in mech], width=0.36,
           color=COLORS["comb"], label="measured $\\Delta$ carrier range")
    ax.bar(x + 0.18, [m["delta_predicted_mpa"] for m in mech], width=0.36,
           color=COLORS["law"], label="predicted $2\\sqrt{\\pi/2}\\,\\sigma_{rider}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$f_{{ice}}/f_w$ = {m['ratio']:g}" for m in mech])
    ax.set_ylabel("carrier range increase [MPa]")
    ax.set_title("(a) mechanism: the rider enlarges the carrier ranges", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    r = np.logspace(-0.6, 0.7, 100)
    for N, style in ((10.0, "-"), (100.0, "--")):
        ax.plot(r, [kappa_enlargement(xx, N, calib["a"], calib["b"], calib["c"]) for xx in r],
                ls=style, color=COLORS["law"], label=f"law, N = {N:g}")
    pts = calib["points"]
    ax.plot([p["r"] for p in pts], [p["kappa_measured"] for p in pts], "o",
            color=COLORS["ice"], label="measured $\\kappa$ (calibration points)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$r = \\sigma_{rider}/\\sigma_{carrier}$")
    ax.set_ylabel("carrier damage amplification $\\kappa$")
    ax.set_title(f"(b) $\\kappa = 1 + a r^b N^c$ with a={calib['a']:.3f}, b={calib['b']:.3f}, "
                 f"c={calib['c']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out, "fig4_mechanism.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig5_spectral_errors(res: dict, out: str) -> str:
    rows = res["criteria"]["b_spectral_methods"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, ratio in zip(axes, (10.0, 1.0)):
        sel = [r for r in rows if r["ratio"] == ratio]
        x = np.arange(len(sel))
        w = 0.26
        ax.bar(x - w, [r["err_dirlik_pct"] for r in sel], width=w, color=COLORS["dirlik"], label="Dirlik")
        ax.bar(x, [r["err_tb_pct"] for r in sel], width=w, color=COLORS["tb"], label="Tovo-Benasciutti")
        ax.bar(x + w, [r["err_nb_pct"] for r in sel], width=w, color=COLORS["nb"], label="narrow band")
        ax.axhline(10.0, color="r", ls="--", lw=1)
        ax.axhline(-10.0, color="r", ls="--", lw=1)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["level"].replace("_", "\n") for r in sel], fontsize=7.5)
        ax.set_title(f"band ratio $f_{{ice}}/f_w$ = {ratio:g}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("damage error vs rainflow truth [%]")
    fig.suptitle("criterion (b): spectral-method damage error on the combined PSD "
                 f"(Dirlik worst {res['criteria']['b_spectral_methods']['worst_err_dirlik_pct']:.1f} %, "
                 f"TB worst {res['criteria']['b_spectral_methods']['worst_err_tb_pct']:.1f} %)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, "fig5_spectral_errors.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig6_whipping(res: dict, out: str) -> str:
    rows = res["criteria"]["c_whipping_bookkeeping"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.22, [r["amplification_A"] for r in rows], width=0.22, color=COLORS["comb"],
           label="measured $A = D_{mis}/D_{true}$")
    ax.bar(x, [r["A_predicted_peak_caliber"] for r in rows], width=0.22, color=COLORS["ice"],
           label="peak caliber (m5528) $1+(s/(1-s))^m$")
    ax.bar(x + 0.22, [r["A_predicted_rms_x_rate"] for r in rows], width=0.22,
           color=COLORS["law"], label="rms $\\times$ cycle-rate caliber")
    ax.set_xticks(x)
    ax.set_xticklabels([r["regime"].replace("_", "\n") for r in rows], fontsize=7)
    ax.set_ylabel("damage amplification of double counting whipping [-]")
    ax.set_title("(a) double-counting bias: measured vs three share calibers", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x - 0.18, [r["share_peak"] for r in rows], width=0.36, color=COLORS["ice"],
           label="peak share (m5528 caliber)")
    ax.bar(x + 0.18, [r["share_rms"] for r in rows], width=0.36, color=COLORS["wave"],
           label="rms share")
    ax.plot(x, [r["damage_share_whip"] for r in rows], "k^", ms=7,
            label="whipping damage share $D_{whip}/D_{true}$")
    ax.set_xticks(x)
    ax.set_xticklabels([r["regime"].replace("_", "\n") for r in rows], fontsize=7)
    ax.set_ylabel("whipping share [-]")
    ax.set_title("(b) the peak share is an extreme-value statistic;\n"
                 "the damage share is what a damage-bias argument needs", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out, "fig6_whipping_calibers.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig7_commutation(res: dict, out: str) -> str:
    d = res["criteria"]["d_rainflow_commutability"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = [f"segmentation\n{c['n_chunks']} chunks" for c in d["segmentation"]]
    vals = [max(c["rel_dev"], 1e-18) for c in d["segmentation"]]
    labels.append("closed-block\npermutation")
    vals.append(max(d["permutation_max_rel_dev"], 1e-18))
    labels.append("naive permuted\nwhole-record count")
    vals.append(d["naive_permutation_spread_rel"])
    colors = [COLORS["law"]] * (len(vals) - 1) + [COLORS["ice"]]
    ax.bar(labels, vals, color=colors)
    ax.axhline(d["permutation_tolerance"], color="r", ls="--", lw=1.2,
               label=f"tolerance {d['permutation_tolerance']:g}")
    ax.set_yscale("log")
    ax.set_ylabel("relative damage deviation [-]")
    ax.set_title("criterion (d): rainflow commutability", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out, "fig7_commutation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "m7361"))
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    results_path = args.results or os.path.join(args.out, "results.json")
    res = _load(results_path)
    made = [
        fig1_time_histories(res, args.out),
        fig2_spectra(res, args.out),
        fig3_combination_law(res, args.out),
        fig4_mechanism(res, args.out),
        fig5_spectral_errors(res, args.out),
        fig6_whipping(res, args.out),
        fig7_commutation(res, args.out),
    ]
    for p in made:
        print(f"wrote {p} ({os.path.getsize(p)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())