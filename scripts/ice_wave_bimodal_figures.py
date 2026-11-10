#!/usr/bin/env python3
"""Figures for node ``P82W5-N09`` from ``artifacts/m8866/results.json``.

Figures (all numbers come from the results file, no hand-typed values):

1. ``fig1_rho_vs_band_ratio.png``   -- sum rule ``rho`` vs ``f_ice/f_w`` (with the
   predeclared +-3 % band and the seed scatter)
2. ``fig2_kappa_map.png``           -- measured carrier amplification ``kappa(r,N)``
   and the calibrated closure / the pure-theory convolution anchor
3. ``fig3_spectral_errors.png``     -- Dirlik / TB / narrow-band / two-band closure
   errors against the +-10 % criterion band
4. ``fig4_sum_rule_envelope.png``   -- the decidable sum-rule envelope ``r_max(N)``
5. ``fig5_spectrum_valley.png``     -- one bimodal PSD with the valley split and the
   two band moment spectra
6. ``fig6_whipping_regimes.png``    -- whipping share vs ``theta`` and inflation ``A``
7. ``fig7_caliber_deviations.png``  -- peak / RMS / moment caliber deviations vs 2 pp
8. ``fig8_commutation.png``         -- streaming-block and block-permutation deviations
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.ice_wave_fatigue.bimodal import band_case  # noqa: E402
from tensorphysics.ice_wave_fatigue.combination import HullGirderCase, WIDTH_LEVELS  # noqa: E402

ART = ROOT / "artifacts" / "m8866"
RES = json.loads((ART / "results.json").read_text())
C = RES["criteria"]


def _save(fig, name: str) -> None:
    out = ART / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig1() -> None:
    rows = RES["criterion_a"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    band = RES["criterion_a"]["band"]
    ax.axhspan(band[0], band[1], color="0.85", label=f"sum rule band {band}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    styles = {"narrow_gamma6": ("o", "C0"), "medium_gamma3p3": ("s", "C1"),
              "wide_bimodal_rp3": ("^", "C2")}
    for width, (marker, color) in styles.items():
        sel = [r for r in rows if r["width"] == width and r["amplitude_mode"] == "equal_damage"]
        sel.sort(key=lambda r: r["ratio_f_ice_over_wave"])
        x = [r["ratio_f_ice_over_wave"] for r in sel]
        y = [r["rho_mean"] for r in sel]
        e = [r["rho_std"] for r in sel]
        ax.errorbar(x, y, yerr=e, marker=marker, color=color, capsize=3,
                    label=f"{width} (equal component damage)")
    rms = [r for r in rows if r["amplitude_mode"] == "equal_rms"]
    ax.scatter([r["ratio_f_ice_over_wave"] for r in rms], [r["rho_mean"] for r in rms],
               marker="x", color="0.4", s=18, label="equal-rms amplitude ratios")
    ax.axvline(10.0, color="C3", ls="--", lw=1.0)
    ax.text(10.5, ax.get_ylim()[0] + 0.02, "criterion (a): band ratio >= 10", color="C3", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("band separation  $f_{ice}/f_{wave}$  [-]")
    ax.set_ylabel(r"$\rho = D_{comb}/(D_{ice}+D_{wave})$  [-]")
    ax.set_title("Plain sum rule is violated at every required band ratio (unsafe side)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, "fig1_rho_vs_band_ratio.png")


def fig2() -> None:
    rows = RES["calibration"]["rows"]
    clos = RES["calibration"]["closure"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    Ns = sorted({r["n_ratio"] for r in rows})
    cmap = plt.get_cmap("viridis")
    for i, N in enumerate(Ns):
        sel = sorted([r for r in rows if r["n_ratio"] == N], key=lambda r: r["r_rider_over_carrier"])
        r = [s["r_rider_over_carrier"] for s in sel]
        ax.plot(r, [s["kappa_truth_dirlik_basis"] for s in sel], "o", color=cmap(i / max(len(Ns) - 1, 1)),
                ms=4, label=f"N={N:g} rainflow truth")
        ax.plot(r, [s["kappa_closure"] for s in sel], "-", color=cmap(i / max(len(Ns) - 1, 1)), lw=1.2)
    sel5 = sorted([r for r in rows if r["n_ratio"] == Ns[0]], key=lambda r: r["r_rider_over_carrier"])
    ax.plot([s["r_rider_over_carrier"] for s in sel5],
            [s["kappa_convolution_theory"] for s in sel5], "k--", lw=1.2,
            label="pure-theory riding convolution (m7361 anchor)")
    ax.set_xlabel(r"$r=\sigma_{rider}/\sigma_{carrier}$  [-]")
    ax.set_ylabel(r"$\kappa$  (carrier damage multiplier)  [-]")
    ax.set_title(f"Carrier amplification map and closure (fit RMS "
                 f"{clos['rms_rel_err_pct']:.1f} %, max {clos['max_rel_err_pct']:.1f} %)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper left")
    _save(fig, "fig2_kappa_map.png")


def fig3() -> None:
    rows = RES["criterion_b"]["rows"]
    labels = [f"{r['level'].split('_')[0]}\nN={r['n_ratio']:.3g}" for r in rows]
    x = np.arange(len(rows))
    w = 0.2
    series = [("err_dirlik_pct", "Dirlik", "C0"), ("err_tb_pct", "Tovo-Benasciutti", "C1"),
              ("err_two_band_pct", "two-band riding", "C3")]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, (key, name, color) in enumerate(series):
        ax.bar(x + (i - 1) * w, [r[key] for r in rows], width=w, label=name, color=color)
    thr = RES["criterion_b"]["threshold_pct"]
    ax.axhline(thr, color="k", ls="--", lw=1.0, label=f"criterion (b) +{thr:.0f} %")
    ax.axhline(-thr, color="k", ls="--", lw=1.0)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("damage error vs rainflow truth  [%]")
    ax.set_title("Single-peak spectral methods fail in the separated regime; the two-band closure holds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig3_spectral_errors.png")


def fig4() -> None:
    rows = RES["envelope"]["rows"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    N = [r["N_freq_ratio"] for r in rows]
    ax.plot(N, [r["r_max_sum_rule"] for r in rows], "o-", color="C0",
            label=f"r_max for sum rule within +-{RES['envelope']['tol'] * 100:.0f} %")
    ax.axvline(RES["settings"]["n_separated"], color="C3", ls="--", lw=1.0,
               label=f"N_separated = {RES['settings']['n_separated']:g}")
    for r in rows:
        ax.annotate(f"{r['r_max_sum_rule']:.3f}", (r["N_freq_ratio"], r["r_max_sum_rule"]),
                    textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("band separation  N = $f_{rider}/f_{carrier}$  [-]")
    ax.set_ylabel("largest riding amplitude ratio r with sum rule ok  [-]")
    ax.set_title("Decidable sum-rule envelope: overlap is only safe for very weak riding")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "fig4_sum_rule_envelope.png")


def fig5() -> None:
    case = HullGirderCase()
    cfg = WIDTH_LEVELS["medium_gamma3p3"]
    out = band_case(case, n_ratio=10.0, wave_kind=cfg["wave_kind"], wave_kwargs=cfg["wave_kwargs"],
                    seed=0, duration=1200.0, sigma_carrier=10.0)
    f = out["freqs"]
    s = out["s_comb"]
    fv = out["split"]["f_valley_hz"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(f, s, "k-", lw=1.2, label="combined PSD  $S_{ice}+S_{wave}$")
    ax.plot(f[f <= fv], s[f <= fv], color="C0", lw=1.0, label="carrier band (wave)")
    ax.plot(f[f > fv], s[f > fv], color="C1", lw=1.0, label="rider band (ice)")
    ax.axvline(fv, color="C3", ls="--", lw=1.0, label=f"valley {fv:.3f} Hz")
    ax.axvline(case.f1_hz, color="0.5", ls=":", lw=1.0, label=f"$f_1$={case.f1_hz:.3f} Hz")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 5.0)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("stress PSD [MPa$^2$/Hz]")
    ax.set_title(f"Bimodal case N={out['n_ratio']:.0f}, r={out['r_rider_over_carrier']:.3f}: "
                 "the valley split defines the two cycle families")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_spectrum_valley.png")


def fig6() -> None:
    rows = RES["criterion_c"]["scan"]["rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for kind, color, marker in (("harmonic", "C0", "o"), ("dem", "C1", "s")):
        sel = [r for r in rows if r["kind"] == kind]
        ax1.plot([r["theta_transition_over_T1"] for r in sel], [r["share_peak"] for r in sel],
                 marker=marker, color=color, ls="-", label=f"{kind} loads")
        ax2.plot([r["share_peak"] for r in sel], [r["A_measured"] for r in sel],
                 marker=marker, color=color, ls="-", label=f"{kind} loads")
    thr = RES["criterion_c"]["scan"]["share_threshold"]
    ax1.axhline(thr, color="C3", ls="--", lw=1.0, label="m5528 claim: share < 8 %")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\theta = \tau_{transition}/T_1$  [-]")
    ax1.set_ylabel("whipping share (peak caliber)  [-]")
    ax1.set_title("Share is a loading-bandwidth quantity")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax2.set_xlabel("whipping share (peak caliber)  [-]")
    ax2.set_ylabel(r"$A = D_{mis}/D_{true}$  [-]")
    ax2.set_title("The same share maps to very different inflations")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    _save(fig, "fig6_whipping_regimes.png")


def fig7() -> None:
    rows = RES["criterion_c"]["scan"]["rows"]
    names = [r["regime"] for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.bar(x - w, [r["dev_peak_pp"] for r in rows], width=w, label="peak caliber (m5528 reading)")
    ax.bar(x, [r["dev_rms_pp"] for r in rows], width=w, label="RMS caliber")
    ax.bar(x + w, [r["dev_moment_pp"] for r in rows], width=w, label="moment (RMS x rate) caliber")
    ax.set_yscale("log")
    ax.axhline(RES["criterion_c"]["consistency"]["tolerance_pp"], color="C3", ls="--", lw=1.0,
               label=f"criterion (c) tolerance {RES['criterion_c']['consistency']['tolerance_pp']:.0f} pp")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("|A predicted - A measured|  [percentage points of A]")
    ax.set_title("Amplitude calibers are invalid outside the quasi-static regime")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "fig7_caliber_deviations.png")


def fig8() -> None:
    stream = RES["criterion_d"]["streaming"]
    perm = RES["criterion_d"]["block_permutation"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    blocks = [r["blocks"] for r in stream["rows"]]
    devs = [max(r["rel_dev"], 1e-18) for r in stream["rows"]]
    ax.plot(blocks, devs, "o-", label="streaming blocks (residue stack)")
    seg = perm["segmentation"]
    ax.plot([s["blocks"] if "blocks" in s else s.get("n_blocks", 0) for s in seg],
            [max(s["rel_dev"], 1e-18) for s in seg], "s--", label="segmented closed blocks")
    ax.axhline(1e-9, color="C3", ls="--", lw=1.0, label="tolerance 1e-9")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of blocks  [-]")
    ax.set_ylabel("relative damage deviation  [-]")
    ax.set_title("Rainflow commutability: block order does not change the damage")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "fig8_commutation.png")


def main() -> int:
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())