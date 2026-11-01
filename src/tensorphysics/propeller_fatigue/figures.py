"""Figures for the W7-N07 report (matplotlib, PNG)."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .hydro_unsteady import sears_gust_function
from .materials_sn import MATERIALS, SNReference, BasquinSN


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def stress_history_figure(out_dir: str, cases: Dict[str, object]) -> Dict[str, str]:
    c = cases["design"]
    h = c["_hist"]
    t = h["t"]
    n_rps = 130.0 / 60.0
    rev_t = 1.0 / n_rps
    m1 = t < 2.2 * rev_t
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.4), sharex=False)
    ax[0].plot(t[m1] * 1e3, h["sigma_hydro"][m1] / 1e6, lw=1.0, label="hydrodynamic alternating")
    ax[0].plot(t[m1] * 1e3, h["sigma_ice"][m1] / 1e6, lw=1.0, label="ice impact")
    ax[0].plot(t[m1] * 1e3, h["sigma"][m1] / 1e6, lw=1.2, color="k", label="combined")
    ax[0].axhline(c["sigma_mean_Pa"] / 1e6, ls="--", color="0.5", label="mean (thrust+torque+CF)")
    ax[0].set_xlabel("time [ms]"); ax[0].set_ylabel("root stress [MPa]")
    ax[0].set_title("Blade root stress, design ice class (2 shafts revs)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    seg = 6.0 * rev_t
    m2 = t < seg
    ax[1].plot(t[m2] * 1e3, h["sigma"][m2] / 1e6, lw=0.8, color="k")
    ax[1].set_xlabel("time [ms]"); ax[1].set_ylabel("combined root stress [MPa]")
    ax[1].set_title("6 revolutions - blade-pass (1P) modulation with ice pulses")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return {"path": _save(fig, out_dir, "fig1_stress_history.png"),
            "caption": "Fig.1 Superposed blade-root stress history: hydrodynamic blade-rate "
                       "alternation, ice impact pulses and the combined signal over 2 and 6 "
                       "shaft revolutions (design ice class)."}


def rainflow_figure(out_dir: str, cases: Dict[str, object]) -> Dict[str, str]:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, col in (("light", "tab:green"), ("design", "tab:orange"), ("heavy", "tab:red")):
        bins = cases[key]["damage_bins"]
        lo = np.array([b["amp_lo_Pa"] for b in bins]) / 1e6
        hi = np.array([b["amp_hi_Pa"] for b in bins]) / 1e6
        d = np.array([b["damage"] for b in bins])
        ax[0].bar((lo + hi) / 2, d, width=(hi - lo) * 0.9, alpha=0.55, color=col,
                  label=f"{key} (D={cases[key]['D_life']:.3g})")
        ax[1].plot((lo + hi) / 2, np.cumsum(d), "o-", color=col, ms=3, label=key)
    ax[0].set_xlabel("equivalent stress amplitude [MPa]")
    ax[0].set_ylabel("Miner damage over design life [-]")
    ax[0].set_yscale("log")
    ax[0].set_title("Rainflow-binned damage spectrum")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("equivalent stress amplitude [MPa]")
    ax[1].set_ylabel("cumulative damage [-]")
    ax[1].set_title("Cumulative damage vs stress level")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return {"path": _save(fig, out_dir, "fig2_rainflow_damage.png"),
            "caption": "Fig.2 Rainflow-counted stress spectrum and the resulting Miner damage "
                       "distribution over the 20-year design exposure for the three ice cases."}


def sn_figure(out_dir: str, result: Dict[str, object]) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    N = np.logspace(3.5, 7.5, 200)
    for key, col in (("NAB", "tab:blue"), ("AL6082", "tab:red")):
        ref = SNReference(key)
        sn = BasquinSN.fit(ref)
        ax.loglog(N, sn.sigma_a(N) / 1e6, "-", color=col,
                  label=f"{key} Basquin fit (b={sn.b:.4f})")
        ax.loglog(ref.N, ref.sigma_a / 1e6, "o", color=col, mfc="none", ms=7,
                  label=f"{key} reference points")
        for n_, s_ in zip(ref.N, ref.sigma_a):
            ax.annotate(f"{(sn.sigma_a(n_)-s_)/s_*100:+.1f}%", (n_, s_ / 1e6),
                        textcoords="offset points", xytext=(6, -10), fontsize=7, color=col)
    ax.set_xlabel("cycles to failure N [-]"); ax.set_ylabel("stress amplitude $\\sigma_a$ [MPa]")
    ax.set_title("S-N (Basquin) fits against reference data")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    return {"path": _save(fig, out_dir, "fig3_sn_curves.png"),
            "caption": "Fig.3 Basquin S-N fits against the reference fatigue data; annotations "
                       "give the relative fit residual at each reference point."}


def dlf_figure(out_dir: str, result: Dict[str, object]) -> Dict[str, str]:
    s = result["dlf_sweep"]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(s["tau_over_T1"], s["dlf_multimode"], "o-", label="multimode closed form (40 modes)")
    ax.plot(s["tau_over_T1"], s["dlf_sdof"], "s--", label="single-DOF (fundamental mode)")
    fd = np.array(s["fd_points"])
    ax.plot(fd[:, 0], fd[:, 1], "^", color="k", ms=9, label="finite-difference PDE")
    ax.axvline(result["beam_impact"]["tau_over_T1"], color="0.4", ls=":",
               label=f"ice pulse $\\tau/T_1$={result['beam_impact']['tau_over_T1']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("pulse duration ratio $\\tau/T_1$ [-]")
    ax.set_ylabel("dynamic load factor $\\sigma_{max}/\\sigma_{static}$ [-]")
    ax.set_title("Impulse response of the equivalent blade span")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    return {"path": _save(fig, out_dir, "fig4_dlf.png"),
            "caption": "Fig.4 Dynamic load factor versus pulse-duration ratio: closed-form "
                       "multimode, single-DOF and independent finite-difference solutions."}


def sears_figure(out_dir: str, result: Dict[str, object]) -> Dict[str, str]:
    rows = result["sears_verification"]["rows"]
    k = np.logspace(-2, 1, 200)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.loglog(k, [abs(sears_gust_function(float(kk))) for kk in k], "-",
              label="Sears $|S(k)|$ (frequency domain)")
    ax.loglog([r["k"] for r in rows], [r["abs_S_timedomain"] for r in rows], "o",
              mfc="none", ms=7, label="Kuessner indicial (time domain)")
    for o in result["hydro"]["ops"]:
        ax.plot(o["reduced_frequency_k_mean"], o["sears_gain_1P"], "D", color="tab:red", ms=7)
        ax.annotate(o["op"], (o["reduced_frequency_k_mean"], o["sears_gain_1P"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color="tab:red")
    ax.set_xlabel("reduced frequency $k=\\pi f c / W$ [-]")
    ax.set_ylabel("gust response magnitude [-]")
    ax.set_title("Unsteady lift response: two independent formulations")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    return {"path": _save(fig, out_dir, "fig5_sears.png"),
            "caption": "Fig.5 Sears gust function (frequency domain) versus the Kuessner "
                       "indicial response integrated in the time domain; diamonds mark the two "
                       "operating points."}


def life_figure(out_dir: str, cases: Dict[str, object]) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    hours = np.logspace(2, 6.2, 200)
    for key, col in (("light", "tab:green"), ("design", "tab:orange"), ("heavy", "tab:red")):
        slope = cases[key]["D_window"] / cases[key]["hours_window"]
        ax.loglog(hours, slope * hours, "-", color=col,
                  label=f"{key}: life {cases[key]['life_hours']:.0f} h, SF={cases[key]['SF_life']:.2f}")
        ax.plot(cases[key]["life_hours"], 1.0, "o", color=col, ms=6)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="D = 1 (crack initiation)")
    ax.axvline(20 * 4000, color="0.4", ls=":", label="20-year design exposure")
    ax.set_xlabel("ice-operating exposure [h]"); ax.set_ylabel("Miner damage D [-]")
    ax.set_title("Crack-initiation life from linear damage accumulation")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.tight_layout()
    return {"path": _save(fig, out_dir, "fig6_life.png"),
            "caption": "Fig.6 Miner damage versus ice-operating exposure for the three ice "
                       "cases; intersections with D=1 give the crack-initiation life."}


def make_all_figures(out_dir: str, result, cases, geom, mat, env, wake, dp) -> List[Dict[str, str]]:
    os.makedirs(out_dir, exist_ok=True)
    figs = [
        stress_history_figure(out_dir, cases),
        rainflow_figure(out_dir, cases),
        sn_figure(out_dir, result),
        dlf_figure(out_dir, result),
        sears_figure(out_dir, result),
        life_figure(out_dir, cases),
    ]
    return [{"name": os.path.basename(f["path"]), "path": f["path"],
             "caption": f["caption"]} for f in figs]