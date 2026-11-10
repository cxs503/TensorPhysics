#!/usr/bin/env python3
"""Figures for the Cowper-Symonds / ice-impact DAF node (p351w1_n05).

Every number plotted is either computed here or read from the benchmark artefact
``benchmarks/rate_plasticity_daf/results.json`` (no hand-entered results).

Outputs (artifacts/p351w1_n05/):
    fig1_shock_spectrum.png   half-sine shock spectrum: closed form vs RK4
    fig2_cs_rate_law.png      Cowper-Symonds flow-stress ratio vs strain rate
    fig3_stress_history.png   stress histories, elastic vs rate-dependent case
    fig4_daf_surface.png      DAF(tau, P) and rate-factor surfaces
    fig5_six_cases.png        six-case DAF validation vs the modal reference
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
for candidate in (_HERE / "src", _HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensorphysics.plasticity.cowper_symonds import (  # noqa: E402
    CowperSymonds,
    uniaxial_tension_constant_rate,
)
from tensorphysics.plasticity.impact_daf import (  # noqa: E402
    BeamHingeFEM,
    ImpactCase,
    beam_modal_elastic_daf,
    sdof_half_sine_daf,
    sdof_half_sine_daf_rk4,
)

RESULTS = _HERE / "benchmarks" / "rate_plasticity_daf" / "results.json"
OUTDIR = _HERE / "artifacts" / "p351w1_n05"


def fig1_shock_spectrum(results: dict) -> pathlib.Path:
    ratios = np.linspace(0.05, 3.0, 240)
    exact = np.array([sdof_half_sine_daf(r) for r in ratios])
    rk4 = np.array([sdof_half_sine_daf_rk4(r, n_steps=6000) for r in ratios])
    peak = results["b_shock_spectrum"]["peak"]
    lit = results["literature_anchors"]["peak_undamped_half_sine_daf"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ratios, exact, "k-", lw=2, label="closed form (exact Duhamel)")
    ax.plot(ratios, rk4, "o", ms=3.0, mfc="none", color="tab:red", label="RK4 (independent)")
    ax.axhline(lit, ls=":", color="tab:gray", label=f"textbook peak = {lit}")
    ax.plot([peak["tau_over_T"]], [peak["daf_max"]], "*", ms=14, color="tab:blue",
            label=f"computed peak = {peak['daf_max']:.4f} at tau/T = {peak['tau_over_T']:.3f}")
    ax.set_xlabel(r"pulse duration / natural period  $\tau/T_1$")
    ax.set_ylabel(r"DAF $=\max|x|/x_{static}$")
    ax.set_title("Undamped half-sine shock spectrum (SDOF reference)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "fig1_shock_spectrum.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig2_cs_rate_law(results: dict) -> pathlib.Path:
    cs = CowperSymonds()
    rates = np.logspace(-6, 4, 300)
    amps = cs.amplification(rates)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(rates, amps, "k-", lw=2, label="Cowper-Symonds  D=40.4 s$^{-1}$, p=5")
    rows = results["a_cowper_symonds"]["perfectly_plastic"]
    ax.plot([r["strain_rate"] for r in rows], [r["sigma_dynamic"] / 235.0e6 for r in rows],
            "s", ms=7, color="tab:blue", label="uniaxial model (return mapping)")
    ax.plot([r["strain_rate"] for r in rows], [r["sigma_analytic"] / 235.0e6 for r in rows],
            "x", ms=9, color="tab:red", label="closed form $\\sigma_0 A(\\dot\\epsilon_p)$")
    lit_rate = 100.0
    lit = results["literature_anchors"]["mild_steel_flow_stress_ratio_100_per_s"]
    ax.plot([lit_rate], [lit], "D", ms=8, mfc="none", color="tab:green",
            label=f"literature anchor at 100 s$^{{-1}}$ (~{lit})")
    ax.plot([lit_rate], [float(cs.amplification(lit_rate))], "D", ms=8, color="tab:green",
            label=f"model at 100 s$^{{-1}}$ = {float(cs.amplification(lit_rate)):.3f}")
    ax.set_xlabel(r"plastic strain rate  $\dot\epsilon_p$  [s$^{-1}$]")
    ax.set_ylabel(r"$\sigma_d/\sigma_0$")
    ax.set_title("Strain-rate sensitivity of the flow stress (mild steel)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "fig2_cs_rate_law.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig3_stress_history() -> pathlib.Path:
    base = ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01)
    p_y = base.yield_pressure
    cases = {
        "elastic, P=0.5 P$_y$, tau=10 ms": ImpactCase(
            pressure_amplitude=0.5 * p_y, pulse_duration=0.010, n_elements=16
        ),
        "rate-dependent, P=0.8 P$_y$, tau=30 ms": ImpactCase(
            pressure_amplitude=0.8 * p_y, pulse_duration=0.030, n_elements=16
        ),
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=False)
    for ax, (label, case) in zip(axes, cases.items()):
        result = BeamHingeFEM(case).daf()
        hist = result["history"]
        t_ms = hist["t"] * 1e3
        for idx, lab in enumerate(hist["hinge_labels"]):
            ax.plot(t_ms, hist["stress"][idx] / 1e6, lw=1.4, label=f"{lab}")
        shape = hist["pressure"] / max(case.pressure_amplitude, 1e-12)
        ax.plot(t_ms, shape * result["sigma_static_global"] / 1e6, "k:", lw=1.2,
                label="pulse shape x static stress")
        ax.axhline(result["sigma_static_global"] / 1e6, color="tab:gray", ls="--", lw=1,
                   label="static stress")
        ax.set_ylabel("section stress [MPa]")
        ax.set_title(
            f"{label}:  DAF = {result['daf_global']:.3f}, "
            f"max $\\dot\\epsilon_p$ = {result['max_plastic_strain_rate']:.1f} s$^{{-1}}$",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("time [ms]")
    fig.suptitle("Plate-strip FE response to a half-sine ice pressure pulse", fontsize=10)
    fig.tight_layout()
    path = OUTDIR / "fig3_stress_history.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig4_daf_surface(results: dict) -> pathlib.Path:
    surface = results["surface"]
    durations = np.array(surface["durations"]) * 1e3
    ratios = np.array(surface["load_ratios"])
    daf = np.array(surface["daf"])
    rate_factor = np.array(surface["daf_rate_factor"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, data, title, cmap in (
        (axes[0], daf, "DAF $=\\sigma_{dyn,max}/\\sigma_{static}$", "viridis"),
        (axes[1], rate_factor, "rate factor DAF(CS)/DAF(rate-independent)", "magma"),
    ):
        cs = ax.contourf(durations, ratios, data, levels=12, cmap=cmap)
        ax.contour(durations, ratios, data, levels=8, colors="w", linewidths=0.5, alpha=0.6)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel(r"pulse duration $\tau$ [ms]")
        ax.set_ylabel(r"pressure amplitude $P/P_y$")
        ax.set_title(title, fontsize=9)
    axes[0].plot(surface["first_period"] * 1e3, 1.0, "r*", ms=12)
    axes[0].annotate(r"$\tau = T_1$", (surface["first_period"] * 1e3, 1.02), color="r", fontsize=8)
    fig.suptitle(
        f"DAF surface, h = {surface['thickness']*1e3:.0f} mm plate strip "
        f"($P_y$ = {surface['yield_pressure']/1e3:.1f} kPa, "
        f"collapse = {surface['collapse_pressure']/1e3:.1f} kPa)",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUTDIR / "fig4_daf_surface.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig5_six_cases(results: dict, fig4_path: pathlib.Path) -> pathlib.Path:
    cases = results["b_daf_six_cases"]["cases"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for thickness, marker, colour in ((0.008, "o", "tab:blue"), (0.016, "s", "tab:orange")):
        sel = [c for c in cases if abs(c["thickness"] - thickness) < 1e-12]
        taus = [c["pulse_duration"] * 1e3 for c in sel]
        fem = [c["daf_fem"] for c in sel]
        ref = [c["daf_modal_reference"] for c in sel]
        ax.plot(taus, fem, marker + "-", color=colour, label=f"FE, h={thickness*1e3:.0f} mm")
        ax.plot(taus, ref, "x", ms=9, color=colour, mfc="none",
                label=f"modal reference, h={thickness*1e3:.0f} mm")
    ax.set_xscale("log")
    ax.set_xlabel(r"pulse duration $\tau$ [ms]")
    ax.set_ylabel("DAF")
    ax.set_title(
        "Six-case DAF validation ($P=0.5P_y$, max error "
        f"{results['b_daf_six_cases']['max_relative_error']*100:.2f} % vs 15 % threshold)",
        fontsize=9,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "fig5_six_cases.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text())
    paths = [
        fig1_shock_spectrum(results),
        fig2_cs_rate_law(results),
        fig3_stress_history(),
        fig4_daf_surface(results),
    ]
    paths.append(fig5_six_cases(results, paths[-1]))
    for path in paths:
        print("wrote", path.relative_to(_HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())