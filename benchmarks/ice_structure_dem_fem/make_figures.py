"""Figures for the p380w2-n06 report, generated from the acceptance ledger/npz.

Numbers and histories are read back from ``artifacts/ice_structure_dem_fem``
(never re-typed), so the figures cannot drift from the JSON ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "ice_structure_dem_fem"


def main() -> int:
    data = json.loads((ART / "verification.json").read_text())
    cond = [(c["h_m"], c["speed_ms"]) for c in data["conditions"]]
    spec = np.load(ART / "dem_spectra.npz")
    rows = data["criteria"]["a_dem_peak_vs_lindqvist"]["rows"]

    fig, axes = plt.subplots(len(cond), 1, figsize=(9, 1.9 * len(cond)))
    for ax, ((h, v), r) in zip(axes, zip(cond, rows)):
        ax.plot(spec[f"t_{h}_{v}"], spec[f"f_{h}_{v}"] / 1e3, lw=0.9, color="tab:blue",
                label="DEM $F_x(t)$")
        ax.axhline(r["lindqvist_peak_per_metre_N_per_m"] / 1e3, color="tab:red", ls="--",
                   label="Lindqvist peak/m")
        ax.axhline(r["lindqvist_mean_per_metre_N_per_m"] / 1e3, color="tab:green", ls=":",
                   label="Lindqvist mean/m")
        ax.axhline(r["analytic_floating_beam_break_N_per_m"] / 1e3, color="0.4", lw=0.8,
                   label="floating beam")
        ax.set_ylabel("kN/m", fontsize=8)
        ax.set_title(f"h = {h} m, v = {v} m/s", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, loc="upper right", ncol=4)
    fig.suptitle("DEM ice-load spectra (2-D slice, unit waterline width)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ART / "fig_load_spectra.png", dpi=130)
    plt.close(fig)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.27, [r["dem_peak_N_per_m"] / 1e3 for r in rows], 0.27,
           label="DEM peak", color="tab:blue")
    ax.bar(x, [r["lindqvist_peak_per_metre_N_per_m"] / 1e3 for r in rows], 0.27,
           label="Lindqvist $R_c+R_b$ / m", color="tab:red")
    ax.bar(x + 0.27, [r["analytic_floating_beam_break_N_per_m"] / 1e3 for r in rows], 0.27,
           label="floating-beam analytic", color="0.5")
    ax.set_xticks(x, [f"h={r['h_m']}\nv={r['speed_ms']}" for r in rows], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("kN/m")
    ax.set_title("Peak ice load: DEM vs Lindqvist vs analytic breaking force")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ART / "fig_peak_comparison.png", dpi=130)
    plt.close(fig)

    hist = np.load(ART / "coupled_history_h1_v3.npz")
    t, p = hist["t"], hist["p"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(t, p / 1e3, color="tab:blue")
    a1.set_ylabel("patch pressure [kPa]")
    a1.set_title("DEM -> FEM transfer (impulse conserving, sub-cycled), h = 1 m, v = 3 m/s")
    a1.grid(alpha=0.3)
    a2.plot(t, hist["sigma"] / 1e6, color="tab:red", label=r"max $|\sigma_x|$")
    a2.plot(t, hist["vm"] / 1e6, color="tab:orange", label="max von Mises")
    a2.axhline(data["ship_panel"]["material"]["sigma_y_Pa"] / 1e6, color="k", ls="--",
               lw=0.8, label=r"$\sigma_y$")
    a2.set_ylabel("stress [MPa]")
    a2.set_xlabel("time [s]")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "fig_coupled_response.png", dpi=130)
    plt.close(fig)

    from tensorphysics.ice_structure_dem_fem.fem_plate import (
        PlateConfig, navier_reference_max_stress, solve_plate_history)
    panel = PlateConfig(a=0.6, b=2.4, t=0.020, nx=12, ny=48)
    qs = np.linspace(0.01e6, 0.09e6, 5)
    fem, nav = [], []
    for q in qs:
        r = solve_plate_history(panel, [0.0, q], patch=(0.0, 0.6, 0.0, 2.4))
        fem.append(float(r.max_abs_sigma_x[-1]))
        nav.append(navier_reference_max_stress(panel, q, patch=(0.0, 0.6, 0.0, 2.4),
                                               n_terms=39)["max_surface_stress_Pa"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(np.array(nav) / 1e6, np.array(fem) / 1e6, "o-", label="FEM (Mindlin, layered J2)")
    lim = [0, max(nav) / 1e6 * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8, label="Navier 1:1")
    ax.set_xlabel("Navier surface stress [MPa]")
    ax.set_ylabel(r"FEM max $|\sigma_x|$ [MPa]")
    ax.set_title("Elastic-limit verification of the elasto-plastic plate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "fig_elastic_verification.png", dpi=130)
    plt.close(fig)

    rows_e = data["criteria"]["e_mesh_convergence"]["rows"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.semilogx([r["n_elements"] for r in rows_e],
                [r["peak_abs_sigma_x_Pa"] / 1e6 for r in rows_e], "s-")
    ax.set_xlabel("number of elements")
    ax.set_ylabel(r"peak $|\sigma_x|$ [MPa]")
    ax.set_title("Plate mesh convergence under the transferred ice load")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(ART / "fig_mesh_convergence.png", dpi=130)
    plt.close(fig)
    print("figures:", sorted(p.name for p in ART.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())