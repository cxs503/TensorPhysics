#!/usr/bin/env python3
"""P210W2-N02: Ice Resistance Maturity — unified runner.

Runs cross-implementation attribution + MCMC calibration, generates figures,
writes paper-quality report.

Usage::

    python benchmarks/ice_resistance_maturity/run_maturity.py

Artifacts:
* ``artifacts/ice_resistance_maturity/maturity_report.json``
* ``artifacts/ice_resistance_maturity/summary.txt``
* ``artifacts/ice_resistance_maturity/figures/*.png``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.ice.ice_resistance_maturity import (
    run_maturity_pipeline,
    write_maturity_artifacts,
)
from tensorphysics.ice.cross_impl_attribution import RefShip, decompose_five
from tensorphysics.ice.resistance import level_ice_resistance, ShipGeometry


REPO = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO / "artifacts" / "ice_resistance_maturity"
FIGDIR = ARTIFACTS / "figures"


def _shrink(p: Path) -> Path:
    """Try to optipng; ignore failure."""
    import subprocess
    try:
        subprocess.run(["optipng", "-quiet", str(p)], timeout=10)
    except Exception:
        pass
    return p


# ---------------------------------------------------------------------------
# Figure 1: Cross-implementation divergence bar chart
# ---------------------------------------------------------------------------


def fig1_divergence_attribution(result) -> Path:
    """Bar chart showing component-wise divergence for each implementation."""
    impls = ["TP_level", "IA_lindqvist", "Lit_original", "Lit_corrected"]
    components = ["R_crush_N", "R_bend_N", "R_momentum_N"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked bar for each implementation
    ax = axes[0]
    x = np.arange(len(impls))
    width = 0.6
    bottoms = np.zeros(len(impls))

    colors = {"R_crush_N": "#d62728", "R_bend_N": "#2ca02c", "R_momentum_N": "#1f77b4"}
    for comp in components:
        vals = [result.decompositions[impl][comp] / 1000.0 for impl in impls]
        ax.bar(x, vals, width, bottom=bottoms, label=comp.replace("_N", "").replace("_", " "), color=colors[comp])
        bottoms += np.array(vals)

    ax.set_xlabel("Implementation")
    ax.set_ylabel("Resistance [kN]")
    ax.set_title("Component-wise Decomposition by Implementation")
    ax.set_xticks(x)
    ax.set_xticklabels(impls, rotation=15, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Right: divergence attribution (fraction of total diff)
    ax = axes[1]
    div_data = result.divergence_ia_vs_tp
    fracs = [div_data["component_fractions"][c] for c in components]
    labels = [c.replace("_N", "").replace("_", " ") for c in components]
    bars = ax.barh(labels, fracs, color=[colors[c] for c in components])
    ax.set_xlabel("Fraction of Total Divergence")
    ax.set_title("Divergence Attribution (IA vs TP)")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{frac:.1%}", va="center", fontsize=8)

    fig.suptitle("P210W2-N02: Cross-Implementation Divergence Attribution", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIGDIR / "fig1_divergence_attribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return _shrink(out)


# ---------------------------------------------------------------------------
# Figure 2: Sensitivity matrix heatmap
# ---------------------------------------------------------------------------


def fig2_sensitivity_matrix(result) -> Path:
    """Heatmap of the component sensitivity matrix."""
    sens = result.sensitivity_condition
    matrix = sens["matrix"]
    params = list(matrix.keys())
    components = list(list(matrix.values())[0].keys())

    S = np.zeros((len(components), len(params)))
    for j, p in enumerate(params):
        for i, c in enumerate(components):
            S[i, j] = matrix[p][c]

    # Normalize for visualization
    S_norm = S / np.max(np.abs(S)) if np.max(np.abs(S)) > 0 else S

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(S_norm, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels([p.replace("_", " ") for p in params], fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels([c.replace("_N", "").replace("_", " ") for c in components], fontsize=8)

    # Add text annotations
    for i in range(len(components)):
        for j in range(len(params)):
            val = S[i, j]
            text = ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=7,
                          color="white" if abs(S_norm[i, j]) > 0.5 else "black")

    ax.set_title(f"Sensitivity Matrix ∂R/∂p (κ={sens['condition_number']:.1f})")
    fig.colorbar(im, ax=ax, label="Normalized sensitivity")
    fig.tight_layout()
    out = FIGDIR / "fig2_sensitivity_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return _shrink(out)


# ---------------------------------------------------------------------------
# Figure 3: MCMC posterior distribution
# ---------------------------------------------------------------------------


def fig3_mcmc_posterior(result) -> Path:
    """Posterior distribution of μ with diagnostics."""
    cal = result.calibration_result
    samples = cal.chain_samples

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left: trace plot
    ax = axes[0]
    ax.plot(samples[:1000], linewidth=0.5, alpha=0.7)
    ax.axhline(cal.mu_posterior_mean, color="r", linestyle="--", label=f"mean={cal.mu_posterior_mean:.4f}")
    ax.axhline(cal.mu_default, color="g", linestyle=":", label=f"default={cal.mu_default:.4f}")
    if not np.isnan(cal.mu_true):
        ax.axhline(cal.mu_true, color="b", linestyle="-", alpha=0.5, label=f"true={cal.mu_true:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("μ")
    ax.set_title("MCMC Trace")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Middle: posterior histogram
    ax = axes[1]
    ax.hist(samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(cal.mu_95_ci_lower, color="r", linestyle="--", label=f"95% CI")
    ax.axvline(cal.mu_95_ci_upper, color="r", linestyle="--")
    ax.axvline(cal.mu_default, color="g", linestyle=":", label="Lindqvist default")
    ax.set_xlabel("μ")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior (CI width={cal.mu_95_ci_upper - cal.mu_95_ci_lower:.4f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Right: convergence diagnostics
    ax = axes[2]
    diagnostics = {
        "R̂": f"{cal.gelman_rubin_rhat:.4f}",
        "ESS": f"{cal.ess}",
        "Acc. rate": f"{cal.acceptance_rate:.3f}",
        "Norm. err": f"{cal.normalization_error:.6f}",
    }
    y_pos = np.arange(len(diagnostics))
    for i, (k, v) in enumerate(diagnostics.items()):
        ax.text(0.1, 0.85 - i * 0.2, f"{k}: {v}", fontsize=10, transform=ax.transAxes,
                fontfamily="monospace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Convergence Diagnostics")
    ax.axis("off")

    fig.suptitle("P210W2-N02: Bayesian MCMC Calibration of μ", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIGDIR / "fig3_mcmc_posterior.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return _shrink(out)


# ---------------------------------------------------------------------------
# Figure 4: Resistance vs speed with calibrated μ
# ---------------------------------------------------------------------------


def fig4_resistance_curves(result) -> Path:
    """Resistance vs speed for different implementations and calibrated μ."""
    ship = RefShip()
    ship_geom = ShipGeometry(L_pp=ship.L_pp, B=ship.B, T=ship.T, bow_angle_deg=ship.bow_angle_deg)

    h = 1.0
    V_grid = np.linspace(0.5, 6.0, 50)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: R vs V for different implementations
    ax = axes[0]
    impls = ["TP_level", "IA_lindqvist", "Lit_original"]
    colors = {"TP_level": "#1f77b4", "IA_lindqvist": "#d62728", "Lit_original": "#2ca02c"}
    for impl in impls:
        R_vals = [decompose_five(impl, h, v, ship)["R_total_N"] / 1000.0 for v in V_grid]
        ax.plot(V_grid, R_vals, label=impl, color=colors[impl], linewidth=1.5)
    ax.set_xlabel("Speed V [m/s]")
    ax.set_ylabel("R_level [kN]")
    ax.set_title(f"Ice Resistance vs Speed (h={h} m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: R vs V with default vs calibrated μ
    ax = axes[1]
    cal = result.calibration_result
    R_default = [level_ice_resistance(h=h, V=v, ship=ship_geom, mu=cal.mu_default)["R_total_N"] / 1000.0 for v in V_grid]
    R_calibrated = [level_ice_resistance(h=h, V=v, ship=ship_geom, mu=cal.mu_posterior_mean)["R_total_N"] / 1000.0 for v in V_grid]
    ax.plot(V_grid, R_default, "g--", label=f"μ={cal.mu_default:.2f} (Lindqvist)", linewidth=1.5)
    ax.plot(V_grid, R_calibrated, "b-", label=f"μ={cal.mu_posterior_mean:.4f} (calibrated)", linewidth=1.5)
    ax.fill_between(V_grid,
                    [level_ice_resistance(h=h, V=v, ship=ship_geom, mu=cal.mu_95_ci_lower)["R_total_N"] / 1000.0 for v in V_grid],
                    [level_ice_resistance(h=h, V=v, ship=ship_geom, mu=cal.mu_95_ci_upper)["R_total_N"] / 1000.0 for v in V_grid],
                    alpha=0.2, color="blue", label="95% CI")
    ax.set_xlabel("Speed V [m/s]")
    ax.set_ylabel("R_level [kN]")
    ax.set_title(f"Effect of μ Calibration (h={h} m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("P210W2-N02: Ice Resistance Curves", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIGDIR / "fig4_resistance_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return _shrink(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print("Running maturity pipeline...")
    result = run_maturity_pipeline(
        h=1.0, V=3.0,
        n_mcmc_steps=30000,
        n_mcmc_chains=4,
        seed=42,
    )

    print("Writing artifacts...")
    paths = write_maturity_artifacts(result)

    print("Generating figures...")
    figs = [
        fig1_divergence_attribution(result),
        fig2_sensitivity_matrix(result),
        fig3_mcmc_posterior(result),
        fig4_resistance_curves(result),
    ]

    # Print summary
    d = result.as_dict()
    print("\n" + "=" * 70)
    print("P210W2-N02: Ice Resistance Maturity Assessment")
    print("=" * 70)
    print(f"Generated: {result.generated_utc}")
    print(f"SHA-256: {result.determinism_sha256}")
    print()
    print("Acceptance Criteria:")
    for k, v in d["acceptance_criteria"].items():
        status = "PASS" if v["passed"] else "FAIL"
        val = v.get("value", "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        print(f"  {k}: {status} (value={val})")
    print()
    print(f"All criteria passed: {d['all_criteria_passed']}")
    print()
    print("Artifacts:")
    print(f"  Report: {paths['report_json']}")
    print(f"  Summary: {paths['summary_txt']}")
    for f in figs:
        print(f"  Figure: {f}")
    print("=" * 70)

    return 0 if d["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
