"""Benchmark driver for the simplified engineering ice-ridge consolidation model.

Generates figures and results for the paper-quality report on campaign node
P46W3-N05: ice ridge consolidation thermodynamics.

Outputs:
  runs/ridge_thermo/results.json          all machine-read numbers
  runs/ridge_thermo/figures/*.png         the report figures

Usage:
    python benchmarks/ridge_thermo/run_engineering_benchmark.py \
        [--out runs/ridge_thermo]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.ice_mechanics.ridge_consolidation_engineering import (
    EngineeringConsolidationParams,
    arrhenius_time_constant,
    consolidation_degree_exponential,
    heat_conduction_residual_1d,
    resistance_linear_coupling,
    run_engineering_verification,
    strength_power_law,
    SIGMA_ICE_REF,
    T_MELT_SEA,
    R_ICE_REF,
)

DAY = 86_400.0


def _env() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        rev = "unknown"
    return {
        "git_revision": rev,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "deterministic": True,
        "random_seed": None,
    }


def _fig_consolidation_degree(out_dir: pathlib.Path) -> None:
    """Figure 1: Consolidation degree C(t) for various temperatures."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    times = np.linspace(0, 90, 300)  # days
    temperatures = [-10.0, -20.0, -30.0, -40.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    
    for T, color in zip(temperatures, colors):
        C_values = [consolidation_degree_exponential(t * DAY, T) for t in times]
        ax.plot(times, C_values, color=color, lw=2, label=f"T = {T}°C")
    
    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Consolidation degree C [-]", fontsize=12)
    ax.set_title("Exponential consolidation law: C(t) = 1 - exp(-t/τ(T))", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_consolidation_degree.png", dpi=160)
    plt.close(fig)


def _fig_arrhenius_time_constant(out_dir: pathlib.Path) -> None:
    """Figure 2: Arrhenius time constant τ(T)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    temperatures = np.linspace(-40, -2, 100)
    tau_values = [arrhenius_time_constant(T) / DAY for T in temperatures]
    
    ax.plot(temperatures, tau_values, color="#d62728", lw=2.5)
    ax.set_xlabel("Temperature [°C]", fontsize=12)
    ax.set_ylabel("Time constant τ [days]", fontsize=12)
    ax.set_title("Arrhenius time constant: τ(T) = τ₀·exp(-Eₐ/R·(1/T - 1/T_ref))", fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(-40, -2)
    ax.set_ylim(0, max(tau_values) * 1.1)
    
    # Annotate reference point
    tau_ref = arrhenius_time_constant(-10.0) / DAY
    ax.plot(-10.0, tau_ref, 'ko', markersize=8)
    ax.annotate(f"T_ref = -10°C\nτ = {tau_ref:.1f} days", 
                xy=(-10.0, tau_ref), xytext=(-25, tau_ref + 5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_arrhenius_time_constant.png", dpi=160)
    plt.close(fig)


def _fig_strength_power_law(out_dir: pathlib.Path) -> None:
    """Figure 3: Strength-consolidation power-law relation."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    C_values = np.linspace(0, 1, 100)
    m_values = [1.5, 1.75, 2.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    
    for m, color in zip(m_values, colors):
        sigma_values = [strength_power_law(C, m=m) / 1e6 for C in C_values]
        ax.plot(C_values, sigma_values, color=color, lw=2, label=f"m = {m}")
    
    ax.set_xlabel("Consolidation degree C [-]", fontsize=12)
    ax.set_ylabel("Compressive strength σ_c [MPa]", fontsize=12)
    ax.set_title("Power-law strength: σ_c = σ₀·C^m", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, SIGMA_ICE_REF / 1e6 * 1.1)
    
    # Annotate reference point
    ax.plot(1.0, SIGMA_ICE_REF / 1e6, 'ko', markersize=8)
    ax.annotate(f"σ₀ = {SIGMA_ICE_REF/1e6:.1f} MPa", 
                xy=(1.0, SIGMA_ICE_REF / 1e6), xytext=(0.7, SIGMA_ICE_REF / 1e6 * 0.9),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_strength_power_law.png", dpi=160)
    plt.close(fig)


def _fig_resistance_coupling(out_dir: pathlib.Path) -> None:
    """Figure 4: Resistance-consolidation linear coupling."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    C_values = np.linspace(0, 1, 100)
    k_values = [0.5, 0.7, 1.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    
    for k, color in zip(k_values, colors):
        R_values = [resistance_linear_coupling(C, k=k) for C in C_values]
        ax.plot(C_values, R_values, color=color, lw=2, label=f"k = {k}")
    
    ax.set_xlabel("Consolidation degree C [-]", fontsize=12)
    ax.set_ylabel("Ridge resistance R_ice [kN]", fontsize=12)
    ax.set_title("Linear resistance coupling: R_ice = R₀·(1 + k·C)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, R_ICE_REF * 2.2)
    
    # Annotate baseline
    ax.plot(0.0, R_ICE_REF, 'ko', markersize=8)
    ax.annotate(f"R₀ = {R_ICE_REF:.0f} kN (unconsolidated)", 
                xy=(0.0, R_ICE_REF), xytext=(0.2, R_ICE_REF * 1.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_resistance_coupling.png", dpi=160)
    plt.close(fig)


def _fig_coupled_evolution(out_dir: pathlib.Path) -> None:
    """Figure 5: Coupled evolution of C, σ_c, and R_ice over time."""
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    times = np.linspace(0, 90, 300)  # days
    T = -20.0
    m = 1.75
    k = 0.7
    
    C_values = [consolidation_degree_exponential(t * DAY, T) for t in times]
    sigma_values = [strength_power_law(C, m=m) / 1e6 for C in C_values]
    R_values = [resistance_linear_coupling(C, k=k) for C in C_values]
    
    # Panel 1: Consolidation degree
    axes[0].plot(times, C_values, color="#1f77b4", lw=2)
    axes[0].set_ylabel("C [-]", fontsize=12)
    axes[0].set_title("Coupled evolution at T = -20°C", fontsize=13)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    # Panel 2: Strength
    axes[1].plot(times, sigma_values, color="#d62728", lw=2)
    axes[1].set_ylabel("σ_c [MPa]", fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, SIGMA_ICE_REF / 1e6 * 1.1)
    
    # Panel 3: Resistance
    axes[2].plot(times, R_values, color="#2ca02c", lw=2)
    axes[2].set_xlabel("Time [days]", fontsize=12)
    axes[2].set_ylabel("R_ice [kN]", fontsize=12)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, R_ICE_REF * 2.2)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_coupled_evolution.png", dpi=160)
    plt.close(fig)


def _fig_heat_conduction_verification(out_dir: pathlib.Path) -> None:
    """Figure 6: Heat conduction residual verification."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    
    n_points_list = [10, 20, 50, 100, 200, 500, 1000]
    residuals = [heat_conduction_residual_1d(-20.0, -5.0, 10.0, 2.0, n) for n in n_points_list]
    
    ax.semilogy(n_points_list, residuals, 'o-', color="#9467bd", lw=2, markersize=8)
    ax.axhline(0.01, color='r', linestyle='--', lw=1.5, label="Threshold: 1%")
    ax.set_xlabel("Number of grid points", fontsize=12)
    ax.set_ylabel("Normalized residual |d²T/dx²| / (ΔT/L²)", fontsize=12)
    ax.set_title("1D steady-state heat conduction verification", fontsize=13)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1100)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_heat_conduction_verification.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Engineering consolidation benchmark")
    parser.add_argument("--out", type=str, default="runs/ridge_thermo",
                        help="Output directory")
    args = parser.parse_args()
    
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {out_dir}")
    print(f"Generating figures...")
    
    # Generate figures
    _fig_consolidation_degree(fig_dir)
    _fig_arrhenius_time_constant(fig_dir)
    _fig_strength_power_law(fig_dir)
    _fig_resistance_coupling(fig_dir)
    _fig_coupled_evolution(fig_dir)
    _fig_heat_conduction_verification(fig_dir)
    
    # Run verification suite
    print("Running verification suite...")
    results = run_engineering_verification()
    results["environment"] = _env()
    results["module"] = "src/tensorphysics/ice_mechanics/ridge_consolidation_engineering.py"
    results["tests"] = "tests/test_ridge_consolidation_engineering.py"
    
    # Write results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to {results_path}")
    print(f"Criteria passed: {results['overall_passed']}")
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
