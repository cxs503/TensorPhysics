#!/usr/bin/env python3
"""Generate publication figures for P295W1-N08: ice-induced frequency lock-in.

Produces:
1. Parameter space map (V vs h) showing lock-in regions
2. Lock-in bandwidth vs ice thickness
3. Amplification factor map
4. Kujala benchmark comparison
5. Ice breaking frequency vs speed for various thicknesses
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ARTIFACTS = WORKTREE / "artifacts" / "p295w1_n08"


def load_results():
    """Load sweep and benchmark results."""
    with open(ARTIFACTS / "sweep_results.json") as f:
        sweep = json.load(f)
    with open(ARTIFACTS / "benchmark_results.json") as f:
        benchmark = json.load(f)
    return sweep, benchmark


def figure_parameter_space_map(sweep, path: Path):
    """Lock-in parameter space map: V vs h."""
    V = np.array(sweep["V_arr"])
    h = np.array(sweep["h_arr"])
    lock_in = np.array(sweep["lock_in_map"])
    amplification = np.array(sweep["amplification_map"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Lock-in map (binary)
    Vg, hg = np.meshgrid(V, h)
    ax = axes[0]
    c = ax.pcolormesh(Vg, hg, lock_in, cmap="RdYlGn_r", shading="auto", vmin=0, vmax=1)
    ax.set_xlabel("Ice speed V [m/s]")
    ax.set_ylabel("Ice thickness h [m]")
    ax.set_title("Lock-in region (dark = locked-in)")
    fig.colorbar(c, ax=ax, label="Lock-in (1=yes, 0=no)")
    
    # Add contour at f_ice = f_n
    f_ice = np.array(sweep["f_ice_map"])
    f_n = sweep["metadata"]["f_n_hz"]
    cs = ax.contour(Vg, hg, f_ice, levels=[f_n], colors="k", linewidths=2, linestyles="--")
    ax.clabel(cs, fmt={f_n: f"$f_{{ice}} = f_n = {f_n}$ Hz"}, fontsize=9)
    
    # Right: Amplification map (log scale)
    ax = axes[1]
    norm = mcolors.LogNorm(vmin=1, vmax=amplification.max())
    c = ax.pcolormesh(Vg, hg, amplification, cmap="viridis", norm=norm, shading="auto")
    ax.set_xlabel("Ice speed V [m/s]")
    ax.set_ylabel("Ice thickness h [m]")
    ax.set_title("Dynamic amplification factor D")
    fig.colorbar(c, ax=ax, label="D = X_dyn / (F₀/k)")
    
    # Overlay lock-in boundary
    ax.contour(Vg, hg, lock_in, levels=[0.5], colors="red", linewidths=2)
    
    fig.suptitle(f"Ice-induced frequency lock-in parameter space (f_n = {f_n} Hz, ζ = {sweep['metadata']['zeta']})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return {
        "name": "parameter_space_map",
        "caption": f"Left: Lock-in regions in the (V, h) parameter space. The dashed line marks f_ice = f_n = {f_n} Hz. "
                   f"Right: Dynamic amplification factor D. The red contour marks the lock-in boundary (|f_ice - f_n|/f_n < 5%). "
                   f"Lock-in fraction: {sweep['lock_in_fraction']:.1%} of the parameter space.",
    }


def figure_bandwidth_vs_thickness(sweep, path: Path):
    """Lock-in bandwidth ΔV vs ice thickness h."""
    bw_data = sweep["bandwidth_data"]
    h_vals = [d["h_m"] for d in bw_data]
    bw_vals = [d["bandwidth_m_s"] for d in bw_data]
    V_min_vals = [d["V_min_m_s"] for d in bw_data]
    V_max_vals = [d["V_max_m_s"] for d in bw_data]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Bandwidth as filled region
    ax.fill_between(h_vals, V_min_vals, V_max_vals, alpha=0.3, color="tab:blue", label="Lock-in speed range")
    ax.plot(h_vals, V_min_vals, "b--", lw=1.5, label="$V_{min}$")
    ax.plot(h_vals, V_max_vals, "b--", lw=1.5, label="$V_{max}$")
    ax.plot(h_vals, bw_vals, "r-o", lw=2, ms=4, label="Bandwidth $\\Delta V$")
    
    ax.set_xlabel("Ice thickness h [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Lock-in bandwidth vs ice thickness")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(sweep["metadata"]["h_range_m"])
    
    # Add text annotation
    if bw_vals:
        max_bw = max(bw_vals)
        max_idx = bw_vals.index(max_bw)
        ax.annotate(f"max $\\Delta V$ = {max_bw:.2f} m/s\nat h = {h_vals[max_idx]:.2f} m",
                    xy=(h_vals[max_idx], max_bw),
                    xytext=(0.7, 0.8), textcoords="axes fraction",
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray"))
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return {
        "name": "bandwidth_vs_thickness",
        "caption": "Lock-in speed bandwidth ΔV as function of ice thickness h. The blue shaded region shows the "
                   "speed range where lock-in occurs. Thicker ice requires higher speeds for lock-in because the "
                   "characteristic ice length scale increases with h.",
    }


def figure_ice_frequency_vs_speed(sweep, path: Path):
    """Ice breaking frequency f_ice vs speed V for various thicknesses."""
    V = np.array(sweep["V_arr"])
    h_arr = np.array(sweep["h_arr"])
    f_ice_map = np.array(sweep["f_ice_map"])
    f_n = sweep["metadata"]["f_n_hz"]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Select a few representative thicknesses
    h_indices = [0, len(h_arr)//4, len(h_arr)//2, 3*len(h_arr)//4, -1]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(h_indices)))
    
    for idx, color in zip(h_indices, colors):
        h = h_arr[idx]
        f_ice = f_ice_map[idx, :]
        ax.plot(V, f_ice, "-", color=color, lw=2, label=f"h = {h:.2f} m")
    
    # Horizontal line at f_n
    ax.axhline(f_n, color="red", ls="--", lw=2, label=f"$f_n$ = {f_n} Hz")
    
    # Lock-in band
    ax.axhspan(f_n * 0.95, f_n * 1.05, color="red", alpha=0.1, label="Lock-in band (±5%)")
    
    ax.set_xlabel("Ice speed V [m/s]")
    ax.set_ylabel("Ice breaking frequency $f_{ice}$ [Hz]")
    ax.set_title("Ice breaking frequency vs speed for various ice thicknesses")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(sweep["metadata"]["V_range_m_s"])
    ax.set_ylim(0, max(3.0, f_n * 2))
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return {
        "name": "ice_frequency_vs_speed",
        "caption": "Ice breaking frequency f_ice = V/L_c as function of ice speed V for various ice thicknesses h. "
                   "The red dashed line and shaded band show the structural natural frequency f_n and the ±5% lock-in band. "
                   "Lock-in occurs where f_ice curves cross the band.",
    }


def figure_kujala_benchmark(benchmark, path: Path):
    """Kujala et al. 2015 benchmark comparison."""
    flex = benchmark["flexural"]
    crush = benchmark["crushing"]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: Speed range comparison
    ax = axes[0]
    h_pos = 0.5
    # Observed range
    ax.barh(h_pos, flex["observed_V_range_m_s"][1] - flex["observed_V_range_m_s"][0],
            left=flex["observed_V_range_m_s"][0], height=0.15, color="tab:green", alpha=0.7,
            label="Kujala 2015 observed")
    # Predicted (flexural)
    ax.barh(h_pos + 0.2, flex["predicted_V_range_m_s"][1] - flex["predicted_V_range_m_s"][0],
            left=flex["predicted_V_range_m_s"][0], height=0.15, color="tab:blue", alpha=0.7,
            label="Model (flexural)")
    # Predicted (crushing)
    ax.barh(h_pos - 0.2, crush["predicted_V_range_m_s"][1] - crush["predicted_V_range_m_s"][0],
            left=crush["predicted_V_range_m_s"][0], height=0.15, color="tab:orange", alpha=0.7,
            label="Model (crushing)")
    
    ax.set_yticks([h_pos - 0.2, h_pos, h_pos + 0.2])
    ax.set_yticklabels(["Crushing", "Observed", "Flexural"])
    ax.set_xlabel("Ice speed V [m/s]")
    ax.set_title(f"Lock-in speed range at h = {flex['h_field_m']} m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    
    # Right: Error summary
    ax = axes[1]
    metrics = ["V_min\nerror", "V_max\nerror", "Frequency\nerror", "Coverage"]
    values = [
        flex["V_min_relative_error"] * 100,
        flex["V_max_relative_error"] * 100,
        flex["frequency_relative_error"] * 100,
        flex["lock_in_coverage"] * 100,
    ]
    colors = ["tab:red" if v > 15 else "tab:green" for v in values[:3]]
    colors.append("tab:green" if values[3] >= 80 else "tab:red")
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.axhline(15, color="red", ls="--", lw=1, label="15% threshold")
    ax.axhline(80, color="green", ls="--", lw=1, label="80% coverage target")
    ax.set_ylabel("Relative error [%] / Coverage [%]")
    ax.set_title("Benchmark metrics (flexural mode)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    
    fig.suptitle("Benchmark against Kujala et al. 2015: ice-induced vibrations on MV Arctic")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return {
        "name": "kujala_benchmark",
        "caption": f"Left: Comparison of observed (Kujala 2015) and predicted lock-in speed ranges at h = {flex['h_field_m']} m. "
                   f"Right: Benchmark metrics. Frequency error = {flex['frequency_relative_error']:.1%} "
                   f"(threshold ≤15%); Coverage = {flex['lock_in_coverage']:.1%} (target ≥80%).",
    }


def figure_amplification_along_sweep(sweep, path: Path):
    """Amplification factor along a constant-thickness cut."""
    V = np.array(sweep["V_arr"])
    h_arr = np.array(sweep["h_arr"])
    amplification = np.array(sweep["amplification_map"])
    f_n = sweep["metadata"]["f_n_hz"]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Select a few thicknesses
    h_indices = [0, len(h_arr)//3, 2*len(h_arr)//3, -1]
    colors = plt.cm.Set2(np.linspace(0, 1, len(h_indices)))
    
    for idx, color in zip(h_indices, colors):
        h = h_arr[idx]
        D = amplification[idx, :]
        ax.semilogy(V, D, "-", color=color, lw=2, label=f"h = {h:.2f} m")
    
    # Threshold line
    ax.axhline(3.0, color="red", ls="--", lw=1.5, label="D = 3 threshold")
    
    ax.set_xlabel("Ice speed V [m/s]")
    ax.set_ylabel("Dynamic amplification D")
    ax.set_title(f"Amplification factor vs ice speed (f_n = {f_n} Hz)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(sweep["metadata"]["V_range_m_s"])
    ax.set_ylim(0.5, max(100, amplification.max()))
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return {
        "name": "amplification_vs_speed",
        "caption": "Dynamic amplification factor D vs ice speed V for various ice thicknesses. "
                   "Peaks occur where f_ice ≈ f_n (lock-in). The red dashed line marks D = 3, "
                   "the threshold for significant amplification.",
    }


def main():
    """Generate all figures."""
    sweep, benchmark = load_results()
    
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    figures_dir = ARTIFACTS / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    figures = []
    
    # 1. Parameter space map
    fig_info = figure_parameter_space_map(sweep, figures_dir / "parameter_space_map.png")
    figures.append(fig_info)
    print(f"  Generated: {fig_info['name']}")
    
    # 2. Bandwidth vs thickness
    fig_info = figure_bandwidth_vs_thickness(sweep, figures_dir / "bandwidth_vs_thickness.png")
    figures.append(fig_info)
    print(f"  Generated: {fig_info['name']}")
    
    # 3. Ice frequency vs speed
    fig_info = figure_ice_frequency_vs_speed(sweep, figures_dir / "ice_frequency_vs_speed.png")
    figures.append(fig_info)
    print(f"  Generated: {fig_info['name']}")
    
    # 4. Kujala benchmark
    fig_info = figure_kujala_benchmark(benchmark, figures_dir / "kujala_benchmark.png")
    figures.append(fig_info)
    print(f"  Generated: {fig_info['name']}")
    
    # 5. Amplification vs speed
    fig_info = figure_amplification_along_sweep(sweep, figures_dir / "amplification_vs_speed.png")
    figures.append(fig_info)
    print(f"  Generated: {fig_info['name']}")
    
    # Save figure metadata
    with open(figures_dir / "figures.json", "w") as f:
        json.dump(figures, f, indent=2)
    
    print(f"\nAll figures saved to {figures_dir}")
    return figures


if __name__ == "__main__":
    main()
