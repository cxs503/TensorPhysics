#!/usr/bin/env python3
"""Generate figures for FSI flutter benchmark report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorphysics.fsi import (
    AirfoilFlutter,
    FlutterConfig,
    LooseCoupling,
    TightCoupling,
)


def generate_figures(output_dir: Path) -> list[dict]:
    """Generate all figures for the report.
    
    Returns list of {name, caption, path} dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    
    config = FlutterConfig(
        chord=1.0,
        mass_ratio=20.0,
        radius_of_gyration=0.5,
        frequency_ratio=0.5,
        reduced_damping_h=0.05,
        reduced_damping_a=0.05,
        dt=0.002,
        n_steps=3000,
        alpha0=0.05,
    )
    
    # Figure 1: Time history - stable case
    print("Generating Figure 1: Stable response time history...")
    flutter = AirfoilFlutter(config)
    result = flutter.run(U=1.0, n_steps=3000)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(result.time, result.pitch * 180 / np.pi, 'b-', linewidth=1)
    axes[0].set_ylabel('Pitch angle [deg]')
    axes[0].set_title('FSI Flutter: Stable Response (U = 1.0 m/s)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result.time, result.plunge, 'r-', linewidth=1)
    axes[1].set_ylabel('Plunge displacement [m]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path1 = output_dir / "fig1_stable_response.png"
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    figures.append({
        "name": "fig1_stable_response",
        "caption": "Time history of pitch and plunge for stable case (U=1.0 m/s, below flutter speed). Response decays due to structural damping.",
        "path": str(path1),
    })
    
    # Figure 2: Loose vs tight coupling comparison
    print("Generating Figure 2: Coupling scheme comparison...")
    flutter_loose = AirfoilFlutter(config)
    loose = LooseCoupling(flutter_loose, exchange_interval=10, flow_velocity=2.0)
    result_loose = loose.run(n_steps=1000)
    
    flutter_tight = AirfoilFlutter(config)
    tight = TightCoupling(flutter_tight, flow_velocity=2.0, tolerance=1e-6)
    result_tight = tight.run(n_steps=1000)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(result_loose.time, result_loose.pitch * 180 / np.pi, 'b-', 
                 label='Loose coupling', linewidth=1, alpha=0.7)
    axes[0].plot(result_tight.time, result_tight.pitch * 180 / np.pi, 'r--', 
                 label='Tight coupling', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('Pitch angle [deg]')
    axes[0].set_title('FSI Coupling Scheme Comparison (U = 2.0 m/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(result_loose.time[::10], np.abs(result_loose.interface_residual) + 1e-15, 
                     'b-o', label='Loose', markersize=3)
    axes[1].semilogy(result_tight.time, np.abs(result_tight.interface_residual) + 1e-15, 
                     'r--', label='Tight', linewidth=1)
    axes[1].set_ylabel('Interface residual')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path2 = output_dir / "fig2_coupling_comparison.png"
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    figures.append({
        "name": "fig2_coupling_comparison",
        "caption": "Comparison of loose and tight coupling schemes. Top: pitch angle time history. Bottom: interface convergence residual (log scale).",
        "path": str(path2),
    })
    
    # Figure 3: Negative control
    print("Generating Figure 3: Negative control (no coupling)...")
    flutter_neg = AirfoilFlutter(config)
    result_neg = flutter_neg.run(U=0.0, n_steps=5000)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result_neg.time, result_neg.pitch * 180 / np.pi, 'g-', linewidth=1)
    ax.axhline(y=0.01 * 180 / np.pi, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax.axhline(y=-0.01 * 180 / np.pi, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Pitch angle [deg]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Negative Control: No Aerodynamic Coupling (U = 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path3 = output_dir / "fig3_negative_control.png"
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    figures.append({
        "name": "fig3_negative_control",
        "caption": "Negative control: with zero flow velocity, the structural response decays to zero due to damping. Red dashed lines show 1% threshold.",
        "path": str(path3),
    })
    
    # Figure 4: Airfoil geometry and surface nodes
    print("Generating Figure 4: Airfoil geometry...")
    from tensorphysics.fsi import LBMForceAdapter
    adapter = LBMForceAdapter(chord=1.0, n_surface_nodes=30)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot airfoil surface
    xc = np.linspace(0, 1, 100)
    t = 0.12
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    
    ax.plot(xc, yt, 'b-', linewidth=2, label='Upper surface')
    ax.plot(xc, -yt, 'b-', linewidth=2, label='Lower surface')
    ax.plot(adapter.surface_nodes[:30, 0], adapter.surface_nodes[:30, 1], 'ro', markersize=4, label='LBM surface nodes (upper)')
    ax.plot(adapter.surface_nodes[30:, 0], adapter.surface_nodes[30:, 1], 'go', markersize=4, label='LBM surface nodes (lower)')
    
    # Mark elastic axis
    ax.axvline(x=0.35, color='k', linestyle='--', alpha=0.5, label='Elastic axis (35% chord)')
    
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title('NACA 0012 Airfoil with FSI Interface Nodes')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path4 = output_dir / "fig4_airfoil_geometry.png"
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    figures.append({
        "name": "fig4_airfoil_geometry",
        "caption": "NACA 0012 airfoil geometry with FSI interface nodes. Red/green dots show surface nodes for force exchange between TensorLBM and TensorFEM.",
        "path": str(path4),
    })
    
    print(f"Generated {len(figures)} figures in {output_dir}")
    return figures


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fsi_flutter/figures"))
    args = parser.parse_args()
    
    figures = generate_figures(args.output_dir)
    for f in figures:
        print(f"  {f['name']}: {f['path']}")
