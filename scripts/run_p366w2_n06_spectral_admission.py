#!/usr/bin/env python3
"""Run the non-Gaussian spectral admission domain study.

Generates:
1. Error map on (γ3, γ4, α₂) grid
2. Figures: error surfaces, admission boundaries
3. JSON report with sha256 for reproducibility
4. Geometry contract validation
"""

import sys
import os
import json
import hashlib
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from tensorphysics.non_gaussian_spectral.admission import (
    compute_error_map,
    validate_geometry_contract,
    fit_admission_boundary,
)
from tensorphysics.non_gaussian_spectral.winterstein import (
    winterstein_hermite_transform,
    _compute_hermite_moments,
)


def main():
    print("=" * 70)
    print("Non-Gaussian Spectral Admission Domain Study")
    print("=" * 70)

    t0 = time.time()

    # Define grid
    gamma3_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
    gamma4_values = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0])
    alpha2_values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95])

    print(f"\nGrid: {len(gamma3_values)} × {len(gamma4_values)} × {len(alpha2_values)} = "
          f"{len(gamma3_values)*len(gamma4_values)*len(alpha2_values)} points")
    print(f"Realizations per point: 3")
    print(f"Duration: 120s per realization")

    # Compute error map
    print("\nComputing error map...")
    grid = compute_error_map(
        gamma3_values=gamma3_values,
        gamma4_values=gamma4_values,
        alpha2_values=alpha2_values,
        C=1e12,
        m=3.0,
        duration_s=120.0,
        fs_hz=20.0,
        n_realizations=3,
        threshold=0.10,
        seed_base=20261110,
    )

    sha256 = grid.sha256()
    print(f"Error map SHA256: {sha256}")

    # Validate geometry contract
    print("\nValidating geometry contract...")
    contract = validate_geometry_contract(grid, method="dirlik_ng")
    print(f"  Monotonicity (γ3): {contract['monotonicity_gamma3']}")
    print(f"  Monotonicity (γ4): {contract['monotonicity_gamma4']}")
    print(f"  Monotonicity pass: {contract['monotonicity_pass']}")
    print(f"  Convexity pass: {contract['convexity_pass']}")
    print(f"  No self-intersection pass: {contract['no_self_intersection_pass']}")
    print(f"  ALL PASS: {contract['all_pass']}")

    # Generate figures
    print("\nGenerating figures...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'p366w2_n06')
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Error surface for dirlik_ng at α₂=0.6
    fig, ax = plt.subplots(figsize=(10, 8))
    k_mid = len(alpha2_values) // 2
    errors_2d = grid.errors['dirlik_ng'][:, :, k_mid]
    im = ax.imshow(errors_2d.T, origin='lower', extent=[
        gamma3_values[0], gamma3_values[-1],
        gamma4_values[0], gamma4_values[-1]
    ], cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Relative error')
    ax.contour(gamma3_values, gamma4_values, errors_2d.T, levels=[0.10],
               colors='red', linewidths=2, linestyles='--')
    ax.set_xlabel('Skewness γ₃', fontsize=12)
    ax.set_ylabel('Kurtosis γ₄', fontsize=12)
    ax.set_title(f'Error Map: Dirlik+Winterstein (α₂={alpha2_values[k_mid]:.2f})\n'
                 f'Red dashed = 10% error boundary', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_surface_dirlik_ng.png'), dpi=150)
    plt.close()
    print(f"  Saved: error_surface_dirlik_ng.png")

    # Figure 2: Error surface for rayleigh_ng at α₂=0.6
    fig, ax = plt.subplots(figsize=(10, 8))
    errors_2d = grid.errors['rayleigh_ng'][:, :, k_mid]
    im = ax.imshow(errors_2d.T, origin='lower', extent=[
        gamma3_values[0], gamma3_values[-1],
        gamma4_values[0], gamma4_values[-1]
    ], cmap='plasma', aspect='auto')
    plt.colorbar(im, ax=ax, label='Relative error')
    ax.contour(gamma3_values, gamma4_values, errors_2d.T, levels=[0.10],
               colors='cyan', linewidths=2, linestyles='--')
    ax.set_xlabel('Skewness γ₃', fontsize=12)
    ax.set_ylabel('Kurtosis γ₄', fontsize=12)
    ax.set_title(f'Error Map: Rayleigh+Winterstein (α₂={alpha2_values[k_mid]:.2f})\n'
                 f'Cyan dashed = 10% error boundary', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_surface_rayleigh_ng.png'), dpi=150)
    plt.close()
    print(f"  Saved: error_surface_rayleigh_ng.png")

    # Figure 3: Admission domain comparison (all methods at α₂=0.6)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = ['dirlik_ng', 'bt_ng', 'rayleigh_ng']
    titles = ['Dirlik+Winterstein', 'BT+Winterstein', 'Rayleigh+Winterstein']
    for idx, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[idx]
        admitted_2d = grid.admitted[method][:, :, k_mid]
        ax.imshow(admitted_2d.T, origin='lower', extent=[
            gamma3_values[0], gamma3_values[-1],
            gamma4_values[0], gamma4_values[-1]
        ], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Skewness γ₃')
        ax.set_ylabel('Kurtosis γ₄')
        ax.set_title(f'{title}\n(α₂={alpha2_values[k_mid]:.2f})')
        # Add text
        n_admitted = np.sum(admitted_2d)
        n_total = admitted_2d.size
        ax.text(0.05, 0.95, f'Admitted: {n_admitted}/{n_total}\n'
                f'({100*n_admitted/n_total:.1f}%)',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'admission_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: admission_comparison.png")

    # Figure 4: Winterstein transform visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: forward transform for different (γ3, γ4)
    ax = axes[0]
    u = np.linspace(-3, 3, 200)
    for g3, g4, color in [(0.0, 3.0, 'gray'), (0.5, 4.0, 'blue'), (1.0, 5.0, 'red')]:
        tfm = winterstein_hermite_transform(g3, g4)
        x = tfm.forward(u)
        ax.plot(u, x, color=color, linewidth=2, label=f'γ₃={g3}, γ₄={g4}')
    ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.3, label='Identity')
    ax.set_xlabel('Gaussian u', fontsize=12)
    ax.set_ylabel('Non-Gaussian x = W(u)', fontsize=12)
    ax.set_title('Winterstein Forward Transform', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: coefficient map
    ax = axes[1]
    g3_range = np.linspace(0, 1.5, 20)
    g4_range = np.linspace(3, 7, 20)
    c3_map = np.zeros((len(g3_range), len(g4_range)))
    c4_map = np.zeros((len(g3_range), len(g4_range)))
    for i, g3 in enumerate(g3_range):
        for j, g4 in enumerate(g4_range):
            tfm = winterstein_hermite_transform(g3, g4)
            c3_map[i, j] = tfm.c3
            c4_map[i, j] = tfm.c4
    im = ax.imshow(c3_map.T, origin='lower', extent=[0, 1.5, 3, 7],
                   cmap='coolwarm', aspect='auto')
    plt.colorbar(im, ax=ax, label='c₃ coefficient')
    ax.set_xlabel('Skewness γ₃', fontsize=12)
    ax.set_ylabel('Kurtosis γ₄', fontsize=12)
    ax.set_title('Winterstein Coefficient c₃', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'winterstein_transform.png'), dpi=150)
    plt.close()
    print(f"  Saved: winterstein_transform.png")

    # Save JSON report
    report = {
        "mission_id": "p366w2-n06-spectral-admission",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "grid": {
            "gamma3_values": gamma3_values.tolist(),
            "gamma4_values": gamma4_values.tolist(),
            "alpha2_values": alpha2_values.tolist(),
            "n_realizations": 3,
            "duration_s": 120.0,
            "fs_hz": 20.0,
            "seed_base": 20261110,
        },
        "sha256": sha256,
        "geometry_contract": contract,
        "errors_summary": {},
        "admission_summary": {},
    }

    for method in grid.errors.keys():
        errors = grid.errors[method]
        admitted = grid.admitted[method]
        report["errors_summary"][method] = {
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
        }
        report["admission_summary"][method] = {
            "n_admitted": int(np.sum(admitted)),
            "n_total": int(admitted.size),
            "admission_rate": float(np.mean(admitted)),
        }

    report_path = os.path.join(os.path.dirname(__file__), '..', 'runs', 'p366w2_n06',
                                'admission_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
