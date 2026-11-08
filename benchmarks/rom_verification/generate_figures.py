"""
Generate figures for ROM verification benchmark.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.rom import NS2DSolver, PODDecomposition


def generate_figures():
    """Generate all figures for the report."""
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'rom_benchmark', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'rom_benchmark', 'results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # =========================================================================
    # Figure 1: Energy Spectrum
    # =========================================================================
    print("Generating Figure 1: Energy Spectrum...")
    
    # Re-run POD to get spectrum
    fom = NS2DSolver(nx=32, ny=32, nu=0.05, forcing_amp=0.5, forcing_kx=2)
    fom.initialize(seed=42)
    fom.u *= 0.1
    fom.v *= 0.1
    
    # Warm-up
    for _ in range(400):
        fom.step_rk4(0.005)
    
    # Collect snapshots
    snapshots_u, snapshots_v, times, energies = fom.run(5.0, 0.005, save_interval=10)
    
    combined = [np.concatenate([u.flatten(), v.flatten()]) for u, v in zip(snapshots_u, snapshots_v)]
    pod = PODDecomposition()
    pod.fit(combined)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = np.arange(1, len(pod.energy_spectrum) + 1)
    ax.semilogy(modes, pod.energy_spectrum, 'bo-', linewidth=2, markersize=6, label='Energy spectrum')
    ax.axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
    ax.set_xlabel('Mode number', fontsize=12)
    ax.set_ylabel('Energy fraction', fontsize=12)
    ax.set_title('POD Energy Spectrum', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_spectrum.png'), dpi=150)
    plt.close()
    
    # =========================================================================
    # Figure 2: Cumulative Energy
    # =========================================================================
    print("Generating Figure 2: Cumulative Energy...")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(modes, pod.cumulative_energy, 'r-', linewidth=2, label='Cumulative energy')
    ax.axhline(y=0.99, color='g', linestyle='--', label='99% threshold')
    k_star = results['parameters']['k_star']
    ax.axvline(x=k_star, color='b', linestyle=':', label=f'k*={k_star}')
    ax.set_xlabel('Number of modes', fontsize=12)
    ax.set_ylabel('Cumulative energy fraction', fontsize=12)
    ax.set_title('POD Cumulative Energy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_energy.png'), dpi=150)
    plt.close()
    
    # =========================================================================
    # Figure 3: Energy Evolution
    # =========================================================================
    print("Generating Figure 3: Energy Evolution...")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, energies, 'b-', linewidth=2, label='FOM energy')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Kinetic energy', fontsize=12)
    ax.set_title('Energy Evolution (FOM)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_evolution.png'), dpi=150)
    plt.close()
    
    # =========================================================================
    # Figure 4: Velocity Field Snapshot
    # =========================================================================
    print("Generating Figure 4: Velocity Field...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # u-velocity
    im1 = axes[0].imshow(np.real(snapshots_u[50]), cmap='RdBu_r', aspect='auto', origin='lower')
    axes[0].set_title('u-velocity', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # v-velocity
    im2 = axes[1].imshow(np.real(snapshots_v[50]), cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1].set_title('v-velocity', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle('Velocity Field Snapshot (t=2.5)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_field.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Figure 5: Verification Summary
    # =========================================================================
    print("Generating Figure 5: Verification Summary...")
    
    criteria = results['criteria']
    names = list(criteria.keys())
    values = []
    thresholds = []
    colors = []
    
    for name in names:
        if name == 'C7':
            values.append(criteria[name]['value'] * 100)
            thresholds.append(criteria[name]['threshold'] * 100)
        else:
            values.append(criteria[name]['value'])
            thresholds.append(criteria[name]['threshold'])
        colors.append('green' if criteria[name].get('pass', not criteria[name].get('honest_fail', False)) else 'red')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values, width, label='Measured', color=colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, thresholds, width, label='Threshold', color='gray', alpha=0.5)
    
    ax.set_xlabel('Criterion', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('ROM Verification Summary', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'verification_summary.png'), dpi=150)
    plt.close()
    
    print(f"Figures saved to: {output_dir}")
    
    return output_dir


if __name__ == '__main__':
    generate_figures()
