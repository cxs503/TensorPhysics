#!/usr/bin/env python3
"""Generate figures for the coverage audit report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load case study results
results_path = ROOT / "artifacts" / "coverage_audit" / "case_studies.json"
with open(results_path) as f:
    results = json.load(f)

fig_dir = ROOT / "artifacts" / "coverage_audit" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Figure 1: Coverage curves for Weibull m (m7272)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
m7272 = results["m7272_weibull_m"]

for idx, level_data in enumerate(m7272["levels"]):
    ax = axes[idx]
    curve = level_data["coverage_curve"]
    n_reps = len(curve)
    reps = np.arange(1, n_reps + 1)
    
    ax.plot(reps, curve, 'b-', linewidth=1.5, label='Empirical coverage')
    ax.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.axhline(y=0.93, color='orange', linestyle=':', linewidth=1, label='Well-calibrated bounds')
    ax.axhline(y=0.97, color='orange', linestyle=':', linewidth=1)
    ax.fill_between([0, n_reps], [0.93, 0.93], [0.97, 0.97], alpha=0.1, color='green')
    
    ax.set_xlabel('Replication number')
    ax.set_ylabel('Running coverage')
    ax.set_title(f'Level {level_data["error_level"]}: {level_data["verdict"]}\n'
                 f'Coverage={level_data["coverage"]:.3f}')
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Case 1: Weibull shape parameter m (m7272) — Coverage convergence\n'
             'n=8, bootstrap B=500, N=500 replications', fontsize=12, fontweight='bold')
plt.tight_layout()
fig1_path = fig_dir / "m7272_coverage_curves.png"
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {fig1_path}")

# ============================================================================
# Figure 2: Coverage curves for σ divergence (m7269)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
m7269 = results["m7269_sigma"]

for idx, level_data in enumerate(m7269["levels"]):
    ax = axes[idx]
    curve = level_data["coverage_curve"]
    n_reps = len(curve)
    reps = np.arange(1, n_reps + 1)
    
    ax.plot(reps, curve, 'b-', linewidth=1.5, label='Empirical coverage')
    ax.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.axhline(y=0.93, color='orange', linestyle=':', linewidth=1, label='Well-calibrated bounds')
    ax.axhline(y=0.97, color='orange', linestyle=':', linewidth=1)
    ax.fill_between([0, n_reps], [0.93, 0.93], [0.97, 0.97], alpha=0.1, color='green')
    
    ax.set_xlabel('Replication number')
    ax.set_ylabel('Running coverage')
    ax.set_title(f'Level {level_data["error_level"]}: {level_data["verdict"]}\n'
                 f'Coverage={level_data["coverage"]:.3f}')
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Case 2: σ divergence (m7269) — Coverage convergence\n'
             'N_peak≈60, chi-squared CI, N=500 replications', fontsize=12, fontweight='bold')
plt.tight_layout()
fig2_path = fig_dir / "m7269_coverage_curves.png"
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {fig2_path}")

# ============================================================================
# Figure 3: Summary comparison bar chart
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 5))

cases = ['m7272 Weibull m', 'm7269 σ divergence']
levels = ['Level 0', 'Level 1', 'Level 2']
colors = ['#2ecc71', '#e67e22', '#e74c3c']  # green, orange, red

x = np.arange(len(cases))
width = 0.25

for i, level_name in enumerate(levels):
    coverages = []
    for case_name in ['m7272_weibull_m', 'm7269_sigma']:
        coverages.append(results[case_name]["levels"][i]["coverage"])
    
    bars = ax.bar(x + i*width - width, coverages, width, label=level_name, 
                  color=colors[i], alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cov:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Nominal 95%')
ax.axhline(y=0.93, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(y=0.97, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.fill_between([-0.5, 1.5], [0.93, 0.93], [0.97, 0.97], alpha=0.05, color='green')

ax.set_xlabel('Case study', fontsize=11)
ax.set_ylabel('Empirical coverage', fontsize=11)
ax.set_title('Coverage Audit Summary: Empirical vs Nominal Coverage\n'
             'Three error-injection levels per case', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=10)
ax.set_ylim([0.5, 1.05])
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig3_path = fig_dir / "coverage_summary.png"
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {fig3_path}")

# ============================================================================
# Figure 4: n_min stopping rule visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

# Wilson half-width vs n for different coverages
n_values = np.arange(10, 1001)
coverages = [0.65, 0.80, 0.95, 0.99]
colors_hw = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']

for cov, color in zip(coverages, colors_hw):
    k_values = (cov * n_values).astype(int)
    hw_values = []
    for k, n in zip(k_values, n_values):
        # Wilson half-width
        z = 1.96
        p_hat = k / n
        denom = 1 + z**2 / n
        centre = p_hat + z**2 / (2 * n)
        spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
        hi = (centre + spread) / denom
        lo = (centre - spread) / denom
        hw_values.append(0.5 * (hi - lo))
    
    ax.plot(n_values, hw_values, linewidth=2, label=f'p={cov:.2f}', color=color)

ax.axhline(y=0.02, color='black', linestyle='--', linewidth=2, label='Target ±0.02')
ax.set_xlabel('Number of replications (n)', fontsize=11)
ax.set_ylabel('Wilson half-width', fontsize=11)
ax.set_title('Stopping Rule: Minimum Replications for ±0.02 Resolution\n'
             'Wilson score interval half-width vs sample size', 
             fontsize=12, fontweight='bold')
ax.set_ylim([0, 0.25])
ax.set_xlim([0, 1000])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4_path = fig_dir / "stopping_rule.png"
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {fig4_path}")

print("\nAll figures generated successfully.")
