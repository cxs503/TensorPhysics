"""Generate figures for P167W2-N07 structural-stability coupling report."""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

from tensorphysics.structural_stability_coupling import (
    compute_gz_curve, wall_sided_gz, propagate_delta_iso,
    PlatePanelContribution, compute_degradation, PlatePanelStructural,
    faulkner_effective_width, cross_validate, negative_control_gate,
)
from tensorphysics.structural_stability_coupling.gz_curve import HullFormData

OUT = "/root/tp-agent/work/m9393/artifacts/m9393"
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# Common hull form (small cargo vessel)
# ============================================================================
hull = HullFormData(
    L=80.0, B=14.0, T=5.5,
    Cb=0.65, Cwp=0.72,
    KG=5.0, KB=2.5, BM_T=4.2,
    LCG=0.0, TCG=0.0,
)

# ============================================================================
# Figure 1: GZ curves — intact vs degraded
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
angles = np.arange(0, 75, 1.0)

gz_intact = compute_gz_curve(hull, angles_deg=angles, method="wall_sided", damage_factor=0.0)
gz_dmg_03 = compute_gz_curve(hull, angles_deg=angles, method="wall_sided", damage_factor=0.3)
gz_dmg_06 = compute_gz_curve(hull, angles_deg=angles, method="wall_sided", damage_factor=0.6)

# For degraded cases, apply damage to GM
for d, color, label in [(0.3, '#e67e22', 'D = 0.3 (moderate yield)'),
                         (0.6, '#c0392b', 'D = 0.6 (severe yield)')]:
    gm_deg = hull.GM * (1 - 0.3 * d)
    gz_d = np.array([wall_sided_gz(phi, gm_deg) for phi in angles])
    ax.plot(angles, gz_d, color=color, lw=2, label=label)

ax.plot(angles, gz_intact.gz_m, 'b-', lw=2.5, label='Intact (D = 0)')
ax.set_xlabel('Heel angle φ [deg]', fontsize=12)
ax.set_ylabel('GZ [m]', fontsize=12)
ax.set_title('GZ Curves: Effect of Structural Stiffness Degradation', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 75)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_gz_degradation.png", dpi=150)
plt.close(fig)

# ============================================================================
# Figure 2: Error propagation δ_iso → GZ correction
# ============================================================================
panels_weight = [
    PlatePanelContribution("bottom_AP", 120, 0.5, 3.0, 0.018, 0.0005),
    PlatePanelContribution("bottom_FP", 110, 0.5, -3.0, 0.018, 0.0005),
    PlatePanelContribution("strength_deck", 150, 9.5, 0.0, 0.016, 0.0004),
    PlatePanelContribution("side_port", 200, 5.0, 6.5, 0.014, 0.0004),
    PlatePanelContribution("side_star", 200, 5.0, -6.5, 0.014, 0.0004),
]

angles_prop = np.arange(0, 45, 1.0)
gz_nom = np.array([wall_sided_gz(phi, hull.GM) for phi in angles_prop])

prop_1sig = propagate_delta_iso(panels_weight, 30000e3, hull.KG, 0.0, gz_nom, angles_prop, 1.0)
prop_2sig = propagate_delta_iso(panels_weight, 30000e3, hull.KG, 0.0, gz_nom, angles_prop, 2.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(angles_prop, gz_nom, 'b-', lw=2, label='Nominal GZ')
ax1.fill_between(angles_prop, prop_1sig.gz_corrected_m, gz_nom, alpha=0.3, color='orange', label='±1σ band')
ax1.fill_between(angles_prop, prop_2sig.gz_corrected_m, gz_nom, alpha=0.15, color='red', label='±2σ band')
ax1.set_xlabel('Heel angle φ [deg]')
ax1.set_ylabel('GZ [m]')
ax1.set_title('GZ with δ_iso Uncertainty Band')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(angles_prop, prop_1sig.relative_change_pct, 'orange', lw=2, label='±1σ')
ax2.plot(angles_prop, prop_2sig.relative_change_pct, 'r--', lw=2, label='±2σ')
ax2.axhline(y=0, color='k', lw=0.5)
ax2.set_xlabel('Heel angle φ [deg]')
ax2.set_ylabel('Relative GZ change [%]')
ax2.set_title('Relative GZ Correction from δ_iso')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Error Propagation: δKG = {prop_1sig.delta_KG_m*1000:.1f} mm, δTCG = {prop_1sig.delta_TCG_m*1000:.1f} mm', fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_error_propagation.png", dpi=150)
plt.close(fig)

# ============================================================================
# Figure 3: Faulkner effective-width curve + degradation
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

beta_range = np.linspace(0.2, 5.0, 100)
phi_range = np.array([faulkner_effective_width(b) for b in beta_range])

ax1.plot(beta_range, phi_range, 'b-', lw=2.5)
ax1.axhline(y=1.0, color='gray', ls='--', lw=0.8, label='Full effective width')
ax1.axvline(x=1.0, color='gray', ls=':', lw=0.8, label='β = 1 (stocky/slender boundary)')
ax1.set_xlabel('Plate slenderness β = (b/t)√(σ_y/E)', fontsize=11)
ax1.set_ylabel('Effective width ratio b_eff/b', fontsize=11)
ax1.set_title('Faulkner Effective-Width Reduction', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)

# GM vs damage parameter
D_range = np.linspace(0, 1, 50)
GM_intact = hull.GM
GM_faulkner = 0.5  # limiting case

nc = negative_control_gate(GM_intact, GM_faulkner, alpha_structural=0.3)
ax2.plot(nc.D_values, nc.GM_values, 'b-', lw=2.5, label='GM(D) = GM₀(1 - αD)')
ax2.axhline(y=GM_faulkner, color='r', ls='--', lw=1.5, label=f'Faulkner limit = {GM_faulkner} m')
ax2.axhline(y=GM_intact, color='g', ls=':', lw=1, label=f'Intact GM = {GM_intact:.2f} m')
ax2.fill_between(nc.D_values, GM_faulkner, nc.GM_values, alpha=0.15, color='blue')
ax2.set_xlabel('Damage parameter D', fontsize=11)
ax2.set_ylabel('GM [m]', fontsize=11)
ax2.set_title('Negative Control: GM vs Structural Damage', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_faulkner_and_negcontrol.png", dpi=150)
plt.close(fig)

# ============================================================================
# Figure 4: Cross-validation — pressure distribution at heel
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

phi_vals = [0, 5, 10, 15, 20]
z_panel = np.linspace(0, hull.T, 50)
rho = 1025.0
g = 9.81

for phi in phi_vals:
    phi_rad = math.radians(phi)
    # Pressure on port side panel (y = B/4)
    y_c = hull.B / 4
    depth = hull.T - z_panel + y_c * math.tan(phi_rad)
    depth = np.maximum(depth, 0)
    p = rho * g * depth / 1e3  # kPa
    ax.plot(p, z_panel, lw=2, label=f'φ = {phi}°')

ax.set_xlabel('Hydrostatic pressure [kPa]', fontsize=11)
ax.set_ylabel('Height above baseline [m]', fontsize=11)
ax.set_title('Pressure Distribution on Side Shell at Various Heel Angles', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, hull.T)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_pressure_distribution.png", dpi=150)
plt.close(fig)

# ============================================================================
# Figure 5: Stiffness degradation — panel-level detail
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

panels_struct = [
    PlatePanelStructural("deck_1", 0.8, 0.016, 315e6, 2.06e11, 9.5, 0.012, 0.0),
    PlatePanelStructural("deck_1", 0.8, 0.016, 315e6, 2.06e11, 9.5, 0.012, 0.5),
    PlatePanelStructural("deck_1", 0.8, 0.016, 315e6, 2.06e11, 9.5, 0.012, 0.8),
    PlatePanelStructural("deck_1", 0.8, 0.016, 315e6, 2.06e11, 9.5, 0.012, 1.0),
    PlatePanelStructural("deck_1", 0.8, 0.016, 315e6, 2.06e11, 9.5, 0.012, 1.2),
]

load_ratios = [p.load_ratio for p in panels_struct]
beta = panels_struct[0].beta
phi_vals = [faulkner_effective_width(beta)] * len(panels_struct)  # constant for same geometry

# Actually compute degradation for each load ratio
gm_values = []
for lr in load_ratios:
    p = PlatePanelStructural("deck_1", 0.8, 0.016, 315e6, 2.06e11, 9.5, 0.012, lr)
    deg = compute_degradation([p], hull.I_T, hull.nabla, hull.KG, hull.KB, 100, 80)
    gm_values.append(deg.GM_degraded_m)

ax.plot(load_ratios, gm_values, 'bo-', lw=2, markersize=8)
ax.axhline(y=hull.GM, color='g', ls='--', label=f'Intact GM = {hull.GM:.2f} m')
ax.set_xlabel('Load ratio (applied / yield)', fontsize=11)
ax.set_ylabel('GM [m]', fontsize=11)
ax.set_title(f'GM Reduction vs Panel Load Ratio (β = {beta:.2f})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_gm_vs_load_ratio.png", dpi=150)
plt.close(fig)

# ============================================================================
# Figure 6: Coupling chain summary diagram
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Boxes
boxes = [
    (1, 4.5, 'N05: FEM Elastoplastic\nPlate Ultimate Strength\n(δ_iso error bound)'),
    (5, 4.5, 'Error Propagation\nδ_iso → δKG → δGZ'),
    (9, 4.5, 'GZ Curve\n(intact + corrected)'),
    (1, 1.5, 'N06: Inclining Test\n3D Inversion\n(waterline geometry)'),
    (5, 1.5, 'Cross-Validation\n3D waterline → plate load\nvs. ultimate capacity'),
    (9, 1.5, 'Negative Control\nstiffness → 0\nFaulkner asymptote'),
]

for x, y, text in boxes:
    ax.add_patch(plt.Rectangle((x-1.2, y-0.7), 2.4, 1.4,
                                facecolor='#3498db' if x < 5 else '#2ecc71' if x < 9 else '#e74c3c',
                                edgecolor='black', lw=1.5, alpha=0.7, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

# Arrows
arrow_kw = dict(arrowstyle='->', lw=2, color='#2c3e50')
ax.annotate('', xy=(3.8, 4.5), xytext=(2.2, 4.5), arrowprops=arrow_kw)
ax.annotate('', xy=(7.8, 4.5), xytext=(6.2, 4.5), arrowprops=arrow_kw)
ax.annotate('', xy=(3.8, 1.5), xytext=(2.2, 1.5), arrowprops=arrow_kw)
ax.annotate('', xy=(7.8, 1.5), xytext=(6.2, 1.5), arrowprops=arrow_kw)
# Cross-coupling arrows
ax.annotate('', xy=(5, 3.6), xytext=(5, 3.0), arrowprops=dict(arrowstyle='<->', lw=2, color='#8e44ad'))

ax.text(5, 3.3, 'Coupling', ha='center', fontsize=9, color='#8e44ad', fontweight='bold')
ax.set_title('Structural–Stability Coupling Verification Chain (P167W2-N07)', fontsize=13, pad=20)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_coupling_chain.png", dpi=150)
plt.close(fig)

print("All figures generated successfully.")
print(f"Output directory: {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png'):
        sz = os.path.getsize(f"{OUT}/{f}")
        print(f"  {f}: {sz/1024:.0f} KB")
