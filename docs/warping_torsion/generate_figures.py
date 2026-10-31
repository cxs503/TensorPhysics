"""
Generate figures for W2-N06 thin-walled torsion & warping report.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
import base64
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.structural_mechanics.thin_walled_torsion import (
    saint_venant_J_rectangular,
    saint_venant_J_closed_box,
    characteristic_length,
)
from tensorphysics.structural_mechanics.warping import (
    bimoment_decay,
    vlasov_ode_solution,
    sectorial_coordinate_closed,
    warping_stress,
)

figures = []

def save_figure(fig, name):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    png_base64 = base64.b64encode(buf.read()).decode('utf-8')
    figures.append({"name": name, "png_base64": png_base64})
    plt.close(fig)

# =============================================================================
# Figure 1: Saint-Venant J vs b/h ratio (Roark comparison)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ratios = np.linspace(1, 15, 100)
J_computed = []
J_roark_beta = []  # Roark reference betas

for r in ratios:
    J_c = saint_venant_J_rectangular(r, 1.0, n_terms=50)
    J_computed.append(J_c)
    # Reference beta from Roark Table 10.1 (interpolated)
    if r <= 1.0:
        beta = 0.1406
    elif r <= 1.5:
        beta = 0.196
    elif r <= 2.0:
        beta = 0.229
    elif r <= 2.5:
        beta = 0.249
    elif r <= 3.0:
        beta = 0.263
    elif r <= 4.0:
        beta = 0.281
    elif r <= 5.0:
        beta = 0.291
    elif r <= 10.0:
        beta = 0.312
    else:
        beta = 1.0/3.0 * (1 - 0.63/r)
    J_roark_beta.append(beta * r * 1.0**3)

ax.plot(ratios, J_computed, 'b-', linewidth=2, label='Computed (series)')
ax.plot(ratios, J_roark_beta, 'r--', linewidth=1.5, label='Roark reference')
ax.set_xlabel('b/h ratio', fontsize=12)
ax.set_ylabel('J / h³', fontsize=12)
ax.set_title('Saint-Venant Torsion Constant: Computed vs Roark Reference', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
save_figure(fig, "fig1_saint_venant_J_vs_ratio")

# =============================================================================
# Figure 2: Bimoment decay along beam length
# =============================================================================
E = 210e9
G = 81e9
Iw = 1e-7
J = 1e-6
L = 10.0
B0 = 1e5

lam = np.sqrt(G * J / (E * Iw))
x = np.linspace(0, L, 200)

B_numerical = bimoment_decay(x, B0, E, Iw, G, J, L)
B_analytical = B0 * np.cosh(lam * (L - x)) / np.cosh(lam * L)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, B_numerical / 1e3, 'b-', linewidth=2, label='Bimoment B(x)')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=L, color='r', linestyle='--', alpha=0.5, label=f'x = L = {L} m')
ax.set_xlabel('Position x [m]', fontsize=12)
ax.set_ylabel('Bimoment B [kN·m²]', fontsize=12)
ax.set_title(f'Bimoment Decay Along Beam (λ = {lam:.3f} m⁻¹)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add characteristic length annotation
ax.axvline(x=1/lam, color='g', linestyle=':', alpha=0.5, label=f'1/λ = {1/lam:.2f} m')
ax.legend(fontsize=10)
save_figure(fig, "fig2_bimoment_decay")

# =============================================================================
# Figure 3: Twist angle distribution (Vlasov solution)
# =============================================================================
T = 1e4
x, theta, theta_prime, bimoment = vlasov_ode_solution(
    L, E, G, Iw, J,
    boundary_condition="fixed-free",
    applied_torque=T,
    n_points=200
)

theta_SV = T * x / (G * J)  # Pure Saint-Venant

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(x, theta * 180 / np.pi, 'b-', linewidth=2, label='Vlasov θ(x)')
ax1.plot(x, theta_SV * 180 / np.pi, 'r--', linewidth=1.5, label='Saint-Venant θ_SV(x)')
ax1.set_xlabel('Position x [m]', fontsize=12)
ax1.set_ylabel('Twist angle [deg]', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Twist Angle: Vlasov vs Pure Saint-Venant', fontsize=13)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(x, bimoment / 1e3, 'g-', linewidth=1.5, alpha=0.7, label='Bimoment B(x)')
ax2.set_ylabel('Bimoment [kN·m²]', fontsize=12, color='g')
ax2.tick_params(axis='y', labelcolor='g')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
save_figure(fig, "fig3_twist_angle_vlasov")

# =============================================================================
# Figure 4: Sectorial coordinate for rectangular box
# =============================================================================
B_box, H_box = 2.0, 1.0
t_w, t_f = 0.05, 0.05

s, omega_bar, A_m = sectorial_coordinate_closed(B_box, H_box, t_w, t_f)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s, omega_bar, 'b-', linewidth=2)
ax.set_xlabel('Arc length s [m]', fontsize=12)
ax.set_ylabel('Normalized sectorial coordinate ω̄ [m²]', fontsize=12)
ax.set_title(f'Sectorial Coordinate for {B_box}×{H_box}m Box Section', fontsize=13)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Mark the four sides
perimeter = 2 * (B_box - t_w) + 2 * (H_box - t_f)
side_lengths = [B_box - t_w, H_box - t_f, B_box - t_w, H_box - t_f]
side_labels = ['Bottom', 'Right', 'Top', 'Left']
s_cum = 0
for sl, label in zip(side_lengths, side_labels):
    ax.axvline(x=s_cum, color='gray', linestyle=':', alpha=0.5)
    ax.text(s_cum + sl/2, ax.get_ylim()[1]*0.9, label, ha='center', fontsize=9, alpha=0.7)
    s_cum += sl
save_figure(fig, "fig4_sectorial_coordinate")

# =============================================================================
# Figure 5: Warping stress distribution
# =============================================================================
# Compute warping stress at different sections
x_sections = [0, L/4, L/2, 3*L/4, L]
B_sections = bimoment_decay(np.array(x_sections), B0, E, Iw, G, J, L)

# Use sectorial coordinate values at key points
omega_points = np.array([0.5, -0.5, 0.25, -0.25, 0.0])  # representative ω̄ values

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['blue', 'green', 'orange', 'red', 'purple']
for i, (xs, Bs) in enumerate(zip(x_sections, B_sections)):
    sigma = warping_stress(omega_points, Bs, Iw)
    ax.plot(omega_points, sigma / 1e6, 'o-', color=colors[i], linewidth=1.5,
            label=f'x = {xs:.1f} m (B = {Bs/1e3:.1f} kN·m²)')

ax.set_xlabel('Normalized sectorial coordinate ω̄ [m²]', fontsize=12)
ax.set_ylabel('Warping stress σ_ω [MPa]', fontsize=12)
ax.set_title('Warping Stress Distribution at Various Sections', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
save_figure(fig, "fig5_warping_stress")

# =============================================================================
# Save figures to disk
# =============================================================================
os.makedirs('/root/tp-agent/work/m765/artifacts', exist_ok=True)
for fig_data in figures:
    png_bytes = base64.b64decode(fig_data['png_base64'])
    with open(f"/root/tp-agent/work/m765/artifacts/{fig_data['name']}.png", 'wb') as f:
        f.write(png_bytes)

# Save figure metadata
with open('/root/tp-agent/work/m765/artifacts/figures.json', 'w') as f:
    json.dump(figures, f)

print(f"Generated {len(figures)} figures")
for fig_data in figures:
    print(f"  - {fig_data['name']}")
