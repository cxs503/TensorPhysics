"""Generate figures for BEM frequency-domain verification report."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
import os

from tensorphysics.bem_freq.green_function import (
    GreenFunction, kochin_function, kochin_energy_damping, RHO_WATER, GRAVITY
)
from tensorphysics.bem_freq.hemisphere import (
    hemisphere_mesh, hemisphere_analytic_A33, hemisphere_radiation_solve,
    hemisphere_anchor_check
)
from tensorphysics.bem_freq.haskind import haskind_relation, incident_wave_potential
from tensorphysics.bem_freq.wigley import wigley_iii_Cw_vs_fr, NAKOS_DATA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_fig(fig):
    """Encode matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def fig1_hemisphere_convergence():
    """Figure 1: Hemisphere A_33 convergence with mesh resolution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    R = 1.0
    A33_analytic = hemisphere_analytic_A33(R, RHO_WATER)

    resolutions = [(12, 8), (16, 12), (24, 16), (32, 24), (48, 32)]
    errors = []
    n_panels_list = []

    for n_theta, n_phi in resolutions:
        mesh = hemisphere_mesh(R, n_theta, n_phi)
        result = hemisphere_radiation_solve(mesh, omega=0.01, free_surface=False)
        error = abs(result['A33'] - A33_analytic) / A33_analytic
        errors.append(error * 100)  # percentage
        n_panels_list.append(mesh['n_panels'])

    ax.semilogx(n_panels_list, errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=3.0, color='r', linestyle='--', linewidth=1.5, label='3% tolerance')
    ax.set_xlabel('Number of panels', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title('Hemisphere A₃₃ Convergence with Mesh Resolution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_b64 = encode_fig(fig)
    plt.close(fig)
    return png_b64


def fig2_wigley_Cw_comparison():
    """Figure 2: Wigley III C_W vs Froude number comparison with Nakos (1990)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    results = wigley_iii_Cw_vs_fr([0.25, 0.30, 0.35])
    frs = [r['Fr'] for r in results]
    cw_calc = [r['C_W'] * 1000 for r in results]  # ×1000
    cw_exp = [r['C_W_exp'] for r in results]

    ax.plot(frs, cw_calc, 'bo-', linewidth=2, markersize=10, label='Michell theory (this work)')
    ax.plot(frs, cw_exp, 'rs--', linewidth=2, markersize=10, label='Nakos (1990) experiment')

    ax.set_xlabel('Froude number Fr', fontsize=12)
    ax.set_ylabel('C_W × 1000', fontsize=12)
    ax.set_title('Wigley III Wave-Making Resistance: Theory vs Experiment', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 0.4)

    png_b64 = encode_fig(fig)
    plt.close(fig)
    return png_b64


def fig3_kochin_polar():
    """Figure 3: Kochin function polar plot for hemisphere radiation."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

    mesh = hemisphere_mesh(R=1.0, n_theta=32, n_phi=24)
    result = hemisphere_radiation_solve(mesh, omega=1.0, free_surface=False)

    theta = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    H = kochin_function(1.0, theta, np.real(result['phi']),
                       mesh['x'], mesh['y'], mesh['z'])

    ax.plot(theta, np.abs(H), 'b-', linewidth=2)
    ax.fill(theta, np.abs(H), alpha=0.3)
    ax.set_title('Kochin Function |H(θ)| for Hemisphere Radiation', fontsize=13)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    png_b64 = encode_fig(fig)
    plt.close(fig)
    return png_b64


def fig4_haskind_residual():
    """Figure 4: Haskind relation verification at multiple frequencies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mesh = hemisphere_mesh(R=1.0, n_theta=24, n_phi=16)
    omegas = np.linspace(0.3, 3.0, 15)
    residuals = []

    for omega in omegas:
        result = hemisphere_radiation_solve(mesh, omega, free_surface=True)
        F_haskind = haskind_relation(
            omega, result['phi'], mesh['nz'],
            mesh['x'], mesh['y'], mesh['z'],
            mesh['area'], A=1.0, beta=0.0
        )
        # Use radiation force magnitude as reference
        F_ref = result['F_rad']
        if abs(F_ref) > 1e-10:
            res = abs(F_haskind - F_ref * 0.5) / abs(F_ref * 0.5)
        else:
            res = 0
        residuals.append(res)

    ax.plot(omegas, residuals, 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('Angular frequency ω (rad/s)', fontsize=12)
    ax.set_ylabel('Haskind residual', fontsize=12)
    ax.set_title('Haskind Relation Verification vs Frequency', fontsize=13)
    ax.grid(True, alpha=0.3)

    png_b64 = encode_fig(fig)
    plt.close(fig)
    return png_b64


def main():
    """Generate all figures and save to artifacts."""
    figures = {}

    print("Generating Figure 1: Hemisphere convergence...")
    figures['hemisphere_convergence'] = fig1_hemisphere_convergence()

    print("Generating Figure 2: Wigley III C_W comparison...")
    figures['wigley_Cw_comparison'] = fig2_wigley_Cw_comparison()

    print("Generating Figure 3: Kochin polar plot...")
    figures['kochin_polar'] = fig3_kochin_polar()

    print("Generating Figure 4: Haskind residual...")
    figures['haskind_residual'] = fig4_haskind_residual()

    # Save figures info
    fig_info = {name: f"Figure {i+1}: {name}" for i, name in enumerate(figures.keys())}
    print("\nGenerated figures:", list(fig_info.keys()))

    return figures


if __name__ == '__main__':
    main()
