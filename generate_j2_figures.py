"""Generate verification figures for J2 flow theory."""

import os
import sys
import base64
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.plasticity.j2_flow import (
    J2FlowMaterial,
    uniaxial_tension_curve,
    thick_cylinder_elastoplastic,
    unloading_path,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts', 'j2')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_stress_strain_figure():
    """C1: Uniaxial tension stress-strain curve."""
    mat = J2FlowMaterial(E=210e9, nu=0.3, sigma_y0=250e6, H=2e9)
    result = uniaxial_tension_curve(mat, eps_max=0.02, n_points=50)

    fig, ax = plt.subplots(figsize=(8, 5))
    strains = np.array(result['strains'])
    sigma_ana = np.array(result['stresses_analytic']) / 1e6  # MPa
    sigma_num = np.array(result['stresses_numerical']) / 1e6  # MPa

    ax.plot(strains * 100, sigma_ana, 'b-', linewidth=2, label='Analytical')
    ax.plot(strains * 100, sigma_num, 'ro', markersize=4, label='Numerical (1D return map)', alpha=0.7)

    eps_y = mat.sigma_y0 / mat.E
    ax.axvline(x=eps_y * 100, color='gray', linestyle='--', alpha=0.5, label=f'Yield strain ({eps_y*100:.3f}%)')
    ax.axhline(y=mat.sigma_y0 / 1e6, color='gray', linestyle='--', alpha=0.5, label=f'Yield stress ({mat.sigma_y0/1e6:.0f} MPa)')

    ax.set_xlabel('Strain (%)', fontsize=12)
    ax.set_ylabel('Stress (MPa)', fontsize=12)
    ax.set_title('J2 Flow Theory: Uniaxial Tension with Isotropic Hardening', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.05, f'Max rel error: {result["max_rel_error"]:.2e}',
            transform=ax.transAxes, fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'stress_strain.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return fig_to_base64(fig)


def generate_thick_cylinder_figure():
    """C2: Thick cylinder stress profiles."""
    result = thick_cylinder_elastoplastic(
        a=0.05, b=0.10, p=200e6, E=210e9, nu=0.3, sigma_y0=250e6, H=0.0
    )

    r_vals = np.array(result['r_vals'])
    sigma_r_el = np.array(result['sigma_r_elastic']) / 1e6
    sigma_t_el = np.array(result['sigma_theta_elastic']) / 1e6
    sigma_r_ep = np.array(result['sigma_r_ep']) / 1e6
    sigma_t_ep = np.array(result['sigma_theta_ep']) / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Radial stress
    ax1.plot(r_vals * 1000, sigma_r_el, 'b-', linewidth=2, label='Elastic')
    ax1.plot(r_vals * 1000, sigma_r_ep, 'r--', linewidth=2, label='Elastoplastic')
    ax1.set_xlabel('Radius (mm)', fontsize=12)
    ax1.set_ylabel('Radial Stress (MPa)', fontsize=12)
    ax1.set_title('Radial Stress Distribution', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Hoop stress
    ax2.plot(r_vals * 1000, sigma_t_el, 'b-', linewidth=2, label='Elastic')
    ax2.plot(r_vals * 1000, sigma_t_ep, 'r--', linewidth=2, label='Elastoplastic')
    ax2.set_xlabel('Radius (mm)', fontsize=12)
    ax2.set_ylabel('Hoop Stress (MPa)', fontsize=12)
    ax2.set_title('Hoop Stress Distribution', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add plastic zone indicator
    if result['r_p_vonmises'] is not None:
        rp = result['r_p_vonmises'] * 1000
        for ax in [ax1, ax2]:
            ax.axvline(x=rp, color='green', linestyle=':', alpha=0.7,
                        label=f'r_p = {result["r_p_vonmises"]*1000:.1f} mm')
            ax.legend(fontsize=10)

    fig.suptitle(f'Thick-Walled Cylinder: a={result["a"]*1000:.0f}mm, b={result["b"]*1000:.0f}mm, p={result["p"]/1e6:.0f}MPa',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'thick_cylinder.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return fig_to_base64(fig)


def generate_unloading_figure():
    """C3/C4: Unloading and re-yield path."""
    mat = J2FlowMaterial(E=210e9, nu=0.3, sigma_y0=250e6, H=2e9)
    result = unloading_path(mat, eps_load_max=0.005, eps_unload=0.0, eps_reload_max=0.008)

    fig, ax = plt.subplots(figsize=(8, 5))

    eps_load = np.array(result['eps_load'])
    sigma_load = np.array(result['sigma_load']) / 1e6
    eps_un = np.array(result['eps_unload'])
    sigma_un = np.array(result['sigma_unload']) / 1e6
    eps_re = np.array(result['eps_reload'])
    sigma_re = np.array(result['sigma_reload']) / 1e6

    ax.plot(eps_load * 100, sigma_load, 'b-', linewidth=2, label='Loading')
    ax.plot(eps_un * 100, sigma_un, 'g-', linewidth=2, label='Unloading')
    ax.plot(eps_re * 100, sigma_re, 'r-', linewidth=2, label='Reloading')

    # Mark key points
    ax.axhline(y=result['historical_max_stress'] / 1e6, color='orange', linestyle='--', alpha=0.5,
               label=f'Historical max: {result["historical_max_stress"]/1e6:.1f} MPa')
    ax.axhline(y=result['reyield_stress'] / 1e6, color='purple', linestyle=':', alpha=0.7,
               label=f'Re-yield: {result["reyield_stress"]/1e6:.1f} MPa')
    ax.axvline(x=result['residual_strain'] * 100, color='gray', linestyle='-.', alpha=0.5,
               label=f'Residual strain: {result["residual_strain"]*100:.4f}%')

    ax.set_xlabel('Strain (%)', fontsize=12)
    ax.set_ylabel('Stress (MPa)', fontsize=12)
    ax.set_title('J2 Flow Theory: Loading-Unloading-Reloading Path', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'unloading.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return fig_to_base64(fig)


if __name__ == '__main__':
    print("Generating J2 flow theory verification figures...")
    b64_1 = generate_stress_strain_figure()
    b64_2 = generate_thick_cylinder_figure()
    b64_3 = generate_unloading_figure()
    print("All figures generated successfully.")
