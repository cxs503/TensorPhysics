"""Generate publication-quality figures for geometric nonlinear verification."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensorphysics.geometric_nonlinear.elastica import ElasticaExact
from tensorphysics.geometric_nonlinear.elastica_fem import ElasticaFEMSolver
from tensorphysics.geometric_nonlinear.von_karman_plate import VonKarmanPlateExact
from tensorphysics.geometric_nonlinear.shallow_arch import VonMisesTruss

rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'figure.figsize': (6, 4.5), 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts', 'm9459')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig1_elastica():
    """Elastica load-deflection curve."""
    fig, ax = plt.subplots()
    exact = ElasticaExact(L=1.0, EI=1.0)
    
    # Exact solution (fewer points for speed)
    theta_B_arr = np.linspace(0.05, 1.4, 20)
    P_norm, delta_L = [], []
    for tb in theta_B_arr:
        sol = exact.compute_from_theta_B(tb)
        P_norm.append(sol.P)
        delta_L.append(sol.y_tip)
    
    ax.plot(delta_L, P_norm, 'b-', linewidth=2, label='Exact (elliptic integrals)')
    
    # FEM points
    P_fem = np.array([0.5, 1.0, 2.0, 3.5, 6.0])
    delta_fem = []
    for P in P_fem:
        solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=80)
        result = solver.solve(P)
        delta_fem.append(result.tip_deflection)
    delta_fem = np.array(delta_fem)
    
    ax.plot(delta_fem, P_fem, 'ro', markersize=6, label='FEM (n=80)')
    
    # Linear theory
    d_lin = np.linspace(0, 0.8, 50)
    ax.plot(d_lin, 3*d_lin, 'k--', alpha=0.5, label='Linear theory')
    
    ax.set_xlabel(r'Tip deflection $\delta/L$')
    ax.set_ylabel(r'Load $PL^2/EI$')
    ax.set_title('Cantilever Elastica: Load-Deflection')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.85); ax.set_ylim(0, 8)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_elastica_load_deflection.png'))
    plt.close(fig)
    print("Fig 1 saved")


def fig2_shapes():
    """Elastica deformed shapes."""
    fig, ax = plt.subplots()
    exact = ElasticaExact(L=1.0, EI=1.0)
    
    for P, color in zip([0.5, 1.5, 3.0, 5.0], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
        sol = exact.compute_shape_at_load(P)
        ax.plot(sol.x_array, sol.y_array, color=color, linewidth=1.5, label=f'P={P:.1f}')
    
    ax.plot(0, 0, 'ks', markersize=8, label='Clamp')
    ax.set_xlabel(r'x/L'); ax.set_ylabel(r'y/L')
    ax.set_title('Elastica: Deformed Shapes')
    ax.legend(loc='lower left'); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal'); ax.invert_yaxis()
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_elastica_shapes.png'))
    plt.close(fig)
    print("Fig 2 saved")


def fig3_convergence():
    """FEM convergence study."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    P = 2.0
    exact = ElasticaExact(L=1.0, EI=1.0)
    sol_exact = exact.compute_shape_at_load(P)
    delta_exact = sol_exact.y_tip
    
    n_list = [10, 20, 40, 80, 160]
    errors, h_vals = [], []
    for n in n_list:
        solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=n)
        result = solver.solve(P)
        errors.append(abs(result.tip_deflection - delta_exact) / delta_exact * 100)
        h_vals.append(1.0 / n)
    
    h_arr = np.array(h_vals)
    err_arr = np.array(errors)
    
    ax1.loglog(h_arr, err_arr, 'bo-', linewidth=2, markersize=6)
    h_ref = np.array([0.005, 0.15])
    ax1.loglog(h_ref, h_ref**2 * err_arr[0]/h_vals[0]**2, 'r--', alpha=0.5, label=r'$O(h^2)$')
    ax1.set_xlabel('Element size h/L'); ax1.set_ylabel('Relative error (%)')
    ax1.set_title('Convergence: Error vs h'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    log_h = np.log(h_arr); log_err = np.log(err_arr/100)
    p = np.polyfit(log_h, log_err, 1)[0]
    ax2.plot(log_h, log_err, 'bo', markersize=6)
    ax2.plot(log_h, np.polyval(np.polyfit(log_h, log_err, 1), log_h), 'r-',
             label=f'Slope = {abs(p):.2f}')
    ax2.set_xlabel('ln(h/L)'); ax2.set_ylabel('ln(error)')
    ax2.set_title(f'Convergence Order: O(h^{abs(p):.2f})')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_elastica_convergence.png'))
    plt.close(fig)
    print("Fig 3 saved")


def fig4_plate():
    """Von Karman plate load-deflection."""
    fig, ax = plt.subplots()
    ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
    
    w0_h = np.linspace(0.1, 6.0, 80)
    _, P_norm, _ = ref.compute_load_deflection_curve(w0_h)
    
    ax.plot(w0_h, P_norm, 'b-', linewidth=2, label='Von Karman (nonlinear)')
    ax.plot(w0_h, w0_h, 'k--', alpha=0.5, label='Linear')
    
    # Camp/Berger points
    w0_pts = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    c = 0.376
    P_camp = w0_pts + c * w0_pts**3
    ax.plot(w0_pts, P_camp, 'ro', markersize=6, label='Camp/Berger ref.')
    
    ax.set_xlabel(r'$w_0/h$'); ax.set_ylabel(r'$Pa^2/(16\pi Dh)$')
    ax.set_title('Circular Plate: Von Karman Large Deflection')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_plate_load_deflection.png'))
    plt.close(fig)
    print("Fig 4 saved")


def fig5_arch():
    """Shallow arch snap-through."""
    fig, ax = plt.subplots()
    
    L, H, EA = 10.0, 1.0, 1e9
    truss = VonMisesTruss(L=L, H=H, EA=EA)
    
    v_range = np.linspace(-0.3*H, 3.0*H, 300)
    P, stability = truss.equilibrium_path(v_range)
    
    stable = stability == 'stable'
    unstable = stability == 'unstable'
    
    ax.plot(v_range[stable]/H, P[stable], 'b-', linewidth=2, label='Stable')
    ax.plot(v_range[unstable]/H, P[unstable], 'r-', linewidth=2, label='Unstable')
    
    result = truss.find_limit_points(n_pts=500)
    ax.plot(result['w_at_upper']/H, result['q_limit_upper'], 'k^', markersize=10,
            label=f'Upper limit')
    ax.plot(result['w_at_lower']/H, result['q_limit_lower'], 'kv', markersize=10,
            label=f'Lower limit')
    
    ax.set_xlabel(r'$w/H$'); ax.set_ylabel('Load P')
    ax.set_title('Von Mises Truss: Snap-Through Path')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_arch_snap_through.png'))
    plt.close(fig)
    print("Fig 5 saved")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_elastica()
    fig2_shapes()
    fig3_convergence()
    fig4_plate()
    fig5_arch()
    print(f"All figures saved to {OUTPUT_DIR}")
