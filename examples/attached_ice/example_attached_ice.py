"""Example: Underwater attached ice growth and resistance penalty.

This example demonstrates the full chain from frazil flux to resistance penalty.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorphysics.ice.attached_ice import (
    RHO_ICE, K_ICE, L_FUSION, T_FREEZE_FW,
    DEFAULT_BOND_STRENGTH_PA, DEFAULT_HULL_ROUGHNESS_M, DEFAULT_SHIP_LENGTH_M,
    frazil_impact_flux,
    IceGrowthODE,
    stefan_analytic,
    detachment_state,
    roughness_state,
    p50_p90_envelope,
    adhesive_strength,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_growth_curves():
    """Figure 1: Ice growth curves for different undercooling."""
    fig, ax = plt.subplots(figsize=(10, 6))

    undercoolings = [0.5, 1.0, 2.0, 3.0, 5.0]
    t_max = 7200.0

    for dT in undercoolings:
        ode = IceGrowthODE(
            undercooling_k=dT,
            u_star=0.02,
            c_f=1e-5,
            eta_stick=1.0,
        )
        result = ode.solve_euler(t_max=t_max, n_steps=3000)

        # Also plot Stefan analytic for comparison
        t_stefan = np.linspace(0, t_max, 200)
        h_stefan = np.array([stefan_analytic(dT, t) for t in t_stefan])

        ax.plot(result.t / 3600, result.h_ice * 1000, '-',
                label=f'dT={dT}K (ODE)', linewidth=2)
        ax.plot(t_stefan / 3600, h_stefan * 1000, '--',
                label=f'dT={dT}K (Stefan)', alpha=0.5)

    ax.set_xlabel('Time [hours]', fontsize=12)
    ax.set_ylabel('Ice thickness [mm]', fontsize=12)
    ax.set_title('Underwater Attached Ice Growth: ODE vs Stefan Analytic',
                 fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "growth_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'growth_curves.png'}")


def plot_detachment_envelope():
    """Figure 2: Detachment envelope (velocity vs ice thickness)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_bond_values = [50e3, 100e3, 200e3, 500e3]
    h_values = np.linspace(0.001, 0.05, 200)

    for tau_bond in tau_bond_values:
        u_detach = np.array([
            np.sqrt(adhesive_strength(h, tau_bond) / 1025)
            for h in h_values
        ])
        ax.plot(h_values * 1000, u_detach, '-',
                label=f'tau_bond={tau_bond/1e3:.0f} kPa', linewidth=2)

    ax.set_xlabel('Ice thickness [mm]', fontsize=12)
    ax.set_ylabel('Critical friction velocity [m/s]', fontsize=12)
    ax.set_title('Detachment Envelope: Velocity vs Ice Thickness', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "detachment_envelope.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'detachment_envelope.png'}")


def plot_roughness_penalty():
    """Figure 3: ITTC-1978 roughness penalty vs ice thickness."""
    fig, ax = plt.subplots(figsize=(10, 6))

    h_values = np.linspace(0, 0.02, 200)
    delta_cf_values = []
    delta_t_values = []

    for h in h_values:
        state = roughness_state(h_ice=h)
        delta_cf_values.append(state.delta_cf)
        delta_t_values.append(state.delta_t_over_t)

    ax.plot(h_values * 1000, np.array(delta_cf_values) * 1e6, '-',
            label='Delta C_f [ppm]', color='blue', linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(h_values * 1000, np.array(delta_t_values) * 100, '-',
             label='Delta T/T [%]', color='red', linewidth=2)

    ax.set_xlabel('Ice thickness [mm]', fontsize=12)
    ax.set_ylabel('Delta C_f [ppm]', fontsize=12, color='blue')
    ax2.set_ylabel('Thrust loss [%]', fontsize=12, color='red')
    ax.set_title('ITTC-1978 Roughness Penalty vs Ice Thickness', fontsize=14)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roughness_penalty.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'roughness_penalty.png'}")


def plot_p50_p90_envelope():
    """Figure 4: P50/P90 envelope for probabilistic assessment."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Monte Carlo simulation
    np.random.seed(42)
    n_samples = 1000
    h_samples = np.random.lognormal(mean=-4.0, sigma=0.8, size=n_samples)
    h_samples = np.clip(h_samples, 0.0001, 0.02)

    env = p50_p90_envelope(h_samples)

    # Plot histogram
    ax.hist(h_samples * 1000, bins=50, alpha=0.6, color='steelblue',
            edgecolor='black', label='Ice thickness samples')

    # Mark P50 and P90
    ax.axvline(env["k_s_p50"] * 1000, color='green', linestyle='--',
               linewidth=2, label=f'P50 k_s = {env["k_s_p50"]*1000:.2f} mm')
    ax.axvline(env["k_s_p90"] * 1000, color='red', linestyle='--',
               linewidth=2, label=f'P90 k_s = {env["k_s_p90"]*1000:.2f} mm')

    ax.set_xlabel('Equivalent sand roughness k_s [mm]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('P50/P90 Envelope: Probabilistic Roughness Assessment',
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "p50_p90_envelope.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'p50_p90_envelope.png'}")


if __name__ == "__main__":
    print("Generating attached ice figures...")
    plot_growth_curves()
    plot_detachment_envelope()
    plot_roughness_penalty()
    plot_p50_p90_envelope()
    print("Done!")
