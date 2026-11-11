#!/usr/bin/env python
"""P226W3-N03 report generator: Pasternak two-parameter elastic foundation for axisymmetric plate FEM.

Generates:
- Deflection profile comparison (FEM vs analytical)
- Load vs G_p curve (shear-layer stiffening)
- Characteristic length scaling diagram
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensorphysics.fem_ice_plate.foundation_plate import (
    plate_flexural_rigidity,
    solve_axisym_plate_foundation,
    pasternak_char_length,
    pasternak_shear_length,
    pasternak_coupling,
    hetenyi_axisym_point_load,
)
from tensorphysics.elastic_foundation.pasternak_plate import pasternak_plate_patch_load

# Parameters
E_ICE = 5.0e9
NU_ICE = 0.33
H_ICE = 1.0
K_W = 1025.0 * 9.81
D = plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
L_C = pasternak_char_length(D, K_W)
PATCH_A = 0.05 * L_C
W_TARGET = 0.001
W_STEPS = np.linspace(0.0, W_TARGET, 5)[1:]

OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts", "p226w3_n03")
os.makedirs(OUTDIR, exist_ok=True)


def fig1_deflection_profile():
    """Figure 1: Deflection profile comparison (FEM vs analytical) for Winkler and Pasternak."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, gamma, title in zip(axes, [0.0, 1.0], ["Winkler (γ=0)", "Pasternak (γ=1)"]):
        G_p = gamma * math.sqrt(D * K_W)
        res = solve_axisym_plate_foundation(
            D, NU_ICE, K_W, PATCH_A, total_load=1.0, n_el=80, G_p=G_p,
            w_center_steps=W_STEPS)
        P_fem = res.total_load[-1]

        r_test = np.linspace(0.0, 5.0 * L_C, 101)
        w_ana = pasternak_plate_patch_load(r_test, P_fem, PATCH_A, D, K_W, G_p)
        w_fem = np.interp(r_test, res.r, res.w)

        ax.plot(r_test / L_C, w_ana * 1e3, "k-", linewidth=2, label="Analytical")
        ax.plot(r_test / L_C, w_fem * 1e3, "r--", linewidth=1.5, label="FEM")
        ax.set_xlabel("r / l_c")
        ax.set_ylabel("w [mm]")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig1_deflection_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig2_load_vs_gp():
    """Figure 2: Load required for fixed deflection vs G_p (shear-layer stiffening)."""
    gammas = np.linspace(0.0, 1.8, 19)
    loads = []
    for gamma in gammas:
        G_p = gamma * math.sqrt(D * K_W)
        res = solve_axisym_plate_foundation(
            D, NU_ICE, K_W, PATCH_A, total_load=1.0, n_el=60, G_p=G_p,
            w_center_steps=W_STEPS)
        loads.append(res.total_load[-1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, np.array(loads) / 1e3, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("Dimensionless coupling γ = G_p / √(Dk)")
    ax.set_ylabel("Load for w(0) = 1 mm [kN]")
    ax.set_title("Shear-layer stiffening: load increases with G_p")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=2.0, color="r", linestyle="--", alpha=0.5, label="γ = 2 (degenerate)")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig2_load_vs_gp.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig3_length_scales():
    """Figure 3: Characteristic length scales vs G_p."""
    Gp_vals = np.logspace(3, 7, 50)
    ls_vals = [pasternak_shear_length(Gp, K_W) for Gp in Gp_vals]
    gamma_vals = [pasternak_coupling(D, K_W, Gp) for Gp in Gp_vals]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogx(Gp_vals, ls_vals, "b-", linewidth=2, label="l_s = √(G_p/k)")
    ax1.axhline(y=L_C, color="r", linestyle="--", linewidth=1.5, label=f"l_c = (D/k)^¼ = {L_C:.1f} m")
    ax1.set_xlabel("G_p [N/m]")
    ax1.set_ylabel("Length [m]", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogx(Gp_vals, gamma_vals, "g--", linewidth=1.5, label="γ = G_p/√(Dk)")
    ax2.set_ylabel("γ", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.axhline(y=2.0, color="r", linestyle=":", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Characteristic length scales vs shear-layer modulus")

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig3_length_scales.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    print("Generating P226W3-N03 report figures...")
    fig1_deflection_profile()
    fig2_load_vs_gp()
    fig3_length_scales()
    print("Done.")
