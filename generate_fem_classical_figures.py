#!/usr/bin/env python3
"""Generate verification figures for IN99-N02 FEM classical structural verification."""

import base64
import io
import json
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.fem_classical import (
    EulerBernoulliBeam2D,
    KirchhoffPlateSSSS,
    CylindricalShellBuckling,
    cantilever_tip_deflection_udl,
    navier_square_plate_center_deflection,
    classical_cylindrical_shell_buckling_stress,
)

# Parameters
BEAM_L = 2.0
BEAM_b = 0.05
BEAM_h = 0.01
BEAM_I = BEAM_b * BEAM_h ** 3 / 12.0
BEAM_E = 2.06e11
BEAM_q = 500.0

PLATE_a = 1.0
PLATE_t = 0.01
PLATE_E = 2.06e11
PLATE_nu = 0.3
PLATE_q = 1.0e4

SHELL_R = 0.5
SHELL_L = 2.0
SHELL_t = 0.002
SHELL_E = 2.06e11
SHELL_nu = 0.3


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_beam_convergence():
    """Figure 1: Beam bending moment L2 error convergence."""
    n_elems = [4, 8, 16, 32, 64]
    errors = []
    for n in n_elems:
        beam = EulerBernoulliBeam2D(BEAM_L, BEAM_E, BEAM_I, n)
        u = beam.solve_udl(BEAM_q)
        L_e = beam.le
        err_sq = 0.0
        for e in range(n):
            x0 = e * L_e
            w0, th0 = u[2*e], u[2*e+1]
            w1, th1 = u[2*e+2], u[2*e+3]
            w_pp_0 = 6*(w1-w0)/L_e**2 - 2*(2*th0+th1)/L_e
            w_pp_1 = 6*(w0-w1)/L_e**2 + 2*(th0+2*th1)/L_e
            gp = np.array([0.5 - np.sqrt(3/5)/2, 0.5, 0.5 + np.sqrt(3/5)/2])
            gw = np.array([5/18, 8/18, 5/18])
            for i in range(3):
                xi = gp[i]
                x = x0 + xi * L_e
                w_pp = w_pp_0 + (w_pp_1 - w_pp_0) * xi
                M_fem = beam.E * beam.I * w_pp
                M_ex = BEAM_q * (BEAM_L - x)**2 / 2
                err_sq += gw[i] * L_e * (M_fem - M_ex)**2
        errors.append(np.sqrt(err_sq))

    fig, ax = plt.subplots(figsize=(7, 5))
    h_vals = np.array([BEAM_L / n for n in n_elems])
    ax.loglog(h_vals, errors, "bo-", linewidth=2, markersize=8, label="FEM L2 error")
    # Reference O(h²) line
    ref = errors[0] * (h_vals / h_vals[0]) ** 2
    ax.loglog(h_vals, ref, "r--", linewidth=1.5, label="O(h²) reference")
    ax.set_xlabel("Element size h [m]", fontsize=12)
    ax.set_ylabel("L2 norm of moment error [N·m]", fontsize=12)
    ax.set_title("Euler-Bernoulli Beam: Bending Moment Convergence (UDL)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    slope = np.polyfit(np.log(h_vals), np.log(errors), 1)[0]
    ax.text(0.05, 0.95, f"Slope = {slope:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    return fig


def make_plate_comparison():
    """Figure 2: Plate center deflection convergence."""
    n_elems = [4, 8, 16, 32]
    w_fems = []
    w_refs = []
    for n in n_elems:
        plate = KirchhoffPlateSSSS(PLATE_a, PLATE_t, PLATE_E, PLATE_nu, n)
        w_fems.append(plate.center_deflection(PLATE_q))
        w_refs.append(plate.navier_reference(PLATE_q))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(n_elems))
    width = 0.35
    ax.bar(x - width/2, [w*1000 for w in w_fems], width, label="FEM", color="steelblue")
    ax.bar(x + width/2, [w*1000 for w in w_refs], width, label="Navier exact", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}×{n}" for n in n_elems])
    ax.set_ylabel("Center deflection [mm]", fontsize=12)
    ax.set_title("Kirchhoff Plate: Center Deflection vs Navier Solution", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    # Add error annotations
    for i, (wf, wr) in enumerate(zip(w_fems, w_refs)):
        err = abs(wf/wr - 1) * 100
        ax.annotate(f"{err:.1f}%", xy=(i, max(wf, wr)*1000),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color="darkred")
    plt.tight_layout()
    return fig


def make_shell_buckling():
    """Figure 3: Shell buckling stress vs circumferential mode."""
    shell = CylindricalShellBuckling(
        SHELL_R, SHELL_L, SHELL_t, SHELL_E, SHELL_nu,
        n_axial=32, n_circ=60
    )
    sigma_classical = shell.classical_buckling_stress()

    n_vals = list(range(5, 61))
    sigma_vals = []
    for n in n_vals:
        N_cr = shell.analytical_Ncr_for_mode(13, n)  # Use m=13 (near optimal)
        sigma_vals.append(N_cr / shell.t)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_vals, [s/1e6 for s in sigma_vals], "b-", linewidth=2, label="Analytical Donnell")
    ax.axhline(y=sigma_classical/1e6, color="r", linestyle="--", linewidth=1.5,
               label=f"Classical formula ({sigma_classical/1e6:.1f} MPa)")
    ax.set_xlabel("Circumferential wave number n", fontsize=12)
    ax.set_ylabel("Critical stress σ_cr [MPa]", fontsize=12)
    ax.set_title("Cylindrical Shell: Buckling Stress vs Circumferential Mode", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # Mark minimum
    min_idx = np.argmin(sigma_vals)
    ax.plot(n_vals[min_idx], sigma_vals[min_idx]/1e6, "go", markersize=10,
            label=f"Min at n={n_vals[min_idx]}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def main():
    figures = []

    for name, maker in [
        ("beam_convergence", make_beam_convergence),
        ("plate_comparison", make_plate_comparison),
        ("shell_buckling", make_shell_buckling),
    ]:
        fig = maker()
        png_b64 = fig_to_base64(fig)
        figures.append({"name": name, "png_base64": png_b64})
        plt.close(fig)
        print(f"Generated {name}")

    # Save figures data
    with open("/tmp/figures_data.json", "w") as f:
        json.dump(figures, f)
    print("Figures saved to /tmp/figures_data.json")


if __name__ == "__main__":
    main()
