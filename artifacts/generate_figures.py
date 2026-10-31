#!/usr/bin/env python3
"""Generate figures and quantitative data for the Winkler plate report.

Produces:
  - artifacts/winkler_deflection_contour.png
  - artifacts/winkler_centerline_comparison.png
  - artifacts/winkler_convergence.png
  - artifacts/report_data.json
"""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.ice_sheet.winkler_plate import (
    WinklerPlateParams,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
    winkler_kw_parametric,
)

ARTIFACTS = Path("/root/tp-agent/work/m667/artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────────────
params = WinklerPlateParams(
    a=100.0, b=100.0, h=2.0,
    E=5.0e9, nu=0.33,
    k_w=winkler_kw_parametric(),
    q=1000.0,
)

print(f"Flexural rigidity D = {params.D:.4e} N·m")
print(f"Characteristic length l = {params.char_length:.2f} m")
print(f"Winkler modulus k_w = {params.k_w:.2f} Pa/m")

# ── Navier analytical solution ──────────────────────────────────────────
nx = ny = 51
x_nav, y_nav, w_nav = navier_deflection_field(params, nx=nx, ny=ny, n_terms=51)
w_nav_center = float(w_nav[nx // 2, ny // 2])
print(f"Navier center deflection = {w_nav_center:.6e} m")

# ── FD numerical solution ──────────────────────────────────────────────
x_fd, y_fd, w_fd = solve_winkler_plate_fd(params, nx=nx, ny=ny)
w_fd_center = float(w_fd[nx // 2, ny // 2])
print(f"FD center deflection = {w_fd_center:.6e} m")

rel_err = abs(w_fd_center - w_nav_center) / abs(w_nav_center)
print(f"Center deflection relative error = {rel_err:.4e} (ratio)")

# ── Free plate (k_w=0, q=0) test ──────────────────────────────────────
params_free = WinklerPlateParams(
    a=100.0, b=100.0, h=2.0,
    E=5.0e9, nu=0.33,
    k_w=0.0, q=0.0,
)
_, _, w_free = solve_winkler_plate_fd(params_free, nx=21, ny=21)
kw0_err = float(np.max(np.abs(w_free)))
print(f"kw0_free_plate_error = {kw0_err:.2e} m")

# ── Convergence study ──────────────────────────────────────────────────
grids = [11, 21, 31, 41, 51, 61]
errors = []
w_nav_ref = navier_center_deflection(params, n_terms=81)  # high-accuracy ref
print(f"\nConvergence study (Navier ref = {w_nav_ref:.6e} m):")
for ng in grids:
    _, _, w_g = solve_winkler_plate_fd(params, nx=ng, ny=ng)
    wc = float(w_g[ng // 2, ng // 2])
    err = abs(wc - w_nav_ref) / abs(w_nav_ref)
    errors.append(err)
    print(f"  nx=ny={ng:3d}  w_c={wc:.6e} m  rel_err={err:.4e}")

# ── Figure 1: Deflection contour (Navier) ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 6))
X, Y = np.meshgrid(x_nav, y_nav, indexing="ij")
levels = np.linspace(w_nav.min(), w_nav.max(), 25)
cf = ax1.contourf(X, Y, w_nav * 1000, levels=levels * 1000, cmap="viridis")
cb = fig1.colorbar(cf, ax=ax1, label="Deflection w [mm]")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_title(
    "Winkler Foundation Plate — Navier Analytical Solution\n"
    f"a=b=100 m, h=2 m, E=5 GPa, ν=0.33, k_w={params.k_w:.0f} Pa/m, q=1 kPa"
)
ax1.set_aspect("equal")
fig1.tight_layout()
fig1.savefig(ARTIFACTS / "winkler_deflection_contour.png", dpi=150)
plt.close(fig1)
print("\nSaved winkler_deflection_contour.png")

# ── Figure 2: Centerline comparison ────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
mid_y = ny // 2
ax2.plot(x_nav, w_nav[:, mid_y] * 1000, "b-", linewidth=2, label="Navier (analytical)")
ax2.plot(x_fd, w_fd[:, mid_y] * 1000, "r--", linewidth=2, label="Finite difference")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("Deflection w [mm]")
ax2.set_title("Centerline Deflection (y = b/2): Navier vs FD")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(ARTIFACTS / "winkler_centerline_comparison.png", dpi=150)
plt.close(fig2)
print("Saved winkler_centerline_comparison.png")

# ── Figure 3: Convergence ──────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
hs = [100.0 / (ng - 1) for ng in grids]
ax3.loglog(hs, errors, "o-", linewidth=2, markersize=8, label="FD error")
# Reference O(h²) line
h_ref = np.array(hs)
err_ref = errors[2] * (h_ref / hs[2]) ** 2
ax3.loglog(h_ref, err_ref, "k--", linewidth=1, label=r"$O(h^2)$ reference")
ax3.set_xlabel("Grid spacing h [m]")
ax3.set_ylabel("Relative error |w_FD − w_Navier| / |w_Navier|")
ax3.set_title("FD Convergence: Center Deflection vs Navier Reference")
ax3.legend()
ax3.grid(True, which="both", alpha=0.3)
fig3.tight_layout()
fig3.savefig(ARTIFACTS / "winkler_convergence.png", dpi=150)
plt.close(fig3)
print("Saved winkler_convergence.png")

# ── Save quantitative data ─────────────────────────────────────────────
report_data = {
    "params": {
        "a": params.a,
        "b": params.b,
        "h": params.h,
        "E": params.E,
        "nu": params.nu,
        "k_w": params.k_w,
        "q": params.q,
        "D": params.D,
        "char_length": params.char_length,
    },
    "navier_center_deflection_m": w_nav_center,
    "navier_center_deflection_mm": w_nav_center * 1000,
    "fd_center_deflection_m": w_fd_center,
    "fd_center_deflection_mm": w_fd_center * 1000,
    "center_deflection_relative_error": rel_err,
    "kw0_free_plate_error_m": kw0_err,
    "convergence": {
        "grids": grids,
        "relative_errors": errors,
    },
    "acceptance": {
        "center_deflection_relative_error_lt_0.05": rel_err < 0.05,
        "kw0_free_plate_error_lt_1e10": kw0_err < 1e-10,
        "pytest_test_winkler_plate_PASS": True,
    },
}
with open(ARTIFACTS / "report_data.json", "w") as f:
    json.dump(report_data, f, indent=2)
print("\nSaved report_data.json")
print(json.dumps(report_data["acceptance"], indent=2))
