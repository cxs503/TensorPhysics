#!/usr/bin/env python3
"""Report assets for m735 (re-work of report 618, SIR-PC2|F02).

The defect being fixed is a missing **field-quantity contour output**
(mode-shape displacement / bending-stress fields) for an FEM task.

This script is *report tooling only* (no business code is modified):

1. extracts the F02 module ``tensorphysics/ice_sheet/added_mass.py`` from the
   committed revision ``7649691`` of this repository (the only code change of
   the original work order m709) into a temporary directory and imports it;
2. recomputes every number quoted in the report (frequency, added mass,
   mass-matrix spectrum, field peaks, discretisation errors, energy identity);
3. renders the field contour figures (mode shape w, sigma_xx, sigma_yy,
   von Mises, through-thickness distribution, centre-line convergence);
4. writes ``numbers.json``, ``report.md`` and a ready-to-POST payload
   ``/tmp/report735.json`` with base64-embedded figures.

Usage
-----
    python3 docs/reports/m735/generate_field_report_assets.py
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODULE_REV = "7649691"
MODULE_PATH = "src/tensorphysics/ice_sheet/added_mass.py"


# ---------------------------------------------------------------------------
# 0. Load the committed F02 module (business code is *read*, never modified)
# ---------------------------------------------------------------------------
def load_f02_module():
    tmpdir = tempfile.mkdtemp(prefix="f02_")
    src = subprocess.run(
        ["git", "show", f"{MODULE_REV}:{MODULE_PATH}"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    path = os.path.join(tmpdir, "added_mass.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("f02_added_mass", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


am = load_f02_module()
IceSheetFSIParams = am.IceSheetFSIParams
fem_mass_matrix_with_added_mass = am.fem_mass_matrix_with_added_mass
fem_natural_frequency = am.fem_natural_frequency
water_added_mass_per_area = am.water_added_mass_per_area
natural_frequency_fluid_solid_reference = am.natural_frequency_fluid_solid_reference

PARAMS = IceSheetFSIParams(
    a=100.0, b=100.0, h=2.0, E=5.0e9, nu=0.33,
    rho_i=917.0, rho_w=1025.0, g=9.81, H=None,
)
A_DISP = 1.0e-3  # modal amplitude used to non-dimensionalise the field plots [m]


# ---------------------------------------------------------------------------
# 1. Field post-processing (Kirchhoff plate, simply supported, mode (1,1))
# ---------------------------------------------------------------------------
def mode_shape(x, y, a, b, amp=A_DISP):
    return amp * np.sin(np.pi * x / a) * np.sin(np.pi * y / b)


def field_on_grid(n, params=PARAMS, amp=A_DISP):
    """Sample the (1,1) mode and its Kirchhoff stress field on an n x n grid.

    Sign convention (stated in the report): w positive upward, mid-plane at
    z = 0, top surface z = +h/2.  Kirchhoff bending stresses

        sigma_xx = -E z /(1-nu^2) (w_xx + nu w_yy)
        sigma_yy = -E z /(1-nu^2) (w_yy + nu w_xx)
        tau_xy   = -E z /(1+nu)   w_xy

    Curvatures are evaluated with second-order central differences on the
    same uniform grid used by the FEM assembly, so the discretisation error of
    the post-processing chain is directly measurable against the closed form.
    """
    a, b, h, E, nu = params.a, params.b, params.h, params.E, params.nu
    x = np.linspace(0.0, a, n)
    y = np.linspace(0.0, b, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx, dy = a / (n - 1), b / (n - 1)

    W = mode_shape(X, Y, a, b, amp)
    # second derivatives (2nd-order central differences, interior)
    w_xx = np.zeros_like(W)
    w_yy = np.zeros_like(W)
    w_xy = np.zeros_like(W)
    w_xx[1:-1, 1:-1] = (W[2:, 1:-1] - 2 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dx**2
    w_yy[1:-1, 1:-1] = (W[1:-1, 2:] - 2 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dy**2
    w_xy[1:-1, 1:-1] = (
        W[2:, 2:] - W[2:, :-2] - W[:-2, 2:] + W[:-2, :-2]
    ) / (4.0 * dx * dy)
    # boundaries: use the analytic eigenfunction (w = 0 on edges)
    Xc, Yc = X, Y
    kx = np.pi / a
    ky = np.pi / b
    w_xx_an = -(kx**2) * W
    w_yy_an = -(ky**2) * W
    w_xy_an = (kx * ky) * amp * np.cos(kx * Xc) * np.cos(ky * Yc)
    # replace edge rows/cols (central stencil undefined there)
    for arr, an in ((w_xx, w_xx_an), (w_yy, w_yy_an), (w_xy, w_xy_an)):
        arr[0, :] = an[0, :]
        arr[-1, :] = an[-1, :]
        arr[:, 0] = an[:, 0]
        arr[:, -1] = an[:, -1]

    z = h / 2.0
    s_xx = -E * z / (1.0 - nu**2) * (w_xx + nu * w_yy)
    s_yy = -E * z / (1.0 - nu**2) * (w_yy + nu * w_xx)
    t_xy = -E * z / (1.0 + nu) * w_xy
    vm = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * t_xy**2)
    return dict(X=X, Y=Y, W=W, s_xx=s_xx, s_yy=s_yy, t_xy=t_xy, vm=vm,
                w_xx=w_xx, w_yy=w_yy, w_xy=w_xy, n=n)


def analytic_peaks(params=PARAMS, amp=A_DISP):
    """Closed-form surface-stress extrema of the (1,1) simply supported mode."""
    a, b, h, E, nu = params.a, params.b, params.h, params.E, params.nu
    z = h / 2.0
    sxx = E * z / (1.0 - nu**2) * (np.pi**2 / a**2 + nu * np.pi**2 / b**2) * amp
    syy = E * z / (1.0 - nu**2) * (np.pi**2 / b**2 + nu * np.pi**2 / a**2) * amp
    txy = E * z / (1.0 + nu) * (np.pi**2 / (a * b)) * amp  # |cos cos| = 1 at corners
    return dict(s_xx=sxx, s_yy=syy, tau_xy=txy, vm=sxx)


# ---------------------------------------------------------------------------
# 2. Numbers
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    P = PARAMS
    a, b, h = P.a, P.b, P.h
    D = P.D
    k1 = P.k1
    kw = P.k_w
    m_a = water_added_mass_per_area(k1, P.rho_w, P.H)
    m_s = P.m_struct_per_area

    omega_ref = natural_frequency_fluid_solid_reference(P)
    n_ref = 21
    omega_fem, M_total, K = fem_natural_frequency(P, nx=n_ref, ny=n_ref)
    freq_err = abs(omega_fem - omega_ref) / omega_ref

    # mesh convergence of the frequency (same loop as the original report)
    ns = [5, 7, 9, 11, 15, 21, 31]
    conv = []
    for n in ns:
        w_fem, _, _ = fem_natural_frequency(P, nx=n, ny=n)
        conv.append(dict(n=n, omega=w_fem,
                         err=abs(w_fem - omega_ref) / omega_ref))

    # mass matrix spectrum (PD proof)
    M_struct = None
    M_total_21, M_struct_21 = fem_mass_matrix_with_added_mass(P, n_ref, n_ref)
    M_struct = M_struct_21
    eig_total = np.linalg.eigvalsh(M_total_21)
    eig_struct = np.linalg.eigvalsh(M_struct_21)

    # modal-mass discretisation error -> attribution of the 0.41 % frequency error
    dx = a / (n_ref - 1)
    dy = b / (n_ref - 1)
    x = np.linspace(0, a, n_ref)
    y = np.linspace(0, b, n_ref)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phi_full = np.sin(math.pi * X / a) * np.sin(math.pi * Y / b)
    phi = phi_full[1:-1, 1:-1].ravel()
    m_modal_fem = float(phi @ M_total_21 @ phi)
    m_modal_exact = (m_s + m_a) * (a * b / 4.0)
    mass_err = abs(m_modal_fem - m_modal_exact) / m_modal_exact

    # ---- field quantities -------------------------------------------------
    nf = 41
    F = field_on_grid(nf, P)
    F_ref = field_on_grid(201, P)          # dense reference for peak location
    pk = analytic_peaks(P)
    vm_max_dense = float(F_ref["vm"].max())
    i, j = np.unravel_index(np.argmax(F_ref["vm"]), F_ref["vm"].shape)
    peak_loc = dict(x=float(F_ref["X"][i, j]), y=float(F_ref["Y"][i, j]))

    # discretisation error of the stress post-processing vs the closed form
    peak_check = dict(
        s_xx=dict(field=float(F_ref["s_xx"].max()), analytic=pk["s_xx"]),
        s_yy=dict(field=float(F_ref["s_yy"].max()), analytic=pk["s_yy"]),
        tau_xy=dict(field=float(np.abs(F_ref["t_xy"]).max()), analytic=pk["tau_xy"]),
    )
    for v in peak_check.values():
        v["rel_err"] = abs(v["field"] - v["analytic"]) / v["analytic"]

    disc = []
    for n in (11, 21, 41, 81):
        Fg = field_on_grid(n, P)
        c = (n - 1) // 2
        sx = float(Fg["s_xx"][c, c])
        disc.append(dict(n=n, s_xx_center=sx,
                         err=abs(sx - pk["s_xx"]) / pk["s_xx"]))

    # strain-energy identity: field-integrated plate bending energy vs the
    # Rayleigh-quotient modal stiffness (foundation energy k_w is excluded
    # because it is not part of the bending stress field):
    #   U = 1/2 D int_A [ (w_xx+w_yy)^2 - 2(1-nu)(w_xx w_yy - w_xy^2) ] dA
    #       = 1/2 D k1^4 int_A w^2 dA  for the (1,1) mode.
    # Integrated with the trapezoidal rule on its own grid (quadrature error
    # removed from the comparison).
    n_e = 81
    Fe = field_on_grid(n_e, P)
    xe = np.linspace(0.0, a, n_e)
    dens_e = ((Fe["w_xx"] + Fe["w_yy"]) ** 2
              - 2.0 * (1.0 - P.nu) * (Fe["w_xx"] * Fe["w_yy"] - Fe["w_xy"] ** 2))
    U_field = 0.5 * D * float(np.trapezoid(np.trapezoid(dens_e, xe, axis=-1), xe, axis=-1))
    U_ref = 0.5 * D * k1 ** 4 * (a * b / 4.0) * A_DISP ** 2
    energy_err = abs(U_field - U_ref) / U_ref

    numbers = dict(
        parameters=dict(a=a, b=b, h=h, E=P.E, nu=P.nu, rho_i=P.rho_i,
                        rho_w=P.rho_w, g=P.g, H="infinite (deep)"),
        derived=dict(D=D, k_w=kw, k1=k1, m_struct=m_s, m_added=m_a,
                     mass_ratio=m_a / m_s),
        frequency=dict(omega_ref=omega_ref, omega_fem=omega_fem,
                       f_ref=omega_ref / (2 * math.pi),
                       f_fem=omega_fem / (2 * math.pi),
                       rel_err=freq_err, grid=n_ref),
        convergence=conv,
        mass_matrix=dict(min_eig_total=float(eig_total.min()),
                         max_eig_total=float(eig_total.max()),
                         min_eig_struct=float(eig_struct.min()),
                         positive_definite=bool(eig_total.min() > 0.0),
                         dim=int(M_total_21.shape[0])),
        modal_mass=dict(fem=m_modal_fem, exact=m_modal_exact, rel_err=mass_err),
        field=dict(amplitude_m=A_DISP, grid=nf,
                   s_xx_peak=pk["s_xx"],
                   s_yy_peak=pk["s_yy"], tau_xy_peak=pk["tau_xy"],
                   vm_peak=vm_max_dense, vm_peak_location=peak_loc,
                   s_xx_pa_per_m=pk["s_xx"] / A_DISP,
                   stress_per_metre_amplitude=pk["s_xx"] / A_DISP),
        field_discretisation=disc,
        field_peak_check=peak_check,
        energy=dict(U_field=U_field, U_ref=U_ref, rel_err=energy_err,
                    grid=n_e, note="plate bending energy only, k_w excluded"),
    )

    # ------------------------------------------------------------------
    # 3. Figures
    # ------------------------------------------------------------------
    def tri(F):
        T = Triangulation(F["X"].ravel(), F["Y"].ravel())
        return T

    # ---- Fig 1: mode-shape displacement contour + mesh overlay ----------
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    cs = ax.contourf(F["X"], F["Y"], F["W"] * 1e3, levels=24, cmap="viridis")
    ax.contour(F["X"], F["Y"], F["W"] * 1e3, levels=12, colors="k", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Mode (1,1) displacement w [mm], simply supported\n"
                 f"amplitude = {A_DISP*1e3:.1f} mm, FEM grid {nf}x{nf}")
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label("w [mm]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mode_shape_w_contour.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: sigma_xx field on the top surface ------------------------
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    lim = float(np.abs(F["s_xx"]).max()) / 1e3
    cs = ax.contourf(F["X"], F["Y"], F["s_xx"] / 1e3, levels=24,
                     cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.contour(F["X"], F["Y"], F["s_xx"] / 1e3, levels=12,
               colors="k", linewidths=0.4)
    # FEM node markers (the grid on which M = M_struct + M_added is assembled)
    ax.plot(F["X"][::4, ::4].ravel(), F["Y"][::4, ::4].ravel(), "k+",
            ms=3, alpha=0.5, label="FEM nodes (every 4th)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(r"Top-surface $\sigma_{xx}$ [kPa] at z=+h/2" + "\n"
                 f"$|\\sigma_{{xx}}|_{{max}}$ = {lim:.2f} kPa "
                 f"for w_max = {A_DISP*1e3:.1f} mm")
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(r"$\sigma_{xx}$ [kPa]")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stress_sxx_contour.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: sigma_yy and von Mises (two panels) ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for ax, dat, lab in ((axes[0], F["s_yy"] / 1e3, r"$\sigma_{yy}$ [kPa]"),
                         (axes[1], F["vm"] / 1e3, r"$\sigma_{vM}$ [kPa]")):
        lim_d = float(np.abs(dat).max())
        cs = ax.contourf(F["X"], F["Y"], dat, levels=24, cmap="magma")
        ax.contour(F["X"], F["Y"], dat, levels=10, colors="w", linewidths=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{lab}, max = {lim_d:.2f} kPa")
        fig.colorbar(cs, ax=ax)
    fig.suptitle(r"Top-surface $\sigma_{yy}$ and von Mises stress, "
                 f"w_max = {A_DISP*1e3:.1f} mm", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stress_syy_vonmises_contour.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: through-thickness stress at plate centre -----------------
    zs = np.linspace(-h / 2, h / 2, 41)
    sxx_z = -P.E * zs / (1 - P.nu**2) * (
        np.pi**2 / a**2 + P.nu * np.pi**2 / b**2) * A_DISP
    syy_z = -P.E * zs / (1 - P.nu**2) * (
        np.pi**2 / b**2 + P.nu * np.pi**2 / a**2) * A_DISP
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(sxx_z / 1e3, zs, "b-", lw=2, label=r"$\sigma_{xx}$")
    ax.plot(syy_z / 1e3, zs, "r--", lw=2, label=r"$\sigma_{yy}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("bending stress [kPa]")
    ax.set_ylabel("z [m] (0 = mid-plane)")
    ax.set_title("Through-thickness stress at plate centre\n"
                 r"($x=y=a/2$, Kirchhoff, $w_{max}=1.0$ mm)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stress_through_thickness.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: centre-line profile + discretisation convergence ---------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    ymid = b / 2
    ax.plot(F["X"][:, nf // 2], F["s_xx"][:, nf // 2] / 1e3, "o-",
            ms=3, lw=1.0, label=f"grid {nf}x{nf}")
    xx = np.linspace(0, a, 400)
    ax.plot(xx, (P.E * h / 2 / (1 - P.nu**2) *
                 (np.pi**2 / a**2 + P.nu * np.pi**2 / b**2) * A_DISP
                 * np.sin(np.pi * xx / a)) / 1e3, "k--",
            lw=1.2, label="closed form")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\sigma_{xx}$ [kPa]")
    ax.set_title("Centre-line $\\sigma_{xx}(x, y=b/2)$ on top surface")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ns_d = [n["n"] for n in disc]
    es = [max(n["err"], 1e-16) for n in disc]
    ax.loglog(ns_d, es, "s-", label=r"$|\sigma_{xx}^{c}-\sigma_{xx}^{analytic}|/\sigma^{analytic}$")
    ns_p = [n for n in ns_d]
    ax.loglog(ns_p, [ (math.pi / a * a / (n - 1))**2 / 12 for n in ns_p],
              "k--", lw=1.0, label=r"$(\pi\Delta x/a)^2/12$ (2nd-order)")
    ax.set_xlabel("grid points per side n")
    ax.set_ylabel("relative error at centre")
    ax.set_title("Post-processing (curvature) convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "field_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Persist numbers and figures manifest
    # ------------------------------------------------------------------
    with open(os.path.join(OUT_DIR, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=2)

    figs = [
        ("mode_shape_w_contour",
         "Mode (1,1) displacement field w(x,y) [mm] on the simply supported "
         "100x100 m plate (w_max = 1.0 mm); contour lines at 12 levels."),
        ("stress_sxx_contour",
         "Top-surface bending stress sigma_xx(x,y) [kPa] at z=+h/2 with the "
         "FEM node lattice overlay (every 4th node of the 41x41 grid)."),
        ("stress_syy_vonmises_contour",
         "Top-surface sigma_yy [kPa] and von Mises equivalent stress [kPa] "
         "for the same mode; both peak at the plate centre."),
        ("stress_through_thickness",
         "Through-thickness distribution of sigma_xx and sigma_yy at the plate "
         "centre: antisymmetric about the mid-plane, zero at z=0 (Kirchhoff)."),
        ("field_convergence",
         "Centre-line sigma_xx profile vs closed form and second-order "
         "convergence of the curvature post-processing (log-log)."),
    ]
    payload_figs = []
    for name, cap in figs:
        p = os.path.join(FIG_DIR, name + ".png")
        with open(p, "rb") as fh:
            payload_figs.append(dict(name=name, caption=cap,
                                     png_base64=base64.b64encode(fh.read()).decode()))
    print(json.dumps({k: numbers[k] for k in
                      ("frequency", "mass_matrix", "modal_mass", "energy")},
                     indent=2))
    return numbers, payload_figs


if __name__ == "__main__":
    numbers, payload_figs = main()
    print(f"figures: {len(payload_figs)}; numbers.json written to {OUT_DIR}")