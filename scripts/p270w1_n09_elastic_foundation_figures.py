#!/usr/bin/env python
"""Figures for node p270w1-n09 (elastic-foundation validation surface).

Produces six PNGs under ``artifacts/p270w1_n09_elastic_foundation/figs`` from
the same kernels used by the verification script (no hand-copied numbers).

Run::

    PYTHONPATH=src python scripts/p270w1_n09_elastic_foundation_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorphysics.elastic_foundation import (  # noqa: E402
    beam_char_length,
    beam_foundation_roots,
    cosine_only_beam_point_load,
    finite_plate_solution,
    hetenyi_beam_point_load,
    pasternak_beam_point_load,
    solve_axisym_plate_fem,
    solve_beam_foundation_fem,
    winkler_plate_char_length,
    winkler_plate_flexural_rigidity,
    winkler_plate_patch_load,
    winkler_plate_patch_moment_r,
    winkler_plate_point_load,
)

OUT = Path("artifacts/p270w1_n09_elastic_foundation")
FIGS = OUT / "figs"
RES = json.loads((OUT / "results.json").read_text())

EI_REF, K_REF, P_REF = 1.0e7, 1.0e6, 1.0e5
E_ICE, NU_ICE = 5.0e9, 0.33
K_ICE = 1025.0 * 9.81
P_ICE = 1.0e5
SIGMA_F = 7.0e5

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 10,
})


def fig01_beam_winkler():
    lb = beam_char_length(EI_REF, K_REF)
    x = np.linspace(-6 * lb, 6 * lb, 601)
    w_het = hetenyi_beam_point_load(x, P_REF, EI_REF, K_REF)
    w_cos = cosine_only_beam_point_load(x, P_REF, EI_REF, K_REF)
    fem = solve_beam_foundation_fem(EI_REF, K_REF, P_REF, half_length=8 * lb,
                                    n_elem=400)
    ana_at_nodes = hetenyi_beam_point_load(fem.x, P_REF, EI_REF, K_REF)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax.plot(x / lb, w_het / w_het.max(), "k-", lw=1.8,
            label="Hetényi closed form (correct)")
    ax.plot(x / lb, w_cos / w_het.max(), "r--", lw=1.4,
            label="cosine-only brief formula (falsified)")
    ax.plot(fem.x / lb, fem.w / w_het.max(), "o", ms=2.4, mfc="none",
            mec="tab:blue", label="Hermite FEM (400 elements)")
    ax.set_ylabel("w / w(0)  [-]")
    ax.set_title("Winkler beam, point load: analytic anchor vs FE "
                 f"(EI = {EI_REF:.1e} N·m², k = {K_REF:.1e} N/m²)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-6, 6)

    with np.errstate(divide="ignore", invalid="ignore"):
        e_fem = np.abs(fem.w / ana_at_nodes - 1.0)
        e_cos = np.abs(w_cos / w_het - 1.0)
    ax2.semilogy(fem.x / lb, np.maximum(e_fem, 1e-16), "-", color="tab:blue",
                 lw=1.2, label="|FEM/Hetényi − 1|")
    ax2.semilogy(x / lb, np.maximum(e_cos, 1e-16), "--", color="crimson",
                 lw=1.2, label="|cosine-only/Hetényi − 1|")
    ax2.set_xlabel("x / l$_b$  (l$_b$ = (4EI/k)$^{1/4}$) [-]")
    ax2.set_ylabel("relative error [-]")
    ax2.set_ylim(1e-8, 1e1)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig01_beam_winkler.png")
    plt.close(fig)


def fig02_beam_pasternak():
    lb = beam_char_length(EI_REF, K_REF)
    gam_ref = np.sqrt(EI_REF * K_REF)
    x = np.linspace(-6 * lb, 6 * lb, 801)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
    for col, gamma in zip(colors, (0.0, 0.5, 1.5, 3.0, 6.0)):
        G = gamma * gam_ref
        w = pasternak_beam_point_load(x, P_REF, EI_REF, K_REF, G)
        roots = beam_foundation_roots(EI_REF, K_REF, G)
        half = max(8.0 * lb, 6.0 / min(abs(roots[0]), abs(roots[1])))
        fem = solve_beam_foundation_fem(EI_REF, K_REF, P_REF,
                                        half_length=half, n_elem=600, G_p=G)
        ax.plot(x / lb, w / w[400], "-", color=col, lw=1.5,
                label=f"γ = {gamma:g} (G$_p$ = {G:.2e} N/m)")
        mask = np.abs(fem.x) <= 6 * lb
        ax.plot(fem.x[mask] / lb,
                fem.w[mask] / float(pasternak_beam_point_load(0.0, P_REF, EI_REF, K_REF, G)[0]),
                "o", ms=2.0, mfc="none", mec=col)
    ax.set_xlabel("x / l$_b$ [-]")
    ax.set_ylabel("w / w(0) [-]")
    ax.set_title("Pasternak beam: γ family (lines = closed form, "
                 "circles = FEM)")
    ax.legend(fontsize=7)
    ax.set_xlim(-6, 6)

    cases = RES["sections"]["S2_beam_pasternak"]["cases"]
    gs = [c["gamma"] for c in cases]
    errs = [c["fem_interior_linf_rel"] for c in cases]
    ax2.semilogy(gs, errs, "s-", color="tab:blue")
    ax2.axhline(1e-3, color="crimson", ls="--", lw=1.0, label="gate 1e-3")
    ax2.set_xlabel("γ = G$_p$/√(EI k) [-]")
    ax2.set_ylabel("FEM interior L∞ rel. error [-]")
    ax2.set_title("FEM vs closed form")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig02_beam_pasternak.png")
    plt.close(fig)


def fig03_plate_radial():
    h = 1.0
    D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
    l = winkler_plate_char_length(D, K_ICE)
    a = 0.05 * l
    res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_ICE,
                                 load_radius=a, n_center=10, n_outer=300,
                                 grading=2.0, bc="free")
    rq = np.linspace(0.0, 8 * l, 241)
    w_fe = res.evaluate(rq)
    w_patch = winkler_plate_patch_load(rq, P_ICE, a, D, K_ICE)
    w_point = winkler_plate_point_load(rq, P_ICE, D, K_ICE)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax.plot(rq / l, w_point / w_point[0], "k-", lw=1.6,
            label="Kelvin closed form (point load)")
    ax.plot(rq / l, w_patch / w_point[0], "--", color="tab:green", lw=1.4,
            label=f"patch load a = {a/l:.2f} l (Hankel integral)")
    ax.plot(rq / l, w_fe / w_point[0], "o", ms=2.2, mfc="none",
            mec="tab:red", label="axisymmetric Kirchhoff FEM")
    ax.set_ylabel("w / w(0) [-]")
    ax.set_title("Winkler plate, ice sheet h = 1 m: analytic anchors vs FE")
    ax.legend(fontsize=8)
    ax2.semilogy(rq / l, np.maximum(np.abs(w_fe / w_patch - 1.0), 1e-17),
                 color="tab:red", lw=1.3, label="|FEM/patch − 1|")
    ax2.semilogy(rq / l, np.maximum(np.abs(w_point / w_patch - 1.0), 1e-17),
                 "k--", lw=1.0, label="|point/patch − 1|")
    ax2.axhline(0.03, color="crimson", ls=":", lw=1.0, label="3 % gate")
    ax2.set_xlabel("r / l  (l = (D/k)$^{1/4}$) [-]")
    ax2.set_ylabel("relative difference [-]")
    ax2.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig03_plate_radial.png")
    plt.close(fig)


def fig04_plate_field():
    h = 1.0
    D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
    l = winkler_plate_char_length(D, K_ICE)
    a = 0.5
    res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_ICE,
                                 load_radius=a, n_center=12, n_outer=320,
                                 grading=2.0, bc="free")
    n = 361
    lim = 3.0 * l
    gx = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(gx, gx)
    rr = np.hypot(X, Y).ravel()
    W = res.evaluate(rr).reshape(X.shape)
    Mr = res.moment_r(rr).reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, field, title, unit in (
            (axes[0], W * 1e3, "deflection w [mm]", "mm"),
            (axes[1], Mr / 1e3, "radial moment M$_r$ [kN·m/m]", "kN·m/m")):
        im = ax.contourf(X / l, Y / l, field, levels=24, cmap="viridis")
        cs = ax.contour(X / l, Y / l, field, levels=8, colors="w", linewidths=0.5)
        ax.clabel(cs, fontsize=6, fmt="%.1f")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(a / l * np.cos(th), a / l * np.sin(th), "r-", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("x / l [-]")
        ax.set_ylabel("y / l [-]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=unit, shrink=0.9)
    fig.suptitle("Field quality: axisymmetric FEM field revolved around the "
                 "load (h = 1 m ice sheet, P = 100 kN, patch a = 0.5 m)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig04_plate_field.png", bbox_inches="tight")
    plt.close(fig)


def fig05_boundary_effect():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"clamped": "tab:blue", "simply_supported": "tab:orange"}
    for case in RES["sections"]["S5_boundary_effect"]["cases"]:
        rows = case["rows"]
        spans = np.array([r["R_over_l"] for r in rows])
        errs = np.abs([r["rel_error_vs_infinite"] for r in rows])
        good = errs > 1e-15
        ax.semilogy(spans[good], errs[good], "o-", ms=4,
                    color=colors[case["bc"]], label=case["bc"].replace("_", " "))
        slope = case["fitted_log_slope"]
        xr = np.linspace(8, 20, 20)
        ref = errs[spans == 8.0][0] * np.exp(-abs(slope) * (xr - 8.0))
        ax.semilogy(xr, ref, ":", lw=1.2, color=colors[case["bc"]],
                    label=f"fit slope −{abs(slope):.2f} / (R/l)")
        ax.annotate(f"fitted |d ln(Δw/w)/d(R/l)| = {abs(slope):.2f}",
                    xy=(9.0, errs[spans == 8.0][0]), fontsize=7,
                    color=colors[case["bc"]])
    r = np.linspace(4, 25, 200)
    ax.semilogy(r, 3e-2 * np.exp(-(1.0 / np.sqrt(2.0)) * (r - 4)), "k--", lw=1.0,
                label="single-mode law exp(−R/(√2 l))")
    ax.semilogy(r, 3e-2 * np.exp(-np.sqrt(2.0) * (r - 4)), "k-.", lw=1.0,
                label="two-condition law exp(−√2 R/l)")
    ax.set_xlabel("R / l [-]")
    ax.set_ylabel("|w$_{centre}$(R) / w$_{centre}$(∞) − 1| [-]")
    ax.set_title("Rigid-edge boundary effect of a finite plate on a Winkler "
                 "foundation")
    ax.set_ylim(1e-16, 1e-1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig05_boundary_effect.png")
    plt.close(fig)


def fig06_ice_scaling():
    rows = RES["sections"]["S6_ice_scaling"]["rows"]
    hs = np.array([r["h"] for r in rows])
    ls = np.array([r["l_analytic"] for r in rows])
    lf = np.array([r["l_fem"] for r in rows])
    ws = np.array([r["w_centre_point"] for r in rows])
    crack = RES["sections"]["S7_first_crack"]["rows"]
    pc = np.array([r["P_crack_analytic"] for r in crack])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    ax = axes[0]
    hfit = np.linspace(0.4, 1.7, 50)
    ax.loglog(hs, ls, "o", ms=6, label="analytic l = (D/k)$^{1/4}$")
    ax.loglog(hs, lf, "x", ms=7, label="FEM-derived l")
    ax.loglog(hfit, ls[1] * (hfit / 1.0) ** 0.75, "k--", lw=1.0,
              label="h$^{3/4}$ reference")
    ax.set_xlabel("ice thickness h [m]")
    ax.set_ylabel("characteristic length l [m]")
    ax.set_title(f"l ∝ h$^{{3/4}}$ (fit: {RES['sections']['S6_ice_scaling']['exponent_l_analytic']:.4f}, "
                 f"FEM: {RES['sections']['S6_ice_scaling']['exponent_l_fem']:.4f})")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.loglog(hs, ws, "s-", ms=5)
    ax.loglog(hfit, ws[1] * (hfit / 1.0) ** -1.5, "k--", lw=1.0,
              label="h$^{-3/2}$ reference")
    ax.set_xlabel("ice thickness h [m]")
    ax.set_ylabel("centre deflection w(0) [m]")
    ax.set_title(f"w(0) = P/(8√(kD)), fit exponent "
                 f"{RES['sections']['S6_ice_scaling']['exponent_w_centre']:.3f}")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.loglog(hs, pc / 1e3, "^-", ms=6, color="tab:red")
    ax.set_xlabel("ice thickness h [m]")
    ax.set_ylabel("first-crack load P$_{cr}$ [kN]")
    ax.set_title("first-crack load, contact patch a = 0.5 m,\n"
                 "σ$_f$ = 0.7 MPa")
    fig.tight_layout()
    fig.savefig(FIGS / "fig06_ice_scaling.png")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig01_beam_winkler()
    fig02_beam_pasternak()
    fig03_plate_radial()
    fig04_plate_field()
    fig05_boundary_effect()
    fig06_ice_scaling()
    for p in sorted(FIGS.glob("*.png")):
        print(f"{p}  {p.stat().st_size/1024:.1f} kB")


if __name__ == "__main__":
    main()