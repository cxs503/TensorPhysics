#!/usr/bin/env python3
"""p387w2-n07 figures: three-tier ice-plate foundation verification surface.

All quantities are recomputed here (or read from ``results.json``), never typed in
by hand.  PNGs land in ``artifacts/p387w2_n07/figs``.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tensorphysics.elastic_foundation.crack_spacing import (  # noqa: E402
    crack_spacing,
    flexural_strength_moment,
    moment_extrema_radii,
)
from tensorphysics.elastic_foundation.halfspace import (  # noqa: E402
    boussinesq_patch_deflection,
    boussinesq_point_deflection,
    halfspace_foundation_modulus,
    halfspace_plate_center_deflection,
    halfspace_plate_moment_r,
    halfspace_plate_patch_load,
)
from tensorphysics.elastic_foundation.pasternak_plate import (  # noqa: E402
    pasternak_plate_center_deflection,
    pasternak_plate_char_length,
    pasternak_plate_moment_r,
    pasternak_plate_patch_load,
    pasternak_plate_point_load,
    pasternak_plate_point_load_hankel,
    vlasov_foundation_params,
)
from tensorphysics.elastic_foundation.plates import (  # noqa: E402
    solve_axisym_plate_fem,
)

ART = os.path.join(HERE, "..", "artifacts", "p387w2_n07")
FIG = os.path.join(ART, "figs")
RES = json.load(open(os.path.join(ART, "results.json")))

ICE = RES["ice"]
SOIL = RES["soil"]
D = ICE["D"]
NU = ICE["nu"]
K_W = ICE["k_water"]
G_P_A = ICE["G_p_shear"]
A = RES["patch_radius"]
P = RES["load"]
C_HS = SOIL["C"]
E_S, NU_S = SOIL["E_s"], SOIL["nu_s"]
K_VL, G_P_VL = SOIL["vlasov"]["k"], SOIL["vlasov"]["G_p"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. three-tier deflection profiles under the same patch load
# ---------------------------------------------------------------------------
def fig_tiers():
    r = np.linspace(0.02 * A, 12.0 * A, 60)
    w_w = pasternak_plate_patch_load(r, P, A, D, K_W, 0.0)
    w_p = pasternak_plate_patch_load(r, P, A, D, K_VL, G_P_VL)
    w_a = pasternak_plate_patch_load(r, P, A, D, K_W, G_P_A)
    w_h = np.array([halfspace_plate_patch_load(rv, P, A, D, E_S, NU_S)[0] for rv in r])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].semilogy(r / A, w_w, label="Winkler (buoyancy k=%.3g Pa/m)" % K_W)
    ax[0].semilogy(r / A, w_p, label="Pasternak/Vlasov (H=%.2f m)" % SOIL["vlasov"]["H"])
    ax[0].semilogy(r / A, w_a, label="Pasternak + ice shear layer")
    ax[0].semilogy(r / A, w_h, label="elastic half-space (Boussinesq)")
    ax[0].set_xlabel("r / a  (a = %.2f m contact patch)" % A)
    ax[0].set_ylabel("deflection w [m]")
    ax[0].set_title("three foundation tiers, same ice plate + patch load")
    ax[0].legend(fontsize=7)
    for arr, lbl in ((w_w, "Winkler"), (w_p, "Pasternak/Vlasov"), (w_h, "half-space")):
        ax[1].plot(r / A, arr / arr[0], label=lbl)
    ax[1].set_xlabel("r / a")
    ax[1].set_ylabel("w(r) / w(0)")
    ax[1].set_title("normalised shapes: the continuum decays fastest")
    ax[1].legend(fontsize=7)
    save(fig, "fig1_three_tiers.png")


# ---------------------------------------------------------------------------
# 2. Pasternak closed form vs independent Hankel integral and FEM
# ---------------------------------------------------------------------------
def fig_pasternak_verification():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, k, G_p, marker in (
            ("Vlasov layer", K_VL, G_P_VL, "o"),
            ("matched kernel", SOIL["matched"]["k"], SOIL["matched"]["G_p"], "s")):
        l = pasternak_plate_char_length(D, k)
        r = np.linspace(0.2 * l, 5.0 * l, 13)
        cf = pasternak_plate_point_load(r, P, D, k, G_p)
        hk = np.array([pasternak_plate_point_load_hankel(rv, P, D, k, G_p) for rv in r])
        mu = np.array([abs(m) for m in
                       __import__("tensorphysics.elastic_foundation.pasternak_plate",
                                  fromlist=["x"]).pasternak_plate_roots(D, k, G_p)[2:]])
        res = solve_axisym_plate_fem(D, NU, k, 60.0 / mu.min(), P=P, n_outer=1200,
                                     grading=2.2, G_p=G_p)
        ax.semilogy(r / l, np.abs(cf / hk - 1.0), marker + "-", label="%s: closed vs Hankel" % name)
        ax.semilogy(r / l, np.abs(cf / res.evaluate(r) - 1.0), marker + "--",
                    label="%s: closed vs axisym FEM" % name)
    ax.axhline(0.05, color="k", ls=":", label="5 % gate")
    ax.set_xlabel("r / l,  l = (D/k)^(1/4)")
    ax.set_ylabel("relative deviation")
    ax.set_title("Pasternak Green function: closed form vs two independent routes")
    ax.legend(fontsize=7)
    save(fig, "fig2_pasternak_verification.png")


# ---------------------------------------------------------------------------
# 3. elastic half-space limits
# ---------------------------------------------------------------------------
def fig_halfspace_limits():
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.5))
    # (a) plate-free limit -> Boussinesq
    betas = np.logspace(-1, -4, 7)
    err = []
    ref = float(boussinesq_patch_deflection(0.0, P / (np.pi * A ** 2), A, E_S, NU_S)[0])
    for beta in betas:
        w0 = halfspace_plate_center_deflection(P, A, beta * C_HS * A ** 3, E_S, NU_S)
        err.append(abs(w0 / ref - 1.0))
    ax[0].loglog(betas, err, "o-", label="spectral solver vs Boussinesq")
    ax[0].loglog(betas, betas, "k--", alpha=0.6, label="O(beta), beta = D/(C a^3)")
    ax[0].axhline(0.05, color="r", ls=":", label="5 % gate")
    ax[0].set_xlabel("D / (C a^3)")
    ax[0].set_ylabel("relative error of w(0)")
    ax[0].invert_xaxis()
    ax[0].set_title("(a) plate-free limit")
    ax[0].legend(fontsize=7)
    # (b) stiff-plate law
    Ds = np.array([1e9, 1e10, 1e11, 1e12])
    w0 = np.array([halfspace_plate_center_deflection(P, A, Dv, E_S, NU_S) for Dv in Ds])
    ax[1].loglog(Ds, w0, "o-", label="solver")
    slope = RES["tier3"]["b3_stiff_law"]["exponent"]
    ax[1].loglog(Ds, w0[0] * (Ds / Ds[0]) ** (-1.0 / 3.0), "k--", alpha=0.6,
                 label="D^(-1/3) reference")
    ax[1].set_xlabel("flexural rigidity D [N m]")
    ax[1].set_ylabel("w_max [m]")
    ax[1].set_title("(b) stiff-plate limit, fitted exponent %.4f" % slope)
    ax[1].legend(fontsize=7)
    # (c) far field -> Boussinesq point load
    r = np.linspace(2 * A, 60 * A, 40)
    w_plate = np.array([halfspace_plate_patch_load(rv, P, A, D, E_S, NU_S)[0] for rv in r])
    w_pt = np.array([boussinesq_point_deflection(P, rv, E_S, NU_S) for rv in r])
    ax[2].loglog(r / A, w_plate, "o", ms=3, label="plate on half-space")
    ax[2].loglog(r / A, w_pt, "-", label="Boussinesq point load")
    ax[2].set_xlabel("r / a")
    ax[2].set_ylabel("w [m]")
    ax[2].set_title("(c) far field, P(1-nu^2)/(pi E r)")
    ax[2].legend(fontsize=7)
    save(fig, "fig3_halfspace_limits.png")


# ---------------------------------------------------------------------------
# 4. ring-crack spacing scaling law
# ---------------------------------------------------------------------------
def fig_crack_scaling():
    d1 = RES["scaling"]["d1_winkler_exponent"]
    ks = np.array(d1["k"])
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax[0].loglog(ks, d1["spacing_closed"], "o-", label="closed form, alpha=%.6f"
                 % d1["alpha_closed"])
    ax[0].loglog(ks, d1["spacing_fem"], "s--", label="axisym FEM, alpha=%.6f"
                 % d1["alpha_fem"])
    l = np.array([pasternak_plate_char_length(D, k) for k in ks])
    ax[0].loglog(ks, d1["Cs_closed"] * l, "k:", alpha=0.6,
                 label="s = %.4f (D/k)^(1/4)" % d1["Cs_closed"])
    ax[0].set_xlabel("Winkler modulus k [Pa/m]")
    ax[0].set_ylabel("ring-crack spacing [m]")
    ax[0].set_title("(a) spacing vs k, theory exponent -1/4")
    ax[0].legend(fontsize=7)
    # moment field with crack radii at the ice-sheet reference modulus
    lw = pasternak_plate_char_length(D, K_W)
    r = np.linspace(0.05 * lw, 11 * lw, 600)
    M_w = pasternak_plate_moment_r(r, P, D, K_W, NU, 0.0)
    M_v = pasternak_plate_moment_r(r, P, D, K_VL, G_P_VL)
    M_f = flexural_strength_moment(7.0e5, ICE["h"])
    rr = moment_extrema_radii(lambda x: pasternak_plate_moment_r(x, P, D, K_W, NU, 0.0),
                              11 * lw, n=6001, n_keep=3)
    ax[1].plot(r / lw, M_w / 1e3, label="Winkler")
    ax[1].plot(r / pasternak_plate_char_length(D, K_VL), M_v / 1e3, label="Pasternak/Vlasov")
    ax[1].axhline(M_f / 1e3, color="r", ls=":", label="M_f = sigma_f h^2/6")
    for rv in rr:
        ax[1].axvline(rv / lw, color="0.6", lw=0.8)
    ax[1].set_xlabel("r / l")
    ax[1].set_ylabel("M_r [kN m/m]")
    ax[1].set_title("(b) M_r field, ring-crack radii (grey) and M_f")
    ax[1].legend(fontsize=7)
    save(fig, "fig4_crack_spacing.png")


# ---------------------------------------------------------------------------
# 5. bearing capacity and the two-parameter fit quality
# ---------------------------------------------------------------------------
def fig_bearing_and_fit():
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    patch = RES["scaling"]["d4_bearing_capacity"]["patch"]
    labels = ["Winkler", "Pasternak\n(Vlasov)", "half-space\n(Boussinesq)"]
    vals = [patch["Winkler"], patch["vlasov"], patch["halfspace_analytic"]]
    ax[0].bar(labels, np.array(vals) / 1e3, color=["#4c72b0", "#dd8452", "#55a868"])
    for i, v in enumerate(vals):
        ax[0].text(i, v / 1e3 * 1.02, "%.3g kN" % (v / 1e3), ha="center", fontsize=8)
    ax[0].set_ylabel("first-crack load P_c [kN]")
    ax[0].set_title("(a) bearing capacity, a = %.2f m patch, M_f = %.4g kN m/m"
                    % (A, flexural_strength_moment(7.0e5, ICE["h"]) / 1e3))
    # (b) Vlasov depth scan: how well can a two-parameter bed mimic the continuum?
    scan = RES["tier3"]["b5_vlasov_depth_scan"]["scan"]
    H = np.array([s["H"] for s in scan])
    e = np.array([s["rel_err"] for s in scan])
    best = RES["tier3"]["b5_best_two_parameter_fit"]
    ax[1].semilogy(H, e * 100, "o-", label="Vlasov family (one free depth H)")
    ax[1].axhline(best["res"] * 100, color="g", ls="--",
                  label="best 2-constant fit: %.2f %%" % (best["res"] * 100))
    ax[1].axhline(5.0, color="r", ls=":", label="5 % gate")
    ax[1].set_xlabel("Vlasov depth H [m]")
    ax[1].set_ylabel("centre-deflection deviation from continuum [%]")
    ax[1].set_title("(b) how close can a two-parameter bed get to the half-space?")
    ax[1].legend(fontsize=7)
    save(fig, "fig5_bearing_and_fit.png")


def main():
    fig_tiers()
    fig_pasternak_verification()
    fig_halfspace_limits()
    fig_crack_scaling()
    fig_bearing_and_fit()


if __name__ == "__main__":
    main()