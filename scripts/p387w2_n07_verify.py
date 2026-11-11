#!/usr/bin/env python3
"""p387w2-n07｜冰板弹性半空间地基三层验证面（Winkler -> Pasternak -> 弹性半空间）。

Computes every acceptance number and writes ``artifacts/p387w2_n07/results.json``.
This is the single source of truth for the node report: the report generator
injects these numbers programmatically (zero copy-by-hand).

Criteria (campaign table):
(a) Pasternak deflection ``w(r)`` vs Hetényi analytic solution, error <= 5%
(b) elastic half-space limit, error <= 5%
(c) point-load ``w_max``, error <= 5%
(d) crack-spacing scaling exponent ``alpha`` vs theory, error <= 20%
(e) two negative controls
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorphysics.elastic_foundation.crack_spacing import (  # noqa: E402
    bearing_capacity_ratio,
    crack_spacing,
    first_crack_load,
    flexural_strength_moment,
    local_log_slope,
    moment_extrema_radii,
    pasternak_ring_crack_period,
    spacing_scaling_exponent,
)
from tensorphysics.elastic_foundation.halfspace import (  # noqa: E402
    boussinesq_patch_deflection,
    boussinesq_point_deflection,
    halfspace_foundation_modulus,
    halfspace_plate_char_length,
    halfspace_plate_center_deflection,
    halfspace_plate_moment_r,
    halfspace_plate_patch_load,
)
from tensorphysics.elastic_foundation.pasternak_plate import (  # noqa: E402
    match_pasternak_to_halfspace,
    pasternak_plate_center_deflection,
    pasternak_plate_char_length,
    pasternak_plate_moment_r,
    pasternak_plate_point_load,
    pasternak_plate_point_load_hankel,
    pasternak_plate_roots,
    vlasov_foundation_params,
)
from tensorphysics.elastic_foundation.plates import (  # noqa: E402
    solve_axisym_plate_fem,
    winkler_plate_point_load,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "artifacts", "p387w2_n07")


def rel(a, b):
    return float(abs(a - b) / abs(b))


# ---------------------------------------------------------------------------
# case definition: level ice sheet
# ---------------------------------------------------------------------------
ICE = dict(E=5.0e9, nu=0.30, h=1.0, sigma_f=7.0e5, rho_w=1025.0, g=9.81)
D_ICE = ICE["E"] * ICE["h"] ** 3 / (12.0 * (1.0 - ICE["nu"] ** 2))
K_WATER = ICE["rho_w"] * ICE["g"]
G_ICE = ICE["E"] / (2.0 * (1.0 + ICE["nu"]))
G_P_SHEAR = G_ICE * ICE["h"]              # case A: ice shear layer, beta = 1
SOIL = dict(E_s=100.0e6, nu_s=0.35)       # frozen silt / river bed
H_VLASOV = 8.011819035441082              # Vlasov depth whose k equals the
                                          # half-space kernel match (below)
C_HS = halfspace_foundation_modulus(**SOIL)
L_HS = halfspace_plate_char_length(D_ICE, C_HS)
K_VL, G_P_VL = vlasov_foundation_params(SOIL["E_s"], SOIL["nu_s"], H_VLASOV)
K_MATCH, G_P_MATCH, ALPHA_REF = match_pasternak_to_halfspace(
    D_ICE, SOIL["E_s"], SOIL["nu_s"])
PATCH_A = 0.5                             # ship-bow contact patch radius [m]
P_LOAD = 1.0e5                            # reference ice load [N]

CASES = {
    "A_ice_shear_layer": dict(k=K_WATER, G_p=G_P_SHEAR,
                              label="ice sheet on water + ice shear layer"),
    "B_vlasov_layer": dict(k=K_VL, G_p=G_P_VL,
                           label="Vlasov compressible layer (H = %.2f m)" % H_VLASOV),
    "C_matched_kernel": dict(k=K_MATCH, G_p=G_P_MATCH,
                             label="two-constant fit of the Boussinesq kernel"),
}


def largest_scale(k, G_p):
    """Slowest decay length ``1/min|mu|`` of the Pasternak kernel [m]."""
    mu1, mu2 = pasternak_plate_roots(D_ICE, k, G_p)[2:]
    return 1.0 / min(abs(mu1), abs(mu2))


# ---------------------------------------------------------------------------
# (a) + (c): Winkler / Pasternak anchors
# ---------------------------------------------------------------------------
def tier12_checks():
    out = {}
    l = pasternak_plate_char_length(D_ICE, K_WATER)
    r_grid = np.linspace(0.0, 5.0 * l, 201)
    w_het = winkler_plate_point_load(r_grid, P_LOAD, D_ICE, K_WATER)
    w_past0 = pasternak_plate_point_load(r_grid, P_LOAD, D_ICE, K_WATER, 0.0)
    out["a1_winkler_limit_max_rel"] = float(np.max(np.abs(w_past0 / w_het - 1.0)))
    out["a1_winkler_limit_n_points"] = int(r_grid.size)
    out["a1_winkler_limit_r_range_over_l"] = [0.0, 5.0]

    # Hetényi centre deflection P/(8 sqrt(k D)) as the reference for (c)
    w0_het = P_LOAD / (8.0 * np.sqrt(K_WATER * D_ICE))
    out["c2_winkler_wmax_rel"] = rel(
        float(pasternak_plate_center_deflection(P_LOAD, D_ICE, K_WATER, 0.0)), w0_het)
    out["c2_winkler_wmax_value"] = w0_het

    per_case = {}
    for name, case in CASES.items():
        k, G_p = case["k"], case["G_p"]
        scale = largest_scale(k, G_p)
        r_fit = np.linspace(0.2 * pasternak_plate_char_length(D_ICE, k),
                            5.0 * pasternak_plate_char_length(D_ICE, k), 15)
        cf = pasternak_plate_point_load(r_fit, P_LOAD, D_ICE, k, G_p)
        hank = np.array([pasternak_plate_point_load_hankel(r, P_LOAD, D_ICE, k, G_p)
                         for r in r_fit])
        res = solve_axisym_plate_fem(D_ICE, ICE["nu"], k, 60.0 * scale, P=P_LOAD,
                                     n_outer=1500, grading=2.2, G_p=G_p)
        fem = res.evaluate(r_fit)
        per_case[name] = dict(
            label=case["label"], k=k, G_p=G_p,
            decay_length=scale,
            char_length=pasternak_plate_char_length(D_ICE, k),
            gamma=float(G_p / np.sqrt(D_ICE * k)),
            a2_closed_vs_hankel_max_rel=float(np.max(np.abs(cf / hank - 1.0))),
            a2_closed_vs_fem_max_rel=float(np.max(np.abs(cf / fem - 1.0))),
            c1_wmax_closed=float(pasternak_plate_center_deflection(P_LOAD, D_ICE, k, G_p)),
            c1_wmax_hankel=float(pasternak_plate_point_load_hankel(0.0, P_LOAD, D_ICE, k, G_p)),
            c1_wmax_fem=float(res.w[0]),
        )
        per_case[name]["c1_wmax_spread_rel"] = float(
            max(per_case[name]["c1_wmax_closed"], per_case[name]["c1_wmax_hankel"],
                per_case[name]["c1_wmax_fem"])
            / min(per_case[name]["c1_wmax_closed"], per_case[name]["c1_wmax_hankel"],
                  per_case[name]["c1_wmax_fem"]) - 1.0)
    out["per_case"] = per_case

    # where does the Winkler placeholder leave the 5 % band?
    def dev(G_p):
        ww = pasternak_plate_point_load(r_grid, P_LOAD, D_ICE, K_WATER, G_p)
        return float(np.max(np.abs(ww / w_het - 1.0)))

    lo, hi = 1.0, 1.0e10
    if dev(hi) > 0.05 > dev(lo):
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if dev(mid) > 0.05:
                hi = mid
            else:
                lo = mid
        G5 = float(np.sqrt(lo * hi))
        out["a3_Gp_at_5pct"] = G5
        out["a3_gamma_at_5pct"] = float(G5 / np.sqrt(D_ICE * K_WATER))
        out["a3_relative_5pct"] = dev(G5)
    return out


# ---------------------------------------------------------------------------
# (b): elastic half-space limit
# ---------------------------------------------------------------------------
def tier3_checks():
    out = {}
    p_patch = P_LOAD / (np.pi * PATCH_A ** 2)
    w_bouss0 = float(boussinesq_patch_deflection(0.0, p_patch, PATCH_A,
                                                SOIL["E_s"], SOIL["nu_s"])[0])
    out["b_boussinesq_centre"] = w_bouss0
    out["b_halfspace_modulus"] = C_HS
    out["b_char_length_hs"] = L_HS

    # b1: plate-free (D -> 0) limit of the spectral solver -> Boussinesq
    conv = []
    for beta in (1.0e-1, 1.0e-2, 1.0e-3):
        D_small = beta * C_HS * PATCH_A ** 3
        w0 = halfspace_plate_center_deflection(P_LOAD, PATCH_A, D_small,
                                               SOIL["E_s"], SOIL["nu_s"])
        conv.append(dict(beta=beta, D=D_small, w0=w0, rel_err=rel(w0, w_bouss0)))
    out["b1_plate_free_convergence"] = conv
    out["b1_plate_free_rel_err"] = conv[-1]["rel_err"]
    # plate-free profile across the whole patch, not just the centre
    r_prof = np.array([0.0, 0.25 * PATCH_A, 0.5 * PATCH_A, 0.75 * PATCH_A, PATCH_A])
    D_small = 1.0e-3 * C_HS * PATCH_A ** 3
    w_spec = np.array([halfspace_plate_center_deflection(P_LOAD, PATCH_A, D_small,
                                                         SOIL["E_s"], SOIL["nu_s"])]
                      + [halfspace_plate_patch_load(r, P_LOAD, PATCH_A, D_small,
                                                    SOIL["E_s"], SOIL["nu_s"])[0]
                         for r in r_prof[1:]])
    w_bouss = boussinesq_patch_deflection(r_prof, p_patch, PATCH_A,
                                          SOIL["E_s"], SOIL["nu_s"])
    out["b1_profile_max_rel"] = float(np.max(np.abs(w_spec / w_bouss - 1.0)))
    out["b1_profile_r"] = r_prof.tolist()
    out["b1_profile_solver"] = w_spec.tolist()
    out["b1_profile_boussinesq"] = w_bouss.tolist()

    # b2: far field of the plate-on-half-space -> Boussinesq point load
    r_far = 50.0 * PATCH_A
    w_far = float(halfspace_plate_patch_load(r_far, P_LOAD, PATCH_A, D_ICE,
                                             SOIL["E_s"], SOIL["nu_s"])[0])
    w_ref = boussinesq_point_deflection(P_LOAD, r_far, SOIL["E_s"], SOIL["nu_s"])
    out["b2_far_field"] = dict(r=r_far, w_plate=w_far, w_boussinesq_point=w_ref,
                               rel_err=rel(w_far, w_ref))

    # b3: stiff-plate limit law w_max ~ D^(-1/3)
    Ds = np.array([1.0e9, 1.0e10, 1.0e11, 1.0e12])
    wmax = np.array([halfspace_plate_center_deflection(P_LOAD, PATCH_A, Dv,
                                                       SOIL["E_s"], SOIL["nu_s"])
                     for Dv in Ds])
    alpha, _C = spacing_scaling_exponent(Ds, wmax)
    out["b3_stiff_law"] = dict(D=Ds.tolist(), w0=wmax.tolist(), exponent=float(alpha),
                               theory=-1.0 / 3.0,
                               rel_err=rel(alpha, -1.0 / 3.0))

    # b4: large-patch continuum limit  w0 -> 2 P (1 - nu^2)/(pi E a)
    big = []
    for aa in (5.0, 10.0, 20.0):
        w0 = halfspace_plate_center_deflection(P_LOAD, aa, D_ICE,
                                               SOIL["E_s"], SOIL["nu_s"])
        ref = 2.0 * P_LOAD * (1.0 - SOIL["nu_s"] ** 2) / (np.pi * SOIL["E_s"] * aa)
        big.append(dict(a=aa, beta=D_ICE / (C_HS * aa ** 3), w0=w0, ref=ref,
                        rel_err=rel(w0, ref)))
    converged = [p["rel_err"] for p in big if p["beta"] <= 1.0e-2]
    out["b4_large_patch_limit"] = dict(points=big,
                                       rel_err=float(max(converged)),
                                       selection="max over beta = D/(C a^3) <= 1e-2")
    # no finite point-load deflection on a plate-free continuum: w0 ~ 1/a
    radii_pf = np.array([0.5, 0.25, 0.125])
    w0_pf = np.array([halfspace_plate_center_deflection(
        P_LOAD, aa, 1.0e-3 * C_HS * aa ** 3, SOIL["E_s"], SOIL["nu_s"])
        for aa in radii_pf])
    exp_pf, _ = spacing_scaling_exponent(radii_pf, w0_pf)
    w0_plate = np.array([halfspace_plate_center_deflection(P_LOAD, aa, D_ICE,
                                                           SOIL["E_s"], SOIL["nu_s"])
                         for aa in radii_pf])
    out["b4_point_load_divergence"] = dict(
        a=radii_pf.tolist(), w0_plate_free=w0_pf.tolist(), w0_ice_plate=w0_plate.tolist(),
        exponent_plate_free=float(exp_pf), theory=-1.0,
        rel_err=rel(exp_pf, -1.0),
        plate_saturates=bool(np.max(np.diff(w0_plate)) / np.max(w0_plate) < 0.5))

    # b5: how far is the two-parameter model from the continuum?  (finding, not a gate)
    w_hs0 = float(halfspace_plate_center_deflection(P_LOAD, PATCH_A, D_ICE,
                                                    SOIL["E_s"], SOIL["nu_s"]))
    w_hs2 = float(halfspace_plate_patch_load(2.0 * PATCH_A, P_LOAD, PATCH_A, D_ICE,
                                             SOIL["E_s"], SOIL["nu_s"])[0])
    from tensorphysics.elastic_foundation.pasternak_plate import (
        pasternak_plate_patch_load,
    )
    model_dev = {}
    for name, case in CASES.items():
        wp0 = float(pasternak_plate_patch_load(0.0, P_LOAD, PATCH_A, D_ICE,
                                               case["k"], case["G_p"])[0])
        wp2 = float(pasternak_plate_patch_load(2.0 * PATCH_A, P_LOAD, PATCH_A,
                                               D_ICE, case["k"], case["G_p"])[0])
        model_dev[name] = dict(w0=wp0, w_2a=wp2,
                               rel_err_centre=rel(wp0, w_hs0),
                               rel_err_2a=rel(wp2, w_hs2))
    out["b5_pasternak_vs_continuum"] = dict(continuum=dict(w0=w_hs0, w_2a=w_hs2),
                                           models=model_dev)
    out["b5_best_rel_err_centre"] = min(v["rel_err_centre"]
                                        for v in model_dev.values())
    # best achievable two-parameter fit of the continuum profile (grid search)
    r_fit = np.array([0.0, PATCH_A, 2.0 * PATCH_A, 5.0 * PATCH_A])
    w_hs_fit = np.array([w_hs0] + [float(halfspace_plate_patch_load(rv, P_LOAD, PATCH_A,
                                                                    D_ICE, SOIL["E_s"],
                                                                    SOIL["nu_s"])[0])
                                   for rv in r_fit[1:]])

    def fit_residual(kk, Gg):
        vals = []
        for rv in r_fit:
            vals.append(float(pasternak_plate_patch_load(rv, P_LOAD, PATCH_A, D_ICE,
                                                         kk, Gg)[0]))
        vals = np.asarray(vals)
        return float(np.max(np.abs(vals / w_hs_fit - 1.0)))

    best_fit = dict(res=fit_residual(K_MATCH, G_P_MATCH), k=K_MATCH, G_p=G_P_MATCH)
    for kf in np.geomspace(K_MATCH / 3.0, K_MATCH * 3.0, 5):
        for gf in np.geomspace(G_P_MATCH / 3.0, G_P_MATCH * 3.0, 5):
            res = fit_residual(kf, gf)
            if res < best_fit["res"]:
                best_fit = dict(res=res, k=float(kf), G_p=float(gf))
    out["b5_best_two_parameter_fit"] = best_fit

    scan = []
    for Hv in np.linspace(3.0, 60.0, 8):
        kv, Gv = vlasov_foundation_params(SOIL["E_s"], SOIL["nu_s"], float(Hv))
        wv = float(pasternak_plate_patch_load(0.0, P_LOAD, PATCH_A, D_ICE, kv, Gv)[0])
        scan.append(dict(H=float(Hv), k=kv, G_p=Gv, w0=wv,
                         rel_err=rel(wv, w_hs0)))
    best = min(scan, key=lambda d: d["rel_err"])
    out["b5_vlasov_depth_scan"] = dict(best_H=best["H"], best_rel_err=best["rel_err"],
                                      best_k=best["k"], best_G_p=best["G_p"],
                                      scan=scan)
    return out


# ---------------------------------------------------------------------------
# (d): crack-spacing scaling law and bearing capacity
# ---------------------------------------------------------------------------
def scaling_checks():
    out = {}
    M_f = flexural_strength_moment(ICE["sigma_f"], ICE["h"])
    out["M_f"] = M_f

    # d1/d2: exponent of s(k) = C (D/k)^(1/4) from closed form and from FEM
    ks = np.array([1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5])
    sp_cf, sp_fem, const_cf = [], [], []
    for k in ks:
        l = pasternak_plate_char_length(D_ICE, k)
        rr = moment_extrema_radii(
            lambda r, _k=k: pasternak_plate_moment_r(r, 1.0, D_ICE, _k, ICE["nu"], 0.0),
            11.0 * l, n=8001, n_keep=5)
        sp_cf.append(float(np.mean(np.diff(rr))))
        const_cf.append(float(sp_cf[-1] / l))
        res = solve_axisym_plate_fem(D_ICE, ICE["nu"], k, 15.0 * l, P=1.0,
                                     n_outer=1200, grading=2.0)
        rr_f = moment_extrema_radii(lambda r: res.moment_r(r), 11.0 * l,
                                    n=6001, n_keep=5)
        sp_fem.append(float(np.mean(np.diff(rr_f))))
    a_cf, C_cf = spacing_scaling_exponent(ks, sp_cf)
    a_fem, C_fem = spacing_scaling_exponent(ks, sp_fem)
    # C_s = s / l is the same constant in both routes (l is the reference length)
    Cs_cf = float(np.mean(np.array(sp_cf) / np.array([pasternak_plate_char_length(D_ICE, k)
                                                      for k in ks])))
    Cs_fem = float(np.mean(np.array(sp_fem) / np.array([pasternak_plate_char_length(D_ICE, k)
                                                        for k in ks])))
    out["d1_winkler_exponent"] = dict(
        k=ks.tolist(), spacing_closed=sp_cf, spacing_fem=sp_fem,
        alpha_closed=float(a_cf), alpha_fem=float(a_fem), theory=-0.25,
        rel_err_closed=rel(a_cf, -0.25), rel_err_fem=rel(a_fem, -0.25),
        Cs_closed=Cs_cf, Cs_fem=Cs_fem, C_fit_closed=C_cf, C_fit_fem=C_fem)

    # d3: Pasternak local exponent (spacing is not a pure power of k any more);
    # theory = analytic period law  s ~ 2 pi / Im(mu(k))
    per_case = {}
    for name in ("B_vlasov_layer", "C_matched_kernel"):
        case = CASES[name]
        k0, G_p = case["k"], case["G_p"]
        delta = 0.02
        ks_p = k0 * np.array([1.0 - delta, 1.0, 1.0 + delta])
        sp = []
        for k in ks_p:
            mu_im = abs(pasternak_plate_roots(D_ICE, k, G_p)[2].imag)
            r_max = 4.0 * (2.0 * np.pi / mu_im)
            rr = moment_extrema_radii(
                lambda r, _k=k: pasternak_plate_moment_r(r, 1.0, D_ICE, _k,
                                                         ICE["nu"], G_p),
                r_max, n=8001, n_keep=3)
            sp.append(float(np.mean(np.diff(rr))))
        a_meas, _ = spacing_scaling_exponent(ks_p, sp)
        a_theory = local_log_slope(
            lambda kk: pasternak_ring_crack_period(D_ICE, kk, G_p), k0)
        per_case[name] = dict(k_values=ks_p.tolist(), spacing=sp,
                              period=pasternak_ring_crack_period(D_ICE, k0, G_p),
                              alpha_measured=float(a_meas),
                              alpha_theory=float(a_theory),
                              rel_err=rel(a_meas, a_theory))
    out["d3_pasternak_exponent"] = per_case

    # d4: first-crack load (bearing capacity) per foundation model
    k = K_WATER
    l = pasternak_plate_char_length(D_ICE, k)
    Pc_w = first_crack_load(
        lambda r: pasternak_plate_moment_r(r, 1.0, D_ICE, k, ICE["nu"], 0.0),
        M_f, 8.0 * l, n=8001)
    # k-independence of the Winkler peak moment (classical Hetenyi result):
    # theory exponent 0, since M_max/P is set by the load, not by the subgrade
    Pc_ks, Pc_vals = [], []
    for kv in (1.0e3, 1.0e4, 1.0e5):
        lv = pasternak_plate_char_length(D_ICE, kv)
        Pc_ks.append(kv)
        Pc_vals.append(first_crack_load(
            lambda r, _k=kv: pasternak_plate_moment_r(r, 1.0, D_ICE, _k, ICE["nu"], 0.0),
            M_f, 8.0 * lv, n=8001))
    exp_Pc, _ = spacing_scaling_exponent(np.array(Pc_ks), np.array(Pc_vals))
    Pc_models = {}
    for name, case in CASES.items():
        lv = pasternak_plate_char_length(D_ICE, case["k"])
        scale = largest_scale(case["k"], case["G_p"])
        Pc_models[name] = first_crack_load(
            lambda r, _c=case: pasternak_plate_moment_r(r, 1.0, D_ICE, _c["k"],
                                                        ICE["nu"], _c["G_p"]),
            M_f, 8.0 * max(lv, scale), n=8001)
    # patch-loaded bearing capacity (smooth field: no point-load singularity)
    from tensorphysics.elastic_foundation.pasternak_plate import (
        pasternak_plate_patch_load,
        pasternak_plate_patch_moment_r,
    )
    r_patch = np.linspace(0.02 * PATCH_A, 4.0 * PATCH_A, 81)

    def Pc_from_profile(w):
        d1 = np.gradient(w, r_patch)
        d2 = np.gradient(d1, r_patch)
        mom = -D_ICE * (d2 + ICE["nu"] * d1 / r_patch)
        return M_f / float(np.max(np.abs(mom)))

    Pc_patch_w = M_f / float(np.max(np.abs(
        pasternak_plate_patch_moment_r(r_patch, 1.0, PATCH_A, D_ICE, K_WATER,
                                       ICE["nu"], 0.0))))
    Pc_patch_vl = M_f / float(np.max(np.abs(
        pasternak_plate_patch_moment_r(r_patch, 1.0, PATCH_A, D_ICE, K_VL,
                                       ICE["nu"], G_P_VL))))
    Pc_patch_w_fd = Pc_from_profile(
        pasternak_plate_patch_load(r_patch, 1.0, PATCH_A, D_ICE, K_WATER, 0.0))
    Pc_hs_analytic = M_f / float(np.max(np.abs(
        halfspace_plate_moment_r(r_patch, 1.0, PATCH_A, D_ICE,
                                 SOIL["E_s"], SOIL["nu_s"], ICE["nu"]))))
    # pitfall documented in the report: finite-difference moments are unusable
    # near the axis (1/r term + coarse grid), hence the spectral moments above
    w_hs_prof = np.array([halfspace_plate_patch_load(rv, 1.0, PATCH_A, D_ICE,
                                                     SOIL["E_s"], SOIL["nu_s"])[0]
                          for rv in r_patch])
    Pc_hs_fd = Pc_from_profile(w_hs_prof)
    out["d4_bearing_capacity"] = dict(
        Winkler_point_load=Pc_w,
        Winkler_k_exponent=float(exp_Pc),
        Winkler_k_exponent_theory=0.0,
        Pasternak_point_load=Pc_models,
        patch=dict(halfspace_analytic=Pc_hs_analytic, halfspace_fd=Pc_hs_fd,
                   fd_rel_err=rel(Pc_hs_fd, Pc_hs_analytic),
                   winkler_fd=Pc_patch_w_fd,
                   winkler_fd_rel_err=rel(Pc_patch_w_fd, Pc_patch_w),
                   Winkler=Pc_patch_w, vlasov=Pc_patch_vl,
                   ratio_hs_over_winkler=bearing_capacity_ratio(Pc_hs_analytic,
                                                                Pc_patch_w),
                   ratio_vlasov_over_winkler=bearing_capacity_ratio(Pc_patch_vl,
                                                                    Pc_patch_w)),
    )
    return out


# ---------------------------------------------------------------------------
# (e): negative controls
# ---------------------------------------------------------------------------
def negative_controls(tier12, tier3):
    out = {}
    # e1: the Winkler placeholder must be *rejected* by the 5 % gate
    k, G_p = CASES["A_ice_shear_layer"]["k"], CASES["A_ice_shear_layer"]["G_p"]
    w_past = pasternak_plate_center_deflection(P_LOAD, D_ICE, k, G_p)
    w_wink = pasternak_plate_center_deflection(P_LOAD, D_ICE, k, 0.0)
    out["e1_winkler_placeholder"] = dict(
        w_pasternak=w_past, w_winkler=w_wink,
        rel_dev=rel(w_wink, w_past), gate=0.05,
        rejected=bool(rel(w_wink, w_past) > 0.05))
    # e2: wrong sign of the shear layer (p = k w + G_p lap w) must be caught
    r_probe = np.array([0.0, 1.0, 2.0])
    try:
        w_wrong = pasternak_plate_point_load(r_probe, P_LOAD, D_ICE, k, -G_p)
        dev = float(np.max(np.abs(w_wrong / pasternak_plate_point_load(
            r_probe, P_LOAD, D_ICE, k, G_p) - 1.0)))
        out["e2_wrong_sign_Gp"] = dict(rel_dev=dev, gate=0.05,
                                       rejected=bool(dev > 0.05 or np.any(w_wrong < 0)))
    except ValueError as exc:                       # pragma: no cover
        out["e2_wrong_sign_Gp"] = dict(rejected=True, error=str(exc))
    # e3: a continuum foundation has no finite point-load deflection
    div = tier3["b4_point_load_divergence"]
    out["e3_continuum_divergence"] = dict(
        exponent_plate_free=div["exponent_plate_free"], theory=div["theory"],
        monotone=bool(np.all(np.diff(div["w0_plate_free"]) > 0.0)),
        plate_saturates=div["plate_saturates"],
        rejected_finite_point_load=bool(np.all(np.diff(div["w0_plate_free"]) > 0.0)
                                        and div["plate_saturates"]))
    # e4: linearity / zero-load degeneracy
    w1 = pasternak_plate_center_deflection(P_LOAD, D_ICE, K_WATER, 0.0)
    w2 = pasternak_plate_center_deflection(2.0 * P_LOAD, D_ICE, K_WATER, 0.0)
    w0 = pasternak_plate_center_deflection(0.0, D_ICE, K_WATER, 0.0)
    out["e4_linearity"] = dict(w_P=w1, w_2P=w2, w_0=w0,
                               rel_dev=rel(w2, 2.0 * w1), zero_ok=bool(abs(w0) < 1e-30))
    return out


def main():
    t0 = time.time()
    tier12 = tier12_checks()
    tier3 = tier3_checks()
    scaling = scaling_checks()
    neg = negative_controls(tier12, tier3)

    cases = tier12["per_case"]
    a_err = max([tier12["a1_winkler_limit_max_rel"]]
                + [v["a2_closed_vs_fem_max_rel"] for v in cases.values()])
    c_err = max([tier12["c2_winkler_wmax_rel"]]
                + [v["c1_wmax_spread_rel"] for v in cases.values()])
    b_err = max(tier3["b1_plate_free_rel_err"], tier3["b1_profile_max_rel"],
                tier3["b2_far_field"]["rel_err"], tier3["b3_stiff_law"]["rel_err"],
                tier3["b4_large_patch_limit"]["rel_err"],
                tier3["b4_point_load_divergence"]["rel_err"])
    d_err = max([scaling["d1_winkler_exponent"]["rel_err_closed"],
                 scaling["d1_winkler_exponent"]["rel_err_fem"]]
                + [v["rel_err"] for v in scaling["d3_pasternak_exponent"].values()])

    criteria = [
        dict(id="a", text="Pasternak w(r) vs Hetényi analytic (Winkler limit) and "
                          "independent FEM", value=float(a_err), threshold=0.05,
             unit="max relative error", passed=bool(a_err <= 0.05)),
        dict(id="b", text="elastic half-space limit (plate-free Boussinesq, far field, "
                          "stiff-plate law, log divergence)", value=float(b_err),
             threshold=0.05, unit="max relative error", passed=bool(b_err <= 0.05)),
        dict(id="c", text="point-load w_max (closed form / Hankel / FEM)",
             value=float(c_err), threshold=0.05, unit="max relative error",
             passed=bool(c_err <= 0.05)),
        dict(id="d", text="crack-spacing scaling exponent alpha vs theory",
             value=float(d_err), threshold=0.20, unit="relative error of exponent",
             passed=bool(d_err <= 0.20)),
        dict(id="e", text="two negative controls",
             value=float(min(neg["e1_winkler_placeholder"]["rel_dev"],
                             abs(neg["e2_wrong_sign_Gp"].get("rel_dev", 1.0)))),
             threshold=0.05, unit="detected deviation",
             passed=bool(neg["e1_winkler_placeholder"]["rejected"]
                         and neg["e2_wrong_sign_Gp"].get("rejected", False)
                         and neg["e3_continuum_divergence"]["monotone"])),
    ]

    payload = dict(
        node="p387w2-n07",
        title="Ice plate on a three-tier foundation: Winkler -> Pasternak -> "
              "elastic half-space",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        runtime_s=round(time.time() - t0, 1),
        ice=dict(ICE, D=D_ICE, k_water=K_WATER, G_ice=G_ICE, G_p_shear=G_P_SHEAR),
        soil=dict(SOIL, C=C_HS, l_hs=L_HS,
                  vlasov=dict(H=H_VLASOV, k=K_VL, G_p=G_P_VL),
                  matched=dict(k=K_MATCH, G_p=G_P_MATCH, alpha_ref=ALPHA_REF)),
        patch_radius=PATCH_A, load=P_LOAD,
        cases={k: dict(v) for k, v in CASES.items()},
        tier12=tier12, tier3=tier3, scaling=scaling, negative_controls=neg,
        criteria=criteria,
        all_passed=bool(all(c["passed"] for c in criteria)),
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(criteria, indent=2))
    print("all_passed:", payload["all_passed"], "runtime", payload["runtime_s"], "s")
    print("written:", os.path.abspath(path))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())