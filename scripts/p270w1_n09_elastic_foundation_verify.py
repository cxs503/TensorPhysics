#!/usr/bin/env python
"""Node p270w1-n09: elastic-foundation validation surface (Winkler/Pasternak).

Runs the full machine-checkable verification campaign and writes every number
to ``artifacts/p270w1_n09_elastic_foundation/results.json`` (the report is
generated from that JSON, so no number is hand-copied).

Run::

    PYTHONPATH=src python scripts/p270w1_n09_elastic_foundation_verify.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tensorphysics.elastic_foundation import (
    beam_char_length,
    beam_foundation_roots,
    cosine_only_beam_point_load,
    finite_plate_center_deflection,
    finite_plate_solution,
    hetenyi_beam_point_load,
    pasternak_beam_point_load,
    point_load_shear_jump,
    solve_axisym_plate_fem,
    solve_beam_foundation_fem,
    winkler_plate_biharmonic_residual,
    winkler_plate_char_length,
    winkler_plate_flexural_rigidity,
    winkler_plate_load_recovery,
    winkler_plate_moment_r,
    winkler_plate_patch_load,
    winkler_plate_patch_moment_r,
    winkler_plate_point_load,
    winkler_plate_point_load_hankel,
)

OUT_DIR = Path("artifacts/p270w1_n09_elastic_foundation")
OUT_JSON = OUT_DIR / "results.json"

# ---------------------------------------------------------------------------
# reference data sets
# ---------------------------------------------------------------------------
# (a) generic beam on a soft foundation
EI_REF, K_REF, P_REF = 1.0e7, 1.0e6, 1.0e5
# (b) floating ice sheet (fresh-water ice, sea-water buoyancy)
E_ICE, NU_ICE, RHO_W, G_ACC = 5.0e9, 0.33, 1025.0, 9.81
K_ICE = RHO_W * G_ACC                        # 10055.25 N/m^3 -> Pa/m
SIGMA_F = 7.0e5                              # ice flexural strength [Pa]
H_LIST = [0.5, 1.0, 1.5]
P_ICE = 1.0e5


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _r(x):
    """Round floats recursively for stable JSON."""
    if isinstance(x, float):
        return float(f"{x:.10g}")
    if isinstance(x, (np.floating,)):
        return float(f"{float(x):.10g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    return x


def _git(args):
    try:
        return subprocess.check_output(["git"] + args, text=True).strip()
    except Exception:  # pragma: no cover
        return ""


# ---------------------------------------------------------------------------
# S1 - infinite Winkler beam
# ---------------------------------------------------------------------------
def section_beam_winkler():
    lb = beam_char_length(EI_REF, K_REF)
    x = np.linspace(-8 * lb, 8 * lb, 401)

    w_het = hetenyi_beam_point_load(x, P_REF, EI_REF, K_REF)
    w_pas = pasternak_beam_point_load(x, P_REF, EI_REF, K_REF, 0.0)
    ident = float(np.max(np.abs(w_pas - w_het)) / np.max(np.abs(w_het)))

    jump_correct = point_load_shear_jump(
        lambda z: pasternak_beam_point_load(z, P_REF, EI_REF, K_REF, 0.0, 3),
        P_REF, EI_REF)
    jump_cos = point_load_shear_jump(
        lambda z: cosine_only_beam_point_load(z, P_REF, EI_REF, K_REF, 3),
        P_REF, EI_REF)

    w0_ana = float(w_het[len(w_het) // 2])
    shape_dev = []
    for fac in (0.5, 1.0, 2.0):
        w_true = float(hetenyi_beam_point_load(fac * lb, P_REF, EI_REF, K_REF)[0])
        w_cos = float(cosine_only_beam_point_load(fac * lb, P_REF, EI_REF, K_REF)[0])
        shape_dev.append({"x_over_lb": fac, "w_hetenyi": w_true,
                          "w_cosine_only": w_cos, "rel_diff": w_cos / w_true - 1.0})

    fem = []
    for n_elem in (100, 200, 400, 800):
        res = solve_beam_foundation_fem(EI_REF, K_REF, P_REF,
                                        half_length=8 * lb, n_elem=n_elem)
        ana = hetenyi_beam_point_load(res.x, P_REF, EI_REF, K_REF)
        interior = np.abs(res.x) <= 4 * lb
        fem.append({
            "n_elem": n_elem,
            "interior_linf_rel": _rel(res.w[interior], ana[interior]),
            "centre_rel": float(res.w[len(res.x) // 2] / w0_ana - 1.0),
            "global_linf_rel": _rel(res.w, ana),
        })

    trunc = []
    for half in (4.0, 6.0, 8.0, 10.0):
        res = solve_beam_foundation_fem(EI_REF, K_REF, P_REF,
                                        half_length=half * lb, n_elem=400)
        ana = hetenyi_beam_point_load(res.x, P_REF, EI_REF, K_REF)
        trunc.append({"half_length_over_lb": half,
                      "global_linf_rel": _rel(res.w, ana),
                      "centre_rel": float(res.w[len(res.x) // 2] / w0_ana - 1.0)})

    return {
        "EI": EI_REF, "k": K_REF, "P": P_REF, "l_b": lb,
        "w_centre_analytic": w0_ana,
        "pasternak_zero_equals_hetenyi_rel": ident,
        "shear_jump_ratio_correct": jump_correct,
        "shear_jump_ratio_cosine_only": jump_cos,
        "cosine_only_shape_deviation": shape_dev,
        "fem_convergence": fem,
        "fem_truncation": trunc,
    }


# ---------------------------------------------------------------------------
# S2 - infinite Pasternak beam
# ---------------------------------------------------------------------------
def section_beam_pasternak():
    lb = beam_char_length(EI_REF, K_REF)
    gam_ref = np.sqrt(EI_REF * K_REF)
    cases = []
    for gamma in (0.0, 0.5, 1.5, 3.0, 6.0):
        G = gamma * gam_ref
        a, b = beam_foundation_roots(EI_REF, K_REF, G)
        # truncation length must resolve the *longest* decay length of the
        # Pasternak regime, otherwise the free-end approximation is the error
        half = max(8.0 * lb, 6.0 / min(abs(a), abs(b)))
        res = solve_beam_foundation_fem(EI_REF, K_REF, P_REF,
                                        half_length=half, n_elem=800, G_p=G)
        ana = pasternak_beam_point_load(res.x, P_REF, EI_REF, K_REF, G)
        interior = np.abs(res.x) <= 4 * lb
        w0 = float(pasternak_beam_point_load(0.0, P_REF, EI_REF, K_REF, G)[0])
        # ODE residual of the closed form away from the load
        xs = np.array([0.4, 1.3, 2.7]) * lb
        w = pasternak_beam_point_load(xs, P_REF, EI_REF, K_REF, G, 0)
        w2 = pasternak_beam_point_load(xs, P_REF, EI_REF, K_REF, G, 2)
        w4 = pasternak_beam_point_load(xs, P_REF, EI_REF, K_REF, G, 4)
        ode = float(np.max(np.abs(EI_REF * w4 - G * w2 + K_REF * w))
                    / (K_REF * np.max(np.abs(w))))
        cases.append({
            "gamma": gamma,
            "G_p": G,
            "a_real": float(a.real), "a_imag": float(a.imag),
            "b_real": float(b.real), "b_imag": float(b.imag),
            "regime": "complex (oscillatory decay)" if abs(a.imag) > 1e-12 * abs(a)
                      else "two real decay lengths",
            "decay_length_long": float(1.0 / min(abs(a), abs(b))),
            "half_length_used": float(half),
            "w_centre": w0,
            "fem_interior_linf_rel": _rel(res.w[interior], ana[interior]),
            "fem_centre_rel": float(res.w[len(res.x) // 2] / w0 - 1.0),
            "ode_residual_rel": ode,
        })
    return {"gamma_reference": gam_ref, "l_b": lb, "cases": cases}


# ---------------------------------------------------------------------------
# S3 - infinite Winkler plate (Kelvin functions)
# ---------------------------------------------------------------------------
def section_plate_infinite(h=1.0):
    D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
    l = winkler_plate_char_length(D, K_ICE)
    rho_list = [0.0, 0.1, 0.3, 1.0, 2.0, 5.0, 8.0]
    hankel = []
    for rho in rho_list:
        r = rho * l
        kelvin = float(winkler_plate_point_load(r, P_ICE, D, K_ICE)[0])
        hank = winkler_plate_point_load_hankel(r, P_ICE, D, K_ICE)
        hankel.append({"r_over_l": rho, "w_kelvin": kelvin,
                       "w_hankel_integral": hank,
                       "rel_diff": (kelvin / hank - 1.0) if hank != 0 else 0.0})
    r = np.array([0.3, 0.7, 1.5, 3.0, 6.0]) * l
    resid = winkler_plate_biharmonic_residual(r, P_ICE, D, K_ICE)
    w0 = float(winkler_plate_point_load(0.0, P_ICE, D, K_ICE)[0])
    return {
        "h": h, "D": D, "l": l, "k": K_ICE,
        "w_centre": w0,
        "w_centre_P_over_8_sqrt_kD": P_ICE / (8.0 * np.sqrt(K_ICE * D)),
        "w_centre_P_over_8_k_l2": P_ICE / (8.0 * K_ICE * l ** 2),
        "w_centre_rel_dev": float(w0 / (P_ICE / (8.0 * np.sqrt(K_ICE * D))) - 1.0),
        "hankel_check": hankel,
        "hankel_max_rel": float(max(abs(h["rel_diff"]) for h in hankel)),
        "load_recovery": winkler_plate_load_recovery(P_ICE, D, K_ICE),
        "biharmonic_residual_max": float(np.max(np.abs(resid))),
    }


# ---------------------------------------------------------------------------
# S4 - axisymmetric Kirchhoff FEM against the closed forms
# ---------------------------------------------------------------------------
def section_plate_fem(h=1.0):
    D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
    l = winkler_plate_char_length(D, K_ICE)
    patch = []
    for (nc, no, gr) in ((8, 200, 2.0), (10, 300, 2.0), (10, 400, 1.5),
                         (16, 480, 2.0)):
        a = 0.05 * l
        res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_ICE,
                                     load_radius=a, n_center=nc, n_outer=no,
                                     grading=gr, bc="free")
        rq = np.linspace(0.0, 8 * l, 81)
        w_fe = res.evaluate(rq)
        w_an = winkler_plate_patch_load(rq, P_ICE, a, D, K_ICE)
        patch.append({
            "n_center": nc, "n_outer": no, "grading": gr, "n_dof": 2 * (nc + no + 1),
            "centre_rel": float(w_fe[0] / w_an[0] - 1.0),
            "profile_l2_rel": float(np.sqrt(np.mean((w_fe - w_an) ** 2))
                                    / np.max(np.abs(w_an))),
            "profile_linf_rel": float(np.max(np.abs(w_fe - w_an))
                                      / np.max(np.abs(w_an))),
        })
    point = []
    for (no, gr) in ((300, 2.0), (400, 2.0), (600, 2.0)):
        res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_ICE,
                                     load_radius=0.0, n_outer=no, grading=gr,
                                     bc="free")
        w0 = float(winkler_plate_point_load(0.0, P_ICE, D, K_ICE)[0])
        point.append({"n_outer": no, "grading": gr, "n_dof": 2 * (no + 1),
                      "centre_rel": float(res.w[0] / w0 - 1.0)})
    # moments at r = 1, 2, 3 l (patch load ~ point load there)
    a = 0.05 * l
    res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_ICE,
                                 load_radius=a, n_center=10, n_outer=300,
                                 grading=2.0, bc="free")
    rq = np.array([1.0, 2.0, 3.0]) * l
    M_fe = res.moment_r(rq)
    M_an = winkler_plate_moment_r(rq, P_ICE, D, K_ICE, NU_ICE)
    moments = [{"r_over_l": float(rho), "M_fem": float(m),
                "M_analytic_point_load": float(ma),
                "rel_diff": float(m / ma - 1.0)}
               for rho, m, ma in zip([1.0, 2.0, 3.0], M_fe, M_an)]
    return {"h": h, "D": D, "l": l, "patch_load": patch,
            "point_load": point, "moments": moments}


# ---------------------------------------------------------------------------
# S5 - finite radius: boundary effect law
# ---------------------------------------------------------------------------
def section_boundary_effect(h=1.0):
    D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
    l = winkler_plate_char_length(D, K_ICE)
    w_inf = float(winkler_plate_point_load(0.0, P_ICE, D, K_ICE)[0])
    spans = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    table = []
    for bc in ("clamped", "simply_supported"):
        rows = []
        for span in spans:
            sol = finite_plate_solution(span * l, P_ICE, D, K_ICE, NU_ICE, bc)
            rows.append({"R_over_l": span,
                         "w_centre_finite": sol.w_center,
                         "rel_error_vs_infinite": sol.w_center / w_inf - 1.0,
                         "c1": sol.c1, "c2": sol.c2})
        # fit the decay slope on the range where the perturbation is between
        # 1e-11 and 1e-2 (above double-precision noise, below the near field)
        # asymptotic band: above the double-precision noise floor (~1e-15)
        # and below the near field where the growing branches are not yet
        # dominated by the slowest one
        pts = [(r["R_over_l"], abs(r["rel_error_vs_infinite"])) for r in rows
               if 1e-12 < abs(r["rel_error_vs_infinite"]) < 1e-4]
        local = []
        for r0, r1 in zip(rows[:-1], rows[1:]):
            e0 = abs(r0["rel_error_vs_infinite"])
            e1 = abs(r1["rel_error_vs_infinite"])
            if e0 > 0.0 and e1 > 0.0 and 1e-12 < e1 < 1e-4:
                local.append({"R_over_l_range": [r0["R_over_l"], r1["R_over_l"]],
                              "log_slope": float(np.log(e1 / e0)
                                                 / (r1["R_over_l"] - r0["R_over_l"]))})
        # exponential decay law: slope of ln|error| against R/l
        x = np.array([p[0] for p in pts])
        y = np.log(np.array([p[1] for p in pts]))
        slope = float(np.polyfit(x, y, 1)[0])
        table.append({"bc": bc, "rows": rows, "fitted_log_slope": slope,
                      "n_fit_points": len(pts), "fit_band": [1e-12, 1e-4],
                      "local_log_slopes": local})
    # FEM cross-check of the boundary condition implementation at R = 12 l
    a = 0.05 * l
    ratio = float(winkler_plate_patch_load(0.0, P_ICE, a, D, K_ICE)[0]) / w_inf
    fem = []
    for bc in ("clamped", "simply_supported"):
        res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 12 * l, P=P_ICE,
                                     load_radius=a, n_center=10, n_outer=400,
                                     grading=2.0, bc=bc)
        ex = finite_plate_solution(12 * l, P_ICE, D, K_ICE, NU_ICE, bc)
        fem.append({"bc": bc, "R_over_l": 12.0,
                    "w_centre_fem": float(res.w[0]),
                    "w_centre_exact_patch_scaled": ex.w_center * ratio,
                    "rel_diff": float(res.w[0] / (ex.w_center * ratio) - 1.0)})
    return {"h": h, "l": l, "w_centre_infinite": w_inf,
            "predicted_slope_minus_1_over_sqrt2": -1.0 / np.sqrt(2.0),
            "predicted_slope_minus_sqrt2": -float(np.sqrt(2.0)),
            "cases": table, "fem_check": fem}


# ---------------------------------------------------------------------------
# S6 - ice-sheet parameters, characteristic-length scaling law
# ---------------------------------------------------------------------------
def section_ice_scaling():
    rows = []
    for h in H_LIST:
        D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
        l = winkler_plate_char_length(D, K_ICE)
        w0 = float(winkler_plate_point_load(0.0, P_ICE, D, K_ICE)[0])
        lb = beam_char_length(E_ICE * h ** 3 / 12.0, K_ICE)
        # FEM-derived characteristic length: centre deflection of a small patch
        a = 0.30                                     # fixed contact radius [m]
        res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_ICE,
                                     load_radius=a, n_center=10, n_outer=300,
                                     grading=2.0, bc="free")
        w0_fem = float(res.w[0])
        patch_ratio = float(winkler_plate_patch_load(0.0, P_ICE, a, D, K_ICE)[0]) / w0
        w0_eff = w0_fem / patch_ratio                  # point-load equivalent
        l_fem = float(np.sqrt(P_ICE / (8.0 * K_ICE * w0_eff)))
        rows.append({"h": h, "D": D, "l_analytic": l, "l_fem": l_fem,
                     "l_fem_rel_dev": l_fem / l - 1.0,
                     "w_centre_point": w0, "w_centre_fem_patch": w0_fem,
                     "patch_to_point_ratio": patch_ratio,
                     "beam_l": lb, "M_first_crack": SIGMA_F * h ** 2 / 6.0})
    hs = np.array([r["h"] for r in rows])
    slope_l = float(np.polyfit(np.log(hs), np.log([r["l_analytic"] for r in rows]), 1)[0])
    slope_l_fem = float(np.polyfit(np.log(hs), np.log([r["l_fem"] for r in rows]), 1)[0])
    slope_w = float(np.polyfit(np.log(hs), np.log([r["w_centre_point"] for r in rows]), 1)[0])
    slope_lb = float(np.polyfit(np.log(hs), np.log([r["beam_l"] for r in rows]), 1)[0])
    return {"E": E_ICE, "nu": NU_ICE, "rho_w": RHO_W, "g": G_ACC, "k": K_ICE,
            "P": P_ICE, "rows": rows,
            "exponent_l_analytic": slope_l,
            "exponent_l_fem": slope_l_fem,
            "exponent_w_centre": slope_w,
            "exponent_beam_l": slope_lb,
            "expected_exponent": 0.75}


# ---------------------------------------------------------------------------
# S7 - first-crack load of a floating ice sheet on a circular contact patch
# ---------------------------------------------------------------------------
def section_first_crack():
    """First-crack load of a floating ice sheet under a circular contact patch.

    The driver is the maximum radial bending moment of the *patch*-loaded
    infinite plate (analytic Hankel-integral moment field, see
    :func:`winkler_plate_patch_moment_r`); the FEM moment maximum at the same
    load is the cross-check.
    """
    a_contact = 0.50
    rows = []
    for h in H_LIST:
        D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
        l = winkler_plate_char_length(D, K_ICE)
        M_f = SIGMA_F * h ** 2 / 6.0
        rq = np.linspace(0.0, 2.0 * l, 401)
        M_unit = winkler_plate_patch_moment_r(rq, 1.0, a_contact, D, K_ICE, NU_ICE)
        i_max = int(np.argmax(np.abs(M_unit)))
        M_unit_max = float(M_unit[i_max])
        P_cr = M_f / abs(M_unit_max)
        res = solve_axisym_plate_fem(D, NU_ICE, K_ICE, 40 * l, P=P_cr,
                                     load_radius=a_contact, n_center=12,
                                     n_outer=320, grading=2.0, bc="free")
        M_fe = np.abs(res.moment_r(rq))
        j_max = int(np.argmax(M_fe))
        rows.append({"h": h, "l": l, "M_first_crack": M_f,
                     "contact_radius": a_contact,
                     "M_analytic_max_per_unit_load": abs(M_unit_max),
                     "r_over_l_at_M_max_analytic": float(rq[i_max] / l),
                     "P_crack_analytic": P_cr,
                     "M_fem_max": float(M_fe[j_max]),
                     "r_over_l_at_M_max_fem": float(rq[j_max] / l),
                     "moment_rel_dev": float(M_fe[j_max] / M_f - 1.0),
                     "P_over_k_l2": float(P_cr / (K_ICE * l ** 2))})
    return {"contact_radius": a_contact, "sigma_f": SIGMA_F, "rows": rows}


# ---------------------------------------------------------------------------
# criteria assembly
# ---------------------------------------------------------------------------
def assemble_criteria(s1, s2, s3, s4, s5, s6, s7):
    beam_worst = max(r["interior_linf_rel"] for r in s1["fem_convergence"])
    pas_worst = max(c["fem_interior_linf_rel"] for c in s2["cases"])
    patch_best = min(s4["patch_load"], key=lambda r: r["profile_l2_rel"])
    point_best = min(s4["point_load"], key=lambda r: abs(r["centre_rel"]))
    mom_worst = max(abs(m["rel_diff"]) for m in s4["moments"])
    slopes = {c["bc"]: abs(c["fitted_log_slope"]) for c in s5["cases"]}
    fem_bc_worst = max(abs(f["rel_diff"]) for f in s5["fem_check"])
    crack_worst = max(abs(r["moment_rel_dev"]) for r in s7["rows"])

    crit = [
        {"id": "G1", "what": "Winkler beam: FEM vs Hetényi, interior L∞ relative error",
         "value": beam_worst, "threshold": 1e-3, "unit": "-",
         "pass": beam_worst < 1e-3},
        {"id": "G2", "what": "Pasternak beam: FEM vs closed form, interior L∞ relative error (γ = 0…6)",
         "value": pas_worst, "threshold": 1e-3, "unit": "-",
         "pass": pas_worst < 1e-3},
        {"id": "G3", "what": "Point-load source strength EI[w''']/P (must be 1; cosine-only brief formula gives 1/2)",
         "value": s1["shear_jump_ratio_correct"], "threshold": [1.0 - 1e-6, 1.0 + 1e-6],
         "unit": "-", "pass": abs(s1["shear_jump_ratio_correct"] - 1.0) < 1e-6},
        {"id": "G4", "what": "Winkler plate: Kelvin closed form vs defining Hankel integral (max rel.)",
         "value": s3["hankel_max_rel"], "threshold": 1e-5, "unit": "-",
         "pass": s3["hankel_max_rel"] < 1e-5},
        {"id": "G5", "what": "Winkler plate: equilibrium load recovery 2πk∫w r dr / P",
         "value": s3["load_recovery"], "threshold": [0.999999, 1.000001], "unit": "-",
         "pass": abs(s3["load_recovery"] - 1.0) < 1e-6},
        {"id": "G6", "what": "Winkler plate: bilaplacian residual of the closed form (max)",
         "value": s3["biharmonic_residual_max"], "threshold": 1e-10, "unit": "-",
         "pass": s3["biharmonic_residual_max"] < 1e-10},
        {"id": "G7", "what": "Axisymmetric Kirchhoff FEM vs analytic patch load: profile L2 relative error",
         "value": patch_best["profile_l2_rel"], "threshold": 0.03, "unit": "-",
         "pass": patch_best["profile_l2_rel"] < 0.03},
        {"id": "G8", "what": "Axisymmetric Kirchhoff FEM point load: centre deflection relative error",
         "value": point_best["centre_rel"], "threshold": 0.03, "unit": "-",
         "pass": abs(point_best["centre_rel"]) < 0.03},
        {"id": "G9", "what": "Axisymmetric FEM bending moments vs analytic (max rel. at r = 1,2,3 l)",
         "value": mom_worst, "threshold": 0.03, "unit": "-",
         "pass": mom_worst < 0.03},
        {"id": "G10",
         "what": ("Boundary effect: |d ln|Δw/w| / d(R/l)| for the clamped edge "
                  "(predictions: 1/sqrt(2) = 0.707 for one edge condition, "
                  "sqrt(2) = 1.414 for the two coupled conditions)"),
         "value": slopes["clamped"], "threshold": [1.1, 1.8], "unit": "-",
         "pass": 1.1 < slopes["clamped"] < 1.8},
        {"id": "G11", "what": "Boundary effect error at R = 40 l below 1e-10 (infinite-plate limit reached)",
         "value": max(abs(r["rel_error_vs_infinite"])
                      for c in s5["cases"] for r in c["rows"] if r["R_over_l"] >= 40.0),
         "threshold": 1e-10, "unit": "-",
         "pass": max(abs(r["rel_error_vs_infinite"]) for c in s5["cases"]
                     for r in c["rows"] if r["R_over_l"] >= 40.0) < 1e-10},
        {"id": "G12", "what": "FEM edge conditions (clamped / simply supported at R = 12 l) vs exact finite-domain solution",
         "value": fem_bc_worst, "threshold": 0.03, "unit": "-",
         "pass": fem_bc_worst < 0.03},
        {"id": "G13", "what": "Scaling law l ∝ h^(3/4): fitted analytic exponent",
         "value": s6["exponent_l_analytic"], "threshold": [0.75 - 1e-9, 0.75 + 1e-9],
         "unit": "-", "pass": abs(s6["exponent_l_analytic"] - 0.75) < 1e-9},
        {"id": "G14", "what": "Scaling law recovered from the FEM deflection field (h = 0.5/1.0/1.5 m)",
         "value": s6["exponent_l_fem"], "threshold": [0.74, 0.76], "unit": "-",
         "pass": 0.74 < s6["exponent_l_fem"] < 0.76},
        {"id": "G15", "what": "First-crack load: FEM moment field reproduces M_f = σ_f h²/6 (max rel. dev.)",
         "value": crack_worst, "threshold": 0.05, "unit": "-",
         "pass": crack_worst < 0.05},
    ]
    return crit


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1 = section_beam_winkler()
    s2 = section_beam_pasternak()
    s3 = section_plate_infinite(1.0)
    s4 = section_plate_fem(1.0)
    s5 = section_boundary_effect(1.0)
    s6 = section_ice_scaling()
    s7 = section_first_crack()
    crit = assemble_criteria(s1, s2, s3, s4, s5, s6, s7)

    payload = {
        "meta": {
            "node": "p270w1-n09-elastic-foundation-validation-maturity",
            "mission_id": 9360,
            "repo": "TensorPhysics",
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git(["rev-parse", "HEAD"]),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "module": "src/tensorphysics/elastic_foundation",
            "tests": "tests/test_elastic_foundation_validation.py",
        },
        "criteria": crit,
        "all_pass": bool(all(c["pass"] for c in crit)),
        "sections": {
            "S1_beam_winkler": s1,
            "S2_beam_pasternak": s2,
            "S3_plate_infinite": s3,
            "S4_plate_fem": s4,
            "S5_boundary_effect": s5,
            "S6_ice_scaling": s6,
            "S7_first_crack": s7,
        },
    }
    OUT_JSON.write_text(json.dumps(_r(payload), indent=2, ensure_ascii=False))
    print(f"wrote {OUT_JSON}")
    for c in crit:
        print(f"  {c['id']} {'PASS' if c['pass'] else 'FAIL'}  "
              f"{c['value']:.6g}  ({c['what']})")
    print("ALL PASS" if payload["all_pass"] else "SOME CRITERIA FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())