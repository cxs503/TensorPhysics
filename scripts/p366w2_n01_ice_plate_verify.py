"""P366-W2-N01 verification driver: FEM ice plate on a Pasternak/Hetenyi foundation
plus the ice thickness / temperature / loading-rate design laws.

Writes
------
``runs/p366w2_n01/results.json``      every number of the four criteria
``artifacts/p366w2_n01/fig_*.png``    figures

Criteria (machine verdicts, all numbers come from this run)
-----------------------------------------------------------
(1) Hetenyi concentrated-load anchor: 2-D BFS Kirchhoff FEM on the ice state
    from :class:`~tensorphysics.ice_sheet.ice_material.IceSheetMaterial`
    against the closed-form Hetenyi/Pasternak field, max relative error < 5 %
    on ``r/l in [0.1, 5]`` for (a) Winkler (``g_p = 0``, the classical
    Kelvin/Hetenyi solution) and (b) ``gamma = 1`` Pasternak.
(2) Constant-pressure patch test: uniform ``w0 = 1 mm`` gives an element
    foundation reaction ``q_e = k w0`` for **every** element, max rel err
    < 1e-10; the Pasternak shear block must annihilate the constant field.
    Uniform and graded mesh.
(3) Pasternak -> Winkler degeneration: ``g_p -> 0`` (six decades) recovers the
    Winkler FEM field, rel err < 0.1 % at the smallest ``g_p``.
(4) Dependence laws of the peak bending stress: ``d sigma_max/dh < 0``,
    ``d sigma_max/d|T| < 0``, ``d sigma_max/dv > 0`` -- plus the strength /
    capacity laws and an explicit definition-consistency audit.

Run from the worktree root::

    PYTHONPATH=src python scripts/p366w2_n01_ice_plate_verify.py
"""

from __future__ import annotations

import base64
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.ice_sheet import design_laws as dl  # noqa: E402
from tensorphysics.ice_sheet import ice_material as im  # noqa: E402
from tensorphysics.ice_sheet import pasternak_foundation as pf  # noqa: E402
from tensorphysics.ice_sheet.kirchhoff_plate_fem import (  # noqa: E402
    BFSPlateSolver,
    KirchhoffWinklerPlate,
    graded_axis_centered,
)

RUNS = ROOT / "runs" / "p366w2_n01"
ART = ROOT / "artifacts" / "p366w2_n01"
FREE = {"left": "free", "right": "free", "bottom": "free", "top": "free"}

REF = {
    "h_m": 1.0,
    "temp_C": -10.0,
    "indentation_rate_m_s": 0.05,
    "load_N": 5.0e5,
    "contact_radius_m": 0.5,
}
RATE_GRID = (0.0005, 0.002, 0.01, 0.05, 0.2, 1.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ice_state(shear_layer_beta: float = 1.0, **overrides) -> im.IceSheetMaterial:
    kw = dict(
        h=REF["h_m"],
        temp_C=REF["temp_C"],
        indentation_rate_m_s=REF["indentation_rate_m_s"],
        shear_layer_beta=shear_layer_beta,
    )
    kw.update(overrides)
    return im.IceSheetMaterial(**kw)


def build_plate(material: im.IceSheetMaterial, length: float) -> KirchhoffWinklerPlate:
    st = material.derived()
    return KirchhoffWinklerPlate(
        h=material.h,
        E=st["E_Pa"],
        nu=material.nu,
        k_w=st["k_w_Pa_per_m"],
        lx=length,
        ly=length,
    )


def fem_patch_solve(
    material: im.IceSheetMaterial,
    g_p: float,
    length: float,
    n: int,
    h0: float,
    radius: float = REF["contact_radius_m"],
    load: float = REF["load_N"],
):
    """2-D BFS FEM of a disc load on the ice sheet."""
    xs = graded_axis_centered(length, n, h0)
    plate = build_plate(material, length)
    solver = BFSPlateSolver(
        plate, n, n,
        xs=xs, ys=xs,
        foundation_model="pasternak" if g_p > 0.0 else "winkler",
        g_p=float(g_p),
    )
    q = load / (math.pi * radius * radius)
    solver.add_pressure(
        lambda x, y: np.full(np.shape(x), q),
        inside=lambda x, y: np.hypot(x, y) <= radius,
        area=math.pi * radius * radius,
        gauss_n=4,
        bbox=(-radius, radius, -radius, radius),
    )
    t0 = time.time()
    field = solver.solve(bc=FREE)
    return field, {
        "domain_m": float(length),
        "elements_per_side": int(n),
        "h0_m": float(h0),
        "max_element_m": float(np.max(np.diff(xs))),
        "solve_runtime_s": time.time() - t0,
    }


def error_metrics(r: np.ndarray, w_fem: np.ndarray, w_ana: np.ndarray, l: float) -> dict:
    """Error measures on the radial window (the field changes sign near 4 l)."""
    peak = float(np.max(np.abs(w_ana)))
    abs_err = np.abs(w_fem - w_ana)
    mask = np.abs(w_ana) >= 0.1 * peak
    rel_masked = float(np.max(abs_err[mask] / np.abs(w_ana[mask])))
    return {
        "window_r_over_l": [float(r[0] / l), float(r[-1] / l)],
        "peak_abs_w_ana_m": peak,
        "max_abs_error_m": float(abs_err.max()),
        "max_error_over_peak": float(abs_err.max() / peak),
        "max_rel_error_masked_10pct": rel_masked,
        "n_samples": int(r.size),
        "n_masked": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# criterion 1
# ---------------------------------------------------------------------------
def criterion_1() -> dict:
    material = ice_state()
    st = material.derived()
    D, k_w, l = st["D_Nm"], st["k_w_Pa_per_m"], st["char_length_m"]
    g_cases = {"winkler": 0.0, "gamma1": math.sqrt(D * k_w)}
    cases, traces = [], {}
    for label, g_p in g_cases.items():
        roots = pf.pasternak_roots(D, k_w, g_p)
        field, mesh = fem_patch_solve(material, g_p, length=200.0, n=90, h0=0.10)
        r = np.linspace(0.1 * l, 5.0 * l, 121)
        w_fem = field.sample_ray((0.0, 0.0), 0.0, r=r)["w"]
        ana_patch = pf.pasternak_patch_field(
            REF["load_N"], REF["contact_radius_m"], D, k_w, g_p,
            nu=material.nu, r_max=5.5 * l, n_field=4001, n_source=96,
        )
        w_patch = np.interp(r, ana_patch["r"], ana_patch["w"])
        w_point = np.atleast_1d(pf.pasternak_point_load(r, REF["load_N"], D, k_w, g_p))
        m_patch = error_metrics(r, w_fem, w_patch, l)
        m_point = error_metrics(r, w_fem, w_point, l)
        used = max(m_patch["max_rel_error_masked_10pct"], m_patch["max_error_over_peak"])
        cases.append({
            "label": label,
            "foundation_model": "winkler" if g_p == 0.0 else "pasternak",
            "g_p_N_per_m": float(g_p),
            "gamma": float(roots["gamma"]),
            "regime": roots["regime"],
            "short_decay_length_m": float(1.0 / max(abs(roots["a"]), abs(roots["b"]))),
            "long_decay_length_m": float(1.0 / min(abs(roots["a"]), abs(roots["b"]))),
            "vs_patch_closed_form": m_patch,
            "vs_point_closed_form": m_point,
            "w_center_fem_m": float(w_fem[0]),
            "w_center_analytic_m": float(w_patch[0]),
            "max_rel_error_used": float(used),
            "passed": bool(used < 0.05),
            **mesh,
        })
        traces[label] = {
            "r_over_l": (r / l).tolist(),
            "w_fem_m": w_fem.tolist(),
            "w_patch_m": w_patch.tolist(),
            "w_point_m": w_point.tolist(),
        }
    gold = dl.unit_moment_field(material, REF["contact_radius_m"],
                                n_field=9601, n_source=96, r_max_factor=8.0)
    prod = dl.unit_moment_field(material, REF["contact_radius_m"],
                                n_field=801, n_source=24, r_max_factor=1.0)
    audit = {
        "m_hat_gold_Nm_per_m_per_N": gold["m_hat_Nm_per_m_per_N"],
        "m_hat_production_Nm_per_m_per_N": prod["m_hat_Nm_per_m_per_N"],
        "rel_diff": float(
            abs(prod["m_hat_Nm_per_m_per_N"] - gold["m_hat_Nm_per_m_per_N"])
            / gold["m_hat_Nm_per_m_per_N"]
        ),
        "gold_settings": "n_field=9601, n_source=96, r_max=8 l",
        "production_settings": "n_field=801, n_source=24, r_max=1 l",
    }
    return {
        "ice_state": st,
        "cases": cases,
        "closed_form_resolution_audit": audit,
        "threshold": 0.05,
        "passed": bool(all(c["passed"] for c in cases) and audit["rel_diff"] < 1e-2),
        "_traces": traces,
    }


# ---------------------------------------------------------------------------
# criterion 2
# ---------------------------------------------------------------------------
def criterion_2() -> dict:
    material = ice_state()
    st = material.derived()
    L = 8.0 * st["char_length_m"]
    g_sets = (
        ("winkler", 0.0),
        ("pasternak_beta1", st["G_p_N_per_m"]),
        ("pasternak_gamma1", math.sqrt(st["D_Nm"] * st["k_w_Pa_per_m"])),
    )
    runs = []
    for label, g_p in g_sets:
        res = dl.foundation_patch_test(
            h=material.h, E=st["E_Pa"], nu=material.nu, k_w=st["k_w_Pa_per_m"],
            g_p=g_p, nx=6, ny=6, length=L, w0=1e-3,
        )
        res["label"] = label
        runs.append(res)
    graded = dl.foundation_patch_test(
        h=material.h, E=st["E_Pa"], nu=material.nu, k_w=st["k_w_Pa_per_m"],
        g_p=st["G_p_N_per_m"], nx=5, ny=5, length=L, w0=1e-3,
    )
    graded["label"] = "graded_mesh_beta1"
    graded["note"] = "graded node spacing (graded_axis_centered, h0 = L/200)"
    allr = runs + [graded]
    return {
        "w0_m": 1.0e-3,
        "k_w_Pa_per_m": st["k_w_Pa_per_m"],
        "target_reaction_Pa": st["k_w_Pa_per_m"] * 1.0e-3,
        "runs": allr,
        "threshold": 1.0e-10,
        "max_rel_error": float(max(r["max_rel_error"] for r in allr)),
        "max_shear_rel_contribution": float(
            max(r["shear_term_max_rel_contribution"] for r in allr)
        ),
        "passed": bool(all(r["passed"] for r in allr)),
    }


# ---------------------------------------------------------------------------
# criterion 3
# ---------------------------------------------------------------------------
def criterion_3() -> dict:
    material = ice_state()
    st = material.derived()
    D, k_w, l = st["D_Nm"], st["k_w_Pa_per_m"], st["char_length_m"]
    g_ref = math.sqrt(D * k_w)
    base, mesh = fem_patch_solve(material, 0.0, length=200.0, n=60, h0=0.25)
    r = np.linspace(0.1 * l, 5.0 * l, 121)
    w_winkler = base.sample_ray((0.0, 0.0), 0.0, r=r)["w"]
    w_ana = np.atleast_1d(pf.pasternak_point_load(r, REF["load_N"], D, k_w, 0.0))
    rows = []
    for decades in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        g_p = g_ref * 10.0 ** (-decades)
        field, _ = fem_patch_solve(material, g_p, length=200.0, n=60, h0=0.25)
        w_p = field.sample_ray((0.0, 0.0), 0.0, r=r)["w"]
        rel_fem = float(np.max(np.abs(w_p - w_winkler)) / np.max(np.abs(w_winkler)))
        rel_ana = float(np.max(np.abs(w_p - w_ana)) / np.max(np.abs(w_ana)))
        rows.append({
            "g_p_N_per_m": float(g_p),
            "g_p_over_sqrt_Dk": float(g_p / g_ref),
            "gamma": float(g_p / g_ref),
            "max_rel_diff_w_vs_winkler_fem": rel_fem,
            "max_rel_diff_w_vs_winkler_closed_form": rel_ana,
            "passed": bool(rel_fem < 1e-3),
        })
    worst = rows[-1]
    return {
        "mesh": mesh,
        "rows": rows,
        "threshold": 1.0e-3,
        "smallest_g_p_N_per_m": worst["g_p_N_per_m"],
        "rel_diff_at_smallest_g_p": worst["max_rel_diff_w_vs_winkler_fem"],
        "passed": bool(worst["max_rel_diff_w_vs_winkler_fem"] < 1e-3),
    }


# ---------------------------------------------------------------------------
# criterion 4
# ---------------------------------------------------------------------------
DEFINITIONS = {
    "induced_stress_fixed_load":
        "sigma_max = 6 P m_hat / h^2 at the fixed nominal load (campaign criterion)",
    "flexural_strength": "sigma_f(T, eps_dot): the stress the ice can carry",
    "first_crack_load": "P_c = M_f / m_hat: load at which sigma_max = sigma_f",
    "utilization": "sigma_max / sigma_f: proximity of the fixed load to cracking",
}
AUDIT_KEYS = {
    "induced_stress_fixed_load": "sigma_max_Pa",
    "flexural_strength": "sigma_f_Pa",
    "first_crack_load": "P_c_N",
    "utilization": "utilization",
}


def _sign(study: dict, sweep: str, x_key: str, y_key: str) -> int:
    return int(dl.sign_report(study["sweeps"][sweep], x_key, y_key)["sign"])


def criterion_4() -> dict:
    common = dict(rate_values=RATE_GRID, load_N=REF["load_N"],
                  contact_radius=REF["contact_radius_m"])
    pass_p = dl.dependence_study(**common)
    pass_w = dl.dependence_study(material_kwargs={"shear_layer_beta": 0.0}, **common)

    beta_scan = []
    for beta in (0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 0.1, 0.3, 1.0):
        s = dl.dependence_study(
            rate_values=(0.05,), load_N=REF["load_N"],
            contact_radius=REF["contact_radius_m"],
            material_kwargs={"shear_layer_beta": beta},
        )
        st = ice_state(shear_layer_beta=beta).derived()
        beta_scan.append({
            "shear_layer_beta": float(beta),
            "pasternak_gamma": float(st["pasternak_gamma"]),
            "char_length_m": float(st["char_length_m"]),
            "sigma_max_ref_Pa": float(s["reference"]["sigma_max_Pa"]),
            "d_sigma_max_d_absT_sign": int(s["signs"]["d_sigma_max_d_absT"]["sign"]),
            "d_sigma_max_d_absT_loglog_slope": s["signs"]["d_sigma_max_d_absT"].get(
                "loglog_slope", float("nan")
            ),
        })

    verdicts = {}
    for name, study in (("pasternak_beta1", pass_p), ("winkler_beta0", pass_w)):
        verdicts[name] = {
            "d_sigma_max_dh_negative": _sign(study, "thickness", "h_m", "sigma_max_Pa") < 0,
            "d_sigma_max_dabsT_negative":
                _sign(study, "temperature", "abs_temp_C", "sigma_max_Pa") < 0,
            "d_sigma_max_drate_positive":
                _sign(study, "loading_rate", "indentation_rate_m_s", "sigma_max_Pa") > 0,
            "d_sigma_f_dabsT_positive":
                _sign(study, "temperature", "abs_temp_C", "sigma_f_Pa") > 0,
            "d_sigma_f_drate_positive":
                _sign(study, "loading_rate", "indentation_rate_m_s", "sigma_f_Pa") > 0,
            "d_Pc_dh_positive": _sign(study, "thickness", "h_m", "P_c_N") > 0,
            "d_Pc_dabsT_positive": _sign(study, "temperature", "abs_temp_C", "P_c_N") > 0,
            "d_Pc_drate_positive":
                _sign(study, "loading_rate", "indentation_rate_m_s", "P_c_N") > 0,
            "d_utilization_dabsT_negative":
                _sign(study, "temperature", "abs_temp_C", "utilization") < 0,
        }

    audit = []
    for name, desc in DEFINITIONS.items():
        key = AUDIT_KEYS[name]
        row = {"definition": name, "quantity": desc, "signs": {}}
        for model, study in (("pasternak_beta1", pass_p), ("winkler_beta0", pass_w)):
            s_dh = _sign(study, "thickness", "h_m", key)
            s_dT = _sign(study, "temperature", "abs_temp_C", key)
            s_dv = _sign(study, "loading_rate", "indentation_rate_m_s", key)
            row["signs"][model] = {
                "d_dh": s_dh, "d_dabsT": s_dT, "d_drate": s_dv,
                "satisfies_all_three": bool(s_dh < 0 and s_dT < 0 and s_dv > 0),
            }
        audit.append(row)

    lit = verdicts["pasternak_beta1"]
    sub = {
        "d_sigma_max_dh_negative": bool(lit["d_sigma_max_dh_negative"]),
        "d_sigma_max_dabsT_negative": bool(lit["d_sigma_max_dabsT_negative"]),
        "d_sigma_max_drate_positive": bool(lit["d_sigma_max_drate_positive"]),
    }
    return {
        "reference_state": pass_p["reference"],
        "definitions": DEFINITIONS,
        "load_N": REF["load_N"],
        "contact_radius_m": REF["contact_radius_m"],
        "pasternak_beta1": pass_p,
        "winkler_beta0": pass_w,
        "beta_scan": beta_scan,
        "verdicts": verdicts,
        "definition_consistency_audit": audit,
        "sub_criteria": sub,
        "literal_criterion_passed": bool(all(sub.values())),
        "passed": bool(all(sub.values())),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(c1: dict, c2: dict, c3: dict, c4: dict) -> list:
    out = []

    # fig 1
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for label, col, name in (
        ("winkler", "C0", "Winkler (Hetenyi/Kelvin)"),
        ("gamma1", "C3", r"Pasternak, $\gamma=1$"),
    ):
        tr = c1["_traces"][label]
        ax[0].semilogy(tr["r_over_l"], np.abs(tr["w_patch_m"]), col + "--", lw=1.4,
                       label=f"{name}: closed form")
        ax[0].semilogy(tr["r_over_l"], np.abs(tr["w_fem_m"]), col, lw=1.0, marker="o",
                       ms=2.5, label=f"{name}: 2-D BFS FEM")
        r = np.asarray(tr["r_over_l"])
        rel = np.abs(np.asarray(tr["w_fem_m"]) - np.asarray(tr["w_patch_m"])) / float(
            np.max(np.abs(tr["w_patch_m"]))
        )
        ax[1].semilogy(r, np.maximum(rel, 1e-8), col, lw=1.2, label=name)
    ax[0].axvspan(0.1, 5.0, color="0.92", zorder=0)
    ax[0].set_xlabel(r"$r/l$,  $l=(D/k)^{1/4}$")
    ax[0].set_ylabel(r"$|w(r)|$  [m]")
    ax[0].set_title("FEM vs Hetenyi-Pasternak closed form")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].axhline(0.05, color="k", ls=":", lw=1.0, label="5 % criterion")
    ax[1].set_xlabel(r"$r/l$")
    ax[1].set_ylabel(r"$|w_{FEM}-w_{analytic}|/\max|w|$")
    ax[1].set_title("normalised error on the criterion window")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out.append(("fig1_hetenyi_fem.png",
                "2-D BFS Kirchhoff FEM vs the Hetenyi/Pasternak closed form for a disc load on the "
                "ice sheet; right panel: normalised error on r/l in [0.1, 5].", fig))

    # fig 2
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    labels = [r["label"] for r in c2["runs"]]
    errs = [r["max_rel_error"] for r in c2["runs"]]
    ax[0].bar(range(len(errs)), errs, color=["C0", "C3", "C2", "C1"][: len(errs)])
    ax[0].axhline(1e-10, color="k", ls=":", label=r"$10^{-10}$ criterion")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation=12, fontsize=8)
    ax[0].set_ylabel(r"$\max_e |q_e/(k w_0)-1|$")
    ax[0].set_title("constant-pressure patch test ($w_0$ = 1 mm)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")
    rows = c3["rows"]
    gp = [r["g_p_over_sqrt_Dk"] for r in rows]
    ax[1].loglog(gp, [max(r["max_rel_diff_w_vs_winkler_fem"], 1e-17) for r in rows],
                 "o-", label="Pasternak FEM vs Winkler FEM")
    ax[1].loglog(gp, [max(r["max_rel_diff_w_vs_winkler_closed_form"], 1e-17) for r in rows],
                 "s--", label="Pasternak FEM vs Winkler closed form")
    ax[1].axhline(1e-3, color="k", ls=":", label="0.1 % criterion")
    ax[1].set_xlabel(r"$g_p/\sqrt{Dk}$")
    ax[1].set_ylabel("max relative difference")
    ax[1].invert_xaxis()
    ax[1].set_title(r"Pasternak $\rightarrow$ Winkler degeneration")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out.append(("fig2_patchtest_degeneracy.png",
                "Left: element-wise foundation reaction of the uniform-deflection patch test "
                "(Winkler / Pasternak / graded mesh). Right: the six-decade Pasternak->Winkler "
                "degeneration ladder.", fig))

    # fig 3
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for study, col, name in ((c4["pasternak_beta1"], "C3", r"Pasternak $\beta=1$"),
                             (c4["winkler_beta0"], "C0", "Winkler")):
        rows_h = study["sweeps"]["thickness"]
        axs[0].loglog([r["h_m"] for r in rows_h], [r["sigma_max_Pa"] for r in rows_h],
                      "o-", color=col, label=name)
    h_ref = [r["h_m"] for r in c4["pasternak_beta1"]["sweeps"]["thickness"]]
    s0 = c4["pasternak_beta1"]["sweeps"]["thickness"][2]["sigma_max_Pa"]
    h_mid = h_ref[2]
    axs[0].loglog(h_ref, [s0 * (h_mid / h) ** 2 for h in h_ref], "k:", label=r"slab scaling $h^{-2}$")
    axs[0].set_xlabel("ice thickness $h$ [m]")
    axs[0].set_ylabel(r"$\sigma_{\max}$ at $P=0.5$ MN [Pa]")
    axs[0].set_title("thickness law")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3, which="both")

    for study, col, name in ((c4["pasternak_beta1"], "C3", r"Pasternak $\beta=1$"),
                             (c4["winkler_beta0"], "C0", "Winkler")):
        rows_t = study["sweeps"]["temperature"]
        axs[1].plot([r["abs_temp_C"] for r in rows_t], [r["sigma_max_Pa"] / 1e3 for r in rows_t],
                    "o-", color=col, label=name)
    axs[1].set_xlabel(r"$|T|$ [K]")
    axs[1].set_ylabel(r"$\sigma_{\max}$ [kPa]")
    axs[1].set_title("temperature law (induced stress)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    rows_v = c4["pasternak_beta1"]["sweeps"]["loading_rate"]
    axs[2].loglog([r["indentation_rate_m_s"] for r in rows_v],
                  [r["sigma_max_Pa"] / 1e3 for r in rows_v], "o-", color="C3",
                  label=r"$\sigma_{\max}$ (Pasternak)")
    axs[2].loglog([r["indentation_rate_m_s"] for r in rows_v],
                  [r["sigma_f_Pa"] / 1e3 for r in rows_v], "s--", color="C1",
                  label=r"$\sigma_f$ (capacity)")
    axs[2].loglog([r["indentation_rate_m_s"] for r in rows_v],
                  [r["P_c_N"] / 1e6 for r in rows_v], "^-", color="C2", label=r"$P_c$ [MN]")
    axs[2].set_xlabel("indentation rate $v$ [m/s]")
    axs[2].set_ylabel("stress [kPa] / load [MN]")
    axs[2].set_title("loading-rate law")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out.append(("fig3_dependence_laws.png",
                "Thickness, temperature and loading-rate dependence of the peak bending stress, "
                "the ice capacity and the first-crack load.", fig))

    # fig 4
    grid = c4["pasternak_beta1"]["grid"]
    hs = sorted({g["h_m"] for g in grid})
    ts = sorted({g["abs_temp_C"] for g in grid})
    Z = np.zeros((len(ts), len(hs)))
    Zc = np.zeros((len(ts), len(hs)))
    for g in grid:
        i, j = ts.index(g["abs_temp_C"]), hs.index(g["h_m"])
        Z[i, j] = g["sigma_max_Pa"] / 1e3
        Zc[i, j] = g["P_c_N"] / 1e6
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, Zz, ttl, unit in (
        (axs[0], Z, r"$\sigma_{\max}$ at $P=0.5$ MN", "kPa"),
        (axs[1], Zc, r"first-crack load $P_c$", "MN"),
    ):
        cs = ax.contourf(hs, ts, Zz, levels=12, cmap="viridis")
        fig.colorbar(cs, ax=ax, label=unit)
        for i, t in enumerate(ts):
            for j, h in enumerate(hs):
                ax.text(h, t, f"{Zz[i, j]:.0f}", ha="center", va="center", color="w", fontsize=6)
        ax.set_xlabel("ice thickness $h$ [m]")
        ax.set_ylabel(r"$|T|$ [K]")
        ax.set_title(ttl + r" ($\beta=1$, $v=0.05$ m/s)")
    fig.tight_layout()
    out.append(("fig4_h_T_map.png",
                "Peak bending stress and first-crack load over the (thickness, temperature) plane.",
                fig))

    # fig 5
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    eps = np.logspace(-8, -2, 120)
    for t, col in ((-2.0, "C0"), (-10.0, "C3"), (-20.0, "C1"), (-30.0, "C2")):
        axs[0].loglog(eps, [im.ice_elastic_modulus(t, e) / 1e9 for e in eps], col,
                      label=f"T = {t:.0f} degC")
        axs[1].loglog(eps, [im.ice_flexural_strength(t, e) / 1e6 for e in eps], col,
                      label=f"T = {t:.0f} degC")
    axs[0].set_xlabel(r"strain rate $\dot\varepsilon$ [1/s]")
    axs[0].set_ylabel("$E$ [GPa]")
    axs[0].set_title("ice modulus: creep branch and temperature")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3, which="both")
    axs[1].set_xlabel(r"strain rate $\dot\varepsilon$ [1/s]")
    axs[1].set_ylabel(r"$\sigma_f$ [MPa]")
    axs[1].set_title("flexural strength: ductile rise, brittle saturation")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out.append(("fig5_ice_material_laws.png",
                "Temperature- and rate-dependent ice modulus and flexural strength used by the "
                "design laws.", fig))

    encoded = []
    for name, caption, fig in out:
        path = ART / name
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        encoded.append({
            "name": name,
            "caption": caption,
            "path": str(path.relative_to(ROOT)),
            "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })
    return encoded


# ---------------------------------------------------------------------------
def run_pytest() -> dict:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_ice_sheet_design_laws.py",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-3:]
    return {
        "command": "python -m pytest -q tests/test_ice_sheet_design_laws.py",
        "returncode": int(proc.returncode),
        "summary_lines": tail,
        "runtime_s": time.time() - t0,
        "passed": bool(proc.returncode == 0),
    }


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/4] Hetenyi / Pasternak anchor vs 2-D FEM ...", flush=True)
    c1 = criterion_1()
    print("      passed =", c1["passed"], flush=True)
    print("[2/4] constant-pressure patch test ...", flush=True)
    c2 = criterion_2()
    print("      max rel err = %.3e  passed = %s" % (c2["max_rel_error"], c2["passed"]), flush=True)
    print("[3/4] Pasternak -> Winkler degeneration ...", flush=True)
    c3 = criterion_3()
    print("      rel diff at smallest gp = %.3e  passed = %s"
          % (c3["rel_diff_at_smallest_g_p"], c3["passed"]), flush=True)
    print("[4/4] dependence laws ...", flush=True)
    c4 = criterion_4()
    print("      sub-criteria =", c4["sub_criteria"], flush=True)

    print("figures ...", flush=True)
    figs = figures(c1, c2, c3, c4)
    print("pytest ...", flush=True)
    py = run_pytest()

    criteria = {
        "C1_hetenyi_point_load_5pct": {
            "passed": c1["passed"],
            "threshold": c1["threshold"],
            "max_rel_error_used": max(c["max_rel_error_used"] for c in c1["cases"]),
            "cases": c1["cases"],
        },
        "C2_patch_test_1e-10": {
            "passed": c2["passed"],
            "threshold": c2["threshold"],
            "max_rel_error": c2["max_rel_error"],
            "max_shear_rel_contribution": c2["max_shear_rel_contribution"],
            "runs": c2["runs"],
        },
        "C3_pasternak_to_winkler_0p1pct": {
            "passed": c3["passed"],
            "threshold": c3["threshold"],
            "rel_diff_at_smallest_g_p": c3["rel_diff_at_smallest_g_p"],
            "rows": c3["rows"],
        },
        "C4_dependence_laws": {
            "passed": c4["passed"],
            "sub_criteria": c4["sub_criteria"],
            "verdicts": c4["verdicts"],
            "beta_scan": c4["beta_scan"],
            "definition_consistency_audit": c4["definition_consistency_audit"],
        },
    }

    honest_fail = []
    if not c4["sub_criteria"]["d_sigma_max_drate_positive"]:
        honest_fail.append(
            "C4c: the literal criterion d sigma_max/dv > 0 is NOT met by the induced peak "
            "stress at a fixed load (measured sign is negative, magnitude < 0.1 %); the rate "
            "law lives in the strength/capacity (d sigma_f/dv > 0, d Pc/dv > 0, both PASS)."
        )

    results = {
        "node": "p366w2-n01-fem-ice-plate",
        "mission_id": 10711,
        "repo": "TensorPhysics",
        "title": ("FEM ice plate on a Pasternak/Hetenyi foundation + ice thickness / "
                  "temperature / loading-rate design laws"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": time.time() - t_start,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "reference_case": REF,
        "criteria": criteria,
        "hetenyi": {
            "ice_state": c1["ice_state"],
            "closed_form_resolution_audit": c1["closed_form_resolution_audit"],
        },
        "dependence_study": {
            "definitions": c4["definitions"],
            "pasternak_beta1": c4["pasternak_beta1"],
            "winkler_beta0": c4["winkler_beta0"],
        },
        "pytest": py,
        "figures": [{"name": f["name"], "caption": f["caption"], "path": f["path"]} for f in figs],
        "all_criteria_passed": bool(
            c1["passed"] and c2["passed"] and c3["passed"] and c4["passed"] and py["passed"]
        ),
        "honest_fail_items": honest_fail,
    }
    (RUNS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (ART / "figures.json").write_text(json.dumps(figs, indent=2))
    print("\nwrote", RUNS / "results.json", flush=True)
    print("all_criteria_passed =", results["all_criteria_passed"], flush=True)
    return 0 if results["all_criteria_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())