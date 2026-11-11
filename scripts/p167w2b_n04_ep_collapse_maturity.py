"""P167W2B-N04 driver: elastoplastic progressive collapse verification surface.

Runs every acceptance check, writes ``artifacts/p167w2b_n04/results.json``
(all numbers consumed by the report are read from that file - zero hand copy),
``artifacts/p167w2b_n04/tables.md`` and five figures.

Usage: python scripts/p167w2b_n04_ep_collapse_maturity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.elastoplastic_collapse import axial_collapse_ep as ep  # noqa: E402

OUT = ROOT / "artifacts" / "p167w2b_n04"
OUT.mkdir(parents=True, exist_ok=True)

MAT = ep.IsotropicHardeningMaterial()
PANEL = ep.StiffenedPanelScantlings()
P0_ICE = 1.0e6  # Pa, ice belt nominal pressure at A0 (model parameter)
A0_ICE = 0.5    # m^2, reference area (model parameter)


# ---------------------------------------------------------------------------
# A. constitutive model: closed form vs incremental vs analytic tangent
# ---------------------------------------------------------------------------
def check_constitutive() -> dict:
    eps_path = np.linspace(0.0, 0.01, 201)
    closed = np.array([ep.uniaxial_closed_form(e, MAT)[0] for e in eps_path])
    incr = ep.uniaxial_incremental(eps_path, MAT)["sigma"]
    ebar_analytic = 0.0
    err_incr = float(np.max(np.abs(incr - closed) / np.maximum(np.abs(closed), 1e-30)))
    # analytic bilinear reference written out explicitly (independent of the module)
    ref = np.where(
        eps_path <= MAT.sigma_y0 / MAT.E,
        MAT.E * eps_path,
        MAT.sigma_y0 + MAT.H * (eps_path - MAT.sigma_y0 / MAT.E) / (1.0 + MAT.H / MAT.E),
    )
    err_vs_analytic = float(np.max(np.abs(closed - ref) / ref[-1]))
    # algorithmic tangent vs central difference of the closed form
    d_eps = 1e-9
    fd = (np.array([ep.uniaxial_closed_form(e + d_eps, MAT)[0] for e in eps_path[10:-1]])
          - np.array([ep.uniaxial_closed_form(e - d_eps, MAT)[0] for e in eps_path[10:-1]])) / (2 * d_eps)
    tang = np.array([ep.uniaxial_closed_form(e, MAT)[1] for e in eps_path[10:-1]])
    err_tang = float(np.max(np.abs(fd - tang) / tang))
    # 3-D return mapping: yield-surface consistency + consistent tangent FD
    strain = np.array([0.004, -0.0012, -0.0012, 0.0015, 0.0, 0.0])
    ep0 = np.zeros(6)
    rm = ep.j2_return_mapping(strain, ep0, 0.0, MAT)
    s3 = np.array([[rm["stress"][0], rm["stress"][3], rm["stress"][5]],
                   [rm["stress"][3], rm["stress"][1], rm["stress"][4]],
                   [rm["stress"][5], rm["stress"][4], rm["stress"][2]]])
    s_dev = s3 - np.trace(s3) / 3.0 * np.eye(3)
    vm = math.sqrt(1.5 * float(np.sum(s_dev * s_dev)))
    sy_new = MAT.sigma_y0 + MAT.H * rm["ebar_p"]
    yield_residual = abs(vm - sy_new) / sy_new
    C = ep.j2_consistent_tangent(strain, ep0, 0.0, MAT)
    C_fd = np.zeros((6, 6))
    h = 1e-9
    base = np.array(rm["stress"], dtype=float)
    for j in range(6):
        dp = np.zeros(6)
        dp[j] = h
        sp = np.array(ep.j2_return_mapping(strain + dp, ep0, 0.0, MAT)["stress"], dtype=float)
        sm = np.array(ep.j2_return_mapping(strain - dp, ep0, 0.0, MAT)["stress"], dtype=float)
        C_fd[:, j] = (sp - sm) / (2 * h)
    scale = np.max(np.abs(C))
    err_cons_tangent = float(np.max(np.abs(C - C_fd)) / scale)
    return {
        "material": {"E": MAT.E, "nu": MAT.nu, "sigma_y0": MAT.sigma_y0, "H": MAT.H,
                     "E_t": MAT.tangent_modulus, "eps_yield": MAT.eps_yield},
        "sigma_at_eps_0p01": float(closed[-1]),
        "ebar_p_at_eps_0p01": float(ep.uniaxial_closed_form(0.01, MAT)[2]),
        "max_rel_err_incremental_vs_closed": err_incr,
        "max_rel_err_closed_vs_analytic_bilinear": err_vs_analytic,
        "max_rel_err_tangent_vs_fd": err_tang,
        "yield_surface_consistency_residual": yield_residual,
        "max_rel_err_consistent_tangent_vs_fd": err_cons_tangent,
        "yielded": bool(rm["yielded"]),
        "_curve": {"eps": eps_path.tolist(), "sigma_closed": closed.tolist(),
                   "sigma_incremental": incr.tolist(), "sigma_analytic": ref.tolist()},
    }


# ---------------------------------------------------------------------------
# B. panel axial collapse vs Paik & Thayamballi simplified method
# ---------------------------------------------------------------------------
BETA_CASES = [
    ("stocky", dict(b=0.40, t=0.020, h_s=0.200, t_w=0.012, b_f=0.100, t_f=0.020)),
    ("benchmark", dict()),
    ("slender", dict(b=0.90, t=0.010, h_s=0.200, t_w=0.010, b_f=0.100, t_f=0.015)),
]
LENGTH_CASES = [("a_2.4m", 2.4), ("a_4.8m", 4.8), ("a_7.2m", 7.2)]


def check_panel() -> dict:
    rows = []
    for tag, kw in BETA_CASES:
        for ltag, a in LENGTH_CASES:
            panel = ep.StiffenedPanelScantlings(a=a, **kw)
            res = ep.smith_panel_axial_collapse(panel, MAT, n_steps=800)
            ref = ep.paik_thayamballi_simplified_ultimate(panel, MAT)
            dev = (res["sigma_u"] - ref["sigma_u"]) / ref["sigma_u"]
            rows.append({
                "case": f"{tag}/{ltag}",
                "a": panel.a, "b": panel.b, "t": panel.t,
                "beta_plate": res["beta_plate"], "phi_faulkner": res["phi_faulkner"],
                "sigma_u_MPa": res["sigma_u"] / 1e6,
                "sigma_u_over_sigma_y": res["sigma_u_over_sigma_y"],
                "eps_u": res["eps_u"],
                "reference_sigma_u_MPa": ref["sigma_u"] / 1e6,
                "reference_sigma_u_over_sigma_y": ref["sigma_u_over_sigma_y"],
                "deviation": dev,
                "post_peak_ratio": res["post_peak_ratio"],
                "sigma_E_stiffener_MPa": res["sigma_E_stiffener"] / 1e6,
                "perry_robertson_stiffener_MPa": res["perry_robertson_stiffener"] / 1e6,
                "eta_imperfection": res["eta_imperfection"],
                "mode": ("stiffener_column" if res["perry_robertson_stiffener"] < 0.5 * MAT.sigma_y0
                         else "plate_dominant"),
            })
    base = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=800)
    conv = []
    for n in (100, 200, 400, 800, 1600):
        r = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=n)
        conv.append({"n_steps": n, "sigma_u_MPa": r["sigma_u"] / 1e6})
    d01 = abs(conv[-1]["sigma_u_MPa"] - conv[-2]["sigma_u_MPa"]) / conv[-2]["sigma_u_MPa"]
    # monotonicity in plate slenderness (b/t increasing -> sigma_u decreasing)
    mono = []
    for b in (0.40, 0.55, 0.70, 0.85, 1.00):
        p = ep.StiffenedPanelScantlings(b=b)
        r = ep.smith_panel_axial_collapse(p, MAT, n_steps=800)
        mono.append({"b": b, "beta_plate": r["beta_plate"],
                     "sigma_u_MPa": r["sigma_u"] / 1e6,
                     "sigma_u_over_sigma_y": r["sigma_u_over_sigma_y"]})
    beta_seq = [m["sigma_u_MPa"] for m in mono]
    monotone_ok = all(beta_seq[i] > beta_seq[i + 1] for i in range(len(beta_seq) - 1))
    return {
        "matrix": rows,
        "max_abs_deviation_vs_reference": float(max(abs(r["deviation"]) for r in rows)),
        "benchmark": {
            "sigma_u_MPa": base["sigma_u"] / 1e6,
            "sigma_u_over_sigma_y": base["sigma_u_over_sigma_y"],
            "eps_u": base["eps_u"],
            "elastic_slope_MPa": base["elastic_slope"] / 1e6,
            "E_MPa": MAT.E / 1e6,
            "elastic_slope_rel_err": abs(base["elastic_slope"] - MAT.E) / MAT.E,
            "post_peak_ratio": base["post_peak_ratio"],
            "sigma_cr_plate_MPa": base["sigma_cr_plate"] / 1e6,
            "beta_plate": base["beta_plate"],
            "phi_faulkner": base["phi_faulkner"],
            "cap_plate_MPa": base["cap_plate"] / 1e6,
            "perry_robertson_stiffener_MPa": base["perry_robertson_stiffener"] / 1e6,
            "eta_imperfection": base["eta_imperfection"],
            "_curve": {"eps": base["eps"].tolist(),
                       "sigma_panel_MPa": (base["sigma_panel"] / 1e6).tolist(),
                       "sigma_plate_MPa": (base["sigma_plate"] / 1e6).tolist(),
                       "sigma_stiffener_MPa": (base["sigma_stiffener"] / 1e6).tolist()},
        },
        "convergence": conv,
        "convergence_rel_diff_800_1600": float(d01),
        "monotonicity_in_beta": mono,
        "monotonicity_ok": bool(monotone_ok),
    }


# ---------------------------------------------------------------------------
# C. IACS pressure-area law
# ---------------------------------------------------------------------------
def check_ice_pressure() -> dict:
    ident = ep.pressure_area_identifiability(P0_ICE, A0_ICE, exponent=-0.3)
    exponents = [ident["clean"]["exponent"]] + [f["exponent"] for f in ident["noisy"]]
    ratios = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    sampled = [{"A_over_A0": r, "P_MPa": ep.iacs_pressure(r * A0_ICE, P0_ICE, A0_ICE) / 1e6}
               for r in ratios]
    return {
        "law": "P = P0 * (max(A, A0)/A0)^-0.3",
        "P0_MPa": P0_ICE / 1e6,
        "A0_m2": A0_ICE,
        "sampled": sampled,
        "clean_fit": ident["clean"],
        "noisy_fits": ident["noisy"],
        "exponent_band": [-0.35, -0.25],
        "max_exponent_deviation_from_minus_0p3": float(max(abs(e + 0.3) for e in exponents)),
        "all_in_band": bool(all(-0.35 <= e <= -0.25 for e in exponents)),
        "regulatory_source_checked": False,
        "evidence_grade": "REAL (numerical inversion) / WITHHELD (regulatory text)",
    }


# ---------------------------------------------------------------------------
# D. plate elastic buckling vs Timoshenko
# ---------------------------------------------------------------------------
def check_buckling() -> dict:
    rows = []
    for aspect in (1.0, 1.5, 2.0, 2.5, 3.0):
        b = PANEL.b
        a = aspect * b
        tim = ep.elastic_plate_buckling_timoshenko(a, b, PANEL.t, MAT)
        fd = ep.fd_plate_buckling_stress(a, b, PANEL.t, MAT, h=b / 4.0, refinements=2)
        sig_fd = fd["sigma_cr"]
        sig_ref = tim["sigma_cr"]
        rows.append({
            "a_over_b": aspect,
            "k_timoshenko": tim["k"],
            "m_mode": tim["m"],
            "sigma_cr_timoshenko_MPa": sig_ref / 1e6,
            "sigma_cr_fd_MPa": sig_fd / 1e6,
            "sigma_cr_fd_richardson_MPa": (None if fd["richardson"] is None
                                           else fd["richardson"] / 1e6),
            "rel_err_finest_grid": abs(sig_fd - sig_ref) / sig_ref,
            "rel_err_richardson": (None if fd["richardson"] is None
                                   else abs(fd["richardson"] - sig_ref) / sig_ref),
            "observed_order": fd["observed_order"],
            "sequence_sigma_cr_MPa": [s["sigma_cr"] / 1e6 for s in fd["sequence"]],
        })
    max_err = max(r["rel_err_finest_grid"] for r in rows)
    max_err_rich = max(r["rel_err_richardson"] for r in rows if r["rel_err_richardson"] is not None)
    return {"rows": rows, "max_rel_err": float(max_err),
            "max_rel_err_richardson": float(max_err_rich)}


def main() -> int:
    cons = check_constitutive()
    panel = check_panel()
    ice = check_ice_pressure()
    buck = check_buckling()

    criteria = {
        "C1_uniaxial_vs_analytic_lt_0p01": {
            "value": cons["max_rel_err_closed_vs_analytic_bilinear"],
            "threshold": 0.01,
            "pass": cons["max_rel_err_closed_vs_analytic_bilinear"] < 0.01,
            "path": "results.json:constitutive.max_rel_err_closed_vs_analytic_bilinear"},
        "C1b_incremental_integrator_lt_1e-6": {
            "value": cons["max_rel_err_incremental_vs_closed"],
            "threshold": 1e-6,
            "pass": cons["max_rel_err_incremental_vs_closed"] < 1e-6,
            "path": "results.json:constitutive.max_rel_err_incremental_vs_closed"},
        "C1c_consistent_tangent_vs_fd_lt_1e-5": {
            "value": cons["max_rel_err_consistent_tangent_vs_fd"],
            "threshold": 1e-5,
            "pass": cons["max_rel_err_consistent_tangent_vs_fd"] < 1e-5,
            "path": "results.json:constitutive.max_rel_err_consistent_tangent_vs_fd"},
        "C1d_yield_surface_consistency_lt_1e-10": {
            "value": cons["yield_surface_consistency_residual"],
            "threshold": 1e-10,
            "pass": cons["yield_surface_consistency_residual"] < 1e-10,
            "path": "results.json:constitutive.yield_surface_consistency_residual"},
        "C2_sigma_u_vs_paik_simplified_lt_0p10": {
            "value": panel["max_abs_deviation_vs_reference"],
            "threshold": 0.10,
            "pass": panel["max_abs_deviation_vs_reference"] < 0.10,
            "path": "results.json:panel.max_abs_deviation_vs_reference"},
        "C2b_collapse_exists_and_below_hardened_yield": {
            "eps_u": panel["benchmark"]["eps_u"],
            "sigma_u_over_sigma_y": panel["benchmark"]["sigma_u_over_sigma_y"],
            "post_peak_ratio": panel["benchmark"]["post_peak_ratio"],
            "pass": bool(0.0 < panel["benchmark"]["eps_u"] < 0.012
                         and panel["benchmark"]["sigma_u_over_sigma_y"] <= 1.0
                         and panel["benchmark"]["post_peak_ratio"] < 1.0),
            "path": "results.json:panel.benchmark"},
        "C2c_step_convergence_lt_0p01": {
            "value": panel["convergence_rel_diff_800_1600"],
            "threshold": 0.01,
            "pass": panel["convergence_rel_diff_800_1600"] < 0.01,
            "path": "results.json:panel.convergence_rel_diff_800_1600"},
        "C2d_monotone_in_plate_slenderness": {
            "pass": panel["monotonicity_ok"],
            "values_MPa": [m["sigma_u_MPa"] for m in panel["monotonicity_in_beta"]],
            "path": "results.json:panel.monotonicity_in_beta"},
        "C2e_elastic_slope_is_E": {
            "value": panel["benchmark"]["elastic_slope_rel_err"],
            "threshold": 0.01,
            "pass": panel["benchmark"]["elastic_slope_rel_err"] < 0.01,
            "path": "results.json:panel.benchmark.elastic_slope_rel_err"},
        "C3_ice_exponent_in_band": {
            "band": [-0.35, -0.25],
            "clean_exponent": ice["clean_fit"]["exponent"],
            "noisy_exponents": [f["exponent"] for f in ice["noisy_fits"]],
            "pass": bool(ice["all_in_band"]),
            "path": "results.json:ice_pressure"},
        "C4_buckling_vs_timoshenko_lt_0p05": {
            "value": buck["max_rel_err"],
            "threshold": 0.05,
            "pass": buck["max_rel_err"] < 0.05,
            "path": "results.json:buckling.max_rel_err"},
        "C4b_richardson_vs_timoshenko_lt_0p01": {
            "value": buck["max_rel_err_richardson"],
            "threshold": 0.01,
            "pass": buck["max_rel_err_richardson"] < 0.01,
            "path": "results.json:buckling.max_rel_err_richardson"},
    }
    overall = all(c["pass"] for c in criteria.values())

    results = {
        "mission": "P167W2B-N04",
        "node": "TensorFEM/TensorPhysics elastoplastic progressive collapse maturity",
        "repository": "TensorPhysics",
        "module": "tensorphysics.elastoplastic_collapse.axial_collapse_ep",
        "domain": "ship structures: ultimate strength of stiffened panels / ice belt",
        "model_constants": {
            "residual_stress_ratio": 0.15,
            "softening_ratio": 0.05,
            "imperfection_ratio_L": 1.0 / 1000.0,
            "n_steps_default": 800,
            "eps_max": 0.012,
            "plate_k": 4.0,
        },
        "benchmark_scantlings": {
            "a": PANEL.a, "b": PANEL.b, "t": PANEL.t, "h_s": PANEL.h_s,
            "t_w": PANEL.t_w, "b_f": PANEL.b_f, "t_f": PANEL.t_f,
            "plate_area": PANEL.plate_area, "stiffener_area": PANEL.stiffener_area,
            "total_area": PANEL.total_area,
        },
        "constitutive": cons,
        "panel": panel,
        "ice_pressure": ice,
        "buckling": buck,
        "criteria": criteria,
        "all_criteria_pass": bool(overall),
        "evidence_grades": {
            "constitutive": "REAL",
            "panel_collapse": "REAL (model) / method-level benchmark vs Paik & Thayamballi simplified method",
            "ice_pressure": "REAL (numerical inversion) / WITHHELD (regulatory text)",
            "buckling": "REAL",
        },
        "limitations": [
            "Panel model is a Smith fiber model: element curves (von Karman post-buckling, "
            "Faulkner ultimate, Perry-Robertson column) are published closed forms, not a "
            "full 3-D shell FE, so tripping/lateral-torsional stiffener buckling and plate "
            "membrane stress redistribution are outside this verification surface.",
            "The Paik & Thayamballi (2003) reference is their Ch.5 simplified method "
            "(rigid-plastic element summation) implemented from published element curves; the "
            "empirical closed-form ultimate-strength equations were NOT used because the "
            "primary text is not available offline (WITHHELD) - no coefficient was copied "
            "from memory.",
            "Post-ultimate shedding slope (0.05 E) and residual stress factor (0.15) are "
            "declared model constants, not fitted to the benchmark.",
            "IACS/ISO regulatory wording is WITHHELD; only the numerical power-law inversion "
            "and its identifiability under 5% pressure scatter are claimed.",
        ],
    }

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    (OUT / "results.json").write_text(json.dumps(_clean(results), indent=2, ensure_ascii=False))

    # ---- figures -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(cons["_curve"]["eps"], [s / 1e6 for s in cons["_curve"]["sigma_closed"]],
            label="closed form (hardening)", lw=2)
    ax.plot(cons["_curve"]["eps"][::10], [s / 1e6 for s in cons["_curve"]["sigma_incremental"][::10]],
            "o", ms=3.5, label="incremental radial return")
    ax.plot(cons["_curve"]["eps"], [s / 1e6 for s in cons["_curve"]["sigma_analytic"]],
            "--", label="analytic bilinear")
    ax.axhline(MAT.sigma_y0 / 1e6, color="grey", ls=":", lw=1, label=r"$\sigma_{y0}$")
    ax.set_xlabel("strain [-]")
    ax.set_ylabel("stress [MPa]")
    ax.set_title("von Mises + isotropic hardening: closed form vs incremental")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_uniaxial.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    c = panel["benchmark"]["_curve"]
    ax.plot(c["eps"], c["sigma_panel_MPa"], lw=2, label=r"panel $\sigma(\varepsilon)$")
    ax.plot(c["eps"], c["sigma_plate_MPa"], "--", label="plating element")
    ax.plot(c["eps"], c["sigma_stiffener_MPa"], "-.", label="stiffener element")
    ax.plot([panel["benchmark"]["eps_u"]], [panel["benchmark"]["sigma_u_MPa"]], "r*", ms=12,
            label=r"$\sigma_u$")
    ax.set_xlabel(r"imposed shortening $\varepsilon$ [-]")
    ax.set_ylabel("stress [MPa]")
    ax.set_ylim(0, max(c["sigma_panel_MPa"]) * 1.25)
    ax.set_title("Smith progressive collapse, elastoplastic element curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_panel_collapse.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in buck["rows"]:
        seq = r["sequence_sigma_cr_MPa"]
        ax.plot([r["a_over_b"]] * len(seq), seq, "o", ms=3, color="C0", alpha=0.6)
    ax.plot([r["a_over_b"] for r in buck["rows"]],
            [r["sigma_cr_timoshenko_MPa"] for r in buck["rows"]], "s-", color="C3",
            label="Timoshenko closed form")
    ax.plot([r["a_over_b"] for r in buck["rows"]],
            [r["sigma_cr_fd_richardson_MPa"] for r in buck["rows"]], "^--", color="C2",
            label="FD + Richardson")
    ax.set_xlabel("aspect ratio a/b")
    ax.set_ylabel(r"$\sigma_{cr}$ [MPa]")
    ax.set_title("Simply supported plate buckling: FD sequence vs Timoshenko")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_buckling.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ratios = np.geomspace(1.0, 10.0, 40)
    ax.loglog(ratios, [ep.iacs_pressure(r * A0_ICE, P0_ICE, A0_ICE) / 1e6 for r in ratios],
              lw=2, label=r"$P=P_0(A/A_0)^{-0.3}$")
    for f in ice["noisy_fits"]:
        ratios_f = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
        p_noisy = np.array([ep.iacs_pressure(r * A0_ICE, P0_ICE, A0_ICE) for r in ratios_f])
        rng = np.random.default_rng(f["seed"])
        p_noisy = p_noisy * np.exp(rng.normal(0.0, 0.05, p_noisy.size))
        ax.loglog(ratios_f, p_noisy / 1e6, "o", ms=3.5, alpha=0.7,
                  label=f"noisy sample (seed {f['seed']})")
        ax.loglog(ratios, f["P0"] * (ratios) ** f["exponent"] / 1e6, ":",
                  label=f"fit n={f['exponent']:.4f}")
    ax.set_xlabel(r"$A/A_0$ [-]")
    ax.set_ylabel("pressure [MPa]")
    ax.set_title("IACS ice-belt pressure-area law and exponent inversion")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_pressure_area.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(panel["matrix"]))
    ax.plot(xs, [r["sigma_u_MPa"] for r in panel["matrix"]], "o-", label="Smith + elastoplastic elements")
    ax.plot(xs, [r["reference_sigma_u_MPa"] for r in panel["matrix"]], "s--",
            label="Paik & Thayamballi simplified")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["case"] for r in panel["matrix"]], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(r"$\sigma_u$ [MPa]")
    ax.set_title("Panel ultimate strength: model vs published simplified method")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_benchmark.png", dpi=160)
    plt.close(fig)

    # ---- tables --------------------------------------------------------
    L = []
    L.append("### constitutive model\n")
    L.append("| quantity | value | criterion |\n|---|---|---|")
    L.append(f"| sigma at eps=1% [MPa] | {cons['sigma_at_eps_0p01']/1e6:.3f} | - |")
    L.append(f"| ebar_p at eps=1% [-] | {cons['ebar_p_at_eps_0p01']:.6e} | - |")
    L.append(f"| incremental vs closed form | {cons['max_rel_err_incremental_vs_closed']:.3e} | < 1e-6 |")
    L.append(f"| closed form vs analytic | {cons['max_rel_err_closed_vs_analytic_bilinear']:.3e} | < 1% |")
    L.append(f"| tangent vs finite difference | {cons['max_rel_err_tangent_vs_fd']:.3e} | < 1e-5 |")
    L.append(f"| consistent tangent vs FD | {cons['max_rel_err_consistent_tangent_vs_fd']:.3e} | < 1e-5 |")
    L.append(f"| yield-surface residual | {cons['yield_surface_consistency_residual']:.3e} | < 1e-10 |\n")
    L.append("### panel collapse matrix\n")
    L.append("| case | beta | Phi | sigma_u [MPa] | sigma_u/sigma_y | eps_u | ref [MPa] | deviation |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in panel["matrix"]:
        L.append(f"| {r['case']} | {r['beta_plate']:.3f} | {r['phi_faulkner']:.3f} | "
                 f"{r['sigma_u_MPa']:.2f} | {r['sigma_u_over_sigma_y']:.4f} | {r['eps_u']:.5f} | "
                 f"{r['reference_sigma_u_MPa']:.2f} | {100*r['deviation']:+.2f}% |")
    L.append(f"\nmax |deviation| = {100*panel['max_abs_deviation_vs_reference']:.2f}% (criterion < 10%)\n")
    L.append("### step convergence\n")
    L.append("| n_steps | sigma_u [MPa] |\n|---|---|")
    for r in panel["convergence"]:
        L.append(f"| {r['n_steps']} | {r['sigma_u_MPa']:.4f} |")
    L.append("")
    L.append("### plate buckling\n")
    L.append("| a/b | k | sigma_cr Timoshenko [MPa] | sigma_cr FD [MPa] | rel err | Richardson err |")
    L.append("|---|---|---|---|---|---|")
    for r in buck["rows"]:
        L.append(f"| {r['a_over_b']:.1f} | {r['k_timoshenko']:.3f} | {r['sigma_cr_timoshenko_MPa']:.3f} | "
                 f"{r['sigma_cr_fd_MPa']:.3f} | {100*r['rel_err_finest_grid']:.3f}% | "
                 f"{100*r['rel_err_richardson']:.3f}% |")
    L.append("")
    L.append("### IACS pressure-area inversion\n")
    L.append("| data set | exponent | P0 [MPa] | r2 |")
    L.append("|---|---|---|---|")
    L.append(f"| clean | {ice['clean_fit']['exponent']:.6f} | {ice['clean_fit']['P0']/1e6:.6f} | "
             f"{ice['clean_fit']['r2']:.6f} |")
    for f in ice["noisy_fits"]:
        L.append(f"| noisy seed {f['seed']} | {f['exponent']:.6f} | {f['P0']/1e6:.4f} | {f['r2']:.4f} |")
    L.append("")
    (OUT / "tables.md").write_text("\n".join(L))

    print(json.dumps({k: v["pass"] for k, v in criteria.items()}, indent=2))
    print("ALL PASS:", overall)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())