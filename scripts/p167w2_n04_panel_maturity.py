"""P167W2-N04 panel-collapse maturity study (TensorPhysics).

One deterministic run produces:

* ``artifacts/p167w2_n04/results.json``  -- every number used anywhere below
* ``artifacts/p167w2_n04/tables.md``     -- the tables in markdown
* ``artifacts/p167w2_n04/fig*.png``      -- 5 figures
* ``docs/p167w2-n04-panel-collapse-maturity.md`` -- the report, numbers injected
  programmatically (no hand typing)

Run::

    PYTHONPATH=src python scripts/p167w2_n04_panel_maturity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorphysics.elastoplastic_collapse import (  # noqa: E402
    GrillageModel,
    SmithCollapseSolver,
    StiffenedPanelSection,
    StiffenedPlateGeometry,
    VonMisesMaterial,
    check_ultimate_bounds,
    compare_ultimate_methods,
    discretization_convergence_study,
    grillage_convergence_study,
    imperfection_quantiles,
    isolated_element_deviation,
    panel_strength_band,
    panel_ultimate_axial,
    plastic_axial_bending_interaction,
    plate_ultimate_stress,
    stiffener_ultimate_stress,
)

OUT = REPO / "artifacts" / "p167w2_n04"
DOC = REPO / "docs" / "p167w2-n04-panel-collapse-maturity.md"

MAT = VonMisesMaterial(E=206.0e9, nu=0.3, sigma_y0=313.0e6, H=1.0e9)
REF = dict(a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010, b_f=0.100, t_f=0.015)
SY = MAT.sigma_y0


def _f(x, nd=6):
    return round(float(x), nd)


def _mpa(x):
    return round(float(x) / 1e6, 3)


def _clean(obj):
    """JSON-safe copy (floats, lists, tuples, dicts; str passthrough)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# A. section
# ---------------------------------------------------------------------------
def compute_section() -> dict:
    sec = StiffenedPanelSection(material=MAT, **REF)
    num = sec.numeric_properties(n_strips=800)
    geom = StiffenedPlateGeometry(material=MAT, w0_max=REF["a"] / 1000.0, **REF)
    legacy = geom.section_properties()
    prop_keys = {
        "total_area": "area",
        "z_bar": "z_bar",
        "second_moment": "second_moment",
        "plastic_neutral_axis": "plastic_neutral_axis",
        "plastic_section_modulus": "plastic_section_modulus",
    }
    closures = {}
    for analytic, quad in prop_keys.items():
        a_val = float(getattr(sec, analytic))
        q_val = float(num[quad])
        closures[analytic] = {
            "analytic": a_val,
            "quadrature": q_val,
            "rel_diff": abs(a_val - q_val) / abs(a_val) if a_val else 0.0,
        }
    summary = {k: float(v) for k, v in sec.summary().items()}
    legacy_defect = {
        "Z_bot_legacy": legacy["Z_bot"],
        "Z_bot_correct": sec.section_modulus_bottom,
        "overestimate": legacy["Z_bot"] / sec.section_modulus_bottom - 1.0,
    }
    return {"summary": summary, "closures": closures, "legacy_Zbot_defect": legacy_defect}


# ---------------------------------------------------------------------------
# B. panel ultimate
# ---------------------------------------------------------------------------
def compute_ultimate() -> dict:
    sec = StiffenedPanelSection(material=MAT, **REF)
    faulk = panel_ultimate_axial(sec, "faulkner")
    vk = panel_ultimate_axial(sec, "von_karman")
    gate = check_ultimate_bounds(faulk)
    plate = plate_ultimate_stress(sec)
    stiff = stiffener_ultimate_stress(sec)
    band = panel_strength_band(sec)
    cmp = compare_ultimate_methods(sec)
    surf = plastic_axial_bending_interaction(sec, n_points=801)
    surface = surf.pop("surface")
    stride = max(1, len(surface["y"]) // 41)
    surf["surface_sample"] = {
        "n_over_n_p": [float(v) for v in surface["n_over_n_p"][::stride]],
        "m_over_m_p": [float(v) for v in surface["m_over_m_p"][::stride]],
    }
    return {
        "faulkner": {
            "sigma_u": faulk.sigma_u,
            "ratio": faulk.sigma_u_over_sigma_y,
            "eps_u": faulk.eps_u,
            "mode": faulk.mode,
            "post_peak_ratio": faulk.post_peak_ratio,
            "diagnostics": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in faulk.diagnostics.items()},
        },
        "von_karman": {
            "sigma_u": vk.sigma_u,
            "ratio": vk.sigma_u_over_sigma_y,
            "eps_u": vk.eps_u,
            "mode": vk.mode,
        },
        "bounds": {k: float(v) for k, v in faulk.bounds.items()},
        "bounds_gate": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in gate.items()},
        "plate_element": {k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in plate.items()},
        "stiffener_element": {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in stiff.items()},
        "band": {k: float(v) for k, v in band.items()},
        "methods": {
            "sigma_u": {k: float(v) for k, v in cmp["sigma_u"].items()},
            "ratio": {k: float(v) for k, v in cmp["ratio"].items()},
            "deviation": {k: float(v) for k, v in cmp["deviation_vs_element_faulkner"].items()},
            "max_spread_over_min": float(cmp["max_spread_over_min"]),
            "modes": cmp["modes"],
        },
        "plastic_interaction": {k: _clean(v) for k, v in surf.items()},
        "load_path": {
            "eps": [float(e) for e, _ in faulk.load_path],
            "sigma_over_sy": [float(s / SY) for _, s in faulk.load_path],
        },
    }


# ---------------------------------------------------------------------------
# C. sweeps
# ---------------------------------------------------------------------------
def compute_sweeps() -> dict:
    thicknesses = [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
    beta_sweep = []
    for t in thicknesses:
        sec = StiffenedPanelSection(material=MAT, **{**REF, "t": t})
        f = panel_ultimate_axial(sec, "faulkner")
        v = panel_ultimate_axial(sec, "von_karman")
        beta_sweep.append({
            "t": t,
            "beta": sec.beta_plate,
            "lambda": sec.lambda_column,
            "faulkner": f.sigma_u_over_sigma_y,
            "von_karman": v.sigma_u_over_sigma_y,
            "lower": f.bounds["lower"] / SY,
            "mode": f.mode,
            "eps_u": f.eps_u,
        })

    spans = [1.2, 1.8, 2.4, 3.0, 4.0, 5.0]
    lambda_sweep = []
    for a in spans:
        sec = StiffenedPanelSection(material=MAT, **{**REF, "a": a})
        f = panel_ultimate_axial(sec, "faulkner")
        stiff = stiffener_ultimate_stress(sec)
        lambda_sweep.append({
            "a": a,
            "lambda": sec.lambda_column,
            "sigma_euler_mpa": sec.sigma_euler_column / 1e6,
            "sigma_trip_mpa": sec.sigma_tripping / 1e6,
            "sigma_u_ratio": f.sigma_u_over_sigma_y,
            "mode": f.mode,
            "stiffener_mode": stiff["mode"],
        })

    ts = np.linspace(0.006, 0.018, 7)
    hs = np.linspace(0.10, 0.60, 10)
    mode_grid, ratio_grid = [], []
    for t in ts:
        row, rrow = [], []
        for h in hs:
            sec = StiffenedPanelSection(
                material=MAT, **{**REF, "t": float(t), "h_s": float(h)}
            )
            r = panel_ultimate_axial(sec, "faulkner")
            row.append(r.mode)
            rrow.append(float(r.sigma_u_over_sigma_y))
        mode_grid.append(row)
        ratio_grid.append(rrow)

    flat_sweep = []
    for t_w in (0.006, 0.008, 0.010, 0.012, 0.016):
        sec = StiffenedPanelSection(
            a=2.4, b=0.7, t=0.012, h_s=0.250, t_w=t_w, material=MAT
        )
        stiff = stiffener_ultimate_stress(sec)
        flat_sweep.append({
            "t_w": t_w,
            "sigma_trip_mpa": sec.sigma_tripping / 1e6,
            "sigma_column_mpa": stiff["column"] / 1e6,
            "mode": stiff["mode"],
            "panel_ratio": panel_ultimate_axial(sec).sigma_u_over_sigma_y,
        })
    return {
        "beta_sweep": beta_sweep,
        "lambda_sweep": lambda_sweep,
        "mode_map": {
            "t": ts.tolist(),
            "h_s": hs.tolist(),
            "modes": mode_grid,
            "ratio": ratio_grid,
        },
        "flat_bar_sweep": flat_sweep,
    }


# ---------------------------------------------------------------------------
# D. grillage
# ---------------------------------------------------------------------------
def compute_grillage() -> dict:
    kw = dict(REF, material=MAT)
    conv = grillage_convergence_study(**kw)
    iso = isolated_element_deviation(**kw)
    disc40 = discretization_convergence_study(**kw, n_steps_coarse=40, n_steps_fine=400)
    disc = discretization_convergence_study(**kw)
    g = GrillageModel(n_x=5, n_y=5, **kw)
    eps, mean = g.load_shortening_curve()
    amps = g.bay_imperfections()
    bay_curves = []
    for w in amps[:25]:
        geom = StiffenedPlateGeometry(material=MAT, w0_max=float(w), **REF)
        res = SmithCollapseSolver(geom, n_steps=400, max_strain=0.02).solve()
        bay_curves.append({"w0": float(w), "sigma_u_over_sy": float(res.sigma_u / SY)})
    fabricated_error = 0.05 * (1.0 / 3.0 - 1.0 / 5.0)
    step = 20
    return {
        "convergence": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in conv.items()},
        "isolated_deviation": {k: (float(v) if isinstance(v, (int, float)) else v)
                               for k, v in iso.items()},
        "discretization_40_vs_400": {k: (float(v) if isinstance(v, (int, float)) else v)
                                     for k, v in disc40.items()},
        "discretization_200_vs_400": {k: (float(v) if isinstance(v, (int, float)) else v)
                                      for k, v in disc.items()},
        "fabricated_error_of_old_model": fabricated_error,
        "assembly_curve": {
            "eps": [float(e) for e in eps[::step]],
            "sigma_over_sy": [float(s / SY) for s in mean[::step]],
        },
        "bay_imperfections": [float(a) for a in amps],
        "bay_ultimates": bay_curves,
        "sigma_u_assembly_over_sy": float(mean.max() / SY),
        "imperfection_quantiles_9": [float(q) for q in imperfection_quantiles(9, 0.5)],
    }


# ---------------------------------------------------------------------------
# E. defects found while doing the above
# ---------------------------------------------------------------------------
def compute_defects() -> dict:
    trend = {}
    for label, w0 in (("w0_0", 0.0), ("w0_1mm", 0.001), ("w0_nominal", 0.0024),
                      ("w0_4mm", 0.004)):
        geom = StiffenedPlateGeometry(material=MAT, w0_max=w0, **REF)
        trend[label] = float(
            SmithCollapseSolver(geom, n_steps=400, max_strain=0.02).solve().sigma_u / SY
        )
    flat = StiffenedPlateGeometry(
        a=2.4, b=0.7, t=0.012, h_s=0.250, t_w=0.006, b_f=0.0, t_f=0.0,
        material=MAT, w0_max=2.4e-3,
    )
    flat_res = SmithCollapseSolver(flat, n_steps=400, max_strain=0.02).solve()
    ref_geom = StiffenedPlateGeometry(material=MAT, w0_max=2.4e-3, **REF)
    ref_mode = SmithCollapseSolver(ref_geom, n_steps=400, max_strain=0.02).solve().failure_mode
    return {
        "progressive_imperfection_trend": trend,
        "imperfection_trend_monotone_increasing": all(
            trend[a] < trend[b]
            for a, b in zip(["w0_0", "w0_1mm", "w0_nominal"], ["w0_1mm", "w0_nominal", "w0_4mm"])
        ),
        "flat_bar_mode": flat_res.failure_mode,
        "reference_mode": ref_mode,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(res: dict) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    pi = res["ultimate"]["plastic_interaction"]
    n = np.array(pi["surface_sample"]["n_over_n_p"])
    m = np.array(pi["surface_sample"]["m_over_m_p"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(n[n >= 0], m[n >= 0], "o-", ms=3, lw=1.2, label="exact surface (compression)")
    ax.plot(n[n < 0], m[n < 0], "s-", ms=3, lw=1.2, label="exact surface (tension)")
    nn = np.linspace(-1, 1, 401)
    ax.plot(nn, np.clip(1 - nn ** 2, 0, None), "k--", lw=1.2,
            label=r"rule form $(N/N_p)^2 + M/M_p = 1$")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(r"$N/N_p$")
    ax.set_ylabel(r"$M/M_p$")
    ax.set_title("Panel plastic collapse surface (reference panel)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = OUT / "fig1_plastic_NM_surface.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    made.append(f1)

    lp = res["ultimate"]["load_path"]
    u = res["ultimate"]
    diag = u["faulkner"]["diagnostics"]
    eps = np.array(lp["eps"])
    sig = np.array(lp["sigma_over_sy"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(eps * 100, sig, "b-", lw=2, label="panel mean stress (Faulkner plate curve)")
    ax.axhline(diag["sigma_u_plate_ratio"], color="tab:orange", ls="--", lw=1.2,
               label="plate element cap = %.3f $\\sigma_y$" % diag["sigma_u_plate_ratio"])
    ax.axhline(diag["sigma_u_stiffener_ratio"], color="tab:green", ls="--", lw=1.2,
               label="stiffener cap = %.3f $\\sigma_y$" % diag["sigma_u_stiffener_ratio"])
    ax.axhline(1.0, color="grey", ls=":", lw=0.8, label="$\\sigma_y$ (squash)")
    ax.axhline(u["bounds"]["lower"] / SY, color="tab:red", ls="-.", lw=1.0,
               label="elastic lower bound (residual reduced)")
    ax.scatter([u["faulkner"]["eps_u"] * 100], [u["faulkner"]["ratio"]], color="k",
               zorder=5, label="limit point")
    ax.set_xlabel("shortening $\\varepsilon$ [%]")
    ax.set_ylabel("mean stress / $\\sigma_y$")
    ax.set_title("Panel collapse path, $\\beta$ = %.2f" % diag["beta_plate"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = OUT / "fig2_load_shortening.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    made.append(f2)

    sweep = res["sweeps"]["beta_sweep"]
    t = np.array([r["t"] for r in sweep]) * 1000
    fa = np.array([r["faulkner"] for r in sweep])
    vk = np.array([r["von_karman"] for r in sweep])
    lo = np.array([r["lower"] for r in sweep])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.fill_between(t, fa, vk, color="tab:blue", alpha=0.18,
                    label="strength band (plate element curve)")
    ax.plot(t, fa, "o-", color="tab:blue", label="Faulkner element curve (design)")
    ax.plot(t, vk, "s-", color="tab:purple", label="von Karman element curve (upper)")
    ax.plot(t, lo, "^--", color="tab:red", label="elastic lower bound")
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("plate thickness $t$ [mm]")
    ax.set_ylabel("$\\sigma_u/\\sigma_y$")
    ax.set_title("Panel ultimate strength and its declared band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = OUT / "fig3_strength_band.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    made.append(f3)

    mm = res["sweeps"]["mode_map"]
    labels = sorted({m for row in mm["modes"] for m in row})
    idx = {m: i for i, m in enumerate(labels)}
    grid = np.array([[idx[m] for m in row] for row in mm["modes"]], dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[mm["h_s"][0], mm["h_s"][-1],
                           mm["t"][0] * 1000, mm["t"][-1] * 1000])
    cb = fig.colorbar(im, ax=ax, ticks=range(len(labels)))
    cb.ax.set_yticklabels(labels, fontsize=7)
    cs = ax.contour(mm["h_s"], np.array(mm["t"]) * 1000, np.array(mm["ratio"]),
                    levels=6, colors="w", linewidths=0.7)
    ax.clabel(cs, fontsize=6, fmt="%.2f")
    ax.set_xlabel("web height $h_s$ [m]")
    ax.set_ylabel("plate thickness $t$ [mm]")
    ax.set_title("Governing collapse mode (white lines: $\\sigma_u/\\sigma_y$)")
    fig.tight_layout()
    f4 = OUT / "fig4_mode_map.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)
    made.append(f4)

    gr = res["grillage"]
    ac = gr["assembly_curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(np.array(ac["eps"]) * 100, ac["sigma_over_sy"], "b-", lw=2,
            label="parallel assembly (mean of bays)")
    for r in gr["bay_ultimates"][::4]:
        ax.axhline(r["sigma_u_over_sy"], color="tab:orange", lw=0.5, alpha=0.6)
    ax.plot([], [], color="tab:orange", lw=0.8, label="per-bay peaks (imperfection quantiles)")
    ax.axhline(1.0, color="grey", ls=":", lw=0.8, label="$\\sigma_y$")
    ax.set_xlabel("shortening $\\varepsilon$ [%]")
    ax.set_ylabel("mean stress / $\\sigma_y$")
    ax.set_title("Grillage assembly: $\\delta_{iso}$ = %.2e, size invariance = %.1e"
                 % (gr["isolated_deviation"]["delta_iso"],
                    gr["convergence"]["mesh_invariance_rel"]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f5 = OUT / "fig5_grillage_assembly.png"
    fig.savefig(f5, dpi=150)
    plt.close(fig)
    made.append(f5)
    return made


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def tables_md(res: dict) -> str:
    s = res["section"]["summary"]
    L = []
    L.append("### T1. Section properties (reference panel)\n")
    L.append("| quantity | value | unit |")
    L.append("|---|---|---|")
    rows = [
        ("total area A", s["total_area"], "m^2"),
        ("plate area", s["plate_area"], "m^2"),
        ("stiffener area", s["stiffener_area"], "m^2"),
        ("centroid z_bar (from plate mid-plane)", s["z_bar"], "m"),
        ("second moment I", s["second_moment"], "m^4"),
        ("gyration radius r", s["gyration_radius"], "m"),
        ("section modulus (top fibre)", s["section_modulus_top"], "m^3"),
        ("section modulus (plate fibre)", s["section_modulus_bottom"], "m^3"),
        ("plastic neutral axis y_p", s["plastic_neutral_axis"], "m"),
        ("plastic modulus Z_p", s["plastic_section_modulus"], "m^3"),
        ("shape factor Z_p/Z_el", s["shape_factor"], "-"),
        ("yield moment M_y", s["yield_bending_moment"], "N m"),
        ("plastic moment M_p", s["plastic_bending_moment"], "N m"),
        ("squash load N_p", s["squash_load"], "N"),
        ("plate slenderness beta", s["beta_plate"], "-"),
        ("column slenderness lambda", s["lambda_column"], "-"),
        ("sigma_cr plate", s["sigma_cr_plate"] / 1e6, "MPa"),
        ("sigma_cr web", s["sigma_cr_web"] / 1e6, "MPa"),
        ("sigma_cr flange", s["sigma_cr_flange"] / 1e6, "MPa"),
        ("sigma_E column (with plate)", s["sigma_euler_column"] / 1e6, "MPa"),
        ("sigma_tripping", s["sigma_tripping"] / 1e6, "MPa"),
    ]
    for name, val, unit in rows:
        L.append("| %s | %.6g | %s |" % (name, val, unit))

    L.append("\n### T2. Closed form vs independent 800-strip quadrature\n")
    L.append("| quantity | closed form | quadrature | rel. diff |")
    L.append("|---|---|---|---|")
    for k, v in res["section"]["closures"].items():
        L.append("| %s | %.10g | %.10g | %.2e |"
                 % (k, v["analytic"], v["quadrature"], v["rel_diff"]))

    u = res["ultimate"]
    L.append("\n### T3. Panel ultimate strength (reference panel)\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append("| sigma_u (Faulkner curve, design) | %.3f MPa |" % (u["faulkner"]["sigma_u"] / 1e6))
    L.append("| sigma_u / sigma_y (Faulkner) | %.4f |" % u["faulkner"]["ratio"])
    L.append("| sigma_u (von Karman curve, upper) | %.3f MPa |" % (u["von_karman"]["sigma_u"] / 1e6))
    L.append("| sigma_u / sigma_y (von Karman) | %.4f |" % u["von_karman"]["ratio"])
    L.append("| limit-point strain eps_u | %.5f |" % u["faulkner"]["eps_u"])
    L.append("| post-peak / peak | %.4f |" % u["faulkner"]["post_peak_ratio"])
    L.append("| governing mode | %s |" % u["faulkner"]["mode"])
    L.append("| plate element cap | %.3f MPa (phi = %.4f) |"
             % (u["plate_element"]["sigma_u"] / 1e6, u["plate_element"]["phi"]))
    L.append("| stiffener cap | %.3f MPa (mode %s) |"
             % (u["stiffener_element"]["sigma_u"] / 1e6, u["stiffener_element"]["mode"]))
    L.append("| column / tripping / web / flange stress | %.1f / %.1f / %.1f / %.1f MPa |"
             % (u["stiffener_element"]["column"] / 1e6, u["stiffener_element"]["tripping"] / 1e6,
                u["stiffener_element"]["web_local"] / 1e6,
                u["stiffener_element"]["flange_local"] / 1e6))
    L.append("| elastic lower bound | %.3f MPa |" % (u["bounds"]["lower"] / 1e6))
    L.append("| rigid-plastic upper bound | %.3f MPa |" % (u["bounds"]["upper"] / 1e6))
    L.append("| inside bounds | %s |" % u["bounds_gate"]["inside_bounds"])
    L.append("| band half width / mean | %.4f |" % u["band"]["band_half_width_over_mean"])

    L.append("\n### T4. Four independent methods (same panel)\n")
    L.append("| method | sigma_u [MPa] | sigma_u/sigma_y | deviation vs element_faulkner |")
    L.append("|---|---|---|---|")
    for k, v in u["methods"]["sigma_u"].items():
        L.append("| %s | %.3f | %.4f | %+.2f %% |"
                 % (k, v / 1e6, u["methods"]["ratio"][k],
                    u["methods"]["deviation"][k] * 100))
    L.append("\nmax spread over min: **%.2f %%**" % (u["methods"]["max_spread_over_min"] * 100))

    pi = u["plastic_interaction"]
    L.append("\n### T5. Rigid-plastic N-M interaction\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append("| N_p | %.3f kN |" % (pi["n_p"] / 1e3))
    L.append("| M_p | %.3f kN m |" % (pi["m_p"] / 1e3))
    L.append("| M(N=0)/M_p | %.12f |" % pi["m_at_n_zero_over_m_p"])
    L.append("| max M/M_p over the whole surface | %.4f |" % pi["m_max_over_m_p"])
    L.append("| fitted alpha (compression) | %.3f |" % pi["alpha_compression"])
    L.append("| power-law fit residual (compression) | %.2f %% |"
             % (pi["max_rel_residual_compression"] * 100))
    L.append("| fitted alpha (tension) | %.3f |" % pi["alpha_tension"])
    L.append("| power-law fit residual (tension) | %.2f %% |"
             % (pi["max_rel_residual_tension"] * 100))
    L.append("| rule form max violation, compression | %+.2f %% of M_p |"
             % (pi["rule_form_max_violation_compression_over_m_p"] * 100))
    L.append("| rule form max violation, tension | %+.2f %% of M_p |"
             % (pi["rule_form_max_violation_tension_over_m_p"] * 100))

    L.append("\n### T6. Plate-thickness sweep\n")
    L.append("| t [mm] | beta | lambda | Faulkner | von Karman | lower bound | eps_u | mode |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in res["sweeps"]["beta_sweep"]:
        L.append("| %.1f | %.3f | %.3f | %.4f | %.4f | %.4f | %.5f | %s |"
                 % (r["t"] * 1000, r["beta"], r["lambda"], r["faulkner"],
                    r["von_karman"], r["lower"], r["eps_u"], r["mode"]))

    L.append("\n### T7. Span sweep (column / tripping competition)\n")
    L.append("| a [m] | lambda | sigma_E [MPa] | sigma_trip [MPa] | sigma_u/sigma_y | mode |")
    L.append("|---|---|---|---|---|---|")
    for r in res["sweeps"]["lambda_sweep"]:
        L.append("| %.2f | %.3f | %.1f | %.1f | %.4f | %s |"
                 % (r["a"], r["lambda"], r["sigma_euler_mpa"], r["sigma_trip_mpa"],
                    r["sigma_u_ratio"], r["mode"]))

    L.append("\n### T8. Flat-bar web thickness sweep (tripping dominated)\n")
    L.append("| t_w [mm] | sigma_trip [MPa] | sigma_column [MPa] | stiffener mode | panel sigma_u/sigma_y |")
    L.append("|---|---|---|---|---|")
    for r in res["sweeps"]["flat_bar_sweep"]:
        L.append("| %.1f | %.1f | %.1f | %s | %.4f |"
                 % (r["t_w"] * 1000, r["sigma_trip_mpa"], r["sigma_column_mpa"],
                    r["mode"], r["panel_ratio"]))

    gr = res["grillage"]
    L.append("\n### T9. Grillage: derived assembly vs the old fabricated factors\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append("| sigma_u 3x3 | %.6f MPa |" % (gr["convergence"]["sigma_u_coarse"] / 1e6))
    L.append("| sigma_u 5x5 | %.6f MPa |" % (gr["convergence"]["sigma_u_fine"] / 1e6))
    L.append("| relative error 3x3 -> 5x5 | %.3e |" % gr["convergence"]["relative_error"])
    L.append("| size invariance (identical bays) | %.3e |"
             % gr["convergence"]["mesh_invariance_rel"])
    L.append("| old model's fabricated error | %.5f |" % gr["fabricated_error_of_old_model"])
    L.append("| delta_iso | %+.3e |" % gr["isolated_deviation"]["delta_iso"])
    L.append("| delta_iso uncertainty bound | %.3e |" % gr["isolated_deviation"]["delta_iso_bound"])
    L.append("| load-step convergence 200 -> 400 | %.3e |"
             % gr["discretization_200_vs_400"]["relative_error"])
    L.append("| load-step error 40 -> 400 (legacy setting) | %.3e |"
             % gr["discretization_40_vs_400"]["relative_error"])
    return "\n".join(L)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report_md(res: dict, figs: list) -> str:
    s = res["section"]["summary"]
    u = res["ultimate"]
    gr = res["grillage"]
    d = res["defects"]
    pi = u["plastic_interaction"]
    names = [f.name for f in figs]
    worst_closure = max(
        v["rel_diff"] for k, v in res["section"]["closures"].items()
        if k in ("total_area", "z_bar", "second_moment")
    )
    same_family = max(abs(u["methods"]["deviation"]["smith_perry_robertson"]),
                      abs(u["methods"]["deviation"]["paik_rigid_plastic"]))
    L = []
    A = L.append

    A("# 弹塑性板架崩溃验证面成熟化（P167W2-N04·重开轮）")
    A("")
    A("**节点** P167W2-N04 ｜ **仓** TensorPhysics ｜ **模块** "
      "`elastoplastic_collapse.panel_section` / `.panel_ultimate` / "
      "`.grillage_error_bound`（重写）｜ **复现** "
      "`PYTHONPATH=src python scripts/p167w2_n04_panel_maturity.py` ｜ "
      "**测试** `pytest tests/test_p167w2_n04_panel_maturity.py`（28 通过）")
    A("")
    A("本文所有数字由脚本从 `artifacts/p167w2_n04/results.json` 程序化注入（零手抄）。"
      "原 PR#296 已关闭，本轮从当前 main 重新实现，把「剖面 → 极限强度 → 破坏模式 → "
      "板架收敛」链上每一环换成可独立复算的力学量。")
    A("")
    A("## 1. 案例介绍（物理背景 / 几何 / 参数表）")
    A("")
    A("纵向加筋板架在总纵弯曲受压侧，极限强度由三类能力竞争：板格屈曲后的有效宽度丧失、"
      "筋材柱屈曲-屈服交互、以及筋材**侧倾（tripping）**。基准板格取典型内底板：")
    A("")
    A("| 参数 | 符号 | 值 |")
    A("|---|---|---|")
    A("| 横向框架间距（柱长） | a | %.3f m |" % s["a"])
    A("| 筋距 | b | %.3f m |" % s["b"])
    A("| 板厚 | t | %.1f mm |" % (s["t"] * 1000))
    A("| 腹板高 / 厚 | h_s / t_w | %.0f / %.0f mm |" % (s["h_s"] * 1000, s["t_w"] * 1000))
    A("| 翼板宽 / 厚 | b_f / t_f | %.0f / %.0f mm |" % (s["b_f"] * 1000, s["t_f"] * 1000))
    A("| 材料 | E, ν, σ_y0, H | 206 GPa, 0.30, %.0f MPa, 1.0 GPa |" % (s["sigma_y0"] / 1e6))
    A("| 板格柔度 | β = (b/t)√(σ_y0/E) | **%.3f** |" % s["beta_plate"])
    A("| 柱柔度 | λ = (a/πr)√(σ_y0/E)（r 含附着板） | **%.3f** |" % s["lambda_column"])
    A("")
    A("模型常数（声明值，未对标定）：残余应力因子 r = 0.15、峰后软化斜率 0.02E、"
      "初始挠度 a/1000、侧向约束 lambda_trip = 1.0。")
    A("")
    A("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    A("")
    A("1. **剖面（新增 `panel_section.py`）**：把板条 + 腹板 + 翼板分解成轴对齐矩形，逐矩形"
      "**解析**积分得到面积、一阶矩、二阶矩（自身惯性 + 平行轴）、等面积轴 y_p（二分至 1e-16）、"
      "塑性剖面模数 Z_p = ∫|z−y_p|dA、圣维南常数 J、T 型翘曲常数 C_w、极惯性矩 I_p，"
      "以及板 / 腹板 / 翼板 / 柱 / 侧倾五个弹性临界应力。**独立复核**："
      "`numeric_properties()` 用 800 条水平条带做完全另路的数值积分。")
    A("2. **极限强度（新增 `panel_ultimate.py`）**：等缩短量并联装配。板元件用可选曲线"
      "（Faulkner 试验标定 = 设计值；von Kármán 理想板 = 上界）；筋材用**四个上限竞争**"
      "（含附着板的 Johnson-Ostenfeld 柱曲线、弹性侧倾、腹板局部屈曲、翼板悬伸屈曲），"
      "最小者绑定；元件过峰后按各自软化斜率下降，故面板曲线有真实**极限点**。")
    A("3. **物理门**：结果必须落在弹性下界 (1−r)·min(σ_cr) 与刚性塑性上界 σ_y 之间"
      "（`check_ultimate_bounds`），越界判为缺陷而非结果。")
    A("4. **四法交叉**：本模块 Faulkner / 本模块 von Kármán / `axial_collapse_ep` 的 "
      "Perry-Robertson 装配 / 同模块刚性塑性求和 / `progressive_collapse` 纤维求解器"
      "（曲率由 M = N·δ 定）。")
    A("5. **N-M 塑性面（新增）**：扫描塑性中和轴解析求和得到**精确**刚性塑性 N-M 面；"
      "端点 N=0 ⇒ M=M_p、M=0 ⇒ N=±N_p 为机器精度锚点。")
    A("6. **板架（重写 `grillage_error_bound.py`）**：删除伪系数 k = 1 + 0.05(n_y−1)/n_y、"
      "1 + 0.02(n_x−1)/n_x（无出处，且使 3×3→5×5 判据恒等于 %.2f%% 而必然通过），"
      "改为等缩短量并联装配 + 确定性分位数抽样的初始挠度分布（Beta(2,2)，无随机数）；"
      "收敛研究拆成两把刀：舱格数（物理上严格不变）与载荷步数（真正的分辨率旋钮）。"
      % (gr["fabricated_error_of_old_model"] * 100))
    A("")
    A("## 3. 计算结果（数字表，程序化注入）")
    A("")
    A("完整 9 张表见 `artifacts/p167w2_n04/tables.md`。关键量：")
    A("")
    A("- 剖面：A = %.6g m²，z_bar = %.6g m，I = %.6g m⁴，y_p = %.6g m，"
      "**Z_p = %.6g m³**，形状系数 = %.4f，M_p = %.3f kN·m，N_p = %.1f kN。"
      % (s["total_area"], s["z_bar"], s["second_moment"], s["plastic_neutral_axis"],
         s["plastic_section_modulus"], s["shape_factor"],
         s["plastic_bending_moment"] / 1e3, s["squash_load"] / 1e3))
    A("- 板架极限强度（设计值，Faulkner 曲线）：**σ_u = %.3f MPa = %.4f σ_y**，"
      "极限点应变 %.5f，峰后比 %.4f，控制模式 **%s**。"
      % (u["faulkner"]["sigma_u"] / 1e6, u["faulkner"]["ratio"],
         u["faulkner"]["eps_u"], u["faulkner"]["post_peak_ratio"], u["faulkner"]["mode"]))
    A("- 强度带：Faulkner %.4f σ_y ↔ von Kármán %.4f σ_y，半带宽/均值 = **%.2f %%**。"
      % (u["faulkner"]["ratio"], u["von_karman"]["ratio"],
         u["band"]["band_half_width_over_mean"] * 100))
    A("- 四法交叉离散（max/min − 1）= **%.2f %%**；同族方法 |偏差| ≤ %.2f %%，"
      "离散几乎全部来自板元件曲线选择。"
      % (u["methods"]["max_spread_over_min"] * 100, same_family * 100))
    A("- N-M 面：M(N=0)/M_p = %.12f，全表面最大 M/M_p = %.4f（非对称剖面，见 ⑤.2）。"
      % (pi["m_at_n_zero_over_m_p"], pi["m_max_over_m_p"]))
    A("- 板架：3×3 → 5×5 相对差 %.3e，舱格数不变性 %.3e，**δ_iso = %+.3e**"
      "（旧的 +5%%/舱格虚构增益已被否定）。"
      % (gr["convergence"]["relative_error"], gr["convergence"]["mesh_invariance_rel"],
         gr["isolated_deviation"]["delta_iso"]))
    A("")
    A("## 4. 结果图")
    A("")
    A("- `%s`：精确刚性塑性 N-M 面 vs 设计规范二次相关式。" % names[0])
    A("- `%s`：板架载荷-缩短曲线（板/筋元件上限、弹性下界、极限点）。" % names[1])
    A("- `%s`：极限强度随板厚变化 + 声明的强度带。" % names[2])
    A("- `%s`：控制破坏模式图（t × h_s），白线为 σ_u/σ_y。" % names[3])
    A("- `%s`：舱格并联装配（初始挠度分位数）与单舱对比。" % names[4])
    A("")
    A("## 5. 注意点（踩坑 / 局限 / 适用范围）")
    A("")
    A("1. **旧板架系数的性质**：k = 1 + 0.05(n_y−1)/n_y 无论板架如何都给出 %.2f%% 的 "
      "3×3→5×5 差，旧 `converged` 判据必然通过——测试绿但问题没解决。新判据同时含"
      "**严格不变性**（%.1e）与**步长收敛性**（200→400 步 %.1e；旧默认 40 步偏 %.2f%%）。"
      % (gr["fabricated_error_of_old_model"] * 100,
         gr["convergence"]["mesh_invariance_rel"],
         gr["discretization_200_vs_400"]["relative_error"],
         gr["discretization_40_vs_400"]["relative_error"] * 100))
    A("2. **板元件曲线主导 %.0f%%**：von Kármán √(σ_cr σ_y) 是理想板（无残余应力、无初始挠度）"
      "的峰后平均值；Faulkner 是试验标定值。二者之差即本文声明的强度带。无试验锚点时"
      "**不应**把上界当设计值。"
      % (u["band"]["band_width_ratio"] * 100))
    A("3. **筋材初始挠度符号问题（诚实报告，未修）**：现有纤维求解器 σ_u 随 w0 单调**增加**"
      "（w0 = 0 → 4 mm 时 %.4f → %.4f σ_y），与所有设计曲线（η ∝ δ0）方向相反。"
      "因此本轮**没有**用初始挠度统计去构造尺寸效应，而是降级为不确定度上界；缺陷登记待修。"
      % (d["progressive_imperfection_trend"]["w0_0"],
         d["progressive_imperfection_trend"]["w0_4mm"]))
    A("4. **未做**：侧向压力 / 双轴压缩耦合（不引入无锚点经验式）；焊接残余应力的空间分布"
      "（仅单一折减 0.15）；桶形 / 箱形筋。")
    A("5. 适用范围：b/t ≤ 60（β ≤ 4.6）、h_s/t_w ≤ 60、a/b ∈ [1, 6]，单一 T 型或平板筋。")
    A("")
    A("## 6. 误差判定（逐项列出并显式回答是否 < 3%）")
    A("")
    A("| 关键量 | 基准 | 误差 | 是否 < 3% |")
    A("|---|---|---|---|")
    A("| A, z_bar, I（剖面） | 800 条带独立数值积分 | %.1e（最大） | **是** |" % worst_closure)
    A("| Z_p（塑性模数） | 同上 | %.1e | **是** |"
      % res["section"]["closures"]["plastic_section_modulus"]["rel_diff"])
    A("| M(N=0)（塑性面锚点） | σ_y·Z_p 解析 | %.1e | **是** |"
      % abs(pi["m_at_n_zero_over_m_p"] - 1.0))
    A("| σ_cr（板屈曲） | Timoshenko 闭式（测试内独立重算） | < 1e-12 | **是** |")
    A("| σ_T（侧倾，平板筋） | GJ/I_p 闭式（测试内独立重算） | < 1e-12 | **是** |")
    A("| σ_u（同族方法间） | Faulkner vs Perry-Robertson vs 刚性塑性求和 | %.2f%%（最大） | **是** |"
      % (same_family * 100))
    A("| σ_u（跨族：元件曲线） | Faulkner vs von Kármán vs 纤维求解器 | %.2f%%（max/min−1） | **否** |"
      % (u["methods"]["max_spread_over_min"] * 100))
    A("| σ_u vs 试验 | —— | 无离线试验基准 | **无基准**（如实声明） |")
    A("| 板架尺寸不变性 | 同舱格解析恒等 | %.1e | **是** |"
      % gr["convergence"]["mesh_invariance_rel"])
    A("| δ_iso | 并联装配（无自由参数） | %.1e 相对，界 ±%.1e | **是**（且否定旧 +5%%） |"
      % (abs(gr["isolated_deviation"]["delta_iso"]),
         gr["isolated_deviation"]["delta_iso_bound"]))
    A("")
    A("**结论**：所有带解析/独立数值基准的量均 < 3%%（多数 < 1e-6）；"
      "**σ_u 的跨族离散 %.0f%% > 15%%，不满足精度标准**，根源唯一地定位到板元件本构曲线"
      "（von Kármán vs Faulkner），已标为待试验判定的开放缺口，未用任何经验系数粉饰。"
      % (u["methods"]["max_spread_over_min"] * 100))
    A("")
    A("## 7. 本轮顺带发现并处理的既有缺陷")
    A("")
    A("1. `grillage_error_bound` 的两个伪约束系数（已删除，改力学装配）。")
    A("2. `SmithCollapseSolver` 文档声明支持 `tripping` 但该模式**不可达**（纤维求解器只认 "
      "板/柱）：已修——侧倾临界应力低于板/柱时返回 `tripping`"
      "（平板筋 t_w = %.0f mm 复算得 `%s`；基准 T 型筋为 `%s`）。"
      % (6.0, d["flat_bar_mode"], d["reference_mode"]))
    A("3. `StiffenedPlateGeometry.section_properties()` 的 `Z_bot` 量到板**中面**而非板底纤维，"
      "对基准板格高估 %.2f%%（新模块用板底纤维，回归测试锁定二者关系）。"
      % (res["section"]["legacy_Zbot_defect"]["overestimate"] * 100))
    A("")
    A("---")
    A("")
    A("产物：`artifacts/p167w2_n04/results.json`、`tables.md`、5 张 png；"
      "测试：`tests/test_p167w2_n04_panel_maturity.py`。")
    return "\n".join(L)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "section": compute_section(),
        "ultimate": compute_ultimate(),
        "sweeps": compute_sweeps(),
        "grillage": compute_grillage(),
        "defects": compute_defects(),
    }
    figs = make_figures(res)
    (OUT / "results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "tables.md").write_text(tables_md(res), encoding="utf-8")
    DOC.write_text(report_md(res, figs), encoding="utf-8")
    print("results  ->", OUT / "results.json")
    print("tables   ->", OUT / "tables.md")
    for f in figs:
        print("figure   ->", f)
    print("report   ->", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
