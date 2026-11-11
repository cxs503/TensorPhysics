"""Run the Smith cross-scale verification chain and emit evidence + figures.

Usage
-----
    PYTHONPATH=src python benchmarks/smith_chain/run_smith_chain.py

Writes
------
``artifacts/smith_chain/smith_chain_results.json``   every number in the report
``artifacts/smith_chain/fig_*.png``                  report figures

The script is deterministic (no randomness) and re-runs the whole chain in
about a minute, so the JSON is the single source of truth for the report - no
number in the report is typed by hand.
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.smith_chain import (  # noqa: E402
    AH32,
    MILD_STEEL,
    SmithSolver,
    analyse_section,
    box_girder,
    csr_min_of_modes_panel_strength,
    exact_box_plastic_modulus,
    grid_convergence_study,
    issc2009_double_bottom,
    paik_panel,
    plate_slenderness,
)
from tensorphysics.smith_chain.benchmarks import effective_width_ultimate_moment  # noqa: E402
from tensorphysics.smith_chain.laws import (  # noqa: E402
    column_strength,
    faulkner_plate_strength_ratio,
)

OUT = ROOT / "artifacts" / "smith_chain"
OUT.mkdir(parents=True, exist_ok=True)

#: pre-declared expectations for the failure-mode gate (declared before the run,
#: derived from element slenderness: beta>1 -> plate buckling, beta<1 -> yield,
#: stiffener capacity below plate strength -> stiffener-governed)
EXPECTED_MODES = {
    "panel_beta2.00": "plate_buckling",
    "box_girder_compact": "yielding",
    "double_bottom_hogging": "plate_buckling",
}

CURVES: dict[str, dict] = {}

#: curvature-peak refinement used in the grid ladders.  The golden-section
#: refinement is switched on for the convergence study so that the differences
#: between levels measure the *element subdivision* error only and not the
#: jitter of the discrete curvature sampling (verified: with the refinement the
#: four ladders are strictly monotone, without it the slender-box and
#: double-bottom ladders reverse by O(1e-5) relative).
PEAK_REFINE = 60


def log(msg: str) -> None:
    print(f"[smith-chain] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Rung 1: stiffened panel beta-series (axial compression)
# ---------------------------------------------------------------------------
def run_panel_rung(betas=(1.5, 2.0, 2.5, 3.0), n_sub=4, n_steps=400):
    rows = []
    for beta in betas:
        sec = paik_panel(beta)
        analysis = analyse_section(sec, n_sub=n_sub, n_steps=n_steps)
        d = analysis.as_dict()
        s = sec.stiffeners[0]
        plate_t = sec.plates[0].thickness
        csr = csr_min_of_modes_panel_strength(
            beta=beta, h=s.h, t_w=s.t_w, span=s.span_a, spacing_b=s.spacing_b,
            plate_t=plate_t, material=sec.material,
        )
        smith = d["sigma_u_over_sigma_y"]
        phi = d["reference"]["faulkner_phi"]
        row = {
            "beta": beta,
            "plate_thickness_mm": plate_t * 1e3,
            "sigma_u_MPa": d["sigma_u_MPa"],
            "sigma_u_over_sigma_y": smith,
            "faulkner_phi": phi,
            "rel_diff_vs_faulkner": abs(smith - phi) / phi,
            "area_weighted_effective_width": d["reference"]["area_weighted_effective_width_ratio"],
            "rel_diff_vs_area_weighted": abs(smith - d["reference"]["area_weighted_effective_width_ratio"])
            / d["reference"]["area_weighted_effective_width_ratio"],
            "csr_min_of_modes": csr["sigma_u_over_sigma_y"],
            "csr_governing_mode": csr["governing_mode"],
            "csr_modes": {"plate": csr["plate_ratio"], "column": csr["column_ratio"],
                          "tripping": csr["tripping_ratio"]},
            "lambda_column": csr["lambda_column"],
            "rel_diff_vs_csr": abs(smith - csr["sigma_u_over_sigma_y"]) / csr["sigma_u_over_sigma_y"],
            "paik_thayamballi_reference_only": d["reference"]["paik_thayamballi_ratio_reference_only"],
            "failure_mode": d["failure_mode"]["dominant"],
            "mode_shares": d["failure_mode"]["shares"],
            "load_shortening": d["load_shortening"],
        }
        rows.append(row)
        CURVES[f"panel_beta{beta:.2f}"] = d["load_shortening"]
        log(f"panel beta={beta:.1f}: sigma_u/sigma_y={smith:.4f} phi={phi:.4f} csr={row['csr_min_of_modes']:.4f}")
    gate_beta = 2.0
    gate = next(r for r in rows if abs(r["beta"] - gate_beta) < 1e-9)
    gate["in_band_0.4_0.8"] = bool(0.4 <= gate["sigma_u_over_sigma_y"] <= 0.8)
    gate["within_10pct_of_csr"] = bool(gate["rel_diff_vs_csr"] <= 0.10)
    gate["within_10pct_of_faulkner"] = bool(gate["rel_diff_vs_faulkner"] <= 0.10)
    return {"rows": rows, "gate_case_beta2": gate}


# ---------------------------------------------------------------------------
# Rung 2: unstiffened box girder
# ---------------------------------------------------------------------------
def run_box_rung(n_sub_levels=(2, 4, 8), n_web_strips=8, n_steps=400):
    out = {}
    for variant in ("compact", "slender"):
        B, H, t = ((0.50, 0.40, 0.020) if variant == "compact" else (1.20, 0.90, 0.012))
        Z_exact = exact_box_plastic_modulus(B, H, t)
        M_exact = MILD_STEEL.sigma_y0 * Z_exact
        analysis = analyse_section(box_girder(variant, n_web_strips=n_web_strips),
                                   n_sub=8, n_steps=n_steps)
        d = analysis.as_dict()
        study = grid_convergence_study(
            lambda lvl: SmithSolver(box_girder(variant, n_web_strips=n_web_strips),
                                    n_sub=int(lvl), n_steps=n_steps,
                                    peak_refine_iters=PEAK_REFINE).solve().M_u,
            levels=n_sub_levels,
        )
        ew = effective_width_ultimate_moment(box_girder(variant, n_web_strips=n_web_strips))
        out[variant] = {
            "B_m": B, "H_m": H, "t_mm": t * 1e3,
            "beta_flange": plate_slenderness(B, t, MILD_STEEL),
            "beta_web": plate_slenderness(H - 2 * t, t, MILD_STEEL),
            "Z_plastic_exact_m3": Z_exact,
            "M_plastic_exact_MNm": M_exact / 1e6,
            "M_u_MNm": d["M_u_MNm"],
            "M_u_over_M_plastic_exact": d["M_u_MNm"] * 1e6 / M_exact,
            "M_first_yield_MNm": d["section"]["M_first_yield_MNm"],
            "M_u_over_M_yield_compression_side": d["diagnostics"]["M_u_over_M_yield_compression_side"],
            "M_EW_MNm": ew["M_u_MNm"],
            "M_u_over_M_EW": d["M_u_MNm"] / ew["M_u_MNm"],
            "z_NA_m": d["z_NA_m"],
            "z_NA_elastic_m": d["section"]["z_neutral_elastic_m"],
            "failure_mode": d["failure_mode"]["dominant"],
            "mode_shares": d["failure_mode"]["shares"],
            "kind_detected": d["section"]["kind"],
            "moment_curvature": d["moment_curvature"],
            "grid_study": study,
        }
        CURVES[f"box_{variant}"] = d["moment_curvature"]
        log(f"box {variant}: M_u={d['M_u_MNm']:.4f} MNm, M_u/M_p,exact="
            f"{d['M_u_MNm']*1e6/M_exact:.5f}, M_u/M_EW={d['M_u_MNm']/ew['M_u_MNm']:.4f}, "
            f"p={study['richardson']['observed_order']:.2f}, "
            f"resid={study['richardson']['residual_finest_vs_extrapolated']*100:.4f}%")
    return out


# ---------------------------------------------------------------------------
# Rung 3: double-bottom hull girder
# ---------------------------------------------------------------------------
def run_double_bottom_rung(n_sub_levels=(2, 4, 8), n_web_strips=8, n_steps=400):
    out = {}
    for bending in ("sagging", "hogging"):
        sec = issc2009_double_bottom(n_web_strips=n_web_strips)
        analysis = analyse_section(sec, n_sub=8, n_steps=n_steps, bending=bending)
        d = analysis.as_dict()
        study = grid_convergence_study(
            lambda lvl: SmithSolver(issc2009_double_bottom(n_web_strips=n_web_strips),
                                    n_sub=int(lvl), n_steps=n_steps, bending=bending,
                                    peak_refine_iters=PEAK_REFINE).solve().M_u,
            levels=n_sub_levels,
        )
        ew = effective_width_ultimate_moment(sec, bending=bending)
        web_ladder = []
        for nw in (2, 4, 8):
            r = SmithSolver(issc2009_double_bottom(n_web_strips=nw), n_sub=8,
                            n_steps=n_steps, bending=bending,
                            peak_refine_iters=PEAK_REFINE).solve()
            web_ladder.append({"n_web_strips": nw, "M_u_MNm": r.M_u / 1e6})
        out[bending] = {
            "M_u_MNm": d["M_u_MNm"],
            "M_EW_MNm": ew["M_u_MNm"],
            "M_u_over_M_EW": d["M_u_MNm"] / ew["M_u_MNm"],
            "M_first_yield_MNm": d["section"]["M_first_yield_MNm"],
            "M_first_yield_compression_side_MNm": d["diagnostics"]["M_first_yield_compression_side_MNm"],
            "M_plastic_MNm": d["section"]["M_plastic_MNm"],
            "M_u_over_M_yield_compression_side": d["diagnostics"]["M_u_over_M_yield_compression_side"],
            "M_u_over_M_plastic": d["diagnostics"]["M_u_over_M_plastic"],
            "z_NA_m": d["z_NA_m"],
            "z_NA_elastic_m": d["section"]["z_neutral_elastic_m"],
            "kappa_u": d["kappa_u"],
            "failure_mode": d["failure_mode"]["dominant"],
            "secondary_mode": d["failure_mode"]["secondary"],
            "mode_shares": d["failure_mode"]["shares"],
            "top_elements": d["failure_mode"]["top_elements"],
            "kind_detected": d["section"]["kind"],
            "moment_curvature": d["moment_curvature"],
            "grid_study": study,
            "web_strip_ladder": web_ladder,
            "elements": d["elements"],
        }
        CURVES[f"db_{bending}"] = d["moment_curvature"]
        log(f"double bottom {bending}: M_u={d['M_u_MNm']:.3f} MNm "
            f"M_u/M_EW={d['M_u_MNm']/ew['M_u_MNm']:.4f} "
            f"p={study['richardson']['observed_order']:.2f} "
            f"resid={study['richardson']['residual_finest_vs_extrapolated']*100:.4f}% "
            f"mode={d['failure_mode']['dominant']}")
    out["section"] = issc2009_double_bottom(n_web_strips=n_web_strips).describe()
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_panel(panel):
    rows = panel["rows"]
    beta = [r["beta"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(beta, [r["sigma_u_over_sigma_y"] for r in rows], "o-", lw=2,
            label="Smith (iso-strain panel analysis)")
    ax.plot(beta, [r["faulkner_phi"] for r in rows], "s--",
            label=r"Faulkner $\phi(\beta)=2/\beta-1/\beta^2$")
    ax.plot(beta, [r["csr_min_of_modes"] for r in rows], "^:",
            label="CSR-style min-of-modes")
    ax.plot(beta, [r["paik_thayamballi_reference_only"] for r in rows], "v-.",
            alpha=0.6, label="Paik & Thayamballi (2003) fit (reference only)")
    ax.set_xlabel(r"plate slenderness $\beta=(b/t)\sqrt{\sigma_y/E}$")
    ax.set_ylabel(r"$\sigma_u/\sigma_y$")
    ax.set_title("Rung 1: stiffened panel, FB 300x20 @ 800 mm, a = 4.0 m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rung1_panel_strength.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in rows:
        eps = [p[0] for p in r["load_shortening"]]
        sig = [p[1] for p in r["load_shortening"]]
        ax.plot(np.array(eps) * 1e3, sig, label=fr"$\beta$={r['beta']:.1f}")
    ax.set_xlabel("average shortening [1e-3]")
    ax.set_ylabel(r"average stress $\sigma_{avg}$ [MPa]")
    ax.set_title("Rung 1: panel load-shortening curves (axial Smith)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rung1_panel_load_shortening.png", dpi=150)
    plt.close(fig)


def fig_box(boxes):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, (variant, b) in zip(axes, boxes.items()):
        k = np.array([p[0] for p in b["moment_curvature"]])
        m = np.array([p[1] for p in b["moment_curvature"]])
        ax.plot(k * 1e3, m, lw=2, label="Smith $M(\\kappa)$")
        ax.axhline(b["M_plastic_exact_MNm"], color="k", ls="--", lw=1,
                   label=f"exact $M_p$ = {b['M_plastic_exact_MNm']:.3f} MNm")
        ax.axhline(b["M_EW_MNm"], color="tab:red", ls=":", lw=1.4,
                   label=f"effective width = {b['M_EW_MNm']:.3f} MNm")
        ax.axhline(b["M_first_yield_MNm"], color="tab:green", ls="-.", lw=1,
                   label=f"first yield = {b['M_first_yield_MNm']:.3f} MNm")
        ax.set_xlabel(r"curvature $\kappa$ [1e-3 m$^{-1}$]")
        ax.set_ylabel("moment [MN m]")
        ax.set_title(f"Rung 2 box girder ({variant}, $\\beta_f$={b['beta_flange']:.2f})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rung2_box_moment_curvature.png", dpi=150)
    plt.close(fig)


def fig_double_bottom(db):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for bending, style in (("sagging", "-"), ("hogging", "--")):
        b = db[bending]
        k = np.array([p[0] for p in b["moment_curvature"]])
        m = np.array([p[1] for p in b["moment_curvature"]])
        ax.plot(k * 1e3, m, style, lw=2, label=f"Smith {bending}")
        ax.plot(abs(b["kappa_u"]) * 1e3, b["M_u_MNm"], "o", color="k", ms=5)
        ax.axhline(b["M_EW_MNm"], ls=":", lw=1.2, alpha=0.8,
                   label=f"effective width ({bending})")
    ax.axhline(db["sagging"]["M_plastic_MNm"], color="grey", ls="-.", lw=1,
               label="idealisation $M_p$")
    ax.set_xlabel(r"curvature $|\kappa|$ [1e-3 m$^{-1}$]")
    ax.set_ylabel("moment [MN m]")
    ax.set_title("Rung 3: ISSC2009-type double bottom, moment-curvature")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rung3_db_moment_curvature.png", dpi=150)
    plt.close(fig)

    # element stress profile at ultimate (hogging)
    b = db["hogging"]
    els = b["elements"]
    z = np.array([e["z_m"] for e in els.values()])
    s = np.array([e["stress_MPa"] for e in els.values()])
    cap = np.array([e["capacity_MPa"] for e in els.values()])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(s, z, "o", label=r"$\sigma$ at $M_u$")
    ax.plot(cap, z, "x", alpha=0.7, label=r"$\sigma_u$ capacity")
    ax.plot(-s, z, "o", mfc="none", alpha=0.5)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axhline(b["z_NA_m"], color="tab:red", ls="--", lw=1.2,
               label=f"$z_{{NA}}$={b['z_NA_m']:.2f} m")
    ax.set_xlabel("axial stress [MPa] (compression > 0)")
    ax.set_ylabel("z [m]")
    ax.set_title("Rung 3: element stresses at ultimate (hogging)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rung3_db_stress_profile.png", dpi=150)
    plt.close(fig)


def fig_convergence(boxes, db):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    series = []
    for name, data in (("box compact", boxes["compact"]), ("box slender", boxes["slender"]),
                       ("double bottom sagging", db["sagging"]),
                       ("double bottom hogging", db["hogging"])):
        st = data["grid_study"]
        vals = np.array(st["values"])
        ext = st["extrapolated"]
        h = 1.0 / np.array(st["levels"], dtype=float)
        err = np.abs(vals - ext) / abs(ext)
        ax.loglog(h, np.maximum(err, 1e-12), "o-", label=f"{name} (p={st['richardson']['observed_order']:.2f})")
        series.append({"name": name, "h": h.tolist(), "rel_err": err.tolist(),
                       "order": st["richardson"]["observed_order"]})
    ax.set_xlabel("discretisation size $h \\propto 1/n_{sub}$")
    ax.set_ylabel("$|M_u(h)-M_{u,ext}|/M_{u,ext}$")
    ax.set_title("Grid convergence (Richardson) of the Smith chain")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergence.png", dpi=150)
    plt.close(fig)
    return series


def fig_modes(db, boxes, panel):
    labels, shares = [], {}
    for name, data in (("panel beta=2.0", None), ("box compact", boxes["compact"]),
                       ("box slender", boxes["slender"]), ("DB sagging", db["sagging"]),
                       ("DB hogging", db["hogging"])):
        if data is None:
            sh = panel["gate_case_beta2"]["mode_shares"]
        else:
            sh = data["mode_shares"]
        labels.append(name)
        shares[name] = sh
    modes = sorted({m for sh in shares.values() for m in sh})
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    bottom = np.zeros(len(labels))
    for m in modes:
        vals = np.array([shares[l].get(m, 0.0) for l in labels])
        ax.bar(labels, vals, bottom=bottom, label=m)
        bottom += vals
    ax.set_ylabel("share of capacity loss at $M_u$")
    ax.set_title("Failure-mode attribution (compression side, at ultimate)")
    ax.legend(fontsize=7.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_failure_modes.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    log("rung 1: stiffened panel beta-series")
    panel = run_panel_rung()
    log("rung 2: unstiffened box girder (compact / slender)")
    boxes = run_box_rung()
    log("rung 3: ISSC2009-type double bottom")
    db = run_double_bottom_rung()

    # C5: failure-mode gate, three profiles, >= 2 must match the pre-declared mode
    mode_checks = {
        "panel_beta2.00": (panel["gate_case_beta2"]["failure_mode"], EXPECTED_MODES["panel_beta2.00"]),
        "box_girder_compact": (boxes["compact"]["failure_mode"], EXPECTED_MODES["box_girder_compact"]),
        "double_bottom_hogging": (db["hogging"]["failure_mode"], EXPECTED_MODES["double_bottom_hogging"]),
    }
    mode_gate = {k: {"identified": v[0], "expected": v[1], "correct": v[0] == v[1]}
                 for k, v in mode_checks.items()}
    n_correct = sum(v["correct"] for v in mode_gate.values())

    # C6: unified interface over the four section archetypes
    kinds = {
        "paik_panel_beta2.0": paik_panel(2.0).kind,
        "box_girder_compact": box_girder("compact").kind,
        "box_girder_slender": box_girder("slender").kind,
        "issc2009_double_bottom": issc2009_double_bottom().kind,
    }
    c6 = {
        "section_kinds_detected": kinds,
        "distinct_kinds": sorted(set(kinds.values())),
        "loadings_auto_selected": {
            "panel": analyse_section(paik_panel(2.0)).loading,
            "box": analyse_section(box_girder("compact")).loading,
            "double_bottom": analyse_section(issc2009_double_bottom()).loading,
        },
    }

    criteria = {
        "C1_grid_convergence_residual_le_1pct": {
            "double_bottom_sagging_residual": db["sagging"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"],
            "double_bottom_hogging_residual": db["hogging"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"],
            "box_slender_residual": boxes["slender"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"],
            "box_compact_residual": boxes["compact"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"],
            "pass": bool(
                max(db["sagging"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"],
                    db["hogging"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"],
                    boxes["slender"]["grid_study"]["richardson"]["residual_finest_vs_extrapolated"]) <= 0.01),
        },
        "C2_convergence_monotone": {
            "double_bottom_sagging": db["sagging"]["grid_study"]["monotone"],
            "double_bottom_hogging": db["hogging"]["grid_study"]["monotone"],
            "box_slender": boxes["slender"]["grid_study"]["monotone"],
            "box_compact": boxes["compact"]["grid_study"]["monotone"],
            "values_db_sagging": db["sagging"]["grid_study"]["values"],
            "values_db_hogging": db["hogging"]["grid_study"]["values"],
            "pass": bool(db["sagging"]["grid_study"]["monotone"]
                         and db["hogging"]["grid_study"]["monotone"]
                         and boxes["slender"]["grid_study"]["monotone"]),
        },
        "C3_panel_magnitude_and_csr": {
            "beta": panel["gate_case_beta2"]["beta"],
            "sigma_u_over_sigma_y": panel["gate_case_beta2"]["sigma_u_over_sigma_y"],
            "in_band_0.4_0.8": panel["gate_case_beta2"]["in_band_0.4_0.8"],
            "csr_min_of_modes": panel["gate_case_beta2"]["csr_min_of_modes"],
            "rel_diff_vs_csr": panel["gate_case_beta2"]["rel_diff_vs_csr"],
            "within_10pct_of_csr": panel["gate_case_beta2"]["within_10pct_of_csr"],
            "pass": bool(panel["gate_case_beta2"]["in_band_0.4_0.8"]
                         and panel["gate_case_beta2"]["within_10pct_of_csr"]),
        },
        "C4_box_girder_magnitude": {
            "compact_ratio_vs_exact_plastic": boxes["compact"]["M_u_over_M_plastic_exact"],
            "slender_ratio_vs_effective_width": boxes["slender"]["M_u_over_M_EW"],
            "slender_Mu_over_Mp": boxes["slender"]["M_u_over_M_plastic_exact"],
            "experimental_anchor": "WITHHELD - Dow (1991) test ultimate moments not shipped offline",
            "pass": bool(0.85 <= boxes["compact"]["M_u_over_M_plastic_exact"] <= 1.15
                         and 0.85 <= boxes["slender"]["M_u_over_M_EW"] <= 1.15),
        },
        "C5_failure_mode": {
            "checks": mode_gate,
            "n_correct_of_3": n_correct,
            "pass": bool(n_correct >= 2),
        },
        "C6_unified_interface": {
            **c6,
            "pass": bool(len(c6["distinct_kinds"]) >= 3
                         and c6["loadings_auto_selected"]["panel"] == "axial"
                         and c6["loadings_auto_selected"]["box"] == "bending"
                         and c6["loadings_auto_selected"]["double_bottom"] == "bending"),
        },
    }

    fig_panel(panel)
    fig_box(boxes)
    fig_double_bottom(db)
    conv_series = fig_convergence(boxes, db)
    fig_modes(db, boxes, panel)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node": "P9263-N03",
        "module": "tensorphysics.smith_chain",
        "script": str(pathlib.Path(__file__).relative_to(ROOT)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "host": platform.node(),
        },
        "runtime_s": None,
        "materials": {
            "AH32": {"E_GPa": AH32.E / 1e9, "nu": AH32.nu, "sigma_y_MPa": AH32.sigma_y0 / 1e6,
                     "H_MPa": AH32.H / 1e6},
            "MILD_STEEL": {"E_GPa": MILD_STEEL.E / 1e9, "nu": MILD_STEEL.nu,
                           "sigma_y_MPa": MILD_STEEL.sigma_y0 / 1e6,
                           "H_MPa": MILD_STEEL.H / 1e6},
        },
        "rung1_panel": panel,
        "rung2_box": {k: v for k, v in boxes.items()},
        "rung3_double_bottom": db,
        "convergence_series": conv_series,
        "criteria": criteria,
        "curves": CURVES,
        "withheld": [
            "ISSC 2009 Committee III.1 double-bottom benchmark coordinates/M_U: not shipped offline "
            "- geometry reconstructed from the structural class, all comparisons made against "
            "self-computed analytic/closed-form references.",
            "Dow (1991) box-girder experimental ultimate moments: not shipped offline - the "
            "experimental ratio band of C4 is answered with the exact analytic plastic limit load "
            "(compact box) and the effective-width closed form (slender box).",
            "Paik & Thayamballi (2003) panel strength coefficients: transcribed from memory, not "
            "re-verifiable offline - reported as reference only, excluded from acceptance gates.",
        ],
    }
    payload["grid_protocol"] = {
        "primary_refinement_parameter": "n_sub (fibres through the thickness of every element), "
                                        "web idealisation frozen at 8 strips",
        "levels": list(db["sagging"]["grid_study"]["levels"]),
        "refinement_ratio": db["sagging"]["grid_study"]["richardson"]["refinement_ratio"],
        "curvature_peak_refinement_iters": PEAK_REFINE,
        "secondary": "web-strip ladder 2/4/8 at n_sub = 8 (sensitivity, not the primary study)",
        "web_ladder_monotone": {
            b: all((db[b]["web_strip_ladder"][i + 1]["M_u_MNm"] - db[b]["web_strip_ladder"][i]["M_u_MNm"]) *
                   (db[b]["web_strip_ladder"][2]["M_u_MNm"] - db[b]["web_strip_ladder"][1]["M_u_MNm"]) > 0
                   for i in (0,))
            for b in ("sagging", "hogging")
        },
    }
    payload["runtime_s"] = time.time() - t0
    out_path = OUT / "smith_chain_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log(f"wrote {out_path.relative_to(ROOT)} ({out_path.stat().st_size/1024:.1f} kB) in {payload['runtime_s']:.1f} s")
    for k, v in criteria.items():
        log(f"{k}: pass={v['pass']}")
    return payload


if __name__ == "__main__":
    main()