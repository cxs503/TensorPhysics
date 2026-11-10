#!/usr/bin/env python
"""Campaign node p351w1_n08: RIF matrix, redundancy ranking, acceptance criteria.

Runs the full component-level redundancy study of the ice-class hull girder
sections in :mod:`tensorphysics.hull_redundancy` and writes

- ``artifacts/p351w1_n08/rif_matrix_<section>_<bending>.json``  full RIF matrix
- ``artifacts/p351w1_n08/redundancy_summary.json``              ranking + sets
- ``artifacts/p351w1_n08/sensitivity.json``                     Spearman rho etc.
- ``artifacts/p351w1_n08/benchmark.json``                       closed-form xcheck
- ``artifacts/p351w1_n08/criteria.json``                        machine acceptance
- ``artifacts/p351w1_n08/figures/*.png``                        report figures

Usage::

    PYTHONPATH=src python scripts/run_p351w1_n08_rif.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.hull_redundancy import (  # noqa: E402
    DamageSpec,
    SmithSectionSolver,
    apply_damage,
    benchmark_vs_smith,
    convergence_study,
    double_hull_side_section,
    most_dangerous,
    negative_control,
    p350w1_n03_combined_section,
    redundancy_set,
    rif_matrix,
    sensitivity_analysis,
    simplified_ultimate_moment,
    worst_case_by_component,
)

OUT = ROOT / "artifacts" / "p351w1_n08"
FIG = OUT / "figures"

SOLVER = {"n_steps": 80, "n_web_fibers": 2, "kappa_max": 6.0e-4}
SECONDARY_EXCLUDE = ("PC1", "PC2")   # closing plating of the side assembly


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def dump(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default,
                               ensure_ascii=False))


def public(matrix: dict) -> dict:
    return {k: v for k, v in matrix.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------
def run_section(section, tag: str, exclude=()) -> dict:
    out = {"section": tag, "directions": {}, "matrices": {}}
    for bending in ("sagging", "hogging"):
        m = rif_matrix(section, solver_kwargs=dict(SOLVER), bending=bending)
        dump(OUT / f"rif_matrix_{tag}_{bending}.json", public(m))
        sens = sensitivity_analysis(m, reference="ultimate")
        if exclude:
            entries = [e for e in m["entries"] if e["component_id"] not in exclude]
            tmp = dict(m)
            tmp["entries"] = entries
            sens_excl = sensitivity_analysis(tmp, reference="ultimate")
            sens["excluding_closing_plating"] = {
                "excluded": list(exclude),
                "pooled": sens_excl["pooled"],
                "plating": sens_excl["plating"],
                "monotone_ok": sens_excl["monotone_ok"],
            }
        nc = negative_control(m)
        rs = redundancy_set(m)
        out["directions"][bending] = {
            "M_u_intact_MNm": m["M_u_intact_MNm"],
            "z_NA_elastic_m": m["z_NA_elastic_m"],
            "z_NA_ultimate_m": m["z_NA_ultimate_m"],
            "kappa_u_intact": m["kappa_u_intact"],
            "rif_min": m["rif_min"],
            "rif_max": m["rif_max"],
            "rif_negative_control": m["rif_negative_control"],
            "rif_in_unit_interval": m["rif_in_unit_interval"],
            "worst_case_ranking": worst_case_by_component(m),
            "top3_most_dangerous": most_dangerous(m, 3),
            "redundancy_set": rs,
            "negative_control": nc,
            "sensitivity": sens,
        }
        out["matrices"][bending] = m
    return out


def make_figures(prim: dict, sec: dict) -> list:
    """Report figures; returns the list of written paths."""
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []

    # --- Fig 1: RIF bar chart, plating strips, both directions -------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, bending in zip(axes, ("sagging", "hogging")):
        m = prim["matrices"][bending]
        rows = [e for e in m["entries"] if e["kind"] == "plate"]
        ids = [e["component_id"] for e in rows]
        rif = [e["rif"] for e in rows]
        ax.bar(range(len(ids)), rif, color="#3b6ea5")
        ax.axhline(0.80, color="crimson", ls="--", lw=1.2, label="RIF = 0.80 gate")
        ax.axhline(0.98, color="seagreen", ls=":", lw=1.2, label="negative-control gate")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90, fontsize=8)
        ax.set_ylim(0.7, 1.02)
        ax.set_title(f"P350W1-N03 combined section - {bending}\nplate-strip tearing RIF")
        ax.set_ylabel("RIF = M_u(damaged)/M_u(intact)")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = FIG / "fig1_rif_plating.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Fig 2: stiffener failure-mode spectrum ---------------------------
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.26
    labels, series = [], {}
    for bending in ("sagging", "hogging"):
        m = prim["matrices"][bending]
        rows = [e for e in m["entries"] if e["kind"] == "stiffener"]
        comps = sorted({e["component_id"] for e in rows})
        for cid in comps:
            key = f"{cid}\n({bending[:3]})"
            labels.append(key)
            for mode in ("remove", "debond", "trip"):
                val = next(e["rif"] for e in rows
                           if e["component_id"] == cid and e["mode"] == mode)
                series.setdefault(mode, []).append(val)
    x = np.arange(len(labels))
    colors = {"remove": "#b23a48", "debond": "#e2a03f", "trip": "#4c956c"}
    for i, mode in enumerate(("remove", "debond", "trip")):
        ax.bar(x + (i - 1) * width, series[mode], width, label=mode, color=colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    ax.axhline(0.80, color="crimson", ls="--", lw=1.0)
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("RIF")
    ax.set_title("Stiffener failure-mode spectrum: removal < loss of composite action < tripping")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig2_mode_spectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Fig 3: RIF vs lever arm (sensitivity) ----------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for bending, marker in (("sagging", "o"), ("hogging", "s")):
        m = prim["matrices"][bending]
        sens = prim["directions"][bending]["sensitivity"]
        for key, colour in (("plating", "#3b6ea5"), ("stiffeners", "#e2a03f")):
            block = sens[key]
            if block["n"] < 3:
                continue
            ax.scatter(block["lever_arm_m"], block["rif"], marker=marker,
                       color=colour, alpha=0.75,
                       label=f"{key} - {bending} (rho={block['spearman_rho']:+.2f})")
    ax.axhline(0.80, color="crimson", ls="--", lw=1.0, label="RIF = 0.80")
    ax.set_xlabel("lever arm |z - z_NA,ultimate| [m]  (compression side)")
    ax.set_ylabel("RIF")
    ax.set_title("RIF sensitivity to failure position (Spearman rho <= -0.8 required)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig3_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Fig 4: moment-curvature curves -----------------------------------
    section = p350w1_n03_combined_section()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cases = [("intact", None, "k-", 2.0)]
    worst_id = prim["directions"]["sagging"]["top3_most_dangerous"][0]["component_id"]
    cases += [
        (f"{worst_id} removed (sagging)", DamageSpec(worst_id, "remove"), "r-", 1.4),
        ("SD1 trip (sagging)", DamageSpec("SD1", "trip"), "g--", 1.4),
        ("SD1 debond (sagging)", DamageSpec("SD1", "debond"), "b-.", 1.4),
    ]
    for label, spec, style, lw in cases:
        dmg = apply_damage(section, spec) if spec else apply_damage(
            section, DamageSpec("PB1", "none"))
        res = SmithSectionSolver(dmg, bending="sagging", **SOLVER).solve()
        k = np.array([p[0] for p in res.moment_curvature])
        M = np.array([p[1] for p in res.moment_curvature]) / 1e6
        ax.plot(k * 1e4, M, style, lw=lw,
                label=f"{label}: M_u = {res.M_u / 1e6:.0f} MNm")
        ax.plot(res.kappa_u * 1e4, res.M_u / 1e6, "k*", ms=9)
    ax.set_xlabel("curvature kappa [1e-4 /m]")
    ax.set_ylabel("bending moment M [MNm]")
    ax.set_title("Moment-curvature: intact vs damaged (sagging)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig4_moment_curvature.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # --- Fig 5: section idealisation --------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    for pl in section.plates:
        thick = pl.thickness * 200.0
        if abs(pl.z) < 0.05 or abs(pl.z - section.depth) < 0.05:          # bottom/deck
            ax.plot([0, 0.5 * pl.width], [pl.z, pl.z], color="#333333",
                    lw=max(1.0, thick * 0.35))
        else:                                                             # side shells
            x0 = 0.0 if pl.id.startswith("PO") else 3.0
            ax.add_patch(plt.Rectangle((x0, pl.z - 0.5 * pl.width),
                                       max(1.0, thick), pl.width,
                                       color="#9fb8d0", alpha=0.9))
    for st in section.stiffeners:
        ax.add_patch(plt.Rectangle((0.9, st.z - 0.5 * st.h), 0.25, st.h,
                                   color="#b23a48"))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, section.depth + 1)
    ax.set_xlabel("idealised transverse coordinate [m]")
    ax.set_ylabel("z above baseline [m]")
    ax.set_title("P350W1-N03 combined section (12 plate strips + 6 stiffener groups)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig5_section.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return [str(p) for p in paths]


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    prim = run_section(p350w1_n03_combined_section(), "p350w1_n03_combined")
    second = run_section(double_hull_side_section(), "double_hull_side",
                         exclude=SECONDARY_EXCLUDE)

    section = p350w1_n03_combined_section()

    # ---- criterion (b): negative control (closest to the neutral axis) ---
    neg = {b: prim["directions"][b]["negative_control"] for b in ("sagging", "hogging")}

    # ---- criterion (d): published closed-form benchmark ------------------
    bench = {}
    for bending in ("sagging", "hogging"):
        m = prim["matrices"][bending]
        rows = {}
        intact_ref = simplified_ultimate_moment(section, DamageSpec("PB1", "none"))
        rows["intact"] = {
            "M_u_smith_MNm": m["M_u_intact_MNm"],
            "M_u_reference_MNm": intact_ref["M_u_MNm"],
            "relative_error": abs(m["M_u_intact_MNm"] - intact_ref["M_u_MNm"])
            / intact_ref["M_u_MNm"],
        }
        worst_id = prim["directions"][bending]["top3_most_dangerous"][0]["component_id"]
        for cid, mode in ((worst_id, "remove"), ("SD1", "remove"), ("SD1", "trip")):
            e = next(x for x in m["entries"]
                     if x["component_id"] == cid and x["mode"] == mode)
            b = benchmark_vs_smith(section, DamageSpec(cid, mode),
                                   smith_M_u=e["M_u_damaged_MNm"] * 1e6)
            rows[f"{cid}:{mode}"] = {
                "M_u_smith_MNm": e["M_u_damaged_MNm"],
                "M_u_reference_MNm": b["M_u_reference_MNm"],
                "relative_error": b["relative_error"],
            }
        bench[bending] = {
            "method": intact_ref["method"],
            "cases": rows,
            "max_relative_error": max(r["relative_error"] for r in rows.values()),
        }
    dump(OUT / "benchmark.json", bench)

    # ---- criterion (e): failure-mode ordering on the compression flange --
    mode_order = {}
    for bending, cid in (("sagging", "SD1"), ("hogging", "SB1")):
        m = prim["matrices"][bending]
        vals = {mode: next(x["rif"] for x in m["entries"]
                           if x["component_id"] == cid and x["mode"] == mode)
                for mode in ("remove", "debond", "trip")}
        mode_order[bending] = {
            "component_id": cid,
            "rif": vals,
            "remove_lt_debond": bool(vals["remove"] < vals["debond"]),
            "debond_lt_trip": bool(vals["debond"] < vals["trip"]),
            "trip_lt_intact": bool(vals["trip"] < 1.0),
            "min_abs_gap": float(min(abs(vals["remove"] - vals["debond"]),
                                     abs(vals["debond"] - vals["trip"]))),
        }
    dump(OUT / "mode_ordering.json", mode_order)

    # ---- criterion (f): discretisation convergence ----------------------
    conv = []
    for bending, cid in (("sagging", "PD1"), ("sagging", "SD1"), ("hogging", "PB1")):
        for mode in ("remove", "trip", "debond"):
            spec = DamageSpec(cid, mode)
            item = convergence_study(section, spec, {
                "n_web_fibers": 2, "n_steps": 60,
                "kappa_max": SOLVER["kappa_max"], "bending": bending})
            conv.append(item)
    dump(OUT / "convergence.json", conv)

    # ---- sensitivity artifact -------------------------------------------
    dump(OUT / "sensitivity.json", {
        tag: {b: data["directions"][b]["sensitivity"] for b in data["directions"]}
        for tag, data in (("p350w1_n03_combined", prim),
                          ("double_hull_side", second))
    })

    # ---- ranking / redundancy summary -----------------------------------
    summary = {}
    for tag, data in (("p350w1_n03_combined", prim), ("double_hull_side", second)):
        summary[tag] = {
            "meta": data["matrices"]["sagging"] and None,
            "directions": {
                b: {
                    "M_u_intact_MNm": data["directions"][b]["M_u_intact_MNm"],
                    "z_NA_elastic_m": data["directions"][b]["z_NA_elastic_m"],
                    "kappa_u_intact": data["directions"][b]["kappa_u_intact"],
                    "rif_min": data["directions"][b]["rif_min"],
                    "rif_max": data["directions"][b]["rif_max"],
                    "z_NA_ultimate_m": data["directions"][b]["z_NA_ultimate_m"],
                    "worst_case_ranking": data["directions"][b]["worst_case_ranking"],
                    "top3_most_dangerous": data["directions"][b]["top3_most_dangerous"],
                    "redundancy_set": data["directions"][b]["redundancy_set"],
                    "negative_control": data["directions"][b]["negative_control"],
                }
                for b in data["directions"]
            },
        }
    summary["p350w1_n03_combined"]["meta"] = {
        "breadth_m": 24.0, "depth_m": 20.0,
        "n_plate_strips": 12, "n_stiffener_groups": 6,
        "frame_spacing_m": 2.4, "sigma_y_MPa": 235.0, "E_GPa": 206.0,
        "solver": SOLVER,
    }
    dump(OUT / "redundancy_summary.json", summary)

    figures = make_figures(prim, second)

    # ---- criteria -------------------------------------------------------
    sag = prim["directions"]["sagging"]
    hog = prim["directions"]["hogging"]
    all_rif = [e["rif"] for b in ("sagging", "hogging")
               for e in prim["matrices"][b]["entries"]]
    sec_all_rif = [e["rif"] for b in ("sagging", "hogging")
                   for e in second["matrices"][b]["entries"]]
    criteria = {
        "node": "p351w1_n08",
        "mission": 10727,
        "repo": "TensorPhysics",
        "module": "src/tensorphysics/hull_redundancy",
        "solver": SOLVER,
        "a_rif_range_and_negative_control": {
            "rif_min": float(min(all_rif)),
            "rif_max": float(max(all_rif)),
            "all_in_unit_interval": bool(
                all(0.0 <= v <= 1.0 + 1e-12 for v in all_rif + sec_all_rif)),
            "self_check_rif_sagging": sag["rif_negative_control"],
            "self_check_rif_hogging": hog["rif_negative_control"],
            "self_check_tolerance": 1e-3,
            "pass": bool(
                all(0.0 <= v <= 1.0 + 1e-12 for v in all_rif + sec_all_rif)
                and abs(sag["rif_negative_control"] - 1.0) <= 1e-3
                and abs(hog["rif_negative_control"] - 1.0) <= 1e-3),
        },
        "b_negative_control_neutral_axis": {
            "threshold": 0.98,
            "sagging": neg["sagging"],
            "hogging": neg["hogging"],
            "pass": bool(neg["sagging"]["ultimate_NA_pass"]
                         and neg["hogging"]["ultimate_NA_pass"]),
            "note": ("components are selected by the lever arm about the ultimate "
                     "(plastic) neutral axis; the elastic-NA reading is reported "
                     "as well because in sagging the plastic axis migrates "
                     "9.53 m -> 5.69 m when the deck sheds"),
        },
        "c_monotonicity_spearman": {
            "sagging_pooled_rho": sag["sensitivity"]["pooled"]["spearman_rho"],
            "sagging_plating_rho": sag["sensitivity"]["plating"]["spearman_rho"],
            "hogging_pooled_rho": hog["sensitivity"]["pooled"]["spearman_rho"],
            "hogging_plating_rho": hog["sensitivity"]["plating"]["spearman_rho"],
            "secondary_sagging_pooled_rho": second["directions"]["sagging"]["sensitivity"]["pooled"]["spearman_rho"],
            "secondary_excl_closing_rho": second["directions"]["sagging"]["sensitivity"]
            .get("excluding_closing_plating", {}).get("pooled", {}).get("spearman_rho"),
            "threshold": -0.8,
            "pass": bool(
                sag["sensitivity"]["pooled"]["spearman_rho"] <= -0.8
                and hog["sensitivity"]["pooled"]["spearman_rho"] <= -0.8
                and sag["sensitivity"]["plating"]["spearman_rho"] <= -0.8
                and hog["sensitivity"]["plating"]["spearman_rho"] <= -0.8),
        },
        "d_benchmark_vs_published_method": {
            "benchmark_type": "published closed-form simplified method "
                              "(Faulkner 1975 + Paik & Thayamballi 2003)",
            "cases": bench,
            "max_relative_error": max(v["max_relative_error"] for v in bench.values()),
            "tolerance": 0.15,
            "pass": bool(max(v["max_relative_error"] for v in bench.values()) <= 0.15),
            "limitation": "method-level benchmark; no experimental damaged-hull-girder "
                          "dataset is bundled in the repository",
        },
        "e_failure_mode_ordering": {
            "cases": mode_order,
            "tolerance_min_gap": 1e-3,
            "pass": bool(all(v["remove_lt_debond"] and v["debond_lt_trip"]
                             and v["trip_lt_intact"] and v["min_abs_gap"] > 1e-3
                             for v in mode_order.values())),
            "expected_order": "RIF(remove) < RIF(debond) < RIF(trip) < 1 "
                              "(removal most severe; tripping between loss of "
                              "composite action and the intact section)",
        },
        "f_discretisation_convergence": {
            "cases": conv,
            "max_relative_change": max(c["relative_change"] for c in conv),
            "tolerance": 0.03,
            "pass": bool(max(c["relative_change"] for c in conv) <= 0.03),
        },
        "g_pytest": {"test_file": "tests/test_hull_redundancy.py",
                     "min_tests_required": 18},
        "figures": figures,
    }
    criteria["all_pass"] = bool(
        criteria["a_rif_range_and_negative_control"]["pass"]
        and criteria["b_negative_control_neutral_axis"]["pass"]
        and criteria["c_monotonicity_spearman"]["pass"]
        and criteria["d_benchmark_vs_published_method"]["pass"]
        and criteria["e_failure_mode_ordering"]["pass"]
        and criteria["f_discretisation_convergence"]["pass"]
    )
    dump(OUT / "criteria.json", criteria)
    print(json.dumps({k: v["pass"] if isinstance(v, dict) and "pass" in v else v
                      for k, v in criteria.items() if k.startswith(("a_", "b_", "c_",
                                                                    "d_", "e_", "f_"))},
                     indent=2))
    print("all_pass:", criteria["all_pass"])
    print("artifacts:", OUT)
    return criteria


if __name__ == "__main__":
    main()