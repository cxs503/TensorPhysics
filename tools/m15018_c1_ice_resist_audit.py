#!/usr/bin/env python3
"""m15018 — 交差审计（bridge audit）of m14881 / C1-ICE-RESIST.

Independent, re-runnable machine check of the delivered three-piece set
(impl_files / run_artifacts / benchmark numbers / repro_cmd) plus an
adversarial cross-examination that the delivery report does not perform:

  A. determinism   — recompute the model outputs from the delivered script and
                     compare with the delivered artifacts (canonical SHA-256).
  B. artifact ledger — which delivered run_artifacts are actually produced by
                     the declared repro_cmd, which are orphans of an older
                     script revision.
  C. cross-implementation divergence — delivered `Lindqvist (1989)` script
                     model vs the repository library model
                     `tensorphysics.ice.resistance.level_ice_resistance`
                     (the repo ships `cross_impl_attribution` for exactly this).
  D. fidelity to the source paper — delivered model vs the published
                     predictions of the same laws carried in the anchor CSV
                     (Hu & Zhou 2016, Table 8).
  E. applicability-domain check — delivered `applicability_domains.json`
                     versus the (h, V) actually used.
  F. dimensional check of the delivered Lindqvist buoyancy term.

Outputs (docs/m15018/ so they survive the merge-time artifacts/ filter):
  docs/m15018/audit_c1_ice_resist.json
  docs/m15018/repro/uikku_comparison.regenerated.json
  docs/m15018/fig_audit_lindqvist_paths.png
  docs/m15018/fig_audit_error_and_domain.png

Run: PYTHONPATH=src python tools/m15018_c1_ice_resist_audit.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.ice.resistance import ShipGeometry, level_ice_resistance  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "m15018"
REPRO = OUT / "repro"
# as-delivered worktree of the audited mission (read-only evidence source)
DELIVERED = Path("/root/tp-agent/work/m14881")
ART = "artifacts/ice_resistance"
# files the declared repro_cmd actually (re)writes — verified by grep of the script
REPRO_WRITES = [
    f"{ART}/uikku_comparison.json",
    f"{ART}/applicability_domains.json",
    f"{ART}/fig_uikku_comparison.png",
    f"{ART}/fig_uikku_scatter.png",
]
DELIVERED_ARTIFACTS = REPRO_WRITES + [
    f"{ART}/uikku_error_summary.json",
    f"{ART}/fig_error_distribution.png",
]


def load_delivered_module():
    p = REPO / "scripts" / "c1_ice_resist_uikku_analysis.py"
    spec = importlib.util.spec_from_file_location("c1_delivered", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def pct_stats(errs):
    e = sorted(errs)
    return {
        "n": len(e),
        "mean_pct": round(statistics.mean(e), 1),
        "median_pct": round(statistics.median(e), 1),
        "max_pct": round(max(e), 1),
        "n_under_25pct": sum(x < 25.0 for x in e),
        "n_under_15pct": sum(x < 15.0 for x in e),
        "pct_under_25pct": round(100.0 * sum(x < 25.0 for x in e) / len(e), 1),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    REPRO.mkdir(parents=True, exist_ok=True)
    c1 = load_delivered_module()
    series, points = c1.load_all_data()
    s = series["ship"]
    ship = ShipGeometry(
        name=s["name"], L_pp=s["L_pp_m"], B=s["B_m"], T=s["T_m"],
        bow_angle_deg=s["bow_angle_deg"], flare_deg=s["flare_deg"],
    )

    rows = []
    for p in points:
        sf, sc = p["sigma_f_kPa"] * 1e3, p["sigma_c_kPa"] * 1e3
        lib = level_ice_resistance(p["h_m"], p["V_ms"], ship, sigma_c=sc, sigma_f=sf, mu=0.15)
        our_lin = c1.lindqvist_buoyancy(
            p["h_m"], p["V_ms"], ship.L_pp, ship.B, s["bow_angle_deg"], sf, sc,
            flare_deg=s["flare_deg"]) / 1e3
        our_kei = c1.keinonen_level(
            p["h_m"], p["V_ms"], ship.L_pp, ship.B, s["bow_angle_deg"], sf, sc) / 1e3
        our_ris = c1.riska_ahead(
            p["h_m"], p["V_ms"], ship.L_pp, ship.B, s["bow_angle_deg"], sf, sc,
            flare_deg=s["flare_deg"]) / 1e3
        rows.append({
            "point_id": p["point_id"], "h_m": p["h_m"], "V_ms": p["V_ms"],
            "sigma_f_kPa": p["sigma_f_kPa"], "sigma_c_kPa": p["sigma_c_kPa"],
            "R_meas_kN": p["R_meas_kN"],
            "our_lindqvist_buoy_kN": round(our_lin, 1),
            "our_keinonen_kN": round(our_kei, 1),
            "our_riska_kN": round(our_ris, 1),
            "lib_level_ice_total_kN": round(lib["R_total_kN"], 1),
            "lit_lindqvist_kN": p["lit_lindqvist_kN"],
            "lit_keinonen_kN": p["lit_keinonen_kN"],
            "lit_riska_kN": p["lit_riska_kN"],
        })

    series_err = {
        "our_lindqvist_buoy": [abs(r["our_lindqvist_buoy_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
        "our_keinonen": [abs(r["our_keinonen_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
        "our_riska": [abs(r["our_riska_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
        "lib_level_ice_total": [abs(r["lib_level_ice_total_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
        "lit_lindqvist": [abs(r["lit_lindqvist_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
        "lit_keinonen": [abs(r["lit_keinonen_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
        "lit_riska": [abs(r["lit_riska_kN"] - r["R_meas_kN"]) / r["R_meas_kN"] * 100 for r in rows],
    }
    stats = {k: pct_stats(v) for k, v in series_err.items()}

    # regenerated artifact copy (survives the merge-time artifacts/ filter)
    # true determinism check: re-execute the delivered evaluation function itself
    delivered_eval = c1.evaluate_all(series, points)
    delivered_stats = c1.compute_stats(
        delivered_eval, ["our_lin", "lit_lin", "our_kei", "lit_kei", "our_ris", "lit_ris"])
    series_dump = {"points": delivered_eval, "statistics": delivered_stats}
    (REPRO / "uikku_comparison.regenerated.json").write_text(json.dumps(series_dump, indent=2))

    # ---- A. determinism against the delivered artifact -------------------
    delivered_cmp = DELIVERED / ART / "uikku_comparison.json"
    det = {"source": str(delivered_cmp), "exists": delivered_cmp.exists()}
    if delivered_cmp.exists():
        d = json.loads(delivered_cmp.read_text())
        regen = json.loads((REPRO / "uikku_comparison.regenerated.json").read_text())
        det.update({
            "schema": d.get("schema"),
            "points_identical": d.get("points") == regen["points"],
            "statistics_identical": d.get("statistics") == regen["statistics"],
            "sha256_delivered_points": canon_sha(d.get("points")),
            "sha256_regenerated_points": canon_sha(regen["points"]),
        })

    # ---- B. artifact ledger ---------------------------------------------
    ledger = []
    for rel in DELIVERED_ARTIFACTS:
        p = DELIVERED / rel
        ledger.append({
            "artifact": rel,
            "present_in_delivered_worktree": p.exists(),
            "bytes": p.stat().st_size if p.exists() else None,
            "produced_by_repro_cmd": rel in REPRO_WRITES,
        })
    stale = [x["artifact"] for x in ledger if x["present_in_delivered_worktree"] and not x["produced_by_repro_cmd"]]
    summary_v2 = DELIVERED / ART / "uikku_error_summary.json"
    stale_meta = None
    if summary_v2.exists():
        sd = json.loads(summary_v2.read_text())
        stale_meta = {"schema": sd.get("schema"), "series": sorted(sd.get("summary", {})),
                      "lit_series_n": {k: v.get("n") for k, v in sd.get("summary", {}).items()
                                       if k.startswith("lit_")}}

    # ---- C. cross-implementation divergence ------------------------------
    div = [abs(r["our_lindqvist_buoy_kN"] - r["lib_level_ice_total_kN"]) / r["lib_level_ice_total_kN"] * 100 for r in rows]
    lit_dev_lin = [abs(r["our_lindqvist_buoy_kN"] - r["lit_lindqvist_kN"]) / r["lit_lindqvist_kN"] * 100 for r in rows]
    ratio_kei = [r["our_keinonen_kN"] / r["lit_keinonen_kN"] for r in rows]
    ratio_lin = [r["our_lindqvist_buoy_kN"] / r["lit_lindqvist_kN"] for r in rows]
    ratio_ris = [r["our_riska_kN"] / r["lit_riska_kN"] for r in rows]

    # ---- E. applicability-domain check ----------------------------------
    domains = json.loads((DELIVERED / ART / "applicability_domains.json").read_text()) \
        if (DELIVERED / ART / "applicability_domains.json").exists() else {}
    applied = {"Lindqvist_1989": "our_lindqvist_buoy", "Keinonen_1983": "our_keinonen", "Riska_1997": "our_riska"}
    violations = []
    for model, dom in domains.items():
        for r in rows:
            if not (dom["h_m"][0] <= r["h_m"] <= dom["h_m"][1]):
                violations.append({"model": model, "point_id": r["point_id"], "var": "h_m", "value": r["h_m"], "domain": dom["h_m"]})
            if not (dom["V_ms"][0] <= r["V_ms"] <= dom["V_ms"][1]):
                violations.append({"model": model, "point_id": r["point_id"], "var": "V_ms", "value": r["V_ms"], "domain": dom["V_ms"]})

    # ---- F. dimensional check of the delivered Lindqvist buoyancy term ---
    dim = {
        "term": "R_f = 0.5 * rho_i * g * h * B * (sin(phi/2) + mu cos(phi/2))",
        "symbolic_dimension": "[kg/m^3]*[m/s^2]*[m]*[m] = kg/(m*s^2) = N/m",
        "expected_dimension": "N",
        "verdict": "DIMENSION_MISMATCH (missing one length factor; the term is a force per unit length)",
        "R_f_term_only_pt103_kN": round(0.5 * 900.0 * 9.80665 * 0.77 * 21.3 * (math.sin(math.radians(10.5)) + 0.15 * math.cos(math.radians(10.5))) / 1e3, 1),
        "total_pt103_kN": round(c1.lindqvist_buoyancy(0.77, 0.2, 150.0, 21.3, 21.0, 724e3, 1748e3, flare_deg=30.0) / 1e3, 1),
    }

    result = {
        "schema": "m15018.c1_ice_resist_audit.v1",
        "audited_mission": 14881,
        "audit_mission": 15018,
        "generated": datetime.now(timezone.utc).isoformat(),
        "anchor": "MTUIKKU-HZ2016 (Hu & Zhou 2016, doi:10.1016/j.ijnaoe.2016.01.004)",
        "points": rows,
        "stats_vs_measured": stats,
        "determinism": det,
        "artifact_ledger": ledger,
        "stale_artifacts": stale,
        "stale_artifact_meta": stale_meta,
        "cross_impl_divergence": {
            "ours_lindqvist_buoy_vs_library": {
                "mean_pct": round(statistics.mean(div), 1),
                "max_pct": round(max(div), 1),
            },
            "ours_lindqvist_buoy_vs_published_lindqvist": {
                "mean_pct": round(statistics.mean(lit_dev_lin), 1),
                "max_pct": round(max(lit_dev_lin), 1),
                "ratio_mean": round(statistics.mean(ratio_lin), 2),
            },
            "ours_keinonen_vs_published_keinonen_ratio": {
                "mean": round(statistics.mean(ratio_kei), 2),
                "min": round(min(ratio_kei), 2),
                "max": round(max(ratio_kei), 2),
            },
            "ours_riska_vs_published_riska_ratio": {
                "mean": round(statistics.mean(ratio_ris), 2),
                "min": round(min(ratio_ris), 2),
                "max": round(max(ratio_ris), 2),
            },
        },
        "applicability_domains": domains,
        "applicability_violations": violations,
        "dimensional_check": dim,
        "prior_art_same_anchor": {
            "doc": "docs/ICE_RESISTANCE_MEASURED_ANCHOR.md",
            "mission": 9425,
            "verdict": "honest-FAIL, median relative error 198.3% (library Lindqvist path)",
            "cited_by_m14881": False,
        },
    }
    (OUT / "audit_c1_ice_resist.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- figures ---------------------------------------------------------
    x = [r["point_id"] for r in rows]
    w = 0.2
    idx = range(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - 1.5 * w for i in idx], [r["R_meas_kN"] for r in rows], w, label="measured (Hu&Zhou Table 3)", color="#2c3e50")
    ax.bar([i - 0.5 * w for i in idx], [r["our_lindqvist_buoy_kN"] for r in rows], w, label="m14881 Lindqvist (buoyancy variant)", color="#2980b9")
    ax.bar([i + 0.5 * w for i in idx], [r["lib_level_ice_total_kN"] for r in rows], w, label="repo library level_ice_resistance", color="#e67e22")
    ax.bar([i + 1.5 * w for i in idx], [r["lit_lindqvist_kN"] for r in rows], w, label="published Lindqvist (Table 8)", color="#27ae60")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([f"{p}\nh={r['h_m']:.2f} V={r['V_ms']:.1f}" for p, r in zip(x, rows)], fontsize=8)
    ax.set_ylabel("resistance [kN]")
    ax.set_title("MT Uikku level ice: three Lindqvist paths diverge (m15018 audit of m14881)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_audit_lindqvist_paths.png", dpi=140)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, lab, col, mk in [
        ("our_lindqvist_buoy", "m14881 Lindqvist(buoy)", "#2980b9", "o"),
        ("our_keinonen", "m14881 Keinonen", "#8e44ad", "s"),
        ("our_riska", "m14881 Riska", "#c0392b", "^"),
        ("lib_level_ice_total", "repo library Lindqvist", "#e67e22", "d"),
        ("lit_lindqvist", "published Lindqvist", "#27ae60", "x"),
    ]:
        ax1.plot(list(idx), series_err[key], marker=mk, label=lab, lw=1.4, ms=5)
    ax1.axhline(15.0, color="k", ls="--", lw=1, label="15% acceptability")
    ax1.set_yscale("log")
    ax1.set_xticks(list(idx))
    ax1.set_xticklabels(x, fontsize=8)
    ax1.set_ylabel("|pred - meas| / meas  [%]  (log)")
    ax1.set_title("per-point relative error vs measurement")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    labels = ["m14881 Lindqvist(buoy)", "repo library Lindqvist", "published Lindqvist"]
    vals = [stats["our_lindqvist_buoy"]["pct_under_25pct"], stats["lib_level_ice_total"]["pct_under_25pct"],
            stats["lit_lindqvist"]["pct_under_25pct"]]
    bars = ax2.bar(labels, vals, color=["#2980b9", "#e67e22", "#27ae60"])
    ax2.axhline(80.0, color="k", ls="--", lw=1, label="m14881 criterion 80%")
    ax2.axhline(100.0 * sum(1 for e in series_err["our_lindqvist_buoy"] if e < 15) / len(rows),
                color="r", ls=":", lw=1.2, label="<15% share (Lindqvist buoy)")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of 10 points under 25% error")
    ax2.set_title("criterion share + 15% share")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_audit_error_and_domain.png", dpi=140)
    plt.close(fig)

    print(json.dumps({k: result[k] for k in
                      ("determinism", "stale_artifacts", "cross_impl_divergence",
                       "dimensional_check")}, indent=2, ensure_ascii=False))
    print("applicability violations:", len(violations))
    print("stats:", json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())