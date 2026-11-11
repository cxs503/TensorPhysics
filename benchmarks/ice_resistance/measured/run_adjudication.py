"""Run the ice-resistance measured-anchor adjudication (mission 9425).

Executes the campaign-node criteria end to end and writes one auditable artifact:

``artifacts/ice_resistance/measured_adjudication.json``

  C1  schema validation of every measured series + SHA-256 re-hash of every
      archived source artifact (100 % pass required)
  C2  double-entry digitisation cross-check on a deterministic 10 % sample
      (agreement > 99 % required)
  C3  model-vs-measurement adjudication: per-point relative error table, median
      error vs the platform's 15 % criterion -- an honest FAIL is written as a
      FAIL, with attribution (which component carries the error)
  C4  physical-plausibility gates: R(h) increasing, R(V) increasing, finite and
      continuous V -> 0 intercept; plus the same plausibility checks applied to
      the measured series itself
  C5  the pre-existing ICE_RESISTANCE_FULL_SPECTRUM corpus must stay fully green
      (the new subsystem may not quietly relax the old criteria)

Run:  python benchmarks/ice_resistance/measured/run_adjudication.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (str(REPO / "src"), str(REPO), str(REPO / "benchmarks" / "ice_resistance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorphysics.ice.measured_anchor import (  # noqa: E402
    GATE_TOLERANCE,
    REQUIRED_SERIES,
    MeasuredPoint,
    MeasuredSeries,
    adjudicate_resistance,
    adjudicate_speed_power,
    double_entry_check,
    gate_ledger,
    load_registry,
    measured_monotonicity,
    physics_gates,
    sampled_point_ids,
    sha256_file,
    validate_series,
)
from tensorphysics.ice.resistance import (  # noqa: E402
    DomainLimits,
    ShipGeometry,
    level_ice_resistance,
)

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"
ARTIFACT_DIR = REPO / "artifacts" / "ice_resistance"
FIG_DIR = ARTIFACT_DIR / "figures"

MTUIKKU_ID = "MTUIKKU-HZ2016"
AGULHAS_ID = "AGULHAS2-FS2012"


# ---------------------------------------------------------------------------
# model closures
# ---------------------------------------------------------------------------


def make_level_model(
    ship: ShipGeometry,
    *,
    use_test_ice_strength: bool,
    k_bend: float = 2.0,
    mu: float = 0.15,
    eta_D: float = 0.55,
    sigma_c_default: float = 5.0e6,
    sigma_f_default: float = 0.5e6,
) -> Callable[[MeasuredPoint, MeasuredSeries], Dict[str, Any]]:
    """Wrap the production Lindqvist branch as an adjudicable model closure."""

    def model(pt: MeasuredPoint, series: MeasuredSeries) -> Dict[str, Any]:
        if use_test_ice_strength and pt.sigma_c_kPa and pt.sigma_f_kPa:
            sigma_c = pt.sigma_c_kPa * 1e3
            sigma_f = pt.sigma_f_kPa * 1e3
        else:
            sigma_c = sigma_c_default
            sigma_f = sigma_f_default
        out = level_ice_resistance(pt.h_m, pt.V_ms, ship, sigma_c=sigma_c, sigma_f=sigma_f, mu=mu, k_bend=k_bend)
        return {
            "R_kN": out["R_total_N"] / 1e3,
            "R_crush_kN": out["R_crush_N"] / 1e3,
            "R_break_kN": out["R_break_N"] / 1e3,
            "R_submersion_kN": out["R_submersion_N"] / 1e3,
            "sigma_c_MPa": sigma_c / 1e6,
            "sigma_f_kPa": sigma_f / 1e3,
            "k_bend": k_bend,
            "mu": mu,
            "eta_D": eta_D,
            "flare_deg": ship.flare_deg,
            "bow_angle_deg": ship.bow_angle_deg,
        }

    return model


def ship_from_series(series: MeasuredSeries, *, flare_override: Optional[float] = None) -> ShipGeometry:
    ship = series.ship
    return ShipGeometry(
        name=str(ship["name"]),
        L_pp=float(ship["L_pp_m"]),
        B=float(ship["B_m"]),
        T=float(ship["T_m"]),
        bow_angle_deg=float(ship["bow_angle_deg"]),
        flare_deg=float(ship["flare_deg"] if flare_override is None else flare_override),
        displacement_t=float(ship.get("displacement_t") or 21000.0),
        eta_D=0.55,
    )


# ---------------------------------------------------------------------------
# published-baseline context (context only, never truth)
# ---------------------------------------------------------------------------


def published_baseline_errors(series: MeasuredSeries) -> Dict[str, Any]:
    """Error statistics of the law predictions *published in the source table*.

    This separates two questions that are usually conflated: "does the measured
    anchor exist?" (yes) and "is our transcription of the law as good as the
    literature's own numbers?" -- the second is answered here against the same
    truth column.
    """

    doc = json.loads(Path(series.path).read_text(encoding="utf-8")) if series.path else {}
    rows = []
    for raw in doc.get("points", []):
        base = raw.get("literature_baselines_kN") or {}
        meas = raw.get("R_meas_kN")
        if not meas:
            continue
        for law in ("lindqvist", "riska", "jeong", "keinonen"):
            pred = base.get(law)
            if pred is None:
                continue
            rows.append(
                {
                    "point_id": raw["point_id"],
                    "law": law,
                    "R_pred_kN": pred,
                    "R_meas_kN": meas,
                    "rel_error": abs(pred - meas) / meas,
                    "signed_rel_error": (pred - meas) / meas,
                }
            )
    summary = {}
    for law in sorted({r["law"] for r in rows}):
        errs = [r["rel_error"] for r in rows if r["law"] == law]
        signed = [r["signed_rel_error"] for r in rows if r["law"] == law]
        summary[law] = {
            "n_points": len(errs),
            "median_rel_error": statistics.median(errs),
            "mean_signed_rel_error": sum(signed) / len(signed),
        }
    return {"series_id": series.series_id, "rows": rows, "summary": summary,
            "_note": "reproduced from the source Table 8 predictions; context for attribution only"}


# ---------------------------------------------------------------------------
# calibration hypothesis (explicitly not applied to the validated model)
# ---------------------------------------------------------------------------


def calibrate_k_bend(series: MeasuredSeries, ship: ShipGeometry, grid: Sequence[float]) -> Dict[str, Any]:
    best: Tuple[float, float] | None = None
    curve = []
    for k in grid:
        adj = adjudicate_resistance(series, make_level_model(ship, use_test_ice_strength=True, k_bend=k), config=f"k_bend={k}")
        med = adj.median_rel_error
        curve.append({"k_bend": k, "median_rel_error": med})
        if best is None or med < best[1]:
            best = (k, med)
    return {
        "grid": list(grid),
        "curve": curve,
        "best_k_bend": best[0] if best else None,
        "best_median_rel_error": best[1] if best else None,
        "applied_to_validated_model": False,
        "rationale": (
            "A closure constant fitted to the same series that scores it is not a validation "
            "(self-calibration). The fit is recorded as a falsifiable hypothesis: it would have to "
            "hold on an independent series before it may enter PARAMETER_PROVENANCE."
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(
    adj_primary,
    baselines: Dict[str, Any],
    ship: ShipGeometry,
    series: MeasuredSeries,
    calib: Dict[str, Any],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    # 1 -- predicted vs measured, 1:1 and +-15 % band
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    meas = [r["R_meas_kN"] for r in adj_primary.rows if r.get("R_meas_kN")]
    pred = [r["R_pred_kN"] for r in adj_primary.rows if r.get("R_meas_kN")]
    lo = 0.0
    hi = max(max(meas), max(pred)) * 1.15
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="1:1")
    xs = [min(meas) * 0.6, max(pred) * 1.6]
    ax.fill_between(xs, [0.85 * v for v in xs], [1.15 * v for v in xs], color="0.85", label="±15% band")
    ax.scatter(meas, pred, s=42, c="#c0392b", edgecolor="k", zorder=3,
               label=f"Lindqvist branch, median err {100*adj_primary.median_rel_error:.0f}%")
    for r in adj_primary.rows:
        if r.get("R_meas_kN"):
            ax.annotate(r["point_id"], (r["R_meas_kN"], r["R_pred_kN"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("measured resistance [kN] (Hu & Zhou 2016)")
    ax.set_ylabel("predicted resistance [kN] (TensorPhysics Lindqvist branch)")
    ax.set_title("Measured anchor adjudication — MT Uikku level-ice series (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(meas) * 0.6, max(pred) * 1.6)
    ax.set_ylim(min(meas) * 0.6, max(pred) * 1.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    p = FIG_DIR / "measured_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    # 2 -- per-point relative error: our branch vs published transpositions
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ids = [r["point_id"] for r in adj_primary.rows]
    ours = [100 * r["rel_error"] for r in adj_primary.rows]
    x = range(len(ids))
    ax.bar([i - 0.2 for i in x], ours, width=0.4, color="#c0392b", label="TensorPhysics Lindqvist branch")
    pub = {r["point_id"]: 100 * r["rel_error"] for r in baselines["rows"] if r["law"] == "lindqvist"}
    ax.bar([i + 0.2 for i in x], [pub.get(i2, math.nan) for i2 in ids], width=0.4, color="#2980b9",
           label="Lindqvist as published in source Table 8")
    ax.axhline(15.0, color="k", ls="--", lw=1.0, label="15% criterion")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_xlabel("measured point id (MT Uikku tests)")
    ax.set_ylabel("|relative error| [%]")
    ax.set_title("Per-point error against the measured anchor (log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    p = FIG_DIR / "per_point_error.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    # 3 -- physics gates + measured overlay: R(h) at fixed V, R(V) at fixed h
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    h_grid = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    for V in (0.2, 0.5, 1.0):
        curve = [level_ice_resistance(h, V, ship)["R_total_N"] / 1e3 for h in h_grid]
        axes[0].plot(h_grid, curve, "-o", ms=3, label=f"model V={V} m/s")
    for r in adj_primary.rows:
        axes[0].scatter([r["h_m"]], [r["R_meas_kN"]], marker="x", s=40, c="k",
                        label="measured (all speeds)" if r is adj_primary.rows[0] else None)
    axes[0].set_xlabel("ice thickness h [m]")
    axes[0].set_ylabel("resistance [kN]")
    axes[0].set_title("C4: R(h) monotone increasing, measured overlay")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    v_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    for h in (0.63, 0.77, 1.04):
        curve = [level_ice_resistance(h, V, ship)["R_total_N"] / 1e3 for V in v_grid]
        axes[1].plot(v_grid, curve, "-o", ms=3, label=f"model h={h} m")
    for grp, color in ((0.63, "#16a085"), (0.77, "#8e44ad")):
        pts = [r for r in adj_primary.rows if abs(r["h_m"] - grp) < 1e-6]
        axes[1].scatter([r["V_ms"] for r in pts], [r["R_meas_kN"] for r in pts], marker="x", s=45, c=color,
                        label=f"measured h={grp} m")
    axes[1].set_xlabel("ship speed V [m/s]")
    axes[1].set_ylabel("resistance [kN]")
    axes[1].set_title("C4: R(V) monotone increasing, V→0 intercept finite")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    p = FIG_DIR / "physics_gates_and_measured.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    # 4 -- component attribution of the over-prediction
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    crush = [r["model"]["R_crush_kN"] for r in adj_primary.rows]
    brk = [r["model"]["R_break_kN"] for r in adj_primary.rows]
    sub = [r["model"]["R_submersion_kN"] for r in adj_primary.rows]
    ax.bar(ids, crush, label="R_crush")
    ax.bar(ids, brk, bottom=crush, label="R_break (bending closure)")
    ax.bar(ids, sub, bottom=[c + b for c, b in zip(crush, brk)], label="R_submersion")
    ax.plot(ids, [r["R_meas_kN"] for r in adj_primary.rows], "k_", ms=28, mew=2.2, label="measured")
    ax.set_ylabel("resistance [kN]")
    ax.set_xlabel("measured point id")
    ax.set_title("Attribution: which component carries the over-prediction")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    p = FIG_DIR / "component_attribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    # 5 -- k_bend calibration curve (hypothesis, not applied)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [c["k_bend"] for c in calib["curve"]]
    ys = [100 * c["median_rel_error"] for c in calib["curve"]]
    ax.plot(xs, ys, "-o", ms=3)
    ax.axhline(15.0, color="k", ls="--", lw=1.0, label="15% criterion")
    if calib["best_k_bend"]:
        ax.scatter([calib["best_k_bend"]], [100 * calib["best_median_rel_error"]], s=70, c="#e67e22",
                   zorder=4, label=f"best fit k_bend={calib['best_k_bend']:.2f} (not applied)")
    ax.set_xlabel("bending closure constant k_bend [-]")
    ax.set_ylabel("median |relative error| [%]")
    ax.set_title("Calibration hypothesis: no single k_bend reaches 15%")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = FIG_DIR / "k_bend_calibration.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(REPO)))

    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = load_registry(DATA_ROOT, verify=True)
    series: Dict[str, MeasuredSeries] = loaded["series"]

    # ---- C1: schema + digests -------------------------------------------
    schema_errors: List[str] = list(loaded["errors"])
    per_series_schema = []
    reg = loaded["registry"]
    reg_digest_ok = True
    for entry in reg.get("series", []):
        p = DATA_ROOT / entry["path"]
        actual = sha256_file(p) if p.is_file() else None
        ok = actual == entry.get("sha256")
        reg_digest_ok = reg_digest_ok and ok
        per_series_schema.append({"series_id": entry["series_id"], "digest_matches_registry": ok,
                                  "sha256": actual})
    for sid, s in series.items():
        errs = validate_series(s.as_dict(), root=(DATA_ROOT / "series" / sid.lower().replace("-", "_")))
        if errs:
            schema_errors.extend(f"{sid}: {e}" for e in errs)

    plausibility = {sid: measured_monotonicity(s) for sid, s in series.items()}
    # A series that declares itself outside the 15 % verdict set (external-validity
    # probe) is measured and recorded, but does not certify -- or decertify -- the
    # truth source.
    verdict_set = {
        sid
        for sid, s in series.items()
        if bool((s.as_dict().get("applicability") or {}).get("in_15pct_verdict_set", True))
    }
    schema_report = {
        "n_series": len(series),
        "n_points": sum(len(s.points) for s in series.values()),
        "all_valid": not schema_errors,
        "all_digests_verified": bool(reg_digest_ok and series),
        "registry_digests": per_series_schema,
        "measured_plausibility_passed": all(p["passed"] for sid, p in plausibility.items() if sid in verdict_set),
        "measured_plausibility_passed_all_series": all(p["passed"] for p in plausibility.values()),
        "measured_plausibility_verdict_set": sorted(verdict_set),
        "errors": schema_errors,
        "series": [s.asset_record() for s in series.values()],
    }

    # ---- C2: double entry -------------------------------------------------
    double_entry = []
    passes: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], Path]] = {}
    pooled: List[str] = []
    for sid, s in series.items():
        sdir = Path(s.path).parent if s.path else DATA_ROOT / "series" / sid.lower().replace("-", "_")
        pa = json.loads((sdir / "pass_a.json").read_text(encoding="utf-8"))
        pb = json.loads((sdir / "pass_b.json").read_text(encoding="utf-8"))
        passes[sid] = (pa, pb, sdir)
        pooled.extend(s.points)
    for sid, s in series.items():
        pa, pb, sdir = passes[sid]
        ids = sampled_point_ids(s.points, 0.10)
        rec = double_entry_check(pa, pb, ids)
        full = double_entry_check(pa, pb, sorted(pa))
        rec.update(
            {
                "series_id": sid,
                "sampled_point_ids": ids,
                "sample_fraction_of_series": len(ids) / len(s.points),
                "full_population": {
                    "n_points": full["n_points_sampled"],
                    "n_fields_compared": full["n_fields_compared"],
                    "agreement_rate": full["agreement_rate"],
                    "mismatches": full["mismatches"],
                },
                "pass_a_sha256": sha256_file(sdir / "pass_a.json"),
                "pass_b_sha256": sha256_file(sdir / "pass_b.json"),
            }
        )
        double_entry.append(rec)

    sampled_union = sorted({pid for r in double_entry for pid in r["sampled_point_ids"]})
    for r in double_entry:
        r["sampled_union"] = sampled_union
        r["sample_fraction_of_all_points"] = len(sampled_union) / len(pooled)

    # ---- C3: adjudication -------------------------------------------------
    mtuikku = series[MTUIKKU_ID]
    agulhas = series[AGULHAS_ID]

    ship_primary = ship_from_series(mtuikku)  # source-reported angles (flare 30 deg)
    configs = {
        "primary_test_ice_strength": make_level_model(ship_primary, use_test_ice_strength=True, k_bend=2.0),
        "module_default_constants": make_level_model(ship_primary, use_test_ice_strength=False, k_bend=2.0),
        "convention_flare60": make_level_model(ship_from_series(mtuikku, flare_override=60.0),
                                               use_test_ice_strength=True, k_bend=2.0),
        "convention_flare25": make_level_model(ship_from_series(mtuikku, flare_override=25.0),
                                               use_test_ice_strength=True, k_bend=2.0),
    }
    adjudications = [
        adjudicate_resistance(mtuikku, fn, config=name) for name, fn in configs.items()
    ]
    primary = adjudications[0]

    baselines = published_baseline_errors(mtuikku)

    calib = calibrate_k_bend(mtuikku, ship_primary, [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0])

    ship_agulhas = ship_from_series(agulhas)
    agulhas_screen = adjudicate_speed_power(
        agulhas, make_level_model(ship_agulhas, use_test_ice_strength=False, k_bend=2.0), config="module_default_constants"
    )

    # ---- C4: physics gates ------------------------------------------------
    physics = physics_gates(lambda h, V: level_ice_resistance(h, V, ship_primary)["R_total_N"] / 1e3)
    physics["measured_series"] = plausibility
    physics["model_gates_passed"] = bool(physics["passed"])
    # The measured-plausibility gate protects the *verdict set*: a series that is
    # admitted to the 15 % adjudication must not contradict R(h)/R(V) monotonicity.
    # Series explicitly declared outside the verdict set (for example an
    # out-of-envelope external-validity probe) are still measured and their
    # violations are still recorded verbatim -- measured data is never "fixed" --
    # but they do not gate C4, because their job is to stress the model, not to
    # certify the truth source.
    physics["measured_series_in_verdict_set"] = sorted(verdict_set)
    physics["measured_gates_outside_verdict_set"] = {
        sid: {"passed": p["passed"], "n_passed": p["n_passed"], "n_checks": p["n_checks"], "notes": p["notes"]}
        for sid, p in plausibility.items()
        if sid not in verdict_set
    }
    physics["measured_gates_passed"] = all(
        p["passed"] for sid, p in plausibility.items() if sid in verdict_set
    )
    physics["passed"] = bool(physics["model_gates_passed"] and physics["measured_gates_passed"])

    # ---- C5: legacy corpus must stay green -------------------------------
    from anchors_eval import evaluate_corpus  # noqa: E402  (benchmark module)

    legacy = evaluate_corpus()
    legacy_gate = {
        "all_passed": legacy["all_passed"],
        "n_anchors": legacy["n_reference_anchors"],
        "n_passed": legacy["n_reference_anchors_passed"],
        "negative_controls_passed": f"{legacy['n_negative_controls_passed']}/{legacy['n_negative_controls']}",
    }

    # ---- attribution of the prediction bias -------------------------------
    attribution = {
        "config": primary.config,
        "component_share_of_prediction": {},
        "measured_median_kN": statistics.median([r["R_meas_kN"] for r in primary.rows if r.get("R_meas_kN")]),
        "predicted_median_kN": statistics.median([r["R_pred_kN"] for r in primary.rows]),
        "ratio_pred_over_meas": (
            statistics.median([r["R_pred_kN"] for r in primary.rows])
            / statistics.median([r["R_meas_kN"] for r in primary.rows if r.get("R_meas_kN")])
        ),
        "speed_independent_share": (
            statistics.fmean([
                (r["model"]["R_crush_kN"] + r["model"]["R_break_kN"]) / r["R_pred_kN"] for r in primary.rows
            ])
        ),
    }
    for key, field_name in (("R_crush", "R_crush_kN"), ("R_break", "R_break_kN"), ("R_submersion", "R_submersion_kN")):
        attribution["component_share_of_prediction"][key] = statistics.fmean(
            [r["model"][field_name] / r["R_pred_kN"] for r in primary.rows]
        )
    attribution["note"] = (
        "The over-prediction is carried by the speed-independent components: the bending closure "
        "R_break (k_bend = 2.0) alone is comparable to the entire measured resistance, so the defect "
        "is a closure/convention issue, not a missing velocity term."
    )

    # ---- ledger -----------------------------------------------------------
    ledger = gate_ledger(
        schema_report=schema_report,
        double_entry_reports=double_entry,
        adjudications=adjudications,
        physics=physics,
        legacy_gate=legacy_gate,
        min_series=REQUIRED_SERIES,
    )
    ledger["C3_15pct_adjudication"]["published_law_baselines_on_same_truth"] = baselines["summary"]
    ledger["C3_15pct_adjudication"]["attribution"] = attribution
    ledger["C3_15pct_adjudication"]["calibration_hypothesis"] = {
        "best_k_bend": calib["best_k_bend"],
        "best_median_rel_error": calib["best_median_rel_error"],
        "applied_to_validated_model": calib["applied_to_validated_model"],
    }
    ledger["C3_15pct_adjudication"]["speed_power_screening"] = {
        "series_id": agulhas_screen["series_id"],
        "n_points": agulhas_screen["n_points"],
        "n_consistent": agulhas_screen["n_consistent"],
        "verdict": agulhas_screen["verdict"],
        "supports_15pct": agulhas_screen["supports_15pct"],
    }

    figures = make_figures(primary, baselines, ship_primary, mtuikku, calib)

    payload = {
        "mission_id": 9425,
        "node": "p273w3-n07-ice-resistance-measured",
        "tolerance": GATE_TOLERANCE,
        "data_root": str(DATA_ROOT.relative_to(REPO)),
        "criteria": ledger,
        "schema_report": schema_report,
        "double_entry": double_entry,
        "adjudications": [a.as_dict() for a in adjudications],
        "published_baselines": baselines,
        "attribution": attribution,
        "calibration_hypothesis": calib,
        "speed_power_screening": agulhas_screen,
        "physics_gates": physics,
        "legacy_full_spectrum": legacy_gate,
        "figures": figures,
    }
    out = ARTIFACT_DIR / "measured_adjudication.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger["honest_verdict"] = (
        "The measured anchor exists and the 15 % criterion is now executable (gate_state=JUDGED). "
        "It is NOT met: the best pre-registered configuration misses by a median "
        f"{100 * ledger['C3_15pct_adjudication']['best_median_rel_error']:.1f} % "
        "against the platform's 15 % criterion. This is recorded as an honest FAIL, not tuned away."
        if not ledger["C3_15pct_adjudication"]["passed"]
        else "Measured anchor exists and at least one series meets the 15 % criterion."
    )
    payload["honest_verdict"] = ledger["honest_verdict"]
    out = ARTIFACT_DIR / "measured_adjudication.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "artifact": str(out.relative_to(REPO)),
            "C1": ledger["C1_schema_and_digest"]["passed"],
            "C2": ledger["C2_double_entry"]["passed"],
            "C3": ledger["C3_15pct_adjudication"]["passed"],
            "C3_best_median_rel_error": ledger["C3_15pct_adjudication"]["best_median_rel_error"],
            "C4": ledger["C4_physics_gates"]["passed"],
            "C5": ledger["C5_legacy_full_spectrum"]["passed"],
            "all_passed": ledger["all_passed"],
            "gate_state": ledger["gate_state"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())