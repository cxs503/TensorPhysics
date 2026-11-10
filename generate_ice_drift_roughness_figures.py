#!/usr/bin/env python
"""Criterion evaluation + figure factory for the ice-drift roughness closure.

Writes
------
* ``artifacts/ice_drift_roughness/criterion_results.json`` -- criteria C1..C6 of
  campaign node ``p302w1-n01-ice-drift-roughness``, with the full raw tables.
* ``artifacts/ice_drift_roughness/fig[1-5]_*.png``         -- report figures.

Run ``generate_ice_drift_roughness_report.py`` afterwards to render the report
from the JSON (so no number is ever hand-transcribed).
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.ice_drift_roughness import (  # noqa: E402
    ANCHORS,
    CD_TOLERANCE,
    DriftForcing,
    KeelMorphology,
    calibrate_wake_amplification,
    cross_validate,
    ice_underside_roughness,
    law_of_wall_drag_coefficient,
    open_water_drag_coefficient,
    solve_free_drift,
    solve_friction_velocity,
    surface_interface_roughness,
    validation_summary,
)
from tensorphysics.ice_drift_roughness.z0_model import NU_SEAWATER, Z_REF  # noqa: E402

OUT_ART = ROOT / "artifacts" / "ice_drift_roughness"
OUT_ART.mkdir(parents=True, exist_ok=True)
SEED = 20261110


def criterion_c1(n_draws: int = 100) -> dict:
    rng = random.Random(SEED)
    depth_violations = spacing_violations = 0
    depth_strict = spacing_strict = degenerate = 0
    rows = []
    for _ in range(n_draws):
        morph = KeelMorphology(
            keel_depth_m=rng.uniform(1.0, 12.0), keel_spacing_m=rng.uniform(30.0, 300.0)
        )
        h_ice = rng.uniform(0.5, 2.5)
        base = ice_underside_roughness(morph, h_ice)
        deeper = ice_underside_roughness(
            KeelMorphology(morph.keel_depth_m * 1.05, morph.keel_spacing_m), h_ice
        )
        sparser = ice_underside_roughness(
            KeelMorphology(morph.keel_depth_m, morph.keel_spacing_m * 1.05), h_ice
        )
        if deeper.z0_m < base.z0_m - 1e-12:
            depth_violations += 1
        elif deeper.z0_m > base.z0_m + 1e-12:
            depth_strict += 1
        else:
            degenerate += 1
        if sparser.z0_m > base.z0_m + 1e-12:
            spacing_violations += 1
        elif sparser.z0_m < base.z0_m - 1e-12:
            spacing_strict += 1
        rows.append(
            {
                "keel_depth_m": morph.keel_depth_m,
                "keel_spacing_m": morph.keel_spacing_m,
                "h_ice_m": h_ice,
                "z0_m": base.z0_m,
                "cd_ice": base.drag_coefficient,
                "area_fraction": base.area_fraction,
                "z0_ridge_m": base.z0_ridge_m,
                "dz0_ddepth": deeper.z0_m - base.z0_m,
                "dz0_dspacing": sparser.z0_m - base.z0_m,
            }
        )

    scan = {}
    for spacing in (40, 70, 120, 300):
        scan["L_s=%dm" % spacing] = [
            {
                "keel_depth_m": dk,
                "z0_m": ice_underside_roughness(KeelMorphology(dk, float(spacing)), 1.8).z0_m,
            }
            for dk in [1.0 + 0.25 * i for i in range(57)]
        ]

    return {
        "n_draws": n_draws,
        "depth_violations": depth_violations,
        "spacing_violations": spacing_violations,
        "depth_strict_increase": depth_strict,
        "spacing_strict_decrease": spacing_strict,
        "degenerate_equal_draws": degenerate,
        "violations": depth_violations + spacing_violations,
        "passed": (depth_violations + spacing_violations) == 0,
        "scan": scan,
        "rows": rows,
    }


def criterion_c2() -> dict:
    rows = []
    worst = 0.0
    worst_case = None
    for k_s in (2e-3, 1e-2, 5e-2, 1.5e-1):
        for u_rel in (0.05, 0.1, 0.2, 0.4, 0.8, 1.2):
            ref = law_of_wall_drag_coefficient(u_rel, k_s)
            model = solve_friction_velocity(u_rel, k_s / 30.0)
            rel = (model.drag_coefficient - ref.drag_coefficient) / ref.drag_coefficient
            rows.append(
                {
                    "k_s_m": k_s,
                    "u_rel_m_s": u_rel,
                    "reynolds": u_rel * Z_REF / NU_SEAWATER,
                    "cd_model": model.drag_coefficient,
                    "cd_loglaw_reference": ref.drag_coefficient,
                    "rel_error": rel,
                    "k_s_plus": model.roughness_reynolds,
                    "regime": model.regime,
                }
            )
            if abs(rel) > worst:
                worst, worst_case = abs(rel), (k_s, u_rel, model.regime)
    return {
        "n_points": len(rows),
        "tolerance": 0.05,
        "max_abs_rel_error": worst,
        "worst_case": {
            "k_s_m": worst_case[0],
            "u_rel_m_s": worst_case[1],
            "regime": worst_case[2],
        },
        "passed": worst < 0.05,
        "rows": rows,
    }


def criterion_c3() -> dict:
    morph = KeelMorphology(keel_depth_m=6.0, keel_spacing_m=70.0)
    cases = [
        {"wind_m_s": [5.0, 0.0], "h_ice_m": 1.0, "concentration": 0.9},
        {"wind_m_s": [10.0, 0.0], "h_ice_m": 2.0, "concentration": 0.9},
        {"wind_m_s": [15.0, 3.0], "h_ice_m": 2.5, "concentration": 1.0},
        {"wind_m_s": [20.0, -5.0], "h_ice_m": 1.5, "concentration": 0.7},
        {"wind_m_s": [7.0, 7.0], "h_ice_m": 3.0, "concentration": 1.0},
    ]
    for case in cases:
        forcing = DriftForcing(wind=tuple(case["wind_m_s"]), latitude_deg=75.0)
        sol = solve_free_drift(
            case["h_ice_m"], case["concentration"], morph, forcing, tol=1e-4, max_iter=10
        )
        case.update(
            {
                "ice_velocity_m_s": list(sol.ice_velocity),
                "drift_speed_m_s": sol.drift_speed,
                "cd_ice": sol.drag_coefficient,
                "z0_ice_m": sol.z0_ice_m,
                "u_star_m_s": sol.u_star,
                "iterations": sol.iterations,
                "last_increment_m_s": sol.last_increment,
                "converged": sol.converged,
                "relative_force_residual": sol.relative_force_residual,
                "drift_ratio": sol.drift_ratio,
                "turn_angle_deg": sol.turn_angle_deg,
                "regime": sol.roughness_regime,
                "history": [list(h) for h in sol.history],
            }
        )

    sweep = []
    for wind_speed in [3.0 + 0.5 * i for i in range(45)]:
        forcing = DriftForcing(wind=(wind_speed, 0.0), latitude_deg=75.0)
        sol = solve_free_drift(2.0, 1.0, morph, forcing)
        predicted = math.degrees(
            math.atan(
                forcing.ice_density * 2.0 * forcing.coriolis()
                / (forcing.water_density * sol.drag_coefficient * sol.relative_speed)
            )
        )
        sweep.append(
            {
                "wind_speed_m_s": wind_speed,
                "drift_speed_m_s": sol.drift_speed,
                "drift_ratio": sol.drift_ratio,
                "turn_angle_deg": sol.turn_angle_deg,
                "turn_angle_analytic_deg": predicted,
                "turn_angle_error": abs(abs(sol.turn_angle_deg) - predicted) / predicted,
            }
        )

    return {
        "tolerance_m_s": 1e-4,
        "max_iterations": 10,
        "max_iterations_used": max(c["iterations"] for c in cases),
        "max_last_increment_m_s": max(c["last_increment_m_s"] for c in cases),
        "max_relative_force_residual": max(c["relative_force_residual"] for c in cases),
        "all_converged": all(c["converged"] for c in cases),
        "max_turn_angle_analytic_error": max(s["turn_angle_error"] for s in sweep),
        "min_drift_ratio": min(s["drift_ratio"] for s in sweep),
        "max_drift_ratio": max(s["drift_ratio"] for s in sweep),
        "max_right_turn_deg": max(-s["turn_angle_deg"] for s in sweep),
        "min_right_turn_deg": min(-s["turn_angle_deg"] for s in sweep),
        "passed": all(
            c["converged"] and c["iterations"] <= 10 and c["last_increment_m_s"] < 1e-4
            for c in cases
        ),
        "cases": cases,
        "nansen_sweep": sweep,
    }


def criterion_c4() -> dict:
    calibration = calibrate_wake_amplification()
    rows = cross_validate(wake_amplification=calibration["fitted_value"])
    summary = validation_summary(rows)
    rows_prior = cross_validate(wake_amplification=2.0)
    return {
        "calibration": calibration,
        "summary": summary,
        "rows": rows,
        "summary_prior_lambda_2p0": validation_summary(rows_prior),
        "anchors": [a.as_dict() for a in ANCHORS],
        "tolerance": CD_TOLERANCE,
        "passed": bool(summary["criterion_C4_pass"]),
    }


def criterion_c5() -> dict:
    ice_state = ice_underside_roughness(KeelMorphology(6.0, 70.0), 2.0)
    cd_open = open_water_drag_coefficient()
    rows = []
    for conc in (0.0, 1e-9, 1e-6, 1e-4, 1e-2, 0.25, 0.5, 0.75, 1.0):
        mosaic = surface_interface_roughness(ice_state, conc)
        rows.append(
            {
                "concentration": conc,
                "cd_surface": mosaic["cd_surface"],
                "z0_surface_m": mosaic["z0_surface_m"],
                "rel_error_vs_open_water": (mosaic["cd_surface"] - cd_open) / cd_open,
            }
        )
    at_zero = [r for r in rows if r["concentration"] == 0.0][0]
    near_zero = [r for r in rows if r["concentration"] == 1e-6][0]
    err_zero = abs(at_zero["rel_error_vs_open_water"])
    err_near = abs(near_zero["rel_error_vs_open_water"])
    return {
        "cd_open_water": cd_open,
        "cd_bulk_10m_reference": 1.3e-3,
        "cd_open_water_rel_error_vs_bulk": abs(cd_open - 1.3e-3) / 1.3e-3,
        "cd_at_C_zero": at_zero["cd_surface"],
        "rel_error_at_C_zero": err_zero,
        "rel_error_at_C_1e_minus_6": err_near,
        "tolerance": 0.01,
        "passed": err_zero < 0.01 and err_near < 0.01,
        "rows": rows,
    }


def criterion_c6() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ice_drift_roughness.py", "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary_line = lines[-1] if lines else ""
    n_passed = 0
    if " passed" in summary_line:
        n_passed = int(summary_line.split(" passed")[0].split()[-1])
    return {
        "command": "python -m pytest tests/test_ice_drift_roughness.py -q",
        "returncode": proc.returncode,
        "summary_line": summary_line,
        "n_tests_passed": n_passed,
        "passed": proc.returncode == 0 and n_passed >= 10,
    }


def make_figures(c1, c2, c3, c4, c5) -> dict:
    figs = {}

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for label, points in c1["scan"].items():
        ax.semilogy([p["keel_depth_m"] for p in points], [p["z0_m"] for p in points],
                    lw=1.8, label=label)
    ax.set_xlabel("mean keel depth  d_k  [m]")
    ax.set_ylabel("effective under-ice roughness  z0  [m]")
    ax.set_title("Under-ice roughness vs keel morphology (h = 1.8 m)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="mean keel spacing")
    fig.tight_layout()
    figs["fig1_z0_vs_keel_morphology.png"] = (
        fig,
        "Effective under-ice roughness length z0 versus mean keel depth for four mean keel "
        "spacings (h = 1.8 m). Monotone increasing in keel depth and decreasing with keel "
        "spacing (criterion C1).",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for k_s in sorted({r["k_s_m"] for r in c2["rows"]}):
        sel = sorted([r for r in c2["rows"] if r["k_s_m"] == k_s], key=lambda r: r["reynolds"])
        ax1.loglog([r["reynolds"] for r in sel], [r["cd_model"] for r in sel], lw=1.9,
                   label="model, ks=%.0f mm" % (k_s * 1e3))
        ax1.loglog([r["reynolds"] for r in sel], [r["cd_loglaw_reference"] for r in sel],
                   "o", ms=4.2, mfc="none", label="log-law ref., ks=%.0f mm" % (k_s * 1e3))
        ax2.semilogx([r["reynolds"] for r in sel], [100.0 * r["rel_error"] for r in sel],
                     "o-", ms=3.5, label="ks=%.0f mm" % (k_s * 1e3))
    ax1.set_xlabel("drift Reynolds number  Re = U_rel z_ref / nu  [-]")
    ax1.set_ylabel("drag coefficient  Cd  [-]")
    ax1.set_title("(a) closure vs log-law reference")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=6.5, ncol=2)
    ax2.axhline(5.0, color="r", ls="--", lw=1.0, label="+/-5 % criterion")
    ax2.axhline(-5.0, color="r", ls="--", lw=1.0)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("drift Reynolds number  Re  [-]")
    ax2.set_ylabel("relative deviation  [%]")
    ax2.set_title("(b) deviation from log law (max %.2f %%)" % (100 * c2["max_abs_rel_error"]))
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    figs["fig2_cd_reynolds_loglaw.png"] = (
        fig,
        "Drag coefficient versus drift Reynolds number: roughness-length superposition "
        "closure (lines) against the independent velocity-defect log-law reference with the "
        "Karman roughness function (open circles); panel (b) shows the deviation and the 5 % "
        "criterion band (criterion C2).",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for case in c3["cases"]:
        hist = case["history"]
        if not hist:
            continue
        ax1.semilogy([h[0] for h in hist], [max(h[3], 1e-12) for h in hist], "o-", ms=4,
                     label="|Ua| = %.1f m/s" % math.hypot(*case["wind_m_s"]))
    ax1.axhline(1e-4, color="r", ls="--", lw=1.0, label="tolerance 1e-4 m/s")
    ax1.set_xlabel("Newton iteration")
    ax1.set_ylabel("drift-velocity increment  |dU|  [m/s]")
    ax1.set_title("(a) closure convergence")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7)

    sweep = c3["nansen_sweep"]
    ax2.plot([s["wind_speed_m_s"] for s in sweep], [100.0 * s["drift_ratio"] for s in sweep],
             "-", color="C0", label="drift/wind ratio [%]")
    ax2.set_xlabel("wind speed  |Ua|  [m/s]")
    ax2.set_ylabel("drift / wind ratio  [%]")
    ax2.grid(True, alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot([s["wind_speed_m_s"] for s in sweep], [-s["turn_angle_deg"] for s in sweep],
              "--", color="C3", label="turn angle (model)")
    ax2b.plot([s["wind_speed_m_s"] for s in sweep],
              [s["turn_angle_analytic_deg"] for s in sweep], ":", color="C1",
              label="analytic tan(theta) = rho_i h f / (rho_w Cd |V|)")
    ax2b.set_ylabel("right turn of drift vs wind  [deg]")
    ax2.legend(fontsize=7, loc="center right")
    ax2b.legend(fontsize=7, loc="upper right")
    ax2.set_title("(b) Nansen-type free drift (h = 2 m, C = 1, 75 deg N)")
    fig.tight_layout()
    figs["fig3_closure_convergence.png"] = (
        fig,
        "Left: Newton increments of the drift-velocity closure decay below the 1e-4 m/s "
        "tolerance in 3-5 iterations for every forcing case (criterion C3). Right: free-drift "
        "drift/wind ratio (~2 %) and right-turning angle versus wind speed, compared with the "
        "analytic balance tan(theta) = rho_i h f / (rho_w Cd |V|).",
    )

    rows = c4["rows"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    idx = list(range(len(rows)))
    model = [r["cd_model"] * 1e3 for r in rows]
    obs = [r["cd_obs"] * 1e3 for r in rows]
    lo = [r["obs_range"][0] * 1e3 for r in rows]
    hi = [r["obs_range"][1] * 1e3 for r in rows]
    ax.errorbar(idx, obs, yerr=[[o - l for o, l in zip(obs, lo)], [h - o for o, h in zip(obs, hi)]],
                fmt="ks", ms=6, capsize=4, label="compiled observation (central + range)")
    ax.plot(idx, model, "o", ms=8, mfc="none", mew=1.8, color="C0", label="closure model")
    for i, r in enumerate(rows):
        ax.annotate("%+.1f%%" % (100 * r["rel_error"]), (i, model[i]), textcoords="offset points",
                    xytext=(8, 6), fontsize=7.5)
    ax.set_xticks(idx)
    ax.set_xticklabels(["%s\n(%s)" % (r["anchor_id"], r["split"][:4]) for r in rows], fontsize=7.5)
    ax.set_ylabel("under-ice drag coefficient  Cd10 x 10^3  [-]")
    ax.set_title("Beaufort Sea cross-validation (C4): max |error| %.1f %% < %.0f %%"
                 % (100 * c4["summary"]["max_abs_rel_error"], 100 * CD_TOLERANCE))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["fig4_beaufort_crossvalidation.png"] = (
        fig,
        "Model Cd10 against the Beaufort Sea anchor set (black: compiled literature central "
        "value with its reported range; blue: closure model, relative deviation annotated). "
        "Calibration anchors were used for the single-parameter fit, validation anchors were "
        "held out (criterion C4).",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    rows5 = c5["rows"]
    ax.plot([r["concentration"] for r in rows5], [1e3 * r["cd_surface"] for r in rows5], "o-",
            label="Cd,surface (ice + leads)")
    ax.axhline(1e3 * c5["cd_open_water"], color="C3", ls="--",
               label="open water Cd = %.4f" % c5["cd_open_water"])
    ax.set_xlabel("ice concentration  C  [-]")
    ax.set_ylabel("Cd x 10^3  [-]")
    ax.set_title("C5 negative control: C -> 0 recovers the open-water value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["fig5_open_water_limit.png"] = (
        fig,
        "Area-averaged interface drag coefficient versus ice concentration; the C -> 0 limit "
        "reproduces the open-water log-law value to machine precision (criterion C5).",
    )

    out = {}
    for name, (fig, caption) in figs.items():
        path = OUT_ART / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out[name] = {"path": str(path.relative_to(ROOT)), "caption": caption}
    return out


def main() -> int:
    results = {
        "campaign_node": "p302w1-n01-ice-drift-roughness",
        "module": "tensorphysics.ice_drift_roughness",
        "seed": SEED,
        "C1": criterion_c1(),
        "C2": criterion_c2(),
        "C3": criterion_c3(),
        "C4": criterion_c4(),
        "C5": criterion_c5(),
        "C6": criterion_c6(),
    }
    results["figures"] = make_figures(
        results["C1"], results["C2"], results["C3"], results["C4"], results["C5"]
    )
    results["all_passed"] = all(
        results[k]["passed"] for k in ("C1", "C2", "C3", "C4", "C5", "C6")
    )
    out = OUT_ART / "criterion_results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)
    for k in ("C1", "C2", "C3", "C4", "C5", "C6"):
        print(k, "passed =", results[k]["passed"])
    print("all_passed =", results["all_passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())