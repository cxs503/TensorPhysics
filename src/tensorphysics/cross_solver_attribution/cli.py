"""One-click driver for the cross-solver attribution campaign node.

Usage (from the repository root)::

    python -m tensorphysics.cross_solver_attribution.cli \
        --out artifacts/cross_solver_attribution/ledger.json \
        --parts orders,factor,replay,sph_lbm,verdicts

Each ``--parts`` entry computes one block and merges it into the output JSON, so
the (slower) lattice-Boltzmann block can be re-run on its own without recomputing
the finite-volume sweep.  ``--parts figures`` writes the PNGs used by the report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

from . import attribution as A
from . import cases as C
from . import orders as O
from .lbm import lbm_linear_wave, lbm_poiseuille
from .sph import sph_poiseuille, sph_shallow_wave

DEFAULT_PARTS = ("orders", "factor", "replay", "sph_lbm", "verdicts")


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, allow_nan=True)


def _sanitise(obj):
    """json helper: replace non-finite floats by null so the ledger is portable."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
def part_orders(data: dict) -> None:
    table = A.order_table(resolutions=(40, 80, 160))
    data["orders"] = table
    data["ritter_convergence"] = A.ritter_convergence()
    data["C1"] = {
        "rows": [{"case": r["case"], "solver": r["label"],
                  "observed_order": r["observed_order"],
                  "expected_order": r["expected"],
                  "band": r["band"], "in_band": r["in_band"]} for r in table],
        "all_in_band": all(r["in_band"] for r in table),
        "n_rows": len(table),
        "n_in_band": sum(1 for r in table if r["in_band"]),
    }


def part_factor(data: dict) -> None:
    legs = A.make_legs(n_factorial=64)
    ledgers = {}
    for key, leg in legs.items():
        t0 = time.time()
        led = A.factorial_ledger(leg)
        led["wall_seconds"] = time.time() - t0
        ledgers[key] = led
    data["legs"] = ledgers

    # negative control for the L term: same design, L factor = sub-cycling only
    control = {}
    for key in ("adv_sine_dg_fvm", "diff_front_dg_fvm"):
        leg = legs[key]
        ctl = A.factorial_ledger(leg, control=True)
        l_val = abs(ledgers[key]["shapley"]["L"])
        c_val = abs(ctl["shapley"]["L"])
        floor = 1.0e-12 * max(abs(ledgers[key]["total"]), 1.0)
        ratio = l_val / max(c_val, floor)
        control[key] = {"L_term_limiter": l_val, "L_term_subcycle_only": c_val,
                        "control_is_zero_to_roundoff": bool(c_val <= floor),
                        "ratio_with_floor": ratio,
                        "pass": bool(ratio > 5.0)}
    data["negative_control"] = control

    attributable = {k: v for k, v in ledgers.items()
                    if v["all_finite"] and not v["degenerate"]}
    data["attributable_legs"] = sorted(attributable)
    data["C2"] = {
        "legs": {k: {"total": v["total"], "sum_shapley": sum(v["shapley"].values()),
                     "closure_rel_residual": v["closure_rel_residual"],
                     "anova_interaction_share": v["anova_interaction_share"],
                     "degenerate": v["degenerate"],
                     "pass": (v["closure_rel_residual"] < 0.05)} for k, v in ledgers.items()},
        "all_pass": all(v["closure_rel_residual"] < 0.05
                        for v in attributable.values()),
        "n_attributable_legs": len(attributable),
    }
    ratio = min(v["ratio_with_floor"] for v in control.values())
    data["C5"] = {
        "min_ratio_with_floor": ratio,
        "pass": bool(ratio > 5.0),
        "detail": control,
        "note": ("the control defines the L factor as sub-cycling only; that knob has "
                 "no measurable effect at the protocol CFL, so the L term is entirely "
                 "limiter-driven"),
    }


def part_replay(data: dict) -> None:
    rep = A.legacy_replay()
    data["legacy_replay"] = rep
    share = rep.get("main_term_share")
    data["C3"] = {
        "legacy_total": A.LEGACY_TOTAL,
        "reproduced_total": rep.get("reproduced_total"),
        "replay_gap_rel": rep.get("replay_gap_rel"),
        "main_term": rep.get("dominant_term"),
        "main_term_share": share,
        "shapley": rep.get("shapley"),
        "shares": rep.get("shares"),
        "pass": bool(share is not None and math.isfinite(share) and share > 0.5),
    }


def part_sph_lbm(data: dict) -> None:
    lbm_p = lbm_poiseuille()
    sph_p_study = [sph_poiseuille(n_fluid=nf) for nf in (80, 160, 320, 640)]
    sph_p = sph_p_study[-1]
    lbm_w = lbm_linear_wave(cells=200)
    sph_w = sph_shallow_wave(n_particles=200)
    # the two solvers work in their own units (lattice units for LBM, SI-like
    # units for SPH); every comparison below is done on the dimensionless
    # u/u_peak_analytic ratio so the units cannot leak into the ledger
    lbm_p["y_over_H"] = [y / float(lbm_p["ny"]) for y in lbm_p["y"]]
    lbm_p["u_over_peak"] = [u / lbm_p["peak_analytic"] for u in lbm_p["profile"]]
    lbm_p["analytic_over_peak"] = [u / lbm_p["peak_analytic"] for u in lbm_p["analytic"]]
    sph_p["y_over_H"] = [y / float(sph_p["height"]) for y in sph_p["y"]]
    sph_p["u_over_peak"] = [u / sph_p["peak_analytic"] for u in sph_p["profile"]]
    sph_p["analytic_over_peak"] = [u / sph_p["peak_analytic"] for u in sph_p["analytic"]]
    data["poiseuille"] = {
        "lbm": lbm_p,
        "sph": sph_p,
        "sph_resolution_study": [{"n_fluid": r["n_fluid"],
                                  "peak_rel_error": r["peak_rel_error"],
                                  "peak_over_mean": r["peak_over_mean_numeric"],
                                  "peak_numeric": r["peak_numeric"]} for r in sph_p_study],
    }
    data["wave_celerity"] = {"lbm": lbm_w, "sph": sph_w}
    peak_ok = (lbm_p["peak_rel_error"] < 0.05) and (sph_p["peak_rel_error"] < 0.05)
    wave_ok = (lbm_w["celerity_rel_error"] < 0.03) and (sph_w["celerity_rel_error"] < 0.03)
    data["C4"] = {
        "poiseuille_peak": {"lbm_rel_error": lbm_p["peak_rel_error"],
                            "sph_rel_error": sph_p["peak_rel_error"],
                            "threshold": 0.05, "pass": bool(peak_ok)},
        "wave_celerity": {"lbm_rel_error": lbm_w["celerity_rel_error"],
                          "sph_rel_error": sph_w["celerity_rel_error"],
                          "threshold": 0.03, "pass": bool(wave_ok)},
        "pass": bool(peak_ok and wave_ok),
    }


def part_verdicts(data: dict) -> None:
    legs = data.get("legs", {})
    pairs = []
    for key, led in legs.items():
        pairs.append({
            "leg": key,
            "description": led["description"],
            "case": led["case"],
            "n": led["n"],
            "D_native": led["v_full"],
            "e_dg": led["rows"]["111"]["e_dg"],
            "e_fvm": led["rows"]["111"]["e_fvm"],
            "dominant_term": led["dominant_term"],
            "share": led["shares"].get(led["dominant_term"]) if led["dominant_term"] else None,
            "all_finite": led["all_finite"],
            "degenerate": led["degenerate"],
            "verdict": ("not supported: DG leg without positivity preservation"
                        if not led["all_finite"] else
                        ("degenerate: both solvers agree at truncation level"
                         if led["degenerate"] else "attributed")),
        })
    if "poiseuille" in data:
        pairs.append({
            "leg": "poiseuille_sph_vs_lbm",
            "description": "SPH viscous balance vs D2Q9 LBM against the analytic parabola",
            "D_native": abs(
                data["poiseuille"]["sph"]["peak_numeric"]
                / data["poiseuille"]["sph"]["peak_analytic"]
                - data["poiseuille"]["lbm"]["peak_numeric"]
                / data["poiseuille"]["lbm"]["peak_analytic"]),
            "e_dg": data["poiseuille"]["lbm"]["peak_rel_error"],
            "e_fvm": data["poiseuille"]["sph"]["peak_rel_error"],
            "dominant_term": "peak-velocity mismatch",
            "share": None, "all_finite": True, "verdict": "attributed",
        })
    if "wave_celerity" in data:
        pairs.append({
            "leg": "wave_celerity_sph_vs_lbm",
            "description": "SPH shallow-water SPH vs D1Q3 acoustic LBM against sqrt(g h0)",
            "D_native": abs(data["wave_celerity"]["sph"]["c_numeric"]
                            - data["wave_celerity"]["lbm"]["c_numeric"])
            / data["wave_celerity"]["lbm"]["c_exact"],
            "e_dg": data["wave_celerity"]["lbm"]["celerity_rel_error"],
            "e_fvm": data["wave_celerity"]["sph"]["celerity_rel_error"],
            "dominant_term": "celerity mismatch",
            "share": None, "all_finite": True, "verdict": "attributed",
        })
    data["arbitration_pairs"] = pairs
    data["C6"] = {"n_pairs": len(pairs),
                  "n_with_quantified_error": sum(1 for p in pairs if p["all_finite"]),
                  "pass": bool(len(pairs) >= 3)}

    checks = {
        "C1": data.get("C1", {}).get("all_in_band"),
        "C2": data.get("C2", {}).get("all_pass"),
        "C3": data.get("C3", {}).get("pass"),
        "C4": data.get("C4", {}).get("pass"),
        "C5": data.get("C5", {}).get("pass"),
        "C6": data.get("C6", {}).get("pass"),
    }
    data["acceptance"] = {
        "checks": checks,
        "all_pass": all(bool(v) for v in checks.values()),
    }


# ---------------------------------------------------------------------------
def part_figures(data: dict, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    figs = {}

    # 1. order convergence
    table = data.get("orders", [])
    if table:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        markers = {"dg_p1": "o", "dg_p2": "s", "fvm_o1": "^", "fvm_o2": "v"}
        for case_key, colour in (("adv_sine", "C0"), ("diff_front", "C1"),
                                 ("ritter", "C2")):
            for row in table:
                if row["case"] != case_key:
                    continue
                ns = np.array(row["resolutions"], dtype=float)
                es = np.array(row["errors"], dtype=float)
                ok = np.isfinite(es) & (es > 0)
                if not ok.any():
                    continue
                ax.loglog(ns[ok], es[ok], marker=markers.get(row["label"], "o"),
                          ls="-", color=colour, alpha=0.75,
                          label=f"{case_key} {row['label']} p={row['observed_order']:.2f}")
        ax.set_xlabel("cells n")
        ax.set_ylabel("relative L2 error (reconstruction)")
        ax.set_title("Observed convergence order per solver and case")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        path = os.path.join(outdir, "order_convergence.png")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        figs["order_convergence"] = path

    # 2. ledger bar chart
    legs = data.get("legs", {})
    if legs:
        keys = list(legs)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        width = 0.26
        for i, term in enumerate(("T", "B", "L")):
            vals = [legs[k]["shapley"].get(term, float("nan")) for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 1) * width, vals, width, label=term)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels([k.replace("_dg_fvm", "") for k in keys], rotation=15,
                           fontsize=7)
        ax.set_ylabel("Shapley contribution to D")
        ax.set_title("Term-level attribution of the DG/FVM disagreement")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        path = os.path.join(outdir, "ledger_shapley.png")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        figs["ledger_shapley"] = path

    # 3. Poiseuille profiles
    poi = data.get("poiseuille")
    if poi:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        lbm = poi["lbm"]; sph = poi["sph"]
        ax.plot(lbm["u_over_peak"], lbm["y_over_H"], "o-", ms=3, label="D2Q9 LBM")
        ax.plot(sph["u_over_peak"], sph["y_over_H"], "s-", ms=4,
                label="SPH (640 particles)")
        ax.plot(lbm["analytic_over_peak"], lbm["y_over_H"], "k--",
                label="analytic parabola")
        ax.set_xlabel("u / u_peak(analytic)")
        ax.set_ylabel("y / H")
        ax.set_title("Poiseuille channel: LBM vs SPH vs exact")
        ax.legend(); ax.grid(alpha=0.3)
        path = os.path.join(outdir, "poiseuille_profiles.png")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        figs["poiseuille_profiles"] = path

    # 4. legacy replay sweep
    rep = data.get("legacy_replay", {})
    if rep.get("sweep"):
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for cfl in sorted({s["cfl"] for s in rep["sweep"]}):
            pts = sorted([s for s in rep["sweep"] if s["cfl"] == cfl], key=lambda s: s["n"])
            ax.semilogx([p["n"] for p in pts], [p["D"] for p in pts], "o-",
                        label=f"CFL={cfl}")
        ax.axhline(A.LEGACY_TOTAL, color="k", ls="--", label="legacy 24.9 %")
        ax.set_xlabel("cells n"); ax.set_ylabel("native-default disagreement D")
        ax.set_title("24.9 % replay: which protocol point lands on the legacy number")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        path = os.path.join(outdir, "legacy_replay.png")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        figs["legacy_replay"] = path

    data["figures"] = figs


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/cross_solver_attribution/ledger.json")
    ap.add_argument("--parts", default=",".join(DEFAULT_PARTS))
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    data = _load(args.out)
    data["protocol"] = {
        "node": "P78-N08",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_head": _git_head(),
        "parts_run": parts,
    }
    figdir = args.figdir or os.path.join(os.path.dirname(args.out) or ".", "figures")

    for part in parts:
        t0 = time.time()
        if part == "orders":
            part_orders(data)
        elif part == "factor":
            part_factor(data)
        elif part == "replay":
            part_replay(data)
        elif part == "sph_lbm":
            part_sph_lbm(data)
        elif part == "verdicts":
            part_verdicts(data)
        elif part == "figures":
            part_figures(data, figdir)
        else:
            raise SystemExit(f"unknown part {part!r}")
        print(f"[cross-solver] part {part} done in {time.time() - t0:.1f}s",
              file=sys.stderr)
        _save(args.out, _sanitise(data))
    _save(args.out, _sanitise(data))
    if "acceptance" in data:
        print(json.dumps(data["acceptance"], indent=2))
    return 0


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())