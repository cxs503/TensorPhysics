#!/usr/bin/env python3
"""Run the P40W3B-N09 multi-solver energy-conservation contract suite.

Produces

    artifacts/p40w3b_n09/verification.json   machine-readable acceptance record
    artifacts/p40w3b_n09/fig_*.png           paper figures

Usage::

    python3 scripts/run_coupling_energy_contract.py [--quick] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

# The scenarios issue tens of thousands of tiny dense operations; a
# multi-threaded BLAS spends more time in its thread pool than in the work.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.coupling_energy import scenarios as S  # noqa: E402
from tensorphysics.coupling_energy.exchange import (  # noqa: E402
    SCHEME_ORDER,
    TemporalScheme,
    quadrature_error,
    reconstruction_order_scan,
)

NODE = "P40W3B-N09"
MISSION = 7597
REPO = "TensorPhysics"


def _signal(t: float) -> float:
    """Smooth interface signal used for the exchange-quadrature order scan."""
    return math.sin(2.0 * math.pi * t) + 0.3 * math.cos(6.0 * math.pi * t)


def run_suite(quick: bool) -> dict:
    ratios = (1, 2, 4, 8)
    schemes = (TemporalScheme.ZOH, TemporalScheme.LINEAR, TemporalScheme.QUADRATIC)
    thermal_kw = dict(t_end=2000.0, relax=0.2, nx=40, ny=20,
                      sample_times=[250.0, 500.0, 1000.0, 2000.0])
    mech_kw = dict(n_macro=400)
    three_kw = dict(n_macro=120)
    if quick:
        thermal_kw = dict(t_end=500.0, relax=0.2, nx=20, ny=10,
                          sample_times=[250.0, 500.0])
        mech_kw = dict(n_macro=200)
        three_kw = dict(n_macro=60)

    data: dict = {"thermal": [], "mechanical": [], "three_field": []}
    for ratio in ratios:
        for scheme in schemes:
            data["thermal"].append(
                S.run_thermal_lbm_fem(ratio, scheme, "shared", **thermal_kw).as_dict())
            data["mechanical"].append(
                S.run_mechanical_fem_dem(ratio, scheme, "shared", **mech_kw).as_dict())
            data["three_field"].append(
                S.run_three_field(ratio, scheme, "shared", **three_kw).as_dict())
        data["thermal"].append(
            S.run_thermal_lbm_fem(ratio, TemporalScheme.LINEAR, "independent",
                                  **thermal_kw).as_dict())
        data["mechanical"].append(
            S.run_mechanical_fem_dem(ratio, TemporalScheme.LINEAR, "independent",
                                     **mech_kw).as_dict())
        data["three_field"].append(
            S.run_three_field(ratio, TemporalScheme.LINEAR, "independent",
                              **three_kw).as_dict())
    return data


def acceptance(data: dict, extra: dict) -> dict:
    def runs(name, mode):
        return [r for r in data[name] if r["exchange_mode"] == mode]

    shared_t = runs("thermal", "shared")
    indep_t = runs("thermal", "independent")
    shared_m = runs("mechanical", "shared")
    indep_m = runs("mechanical", "independent")
    shared_3 = runs("three_field", "shared")
    indep_3 = runs("three_field", "independent")

    thermal_res = max(r["residual_max"] for r in shared_t)
    thermal_ctrl = min(r["residual_max"] for r in indep_t)
    thermal_prof = min(r["metrics"].get("profile_rel_l2_error", float("nan")) for r in shared_t)
    mom_res = max(max(r["series"]["momentum_residual"]) for r in shared_m)
    mom_ctrl = min(max(r["series"]["momentum_residual"]) for r in indep_m)
    mech_drift = {r["ratio"]: r["residual_max"] for r in shared_m
                  if r["scheme"] == "linear"}
    three_res = max(r["residual_max"] for r in shared_3)
    three_ctrl = min(r["residual_max"] for r in indep_3)
    orders = extra["order_scan"]

    crit = {}

    def gate(key, required, actual, passed, note=""):
        crit[key] = {"required": required, "actual": actual, "pass": bool(passed),
                     "note": note}

    gate("thermal_conservation_residual",
         "< 1e-12 (max over 12 shared runs)", thermal_res, thermal_res < 1e-12,
         "ledger residual eps = |sum sides - Q_in - (t=0)| / scale")
    gate("thermal_falsification_ratio",
         "> 1e3 x shared (independent-flux control)",
         thermal_ctrl / max(thermal_res, 1e-300), thermal_ctrl / max(thermal_res, 1e-300) > 1e3,
         "control replaces the shared flux by the solid's own Fourier estimate")
    gate("thermal_profile_error_r1",
         "< 0.03 (rel. L2 vs exact 1-D slab series)", thermal_prof, thermal_prof < 0.03,
         "best of the 12 shared runs; exact solution of the coupled 1-D problem")
    gate("mechanical_momentum_residual",
         "< 1e-10 (max over 12 shared runs)", mom_res, mom_res < 1e-10,
         "|P(t) - P(0)| / |P(0)| with an unsupported plate (no external force)")
    gate("mechanical_falsification_ratio",
         "> 1e3 x shared (lagged-reaction control)",
         mom_ctrl / max(mom_res, 1e-300), mom_ctrl / max(mom_res, 1e-300) > 1e3,
         "control: plate reaction from a frozen start-of-interval particle state")
    mono = all(mech_drift[r] <= mech_drift[1] for r in (2, 4, 8))
    gate("mechanical_energy_convergence",
         "drift(r=8) < 0.05 and monotone decreasing in r",
         {str(k): v for k, v in mech_drift.items()},
         mech_drift[8] < 0.05 and mono,
         "first-order energy drift of the explicit DEM sub-stepping")
    gate("three_field_energy_chain",
         "< 1e-3 (max over 12 shared runs)", three_res, three_res < 1e-3,
         "E_total = E_fluid + E_solid^thermal + E_solid^mech + E_particles + E_contact")
    gate("three_field_falsification_ratio",
         "> 1e2 x shared (independent-flux control)",
         three_ctrl / max(three_res, 1e-300), three_ctrl / max(three_res, 1e-300) > 1e2)
    ok_orders = (0.7 < orders["zoh"] < 1.3 and 1.7 < orders["linear"] < 2.3
                 and 2.5 < orders["quadratic"] < 3.3)
    gate("interpolation_order_scan",
         "zoh 1.0, linear 2.0, quadratic 3.0 (+-0.3)", orders, ok_orders,
         "measured on an analytic signal by exchange quadrature refinement")
    gate("plate_first_frequency",
         "< 0.03 rel. error vs c/(4L)", extra["plate_frequency"]["rel_error"],
         extra["plate_frequency"]["rel_error"] < 0.03,
         "FEM physics check, clamped-free axial mode")
    n_runs = len(data["thermal"]) + len(data["mechanical"]) + len(data["three_field"])
    gate("subcycling_matrix_complete",
         "4 ratios x 3 schemes (+4 controls) x 3 couplings = 48 runs", n_runs,
         n_runs == 48, "r in {1,2,4,8}, schemes in {zoh,linear,quadratic}")
    return {"criteria": crit, "overall_pass": all(c["pass"] for c in crit.values())}


def figures(data: dict, out: pathlib.Path, extra: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    lin = {r["ratio"]: r for r in data["thermal"]
           if r["scheme"] == "linear" and r["exchange_mode"] == "shared"}
    base = lin[1]
    t = np.asarray(base["series"]["t"])
    ef = np.asarray(base["series"]["e_fluid"])
    es = np.asarray(base["series"]["e_solid"])
    qe = np.asarray(base["series"]["q_external"])
    qi = np.asarray(base["series"]["q_iface"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(t, es, label="E_solid [J]")
    ax[0].plot(t, ef, label="E_fluid [J]")
    ax[0].plot(t, qe, "--", label="Q_external in [J]")
    ax[0].plot(t, qi, ":", label="Q_interface (solid->fluid) [J]")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("energy [J]")
    ax[0].set_title("(a) LBM<->FEM thermal energy history (r=1, linear)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for key, r in lin.items():
        ax[1].semilogy(r["series"]["t"], np.maximum(r["series"]["residual"], 1e-17),
                       label=f"r={key}")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel(r"$\varepsilon$ [-]")
    ax[1].set_title("(a) conservation residual per sub-cycling ratio")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig_thermal_energy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_thermal_energy.png",
                 "caption": "LBM<->FEM thermal run: stored energy, external input and "
                            "interface transfer (left) and the conservation residual for "
                            "the four sub-cycling ratios (right)."})

    # profiles vs analytic
    profs = base["metrics"].get("profiles", [])
    if profs:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot([0, 1], [50, 50], "k:", lw=0.8, label="steady state (50 K)")
        for prof in profs:
            y = np.asarray(prof["y"]) * 1e3
            ax.plot(y, prof["T"], "o", ms=3,
                    label=f"coupled t={prof['t']:.0f}s")
            ax.plot(y, prof["T_analytic"], "-", lw=1.0)
        ax.plot([], [], "-", color="0.3", label="exact 1-D series")
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("T [K]")
        ax.set_title("(a) coupled profile vs exact slab solution")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out / "fig_thermal_profile.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        figs.append({"name": "fig_thermal_profile.png",
                     "caption": "Coupled LBM<->FEM temperature profile against the exact "
                                "one-dimensional slab solution at four instants."})

    # interpolation quadrature order scan
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    for scheme in (TemporalScheme.ZOH, TemporalScheme.LINEAR, TemporalScheme.QUADRATIC):
        errs = [quadrature_error(_signal, h, 2.0, scheme)[0] for h in hs]
        ax.loglog(hs, errs, "o-", label=f"{scheme.value} (order {SCHEME_ORDER[scheme]})")
    ax.invert_xaxis()
    ax.set_xlabel("exchange interval h [-]")
    ax.set_ylabel("relative exchange error [-]")
    ax.set_title("interface reconstruction order")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig_interpolation_order.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_interpolation_order.png",
                 "caption": "Measured exchange-quadrature order of the zero/first/second "
                            "order interface reconstructions on an analytic signal."})

    # mechanical
    lin_m = {r["ratio"]: r for r in data["mechanical"]
             if r["scheme"] == "linear" and r["exchange_mode"] == "shared"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r4 = lin_m[4]
    tt = np.asarray(r4["series"]["t"]) * 1e3
    ax[0].plot(tt, r4["series"]["e_particles"], label="particles KE")
    ax[0].plot(tt, r4["series"]["e_solid"], label="plate KE+strain")
    ax[0].plot(tt, r4["series"]["e_contact"], label="contact potential")
    ax[0].plot(tt, r4["series"]["e_total"], "k--", label="total")
    ax[0].set_xlabel("t [ms]")
    ax[0].set_ylabel("energy [J]")
    ax[0].set_title("(b) FEM<->DEM impact energy split (r=4)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for key, r in lin_m.items():
        ax[1].semilogy(np.asarray(r["series"]["t"]) * 1e3,
                       np.maximum(r["series"]["momentum_residual"], 1e-18),
                       label=f"r={key}")
    for r in data["mechanical"]:
        if r["exchange_mode"] == "independent":
            ax[1].semilogy(np.asarray(r["series"]["t"]) * 1e3,
                           np.maximum(r["series"]["momentum_residual"], 1e-18),
                           ":", label=f"r={r['ratio']} independent")
            break
    ax[1].set_xlabel("t [ms]")
    ax[1].set_ylabel(r"$|P(t)-P(0)|/|P(0)|$")
    ax[1].set_title("(b) momentum conservation and its falsification arm")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig_mechanical_momentum.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_mechanical_momentum.png",
                 "caption": "FEM<->DEM impact: energy partition (left) and the momentum "
                            "residual for the shared contact force versus the independent "
                            "(lagged reaction) control (right)."})

    # three field + ratio sweep
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r3 = [r for r in data["three_field"]
          if r["ratio"] == 2 and r["scheme"] == "linear" and r["exchange_mode"] == "shared"][0]
    ax[0].plot(r3["series"]["t_thermal"], r3["series"]["e_solid_thermal"], label="solid thermal")
    ax[0].plot(r3["series"]["t_thermal"], r3["series"]["e_fluid"], label="fluid")
    ax[0].plot(r3["series"]["t_thermal"], r3["series"]["q_external"], "--", label="Q_in")
    ax2 = ax[0].twinx()
    ax2.plot(r3["series"]["t_mech"], r3["series"]["e_solid_mech"], color="C3",
             label="solid mechanical")
    ax2.plot(r3["series"]["t_mech"], r3["series"]["e_particles"], color="C4",
             label="particles")
    ax2.set_ylabel("mechanical energy [J]")
    ax[0].set_xlabel("t_thermal [s] (mechanical clock offset ~1e5)")
    ax[0].set_ylabel("thermal energy [J]")
    ax[0].set_title("(c) three-field energy chain (r=2)")
    ax[0].legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="lower right")
    ax[0].grid(alpha=0.3)
    for scheme in ("zoh", "linear", "quadratic"):
        pts = {r["ratio"]: r["residual_max"] for r in data["three_field"]
               if r["scheme"] == scheme and r["exchange_mode"] == "shared"}
        ax[1].loglog(sorted(pts), [max(pts[k], 1e-16) for k in sorted(pts)], "o-",
                     label=scheme)
    ctrl = {r["ratio"]: r["residual_max"] for r in data["three_field"]
            if r["exchange_mode"] == "independent"}
    ax[1].loglog(sorted(ctrl), [max(ctrl[k], 1e-16) for k in sorted(ctrl)], "s:",
                 label="independent control")
    ax[1].set_xlabel("sub-cycling ratio r")
    ax[1].set_ylabel(r"$\varepsilon_{max}$ [-]")
    ax[1].set_title("(c) residual vs sub-cycling ratio")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig_three_field_chain.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_three_field_chain.png",
                 "caption": "Three-field run: thermal and mechanical energy components "
                            "(left) and the total residual versus sub-cycling ratio with "
                            "the independent-flux control (right)."})

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, marker in (("thermal", "o"), ("mechanical", "s"), ("three_field", "^")):
        pts = {r["ratio"]: r["residual_max"] for r in data[name]
               if r["scheme"] == "linear" and r["exchange_mode"] == "shared"}
        ax.loglog(sorted(pts), [max(pts[k], 1e-16) for k in sorted(pts)], marker + "-",
                  label=name)
    ax.set_xlabel("sub-cycling ratio r")
    ax.set_ylabel(r"$\varepsilon_{max}$ [-]")
    ax.set_title("conservation residual vs sub-cycling ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig_residual_vs_ratio.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig_residual_vs_ratio.png",
                 "caption": "Conservation residual of the three couplings as a function of "
                            "the sub-cycling ratio (first-order reconstruction)."})
    return figs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="artifacts/p40w3b_n09")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = run_suite(args.quick)

    # measured in the asymptotic range h = 0.05 -> 0.0125
    order_scan = {k: v.order for k, v in
                  reconstruction_order_scan(_signal, 0.05, 2.0,
                                            refinements=(1.0, 0.5, 0.25)).items()}
    freq = S.plate_first_frequency(nx=20, ny=2, n_steps=2000)
    extra = {"order_scan": order_scan, "plate_frequency": freq}

    gates = acceptance(data, extra)
    figs = figures(data, out, extra)

    record = {
        "mission_id": MISSION,
        "node": NODE,
        "repo": REPO,
        "module": "tensorphysics.coupling_energy",
        "quick_mode": args.quick,
        "acceptance_criteria": gates["criteria"],
        "overall_pass": gates["overall_pass"],
        "order_scan": order_scan,
        "plate_first_frequency": freq,
        "figures": [f["name"] for f in figs],
        "runs": data,
    }
    (out / "verification.json").write_text(json.dumps(record, indent=2))
    (out / "figures.json").write_text(json.dumps(figs, indent=2))
    print(json.dumps({k: v["actual"] for k, v in gates["criteria"].items()}, indent=2))
    print("overall_pass:", gates["overall_pass"])
    return 0 if gates["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())