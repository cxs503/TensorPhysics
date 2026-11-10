#!/usr/bin/env python
"""Generate the grounding-damage artefacts: verification table, case results,
parameter sweep, and figures.

Usage::

    python scripts/generate_grounding_damage_artifacts.py [--skip-fem]

Everything written by this script is programmatically injected into
``artifacts/grounding_damage/`` (JSON + PNG) so that no number in the report is
hand copied.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.grounding_damage import (  # noqa: E402
    DEFAULT_PANEL, GroundingCase, design_panel, parameter_sweep, run_grounding_case,
)
from tensorphysics.grounding_damage import verification as V  # noqa: E402
from tensorphysics.grounding_damage.contact import HertzContact, hertz_sphere_on_plane  # noqa: E402

OUT = ROOT / "artifacts" / "grounding_damage"
OUT.mkdir(parents=True, exist_ok=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def main(skip_fem: bool = False) -> None:
    t_start = time.time()
    results = {}

    # ---------------------------------------------------------------- panel
    t0 = time.time()
    response = design_panel(DEFAULT_PANEL, delta_max=0.35, steps=12, n_x=21, n_y=21,
                            w_modes=3, uv_modes=2, n_layers=3)
    results["panel_curve"] = {
        "seconds": time.time() - t0,
        "spec": {"a": response.spec.a, "b": response.spec.b, "t": response.spec.t,
                 "sigma_y": response.spec.grade.sigma_y,
                 "eps_f": response.spec.grade.eps_f, "bc": response.spec.bc},
        "rows": [{"delta": float(d), "pressure": float(q), "eps_pl": float(e),
                  "torn_fraction": float(f), "energy": float(en), "volume": float(v)}
                 for d, q, e, f, en, v in zip(response.deflection, response.pressure,
                                             response.eps_pl_max, response.torn_fraction,
                                             response.energy, response.volume)],
        "tes": float(np.interp(response.spec.grade.eps_f,
                               response.eps_pl_max, response.deflection))
        if response.eps_pl_max[-1] >= response.spec.grade.eps_f else None,
    }

    # --------------------------------------------------------- verification
    results["material"] = V.material_verification(n=200)
    results["flooding"] = V.flooding_verification()
    results["stability"] = V.stability_verification()
    results["panel_linear"] = V.panel_linear_verification()
    print("verification done", flush=True)
    results["hertz"] = V.hertz_verification()
    results["stiffener"] = V.stiffener_verification()
    results["negative_controls"] = V.negative_controls(response)

    fem_path = OUT / "fem_reference.json"
    if not skip_fem:
        t0 = time.time()
        fem = V.panel_fem_reference(nx=14, ny=4, max_steps=18, immovable=True, n_layers=4)
        np.savez(OUT / "fem_reference.npz", **{k: np.asarray(v) for k, v in fem.items()})
        json.dump(jsonable({k: np.asarray(v) for k, v in fem.items()}),
                  open(fem_path, "w"), indent=1)
        results["fem_seconds"] = time.time() - t0
    if fem_path.exists():
        data = json.load(open(fem_path))
        fem = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        results["panel_fem"] = V.panel_fem_comparison(response, fem,
                                                      loads=(5e5, 1e6, 2e6))
        results["fem_reference"] = {"pressure": fem["pressure"].tolist(),
                                    "w_max": fem["w_max"].tolist()}

    # ------------------------------------------------------------ case (C7)
    case = GroundingCase(velocity_kn=3.0, ridge_height=4.0, ridge_slope_deg=12.0)
    results["case_c7"] = run_grounding_case(case, response)
    results["gz_by_speed"] = {
        str(v): run_grounding_case(GroundingCase(velocity_kn=v), response)["stability"]["gz"]
        for v in (1.0, 3.0, 5.0)
    }

    # ---------------------------------------------------------- parameter scan
    t0 = time.time()
    rows = parameter_sweep(response,
                           velocities=(1.0, 3.0, 5.0),
                           ridge_heights=(2.0, 4.0, 6.0),
                           slopes=(5.0, 12.0, 20.0))
    results["sweep"] = {"seconds": time.time() - t0, "rows": rows}

    results["meta"] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()),
                       "total_seconds": time.time() - t_start}
    json.dump(jsonable(results), open(OUT / "results.json", "w"), indent=1)

    # ------------------------------------------------------------- figures
    make_figures(response, results)
    print(f"artefacts written to {OUT} in {time.time() - t_start:.1f} s")


def make_figures(response, results) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. panel resistance curve -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    d = np.asarray(results["panel_curve"]["rows"])
    ax[0].plot([r["delta"] for r in d], [r["pressure"] / 1e6 for r in d], "o-",
               label="Ritz panel (elastoplastic)")
    if "fem_reference" in results:
        ax[0].plot(results["fem_reference"]["w_max"],
                   np.asarray(results["fem_reference"]["pressure"]) / 1e6, "s--",
                   label="FE reference (Q4+Hermite)")
    ax[0].set_xlabel("centre deflection $\\delta$ (m)")
    ax[0].set_ylabel("contact pressure $q$ (MPa)")
    ax[0].set_title("Bottom panel resistance curve")
    ax[0].grid(alpha=.3)
    ax[0].legend()
    ax[1].plot([r["delta"] for r in d], [r["eps_pl"] for r in d], "o-", color="C3")
    ax[1].axhline(results["panel_curve"]["spec"]["eps_f"], color="k", ls=":",
                  label="$\\varepsilon_f$")
    ax[1].set_xlabel("centre deflection (m)")
    ax[1].set_ylabel("peak equivalent plastic strain $\\varepsilon_{pl}$")
    ax[1].set_title("Tearing criterion")
    ax[1].grid(alpha=.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "panel_curve.png", dpi=130)
    plt.close(fig)

    # 2. Hertz verification -----------------------------------------------------
    R, E, nu = 0.5, 2.06e11, 0.3
    solver = HertzContact(R, E, nu, n_rings=80)
    sol = solver.solve(1e-3)
    ref = hertz_sphere_on_plane(sol["load"], R, E, nu)
    rr = np.linspace(0, 1.05 * ref.radius, 200)
    p_hertz = ref.peak_pressure * np.sqrt(np.maximum(1 - (rr / ref.radius) ** 2, 0))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(sol["r"] / ref.radius, sol["pressure"] / 1e6, "o", ms=3,
               label="half-space contact solver")
    ax[0].plot(rr / ref.radius, p_hertz / 1e6, "k-", label="Hertz (closed form)")
    ax[0].set_xlim(0, 1.6)
    ax[0].set_xlabel("$r/a_{Hertz}$")
    ax[0].set_ylabel("contact pressure (MPa)")
    ax[0].set_title("C1: contact pressure, $\\delta$ = 1 mm")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    rows = results["hertz"]["rows"]
    ax[1].plot([r["delta"] for r in rows], [r["radius_error_pct"] for r in rows], "o-",
               label="contact radius")
    ax[1].plot([r["delta"] for r in rows], [r["p0_error_pct"] for r in rows], "s-",
               label="peak pressure")
    ax[1].axhline(10, color="r", ls="--", lw=1, label="10 % acceptance")
    ax[1].axhline(-10, color="r", ls="--", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("approach $\\delta$ (m)")
    ax[1].set_ylabel("error vs Hertz (%)")
    ax[1].set_title("C1 verification")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "hertz.png", dpi=130)
    plt.close(fig)

    # 3. damage profile of the C7 case ------------------------------------------
    case = results["case_c7"]
    prof = case["contact"]["penetration_profile"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].fill_between(np.asarray(prof["y"]), 0,
                       np.asarray(prof["pressure"]) / 1e6, step="mid", alpha=.4,
                       color="C0")
    ax[0].plot(np.asarray(prof["y"]), np.asarray(prof["delta"]) * 1e3, "k-",
               label="penetration (mm)")
    ax[0].set_xlabel("lateral position $y$ (m)")
    ax[0].set_ylabel("contact pressure (MPa) / penetration (mm)")
    ax[0].set_title("Ridge contact pressure across the bottom (C7)")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    rows = results["sweep"]["rows"]
    h = 4.0
    for alpha, marker in ((5.0, "o"), (12.0, "s"), (20.0, "^")):
        sel = [r for r in rows if r["ridge_height"] == h and r["slope_deg"] == alpha]
        ax[1].plot([r["velocity_kn"] for r in sel], [r["opening_area"] for r in sel],
                   marker + "-", label=f"$\\alpha$ = {alpha:.0f}$^\\circ$")
    ax[1].set_xlabel("grounding speed (kn)")
    ax[1].set_ylabel("damage opening area $A_{damage}$ (m$^2$)")
    ax[1].set_title("Opening area vs speed ($h_{ridge}$ = 4 m)")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "damage.png", dpi=130)
    plt.close(fig)

    # 4. residual GZ curves -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for vkey, gz in results.get("gz_by_speed", {}).items():
        ax[0].plot(gz["angle_deg"], gz["gz"], "o-", label=f"V = {float(vkey):.0f} kn")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xlabel("heel angle (deg)")
    ax[0].set_ylabel("GZ (m)")
    ax[0].set_title("Residual GZ curve after flooding")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    # cumulative area under GZ
    res = results["case_c7"]
    gz = res["stability"]["gz"]
    ang = np.deg2rad(np.asarray(gz["angle_deg"]))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (np.asarray(gz["gz"])[1:] +
                                                  np.asarray(gz["gz"])[:-1]) * np.diff(ang))])
    ax[1].plot(np.rad2deg(ang), cum, "o-", label="C7 case (3 kn)")
    ax[1].axhline(0.055, color="r", ls="--", label="IMO 0.055 m rad at 30$^\\circ$")
    ax[1].axvline(30, color="gray", ls=":")
    ax[1].set_xlabel("heel angle (deg)")
    ax[1].set_ylabel("area under GZ (m rad)")
    ax[1].set_title("C5: damage-stability area criterion")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "stability.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fem", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    if args.figures_only:
        results = json.load(open(OUT / "results.json"))
        make_figures(None, results)
        print("figures regenerated")
    else:
        main(skip_fem=args.skip_fem)