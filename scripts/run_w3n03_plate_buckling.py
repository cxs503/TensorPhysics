#!/usr/bin/env python3
"""W3-N03 verification campaign: ice-loaded hull plate bucking / ultimate strength.

Runs the machine-checkable gates and writes machine-readable artefacts plus the
figures used by the report:

* gate (a) linear buckling vs an independent Rayleigh-Ritz reference for four
  boundary-condition combinations;
* gate (b) nonlinear ultimate strength (arc-length limit point) vs the analytic
  von Karman / Faulkner effective-width benchmark for six slenderness values;
* imperfection sensitivity (slight / average / severe welding distortion);
* IACS UR I ice patch + lateral water pressure ultimate-limit-state check;
* gate (c) pytest outcome (read from the pytest log if available).

All numbers in the JSON are computed by the code in this repository -- nothing
is typed by hand into the report.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorphysics.plate_buckling import PlateGeometry, PlateModel
from tensorphysics.plate_buckling.benchmarks import (
    euler_buckling_stress,
    faulkner_ultimate,
    ritz_buckling_stress,
)
from tensorphysics.plate_buckling.imperfection import (
    ImperfectionSpec,
    apply_imperfection,
    deflection_amplitude,
)
from tensorphysics.plate_buckling.loads import IcePatchLoad, ice_patch_pressure
from tensorphysics.plate_buckling.uls_check import uls_check

SIGMA_Y = 313.0e6
E = 206.0e9
NU = 0.3
BETA_COEF = float(np.sqrt(SIGMA_Y / E))     # t = b * BETA_COEF / beta
OUT = ROOT / "artifacts" / "plate_buckling"
FIG = OUT / "figures"


def thickness_for(beta: float, b: float = 1.0) -> float:
    return b * BETA_COEF / beta


def gate_a(mesh=(24, 6), a=4.0):
    """Linear buckling vs independent Rayleigh-Ritz reference (4 BCs)."""
    geom = PlateGeometry(a=a, b=1.0, t=0.02, sigma_y=SIGMA_Y, nx=mesh[0], ny=mesh[1])
    rows = []
    for bc in ("SSSS", "CCCC", "SCSC", "CSCS"):
        model = PlateModel(geom, bc)
        res = model.linear_buckling(1.0)
        k_fem = res.sigma_cr * geom.t * geom.b**2 / (np.pi**2 * geom.flexural_rigidity)
        ritz = ritz_buckling_stress(geom, bc, n_terms=8)
        euler = euler_buckling_stress(geom, bc)
        rows.append({
            "bc": bc,
            "sigma_cr_fem_MPa": res.sigma_cr / 1e6,
            "k_fem": float(k_fem),
            "k_ritz": ritz["k"],
            "rel_err_vs_ritz": float(k_fem / ritz["k"] - 1.0),
            "k_euler_textbook": euler["k"],
            "rel_err_vs_euler": float(k_fem / euler["k"] - 1.0),
        })
    return geom, rows


def ultimate_strength(betas=(1.0, 1.25, 1.5, 2.0, 2.5, 3.0), mesh=(8, 4),
                      level="average", sigma_r_ratio=0.0, ds0=6e-4,
                      max_steps=14):
    out = []
    for beta in betas:
        geom = PlateGeometry(a=3.0, b=1.0, t=thickness_for(beta), sigma_y=SIGMA_Y,
                             nx=mesh[0], ny=mesh[1])
        model = PlateModel(geom, "SSSS")
        spec = ImperfectionSpec.standard(geom, level=level,
                                         sigma_r_ratio=sigma_r_ratio)
        apply_imperfection(model, spec)
        t0 = time.time()
        res = model.solve_arc_length(model.compression_load(SIGMA_Y),
                                     max_steps=max_steps, sigma_ref=SIGMA_Y,
                                     ds0=ds0)
        ref = faulkner_ultimate(beta)
        out.append({
            "beta": float(beta),
            "t_mm": geom.t * 1e3,
            "sigma_cr_MPa": model.linear_buckling(1.0).sigma_cr / 1e6,
            "sigma_u_over_sigma_y": float(res["lam_ultimate"]),
            "sigma_u_MPa": float(res["lambda"] * 0 if False else
                                 res["lam_ultimate"] * SIGMA_Y / 1e6),
            "reference_faulkner": ref,
            "rel_err": float(res["lam_ultimate"] / ref - 1.0),
            "n_steps": int(res["n_steps"]),
            "peak_vm_MPa": float(res["peak"]["sigma_vm_max"]) / 1e6,
            "w_peak_mm": float(res["peak"]["w_monitor"]) * 1e3,
            "runtime_s": float(time.time() - t0),
            "_path": [(p["lam"], p["w_monitor"] * 1e3) for p in res["path"]],
        })
    return out


def imperfection_sensitivity(beta=2.0, mesh=(8, 4)):
    rows = []
    paths = {}
    for level in ("slight", "average", "severe"):
        geom = PlateGeometry(a=3.0, b=1.0, t=thickness_for(beta), sigma_y=SIGMA_Y,
                             nx=mesh[0], ny=mesh[1])
        model = PlateModel(geom, "SSSS")
        spec = ImperfectionSpec.standard(geom, level=level, sigma_r_ratio=0.0)
        apply_imperfection(model, spec)
        res = model.solve_arc_length(model.compression_load(SIGMA_Y),
                                     max_steps=14, sigma_ref=SIGMA_Y, ds0=6e-4)
        rows.append({
            "level": level,
            "w0max_mm": spec.w0max * 1e3,
            "sigma_u_over_sigma_y": float(res["lam_ultimate"]),
        })
        paths[level] = ([(p["lam"], p["w_monitor"] * 1e3) for p in res["path"]])
    return rows, paths


def ice_load_uls(h_ice=1.0, p_ice_ref=5.0e6, p_water=0.1e6, drain=8.0):
    """Ice patch + lateral water pressure ULS check (transverse load path)."""
    beta = 2.0
    geom = PlateGeometry(a=3.0, b=1.0, t=thickness_for(beta), sigma_y=SIGMA_Y,
                         nx=12, ny=5)
    model = PlateModel(geom, "SSSS")
    apply_imperfection(model, ImperfectionSpec.standard(geom, "average",
                                                        sigma_r_ratio=0.0))
    p_ice_unit = 1.0e5          # reference (0.1 MPa) ice pressure for the trace
    patch = IcePatchLoad(h_ice=h_ice, p_ice=p_ice_unit)
    f_ice = patch.nodal_load(model)
    f_water = model.pressure_load(p_water)
    res = model.solve_arc_length(f_ice, F_const=f_water, max_steps=24,
                                 ds0=2e-4, monitor_node=None)
    p_ult = res["lam_ultimate"] * p_ice_unit
    p_design = ice_patch_pressure(h_ice, p_ref=p_ice_ref, h_ref=1.0)
    check = uls_check(SIGMA_Y, res["peak"]["sigma_vm_max"], p_design, p_water, p_ult)
    return {
        "geometry": {"a_m": geom.a, "b_m": geom.b, "t_mm": geom.t * 1e3,
                     "beta": geom.beta, "a_over_b": geom.aspect},
        "ice": patch.as_dict() | {"p_design_MPa": p_design / 1e6},
        "water_pressure_MPa": p_water / 1e6,
        "p_ult_MPa": p_ult / 1e6,
        "lam_ultimate": float(res["lam_ultimate"]),
        "n_steps": int(res["n_steps"]),
        "uls": check.as_dict(),
        "_path": [(p["lam"] * p_ice_unit / 1e6, p["w_monitor"] * 1e3)
                  for p in res["path"]],
    }


def pytest_summary():
    log = pathlib.Path("/tmp/pytest_pb.log")
    if not log.exists():
        return {"available": False}
    text = log.read_text()
    import re
    m = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) failed", text)
    return {"available": True, "passed": int(m.group(1)) if m else None,
            "failed": int(failed.group(1)) if failed else 0,
            "tail": text.strip().splitlines()[-1][:200]}


def make_figures(beta_rows, imperf, imperf_paths, uls, gate_rows, geom_a):
    FIG.mkdir(parents=True, exist_ok=True)
    figs = []

    # 1. load-deflection (average stress vs central deflection) for the sweep
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in beta_rows:
        lam, w = zip(*row["_path"])
        ax.plot(np.abs(w), np.array(lam) * SIGMA_Y / 1e6, "-o", ms=2.5,
                label=f"beta={row['beta']:.2f}")
    ax.set_xlabel("central deflection |w| [mm]")
    ax.set_ylabel("mean axial stress [MPa]")
    ax.set_title("Compression: load-deflection paths (arc-length)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    p = FIG / "load_deflection_paths.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append(("load_deflection_paths.png", "Load-deflection paths from the arc-length continuation"))

    # 2. ultimate strength vs beta
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    betas = np.array([r["beta"] for r in beta_rows])
    fe = np.array([r["sigma_u_over_sigma_y"] for r in beta_rows])
    ref = np.array([r["reference_faulkner"] for r in beta_rows])
    bb = np.linspace(1.0, 3.2, 200)
    ax.plot(bb, [faulkner_ultimate(b) for b in bb], "k-", lw=1.2,
            label="von Karman / Faulkner")
    ax.plot(betas, fe, "o", ms=6, label="nonlinear FE (limit point)")
    ax.set_xlabel("plate slenderness beta = (b/t) sqrt(sigma_y/E)")
    ax.set_ylabel("sigma_u / sigma_y")
    ax.set_title("Ultimate strength vs slenderness")
    ax.grid(alpha=0.3); ax.legend()
    p = FIG / "ultimate_strength_vs_beta.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append(("ultimate_strength_vs_beta.png", "Ultimate strength vs plate slenderness (FE vs Faulkner)"))

    # 3. imperfection sensitivity
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    names = [r["level"] for r in imperf]
    vals = [r["sigma_u_over_sigma_y"] for r in imperf]
    ax.bar(names, vals, color=["#7fb3d5", "#f5b041", "#cb4335"])
    ax.set_ylabel("sigma_u / sigma_y")
    ax.set_title(f"Imperfection sensitivity (beta=2.0)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    p = FIG / "imperfection_sensitivity.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append(("imperfection_sensitivity.png", "Welding imperfection sensitivity of the ultimate strength"))

    # 4. buckling mode (first eigenmode) contour
    model = PlateModel(geom_a, "SSSS")
    res = model.linear_buckling(1.0)
    w = res.mode[6 * np.arange(geom_a.n_nodes) + 2].reshape(geom_a.ny + 1,
                                                           geom_a.nx + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    cs = ax.contourf(model.xs, model.ys, w, levels=20, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax, label="w (mode shape, arbitrary units)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"First buckling mode ({geom_a.aspect:.0f}:1 plate, SSSS)")
    p = FIG / "buckling_mode.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append(("buckling_mode.png", "First elastic buckling mode of the 4:1 panel"))

    # 5. ice patch pressure-deflection path
    if uls.get("_path"):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        pz, w = zip(*uls["_path"])
        ax.plot(np.abs(w), pz, "-o", ms=2.5, color="#154360")
        ax.set_xlabel("central deflection |w| [mm]")
        ax.set_ylabel("ice patch pressure [MPa]")
        ax.set_title("Ice patch load + water pressure: transverse load path")
        ax.grid(alpha=0.3)
        p = FIG / "ice_patch_load_path.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        figs.append(("ice_patch_load_path.png", "Transverse load path under the IACS UR I ice patch"))

    return figs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    geom_a, gate_rows = gate_a()
    print("gate A done", flush=True)
    beta_rows = ultimate_strength()
    print("ultimate sweep done", flush=True)
    imperf, imperf_paths = imperfection_sensitivity()
    print("imperfection done", flush=True)
    uls = ice_load_uls()
    print("uls done", flush=True)
    figs = make_figures(beta_rows, imperf, imperf_paths, uls, gate_rows, geom_a)

    gate_a_max = max(abs(r["rel_err_vs_ritz"]) for r in gate_rows)
    gate_b_max = max(abs(r["rel_err"]) for r in beta_rows)
    w0_levels = {r["level"]: r["w0max_mm"] for r in imperf}
    payload = {
        "campaign": "W3-N03",
        "title": "Hull plate buckling and ultimate strength under ice load",
        "material": {"E_GPa": E / 1e9, "nu": NU, "sigma_y_MPa": SIGMA_Y / 1e6},
        "gate_a_linear_buckling": {
            "reference": "Rayleigh-Ritz (independent) + textbook Euler k",
            "cases": gate_rows,
            "max_rel_err_vs_ritz": gate_a_max,
            "threshold": 0.02,
            "passed": bool(gate_a_max < 0.02),
        },
        "gate_b_ultimate_strength": {
            "reference": "von Karman / Faulkner effective width 2/beta - 1/beta^2 "
                         "(the analytic curve that ALPS/Paik-type fits reproduce; "
                         "Paik et al. (2008) tabulated values are not available "
                         "offline -- stated as a limitation)",
            "imperfection": "average welding distortion 0.1 beta^2 t, critical mode, "
                            "no residual stress",
            "cases": [{k: v for k, v in r.items() if k != "_path"} for r in beta_rows],
            "max_rel_err": gate_b_max,
            "threshold": 0.10,
            "passed": bool(gate_b_max < 0.10),
        },
        "imperfection_sensitivity": imperf,
        "uls_ice_load": {k: v for k, v in uls.items() if not k.startswith("_")},
        "gate_c_pytest": pytest_summary(),
        "figures": [{"name": n, "caption": c} for n, c in figs],
        "runtime_s": time.time() - t0,
        "code": {
            "module": "src/tensorphysics/plate_buckling",
            "tests": ["tests/test_plate_buckling_fem.py",
                      "tests/test_plate_buckling_uls.py"],
        },
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2,
                                                      ensure_ascii=False))
    print(json.dumps({
        "gate_a_max_rel_err": gate_a_max,
        "gate_b_max_rel_err": gate_b_max,
        "gate_a_passed": payload["gate_a_linear_buckling"]["passed"],
        "gate_b_passed": payload["gate_b_ultimate_strength"]["passed"],
        "uls_passed": uls["uls"]["passed"],
        "uls_utilisation": uls["uls"]["utilisation"],
        "runtime_s": payload["runtime_s"],
        "figures": [n for n, _ in figs],
    }, indent=2))


if __name__ == "__main__":
    main()