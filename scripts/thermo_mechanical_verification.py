#!/usr/bin/env python
"""Thermo-mechanical coupling verification driver (campaign node p315w1-n05).

Runs the three analytical anchors (bimetal strip / thermal stress concentration
at a circular hole / transient thermal shock), the degenerate limits and the
mesh-convergence studies, then writes

* ``artifacts/thermo_mechanical/verification.json`` -- machine-readable verdict
  (every acceptance number + artifact paths, programmatically generated, no
  hand-copied digits),
* ``docs/figures/thermo_mechanical_*.png`` -- field plots and comparison curves,
* ``docs/thermo_mechanical_verification.md`` -- report body (numbers injected).

Usage: ``python scripts/thermo_mechanical_verification.py``
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.thermo_mechanical import (  # noqa: E402
    DEFAULT_MATERIAL,
    annulus_thermal_stress,
    arc_end_deflection,
    bimetal_beam_fe,
    bimetal_q4_fe,
    bimetal_q8_fe,
    hole_fe_sweep,
    thermal_shock_fe,
    thermal_stress_concentration,
    timoshenko_curvature,
    timoshenko_end_deflection,
)

ART = ROOT / "artifacts" / "thermo_mechanical"
FIG = ROOT / "docs" / "figures"
H1 = H2 = 1.0e-3
E1 = E2 = 200.0e9
ALPHA2 = 6.0e-6
DT = 100.0
LENGTH = 0.10
RATIOS = (2.0, 3.0, 5.0)


def pct(x):
    return float(100.0 * x)


def bimetal_block():
    rows = []
    for ratio in RATIOS:
        a1 = ratio * ALPHA2
        kappa_ref = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
        from tensorphysics.thermo_mechanical import exact_composite_solution
        sec = exact_composite_solution(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH)
        beam = bimetal_beam_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH, n_elems=8)
        q8 = bimetal_q8_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH,
                           nx=25, ny1=2, ny2=2)
        tip_ref = timoshenko_end_deflection(kappa_ref, LENGTH)
        rows.append({
            "alpha1_over_alpha2": ratio,
            "alpha1": a1,
            "kappa_timoshenko_1925": kappa_ref,
            "kappa_layered_section": sec["kappa"],
            "kappa_rel_err_section": abs(sec["kappa"] - kappa_ref) / kappa_ref,
            "kappa_frame_fe": beam["kappa_mean"],
            "kappa_frame_fe_rel_err": abs(beam["kappa_mean"] - kappa_ref) / kappa_ref,
            "kappa_q8_2d": q8["kappa"],
            "kappa_q8_rel_err": abs(q8["kappa"] - kappa_ref) / kappa_ref,
            "tip_deflection_analytic": tip_ref,
            "tip_deflection_frame_fe": abs(beam["tip_deflection"]),
            "tip_deflection_frame_fe_rel_err": abs(abs(beam["tip_deflection"]) - tip_ref) / tip_ref,
            "tip_deflection_q8_2d": abs(q8["tip_deflection"]),
            "tip_deflection_q8_rel_err": abs(abs(q8["tip_deflection"]) - tip_ref) / tip_ref,
            "tip_deflection_elastica": arc_end_deflection(kappa_ref, LENGTH),
        })
    return rows


def bimetal_convergence():
    a1 = 2.0 * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
    q8_rows, q4_rows = [], []
    for nx, ny in ((10, 1), (20, 2), (40, 4)):
        res = bimetal_q8_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH,
                            nx=nx, ny1=ny, ny2=ny)
        q8_rows.append({"nx": nx, "ny": ny, "n_dof": res["n_dof"],
                        "h": LENGTH / nx, "kappa": res["kappa"],
                        "rel_err": abs(res["kappa"] - kappa_ref) / kappa_ref,
                        "tip_deflection": res["tip_deflection"]})
    for nx, ny in ((25, 2), (50, 4)):
        res = bimetal_q4_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH,
                            nx=nx, ny1=ny, ny2=ny)
        q4_rows.append({"nx": nx, "ny": ny, "n_dof": res["n_dof"],
                        "h": LENGTH / nx, "kappa": res["kappa"],
                        "rel_err": abs(res["kappa"] - kappa_ref) / kappa_ref})
    rates = [float(np.log2(q8_rows[i]["rel_err"] / q8_rows[i + 1]["rel_err"]))
             for i in range(len(q8_rows) - 1)]
    q4_rates = [float(np.log2(q4_rows[i]["rel_err"] / q4_rows[i + 1]["rel_err"]))
                for i in range(len(q4_rows) - 1)]
    return {"kappa_ref": kappa_ref, "q8": q8_rows, "q8_rates": rates,
            "q4": q4_rows, "q4_rates": q4_rates}


def hole_block():
    a, b, Ta, Tb, E, alpha, nu = 0.01, 0.20, 100.0, 0.0, 200.0e9, 12.0e-6, 0.3
    ex = thermal_stress_concentration(a, b, Ta, Tb, E, alpha)
    rows = []
    for ne in (32, 64, 128, 256):
        fe = hole_fe_sweep(a, b, Ta, Tb, E, alpha, nu=nu, n_elems=ne)
        rows.append({
            "n_elems": ne,
            "K_t_fe_same_radius": fe["K_t_fe"],
            "K_t_exact_same_radius": fe["K_t_exact"],
            "rel_err_same_radius": fe["rel_error_Kt"],
            "K_t_fe_hole_edge": fe["K_t_fe_edge"],
            "K_t_exact_hole_edge": fe["K_t_exact_edge"],
            "rel_err_hole_edge": fe["rel_error_Kt_edge"],
            "max_rel_err_profile": fe["max_rel_error_profile"],
        })
    rates = [float(np.log2(rows[i]["rel_err_same_radius"] / rows[i + 1]["rel_err_same_radius"]))
             for i in range(len(rows) - 1)]
    return {"a": a, "b": b, "Ta": Ta, "Tb": Tb, "E": E, "alpha": alpha, "nu": nu,
            "K_t_analytic": ex["K_t"], "sigma_theta_hole_edge": ex["sigma_theta"],
            "rows": rows, "rates": rates}


def shock_block():
    out = thermal_shock_fe([0.1, 1.0, 10.0])
    scale = abs(out["surface_stress_exact"][0])
    per_time = []
    y = out["y"]
    for k, t in enumerate(out["times"]):
        sig = out["sigma_xx"][k]
        sig_ex = out["sigma_xx_exact"][k]
        probes = {}
        for mult in (0.0, 0.5, 1.0, 2.0, 3.0):
            i = int(np.argmin(np.abs(y - mult * np.sqrt(out["kappa"] * t))))
            probes[f"y_over_sqrt_kappa_t_{mult}"] = {
                "y": float(y[i]),
                "sigma_fe": float(sig[i]),
                "sigma_exact": float(sig_ex[i]),
                "rel_err": float(abs(sig[i] - sig_ex[i]) / scale),
            }
        per_time.append({
            "t": float(t),
            "sqrt_kappa_t": float(np.sqrt(out["kappa"] * t)),
            "surface_stress_fe": float(sig[0]),
            "surface_stress_exact": float(out["surface_stress_exact"][k]),
            "surface_rel_err": float(abs(sig[0] - out["surface_stress_exact"][k]) / scale),
            "temperature_max_rel_err": float(np.max(np.abs(out["T"][k] - out["T_exact"][k])) / 20.0),
            "stress_profile_max_rel_err": float(out["max_rel_err_profile"][k]),
            "probes": probes,
        })
    return {"material": DEFAULT_MATERIAL, "kappa": out["kappa"],
            "max_sigma_yy_over_scale": out["max_sigma_yy"] / scale,
            "per_time": per_time, "_data": out}


def degenerate_block():
    from tensorphysics.thermo_mechanical import bimetal_section

    P = 1.0
    sec = bimetal_section(H1, H2, E1, E2, 12e-6, 6e-6, width=1.0, dT=0.0)
    pure = bimetal_beam_fe(H1, H2, E1, E2, 12e-6, 6e-6, 0.0, length=LENGTH,
                           n_elems=16, tip_load=P)
    ref = P * LENGTH ** 3 / (3.0 * sec.EI)
    zero_dT = bimetal_beam_fe(H1, H2, E1, E2, 12e-6, 6e-6, 0.0, length=LENGTH, n_elems=8)
    equal_alpha = bimetal_q8_fe(H1, H2, E1, E2, 8e-6, 8e-6, DT, length=LENGTH,
                                nx=20, ny1=2, ny2=2)
    zero_alpha = bimetal_q8_fe(H1, H2, E1, E2, 0.0, 0.0, DT, length=LENGTH,
                               nx=10, ny1=1, ny2=1)
    return {
        "pure_elastic_tip_load": {
            "P": P, "EI": sec["EI"] if isinstance(sec, dict) else sec.EI,
            "delta_fe": pure["tip_deflection"], "delta_analytic": ref,
            "rel_err": abs(pure["tip_deflection"] - ref) / ref,
        },
        "dT_zero_curvature": {"kappa_mean": zero_dT["kappa_mean"],
                              "tip_deflection": zero_dT["tip_deflection"]},
        "equal_alpha_curvature": {"kappa": equal_alpha["kappa"],
                                  "tip_deflection": equal_alpha["tip_deflection"]},
        "zero_alpha_stress": {
            "max_sigma_over_EalphaDT": float(np.max(np.abs(zero_alpha["sigma"]))
                                             / (E1 * 12e-6 * DT)),
        },
    }


def make_figures(bim, conv, hole, shock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    paths = {}

    # ---- figure 1: bimetal ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ratios = [r["alpha1_over_alpha2"] for r in bim]
    ax.plot(ratios, [r["kappa_timoshenko_1925"] for r in bim], "k-", lw=2,
            label="Timoshenko (1925)")
    ax.plot(ratios, [r["kappa_frame_fe"] for r in bim], "o", ms=9, mfc="none",
            label="frame FE")
    ax.plot(ratios, [r["kappa_q8_2d"] for r in bim], "s", ms=6, label="2-D Q8 FE")
    ax.set_xlabel(r"$\alpha_1/\alpha_2$")
    ax.set_ylabel(r"curvature $\kappa$  [1/m]")
    ax.set_title(r"bimetal curvature ($\Delta T=100$ K)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    h = [r["h"] * 1e3 for r in conv["q8"]]
    ax.loglog(h, [r["rel_err"] for r in conv["q8"]], "o-", label="Q8 (quadratic)")
    ax.loglog([r["h"] * 1e3 for r in conv["q4"]], [r["rel_err"] for r in conv["q4"]],
              "s--", label="Q4 (bilinear, locks)")
    ref = np.array([r["rel_err"] for r in conv["q8"]])
    ax.loglog(h, ref[0] * (np.array(h) / h[0]) ** 2, "k:", label=r"$O(h^2)$ guide")
    ax.set_xlabel("element size $h$ [mm]")
    ax.set_ylabel(r"$|\kappa_{FE}-\kappa_{Timoshenko}|/\kappa$")
    ax.set_title("mesh convergence of curvature")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[2]
    for ratio in (2.0, 3.0):
        a1 = ratio * ALPHA2
        q8 = bimetal_q8_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH,
                           nx=25, ny1=2, ny2=2)
        nodes, u = q8["nodes"], q8["u"]
        ax.plot(nodes[:, 0] * 1e3, u[:, 1] * 1e3, ".", ms=2,
                label=fr"Q8 $\alpha_1/\alpha_2$={ratio:g}")
        kappa = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
        xx = np.linspace(0, LENGTH, 50)
        ax.plot(xx * 1e3, -0.5 * kappa * xx ** 2 * 1e3, "-",
                label=fr"$-\kappa x^2/2$, $\alpha_1/\alpha_2$={ratio:g}")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("deflection w [mm]")
    ax.set_title("deflected shape (cantilever)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths["bimetal"] = FIG / "thermo_mechanical_bimetal.png"
    fig.savefig(paths["bimetal"], dpi=130)
    plt.close(fig)

    # ---- figure 2: thermal stress concentration ----
    a, b, Ta, Tb = hole["a"], hole["b"], hole["Ta"], hole["Tb"]
    E, alpha = hole["E"], hole["alpha"]
    r = np.linspace(a, b, 600)
    s_r, s_t = annulus_thermal_stress(r, a, b, Ta, Tb, E, alpha)
    fe = hole_fe_sweep(a, b, Ta, Tb, E, alpha, nu=0.3, n_elems=256)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    scale = E * alpha * (Ta - Tb)
    ax.plot(r / a, s_t / scale, "k-", lw=2, label=r"exact $\sigma_\theta$")
    ax.plot(r / a, s_r / scale, "b--", lw=1.5, label=r"exact $\sigma_r$")
    ax.plot(fe["r_mid"] / a, fe["sigma_theta_fe"] / scale, "o", ms=4, mfc="none",
            label=r"ring FE $\sigma_\theta$ (256 el.)")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$\sigma/(E\alpha\Delta T)$")
    ax.set_title("radial gradient around a circular hole")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    ne = [row["n_elems"] for row in hole["rows"]]
    ax.loglog(ne, [row["rel_err_same_radius"] for row in hole["rows"]], "o-",
              label=r"$K_t$ error (same radius)")
    ax.loglog(ne, [row["rel_err_hole_edge"] for row in hole["rows"]], "s--",
              label=r"$K_t$ error (hole-edge extrapolation)")
    n0 = hole["rows"][0]["rel_err_same_radius"]
    ax.loglog(ne, n0 * (np.array(ne, dtype=float) / ne[0]) ** -2, "k:",
              label=r"$O(h^2)$ guide")
    ax.set_xlabel("number of ring elements")
    ax.set_ylabel("relative error")
    ax.set_title(r"convergence of $K_t$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths["hole"] = FIG / "thermo_mechanical_hole.png"
    fig.savefig(paths["hole"], dpi=130)
    plt.close(fig)

    # ---- figure 3: thermal shock ----
    data = shock["_data"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for k, t in enumerate(data["times"]):
        ax.plot(data["y"] * 1e3, data["T"][k], "-", lw=1.5,
                label=fr"FE t={t:g} s")
        ax.plot(data["y"] * 1e3, data["T_exact"][k], "k:", lw=1.0)
    ax.set_xlim(0, 40)
    ax.set_xlabel("depth y [mm]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title("thermal shock: FE (lines) vs erfc (dotted)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    scale = abs(data["surface_stress_exact"][0]) / 1e6
    for k, t in enumerate(data["times"]):
        ax.plot(data["y"] / np.sqrt(data["kappa"] * t),
                data["sigma_xx"][k] / 1e6, "-", lw=1.5, label=fr"FE t={t:g} s")
        ax.plot(data["y"] / np.sqrt(data["kappa"] * t),
                data["sigma_xx_exact"][k] / 1e6, "k:", lw=1.0)
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$y/\sqrt{\kappa t}$")
    ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax.set_title(f"in-plane thermal stress (surface {scale:.1f} MPa)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths["shock"] = FIG / "thermo_mechanical_shock.png"
    fig.savefig(paths["shock"], dpi=130)
    plt.close(fig)
    return {k: str(v.relative_to(ROOT)) for k, v in paths.items()}


def main():
    ART.mkdir(parents=True, exist_ok=True)
    bim = bimetal_block()
    conv = bimetal_convergence()
    hole = hole_block()
    shock = shock_block()
    degen = degenerate_block()
    figures = make_figures(bim, conv, hole, shock)

    shock_slim = {k: v for k, v in shock.items() if not k.startswith("_")}
    criteria = {
        "a_bimetal_curvature_le_3pct": {
            "value_pct": pct(max(r["kappa_q8_rel_err"] for r in bim)),
            "frame_fe_pct": pct(max(r["kappa_frame_fe_rel_err"] for r in bim)),
            "threshold_pct": 3.0,
            "pass": bool(max(max(r["kappa_q8_rel_err"] for r in bim),
                             max(r["kappa_frame_fe_rel_err"] for r in bim)) <= 0.03),
        },
        "b_bimetal_tip_deflection_le_3pct": {
            "value_pct": pct(max(r["tip_deflection_q8_rel_err"] for r in bim)),
            "frame_fe_pct": pct(max(r["tip_deflection_frame_fe_rel_err"] for r in bim)),
            "threshold_pct": 3.0,
            "pass": bool(max(max(r["tip_deflection_q8_rel_err"] for r in bim),
                             max(r["tip_deflection_frame_fe_rel_err"] for r in bim)) <= 0.03),
        },
        "c_hole_kt_le_5pct": {
            "value_pct": pct(hole["rows"][-1]["rel_err_same_radius"]),
            "hole_edge_pct": pct(hole["rows"][-1]["rel_err_hole_edge"]),
            "threshold_pct": 5.0,
            "pass": bool(hole["rows"][-1]["rel_err_hole_edge"] <= 0.05),
        },
        "d_shock_surface_stress_le_5pct": {
            "value_pct": pct(max(t["surface_rel_err"] for t in shock["per_time"])),
            "profile_pct": pct(max(t["stress_profile_max_rel_err"] for t in shock["per_time"])),
            "threshold_pct": 5.0,
            "pass": bool(max(max(t["surface_rel_err"] for t in shock["per_time"]),
                             max(t["stress_profile_max_rel_err"] for t in shock["per_time"])) <= 0.05),
        },
        "e_degenerate_pure_elastic_le_1pct": {
            "value_pct": pct(degen["pure_elastic_tip_load"]["rel_err"]),
            "threshold_pct": 1.0,
            "pass": bool(degen["pure_elastic_tip_load"]["rel_err"] <= 0.01),
        },
        "f_mesh_convergence_second_order": {
            "q8_rates": conv["q8_rates"],
            "q4_rates": conv["q4_rates"],
            "min_rate": float(min(conv["q8_rates"])),
            "pass": bool(min(conv["q8_rates"]) >= 1.8),
        },
    }
    verdict = {
        "node": "p315w1-n05-fem-thermo-mechanical",
        "repo": "TensorPhysics",
        "module": "src/tensorphysics/thermo_mechanical",
        "bimetal": bim,
        "bimetal_convergence": conv,
        "hole_thermal": hole,
        "thermal_shock": shock_slim,
        "degenerate": degen,
        "criteria": criteria,
        "all_pass": bool(all(v["pass"] for v in criteria.values())),
        "figures": figures,
        "artifacts": {
            "verification_json": "artifacts/thermo_mechanical/verification.json",
            "report_md": "docs/thermo_mechanical_verification.md",
            "tests": [
                "tests/test_thermo_mechanical_bimetal.py",
                "tests/test_thermo_mechanical_hole.py",
                "tests/test_thermo_mechanical_shock.py",
                "tests/test_thermo_mechanical_degenerate.py",
                "tests/test_thermo_mechanical_convergence.py",
            ],
        },
    }
    out = ART / "verification.json"
    out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"all_pass": verdict["all_pass"],
                      "criteria": {k: v["pass"] for k, v in criteria.items()}},
                     indent=2))
    print("wrote", out.relative_to(ROOT))
    return verdict


if __name__ == "__main__":
    main()