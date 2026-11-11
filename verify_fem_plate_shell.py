#!/usr/bin/env python
"""TensorFEM plate/shell static verification driver (campaign node p405w1-n05).

Runs the whole verification face implemented in
:mod:`tensorphysics.plate_shell` and writes

* ``<outdir>/verify_fem_plate_shell.json``  -- all numbers, gate verdicts,
  negative controls (machine-auditable, nothing typed by hand);
* ``<outdir>/fig1_plate_deflection_field.png``  -- FE vs Navier deflection field
  and the relative error map;
* ``<outdir>/fig2_plate_moments_shear.png``  -- ``Mx``, ``My``, ``Qx`` profiles
  along the plate centre lines, FE vs the Navier series;
* ``<outdir>/fig3_convergence_orders.png``  -- h-refinement (log-log) with the
  fitted orders;
* ``<outdir>/fig4_shell_pressure_edge_effect.png`` -- cylindrical shell under
  internal pressure: hoop-stress distribution, edge bending moment and the
  measured ``exp(-beta x)`` boundary layer (``beta`` vs ``sqrt(R t)``).

Usage::

    python verify_fem_plate_shell.py                # full run + figures
    python verify_fem_plate_shell.py --quick        # coarser, for smoke tests
    python verify_fem_plate_shell.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from tensorphysics.plate_shell import (  # noqa: E402
    SSSSPlateFE,
    CylinderPressureFE,
    beta_edge,
    cylinder_edge_solution,
    field_error,
    navier_plate_fields,
    run_verification,
)


def _figures(payload, outdir: pathlib.Path, plate_mesh: int = 32,
             shell_mesh: int = 128) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    pc = payload["plate"]["case"]
    sc = payload["shell"]["case"]

    # ---------------------------------------------------------------- fig 1
    fe = SSSSPlateFE(a=pc["a"], b=pc["b"], t=pc["t"], E=pc["E"], nu=pc["nu"],
                     nx=plate_mesh, ny=plate_mesh)
    fe.solve(pc["q"])
    fld = field_error(pc["a"], pc["b"], pc["t"], pc["E"], pc["nu"], pc["q"],
                      fe, n_sample=41)
    fe_fig = SSSSPlateFE(a=pc["a"], b=pc["b"], t=pc["t"], E=pc["E"],
                         nu=pc["nu"], nx=8, ny=8)
    fe_fig.solve(pc["q"])
    fld = field_error(pc["a"], pc["b"], pc["t"], pc["E"], pc["nu"], pc["q"],
                      fe_fig, n_sample=61)
    err_max = float(np.max(np.abs(fld["rel_w"])))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    scale = 1e3  # mm
    cs0 = axes[0].contourf(fld["x"], fld["y"], fld["w_fe"] * scale, levels=20,
                           cmap="viridis")
    fig.colorbar(cs0, ax=axes[0], label="w FE [mm]")
    axes[0].set_title("FE deflection, 8x8 mesh")
    cs1 = axes[1].contourf(fld["x"], fld["y"], fld["w_ref"] * scale, levels=20,
                           cmap="viridis")
    fig.colorbar(cs1, ax=axes[1], label="w Navier [mm]")
    axes[1].set_title("Navier series (exact)")
    cs2 = axes[2].contourf(fld["x"], fld["y"], fld["rel_w"] / err_max,
                           levels=20, cmap="RdBu_r")
    fig.colorbar(cs2, ax=axes[2], label="normalised error [-]")
    axes[2].set_title(f"8x8 error pattern, max $|\\Delta w|/w_{{max}}$ = "
                      f"{err_max * 100:.2e} %")
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    fig.suptitle("Uniformly loaded simply supported square plate "
                 f"(a={pc['a']} m, t={pc['t']} m, q={pc['q']:.0f} Pa)")
    fig.tight_layout()
    p1 = outdir / "fig1_plate_deflection_field.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    paths.append(p1)

    # ---------------------------------------------------------------- fig 2
    xs = np.linspace(0.0, pc["a"], 121)
    ys = np.linspace(0.0, pc["b"], 121)
    line_x = fe.fields_at_points(xs, np.full_like(xs, 0.5 * pc["b"]))
    line_y = fe.fields_at_points(np.full_like(ys, 0.5 * pc["a"]), ys)
    ref_x = navier_plate_fields(pc["a"], pc["b"], pc["t"], pc["E"], pc["nu"],
                                pc["q"], xs, np.full_like(xs, 0.5 * pc["b"]))
    ref_y = navier_plate_fields(pc["a"], pc["b"], pc["t"], pc["E"], pc["nu"],
                                pc["q"], np.full_like(ys, 0.5 * pc["a"]), ys)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    axes[0].plot(xs, ref_x["Mx"] / 1e3, "-", color="k", label="Navier $M_x$")
    axes[0].plot(xs, line_x["Mx"] / 1e3, "--", color="C3",
                 label=f"FE ({plate_mesh}x{plate_mesh})")
    axes[0].plot(xs, ref_x["My"] / 1e3, "-", color="0.6", label="Navier $M_y$")
    axes[0].plot(xs, line_x["My"] / 1e3, ":", color="C0", label="FE $M_y$")
    axes[0].set_xlabel("x [m] (y = b/2)")
    axes[0].set_ylabel("moment [kN m/m]")
    axes[0].legend(fontsize=7)
    axes[0].set_title("bending moments")

    axes[1].plot(xs, ref_x["Qx"] / 1e3, "-", color="k", label="Navier exact")
    axes[1].plot(xs, line_x["Qx"] / 1e3, "--", color="C3", label="FE")
    axes[1].set_xlabel("x [m] (y = b/2)")
    axes[1].set_ylabel("$Q_x$ [kN/m]")
    axes[1].legend(fontsize=7)
    axes[1].set_title("shear force")

    axes[2].plot(ys, ref_y["My"] / 1e3, "-", color="k", label="Navier exact")
    axes[2].plot(ys, line_y["My"] / 1e3, "--", color="C3", label="FE")
    axes[2].set_xlabel("y [m] (x = a/2)")
    axes[2].set_ylabel("$M_y$ [kN m/m]")
    axes[2].legend(fontsize=7)
    axes[2].set_title("$M_y$ across the centre line")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = outdir / "fig2_plate_moments_shear.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    paths.append(p2)

    # ---------------------------------------------------------------- fig 3
    rows = payload["plate"]["meshes"]
    h = np.array([r["h"] for r in rows])
    orders = payload["plate"]["orders"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    series = [("err_w", "$w_{max}$", "o"), ("l2_w", "$L_2$ error of $w$", "s"),
              ("err_Mx", "$M_x$ centre", "^"), ("err_Qx", "$Q_x(a/4,b/2)$", "d")]
    for key, label, mk in series:
        err = np.array([r[key] for r in rows])
        axes[0].loglog(h, err, mk + "-",
                       label=f"{label} (fit p={orders[key]:.2f})")
    href = h / h[0]
    for p, style in ((2.0, ":"), (4.0, "--")):
        axes[0].loglog(h, 0.05 * href ** p, style, color="0.6",
                       label=f"$h^{{{int(p)}}}$ reference")
    axes[0].set_xlabel("element size h [m]")
    axes[0].set_ylabel("relative error")
    axes[0].set_ylim(1e-9, 1)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title("plate h-refinement vs Navier series")

    sr = payload["shell"]["meshes"]
    sh = np.array([r["h"] for r in sr])
    so = payload["shell"]["orders"]
    axes[1].loglog(sh, [r["err_Mx_edge"] for r in sr], "o-",
                   label=f"edge $M_x$ (p={so['err_Mx_edge']:.2f})")
    axes[1].loglog(sh, [r["err_sigma_theta_eq_pointwise"] for r in sr], "s-",
                   label="pointwise $\\sigma_\\theta$ (p="
                         f"{so['err_sigma_theta_eq_pointwise']:.2f})")
    axes[1].loglog(sh, [r["err_sigma_theta_pointwise"] for r in sr], "^-",
                   label="$\\sigma_\\theta$ const. recovery (p="
                         f"{so['err_sigma_theta_pointwise']:.2f})")
    axes[1].axhline(1e-2, color="0.5", ls=":", label="1% gate")
    axes[1].set_xlabel("element length h [m]")
    axes[1].set_ylabel("relative error")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].set_title("cylindrical shell h-refinement vs closed form")
    fig.tight_layout()
    p3 = outdir / "fig3_convergence_orders.png"
    fig.savefig(p3, dpi=110)
    plt.close(fig)
    paths.append(p3)

    # ---------------------------------------------------------------- fig 4
    beta = beta_edge(sc["R"], sc["t"], sc["E"], sc["nu"])
    fe_s = CylinderPressureFE(R=sc["R"], L=sc["L"], t=sc["t"], E=sc["E"],
                              nu=sc["nu"], n_elem=shell_mesh)
    fe_s.solve(sc["p"])
    xg = np.linspace(0.0, sc["L"], 401)
    f = fe_s.fields_at(xg)
    ref = cylinder_edge_solution(sc["R"], sc["L"], sc["t"], sc["E"], sc["nu"],
                                 sc["p"], xg)
    xu = xg / np.sqrt(sc["R"] * sc["t"])
    s_mem = sc["p"] * sc["R"] / sc["t"]
    w_p = sc["p"] * sc["R"] ** 2 / (sc["E"] * sc["t"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    axes[0].plot(xu, ref["sigma_theta"] / s_mem, "-", color="k",
                 label="Donnell closed form")
    axes[0].plot(xu, f["sigma_theta_eq"] / s_mem, "--", color="C3",
                 label=f"FE ({shell_mesh} el.)")
    axes[0].axhline(1.0, color="0.6", ls=":", label="$pR/t$ (membrane)")
    axes[0].set_xlabel("x / $\\sqrt{Rt}$")
    axes[0].set_ylabel("$\\sigma_\\theta / (pR/t)$")
    axes[0].legend(fontsize=7)
    axes[0].set_title("hoop stress: membrane + edge effect")
    axes[0].grid(alpha=0.3)

    axes[1].plot(xu, ref["Mx"] / 1e3, "-", color="k", label="closed form")
    axes[1].plot(xu, f["Mx"] / 1e3, "--", color="C3", label="FE")
    axes[1].set_xlabel("x / $\\sqrt{Rt}$")
    axes[1].set_ylabel("$M_x$ [kN m/m]")
    axes[1].legend(fontsize=7)
    axes[1].set_title("edge bending moment $M_x$")
    axes[1].grid(alpha=0.3)

    dec = payload["shell"]["edge_decay_fit"]
    axes[2].semilogy(xu, np.abs(ref["w"] - w_p) / w_p, "-", color="k",
                     label="exact $|w - w_p|/w_p$")
    axes[2].semilogy(xu, np.abs(f["w"] - w_p) / w_p, "--", color="C3",
                     label="FE")
    xx = np.linspace(0.0, xu[-1], 200)
    axes[2].semilogy(xx, np.abs(dec["R0"]) / w_p
                     * np.exp(-dec["beta_fit"] * xx
                              * np.sqrt(sc["R"] * sc["t"])),
                     ":", color="C0",
                     label="fitted $e^{-\\beta x}$, "
                           f"$\\beta\\sqrt{{Rt}}$={dec['beta_sqrt_Rt']:.3f}")
    axes[2].set_xlabel("x / $\\sqrt{Rt}$")
    axes[2].set_ylabel("relative deviation from membrane")
    axes[2].legend(fontsize=7)
    axes[2].set_title("boundary layer: 1/$\\beta$ = "
                      f"{dec['decay_length_over_sqrt_Rt']:.3f}$\\sqrt{{Rt}}$")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Cylindrical shell under internal pressure "
                 f"(R={sc['R']} m, t={sc['t']} m, "
                 f"R/t={sc['R'] / sc['t']:.0f}, p={sc['p'] / 1e6:.1f} MPa)")
    fig.tight_layout()
    p4 = outdir / "fig4_shell_pressure_edge_effect.png"
    fig.savefig(p4, dpi=110)
    plt.close(fig)
    paths.append(p4)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="artifacts/plate_shell")
    ap.add_argument("--quick", action="store_true",
                    help="coarser meshes (smoke test only, gates not valid)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        payload = run_verification(plate_meshes=(4, 8, 16),
                                   shell_meshes=(8, 32), with_gates=False)
        plate_mesh, shell_mesh = 8, 32
    else:
        payload = run_verification()
        plate_mesh, shell_mesh = 32, 128

    figs = []
    if not args.no_figures:
        figs = [str(p) for p in _figures(payload, outdir, plate_mesh,
                                         shell_mesh)]
    payload["figures"] = figs

    jpath = outdir / "verify_fem_plate_shell.json"
    jpath.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"verdict: {payload['verdict']} "
          f"({payload['n_gates_pass']}/{payload['n_gates']} gates)")
    for r in payload["gate_table"]:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['gate']}: "
              f"{r['criterion']}")
    print(f"json: {jpath}")
    for f in figs:
        print(f"figure: {f}")
    return 0 if payload["verdict"] in ("PASS", "SMOKE") else 1


if __name__ == "__main__":
    raise SystemExit(main())