"""Run the ice-belt FEM cross-check and render its report figures.

Usage
-----
    python scripts/run_hull_icebelt_fem_xcheck.py [--out artifacts/]

Produces
--------
* ``artifacts/hull_icebelt_fem_xcheck.json`` -- the machine-checked artifact
  (verification deviations, mesh convergence, dual-chain deviation table,
  attribution, criteria, audit chain, limitations);
* ``artifacts/figs/icebelt_*.png`` -- the report figures.

Every number in the figures comes from the same run that writes the JSON; no
value is typed by hand in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.hull_icebelt_fem import (  # noqa: E402
    analytic,
    iacs_pressure,
    panel_fem,
    xcheck,
)


def _element_centre_fields(panel: panel_fem.IceBeltPanel, pressure: float,
                           box, load_rule: str = "consistent"):
    """Element-centre bending stresses and deflection for plotting."""
    from tensorphysics.plate_buckling import PlateModel

    g = panel.geometry()
    model = PlateModel(g, bc=panel.bc, validate=(panel.gate == "strict"))
    sol = panel.solve(pressure, patch_box=box, model=model, load_rule=load_rule)
    Db = (g.E * g.t ** 3 / (12.0 * (1.0 - g.nu ** 2))) * np.array(
        [[1.0, g.nu, 0.0], [g.nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - g.nu)]])
    u_b = sol.u[model.edof_bend]
    kappa = np.einsum("gci,ei->egc", model.kern["Bb"], u_b)
    M = kappa @ Db.T
    sig = 6.0 * M / g.t ** 2
    xs = np.array([[xc for (xc, _) in model.elem_xy]])[0].reshape(g.ny, g.nx)
    ys = np.array([[yc for (_, yc) in model.elem_xy]])[0].reshape(g.ny, g.nx)
    return {
        "x": xs, "y": ys,
        "sigma_x": sig[..., 0].mean(axis=1).reshape(g.ny, g.nx) / 1e6,
        "sigma_y": sig[..., 1].mean(axis=1).reshape(g.ny, g.nx) / 1e6,
        "w": sol.u[6 * np.arange(g.n_nodes) + 2].reshape(g.ny + 1, g.nx + 1),
        "solution": sol,
        "geom": g,
    }


def fig_pressure_area(art: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    areas = np.logspace(-1.4, 1.0, 200)
    p = [iacs_pressure.pressure_area_law(a) / 1e6 for a in areas]
    ax.loglog(areas, p, "-", lw=2, label="chain A: IACS pressure-area law "
                                         r"($p=1.2\,(A/1\,\mathrm{m^2})^{-0.5}$ MPa)")
    rows = art["dual_chain"]["rows"]
    a1 = [r["design_area_on_panel_m2"] for r in rows]
    p1 = [r["chain_a"]["pressure_MPa"] for r in rows]
    a2 = [r["design_area_contact_patch_m2"] for r in rows]
    p2 = [r["chain_a"]["pressure_contact_patch_convention_MPa"] for r in rows]
    ax.loglog(a1, p1, "o", ms=6, label="chain A, panel-area convention")
    ax.loglog(a2, p2, "s", ms=6, mfc="none",
              label="chain A, contact-patch convention")
    if art["dual_chain"]["chain_b_status"] == "AVAILABLE":
        pB = [r["chain_b"]["p_patch_MPa"] for r in rows]
        ax.axhline(pB[0], color="crimson", lw=2, ls="--",
                   label=f"chain B (B1 ice_fullscale): {pB[0]:.2f} MPa, "
                         "area-independent")
        for r, a, pv in zip(rows, a1, p1):
            ax.annotate(f"h={r['h_ice_m']:.2f} m", (a, pv),
                        textcoords="offset points", xytext=(6, -12), fontsize=7)
    ax.set_xlabel("design patch area A [m$^2$]")
    ax.set_ylabel("ice patch pressure p [MPa]")
    ax.set_title("Pressure-area law: chain A vs chain B (no area reduction)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    path = out / "icebelt_pressure_vs_area.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_fem_vs_analytic(art: dict, out: Path) -> Path:
    panel = xcheck.VERIFY_PANEL
    box = panel.patch_box(xcheck.VERIFY_PATCH_THICKNESS)
    q = xcheck.PATCH_TEST_PRESSURE
    data = _element_centre_fields(panel, q, box)
    g = data["geom"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    xs_n = np.linspace(0.0, g.a, g.nx + 1)
    ax.plot(xs_n, data["w"][g.ny // 2, :] * 1e3, "o-", ms=3, lw=1.0,
            label="FEM (H3 Hermite, 48x16)")
    xs = np.linspace(0.0, g.a, 201)
    prof = analytic.navier_grid(g, [box], q, xs, [g.b / 2], n_terms=81)
    ax.plot(xs, prof["w"][:, 0] * 1e3, "-", lw=2.0, label="closed form (Navier)")
    ax.set_xlabel("x [m]  (y = b/2)")
    ax.set_ylabel("deflection w [mm]")
    ax.set_title("Patch load: centreline deflection")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    names, wdev, sdev = [], [], []
    for c in art["verification"]["cases"]:
        if not c.get("acceptance", True):
            continue
        names.append(c["name"].replace("_", " "))
        wdev.append(c["deviation_pct"]["w_centre"])
        sdev.append(c["deviation_pct"]["sigma_centre"])
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, wdev, 0.36, label=r"$\Delta w_{centre}$")
    ax.bar(xpos + 0.18, sdev, 0.36, label=r"$\Delta \sigma_{centre}$")
    ax.axhline(5.0, color="crimson", ls="--", lw=1.2, label="5 % gate")
    ax.axhline(-5.0, color="crimson", ls="--", lw=1.2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=7, rotation=12)
    ax.set_ylabel("deviation vs closed form [%]")
    ax.set_title("FEM vs closed-form agreement")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "icebelt_fem_vs_analytic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_stress_fields(art: dict, out: Path) -> Path:
    design = art["design_case"]
    panel = xcheck.DESIGN_PANEL
    box = panel.patch_box(design["pressure"]["h_ice_m"])
    q = design["pressure"]["pressure_Pa"]
    fem = _element_centre_fields(panel, q, box)
    g = fem["geom"]
    xs = np.linspace(0.0, g.a, 121)
    ys = np.linspace(0.0, g.b, 61)
    ana = analytic.navier_grid(g, [box], q, xs, ys, n_terms=61)
    sig_a = 6.0 * np.abs(ana["Mx"]) / g.t ** 2 / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    vmax = max(np.abs(fem["sigma_x"]).max(), sig_a.max())
    im = axes[0].pcolormesh(fem["x"], fem["y"], fem["sigma_x"], cmap="viridis",
                            vmin=0, vmax=vmax, shading="nearest")
    axes[0].set_title("FEM $\\sigma_x$ [MPa] (PC6 design patch)")
    fig.colorbar(im, ax=axes[0])
    im = axes[1].pcolormesh(xs, ys, np.abs(sig_a.T), cmap="viridis", vmin=0,
                            vmax=vmax, shading="auto")
    axes[1].set_title("closed-form $|\\sigma_x|$ [MPa]")
    fig.colorbar(im, ax=axes[1])
    im = axes[2].pcolormesh(fem["x"], fem["y"], fem["sigma_y"], cmap="magma",
                            shading="nearest")
    axes[2].set_title("FEM $\\sigma_y$ [MPa]")
    fig.colorbar(im, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.tight_layout()
    path = out / "icebelt_stress_fields.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_dual_chain(art: dict, out: Path) -> Path:
    rows = art["dual_chain"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    h = [r["h_ice_m"] for r in rows]
    ax.plot(h, [r["chain_a"]["pressure_MPa"] for r in rows], "o-",
            label="chain A (panel-area convention)")
    ax.plot(h, [r["chain_a"]["pressure_contact_patch_convention_MPa"]
                for r in rows], "s--", label="chain A (contact-patch convention)")
    if art["dual_chain"]["chain_b_status"] == "AVAILABLE":
        ax.plot(h, [r["chain_b"]["p_patch_MPa"] for r in rows], "^-",
                lw=2, color="crimson", label="chain B (B1 local line load)")
        ax.plot(h, [r["chain_a"]["sigma_x_max_MPa"] for r in rows], "o:",
                color="tab:blue", alpha=0.6,
                label="chain A panel $\\sigma_x^{max}$ [MPa]")
    ax.set_xlabel("design ice thickness $h_I$ [m]")
    ax.set_ylabel("pressure [MPa]")
    ax.set_title("Same ice condition, both chains")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    dev = [r["deviation_pct"]["p_B_vs_A"] for r in rows]
    devc = [r["deviation_contact_patch_convention_pct"] for r in rows]
    xpos = np.arange(len(h))
    ax.bar(xpos - 0.2, dev, 0.4, label="panel-area convention")
    ax.bar(xpos + 0.2, devc, 0.4, label="contact-patch convention")
    ax.axhline(10.0, color="crimson", ls="--", lw=1.2,
               label="10 % attribution threshold")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{v:.2f}" for v in h])
    ax.set_xlabel("design ice thickness $h_I$ [m]")
    ax.set_ylabel(r"$\Delta p = (p_B-p_A)/p_A$ [%]")
    ax.set_title("Dual-chain deviation (chain B over-predicts)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "icebelt_dual_chain.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    out = Path(args.out)
    figs_dir = out / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    artifact = out / "hull_icebelt_fem_xcheck.json"
    art = xcheck.run_crosscheck(out_path=artifact, repo_root=ROOT)

    figures = []
    for fn, caption in (
        (fig_pressure_area, "IACS pressure-area law vs the B1 chain (chain B has "
                            "no area reduction)"),
        (fig_fem_vs_analytic, "FEM (H3 Hermite) vs closed-form Navier solution "
                              "for the ice patch load"),
        (fig_stress_fields, "PC6 design patch: FEM bending stress fields vs the "
                            "closed-form field"),
        (fig_dual_chain, "Dual-chain deviation over the design ice thickness "
                         "sweep"),
    ):
        path = fn(art, figs_dir)
        figures.append({"path": str(path.relative_to(ROOT)), "caption": caption})
        print("figure:", path)

    summary = {
        "artifact": str(artifact),
        "criteria": {k: bool(v["pass"]) for k, v in art["criteria"].items()},
        "all_criteria_pass": art["all_criteria_pass"],
        "max_fem_vs_closed_form_pct": art["verification"]["max_abs_deviation_pct"],
        "chain_b_status": art["dual_chain"]["chain_b_status"],
        "figures": figures,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if art["all_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())