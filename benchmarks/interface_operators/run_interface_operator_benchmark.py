"""Benchmark + figure generator for the interface-operator conservation contract.

Produces, under ``artifacts/m5207/``:

* ``interface_operator_acceptance.json``  -- every acceptance number (C1..C6),
  the per-(operator, field) table, the two chains and the environment block;
* ``fig1_order_convergence.png``          -- L2 error vs h for the five fields;
* ``fig2_conservation_residuals.png``     -- conservation residual per operator/field;
* ``fig3_boundary_layer_transfer.png``    -- transferred vs analytic steep field;
* ``fig4_chain_field_diff.png``           -- chain field differences vs the 2 % gate.

Run:  PYTHONPATH=src python3 benchmarks/interface_operators/run_interface_operator_benchmark.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # noqa: E402  (headless)
import matplotlib.pyplot as plt             # noqa: E402

from tensorphysics.interface_operators.acceptance import (  # noqa: E402
    RESOLUTIONS, fvm_lbm_chain, hierarchy, lbm_dem_chain, operator_suite,
    run_acceptance)
from tensorphysics.interface_operators.core import (  # noqa: E402
    ANALYTIC_FIELDS, exact_values, uniform_mesh)

COLORS = {"mortar": "#1f77b4", "scatter": "#d62728", "me": "#2ca02c"}
LABELS = {"mortar": "mortar projection", "scatter": "RBF scatter",
          "me": "LBM-DEM ME"}
FIELD_STYLE = {
    "constant": ("o", "-"),
    "linear": ("s", "--"),
    "quadratic": ("^", "-."),
    "harmonic": ("d", ":"),
    "boundary_layer": ("v", "-"),
}


def order_curves(name: str, op) -> dict:
    """(resolution, h, L2 error) per field for one operator."""
    pairs = []
    for n in RESOLUTIONS:
        if name == "me":
            from tensorphysics.interface_operators.acceptance import particle_lattice
            src = uniform_mesh(0.2, 0.8, n)
            pairs.append((src, op.dem_pointset(src, particle_lattice(n, 0.2, 0.8))))
        else:
            pairs.append(hierarchy(n))
    out = {}
    for fld in ANALYTIC_FIELDS:
        est = op.observed_order(fld, pairs)
        out[fld.name] = {
            "resolutions": list(est.resolutions),
            "errors": list(est.errors),
            "h": [float(np.mean(t.measures)) if hasattr(t, "widths")
                  else float(np.mean(t.measures)) for _, t in pairs],
            "order_fit": est.order,
            "exact": est.exact,
        }
    return out


def fig_order(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axes, ("mortar", "scatter")):
        data = curves[name]
        for fld in ANALYTIC_FIELDS:
            d = data[fld.name]
            marker, ls = FIELD_STYLE[fld.name]
            ax.loglog(d["h"], np.maximum(d["errors"], 1e-17), marker=marker,
                      ls=ls, label=f"{fld.name}", ms=5, lw=1.4)
        h = np.array(sorted(data["harmonic"]["h"]))
        ax.loglog(h, 3e-3 * (h / h[0]) ** 2, "k--", lw=0.8, label=r"$h^2$ reference")
        ax.loglog(h, 3e-2 * (h / h[0]), "k:", lw=0.8, label=r"$h$ reference")
        ax.set_xlabel("interface cell size h")
        ax.set_ylabel(r"relative $L_2$ error")
        ax.set_title(f"{LABELS[name]}: observed order")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Interface transfer error vs resolution (analytic fields)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_conservation(payload: dict, path: Path) -> None:
    fields = [f.name for f in ANALYTIC_FIELDS]
    ops = list(payload["operator_table"])
    x = np.arange(len(fields))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, name in enumerate(ops):
        vals = [max(payload["operator_table"][name][f]["conservation_residual"], 1e-18)
                for f in fields]
        ax.bar(x + (k - 1) * width, vals, width, label=LABELS[name],
               color=COLORS[name], log=True)
    ax.axhline(1e-14, color="k", ls="--", lw=1.0, label="C1 gate 1e-14")
    ax.axhline(1e-12, color="r", ls=":", lw=1.0, label="C3 gate 1e-12")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fields], fontsize=9)
    ax.set_ylabel("|flux_out - flux_in| / sum|flux|")
    ax.set_ylim(1e-18, 1e-8)
    ax.set_title("Conservation residual per operator and test field")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_boundary_layer(path: Path) -> dict:
    """Steep field transferred from a graded donor to a uniform acceptor."""
    fld = ANALYTIC_FIELDS[4]
    src, tgt = hierarchy(RESOLUTIONS[-1])
    u = exact_values(src, fld)
    ops = operator_suite()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xx = np.linspace(0.0, 0.12, 400)
    ax.plot(xx, fld.func(xx), "k-", lw=2.0, label="analytic exp(-x/0.06)")
    extra = {}
    for name in ("mortar", "scatter"):
        v = ops[name].transfer(src, u, tgt)
        ax.plot(tgt.centers, v, "o--", ms=4, lw=1.2, color=COLORS[name],
                label=f"{LABELS[name]} (L2={ops[name].l2_error(tgt, v, fld):.1e})")
        extra[name] = float(ops[name].l2_error(tgt, v, fld))
    nodes = uniform_mesh(0.2, 0.8, 160)
    from tensorphysics.interface_operators.acceptance import particle_lattice
    parts = ops["me"].dem_pointset(nodes, particle_lattice(160, 0.2, 0.8))
    v = ops["me"].gather(nodes, exact_values(nodes, fld), parts)
    ax.plot(parts.centers, v, ".", ms=3, color=COLORS["me"], alpha=0.7,
            label=f"{LABELS['me']} (L2={ops['me'].l2_error(parts, v, fld):.1e})")
    extra["me"] = float(ops["me"].l2_error(parts, v, fld))
    ax.set_xlim(0.0, 0.12)
    ax.set_xlabel("interface coordinate x")
    ax.set_ylabel("u")
    ax.set_title("Steep (boundary-layer) field: transfer vs analytic reference")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return extra


def fig_chains(payload: dict, path: Path) -> None:
    labels, values, colors = [], [], []
    for row in payload["chains"]:
        if row["field"] != "boundary_layer":
            continue
        labels.append(row["chain"] + "\n(boundary layer)")
        values.append(row["field_diff"])
        colors.append("#1f77b4")
    for row in payload["chain_sensitivity_clustered_dem"]:
        if row["field"] != "boundary_layer":
            continue
        labels.append("LBM<->DEM clustered\n(boundary layer)")
        values.append(row["field_diff"])
        colors.append("#ff7f0e")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.02, color="r", ls="--", lw=1.2, label="C5 gate 2 %")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.2%}",
                ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("post-exchange relative field difference")
    ax.set_title("Two-solver chains: field difference after conservative exchange")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/m5207")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = run_acceptance(verbose=True)
    ops = operator_suite()
    curves = {name: order_curves(name, op) for name, op in ops.items()}
    payload["order_curves"] = curves

    fig_order(curves, out / "fig1_order_convergence.png")
    fig_conservation(payload, out / "fig2_conservation_residuals.png")
    bl = fig_boundary_layer(out / "fig3_boundary_layer_transfer.png")
    fig_chains(payload, out / "fig4_chain_field_diff.png")

    payload["artifacts"] = {
        "json": str(out / "interface_operator_acceptance.json"),
        "figures": [
            {"name": "fig1_order_convergence.png",
             "caption": "relative L2 transfer error vs interface cell size for the five "
                        "analytic fields; h^2 and h reference slopes"},
            {"name": "fig2_conservation_residuals.png",
             "caption": "flux conservation residual per operator and field against the "
                        "C1 (1e-14) and C3 (1e-12) gates"},
            {"name": "fig3_boundary_layer_transfer.png",
             "caption": "steep boundary-layer field: transferred values vs the analytic "
                        "profile (graded donor -> uniform acceptor)"},
            {"name": "fig4_chain_field_diff.png",
             "caption": "post-exchange field difference for the FVM<->LBM and LBM<->DEM "
                        "chains against the 2 % C5 gate"},
        ],
    }
    payload["figure_l2_errors_boundary_layer"] = bl
    (out / "interface_operator_acceptance.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"out": str(out), "all_passed": payload["all_passed"],
                      "artifacts": [p.name for p in sorted(out.iterdir())]}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())