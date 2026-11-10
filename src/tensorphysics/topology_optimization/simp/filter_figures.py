"""Figures for the SIMP density-filter acceptance node.

Everything is generated from the acceptance JSON / ``designs.npz`` written by
:mod:`.filter_acceptance`, so no number in a figure is hand-copied.

Usage::

    PYTHONPATH=src python -m tensorphysics.topology_optimization.simp.filter_figures \
        artifacts/p156w2_n02_simp_filter
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["make_all"]

_CB = {"unfiltered": "viridis", "filtered": "viridis"}


def _checker_map(rho2d: np.ndarray) -> np.ndarray:
    """|mixed second difference|/2 field (0 = smooth, 0.5 = perfect checker)."""
    d = rho2d[:-1, :-1] - rho2d[:-1, 1:] - rho2d[1:, :-1] + rho2d[1:, 1:]
    out = np.zeros_like(rho2d)
    out[:-1, :-1] = np.abs(d) / 2.0
    return out


def _design(ax, rho2d: np.ndarray, title: str) -> None:
    ax.imshow(rho2d, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest",
              origin="lower", aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def make_all(outdir: str | pathlib.Path) -> list[str]:
    out = pathlib.Path(outdir)
    rec: dict[str, Any] = json.loads((out / "acceptance.json").read_text())
    d = np.load(out / "designs.npz")
    runs = rec["runs"]
    crit = rec["criteria"]
    paths: list[str] = []

    def save(fig, name: str) -> None:
        p = out / name
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))

    levels = ("coarse", "medium", "fine")

    # -- fig 1: MBB designs, protocol B ------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for j, lv in enumerate(levels):
        nx, ny = runs["mbb_beam"][lv]["nelx"], runs["mbb_beam"][lv]["nely"]
        u = d[f"mbb_beam_{lv}_unfiltered"].reshape(ny, nx)
        f = d[f"mbb_beam_{lv}_B"].reshape(ny, nx)
        _design(axes[0, j], u,
                f"MBB {nx}x{ny} unfiltered\nchecker={runs['mbb_beam'][lv]['unfiltered']['checkerboard_index']:.3f}")
        _design(axes[1, j], f,
                f"MBB {nx}x{ny} PDE+Heaviside\nchecker={runs['mbb_beam'][lv]['filtered_B']['checkerboard_index']:.3f}")
    axes[0, 0].set_ylabel("no filter", fontsize=10)
    axes[1, 0].set_ylabel("r_min fixed (B)", fontsize=10)
    fig.suptitle("MBB beam (L/H=3, volfrac=0.4): checkerboard vs regularised design",
                 fontsize=11)
    save(fig, "fig1_mbb_designs.png")

    # -- fig 2: cantilever designs, protocol B -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for j, lv in enumerate(levels):
        nx, ny = runs["cantilever"][lv]["nelx"], runs["cantilever"][lv]["nely"]
        f = d[f"cantilever_{lv}_B"].reshape(ny, nx)
        _design(axes[j], f,
                f"cantilever {nx}x{ny}\nchecker={runs['cantilever'][lv]['filtered_B']['checkerboard_index']:.3f}")
    fig.suptitle("Cantilever (L/H=2, volfrac=0.5), fixed physical r_min (protocol B)",
                 fontsize=11)
    save(fig, "fig2_cantilever_designs.png")

    # -- fig 3: checkerboard index maps ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    nx, ny = runs["mbb_beam"]["fine"]["nelx"], runs["mbb_beam"]["fine"]["nely"]
    for ax, key, label in (
        (axes[0], "mbb_beam_fine_unfiltered", "unfiltered"),
        (axes[1], "mbb_beam_fine_B", "PDE+Heaviside (B)"),
    ):
        field = d[key].reshape(ny, nx)
        im = ax.imshow(_checker_map(field), cmap="inferno", vmin=0.0, vmax=0.5,
                       origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(f"|mixed 2nd difference|/2, MBB 96x32 {label}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, label="alternation index per cell")
    save(fig, "fig3_checkerboard_map.png")

    # -- fig 4: compliance histories ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, name in zip(axes, ("mbb_beam", "cantilever")):
        for lv in levels:
            c = runs[name][lv]["filtered_B"]["compliance"]
            ax.axhline(c, lw=1.0, ls="--", label=f"{lv} filtered")
        ax.set_yscale("log")
        ax.set_xlabel("protocol-A/B runs (reference lines)")
        ax.set_title(f"{name}: converged compliance per level", fontsize=9)
        ax.set_xticks([])
        ax.legend(fontsize=7)
    save(fig, "fig4_compliance_levels.png")

    # -- fig 5: mesh independence (Jaccard + convergence) -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    width = 0.35
    labels = list(rec["benchmarks"].keys())
    for k, tag in enumerate(("B", "A")):
        vals = [crit["C3"]["cases"][n][tag]["jaccard_min"] for n in labels]
        ax.bar(np.arange(len(labels)) + (k - 0.5) * width, vals, width,
               label=("fixed physical r_min (B)" if tag == "B"
                      else "fixed r/h = 1.5 (A)"))
    ax.axhline(0.85, color="crimson", ls="--", lw=1.2, label="criterion 0.85")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("min Jaccard over the three mesh pairs")
    ax.set_title("topology similarity across meshes", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    for name, style in zip(labels, ("o-", "s-")):
        c = [runs[name][lv]["filtered_B"]["compliance"] for lv in levels]
        h = [1.0 / runs[name][lv]["nelx"] for lv in levels]
        ax.loglog(h, c, style, label=f"{name} (B)")
        order = crit["C3"]["cases"][name]["B"]["order_optimised_design"]["order"]
        ax.annotate(f"p={order:.2f}", xy=(h[-1], c[-1]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("h / L")
    ax.set_ylabel("optimal compliance C")
    ax.set_title("compliance vs mesh size (Richardson order)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "fig5_mesh_independence.png")

    # -- fig 6: sensitivity consistency ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, name in zip(axes, labels):
        row = crit["C2"]["cases"][name]["rows"]["filtered_B"]
        ax.semilogy([1], [row["l2_rel"]], "o", label="l2 relative error")
        ax.semilogy([1], [row["mean_rel_top1pct"]], "s", label="mean rel. (active 1%)")
        ax.semilogy([1], [row["max_rel_top1pct"]], "^", label="max rel. (active 1%)")
        ax.axhline(1e-2, color="crimson", ls="--", lw=1.2, label="1% criterion")
        ax.set_xticks([1], [f"{crit['C2']['cases'][name]['n_elems']} elements"])
        ax.set_ylabel("relative error")
        ax.set_title(f"{name}: adjoint vs central FD", fontsize=9)
        ax.legend(fontsize=7)
    save(fig, "fig6_sensitivity_consistency.png")

    # -- fig 7: filter comparison at the medium level ----------------------
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for i, name in enumerate(labels):
        r = runs[name]["medium"]
        nx, ny = r["nelx"], r["nely"]
        for j, (key, dkey) in enumerate((
                ("unfiltered", "unfiltered"), ("linear_hat", "linear_hat"),
                ("density_heaviside", "density_heaviside"), ("filtered_B", "B"))):
            rho = d[f"{name}_medium_{dkey}"].reshape(ny, nx)
            _design(axes[i, j], rho,
                    f"{name} {key}\nchecker={r[key]['checkerboard_index']:.3f} "
                    f"grey={r[key]['grey_fraction']:.2f}")
    fig.suptitle("regulariser comparison at the medium level "
                 "(checkerboard index and grey fraction are measured)", fontsize=11)
    save(fig, "fig7_regularisers.png")

    # -- fig 8: C5 negative control ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    nx = runs["mbb_beam"]["medium"]["nelx"]
    ny = runs["mbb_beam"]["medium"]["nely"]
    for ax, key, lab in (
        (axes[0], "c5_radius_below_h", "r_min/h = %.1f" % crit["C5"]["radius_below_h"]),
        (axes[1], "c5_radius_above_h", "r_min/h = %.1f" % crit["C5"]["radius_above_h"]),
    ):
        _design(ax, d[key].reshape(ny, nx), f"MBB 72x24 {lab}")
    fig.suptitle("negative control: filter radius below one element size", fontsize=10)
    save(fig, "fig8_radius_below_h.png")
    return paths


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts/p156w2_n02_simp_filter"
    for p in make_all(target):
        print(p)