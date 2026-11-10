#!/usr/bin/env python3
"""IN157-N05 study driver: acceptance run + publication figures.

    PYTHONPATH=src python3 scripts/simp_in157_study.py --workers 8

Writes ``artifacts/simp_in157_n05/verification.json``, ``designs.npz`` and five
figures.  Every number in the figures comes from that JSON / NPZ, so the report
cannot drift from the run.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorphysics.topology_optimization.simp.acceptance_in157 import (  # noqa: E402
    alternating_cell_fraction,
    run_acceptance,
)

OUT = pathlib.Path("artifacts/simp_in157_n05")
REF_RADIUS = 1.5


def _alt_map(field: np.ndarray) -> np.ndarray:
    """Cell-wise alternation indicator (same rule as the scalar index)."""
    a, b = field[:-1, :-1], field[1:, 1:]
    c, d = field[:-1, 1:], field[1:, :-1]
    block = (np.abs(a - b) < 0.5) & (np.abs(c - d) < 0.5) & (np.abs(a - c) >= 0.5)
    mask = np.zeros_like(field, dtype=float)
    mask[:-1, :-1] += block
    mask[1:, 1:] += block
    mask[:-1, 1:] += block
    mask[1:, :-1] += block
    return mask


def figure_designs(rec, arrays, path):
    unf = next(r for r in rec["runs"] if r["tag"] == "unfiltered" and r["seed"] == 0)
    fil = next(r for r in rec["runs"] if r["tag"].startswith("filtered_r"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, run, label in (
        (axes[0], unf, "no filter"),
        (axes[1], fil, f"density filter r = {REF_RADIUS:g}"),
    ):
        rho = arrays[f"{run['tag']}_seed{run['seed']}_rho"]
        im = ax.imshow(rho, cmap="gray_r", vmin=0, vmax=1, aspect="equal",
                       interpolation="nearest")
        ax.set_title(
            f"{label}: C = {run['compliance']:.2f}, "
            f"alternating index = {run['alternating_fraction_raw']:.4f}, "
            f"grey fraction = {run['grey_fraction']:.3f}"
        )
        ax.set_xlabel("element x")
        ax.set_ylabel("element y")
        fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\rho$")
    fig.suptitle("MBB beam 100x50, volfrac 0.5, SIMP p = 3 (IN157-N05)", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_alternation(rec, arrays, path):
    runs = [
        next(r for r in rec["runs"] if r["tag"] == "unfiltered" and r["seed"] == 0),
        next(r for r in rec["runs"] if r["tag"].startswith("filtered_r")),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, run, label in (
        (axes[0], runs[0], "no filter"),
        (axes[1], runs[1], f"density filter r = {REF_RADIUS:g}"),
    ):
        rho = arrays[f"{run['tag']}_seed{run['seed']}_rho"]
        mask = _alt_map(rho)
        im = ax.imshow(mask, cmap="inferno", vmin=0, vmax=1, aspect="equal",
                       interpolation="nearest")
        ax.set_title(
            f"{label}: alternating-cell map "
            f"(index = {alternating_cell_fraction(rho):.4f})"
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="2x2 alternating block membership")
    fig.suptitle("Checkerboard detector field: literal 2x2 alternation map", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_radius_scan(rec, arrays, path):
    c3 = rec["criteria"]["C3"]
    radii = np.asarray(c3["radii"], dtype=float)
    comp = np.asarray(c3["compliance"], dtype=float)
    idx = [next(run for run in rec["runs"] if run["tag"] == f"scan_r{r:g}")["alternating_fraction_raw"]
           for r in radii]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(radii, comp, "o-", color="tab:blue", label="compliance C(r)")
    ax.set_xlabel("filter radius r / element length")
    ax.set_ylabel("compliance C", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(radii, idx, "s--", color="tab:red", label="alternating index")
    ax2.set_ylabel("alternating-cell fraction", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title(
        f"Radius scan (Spearman rho = {c3['spearman_rho']:.4f}, "
        f"monotone increasing = {c3['monotone_increasing']})"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_history(rec, arrays, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), constrained_layout=True, sharex=True)
    series = [
        (next(r for r in rec["runs"] if r["tag"] == "unfiltered" and r["seed"] == 0),
         "no filter", "tab:red"),
        (next(r for r in rec["runs"] if r["tag"].startswith("filtered_r")),
         f"r = {REF_RADIUS:g}", "tab:blue"),
    ]
    for run, label, colour in series:
        hist = run["history"]
        it = [h["iteration"] for h in hist]
        axes[0].plot(it, [h["compliance"] for h in hist], color=colour, label=label)
        axes[1].plot(it, [h["checkerboard_index"] for h in hist], color=colour, label=label)
    axes[0].set_ylabel("compliance C")
    axes[1].set_ylabel("continuous checkerboard index")
    axes[1].set_xlabel("iteration")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Convergence history: compliance and roughness")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_negative_control(rec, arrays, path):
    neg = next(r for r in rec["runs"] if r["tag"] == "negative_control")
    unf = next(r for r in rec["runs"] if r["tag"] == "unfiltered" and r["seed"] == 0)
    rho_neg = arrays[f"negative_control_seed{neg['seed']}_rho"]
    rho_unf = arrays[f"unfiltered_seed{unf['seed']}_rho"]
    diff = np.abs(rho_neg - rho_unf)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    im = axes[0].imshow(diff, cmap="viridis", aspect="equal", interpolation="nearest")
    axes[0].set_title(
        f"|rho(r->0) - rho(no filter)| max = {diff.max():.2e}"
    )
    axes[0].set_xlabel("element x")
    axes[0].set_ylabel("element y")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="|density difference|")
    c4 = rec["criteria"]["C4"]
    axes[1].bar(["r -> 0", "no filter"], [c4["compliance"], c4["compliance_reference"]],
                color=["tab:orange", "tab:gray"])
    axes[1].set_ylabel("compliance C")
    axes[1].set_title(f"negative control: rel. dev = {c4['relative_deviation']:.2e}")
    axes[1].grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tol", type=float, default=5e-3)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.outdir)
    if args.figures_only:
        rec = json.loads((out / "verification.json").read_text())
    else:
        rec = run_acceptance(out, workers=args.workers, iters=args.iters, tol=args.tol)
    arrays = np.load(out / "designs.npz")

    figure_designs(rec, arrays, out / "fig1_mbb_designs.png")
    figure_alternation(rec, arrays, out / "fig2_alternation_map.png")
    figure_radius_scan(rec, arrays, out / "fig3_radius_scan.png")
    figure_history(rec, arrays, out / "fig4_history.png")
    figure_negative_control(rec, arrays, out / "fig5_negative_control.png")

    summary = {
        "verdict": rec["verdict"],
        "criteria": {
            k: {kk: v.get(kk) for kk in
                ("status", "ratio", "ratio_mean", "index_no_filter_mean", "index_filtered",
                 "suppression_ratio", "spearman_rho", "monotone_increasing",
                 "relative_deviation", "value")}
            for k, v in rec["criteria"].items()
        },
        "figures": [str(out / name) for name in
                    ("fig1_mbb_designs.png", "fig2_alternation_map.png",
                     "fig3_radius_scan.png", "fig4_history.png",
                     "fig5_negative_control.png")],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())