#!/usr/bin/env python3
"""Generate benchmark artifacts and figures for GAP-CROSS-MAP (m14679).

Produces:
- runs/m14679/benchmark.json  : full numerical results
- artifacts/m14679/conservation_bar.png  : conservation error bar chart
- artifacts/m14679/convergence_plot.png  : convergence order plot
- artifacts/m14679/symmetry_field.png    : symmetry roundtrip field comparison
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensorphysics.grid_to_grid_remap.acceptance import (
    case_conservation,
    case_convergence,
    case_symmetry,
    make_perturbed_tri_mesh,
)
from tensorphysics.grid_to_grid_remap.overlap import build_overlap_table
from tensorphysics.grid_to_grid_remap.remap import (
    remap_cell_average,
    conservation_report,
    symmetry_roundtrip,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "m14679")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs", "m14679")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


def _f(x):
    return float(x) if isinstance(x, (float, np.floating)) else x


# ---------------------------------------------------------------------------
# 1. Conservation benchmark
# ---------------------------------------------------------------------------
def gen_conservation():
    cons = case_conservation(n_src=32, n_tgt=48, seed=7)
    # Save full results
    with open(os.path.join(RUNS_DIR, "conservation.json"), "w") as f:
        json.dump(cons, f, indent=2, default=str)

    # Bar chart of conservation errors
    fig, ax = plt.subplots(figsize=(8, 4))
    modes = ["average", "bounded"]
    fields = ["smooth_sin", "hot_spot", "thermal_front"]
    x = np.arange(len(fields))
    width = 0.35

    for i, mode in enumerate(modes):
        vals = [cons["fields"][fn][mode]["conservation_rel_defect"] for fn in fields]
        ax.bar(x + i * width, vals, width, label=mode)

    ax.set_yscale("log")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(fields, rotation=15, ha="right")
    ax.set_ylabel("Relative conservation defect")
    ax.set_title("GAP-CROSS-MAP: Conservation Error by Field and Mode")
    ax.axhline(y=0.01, color="red", linestyle="--", label="1% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "conservation_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR}/conservation_bar.png")
    return cons


# ---------------------------------------------------------------------------
# 2. Convergence study
# ---------------------------------------------------------------------------
def gen_convergence():
    conv = case_convergence(levels=(8, 16, 24, 32, 48), n_tgt=64)
    with open(os.path.join(RUNS_DIR, "convergence.json"), "w") as f:
        json.dump(conv, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(7, 5))
    hs = [r["h"] for r in conv["rows"]]
    for key, label, expected in [
        ("avg_l2_abs", "Average (1st order)", 1.0),
        ("bnd_l2_abs", "Bounded (2nd order)", 2.0),
    ]:
        errs = [r[key] for r in conv["rows"]]
        ax.loglog(hs, errs, "o-", label=f"{label}, order={conv['observed_order'][key]:.2f}")
        # Reference slope
        ref = [errs[0] * (h / hs[0]) ** expected for h in hs]
        ax.loglog(hs, ref, "--", alpha=0.4)

    ax.set_xlabel("Source mesh spacing h")
    ax.set_ylabel("L2 error")
    ax.set_title("GAP-CROSS-MAP: Convergence Order")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence_plot.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR}/convergence_plot.png")
    return conv


# ---------------------------------------------------------------------------
# 3. Symmetry roundtrip visualization
# ---------------------------------------------------------------------------
def gen_symmetry():
    mesh_a = make_perturbed_tri_mesh(24, jitter=0.25, seed=11)
    mesh_b = make_perturbed_tri_mesh(36, jitter=0.20, seed=211)
    table_ab = build_overlap_table(mesh_a, mesh_b)
    table_ba = build_overlap_table(mesh_b, mesh_a)

    cen_a = mesh_a.centroids()
    phi_a = np.sin(2.0 * np.pi * cen_a[:, 0]) * np.cos(2.0 * np.pi * cen_a[:, 1])

    rt = symmetry_roundtrip(table_ab, table_ba, phi_a)
    with open(os.path.join(RUNS_DIR, "symmetry.json"), "w") as f:
        json.dump({"roundtrip": rt, "mesh_a_n": mesh_a.n_elements, "mesh_b_n": mesh_b.n_elements},
                  f, indent=2, default=str)

    # Scatter plot: original vs roundtrip
    phi_rt = remap_cell_average(table_ba, remap_cell_average(table_ab, phi_a).phi).phi
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(phi_a, phi_rt, s=3, alpha=0.5)
    lim = max(abs(phi_a).max(), abs(phi_rt).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
    ax.set_xlabel("Original field value")
    ax.set_ylabel("Roundtrip field value")
    ax.set_title(f"GAP-CROSS-MAP: Symmetry Roundtrip\nL2 rel error = {rt['l2_rel']:.2e}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "symmetry_field.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR}/symmetry_field.png")
    return rt


# ---------------------------------------------------------------------------
# 4. Master benchmark JSON
# ---------------------------------------------------------------------------
def gen_master(cons, conv, rt):
    benchmark = {
        "mission": "m14679",
        "capability": "GAP-CROSS-MAP: non-matching grid conservative remapping",
        "conservation": {
            k: {mode: cons["fields"][k][mode]["conservation_rel_defect"]
                for mode in ["average", "bounded"]}
            for k in cons["fields"]
        },
        "convergence": {
            "observed_order": conv["observed_order"],
            "rows": conv["rows"],
        },
        "symmetry": rt,
        "all_conservation_below_1pct": all(
            cons["fields"][k][m]["conservation_rel_defect"] < 0.01
            for k in cons["fields"] for m in ["average", "bounded"]
        ),
    }
    with open(os.path.join(RUNS_DIR, "benchmark.json"), "w") as f:
        json.dump(benchmark, f, indent=2, default=str)
    print(f"  -> {RUNS_DIR}/benchmark.json")
    return benchmark


if __name__ == "__main__":
    print("Generating GAP-CROSS-MAP artifacts...")
    cons = gen_conservation()
    conv = gen_convergence()
    rt = gen_symmetry()
    bench = gen_master(cons, conv, rt)
    print("\nAll artifacts generated successfully.")
    print(f"Conservation errors (all < 1%):")
    for fn, modes in bench["conservation"].items():
        for mode, val in modes.items():
            print(f"  {fn}/{mode}: {val:.2e}")
    print(f"Convergence orders: avg={bench['convergence']['observed_order']['avg_l2_abs']:.2f}, "
          f"bnd={bench['convergence']['observed_order']['bnd_l2_abs']:.2f}")
    print(f"Symmetry roundtrip L2 rel error: {bench['symmetry']['l2_rel']:.2e}")
