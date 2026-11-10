"""Figures for the SIMP acceptance report (matplotlib, headless Agg).

Everything is generated from the artefacts written by
:mod:`tensorphysics.topology_optimization.simp.verification` -- no number in a
figure is typed by hand.
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence

import numpy as np

__all__ = ["make_all"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _checkerboard_map(rho2d: np.ndarray) -> np.ndarray:
    d = np.zeros_like(rho2d)
    d[:-1, :-1] = rho2d[:-1, :-1] - rho2d[:-1, 1:] - rho2d[1:, :-1] + rho2d[1:, 1:]
    return np.abs(d) / 2.0


def make_all(
    npz_path: str | pathlib.Path,
    json_path: str | pathlib.Path,
    outdir: str | pathlib.Path,
) -> list[str]:
    plt = _plt()
    npz = np.load(npz_path)
    record = json.loads(pathlib.Path(json_path).read_text())
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def save(fig, name: str) -> None:
        path = out / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(str(path))

    # 1. MBB beam density fields -------------------------------------------
    panels = [
        ("mbb_unfiltered", npz["mbb_unfiltered"], "no filter, seed 0"),
        ("mbb_unfiltered_seed1", npz["mbb_unfiltered_seed1"], "no filter, seed 1"),
        ("mbb_filtered", npz["mbb_filtered"], "density filter r = 1.5"),
    ]
    ci = {r["seed"] if r["radius"] is None else "f": r["checkerboard_index"]
          for r in record["runs"]["mbb"] if r["nelx"] == record["mbb"]["nelx"]}
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    for ax, (key, arr, label) in zip(axes, panels):
        nelx = record["mbb"]["nelx"]
        nely = record["mbb"]["nely"]
        ax.imshow(arr.reshape(nely, nelx), cmap="gray_r", origin="lower",
                  extent=(0, nelx / nely, 0, 1), vmin=0, vmax=1, aspect="equal")
        c = ci.get(0 if "seed 1" not in label else 1, np.nan) if "no filter" in label else ci.get("f", np.nan)
        ax.set_title(f"{label}  (checkerboard index = {c:.3f})", fontsize=10)
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x  (length / height = 2)")
    fig.suptitle("MBB beam, 100x50, volfrac 0.5, p = 3: density filter suppresses "
                 "element-wise alternation", fontsize=11)
    save(fig, "fig1_mbb_designs.png")

    # 2. checkerboard indicator maps ---------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
    nelx, nely = record["mbb"]["nelx"], record["mbb"]["nely"]
    for ax, key, label in (
        (axes[0], "mbb_unfiltered", "no filter"),
        (axes[1], "mbb_filtered", "density filter r = 1.5"),
    ):
        m = _checkerboard_map(npz[key].reshape(nely, nelx))
        im = ax.imshow(m, cmap="magma", origin="lower", vmin=0, vmax=1,
                       extent=(0, nelx / nely, 0, 1), aspect="equal")
        ax.set_title(f"{label}: |mixed 2nd difference| / 2 (checkerboard indicator)",
                     fontsize=10)
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.8, label="local alternation")
    save(fig, "fig2_checkerboard_map.png")

    # 3. compliance history -------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run in record["runs"]["mbb"]:
        h = run["history"]
        it = [r["iteration"] for r in h]
        tag = (f"{run['nelx']}x{run['nely']} "
               f"{'filtered' if run['radius'] else 'no filter'} seed {run['seed']}")
        ax1.plot(it, [r["compliance"] for r in h], label=tag)
        ax2.plot(it, [r["checkerboard_index"] for r in h], label=tag)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("compliance")
    ax1.set_yscale("log")
    ax1.set_title("compliance history")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("checkerboard index")
    ax2.set_title("checkerboard index history")
    ax2.legend(fontsize=7)
    save(fig, "fig3_history.png")

    # 4. sensitivity scatter + error histogram ------------------------------
    sens = record["sensitivity"]
    an = npz["cantilever_sensitivity"]
    fd = npz["cantilever_fd"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    pos = (np.abs(an) > 0) & (np.abs(fd) > 0)
    ax1.loglog(np.abs(an[pos]), np.abs(fd[pos]), ".", ms=3, alpha=0.6)
    lim = [max(np.abs(an[pos]).min(), 1e-12), np.abs(an).max() * 1.2]
    ax1.plot(lim, lim, "k--", lw=1, label="y = x")
    ax1.set_xlabel("|dC/drho| adjoint")
    ax1.set_ylabel("|dC/drho| central FD (Delta rho = 1e-6)")
    ax1.set_title(f"{sens['design']['nelx']}x{sens['design']['nely']} cantilever, "
                  f"{sens['n_elements']} elements")
    ax1.legend()
    err = np.abs(an - fd)
    ax2.hist(err, bins=50, color="steelblue")
    ax2.axvline(sens["with_density_filter"]["mean_abs"], color="k", ls="--",
                label=f"mean = {sens['with_density_filter']['mean_abs']:.2e}")
    ax2.axvline(1e-4, color="r", ls=":", label="C3 threshold 1e-4")
    ax2.set_xlabel("|analytic - FD|")
    ax2.set_ylabel("element count")
    ax2.set_yscale("log")
    ax2.set_title("absolute sensitivity error")
    ax2.legend(fontsize=8)
    save(fig, "fig4_sensitivity_scatter.png")

    # 5. spatial error map ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.2))
    m = err.reshape(sens["design"]["nely"], sens["design"]["nelx"])
    im = ax.imshow(np.log10(np.maximum(m, 1e-12)), cmap="viridis", origin="lower",
                   aspect="auto")
    fig.colorbar(im, ax=ax, label="log10 |analytic - FD|")
    ax.set_title("adjoint vs central-FD sensitivity error (density-filtered cantilever)")
    ax.set_xlabel("element x index")
    ax.set_ylabel("element y index")
    save(fig, "fig5_error_map.png")

    # 6. step convergence ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    real = record["criteria"]["C5"]["scheme_real_forward"]["rows"]
    cx = record["criteria"]["C5"]["scheme_complex_step"]["rows"]
    ax.loglog([r["delta"] for r in real], [r["mean_abs"] for r in real], "o-",
              label="real forward difference")
    ax.loglog([r["delta"] for r in cx], [r["mean_abs"] for r in cx], "s-",
              label="complex-step difference")
    d = np.array([r["delta"] for r in cx])
    ax.loglog(d, d * (cx[0]["mean_abs"] / d[0]), "k:", lw=1, label="O(Delta rho)")
    ax.set_xlabel(r"$\Delta\rho$")
    ax.set_ylabel("mean |analytic - FD|")
    ax.set_title("finite-difference step refinement (50x25 cantilever)")
    ax.invert_xaxis()
    ax.legend()
    save(fig, "fig6_convergence.png")

    # 7. cantilever design ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(npz["cantilever_filtered"].reshape(sens["design"]["nely"],
                                                 sens["design"]["nelx"]),
              cmap="gray_r", origin="lower", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"cantilever 50x25, volfrac 0.4, p = 3, r = 1.5 "
                 f"(C = {sens['design']['compliance']:.2f})")
    ax.set_xlabel("element x index")
    ax.set_ylabel("element y index")
    save(fig, "fig7_cantilever_design.png")

    # 8. mesh study ----------------------------------------------------------
    ms = record["mesh_study"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = [f"{ms['coarse_mesh'][0]}x{ms['coarse_mesh'][1]}",
              f"{ms['fine_mesh'][0]}x{ms['fine_mesh'][1]}"]
    x = np.arange(2)
    ax.bar(x - 0.18, [ms["unfiltered"]["C_coarse"], ms["unfiltered"]["C_fine"]],
           width=0.36, label="no filter")
    ax.bar(x + 0.18, [ms["filtered"]["C_coarse"], ms["filtered"]["C_fine"]],
           width=0.36, label="filtered r = 1.5")
    ax.set_xticks(x, labels)
    ax.set_ylabel("compliance")
    ax.set_title("mesh dependence of the optimum (MBB, volfrac 0.5)")
    ax.legend()
    save(fig, "fig8_mesh_study.png")

    return made


if __name__ == "__main__":  # pragma: no cover
    import sys

    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/simp_m9136")
    print(make_all(base / "designs.npz", base / "verification.json", base))