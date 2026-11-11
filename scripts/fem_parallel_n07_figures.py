#!/usr/bin/env python3
"""Figures for the P284W3-N07 scaling study (all data from the JSON artifact).

Usage::

    PYTHONPATH=src python scripts/fem_parallel_n07_figures.py \
        --json artifacts/fem_parallel_n07/scaling.json --outdir docs/figures/n07

Every number plotted is read from ``scaling.json``; the only computed extra is
the analytic manufacturing-solution field used for the error map, which is
recomputed from the problem definition (never hand-copied).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _pts(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(section.get("points", []), key=lambda p: p["ranks"])


def fig_strong(section, path):
    pts = _pts(section)
    P = [p["ranks"] for p in pts]
    S = [p["speedup"] for p in pts]
    E = [p["efficiency"] for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(P, P, "k--", lw=1, label="ideal $S(P)=P$")
    ax[0].plot(P, S, "o-", label="measured $S(P)$")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("ranks $P$")
    ax[0].set_ylabel("speedup $S(P)=T(1)/T(P)$")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend()
    ax[1].plot(P, E, "s-", color="tab:red")
    ax[1].axhline(0.60, ls=":", color="k", lw=1)
    ax[1].annotate("acceptance 0.60", xy=(P[-1], 0.60), xytext=(-70, 6),
                   textcoords="offset points", fontsize=8)
    ax[1].set_xscale("log", base=2)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("ranks $P$")
    ax[1].set_ylabel("efficiency $E(P)=S(P)/P$")
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Strong scaling, N=%d DOF, %d fixed iterations"
                 % (section["n_dofs"], section["iters"]), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_comm(section, path):
    pts = _pts(section)
    P = [p["ranks"] for p in pts]
    frac = [100 * (p["comm_fraction"] or 0.0) for p in pts]
    per_iter = [p["per_iter_ms"] for p in pts]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(P, frac, "o-", color="tab:purple")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ranks $P$")
    ax.set_ylabel("communication share $T_{comm}/T_{solve}$ [%]")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.plot(P, per_iter, "s--", color="tab:green", alpha=0.7)
    ax2.set_ylabel("solver time per iteration [ms]", color="tab:green")
    ax.set_title("Communication overhead (fused collectives)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_noise(section, path):
    pts = _pts(section)
    P = [p["ranks"] for p in pts]
    noise = [p["noise_ratio"] for p in pts]
    load = [p["load1"] for p in pts]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(P, noise, "o-", color="tab:brown", label="max/min over repetitions")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ranks $P$")
    ax.set_ylabel("timing spread (max/min)")
    ax.axhline(1.0, ls=":", color="k", lw=1)
    ax2 = ax.twinx()
    ax2.plot(P, load, "s--", color="tab:gray", alpha=0.7)
    ax2.set_ylabel("host load average at measure time", color="tab:gray")
    ax.set_title("Contention diagnostics: why min-of-reps is required", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_weak(section, path):
    pts = _pts(section)
    P = [p["ranks"] for p in pts]
    W = [p["weak_ratio"] for p in pts]
    Wi = [p["weak_ratio_per_iter"] or 1.0 for p in pts]
    iters = [p["iterations"] for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(P, W, "o-", label="converged solve $T(P)/T(1)$")
    ax[0].plot(P, Wi, "s--", label="per iteration (parallel overhead only)")
    ax[0].axhline(1.0, ls=":", color="k", lw=1)
    ax[0].axhline(1.5, ls=":", color="tab:red", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("ranks $P$")
    ax[0].set_ylabel("weak-scaling ratio $W(P)$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(P, iters, "o-", color="tab:orange")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("ranks $P$")
    ax[1].set_ylabel("Krylov iterations to $10^{-8}$")
    ax[1].set_title("iteration count vs $P$ (N/P fixed)", fontsize=9)
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("Weak scaling, %d DOF per rank, %s preconditioner"
                 % (section["n_per_rank"], section["precond"]), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_3d(section, path):
    pts = _pts(section)
    P = [p["ranks"] for p in pts]
    E = [p["efficiency"] for p in pts]
    S = [p["speedup"] for p in pts]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(P, P, "k--", lw=1, label="ideal")
    ax.plot(P, S, "o-", label="$S(P)$")
    ax.plot(P, E, "s-", label="$E(P)$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ranks $P$ (Cartesian %s decomposition)"
                  % "/".join("%d" % v for v in pts[-1]["decomposition"]))
    ax.set_ylabel("speedup / efficiency")
    ax.set_title("3-D Poisson, grid %s (N=%d DOF)"
                 % ("x".join(str(v) for v in section["grid"]), section["n_dofs"]), fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_fields(outdir: str) -> str:
    """Parallel solution field, exact field and the error map (manufactured case)."""
    from tensorphysics.fem_parallel.comm import SerialComm
    from tensorphysics.fem_parallel.operator import DistributedSystem
    from tensorphysics.fem_parallel.partition import CartesianPartition
    from tensorphysics.fem_parallel.pcg import JacobiPreconditioner, cg
    from tensorphysics.fem_parallel.problems import make_problem

    pr = make_problem("poisson_unit_source", (129, 129))
    system = DistributedSystem(SerialComm(), pr.grid, CartesianPartition(pr.grid.n, 1), pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    res = cg(A, b, system, tol=1e-12, maxiter=20000,
             precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
    u = res.x.reshape(pr.grid.n)
    xy = pr.grid.coords(np.arange(pr.grid.num_nodes))
    exact = np.asarray(pr.exact(xy)).reshape(pr.grid.n)
    err = u - exact
    X = xy[:, 0].reshape(pr.grid.n)
    Y = xy[:, 1].reshape(pr.grid.n)
    rel = float(np.linalg.norm(err) / np.linalg.norm(exact))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    cs = ax[0].contourf(X, Y, u, levels=24)
    fig.colorbar(cs, ax=ax[0]); ax[0].set_title("Q1 solution $u_h$")
    cs = ax[1].contourf(X, Y, exact, levels=24)
    fig.colorbar(cs, ax=ax[1]); ax[1].set_title("analytic $u$")
    cs = ax[2].contourf(X, Y, err, levels=24, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax[2])
    ax[2].set_title("error $u_h-u$ (rel. $L_2$=%.2e)" % rel)
    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(outdir, "fields.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/fem_parallel_n07/scaling.json")
    ap.add_argument("--outdir", default="docs/figures/n07")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    data = json.load(open(args.json))
    made = []
    if data.get("strong"):
        p = os.path.join(args.outdir, "strong_scaling.png")
        fig_strong(data["strong"], p); made.append(p)
        p = os.path.join(args.outdir, "communication.png")
        fig_comm(data["strong"], p); made.append(p)
        p = os.path.join(args.outdir, "noise.png")
        fig_noise(data["strong"], p); made.append(p)
    if data.get("weak"):
        p = os.path.join(args.outdir, "weak_scaling.png")
        fig_weak(data["weak"], p); made.append(p)
    if data.get("three_d"):
        p = os.path.join(args.outdir, "strong_3d.png")
        fig_3d(data["three_d"], p); made.append(p)
    made.append(fig_fields(args.outdir))
    print(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())