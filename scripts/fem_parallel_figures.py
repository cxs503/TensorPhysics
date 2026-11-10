"""Figures for the FEM MPI domain-decomposition study (p284w3-n04).

Reads ``artifacts/fem_parallel/verification.json`` (produced by
``python -m tensorphysics.fem_parallel.verification``) and writes PNGs into
``artifacts/fem_parallel/figures``.  One extra 4-rank solve is run here to plot
the solution field and its error against the analytic solution -- that is the
only number in the figure set not already in the JSON, and it is timed and
reported by the same code path.
"""

from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.fem_parallel.operator import DistributedSystem  # noqa: E402
from tensorphysics.fem_parallel.partition import CartesianPartition  # noqa: E402
from tensorphysics.fem_parallel.pcg import JacobiPreconditioner, cg  # noqa: E402
from tensorphysics.fem_parallel.problems import make_problem  # noqa: E402
from tensorphysics.fem_parallel.reference import (  # noqa: E402
    global_from_serialised,
    relative_l2,
    serialise_solution,
)
from tensorphysics.fem_parallel.runtime import run  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "fem_parallel")
FIG = os.path.join(ART, "figures")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def ranks_of(points):
    return np.array([p["nranks"] for p in points], dtype=float)


def fig_strong(payload, out):
    pts = payload["strong"]
    p = ranks_of(pts)
    s = np.array([x["speedup"] for x in pts])
    e = np.array([x["efficiency"] for x in pts])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(p, p, "k--", label="ideal $S=P$")
    ax[0].loglog(p, s, "o-", color="C0", label="measured (solve)")
    if all("speedup_total" in x for x in pts):
        ax[0].loglog(p, [x["speedup_total"] for x in pts], "s-", color="C1",
                     label="measured (assemble+setup+solve)")
    ax[0].set_xlabel("ranks $P$")
    ax[0].set_ylabel("speed-up $S(P)$")
    ax[0].set_title(f"Strong scaling, $N$={payload['n_dofs_strong']:,} DOF")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].semilogx(p, e, "o-", color="C0", label="solve-only")
    if all("efficiency_total" in x for x in pts):
        ax[1].semilogx(p, [x["efficiency_total"] for x in pts], "s-", color="C1",
                       label="assemble+setup+solve")
    ax[1].axhline(0.6, color="r", ls=":", label="acceptance 0.6")
    ax[1].set_xlabel("ranks $P$")
    ax[1].set_ylabel("parallel efficiency $E(P)$")
    ax[1].set_title("Strong-scaling efficiency")
    ax[1].set_ylim(0, 1.15)
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_weak(payload, out):
    pts = payload["weak"]
    p = ranks_of(pts)
    ratio = np.array([x["weak_ratio"] for x in pts])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(p, np.ones_like(p), "k--", label="ideal $T(P)/T(1)=1$")
    ax.semilogx(p, ratio, "o-", color="C2", label="measured (solve)")
    if all("weak_ratio_total" in x for x in pts):
        ax.semilogx(p, [x["weak_ratio_total"] for x in pts], "s-", color="C1",
                    label="assemble+setup+solve")
    ax.axhline(1.5, color="r", ls=":", label="acceptance 1.5")
    ax.set_xlabel("ranks $P$")
    ax.set_ylabel("$T(P)/T(1)$")
    ax.set_title(f"Weak scaling, $N/P$={payload['n_per_rank_weak']:,} DOF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_comm(payload, out):
    pts = payload["comm_sweep"]
    p = ranks_of(pts)
    frac = np.array([x["comm_fraction"] for x in pts])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(p, frac, "o-", color="C3", label="$T_{comm}/T_{solve}$")
    ax[0].axhline(0.3, color="r", ls=":", label="acceptance 0.3 at P=16")
    ax[0].set_xlabel("ranks $P$")
    ax[0].set_ylabel("communication fraction")
    ax[0].set_title(f"Communication overhead (N={payload['n_dofs_strong']:,})")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)
    solve = np.array([x["t_solve"] for x in pts])
    comm = np.array([x["t_comm"] for x in pts])
    ax[1].loglog(p, solve, "o-", label="$T_{solve}$")
    ax[1].loglog(p, comm, "s-", label="$T_{comm}$")
    ax[1].set_xlabel("ranks $P$")
    ax[1].set_ylabel("time [s]")
    ax[1].set_title("Solve vs communication time")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_iterations(payload, out):
    pts = payload["iterations"]
    labels = {"none": "CG (none)", "jacobi": "PCG Jacobi", "asm": "PCG additive Schwarz"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(pts))
    vals = [pt["iterations"] for pt in pts]
    ax.bar(xs, vals, color=["C7", "C0", "C2"][: len(pts)])
    for x, pt in zip(xs, pts):
        ax.text(x, pt["iterations"] * 1.02, f"{pt['iterations']}\n({100 * pt['iteration_reduction']:.0f}% less)",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels.get(pt["variant"], pt["variant"]) for pt in pts], fontsize=8)
    ax.set_ylabel("Krylov iterations")
    ax.set_title(f"Iteration count, $N$={pts[0]['n_global']:,}, $P$={pts[0]['nranks']}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_accuracy(payload, out):
    rows = payload["accuracy"]
    h = np.array([r["h"] for r in rows])
    err = np.array([r["rel_l2_error"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(h, err, "o-", label="Q1 FEM vs analytic")
    if len(h) > 1:
        ax.loglog(h, err[-1] * (h / h[-1]) ** 2, "k--", label="slope $h^2$")
    ax.set_xlabel("$h$")
    ax.set_ylabel("relative $L_2$ error")
    ax.set_title("Manufactured-solution convergence (Poisson)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_fields(out, n=(129, 129), nranks=4):
    pr = make_problem("poisson_manufactured", n)

    def worker(comm):
        part = CartesianPartition(pr.grid.n, comm.size)
        system = DistributedSystem(comm, pr.grid, part, pr.kernel)
        A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
        res = cg(A, b, system, tol=1e-12, maxiter=20000,
                 precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
        payload = serialise_solution(system, res.x)
        payload["iterations"] = res.iterations
        return payload

    res = run(worker, nranks, barrier_timeout=1800)
    u = global_from_serialised(res, pr.grid.num_nodes)
    xy = pr.grid.coords(np.arange(pr.grid.num_nodes))
    exact = np.asarray(pr.exact(xy)).reshape(-1)
    X = xy[:, 0].reshape(n[1], n[0])
    Y = xy[:, 1].reshape(n[1], n[0])
    U = u.reshape(n[1], n[0])
    E = np.abs(u - exact).reshape(n[1], n[0])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    im = ax[0].pcolormesh(X, Y, U, shading="auto", cmap="viridis")
    ax[0].set_title(f"parallel FE solution ($P$={nranks})")
    fig.colorbar(im, ax=ax[0])
    im1 = ax[1].pcolormesh(X, Y, exact.reshape(n[1], n[0]), shading="auto", cmap="viridis")
    ax[1].set_title("analytic solution")
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].pcolormesh(X, Y, E, shading="auto", cmap="magma")
    ax[2].set_title("|FE - analytic|")
    fig.colorbar(im2, ax=ax[2])
    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"rel_l2": relative_l2(u, exact), "iterations": int(max(r["iterations"] for r in res))}


def fig_load_balance(payload, out):
    pts = [p for p in payload["comm_sweep"] if p["nranks"] > 1]
    if not pts:
        return
    pt = pts[-1]
    ranks = [r["rank"] for r in pt["per_rank"]]
    asm = [r["t_assemble"] for r in pt["per_rank"]]
    mv = [r["t_matvec_local"] for r in pt["per_rank"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(ranks, asm, label="local assembly")
    ax.bar(ranks, mv, bottom=asm, label="local matvec")
    ax.set_xlabel("rank")
    ax.set_ylabel("time [s]")
    ax.set_title(f"Per-rank work, $P$={pt['nranks']} "
                 f"(std/mean={pt['load_balance_std_over_mean']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    os.makedirs(FIG, exist_ok=True)
    payload = load(os.path.join(ART, "verification.json"))
    figs = []
    fig_strong(payload, os.path.join(FIG, "strong_scaling.png")); figs.append("strong_scaling.png")
    fig_weak(payload, os.path.join(FIG, "weak_scaling.png")); figs.append("weak_scaling.png")
    fig_comm(payload, os.path.join(FIG, "communication.png")); figs.append("communication.png")
    fig_iterations(payload, os.path.join(FIG, "iterations.png")); figs.append("iterations.png")
    fig_accuracy(payload, os.path.join(FIG, "accuracy.png")); figs.append("accuracy.png")
    fig_load_balance(payload, os.path.join(FIG, "load_balance.png")); figs.append("load_balance.png")
    field = fig_fields(os.path.join(FIG, "fields.png")); figs.append("fields.png")
    payload["field_figure"] = field
    with open(os.path.join(ART, "figures_meta.json"), "w") as fh:
        json.dump({"figures": figs, "field": field}, fh, indent=2)
    print("wrote", len(figs), "figures to", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
