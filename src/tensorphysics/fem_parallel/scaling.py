"""Strong/weak scaling and communication-overhead measurement harness.

A *point* of the study is one ``(problem, grid, nranks, preconditioner)`` tuple.
:func:`run_point` launches the ranks through
:func:`tensorphysics.fem_parallel.runtime.run` and returns the measured
breakdown (assembly, preconditioner setup, solve, communication), the Krylov
iteration count and the per-rank work times used for the load-balance statistic.

Aggregation rules (stated once, used everywhere):

* ``T(P) = max_rank(t_assemble + t_setup + t_solve)`` -- the wall time of the
  parallel run is the slowest rank, not the average.
* strong scaling: ``S(P) = T(1) / T(P)``, ``E(P) = S(P) / P``.
* weak scaling: ``W(P) = T(P) / T(1)`` with the global DOF count growing as
  ``P * N_per_rank``; ideal weak scaling gives ``W(P) = 1``.
* communication fraction: ``T_comm / T_solve`` where ``T_comm`` is the sum of
  the time spent inside the collective calls of the slowest rank.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .operator import DistributedSystem
from .partition import CartesianPartition
from .pcg import ASMPreconditioner, JacobiPreconditioner, cg, cg_latency_reduced
from .problems import make_problem
from .runtime import run

__all__ = [
    "grid_for_dofs",
    "solve_worker",
    "run_point",
    "strong_scaling",
    "weak_scaling",
    "iteration_study",
    "aggregate",
    "save_json",
]


def grid_for_dofs(n_dofs: int, ndim: int = 2, aspect: Sequence[float] = None) -> Tuple[int, ...]:
    """Tensor grid with (about) ``n_dofs`` nodes in ``ndim`` dimensions."""
    aspect = list(aspect or [1.0] * ndim)
    vol = float(np.prod(aspect)) ** (1.0 / ndim)
    per_dim = (float(n_dofs) / vol) ** (1.0 / ndim)
    return tuple(max(2, int(round(per_dim * a))) for a in aspect)


def solve_worker(
    comm,
    name: str = "poisson_unit_source",
    n: Sequence[int] = (129, 129),
    tol: float = 1e-8,
    maxiter: int = 4000,
    precond: str = "jacobi",
    local_reps: int = 5,
    variant: str = "cg",
) -> Dict[str, Any]:
    """One benchmark point on one rank (module-level so it can be pickled)."""
    pr = make_problem(name, tuple(int(v) for v in n))
    grid = pr.grid
    part = CartesianPartition(grid.n, comm.size)
    t0 = time.perf_counter()
    system = DistributedSystem(comm, grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    t_assemble = time.perf_counter() - t0

    t0 = time.perf_counter()
    if precond == "asm":
        pc = ASMPreconditioner(A, system)
    elif precond == "jacobi":
        pc = JacobiPreconditioner(A, system)
    elif precond in ("none", None):
        pc = None
    else:  # pragma: no cover - operator input
        raise ValueError(f"unknown preconditioner {precond!r}")
    t_setup = time.perf_counter() - t0

    # local (communication-free) work: load balance indicator
    x = np.ones(system.ndof)
    t0 = time.perf_counter()
    for _ in range(int(local_reps)):
        A @ x
    t_matvec_local = (time.perf_counter() - t0) / max(int(local_reps), 1)

    # warm-up collective (page in the shared arena, touch ghost buffers)
    system.halo.exchange(comm, b.copy())
    comm.barrier()

    comm.reset_stats()
    solver = cg_latency_reduced if variant == "reordered" else cg
    res = solver(
        A, b, system,
        tol=tol, maxiter=maxiter, precond=pc, fixed_dof=system.bc_dof,
        record_every=0,
    )
    return {
        "rank": int(comm.rank),
        "t_assemble": t_assemble,
        "t_setup": t_setup,
        "t_solve": res.t_total,
        "t_matvec": res.t_matvec,
        "t_precond": res.t_precond,
        "t_comm": res.t_comm,
        "t_comm_stats": float(comm.comm_time()),
        "t_barrier": float(comm.stats["barrier_time"]),
        "t_matvec_local": float(t_matvec_local),
        "n_roll_call": int(comm.stats["exchange_calls"] + comm.stats["allreduce_calls"]),
        "iterations": int(res.iterations),
        "converged": bool(res.converged),
        "n_owned": int(system.n_owned),
        "n_box": int(system.ndof),
        "n_global": int(system.n_global),
        "nnz_local": int(A.nnz),
        "bytes_exchanged": int(comm.stats["exchange_bytes"]),
    }


def run_point(
    name: str = "poisson_unit_source",
    n: Sequence[int] = (129, 129),
    nranks: int = 1,
    tol: float = 1e-8,
    maxiter: int = 4000,
    precond: str = "jacobi",
    backend: str = "auto",
    slot_floats: int = 1 << 18,
    barrier_timeout: float = 1800.0,
    variant: str = "cg",
) -> Dict[str, Any]:
    """Run one benchmark point and aggregate the rank-level measurements."""
    n = tuple(int(v) for v in n)
    t_wall = time.perf_counter()
    ranks = run(
        solve_worker, nranks, backend=backend, slot_floats=slot_floats,
        kwargs={"name": name, "n": n, "tol": tol, "maxiter": maxiter, "precond": precond,
                "variant": variant},
        barrier_timeout=barrier_timeout,
    )
    wall = time.perf_counter() - t_wall
    agg = aggregate(ranks)
    agg.update(
        {
            "problem": name,
            "grid": list(n),
            "nranks": int(nranks),
            "precond": precond,
            "solver_variant": variant,
            "tol": tol,
            "backend": backend,
            "wall_including_launch": wall,
        }
    )
    return agg


def aggregate(ranks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn per-rank measurements into the point-level summary."""
    def col(key):
        return np.array([float(r[key]) for r in ranks])

    t_assemble = float(col("t_assemble").max())
    t_setup = float(col("t_setup").max())
    t_solve = float(col("t_solve").max())
    t_comm = float(col("t_comm").max())
    mv = col("t_matvec_local")
    asm_t = col("t_assemble")
    work = asm_t + mv  # per-rank computation time (no collectives)
    lb_std = float(work.std() / work.mean()) if work.mean() > 0 else 0.0
    return {
        "t_assemble": t_assemble,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_total": t_assemble + t_setup + t_solve,
        "t_comm": t_comm,
        "comm_fraction": float(t_comm / t_solve) if t_solve > 0 else 0.0,
        "iterations": int(max(r["iterations"] for r in ranks)),
        "converged": bool(all(r["converged"] for r in ranks)),
        "n_owned_min": int(min(r["n_owned"] for r in ranks)),
        "n_owned_max": int(max(r["n_owned"] for r in ranks)),
        "n_global": int(max(r["n_global"] for r in ranks)),
        "nnz_local": int(max(r["nnz_local"] for r in ranks)),
        "load_balance_std_over_mean": lb_std,
        "per_rank": [
            {k: r[k] for k in ("rank", "t_assemble", "t_solve", "t_comm", "n_owned",
                               "t_matvec_local", "iterations")}
            for r in ranks
        ],
    }


def strong_scaling(
    name: str = "poisson_unit_source",
    n: Sequence[int] = (1000, 1000),
    ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    **kwargs,
) -> List[Dict[str, Any]]:
    points = []
    for p in ranks:
        pt = run_point(name=name, n=n, nranks=p, **kwargs)
        points.append(pt)
    base = points[0]["t_solve"] if points else 1.0
    base_total = points[0]["t_total"] if points else 1.0
    for pt in points:
        pt["speedup"] = base / pt["t_solve"] if pt["t_solve"] else 0.0
        pt["efficiency"] = pt["speedup"] / pt["nranks"]
        pt["speedup_total"] = base_total / pt["t_total"] if pt["t_total"] else 0.0
        pt["efficiency_total"] = pt["speedup_total"] / pt["nranks"]
    return points


def weak_scaling(
    name: str = "poisson_unit_source",
    n_per_rank: int = 100_000,
    ranks: Iterable[int] = (1, 2, 4, 8, 16, 32),
    ndim: int = 2,
    **kwargs,
) -> List[Dict[str, Any]]:
    points = []
    for p in ranks:
        n = grid_for_dofs(int(n_per_rank) * int(p), ndim=ndim)
        pt = run_point(name=name, n=n, nranks=p, **kwargs)
        pt["n_per_rank_target"] = int(n_per_rank)
        points.append(pt)
    base = points[0]["t_solve"] if points else 1.0
    base_total = points[0]["t_total"] if points else 1.0
    for pt in points:
        pt["weak_ratio"] = pt["t_solve"] / base if base else 0.0
        pt["weak_ratio_total"] = pt["t_total"] / base_total if base_total else 0.0
    return points


def iteration_study(
    name: str = "poisson_unit_source",
    n: Sequence[int] = (317, 317),
    nranks: int = 4,
    variants: Sequence[str] = ("none", "jacobi", "asm"),
    tol: float = 1e-8,
    **kwargs,
) -> List[Dict[str, Any]]:
    """CG vs preconditioned CG iteration counts on one and the same system."""
    out = []
    for v in variants:
        pt = run_point(name=name, n=n, nranks=nranks, precond=v, tol=tol, **kwargs)
        pt["variant"] = v
        out.append(pt)
    base = next((p["iterations"] for p in out if p["variant"] == "none"), None)
    for pt in out:
        pt["iteration_reduction"] = (
            1.0 - pt["iterations"] / base if base else 0.0
        )
    return out


def save_json(payload: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def main(argv=None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="parallel FEM scaling study")
    ap.add_argument("--out", default="artifacts/fem_parallel/scaling.json")
    ap.add_argument("--n-dofs", type=int, default=1_000_000)
    ap.add_argument("--ranks", default="1,2,4,8,16,32")
    ap.add_argument("--ndim", type=int, default=2)
    ap.add_argument("--problem", default="poisson_unit_source")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--only", default="all", choices=["all", "strong", "weak", "iter"])
    args = ap.parse_args(argv)
    rr = [int(x) for x in args.ranks.split(",") if x]
    payload: Dict[str, Any] = {"n_dofs": args.n_dofs, "ranks": rr, "problem": args.problem,
                               "ndim": args.ndim, "tol": args.tol, "backend": args.backend}
    if args.only in ("all", "strong"):
        n = grid_for_dofs(args.n_dofs, ndim=args.ndim)
        payload["strong"] = strong_scaling(name=args.problem, n=n, ranks=rr, tol=args.tol,
                                           backend=args.backend)
    if args.only in ("all", "weak"):
        payload["weak"] = weak_scaling(name=args.problem,
                                       n_per_rank=args.n_dofs // max(rr[-1], 1),
                                       ranks=rr, ndim=args.ndim, tol=args.tol,
                                       backend=args.backend)
    if args.only in ("all", "iter"):
        payload["iterations"] = iteration_study(name=args.problem,
                                                n=grid_for_dofs(args.n_dofs // 10, ndim=args.ndim),
                                                nranks=max(1, min(rr[-1], 8)),
                                                tol=args.tol, backend=args.backend)
    print(save_json(payload, args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
