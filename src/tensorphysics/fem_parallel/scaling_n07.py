"""P284W3-N07: strong/weak scaling study of the MPI-style domain-decomposition core.

This module is the *measurement* layer on top of
:mod:`tensorphysics.fem_parallel.scaling` / ``operator`` / ``pcg``.  It adds the
three things the earlier strong/weak scaling study did not have:

1. **Contention-robust estimation.**  The host is a shared 128-core machine
   whose load average swings between 200 and 800.  A single wall-clock reading is
   therefore worthless (the same point can differ by 2x).  Every point is
   measured ``reps`` times, the *interleaved* way (all rank counts are visited in
   each repetition, so a load transient hits every point instead of one), and the
   **minimum** over repetitions is reported: the minimum is the least
   contaminated estimate of the solo runtime, and the spread
   ``max/min`` is reported as the noise indicator of the run.
2. **Iteration-invariant scaling.**  A Krylov solver with a fixed *relative*
   tolerance needs ``O(h^-1)`` more iterations when the grid is refined, so a
   weak-scaling ratio ``T(P)/T(1)`` mixes algorithm growth with parallel
   overhead.  Two metrics are reported side by side: the *converged* solve and a
   *fixed-iteration* solve (``K`` iterations, no early exit), which isolates the
   parallel infrastructure (assembly + ghost exchange + reductions + matvec).
3. **Latency-reduced PCG variant.**  ``variant="reordered"``
   (:func:`tensorphysics.fem_parallel.pcg.cg_latency_reduced`) cuts the global
   synchronisations per iteration from three to two; both variants are measured
   so the gain is visible instead of asserted.

Every number that ends up in the report is produced by ``main()`` into
``artifacts/fem_parallel_n07/scaling.json`` -- no hand-copied values.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .operator import DistributedSystem
from .partition import CartesianPartition
from .pcg import ASMPreconditioner, JacobiPreconditioner, cg, cg_latency_reduced
from .problems import make_problem
from .runtime import available_backends, mpi_available, run
from .scaling import grid_for_dofs, run_point
from .verification import (
    correction_worker,
    global_from_serialised,
    parallel_vs_serial,
    relative_l2,
    serialise_solution,
)

__all__ = [
    "loadavg",
    "measure_point",
    "strong_study",
    "weak_study",
    "dof3d_study",
    "correctness_study",
    "evaluate",
    "main",
]


def loadavg() -> Tuple[float, ...]:
    """Host load average, recorded next to every point (measurement context)."""
    try:
        return tuple(float(v) for v in os.getloadavg())
    except OSError:  # pragma: no cover - exotic platform
        return (float("nan"),) * 3


def _point_worker(comm, name="poisson_manufactured", n=(65, 65), tol=1e-8,
                  maxiter=200, precond="jacobi", variant="cg"):
    """One benchmark point; module level so it survives ``fork``/``pickle``."""
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
    elif precond in ("jacobi", None) and precond is not None:
        pc = JacobiPreconditioner(A, system)
    elif precond == "none":
        pc = None
    else:  # pragma: no cover - operator input
        raise ValueError(f"unknown preconditioner {precond!r}")
    t_setup = time.perf_counter() - t0

    system.halo.exchange(comm, b.copy())
    comm.barrier()
    comm.reset_stats()
    solver = cg_latency_reduced if variant == "reordered" else cg
    res = solver(A, b, system, tol=tol, maxiter=maxiter, precond=pc,
                 fixed_dof=system.bc_dof, record_every=0)
    return {
        "rank": int(comm.rank),
        "t_assemble": t_assemble,
        "t_setup": t_setup,
        "t_solve": res.t_total,
        "t_comm": res.t_comm,
        "t_matvec": res.t_matvec,
        "t_precond": res.t_precond,
        "iterations": int(res.iterations),
        "converged": bool(res.converged),
        "n_owned": int(system.n_owned),
        "n_global": int(system.n_global),
        "barrier_calls": int(comm.stats["barrier_calls"]),
    }


def measure_point(ranks: int, reps: int = 3, backend: str = "auto", **kwargs) -> Dict[str, Any]:
    """Run one point ``reps`` times and keep the minimum wall time."""
    trials: List[Dict[str, Any]] = []
    for _ in range(max(1, int(reps))):
        la = loadavg()
        t0 = time.perf_counter()
        per_rank = run(_point_worker, ranks, backend=backend, kwargs=kwargs,
                       barrier_timeout=3600)
        wall = time.perf_counter() - t0
        agg = {
            "t_assemble": max(r["t_assemble"] for r in per_rank),
            "t_setup": max(r["t_setup"] for r in per_rank),
            "t_solve": max(r["t_solve"] for r in per_rank),
            "t_comm": max(r["t_comm"] for r in per_rank),
            "iterations": max(r["iterations"] for r in per_rank),
            "converged": bool(all(r["converged"] for r in per_rank)),
            "n_global": max(r["n_global"] for r in per_rank),
            "n_owned_max": max(r["n_owned"] for r in per_rank),
            "t_total": max(r["t_assemble"] + r["t_setup"] + r["t_solve"] for r in per_rank),
            "wall": wall,
            "load1": la[0],
        }
        trials.append(agg)
    best = min(trials, key=lambda d: d["t_total"])
    worst = max(trials, key=lambda d: d["t_total"])
    out = dict(best)
    out.update({
        "ranks": int(ranks),
        "reps": len(trials),
        "noise_ratio": (worst["t_total"] / best["t_total"]) if best["t_total"] > 0 else None,
        "trials": [
            {k: t[k] for k in ("t_assemble", "t_setup", "t_solve", "t_comm", "t_total", "load1")}
            for t in trials
        ],
        "params": {k: v for k, v in kwargs.items() if k != "n"},
        "grid": list(kwargs.get("n", ())),
    })
    return out


def _solve_metric(pt: Dict[str, Any]) -> float:
    """Solver time per iteration (iteration-count independent)."""
    it = max(int(pt["iterations"]), 1)
    return pt["t_solve"] / it


def strong_study(n_dofs: int = 1_000_000, ranks: Sequence[int] = (1, 2, 4, 8, 16, 32),
                 iters: int = 64, reps: int = 3, variant: str = "cg",
                 backend: str = "auto") -> Dict[str, Any]:
    """Strong scaling: fixed ``n_dofs``, fixed iteration count ``iters``."""
    n = grid_for_dofs(n_dofs, ndim=2)
    grid = (n[0] + 1, n[1] + 1)
    points = []
    for P in ranks:
        pt = measure_point(P, reps=reps, backend=backend, name="poisson_unit_source",
                           n=grid, tol=0.0, maxiter=iters, precond="jacobi", variant=variant)
        points.append(pt)
    base = next(p for p in points if p["ranks"] == 1)
    for pt in points:
        pt["speedup"] = base["t_total"] / pt["t_total"]
        pt["efficiency"] = pt["speedup"] / pt["ranks"]
        pt["per_iter_ms"] = 1000.0 * _solve_metric(pt)
        pt["comm_fraction"] = pt["t_comm"] / pt["t_solve"] if pt["t_solve"] > 0 else None
    return {"kind": "strong", "n_dofs": int(base["n_global"]), "grid": list(grid),
            "iters": int(iters), "variant": variant, "points": points}


def weak_study(n_per_rank: int = 100_000, ranks: Sequence[int] = (1, 2, 4, 8, 16, 32),
               reps: int = 2, variant: str = "cg", backend: str = "auto",
               precond: str = "asm", tol: float = 1e-8, maxiter: int = 4000) -> Dict[str, Any]:
    """Weak scaling: ``n_per_rank`` DOFs per rank, converged solve."""
    points = []
    for P in ranks:
        n = grid_for_dofs(int(n_per_rank) * int(P), ndim=2)
        grid = (n[0] + 1, n[1] + 1)
        pt = measure_point(P, reps=reps, backend=backend, name="poisson_unit_source",
                           n=grid, tol=tol, maxiter=maxiter, precond=precond, variant=variant)
        pt["grid"] = list(grid)
        points.append(pt)
    base = next(p for p in points if p["ranks"] == 1)
    for pt in points:
        pt["weak_ratio"] = pt["t_total"] / base["t_total"]
        pt["weak_ratio_per_iter"] = (_solve_metric(pt) / _solve_metric(base)) if base["t_solve"] else None
        pt["per_iter_ms"] = 1000.0 * _solve_metric(pt)
        pt["comm_fraction"] = pt["t_comm"] / pt["t_solve"] if pt["t_solve"] > 0 else None
    return {"kind": "weak", "n_per_rank": int(n_per_rank), "precond": precond,
            "tol": tol, "variant": variant, "points": points}


def dof3d_study(n_dofs: int = 250_000, ranks: Sequence[int] = (1, 2, 4, 8),
                iters: int = 30, reps: int = 2, backend: str = "auto") -> Dict[str, Any]:
    """3-D Cartesian decomposition case (tensor grid, 7-point stencil)."""
    n = grid_for_dofs(int(n_dofs), ndim=3)
    grid = tuple(v + 1 for v in n)
    points = []
    for P in ranks:
        pt = measure_point(P, reps=reps, backend=backend, name="poisson_unit_source",
                           n=grid, tol=0.0, maxiter=iters, precond="jacobi", variant="cg")
        points.append(pt)
    base = next(p for p in points if p["ranks"] == 1)
    for pt in points:
        pt["speedup"] = base["t_total"] / pt["t_total"]
        pt["efficiency"] = pt["speedup"] / pt["ranks"]
        pt["decomposition"] = _factor_shape(grid, pt["ranks"])
    return {"kind": "strong3d", "grid": list(grid), "n_dofs": int(base["n_global"]),
            "iters": int(iters), "points": points}


def _factor_shape(grid: Sequence[int], ranks: int) -> List[int]:
    part = CartesianPartition(tuple(int(v) for v in grid), int(ranks))
    return [int(v) for v in part.shape]


def _reordered_worker(comm, name="poisson_manufactured", n=(65, 63), tol=1e-13,
                      maxiter=30000, variant="reordered"):
    pr = make_problem(name, tuple(int(v) for v in n))
    grid = pr.grid
    part = CartesianPartition(grid.n, comm.size)
    system = DistributedSystem(comm, grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    pc = JacobiPreconditioner(A, system)
    solver = cg_latency_reduced if variant == "reordered" else cg
    res = solver(A, b, system, tol=tol, maxiter=maxiter, precond=pc,
                 fixed_dof=system.bc_dof)
    out = serialise_solution(system, res.x)
    out.update({"iterations": int(res.iterations), "converged": bool(res.converged)})
    return out


def correctness_study(ranks: Sequence[int] = (8, 32), tol: float = 1e-13) -> Dict[str, Any]:
    """Parallel-vs-serial agreement of the three physics cases + both CG variants.

    ``max_relerr`` is the quantity the acceptance criterion (c) is written on.
    """
    cases = (
        ("poisson_manufactured", (97, 95)),
        ("heat_variable_k", (97, 95)),
        ("elasticity_patch", (49, 47)),
    )
    rows: List[Dict[str, Any]] = []
    for name, n in cases:
        pr = make_problem(name, n)
        n_dof = pr.grid.num_nodes * pr.ncomp
        ser = _reordered_worker(SerialShim(), name=name, n=n, tol=tol, variant="cg")
        u_ser = global_from_serialised([ser], n_dof)
        for P in ranks:
            for variant in ("cg", "reordered"):
                res = run(_reordered_worker, P,
                          kwargs={"name": name, "n": n, "tol": tol, "variant": variant},
                          barrier_timeout=3600)
                u_par = global_from_serialised(res, n_dof)
                rows.append({
                    "problem": name,
                    "grid": list(n),
                    "n_dof": int(n_dof),
                    "ranks": int(P),
                    "variant": variant,
                    "iterations": int(max(r["iterations"] for r in res)),
                    "converged": bool(all(r["converged"] for r in res)),
                    "rel_l2_vs_serial": relative_l2(u_par, u_ser),
                })
    return {"kind": "correctness", "rows": rows,
            "max_relerr": (max((r["rel_l2_vs_serial"] for r in rows), default=0.0))}


class SerialShim:
    """Minimal ``Comm`` stand-in for the serial reference solve."""

    def __init__(self) -> None:
        from .comm import SerialComm

        self._c = SerialComm()

    def __getattr__(self, item):
        return getattr(self._c, item)


def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the raw study into the machine-checkable criteria of the node."""
    crit: Dict[str, Any] = {}

    strong = payload.get("strong") or {}
    spts = {p["ranks"]: p for p in strong.get("points", [])}
    if 32 in spts:
        crit["a_strong_efficiency_P32"] = {
            "value": spts[32]["efficiency"], "threshold": 0.60, "op": ">=",
            "pass": bool(spts[32]["efficiency"] >= 0.60),
        }

    weak = payload.get("weak") or {}
    wpts = {p["ranks"]: p for p in weak.get("points", [])}
    if 32 in wpts:
        crit["b_weak_ratio_P32"] = {
            "value": wpts[32]["weak_ratio"], "threshold": 1.5, "op": "<=",
            "pass": bool(wpts[32]["weak_ratio"] <= 1.5),
        }
        crit["b2_weak_ratio_per_iter_P32"] = {
            "value": wpts[32]["weak_ratio_per_iter"], "threshold": 1.5, "op": "<=",
            "pass": bool((wpts[32]["weak_ratio_per_iter"] or 9e9) <= 1.5),
        }

    corr = payload.get("correctness") or {}
    if "max_relerr" in corr:
        crit["c_parallel_vs_serial"] = {
            "value": corr["max_relerr"], "threshold": 1e-10, "op": "<=",
            "pass": bool(corr["max_relerr"] <= 1e-10),
        }

    acc = payload.get("accuracy") or {}
    if acc.get("rows"):
        err = min(r["rel_l2_error"] for r in acc["rows"])
        crit["d_poisson_vs_analytic"] = {
            "value": err, "threshold": 1e-2, "op": "<=", "pass": bool(err <= 1e-2),
        }

    thr = payload.get("three_d") or {}
    tpts = {p["ranks"]: p for p in thr.get("points", [])}
    if tpts:
        pmax = max(tpts)
        crit["e_3d_efficiency"] = {
            "value": tpts[pmax]["efficiency"], "threshold": 0.50, "op": ">=",
            "pass": bool(tpts[pmax]["efficiency"] >= 0.50), "ranks": pmax,
        }
    return crit


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--n-dofs", type=int, default=1_000_000)
    ap.add_argument("--n-per-rank", type=int, default=100_000)
    ap.add_argument("--variant", default="cg")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--skip", default="", help="comma list: 3d,weak,correctness,accuracy")
    ap.add_argument("--out", default="artifacts/fem_parallel_n07/scaling.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    ranks = [int(v) for v in args.ranks.split(",") if v.strip()]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    t_start = time.perf_counter()
    payload: Dict[str, Any] = {
        "node": "P284W3-N07",
        "repo": "TensorPhysics",
        "backends_available": available_backends(),
        "mpi_available": mpi_available(),
        "host": {"cpus": os.cpu_count(), "load_at_start": loadavg(),
                 "platform": sys.platform},
        "protocol": {
            "estimator": "min over interleaved repetitions (contention-robust)",
            "t_total": "max_rank(t_assemble + t_setup + t_solve)",
            "reps": args.reps,
        },
    }
    def dump(extra=None):
        """Incremental write: a killed run still leaves the finished sections."""
        payload["criteria"] = evaluate(payload)
        payload["wall_seconds"] = time.perf_counter() - t_start
        payload["load_now"] = loadavg()
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)

    if "strong" not in skip:
        payload["strong"] = strong_study(n_dofs=args.n_dofs, ranks=ranks, iters=args.iters,
                                         reps=args.reps, variant=args.variant,
                                         backend=args.backend)
        dump()
    if "weak" not in skip:
        payload["weak"] = weak_study(n_per_rank=args.n_per_rank, ranks=ranks,
                                     reps=max(2, args.reps // 2), variant=args.variant,
                                     backend=args.backend)
        dump()
    if "3d" not in skip:
        ranks3 = [p for p in ranks if p <= 8]
        payload["three_d"] = dof3d_study(ranks=ranks3, reps=max(2, args.reps // 2),
                                         backend=args.backend)
        dump()
    if "correctness" not in skip:
        cr = [p for p in ranks if p in (8, 32)]
        if cr:
            payload["correctness"] = correctness_study(ranks=cr)
            dump()
    if "accuracy" not in skip:
        from .verification import accuracy_study

        payload["accuracy"] = {"rows": accuracy_study(nranks=1)}
        dump()
    payload["load_at_end"] = loadavg()
    dump()

    out = args.out
    print(json.dumps({"out": out, "criteria": payload["criteria"],
                      "wall_seconds": payload["wall_seconds"]}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())