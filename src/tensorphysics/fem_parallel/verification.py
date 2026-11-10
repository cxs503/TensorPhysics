"""Acceptance-criteria driver for the MPI domain-decomposition node.

Criteria (from the campaign node ``p284w3-n04-fem-mpi-parallel``):

======  ==================================================================
C1      strong scaling: ``E(P=16) >= 0.6`` (Poisson, ``N = 10**6`` DOFs)
C2      weak scaling: ``T(32)/T(1) <= 1.5`` (Poisson, ``N/P = 10**5``)
C3      parallel vs serial solution: relative L2 ``< 1e-10`` (Poisson,
        variable-conductivity diffusion, plane-strain elasticity)
C4      communication overhead: ``T_comm/T_total < 0.3`` at ``P = 16``
C5      PCG vs CG: iteration count reduced by ``>= 50 %``
C6      load balance: per-rank computation time ``std/mean < 0.1``
C7      ``>= 15`` green pytest cases (reported by CI, see ``docs/``)
======  ==================================================================

Every number in the produced JSON is produced by the run itself (no hand
transcription); the report generator reads the JSON.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .operator import DistributedSystem
from .partition import CartesianPartition
from .pcg import ASMPreconditioner, JacobiPreconditioner, cg
from .comm import SerialComm
from .problems import make_problem
from .reference import global_from_serialised, relative_l2, serialise_solution
from .runtime import run
from .scaling import grid_for_dofs, iteration_study, save_json, strong_scaling, weak_scaling

__all__ = [
    "correction_worker",
    "parallel_vs_serial",
    "accuracy_study",
    "run_verification",
    "CRITERIA",
]

CRITERIA = {
    "C1_strong_efficiency": 0.6,      # at P=16
    "C2_weak_ratio": 1.5,             # at P=32
    "C3_parallel_vs_serial_l2": 1e-10,
    "C4_comm_fraction": 0.3,          # at P=16
    "C5_iteration_reduction": 0.5,
    "C6_load_balance": 0.1,
}
TARGET_RANKS = {"C1_strong_efficiency": 16, "C2_weak_ratio": 32, "C4_comm_fraction": 16}


def correction_worker(
    comm,
    name: str = "poisson_manufactured",
    n: Sequence[int] = (65, 63),
    tol: float = 1e-13,
    precond: str = "jacobi",
    maxiter: int = 30000,
) -> Dict[str, Any]:
    """Solve one case and return the owned part of the solution (module level)."""
    pr = make_problem(name, tuple(int(v) for v in n))
    grid = pr.grid
    part = CartesianPartition(grid.n, comm.size)
    system = DistributedSystem(comm, grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    pc = ASMPreconditioner(A, system) if precond == "asm" else JacobiPreconditioner(A, system)
    res = cg(A, b, system, tol=tol, maxiter=maxiter, precond=pc, fixed_dof=system.bc_dof)
    out = serialise_solution(system, res.x)
    out.update({"ncomp": system.ncomp, "iterations": res.iterations,
                "converged": res.converged})
    return out


def parallel_vs_serial(
    cases: Sequence[Tuple[str, Tuple[int, ...]]] = (
        ("poisson_manufactured", (65, 63)),
        ("heat_variable_k", (65, 63)),
        ("elasticity_patch", (33, 31)),
    ),
    ranks: Iterable[int] = (4, 6, 8),
    tol: float = 1e-13,
) -> List[Dict[str, Any]]:
    """Compare the parallel solution against a serial solve of the same system."""
    rows = []
    for name, n in cases:
        pr = make_problem(name, n)
        ncomp = pr.ncomp
        n_dof = pr.grid.num_nodes * ncomp
        ser = correction_worker(SerialComm(), name=name, n=n, tol=tol)
        u_ser = global_from_serialised([ser], n_dof)
        exact = None
        if pr.exact is not None:
            exact = np.asarray(pr.exact(pr.grid.coords(np.arange(pr.grid.num_nodes)))).reshape(-1)
        for p in ranks:
            res = run(correction_worker, p, kwargs={"name": name, "n": n, "tol": tol},
                      barrier_timeout=3600)
            u_par = global_from_serialised(res, n_dof)
            row = {
                "problem": name,
                "grid": list(n),
                "n_dof": int(n_dof),
                "nranks": int(p),
                "iterations": int(max(r["iterations"] for r in res)),
                "converged": bool(all(r["converged"] for r in res)),
                "rel_l2_parallel_vs_serial": relative_l2(u_par, u_ser),
            }
            if exact is not None:
                row["rel_l2_parallel_vs_exact"] = relative_l2(u_par, exact)
                row["rel_l2_serial_vs_exact"] = relative_l2(u_ser, exact)
            rows.append(row)
    return rows


def accuracy_study(
    name: str = "poisson_manufactured",
    levels: Sequence[Tuple[int, ...]] = ((33, 33), (65, 65), (129, 129)),
    nranks: int = 1,
) -> List[Dict[str, Any]]:
    """Manufactured-solution convergence: relative L2 error vs ``h``."""
    rows = []
    for n in levels:
        pr = make_problem(name, n)
        res = run(
            _accuracy_worker, nranks,
            kwargs={"name": name, "n": n}, barrier_timeout=3600,
        )
        u = global_from_serialised(res, pr.grid.num_nodes * pr.ncomp)
        xy = pr.grid.coords(np.arange(pr.grid.num_nodes))
        ex = np.asarray(pr.exact(xy)).reshape(-1)
        rows.append({
            "grid": list(n),
            "h": float(pr.grid.h[0]),
            "rel_l2_error": relative_l2(u, ex),
            "iterations": int(max(r["iterations"] for r in res)),
        })
    for prev, cur in zip(rows, rows[1:]):
        ratio = prev["rel_l2_error"] / cur["rel_l2_error"] if cur["rel_l2_error"] else 0.0
        cur["observed_order"] = float(np.log2(ratio) / np.log2(prev["h"] / cur["h"])) if ratio > 0 else 0.0
    return rows


def _accuracy_worker(comm, name: str, n: Sequence[int]) -> Dict[str, Any]:
    pr = make_problem(name, tuple(int(v) for v in n))
    part = CartesianPartition(pr.grid.n, comm.size)
    system = DistributedSystem(comm, pr.grid, part, pr.kernel)
    A, b = system.assemble(bc=True, bc_fn=pr.dirichlet)
    res = cg(A, b, system, tol=1e-12, maxiter=40000,
             precond=JacobiPreconditioner(A, system), fixed_dof=system.bc_dof)
    out = serialise_solution(system, res.x)
    out["iterations"] = res.iterations
    return out


def run_verification(
    n_dofs: int = 1_000_000,
    ranks: Sequence[int] = (1, 2, 4, 8, 16, 32),
    n_per_rank: int = 100_000,
    backend: str = "auto",
    tol: float = 1e-8,
    out_path: Optional[str] = None,
    commutator_ranks: int = 16,
    iteration_n_dofs: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the full acceptance-criteria suite and return the JSON payload."""
    payload: Dict[str, Any] = {
        "node": "p284w3-n04-fem-mpi-parallel",
        "backend": backend,
        "n_dofs_strong": int(n_dofs),
        "n_per_rank_weak": int(n_per_rank),
        "ranks": list(ranks),
        "tol": tol,
    }

    grid = grid_for_dofs(n_dofs, ndim=2)
    payload["grid_strong"] = list(grid)
    payload["strong"] = strong_scaling(name="poisson_unit_source", n=grid, ranks=ranks,
                                       tol=tol, backend=backend)
    payload["weak"] = weak_scaling(name="poisson_unit_source", n_per_rank=n_per_rank,
                                   ranks=ranks, ndim=2, tol=tol, backend=backend)
    # the communication counters are collected inside the strong-scaling points,
    # so no extra runs are needed for the comm-overhead criterion
    payload["comm_sweep"] = payload["strong"]
    it_n = grid_for_dofs(iteration_n_dofs or max(n_dofs // 10, 10_000), ndim=2)
    payload["iterations"] = iteration_study(
        name="poisson_unit_source", n=it_n, nranks=min(8, max(ranks)), tol=tol, backend=backend
    )
    payload["correctness"] = parallel_vs_serial(ranks=tuple(r for r in ranks if r in (2, 4, 6, 8)))
    payload["accuracy"] = accuracy_study()

    def find(points, key, value):
        for pt in points:
            if pt.get(key) == value:
                return pt
        return None

    p16 = find(payload["strong"], "nranks", TARGET_RANKS["C1_strong_efficiency"])
    w32 = find(payload["weak"], "nranks", TARGET_RANKS["C2_weak_ratio"])
    c16 = find(payload["comm_sweep"], "nranks", TARGET_RANKS["C4_comm_fraction"])
    it = {pt["variant"]: pt for pt in payload["iterations"]}
    best_precond = min(
        (v for v in it if v != "none"), key=lambda v: it[v]["iterations"], default=None
    )
    checks = {}
    if p16 is not None:
        checks["C1_strong_efficiency"] = {
            "value": p16["efficiency"], "limit": CRITERIA["C1_strong_efficiency"],
            "pass": bool(p16["efficiency"] >= CRITERIA["C1_strong_efficiency"]),
            "at_p": p16["nranks"],
            "t1_solve": payload["strong"][0]["t_solve"], "t16_solve": p16["t_solve"],
            "iterations": p16["iterations"],
        }
    if w32 is not None:
        checks["C2_weak_ratio"] = {
            "value": w32["weak_ratio"], "limit": CRITERIA["C2_weak_ratio"],
            "pass": bool(w32["weak_ratio"] <= CRITERIA["C2_weak_ratio"]),
            "grid": w32["grid"], "at_p": w32["nranks"],
        }
    if payload["correctness"]:
        worst = max(r["rel_l2_parallel_vs_serial"] for r in payload["correctness"])
        checks["C3_parallel_vs_serial_l2"] = {
            "value": worst, "limit": CRITERIA["C3_parallel_vs_serial_l2"],
            "pass": bool(worst < CRITERIA["C3_parallel_vs_serial_l2"]),
            "cases": len(payload["correctness"]),
        }
    if c16 is not None:
        checks["C4_comm_fraction"] = {
            "value": c16["comm_fraction"], "limit": CRITERIA["C4_comm_fraction"],
            "pass": bool(c16["comm_fraction"] < CRITERIA["C4_comm_fraction"]),
            "at_p": c16["nranks"], "t_comm": c16["t_comm"], "t_solve": c16["t_solve"],
        }
    if best_precond is not None:
        checks["C5_iteration_reduction"] = {
            "value": it[best_precond]["iteration_reduction"],
            "limit": CRITERIA["C5_iteration_reduction"],
            "pass": bool(it[best_precond]["iteration_reduction"] >= CRITERIA["C5_iteration_reduction"]),
            "variant": best_precond,
            "cg_iterations": it["none"]["iterations"],
            "pcg_iterations": it[best_precond]["iterations"],
        }
    lb_points = [pt for pt in payload["comm_sweep"] if pt["nranks"] > 1]
    if lb_points:
        worst_lb = max(pt["load_balance_std_over_mean"] for pt in lb_points)
        checks["C6_load_balance"] = {
            "value": worst_lb, "limit": CRITERIA["C6_load_balance"],
            "pass": bool(worst_lb < CRITERIA["C6_load_balance"]),
            "at_p": max(pt["nranks"] for pt in lb_points),
            "points": len(lb_points),
        }
    missing = [k for k in CRITERIA if k not in checks]
    payload["checks"] = checks
    payload["missing_checks"] = missing
    payload["all_pass"] = bool(
        checks and not missing and all(c["pass"] for c in checks.values())
    )
    if out_path:
        save_json(payload, out_path)
    return payload


def main(argv=None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="parallel FEM acceptance suite")
    ap.add_argument("--n-dofs", type=int, default=1_000_000)
    ap.add_argument("--n-per-rank", type=int, default=100_000)
    ap.add_argument("--ranks", default="1,2,4,8,16,32")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default="artifacts/fem_parallel/verification.json")
    args = ap.parse_args(argv)
    payload = run_verification(
        n_dofs=args.n_dofs,
        ranks=tuple(int(x) for x in args.ranks.split(",") if x),
        n_per_rank=args.n_per_rank,
        backend=args.backend,
        tol=args.tol,
        out_path=args.out,
    )
    print(json.dumps(payload.get("checks", {}), indent=2))
    print(json.dumps({k: v.get("pass") if isinstance(v, dict) else v
                      for k, v in payload.get("checks", {}).items()}, indent=2))
    print("ALL PASS" if payload["all_pass"] else "SOME CRITERIA FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
