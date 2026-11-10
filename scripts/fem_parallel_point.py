"""Run one or more scaling points and write a partial result JSON.

Used to spread the acceptance study over several cluster jobs (each job gets its
own core allocation, which schedules faster than one 32-core block); the partial
files are merged by ``scripts/fem_parallel_merge.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorphysics.fem_parallel.scaling import (  # noqa: E402
    grid_for_dofs,
    run_point,
    save_json,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="strong", choices=["strong", "weak", "iter"])
    ap.add_argument("--n-dofs", type=int, default=1_000_000)
    ap.add_argument("--n-per-rank", type=int, default=100_000)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--problem", default="poisson_unit_source")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ranks = [int(x) for x in args.ranks.split(",") if x]
    payload = {"kind": args.kind, "n_dofs": args.n_dofs, "tol": args.tol,
               "problem": args.problem, "points": []}
    if args.kind == "strong":
        grid = grid_for_dofs(args.n_dofs, ndim=2)
        payload["grid"] = list(grid)
        for p in ranks:
            pt = run_point(name=args.problem, n=grid, nranks=p, tol=args.tol)
            payload["points"].append(pt)
            print(json.dumps({k: pt[k] for k in ("nranks", "t_solve", "iterations",
                                                 "comm_fraction", "load_balance_std_over_mean")}),
                  flush=True)
        base = payload["points"][0]["t_solve"]
        for pt in payload["points"]:
            pt["speedup"] = base / pt["t_solve"] if pt["t_solve"] else 0.0
            pt["efficiency"] = pt["speedup"] / pt["nranks"]
    elif args.kind == "weak":
        for p in ranks:
            n = grid_for_dofs(int(args.n_per_rank) * p, ndim=2)
            pt = run_point(name=args.problem, n=n, nranks=p, tol=args.tol)
            pt["n_per_rank_target"] = int(args.n_per_rank)
            payload["points"].append(pt)
            print(json.dumps({"nranks": p, "grid": pt["grid"], "t_solve": pt["t_solve"],
                              "iterations": pt["iterations"]}), flush=True)
    else:
        from tensorphysics.fem_parallel.scaling import iteration_study

        n = grid_for_dofs(args.n_dofs, ndim=2)
        payload["points"] = iteration_study(name=args.problem, n=n, nranks=max(ranks),
                                            tol=args.tol)
        payload["n_ranks_used"] = max(ranks)
    save_json(payload, args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
