"""Command-line front end for the uncertainty-budget gate.

Three predeclared modes (mutually exclusive flags, so a script cannot silently
run something other than what it says):

``--pce``
    Chaos route only: sample, fit, report the surrogate LOO quality ``Q^2``,
    the analytic Sobol' indices of the surrogate and the output band.
``--sobol``
    Chaos route **plus** the independent Saltelli/Jansen cross-check; the
    PCE-vs-MC agreement is part of the report (this is the default).
``--montecarlo``
    Sampled route only: no surrogate is fitted, the pick-freeze estimate is
    primary (what the fail-safe degrades to).

Examples
--------
::

    python -m tensorphysics.uq.cli --sobol --benchmark ishigami --json out.json
    python -m tensorphysics.uq.cli --pce --benchmark holton --dim 8 --degree 2
    python -m tensorphysics.uq.cli --montecarlo --benchmark ishigami --n-base 8192
    python -m tensorphysics.uq.cli --pce --failsafe --benchmark ishigami --n 8
    python -m tensorphysics.uq.cli --benchmark-json data/cross_repo_benchmarks/\\
        tensorfem_finite_width_hole_audit.json --list-qoi

Exit codes: ``0`` normal, ``2`` usage/schema error, ``3`` ``--require-pass`` was
given and the gate did not pass.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .benchmarks import (
    holtrop_type_benchmark,
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)
from .failsafe import DEFAULT_Q2_MIN, indices_from_mc, loo_q2, run_with_failsafe
from .holton import holton_analytic, holton_spec, make_holton
from .pce import fit_pce
from .sample_spec import SampleSpec
from .sampler import sample
from .sobol_mc import saltelli_indices
from .solver_outputs import combine_budget, parse_benchmarks
from .verdict import DEFAULT_EPS_LOO_MAX, PASS, decide, output_budget

__all__ = ["build_parser", "main", "run"]

_NO_UNIFORM = "U(0,1)"


# ---------------------------------------------------------------------------
# benchmark registry
# ---------------------------------------------------------------------------
def _bench_ishigami(args: argparse.Namespace) -> tuple[str, Callable, SampleSpec, Any]:
    return "ishigami", ishigami, ishigami_spec(), ishigami_analytic()


def _bench_holton(args: argparse.Namespace) -> tuple[str, Callable, SampleSpec, Any]:
    model = make_holton(args.dim)
    return "holton", model, holton_spec(args.dim), holton_analytic(args.dim)


def _bench_sobol_g(args: argparse.Namespace) -> tuple[str, Callable, SampleSpec, Any]:
    a = tuple(2.0**-k for k in range(args.dim))
    return (
        "sobol-g",
        lambda p: sobol_g(p, a=a),
        sobol_g_spec(args.dim),
        sobol_g_analytic(a=a),
    )


def _bench_holtrop(args: argparse.Namespace) -> tuple[str, Callable, SampleSpec, Any]:
    bench = holtrop_type_benchmark()
    return "holtrop-type", bench.resistance_kn, bench.spec, None


BENCHMARKS: dict[str, Callable[[argparse.Namespace], tuple[str, Callable, SampleSpec, Any]]] = {
    "ishigami": _bench_ishigami,
    "holton": _bench_holton,
    "sobol-g": _bench_sobol_g,
    "holtrop-type": _bench_holtrop,
}


# ---------------------------------------------------------------------------
# analytic comparison
# ---------------------------------------------------------------------------
def analytic_comparison(indices, analytic: Any) -> dict[str, Any]:
    """Compare computed indices with a closed-form decomposition (when known)."""
    if analytic is None:
        return {}
    first_ref = np.asarray(analytic["first"], dtype=float)
    total_ref = np.asarray(analytic["total"], dtype=float)
    first = np.asarray(indices.first, dtype=float)
    total = np.asarray(indices.total, dtype=float)
    if first.size != first_ref.size:
        return {"error": "index length mismatch"}
    first_err = np.abs(first - first_ref)
    total_err = np.abs(total - total_ref)
    second_ref = analytic.get("second") or {}
    second_err = {
        f"{i}*{j}": float(abs(indices.second.get((i, j), math.nan) - v))
        for (i, j), v in second_ref.items()
        if (i, j) in indices.second
    }
    return {
        "first_max_abs_error": float(np.max(first_err)) if first.size else 0.0,
        "total_max_abs_error": float(np.max(total_err)) if total.size else 0.0,
        "second_max_abs_error": max(second_err.values()) if second_err else None,
        "per_input": {
            f"x{i + 1}": {
                "S_ref": float(first_ref[i]),
                "S": float(first[i]),
                "ST_ref": float(total_ref[i]),
                "ST": float(total[i]),
            }
            for i in range(first.size)
        },
        "second": second_err,
    }


# ---------------------------------------------------------------------------
# run modes
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the requested mode and return a JSON-serialisable report."""
    if args.benchmark_json:
        return _run_benchmark_json(args)
    factory = BENCHMARKS[args.benchmark]
    name, solver, spec, analytic = factory(args)
    if args.mode == "montecarlo":
        report = _run_montecarlo(args, solver, spec, name)
    else:
        report = _run_chaos(args, solver, spec, name, analytic, cross_check=args.mode == "sobol")
    report["mode"] = args.mode
    report["benchmark"] = name
    report["analytic_comparison"] = analytic_comparison(report.pop("_indices"), analytic)
    return report


def _run_chaos(
    args: argparse.Namespace,
    solver: Callable,
    spec: SampleSpec,
    name: str,
    analytic: Any,
    *,
    cross_check: bool,
) -> dict[str, Any]:
    design = sample(spec, args.n, rule=args.rule, seed=args.seed)
    points, coords = spec.from_latent(design.latent)
    values = np.asarray(solver(points), dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=np.nan)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        raise SystemExit(f"only {int(valid.sum())} finite solver outputs: cannot fit")
    surrogate = fit_pce(
        coords[valid],
        values[valid],
        degree=args.degree,
        families=spec.families,
        method=args.method,
    )
    pce_indices = surrogate.sobol()
    band = output_budget(surrogate, spec, n_mc=args.budget_n_mc, seed=args.seed)
    report: dict[str, Any] = {
        "status_source": "pce",
        "n_samples": int(args.n),
        "n_valid": int(valid.sum()),
        "degree": int(args.degree),
        "method": args.method,
        "q2": loo_q2(surrogate),
        "eps_loo": float(surrogate.eps_loo),
        "indices": pce_indices.as_dict(names=spec.names),
        "budget": band.as_dict(),
    }
    mc = None
    if cross_check:
        def _evaluate(unit_points: np.ndarray) -> np.ndarray:
            phys, _ = spec.transform(unit_points)
            return np.asarray(solver(phys), dtype=float).reshape(-1)

        mc = saltelli_indices(
            _evaluate, spec.dim, n_base=args.n_base, seed=args.seed + 3,
            second_order=True, bootstrap=args.bootstrap,
        )
        report["montecarlo"] = mc.as_dict(names=spec.names)
        report["pce_vs_mc"] = {
            "first_max_abs_diff": float(np.max(np.abs(pce_indices.first - mc.first))),
            "total_max_abs_diff": float(np.max(np.abs(pce_indices.total - mc.total))),
        }
        verdict = decide(
            budget=band,
            threshold=args.threshold,
            surrogate=surrogate,
            mc=mc,
            cross_check=None,
            require_cross_check=False,
        )
        report["gate"] = verdict.as_dict()
    else:
        verdict = decide(
            budget=band,
            threshold=args.threshold,
            surrogate=surrogate,
            mc=None,
            cross_check=None,
            require_cross_check=False,
            eps_loo_max=args.eps_loo_max,
        )
        report["gate"] = verdict.as_dict()
    report["indices_source"] = "pce"
    report["_indices"] = pce_indices
    if args.failsafe:
        study = run_with_failsafe(
            solver,
            spec,
            n_samples=args.n,
            degree=args.degree,
            method=args.method,
            seed=args.seed,
            rule=args.rule,
            q2_min=args.q2_min,
            threshold=args.threshold,
            run_mc=True,
            n_base_mc=args.n_base,
        )
        report["failsafe"] = study.as_dict()
    return report


def _run_montecarlo(
    args: argparse.Namespace, solver: Callable, spec: SampleSpec, name: str
) -> dict[str, Any]:
    def _evaluate(unit_points: np.ndarray) -> np.ndarray:
        phys, _ = spec.transform(unit_points)
        return np.asarray(solver(phys), dtype=float).reshape(-1)

    mc = saltelli_indices(
        _evaluate, spec.dim, n_base=args.n_base, seed=args.seed,
        second_order=True, bootstrap=args.bootstrap,
    )
    from .failsafe import direct_budget

    class _Box:
        def __init__(self, fn: Callable) -> None:
            self._fn = fn

        def evaluate(self, points: np.ndarray) -> np.ndarray:
            return np.asarray(self._fn(points), dtype=float).reshape(-1)

    band = direct_budget(_Box(solver), spec, n_mc=args.budget_n_mc, seed=args.seed + 29)
    verdict = decide(
        budget=band,
        threshold=args.threshold,
        surrogate=None,
        mc=mc,
        cross_check=None,
        require_cross_check=False,
        min_samples=0,
        budget_certified_directly=True,
    )
    report = {
        "status_source": "montecarlo",
        "n_base": int(args.n_base),
        "n_model_evaluations": int(mc.n_model_evaluations),
        "indices": mc.as_dict(names=spec.names),
        "budget": band.as_dict(),
        "gate": verdict.as_dict(),
        "indices_source": "montecarlo",
        "_indices": indices_from_mc(mc),
    }
    if args.failsafe:
        report["failsafe"] = {
            "note": "the montecarlo mode already is the fail-safe route",
            "mode": "montecarlo",
        }
    return report


def _run_benchmark_json(args: argparse.Namespace) -> dict[str, Any]:
    artifact, records = parse_benchmarks(args.benchmark_json)
    out: dict[str, Any] = {
        "mode": "benchmark-json",
        "artifact": artifact.as_dict(),
        "n_records": len(records),
        "records": [r.as_dict() for r in records],
    }
    if args.list_qoi or not args.study_json:
        out["qoi_ids"] = [r.qoi_id for r in records]
        return out
    payload = json.loads(Path(args.study_json).read_text(encoding="utf-8"))
    rel_width = float(payload.get("budget", {}).get("relative_width", math.nan))
    if not math.isfinite(rel_width):
        raise SystemExit(f"{args.study_json} carries no budget.relative_width")
    budget = combine_budget(rel_width, records, threshold=args.threshold)
    out["combined_budget"] = budget.as_dict()
    return out


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tensorphysics.uq.cli",
        description="Uncertainty budget / Sobol' sensitivity CLI (PCE, sampled, fail-safe).",
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--pce", dest="mode", action="store_const", const="pce",
                       help="chaos surrogate only (analytic Sobol' indices)")
    modes.add_argument("--sobol", dest="mode", action="store_const", const="sobol",
                       help="chaos surrogate plus independent Saltelli cross-check (default)")
    modes.add_argument("--montecarlo", dest="mode", action="store_const", const="montecarlo",
                       help="quasi-Monte-Carlo indices only (no surrogate)")
    p.add_argument("--mode", choices=("pce", "sobol", "montecarlo"),
                   help="long form of the mode flag (--pce/--sobol/--montecarlo)")
    p.add_argument("--benchmark", default="ishigami", choices=tuple(BENCHMARKS),
                   help="benchmark / test function to analyse (default: ishigami)")
    p.add_argument("--benchmark-json", default=None,
                   help="parse a TensorLBM/TensorFVM/TensorFEM benchmark artefact instead")
    p.add_argument("--study-json", default=None,
                   help="with --benchmark-json: a UQ study JSON to combine budgets with")
    p.add_argument("--list-qoi", action="store_true", help="list extracted QoI ids and exit")
    p.add_argument("--dim", type=int, default=8, help="input dimension for holton/sobol-g")
    p.add_argument("--n", type=int, default=512, help="number of solver samples for the PCE fit")
    p.add_argument("--degree", type=int, default=3, help="total polynomial degree")
    p.add_argument("--method", default="lars", choices=("lars", "ols"), help="sparse fit method")
    p.add_argument("--n-base", type=int, default=1 << 13, help="pick-freeze base sample size")
    p.add_argument("--budget-n-mc", type=int, default=1 << 14,
                   help="samples for the output prediction interval")
    p.add_argument("--bootstrap", type=int, default=0, help="MC bootstrap resamples (0 = off)")
    p.add_argument("--seed", type=int, default=20240, help="master seed")
    p.add_argument("--rule", default="sobol", choices=("sobol", "lhs", "mc"),
                   help="sampling rule for the PCE training design")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="engineering threshold on the relative 95%% band (default 0.15)")
    p.add_argument("--q2-min", type=float, default=DEFAULT_Q2_MIN,
                   help="fail-safe quality floor on Q^2 (default 0.5)")
    p.add_argument("--eps-loo-max", type=float, default=DEFAULT_EPS_LOO_MAX,
                   help="surrogate LOO gate (default 0.1)")
    p.add_argument("--failsafe", action="store_true",
                   help="run the predeclared fail-safe policy in addition to the chosen mode")
    p.add_argument("--require-pass", action="store_true",
                   help="exit 3 when the gate status is not PASS")
    p.add_argument("--json", default=None, help="write the full report to this path")
    p.add_argument("--quiet", action="store_true", help="suppress the human-readable summary")
    return p


def _print_summary(report: dict[str, Any]) -> None:
    mode = report.get("mode", "?")
    print(f"mode: {mode}   benchmark: {report.get('benchmark', report.get('artifact', {}).get('path'))}")
    gate = report.get("gate") or (report.get("failsafe") or {}).get("verdict")
    if gate:
        print(f"gate: {gate.get('status')}  reasons: "
              f"{[r.get('code') for r in gate.get('reasons', [])]}")
    indices = report.get("indices") or (report.get("failsafe") or {}).get("indices")
    if isinstance(indices, dict):
        first = indices.get("first") or indices.get("first_jansen")
        total = indices.get("total")
        if first and total:
            print("  input            S_i        S_Ti")
            for nm in first:
                print(f"  {nm:<12} {first[nm]:10.4f} {total[nm]:10.4f}")
    comp = report.get("analytic_comparison") or {}
    if comp:
        print(f"vs analytic: max|dS|={comp['first_max_abs_error']:.3e} "
              f"max|dST|={comp['total_max_abs_error']:.3e}")
    budget = report.get("budget") or (report.get("failsafe") or {}).get("budget_primary")
    if budget:
        print(f"95% band: {budget['relative_width']:.4g} of the mean "
              f"(threshold {report.get('threshold', 0.15):.3g})")
    if report.get("qoi_ids"):
        print("QoI ids:")
        for q in report["qoi_ids"]:
            print(f"  {q}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = "sobol"
    if args.mode == "sobol" and args.benchmark_json:
        pass
    try:
        report = run(args)
    except SystemExit as exc:  # argparse-style error with a message
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report["threshold"] = args.threshold
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float) + "\n",
                                   encoding="utf-8")
    if not args.quiet:
        _print_summary(report)
    if args.require_pass:
        status = report.get("status") or (report.get("gate") or {}).get("status")
        if status is None:
            status = (report.get("failsafe") or {}).get("status")
        if status != PASS:
            return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())