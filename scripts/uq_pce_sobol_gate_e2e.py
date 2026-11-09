#!/usr/bin/env python3
"""End-to-end demonstration of the PCE Sobol' gate.

Runs the full UQ pipeline on two benchmark problems:

1. **Ishigami function** (d=3): analytic Sobol' indices are known in closed
   form, providing an absolute accuracy check.
2. **Sobol' G-function** (d=4): a standard sensitivity analysis benchmark
   with known first-order indices.

Outputs
-------
- ``<out>/ishigami_study.json``: full study results for Ishigami
- ``<out>/g_function_study.json``: full study results for G-function
- ``<out>/figures/``: convergence and comparison plots (PNG)

Usage::

    python scripts/uq_pce_sobol_gate_e2e.py [--out artifacts/uq_gate]

Exit code 0 if all acceptance criteria pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from functools import partial
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.uq import (  # noqa: E402
    ishigami,
    ishigami_analytic,
    ishigami_spec,
    run_uq_study,
    sobol_g,
    sobol_g_analytic,
    sobol_g_spec,
)

# Standard G-function coefficients (common benchmark)
G_FUNCTION_A = [0.0, 1.0, 2.0, 3.0]
G_FUNCTION_DIM = 4


def _round(value: float, digits: int = 6) -> float:
    return float(np.round(float(value), digits))


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Ishigami benchmark
# ---------------------------------------------------------------------------
def run_ishigami(out: pathlib.Path) -> dict[str, Any]:
    """Run the Ishigami benchmark and return the study payload."""
    print("=" * 72)
    print("Ishigami function (d=3)")
    print("=" * 72)

    spec = ishigami_spec()
    ref = ishigami_analytic()

    t0 = time.perf_counter()
    study = run_uq_study(
        ishigami,
        spec,
        n_samples=1024,
        degree=8,
        n_boot=200,
        bootstrap_seed=777,
        n_base_mc=1 << 14,
        mc_seed=5150,
        bootstrap_mc=100,
        budget_n_mc=10000,
    )
    elapsed = time.perf_counter() - t0

    pce = study.pce_sobol
    mc = study.mc
    assert pce is not None

    # Build results table
    verdict_dict = study.verdict.as_dict() if study.verdict else {}
    reason_codes = verdict_dict.get("reason_codes", [])
    reason_str = ", ".join(reason_codes) if reason_codes else "none"

    results: dict[str, Any] = {
        "benchmark": "ishigami",
        "elapsed_seconds": _round(elapsed, 2),
        "status": study.status,
        "verdict_reason": reason_str,
        "surrogate": {
            "eps_loo": _round(study.surrogate.eps_loo),
            "r2": _round(study.surrogate.r2),
            "n_terms": study.surrogate.n_terms,
            "n_samples": study.surrogate.n_samples,
        },
        "sample_hash": study.samples.hash,
        "indices": {},
    }

    # First-order indices
    print("\nFirst-order Sobol' indices:")
    print(f"{'Input':>8s}  {'Analytic':>10s}  {'PCE':>10s}  {'|Δ|':>10s}  {'MC (Jansen)':>12s}")
    print("-" * 58)
    for i, name in enumerate(["x1", "x2", "x3"]):
        exact = float(ref["first"][i])
        pce_val = float(pce.first[i])
        mc_val = float(mc.first[i]) if mc else float("nan")
        err = abs(pce_val - exact)
        results["indices"][name] = {
            "analytic": _round(exact),
            "pce": _round(pce_val),
            "abs_error": _round(err),
            "mc_jansen": _round(mc_val),
        }
        print(f"{name:>8s}  {exact:10.6f}  {pce_val:10.6f}  {err:10.2e}  {mc_val:12.6f}")

    # Total-effect indices
    print("\nTotal-effect Sobol' indices:")
    print(f"{'Input':>8s}  {'Analytic':>10s}  {'PCE':>10s}  {'|Δ|':>10s}  {'MC (Jansen)':>12s}")
    print("-" * 58)
    for i, name in enumerate(["x1", "x2", "x3"]):
        exact = float(ref["total"][i])
        pce_val = float(pce.total[i])
        mc_val = float(mc.total[i]) if mc else float("nan")
        err = abs(pce_val - exact)
        results["indices"][f"{name}_total"] = {
            "analytic": _round(exact),
            "pce": _round(pce_val),
            "abs_error": _round(err),
            "mc_jansen": _round(mc_val),
        }
        print(f"{name:>8s}  {exact:10.6f}  {pce_val:10.6f}  {err:10.2e}  {mc_val:12.6f}")

    # Second-order indices
    print("\nSecond-order Sobol' indices:")
    print(f"{'Pair':>8s}  {'Analytic':>10s}  {'PCE':>10s}  {'|Δ|':>10s}")
    print("-" * 44)
    for (i, j), name in [((0, 2), "x1*x3"), ((0, 1), "x1*x2"), ((1, 2), "x2*x3")]:
        exact = float(ref["second"].get((i, j), 0.0))
        pce_val = float(pce.second.get((i, j), 0.0))
        err = abs(pce_val - exact)
        results["indices"][name] = {
            "analytic": _round(exact),
            "pce": _round(pce_val),
            "abs_error": _round(err),
        }
        print(f"{name:>8s}  {exact:10.6f}  {pce_val:10.6f}  {err:10.2e}")

    # Acceptance checks
    worst_pce = max(
        abs(float(pce.first[0]) - float(ref["first"][0])),
        abs(float(pce.first[1]) - float(ref["first"][1])),
        abs(float(pce.total[0]) - float(ref["total"][0])),
        abs(float(pce.total[2]) - float(ref["total"][2])),
        abs(float(pce.second.get((0, 2), 0.0)) - float(ref["second"][(0, 2)])),
    )
    worst_mc = max(
        _safe_div(abs(float(mc.first[0]) - float(ref["first"][0])), float(ref["first"][0])),
        _safe_div(abs(float(mc.first[1]) - float(ref["first"][1])), float(ref["first"][1])),
        _safe_div(abs(float(mc.total[0]) - float(ref["total"][0])), float(ref["total"][0])),
        _safe_div(abs(float(mc.total[2]) - float(ref["total"][2])), float(ref["total"][2])),
    ) if mc else 0.0

    results["acceptance"] = {
        "A1_pce_vs_analytic": {
            "worst_abs_error": _round(worst_pce),
            "limit": 1e-3,
            "pass": bool(worst_pce <= 1e-3),
        },
        "A2_mc_vs_analytic": {
            "worst_rel_error": _round(worst_mc),
            "limit": 0.05,
            "pass": bool(worst_mc <= 0.05),
        },
    }

    # Cross-check
    if study.cross_check:
        results["cross_check"] = study.cross_check.as_dict()
        results["acceptance"]["A7_cross_check"] = {
            "max_rel_deviation": _round(study.cross_check.max_relative_deviation),
            "limit": 0.05,
            "pass": bool(study.cross_check.agreement and study.cross_check.max_relative_deviation <= 0.05),
        }

    # Output budget
    results["output_budget"] = study.budget.as_dict()

    # Write JSON
    out_path = out / "ishigami_study.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Sobol' G-function benchmark
# ---------------------------------------------------------------------------
def run_g_function(out: pathlib.Path) -> dict[str, Any]:
    """Run the Sobol' G-function benchmark and return the study payload."""
    print("\n" + "=" * 72)
    print(f"Sobol' G-function (d={G_FUNCTION_DIM}, a={G_FUNCTION_A})")
    print("=" * 72)

    spec = sobol_g_spec(G_FUNCTION_DIM)
    ref = sobol_g_analytic(G_FUNCTION_A)

    # Create a solver wrapper that binds the 'a' parameter
    solver = partial(sobol_g, a=G_FUNCTION_A)

    t0 = time.perf_counter()
    study = run_uq_study(
        solver,
        spec,
        n_samples=1024,
        degree=6,
        n_boot=200,
        bootstrap_seed=777,
        run_mc=False,
        budget_n_mc=10000,
    )
    elapsed = time.perf_counter() - t0

    pce = study.pce_sobol
    assert pce is not None

    # Build results table
    verdict_dict = study.verdict.as_dict() if study.verdict else {}
    reason_codes = verdict_dict.get("reason_codes", [])
    reason_str = ", ".join(reason_codes) if reason_codes else "none"

    results: dict[str, Any] = {
        "benchmark": "sobol_g",
        "dim": G_FUNCTION_DIM,
        "a_coefficients": G_FUNCTION_A,
        "elapsed_seconds": _round(elapsed, 2),
        "status": study.status,
        "verdict_reason": reason_str,
        "surrogate": {
            "eps_loo": _round(study.surrogate.eps_loo),
            "r2": _round(study.surrogate.r2),
            "n_terms": study.surrogate.n_terms,
            "n_samples": study.surrogate.n_samples,
        },
        "sample_hash": study.samples.hash,
        "indices": {},
    }

    # First-order indices
    print("\nFirst-order Sobol' indices:")
    print(f"{'Input':>8s}  {'Analytic':>10s}  {'PCE':>10s}  {'|Δ|':>10s}")
    print("-" * 44)
    for i, name in enumerate([f"x{j+1}" for j in range(G_FUNCTION_DIM)]):
        exact = float(ref["first"][i])
        pce_val = float(pce.first[i])
        err = abs(pce_val - exact)
        results["indices"][name] = {
            "analytic": _round(exact),
            "pce": _round(pce_val),
            "abs_error": _round(err),
        }
        print(f"{name:>8s}  {exact:10.6f}  {pce_val:10.6f}  {err:10.2e}")

    # Total-effect indices
    print("\nTotal-effect Sobol' indices:")
    print(f"{'Input':>8s}  {'Analytic':>10s}  {'PCE':>10s}  {'|Δ|':>10s}")
    print("-" * 44)
    for i, name in enumerate([f"x{j+1}" for j in range(G_FUNCTION_DIM)]):
        exact = float(ref["total"][i])
        pce_val = float(pce.total[i])
        err = abs(pce_val - exact)
        results["indices"][f"{name}_total"] = {
            "analytic": _round(exact),
            "pce": _round(pce_val),
            "abs_error": _round(err),
        }
        print(f"{name:>8s}  {exact:10.6f}  {pce_val:10.6f}  {err:10.2e}")

    # Acceptance check
    worst_pce = max(
        abs(float(pce.first[i]) - float(ref["first"][i])) for i in range(G_FUNCTION_DIM)
    )
    results["acceptance"] = {
        "A9_g_function": {
            "worst_abs_error": _round(worst_pce),
            "limit": 5e-3,
            "pass": bool(worst_pce <= 5e-3),
        },
    }

    results["output_budget"] = study.budget.as_dict()

    # Write JSON
    out_path = out / "g_function_study.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def generate_figures(out: pathlib.Path, ishigami_results: dict, g_results: dict) -> None:
    """Generate comparison and convergence figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping figure generation")
        return

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Ishigami first-order indices comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["x1", "x2", "x3"]
    analytic = [ishigami_results["indices"][n]["analytic"] for n in names]
    pce_vals = [ishigami_results["indices"][n]["pce"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, analytic, width, label="Analytic", color="steelblue")
    ax.bar(x + width / 2, pce_vals, width, label="PCE", color="coral")
    ax.set_xlabel("Input")
    ax.set_ylabel("First-order Sobol' index")
    ax.set_title("Ishigami function: first-order Sobol' indices")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "ishigami_first_order.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'ishigami_first_order.png'}")

    # Figure 2: Ishigami total-effect indices comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    analytic_t = [ishigami_results["indices"][f"{n}_total"]["analytic"] for n in names]
    pce_t = [ishigami_results["indices"][f"{n}_total"]["pce"] for n in names]

    ax.bar(x - width / 2, analytic_t, width, label="Analytic", color="steelblue")
    ax.bar(x + width / 2, pce_t, width, label="PCE", color="coral")
    ax.set_xlabel("Input")
    ax.set_ylabel("Total-effect Sobol' index")
    ax.set_title("Ishigami function: total-effect Sobol' indices")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "ishigami_total_effect.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'ishigami_total_effect.png'}")

    # Figure 3: G-function first-order indices
    fig, ax = plt.subplots(figsize=(8, 5))
    g_names = [f"x{i+1}" for i in range(g_results.get("dim", 4))]
    g_analytic = [g_results["indices"][n]["analytic"] for n in g_names]
    g_pce = [g_results["indices"][n]["pce"] for n in g_names]

    x_g = np.arange(len(g_names))
    ax.bar(x_g - width / 2, g_analytic, width, label="Analytic", color="steelblue")
    ax.bar(x_g + width / 2, g_pce, width, label="PCE", color="coral")
    ax.set_xlabel("Input")
    ax.set_ylabel("First-order Sobol' index")
    ax.set_title("Sobol' G-function: first-order Sobol' indices")
    ax.set_xticks(x_g)
    ax.set_xticklabels(g_names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "g_function_first_order.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'g_function_first_order.png'}")

    # Figure 4: Surrogate quality summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ishigami surrogate
    ax = axes[0]
    metrics = ["ε_LOO", "R²"]
    values = [ishigami_results["surrogate"]["eps_loo"], ishigami_results["surrogate"]["r2"]]
    colors = ["coral" if v > 0.1 else "steelblue" for v in values]
    ax.bar(metrics, values, color=colors)
    ax.set_title(f"Ishigami surrogate quality\n(n={ishigami_results['surrogate']['n_samples']}, "
                 f"terms={ishigami_results['surrogate']['n_terms']})")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="ε_LOO threshold")
    ax.legend()

    # G-function surrogate
    ax = axes[1]
    values_g = [g_results["surrogate"]["eps_loo"], g_results["surrogate"]["r2"]]
    colors_g = ["coral" if v > 0.1 else "steelblue" for v in values_g]
    ax.bar(metrics, values_g, color=colors_g)
    ax.set_title(f"G-function surrogate quality\n(n={g_results['surrogate']['n_samples']}, "
                 f"terms={g_results['surrogate']['n_terms']})")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="ε_LOO threshold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "surrogate_quality.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'surrogate_quality.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="PCE Sobol' gate end-to-end demo")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "artifacts" / "uq_gate",
                        help="Output directory (default: artifacts/uq_gate)")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out}")
    print()

    # Run benchmarks
    ishigami_results = run_ishigami(out)
    g_results = run_g_function(out)

    # Generate figures
    generate_figures(out, ishigami_results, g_results)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    all_pass = True
    for name, results in [("Ishigami", ishigami_results), ("G-function", g_results)]:
        print(f"\n{name}:")
        print(f"  Status: {results['status']}")
        for crit_name, crit in results["acceptance"].items():
            status = "PASS" if crit["pass"] else "FAIL"
            measured = crit.get("worst_abs_error", crit.get("worst_rel_error", crit.get("max_rel_deviation", "N/A")))
            print(f"  {crit_name}: {status} (measured={measured}, limit={crit['limit']})")
            if not crit["pass"]:
                all_pass = False

    print()
    if all_pass:
        print("All acceptance criteria PASSED.")
        return 0
    else:
        print("Some acceptance criteria FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
