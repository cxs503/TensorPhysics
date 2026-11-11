"""Run the multi-fidelity selection verification and write the published artifacts.

Usage (from the repository root)::

    python benchmarks/multifidelity/run_verification.py

Outputs
-------
``artifacts/multifidelity_selection/p9003w1_n05_results.json``
    every acceptance number, the backtest metric tables, the Pareto front, the
    physical anchor and the calibration provenance (machine-readable).
``artifacts/multifidelity_selection/fig_*.png``
    the six figures referenced by the report.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.multifidelity import run_verification  # noqa: E402
from tensorphysics.multifidelity.figures import (  # noqa: E402
    plot_budget_response,
    plot_confidence_sweep,
    plot_coverage,
    plot_error_ratio,
    plot_pareto_frontier,
    plot_recommendation_histogram,
)

OUT_DIR = ROOT / "artifacts" / "multifidelity_selection"
FIG_DIR = OUT_DIR / "figures"


def run_pytest() -> dict:
    """Run the module's test file and record the machine-checked count."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_multifidelity.py", "-q"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    match = re.search(r"(\d+) passed", proc.stdout)
    failed = re.search(r"(\d+) failed", proc.stdout)
    return {"command": " ".join(cmd), "returncode": proc.returncode,
            "n_tests": int(match.group(1)) if match else 0,
            "n_failed": int(failed.group(1)) if failed else 0,
            "last_line": tail[0]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pytest_summary = run_pytest()
    result = run_verification(pytest_summary=pytest_summary)
    artifact = result.artifact

    deployment = result.metrics["deployment"]
    figures = []
    figures.append(plot_recommendation_histogram(
        deployment, "deployment", FIG_DIR / "fig1_recommendation_histogram.png"))
    figures.append(plot_pareto_frontier(
        artifact["pareto"]["frontier"], artifact["pareto"]["all_models"],
        artifact["pareto"]["budget"], FIG_DIR / "fig2_pareto_frontier.png"))
    figures.append(plot_coverage(
        {truth: {v: m for v, m in variants.items()} for truth, variants in result.metrics.items()},
        nominal=artifact["protocol"]["interval_level"],
        threshold=0.85, out_path=FIG_DIR / "fig3_interval_coverage.png"))
    figures.append(plot_budget_response(
        [], {"prior_only": result.records["deployment"]["prior_only"],
             "conformal_stratified": result.records["deployment"]["conformal_stratified"]},
        {"accuracy": 0.90, "cost_efficiency": 0.90},
        FIG_DIR / "fig4_budget_response.png"))
    figures.append(plot_error_ratio(
        deployment, result.records["deployment"], FIG_DIR / "fig5_error_ratio.png"))
    figures.append(plot_confidence_sweep(
        artifact["confidence_sweep"], FIG_DIR / "fig6_confidence_sweep.png"))

    artifact["figures"] = [str(p.relative_to(ROOT)) for p in figures]
    artifact["pytest"] = pytest_summary
    out_json = OUT_DIR / "p9003w1_n05_results.json"
    out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"\nartifact -> {out_json.relative_to(ROOT)}")
    for p in figures:
        print(f"figure   -> {p.relative_to(ROOT)}")
    print("\nacceptance:")
    for key, value in artifact["acceptance"].items():
        if isinstance(value, bool):
            print(f"  {key:32s} {value}")
        else:
            print(f"  {key:32s} pass={value['pass']:<5} value={value['value']}")
    print("\nkey metrics (deployment truth):")
    header = f"  {'variant':26s} {'rec':>5s} {'acc':>6s} {'cost':>6s} {'cov':>6s} {'worst':>6s} {'oracle':>6s}"
    print(header)
    for name, m in deployment.items():
        print(f"  {name:26s} {m['recommendation_rate']:5.2f} {m['accuracy']:6.3f} "
              f"{m['cost_efficiency']:6.3f} {m['coverage']:6.3f} {m['coverage_worst_stratum']:6.3f} "
              f"{(m['oracle_optimal_rate'] or 0):6.2f}")
    return 0 if artifact["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
