"""ISO 19906 partial-factor calibration benchmark runner (P46W3-N09 maturity).

Runs the full calibration study and writes:
- artifacts/iso19906/results.json (all numbers)
- artifacts/iso19906/summary.md (machine-readable summary)

Usage:
    PYTHONPATH=src python benchmarks/iso19906/run_benchmark.py [--quick]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tensorphysics.iso19906_calibration import (
    run_full_calibration,
    export_results_json,
    summary_table,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv

    out_dir = os.path.join(REPO_ROOT, "artifacts", "iso19906")
    os.makedirs(out_dir, exist_ok=True)

    config = {}
    if quick:
        config = {"samples_mc": 200_000, "samples_is": 50_000}

    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting ISO 19906 calibration...")
    results = run_full_calibration(config=config)

    json_path = os.path.join(out_dir, "results.json")
    export_results_json(results, json_path)

    # Write summary
    sol = results["inverse"]["solution"]
    summary = results["criteria_summary"]
    lines = [
        "# ISO 19906 Partial-Factor Calibration — Benchmark Summary",
        "",
        f"**Date**: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"**Node**: P46W3-N09 (maturity)",
        f"**Target β_T**: {results['study']['target_beta']:.2f}",
        "",
        "## Calibrated Factors",
        "",
        f"| Factor | Value |",
        f"|---|---|",
        f"| γ_f | {sol['gamma_f']:.6f} |",
        f"| ψ | {sol['psi']:.6f} |",
        f"| κ = γ_f·ψ | {sol['kappa_star']:.6f} |",
        f"| β achieved | {sol['beta_achieved']:.12f} |",
        f"| |β − β_T| | {abs(sol['beta_achieved'] - 4.0):.2e} |",
        "",
        "## Machine Criteria",
        "",
        summary_table(results),
        "",
        f"**Summary**: {summary['passed']}/{summary['total']} passed",
        f"**All passed**: {summary['all_passed']}",
        "",
        "## Files",
        "",
        f"- Full results: `{json_path}`",
    ]

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"γ_f = {sol['gamma_f']:.6f}  ψ = {sol['psi']:.6f}  "
          f"κ = {sol['kappa_star']:.6f}  β = {sol['beta_achieved']:.9f}")
    print(f"criteria: {summary['passed']}/{summary['total']} passed"
          + ("" if summary["all_passed"] else f"  failed={summary['failed']}"))
    print(f"json    : {json_path}")
    print(f"summary : {summary_path}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
