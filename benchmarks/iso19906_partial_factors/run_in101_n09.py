"""IN101-N09 runner — ISO 19906 ice-force partial-factor back-calibration (mature).

Executes :func:`tensorphysics.ice_partial_factors.target_inverse_study.run_study`
and writes

* ``artifacts/iso19906_target_inverse/in101_n09_results.json``  (every number)
* ``artifacts/iso19906_target_inverse/fig*.png``                (six figures)
* ``docs/reports/in101_n09/iso19906_target_inverse.md``         (the report)

The report text is generated from the JSON, never from hand-typed values.

Usage::

    PYTHONPATH=src python3 benchmarks/iso19906_partial_factors/run_in101_n09.py [--quick]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, HERE)

from tensorphysics.ice_partial_factors.target_inverse_study import run_study  # noqa: E402

import in101_n09_figures  # noqa: E402
import in101_n09_report  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "artifacts", "iso19906_target_inverse")
REPORT_DIR = os.path.join(REPO_ROOT, "docs", "reports", "in101_n09")
JSON_PATH = os.path.join(OUT_DIR, "in101_n09_results.json")
REPORT_PATH = os.path.join(REPORT_DIR, "iso19906_target_inverse.md")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    config = {}
    if quick:
        config = {"samples_mc": 200_000, "samples_is": 50_000}

    results = run_study(config)
    results["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runner": "benchmarks/iso19906_partial_factors/run_in101_n09.py",
        "quick": quick,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    figures = in101_n09_figures.make_all(results, OUT_DIR)
    report = in101_n09_report.build_report(results, figures, JSON_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)

    summary = results["criteria_summary"]
    solution = results["inverse"]["solution"]
    print(f"gamma_f = {solution['gamma_f']:.6f}  psi = {solution['psi']:.6f}  "
          f"kappa = {solution['kappa_star']:.6f}  beta = {solution['beta_achieved']:.9f}")
    print(f"criteria: {summary['passed']}/{summary['total']} passed"
          + ("" if summary["all_passed"] else f"  failed={summary['failed']}"))
    print(f"json    : {JSON_PATH}")
    print(f"report  : {REPORT_PATH}")
    print(f"figures : {len(figures)}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())