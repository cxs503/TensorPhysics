#!/usr/bin/env python3
"""Benchmark / acceptance runner for the Cowper-Symonds ice-impact DAF module.

Runs every machine-checkable acceptance criterion of node ``p351w1_n05`` and
writes the numbers to ``benchmarks/rate_plasticity_daf/results.json`` so that
every value quoted in the report is machine-read from the artefact.

Usage
-----
    python benchmarks/rate_plasticity_daf/run_verification.py            # full
    python benchmarks/rate_plasticity_daf/run_verification.py --quick    # no surface
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for candidate in (_REPO / "src", _REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

from tensorphysics.plasticity.verification_daf import (  # noqa: E402
    LITERATURE_DAF_MAX_HALF_SINE,
    LITERATURE_FLOW_STRESS_RATIO_AT_100,
    daf_surface,
    run_all_criteria,
)

OUT = _HERE / "results.json"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the DAF(tau,P) surface")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    started = time.time()
    report = run_all_criteria(include_surface=not args.quick)
    report["runner"] = {
        "script": str(pathlib.Path(__file__).relative_to(_REPO)),
        "elapsed_s": round(time.time() - started, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "quick": bool(args.quick),
    }
    report["literature_anchors"] = {
        "cowper_symonds_D_p_mild_steel": [40.4, 5.0],
        "peak_undamped_half_sine_daf": LITERATURE_DAF_MAX_HALF_SINE,
        "mild_steel_flow_stress_ratio_100_per_s": LITERATURE_FLOW_STRESS_RATIO_AT_100,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=_json_default))

    print(f"criteria passed: {len(report['passed'])} / {len(report['criteria'])}")
    if report["failed"]:
        print("FAILED criteria:")
        for name in report["failed"]:
            print("  -", name)
    for crit in report["criteria"]:
        print(
            f"  [{'PASS' if crit['passed'] else 'FAIL'}] {crit['name']:42s} "
            f"err={crit['relative_error']:.3e} tol={crit['tolerance']:.3g}"
        )
    print(f"artefact: {out}")
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())