#!/usr/bin/env python3
"""Run the W3-N06 case: ice-water propeller cavitation and radiated noise.

Usage::

    PYTHONPATH=src python3 scenarios/ice_cavitation_noise/run_case.py \
        [--outdir artifacts/ice_cavitation_noise]

Writes JSON artifacts and figures, prints the acceptance criteria.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.domains.ice_cavitation_noise.case_runner import (  # noqa: E402
    CaseConfig,
    run_case,
    write_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "ice_cavitation_noise"))
    parser.add_argument("--n-cells", type=int, default=120)
    args = parser.parse_args()

    config = CaseConfig(n_cells=args.n_cells)
    result, spectra, series = run_case(config)
    paths = write_artifacts(result, spectra, series, args.outdir)

    print(json.dumps(result.criteria, indent=2))
    print(json.dumps({"files": paths["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())