"""End-to-end runner for the DEM-FEM ice-structure acceptance harness.

Usage::

    PYTHONPATH=src python benchmarks/ice_structure_dem_fem/run_verification.py

Stages (each writes into ``artifacts/ice_structure_dem_fem``):

1. ``run_fast.py``      -- criteria (a)-(f) with an incremental JSON ledger
                           (heavy: several minutes on one core; run it on the
                           compute cluster for the full-length DEM spectra via
                           ``tensorphysics.ice_structure_dem_fem.verification``)
2. ``make_figures.py``  -- report figures from the ledger / saved histories
3. ``write_docs.py``    -- ``docs/ICE_STRUCTURE_DEM_FEM_P380W2N06.md`` with every
                           number injected from the ledger

``publish_report.py`` then posts the paper-grade report to TensorPlatform.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    env = {"PYTHONPATH": str(ROOT / "src")}
    for script in ("run_fast.py", "make_figures.py", "write_docs.py"):
        print(f"== {script}", flush=True)
        rc = subprocess.call([sys.executable, "-W", "ignore", str(HERE / script)],
                             cwd=str(ROOT), env={**env, "PATH": __import__("os").environ["PATH"]})
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
