#!/usr/bin/env python
"""One-click recomputation of the P78-N08 cross-solver attribution ledger.

Run from the repository root::

    python scripts/recompute_cross_solver_attribution.py            # full run
    python scripts/recompute_cross_solver_attribution.py --parts orders,factor
    python scripts/recompute_cross_solver_attribution.py --check    # full + gate

The script is a thin wrapper: all numbers come from
``tensorphysics.cross_solver_attribution.cli``, which merges the requested blocks
into ``artifacts/cross_solver_attribution/ledger.json``.  Nothing is typed in by
hand; the report generator reads the same JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/cross_solver_attribution/ledger.json")
    ap.add_argument("--parts", default="orders,factor,replay,sph_lbm,verdicts,figures")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero when any acceptance check fails")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from tensorphysics.cross_solver_attribution.cli import main as cli_main

    cli_main(["--out", args.out, "--parts", args.parts])

    with open(args.out) as fh:
        data = json.load(fh)
    acc = data.get("acceptance")
    if acc is None:
        print("acceptance block missing: run with --parts ...,verdicts", file=sys.stderr)
        return 1
    print(json.dumps(acc["checks"], indent=2))
    if args.check and not acc["all_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())