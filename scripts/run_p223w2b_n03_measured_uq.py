"""Run the measured-data UQ study of the ice-going digital twin (node p223w2b-n03).

Usage::

    PYTHONPATH=src python scripts/run_p223w2b_n03_measured_uq.py [--out DIR]

Writes ``report.json`` (all numbers, single source of truth for the figures and
for the published report) into ``artifacts/p223w2b_n03_measured_uq/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from tensorphysics.ice_digital_twin import measured_uq as muq

DEFAULT_OUT = os.path.join("artifacts", "p223w2b_n03_measured_uq")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--calib-steps", type=int, default=12000)
    ap.add_argument("--calib-burn", type=int, default=3000)
    ap.add_argument("--loo-steps", type=int, default=5000)
    ap.add_argument("--loo-burn", type=int, default=1500)
    ap.add_argument("--sob-order", type=int, default=16)
    ap.add_argument("--sob-base", type=int, default=1 << 14)
    ap.add_argument("--seed", type=int, default=20261)
    args = ap.parse_args(argv)

    camp = muq.load_measured_campaign()
    report = muq.run_measured_uq(
        campaign=camp,
        calib_steps=args.calib_steps,
        calib_burn=args.calib_burn,
        loo_steps=args.loo_steps,
        loo_burn=args.loo_burn,
        sob_order=args.sob_order,
        sob_n_base=args.sob_base,
        seed=args.seed,
    )
    report["double_entry"] = muq.double_entry_check(camp)
    report["run_settings"] = {
        "calib_steps": args.calib_steps,
        "calib_burn": args.calib_burn,
        "loo_steps": args.loo_steps,
        "loo_burn": args.loo_burn,
        "sob_order": args.sob_order,
        "sob_n_base": args.sob_base,
        "seed": args.seed,
    }
    path = muq.write_report(report, args.out)
    print(f"report written: {path}")
    for cid, crit in sorted(report["criteria"].items()):
        blob = json.dumps(crit["value"], default=float)
        print(f"criterion {cid}: {crit['status']} :: {blob[:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
