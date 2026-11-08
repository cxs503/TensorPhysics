#!/usr/bin/env python3
"""Run the P35 gate's declared regression probes and record them.

Long probes (cavity_3d_full at 64³/100k steps) outlive one agent shell call, so
they are executed from here, detached, and appended to the records file after
each case.  The verdict is rendered separately from ``--snapshot`` + ``--records``
(see ``docs/P35_CAMPAIGN_INTEGRATION_GATE.md``).

    python3 scripts/p35_gate_probes.py --repo-root TensorLBM=/path/to/TensorLBM \
        --out artifacts/p35_gate --only couette_3d --timeout 900
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.campaign.p35_gate import collect as C  # noqa: E402
from tensorphysics.campaign.p35_gate import regression as R  # noqa: E402
from tensorphysics.campaign.p35_gate import spec as S  # noqa: E402

DEFAULT_OUT = "artifacts/p35_gate"
RECORDS_NAME = "regression_records.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", action="append", default=[], help="case name (repeatable)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    roots = C.parse_repo_roots(args.repo_root)
    records_path = Path(args.out) / RECORDS_NAME
    cases = [c for c in S.REGRESSION_CASES if not args.only or c.case in args.only]
    for case in cases:
        root = roots.get(case.repo) or roots.get(case.repo.lower())
        if not root:
            print(f"[skip] {case.case}: repository {case.repo} not bound", flush=True)
            continue
        print(f"[probe] {case.case}: running in {root} (timeout {args.timeout:.0f}s)", flush=True)
        result = R.run_probe(case, root, Path(args.out) / "probes", timeout=args.timeout)
        R.append_record(records_path, result)
        print(f"[probe] {case.case}: {result.status} | {result.detail}", flush=True)
    print(f"records: {records_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())