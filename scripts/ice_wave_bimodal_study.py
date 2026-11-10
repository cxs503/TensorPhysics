#!/usr/bin/env python3
"""Run node ``P82W5-N09`` and write ``artifacts/m8866/results.json``.

Usage::

    python3 scripts/ice_wave_bimodal_study.py            # full published run
    python3 scripts/ice_wave_bimodal_study.py --quick    # smoke run (<1 min)
    python3 scripts/ice_wave_bimodal_study.py --out artifacts/m8866/results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.ice_wave_fatigue.w5n09 import run_w5n09_study, summarise_criteria  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "m8866" / "results.json"))
    args = ap.parse_args()
    t0 = time.time()
    result = run_w5n09_study(quick=args.quick)
    result["criteria"] = summarise_criteria(result)
    result["wall_time_s"] = time.time() - t0
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {out} in {result['wall_time_s']:.1f} s")
    for key, val in result["criteria"].items():
        print(f"{key}: {json.dumps(val, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())