"""CLI: ``python -m tensorphysics.metrology.criteria <command>``.

Commands
--------
``replay``   re-judge the repository's historical criteria and write the degeneracy list
``controls`` run the 24 negative controls (12 degenerate + 12 benign)
``labeled``  run the 40-record labelled agreement set (pre-registered stop rule)
``all``      all of the above into one report JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .controls import evaluate_controls, evaluate_labeled
from .replay import json_safe, summarize


def _dump(payload: Any, path: str | None) -> None:
    text = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {path}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tensorphysics.metrology.criteria", description=__doc__)
    ap.add_argument("command", choices=("replay", "controls", "labeled", "all"))
    ap.add_argument("--root", default=os.getcwd(), help="repository root to harvest (read-only)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--out-dir", default=None, help="directory for per-command JSON files")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "controls":
        payload = evaluate_controls()
        payload.pop("_rows", None)
        _dump(payload, args.out)
        return 0

    if args.command == "labeled":
        _dump(evaluate_labeled(), args.out)
        return 0

    out = args.out
    if args.command == "all" and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        controls = evaluate_controls()
        controls.pop("_rows", None)
        _dump(controls, os.path.join(args.out_dir, "criteria_controls.json"))
        _dump(evaluate_labeled(), os.path.join(args.out_dir, "criteria_labeled40.json"))
        out = out or os.path.join(args.out_dir, "criteria_replay.json")
    payload = summarize(args.root, out_path=out)
    if args.quiet:
        kpi = payload["kpi"]
        print(json.dumps({k: kpi[k] for k in ("n_records", "n_degraded", "degraded_fraction",
                                             "n_flips", "alternative_coverage", "meets_corpus_size",
                                             "meets_degraded_fraction")}, indent=2))
        print(json.dumps(payload["controls"]["degenerate_hit_rate"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())