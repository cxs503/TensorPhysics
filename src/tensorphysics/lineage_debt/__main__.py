"""Command line surface of the lineage-inheritance debt ledger.

Three verbs, all deterministic:

``replay``    snapshot bundle -> per-plan ledger JSON files (+ gate verdicts)
``query``     filter one ledger (state / coverage / domain / quantity / ...)
``selfcheck`` deny-list scan proving the logic carries no domain vocabulary

Examples
--------
``python -m tensorphysics.lineage_debt replay --bundle B.json --out runs/lineage_debt``
``python -m tensorphysics.lineage_debt query --ledger runs/lineage_debt/ledger_plan173.json --state debt_open``
``python -m tensorphysics.lineage_debt selfcheck``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .gate import seeding_check
from .model import canonical_json, dedupe_debts
from .query import load_ledger, query
from .replay import build_ledger, load_bundle, write_ledger
from .selfcheck import selfcheck


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    deferrals: dict[str, str] = {}
    if args.deferrals:
        deferrals = json.loads(Path(args.deferrals).read_text(encoding="utf-8")).get("deferrals", {})
    bundle = load_bundle(args.bundle)
    plans = [int(p) for p in args.plans.split(",")] if args.plans else sorted(bundle.plans)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for plan_id in plans:
        ledger = build_ledger(bundle, plan_id, cfg, deferrals)
        path = write_ledger(ledger, out_dir / f"ledger_plan{plan_id}.json")
        gate = ledger["seed_gate"]
        summary.append({
            "plan_id": plan_id,
            "ledger": str(path),
            "debts": ledger["summary"]["total"],
            "coverage": ledger["summary"]["by_coverage"],
            "verdict": gate["verdict"],
            "missing": len(gate["missing"]),
        })
    print(json.dumps({"schema": "lineage_debt_replay.v1", "plans": summary}, ensure_ascii=False,
                     indent=2, sort_keys=True))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    ledger = load_ledger(args.ledger)
    rows = query(ledger, plan_id=args.plan_id, state=args.state, coverage=args.coverage,
                 domain=args.domain, quantity=args.quantity, source_kind=args.source_kind,
                 min_abs_err_pct=args.min_abs_err_pct, node_id=args.node_id,
                 mission_id=args.mission_id, report_id=args.report_id)
    if args.json:
        sys.stdout.write(canonical_json({"count": len(rows), "rows": rows}))
        return 0
    print(f"{'key':44s} {'state':12s} {'coverage':18s} {'err%':>8s} report")
    for row in rows:
        err = "" if row.get("err_pct") is None else f"{row['err_pct']:8.2f}"
        print(f"{str(row.get('key'))[:44]:44s} {str(row.get('state'))[:12]:12s} "
              f"{str(row.get('coverage'))[:18]:18s} {err} {row.get('report_id')}")
    print(f"-- {len(rows)} row(s)")
    return 0


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = selfcheck(cfg, root=Path(args.root) if args.root else None)
    sys.stdout.write(canonical_json(report))
    return 0 if report["verdict"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tensorphysics.lineage_debt", description=__doc__)
    p.add_argument("--config", default=None, help="vocabulary JSON (default: shipped vocab)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="snapshot bundle -> per-plan debt ledgers")
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--plans", default=None, help="comma separated plan ids (default: all in bundle)")
    r.add_argument("--deferrals", default=None, help="JSON file with {'deferrals': {key: reason}}")
    r.set_defaults(func=_cmd_replay)

    q = sub.add_parser("query", help="filter one ledger")
    q.add_argument("--ledger", required=True)
    q.add_argument("--plan-id", type=int, default=None)
    q.add_argument("--state", default=None)
    q.add_argument("--coverage", default=None)
    q.add_argument("--domain", default=None)
    q.add_argument("--quantity", default=None)
    q.add_argument("--source-kind", dest="source_kind", default=None)
    q.add_argument("--min-abs-err-pct", dest="min_abs_err_pct", type=float, default=None)
    q.add_argument("--node-id", dest="node_id", type=int, default=None)
    q.add_argument("--mission-id", dest="mission_id", type=int, default=None)
    q.add_argument("--report-id", dest="report_id", type=int, default=None)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=_cmd_query)

    s = sub.add_parser("selfcheck", help="deny-list scan of the package logic")
    s.add_argument("--root", default=None)
    s.set_defaults(func=_cmd_selfcheck)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())