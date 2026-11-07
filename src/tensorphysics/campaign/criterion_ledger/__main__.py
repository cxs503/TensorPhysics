"""CLI for the criterion-ledger mutation runner (cluster sharding friendly).

Examples
--------
Single shard, one gate::

    python3 -m tensorphysics.campaign.criterion_ledger.runner \
        --gate <GATE_ID> --work-root /path/to/upstream-tree \
        --audit-tool /path/to/upstream-tree/tools/audit/<audit tool>.py \
        --classes native --out /path/to/out --shard 0/1 --jobs 16

M1 monotonic curve (one eps level per invocation)::

    ... --classes M1_numeric_perturb --m1-eps 0.01 --out /path/out_eps01

Aggregate shards::

    python3 -m tensorphysics.campaign.criterion_ledger.runner \
        --aggregate /path/to/out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import gates as G
from .adapter import Ledger
from .runner import (
    aggregate,
    aggregate_full_ledger,
    run_full_ledger_shard,
    run_gate,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="criterion-ledger mutation runner")
    ap.add_argument("--gate", default=None, help="gate_id (default ledger)")
    ap.add_argument("--work-root", required=True,
                    help="root of the tree containing the gate layout")
    ap.add_argument("--tp-root", default=None,
                    help="pinned upstream tree root (upstream gate criteria)")
    ap.add_argument("--audit-tool", default=None,
                    help="path to the upstream mutation-audit tool")
    ap.add_argument("--classes", default=None,
                    help="comma-separated mutant classes (default: gate spec)")
    ap.add_argument("--m1-eps", type=float, default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=None, help="output dir for shard JSON")
    ap.add_argument("--shard", default="0/1", help="i/n shard selector")
    ap.add_argument("--aggregate", default=None,
                    help="aggregate per-gate shard JSONs in this dir and exit")
    ap.add_argument("--full-ledger", action="store_true",
                    help="run one shard of the full ledger (all gates)")
    ap.add_argument("--aggregate-full", default=None,
                    help="aggregate full-ledger shards in this dir and exit")
    a = ap.parse_args(argv)

    if a.aggregate:
        res = aggregate(Path(a.aggregate))
        print(json.dumps({k: res[k] for k in
                          ("gate", "total", "killed", "survived", "kill_rate")},
                         ensure_ascii=False, indent=2))
        return 0

    if a.aggregate_full:
        full = aggregate_full_ledger(Path(a.aggregate_full))
        kpi = full["kpi"]
        print(json.dumps({
            "gates": kpi["gates"], "total_mutants": kpi["total_mutants"],
            "killed": kpi["killed"],
            "overall_kill_rate": kpi["overall_kill_rate"],
            "per_gate": {g: {"total": v["total"], "killed": v["killed"],
                             "kill_rate": v["kill_rate"]}
                         for g, v in full["per_gate"].items()},
        }, ensure_ascii=False, indent=2))
        return 0

    repo_root = Path(a.work_root)
    tp_root = Path(a.tp_root) if a.tp_root else None
    led = G.build_default_ledger(repo_root, tp_root=tp_root)

    if a.full_ledger:
        if not a.out:
            ap.error("--out is required with --full-ledger")
        work_roots = {}
        for g in led.gates():
            work_roots[g.gate_id] = (tp_root if g.upstream_repo is not None
                                     else repo_root)
        shard_id, n_shards = (int(x) for x in a.shard.split("/"))
        run_full_ledger_shard(led, work_roots, shard_id=shard_id,
                             n_shards=n_shards, out_dir=Path(a.out),
                             jobs=a.jobs, timeout=a.timeout,
                             audit_tool=Path(a.audit_tool) if a.audit_tool
                             else None)
        print(json.dumps({"shard": a.shard, "out": a.out}))
        return 0

    if not a.gate:
        ap.error("--gate is required (or use --full-ledger/--aggregate)")
    gate = led.gate(a.gate)
    # the gate's layout root: upstream gates live in the pinned upstream
    # tree, in-repo gates live in this repo's worktree
    work_root = tp_root if gate.upstream_repo is not None else repo_root
    classes = (tuple(c.strip() for c in a.classes.split(",") if c.strip())
               if a.classes else None)
    shard_id, n_shards = (int(x) for x in a.shard.split("/"))
    res = run_gate(gate, work_root, jobs=a.jobs, timeout=a.timeout,
                   out_dir=Path(a.out) if a.out else None,
                   shard_id=shard_id, n_shards=n_shards, classes=classes,
                   audit_tool=Path(a.audit_tool) if a.audit_tool else None,
                   per_class=a.per_class, m1_eps=a.m1_eps)
    print(json.dumps({k: res[k] for k in
                      ("gate", "total", "killed", "survived")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
