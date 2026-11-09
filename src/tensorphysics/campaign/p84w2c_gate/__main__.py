"""CLI: ``python -m tensorphysics.campaign.p84w2c_gate``.

Examples
--------
Collect every node from the live worktrees and publish the verdict::

    python -m tensorphysics.campaign.p84w2c_gate \
        --out artifacts/p84w2c_gate --doc docs/P84W2C_CAMPAIGN_GATE.md \
        --auto-root-template /root/tp-agent/work/m{mission_id} \
        --consumer-root ship-performance-platform=/root/tp-agent/work/m8018 \
        --api auto

Offline / CI mode (no platform evidence, synthetic or absent node roots)::

    python -m tensorphysics.campaign.p84w2c_gate --out /tmp/gate --no-pytest --no-api
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import collect as C
from .gate import run_gate, summarise
from .spec import NODES, PARENT_PLAN_ID

REPRODUCE = (
    "python -m tensorphysics.campaign.p84w2c_gate \\",
    "    --out artifacts/p84w2c_gate --doc docs/P84W2C_CAMPAIGN_GATE.md \\",
    "    --auto-root-template '/root/tp-agent/work/m{mission_id}' \\",
    "    --consumer-root ship-performance-platform=<ship-platform 检出> \\",
    "    --plan-id 385 --parent-plan-id 365 --api auto",
    "python -m pytest tests/test_p84w2c_gate.py -q",
)


def _parse_kv(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--node-root/--consumer-root expects KEY=PATH, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/p84w2c_gate", help="output directory")
    parser.add_argument("--doc", default="docs/P84W2C_CAMPAIGN_GATE.md",
                        help="campaign markdown path")
    parser.add_argument("--node-root", action="append", metavar="KEY=PATH",
                        help="explicit node worktree, repeatable (KEY = P84W2C-N01 ..)")
    parser.add_argument("--auto-root-template", default=None, metavar="TEMPLATE",
                        help="fallback path template using {mission_id}")
    parser.add_argument("--consumer-root", action="append", metavar="REPO=PATH",
                        help="consumer repository checkout (ship-performance-platform=...)")
    parser.add_argument("--plan-id", type=int, default=385)
    parser.add_argument("--parent-plan-id", type=int, default=PARENT_PLAN_ID)
    parser.add_argument("--plan-title", default="P84-W2C seed (谱系继承波·Plan 84 Wave 2C)")
    parser.add_argument("--api", choices=("auto", "off"), default="auto",
                        help="auto: login with TENSOR_AGENT_USER/PASS and fetch plan state")
    parser.add_argument("--no-pytest", action="store_true", help="skip suite execution")
    parser.add_argument("--pytest-timeout", type=int, default=900)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--summary-json", default=None,
                        help="also write the one-line summary to this path")
    parser.add_argument("--pytest-only", action="store_true",
                        help="only execute/normalise the selected node suites, then exit")
    parser.add_argument("--only", default=None,
                        help="comma separated node keys for --pytest-only "
                             "(default: all nodes)")
    args = parser.parse_args(argv)

    explicit = _parse_kv(args.node_root)
    roots: dict[str, Path] = {}
    for spec in NODES:
        if spec.node_key in explicit:
            roots[spec.node_key] = Path(explicit[spec.node_key])
        elif args.auto_root_template:
            roots[spec.node_key] = Path(args.auto_root_template.format(mission_id=spec.mission_id))

    consumers = {k: Path(v) for k, v in _parse_kv(args.consumer_root).items()}

    if args.pytest_only:
        selected = [item.strip() for item in args.only.split(",")] if args.only else \
            [spec.node_key for spec in NODES]
        evidence = Path(args.out) / "node_evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        for spec in NODES:
            if spec.node_key not in selected:
                continue
            root = roots.get(spec.node_key)
            if root is None:
                continue
            record = C.run_pytest(root, spec.tests, evidence / f"{spec.node_key}.pytest.json",
                                  timeout=args.pytest_timeout)
            print(f"{spec.node_key}: mode={record['mode']} status={record['status']} "
                  f"collected={record.get('collected')} passed={record.get('passed')} "
                  f"failed={record.get('failed')} errors={record.get('errors')}")
        return 0

    platform = None
    parent_nodes = None
    if args.api == "auto":
        base, token = C.env_token()
        if token:
            platform = C.collect_platform(base, token, args.plan_id,
                                          [spec.mission_id for spec in NODES])
            parent = C.collect_platform(base, token, args.parent_plan_id, [])
            parent_nodes = parent.get("nodes", {}) if parent.get("available") else None
        else:
            platform = {"available": False, "plan_id": args.plan_id,
                        "reason": "login unavailable (env token missing or rejected)"}

    report = run_gate(
        node_roots=roots,
        out_dir=Path(args.out),
        consumer_roots=consumers,
        plan_id=args.plan_id,
        plan_title=args.plan_title,
        platform=platform,
        parent_nodes=parent_nodes,
        doc_path=Path(args.doc),
        run_pytest=not args.no_pytest,
        pytest_timeout=args.pytest_timeout,
        make_figure=not args.no_figure,
        reproduce=REPRODUCE,
    )
    line = summarise(report)
    print(line)
    print(f"json  -> {Path(args.out) / 'campaign_gate_report.json'}")
    print(f"md    -> {args.doc}")
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps({"summary": line}, ensure_ascii=False),
                                           encoding="utf-8")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())