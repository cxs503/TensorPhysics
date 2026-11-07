#!/usr/bin/env python3
"""P34-G1 gate driver: collect evidence -> run regression slices -> judge.

Every subcommand is resumable and bounded, because the agent shell caps each call
at ~120 s: ``regression`` stops at ``--budget`` seconds and persists its per-file
records, so the same command can simply be called again to continue.

Typical sequence (see docs/p34_campaign_summary.md for the executed commands)::

    python scripts/p34_gate_run.py plan --ship-root /root/tp-agent/work/m5028 \
        --plan artifacts/p34_integration/regression_plan.json
    python scripts/p34_gate_run.py regression --plan ... --out artifacts/p34_integration --budget 100
    python scripts/p34_gate_run.py ingest-junit --out ... --suite platform_core --junit pc.xml
    python scripts/p34_gate_run.py assemble --plan ... --out ...
    python scripts/p34_gate_run.py collect --out artifacts/p34_integration --run-tests
    python scripts/p34_gate_run.py gate --out artifacts/p34_integration --report docs/p34_campaign_summary.md
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.p34_gate import (  # noqa: E402
    NODES, collect_campaign, run_gate, spec as S)
from tensorphysics.campaign.p34_gate.collect import check_png, sha256_file  # noqa: E402
from tensorphysics.campaign.p34_gate import regression as R  # noqa: E402

#: Capability families required by C3, mapped to file-name patterns in the ship
#: repository.  Only *pre-existing* test files are selected (no P34 node files),
#: so the suite measures regression, not new work.
FAMILY_PATTERNS = {
    "resistance": r"(resistance|holtrop|added_resistance)",
    "propulsion": r"(propuls|openwater|propeller|wake)",
    "seakeeping": r"(seakeep|seakeeping|motion|rao)",
    "maneuvering": r"(maneuver|manoeuv)",
    "stability": r"(stabilit|hydrostat|gz_)",
}
#: Files that need a live server / database are excluded from the offline suite.
EXCLUDE_PATTERNS = r"(platform_audit|api_client|deploy|frontend|smoke|live_|endpoint)"


def _cmd_plan(args) -> int:
    ship_root = Path(args.ship_root)
    tests_dir = ship_root / "tests"
    names = sorted(p.name for p in tests_dir.glob("test_*.py"))
    suites = []
    for fam, pat in FAMILY_PATTERNS.items():
        sel = [f"tests/{n}" for n in names
               if re.search(pat, n, re.I) and not re.search(EXCLUDE_PATTERNS, n, re.I)]
        sel = sel[: args.max_per_family]
        suites.append({"name": f"ship_{fam}", "family": fam,
                       "repo": "ship-performance-platform",
                       "root": str(ship_root), "tests": sel,
                       "note": "pre-existing test files selected by name pattern "
                               f"{pat!r}; P34 node files excluded"})
    core = [f"tests/{p.name}" for p in sorted((ROOT / "tests").glob("test_*.py"))]
    suites.append({"name": "platform_core", "family": "platform_core",
                   "repo": "TensorPhysics", "root": str(ROOT), "tests": core,
                   "note": "full TensorPhysics test suite (workflow / validation logic), "
                           "every file of tests/ run on this host and recorded per batch"})
    plan = {"schema": "p34_regression_plan.v1", "ship_root": str(ship_root),
            "suites": suites}
    out = Path(args.plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"plan": str(out),
                      "suites": {s["name"]: len(s["tests"]) for s in suites}},
                     ensure_ascii=False, indent=2))
    return 0


def _load_specs(plan_path: Path) -> list[R.SuiteSpec]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    return [R.SuiteSpec(name=s["name"], family=s["family"], repo=s["repo"],
                        root=Path(s["root"]), tests=tuple(s["tests"]),
                        note=s.get("note", ""))
            for s in plan["suites"]]


def _cmd_regression(args) -> int:
    specs = _load_specs(Path(args.plan))
    out = Path(args.out)
    results = []
    for spec in specs:
        if args.only and spec.name not in args.only:
            continue
        if spec.name == "platform_core" and not args.include_platform_core:
            continue
        results.append(R.run_suite(spec, out, budget_s=args.budget,
                                   per_file_timeout=args.per_file_timeout,
                                   group_size=args.group_size))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _cmd_ingest_junit(args) -> int:
    out = Path(args.out)
    d = out / args.suite
    (d / "junit").mkdir(parents=True, exist_ok=True)
    src = Path(args.junit)
    dst = d / "junit" / src.name
    shutil.copyfile(src, dst)
    from tensorphysics.campaign.p34_gate.collect import _parse_junit
    counts = _parse_junit(dst)
    rec_path = d / "records.json"
    records = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else []
    records = [r for r in records if r["test"] != args.test]
    records.append({"test": args.test, **counts, "exit_code": args.exit_code,
                    "duration_s": args.duration_s, "arch": args.arch,
                    "head_sha": args.head_sha, "junit": str(dst),
                    "note": args.note, "family": args.family})
    rec_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"suite": args.suite, "test": args.test, **counts,
                      "exit_code": args.exit_code}, ensure_ascii=False))
    return 0


def _cmd_assemble(args) -> int:
    specs = _load_specs(Path(args.plan))
    doc = R.write_regression(specs, Path(args.out), Path(args.out) / "regression.json")
    print(json.dumps({"n_suites": doc["n_suites"], "n_green": doc["n_green"],
                      "suites": [{k: s[k] for k in
                                  ("name", "family", "exit_code", "collected", "passed",
                                   "failed", "errors", "n_covered", "n_missing")}
                                 for s in doc["suites"]]}, ensure_ascii=False, indent=2))
    return 0


def _cmd_collect(args) -> int:
    out = Path(args.out)
    reports = {}
    if args.report_index and Path(args.report_index).is_file():
        reports = json.loads(Path(args.report_index).read_text(encoding="utf-8"))
    nodes = NODES
    if args.node:
        nodes = tuple(n for n in NODES if n.ext_id in set(args.node))
    roots = {k: Path(v) for k, v in json.loads(args.roots).items()} if args.roots else None
    index = collect_campaign(nodes=nodes, roots=roots, template=args.template,
                             out_dir=out, run_tests=args.run_tests,
                             test_timeout=args.node_timeout, reports=reports)
    print(json.dumps(index["summary"], ensure_ascii=False, indent=2))
    for n in index["nodes"]:
        print(f"{n['ext_id']}: worktree={n['worktree']} gaps={n['gaps']}")
    return 0


def _cmd_gate(args) -> int:
    out = Path(args.out)
    collection = json.loads((out / "collection_index.json").read_text(encoding="utf-8"))
    reg = None
    if (out / "regression.json").is_file():
        reg = json.loads((out / "regression.json").read_text(encoding="utf-8"))
    report = Path(args.report) if args.report else None
    deliverables = [p for p in (out / "gate_verdict.json", out / "collection_index.json",
                                report) if p and Path(p).is_file()]
    verdict = run_gate(collection=collection, regression=reg, report_path=report,
                       pkg_dir=ROOT / "src/tensorphysics/campaign/p34_gate",
                       deliverables=deliverables, repo_root=ROOT, out_dir=out)
    (out / "gate_verdict.md").write_text(verdict["markdown"], encoding="utf-8")
    print(json.dumps({"overall_status": verdict["overall_status"],
                      "criterion_summary": verdict["criterion_summary"],
                      "blocking": verdict["blocking_criteria"]}, ensure_ascii=False, indent=2))
    for c in verdict["criteria"]:
        print(f"{c['criterion_id']} {c['status']:9s} {c['detail']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--ship-root", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--max-per-family", type=int, default=8)
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("regression")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=100.0)
    p.add_argument("--per-file-timeout", type=float, default=240.0)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--include-platform-core", action="store_true", default=True)
    p.add_argument("--group-size", type=int, default=2)
    p.set_defaults(func=_cmd_regression)

    p = sub.add_parser("ingest-junit")
    p.add_argument("--out", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--junit", required=True)
    p.add_argument("--exit-code", type=int, default=0)
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--arch", default="")
    p.add_argument("--head-sha", default="")
    p.add_argument("--note", default="")
    p.set_defaults(func=_cmd_ingest_junit)

    p = sub.add_parser("assemble")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_assemble)

    p = sub.add_parser("collect")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default="/root/tp-agent/work/m{mission_id}")
    p.add_argument("--roots", default=None, help="JSON map ext_id -> worktree path")
    p.add_argument("--node", nargs="*", default=None)
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--node-timeout", type=float, default=600.0)
    p.add_argument("--report-index", default=None)
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("gate")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    p.set_defaults(func=_cmd_gate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())