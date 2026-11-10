"""Command line surface of the revival registry.

::

    python -m tensorphysics.fail_revival extract    [--corpus DIR]
    python -m tensorphysics.fail_revival graph      [--corpus DIR]
    python -m tensorphysics.fail_revival acceptance [--out-dir DIR]
    python -m tensorphysics.fail_revival query      --db FILE (--fail KEY | --task KEY | --criterion SHA)
    python -m tensorphysics.fail_revival gate       --payload FILE [--corpus DIR]

Every command prints canonical JSON on stdout; nothing is written unless asked
for, so the tool is safe to run inside an audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import acceptance as acceptance_mod
from . import gate as gate_mod
from . import lineage as lin
from .blockers import BlockerState, build_graph, load_declarations
from .extract import extract_directory
from .model import canonical_json

DEFAULT_CORPUS = Path("tensor_reports") / "campaign_nodes"


def _registry(corpus: Path):
    extraction = extract_directory(corpus)
    records = list(extraction.records)
    blockers, _ = load_declarations(corpus / "revival_blockers.json")
    deps: dict[int, tuple[int, ...]] = {}
    for path in sorted(corpus.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") == "campaign_node_artifact.v1" and data.get("depends_on_missions"):
            deps[int(data["mission_id"])] = tuple(int(d) for d in data["depends_on_missions"])
    graph = build_graph(records, blockers, mission_deps=deps,
                        provenance={"corpus": str(corpus)})
    return extraction, records, blockers, graph


def cmd_extract(args) -> dict:
    extraction = extract_directory(args.corpus)
    return extraction.to_dict()


def cmd_graph(args) -> dict:
    _, records, blockers, graph = _registry(args.corpus)
    out = graph.to_dict()
    out["resolvability_now"] = graph.resolvability(BlockerState())
    out["records"] = len(records)
    out["blockers"] = len(blockers)
    return out


def cmd_acceptance(args) -> dict:
    result = acceptance_mod.run(out_dir=args.out_dir, sample=args.sample)
    return {"summary": result["summary"], "acceptance_path": result["acceptance_path"]}


def cmd_query(args) -> dict:
    conn = lin.connect(args.db)
    if args.fail:
        return {"fail_key": args.fail, "edges": lin.lineage_of(conn, args.fail)}
    if args.task:
        return {"task_key": args.task, "edges": lin.edge_of_task(conn, args.task)}
    if args.criterion:
        return {"criterion_sha256": args.criterion,
                "edges": lin.lineage_of_criterion(conn, args.criterion)}
    return {"audit_ledger": lin.ledger(conn)}


def cmd_gate(args) -> dict:
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    _, records, _, _ = _registry(args.corpus)
    registry = {r.fail_id: r for r in records}
    return gate_mod.zero_rewrite_gate(gate_mod.request_from_payload(payload), registry).to_dict()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tensorphysics.fail_revival")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("graph")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("acceptance")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts") / "fail_revival")
    p.add_argument("--sample", type=int, default=20)
    p.set_defaults(func=cmd_acceptance)

    p = sub.add_parser("query")
    p.add_argument("--db", type=Path, default=Path("artifacts") / "fail_revival" / "lineage.sqlite3")
    p.add_argument("--fail")
    p.add_argument("--task")
    p.add_argument("--criterion")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("gate")
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.set_defaults(func=cmd_gate)

    args = ap.parse_args(argv)
    try:
        out = args.func(args)
    except FileNotFoundError as exc:
        print(canonical_json({"error": "missing_input", "detail": str(exc)}))
        return 2
    print(canonical_json(out))
    if args.cmd == "acceptance":
        return 0 if out["summary"]["all_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())