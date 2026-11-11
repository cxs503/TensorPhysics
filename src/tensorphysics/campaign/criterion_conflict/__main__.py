"""CLI: build a criterion registry, list conflicts, or replay arbitration.

Examples::

    python -m tensorphysics.campaign.criterion_conflict build \
        --corpus corpus.json --config metrics.json --out registry.json
    python -m tensorphysics.campaign.criterion_conflict arbitrate \
        --registry registry.json --config metrics.json --decisions decisions.json \
        --ledger arbitration.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arbitration import ArbitrationLedger
from .conflict import Conflict
from .model import Criterion
from .registry import CorpusEntry, CriterionRegistry


def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cmd_build(args: argparse.Namespace) -> int:
    corpus = _load_json(args.corpus)
    entries = corpus.get("entries", corpus) if isinstance(corpus, dict) else corpus
    registry = CriterionRegistry.build(entries, args.config)
    payload = registry.to_dict()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "registry_hash": payload["registry_hash"],
        "criteria": payload["criterion_count"],
        "mapped": payload["mapped_count"],
        "conflicts": len(payload["conflicts"]),
        "out": args.out,
    }, ensure_ascii=False))
    return 0


def _cmd_arbitrate(args: argparse.Namespace) -> int:
    registry = CriterionRegistry.from_json(args.registry, args.config)
    conflicts = registry.conflicts()
    by_id = {c.conflict_id: c for c in conflicts}
    decisions = _load_json(args.decisions)
    ledger = ArbitrationLedger(known_conflicts=by_id.keys(), path=args.ledger)
    for item in decisions:
        cid = item["conflict_id"]
        if cid not in by_id:
            raise SystemExit(f"unknown conflict_id {cid}")
        ledger.arbitrate(by_id[cid], item["decision"], item["arbiter"],
                         item["rationale"], item["timestamp"],
                         item.get("chosen_threshold"), item.get("chosen_unit"))
    print(json.dumps({
        "records": len(ledger.records),
        "chain_valid": ledger.verify_chain(),
        "replay_digest": ledger.replay_digest(),
        "state": ledger.state(),
    }, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="criterion-conflict")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="extract criteria and detect conflicts")
    build.add_argument("--corpus", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--out", required=True)
    build.set_defaults(func=_cmd_build)

    arb = sub.add_parser("arbitrate", help="apply arbitration decisions")
    arb.add_argument("--registry", required=True)
    arb.add_argument("--config", required=True)
    arb.add_argument("--decisions", required=True)
    arb.add_argument("--ledger", required=True)
    arb.set_defaults(func=_cmd_arbitrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
