"""CLI: ``python -m tensorphysics.capability_reuse build|hints|selfcheck``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import build_graph
from .audit import load_criteria, run_audit, sample_edges, write_json
from .extract import evidence_edges, extract_code_edges, map_capabilities
from .gold import load_gold, score_extraction
from .hints import build_hints
from .indexer import build_index, load_snapshot
from .model import canonical_hash, edge_rows

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "src/tensorphysics/capability_reuse/configs"


def _sample_review_path(out_dir: Path) -> Path:
    return out_dir / "edge_sample_review.json"


def _load_repo_config() -> dict:
    return json.loads((CFG / "repos.json").read_text(encoding="utf-8"))


def build(args) -> int:
    snap = load_snapshot(args.snapshot)
    repos = build_index(snap)
    repo_cfg = _load_repo_config()
    params = repo_cfg.get("extraction_params") or {}
    criteria_cfg = json.loads((CFG / "audit_criteria.json").read_text(encoding="utf-8"))
    strength = criteria_cfg.get("edge_strength") or {}
    graph, stats = build_graph(snap, repos, None, strength=strength, params=params)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    edge_payload = {
        "schema": stats["schema"],
        "snapshot_hash": snap["provenance"].get("snapshot_content_sha256"),
        "edges": edge_rows(graph.edges),
    }
    write_json(out / "edges.json", edge_payload)
    stats_out = {k: v for k, v in stats.items()}
    write_json(out / "graph_stats.json", stats_out)
    hints = build_hints(stats, top_k=int(args.top_k))
    hints["source"] = {"snapshot_hash": snap["provenance"].get("snapshot_content_sha256"),
                       "edges_sha256": canonical_hash(edge_payload["edges"]),
                       "graph_sha256": canonical_hash(stats["counts"])}
    write_json(out / "seeding_hints.json", hints)

    criteria = load_criteria(CFG / "audit_criteria.json")
    gold = load_gold(args.gold) if args.gold else None
    gold_score = score_extraction(gold, graph.edge_pairs()) if gold else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "gold_edges": 0, "positives": 0,
        "negatives": 0, "repos_covered": 0, "repos": [], "true_positives": 0,
        "false_negatives": 0, "false_positives_on_negatives": 0, "missing": [], "spurious": []}

    pairs = sorted(graph.edge_pairs())
    review_path = _sample_review_path(out)
    if review_path.exists():
        review = json.loads(review_path.read_text())
        sample_rows = review["rows"]
    else:
        sample = sample_edges(pairs, n=int(criteria["sample_size"]), salt=snap["provenance"].get(
            "snapshot_content_sha256", ""))
        by_pair = {}
        for e in graph.edges:
            by_pair.setdefault((e.src, e.dst), []).append(e)
        sample_rows = []
        for src, dst in sample:
            evs = sorted(by_pair[(src, dst)], key=lambda e: (e.file, e.line))
            sample_rows.append({"src": src, "dst": dst, "verdict": "unreviewed",
                                "evidence": [f"{e.file}:{e.line} [{e.kind}] {e.detail}" for e in evs[:2]]})
        review = {"schema": "capability-reuse-edge-review/1", "sample_size": len(sample_rows),
                  "review_file": str(review_path.relative_to(ROOT)) if review_path.is_relative_to(ROOT) else str(review_path),
                  "unsupported": 0, "rows": sample_rows}
        write_json(review_path, review)

    determinism = {"runs": 1, "identical": None, "note": "single build; run scripts/capability_reuse_audit.py for the two-run proof"}
    if args.determinism_probe:
        h1 = canonical_hash(edge_payload["edges"])
        repos2 = build_index(snap)
        graph2, _stats2 = build_graph(snap, repos2, None, strength=strength, params=params)
        h2 = canonical_hash(edge_rows(graph2.edges))
        determinism = {"runs": 2, "identical": h1 == h2, "edge_hash_run1": h1, "edge_hash_run2": h2}

    write_json(out / "determinism.json", determinism)
    report = run_audit(stats=stats, gold_score=gold_score, gold=gold or {},
                       criteria=criteria, determinism=determinism, sample_review=review,
                       hints=hints, snapshot_hash=snap["provenance"].get("snapshot_content_sha256", ""),
                       figures=[])
    write_json(out / "audit_report.json", report)
    print(json.dumps({"edges": stats["counts"]["edges"], "reuse_rate": stats["reuse"]["reuse_rate"],
                      "jaccard": stats["calibers"]["capability_level"]["jaccard"],
                      "gold": {"p": gold_score["precision"], "r": gold_score["recall"]},
                      "verdict": report["verdict"]["status"], "out": str(out)}, indent=1, ensure_ascii=False))
    return 0


def reaudit(args) -> int:
    """Re-assemble the audit from the persisted edge set (no re-extraction)."""
    out = Path(args.out_dir)
    payload = json.loads((out / "edges.json").read_text())
    stats = json.loads((out / "graph_stats.json").read_text())
    hints = json.loads((out / "seeding_hints.json").read_text())
    from .gold import load_gold, score_extraction
    pairs = {(e["src"], e["dst"]) for e in payload["edges"]}
    criteria_cfg = json.loads((CFG / "audit_criteria.json").read_text(encoding="utf-8"))
    criteria = criteria_cfg["criteria"]
    gold = load_gold(args.gold) if args.gold else None
    gold_score = score_extraction(gold, pairs) if gold else None
    review = json.loads((out / "edge_sample_review.json").read_text())
    det_path = out / "determinism.json"
    determinism = json.loads(det_path.read_text()) if det_path.exists() else {
        "runs": 1, "identical": None, "note": "no determinism probe recorded"}
    write_json(out / "determinism.json", determinism)
    report = run_audit(stats=stats, gold_score=gold_score, gold=gold or {}, criteria=criteria,
                       determinism=determinism, sample_review=review, hints=hints,
                       snapshot_hash=payload.get("snapshot_hash", ""), figures=[])
    write_json(out / "audit_report.json", report)
    print(json.dumps({"gold": {"p": gold_score["precision"], "r": gold_score["recall"]},
                      "jaccard": stats["calibers"]["capability_level"]["jaccard"],
                      "verdict": report["verdict"]}, indent=1, ensure_ascii=False)[:700])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tensorphysics.capability_reuse")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the reuse graph, hints and audit report")
    b.add_argument("--snapshot", default=str(ROOT / "artifacts/capability_reuse/snapshot.json"))
    b.add_argument("--out-dir", default=str(ROOT / "artifacts/capability_reuse"))
    b.add_argument("--gold", default=str(ROOT / "artifacts/capability_reuse/gold_standard.json"))
    b.add_argument("--top-k", type=int, default=25)
    b.add_argument("--determinism-probe", action="store_true", help="build twice and compare edge hashes")
    b.set_defaults(func=build)
    r = sub.add_parser("reaudit", help="re-assemble audit_report.json from the persisted edge set")
    r.add_argument("--out-dir", default=str(ROOT / "artifacts/capability_reuse"))
    r.add_argument("--gold", default=str(ROOT / "artifacts/capability_reuse/gold_standard.json"))
    r.set_defaults(func=reaudit)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())