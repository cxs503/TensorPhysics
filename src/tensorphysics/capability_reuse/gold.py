"""Gold-standard scoring: human-labelled capability-level reuse edges."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .model import Edge

GOLD_SCHEMA = "capability-reuse-gold/1"


def load_gold(path: str | Path) -> dict:
    gold = json.loads(Path(path).read_text(encoding="utf-8"))
    if gold.get("schema") != GOLD_SCHEMA:
        raise ValueError(f"unexpected gold schema {gold.get('schema')!r}, want {GOLD_SCHEMA!r}")
    for row in gold["edges"]:
        if row["label"] not in ("positive", "negative"):
            raise ValueError(f"bad label {row['label']!r} in gold edge {row}")
        if row["label"] == "positive" and not row.get("evidence"):
            raise ValueError(f"positive gold edge without evidence: {row}")
    return gold


def gold_pairs(gold: dict) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    pos = {(e["src"], e["dst"]) for e in gold["edges"] if e["label"] == "positive"}
    neg = {(e["src"], e["dst"]) for e in gold["edges"] if e["label"] == "negative"}
    return pos, neg


def score_extraction(gold: dict, predicted: Iterable[Tuple[str, str]]) -> Dict[str, object]:
    pred = set(predicted)
    pos, neg = gold_pairs(gold)
    tp = sorted(pos & pred)
    fn = sorted(pos - pred)
    fp_on_neg = sorted(neg & pred)
    fp = fp_on_neg                      # only the labelled negatives count as errors
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    repos = sorted({e.get("repo", "") for e in gold["edges"]} - {""})
    return {
        "gold_edges": len(gold["edges"]),
        "positives": len(pos),
        "negatives": len(neg),
        "repos_covered": len(repos),
        "repos": repos,
        "true_positives": len(tp),
        "false_negatives": len(fn),
        "false_positives_on_negatives": len(fp_on_neg),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "missing": [list(x) for x in fn],
        "spurious": [list(x) for x in fp_on_neg],
        "tp_pairs": [list(x) for x in tp],
    }


def sampled_edge_support(edges: Sequence[Edge], sample: Sequence[Tuple[str, str]],
                         limit: int = 20) -> Dict[str, object]:
    """For a sample of (src, dst) capability pairs report the best file:line evidence."""
    by_pair: Dict[Tuple[str, str], List[Edge]] = {}
    for e in edges:
        by_pair.setdefault((e.src, e.dst), []).append(e)
    rows = []
    for src, dst in sample:
        evs = sorted(by_pair.get((src, dst), []), key=lambda e: (e.file, e.line))
        rows.append({"src": src, "dst": dst,
                     "supported": bool(evs),
                     "best_evidence": [f"{e.file}:{e.line} [{e.kind}] {e.detail}" for e in evs[:3]]})
    return {"sample_size": len(rows), "unsupported": sum(1 for r in rows if not r["supported"]),
            "rows": rows}