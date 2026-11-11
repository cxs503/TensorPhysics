"""Machine-checkable acceptance criteria of the capability-reuse audit."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import AUDIT_SCHEMA, canonical_hash, canonical_json


def load_criteria(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return cfg["criteria"]


def write_json(path: str | Path, obj) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1)
    p.write_text(text + "\n", encoding="utf-8")
    return str(p)


def sample_edges(pairs: Sequence[Tuple[str, str]], n: int = 20, salt: str = "") -> List[Tuple[str, str]]:
    """Deterministic pseudo-random sample: order by sha256(salt|src|dst)."""
    def key(p):
        return hashlib.sha256(f"{salt}|{p[0]}|{p[1]}".encode()).hexdigest()
    return sorted(pairs, key=key)[:n]


def run_audit(*, stats: Dict[str, object], gold_score: Dict[str, object], gold: dict,
              criteria: Dict[str, object], determinism: Dict[str, object],
              sample_review: Optional[dict], hints: dict, snapshot_hash: str,
              figures: Optional[List[str]] = None) -> Dict[str, object]:
    counts = stats["counts"]                     # type: ignore[index]
    per_repo = stats["per_repo"]                 # type: ignore[index]
    calibers = stats["calibers"]                 # type: ignore[index]
    usage = stats["reuse"]                       # type: ignore[index]
    cap_repo = stats.get("cap_repo") or {}
    cap_class = {c: (row or {}).get("cap_class") or "unclassified"
                 for c, row in (stats.get("cap_class") or {}).items()}
    classes_per_repo: Dict[str, set] = {}
    for cid, repo in cap_repo.items():
        classes_per_repo.setdefault(repo, set()).add(cap_class.get(cid) or "unclassified")
    classes_per_repo = {k: v for k, v in classes_per_repo.items()}

    # (a) gold standard -------------------------------------------------------
    a_pass = (gold_score["precision"] >= criteria["gold_precision_min"]
              and gold_score["recall"] >= criteria["gold_recall_min"]
              and gold_score["gold_edges"] >= criteria["gold_min_edges"]
              and gold_score["repos_covered"] >= criteria["gold_min_repos"]
              and gold_score["negatives"] >= criteria["gold_min_negatives"])

    # (b) dual caliber --------------------------------------------------------
    jac = calibers["capability_level"]["jaccard"]
    b_pass = jac >= criteria["dual_caliber_jaccard_min"]

    # (c) determinism ---------------------------------------------------------
    c_pass = bool(determinism.get("identical")) and determinism.get("runs", 0) >= 2

    # (d) coverage ------------------------------------------------------------
    repos_covered = sorted(r for r, row in per_repo.items() if row.get("capabilities"))
    repos_ok = [r for r in repos_covered if len(classes_per_repo.get(r, ())) >= criteria["min_classes_per_repo"]]
    d_pass = len(repos_covered) >= criteria["min_repos"] and len(repos_ok) >= criteria["min_repos"]
    d_separation = usage["orphan_capabilities"] + usage["reused_capabilities"] == counts["determined"]

    # (e) sampled edge review -------------------------------------------------
    if sample_review:
        n = int(sample_review["sample_size"])
        unsupported = int(sample_review["unsupported"])
        fp_rate = unsupported / n if n else 0.0
        e_pass = fp_rate <= criteria["sample_false_positive_max"]
    else:  # not run -> unknown, never "passed"
        n, unsupported, fp_rate, e_pass = 0, 0, None, None

    # (f) hints + tests (pytest verdict is supplied by the driver) -----------
    f_pass = hints.get("schema") is not None and "orphans" in hints and "hubs" in hints

    report = {
        "schema": AUDIT_SCHEMA,
        "snapshot_hash": snapshot_hash,
        "criteria": criteria,
        "a_gold_standard": {
            "pass": a_pass, "precision": gold_score["precision"], "recall": gold_score["recall"],
            "f1": gold_score["f1"], "gold_edges": gold_score["gold_edges"],
            "positives": gold_score["positives"], "negatives": gold_score["negatives"],
            "repos_covered": gold_score["repos_covered"], "repos": gold_score["repos"],
            "false_negatives": gold_score["false_negatives"],
            "false_positives_on_negatives": gold_score["false_positives_on_negatives"],
            "missing": gold_score["missing"], "spurious": gold_score["spurious"],
            "gold_authorship": gold.get("authorship", ""),
        },
        "b_dual_caliber": {
            "pass": b_pass, "jaccard": jac,
            "ast_edges": calibers["capability_level"]["ast_edges"],
            "ref_edges": calibers["capability_level"]["ref_edges"],
            "intersection": calibers["capability_level"]["intersection"],
            "union": calibers["capability_level"]["union"],
            "ast_only": calibers["capability_level"]["ast_only"],
            "ref_only": calibers["capability_level"]["ref_only"],
            "disagreements": len(calibers["capability_level"]["ast_only"]) + len(calibers["capability_level"]["ref_only"]),
        },
        "c_determinism": dict(determinism, **{"pass": c_pass}),
        "d_coverage": {
            "pass": bool(d_pass and d_separation),
            "repos_covered": len(repos_covered), "repos": repos_covered,
            "repos_with_min_classes": sorted(repos_ok),
            "min_classes_per_repo": criteria["min_classes_per_repo"],
            "classes_per_repo": {k: sorted(v) for k, v in sorted(classes_per_repo.items())},
            "bucket_separation": {
                "determined": counts["determined"], "undetermined": counts["undetermined"],
                "orphan_denominator_is_determined_only": d_separation,
            },
        },
        "e_edge_sample": {
            "pass": e_pass, "sample_size": n, "unsupported": unsupported,
            "false_positive_rate": fp_rate,
            "threshold": criteria["sample_false_positive_max"],
            "review_file": (sample_review or {}).get("review_file"),
        },
        "f_seeding_hints": {"pass": f_pass, "orphans": hints["summary"]["orphans"],
                            "hubs": len(hints["hubs"]), "gaps": len(hints["gaps"])},
        "g_report": {"figures": figures or []},
        "reuse_metrics": usage,
        "graph_counts": counts,
    }
    report["verdict"] = {
        "criteria_passed": [k for k in ("a", "b", "c", "d", "e", "f")
                            if report[f"{k}_gold_standard" if k == "a" else
                                        f"{k}_dual_caliber" if k == "b" else
                                        f"{k}_determinism" if k == "c" else
                                        f"{k}_coverage" if k == "d" else
                                        f"{k}_edge_sample" if k == "e" else "f_seeding_hints"]["pass"]],
        "unknown": [k for k in ("e",) if report["e_edge_sample"]["pass"] is None],
    }
    failed = [k for k in ("a_gold_standard", "b_dual_caliber", "c_determinism", "d_coverage",
                          "e_edge_sample", "f_seeding_hints") if report[k]["pass"] is False]
    report["verdict"]["failed"] = failed
    report["verdict"]["status"] = "PASS" if not failed else "FAIL"
    report["content_sha256"] = canonical_hash({k: v for k, v in report.items() if k != "content_sha256"})
    return report