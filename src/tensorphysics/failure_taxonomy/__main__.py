"""Back-test driver for the failure taxonomy (criterion harness).

Runs, on the pinned frozen corpus:

* annotator A (structured signal rules) vs the analyst's gold labels  -> C2
* annotator B (unsupervised seed prototypes) vs gold                  -> robustness
* kappa between the two annotators (C1 primary pair: analyst vs A; second pair:
  A vs B)                                                             -> C1
* negative control 1: bodies replaced by same-length non-cue decoys    -> C3a
* negative control 2: gold labels permuted, train/test centroid arm    -> C3b
* decision-table schema coverage                                       -> C4

Usage::

    PYTHONPATH=src python -m tensorphysics.failure_taxonomy --out /tmp/failure_taxonomy_backtest.json
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from .agreement import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    per_class_recall,
    permute_labels,
    replace_bodies,
)
from .dataset import (
    load_cue_lexicon,
    load_decoys,
    load_frozen_set,
    load_manifest,
    load_rule_weights,
    load_seeds,
    stratum_split,
)
from .decide import table_coverage
from .labelers import SeedPrototypeLabeler, SignalRuleLabeler, SupervisedCentroidLabeler
from .taxonomy import CLASS_CODES, RANDOM_BASELINE


N_PERMUTATIONS = 25
N_DECOY_DRAWS = 10


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _labels(labeler, bodies) -> list[str]:
    return [labeler.label(b).code for b in bodies]


def run_backtest(seed: int = 20261110) -> dict[str, Any]:
    records = load_frozen_set()
    manifest = load_manifest()
    bodies = [r.body for r in records]
    gold = [r.gold_class for r in records]

    annotator_a = SignalRuleLabeler(load_rule_weights(), load_cue_lexicon())
    annotator_b = SeedPrototypeLabeler(load_seeds())

    pred_a = _labels(annotator_a, bodies)
    pred_b = _labels(annotator_b, bodies)

    train, test = stratum_split(records, mod=3, remainder=2)
    supervised = SupervisedCentroidLabeler().fit((r.body, r.gold_class) for r in train)
    pred_train_arm = [supervised.label(r.body).code for r in test]
    gold_test = [r.gold_class for r in test]

    # Negative control C3b: repeat the permutation so the reported number is a
    # mean over draws, not a single lucky shuffle (n_test is small).
    perm_acc: list[float] = []
    perm_kappa: list[float] = []
    for k in range(N_PERMUTATIONS):
        permuted = permute_labels([r.gold_class for r in train], seed=seed + k)
        shuffled = SupervisedCentroidLabeler().fit(
            (r.body, code) for r, code in zip(train, permuted)
        )
        pred_shuffled = [shuffled.label(r.body).code for r in test]
        perm_acc.append(accuracy(pred_shuffled, gold_test))
        perm_kappa.append(cohen_kappa([shuffled.label(r.body).code for r in test], gold_test))

    # Negative control C3a: same averaging over decoy draws.
    decoy_acc: list[float] = []
    decoys = load_decoys()
    for k in range(N_DECOY_DRAWS):
        decoy_bodies = replace_bodies(bodies, decoys, seed=seed + k)
        decoy_acc.append(accuracy(_labels(annotator_a, decoy_bodies), gold))

    metrics: dict[str, Any] = {
        "corpus": {
            "n_records": len(records),
            "n_decoys": len(load_decoys()),
            "per_class": manifest["per_class"],
            "sha256": manifest["sha256"],
            "annotator": manifest["annotator"],
            "protocol": manifest["protocol"],
        },
        "C1_dual_annotation": {
            "kappa_analyst_vs_structured": cohen_kappa(gold, pred_a),
            "agreement_analyst_vs_structured": accuracy(pred_a, gold),
            "kappa_structured_vs_prototype": cohen_kappa(pred_a, pred_b),
            "agreement_structured_vs_prototype": accuracy(pred_b, pred_a),
            "kappa_analyst_vs_prototype": cohen_kappa(gold, pred_b),
        },
        "C2_frozen_set_consistency": {
            "accuracy_structured": accuracy(pred_a, gold),
            "accuracy_prototype": accuracy(pred_b, gold),
            "macro_f1_structured": macro_f1(gold, pred_a),
            "per_class_recall_structured": per_class_recall(gold, pred_a),
            "confusion_structured": confusion_matrix(gold, pred_a),
            "random_baseline": RANDOM_BASELINE,
        },
        "C3a_decoy_negative_control": {
            "accuracy_on_decoys": _mean(decoy_acc),
            "accuracy_max_draw": max(decoy_acc),
            "n_draws": len(decoy_acc),
            "n": len(bodies),
        },
        "C3b_label_permutation_control": {
            "n_train": len(train),
            "n_test": len(test),
            "n_permutations": len(perm_acc),
            "test_accuracy_true_labels": accuracy(pred_train_arm, gold_test),
            "test_accuracy_permuted_labels": _mean(perm_acc),
            "test_accuracy_permuted_max_draw": max(perm_acc),
            "test_kappa_permuted": _mean(perm_kappa),
            "chance_level": RANDOM_BASELINE,
        },
        "C4_decision_table": table_coverage(),
        "predictions": [
            {"record_id": r.record_id, "gold": r.gold_class, "structured": a, "prototype": b}
            for r, a, b in zip(records, pred_a, pred_b)
        ],
    }

    thresholds = {
        "C1_kappa_min": 0.6,
        "C2_accuracy_min": 0.70,
        "C3a_accuracy_max": 0.35,
        "C3b_accuracy_max": 0.30,
    }
    verdicts = {
        "C1": metrics["C1_dual_annotation"]["kappa_analyst_vs_structured"] >= thresholds["C1_kappa_min"],
        "C2": metrics["C2_frozen_set_consistency"]["accuracy_structured"] >= thresholds["C2_accuracy_min"],
        "C3a": metrics["C3a_decoy_negative_control"]["accuracy_on_decoys"] <= thresholds["C3a_accuracy_max"],
        "C3b": metrics["C3b_label_permutation_control"]["test_accuracy_permuted_labels"]
        <= thresholds["C3b_accuracy_max"],
        "C4": all(v >= 1 for v in metrics["C4_decision_table"]["trigger_counts"].values()),
    }
    metrics["thresholds"] = thresholds
    metrics["verdicts"] = verdicts
    metrics["all_pass"] = all(verdicts.values())
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args(argv)
    metrics = run_backtest(seed=args.seed)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    slim = {
        "C1": metrics["C1_dual_annotation"],
        "C2": {k: v for k, v in metrics["C2_frozen_set_consistency"].items() if k != "confusion_structured"},
        "C3a": metrics["C3a_decoy_negative_control"],
        "C3b": metrics["C3b_label_permutation_control"],
        "verdicts": metrics["verdicts"],
        "all_pass": metrics["all_pass"],
    }
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    return 0 if metrics["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())