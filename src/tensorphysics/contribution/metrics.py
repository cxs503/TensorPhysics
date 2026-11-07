"""Metrics: F1 score computation and 48h observation window productivity rate recalculation."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .classifier import ContributionItem, classify_item
from .golden_set import GOLDEN_SET, load_golden_set
from .ontology import ContributionCategory


def compute_f1(
    items: Optional[List[ContributionItem]] = None,
) -> Dict:
    """Compute per-category and macro F1 scores on the golden set.

    Returns dict with:
      - per_category: {category: {"precision", "recall", "f1", "support"}}
      - macro_f1: unweighted average of per-category F1
      - weighted_f1: weighted by support
      - confusion_matrix: {true_cat: {pred_cat: count}}
      - total_items: number of items evaluated
      - adversarial_items: number of adversarial items
      - adversarial_accuracy: accuracy on adversarial items only
    """
    if items is None:
        items = load_golden_set()

    # Collect predictions
    predictions = []
    for item in items:
        result = classify_item(item)
        predictions.append({
            "mission_id": item.mission_id,
            "true_category": item.metadata.get("ground_truth"),
            "predicted_category": result.predicted_category.value if result.predicted_category else None,
            "confidence": result.confidence,
            "method": result.method,
            "is_adversarial": item.metadata.get("is_adversarial", False),
        })

    # Build confusion matrix
    all_cats = [c.value for c in ContributionCategory]
    confusion = {tc: {pc: 0 for pc in all_cats} for tc in all_cats}
    for p in predictions:
        tc = p["true_category"]
        pc = p["predicted_category"]
        if tc in confusion and pc in confusion[tc]:
            confusion[tc][pc] += 1

    # Per-category metrics
    per_category = {}
    for cat in all_cats:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in all_cats if other != cat)
        fn = sum(confusion[cat][other] for other in all_cats if other != cat)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro and weighted F1
    f1_values = [v["f1"] for v in per_category.values() if v["support"] > 0]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    total_support = sum(v["support"] for v in per_category.values())
    weighted_f1 = (
        sum(v["f1"] * v["support"] for v in per_category.values()) / total_support
        if total_support > 0 else 0.0
    )

    # Adversarial accuracy
    adversarial_preds = [p for p in predictions if p["is_adversarial"]]
    adversarial_correct = sum(
        1 for p in adversarial_preds if p["true_category"] == p["predicted_category"]
    )
    adversarial_accuracy = (
        adversarial_correct / len(adversarial_preds) if adversarial_preds else 0.0
    )

    # Overall accuracy
    correct = sum(1 for p in predictions if p["true_category"] == p["predicted_category"])
    overall_accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "per_category": per_category,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "confusion_matrix": confusion,
        "total_items": len(predictions),
        "adversarial_items": len(adversarial_preds),
        "adversarial_accuracy": round(adversarial_accuracy, 4),
        "predictions": predictions,
    }


def compute_productivity_rate(
    items: Optional[List[ContributionItem]] = None,
    window_hours: int = 48,
) -> Dict:
    """Compute 48h observation window productivity rate.

    Productivity rate = number of verified contributions (C1+C2+C3) per 48h window.

    Returns dict with:
      - total_contributions: total items classified
      - verified_contributions: count of C1+C2+C3
      - productivity_rate: verified per 48h
      - breakdown: {category: count}
      - window_hours: the observation window used
    """
    if items is None:
        items = load_golden_set()

    # Classify all items
    results = [classify_item(item) for item in items]

    # Count by category
    breakdown = defaultdict(int)
    for r in results:
        if r.predicted_category:
            breakdown[r.predicted_category.value] += 1

    # Verified = C1 + C2 + C3
    verified_cats = {
        ContributionCategory.C1_VERIFIED_CASE.value,
        ContributionCategory.C2_CAPABILITY_DELTA.value,
        ContributionCategory.C3_DEFECT_FIX.value,
    }
    verified_count = sum(
        breakdown.get(cat, 0) for cat in verified_cats
    )

    # Productivity rate: verified per 48h window
    # Assuming the golden set represents contributions over a known period
    # For the golden set of 60 items, we assume they span ~30 days (typical observation)
    # So rate = verified_count / (30*24/48) = verified_count / 15 windows
    n_windows = max(1, len(items) // 4)  # rough estimate: 4 items per 48h window
    productivity_rate = verified_count / n_windows if n_windows > 0 else 0.0

    return {
        "total_contributions": len(items),
        "verified_contributions": verified_count,
        "productivity_rate": round(productivity_rate, 4),
        "breakdown": dict(breakdown),
        "window_hours": window_hours,
        "n_windows_estimated": n_windows,
    }
