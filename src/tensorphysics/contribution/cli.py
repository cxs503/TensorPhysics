"""CLI for contribution classification and reporting.

Usage:
    python -m tensorphysics.contribution classify --mission-id m103
    python -m tensorphysics.contribution classify --batch golden_set.json
    python -m tensorphysics.contribution report --output report.json
    python -m tensorphysics.contribution ontology
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .classifier import ContributionItem, classify_item
from .golden_set import GOLDEN_SET, load_golden_set
from .metrics import compute_f1, compute_productivity_rate
from .ontology import CONTRIBUTION_ONTOLOGY, ContributionCategory, list_categories


def cmd_classify(args: argparse.Namespace) -> int:
    """Handle 'classify' subcommand."""
    if args.mission_id:
        # Single item classification (demo mode with synthetic data)
        item = ContributionItem(
            mission_id=args.mission_id,
            title=args.title or "",
            summary=args.summary or "",
            tags=args.tags.split(",") if args.tags else [],
        )
        result = classify_item(item)
        print(json.dumps({
            "mission_id": result.mission_id,
            "predicted_category": result.predicted_category.value if result.predicted_category else None,
            "confidence": result.confidence,
            "method": result.method,
            "evidence_fields_present": result.evidence_fields_present,
            "evidence_fields_missing": result.evidence_fields_missing,
            "positive_signals": result.positive_signals,
            "counter_signals": result.counter_signals,
            "reasoning": result.reasoning,
        }, indent=2))
        return 0

    if args.batch:
        # Batch classification from JSON file
        with open(args.batch) as f:
            data = json.load(f)

        items = []
        for d in data:
            items.append(ContributionItem(
                mission_id=d.get("mission_id", ""),
                title=d.get("title", ""),
                summary=d.get("summary", ""),
                tags=d.get("tags", []),
                evidence=d.get("evidence", {}),
            ))

        results = []
        for item in items:
            r = classify_item(item)
            results.append({
                "mission_id": r.mission_id,
                "predicted_category": r.predicted_category.value if r.predicted_category else None,
                "confidence": r.confidence,
                "method": r.method,
            })

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Wrote {len(results)} classifications to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        return 0

    # Default: classify golden set and show summary
    items = load_golden_set()
    metrics = compute_f1(items)
    print(json.dumps({
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "overall_accuracy": metrics["overall_accuracy"],
        "total_items": metrics["total_items"],
        "per_category": metrics["per_category"],
    }, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle 'report' subcommand."""
    items = load_golden_set()
    f1_metrics = compute_f1(items)
    prod_metrics = compute_productivity_rate(items)

    report = {
        "title": "Scientific Contribution Measurement Unit — Evaluation Report",
        "f1_metrics": f1_metrics,
        "productivity_metrics": prod_metrics,
        "ontology_summary": list_categories(),
        "golden_set_size": len(GOLDEN_SET),
        "adversarial_count": sum(1 for d in GOLDEN_SET if d.get("is_adversarial")),
    }

    output_path = args.output or "contribution_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {output_path}")
    return 0


def cmd_ontology(args: argparse.Namespace) -> int:
    """Handle 'ontology' subcommand."""
    cats = list_categories()
    for cat in cats:
        print(f"\n{'='*60}")
        print(f"  {cat['name']}: {cat['label']}")
        print(f"{'='*60}")
        print(f"  Description: {cat['description']}")
        print(f"  Required evidence: {', '.join(cat['min_evidence'])}")
        print(f"  Threshold: {cat['evidence_threshold']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tensorphysics.contribution",
        description="Scientific Contribution Measurement Unit CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify contribution items")
    p_classify.add_argument("--mission-id", help="Single mission ID to classify")
    p_classify.add_argument("--title", help="Item title (with --mission-id)")
    p_classify.add_argument("--summary", help="Item summary (with --mission-id)")
    p_classify.add_argument("--tags", help="Comma-separated tags (with --mission-id)")
    p_classify.add_argument("--batch", help="JSON file with items to classify")
    p_classify.add_argument("--output", "-o", help="Output file path")
    p_classify.set_defaults(func=cmd_classify)

    # report
    p_report = subparsers.add_parser("report", help="Generate evaluation report")
    p_report.add_argument("--output", "-o", help="Output file path", default="contribution_report.json")
    p_report.set_defaults(func=cmd_report)

    # ontology
    p_ontology = subparsers.add_parser("ontology", help="Display ontology definitions")
    p_ontology.set_defaults(func=cmd_ontology)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
