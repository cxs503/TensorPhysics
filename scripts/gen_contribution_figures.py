"""Generate figures for the contribution measurement report."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tensorphysics.contribution.metrics import compute_f1, compute_productivity_rate
from tensorphysics.contribution.ontology import ContributionCategory


def generate_confusion_matrix(output_path: str):
    """Generate confusion matrix heatmap."""
    metrics = compute_f1()
    cm = metrics["confusion_matrix"]
    cats = [c.value for c in ContributionCategory]
    cat_labels = [c.replace("C", "C").replace("_", "\n") for c in cats]

    matrix = np.zeros((6, 6), dtype=int)
    for i, tc in enumerate(cats):
        for j, pc in enumerate(cats):
            matrix[i][j] = cm[tc][pc]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(range(6))
    ax.set_xticklabels(cat_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(cat_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Contribution Classifier (Golden Set, N=60)", fontsize=14)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            val = matrix[i][j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white" if val > 5 else "black")

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def generate_f1_bar_chart(output_path: str):
    """Generate per-category F1 bar chart."""
    metrics = compute_f1()
    cats = [c.value for c in ContributionCategory]
    cat_labels = [c.replace("_", " ").title() for c in cats]

    f1_vals = [metrics["per_category"][c]["f1"] for c in cats]
    precision_vals = [metrics["per_category"][c]["precision"] for c in cats]
    recall_vals = [metrics["per_category"][c]["recall"] for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision_vals, width, label="Precision", color="#2196F3")
    bars2 = ax.bar(x, recall_vals, width, label="Recall", color="#4CAF50")
    bars3 = ax.bar(x + width, f1_vals, width, label="F1", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Category Precision, Recall, and F1 (Golden Set, N=60)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved F1 bar chart to {output_path}")


def generate_productivity_chart(output_path: str):
    """Generate productivity rate breakdown chart."""
    prod = compute_productivity_rate()
    breakdown = prod["breakdown"]

    cats = [c.value for c in ContributionCategory]
    cat_labels = [c.replace("_", " ").title() for c in cats]
    counts = [breakdown.get(c, 0) for c in cats]

    # Color by verified vs non-verified
    verified_cats = {"C1_verified_case", "C2_capability_delta", "C3_defect_fix"}
    colors = ["#4CAF50" if c in verified_cats else "#FF9800" for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cat_labels, counts, color=colors, edgecolor="white", linewidth=1.5)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Contribution Breakdown (N={prod['total_contributions']}, "
                 f"Verified={prod['verified_contributions']}, "
                 f"Rate={prod['productivity_rate']:.2f}/48h)", fontsize=14)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Verified (C1+C2+C3)"),
        Patch(facecolor="#FF9800", label="Other (C4+C5+C6)"),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved productivity chart to {output_path}")


if __name__ == "__main__":
    output_dir = Path("/root/tp-agent/work/m5252/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_confusion_matrix(str(output_dir / "confusion_matrix.png"))
    generate_f1_bar_chart(str(output_dir / "f1_bar_chart.png"))
    generate_productivity_chart(str(output_dir / "productivity_breakdown.png"))
