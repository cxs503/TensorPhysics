"""Generate figures for the Reference-Truth Certificate system documentation."""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def generate_architecture_figure(output_path: str):
    """Generate architecture diagram of the certificate system."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(6, 7.5, "Reference-Truth Certificate System Architecture",
            ha="center", va="top", fontsize=14, fontweight="bold")

    # Components
    components = {
        "schema": (1, 5.5, 2.5, 1.2, "Schema\n(schema.py)", "#4CAF50"),
        "validator": (4.5, 5.5, 2.5, 1.2, "Validator\n(validator.py)", "#2196F3"),
        "store": (8, 5.5, 2.5, 1.2, "Store\n(store.py)", "#FF9800"),
        "gate": (4.5, 3.5, 2.5, 1.2, "Reference Gate\n(gate.py)", "#9C27B0"),
        "backfill": (1, 3.5, 2.5, 1.2, "Backfill\n(backfill.py)", "#795548"),
        "certificates": (8, 3.5, 2.5, 1.2, "Certificates\n(JSON)", "#607D8B"),
    }

    for key, (x, y, w, h, label, color) in components.items():
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=10, fontweight="bold")

    # Arrows
    arrow_style = dict(arrowstyle="->", color="#333333", lw=1.5)

    # Schema → Validator
    ax.annotate("", xy=(4.5, 6.1), xytext=(3.5, 6.1), arrowprops=arrow_style)
    ax.text(4.0, 6.3, "validate", fontsize=8, ha="center")

    # Validator → Store
    ax.annotate("", xy=(8, 6.1), xytext=(7, 6.1), arrowprops=arrow_style)
    ax.text(7.5, 6.3, "add", fontsize=8, ha="center")

    # Store → Gate
    ax.annotate("", xy=(5.75, 4.7), xytext=(9.25, 5.5), arrowprops=arrow_style)
    ax.text(7.8, 5.2, "query", fontsize=8, ha="center")

    # Gate → Decision
    ax.annotate("", xy=(5.75, 2.5), xytext=(5.75, 3.5), arrowprops=arrow_style)
    ax.text(6.5, 3.0, "ALLOW/BLOCK", fontsize=8, ha="center")

    # Backfill → Store
    ax.annotate("", xy=(8, 4.1), xytext=(3.5, 4.1), arrowprops=arrow_style)
    ax.text(5.75, 4.3, "populate", fontsize=8, ha="center")

    # Store → Certificates (persistence)
    ax.annotate("", xy=(9.25, 4.7), xytext=(9.25, 5.5),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    ax.text(9.8, 5.1, "save/load", fontsize=8, ha="center")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", alpha=0.3, label="Schema: data model"),
        mpatches.Patch(facecolor="#2196F3", alpha=0.3, label="Validator: type-specific rules"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.3, label="Store: persistence"),
        mpatches.Patch(facecolor="#9C27B0", alpha=0.3, label="Gate: enforcement"),
        mpatches.Patch(facecolor="#795548", alpha=0.3, label="Backfill: gold-standard"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

    # Key principle
    ax.text(6, 1.5, "Key Principle: Un-certified references cannot be used as acceptance criteria",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", edgecolor="#F57F17", linewidth=2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved architecture figure to {output_path}")


def generate_certificate_statistics_figure(output_path: str, cert_file: str):
    """Generate statistics figure from the certificate store."""
    with open(cert_file, "r") as f:
        data = json.load(f)

    certs = data["certificates"]

    # Count by type
    type_counts = {}
    status_counts = {}
    for cert in certs:
        t = cert["certificate_type"]
        s = cert["status"]
        type_counts[t] = type_counts.get(t, 0) + 1
        status_counts[s] = status_counts.get(s, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart: certificate types
    ax1 = axes[0]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    ax1.pie(sizes, labels=labels, colors=colors[:len(labels)],
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax1.set_title("Certificate Types Distribution", fontsize=12, fontweight="bold")

    # Bar chart: status
    ax2 = axes[1]
    status_colors = {"valid": "#4CAF50", "draft": "#9E9E9E", "insufficient": "#F44336",
                     "expired": "#FF9800", "revoked": "#9C27B0"}
    bars = ax2.bar(status_counts.keys(), status_counts.values(),
                   color=[status_colors.get(s, "#607D8B") for s in status_counts.keys()])
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Certificate Status Distribution", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(status_counts.values()) * 1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # Overall stats
    total = len(certs)
    valid_rate = status_counts.get("valid", 0) / total * 100
    fig.suptitle(f"Reference-Truth Certificate Statistics (N={total}, Valid Rate={valid_rate:.0f}%)",
                fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved statistics figure to {output_path}")


def generate_validation_flow_figure(output_path: str):
    """Generate validation flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(7, 5.5, "Certificate Validation Flow by Type",
            ha="center", va="top", fontsize=13, fontweight="bold")

    # Four validation paths
    paths = [
        {
            "type": "ANALYTICAL",
            "color": "#4CAF50",
            "x": 0.5,
            "checks": [
                "Has exact_formula\nor symbolic_equivalence?",
                "symbolic_equivalence\n== True?",
                "reference_value\nis finite?",
            ],
            "result": "VALID",
        },
        {
            "type": "SELF_CONVERGENCE",
            "color": "#2196F3",
            "x": 3.8,
            "checks": [
                "≥3 refinement\nlevels?",
                "observed_order_p\n≥ 0.5?",
                "GCI computed\nfor finest pair?",
            ],
            "result": "VALID",
        },
        {
            "type": "EXPERIMENTAL",
            "color": "#FF9800",
            "x": 7.1,
            "checks": [
                "source_reference\nprovided?",
                "stated_uncertainty\n> 0?",
                "confidence_level\n≥ 0.90?",
            ],
            "result": "VALID",
        },
        {
            "type": "STATISTICAL",
            "color": "#9C27B0",
            "x": 10.4,
            "checks": [
                "n_samples ≥ 30?",
                "CI brackets\nthe mean?",
                "confidence_level\n≥ 0.90?",
            ],
            "result": "VALID",
        },
    ]

    for path in paths:
        x = path["x"]
        color = path["color"]

        # Type header
        rect = mpatches.FancyBboxPatch((x, 4.2), 2.8, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.4, 4.6, path["type"], ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

        # Check boxes
        for i, check in enumerate(path["checks"]):
            y = 3.2 - i * 1.0
            rect = mpatches.FancyBboxPatch((x + 0.2, y), 2.4, 0.7,
                                            boxstyle="round,pad=0.05",
                                            facecolor="white",
                                            edgecolor="#666666", linewidth=1)
            ax.add_patch(rect)
            ax.text(x + 1.4, y + 0.35, check, ha="center", va="center", fontsize=8)

            # Arrow to next
            if i < len(path["checks"]) - 1:
                ax.annotate("", xy=(x + 1.4, y - 0.05),
                           xytext=(x + 1.4, y - 0.25),
                           arrowprops=dict(arrowstyle="->", color="#666666"))

        # Result
        y_result = 3.2 - len(path["checks"]) * 1.0 + 0.2
        ax.annotate("", xy=(x + 1.4, y_result),
                   xytext=(x + 1.4, y_result + 0.2),
                   arrowprops=dict(arrowstyle="->", color="#666666"))

        rect = mpatches.FancyBboxPatch((x + 0.4, y_result - 0.6), 2.0, 0.5,
                                        boxstyle="round,pad=0.05",
                                        facecolor="#E8F5E9",
                                        edgecolor="#4CAF50", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.4, y_result - 0.35, "✓ VALID", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#4CAF50")

    # Failure path
    ax.text(7, 0.3, "Any check fails → status = INSUFFICIENT → Gate BLOCKS citation",
            ha="center", va="center", fontsize=10, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#F44336"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved validation flow figure to {output_path}")


if __name__ == "__main__":
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    generate_architecture_figure(str(output_dir / "certificate_architecture.png"))
    generate_certificate_statistics_figure(
        str(output_dir / "certificate_statistics.png"),
        str(output_dir / "reference_truth_certificates.json")
    )
    generate_validation_flow_figure(str(output_dir / "certificate_validation_flow.png"))

    print("All figures generated successfully.")
