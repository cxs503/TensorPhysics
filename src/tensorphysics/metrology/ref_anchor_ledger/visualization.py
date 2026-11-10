"""
Visualization for the reference anchor uncertainty ledger.

Generates:
1. Uncertainty bar chart for all anchors (grouped by domain)
2. Three-state judgment interval plot
3. Uncertainty propagation waterfall chart
"""

from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_anchor_uncertainties(
    anchors: list,
    title: str = "Reference Anchor Uncertainties (U_ref at k=2)",
    figsize: Tuple[int, int] = (14, 10),
    output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Plot uncertainty bar chart for all anchors.

    Parameters
    ----------
    anchors : list of RefAnchor
        Anchors to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    output_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    str or None
        Base64-encoded PNG if matplotlib available, else None
    """
    if not HAS_MPL:
        return None

    # Group by domain
    domains = sorted(set(a.domain for a in anchors))
    colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
    domain_colors = {d: colors[i] for i, d in enumerate(domains)}

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by relative uncertainty
    sorted_anchors = sorted(anchors, key=lambda a: a.relative_uncertainty, reverse=True)

    y_pos = np.arange(len(sorted_anchors))
    values = [a.value for a in sorted_anchors]
    U_refs = [a.U_ref for a in sorted_anchors]
    labels = [f"{a.anchor_id}: {a.name[:40]}" for a in sorted_anchors]
    bar_colors = [domain_colors[a.domain] for a in sorted_anchors]

    bars = ax.barh(y_pos, values, xerr=U_refs, color=bar_colors, alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Value ± U_ref (k=2)")
    ax.set_title(title)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Legend
    patches = [mpatches.Patch(color=domain_colors[d], label=d) for d in domains]
    ax.legend(handles=patches, loc='lower right', fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_judgment_intervals(
    judgments: list,
    title: str = "Three-State Judgment Intervals",
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Plot judgment intervals showing Δ vs thresholds.

    Parameters
    ----------
    judgments : list of JudgmentResult
        Judgment results to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    output_path : Path, optional
        Save path

    Returns
    -------
    str or None
        Base64-encoded PNG
    """
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    n = len(judgments)
    y_pos = np.arange(n)

    # Color by judgment state
    color_map = {
        "robust": "#2ecc71",      # green
        "critical": "#f39c12",    # orange
        "indeterminate": "#e74c3c",  # red
    }

    for i, j in enumerate(judgments):
        color = color_map[j.judgment.value]

        # Plot threshold zones
        ax.barh(i, j.threshold_robust, color=color, alpha=0.3, label='_nolegend_')
        ax.barh(i, j.threshold_critical - j.threshold_robust,
                left=j.threshold_robust, color=color, alpha=0.15, label='_nolegend_')

        # Plot actual delta
        ax.plot(j.delta, i, 'ko', markersize=8, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Case {i+1}" for i in range(n)], fontsize=9)
    ax.set_xlabel("|Δ| (absolute difference)")
    ax.set_title(title)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_map["robust"], alpha=0.5, label="ROBUST zone"),
        mpatches.Patch(facecolor=color_map["critical"], alpha=0.3, label="CRITICAL zone"),
        mpatches.Patch(facecolor=color_map["indeterminate"], alpha=0.3, label="INDETERMINATE"),
        plt.Line2D([0], [0], marker='o', color='k', linestyle='None',
                   markersize=8, label="|Δ| actual"),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_propagation_waterfall(
    contributions: Dict[str, float],
    result_nominal: float,
    U_result: float,
    title: str = "Uncertainty Propagation Waterfall",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Plot uncertainty propagation waterfall chart.

    Shows how each anchor contributes to the total result uncertainty.

    Parameters
    ----------
    contributions : dict
        {anchor_id: variance_contribution}
    result_nominal : float
        Nominal result value
    U_result : float
        Total expanded uncertainty
    title : str
        Plot title
    figsize : tuple
        Figure size
    output_path : Path, optional
        Save path

    Returns
    -------
    str or None
        Base64-encoded PNG
    """
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by contribution
    sorted_items = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    total_var = sum(values)

    # Cumulative bar chart
    cumulative = np.cumsum(values)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Variance Contribution")
    ax.set_title(f"{title}\nTotal U_result = {U_result:.4g} (k=2)")

    # Add cumulative line
    ax2 = ax.twinx()
    ax2.plot(range(len(labels)), cumulative / total_var * 100,
             'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 110)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_all_figures(
    ledger,
    judgments: list,
    contributions: Dict[str, float],
    result_nominal: float,
    U_result: float,
    output_dir: Path,
) -> Dict[str, str]:
    """
    Generate all visualization figures.

    Returns
    -------
    dict
        {figure_name: base64_png}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # 1. Anchor uncertainties
    fig1 = plot_anchor_uncertainties(
        ledger.anchors,
        output_path=output_dir / "anchor_uncertainties.png"
    )
    if fig1:
        figures["anchor_uncertainties"] = fig1

    # 2. Judgment intervals
    fig2 = plot_judgment_intervals(
        judgments,
        output_path=output_dir / "judgment_intervals.png"
    )
    if fig2:
        figures["judgment_intervals"] = fig2

    # 3. Propagation waterfall
    fig3 = plot_propagation_waterfall(
        contributions, result_nominal, U_result,
        output_path=output_dir / "propagation_waterfall.png"
    )
    if fig3:
        figures["propagation_waterfall"] = fig3

    return figures
