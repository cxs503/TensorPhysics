"""Figures for the multi-fidelity selection verification (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: the artifacts are PNGs, never interactive windows
import matplotlib.pyplot as plt  # noqa: E402

VARIANT_STYLE = {
    "prior_only": ("#b03030", "registered priors only"),
    "conformal_pooled": ("#d08a1e", "conformal, pooled residuals"),
    "conformal_stratified": ("#2b7a3d", "conformal, stratified residuals"),
    "conformal_mispartitioned": ("#7a5aa8", "conformal, wrong partition (probe)"),
}
MODEL_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def _variant_style(name: str):
    return VARIANT_STYLE.get(name, ("#444444", name))


def plot_recommendation_histogram(metrics_by_variant: Mapping[str, Mapping[str, Any]],
                                  truth_label: str, out_path: Path) -> Path:
    """Stacked histogram: which fidelity level does each configuration recommend?"""
    variants = [v for v in metrics_by_variant if v in VARIANT_STYLE or True]
    models: List[str] = []
    for v in variants:
        for mid in metrics_by_variant[v]["recommendation_histogram"]:
            if mid not in models:
                models.append(mid)
    models = sorted(models)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.8 / max(len(variants), 1)
    for k, variant in enumerate(variants):
        hist = metrics_by_variant[variant]["recommendation_histogram"]
        total = sum(hist.values()) or 1
        bottom = 0.0
        for j, mid in enumerate(models):
            share = 100.0 * hist.get(mid, 0) / total
            ax.bar(k * width, share, width=width, bottom=bottom,
                   color=MODEL_COLORS[j % len(MODEL_COLORS)],
                   label=mid if k == 0 else None, edgecolor="white", linewidth=0.4)
            bottom += share
    ax.set_xticks([i * width for i in range(len(variants))])
    ax.set_xticklabels([_variant_style(v)[1] for v in variants], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("share of recommendations [%]")
    ax.set_title(f"Recommended fidelity level per configuration ({truth_label} truth)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pareto_frontier(frontier: Sequence[Mapping[str, Any]],
                         all_models: Sequence[Mapping[str, Any]],
                         budget: Mapping[str, Any], out_path: Path) -> Path:
    """Error-cost plane with the Pareto staircase and the two budget constraints."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in all_models:
        ax.scatter(row["cost_median_s"], 100 * row["error_median"],
                   s=46, color="#9aa4b0", zorder=2, marker="o")
        ax.annotate(row["model_id"], (row["cost_median_s"], 100 * row["error_median"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7, color="#4a4a4a")
    costs = [r["cost_median_s"] for r in frontier]
    errs = [100 * r["error_median"] for r in frontier]
    ax.step(costs, errs, where="post", color="#2b7a3d", lw=1.6, label="Pareto front (medians)")
    ax.scatter(costs, errs, s=70, facecolors="none", edgecolors="#2b7a3d", lw=1.6, zorder=3)
    up_costs = [r["cost_median_s"] for r in frontier]
    up_errs = [100 * r["error_upper"] for r in frontier]
    ax.step(up_costs, up_errs, where="post", color="#2b7a3d", lw=1.0, ls="--",
            label="Pareto front (90 % upper bounds)")
    if budget.get("epsilon") is not None:
        ax.axhline(100 * budget["epsilon"], color="#b03030", ls=":", lw=1.2,
                   label=f"accuracy budget ε = {100 * budget['epsilon']:.1f} %")
    if budget.get("cost") is not None:
        ax.axvline(budget["cost"], color="#4c72b0", ls=":", lw=1.2,
                   label=f"compute budget C = {budget['cost']:.3g} s")
    ax.set_xscale("log")
    ax.set_xlabel("median cost [s, log scale]")
    ax.set_ylabel("relative error on R_T [%]")
    ax.set_title("Error–cost Pareto front of the registered model set")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coverage(metrics_by_truth: Mapping[str, Mapping[str, Mapping[str, Any]]],
                  nominal: float, threshold: float, out_path: Path) -> Path:
    """Nominal vs realised interval coverage, plus the worst-stratum coverage."""
    truths = list(metrics_by_truth)
    variants = list({v for t in truths for v in metrics_by_truth[t]})
    variants = [v for v in VARIANT_STYLE if v in variants] or variants
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    width = 0.8 / max(len(truths), 1)
    for k, truth in enumerate(truths):
        for j, variant in enumerate(variants):
            m = metrics_by_truth[truth].get(variant)
            if m is None:
                continue
            x = k + (j - (len(variants) - 1) / 2.0) * (width / max(len(variants), 1))
            colour = _variant_style(variant)[0]
            ax.bar(x, 100 * m["coverage"], width=width / max(len(variants), 1) * 0.95,
                   color=colour, alpha=0.85,
                   label=_variant_style(variant)[1] if k == 0 else None)
            ax.plot([x - 0.03, x + 0.03], [100 * m["coverage_worst_stratum"]] * 2,
                    color="black", lw=1.6, marker="_", ms=9)
            lo, hi = m["coverage_wilson95"]
            ax.errorbar(x, 100 * m["coverage"], yerr=[[100 * (m["coverage"] - lo)],
                                                      [100 * (hi - m["coverage"])]],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(100 * nominal, color="#2b7a3d", ls="--", lw=1.1,
               label=f"nominal level {100 * nominal:.0f} %")
    ax.axhline(100 * threshold, color="#b03030", ls=":", lw=1.2,
               label=f"acceptance floor {100 * threshold:.0f} %")
    ax.set_xticks(range(len(truths)))
    ax.set_xticklabels(truths)
    ax.set_ylabel("interval coverage [%]")
    ax.set_ylim(0, 105)
    ax.set_title("One-sided 90 % error-bound coverage (bar) and worst stratum (black tick)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_budget_response(records: Sequence[Mapping[str, Any]],
                         labels: Mapping[str, Sequence[Mapping[str, Any]]],
                         thresholds: Mapping[str, float], out_path: Path) -> Path:
    """Accuracy and cost efficiency as a function of the accuracy budget ε."""
    bins = [0.015, 0.025, 0.04, 0.06, 0.09, 0.13, 0.20, 0.35]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for key, recs in labels.items():
        colour, label = _variant_style(key)
        recs = [r for r in recs if r["realised_error"] is not None]
        xs, acc, ce = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sub = [r for r in recs if lo <= r["epsilon"] < hi]
            if not sub:
                continue
            xs.append(0.5 * (lo + hi))
            acc.append(100 * sum(1 for r in sub if r["realised_error"] <= r["epsilon"]) / len(sub))
            ce.append(100 * sum(1 for r in sub if r["realised_cost"] <= r["cost_budget"]) / len(sub))
        axes[0].plot(xs, acc, "o-", color=colour, lw=1.5, ms=4, label=label)
        axes[1].plot(xs, ce, "s-", color=colour, lw=1.5, ms=4, label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("accuracy budget ε (relative error)")
        ax.set_ylim(40, 102)
    axes[0].axhline(100 * thresholds["accuracy"], color="#b03030", ls=":", lw=1.1,
                    label="acceptance 90 %")
    axes[0].set_ylabel("P(realised error ≤ ε) [%]")
    axes[0].set_title("Accuracy vs accuracy budget")
    axes[1].axhline(100 * thresholds["cost_efficiency"], color="#b03030", ls=":", lw=1.1,
                    label="acceptance 90 %")
    axes[1].set_ylabel("P(realised cost ≤ C) [%]")
    axes[1].set_title("Cost efficiency vs accuracy budget")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_error_ratio(metrics_by_variant: Mapping[str, Mapping[str, Any]],
                     records_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
                     out_path: Path) -> Path:
    """Distribution of realised/predicted bound: >1 means the bound was violated."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    variants = [v for v in VARIANT_STYLE if v in records_by_variant]
    data = []
    for variant in variants:
        ratios = [r["realised_error"] / max(r["error_upper"], 1e-15)
                  for r in records_by_variant[variant] if r["realised_error"] is not None]
        data.append(ratios)
    bp = ax.boxplot(data, labels=[_variant_style(v)[1] for v in variants], showfliers=False,
                    patch_artist=True, widths=0.55)
    for patch, variant in zip(bp["boxes"], variants):
        patch.set_facecolor(_variant_style(variant)[0])
        patch.set_alpha(0.55)
    ax.axhline(1.0, color="#b03030", ls="--", lw=1.2, label="bound violated above this line")
    ax.set_yscale("log")
    ax.set_ylabel("realised error / predicted 90 % bound")
    ax.set_title("Calibration of the reported error bound (box = IQR, whiskers = 5–95 %)")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_confidence_sweep(rows: Sequence[Mapping[str, Any]], out_path: Path) -> Path:
    """Price of the guarantee: recommendation rate, cost ratio and coverage vs alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    variants = sorted({r["variant"] for r in rows})
    for variant in variants:
        colour, label = _variant_style(variant)
        sub = sorted([r for r in rows if r["variant"] == variant], key=lambda r: r["alpha"])
        alphas = [100 * r["alpha"] for r in sub]
        axes[0].plot(alphas, [100 * r["recommendation_rate"] for r in sub], "o-", color=colour, label=label)
        axes[1].plot(alphas, [100 * r["accuracy"] for r in sub], "o-", color=colour, label=label)
        axes[2].plot(alphas, [r["median_cost_ratio_vs_oracle"] or 1.0 for r in sub], "o-",
                     color=colour, label=label)
    axes[0].set_title("recommendation rate")
    axes[0].set_ylabel("% of queries answered with a guarantee")
    axes[1].set_title("accuracy  P(error ≤ ε)")
    axes[1].set_ylabel("%")
    axes[1].axhline(90, color="#b03030", ls=":", lw=1.1)
    axes[2].set_title("price of the guarantee")
    axes[2].set_ylabel("median cost / ex-post oracle cost")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("demanded confidence α [%]")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Confidence sweep on the deployment truth (same 2000 queries)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
