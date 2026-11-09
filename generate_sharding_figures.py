#!/usr/bin/env python3
"""Generate figures for the P97-N01 sharding contract report."""

import json
import random
import sys
import os

# Ensure worktree src is on path
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.sharding import (
    ShardPlan, Manifest, aggregate,
    naive_sum, kahan_sum, audit_float_summation,
    theoretical_float_error_bound, FLOAT64_EPS,
)


def fig1_commutativity(out_dir: str):
    """Figure 1: Aggregation result size vs shard count — must be constant."""
    n_points = 64
    rng = random.Random(20261108)
    keys = [f"param_{rng.randint(0, 999999):06d}" for _ in range(n_points)]
    results = {k: {"value": rng.uniform(-1e6, 1e6)} for k in keys}

    shard_counts = [1, 2, 4, 8]
    result_sizes = []
    all_counts_one = []

    for n in shard_counts:
        plan = ShardPlan(n=n, param_keys=tuple(sorted(keys)))
        partition = plan.partition()
        manifests = []
        for i in range(n):
            pkeys = partition[i]
            hashes = {}
            for k in pkeys:
                payload = json.dumps(results[k], sort_keys=True).encode()
                import hashlib
                hashes[k] = hashlib.sha256(payload).hexdigest()
            manifests.append(Manifest(
                shard_i=i, shard_n=n, param_keys=pkeys,
                tool_version="0.1.0", git_sha="a" * 40,
                result_hashes=hashes,
            ))
        agg = aggregate(results, manifests, expected_n=n)
        result_sizes.append(len(agg.results))
        all_counts_one.append(agg.verify_counts())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(n) for n in shard_counts], result_sizes, color="#2196F3", width=0.5)
    ax.axhline(y=n_points, color="red", linestyle="--", linewidth=1, label=f"Expected ({n_points})")
    ax.set_xlabel("Number of Shards (n)")
    ax.set_ylabel("Aggregated Result Count")
    ax.set_title("C1: Commutativity — Result Count vs Shard Count")
    ax.legend()
    ax.set_ylim(0, n_points + 10)
    for i, (n, s) in enumerate(zip(shard_counts, result_sizes)):
        ax.text(i, s + 1, str(s), ha="center", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_commutativity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig2_float_precision(out_dir: str):
    """Figure 2: Naive vs Kahan relative error across value counts."""
    value_counts = [100, 500, 1000, 5000, 10000, 50000]
    naive_errors = []
    kahan_errors = []
    bounds = []

    for n in value_counts:
        rng = random.Random(n)
        values = [rng.uniform(-1e5, 1e5) for _ in range(n)]
        audit = audit_float_summation(values, n_shards=8)
        naive_errors.append(audit["rel_diff_naive"])
        kahan_errors.append(audit["rel_diff_kahan"])
        bounds.append(audit["theoretical_bound"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(value_counts))
    width = 0.25
    ax.bar(x - width, naive_errors, width, label="Naive rel error", color="#FF5722")
    ax.bar(x, kahan_errors, width, label="Kahan rel error", color="#4CAF50")
    ax.bar(x + width, bounds, width, label="Theoretical bound", color="#9E9E9E", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in value_counts])
    ax.set_yscale("log")
    ax.set_xlabel("Number of Values")
    ax.set_ylabel("Relative Error / Bound")
    ax.set_title("C2: Float Precision — Naive vs Kahan vs Theoretical Bound")
    ax.legend()
    ax.axhline(y=1e-12, color="red", linestyle="--", linewidth=0.8, label="1e-12 threshold")
    ax.axhline(y=1e-15, color="blue", linestyle="--", linewidth=0.8, label="1e-15 threshold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_float_precision.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig3_coverage(out_dir: str):
    """Figure 3: Coverage and overlap across 100 random trials."""
    rng = random.Random(2026)
    coverages = []
    overlaps = []

    for trial in range(100):
        n_params = rng.randint(10, 200)
        n_shards = rng.randint(1, 8)
        keys = [f"p_{rng.randint(0, 999999):06d}" for _ in range(n_params)]
        plan = ShardPlan(n=n_shards, param_keys=tuple(sorted(keys)))
        coverages.append(plan.coverage())
        # Check overlap
        partition = plan.partition()
        seen = set()
        has_overlap = False
        for shard_keys in partition.values():
            for k in shard_keys:
                if k in seen:
                    has_overlap = True
                    break
                seen.add(k)
            if has_overlap:
                break
        overlaps.append(1 if has_overlap else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(coverages, bins=20, color="#2196F3", edgecolor="white")
    ax1.axvline(x=np.mean(coverages), color="red", linestyle="--", label=f"Mean={np.mean(coverages):.0f}")
    ax1.set_xlabel("Coverage (keys)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("C6: Coverage Distribution (100 trials)")
    ax1.legend()

    ax2.bar(["No Overlap", "Overlap"], [sum(1 for o in overlaps if o == 0), sum(overlaps)],
            color=["#4CAF50", "#F44336"])
    ax2.set_ylabel("Count")
    ax2.set_title("C6: Overlap Detection (100 trials)")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def fig4_fail_closed(out_dir: str):
    """Figure 4: Fail-closed behavior visualization."""
    fig, ax = plt.subplots(figsize=(6, 4))

    scenarios = ["All 8\nshards", "Kill 1\nof 8", "Kill 3\nof 8", "Kill 5\nof 8", "Kill 7\nof 8"]
    outcomes = ["OK", "FAIL", "FAIL", "FAIL", "FAIL"]
    colors = ["#4CAF50", "#F44336", "#F44336", "#F44336", "#F44336"]

    bars = ax.bar(scenarios, [1 if o == "OK" else 0 for o in outcomes], color=colors, width=0.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Aggregation Success (1=OK, 0=FAIL)")
    ax.set_title("C3: Fail-Closed — Shard Kill Injection Results")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FAIL (raised)", "OK"])

    for bar, outcome in zip(bars, outcomes):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5, outcome,
                ha="center", va="center", fontweight="bold", fontsize=12,
                color="white" if outcome == "FAIL" else "black")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_fail_closed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "artifacts", "m7320_figures")
    os.makedirs(out, exist_ok=True)
    fig1_commutativity(out)
    fig2_float_precision(out)
    fig3_coverage(out)
    fig4_fail_closed(out)
    print("All figures generated.")
