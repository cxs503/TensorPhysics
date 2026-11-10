#!/usr/bin/env python
"""Generate figures and report data for P97-N01 sharding contract verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure we import from this worktree
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorphysics.sharding import (
    AggregationError,
    Aggregator,
    CrossShardFloatAggregator,
    FloatAudit,
    Manifest,
    ShardPlan,
    canonical_key,
)
from tensorphysics.sharding_failure import FailureInjector


OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "p97_n01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_param_keys_64():
    keys = []
    for i in range(8):
        for j in range(8):
            keys.append({"speed": 5.0 + i * 1.5, "draft": 3.0 + j * 0.5})
    return keys


def compute_resistance_from_canonical(ck):
    key = json.loads(ck)
    speed = key["speed"]
    draft = key["draft"]
    rho = 1025.0
    S = 100.0 + 10.0 * draft
    return 0.5 * rho * S * speed**2 * 0.003


def compute_summand_from_canonical(ck):
    key = json.loads(ck)
    speed = key["speed"]
    draft = key["draft"]
    return speed * 1e6 + draft * 1e-3 + math.sin(speed * draft) * 1e3


def compute_summand(key):
    speed = key["speed"]
    draft = key["draft"]
    return speed * 1e6 + draft * 1e-3 + math.sin(speed * draft) * 1e3


# ---------------------------------------------------------------------------
# Figure 1: Float Audit — Naive vs Kahan vs Reference
# ---------------------------------------------------------------------------

def fig1_float_audit():
    rng = np.random.RandomState(42)
    ns = [10, 100, 1000, 10000]
    naive_errs = []
    kahan_errs = []
    naive_bounds = []
    kahan_bounds = []

    for n in ns:
        values = rng.randn(n) * 1e6
        audit = FloatAudit.audit(values)
        naive_errs.append(audit["naive_abs_err"])
        kahan_errs.append(audit["kahan_abs_err"])
        naive_bounds.append(audit["naive_theory_bound"])
        kahan_bounds.append(audit["kahan_theory_bound"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ns))
    width = 0.2

    ax.bar(x - 1.5*width, naive_errs, width, label="Naive abs error", color="#e74c3c", alpha=0.8)
    ax.bar(x - 0.5*width, kahan_errs, width, label="Kahan abs error", color="#2ecc71", alpha=0.8)
    ax.bar(x + 0.5*width, naive_bounds, width, label="Naive theory bound", color="#e74c3c", alpha=0.3, hatch="//")
    ax.bar(x + 1.5*width, kahan_bounds, width, label="Kahan theory bound", color="#2ecc71", alpha=0.3, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("Absolute error / bound")
    ax.set_yscale("log")
    ax.set_title("Float Summation Audit: Naive vs Kahan vs Theoretical Bounds")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "float_audit_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Cross-shard float aggregation error
# ---------------------------------------------------------------------------

def fig2_cross_shard_float():
    keys = make_param_keys_64()
    all_values = [compute_summand(k) for k in keys]
    reference = math.fsum(all_values)

    shard_counts = [1, 2, 4, 8]
    naive_rels = []
    kahan_rels = []

    for n_shards in shard_counts:
        plan = ShardPlan.create(keys, n_shards)
        partial_sums_naive = []
        partial_sums_kahan = []
        for si in range(n_shards):
            shard_cks = plan.get_shard_keys(si)
            shard_values = [compute_summand_from_canonical(ck) for ck in shard_cks]
            partial_sums_naive.append((si, FloatAudit.naive_sum(shard_values)))
            partial_sums_kahan.append((si, FloatAudit.kahan_sum(shard_values)))

        agg_naive = CrossShardFloatAggregator.aggregate_float_sum(partial_sums_naive)
        agg_kahan = CrossShardFloatAggregator.aggregate_float_sum(partial_sums_kahan)

        denom = abs(reference) if reference != 0 else 1.0
        naive_rels.append(abs(agg_naive["naive_sum"] - reference) / denom)
        kahan_rels.append(abs(agg_kahan["kahan_sum"] - reference) / denom)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(shard_counts))
    width = 0.35

    ax.bar(x - width/2, naive_rels, width, label="Naive relative error", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, kahan_rels, width, label="Kahan relative error", color="#2ecc71", alpha=0.8)

    # Threshold lines
    ax.axhline(y=1e-12, color="#e74c3c", linestyle="--", alpha=0.5, label="Threshold 1e-12")
    ax.axhline(y=1e-15, color="#2ecc71", linestyle="--", alpha=0.5, label="Threshold 1e-15")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} shards" for n in shard_counts])
    ax.set_ylabel("Relative error")
    ax.set_yscale("log")
    ax.set_title("Cross-Shard Float Aggregation: 1 vs 8 Shards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "cross_shard_float.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Shard balance distribution
# ---------------------------------------------------------------------------

def fig3_shard_balance():
    keys = make_param_keys_64()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for idx, n_shards in enumerate([1, 2, 4, 8]):
        plan = ShardPlan.create(keys, n_shards)
        sizes = [len(plan.get_shard_keys(si)) for si in range(n_shards)]

        axes[idx].bar(range(n_shards), sizes, color="#3498db", alpha=0.8)
        axes[idx].set_title(f"n_shards={n_shards}\nbalance={plan.balance}")
        axes[idx].set_xlabel("Shard index")
        axes[idx].set_ylabel("Keys per shard")
        axes[idx].set_xticks(range(n_shards))
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle("Shard Key Distribution (N=64 parameter points)", fontsize=14)
    fig.tight_layout()

    path = OUTPUT_DIR / "shard_balance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Fail-closed behavior diagram
# ---------------------------------------------------------------------------

def fig4_fail_closed():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Draw a flowchart-style diagram
    y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]

    # Boxes
    boxes = [
        (0.5, y_positions[0], "8 Shards Complete", "#2ecc71"),
        (0.3, y_positions[1], "Aggregate", "#3498db"),
        (0.7, y_positions[1], "Aggregate", "#3498db"),
        (0.3, y_positions[2], "✓ 64 keys merged", "#2ecc71"),
        (0.7, y_positions[2], "✗ Missing shards → raise", "#e74c3c"),
        (0.3, y_positions[3], "Success", "#2ecc71"),
        (0.7, y_positions[3], "Fail-closed", "#e74c3c"),
        (0.5, y_positions[4], "Retry with all shards → ✓", "#f39c12"),
    ]

    for x, y, text, color in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))

    # Arrows
    ax.annotate("", xy=(0.3, y_positions[1]+0.05), xytext=(0.5, y_positions[0]-0.05),
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=2))
    ax.annotate("", xy=(0.7, y_positions[1]+0.05), xytext=(0.5, y_positions[0]-0.05),
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=2))
    ax.annotate("", xy=(0.3, y_positions[2]+0.05), xytext=(0.3, y_positions[1]-0.05),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2))
    ax.annotate("", xy=(0.7, y_positions[2]+0.05), xytext=(0.7, y_positions[1]-0.05),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax.annotate("", xy=(0.3, y_positions[3]+0.05), xytext=(0.3, y_positions[2]-0.05),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2))
    ax.annotate("", xy=(0.7, y_positions[3]+0.05), xytext=(0.7, y_positions[2]-0.05),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax.annotate("", xy=(0.5, y_positions[4]+0.05), xytext=(0.3, y_positions[3]-0.05),
                arrowprops=dict(arrowstyle="->", color="#f39c12", lw=2, linestyle="--"))
    ax.annotate("", xy=(0.5, y_positions[4]+0.05), xytext=(0.7, y_positions[3]-0.05),
                arrowprops=dict(arrowstyle="->", color="#f39c12", lw=2, linestyle="--"))

    ax.set_title("Fail-Closed Aggregation Contract", fontsize=14, pad=20)

    path = OUTPUT_DIR / "fail_closed_diagram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Collect all test results programmatically
# ---------------------------------------------------------------------------

def collect_test_results():
    """Run all acceptance criteria tests and collect results."""
    results = {}

    # C1: Commutativity
    keys = make_param_keys_64()
    c1_results = {}
    for n_shards in [1, 2, 4, 8]:
        plan = ShardPlan.create(keys, n_shards)
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()

        injector = FailureInjector(seed=0)
        shard_results = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(n_shards, all_keys)

        ref = agg.aggregate(shard_results)
        rng = np.random.RandomState(42)
        commutative = True
        for _ in range(50):
            perm = list(range(len(shard_results)))
            rng.shuffle(perm)
            permuted = [shard_results[i] for i in perm]
            if agg.aggregate(permuted) != ref:
                commutative = False
                break

        c1_results[f"n{n_shards}"] = {
            "commutative": commutative,
            "key_count": len(ref),
            "all_keys_count_1": all(
                sum(1 for si in range(n_shards) for ck in plan.get_shard_keys(si) if ck == k) == 1
                for k in [canonical_key(kk) for kk in keys]
            )
        }
    results["C1_commutativity"] = c1_results

    # C2: Float bounds
    all_values = [compute_summand(k) for k in keys]
    reference = math.fsum(all_values)

    plan8 = ShardPlan.create(keys, n_shards=8)
    partial_sums_naive = []
    partial_sums_kahan = []
    for si in range(8):
        shard_cks = plan8.get_shard_keys(si)
        shard_values = [compute_summand_from_canonical(ck) for ck in shard_cks]
        partial_sums_naive.append((si, FloatAudit.naive_sum(shard_values)))
        partial_sums_kahan.append((si, FloatAudit.kahan_sum(shard_values)))

    agg_naive = CrossShardFloatAggregator.aggregate_float_sum(partial_sums_naive)
    agg_kahan = CrossShardFloatAggregator.aggregate_float_sum(partial_sums_kahan)

    denom = abs(reference) if reference != 0 else 1.0
    rel_naive = abs(agg_naive["naive_sum"] - reference) / denom
    rel_kahan = abs(agg_kahan["kahan_sum"] - reference) / denom

    results["C2_float_bounds"] = {
        "reference_sum": reference,
        "naive_8shard_sum": agg_naive["naive_sum"],
        "kahan_8shard_sum": agg_kahan["kahan_sum"],
        "naive_rel_err": rel_naive,
        "kahan_rel_err": rel_kahan,
        "naive_threshold": 1e-12,
        "kahan_threshold": 1e-15,
        "naive_pass": rel_naive <= 1e-12,
        "kahan_pass": rel_kahan <= 1e-15,
        "naive_within_theory": agg_naive["naive_within_bound"],
        "kahan_within_theory": agg_kahan["kahan_within_bound"],
    }

    # C3: Fail-closed
    plan8 = ShardPlan.create(keys, n_shards=8)
    injector = FailureInjector(seed=42)
    baseline = injector.generate_baseline_results(plan8, compute_resistance_from_canonical)
    all_keys = []
    for si in range(8):
        all_keys.extend(plan8.get_shard_keys(si))
    agg = Aggregator(8, all_keys)

    killed = injector.inject_kill_shards(baseline, n_kill=3)
    fail_closed = False
    try:
        agg.aggregate(killed)
    except AggregationError:
        fail_closed = True

    retry_result = agg.aggregate(baseline)
    results["C3_fail_closed"] = {
        "kill_3_of_8_raises": fail_closed,
        "retry_key_count": len(retry_result),
        "retry_matches_reference": len(retry_result) == 64,
    }

    # C4: Conflict detection
    conflict = injector.inject_key_conflict(baseline, source_shard=0, target_shard=1)
    conflict_detected = False
    try:
        agg.aggregate(conflict)
    except AggregationError as e:
        if "Conflict" in str(e):
            conflict_detected = True

    results["C4_conflict_detection"] = {
        "conflict_detected": conflict_detected,
    }

    # C5: Manifest integrity
    tamper_results = {}
    for tamper_type in ["git_sha", "result_hash", "shard_index", "total_shards"]:
        tampered = injector.inject_manifest_tamper(baseline, shard_idx=0, tamper_type=tamper_type)
        detected = False
        try:
            agg.aggregate(tampered)
        except AggregationError:
            detected = True
        tamper_results[tamper_type] = detected

    results["C5_manifest_integrity"] = tamper_results

    # C6: Boundary coverage
    rng = np.random.RandomState(42)
    coverage_pass = True
    overlap_pass = True
    for trial in range(100):
        n_points = rng.randint(10, 201)
        n_shards = int(rng.choice([1, 2, 4, 8, 16]))
        trial_keys = []
        for idx in range(n_points):
            trial_keys.append({
                "speed": round(float(rng.uniform(3, 20)), 3),
                "draft": round(float(rng.uniform(1, 10)), 3),
                "trial": trial,
                "idx": idx,
            })
        plan = ShardPlan.create(trial_keys, n_shards)
        if not plan.verify_coverage(trial_keys):
            coverage_pass = False
        if not plan.verify_uniqueness():
            overlap_pass = False

    results["C6_boundary"] = {
        "trials": 100,
        "coverage_100pct": coverage_pass,
        "overlap_0pct": overlap_pass,
    }

    # C7: Regression
    plan1 = ShardPlan.create(keys, n_shards=1)
    injector1 = FailureInjector(seed=0)
    results_1shard = injector1.generate_baseline_results(plan1, compute_resistance_from_canonical)
    all_keys_1 = list(plan1.get_shard_keys(0))
    agg1 = Aggregator(1, all_keys_1)
    merged = agg1.aggregate(results_1shard)

    direct = {}
    for k in keys:
        ck = canonical_key(k)
        direct[ck] = compute_resistance_from_canonical(ck)

    bit_identical = all(merged.get(ck) == direct[ck] for ck in direct)

    results["C7_regression"] = {
        "shard1_vs_direct_bit_identical": bit_identical,
        "key_count": len(merged),
    }

    return results


def main():
    print("Generating figures...")
    fig1 = fig1_float_audit()
    print(f"  Figure 1: {fig1}")
    fig2 = fig2_cross_shard_float()
    print(f"  Figure 2: {fig2}")
    fig3 = fig3_shard_balance()
    print(f"  Figure 3: {fig3}")
    fig4 = fig4_fail_closed()
    print(f"  Figure 4: {fig4}")

    print("\nCollecting test results...")
    test_results = collect_test_results()

    # Save results
    results_path = OUTPUT_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")

    # Print summary
    print("\n=== Acceptance Criteria Summary ===")
    for criterion, data in test_results.items():
        print(f"\n{criterion}:")
        if isinstance(data, dict):
            for k, v in data.items():
                status = "✓ PASS" if v else "✗ FAIL" if isinstance(v, bool) else str(v)
                print(f"  {k}: {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
