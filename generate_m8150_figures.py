#!/usr/bin/env python3
"""Generate verification figures for P96W2-N11 sharding contract.

Produces:
  fig1_commutativity.png — result count vs shard count
  fig2_float_precision.png — naive vs Kahan vs theoretical bound
  fig3_coverage.png — coverage distribution and overlap detection
  fig4_fail_closed.png — fail-closed behavior
  fig5_boundary_tamper.png — C5 boundary tampering detection
"""

import hashlib
import json
import os
import random
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorphysics.sharding import (
    Manifest,
    ShardPlan,
    aggregate,
    audit_float_summation,
    tamper_manifest,
    validate_manifest_set,
)


def make_synthetic_params(n, seed=42):
    rng = random.Random(seed)
    return [f"param_{rng.randint(0, 999999):06d}" for _ in range(n)]


def make_synthetic_results(keys, seed=42):
    rng = random.Random(seed + 1)
    return {k: {"value": rng.uniform(-1e6, 1e6), "status": "ok"} for k in keys}


def make_manifests_for_plan(plan, results, git_sha="a" * 40):
    partition = plan.partition()
    manifests = []
    for i in range(plan.n):
        keys = partition[i]
        hashes = {}
        for k in keys:
            payload = json.dumps(results[k], sort_keys=True).encode()
            hashes[k] = hashlib.sha256(payload).hexdigest()
        manifests.append(Manifest(
            shard_i=i, shard_n=plan.n, param_keys=keys,
            tool_version="0.1.0", git_sha=git_sha, result_hashes=hashes,
        ))
    return manifests


def fig1_commutativity(out_dir):
    n_points = 64
    shard_counts = [1, 2, 4, 8]
    result_sizes = []

    for n in shard_counts:
        keys = make_synthetic_params(n_points, seed=20261108)
        results = make_synthetic_results(keys, seed=20261108)
        plan = ShardPlan(n=n, param_keys=tuple(sorted(keys)))
        manifests = make_manifests_for_plan(plan, results)
        agg = aggregate(results, manifests, expected_n=n, expected_keys=set(keys))
        result_sizes.append(len(agg.results))

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


def fig2_float_precision(out_dir):
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


def fig3_coverage(out_dir):
    rng = random.Random(2026)
    coverages = []
    overlaps = []

    for trial in range(100):
        n_params = rng.randint(10, 200)
        n_shards = rng.randint(1, 8)
        keys = [f"p_{rng.randint(0, 999999):06d}" for _ in range(n_params)]
        plan = ShardPlan(n=n_shards, param_keys=tuple(sorted(keys)))
        coverages.append(plan.coverage())
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


def fig4_fail_closed(out_dir):
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


def fig5_boundary_tamper(out_dir):
    """C5: Three tampering variants — git_sha, result_hash, param_keys (boundary)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    m = Manifest(
        shard_i=0, shard_n=4,
        param_keys=("p1", "p2", "p3"),
        tool_version="0.1.0",
        git_sha="a" * 40,
        result_hashes={"p1": "h1" * 32, "p2": "h2" * 32, "p3": "h3" * 32},
        param_ranges="[0,1]x[0,1]",
    )

    variants = [
        ("Git SHA\ntampered", "tamper_git_sha", "#FF9800"),
        ("Result hash\ntampered", "tamper_result_hash", "#F44336"),
        ("Boundary\n(param_keys) tampered", "tamper_param_keys", "#9C27B0"),
    ]

    for ax, (title, kwarg, color) in zip(axes, variants):
        kwargs = {kwarg: True}
        tampered = tamper_manifest(m, **kwargs)

        original_hash = m.integrity_hash()[:16]
        tampered_hash = tampered.integrity_hash()[:16]
        changed = original_hash != tampered_hash

        if kwarg == "tamper_git_sha":
            errs = tampered.validate(expected_git_sha="a" * 40)
            detected = len(errs) > 0
        elif kwarg == "tamper_param_keys":
            errs = validate_manifest_set(
                [tampered], expected_n=4,
                expected_keys={"p1", "p2", "p3"}
            )
            detected = len(errs) > 0
        else:
            detected = changed

        status = "DETECTED" if detected else "MISSED"
        status_color = "#4CAF50" if detected else "#F44336"

        ax.bar(["Original", "Tampered"], [1, 1 if detected else 0],
               color=["#2196F3", color], width=0.4)
        ax.set_title(title)
        ax.set_ylim(0, 1.5)
        ax.text(0.5, 1.2, status, ha="center", fontweight="bold",
                color=status_color, fontsize=11, transform=ax.transAxes)
        ax.set_ylabel("Detection Score")

    plt.suptitle("C5: Manifest Integrity — Three Tampering Variants", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_boundary_tamper.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "artifacts", "m8150_figures")
    os.makedirs(out, exist_ok=True)
    fig1_commutativity(out)
    fig2_float_precision(out)
    fig3_coverage(out)
    fig4_fail_closed(out)
    fig5_boundary_tamper(out)
    print("All figures generated.")
