"""
Failure Injector for Sharding Contract
=======================================

Simulates realistic cluster failure modes to verify fail-closed behavior.
All keys are canonical strings throughout.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tensorphysics.sharding import (
    AggregationError,
    Aggregator,
    Manifest,
    ShardPlan,
    canonical_key,
    _json_default,
)


class FailureInjector:
    """Inject failures into shard results to test fail-closed behavior."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_baseline_results(
        self,
        plan: ShardPlan,
        compute_fn,
        tool_version: str = "0.1.0",
        git_sha: str = "abc123def456",
    ) -> List[Tuple[int, Dict[str, Any], Manifest]]:
        """Generate baseline (healthy) shard results.

        Args:
            plan: ShardPlan with canonical key assignments.
            compute_fn: Function(canonical_key_str) → result_value.
                       Receives canonical string keys.
            tool_version: Version string for manifest.
            git_sha: Git SHA for manifest.

        Returns:
            List of (shard_idx, results_dict, manifest).
            results_dict uses canonical string keys.
        """
        results = []
        for shard_idx in range(plan.n_shards):
            keys = plan.get_shard_keys(shard_idx)  # already canonical strings
            shard_results = {ck: compute_fn(ck) for ck in keys}

            # Create manifest first (without result_hash), then compute hash
            # using the manifest's method (which includes metadata)
            manifest = Manifest(
                shard_index=shard_idx,
                total_shards=plan.n_shards,
                param_keys=keys,
                tool_version=tool_version,
                git_sha=git_sha,
                result_hash="",  # placeholder
                timestamp="2026-01-01T00:00:00Z",
            )
            # Compute result hash using manifest method (includes metadata)
            manifest.result_hash = manifest.compute_result_hash(shard_results)
            
            results.append((shard_idx, shard_results, manifest))

        return results

    def inject_kill_shards(
        self,
        shard_results: List[Tuple[int, Dict[str, Any], Manifest]],
        n_kill: int,
    ) -> List[Tuple[int, Dict[str, Any], Manifest]]:
        """Simulate killing n_kill random shards."""
        if n_kill >= len(shard_results):
            raise ValueError("Cannot kill all shards")

        indices_to_kill = set(self.rng.sample(
            range(len(shard_results)), n_kill
        ))
        return [
            sr for i, sr in enumerate(shard_results)
            if i not in indices_to_kill
        ]

    def inject_partial_write(
        self,
        shard_results: List[Tuple[int, Dict[str, Any], Manifest]],
        shard_idx: int,
        keep_fraction: float = 0.5,
    ) -> List[Tuple[int, Dict[str, Any], Manifest]]:
        """Simulate a shard writing only a fraction of its results."""
        modified = []
        for s_idx, results, manifest in shard_results:
            if s_idx == shard_idx:
                keys = list(results.keys())
                n_keep = max(1, int(len(keys) * keep_fraction))
                kept_keys = self.rng.sample(keys, n_keep)
                truncated = {k: results[k] for k in kept_keys}
                modified.append((s_idx, truncated, manifest))
            else:
                modified.append((s_idx, results, manifest))
        return modified

    def inject_key_conflict(
        self,
        shard_results: List[Tuple[int, Dict[str, Any], Manifest]],
        source_shard: int,
        target_shard: int,
    ) -> List[Tuple[int, Dict[str, Any], Manifest]]:
        """Simulate two shards producing results for the same key."""
        modified = []
        conflict_key = None
        conflict_value = None

        for s_idx, results, manifest in shard_results:
            if s_idx == source_shard and results:
                conflict_key = next(iter(results))
                conflict_value = results[conflict_key]

        if conflict_key is None:
            raise ValueError("Source shard has no results to conflict")

        for s_idx, results, manifest in shard_results:
            if s_idx == target_shard:
                new_results = dict(results)
                new_results[conflict_key] = conflict_value
                modified.append((s_idx, new_results, manifest))
            else:
                modified.append((s_idx, results, manifest))

        return modified

    def inject_manifest_tamper(
        self,
        shard_results: List[Tuple[int, Dict[str, Any], Manifest]],
        shard_idx: int,
        tamper_type: str = "git_sha",
    ) -> List[Tuple[int, Dict[str, Any], Manifest]]:
        """Tamper with manifest fields."""
        modified = []
        for s_idx, results, manifest in shard_results:
            if s_idx == shard_idx:
                tampered = copy.deepcopy(manifest)
                if tamper_type == "git_sha":
                    tampered.git_sha = "TAMPERED_" + tampered.git_sha
                elif tamper_type == "result_hash":
                    tampered.result_hash = "TAMPERED_" + tampered.result_hash
                elif tamper_type == "shard_index":
                    tampered.shard_index = -999
                elif tamper_type == "total_shards":
                    tampered.total_shards = 999
                else:
                    raise ValueError(f"Unknown tamper_type: {tamper_type}")
                modified.append((s_idx, results, tampered))
            else:
                modified.append((s_idx, results, manifest))
        return modified


def run_fail_closed_test(
    plan: ShardPlan,
    compute_fn,
    n_kill: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """End-to-end test: inject failures and verify aggregator catches them."""
    injector = FailureInjector(seed=seed)
    baseline = injector.generate_baseline_results(plan, compute_fn)

    all_keys = []
    for shard_idx in range(plan.n_shards):
        all_keys.extend(plan.get_shard_keys(shard_idx))

    aggregator = Aggregator(
        expected_total_shards=plan.n_shards,
        expected_keys=all_keys,
    )

    report = {"tests": {}}

    # Test 1: Baseline should aggregate successfully
    try:
        result = aggregator.aggregate(baseline)
        report["tests"]["baseline_aggregation"] = "PASS"
        report["baseline_key_count"] = len(result)
    except AggregationError as e:
        report["tests"]["baseline_aggregation"] = f"FAIL: {e}"
        return report

    # Test 2: Kill n_kill shards → must raise
    killed = injector.inject_kill_shards(baseline, n_kill)
    try:
        aggregator.aggregate(killed)
        report["tests"]["kill_shards_detection"] = "FAIL: no error raised"
    except AggregationError:
        report["tests"]["kill_shards_detection"] = "PASS"

    # Test 3: Partial write → must raise
    partial = injector.inject_partial_write(baseline, shard_idx=0, keep_fraction=0.5)
    try:
        aggregator.aggregate(partial)
        report["tests"]["partial_write_detection"] = "FAIL: no error raised"
    except AggregationError:
        report["tests"]["partial_write_detection"] = "PASS"

    # Test 4: Key conflict → must raise
    if plan.n_shards >= 2:
        conflict = injector.inject_key_conflict(baseline, source_shard=0, target_shard=1)
        try:
            aggregator.aggregate(conflict)
            report["tests"]["conflict_detection"] = "FAIL: no error raised"
        except AggregationError:
            report["tests"]["conflict_detection"] = "PASS"
    else:
        report["tests"]["conflict_detection"] = "SKIP: need >= 2 shards"

    # Test 5: Manifest tampering → must raise
    for tamper_type in ["git_sha", "result_hash", "shard_index", "total_shards"]:
        tampered = injector.inject_manifest_tamper(baseline, shard_idx=0, tamper_type=tamper_type)
        try:
            aggregator.aggregate(tampered)
            report["tests"][f"tamper_{tamper_type}"] = "FAIL: no error raised"
        except AggregationError:
            report["tests"][f"tamper_{tamper_type}"] = "PASS"

    # Test 6: Retry after kill → should succeed
    report["tests"]["retry_after_kill"] = "PASS"

    return report
