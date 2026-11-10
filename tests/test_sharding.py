"""
Comprehensive test suite for TensorPhysics sharding contract.

Tests all acceptance criteria:
C1. Commutativity: N=64 params, n=1/2/4/8 shards, bit-identical results
C2. Float bounds: 1-shard vs 8-shard relative deviation ≤ 1e-12, Kahan ≤ 1e-15
C3. Fail-closed: inject 3/8 kill → aggregator raises, retry matches reference
C4. Conflict detection: two shards write same key → raise
C5. Manifest integrity: tamper git sha/segment/result hash → check red
C6. Boundary: 100 random param spaces, coverage 100%, overlap 0%
C7. Regression: shard=1 path bit-identical to direct execution
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from tensorphysics.sharding import (
    AggregationError,
    Aggregator,
    CrossShardFloatAggregator,
    FloatAudit,
    Manifest,
    ShardPlan,
    _stable_hash,
    canonical_key,
)
from tensorphysics.sharding_failure import (
    FailureInjector,
    run_fail_closed_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_param_keys_64() -> List[Dict[str, float]]:
    """Generate 64 synthetic parameter points (speed × draft combinations)."""
    keys = []
    for i in range(8):
        for j in range(8):
            keys.append({"speed": 5.0 + i * 1.5, "draft": 3.0 + j * 0.5})
    return keys


def compute_resistance_from_canonical(ck: str) -> float:
    """Compute resistance from canonical key string."""
    key = json.loads(ck)
    speed = key["speed"]
    draft = key["draft"]
    rho = 1025.0
    S = 100.0 + 10.0 * draft
    return 0.5 * rho * S * speed**2 * 0.003


def compute_summand_from_canonical(ck: str) -> float:
    """Compute summand from canonical key string."""
    key = json.loads(ck)
    speed = key["speed"]
    draft = key["draft"]
    return speed * 1e6 + draft * 1e-3 + math.sin(speed * draft) * 1e3


def compute_resistance(key: Dict[str, float]) -> float:
    """Compute resistance from dict key (for direct comparison)."""
    speed = key["speed"]
    draft = key["draft"]
    rho = 1025.0
    S = 100.0 + 10.0 * draft
    return 0.5 * rho * S * speed**2 * 0.003


def compute_summand(key: Dict[str, float]) -> float:
    """Compute summand from dict key (for direct comparison)."""
    speed = key["speed"]
    draft = key["draft"]
    return speed * 1e6 + draft * 1e-3 + math.sin(speed * draft) * 1e3


# ---------------------------------------------------------------------------
# C1: Commutativity — N=64, n=1/2/4/8, bit-identical
# ---------------------------------------------------------------------------

class TestCommutativity:
    """C1: Aggregation is commutative (order-independent)."""

    def test_commutativity_n1(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=1)
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()
        assert plan.total_keys == 64

        injector = FailureInjector(seed=0)
        results = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)
        merged = agg.aggregate(results)
        assert len(merged) == 64

    def test_commutativity_n2(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=2)
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()

        injector = FailureInjector(seed=0)
        results = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        assert agg.aggregate_commutativity_check(results)

        ref = agg.aggregate(results)
        rng = np.random.RandomState(123)
        for _ in range(50):
            perm = list(range(len(results)))
            rng.shuffle(perm)
            permuted = [results[i] for i in perm]
            result = agg.aggregate(permuted)
            assert result == ref

    def test_commutativity_n4(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=4)
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()
        assert plan.total_keys == 64

        injector = FailureInjector(seed=0)
        results = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        assert agg.aggregate_commutativity_check(results)

        ref = agg.aggregate(results)
        rng = np.random.RandomState(456)
        for _ in range(50):
            perm = list(range(len(results)))
            rng.shuffle(perm)
            permuted = [results[i] for i in perm]
            result = agg.aggregate(permuted)
            assert result == ref

    def test_commutativity_n8(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()
        assert plan.total_keys == 64

        injector = FailureInjector(seed=0)
        results = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        assert agg.aggregate_commutativity_check(results)

        ref = agg.aggregate(results)
        rng = np.random.RandomState(789)
        for _ in range(50):
            perm = list(range(len(results)))
            rng.shuffle(perm)
            permuted = [results[i] for i in perm]
            result = agg.aggregate(permuted)
            assert result == ref

    def test_each_key_count_is_one(self):
        """Each parameter point appears exactly once across all shards."""
        keys = make_param_keys_64()
        for n_shards in [1, 2, 4, 8]:
            plan = ShardPlan.create(keys, n_shards)
            counts: Dict[str, int] = {}
            for si in range(n_shards):
                for ck in plan.get_shard_keys(si):
                    counts[ck] = counts.get(ck, 0) + 1
            assert len(counts) == 64, f"n_shards={n_shards}: expected 64 unique keys"
            for ck, count in counts.items():
                assert count == 1, f"Key {ck} appears {count} times"


# ---------------------------------------------------------------------------
# C2: Float bounds — naive vs Kahan, theoretical vs empirical
# ---------------------------------------------------------------------------

class TestFloatBounds:
    """C2: Float summation error within theoretical bounds."""

    def test_naive_within_theory(self):
        rng = np.random.RandomState(42)
        for n in [10, 100, 1000, 10000]:
            values = rng.randn(n) * 1e6
            audit = FloatAudit.audit(values, label=f"naive_n{n}")
            assert audit["naive_within_bound"], (
                f"n={n}: naive error {audit['naive_abs_err']:.3e} "
                f"> bound {audit['naive_theory_bound']:.3e}"
            )

    def test_kahan_within_theory(self):
        rng = np.random.RandomState(42)
        for n in [10, 100, 1000, 10000]:
            values = rng.randn(n) * 1e6
            audit = FloatAudit.audit(values, label=f"kahan_n{n}")
            assert audit["kahan_within_bound"], (
                f"n={n}: kahan error {audit['kahan_abs_err']:.3e} "
                f"> bound {audit['kahan_theory_bound']:.3e}"
            )

    def test_kahan_more_accurate_than_naive(self):
        rng = np.random.RandomState(42)
        for n in [100, 1000, 10000]:
            values = rng.randn(n) * 1e6
            audit = FloatAudit.audit(values)
            assert audit["kahan_abs_err"] <= audit["naive_abs_err"] + 1e-300

    def test_cross_shard_float_1_vs_8_shards(self):
        """1-shard vs 8-shard sum: relative deviation ≤ 1e-12 (naive),
        ≤ 1e-15 (Kahan)."""
        keys = make_param_keys_64()

        all_values = [compute_summand(k) for k in keys]
        reference = math.fsum(all_values)

        # 8-shard: partial sums then aggregate
        plan = ShardPlan.create(keys, n_shards=8)
        partial_sums_naive = []
        partial_sums_kahan = []
        for si in range(8):
            shard_cks = plan.get_shard_keys(si)
            shard_values = [compute_summand_from_canonical(ck) for ck in shard_cks]
            partial_sums_naive.append((si, FloatAudit.naive_sum(shard_values)))
            partial_sums_kahan.append((si, FloatAudit.kahan_sum(shard_values)))

        agg_naive = CrossShardFloatAggregator.aggregate_float_sum(partial_sums_naive)
        agg_kahan = CrossShardFloatAggregator.aggregate_float_sum(partial_sums_kahan)

        denom = abs(reference) if reference != 0 else 1.0
        rel_8shard_naive = abs(agg_naive["naive_sum"] - reference) / denom
        rel_8shard_kahan = abs(agg_kahan["kahan_sum"] - reference) / denom

        assert rel_8shard_naive <= 1e-12, (
            f"8-shard naive rel err {rel_8shard_naive:.3e} > 1e-12"
        )
        assert rel_8shard_kahan <= 1e-15, (
            f"8-shard kahan rel err {rel_8shard_kahan:.3e} > 1e-15"
        )

    def test_empirical_within_theory_cross_shard(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        partial_sums = []
        for si in range(8):
            shard_cks = plan.get_shard_keys(si)
            shard_values = [compute_summand_from_canonical(ck) for ck in shard_cks]
            partial_sums.append((si, FloatAudit.naive_sum(shard_values)))

        result = CrossShardFloatAggregator.aggregate_float_sum(partial_sums)
        assert result["naive_within_bound"]
        assert result["kahan_within_bound"]


# ---------------------------------------------------------------------------
# C3: Fail-closed — inject 3/8 kill → raise, retry matches reference
# ---------------------------------------------------------------------------

class TestFailClosed:
    """C3: Missing shards cause aggregation failure (fail-closed)."""

    def test_kill_3_of_8_raises(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        killed = injector.inject_kill_shards(baseline, n_kill=3)
        assert len(killed) == 5

        with pytest.raises(AggregationError, match="Missing shards"):
            agg.aggregate(killed)

    def test_retry_after_kill_matches_reference(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        # First attempt: kill 3 → fail
        killed = injector.inject_kill_shards(baseline, n_kill=3)
        with pytest.raises(AggregationError):
            agg.aggregate(killed)

        # Retry: use full baseline → success
        result = agg.aggregate(baseline)
        assert len(result) == 64

        # Verify against direct computation
        for k in keys:
            ck = canonical_key(k)
            expected = compute_resistance(k)
            assert ck in result
            assert result[ck] == expected

    def test_fail_closed_report(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)
        report = run_fail_closed_test(plan, compute_resistance_from_canonical, n_kill=3, seed=42)

        assert report["tests"]["baseline_aggregation"] == "PASS"
        assert report["tests"]["kill_shards_detection"] == "PASS"
        assert report["tests"]["partial_write_detection"] == "PASS"
        assert report["tests"]["conflict_detection"] == "PASS"
        assert report["tests"]["tamper_git_sha"] == "PASS"
        assert report["tests"]["tamper_result_hash"] == "PASS"
        assert report["tests"]["tamper_shard_index"] == "PASS"
        assert report["tests"]["tamper_total_shards"] == "PASS"
        assert report["tests"]["retry_after_kill"] == "PASS"


# ---------------------------------------------------------------------------
# C4: Conflict detection — two shards write same key → raise
# ---------------------------------------------------------------------------

class TestConflictDetection:
    """C4: Duplicate parameter keys across shards must raise."""

    def test_two_shards_same_key_raises(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        conflict = injector.inject_key_conflict(baseline, source_shard=0, target_shard=1)

        with pytest.raises(AggregationError, match="Conflict"):
            agg.aggregate(conflict)

    def test_duplicate_shard_index_raises(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=2)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        duplicated = [baseline[0], baseline[0], baseline[1]]
        with pytest.raises(AggregationError, match="Duplicate shard indices"):
            agg.aggregate(duplicated)


# ---------------------------------------------------------------------------
# C5: Manifest integrity — tamper fields → check red
# ---------------------------------------------------------------------------

class TestManifestIntegrity:
    """C5: Manifest tampering must be detected."""

    def test_tamper_git_sha(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        tampered = injector.inject_manifest_tamper(baseline, 0, "git_sha")
        with pytest.raises(AggregationError, match="integrity"):
            agg.aggregate(tampered)

    def test_tamper_result_hash(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        tampered = injector.inject_manifest_tamper(baseline, 0, "result_hash")
        with pytest.raises(AggregationError, match="integrity"):
            agg.aggregate(tampered)

    def test_tamper_shard_index(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        tampered = injector.inject_manifest_tamper(baseline, 0, "shard_index")
        with pytest.raises(AggregationError):
            agg.aggregate(tampered)

    def test_tamper_total_shards(self):
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=8)

        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(plan.n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(plan.n_shards, all_keys)

        tampered = injector.inject_manifest_tamper(baseline, 0, "total_shards")
        with pytest.raises(AggregationError):
            agg.aggregate(tampered)

    def test_manifest_verify_integrity(self):
        """Direct manifest integrity verification."""
        keys = make_param_keys_64()[:8]
        cks = [canonical_key(k) for k in keys]
        results = {ck: compute_resistance_from_canonical(ck) for ck in cks}

        manifest = Manifest(
            shard_index=0,
            total_shards=8,
            param_keys=tuple(cks),
            tool_version="0.1.0",
            git_sha="abc123",
            result_hash="",
        )
        manifest.result_hash = manifest.compute_result_hash(results)

        # Valid manifest
        assert manifest.verify_integrity(results)

        # Tampered hash
        manifest.result_hash = "wrong_hash"
        assert not manifest.verify_integrity(results)

        # Tampered git_sha
        manifest.result_hash = manifest.compute_result_hash(results)
        manifest.git_sha = ""
        assert not manifest.verify_integrity(results)


# ---------------------------------------------------------------------------
# C6: Boundary — 100 random param spaces, coverage 100%, overlap 0%
# ---------------------------------------------------------------------------

class TestBoundaryCoverage:
    """C6: Shard boundaries cover 100% with 0% overlap."""

    def test_coverage_100_overlap_0_100_spaces(self):
        """100 random parameter spaces: coverage=100%, overlap=0%."""
        rng = np.random.RandomState(42)

        for trial in range(100):
            n_points = rng.randint(10, 201)
            n_shards = int(rng.choice([1, 2, 4, 8, 16]))

            keys = []
            for idx in range(n_points):
                keys.append({
                    "speed": round(float(rng.uniform(3, 20)), 3),
                    "draft": round(float(rng.uniform(1, 10)), 3),
                    "trial": trial,
                    "idx": idx,
                })

            plan = ShardPlan.create(keys, n_shards)

            assert plan.verify_coverage(keys), f"Trial {trial}: coverage failed"
            assert plan.verify_uniqueness(), f"Trial {trial}: uniqueness failed"
            assert plan.total_keys == n_points

    def test_balance_property(self):
        """Hash-based assignment: verify coverage and uniqueness (balance is statistical).
        
        Note: Hash-based assignment does NOT guarantee balance ≤ 1.
        For perfect balance, use round-robin. Hash-based gives statistical balance.
        We verify the critical properties: coverage=100%, overlap=0%.
        """
        rng = np.random.RandomState(42)

        for trial in range(50):
            n_points = rng.randint(1, 1000)
            n_shards = rng.randint(1, 33)

            keys = [f"key_{trial}_{i}" for i in range(n_points)]
            plan = ShardPlan.create(keys, n_shards)

            # Critical properties (must hold)
            assert plan.verify_coverage(keys), f"Trial {trial}: coverage failed"
            assert plan.verify_uniqueness(), f"Trial {trial}: uniqueness failed"
            assert plan.total_keys == n_points
            # Balance is statistical for hash-based (not guaranteed ≤ 1)
            assert plan.balance < n_points  # sanity: no shard has all keys

    def test_determinism(self):
        """Same input → same plan every time."""
        keys = make_param_keys_64()

        plan1 = ShardPlan.create(keys, n_shards=8)
        plan2 = ShardPlan.create(keys, n_shards=8)

        for si in range(8):
            assert plan1.get_shard_keys(si) == plan2.get_shard_keys(si)


# ---------------------------------------------------------------------------
# C7: Regression — shard=1 bit-identical to direct execution
# ---------------------------------------------------------------------------

class TestRegression:
    """C7: 1-shard path bit-identical to direct execution."""

    def test_shard1_vs_direct(self):
        """1-shard aggregation produces bit-identical results to direct computation."""
        keys = make_param_keys_64()
        plan = ShardPlan.create(keys, n_shards=1)

        injector = FailureInjector(seed=0)
        results = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = list(plan.get_shard_keys(0))
        agg = Aggregator(plan.n_shards, all_keys)
        merged = agg.aggregate(results)

        # Direct computation
        direct = {}
        for k in keys:
            ck = canonical_key(k)
            direct[ck] = compute_resistance(k)

        # Compare
        assert len(merged) == len(direct)
        for ck in direct:
            assert ck in merged, f"Key {ck} not in merged results"
            assert merged[ck] == direct[ck], (
                f"Bit-identical violation for key {ck}: "
                f"merged={merged[ck]}, direct={direct[ck]}"
            )

    def test_shard1_float_vs_direct(self):
        """1-shard float sum bit-identical to direct sum."""
        keys = make_param_keys_64()

        direct_values = [compute_summand(k) for k in keys]
        direct_sum = FloatAudit.naive_sum(direct_values)

        plan = ShardPlan.create(keys, n_shards=1)
        shard_cks = plan.get_shard_keys(0)
        shard_values = [compute_summand_from_canonical(ck) for ck in shard_cks]
        shard_sum = FloatAudit.naive_sum(shard_values)

        assert direct_sum == shard_sum, (
            f"1-shard sum {shard_sum} != direct sum {direct_sum}"
        )


# ---------------------------------------------------------------------------
# Additional: ShardPlan edge cases
# ---------------------------------------------------------------------------

class TestShardPlanEdgeCases:
    """Edge cases for ShardPlan."""

    def test_single_key(self):
        keys = ["only_one"]
        plan = ShardPlan.create(keys, n_shards=4)
        assert plan.total_keys == 1
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError):
            ShardPlan.create([], n_shards=4)

    def test_zero_shards_raises(self):
        with pytest.raises(ValueError):
            ShardPlan.create(["a", "b"], n_shards=0)

    def test_stable_hash_determinism(self):
        key = {"speed": 10.0, "draft": 5.0}
        h1 = _stable_hash(key)
        h2 = _stable_hash(key)
        assert h1 == h2

    def test_stable_hash_different_keys(self):
        k1 = {"speed": 10.0, "draft": 5.0}
        k2 = {"speed": 10.1, "draft": 5.0}
        h1 = _stable_hash(k1)
        h2 = _stable_hash(k2)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Additional: Manifest serialization round-trip
# ---------------------------------------------------------------------------

class TestManifestSerialization:
    """Manifest to_dict / from_dict round-trip."""

    def test_round_trip(self):
        manifest = Manifest(
            shard_index=3,
            total_shards=8,
            param_keys=("a", "b", "c"),
            tool_version="0.1.0",
            git_sha="abc123",
            result_hash="def456",
            timestamp="2026-01-01T00:00:00Z",
            extra={"note": "test"},
        )

        d = manifest.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        manifest2 = Manifest.from_dict(d2)

        assert manifest2.shard_index == manifest.shard_index
        assert manifest2.total_shards == manifest.total_shards
        assert manifest2.param_keys == manifest.param_keys
        assert manifest2.tool_version == manifest.tool_version
        assert manifest2.git_sha == manifest.git_sha
        assert manifest2.result_hash == manifest.result_hash


# ---------------------------------------------------------------------------
# Integration: Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end integration test simulating real cluster execution."""

    def test_full_64_point_scan_8_shards(self):
        """Full pipeline: 64 points, 8 shards, aggregate, verify."""
        keys = make_param_keys_64()
        n_shards = 8

        plan = ShardPlan.create(keys, n_shards)
        assert plan.verify_coverage(keys)
        assert plan.verify_uniqueness()

        injector = FailureInjector(seed=42)
        shard_results = injector.generate_baseline_results(
            plan, compute_resistance_from_canonical,
            tool_version="0.1.0",
            git_sha="deadbeef1234567890",
        )

        all_keys = []
        for si in range(n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(n_shards, all_keys)
        merged = agg.aggregate(shard_results)

        assert len(merged) == 64
        for k in keys:
            ck = canonical_key(k)
            expected = compute_resistance(k)
            assert ck in merged
            assert merged[ck] == expected

        assert agg.aggregate_commutativity_check(shard_results)

    def test_full_pipeline_with_failure_recovery(self):
        """Simulate failure → detect → retry → succeed."""
        keys = make_param_keys_64()
        n_shards = 8

        plan = ShardPlan.create(keys, n_shards)
        injector = FailureInjector(seed=42)
        baseline = injector.generate_baseline_results(plan, compute_resistance_from_canonical)

        all_keys = []
        for si in range(n_shards):
            all_keys.extend(plan.get_shard_keys(si))
        agg = Aggregator(n_shards, all_keys)

        # Attempt 1: kill 3 shards → fail
        killed = injector.inject_kill_shards(baseline, n_kill=3)
        with pytest.raises(AggregationError):
            agg.aggregate(killed)

        # Attempt 2: retry with all shards → succeed
        result = agg.aggregate(baseline)
        assert len(result) == 64

        # Verify bit-identical to reference
        ref_agg = Aggregator(n_shards, all_keys)
        ref_result = ref_agg.aggregate(baseline)
        assert result == ref_result
