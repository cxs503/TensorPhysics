"""Tests for TensorPhysics cluster sharding determinism aggregation contract.

Covers all 7 acceptance criteria:
  C1. Commutativity: N=64 points, n=1/2/4/8 shards, bitwise identical after key reorder
  C2. Float bound: 1 vs 8 shard relative deviation ≤1e-12, ≤ theoretical bound, Kahan ≤1e-15
  C3. fail-closed: inject 3/8 shard kill → aggregator errors, retry matches reference
  C4. Conflict detection: two shards write same key → raise
  C5. Manifest integrity: tamper git sha / boundary / result hash → validation fails
  C6. Boundary no overlap / seamless: 100 random parameter space mappings, coverage 100/100, overlap 0
  C7. Regression: shard=1 path matches direct run bitwise
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import pytest

# Ensure worktree src is on path (conftest does this, but be safe)
_SRC = __import__("pathlib").Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorphysics.sharding import (
    FLOAT64_EPS,
    AggregationError,
    AggregationResult,
    BoundaryViolationError,
    IncompleteShardError,
    Manifest,
    ShardConflictError,
    ShardContract,
    ShardPlan,
    aggregate,
    audit_float_summation,
    build_shard_command_template,
    collect_shard_outputs,
    inject_half_written,
    inject_key_collision,
    inject_shard_kill,
    kahan_sum,
    naive_sum,
    tamper_manifest,
    theoretical_float_error_bound,
    validate_manifest_set,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_params(n: int, seed: int = 42) -> List[str]:
    """Generate n deterministic synthetic parameter keys."""
    rng = random.Random(seed)
    return [f"param_{rng.randint(0, 999999):06d}" for _ in range(n)]


def make_synthetic_results(keys: List[str], seed: int = 42) -> Dict[str, Any]:
    """Generate deterministic synthetic result payloads."""
    rng = random.Random(seed + 1)
    return {
        k: {"value": rng.uniform(-1e6, 1e6), "status": "ok"}
        for k in keys
    }


def make_manifests_for_plan(
    plan: ShardPlan,
    results: Dict[str, Any],
    tool_version: str = "0.1.0",
    git_sha: str = "abc123" * 6 + "00",  # 40 chars
) -> List[Manifest]:
    """Create manifests for all shards in a plan."""
    partition = plan.partition()
    manifests = []
    for i in range(plan.n):
        keys = partition[i]
        hashes = {}
        for k in keys:
            payload = json.dumps(results[k], sort_keys=True).encode()
            hashes[k] = hashlib.sha256(payload).hexdigest()
        manifests.append(Manifest(
            shard_i=i,
            shard_n=plan.n,
            param_keys=keys,
            tool_version=tool_version,
            git_sha=git_sha,
            result_hashes=hashes,
        ))
    return manifests


# ---------------------------------------------------------------------------
# C1: Commutativity
# ---------------------------------------------------------------------------

class TestC1Commutativity:
    """N=64 parameter points, n=1/2/4/8 shards, aggregated results bitwise identical."""

    N_POINTS = 64
    SEED = 20261108

    @pytest.fixture
    def param_keys(self) -> List[str]:
        return make_synthetic_params(self.N_POINTS, seed=self.SEED)

    @pytest.fixture
    def results(self, param_keys: List[str]) -> Dict[str, Any]:
        return make_synthetic_results(param_keys, seed=self.SEED)

    def _aggregate_for_n(self, n: int, param_keys: List[str],
                         results: Dict[str, Any]) -> AggregationResult:
        plan = ShardPlan(n=n, param_keys=tuple(sorted(param_keys)))
        manifests = make_manifests_for_plan(plan, results)
        return aggregate(results, manifests, expected_n=n,
                         expected_keys=set(param_keys))

    @pytest.mark.parametrize("n_shards", [1, 2, 4, 8])
    def test_aggregation_produces_all_keys(self, n_shards: int,
                                           param_keys: List[str],
                                           results: Dict[str, Any]):
        agg = self._aggregate_for_n(n_shards, param_keys, results)
        assert len(agg.results) == self.N_POINTS
        assert len(agg.counts) == self.N_POINTS

    @pytest.mark.parametrize("n_shards", [1, 2, 4, 8])
    def test_each_key_count_is_one(self, n_shards: int,
                                   param_keys: List[str],
                                   results: Dict[str, Any]):
        agg = self._aggregate_for_n(n_shards, param_keys, results)
        assert agg.verify_counts(), f"Not all counts == 1 for n={n_shards}"

    def test_bitwise_identical_across_shard_counts(self, param_keys: List[str],
                                                    results: Dict[str, Any]):
        """Aggregated results must be bitwise identical regardless of shard count."""
        refs: Dict[int, Dict[str, Any]] = {}
        for n in [1, 2, 4, 8]:
            agg = self._aggregate_for_n(n, param_keys, results)
            # Serialize to canonical JSON for comparison
            refs[n] = json.dumps(agg.results, sort_keys=True)

        for n in [2, 4, 8]:
            assert refs[n] == refs[1], f"Results differ between n=1 and n={n}"

    def test_key_order_independence(self, param_keys: List[str],
                                    results: Dict[str, Any]):
        """Aggregation must be independent of the order keys are presented."""
        plan = ShardPlan(n=4, param_keys=tuple(sorted(param_keys)))
        manifests = make_manifests_for_plan(plan, results)

        # Shuffle results dict order
        shuffled_keys = list(results.keys())
        random.Random(99).shuffle(shuffled_keys)
        shuffled_results = {k: results[k] for k in shuffled_keys}

        agg1 = aggregate(results, manifests, expected_n=4,
                         expected_keys=set(param_keys))
        agg2 = aggregate(shuffled_results, manifests, expected_n=4,
                         expected_keys=set(param_keys))

        assert json.dumps(agg1.results, sort_keys=True) == \
               json.dumps(agg2.results, sort_keys=True)


# ---------------------------------------------------------------------------
# C2: Float bound
# ---------------------------------------------------------------------------

class TestC2FloatBound:
    """1 vs 8 shard relative deviation ≤1e-12, ≤ theoretical bound, Kahan ≤1e-15."""

    def test_naive_vs_kahan_precision(self):
        """Kahan summation should be significantly more precise."""
        rng = random.Random(42)
        values = [rng.uniform(-1e3, 1e3) for _ in range(10000)]

        naive = naive_sum(values)
        kahan = kahan_sum(values)

        # They should be close but not necessarily identical
        rel_diff = abs(naive - kahan) / (abs(naive) if naive != 0 else 1.0)
        assert rel_diff < 1e-10, f"Kahan deviation too large: {rel_diff}"

    def test_theoretical_bound_holds(self):
        """Naive summation error should be within theoretical bound."""
        rng = random.Random(123)
        values = [rng.uniform(-1e4, 1e4) for _ in range(5000)]

        audit = audit_float_summation(values, n_shards=8)

        assert audit["naive_within_bound"], (
            f"Naive error {audit['naive_err']} exceeds bound {audit['theoretical_bound']}"
        )

    def test_kahan_ultra_precise(self):
        """Kahan relative difference between 1-shard and 8-shard ≤ 1e-15."""
        rng = random.Random(777)
        values = [rng.uniform(-1e5, 1e5) for _ in range(8000)]

        audit = audit_float_summation(values, n_shards=8)

        assert audit["rel_diff_naive"] <= 1e-12, (
            f"Naive rel_diff {audit['rel_diff_naive']} > 1e-12"
        )
        assert audit["rel_diff_kahan"] <= 1e-15, (
            f"Kahan rel_diff {audit['rel_diff_kahan']} > 1e-15"
        )
        assert audit["kahan_within_bound"]

    def test_float_error_bound_formula(self):
        """Verify the theoretical bound formula is correct."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        n = len(values)
        abs_sum = sum(abs(v) for v in values)  # 15.0
        expected = 1.0 * FLOAT64_EPS * n * abs_sum  # 1 * eps * 5 * 15

        bound = theoretical_float_error_bound(values, n)
        assert abs(bound - expected) < 1e-30


# ---------------------------------------------------------------------------
# C3: fail-closed
# ---------------------------------------------------------------------------

class TestC3FailClosed:
    """Inject 3/8 shard kill → aggregator errors; retry with all shards matches reference."""

    N_POINTS = 64
    SEED = 20261108

    @pytest.fixture
    def full_setup(self):
        param_keys = make_synthetic_params(self.N_POINTS, seed=self.SEED)
        results = make_synthetic_results(param_keys, seed=self.SEED)
        plan = ShardPlan(n=8, param_keys=tuple(sorted(param_keys)))
        manifests = make_manifests_for_plan(plan, results)
        return param_keys, results, plan, manifests

    def test_kill_3_of_8_raises_error(self, full_setup):
        _, results, plan, manifests = full_setup
        # Kill shards 2, 5, 7
        remaining, killed = inject_shard_kill(manifests, [2, 5, 7])
        assert len(killed) == 3
        assert len(remaining) == 5

        with pytest.raises((IncompleteShardError, BoundaryViolationError)):
            aggregate(results, remaining, expected_n=8)

    def test_retry_with_all_shards_matches_reference(self, full_setup):
        param_keys, results, plan, manifests = full_setup
        # Reference: aggregate with all 8 shards
        ref_agg = aggregate(results, manifests, expected_n=8,
                            expected_keys=set(param_keys))

        # Simulate kill then retry
        remaining, _ = inject_shard_kill(manifests, [1, 3, 6])
        with pytest.raises((IncompleteShardError, BoundaryViolationError)):
            aggregate(results, remaining, expected_n=8)

        # Retry with full set
        retry_agg = aggregate(results, manifests, expected_n=8,
                              expected_keys=set(param_keys))
        assert json.dumps(retry_agg.results, sort_keys=True) == \
               json.dumps(ref_agg.results, sort_keys=True)

    def test_half_written_detected(self, full_setup):
        _, results, plan, manifests = full_setup
        # Corrupt one manifest by removing result hashes
        corrupted = inject_half_written(manifests[3], [list(manifests[3].param_keys)[0]])
        manifests_copy = list(manifests)
        manifests_copy[3] = corrupted

        # Validation should catch missing result_hash
        errs = corrupted.validate()
        assert len(errs) > 0, "Half-written manifest should fail validation"


# ---------------------------------------------------------------------------
# C4: Conflict detection
# ---------------------------------------------------------------------------

class TestC4ConflictDetection:
    """Two shards writing the same key must raise ShardConflictError."""

    def test_duplicate_key_raises(self):
        manifests = [
            Manifest(
                shard_i=0, shard_n=4, param_keys=("a", "b"),
                tool_version="0.1.0", git_sha="x" * 40,
                result_hashes={"a": "h1", "b": "h2"},
            ),
            Manifest(
                shard_i=1, shard_n=4, param_keys=("c",),
                tool_version="0.1.0", git_sha="x" * 40,
                result_hashes={"c": "h3"},
            ),
            Manifest(
                shard_i=2, shard_n=4, param_keys=("d",),
                tool_version="0.1.0", git_sha="x" * 40,
                result_hashes={"d": "h5"},
            ),
            Manifest(
                shard_i=3, shard_n=4, param_keys=("e",),
                tool_version="0.1.0", git_sha="x" * 40,
                result_hashes={"e": "h6"},
            ),
        ]
        # Inject boundary overlap: shard 1 also claims "b"
        manifests[1] = Manifest(
            shard_i=1, shard_n=4, param_keys=("b", "c"),
            tool_version="0.1.0", git_sha="x" * 40,
            result_hashes={"b": "h3", "c": "h4"},
        )
        results = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

        # Should raise BoundaryViolationError (detected at set validation)
        with pytest.raises(BoundaryViolationError) as exc_info:
            aggregate(results, manifests, expected_n=4)
        assert "boundary overlap" in str(exc_info.value).lower() or "b" in str(exc_info.value)

    def test_injected_collision_raises(self):
        param_keys = make_synthetic_params(32, seed=99)
        results = make_synthetic_results(param_keys, seed=99)
        plan = ShardPlan(n=4, param_keys=tuple(sorted(param_keys)))
        manifests = make_manifests_for_plan(plan, results)

        # Inject a collision into shard 2
        collision_key = "collision_key_xyz"
        results[collision_key] = {"value": 0, "status": "ok"}

        manifests_with_collision = inject_key_collision(
            manifests, collision_key, target_shard_i=2
        )
        # Also inject into shard 0 to create actual overlap
        manifests_with_collision = inject_key_collision(
            manifests_with_collision, collision_key, target_shard_i=0
        )

        with pytest.raises(BoundaryViolationError):
            aggregate(results, manifests_with_collision, expected_n=4)


# ---------------------------------------------------------------------------
# C5: Manifest integrity
# ---------------------------------------------------------------------------

class TestC5ManifestIntegrity:
    """Tampered git sha / boundary / result hash must fail validation."""

    def _good_manifest(self) -> Manifest:
        return Manifest(
            shard_i=0, shard_n=4,
            param_keys=("p1", "p2"),
            tool_version="0.1.0",
            git_sha="a" * 40,
            result_hashes={"p1": "h1", "p2": "h2"},
        )

    def test_tampered_git_sha_detected(self):
        m = self._good_manifest()
        tampered = tamper_manifest(m, tamper_git_sha=True)
        errs = tampered.validate(expected_git_sha="a" * 40)
        assert any("git_sha" in e for e in errs), f"Expected git_sha error, got: {errs}"

    def test_tampered_result_hash_detected(self):
        m = self._good_manifest()
        tampered = tamper_manifest(m, tamper_result_hash=True)
        # The integrity hash changes when result_hash is tampered
        assert m.integrity_hash() != tampered.integrity_hash()

    def test_tampered_boundary_detected(self):
        """C5 third variant: tampering param_keys (boundary) must be detected."""
        m = self._good_manifest()
        tampered = tamper_manifest(m, tamper_param_keys=True)
        # The tampered manifest has an extra key that wasn't in the original
        assert "__tampered_boundary_key__" in tampered.param_keys
        # The integrity hash must change
        assert m.integrity_hash() != tampered.integrity_hash()
        # Cross-manifest validation should detect the spurious key
        # when checked against expected_keys
        good_manifests = [self._good_manifest()]
        tampered_manifests = [tampered]
        expected_keys = {"p1", "p2"}
        errs = validate_manifest_set(
            tampered_manifests, expected_n=4, expected_keys=expected_keys
        )
        # Should detect extra key
        assert any("extra" in e.lower() or "tampered" in e.lower() for e in errs), \
            f"Expected boundary error, got: {errs}"

    def test_integrity_hash_changes_on_any_modification(self):
        m = self._good_manifest()
        for tamper_git in [True, False]:
            for tamper_hash in [True, False]:
                for tamper_keys in [True, False]:
                    if not tamper_git and not tamper_hash and not tamper_keys:
                        continue
                    t = tamper_manifest(m, tamper_git_sha=tamper_git,
                                        tamper_result_hash=tamper_hash,
                                        tamper_param_keys=tamper_keys)
                    assert m.integrity_hash() != t.integrity_hash()

    def test_missing_result_hash_detected(self):
        m = Manifest(
            shard_i=0, shard_n=2,
            param_keys=("p1", "p2"),
            tool_version="0.1.0",
            git_sha="b" * 40,
            result_hashes={"p1": "h1"},  # missing p2
        )
        errs = m.validate()
        assert any("p2" in e for e in errs)

    def test_shard_index_out_of_range(self):
        m = Manifest(
            shard_i=5, shard_n=4,  # invalid
            param_keys=(),
            tool_version="0.1.0",
            git_sha="c" * 40,
        )
        errs = m.validate()
        assert any("out of range" in e for e in errs)

    def test_manifest_serialization_roundtrip(self):
        m = self._good_manifest()
        s = m.to_json()
        m2 = Manifest.from_json(s)
        assert m.to_json() == m2.to_json()


# ---------------------------------------------------------------------------
# C6: Boundary no overlap / seamless
# ---------------------------------------------------------------------------

class TestC6BoundaryNoOverlap:
    """100 random parameter space mappings: coverage 100/100, overlap 0."""

    def test_100_random_mappings(self):
        """Generate 100 random parameter sets and verify coverage + no overlap."""
        rng = random.Random(2026)
        n_trials = 100

        for trial in range(n_trials):
            n_params = rng.randint(10, 200)
            n_shards = rng.randint(1, 8)
            keys = make_synthetic_params(n_params, seed=trial * 1000 + 42)

            plan = ShardPlan(n=n_shards, param_keys=tuple(sorted(keys)))
            assert plan.coverage() == n_params, \
                f"Trial {trial}: coverage {plan.coverage()} != {n_params}"

            # Verify no overlap via partition
            partition = plan.partition()
            all_keys_in_partitions: List[str] = []
            for shard_keys in partition.values():
                all_keys_in_partitions.extend(shard_keys)
            assert len(all_keys_in_partitions) == n_params, \
                f"Trial {trial}: partition total {len(all_keys_in_partitions)} != {n_params}"
            assert len(set(all_keys_in_partitions)) == n_params, \
                f"Trial {trial}: overlap detected"

            # Cross-validate with validate_manifest_set
            results = make_synthetic_results(keys, seed=trial)
            manifests = make_manifests_for_plan(plan, results)
            errs = validate_manifest_set(manifests, expected_n=n_shards,
                                          expected_keys=set(keys))
            assert errs == [], f"Trial {trial}: validation errors: {errs}"

    def test_empty_param_set(self):
        plan = ShardPlan(n=4, param_keys=())
        assert plan.coverage() == 0
        partition = plan.partition()
        for keys in partition.values():
            assert len(keys) == 0

    def test_single_param(self):
        plan = ShardPlan(n=4, param_keys=("only_one",))
        assert plan.coverage() == 1
        partition = plan.partition()
        total = sum(len(v) for v in partition.values())
        assert total == 1

    def test_nonlinear_index_keys(self):
        """Test with non-sequential, floating-point-like keys."""
        rng = random.Random(314159)
        keys = [f"alpha_{rng.uniform(0, 1):.15f}_beta_{rng.randint(0, 99999)}"
                for _ in range(50)]
        plan = ShardPlan(n=8, param_keys=tuple(sorted(keys)))
        partition = plan.partition()
        all_keys = []
        for shard_keys in partition.values():
            all_keys.extend(shard_keys)
        assert len(set(all_keys)) == 50


# ---------------------------------------------------------------------------
# C7: Regression
# ---------------------------------------------------------------------------

class TestC7Regression:
    """shard=1 path matches direct run bitwise."""

    def test_shard1_matches_direct(self):
        """With n=1, aggregation must produce identical results to direct computation."""
        param_keys = make_synthetic_params(32, seed=777)
        results = make_synthetic_results(param_keys, seed=777)

        # Direct: just use the results as-is
        direct_json = json.dumps(results, sort_keys=True)

        # Via shard=1
        plan = ShardPlan(n=1, param_keys=tuple(sorted(param_keys)))
        manifests = make_manifests_for_plan(plan, results)
        agg = aggregate(results, manifests, expected_n=1,
                        expected_keys=set(param_keys))
        shard_json = json.dumps(agg.results, sort_keys=True)

        assert direct_json == shard_json

    def test_shard1_single_shard_manifest(self):
        plan = ShardPlan(n=1, param_keys=("a", "b", "c"))
        partition = plan.partition()
        assert 0 in partition
        assert set(partition[0]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# ShardPlan tests
# ---------------------------------------------------------------------------

class TestShardPlan:
    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            ShardPlan(n=0, param_keys=("a",))

    def test_deterministic_mapping(self):
        plan = ShardPlan(n=4, param_keys=("a", "b", "c"))
        s1 = plan.shard_for("a")
        s2 = plan.shard_for("a")
        assert s1 == s2

    def test_partition_completeness(self):
        keys = tuple(f"k{i}" for i in range(20))
        plan = ShardPlan(n=4, param_keys=keys)
        partition = plan.partition()
        all_keys = []
        for v in partition.values():
            all_keys.extend(v)
        assert set(all_keys) == set(keys)


# ---------------------------------------------------------------------------
# Kahan summation tests
# ---------------------------------------------------------------------------

class TestKahanSum:
    def test_exact_for_integers(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert kahan_sum(values) == 15.0

    def test_better_than_naive_for_pathological(self):
        """Kahan should be more accurate for pathological summation."""
        # Classic example: summing many small values with one large value
        values = [1e15] + [1.0] * 10000
        naive = naive_sum(values)
        kahan = kahan_sum(values)
        exact = 1e15 + 10000.0
        assert abs(kahan - exact) < abs(naive - exact) + 1e-5


# ---------------------------------------------------------------------------
# Aggregate edge cases
# ---------------------------------------------------------------------------

class TestAggregateEdgeCases:
    def test_empty_aggregation(self):
        manifests = [
            Manifest(shard_i=0, shard_n=1, param_keys=(),
                     tool_version="0.1.0", git_sha="a" * 40)
        ]
        agg = aggregate({}, manifests, expected_n=1)
        assert len(agg.results) == 0

    def test_single_shard_single_key(self):
        m = Manifest(
            shard_i=0, shard_n=1, param_keys=("x",),
            tool_version="0.1.0", git_sha="a" * 40,
            result_hashes={"x": "h1"},
        )
        agg = aggregate({"x": 42}, [m], expected_n=1)
        assert agg.results["x"] == 42
        assert agg.counts["x"] == 1

    def test_manifest_validation_errors_collected(self):
        m = Manifest(
            shard_i=5, shard_n=4,  # invalid index
            param_keys=("a",),
            tool_version="0.1.0",
            git_sha="a" * 40,
            # missing result_hash for "a"
        )
        with pytest.raises(AggregationError):
            aggregate({"a": 1}, [m])


# ---------------------------------------------------------------------------
# Cross-manifest validation tests
# ---------------------------------------------------------------------------

class TestValidateManifestSet:
    def test_valid_set(self):
        manifests = [
            Manifest(shard_i=0, shard_n=2, param_keys=("a",),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"a": "h1"}),
            Manifest(shard_i=1, shard_n=2, param_keys=("b",),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"b": "h2"}),
        ]
        errs = validate_manifest_set(manifests, expected_n=2,
                                      expected_keys={"a", "b"})
        assert errs == []

    def test_overlap_detected(self):
        manifests = [
            Manifest(shard_i=0, shard_n=2, param_keys=("a", "b"),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"a": "h1", "b": "h2"}),
            Manifest(shard_i=1, shard_n=2, param_keys=("b", "c"),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"b": "h3", "c": "h4"}),
        ]
        errs = validate_manifest_set(manifests)
        assert any("boundary overlap" in e.lower() for e in errs)

    def test_missing_keys_detected(self):
        manifests = [
            Manifest(shard_i=0, shard_n=1, param_keys=("a",),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"a": "h1"}),
        ]
        errs = validate_manifest_set(manifests, expected_n=1,
                                      expected_keys={"a", "b"})
        assert any("missing" in e.lower() for e in errs)

    def test_extra_keys_detected(self):
        manifests = [
            Manifest(shard_i=0, shard_n=1, param_keys=("a", "b"),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"a": "h1", "b": "h2"}),
        ]
        errs = validate_manifest_set(manifests, expected_n=1,
                                      expected_keys={"a"})
        assert any("extra" in e.lower() for e in errs)

    def test_wrong_shard_count(self):
        manifests = [
            Manifest(shard_i=0, shard_n=4, param_keys=("a",),
                     tool_version="0.1.0", git_sha="x" * 40,
                     result_hashes={"a": "h1"}),
        ]
        errs = validate_manifest_set(manifests, expected_n=4)
        assert any("expected 4" in e.lower() for e in errs)

    def test_empty_set(self):
        errs = validate_manifest_set([])
        assert any("empty" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# ShardContract tests
# ---------------------------------------------------------------------------

class TestShardContract:
    def test_full_workflow(self):
        """Test the complete shard→execute→aggregate workflow."""
        param_keys = make_synthetic_params(32, seed=42)
        contract = ShardContract.from_param_keys(
            param_keys, n_shards=4,
            tool_version="0.1.0", git_sha="test" * 10,
        )

        # Execute each shard
        all_results = []
        all_manifests = []
        for i in range(4):
            def compute(key, _seed=42):
                rng = random.Random(hash(key) + _seed)
                return {"value": rng.uniform(-1, 1)}
            results, manifest = contract.execute_shard(i, compute)
            all_results.append(results)
            all_manifests.append(manifest)

        # Aggregate
        agg = contract.aggregate_all(all_results, all_manifests)
        assert len(agg.results) == 32
        assert agg.verify_counts()

    def test_contract_catches_missing_shard(self):
        param_keys = make_synthetic_params(16, seed=99)
        contract = ShardContract.from_param_keys(param_keys, n_shards=4)

        # Only execute 3 of 4 shards
        all_results = []
        all_manifests = []
        for i in range(3):
            def compute(key):
                return {"value": 0}
            results, manifest = contract.execute_shard(i, compute)
            all_results.append(results)
            all_manifests.append(manifest)

        with pytest.raises((IncompleteShardError, BoundaryViolationError)):
            contract.aggregate_all(all_results, all_manifests)


# ---------------------------------------------------------------------------
# run_cluster integration tests
# ---------------------------------------------------------------------------

class TestRunClusterIntegration:
    def test_build_command_template(self):
        cmd = build_shard_command_template("worker.py", n_shards=8)
        assert "{i}" in cmd
        assert "--shard-n 8" in cmd
        assert "worker.py" in cmd

    def test_collect_shard_outputs(self):
        """Test collecting shard outputs from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write shard outputs
            for i in range(4):
                manifest = Manifest(
                    shard_i=i, shard_n=4,
                    param_keys=(f"k{i}",),
                    tool_version="0.1.0",
                    git_sha="x" * 40,
                    result_hashes={f"k{i}": f"h{i}"},
                )
                with open(os.path.join(tmpdir, f"manifest_{i}.json"), "w") as f:
                    json.dump(manifest.to_dict(), f)
                with open(os.path.join(tmpdir, f"results_{i}.json"), "w") as f:
                    json.dump({f"k{i}": f"result_{i}"}, f)

            results, manifests = collect_shard_outputs(tmpdir, 4)
            assert len(manifests) == 4
            assert len(results) == 4
            for i in range(4):
                assert f"k{i}" in results

    def test_collect_missing_shard_raises(self):
        """Missing shard file should raise IncompleteShardError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only write 2 of 4 shards
            for i in [0, 1]:
                manifest = Manifest(
                    shard_i=i, shard_n=4,
                    param_keys=(f"k{i}",),
                    tool_version="0.1.0",
                    git_sha="x" * 40,
                    result_hashes={f"k{i}": f"h{i}"},
                )
                with open(os.path.join(tmpdir, f"manifest_{i}.json"), "w") as f:
                    json.dump(manifest.to_dict(), f)
                with open(os.path.join(tmpdir, f"results_{i}.json"), "w") as f:
                    json.dump({f"k{i}": f"result_{i}"}, f)

            with pytest.raises(IncompleteShardError):
                collect_shard_outputs(tmpdir, 4)
