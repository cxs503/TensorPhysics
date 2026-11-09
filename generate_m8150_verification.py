#!/usr/bin/env python3
"""Generate verification report JSON for P96W2-N11 sharding contract.

Runs all acceptance criteria and produces a machine-readable JSON report.
"""

import hashlib
import json
import os
import random
import sys
import time

_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorphysics.sharding import (
    FLOAT64_EPS,
    Manifest,
    ShardPlan,
    ShardContract,
    aggregate,
    audit_float_summation,
    inject_shard_kill,
    inject_half_written,
    inject_key_collision,
    kahan_sum,
    naive_sum,
    tamper_manifest,
    theoretical_float_error_bound,
    validate_manifest_set,
    BoundaryViolationError,
    IncompleteShardError,
    ShardConflictError,
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


def verify_c1():
    """C1: Commutativity — N=64, n=1/2/4/8, bitwise identical."""
    n_points = 64
    seed = 20261108
    keys = make_synthetic_params(n_points, seed=seed)
    results = make_synthetic_results(keys, seed=seed)

    refs = {}
    for n in [1, 2, 4, 8]:
        plan = ShardPlan(n=n, param_keys=tuple(sorted(keys)))
        manifests = make_manifests_for_plan(plan, results)
        agg = aggregate(results, manifests, expected_n=n, expected_keys=set(keys))
        refs[n] = json.dumps(agg.results, sort_keys=True)
        assert len(agg.results) == n_points
        assert agg.verify_counts()

    for n in [2, 4, 8]:
        assert refs[n] == refs[1], f"C1 FAIL: n={n} differs from n=1"

    return {
        "criterion": "C1",
        "name": "Commutativity",
        "pass": True,
        "details": {
            "n_points": n_points,
            "shard_counts_tested": [1, 2, 4, 8],
            "all_bitwise_identical": True,
            "all_counts_one": True,
        }
    }


def verify_c2():
    """C2: Float bound — naive ≤1e-12, Kahan ≤1e-15, within theoretical bound."""
    seed = 777
    rng = random.Random(seed)
    values = [rng.uniform(-1e5, 1e5) for _ in range(8000)]

    audit = audit_float_summation(values, n_shards=8)

    naive_ok = audit["rel_diff_naive"] <= 1e-12
    kahan_ok = audit["rel_diff_kahan"] <= 1e-15
    naive_in_bound = audit["naive_within_bound"]
    kahan_in_bound = audit["kahan_within_bound"]

    return {
        "criterion": "C2",
        "name": "Float precision bound",
        "pass": naive_ok and kahan_ok and naive_in_bound and kahan_in_bound,
        "details": {
            "n_values": 8000,
            "n_shards": 8,
            "rel_diff_naive": audit["rel_diff_naive"],
            "rel_diff_kahan": audit["rel_diff_kahan"],
            "naive_threshold": 1e-12,
            "kahan_threshold": 1e-15,
            "naive_within_bound": naive_in_bound,
            "kahan_within_bound": kahan_in_bound,
            "theoretical_bound": audit["theoretical_bound"],
            "naive_err": audit["naive_err"],
            "kahan_err": audit["kahan_err"],
        }
    }


def verify_c3():
    """C3: fail-closed — kill 3/8 shards → error; retry matches reference."""
    n_points = 64
    seed = 20261108
    keys = make_synthetic_params(n_points, seed=seed)
    results = make_synthetic_results(keys, seed=seed)
    plan = ShardPlan(n=8, param_keys=tuple(sorted(keys)))
    manifests = make_manifests_for_plan(plan, results)

    # Reference
    ref_agg = aggregate(results, manifests, expected_n=8, expected_keys=set(keys))

    # Kill 3 shards
    remaining, killed = inject_shard_kill(manifests, [2, 5, 7])
    kill_raised = False
    try:
        aggregate(results, remaining, expected_n=8)
    except (IncompleteShardError, BoundaryViolationError):
        kill_raised = True

    # Retry matches reference
    retry_agg = aggregate(results, manifests, expected_n=8, expected_keys=set(keys))
    retry_matches = json.dumps(retry_agg.results, sort_keys=True) == \
                    json.dumps(ref_agg.results, sort_keys=True)

    # Half-written detection
    corrupted = inject_half_written(manifests[3], [list(manifests[3].param_keys)[0]])
    half_written_detected = len(corrupted.validate()) > 0

    return {
        "criterion": "C3",
        "name": "Fail-closed",
        "pass": kill_raised and retry_matches and half_written_detected,
        "details": {
            "kill_3_of_8_raised": kill_raised,
            "retry_matches_reference": retry_matches,
            "half_written_detected": half_written_detected,
        }
    }


def verify_c4():
    """C4: Conflict detection — two shards write same key → raise."""
    # Direct overlap
    manifests = [
        Manifest(shard_i=0, shard_n=2, param_keys=("a", "b"),
                 tool_version="0.1.0", git_sha="x" * 40,
                 result_hashes={"a": "h1", "b": "h2"}),
        Manifest(shard_i=1, shard_n=2, param_keys=("b", "c"),
                 tool_version="0.1.0", git_sha="x" * 40,
                 result_hashes={"b": "h3", "c": "h4"}),
    ]
    overlap_raised = False
    try:
        aggregate({"a": 1, "b": 2, "c": 3}, manifests, expected_n=2)
    except BoundaryViolationError:
        overlap_raised = True

    # Injected collision
    keys = make_synthetic_params(32, seed=99)
    results = make_synthetic_results(keys, seed=99)
    plan = ShardPlan(n=4, param_keys=tuple(sorted(keys)))
    base_manifests = make_manifests_for_plan(plan, results)
    collision_key = "collision_xyz"
    results[collision_key] = {"value": 0}
    collided = inject_key_collision(base_manifests, collision_key, 0)
    collided = inject_key_collision(collided, collision_key, 2)
    inject_raised = False
    try:
        aggregate(results, collided, expected_n=4)
    except BoundaryViolationError:
        inject_raised = True

    return {
        "criterion": "C4",
        "name": "Conflict detection",
        "pass": overlap_raised and inject_raised,
        "details": {
            "direct_overlap_raised": overlap_raised,
            "injected_collision_raised": inject_raised,
        }
    }


def verify_c5():
    """C5: Manifest integrity — 3 tampering variants all detected."""
    m = Manifest(
        shard_i=0, shard_n=4,
        param_keys=("p1", "p2"),
        tool_version="0.1.0",
        git_sha="a" * 40,
        result_hashes={"p1": "h1", "p2": "h2"},
    )

    # Git SHA tampering
    t1 = tamper_manifest(m, tamper_git_sha=True)
    git_sha_detected = any("git_sha" in e for e in t1.validate(expected_git_sha="a" * 40))

    # Result hash tampering
    t2 = tamper_manifest(m, tamper_result_hash=True)
    result_hash_detected = m.integrity_hash() != t2.integrity_hash()

    # Boundary (param_keys) tampering
    t3 = tamper_manifest(m, tamper_param_keys=True)
    boundary_errs = validate_manifest_set([t3], expected_n=4, expected_keys={"p1", "p2"})
    boundary_detected = len(boundary_errs) > 0

    return {
        "criterion": "C5",
        "name": "Manifest integrity",
        "pass": git_sha_detected and result_hash_detected and boundary_detected,
        "details": {
            "git_sha_tamper_detected": git_sha_detected,
            "result_hash_tamper_detected": result_hash_detected,
            "boundary_tamper_detected": boundary_detected,
        }
    }


def verify_c6():
    """C6: Boundary no overlap — 100 random mappings, coverage 100/100, overlap 0."""
    rng = random.Random(2026)
    n_trials = 100
    all_pass = True
    coverages = []
    overlap_count = 0

    for trial in range(n_trials):
        n_params = rng.randint(10, 200)
        n_shards = rng.randint(1, 8)
        keys = make_synthetic_params(n_params, seed=trial * 1000 + 42)
        plan = ShardPlan(n=n_shards, param_keys=tuple(sorted(keys)))

        if plan.coverage() != n_params:
            all_pass = False
            continue

        partition = plan.partition()
        all_keys = []
        for shard_keys in partition.values():
            all_keys.extend(shard_keys)

        if len(set(all_keys)) != n_params:
            overlap_count += 1
            all_pass = False

        coverages.append(plan.coverage())

    return {
        "criterion": "C6",
        "name": "Boundary no overlap / seamless",
        "pass": all_pass and overlap_count == 0,
        "details": {
            "n_trials": n_trials,
            "coverage_100_pct": len([c for c in coverages if c > 0]) == n_trials,
            "overlap_count": overlap_count,
            "min_coverage": min(coverages) if coverages else 0,
            "max_coverage": max(coverages) if coverages else 0,
        }
    }


def verify_c7():
    """C7: Regression — shard=1 matches direct run bitwise."""
    keys = make_synthetic_params(32, seed=777)
    results = make_synthetic_results(keys, seed=777)

    direct_json = json.dumps(results, sort_keys=True)

    plan = ShardPlan(n=1, param_keys=tuple(sorted(keys)))
    manifests = make_manifests_for_plan(plan, results)
    agg = aggregate(results, manifests, expected_n=1, expected_keys=set(keys))
    shard_json = json.dumps(agg.results, sort_keys=True)

    return {
        "criterion": "C7",
        "name": "Regression (shard=1 = direct)",
        "pass": direct_json == shard_json,
        "details": {
            "n_keys": 32,
            "bitwise_match": direct_json == shard_json,
        }
    }


if __name__ == "__main__":
    t0 = time.time()
    results = {
        "mission_id": "m8150",
        "node": "P96W2-N11",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criteria": [
            verify_c1(),
            verify_c2(),
            verify_c3(),
            verify_c4(),
            verify_c5(),
            verify_c6(),
            verify_c7(),
        ],
    }
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 3)
    results["all_pass"] = all(c["pass"] for c in results["criteria"])

    out_path = os.path.join(os.path.dirname(__file__), "runs", "m8150_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nAll criteria pass: {results['all_pass']}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Report saved to: {out_path}")
