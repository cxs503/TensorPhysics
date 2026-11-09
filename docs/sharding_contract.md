# Sharding Aggregation Contract

## Overview

The `tensorphysics.sharding` module provides a deterministic aggregation contract
for cluster-sharded parameter scans. It ensures:

1. **Commutativity**: Same results regardless of shard count (n=1/2/4/8)
2. **Fail-closed**: Missing/killed shards cause aggregation to raise (no silent data loss)
3. **Conflict detection**: Duplicate parameter keys across shards raise immediately
4. **Manifest integrity**: Tampering with git SHA, result hashes, or boundaries is detected
5. **Float precision**: Naive vs Kahan summation with theoretical error bounds
6. **Coverage**: All parameter points covered exactly once (no gaps, no overlaps)

## Core Components

### ShardPlan

Deterministic mapping of parameter keys to shards using SHA-256 hashing:

```python
plan = ShardPlan(n=8, param_keys=tuple(sorted(keys)))
partition = plan.partition()  # {0: ("k1", "k3"), 1: ("k2",), ...}
```

### Manifest

Per-shard metadata with integrity hashes:

```python
manifest = Manifest(
    shard_i=0, shard_n=8,
    param_keys=("k1", "k3"),
    tool_version="0.1.0",
    git_sha="abc123...",
    result_hashes={"k1": "sha256...", "k3": "sha256..."},
)
```

### Aggregator

Order-independent merge with conflict detection:

```python
result = aggregate(shard_results, manifests, expected_n=8, expected_keys=set(keys))
# Raises: BoundaryViolationError, ShardConflictError, IncompleteShardError
```

### ShardContract

High-level workflow wrapper:

```python
contract = ShardContract.from_param_keys(keys, n_shards=8)
results_i, manifest_i = contract.execute_shard(i, compute_fn)
aggregated = contract.aggregate_all(all_results, all_manifests)
```

### Float Audit

Kahan-compensated summation with theoretical error bounds:

```python
audit = audit_float_summation(values, n_shards=8)
# Returns: naive_1, naive_n, kahan_1, kahan_n, rel_diff_naive, rel_diff_kahan,
#          theoretical_bound, naive_within_bound, kahan_within_bound
```

### run_cluster Integration

```python
cmd_tpl = build_shard_command_template("worker.py", n_shards=8)
# Use with run_cluster(action="shard", cmd_tpl=cmd_tpl, shards=8)

results, manifests = collect_shard_outputs(output_dir, n_shards=8)
aggregated = aggregate(results, manifests, expected_n=8)
```

## Acceptance Criteria

| Criterion | Description | Threshold |
|---|---|---|
| C1 | Commutativity | Bitwise identical across n=1/2/4/8 |
| C2 | Float bound | Naive ≤1e-12, Kahan ≤1e-15 |
| C3 | Fail-closed | Kill 3/8 → raise; retry matches |
| C4 | Conflict detection | Duplicate key → raise |
| C5 | Manifest integrity | 3 tamper variants detected |
| C6 | No overlap | 100 trials, 0 overlaps |
| C7 | Regression | shard=1 matches direct run |

## Testing

```bash
pytest tests/test_sharding.py -v  # 51 tests, ~1s
```
