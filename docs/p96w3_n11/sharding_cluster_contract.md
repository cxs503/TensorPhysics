# Cluster shard channel — deterministic aggregation contract

**Module:** `src/tensorphysics/sharding_cluster.py`
**Layer:** sits on top of `tensorphysics.sharding` (`ShardPlan` / `Manifest` /
`Aggregator` / `FloatAudit`), which defines the in-memory aggregation contract.
**Status:** new capability for P96W3-N11 (TensorPhysics, cluster execution face).

---

## 1. Problem

A parameter scan launched as `--shards N` must produce **one** result set,
independent of `N`, and must **never** produce a plausible-looking subset when a
shard dies. Before this module the repository had a verified in-memory
aggregator (`sharding.py`) but no *channel*: no `{i}/{n}` command rendering, no
on-disk shard artifact format, no fail-closed fetch, and no calibrated float
bound on the sums that cross shard boundaries. Those are exactly the places
where silent corruption enters distributed runs: a half-flushed JSON file, a
shard that ran an older checkout, a duplicate key that "last write wins".

## 2. Contract

### 2.1 Command rendering

```python
render_shard_cmd(cmd_tpl, shard_index, total_shards,
                 one_based=True, plan_path=..., stage_dir=...) -> str
```

* `cmd_tpl` **must** contain `{i}` and `{n}`; anything else unresolved after
  rendering (`{plan}`, `{out}`, `{stage_dir}` when not supplied) raises
  `ShardChannelError`.
* Cluster convention: `{i}` is rendered 1..n (`one_based=True`, the default);
  `one_based=False` renders 0..n-1. The plan records which convention it used,
  and the CLI translates the rendered ordinal back to a 0-based index, so a
  plan can never be executed with the wrong offset.
* Out-of-range ordinals raise (`shard_index 5` for `n=4`).

### 2.2 Plan document

```python
plan = build_plan(param_keys, n_shards, cmd_tpl, tool_version=..., git_sha=...,
                  compute="pkg.module:function", mapping="balanced")
write_plan(path, plan); plan = load_plan(path)
```

Frozen, digestible JSON (`schema: tensorphysics.shard_plan/1`) containing, per
shard: index, total, **sorted canonical parameter keys**, key-set digest, and
the rendered command; plus the tool version, git SHA, compute entry point and
`plan_digest` over the whole document. `load_plan` recomputes the digest, so a
plan modified in flight is rejected.

Mapping strategies:

| `mapping` | rule | properties |
|---|---|---|
| `balanced` (default) | sort keys by `(sha256, key)`, deal round-robin | coverage, uniqueness **and** `max-min <= 1` by construction |
| `hash_mod` | `sha256 mod n` (legacy `ShardPlan`) | deterministic, sizes random (skew possible) |

Balance matters on a cluster: a skewed shard becomes the critical path, so the
channel's default mapping is size-balanced while remaining deterministic for
fixed `(keys, n)`.

### 2.3 Shard artifact (what a shard writes)

`write_artifact` / `read_artifact` / `verify_artifact`, file
`shard_{i:03d}_of_{n:03d}.json`:

```
schema, status="complete", shard_index, total_shards, tool_version, git_sha,
plan_digest, n_expected, n_written, keys_digest, results{canonical key: value},
results_hash = sha256(canonical results),
envelope_hash = sha256(canonical envelope minus envelope_hash)
```

* Atomic write (`tmp` + `fsync` + `os.replace`): a reader never observes a
  partially written file, and if one does (killed process, truncated transfer)
  `read_artifact` raises instead of parsing a prefix.
* The writer **refuses** to emit an artifact whose key set differs from the
  plan's assignment for that shard — boundary violations are caught at the
  source, not at merge time.

### 2.4 Fail-closed aggregation

```python
merged = aggregate_from_dir(stage_dir, plan)     # or `python -m ... aggregate`
```

Verifies, per shard: presence, `status`, index, total, tool version, git SHA,
`plan_digest`, `n_written == len(results)`, key-set digest == *plan assignment*
digest, `results_hash`, `envelope_hash`; then hands the verified shards to
`sharding.Aggregator`, which enforces `count == 1` per key and a gap-free shard
index set. Post-condition: `set(merged) == planned keys`.

Failure modes and their behaviour:

| failure | behaviour |
|---|---|
| shard killed / artifact missing | `ShardChannelError` (no partial dict) |
| truncated / empty file | `ShardChannelError` |
| complete JSON, `status != complete` | `ShardChannelError` |
| two shards report the same key | `ShardChannelError` |
| key moved across the segment boundary | `ShardChannelError` |
| shard ran another checkout (`git_sha`) or other `n`/`tool_version` | `ShardChannelError` |
| payload / envelope / digest tampering | `ShardChannelError` |

There is **no** last-write-wins path and no "best effort" mode by design; retry
is the only recovery (re-run exactly the missing ordinals, then aggregate
again).

### 2.5 Float contract

Sum-sensitive quantities that cross the shard boundary (total energy, KPI mean)
are aggregated per shard and then across shards. Two bounds are hardcoded and
checked against measurement:

```
naive:  |err| <= K_NAIVE * eps * n * sum|x|,  K_NAIVE = 1.0   (classical worst case, (n-1)/n < 1)
Kahan:  |err| <= K_KAHAN * eps * sum|x|,      K_KAHAN = 2.0   (Neumaier compensated variant)
```

`K` is *measured* (`FloatContract.calibrate` on a labelled corpus) **and**
hardcoded conservatively above the measurement; a disjoint validation corpus
then asserts `observed <= K`. The empirical crossover study (see report)
quantifies where each method stops meeting its bar as a function of the
conditioning ratio `R = sum|x| / |sum x|`.

## 3. Integration recipe (cluster channel)

```bash
# 1. freeze the plan locally
python -c "... build_plan(...); write_plan('.../plan.json', plan)"
# 2. one job, N shards, the channel renders {i}/{n}
#    run_cluster(cmd_tpl="python scripts/..._cluster_shard.py run-shard \
#        --plan .../plan.json --i {i} --n {n} --out .../out", shards=N,
#        stage_dir=<worktree>)
# 3. fail-closed merge
python scripts/..._cluster_shard.py aggregate --plan .../plan.json --out .../out
```

`scripts/p96w3_n11_cluster_shard.py` loads the shard/channel modules without
executing the package `__init__` (which eagerly imports numpy/scipy — ~10 s on a
cold node) and resolves relative `--plan`/`--out` against the repository root so
the same template works from any node working directory.

## 4. Applicability and limitations

* **Well-conditioned aggregates only.** The 1e-12 / 1e-15 bars are attainable
  for `R` up to the measured crossover (~1e5 naive, ~1e2 Kahan, n=64). For more
  cancelling aggregates no float64 summation meets 1e-15; use exact/compensated
  accumulation of partial sums with wider precision instead.
* **Honest shards.** The channel detects crashes, truncation, version drift,
  boundary violations and duplicate keys; it does not defend against a shard
  that fabricates self-consistent values (SHA-256 is an integrity check, not a
  signature).
* **Single writer per shard.** Two concurrent processes writing the same
  artifact path race on `os.replace`; the last one wins atomically and the
  other's keys are simply absent — which the verifier then rejects. Give each
  attempt/retry its own stage directory.
* **Retry, don't repair.** There is no repair mode; a failed shard must be
  re-executed.
* The synthetic compute (`synthetic_scan_energy`) is a float-arithmetic fixture,
  not a physics model; domain compute is injected through `plan["compute"]`.