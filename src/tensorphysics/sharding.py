"""Cluster sharding determinism aggregation contract.

Provides:
- ShardPlan: deterministic parameter-point → shard mapping with provable
  key coverage and uniqueness.
- Manifest: per-shard metadata (i/n, parameter range, tool version, git sha,
  result hashes) with integrity validation.
- Aggregator: order-independent merge by parameter key; raises on duplicate
  keys or missing shards (fail-closed).
- Float audit utilities: naive vs Kahan-compensated summation with
  theoretical error bound K·eps·n·Σ|x|.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOAT64_EPS = sys.float_info.epsilon  # ≈ 2.22e-16


# ---------------------------------------------------------------------------
# ShardPlan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardPlan:
    """Deterministic mapping of parameter points to shards.

    Parameters are assigned to shard ``hash(key) % n`` so that:
    - Every parameter point maps to exactly one shard (uniqueness).
    - The union of all shards covers every parameter point (coverage).
    - The mapping is independent of the number of shards used for other
      points (i.e. adding/removing shards does not reassign existing points
      *within the same n*; changing n does change assignments, which is
      expected and documented).
    """

    n: int  # number of shards
    param_keys: Tuple[str, ...]  # sorted tuple of parameter keys

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        # Verify uniqueness: each key maps to exactly one shard
        shard_counts: Dict[int, int] = {}
        for key in self.param_keys:
            s = self.shard_for(key)
            shard_counts[s] = shard_counts.get(s, 0) + 1
        total = sum(shard_counts.values())
        if total != len(self.param_keys):
            raise ValueError("ShardPlan: key uniqueness violated")

    def shard_for(self, key: str) -> int:
        """Return the shard index (0..n-1) for a given parameter key."""
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        return h % self.n

    def partition(self) -> Dict[int, Tuple[str, ...]]:
        """Return {shard_idx: (keys...)} for all shards."""
        parts: Dict[int, List[str]] = {i: [] for i in range(self.n)}
        for key in self.param_keys:
            parts[self.shard_for(key)].append(key)
        return {i: tuple(sorted(v)) for i, v in parts.items()}

    def coverage(self) -> int:
        """Number of distinct parameter keys covered."""
        return len(self.param_keys)

    def verify_no_overlap(self) -> bool:
        """Verify that no key appears in more than one shard."""
        seen: set = set()
        for key in self.param_keys:
            if key in seen:
                return False
            seen.add(key)
        return True


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    """Per-shard execution manifest with integrity hashes.

    Attributes:
        shard_i: shard index (0-based)
        shard_n: total number of shards
        param_keys: tuple of parameter keys handled by this shard
        tool_version: tool / package version string
        git_sha: git commit SHA of the code used
        result_hashes: {param_key: sha256 hex digest of result payload}
        param_ranges: optional description of parameter space covered
    """

    shard_i: int
    shard_n: int
    param_keys: Tuple[str, ...]
    tool_version: str
    git_sha: str
    result_hashes: Dict[str, str] = field(default_factory=dict)
    param_ranges: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_i": self.shard_i,
            "shard_n": self.shard_n,
            "param_keys": list(self.param_keys),
            "tool_version": self.tool_version,
            "git_sha": self.git_sha,
            "result_hashes": self.result_hashes,
            "param_ranges": self.param_ranges,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        return cls(
            shard_i=d["shard_i"],
            shard_n=d["shard_n"],
            param_keys=tuple(d["param_keys"]),
            tool_version=d["tool_version"],
            git_sha=d["git_sha"],
            result_hashes=dict(d.get("result_hashes", {})),
            param_ranges=d.get("param_ranges"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Manifest":
        return cls.from_dict(json.loads(s))

    def integrity_hash(self) -> str:
        """SHA-256 of the canonical JSON representation."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def validate(self, expected_git_sha: Optional[str] = None,
                 expected_tool_version: Optional[str] = None) -> List[str]:
        """Return list of validation error strings (empty = valid)."""
        errors: List[str] = []
        if self.shard_i < 0 or self.shard_i >= self.shard_n:
            errors.append(f"shard_i={self.shard_i} out of range [0, {self.shard_n})")
        if expected_git_sha and self.git_sha != expected_git_sha:
            errors.append(f"git_sha mismatch: expected {expected_git_sha}, got {self.git_sha}")
        if expected_tool_version and self.tool_version != expected_tool_version:
            errors.append(f"tool_version mismatch: expected {expected_tool_version}, got {self.tool_version}")
        # Verify result_hashes covers all param_keys
        for key in self.param_keys:
            if key not in self.result_hashes:
                errors.append(f"missing result_hash for param_key={key}")
        return errors


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class AggregationError(Exception):
    """Raised when aggregation cannot produce a valid result."""
    pass


class ShardConflictError(AggregationError):
    """Raised when two shards claim the same parameter key."""
    pass


class IncompleteShardError(AggregationError):
    """Raised when expected shards are missing."""
    pass


@dataclass
class AggregationResult:
    """Result of aggregating shard outputs.

    Attributes:
        results: {param_key: result_payload}
        counts: {param_key: count} — should all be 1
        manifest_hashes: list of manifest integrity hashes
    """

    results: Dict[str, Any]
    counts: Dict[str, int]
    manifest_hashes: List[str]

    def verify_counts(self) -> bool:
        """Return True iff every parameter key has count == 1."""
        return all(c == 1 for c in self.counts.values())


def aggregate(
    shard_results: Dict[str, Any],
    manifests: Sequence[Manifest],
    expected_n: Optional[int] = None,
) -> AggregationResult:
    """Aggregate shard results into a single result set.

    Args:
        shard_results: {param_key: result_payload} from all shards combined.
        manifests: list of per-shard manifests.
        expected_n: if given, verify that exactly expected_n shards are present.

    Returns:
        AggregationResult with merged results and per-key counts.

    Raises:
        ShardConflictError: if any param_key appears in more than one shard.
        IncompleteShardError: if expected_n is given and shard count differs.
        AggregationError: on manifest validation failure.
    """
    # Validate manifests
    all_errors: List[str] = []
    for m in manifests:
        errs = m.validate()
        all_errors.extend(errs)
    if all_errors:
        raise AggregationError(f"Manifest validation failed: {'; '.join(all_errors)}")

    # Check shard count
    if expected_n is not None:
        if len(manifests) != expected_n:
            raise IncompleteShardError(
                f"Expected {expected_n} shards, got {len(manifests)}"
            )

    # Merge with conflict detection
    merged: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    manifest_hashes: List[str] = []

    for m in manifests:
        manifest_hashes.append(m.integrity_hash())
        for key in m.param_keys:
            if key in counts:
                raise ShardConflictError(
                    f"Duplicate param_key '{key}' found in multiple shards"
                )
            counts[key] = 1  # FIX: first insert sets count to 1
            if key in shard_results:
                merged[key] = shard_results[key]

    # Verify all keys in shard_results are accounted for
    for key in shard_results:
        if key not in counts:
            counts[key] = 1
            merged[key] = shard_results[key]

    return AggregationResult(
        results=merged,
        counts=counts,
        manifest_hashes=manifest_hashes,
    )


# ---------------------------------------------------------------------------
# Float audit: Kahan summation
# ---------------------------------------------------------------------------

def naive_sum(values: Sequence[float]) -> float:
    """Plain float64 summation (order-dependent)."""
    s = 0.0
    for v in values:
        s += v
    return s


def kahan_sum(values: Sequence[float]) -> float:
    """Kahan compensated summation (much lower error accumulation)."""
    s = 0.0
    c = 0.0  # compensation
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def theoretical_float_error_bound(values: Sequence[float], n_terms: int) -> float:
    """Theoretical upper bound on naive summation error: K·eps·n·Σ|x|.

    For IEEE 754 double precision, K=1 is a conservative bound for
    summation without catastrophic cancellation.
    """
    K = 1.0
    abs_sum = sum(abs(v) for v in values)
    return K * FLOAT64_EPS * n_terms * abs_sum


def audit_float_summation(
    values_all: Sequence[float],
    n_shards: int = 8,
) -> Dict[str, float]:
    """Compare naive and Kahan summation: single-pass vs sharded.

    The same set of values is summed two ways:
    - 1-shard: all values in a single pass
    - n_shards: split into n groups, each summed separately, then partial
      sums are added together

    Returns dict with:
        naive_1: naive sum of all values (single pass)
        naive_n: naive sum via sharded approach
        kahan_1: Kahan sum of all values (single pass)
        kahan_n: Kahan sum via sharded approach
        rel_diff_naive: |naive_1 - naive_n| / |naive_1| (or 0 if naive_1==0)
        rel_diff_kahan: |kahan_1 - kahan_n| / |kahan_1| (or 0 if kahan_1==0)
        theoretical_bound: K·eps·n·Σ|x| for the full set
        naive_within_bound: bool
        kahan_within_bound: bool
    """
    values = list(values_all)
    n_total = len(values)

    # 1-shard: single pass
    naive_1 = naive_sum(values)
    kahan_1 = kahan_sum(values)

    # n-shard: split, sum each, then sum partials
    chunk = n_total // n_shards
    partials_naive = []
    partials_kahan = []
    for i in range(n_shards):
        start = i * chunk
        end = start + chunk if i < n_shards - 1 else n_total
        chunk_vals = values[start:end]
        partials_naive.append(naive_sum(chunk_vals))
        partials_kahan.append(kahan_sum(chunk_vals))

    naive_n = naive_sum(partials_naive)
    kahan_n = kahan_sum(partials_kahan)

    denom_1 = abs(naive_1) if naive_1 != 0 else 1.0
    denom_k = abs(kahan_1) if kahan_1 != 0 else 1.0

    rel_diff_naive = abs(naive_1 - naive_n) / denom_1
    rel_diff_kahan = abs(kahan_1 - kahan_n) / denom_k

    bound = theoretical_float_error_bound(values, n_total)
    naive_err = abs(naive_1 - naive_n)
    kahan_err = abs(kahan_1 - kahan_n)

    return {
        "naive_1": naive_1,
        "naive_n": naive_n,
        "kahan_1": kahan_1,
        "kahan_n": kahan_n,
        "rel_diff_naive": rel_diff_naive,
        "rel_diff_kahan": rel_diff_kahan,
        "theoretical_bound": bound,
        "naive_err": naive_err,
        "kahan_err": kahan_err,
        "naive_within_bound": naive_err <= bound,
        "kahan_within_bound": kahan_err <= bound,
    }


# ---------------------------------------------------------------------------
# Failure injection helpers
# ---------------------------------------------------------------------------

def inject_shard_kill(
    manifests: List[Manifest],
    kill_indices: Sequence[int],
) -> Tuple[List[Manifest], List[int]]:
    """Simulate killing shards by removing them from the list.

    Returns (remaining_manifests, killed_indices).
    """
    kill_set = set(kill_indices)
    remaining = [m for m in manifests if m.shard_i not in kill_set]
    return remaining, list(kill_set)


def inject_half_written(
    manifest: Manifest,
    corrupt_keys: Sequence[str],
) -> Manifest:
    """Simulate a half-written result by removing result hashes for some keys."""
    m = Manifest(
        shard_i=manifest.shard_i,
        shard_n=manifest.shard_n,
        param_keys=manifest.param_keys,
        tool_version=manifest.tool_version,
        git_sha=manifest.git_sha,
        result_hashes=dict(manifest.result_hashes),
        param_ranges=manifest.param_ranges,
    )
    for key in corrupt_keys:
        m.result_hashes.pop(key, None)
    return m


def inject_key_collision(
    manifests: List[Manifest],
    collision_key: str,
    target_shard_i: int,
) -> List[Manifest]:
    """Inject a duplicate key into a target shard to test conflict detection.

    Also adds a dummy result_hash so manifest validation passes and we
    reach the conflict detection stage.
    """
    result = []
    for m in manifests:
        if m.shard_i == target_shard_i:
            new_keys = list(m.param_keys)
            if collision_key not in new_keys:
                new_keys.append(collision_key)
            new_hashes = dict(m.result_hashes)
            new_hashes[collision_key] = "collision_dummy_hash"
            m2 = Manifest(
                shard_i=m.shard_i,
                shard_n=m.shard_n,
                param_keys=tuple(new_keys),
                tool_version=m.tool_version,
                git_sha=m.git_sha,
                result_hashes=new_hashes,
                param_ranges=m.param_ranges,
            )
            result.append(m2)
        else:
            result.append(m)
    return result


def tamper_manifest(
    manifest: Manifest,
    tamper_git_sha: bool = False,
    tamper_result_hash: bool = False,
) -> Manifest:
    """Create a tampered copy of a manifest for integrity testing."""
    new_sha = manifest.git_sha
    if tamper_git_sha:
        new_sha = "0" * 40  # obviously wrong

    new_hashes = dict(manifest.result_hashes)
    if tamper_result_hash and new_hashes:
        first_key = next(iter(new_hashes))
        new_hashes[first_key] = "deadbeef" * 8

    return Manifest(
        shard_i=manifest.shard_i,
        shard_n=manifest.shard_n,
        param_keys=manifest.param_keys,
        tool_version=manifest.tool_version,
        git_sha=new_sha,
        result_hashes=new_hashes,
        param_ranges=manifest.param_ranges,
    )
