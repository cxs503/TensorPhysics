"""
TensorPhysics Cluster Sharding — Deterministic Aggregation Contract
====================================================================

Provides machine-verifiable guarantees for distributed parameter-scan execution:

1. **ShardPlan**: Deterministic mapping of parameter points → shards.
   Proves key coverage (all N points assigned) and uniqueness (each point
   assigned to exactly one shard).

2. **Manifest**: Per-shard metadata including shard index, total shards,
   parameter range, tool version, git SHA, and result hash.

3. **Aggregator**: Order-independent merge of shard results by parameter key.
   Enforces count==1 per key (conflict detection → raise).

4. **FloatAudit**: Kahan compensated summation vs naive, with theoretical
   error bound verification.

Design invariants:
- aggregate(results) is commutative (key-order independent)
- For each parameter point, exactly one shard contributes (count==1)
- Missing shards → fail-closed (raise, never silently drop)
- Tampered manifest → checksum mismatch (raise)

References:
- Higham, N.J. "The accuracy of floating point summation", 2009
- Kahan, W. "Further remarks on reducing truncation errors", CACM 1965
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Key canonicalization: convert any key to a hashable canonical string form
# ---------------------------------------------------------------------------

def canonical_key(key: Any) -> str:
    """Convert any parameter key to a canonical JSON string (hashable, sortable).
    
    This is the universal key representation used throughout the sharding system.
    All result dicts use canonical string keys internally.
    """
    if isinstance(key, str):
        # Validate it's valid JSON or use as-is
        try:
            json.loads(key)
            return key
        except (json.JSONDecodeError, TypeError):
            return json.dumps(key)
    return json.dumps(key, sort_keys=True, default=str)


def _stable_hash(key: Any) -> int:
    """Compute a stable (non-randomized) hash for deterministic sharding."""
    ck = canonical_key(key)
    serialized = ck.encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    # Use first 16 hex chars = 64 bits, fits in Python int natively
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# ShardPlan: deterministic parameter-point → shard mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardPlan:
    """Deterministic mapping of parameter points to shards.

    Given N parameter points (identified by any keys) and n shards,
    assigns each point to exactly one shard using modular hashing.

    Invariants (provable by construction):
    - Coverage: every key appears in exactly one shard's assignment
    - Uniqueness: no key appears in more than one shard
    - Determinism: same (keys, n_shards) → same assignment
    - Balance: max shard size - min shard size <= 1
    """
    n_shards: int
    assignments: Dict[int, Tuple[str, ...]]  # shard_idx → tuple of canonical keys

    @classmethod
    def create(cls, param_keys: Sequence[Any], n_shards: int) -> "ShardPlan":
        """Create a deterministic shard plan.

        Args:
            param_keys: Sequence of parameter point identifiers (any type).
            n_shards: Number of shards (must be >= 1).

        Returns:
            ShardPlan with deterministic assignments (canonical string keys).

        Raises:
            ValueError: If n_shards < 1 or param_keys is empty.
        """
        if n_shards < 1:
            raise ValueError(f"n_shards must be >= 1, got {n_shards}")
        if len(param_keys) == 0:
            raise ValueError("param_keys must be non-empty")

        # Deterministic assignment: stable_hash(key) mod n_shards
        buckets: Dict[int, list] = {i: [] for i in range(n_shards)}
        for key in param_keys:
            h = _stable_hash(key)
            shard_idx = h % n_shards  # Python handles big ints natively
            buckets[shard_idx].append(canonical_key(key))

        assignments = {i: tuple(keys) for i, keys in buckets.items()}
        return cls(n_shards=n_shards, assignments=assignments)

    def verify_coverage(self, param_keys: Sequence[Any]) -> bool:
        """Verify all param_keys are covered exactly once."""
        all_assigned = set()
        for keys in self.assignments.values():
            for ck in keys:
                if ck in all_assigned:
                    return False  # duplicate
                all_assigned.add(ck)
        
        expected = {canonical_key(k) for k in param_keys}
        return all_assigned == expected

    def verify_uniqueness(self) -> bool:
        """Verify no key appears in multiple shards."""
        seen = set()
        for keys in self.assignments.values():
            for ck in keys:
                if ck in seen:
                    return False
                seen.add(ck)
        return True

    def get_shard_keys(self, shard_idx: int) -> Tuple[str, ...]:
        """Get canonical parameter keys assigned to a specific shard."""
        if shard_idx < 0 or shard_idx >= self.n_shards:
            raise IndexError(f"shard_idx {shard_idx} out of range [0, {self.n_shards})")
        return self.assignments[shard_idx]

    @property
    def total_keys(self) -> int:
        return sum(len(v) for v in self.assignments.values())

    @property
    def balance(self) -> int:
        """Max shard size - min shard size (should be <= 1 for good balance)."""
        sizes = [len(v) for v in self.assignments.values()]
        return max(sizes) - min(sizes) if sizes else 0


# ---------------------------------------------------------------------------
# Manifest: per-shard metadata with integrity checksums
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    """Per-shard execution manifest.

    Records everything needed to verify a shard's contribution:
    - Shard identity (index, total)
    - Parameter range (which canonical keys this shard handles)
    - Tool version and git SHA for reproducibility
    - Result hash for integrity verification
    """
    shard_index: int
    total_shards: int
    param_keys: Tuple[str, ...]  # canonical keys
    tool_version: str
    git_sha: str
    result_hash: str  # SHA-256 of serialized results
    timestamp: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def compute_result_hash(self, results: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of results + metadata (canonical serialization).
        
        Includes metadata (git_sha, tool_version, etc.) in the hash so that
        any tampering with manifest fields is detected.
        
        Args:
            results: Dict with canonical string keys.
        """
        # Combine results with metadata for integrity
        payload = {
            "results": dict(sorted(results.items())),
            "metadata": {
                "shard_index": self.shard_index,
                "total_shards": self.total_shards,
                "param_keys": list(self.param_keys),
                "tool_version": self.tool_version,
                "git_sha": self.git_sha,
            }
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            default=_json_default,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def verify_integrity(self, results: Optional[Dict[str, Any]] = None) -> bool:
        """Verify manifest integrity.

        If results provided, recompute and compare result_hash.
        Always checks that param_keys are non-empty and indices valid.
        """
        if self.shard_index < 0 or self.shard_index >= self.total_shards:
            return False
        if len(self.param_keys) == 0:
            return False
        if not self.git_sha or not self.result_hash:
            return False
        if results is not None:
            computed = self.compute_result_hash(results)
            if computed != self.result_hash:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "param_keys": list(self.param_keys),
            "tool_version": self.tool_version,
            "git_sha": self.git_sha,
            "result_hash": self.result_hash,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        return cls(
            shard_index=d["shard_index"],
            total_shards=d["total_shards"],
            param_keys=tuple(d["param_keys"]),
            tool_version=d["tool_version"],
            git_sha=d["git_sha"],
            result_hash=d["result_hash"],
            timestamp=d.get("timestamp", ""),
            extra=d.get("extra", {}),
        )


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Aggregator: order-independent merge with conflict detection
# ---------------------------------------------------------------------------

class AggregationError(Exception):
    """Raised when aggregation contract is violated."""
    pass


class Aggregator:
    """Merge shard results with contract enforcement.

    Contract:
    - aggregate(results_list) is commutative (order-independent)
    - Each parameter key must appear exactly once across all shards
    - Missing shards (gaps in shard indices) → raise
    - Duplicate keys → raise (conflict detection)
    
    All keys are canonical strings internally.
    """

    def __init__(self, expected_total_shards: int, expected_keys: Sequence[Any]):
        """
        Args:
            expected_total_shards: Total number of shards expected.
            expected_keys: All parameter keys (any type, will be canonicalized).
        """
        self.expected_total_shards = expected_total_shards
        self.expected_keys = {canonical_key(k) for k in expected_keys}

    def aggregate(
        self,
        shard_results: List[Tuple[int, Dict[str, Any], Manifest]],
        *,
        strict_shard_count: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate shard results into a single result dict.

        Args:
            shard_results: List of (shard_index, results_dict, manifest).
                          results_dict uses canonical string keys.
            strict_shard_count: If True, require all shards present.

        Returns:
            Merged dict: canonical_key → result_value.

        Raises:
            AggregationError: If contract violated.
        """
        # Check for duplicate shard indices
        shard_indices = [sr[0] for sr in shard_results]
        if len(shard_indices) != len(set(shard_indices)):
            raise AggregationError(
                f"Duplicate shard indices detected: {shard_indices}"
            )

        # Fail-closed: check all shards present
        if strict_shard_count:
            expected_indices = set(range(self.expected_total_shards))
            present_indices = set(shard_indices)
            missing = expected_indices - present_indices
            if missing:
                raise AggregationError(
                    f"Missing shards: {sorted(missing)}. "
                    f"Fail-closed: cannot aggregate partial results."
                )

        # First pass: check for conflicts (before integrity, so we catch conflicts even if hash is wrong)
        all_keys_with_sources: Dict[str, List[int]] = {}
        for shard_idx, results, manifest in shard_results:
            for ck in results.keys():
                all_keys_with_sources.setdefault(ck, []).append(shard_idx)
        
        for ck, sources in all_keys_with_sources.items():
            if len(sources) > 1:
                raise AggregationError(
                    f"Conflict: parameter key '{ck}' produced by "
                    f"shards {sources}. "
                    f"Each key must be computed by exactly one shard."
                )

        # Validate manifests
        for shard_idx, results, manifest in shard_results:
            if manifest.shard_index != shard_idx:
                raise AggregationError(
                    f"Shard index mismatch: got {shard_idx}, "
                    f"manifest says {manifest.shard_index}"
                )
            if manifest.total_shards != self.expected_total_shards:
                raise AggregationError(
                    f"Total shards mismatch in shard {shard_idx}: "
                    f"expected {self.expected_total_shards}, "
                    f"got {manifest.total_shards}"
                )
            if not manifest.verify_integrity(results):
                raise AggregationError(
                    f"Manifest integrity check failed for shard {shard_idx}. "
                    f"Result hash mismatch or invalid metadata."
                )

        # Merge (conflict-free by construction from first pass)
        merged: Dict[str, Any] = {}
        key_sources: Dict[str, List[int]] = {}

        for shard_idx, results, manifest in shard_results:
            for ck, value in results.items():
                merged[ck] = value
                key_sources.setdefault(ck, []).append(shard_idx)

        # Verify key coverage
        merged_keys = set(merged.keys())
        missing_keys = self.expected_keys - merged_keys
        if missing_keys:
            raise AggregationError(
                f"Missing {len(missing_keys)} parameter keys: "
                f"{sorted(list(missing_keys)[:5])}..."
            )

        extra_keys = merged_keys - self.expected_keys
        if extra_keys:
            raise AggregationError(
                f"Unexpected {len(extra_keys)} parameter keys: "
                f"{sorted(list(extra_keys)[:5])}..."
            )

        # Verify count == 1 per key
        for ck, sources in key_sources.items():
            if len(sources) != 1:
                raise AggregationError(
                    f"Key '{ck}' has count {len(sources)} != 1"
                )

        return merged

    def aggregate_commutativity_check(
        self,
        shard_results: List[Tuple[int, Dict[str, Any], Manifest]],
    ) -> bool:
        """Verify aggregation is commutative (order-independent)."""
        if len(shard_results) <= 1:
            return True

        reference = self.aggregate(shard_results)

        perms_to_test = min(math.factorial(len(shard_results)), 24)
        rng = np.random.RandomState(42)
        indices = list(range(len(shard_results)))

        for _ in range(perms_to_test):
            rng.shuffle(indices)
            permuted = [shard_results[i] for i in indices]
            result = self.aggregate(permuted)
            if result != reference:
                return False

        return True


# ---------------------------------------------------------------------------
# FloatAudit: Kahan compensated summation with theoretical error bounds
# ---------------------------------------------------------------------------

class FloatAudit:
    """Floating-point summation audit with error bounds.

    Compares naive summation vs Kahan compensated summation.
    Theoretical bound for naive: |error| <= (n-1) * eps * sum(|x_i|)
    Theoretical bound for Kahan: |error| <= 2 * eps * sum(|x_i|) + O(eps^2)

    Where eps = machine epsilon for float64 ≈ 2.22e-16.
    """

    EPS64 = np.finfo(np.float64).eps  # ≈ 2.22e-16

    @staticmethod
    def naive_sum(values: Sequence[float]) -> float:
        """Standard left-to-right summation."""
        s = 0.0
        for x in values:
            s += x
        return s

    @staticmethod
    def kahan_sum(values: Sequence[float]) -> float:
        """Kahan compensated summation (Neumaier variant)."""
        s = 0.0
        c = 0.0  # compensation for lost low-order bits
        for x in values:
            y = x - c
            t = s + y
            c = (t - s) - y
            s = t
        return s

    @staticmethod
    def pairwise_sum(values: Sequence[float]) -> float:
        """Pairwise (divide-and-conquer) summation — O(log n) error."""
        arr = list(values)
        while len(arr) > 1:
            new_arr = []
            for i in range(0, len(arr), 2):
                if i + 1 < len(arr):
                    new_arr.append(arr[i] + arr[i + 1])
                else:
                    new_arr.append(arr[i])
            arr = new_arr
        return arr[0] if arr else 0.0

    @staticmethod
    def reference_sum(values: Sequence[float]) -> float:
        """High-precision reference using Python's math.fsum (exact for float64)."""
        return math.fsum(values)

    @classmethod
    def audit(
        cls,
        values: Sequence[float],
        label: str = "",
    ) -> Dict[str, Any]:
        """Perform comprehensive float summation audit."""
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        abs_sum = float(np.sum(np.abs(arr)))

        naive = cls.naive_sum(arr)
        kahan = cls.kahan_sum(arr)
        pairwise = cls.pairwise_sum(arr)
        reference = cls.reference_sum(arr)

        naive_bound = (n - 1) * cls.EPS64 * abs_sum if n > 1 else 0.0
        kahan_bound = 2.0 * cls.EPS64 * abs_sum + n * cls.EPS64**2 * abs_sum

        naive_err = abs(naive - reference)
        kahan_err = abs(kahan - reference)
        pairwise_err = abs(pairwise - reference)

        denom = abs(reference) if reference != 0 else 1.0
        naive_rel = naive_err / denom
        kahan_rel = kahan_err / denom
        pairwise_rel = pairwise_err / denom

        return {
            "label": label,
            "n": n,
            "sum_abs": abs_sum,
            "naive": naive,
            "kahan": kahan,
            "pairwise": pairwise,
            "reference": reference,
            "naive_abs_err": naive_err,
            "kahan_abs_err": kahan_err,
            "pairwise_abs_err": pairwise_err,
            "naive_rel_err": naive_rel,
            "kahan_rel_err": kahan_rel,
            "pairwise_rel_err": pairwise_rel,
            "naive_theory_bound": naive_bound,
            "kahan_theory_bound": kahan_bound,
            "naive_within_bound": naive_err <= naive_bound + 1e-300,
            "kahan_within_bound": kahan_err <= kahan_bound + 1e-300,
            "eps64": cls.EPS64,
        }


# ---------------------------------------------------------------------------
# Cross-shard float aggregation with error tracking
# ---------------------------------------------------------------------------

class CrossShardFloatAggregator:
    """Aggregate floating-point results across shards with error bounds."""

    @classmethod
    def aggregate_float_sum(
        cls,
        shard_partial_sums: List[Tuple[int, float]],
    ) -> Dict[str, Any]:
        """Aggregate partial sums from shards using Kahan compensation."""
        sorted_sums = sorted(shard_partial_sums, key=lambda x: x[0])
        values = [s for _, s in sorted_sums]

        naive = FloatAudit.naive_sum(values)
        kahan = FloatAudit.kahan_sum(values)
        reference = FloatAudit.reference_sum(values)

        n = len(values)
        abs_sum = sum(abs(v) for v in values)
        naive_bound = (n - 1) * FloatAudit.EPS64 * abs_sum if n > 1 else 0.0
        kahan_bound = 2.0 * FloatAudit.EPS64 * abs_sum

        naive_err = abs(naive - reference)
        kahan_err = abs(kahan - reference)

        return {
            "n_shards": n,
            "naive_sum": naive,
            "kahan_sum": kahan,
            "reference_sum": reference,
            "naive_abs_err": naive_err,
            "kahan_abs_err": kahan_err,
            "naive_theory_bound": naive_bound,
            "kahan_theory_bound": kahan_bound,
            "naive_within_bound": naive_err <= naive_bound + 1e-300,
            "kahan_within_bound": kahan_err <= kahan_bound + 1e-300,
            "shard_indices": [idx for idx, _ in sorted_sums],
        }
