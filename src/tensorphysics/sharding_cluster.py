"""Cluster shard channel — ``{i}/{n}`` templates, on-disk shard artifacts, fail-closed fetch.

This module sits *between* :mod:`tensorphysics.sharding` (the in-memory
deterministic aggregation contract: ``ShardPlan`` / ``Manifest`` /
``Aggregator``) and a real cluster job submission channel that speaks the
``--cmd-tpl "... {i}/{n} ..." --shards N`` dialect (one command template,
rendered once per shard ordinal, artifacts written back to a shared stage
directory and later fetched).

Guarantees added here (all machine-checkable, all fail-closed):

1. **Command rendering** — ``render_shard_cmd`` validates the template contains
   exactly the ``{i}`` and ``{n}`` placeholders and renders ordinals in
   ``1..n`` (cluster convention) or ``0..n-1``.
2. **Plan document** — ``build_plan`` freezes the parameter-point → shard
   mapping, the tool version, the git SHA and the compute entry point into a
   single digestible JSON document (``plan_digest``).  Any shard that runs a
   different plan is detectable.
3. **Atomic shard artifacts** — ``write_artifact`` writes an envelope with a
   result hash *and* an envelope hash via ``tmp + os.replace``; a truncated or
   half-flushed file is rejected on read (never silently accepted).
4. **Fail-closed fetch** — ``aggregate_from_dir`` refuses to return a partial
   mapping: missing artifact, half-written artifact, foreign/extra key,
   boundary mismatch, version drift or hash mismatch all raise
   :class:`ShardChannelError`.  It never falls back to last-write-wins.
5. **Calibrated float bounds** — :class:`FloatContract` records the naive
   summation error bound ``K·eps·n·Σ|x|`` with a *measured* calibration
   constant and a conservative hardcoded cap, and audits shard-count
   invariance of sum-sensitive scan aggregates.

The synthetic compute used by the CLI (``synthetic_scan_energy``) is explicitly
a **float-arithmetic fixture**, not a physics model: it is a deterministic
101-term alternating series with strong cancellation, chosen because it makes
summation-order effects observable.  Domain compute is injected by the plan's
``compute`` reference, so no domain semantics are hardcoded here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


from tensorphysics.sharding import (
    AggregationError,
    Aggregator,
    FloatAudit,
    Manifest,
    ShardPlan,
    _json_default,
    _stable_hash,
    canonical_key,
)

PLAN_SCHEMA = "tensorphysics.shard_plan/1"
ARTIFACT_SCHEMA = "tensorphysics.shard_artifact/1"

__all__ = [
    "ShardChannelError",
    "ShardSpec",
    "render_shard_cmd",
    "shard_commands",
    "canonical_digest",
    "keys_digest",
    "balanced_buckets",
    "build_plan",
    "write_plan",
    "load_plan",
    "artifact_path",
    "write_artifact",
    "read_artifact",
    "verify_artifact",
    "aggregate_from_dir",
    "FloatContract",
    "synthetic_scan_energy",
    "resolve_compute",
    "run_shard",
    "main",
]


class ShardChannelError(AggregationError):
    """Raised whenever the cluster shard channel contract is violated."""


# ---------------------------------------------------------------------------
# Digests and command rendering
# ---------------------------------------------------------------------------

def canonical_digest(payload: Any) -> str:
    """SHA-256 of a canonical JSON serialisation (sorted keys, fixed encoding)."""
    blob = json.dumps(payload, sort_keys=True, default=_json_default,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def keys_digest(keys: Iterable[str]) -> str:
    """SHA-256 over the *sorted* canonical keys (order-insensitive, set-exact)."""
    return canonical_digest(sorted(set(canonical_key(k) for k in keys)))


def render_shard_cmd(
    cmd_tpl: str,
    shard_index: int,
    total_shards: int,
    *,
    one_based: bool = True,
    plan_path: Optional[os.PathLike | str] = None,
    stage_dir: Optional[os.PathLike | str] = None,
    defer: Sequence[str] = (),
) -> str:
    """Render a cluster ``{i}/{n}`` command template for one shard ordinal.

    Args:
        cmd_tpl: Template containing both ``{i}`` and ``{n}``.
        shard_index: Zero-based shard index.
        total_shards: Total number of shards (``n``).
        one_based: Cluster convention — ``{i}`` becomes ``1..n``.

    Raises:
        ShardChannelError: on a malformed template or out-of-range index.
    """
    if not isinstance(cmd_tpl, str) or "{i}" not in cmd_tpl or "{n}" not in cmd_tpl:
        raise ShardChannelError(
            f"cmd_tpl must contain both '{{i}}' and '{{n}}': {cmd_tpl!r}"
        )
    if total_shards < 1:
        raise ShardChannelError(f"total_shards must be >= 1, got {total_shards}")
    if not (0 <= shard_index < total_shards):
        raise ShardChannelError(
            f"shard_index {shard_index} out of range for n={total_shards}"
        )
    ordinal = shard_index + 1 if one_based else shard_index
    cmd = cmd_tpl.replace("{i}", str(ordinal)).replace("{n}", str(total_shards))
    if plan_path is not None:
        cmd = cmd.replace("{plan}", str(plan_path))
    if stage_dir is not None:
        cmd = cmd.replace("{out}", str(stage_dir)).replace("{stage_dir}", str(stage_dir))
    deferred = {f"{{{name.strip('{}')}}}" for name in defer}
    leftover = [
        ph
        for ph in ("{i}", "{n}", "{plan}", "{out}", "{stage_dir}")
        if ph in cmd and ph not in deferred
    ]
    if leftover:
        raise ShardChannelError(
            f"unrendered placeholders {leftover} remain in {cmd!r}"
        )
    return cmd


def shard_commands(
    plan: Dict[str, Any],
    plan_path: os.PathLike | str,
    stage_dir: os.PathLike | str,
) -> List[str]:
    """Concrete per-shard commands for a plan (one entry per shard ordinal)."""
    return [
        render_shard_cmd(
            plan["cmd_tpl"],
            spec.shard_index,
            int(plan["n_shards"]),
            one_based=bool(plan.get("one_based", True)),
            plan_path=plan_path,
            stage_dir=stage_dir,
        )
        for spec in plan_specs(plan)
    ]


# ---------------------------------------------------------------------------
# Plan document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardSpec:
    """One shard's execution spec inside a plan document."""

    shard_index: int
    total_shards: int
    param_keys: Tuple[str, ...]
    cmd: str

    @property
    def keys_digest(self) -> str:
        return keys_digest(self.param_keys)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "cmd": self.cmd,
            "param_keys": list(self.param_keys),
            "keys_digest": self.keys_digest,
        }


def balanced_buckets(
    canon_keys: Sequence[str],
    n_shards: int,
    *,
    mapping: str = "balanced",
) -> List[List[str]]:
    """Deterministic parameter-point → shard mapping.

    ``mapping="balanced"`` (default, used by the cluster channel): sort canonical
    keys by ``(sha256, key)`` and deal them round-robin.  Guarantees
    *coverage* (every key once), *uniqueness* (no repeats) and
    *balance* (``max size - min size <= 1``) **by construction**, which matters
    on a cluster where a skewed shard becomes the critical path.
    ``mapping="hash_mod"`` reproduces :class:`~tensorphysics.sharding.ShardPlan`
    (``sha256 mod n``), which is deterministic but not size-balanced.
    """
    if n_shards < 1:
        raise ShardChannelError(f"n_shards must be >= 1, got {n_shards}")
    canon = sorted(set(canon_keys), key=lambda k: (_stable_hash(k), k))
    if len(canon) != len(list(canon_keys)):
        raise ShardChannelError("duplicate parameter keys in plan")

    buckets: List[List[str]] = [[] for _ in range(n_shards)]
    if mapping == "balanced":
        for rank, key in enumerate(canon):
            buckets[rank % n_shards].append(key)
    elif mapping == "hash_mod":
        for key in canon:
            buckets[_stable_hash(key) % n_shards].append(key)
    else:
        raise ShardChannelError(f"unknown mapping {mapping!r}")

    buckets = [sorted(b) for b in buckets]
    flat = [k for b in buckets for k in b]
    if sorted(flat) != sorted(canon):
        raise ShardChannelError("mapping violated coverage/uniqueness")
    return buckets


def build_plan(
    param_keys: Sequence[Any],
    n_shards: int,
    cmd_tpl: str,
    *,
    tool_version: str,
    git_sha: str,
    compute: str = "tensorphysics.sharding_cluster:synthetic_scan_energy",
    one_based: bool = True,
    stage_dir: str = "",
    mapping: str = "balanced",
) -> Dict[str, Any]:
    """Build the frozen plan document shared by every shard.

    The mapping itself comes from :class:`tensorphysics.sharding.ShardPlan`
    (deterministic SHA-256 bucketing), so coverage/uniqueness properties are
    inherited.  Here we additionally freeze *what* runs (``compute``),
    *which code* runs (``tool_version`` + ``git_sha``) and *how* it is
    launched (rendered ``cmd`` per shard).

    Raises:
        ShardChannelError: if the mapping does not cover every key exactly once
            or if any rendered command is degenerate.
    """
    if not git_sha or not str(git_sha).strip():
        raise ShardChannelError("git_sha must be a non-empty string")
    if not tool_version or not str(tool_version).strip():
        raise ShardChannelError("tool_version must be a non-empty string")

    canon = [canonical_key(k) for k in param_keys]
    if not canon:
        raise ShardChannelError("param_keys must be non-empty")

    buckets = balanced_buckets(canon, n_shards, mapping=mapping)

    specs: List[ShardSpec] = []
    for i in range(n_shards):
        keys = tuple(buckets[i])
        specs.append(
            ShardSpec(
                shard_index=i,
                total_shards=n_shards,
                param_keys=keys,
                cmd=render_shard_cmd(
                    cmd_tpl,
                    i,
                    n_shards,
                    one_based=one_based,
                    plan_path="{plan}",
                    stage_dir="{out}",
                    defer=("plan", "out"),
                ),
            )
        )

    body = {
        "schema": PLAN_SCHEMA,
        "n_shards": n_shards,
        "one_based": one_based,
        "cmd_tpl": cmd_tpl,
        "tool_version": str(tool_version),
        "git_sha": str(git_sha),
        "compute": compute,
        "stage_dir": stage_dir,
        "mapping": mapping,
        "total_keys": len(canon),
        "shards": [s.to_dict() for s in specs],
    }
    body["plan_digest"] = canonical_digest(body)
    return body


def write_plan(path: os.PathLike | str, plan: Dict[str, Any]) -> Path:
    """Atomically write a plan document."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(p, plan)
    return p


def load_plan(path: os.PathLike | str) -> Dict[str, Any]:
    """Load and integrity-check a plan document.

    Raises:
        ShardChannelError: on unreadable JSON, schema mismatch or digest
            mismatch (a tampered plan is never trusted).
    """
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except FileNotFoundError as exc:
        raise ShardChannelError(f"plan not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ShardChannelError(f"plan is not valid JSON: {p}") from exc

    if doc.get("schema") != PLAN_SCHEMA:
        raise ShardChannelError(f"plan schema mismatch: {doc.get('schema')!r}")
    body = {k: v for k, v in doc.items() if k != "plan_digest"}
    if canonical_digest(body) != doc.get("plan_digest"):
        raise ShardChannelError("plan digest mismatch (plan was modified)")
    if len(doc.get("shards", [])) != doc.get("n_shards"):
        raise ShardChannelError("plan shard list length != n_shards")
    return doc


def plan_specs(plan: Dict[str, Any]) -> List[ShardSpec]:
    return [
        ShardSpec(
            shard_index=int(s["shard_index"]),
            total_shards=int(s["total_shards"]),
            param_keys=tuple(s["param_keys"]),
            cmd=str(s["cmd"]),
        )
        for s in plan["shards"]
    ]


# ---------------------------------------------------------------------------
# On-disk shard artifacts (atomic write, fail-closed read)
# ---------------------------------------------------------------------------

def artifact_path(out_dir: os.PathLike | str, shard_index: int, total_shards: int) -> Path:
    return Path(out_dir) / f"shard_{shard_index:03d}_of_{total_shards:03d}.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via tmp file + fsync + os.replace (never a half file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, sort_keys=True, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_artifact(
    out_dir: os.PathLike | str,
    *,
    plan: Dict[str, Any],
    shard_index: int,
    results: Dict[str, Any],
) -> Path:
    """Write one shard's results as an atomically-flushed artifact.

    The writer refuses (fail-closed at source) to emit an artifact whose key set
    differs from the plan for this shard — that is what makes a foreign-key or
    boundary violation detectable rather than silently merged later.
    """
    specs = plan_specs(plan)
    spec = specs[shard_index]
    got = {canonical_key(k) for k in results}
    want = set(spec.param_keys)
    if got != want:
        raise ShardChannelError(
            "refusing to write artifact: key set mismatch "
            f"(missing={sorted(want - got)[:3]}, extra={sorted(got - want)[:3]})"
        )

    envelope = {
        "schema": ARTIFACT_SCHEMA,
        "status": "complete",
        "shard_index": shard_index,
        "total_shards": plan["n_shards"],
        "tool_version": plan["tool_version"],
        "git_sha": plan["git_sha"],
        "plan_digest": plan["plan_digest"],
        "n_expected": len(want),
        "n_written": len(results),
        "keys_digest": keys_digest(want),
        "results": {canonical_key(k): results[k] for k in results},
    }
    envelope["results_hash"] = canonical_digest(envelope["results"])
    envelope["envelope_hash"] = canonical_digest(
        {k: v for k, v in envelope.items() if k != "envelope_hash"}
    )
    path = artifact_path(out_dir, shard_index, plan["n_shards"])
    _atomic_write_json(path, envelope)
    return path


def read_artifact(path: os.PathLike | str) -> Dict[str, Any]:
    """Read an artifact, rejecting missing / truncated / malformed files."""
    p = Path(path)
    try:
        raw = p.read_text()
    except FileNotFoundError as exc:
        raise ShardChannelError(f"shard artifact missing (killed shard?): {p}") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ShardChannelError(
            f"shard artifact truncated/corrupt (half-written file): {p}"
        ) from exc
    if not isinstance(doc, dict) or doc.get("schema") != ARTIFACT_SCHEMA:
        raise ShardChannelError(f"artifact schema mismatch: {p}")
    return doc


def verify_artifact(
    doc: Dict[str, Any],
    *,
    plan: Dict[str, Any],
    shard_index: int,
) -> Dict[str, Any]:
    """Verify one in-memory artifact against the plan.

    Checks (each individually sufficient to raise):
    status complete / shard index / total shards / tool version / git SHA /
    plan digest / written==expected / key-set digest / results hash /
    envelope hash.

    Returns:
        The artifact's ``results`` mapping (canonical keys).

    Raises:
        ShardChannelError: on any inconsistency.
    """
    spec = plan_specs(plan)[shard_index]
    where = f"shard {shard_index}/{plan['n_shards']}"

    if doc.get("status") != "complete":
        raise ShardChannelError(f"{where}: status={doc.get('status')!r} (incomplete)")
    if int(doc.get("shard_index", -1)) != shard_index:
        raise ShardChannelError(
            f"{where}: artifact claims shard_index={doc.get('shard_index')}"
        )
    if int(doc.get("total_shards", -1)) != int(plan["n_shards"]):
        raise ShardChannelError(
            f"{where}: total_shards {doc.get('total_shards')} != plan {plan['n_shards']}"
        )
    if doc.get("tool_version") != plan["tool_version"]:
        raise ShardChannelError(
            f"{where}: tool_version {doc.get('tool_version')!r} != plan "
            f"{plan['tool_version']!r}"
        )
    if doc.get("git_sha") != plan["git_sha"]:
        raise ShardChannelError(
            f"{where}: git_sha {doc.get('git_sha')!r} != plan {plan['git_sha']!r}"
        )
    if doc.get("plan_digest") != plan["plan_digest"]:
        raise ShardChannelError(f"{where}: plan_digest mismatch (shard ran another plan)")

    results = doc.get("results")
    if not isinstance(results, dict):
        raise ShardChannelError(f"{where}: results missing or not an object")
    if int(doc.get("n_written", -1)) != len(results):
        raise ShardChannelError(
            f"{where}: n_written {doc.get('n_written')} != actual {len(results)}"
        )
    if len(results) != len(spec.param_keys):
        raise ShardChannelError(
            f"{where}: {len(results)} keys written, plan expects {len(spec.param_keys)}"
        )
    if doc.get("keys_digest") != keys_digest(results.keys()):
        raise ShardChannelError(f"{where}: key-set digest mismatch (boundary tamper)")
    if doc.get("keys_digest") != spec.keys_digest:
        raise ShardChannelError(f"{where}: key set differs from plan assignment")
    if doc.get("results_hash") != canonical_digest(results):
        raise ShardChannelError(f"{where}: results hash mismatch (payload tamper)")

    body = {k: v for k, v in doc.items() if k != "envelope_hash"}
    if doc.get("envelope_hash") != canonical_digest(body):
        raise ShardChannelError(f"{where}: envelope hash mismatch (envelope tamper)")
    return results


def aggregate_from_dir(
    stage_dir: os.PathLike | str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Fail-closed aggregation of shard artifacts fetched from a stage dir.

    Returns the merged mapping **only** if every shard artifact is present,
    complete and consistent with the plan; otherwise raises
    :class:`ShardChannelError` (never returns a partial mapping, never uses
    last-write-wins).

    Raises:
        ShardChannelError: on any missing/inconsistent shard.
    """
    n = int(plan["n_shards"])
    triples: List[Tuple[int, Dict[str, Any], Manifest]] = []
    all_keys: List[str] = []

    for spec in plan_specs(plan):
        doc = read_artifact(artifact_path(stage_dir, spec.shard_index, n))
        results = verify_artifact(doc, plan=plan, shard_index=spec.shard_index)
        manifest = Manifest(
            shard_index=spec.shard_index,
            total_shards=n,
            param_keys=tuple(sorted(results.keys())),
            tool_version=plan["tool_version"],
            git_sha=plan["git_sha"],
            result_hash="",
            timestamp=str(doc.get("written_at", "")),
            extra={
                "plan_digest": plan["plan_digest"],
                "artifact_envelope_hash": doc.get("envelope_hash", ""),
            },
        )
        manifest.result_hash = manifest.compute_result_hash(results)
        triples.append((spec.shard_index, results, manifest))
        all_keys.extend(results.keys())

    aggregator = Aggregator(expected_total_shards=n, expected_keys=all_keys)
    merged = aggregator.aggregate(triples)

    # Explicit post-condition: exactly one contribution per planned key.
    planned = {k for spec in plan_specs(plan) for k in spec.param_keys}
    if set(merged) != planned:
        raise ShardChannelError("aggregated key set != planned key set")
    return merged


# ---------------------------------------------------------------------------
# Float contract: calibrated bounds + shard-count invariance audit
# ---------------------------------------------------------------------------

class FloatContract:
    """Calibrated floating-point aggregation contract.

    Naive summation error bound (Higham 1993)::

        |naive - exact| <= K_NAIVE * eps * n * sum|x|

    with ``K_NAIVE`` hardcoded to the classical worst-case cap ``1`` (the
    provable bound is ``(n-1)*eps/(1-(n-1)*eps) * sum|x|``, i.e. ``K < 1``).
    ``calibrate()`` measures the *observed* maximum of
    ``err / (eps*n*sum|x|)`` on a labelled corpus so the hardcoded ``K`` is
    both measured and conservatively above the measurement; validation on a
    disjoint corpus then checks ``observed <= K``.

    Kahan compensated summation bound: ``|kahan - exact| <= K_KAHAN*eps*sum|x|``
    with ``K_KAHAN = 2`` (Neumaier variant), equivalently ``2/n`` in the
    ``K*eps*n*sum|x|`` normalisation used for naive.
    """

    K_NAIVE = 1.0
    K_KAHAN = 2.0
    EPS64 = FloatAudit.EPS64

    #: measured maximum ratio on the calibration corpus (filled by calibrate())
    CALIBRATION: Dict[str, Any] = {}

    @classmethod
    def bound_naive(cls, n: int, abs_sum: float) -> float:
        return cls.K_NAIVE * cls.EPS64 * n * abs_sum

    @classmethod
    def bound_kahan(cls, abs_sum: float) -> float:
        return cls.K_KAHAN * cls.EPS64 * abs_sum

    @classmethod
    def audit_values(cls, values: Sequence[float], label: str = "") -> Dict[str, Any]:
        """Per-vector audit of naive vs Kahan vs exact (fsum) summation."""
        arr = [float(v) for v in values]
        n = int(len(arr))
        abs_sum = math.fsum(abs(v) for v in arr)
        naive = FloatAudit.naive_sum(arr)
        kahan = FloatAudit.kahan_sum(arr)
        exact = FloatAudit.reference_sum(arr)
        naive_err = abs(naive - exact)
        kahan_err = abs(kahan - exact)
        nb = cls.bound_naive(n, abs_sum)
        kb = cls.bound_kahan(abs_sum)
        return {
            "label": label,
            "n": n,
            "sum_abs": abs_sum,
            "naive": naive,
            "kahan": kahan,
            "exact": exact,
            "naive_abs_err": naive_err,
            "kahan_abs_err": kahan_err,
            "naive_ratio": (naive_err / nb) if nb > 0 else 0.0,
            "kahan_ratio": (kahan_err / kb) if kb > 0 else 0.0,
            "naive_bound": nb,
            "kahan_bound": kb,
            "naive_within_bound": naive_err <= nb + 1e-300,
            "kahan_within_bound": kahan_err <= kb + 1e-300,
        }

    @classmethod
    def calibrate(cls, corpora: Sequence[Sequence[float]], label: str = "calibration") -> Dict[str, Any]:
        """Measure the empirical maximum of err/(K*eps*n*sum|x|) for naive/Kahan."""
        naive_ratios, kahan_ratios = [], []
        for idx, values in enumerate(corpora):
            a = cls.audit_values(values, label=f"{label}[{idx}]")
            if a["n"] > 1 and a["sum_abs"] > 0:
                naive_ratios.append(a["naive_ratio"])
                kahan_ratios.append(a["kahan_ratio"])
        out = {
            "label": label,
            "n_corpora": len(corpora),
            "naive_max_ratio": max(naive_ratios) if naive_ratios else 0.0,
            "kahan_max_ratio": max(kahan_ratios) if kahan_ratios else 0.0,
            "K_NAIVE_hardcoded": cls.K_NAIVE,
            "K_KAHAN_hardcoded": cls.K_KAHAN,
        }
        out["naive_bound_holds"] = out["naive_max_ratio"] <= cls.K_NAIVE
        out["kahan_bound_holds"] = out["kahan_max_ratio"] <= cls.K_KAHAN
        out["eps64"] = cls.EPS64
        return out

    @classmethod
    def scan_total(
        cls,
        shard_key_sets: Sequence[Sequence[str]],
        values: Dict[str, float],
        *,
        mode: str = "naive",
    ) -> float:
        """Total of a sum-sensitive quantity as it would be aggregated.

        ``mode="naive"``: sum within each shard (in the shard's key order),
        then sum the shard partial sums in shard-index order.
        ``mode="kahan"``: Kahan-compensated within shard, Kahan across shards.
        ``mode="exact"``: ``math.fsum`` over every point (reference).
        """
        if mode == "exact":
            return math.fsum(values[k] for keys in shard_key_sets for k in keys)
        partials = []
        for keys in shard_key_sets:
            arr = [values[k] for k in keys]
            partials.append(
                FloatAudit.naive_sum(arr) if mode == "naive" else FloatAudit.kahan_sum(arr)
            )
        return (
            FloatAudit.naive_sum(partials)
            if mode == "naive"
            else FloatAudit.kahan_sum(partials)
        )

    @classmethod
    def shard_count_invariance(
        cls,
        values: Dict[str, float],
        shard_key_sets_by_n: Dict[int, Sequence[Sequence[str]]],
        *,
        reference_n: int = 1,
    ) -> Dict[str, Any]:
        """Audit that totals are invariant to the number of shards.

        For every ``n`` present, computes naive / Kahan / exact totals and the
        relative deviation against ``reference_n`` (bit-level comparison of the
        aggregated *results* is covered by the aggregation tests; here we bound
        the float-level drift of the derived sums).
        """
        totals = {}
        for n, keysets in shard_key_sets_by_n.items():
            totals[n] = {
                "naive": cls.scan_total(keysets, values, mode="naive"),
                "kahan": cls.scan_total(keysets, values, mode="kahan"),
                "exact": cls.scan_total(keysets, values, mode="exact"),
            }

        abs_sum = float(sum(abs(v) for v in values.values()))
        report: Dict[str, Any] = {
            "reference_n": reference_n,
            "sum_abs": abs_sum,
            "n_shards": sorted(shard_key_sets_by_n),
            "totals": totals,
            "points": len(values),
        }

        ref = totals[reference_n]
        per_n = {}
        for n, t in totals.items():
            denom = abs(ref["exact"]) if ref["exact"] != 0 else 1.0
            per_n[str(n)] = {
                "naive_rel_dev_vs_ref": abs(t["naive"] - ref["exact"]) / denom,
                "kahan_rel_dev_vs_ref": abs(t["kahan"] - ref["exact"]) / denom,
                "naive_vs_kahan_rel_dev": (
                    abs(t["naive"] - t["kahan"]) / denom
                ),
                "naive_abs_err": abs(t["naive"] - t["exact"]),
                "kahan_abs_err": abs(t["kahan"] - t["exact"]),
                "naive_bound": cls.bound_naive(len(values), abs_sum),
                "kahan_bound": cls.bound_kahan(abs_sum),
            }
        report["per_n"] = per_n
        return report


# ---------------------------------------------------------------------------
# Compute resolution + shard runner (the thing the cluster command runs)
# ---------------------------------------------------------------------------

def synthetic_scan_energy(ck: str) -> Dict[str, float]:
    """Deterministic float-arithmetic fixture for one parameter point.

    **Not a physics model** — a synthetic 101-term alternating series with
    strong cancellation, used to make summation-order effects observable.
    The key must be JSON with numeric fields ``x`` and ``y``.
    """
    key = json.loads(ck)
    x = float(key["x"])
    y = float(key["y"])
    energy = 0.0
    for k in range(101):
        energy += (x ** 1.5) * math.exp(-0.13 * k) * (1.0 if k % 2 == 0 else -1.0)
        energy += y * math.sin(0.7 * k + 0.1) * 1e-3
    kpi = energy / (1.0 + 0.5 * abs(x))
    return {"energy": energy, "kpi": kpi}


def resolve_compute(ref: str) -> Callable[[str], Any]:
    """Resolve ``"module:function"`` (or ``"module.function"``) to a callable."""
    if ":" in ref:
        mod_name, fn_name = ref.split(":", 1)
    elif "." in ref:
        mod_name, fn_name = ref.rsplit(".", 1)
    else:
        raise ShardChannelError(f"compute ref must be 'module:function', got {ref!r}")
    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ShardChannelError(f"cannot import compute module {mod_name!r}") from exc
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ShardChannelError(f"compute {ref!r} is not callable")
    return fn


def run_shard(
    plan_path: os.PathLike | str,
    shard_index: int,
    out_dir: os.PathLike | str,
    *,
    compute: Optional[Callable[[str], Any]] = None,
) -> Path:
    """Execute one shard of a plan and write its artifact.

    This is the body of the cluster command: the command template rendered by
    :func:`render_shard_cmd` passes ``--plan ... --i I --n N --out DIR``.
    ``shard_index`` here is the **zero-based** index; the CLI translates the
    rendered ordinal (``1..n`` for ``one_based`` plans) before calling this.
    """
    plan = load_plan(plan_path)
    n = int(plan["n_shards"])
    if not (0 <= shard_index < n):
        raise ShardChannelError(f"shard_index {shard_index} out of range for n={n}")
    fn = compute if compute is not None else resolve_compute(plan["compute"])
    spec = plan_specs(plan)[shard_index]
    results = {}
    for ck in spec.param_keys:
        value = fn(ck)
        if not isinstance(value, dict):
            raise ShardChannelError(
                f"compute must return a mapping for key {ck!r}, got {type(value).__name__}"
            )
        results[ck] = value
    return write_artifact(out_dir, plan=plan, shard_index=shard_index, results=results)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(
        description="TensorPhysics cluster shard channel ({i}/{n} execution)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run-shard", help="execute one shard and write its artifact")
    p_run.add_argument("--plan", required=True)
    p_run.add_argument(
        "--i",
        type=int,
        required=True,
        help="shard ordinal as rendered into the command template "
        "(1..n when the plan is one_based, else 0..n-1)",
    )
    p_run.add_argument("--n", type=int, required=True)
    p_run.add_argument("--out", required=True, help="stage directory for artifacts")

    p_agg = sub.add_parser("aggregate", help="fail-closed aggregation of a stage dir")
    p_agg.add_argument("--plan", required=True)
    p_agg.add_argument("--out", required=True)
    p_agg.add_argument("--json-out", default="")

    args = parser.parse_args(argv)

    if args.cmd == "run-shard":
        plan = load_plan(args.plan)
        if args.n != int(plan["n_shards"]):
            raise SystemExit(
                f"--n {args.n} disagrees with plan n_shards {plan['n_shards']}"
            )
        index = args.i - 1 if plan.get("one_based", True) else args.i
        path = run_shard(args.plan, index, args.out)
        print(path)
        return 0

    merged = aggregate_from_dir(args.out, load_plan(args.plan))
    payload = {"n_keys": len(merged), "keys_digest": keys_digest(merged.keys())}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())