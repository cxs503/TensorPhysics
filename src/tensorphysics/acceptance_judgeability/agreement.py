"""Agreement statistics and deterministic stratified sampling.

The census claims a machine classification; that claim is only worth something if
it agrees with a human reading of the same clauses. This module supplies the two
pieces such a check needs, with no randomness that could drift between runs:

* :func:`stratified_sample` -- hash-ordered, per-stratum selection, so the same
  corpus and seed always yields the same 100 clauses;
* :func:`cohen_kappa` / :func:`confusion_matrix` -- multi-class agreement for an
  arbitrary label set (here the four judgeability classes).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence


def _order_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def stratified_sample(
    items: Sequence[Any],
    *,
    stratum: Callable[[Any], str],
    ident: Callable[[Any], str],
    n: int,
    seed: str = "p106w3a2-n08",
    min_per_stratum: int = 1,
) -> list[Any]:
    """Deterministically pick ``n`` items, proportionally across strata.

    Selection inside a stratum is by ``sha256(seed|ident)`` order, so the sample is
    stable across processes, machines and Python versions (no ``random`` state).
    """
    if n <= 0 or not items:
        return []
    buckets: dict[str, list[Any]] = {}
    for item in items:
        buckets.setdefault(str(stratum(item)), []).append(item)
    for key in buckets:
        buckets[key].sort(key=lambda it: _order_key(seed, str(ident(it))))

    total = len(items)
    planned: dict[str, int] = {}
    for key, bucket in buckets.items():
        share = len(bucket) / total
        planned[key] = max(min_per_stratum, int(round(share * n)))
        planned[key] = min(planned[key], len(bucket))

    # trim/grow to exactly n without dropping a stratum below one item
    while sum(planned.values()) > n:
        key = max((k for k in planned if planned[k] > min_per_stratum), key=lambda k: planned[k], default=None)
        if key is None:
            break
        planned[key] -= 1
    while sum(planned.values()) < n:
        key = max(
            (k for k in planned if planned[k] < len(buckets[k])),
            key=lambda k: len(buckets[k]) - planned[k],
            default=None,
        )
        if key is None:
            break
        planned[key] += 1

    out: list[Any] = []
    for key in sorted(buckets):
        out.extend(buckets[key][: planned[key]])
    out.sort(key=lambda it: _order_key(seed, str(ident(it))))
    return out


def confusion_matrix(
    truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, dict[str, int]]:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    labs = list(labels) if labels is not None else sorted(set(truth) | set(predicted))
    m = {a: {b: 0 for b in labs} for a in labs}
    for t, p in zip(truth, predicted):
        m.setdefault(t, {b: 0 for b in labs})[p] = m.setdefault(t, {}).get(p, 0) + 1
    return m


def cohen_kappa(truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Cohen's kappa for arbitrary label sets (``1.0`` perfect, ``0.0`` chance)."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    n = len(truth)
    if n == 0:
        raise ValueError("kappa undefined for an empty sample")
    labs = list(labels) if labels is not None else sorted(set(truth) | set(predicted))
    po = sum(1 for t, p in zip(truth, predicted) if t == p) / n
    ct, cp = Counter(truth), Counter(predicted)
    pe = sum((ct.get(lab, 0) / n) * (cp.get(lab, 0) / n) for lab in labs)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def per_class_agreement(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, dict[str, float]]:
    """Recall/precision per label, for the report's error analysis."""
    labs = sorted(set(truth) | set(predicted))
    out: dict[str, dict[str, float]] = {}
    for lab in labs:
        tp = sum(1 for t, p in zip(truth, predicted) if t == lab and p == lab)
        support = sum(1 for t in truth if t == lab)
        called = sum(1 for p in predicted if p == lab)
        out[lab] = {
            "recall": (tp / support) if support else float("nan"),
            "precision": (tp / called) if called else float("nan"),
            "support": support,
            "called": called,
        }
    return out