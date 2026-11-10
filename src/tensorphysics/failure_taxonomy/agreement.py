"""Agreement statistics for the dual-annotation back-test.

Implements Cohen's kappa, observed agreement, per-class recall, a confusion
matrix, and the two negative-control transforms used by criteria C3:

* :func:`replace_bodies` -- substitute each prose body with a *same-length*
  random passage drawn from a decoy pool of non-FAIL report text;
* :func:`permute_labels` -- random re-assignment of the gold labels (seeded and
  reproducible), which must collapse any supervised fit to chance level.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Mapping, Sequence

from .taxonomy import CLASS_CODES


def observed_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two label sequences over the same items."""
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n = len(a)
    if n == 0:
        return 0.0
    po = observed_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def accuracy(pred: Sequence[str], gold: Sequence[str]) -> float:
    if len(pred) != len(gold):
        raise ValueError("label sequences differ in length")
    if not gold:
        return 0.0
    return sum(1 for p, g in zip(pred, gold) if p == g) / len(gold)


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], classes: Sequence[str] = CLASS_CODES
) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0] * len(classes) for _ in classes]
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            mat[idx[g]][idx[p]] += 1
    return mat


def per_class_recall(
    gold: Sequence[str], pred: Sequence[str], classes: Sequence[str] = CLASS_CODES
) -> dict[str, float]:
    out: dict[str, float] = {}
    for code in classes:
        total = sum(1 for g in gold if g == code)
        hit = sum(1 for g, p in zip(gold, pred) if g == code and p == code)
        out[code] = (hit / total) if total else 0.0
    return out


def permute_labels(labels: Sequence[str], seed: int = 0) -> list[str]:
    """Deterministically shuffle the label sequence (negative control C3b)."""
    out = list(labels)
    random.Random(seed).shuffle(out)
    return out


def replace_bodies(
    bodies: Sequence[str], decoys: Sequence[str], seed: int = 0
) -> list[str]:
    """Replace every body with a same-length decoy passage (negative control C3a).

    The decoy pool holds passages harvested from the same reports but carrying
    no failure cue.  For each body the closest-length decoy is taken, drawn
    deterministically from the pool.
    """
    if not decoys:
        raise ValueError("decoy pool is empty")
    by_len = sorted(decoys, key=len)
    lengths = [len(b) for b in by_len]
    rng = random.Random(seed)
    out: list[str] = []
    for body in bodies:
        target = len(body)
        pos = min(range(len(lengths)), key=lambda i: abs(lengths[i] - target))
        window = by_len[max(0, pos - 3) : min(len(by_len), pos + 4)]
        out.append(rng.choice(window))
    return out


def summarise(
    gold: Sequence[str],
    pred: Sequence[str],
    other: Sequence[str] | None = None,
) -> dict:
    """One-call metrics bundle for the report JSON."""
    out = {
        "n": len(gold),
        "accuracy": accuracy(pred, gold),
        "agreement": observed_agreement(pred, gold),
        "kappa": cohen_kappa(other, pred) if other is not None else cohen_kappa(gold, pred),
        "recall": per_class_recall(gold, pred),
        "confusion": confusion_matrix(gold, pred),
    }
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], classes: Sequence[str] = CLASS_CODES) -> float:
    scores = []
    for code in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == code and p == code)
        fp = sum(1 for g, p in zip(gold, pred) if g != code and p == code)
        fn = sum(1 for g, p in zip(gold, pred) if g == code and p != code)
        if tp == 0 and (fp or fn):
            scores.append(0.0)
            continue
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


__all__ = [
    "accuracy",
    "cohen_kappa",
    "confusion_matrix",
    "macro_f1",
    "observed_agreement",
    "per_class_recall",
    "permute_labels",
    "replace_bodies",
    "summarise",
]