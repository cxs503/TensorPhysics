"""Small statistics helpers used by the criteria metrology (stdlib only)."""
from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["agreement", "cohen_kappa", "confusion_matrix"]


def agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """Raw agreement rate of two label sequences.

    >>> agreement(["x", "y", "x"], ["x", "y", "y"])
    0.6666666666666666
    >>> agreement([], [])
    0.0
    """
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def confusion_matrix(a: Sequence[str], b: Sequence[str], labels: Iterable[str] | None = None) -> dict[str, dict[str, int]]:
    """Counts ``matrix[gold][predicted]`` over the given label vocabulary."""
    vocab = list(labels) if labels is not None else sorted(set(a) | set(b))
    matrix = {g: {p: 0 for p in vocab} for g in vocab}
    for x, y in zip(a, b):
        matrix.setdefault(x, {p: 0 for p in vocab}).setdefault(y, 0)
        matrix[x][y] += 1
    return matrix


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Iterable[str] | None = None) -> float:
    """Cohen's kappa for two annotators.

    >>> round(cohen_kappa(["x", "y"], ["x", "y"]), 6)
    1.0
    >>> cohen_kappa([], [])
    0.0
    """
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n = len(a)
    if n == 0:
        return 0.0
    vocab = list(labels) if labels is not None else sorted(set(a) | set(b))
    matrix = confusion_matrix(a, b, vocab)
    p_obs = sum(matrix[g][g] for g in vocab) / n
    p_exp = sum((sum(matrix[g].values()) / n) * (sum(matrix[h][g] for h in matrix) / n) for g in vocab)
    if p_exp >= 1.0:
        return 1.0 if p_obs >= 1.0 else 0.0
    return (p_obs - p_exp) / (1.0 - p_exp)