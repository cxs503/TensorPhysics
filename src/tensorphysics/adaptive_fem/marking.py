"""Marking strategies for adaptive mesh refinement."""
from __future__ import annotations

import numpy as np

__all__ = ["dorfler_mark", "mark_fraction", "mark_above_mean"]


def dorfler_mark(eta: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Dörfler (bulk) marking on the *energy* (squared) indicators.

    Returns the indices of the smallest set ``M`` such that
    ``sum_{K in M} eta_K^2 >= theta * sum_K eta_K^2``.  With ``theta = 0.5``
    half of the total estimated error energy is covered by the marked set; this
    is the marking rule used in the adaptive runs reported for this module.
    """
    eta = np.asarray(eta, dtype=float)
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in (0, 1]")
    e2 = eta**2
    total = float(e2.sum())
    if total <= 0.0 or eta.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-e2, kind="stable")
    cum = np.cumsum(e2[order])
    k = int(np.searchsorted(cum, theta * total, side="left")) + 1
    return np.sort(order[:k]).astype(np.int64)


def mark_fraction(eta: np.ndarray, fraction: float = 0.3) -> np.ndarray:
    """Mark the given fraction of elements with the largest indicators."""
    eta = np.asarray(eta, dtype=float)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    n = max(1, int(np.ceil(fraction * eta.size)))
    return np.sort(np.argsort(-eta, kind="stable")[:n]).astype(np.int64)


def mark_above_mean(eta: np.ndarray) -> np.ndarray:
    """Mark every element whose indicator exceeds the mean indicator."""
    eta = np.asarray(eta, dtype=float)
    if eta.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(eta >= eta.mean()).astype(np.int64)