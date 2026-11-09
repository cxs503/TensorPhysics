"""Monte-Carlo (Saltelli/Jansen) Sobol' estimators -- the independent cross-check.

The PCE indices are analytic but conditional on a chosen basis and support; the
only way to know that the surrogate respects the actual model is to estimate the
same indices from *model evaluations alone*.  This module implements the
Saltelli (2010) design with the Jansen (1999) estimators

    S1_i = mean_A[ f(B) * (f(A_B^i) - f(A)) ] / Var(f),
    ST_i = mean_A[ (f(A) - f(A_B^i))^2 ] / (2 Var(f)),

and gives every index a 95 % percentile interval from a bootstrap over the base
samples (resampling the ``n_base`` base points, which is the statistically
consistent unit because ``A``, ``B`` and all ``A_B^i`` share the base index).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .indices import SobolIndices
from .sampling import saltelli_design, SampleSet

__all__ = ["saltelli_indices", "jansen_estimators", "bootstrap_ci"]


def jansen_estimators(
    fA: np.ndarray, fB: np.ndarray, fAB: Sequence[np.ndarray], *, idx: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point estimates ``(S1, ST, Var)`` of the Jansen/Saltelli estimators.

    ``idx`` optionally restricts the computation to a subset of base samples
    (used by the bootstrap).
    """
    A = np.asarray(fA, dtype=float).ravel()
    B = np.asarray(fB, dtype=float).ravel()
    ABs = [np.asarray(f, dtype=float).ravel() for f in fAB]
    if idx is not None:
        # slice every array exactly once: a double application of ``idx`` would
        # silently permute the ``A_B^i`` samples against ``A`` and ``B`` and
        # wreck the bootstrap distribution (it is centred on the estimate only
        # when the pairing is preserved)
        A = A[idx]
        B = B[idx]
        ABs = [f[idx] for f in ABs]
    d = len(ABs)
    pooled = np.concatenate([A, B])
    var = float(np.var(pooled, ddof=1))
    s1 = np.empty(d)
    st = np.empty(d)
    if var <= 0.0:  # pragma: no cover - degenerate model
        return np.zeros(d), np.zeros(d), 0.0
    for i, AB in enumerate(ABs):
        ABf = AB
        s1[i] = float(np.mean(B * (ABf - A)) / var)
        st[i] = float(np.mean((A - ABf) ** 2) / (2.0 * var))
    return s1, st, var


def bootstrap_ci(
    fA: np.ndarray,
    fB: np.ndarray,
    fAB: Sequence[np.ndarray],
    *,
    n_boot: int = 400,
    ci_level: float = 0.95,
    seed: int = 0,
    block: int = 1,
) -> Dict[str, np.ndarray]:
    """Percentile bootstrap intervals for the Jansen estimators.

    ``block`` allows a moving-block bootstrap: the Saltelli design reuses the
    same base point across ``d+2`` evaluations, so a block bootstrap over
    contiguous base points preserves that within-point dependence more honestly
    than an iid resample when the sampler is a low-discrepancy sequence.
    """
    fA = np.asarray(fA, dtype=float).ravel()
    n = fA.size
    rng = np.random.default_rng(int(seed))
    s1_d = np.empty((int(n_boot), len(fAB)))
    st_d = np.empty((int(n_boot), len(fAB)))
    n_blocks = max(n // max(block, 1), 1)
    for k in range(int(n_boot)):
        if block <= 1:
            idx = rng.integers(0, n, size=n)
        else:
            starts = rng.integers(0, max(n - block, 1), size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        s1_d[k], st_d[k], _ = jansen_estimators(fA, fB, fAB, idx=idx)
    low = (1.0 - ci_level) / 2.0
    return {
        "S1_low": np.quantile(s1_d, low, axis=0),
        "S1_high": np.quantile(s1_d, 1.0 - low, axis=0),
        "ST_low": np.quantile(st_d, low, axis=0),
        "ST_high": np.quantile(st_d, 1.0 - low, axis=0),
    }


def saltelli_indices(
    model,
    evaluate: Callable[[np.ndarray], np.ndarray],
    *,
    n_base: int = 1024,
    method: str = "sobol",
    seed: int = 0,
    n_boot: int = 400,
    ci_level: float = 0.95,
    block: int = 1,
) -> tuple[SobolIndices, Dict[str, Any]]:
    """Saltelli/Jansen Sobol' indices of ``evaluate`` under ``model`` with bootstrap CIs.

    Returns the index set and the design/evaluation provenance.
    """
    design = saltelli_design(model, n_base, method=method, seed=seed)
    fA = np.asarray(evaluate(design["A"]), dtype=float).ravel()
    fB = np.asarray(evaluate(design["B"]), dtype=float).ravel()
    fAB = [np.asarray(evaluate(X), dtype=float).ravel() for X in design["AB"]]
    if not (np.all(np.isfinite(fA)) and np.all(np.isfinite(fB)) and all(np.all(np.isfinite(f)) for f in fAB)):
        raise ValueError("Saltelli estimator received non-finite model output")
    s1, st, var = jansen_estimators(fA, fB, fAB)
    ci = bootstrap_ci(fA, fB, fAB, n_boot=n_boot, ci_level=ci_level, seed=seed, block=block)
    indices = SobolIndices(
        names=list(model.names),
        S1=s1,
        ST=st,
        variance=var,
        estimator="saltelli-jansen",
        ci_level=ci_level,
        S1_low=ci["S1_low"],
        S1_high=ci["S1_high"],
        ST_low=ci["ST_low"],
        ST_high=ci["ST_high"],
        meta={
            "n_base": int(n_base),
            "n_boot": int(n_boot),
            "bootstrap_block": int(block),
            "sampler": method,
            "seed": int(seed),
            "design_digest_sha256": design["digest"],
            "evaluations": int(design["evaluations"]),
            "ab_max_abs_column_correlation": float(design.get("ab_max_abs_column_correlation", float("nan"))),
            "ci_method": "percentile bootstrap over base samples",
        },
    )
    return indices, design