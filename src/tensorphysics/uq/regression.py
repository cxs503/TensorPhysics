"""Sparse polynomial regression by orthogonal matching pursuit with LOO selection.

Why not plain least squares on the full basis?  The number of terms of a
total-degree-``p`` basis in ``d`` dimensions grows as ``C(p+d, d)``: at ``d=3``,
``p=14`` that is 680 terms, and an ordinary least-squares fit needs several
thousand model evaluations *per candidate degree*.  Because engineering models
are usually of low effective dimension, most of those coefficients are zero.
Orthogonal matching pursuit (OMP) recovers the sparse support and pays the cost
in the number of retained terms instead of the basis size.

Model selection uses the **leave-one-out (LOO) error of the selected model**,
computable in closed form from the hat matrix diagonal,

    eps_LOO = sum_i (r_i / (1 - h_ii))^2 / sum_i (y_i - mean(y))^2 ,

so no extra model evaluations are needed.  This is the Blatman-Sudret
(2011) construction with OMP substituted for least-angle regression: both are
greedy support-selection paths, and both are heuristic with respect to the
post-selection nature of the error estimate -- reported as such.

The constant term is always retained; the path is built on the *centered*
design so that ``h_ii = 1/n + sum_k q_ik^2`` with ``q`` the orthonormal basis of
the centered selected columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

__all__ = ["OMPPath", "OMPModel", "omp_path", "loo_error_from_hat"]


def loo_error_from_hat(residuals: np.ndarray, hat_diag: np.ndarray) -> float:
    """LOO sum of squares ``sum_i (r_i / (1 - h_ii))^2``."""
    r = np.asarray(residuals, dtype=float)
    h = np.clip(np.asarray(hat_diag, dtype=float), 0.0, 1.0 - 1e-12)
    return float(np.sum((r / (1.0 - h)) ** 2))


@dataclass
class OMPModel:
    """One point on the OMP path: selected columns, coefficients, LOO error."""

    selected: np.ndarray  # column indices into the design matrix (ASCENDING)
    coefficients: np.ndarray  # coefficients of the selected columns
    loo_sse: float
    loo_error: float  # loo_sse / sum (y - mean)^2
    r2: float
    n_samples: int

    @property
    def n_terms(self) -> int:
        return int(len(self.selected))

    def predict(self, design: np.ndarray) -> np.ndarray:
        return design[:, self.selected] @ self.coefficients

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_terms": self.n_terms,
            "loo_error": float(self.loo_error),
            "r2": float(self.r2),
            "n_samples": int(self.n_samples),
        }


@dataclass
class OMPPath:
    """The full greedy path, ordered by increasing number of retained terms."""

    models: List[OMPModel] = field(default_factory=list)
    best: OMPModel | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": [m.as_dict() for m in self.models],
            "best_terms": self.best.n_terms if self.best else 0,
            "best_loo_error": float(self.best.loo_error) if self.best else float("nan"),
        }


def omp_path(
    design: np.ndarray,
    y: np.ndarray,
    *,
    max_terms: int | None = None,
    tol: float = 1e-10,
    selection: str = "min_loo",
    complexity_penalty: float = 0.0,
) -> OMPPath:
    """Greedy OMP path with the closed-form LOO error at every step.

    Column 0 of ``design`` is assumed to be the constant term; it is always
    retained and never selected twice.  ``selection="min_loo"`` takes the path
    step with the smallest LOO error; ``complexity_penalty`` adds
    ``complexity_penalty * terms / n`` to that error before comparing (a
    parsimony tie-break).
    """
    A = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, m = A.shape
    if A.shape[0] != y.size:
        raise ValueError("design and y disagree on the number of samples")
    if m < 1:
        raise ValueError("design needs at least the constant column")

    ybar = float(y.mean())
    yc = y - ybar
    sst = float(np.sum(yc**2))
    Ac = A[:, 1:] - A[:, 1:].mean(axis=0, keepdims=True)
    col_norm = np.sqrt(np.sum(Ac**2, axis=0))
    available = col_norm > 1e-14 * max(1.0, float(np.abs(Ac).max(initial=0.0)))

    budget = int(m - 1) if max_terms is None else min(int(max_terms), m - 1)
    budget = max(budget, 0)

    Q: List[np.ndarray] = []  # orthonormalised centered selected columns
    sel: List[int] = []
    q2 = np.zeros(n, dtype=float)  # sum_k q_k^2
    pred = np.zeros(n, dtype=float)
    resid = yc.copy()
    path = OMPPath()
    # R is the upper-triangular factor of the Gram-Schmidt process on the
    # *centered* selected columns; z = Q^T yc.  Solving R beta = z gives the
    # coefficients in the original (selected) basis in O(k^2), so the whole
    # greedy path costs O(n m k) instead of one least-squares solve per step.
    R = np.zeros((budget + 1, budget + 1), dtype=float)
    z = np.zeros(budget + 1, dtype=float)

    for _step in range(budget + 1):
        # LOO of the current selection (constant term always in).
        hat = 1.0 / n + q2
        r = yc - pred
        loo_sse = loo_error_from_hat(r, hat)
        loo_err = loo_sse / sst if sst > 0 else float("inf")
        rss = float(np.sum(r**2))
        selected = np.array([0] + [s + 1 for s in sel], dtype=int)
        k = len(sel)
        if k:
            beta = np.linalg.solve(R[:k, :k], z[:k])
        else:
            beta = np.zeros(0, dtype=float)
        model = OMPModel(
            selected=selected,
            coefficients=np.concatenate([[ybar], beta]),
            loo_sse=loo_sse,
            loo_error=loo_err,
            r2=1.0 - rss / sst if sst > 0 else float("nan"),
            n_samples=n,
        )
        path.models.append(model)

        # greedy direction
        if _step >= budget or not available.any() or sst <= 0:
            break
        corr = np.abs(Ac.T @ resid)
        corr[~available] = -np.inf
        if sel:
            corr[np.array(sel, dtype=int)] = -np.inf
        j = int(np.argmax(corr))
        if not np.isfinite(corr[j]) or corr[j] <= tol * max(1.0, float(np.abs(yc).sum())):
            break
        q = Ac[:, j].copy()
        for _reorth in range(2):  # classical GS with one reorthogonalisation
            for idx_q, qk in enumerate(Q):
                coef = qk @ q
                q -= coef * qk
                R[idx_q, k] += coef
        nq = float(np.linalg.norm(q))
        if nq <= 1e-14 * max(1.0, float(col_norm[j])):
            available[j] = False
            continue
        q /= nq
        R[k, k] = nq
        Q.append(q)
        sel.append(j)
        z[k] = float(q @ yc)
        q2 += q * q
        pred += z[k] * q
        resid = yc - pred

    scored = [
        (m.loo_error + complexity_penalty * m.n_terms / max(m.n_samples, 1), m)
        for m in path.models
    ]
    scored.sort(key=lambda item: (item[0], item[1].n_terms))
    path.best = scored[0][1] if scored else None
    return path