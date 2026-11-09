"""Non-intrusive polynomial chaos expansion (PCE) with analytical Sobol' indices.

Construction
------------
A truncated polynomial chaos expansion approximates the model output as

    Y ~= M(X) = sum_{alpha in A} c_alpha Psi_alpha(xi(X)),

``xi`` being the canonical iid coordinates of :mod:`tensorphysics.uq.distributions`
(``2F(x)-1`` for a Legendre basis, ``Phi^{-1}(F(x))`` for Hermite).  The
multi-index set ``A`` is truncated by total degree ``p`` optionally with a
hyperbolic ``q``-norm (see :mod:`tensorphysics.uq.basis`).

Because the basis is orthonormal, the coefficients *are* the variance
decomposition:

    Var(Y) = sum_{alpha != 0} c_alpha^2,   S1_i = sum_{alpha in A_i} c_alpha^2 / Var(Y),
    ST_i = sum_{alpha: alpha_i > 0} c_alpha^2 / Var(Y),

with ``A_i = {alpha : alpha_i > 0, alpha_{j != i} = 0}``.  This is the reason
PCE is the cheap, noise-free route to Sobol' indices: the indices come out of
one least-squares problem instead of ``N(d+2)`` model evaluations, and their
uncertainty can be propagated analytically (see ``coefficient_ci``).

Two honest caveats, both reported by the gate
---------------------------------------------
* the LOO error is computed for the *selected* sparse support, so it is a
  post-selection estimate (Blatman & Sudret 2011 make the same assumption);
* the coefficient covariance from ``sigma^2 (A^T A)^{-1}`` measures regression
  noise only -- truncation bias is what ``eps_LOO`` measures, and the two are
  reported separately, never merged into one "uncertainty".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .basis import LEGENDRE, MultiIndexSet, orthonormal_basis
from .indices import SobolIndices
from .regression import OMPPath, omp_path

__all__ = [
    "PCEFit",
    "PCESweep",
    "fit_pce",
    "fit_pce_sweep",
    "spearman_rank_correlation",
]


@dataclass
class PCEFit:
    """One fitted PCE at a fixed total degree."""

    index_set: MultiIndexSet
    coefficients: np.ndarray
    selected: np.ndarray
    degree: int
    family: str
    n_samples: int
    loo_error: float
    r2: float
    residual_variance: float
    output_variance: float
    coefficient_covariance: np.ndarray | None = None
    names: Sequence[str] = field(default_factory=tuple)

    # -- prediction -------------------------------------------------------
    def predict(self, scores: np.ndarray) -> np.ndarray:
        A = orthonormal_basis(scores, self.index_set)
        return A[:, self.selected] @ self.coefficients

    # -- diagnostics ------------------------------------------------------
    @property
    def basis_size(self) -> int:
        return self.index_set.size

    @property
    def n_terms(self) -> int:
        return int(len(self.selected))

    @property
    def mean(self) -> float:
        const = np.flatnonzero(self.index_set.orders() == 0)
        if const.size and np.isin(const[0], self.selected):
            k = int(np.flatnonzero(self.selected == const[0])[0])
            return float(self.coefficients[k])
        return 0.0

    @property
    def surrogate_variance(self) -> float:
        """``sum_{alpha != 0} c_alpha^2`` -- the variance the surrogate resolves."""
        orders = self.index_set.orders()[self.selected]
        terms = self.coefficients[orders > 0]
        return float(np.sum(terms**2))

    def term_variances(self) -> np.ndarray:
        return self.coefficients**2

    # -- Sobol' indices ---------------------------------------------------
    def sobol(
        self,
        *,
        ci_level: float = 0.95,
        n_draws: int = 2000,
        seed: int = 0,
        second_order: bool = True,
        max_pairs: int = 15,
    ) -> SobolIndices:
        """Analytical Sobol' indices of the PCE with coefficient-covariance CIs."""
        idx = self.index_set.array[self.selected]
        var = self.coefficients**2
        total = float(np.sum(var[idx.sum(axis=1) > 0]))
        dim = self.index_set.dim
        names = list(self.names) if len(self.names) == dim else [f"x{i + 1}" for i in range(dim)]
        if total <= 0.0:
            z = np.zeros(dim)
            return SobolIndices(names=names, S1=z, ST=z.copy(), variance=0.0, estimator="pce", ci_level=ci_level)

        def _decompose(coef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
            v = coef**2
            tot = float(np.sum(v[idx.sum(axis=1) > 0])) or 1.0
            s1 = np.zeros(dim)
            st = np.zeros(dim)
            for j in range(dim):
                st[j] = float(np.sum(v[idx[:, j] > 0])) / tot
                first = np.all(
                    (idx > 0) == (np.eye(1, dim, j, dtype=int)), axis=1
                )
                s1[j] = float(np.sum(v[first])) / tot
            s2: Dict[str, float] = {}
            if second_order and dim >= 2:
                pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
                orders = idx.sum(axis=1)
                for i, j in pairs[:max_pairs]:
                    others = [k for k in range(dim) if k not in (i, j)]
                    # closed second-order index: variance of the conditional
                    # expectation E[Y | x_i, x_j] minus the two first-order
                    # contributions, i.e. the sum over every term whose support
                    # is a subset of {i, j}
                    mask = np.ones(idx.shape[0], dtype=bool)
                    if others:
                        mask = np.all(idx[:, others] == 0, axis=1)
                    mask &= orders > 0
                    vij = float(np.sum(v[mask])) / tot
                    s2[f"{names[i]}-{names[j]}"] = float(vij - s1[i] - s1[j])
            return s1, st, s2

        S1, ST, S2 = _decompose(self.coefficients)
        low_q = (1.0 - ci_level) / 2.0
        out = SobolIndices(
            names=names,
            S1=S1,
            ST=ST,
            variance=total,
            estimator="pce",
            ci_level=ci_level,
            S2=S2,
            meta={
                "degree": int(self.degree),
                "family": self.family,
                "n_terms": self.n_terms,
                "basis_size": self.basis_size,
                "loo_error": float(self.loo_error),
                "r2": float(self.r2),
                "ci_method": "coefficient-covariance Monte-Carlo (regression noise only)",
                "n_draws": int(n_draws),
            },
        )
        if self.coefficient_covariance is not None and n_draws > 0:
            try:
                L = np.linalg.cholesky(self.coefficient_covariance + 1e-300 * np.eye(self.n_terms))
                rng = np.random.default_rng(int(seed))
                draws = self.coefficients[:, None] + L @ rng.standard_normal((self.n_terms, int(n_draws)))
                s1_d = np.empty((int(n_draws), dim))
                st_d = np.empty((int(n_draws), dim))
                for k in range(int(n_draws)):
                    a, b, _ = _decompose(draws[:, k])
                    s1_d[k] = a
                    st_d[k] = b
                out.S1_low = np.quantile(s1_d, low_q, axis=0)
                out.S1_high = np.quantile(s1_d, 1.0 - low_q, axis=0)
                out.ST_low = np.quantile(st_d, low_q, axis=0)
                out.ST_high = np.quantile(st_d, 1.0 - low_q, axis=0)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate covariance
                out.meta["ci_method"] = "unavailable (coefficient covariance not positive definite)"
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "degree": int(self.degree),
            "family": self.family,
            "n_samples": int(self.n_samples),
            "basis_size": int(self.basis_size),
            "n_terms": int(self.n_terms),
            "loo_error": float(self.loo_error),
            "r2": float(self.r2),
            "output_variance": float(self.output_variance),
            "surrogate_variance": float(self.surrogate_variance),
            "residual_variance": float(self.residual_variance),
            "index_set": self.index_set.as_dict(),
        }


@dataclass
class PCESweep:
    """Degree sweep with the selected model and the convergence record."""

    best: PCEFit
    records: List[PCEFit] = field(default_factory=list)
    requested_degrees: Sequence[int] = field(default_factory=tuple)

    def loo_curve(self) -> List[Tuple[int, float]]:
        return [(int(f.degree), float(f.loo_error)) for f in self.records]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best.as_dict(),
            "loo_curve": [
                {
                    "degree": int(f.degree),
                    "basis_size": int(f.basis_size),
                    "n_terms": int(f.n_terms),
                    "loo_error": float(f.loo_error),
                    "r2": float(f.r2),
                }
                for f in self.records
            ],
            "requested_degrees": [int(d) for d in self.requested_degrees],
        }


def _fit_one(
    scores: np.ndarray,
    y: np.ndarray,
    index_set: MultiIndexSet,
    *,
    names: Sequence[str],
    max_terms: int | None,
    complexity_penalty: float,
) -> PCEFit | None:
    A = orthonormal_basis(scores, index_set)
    if max_terms is None:
        # the greedy path only has to reach the sparse support, not the full
        # basis: cap it so the O(n * basis * steps) work stays bounded
        max_terms = int(min(index_set.size - 1, max(40, 8 * int(index_set.degree))))
    path: OMPPath = omp_path(A, y, max_terms=max_terms, complexity_penalty=complexity_penalty)
    if path.best is None:  # pragma: no cover - defensive
        return None
    sel = np.asarray(path.best.selected, dtype=int)
    X = A[:, sel]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, m = X.shape
    rss = float(resid @ resid)
    ybar = float(y.mean())
    sst = float(np.sum((y - ybar) ** 2))
    dof = max(n - m, 1)
    sigma2 = rss / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:  # pragma: no cover - singular Gram
        cov = None
    return PCEFit(
        index_set=index_set,
        coefficients=coef,
        selected=sel,
        degree=int(index_set.degree),
        family=index_set.family,
        n_samples=int(n),
        loo_error=float(path.best.loo_error),
        r2=1.0 - rss / sst if sst > 0 else float("nan"),
        residual_variance=float(rss / max(n - 1, 1)),
        output_variance=float(sst / max(n - 1, 1)),
        coefficient_covariance=cov,
        names=list(names),
    )


def fit_pce(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    names: Sequence[str],
    family: str = LEGENDRE,
    degree: int = 4,
    q: float = 1.0,
    interaction: int | None = None,
    max_terms: int | None = None,
    complexity_penalty: float = 0.0,
) -> PCEFit:
    """Fit a single PCE at the given total degree (sparse support via OMP)."""
    idx = MultiIndexSet.build(np.asarray(scores).shape[1], degree, family, q=q, interaction=interaction)
    fit = _fit_one(
        np.asarray(scores, dtype=float),
        np.asarray(y, dtype=float).ravel(),
        idx,
        names=names,
        max_terms=max_terms,
        complexity_penalty=complexity_penalty,
    )
    if fit is None:  # pragma: no cover - defensive
        raise RuntimeError("PCE fit failed (OMP produced no model)")
    return fit


def fit_pce_sweep(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    names: Sequence[str],
    family: str = LEGENDRE,
    degrees: Iterable[int] = (2, 3, 4, 5, 6, 8, 10, 12),
    q: float = 1.0,
    interaction: int | None = None,
    max_terms: int | None = None,
    min_oversampling: float = 1.5,
    complexity_penalty: float = 0.0,
) -> PCESweep:
    """Sweep the total degree, keep the model with the smallest LOO error.

    Candidate degrees whose basis is too large for the available sample count
    (``n < min_oversampling * basis_size``) are skipped, so the sweep degrades to
    a lower degree instead of returning a regression that cannot be trusted.
    """
    S = np.asarray(scores, dtype=float)
    yv = np.asarray(y, dtype=float).ravel()
    n = S.shape[0]
    records: List[PCEFit] = []
    used: List[int] = []
    for degree in degrees:
        idx = MultiIndexSet.build(S.shape[1], int(degree), family, q=q, interaction=interaction)
        if n < min_oversampling * idx.size:
            continue
        fit = _fit_one(S, yv, idx, names=names, max_terms=max_terms, complexity_penalty=complexity_penalty)
        if fit is None:
            continue
        records.append(fit)
        used.append(int(degree))
    if not records:
        raise ValueError(
            f"no degree in the sweep fits inside n={n} samples "
            f"(min_oversampling={min_oversampling}); increase the sample count"
        )
    best = min(records, key=lambda f: (f.loo_error, f.n_terms))
    return PCESweep(best=best, records=records, requested_degrees=used)


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy dependency, ties averaged)."""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("need two equally sized vectors with >= 2 entries")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        # average ties
        vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            sums = np.zeros(vals.size)
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra**2) * np.sum(rb**2)))
    return float(np.sum(ra * rb) / denom) if denom > 0 else float("nan")