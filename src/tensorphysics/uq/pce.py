"""Non-intrusive polynomial-chaos surrogate and analytic Sobol' indices.

The surrogate is a total-degree expansion in an **orthonormal** tensor-product
basis (normalised Legendre polynomials for uniform marginals, probabilists'
Hermite polynomials for Gaussian/lognormal marginals)::

    y(xi) ~ sum_{alpha in A} c_alpha * prod_i phi_{alpha_i}(xi_i)

Because the basis is orthonormal with respect to the input measure, the
expansion variance and all Sobol' indices are *algebraic* in the
coefficients:

    V        = sum_{alpha != 0} c_alpha**2
    S_u      = (1/V) * sum_{alpha: supp(alpha) = u} c_alpha**2
    S_T,i    = (1/V) * sum_{alpha: alpha_i > 0}        c_alpha**2

which is the classical result of Sobol' (1993) / Sudret (2008).  No extra
model evaluations are needed to read the indices off a fitted chaos.

Sparsity
--------
The number of total-degree terms grows as ``C(d + p, p)``.  With a budget of
``N`` solver runs, ``N`` may be comparable to (or below) that number, so the
coefficients are obtained by greedy **least-angle-style forward selection**
(orthogonal matching pursuit on the orthonormal design matrix): at every step
the term most correlated with the residual is added and the active-set least
squares is refitted.  The active set is chosen by the leave-one-out error
``eps_LOO`` computed analytically from the hat matrix of the active set

    eps_LOO**2 = (1/N) sum_i ( r_i / (1 - h_ii) )**2 ,   h_ii = x_i^T (X^T X)^{-1} x_i

(the PRESS statistic), whose closed form avoids refitting ``N`` surrogates.
``eps_loo`` is reported normalised by ``std(y)`` (a dimensionless "fraction of
one output standard deviation"): values below 0.1 mean the surrogate resolves
at least 99% of the output variance, which is the surrogate-quality floor the
UQ verdict enforces.

``method="ols"`` fits all terms of the truncation (used as the reference /
regression test of the selector, and required when ``N >= 4 P``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "PCESurrogate",
    "SobolIndices",
    "basis_matrix",
    "basis_values",
    "bootstrap_sobol",
    "fit_pce",
    "hermite_orthonormal",
    "legendre_orthonormal",
    "multi_indices",
]

_HAT_FLOOR = 1.0e-12


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------
def legendre_orthonormal(x: np.ndarray, degree: int) -> np.ndarray:
    """Normalised Legendre polynomials ``sqrt(2k+1) P_k(x)`` for ``k = 0..p``.

    Orthonormal on ``[-1, 1]`` with the uniform probability measure:
    ``E[phi_k phi_l] = delta_kl``.
    """
    x = np.asarray(x, dtype=float)
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    out = np.empty((x.size, degree + 1), dtype=float)
    out[:, 0] = 1.0
    if degree >= 1:
        out[:, 1] = x
    for k in range(1, degree):
        out[:, k + 1] = ((2 * k + 1) * x * out[:, k] - k * out[:, k - 1]) / (k + 1)
    return out * np.sqrt(2.0 * np.arange(degree + 1) + 1.0)


def hermite_orthonormal(z: np.ndarray, degree: int) -> np.ndarray:
    """Normalised probabilists' Hermite polynomials ``He_k(z)/sqrt(k!)``.

    Orthonormal with respect to the standard normal measure.
    """
    z = np.asarray(z, dtype=float)
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    out = np.empty((z.size, degree + 1), dtype=float)
    out[:, 0] = 1.0
    if degree >= 1:
        out[:, 1] = z
    for k in range(1, degree):
        # probabilists' recurrence: He_{k+1} = z He_k - k He_{k-1} (no division)
        out[:, k + 1] = z * out[:, k] - k * out[:, k - 1]
    # accumulate the 1/sqrt(k!) scaling in a stable way
    scale = np.array([1.0 / math.sqrt(math.gamma(k + 1)) for k in range(degree + 1)])
    return out * scale


def basis_values(x: np.ndarray, family: str, degree: int) -> np.ndarray:
    """Dispatch to the orthonormal basis family (``legendre`` / ``hermite``)."""
    if family == "legendre":
        return legendre_orthonormal(x, degree)
    if family == "hermite":
        return hermite_orthonormal(x, degree)
    raise ValueError(f"unknown basis family {family!r}; expected 'legendre' or 'hermite'")


def multi_indices(dim: int, degree: int) -> np.ndarray:
    """All multi-indices with total degree ``<= degree``, ordered by degree.

    Row 0 is always the zero multi-index (the constant term); the ordering is
    deterministic (total degree, then lexicographic) so that the coefficient
    vector is reproducible.
    """
    if dim < 1 or degree < 0:
        raise ValueError(f"dim must be >= 1 and degree >= 0, got {dim}, {degree}")
    rows: list[tuple[int, ...]] = []

    def recurse(prefix: list[int], remaining: int, slots: int, total: int) -> None:
        if slots == 1:
            rows.append(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value, slots - 1, total)

    for total in range(degree + 1):
        recurse([], total, dim, total)
    rows.sort(key=lambda a: (sum(a), a))
    return np.asarray(rows, dtype=int)


def basis_matrix(
    coords: np.ndarray, families: Sequence[str], multi_index: np.ndarray
) -> np.ndarray:
    """Tensor-product orthonormal design matrix ``(n, P)``."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    dim = coords.shape[1]
    if len(families) != dim:
        raise ValueError(
            f"{len(families)} basis families for {dim} coordinates"
        )
    max_deg = int(multi_index.max()) if multi_index.size else 0
    per_dim = [basis_values(coords[:, k], families[k], max_deg) for k in range(dim)]
    n = coords.shape[0]
    out = np.ones((n, multi_index.shape[0]), dtype=float)
    for j, alpha in enumerate(multi_index):
        col = per_dim[0][:, alpha[0]]
        for k in range(1, dim):
            col = col * per_dim[k][:, alpha[k]]
        out[:, j] = col
    return out


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SobolIndices:
    """Sobol' sensitivity indices read off a polynomial chaos surrogate."""

    first: np.ndarray = field(repr=False)
    total: np.ndarray = field(repr=False)
    second: dict[tuple[int, int], float] = field(default_factory=dict)
    variance: float = 0.0
    mean: float = 0.0
    first_ci: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    total_ci: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    @property
    def first_sum(self) -> float:
        return float(np.sum(self.first))

    @property
    def residual_interaction(self) -> float:
        """``1 - sum_i S_i`` -- the interaction share the first-order set misses."""
        return float(1.0 - np.sum(self.first))

    def as_dict(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        d = int(self.first.size)
        names = tuple(names) if names is not None else tuple(f"x{i}" for i in range(d))
        out: dict[str, Any] = {
            "mean": self.mean,
            "variance": self.variance,
            "first": {names[i]: float(self.first[i]) for i in range(d)},
            "total": {names[i]: float(self.total[i]) for i in range(d)},
            "second": {
                f"{names[i]}*{names[j]}": float(v) for (i, j), v in self.second.items()
            },
            "first_sum": self.first_sum,
        }
        if self.first_ci is not None:
            lo, hi = self.first_ci
            out["first_ci95"] = {
                names[i]: [float(lo[i]), float(hi[i])] for i in range(d)
            }
        if self.total_ci is not None:
            lo, hi = self.total_ci
            out["total_ci95"] = {
                names[i]: [float(lo[i]), float(hi[i])] for i in range(d)
            }
        return out


@dataclass(frozen=True, eq=False)
class PCESurrogate:
    """A fitted non-intrusive polynomial-chaos surrogate."""

    degree: int
    families: tuple[str, ...]
    multi_index: np.ndarray = field(repr=False)
    coef: np.ndarray = field(repr=False)
    active: tuple[int, ...]
    eps_loo: float
    eps_loo_abs: float
    rmse: float
    r2: float
    n_samples: int
    n_terms: int
    method: str
    screen_path: tuple[tuple[int, float], ...] = ()

    # ---- evaluation -----------------------------------------------------------
    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """Surrogate value at basis coordinates ``coords`` ``(n, d)``."""
        A = basis_matrix(coords, self.families, self.multi_index)
        return A @ self.coef

    @property
    def mean(self) -> float:
        return float(self.coef[0])

    @property
    def variance(self) -> float:
        return float(np.sum(np.square(self.coef[1:])))

    @property
    def dim(self) -> int:
        return len(self.families)

    # ---- sensitivity ----------------------------------------------------------
    def sobol(self, order: int = 2) -> SobolIndices:
        """Analytic Sobol' indices up to ``order`` (1, 2, ...)."""
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        c2 = np.square(self.coef)
        c2[0] = 0.0
        variance = float(np.sum(c2))
        dim = self.dim
        if variance <= 0.0:
            zeros = np.zeros(dim)
            return SobolIndices(first=zeros, total=zeros.copy(), variance=0.0, mean=self.mean)
        support = self.multi_index > 0
        n_support = support.sum(axis=1)
        first = np.empty(dim)
        total = np.empty(dim)
        for i in range(dim):
            first[i] = float(np.sum(c2[support[:, i] & (n_support == 1)]))
            total[i] = float(np.sum(c2[support[:, i]]))
        first /= variance
        total /= variance
        second: dict[tuple[int, int], float] = {}
        if order >= 2:
            for i in range(dim):
                for j in range(i + 1, dim):
                    mask = (
                        support[:, i]
                        & support[:, j]
                        & (n_support == 2)
                    )
                    second[(i, j)] = float(np.sum(c2[mask]) / variance)
        return SobolIndices(
            first=first,
            total=total,
            second=second,
            variance=variance,
            mean=self.mean,
        )

    def as_dict(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        return {
            "method": self.method,
            "degree": self.degree,
            "families": list(self.families),
            "n_samples": self.n_samples,
            "n_terms_total": int(self.multi_index.shape[0]),
            "n_terms_active": int(len(self.active)),
            "eps_loo": self.eps_loo,
            "eps_loo_abs": self.eps_loo_abs,
            "rmse": self.rmse,
            "r2": self.r2,
            "sobol": self.sobol().as_dict(names),
        }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def _gram(A: np.ndarray, *, blas: bool = True) -> np.ndarray:
    """``A^T A`` with a platform-independent fallback.

    Some BLAS builds silently return NaN for perfectly finite inputs (observed
    on loongarch64 / OpenBLAS 0.3.27 for the ``(1024, 8)`` Gram matrix of a
    Legendre design).  A NaN in the Gram matrix poisons the whole forward
    selection path, so the result is validated and, when it is not finite,
    recomputed with an ``einsum`` reduction that never enters the BLAS kernel.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        G = A.T @ A if blas else np.einsum("ij,ik->jk", A, A)
    if not np.isfinite(G).all():
        G = np.einsum("ij,ik->jk", A, A)
    return G


def _projection(A: np.ndarray, coef: np.ndarray, *, blas: bool = True) -> np.ndarray:
    """``A @ coef`` with the same non-finite fallback as :func:`_gram`."""
    with np.errstate(invalid="ignore", over="ignore"):
        values = A @ coef if blas else np.einsum("ij,j->i", A, coef)
    if not np.isfinite(values).all():
        values = np.einsum("ij,j->i", A, coef)
    return values


def _matmul(A: np.ndarray, B: np.ndarray, *, blas: bool = True) -> np.ndarray:
    """``A @ B`` with the same non-finite fallback as :func:`_gram`."""
    with np.errstate(invalid="ignore", over="ignore"):
        out = A @ B if blas else np.einsum("ij,jk->ik", A, B)
    if not np.isfinite(out).all():
        out = np.einsum("ij,jk->ik", A, B)
    return out


def _active_fit_impl(
    A: np.ndarray, y: np.ndarray, ridge: float, *, blas: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    G = _gram(A, blas=blas)
    if ridge > 0.0:
        G = G + ridge * np.eye(G.shape[0])
    with np.errstate(invalid="ignore", over="ignore"):
        b = A.T @ y
    if not np.isfinite(b).all():
        b = np.einsum("ij,i->j", A, y)
    try:
        coef = np.linalg.solve(G, b)
        Ginv = np.linalg.solve(G, np.eye(G.shape[0]))
    except np.linalg.LinAlgError:  # pragma: no cover - rank-deficient guard
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        Ginv = np.linalg.pinv(G)
    resid = y - _projection(A, coef, blas=blas)
    hat = np.sum(_matmul(A, Ginv, blas=blas) * A, axis=1)
    denom = np.clip(1.0 - hat, _HAT_FLOOR, None)
    loo = float(np.sqrt(np.mean(np.square(resid / denom))))
    return coef, resid, loo


def _active_fit(A: np.ndarray, y: np.ndarray, ridge: float = 0.0):
    """Least squares on the active columns; returns coef, resid, loo (absolute).

    The BLAS path is tried first (fast); if any intermediate is non-finite the
    fit is repeated without BLAS reductions (see :func:`_gram`).
    """
    coef, resid, loo = _active_fit_impl(A, y, ridge, blas=True)
    if not (np.isfinite(coef).all() and np.isfinite(resid).all() and np.isfinite(loo)):
        coef, resid, loo = _active_fit_impl(A, y, ridge, blas=False)
    return coef, resid, loo


def fit_pce(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    degree: int,
    families: Sequence[str],
    method: str = "lars",
    max_terms: int | None = None,
    patience: int = 2,
    ridge: float = 1.0e-12,
) -> PCESurrogate:
    """Fit a total-degree polynomial chaos expansion to ``values``.

    Parameters
    ----------
    coords:
        ``(N, d)`` basis coordinates (see :mod:`tensorphysics.uq.sample_spec`).
    values:
        ``(N,)`` finite solver outputs.
    degree:
        Total polynomial degree of the truncation.
    families:
        Basis family per coordinate (``"legendre"`` / ``"hermite"``).
    method:
        ``"lars"`` (greedy forward selection on the LOO criterion, default) or
        ``"ols"`` (all terms; needs ``N >= 4 P`` for a meaningful PRESS).
    max_terms:
        Maximum number of active terms for the greedy selector (default
        ``min(P, max(20, P // 2))``).
    patience:
        Stop the forward path after ``patience`` consecutive non-improving
        steps.
    ridge:
        Tiny Tikhonov term that keeps the normal equations solvable when the
        active set gets close to the sample count.
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(values, dtype=float).reshape(-1)
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    if y.size != coords.shape[0]:
        raise ValueError(
            f"{y.size} values for {coords.shape[0]} coordinate rows"
        )
    if not np.all(np.isfinite(y)):
        bad = int(np.count_nonzero(~np.isfinite(y)))
        raise ValueError(
            f"{bad} non-finite samples in the training values; the caller must "
            "withhold the UQ verdict instead of fitting a surrogate to them"
        )
    n, dim = coords.shape
    if len(families) != dim:
        raise ValueError(f"{len(families)} families for {dim} coordinates")
    if method not in ("lars", "ols"):
        raise ValueError(f"unknown fitting method {method!r}")
    multi = multi_indices(dim, degree)
    A = basis_matrix(coords, families, multi)
    P = A.shape[1]
    sd = float(np.std(y))
    scale = sd if sd > 0.0 else 1.0

    if method == "ols":
        if n < P:
            raise ValueError(
                f"ols needs at least P={P} samples for degree {degree} in {dim} dims, got {n}"
            )
        coef, resid, loo_abs = _active_fit(A, y, ridge=ridge)
        active = tuple(range(P))
        path: tuple[tuple[int, float], ...] = ((P, loo_abs / scale),)
    else:
        limit = max_terms if max_terms is not None else min(P, max(20, P // 2))
        limit = int(min(limit, P, max(1, n - 1)))
        active_list = [0]
        best_active = list(active_list)
        best_loo = math.inf
        best_coef = np.zeros(P)
        path_list: list[tuple[int, float]] = []
        stall = 0
        for _step in range(limit):
            idx = np.asarray(active_list, dtype=int)
            coef_a, resid, loo_abs = _active_fit(A[:, idx], y, ridge=ridge)
            loo_rel = loo_abs / scale
            path_list.append((len(active_list), loo_rel))
            if loo_rel < best_loo:
                best_loo = loo_rel
                best_active = list(active_list)
                coef_full = np.zeros(P)
                coef_full[idx] = coef_a
                best_coef = coef_full
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break
            if len(active_list) >= limit:
                break
            with np.errstate(invalid="ignore", over="ignore"):
                corr = np.abs(A.T @ resid)
            if not np.isfinite(corr).all():
                # BLAS GEMM produced non-finite values from finite inputs
                corr = np.abs(np.einsum("ij,i->j", A, resid))
            corr[idx] = -np.inf
            nxt = int(np.argmax(corr))
            if not np.isfinite(corr[nxt]) or corr[nxt] <= 0.0:
                break
            active_list.append(nxt)
        coef = best_coef
        active = tuple(sorted(best_active))
        resid = y - A @ coef
        loo_abs = best_loo * scale
        path = tuple(path_list)

    rmse = float(np.sqrt(np.mean(np.square(resid))))
    var_y = float(np.var(y))
    r2 = float(1.0 - np.sum(np.square(resid)) / (var_y * y.size)) if var_y > 0.0 else 1.0
    return PCESurrogate(
        degree=int(degree),
        families=tuple(families),
        multi_index=multi,
        coef=coef,
        active=active,
        eps_loo=float(loo_abs / scale),
        eps_loo_abs=float(loo_abs),
        rmse=rmse,
        r2=r2,
        n_samples=int(n),
        n_terms=int(len(active)),
        method=method,
        screen_path=path,
    )


def bootstrap_sobol(
    surrogate: PCESurrogate,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 20240,
    order: int = 2,
    level: float = 0.95,
) -> SobolIndices:
    """Bootstrap confidence intervals for the surrogate Sobol' indices.

    The active multi-index set of ``surrogate`` is held fixed and its
    coefficients are refitted on ``n_boot`` bootstrap resamples of the samples;
    the percentile interval therefore measures the *coefficient estimation*
    uncertainty at the chosen truncation.  Model-selection (truncation) error
    is **not** included -- it is controlled separately by the LOO-error gate.
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("bootstrap requires finite sample values")
    if n_boot < 10:
        raise ValueError(f"n_boot must be >= 10 for a percentile interval, got {n_boot}")
    A = basis_matrix(coords, surrogate.families, surrogate.multi_index)
    idx_active = np.asarray(surrogate.active, dtype=int)
    A_act = A[:, idx_active]
    rng = np.random.default_rng(seed)
    n = y.size
    first = np.empty((n_boot, len(surrogate.families)))
    total = np.empty_like(first)
    base = surrogate.sobol(order=order)
    for b in range(n_boot):
        take = rng.integers(0, n, size=n)
        A_b = A_act[take]
        y_b = y[take]
        try:
            # no hat matrix is needed here: one normal-equation solve per
            # replicate keeps 200 bootstrap refits well under a second
            gram = _gram(A_b) + 1.0e-12 * np.eye(A_b.shape[1])
            with np.errstate(invalid="ignore", over="ignore"):
                rhs = A_b.T @ y_b
            if not np.isfinite(rhs).all():
                rhs = np.einsum("ij,i->j", A_b, y_b)
            coef_a = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            first[b] = np.nan
            total[b] = np.nan
            continue
        full = np.zeros(A.shape[1])
        full[idx_active] = coef_a
        tmp = PCESurrogate(
            degree=surrogate.degree,
            families=surrogate.families,
            multi_index=surrogate.multi_index,
            coef=full,
            active=surrogate.active,
            eps_loo=surrogate.eps_loo,
            eps_loo_abs=surrogate.eps_loo_abs,
            rmse=surrogate.rmse,
            r2=surrogate.r2,
            n_samples=surrogate.n_samples,
            n_terms=surrogate.n_terms,
            method=surrogate.method,
        )
        ind = tmp.sobol(order=order)
        first[b] = ind.first
        total[b] = ind.total
    alpha = 0.5 * (1.0 - level)
    lo = np.nanpercentile(first, 100.0 * alpha, axis=0)
    hi = np.nanpercentile(first, 100.0 * (1.0 - alpha), axis=0)
    tlo = np.nanpercentile(total, 100.0 * alpha, axis=0)
    thi = np.nanpercentile(total, 100.0 * (1.0 - alpha), axis=0)
    return SobolIndices(
        first=base.first,
        total=base.total,
        second=base.second,
        variance=base.variance,
        mean=base.mean,
        first_ci=(lo, hi),
        total_ci=(tlo, thi),
    )