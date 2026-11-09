"""Orthogonal polynomial bases and multi-index sets for polynomial chaos.

Two classical families are implemented; both are orthonormal with respect to
the measure induced by the canonical iid space of
:mod:`tensorphysics.uq.distributions`:

===============  ==============================  ============================
family           canonical variable              measure
===============  ==============================  ============================
``legendre``     ``xi = 2F(x) - 1 in [-1, 1]``   ``Uniform(-1, 1)``, ``E[P_k^2] = 1/(2k+1)``
``hermite``      ``z = Phi^{-1}(F(x))``          ``N(0, 1)``, ``E[He_k^2] = k!``
===============  ==============================  ============================

Because ``F`` is a monotone bijection, a Legendre basis built on ``xi`` produces
the *same* Sobol' indices as a basis built on ``x`` -- the variance decomposition
is invariant under monotone transformation of independent inputs.  This is what
lets one generic basis serve arbitrary marginals.

Multi-index sets
----------------
A total-degree set ``{alpha : sum(alpha) <= p}`` is the standard choice; the
hyperbolic (``q``-norm) truncation ``|alpha|_q <= p`` removes the pure
high-order interaction terms that a low-effective-dimension model does not need.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "HERMITE",
    "gauss_rule",
    "LEGENDRE",
    "FAMILIES",
    "MultiIndexSet",
    "basis_matrix",
    "check_orthonormality",
    "hyperbolic_truncate",
    "multi_indices",
    "orthonormal_basis",
    "univariate_norms",
    "univariate_table",
]

LEGENDRE = "legendre"
HERMITE = "hermite"
FAMILIES = (LEGENDRE, HERMITE)


def univariate_table(scores: np.ndarray, degree: int, family: str) -> np.ndarray:
    """Values of the first ``degree+1`` orthonormal polynomials at ``scores``.

    Returns an ``(n, degree+1)`` array; columns are ``phi_0 .. phi_degree``.
    """
    x = np.asarray(scores, dtype=float).ravel()
    deg = int(degree)
    P = np.empty((x.size, deg + 1), dtype=float)
    P[:, 0] = 1.0
    if deg >= 1:
        if family == LEGENDRE:
            P[:, 1] = x
            for k in range(2, deg + 1):
                P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
        elif family == HERMITE:
            P[:, 1] = x
            for k in range(2, deg + 1):
                P[:, k] = x * P[:, k - 1] - (k - 1) * P[:, k - 2]
        else:
            raise ValueError(f"unknown basis family {family!r}; known: {FAMILIES}")
    norms = univariate_norms(deg, family)
    return P * (1.0 / np.sqrt(norms))[None, :]


def univariate_norms(degree: int, family: str) -> np.ndarray:
    """``E[psi_k^2]`` for the *unnormalized* classical polynomials."""
    deg = int(degree)
    k = np.arange(deg + 1, dtype=float)
    if family == LEGENDRE:
        return 1.0 / (2.0 * k + 1.0)
    if family == HERMITE:
        return np.array([math.factorial(int(i)) for i in range(deg + 1)], dtype=float)
    raise ValueError(f"unknown basis family {family!r}; known: {FAMILIES}")


def multi_indices(dim: int, degree: int, *, q: float = 1.0, interaction: int | None = None) -> np.ndarray:
    """Total-degree (optionally hyperbolic / bounded-interaction) multi-index set.

    Parameters
    ----------
    dim, degree:
        Number of inputs and the maximum total degree ``p``.
    q:
        Hyperbolic truncation norm; ``q=1`` is the full total-degree set, ``q<1``
        keeps only ``|alpha|_q <= p``.
    interaction:
        Maximum number of non-zero components per term (``None`` = unbounded).
    """
    dim = int(dim)
    degree = int(degree)
    if dim < 1 or degree < 0:
        raise ValueError("dim >= 1 and degree >= 0 are required")
    idx: list[Tuple[int, ...]] = []
    for alpha in itertools.product(range(degree + 1), repeat=dim):
        s = sum(alpha)
        if s > degree:
            continue
        if q < 1.0 and s > 0.0:
            if sum(a**q for a in alpha) ** (1.0 / q) > degree + 1e-12:
                continue
        if interaction is not None and sum(1 for a in alpha if a > 0) > interaction:
            continue
        idx.append(alpha)
    idx.sort(key=lambda a: (sum(a), a))
    return np.array(idx, dtype=int).reshape(len(idx), dim)


def hyperbolic_truncate(index_set: np.ndarray, degree: int, q: float) -> np.ndarray:
    """Apply a hyperbolic ``q``-norm truncation to an existing index set."""
    A = np.asarray(index_set, dtype=int)
    if A.ndim != 2:
        raise ValueError("index_set must be 2-D (m, dim)")
    keep = []
    for a in A:
        s = float(np.sum(a))
        if s <= 1e-12:
            keep.append(True)
        else:
            keep.append(float(np.sum(a.astype(float) ** q)) ** (1.0 / q) <= degree + 1e-12)
    return A[np.array(keep, dtype=bool)]


@dataclass(frozen=True)
class MultiIndexSet:
    """A multi-index set with cached per-term norms and order statistics."""

    array: np.ndarray
    family: str
    degree: int

    @classmethod
    def build(
        cls,
        dim: int,
        degree: int,
        family: str = LEGENDRE,
        *,
        q: float = 1.0,
        interaction: int | None = None,
    ) -> "MultiIndexSet":
        if family not in FAMILIES:
            raise ValueError(f"unknown basis family {family!r}")
        A = multi_indices(dim, degree, q=q, interaction=interaction)
        return cls(array=A, family=family, degree=int(degree))

    @property
    def size(self) -> int:
        return int(self.array.shape[0])

    @property
    def dim(self) -> int:
        return int(self.array.shape[1])

    @property
    def norms(self) -> np.ndarray:
        """``E[Psi_alpha^2]`` of the *normalised* tensor products (all ones).

        The basis returned by :func:`orthonormal_basis` is orthonormal, so the
        norm of every term is 1; the unnormalized norms of the underlying
        classical polynomials are available from :func:`univariate_norms`.
        """
        return np.ones(self.size, dtype=float)

    def orders(self) -> np.ndarray:
        return np.sum(self.array, axis=1)

    def as_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "degree": int(self.degree),
            "size": self.size,
            "dim": self.dim,
            "max_order": int(self.orders().max()) if self.size else 0,
        }


def orthonormal_basis(scores: np.ndarray, index_set: MultiIndexSet) -> np.ndarray:
    """Design matrix ``A[i, k] = Psi_{alpha_k}(scores[i])`` (orthonormal terms).

    ``scores`` is ``(n, dim)`` in the canonical space of the family.
    """
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("scores must be 2-D (n, dim)")
    if S.shape[1] != index_set.dim:
        raise ValueError(f"scores have {S.shape[1]} columns but the index set expects {index_set.dim}")
    deg = int(index_set.array.max()) if index_set.size else 0
    tables = [univariate_table(S[:, j], deg, index_set.family) for j in range(index_set.dim)]
    A = np.ones((S.shape[0], index_set.size), dtype=float)
    for k, alpha in enumerate(index_set.array):
        col = A[:, k]
        for j, a in enumerate(alpha):
            if a:
                col = col * tables[j][:, int(a)]
        A[:, k] = col
    return A


#: Alias kept for the short call sites.
basis_matrix = orthonormal_basis


def gauss_rule(n_nodes: int, family: str) -> Tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss quadrature exact for the family's orthogonality measure.

    For ``legendre`` the measure is ``Uniform(-1, 1)`` (weight 1/2), for
    ``hermite`` it is ``N(0, 1)``.  ``n_nodes`` nodes per dimension integrate
    polynomials of degree ``2*n_nodes - 1`` exactly, so a Gram matrix of the
    basis truncated at total degree ``p`` is reproduced to machine precision with
    ``n_nodes = p + 1``.
    """
    if family == LEGENDRE:
        x, w = np.polynomial.legendre.leggauss(int(n_nodes))
        w = w / 2.0
    elif family == HERMITE:
        x, w = np.polynomial.hermite.hermgauss(int(n_nodes))
        x = np.sqrt(2.0) * x
        w = w / np.sqrt(np.pi)
    else:
        raise ValueError(f"unknown basis family {family!r}")
    return x, w


def check_orthonormality(
    index_set: MultiIndexSet,
    n: int | None = None,
    seed: int = 20260101,
    tol: float = 1e-10,
) -> Dict[str, float]:
    """Check ``E[Psi_a Psi_b] = delta_ab`` under the basis measure.

    Uses Gauss quadrature (exact, machine precision) when the tensor quadrature
    fits (``dim <= 4``), otherwise a Monte-Carlo estimate with the requested
    ``n`` samples (whose error then sets the tolerance).  A mismatch between
    basis and measure silently biases every Sobol' index downstream, so this is
    the numerical proof that the two agree.
    """
    max_degree = int(index_set.orders().max()) if index_set.size else 0
    q = max_degree + 1
    use_quad = index_set.dim <= 4 and q**index_set.dim <= 50000
    if use_quad:
        x, w = gauss_rule(q, index_set.family)
        grids = np.meshgrid(*([x] * index_set.dim), indexing="ij")
        scores = np.stack([g.ravel() for g in grids], axis=1)
        weights = np.prod(np.meshgrid(*([w] * index_set.dim), indexing="ij"), axis=0).ravel()
        method = "gauss-quadrature"
    else:
        from .sampling import draw_unit

        n = int(n or 50000)
        U = draw_unit(n, index_set.dim, method="mc", seed=seed)
        if index_set.family == LEGENDRE:
            scores = 2.0 * U - 1.0
        else:
            scores = np.sqrt(2.0) * _erfinv(2.0 * U - 1.0)
        weights = np.full(n, 1.0 / n)
        method = "monte-carlo"
        tol = max(tol, 0.1 / math.sqrt(n))
    A = orthonormal_basis(scores, index_set)
    G = A.T @ (A * weights[:, None])
    off = np.abs(G - np.eye(index_set.size))
    return {
        "method": method,  # type: ignore[dict-item]
        "n_nodes": float(scores.shape[0]),
        "max_abs_off_diagonal": float(off.max()),
        "max_diag_error": float(np.abs(np.diag(G) - 1.0).max()),
        "tolerance": float(tol),
        "ok": bool(off.max() <= tol),
    }


def _erfinv(y: np.ndarray) -> np.ndarray:
    from scipy.special import erfinv

    return erfinv(np.asarray(y, dtype=float))
