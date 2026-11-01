"""Euler-Bernoulli beam response of the hull girder to a distributed load.

Support conventions
------------------
Two linear conventions are provided, because they answer two different
questions:

``support="free"`` (default, physical for a floating hull)
    The hull floats: the rigid-body part of the added load (net force and net
    moment) is carried by the change of buoyancy / heave and trim
    redistribution and produces no bending, so it is removed before
    integration::

        q_bal(x) = q(x) - (a + b x),  int q_bal dx = 0,  int x q_bal dx = 0
        V(x) = int_{-L/2}^{x} q_bal ds,   M(x) = int_{-L/2}^{x} V ds

``support="simple"``
    Classical simply supported beam (reactions at the perpendiculars),
    ``V(0) = R0``, ``R0 = int q (L - s) ds / L`` so that ``M = 0`` at both
    ends.  This is the convention behind the analytic degeneracy checks
    (uniform load -> ``M_mid = q L^2 / 8``; independent segments -> the
    discrete influence-line variance).

Both conventions are linear operators on ``q``; :func:`moment_matrix` returns
the dense matrix ``A`` with ``M = A q`` so that the covariance of a random
pressure field can be propagated analytically,
``Var(M_i) = (A C A^T)_ii``, without Monte-Carlo noise.  The same operator is
used for the analytic limits and for the Monte-Carlo statistics, so the two are
directly comparable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SUPPORT_CONVENTIONS",
    "balance_load",
    "shear_and_moment",
    "moment_matrix",
    "moment_statistics",
    "independent_segment_variance",
    "uniform_load_midspan_moment",
]

SUPPORT_CONVENTIONS = ("free", "simple")


def balance_load(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Remove the rigid-body part of the load; returns ``(q_bal, a, b)``."""
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    if x.shape != q.shape:
        raise ValueError("x and q must have the same shape")
    length = float(x[-1] - x[0])
    a = float(np.trapezoid(q, x) / length)
    xx = float(np.trapezoid(x * x, x))
    b = float(np.trapezoid(x * q, x) / xx) if xx > 0.0 else 0.0
    return q - (a + b * x), a, b


def shear_and_moment(
    x: np.ndarray, q: np.ndarray, support: str = "free"
) -> tuple[np.ndarray, np.ndarray]:
    """Shear force ``V(x)`` and bending moment ``M(x)`` by cumulative trapezoid."""
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    if x.shape != q.shape:
        raise ValueError("x and q must have the same shape")
    if support not in SUPPORT_CONVENTIONS:
        raise ValueError(f"support must be one of {SUPPORT_CONVENTIONS}")
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("x must be strictly increasing")
    length = float(x[-1] - x[0])

    if support == "free":
        q_used = balance_load(x, q)[0]
        v = np.concatenate(([0.0], np.cumsum(0.5 * (q_used[1:] + q_used[:-1]) * dx)))
    else:  # simply supported: left reaction from global moment balance
        r0 = float(np.trapezoid(q * (x[-1] - x), x) / length)
        increments = 0.5 * (q[1:] + q[:-1]) * dx
        v = r0 - np.concatenate(([0.0], np.cumsum(increments)))
    m = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dx)))
    return v, m


def moment_matrix(x: np.ndarray, support: str = "free") -> np.ndarray:
    """Linear operator ``A`` with ``M = A q`` (shape ``(n, n)``)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    a = np.empty((n, n), dtype=float)
    for i in range(n):
        unit = np.zeros(n)
        unit[i] = 1.0
        a[:, i] = shear_and_moment(x, unit, support=support)[1]
    return a


def moment_statistics(
    x: np.ndarray,
    covariance: np.ndarray,
    matrix: np.ndarray | None = None,
    support: str = "free",
) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation of a zero-mean bending moment from a load covariance.

    Returns ``(std_M, A)``.
    """
    x = np.asarray(x, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (x.size, x.size):
        raise ValueError("covariance must be (n, n) on the beam grid")
    a = moment_matrix(x, support=support) if matrix is None else matrix
    act = a @ cov
    var = np.einsum("ij,ij->i", act, a)
    return np.sqrt(np.clip(var, 0.0, None)), a


def uniform_load_midspan_moment(load_per_length: float, length: float) -> float:
    """Analytic simply supported midspan moment for a uniform load: ``q L^2 / 8``."""
    return float(load_per_length * length ** 2 / 8.0)


def independent_segment_variance(
    x: np.ndarray,
    n_segments: int,
    sigma: float,
    support: str = "simple",
) -> dict:
    """Variance of ``M(x)`` for *uncorrelated* pressure over ``n_segments`` segments.

    Each segment ``i`` of length ``l`` carries an independent pressure of
    standard deviation ``sigma``; with the segment influence integral
    ``I_i(x) = int_{segment i} g(x, s) ds`` the moment variance is exactly::

        Var(M) = sigma^2 * sum_i I_i(x)^2

    The acceptance criterion quotes the scaling law ``n * sigma^2 * l^2`` (the
    dimensionally normalised form, i.e. with unit influence slope); this routine
    returns both the exact variance and the ratio
    ``Var / (n sigma^2 l^2) = <(I_i / l)^2>`` so that the law can be checked
    against its closed form.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    length = float(x[-1] - x[0])
    l = float(length / n_segments)
    a = moment_matrix(x, support=support)
    var = np.zeros(n)
    for i in range(n_segments):
        lo = x[0] + i * l
        hi = lo + l
        mask = ((x >= lo - 1e-12) & (x <= hi + 1e-12)).astype(float)
        contribution = a @ mask  # I_i(x)
        var += (sigma ** 2) * contribution ** 2
    scale = n_segments * sigma ** 2 * l ** 2
    return {
        "std": np.sqrt(np.clip(var, 0.0, None)),
        "variance": var,
        "scale_n_sigma2_l2": scale,
        "ratio_to_scale": float(np.max(var) / scale) if scale > 0 else float("nan"),
        "segment_length": l,
        "n_segments": int(n_segments),
    }