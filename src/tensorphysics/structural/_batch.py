"""Optional NumPy fast path for large simulation runs.

The reliability core is standard-library only; these helpers are imported lazily
so that ``tensorphysics.structural`` keeps working without NumPy. They only
accelerate sampling (Monte Carlo / importance sampling), never the FORM/SORM
iterations, and the fast path is cross-validated against the scalar path in the
test suite.
"""
from __future__ import annotations

from typing import Sequence

from .distributions import (
    Distribution,
    ExponentialDistribution,
    GumbelMaxDistribution,
    LognormalDistribution,
    NormalDistribution,
)

__all__ = [
    "numpy_available",
    "batch_ppf_z",
    "batch_u_to_x",
    "batch_norm_cdf",
]

# Abramowitz & Stegun 26.2.17 upper-tail coefficients (|eps| < 7.5e-8)
_AS_P = 0.2316419
_AS_B = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)


def numpy_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import numpy  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def batch_norm_cdf(z):  # pragma: no cover - requires numpy
    """Vectorised standard normal CDF (A&S 26.2.17, |err| < 7.5e-8)."""
    import numpy as np

    z = np.asarray(z, dtype=float)
    x = np.abs(z)
    t = 1.0 / (1.0 + _AS_P * x)
    poly = t * (_AS_B[0] + t * (_AS_B[1] + t * (_AS_B[2] + t * (_AS_B[3] + t * _AS_B[4]))))
    tail = poly * np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return np.where(z >= 0.0, 1.0 - tail, tail)


def batch_ppf_z(marginal: Distribution, z):  # pragma: no cover - requires numpy
    """Vectorised ``ppf(Phi(z))`` for the supported marginals."""
    import numpy as np

    z = np.asarray(z, dtype=float)
    if isinstance(marginal, NormalDistribution):
        return marginal.mu + marginal.sigma * z
    if isinstance(marginal, LognormalDistribution):
        return np.exp(marginal.mu_ln + marginal.sigma_ln * z)
    if isinstance(marginal, ExponentialDistribution):
        return -np.log(batch_norm_cdf(-z)) / marginal.lam
    if isinstance(marginal, GumbelMaxDistribution):
        upper = batch_norm_cdf(-z)  # Phi(-z), accurate in the upper tail
        lower = batch_norm_cdf(z)
        u = np.where(z <= 0.0, -np.log(lower), -np.log1p(-upper))
        return marginal.loc - marginal.scale * np.log(u)
    return np.frompyfunc(marginal.ppf_z, 1, 1)(z).astype(float)


def batch_u_to_x(transform, w):  # pragma: no cover - requires numpy
    """Map an ``(n, d)`` array of independent standard normals to physical space."""
    import numpy as np

    w = np.asarray(w, dtype=float)
    lower = np.asarray(transform._lower, dtype=float)
    z = w @ lower.T
    out = np.empty_like(z)
    for i, marginal in enumerate(transform.marginals):
        out[:, i] = batch_ppf_z(marginal, z[:, i])
    return out


def batch_rows_to_list(x: Sequence[Sequence[float]]):  # pragma: no cover - requires numpy
    return [list(row) for row in x]