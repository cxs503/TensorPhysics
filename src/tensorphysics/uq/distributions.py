"""Input uncertainty model: marginals, Gaussian-copula correlation, isoprobabilistic map.

The module is deliberately *solver agnostic*: it only describes the random
input vector ``X = (X_1, ..., X_d)`` and provides the two maps a
non-intrusive uncertainty-quantification (UQ) driver needs:

* ``unit_to_physical(U)``  -- ``U in [0,1]^d`` (the canonical iid space, drawn by
  :mod:`tensorphysics.uq.sampling`) to the physical input vector ``X``;
* ``physical_to_unit(X)``  -- the inverse (probability integral transform).

Independence and correlation
----------------------------
With ``correlation=None`` the components are independent and
``unit_to_physical`` acts component-wise through each marginal inverse CDF, so
the map is a monotone bijection and the Sobol'-Hoeffding decomposition of any
model output in ``U``-space is *identical* to the one in ``X``-space.  When a
latent correlation matrix ``R`` is supplied the model is the **Gaussian copula**
``X_i = F_i^{-1}(Phi(z_i))``, ``z ~ N(0, R)``; ``R`` is therefore interpreted as
the latent-Gaussian correlation (no Nataf correction is applied -- that
approximation is a property of the *marginals* and would silently change the
user's matrix, so it is left to the caller).

Correlated inputs invalidate the *independent* polynomial-chaos construction
(:mod:`tensorphysics.uq.pce`); :mod:`tensorphysics.uq.gate` refuses to certify
such a study instead of silently reporting wrong indices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

__all__ = [
    "Beta",
    "GaussianCopulaError",
    "InputModel",
    "LogNormal",
    "Marginal",
    "Normal",
    "Uniform",
    "marginal_from_dict",
    "unit_scale",
]

_UNIT_EPS = 1e-12


class GaussianCopulaError(ValueError):
    """Raised when a supplied latent correlation matrix is not usable."""


class Marginal:
    """Base class of a one-dimensional input distribution."""

    kind = "abstract"

    #: Subclasses set this to their parameter names for provenance output.
    param_names: Tuple[str, ...] = ()

    def ppf(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def mean(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def std(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "mean": self.mean(),
            "std": self.std(),
            **{name: getattr(self, name) for name in self.param_names},
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.as_dict()})"


@dataclass(frozen=True)
class Uniform(Marginal):
    """Uniform distribution on ``[low, high]``."""

    low: float
    high: float
    kind = "uniform"
    param_names = ("low", "high")

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError("Uniform bounds must be finite")
        if not self.high > self.low:
            raise ValueError(f"Uniform requires high > low, got {self.low}, {self.high}")

    def ppf(self, u):
        return self.low + (self.high - self.low) * np.asarray(u, dtype=float)

    def cdf(self, x):
        return (np.asarray(x, dtype=float) - self.low) / (self.high - self.low)

    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    def std(self) -> float:
        return (self.high - self.low) / math.sqrt(12.0)


@dataclass(frozen=True)
class Normal(Marginal):
    """Gaussian distribution ``N(mu, sigma^2)``."""

    mu: float
    sigma: float
    kind = "normal"
    param_names = ("mu", "sigma")

    def __post_init__(self) -> None:
        if not self.sigma > 0.0:
            raise ValueError(f"Normal requires sigma > 0, got {self.sigma}")

    def ppf(self, u):
        return self.mu + self.sigma * ndtri(np.clip(np.asarray(u, dtype=float), _UNIT_EPS, 1 - _UNIT_EPS))

    def cdf(self, x):
        return ndtr((np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def mean(self) -> float:
        return self.mu

    def std(self) -> float:
        return self.sigma


@dataclass(frozen=True)
class LogNormal(Marginal):
    """Lognormal distribution with log-parameters ``(mu_log, sigma_log)``."""

    mu_log: float
    sigma_log: float
    kind = "lognormal"
    param_names = ("mu_log", "sigma_log")

    def __post_init__(self) -> None:
        if not self.sigma_log > 0.0:
            raise ValueError(f"LogNormal requires sigma_log > 0, got {self.sigma_log}")

    def ppf(self, u):
        u = np.clip(np.asarray(u, dtype=float), _UNIT_EPS, 1 - _UNIT_EPS)
        return np.exp(self.mu_log + self.sigma_log * ndtri(u))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0.0):
            raise ValueError("LogNormal cdf requires x > 0")
        return ndtr((np.log(x) - self.mu_log) / self.sigma_log)

    def mean(self) -> float:
        return math.exp(self.mu_log + 0.5 * self.sigma_log**2)

    def std(self) -> float:
        var = (math.exp(self.sigma_log**2) - 1.0) * math.exp(2 * self.mu_log + self.sigma_log**2)
        return math.sqrt(var)


@dataclass(frozen=True)
class Beta(Marginal):
    """Beta distribution on ``[low, high]`` with shape parameters ``(a, b)``.

    Evoked for bounded-but-asymmetric engineering parameters (e.g. a
    semi-empirical coefficient known from a literature interval with a most
    likely value away from the midpoint).
    """

    a: float
    b: float
    low: float = 0.0
    high: float = 1.0
    kind = "beta"
    param_names = ("a", "b", "low", "high")

    def __post_init__(self) -> None:
        if not (self.a > 0.0 and self.b > 0.0):
            raise ValueError("Beta requires a > 0 and b > 0")
        if not self.high > self.low:
            raise ValueError("Beta requires high > low")
        from scipy.stats import beta as _beta  # local import: only needed here

        object.__setattr__(
            self, "_dist", _beta(self.a, self.b, loc=self.low, scale=self.high - self.low)
        )

    def ppf(self, u):
        return self._dist.ppf(np.clip(np.asarray(u, dtype=float), _UNIT_EPS, 1 - _UNIT_EPS))

    def cdf(self, x):
        return self._dist.cdf(np.asarray(x, dtype=float))

    def mean(self) -> float:
        return float(self._dist.mean())

    def std(self) -> float:
        return float(self._dist.std())


MARGINAL_TYPES = {cls.kind: cls for cls in (Uniform, Normal, LogNormal, Beta)}


def marginal_from_dict(spec: Mapping[str, Any]) -> Marginal:
    """Build a :class:`Marginal` from a JSON-able specification.

    ``{"kind": "uniform", "low": 0.5, "high": 1.5}`` etc.
    """
    kind = str(spec.get("kind", "")).lower()
    if kind not in MARGINAL_TYPES:
        raise ValueError(f"Unknown marginal kind {kind!r}; known: {sorted(MARGINAL_TYPES)}")
    cls = MARGINAL_TYPES[kind]
    params = {k: v for k, v in spec.items() if k != "kind"}
    return cls(**params)  # type: ignore[arg-type]


def unit_scale(d: int) -> np.ndarray:
    """Return the ``[0, 1]`` lower corner vector (helper for adapters)."""
    return np.zeros(int(d), dtype=float)


class InputModel:
    """Random input vector: independent marginals plus an optional Gaussian copula.

    Parameters
    ----------
    names:
        One name per component, used in every reported index.
    marginals:
        One :class:`Marginal` per component.
    correlation:
        Optional ``(d, d)`` latent-Gaussian correlation matrix (unit diagonal,
        positive definite).  ``None`` means independent components.
    """

    def __init__(
        self,
        names: Sequence[str],
        marginals: Sequence[Marginal],
        correlation: np.ndarray | None = None,
    ) -> None:
        names = list(names)
        marginals = list(marginals)
        if len(names) != len(marginals):
            raise ValueError("names and marginals must have the same length")
        if not names:
            raise ValueError("InputModel needs at least one component")
        self.names = names
        self.marginals = marginals
        self.dim = len(names)
        self.correlation = self._check_correlation(correlation)
        self._chol: np.ndarray | None = None
        if self.correlation is not None:
            try:
                self._chol = np.linalg.cholesky(self.correlation)
            except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
                raise GaussianCopulaError(f"correlation matrix is not positive definite: {exc}") from exc

    def _check_correlation(self, correlation) -> np.ndarray | None:
        if correlation is None:
            return None
        R = np.asarray(correlation, dtype=float)
        if R.shape != (self.dim, self.dim):
            raise GaussianCopulaError(f"correlation must be ({self.dim}, {self.dim}), got {R.shape}")
        if not np.allclose(np.diag(R), 1.0, atol=1e-12):
            raise GaussianCopulaError("correlation diagonal must be 1")
        if not np.allclose(R, R.T, atol=1e-12):
            raise GaussianCopulaError("correlation must be symmetric")
        if np.any(np.abs(R) > 1.0 + 1e-12):
            raise GaussianCopulaError("correlation entries must lie in [-1, 1]")
        return R

    # -- diagnostics ------------------------------------------------------
    @property
    def correlated(self) -> bool:
        return self.correlation is not None

    @property
    def pce_compatible(self) -> bool:
        """Independent components only (the PCE basis requires independence)."""
        return not self.correlated

    # -- maps -------------------------------------------------------------
    def unit_to_physical(self, unit: np.ndarray) -> np.ndarray:
        U = np.asarray(unit, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.dim:
            raise ValueError(f"unit samples must be (n, {self.dim}), got {U.shape}")
        if self._chol is not None:
            U = ndtr(ndtri(np.clip(U, _UNIT_EPS, 1 - _UNIT_EPS)) @ self._chol.T)
        cols = [m.ppf(U[:, j]) for j, m in enumerate(self.marginals)]
        return np.column_stack(cols)

    def physical_to_unit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"physical samples must be (n, {self.dim}), got {X.shape}")
        cols = [m.cdf(X[:, j]) for j, m in enumerate(self.marginals)]
        U = np.clip(np.column_stack(cols), _UNIT_EPS, 1 - _UNIT_EPS)
        if self._chol is not None:
            z = ndtri(U)
            L = self._chol
            # solve L z_unit = z  => latent independent standard normals
            z_unit = np.linalg.solve(L, z.T).T
            U = ndtr(z_unit)
        return U

    def standard_scores(self, unit: np.ndarray, family: str) -> np.ndarray:
        """Map canonical uniforms to the iid space of the PCE basis family."""
        U = np.asarray(unit, dtype=float)
        if self._chol is not None:
            raise GaussianCopulaError(
                "PCE requires independent inputs; correlated InputModel has no "
                "product basis in this package"
            )
        if family == "legendre":
            return 2.0 * U - 1.0
        if family == "hermite":
            return ndtri(np.clip(U, _UNIT_EPS, 1 - _UNIT_EPS))
        raise ValueError(f"unknown basis family {family!r}")

    # -- moments ----------------------------------------------------------
    def mean_vector(self) -> np.ndarray:
        return np.array([m.mean() for m in self.marginals], dtype=float)

    def std_vector(self) -> np.ndarray:
        return np.array([m.std() for m in self.marginals], dtype=float)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "names": list(self.names),
            "marginals": [m.as_dict() for m in self.marginals],
            "correlated": self.correlated,
        }
        if self.correlation is not None:
            out["gaussian_copula_correlation"] = self.correlation.tolist()
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"InputModel(d={self.dim}, names={self.names}, correlated={self.correlated})"