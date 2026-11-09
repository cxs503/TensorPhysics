"""Uncertain-input specification for the uncertainty-budget contract.

The module defines the *input side* of the UQ contract that
:mod:`tensorphysics.uq` hands to any solver: a list of
:class:`Marginal` distributions (uniform / normal / lognormal) plus an
optional Gaussian-copula correlation matrix, and the deterministic map from
a point of the unit hypercube (what a sampler produces) to the physical
parameter vector and to the polynomial basis coordinate.

Coordinate conventions
----------------------
Every marginal is parametrised through a **standard Gaussian latent**
``z`` (``u = Phi(z)`` for a uniform draw ``u`` in ``(0, 1)``).  For each
marginal the *basis coordinate* ``xi`` is a measure-preserving map of
``z``:

===============  ==========================  ==================================
kind             basis family                ``xi``
===============  ==========================  ==================================
``uniform``      Legendre on ``[-1, 1]``     ``xi = 2*Phi(z) - 1``
``normal``       probabilists' Hermite       ``xi = z``
``lognormal``    probabilists' Hermite       ``xi = z``  (``x = exp(mu+s*z)``)
===============  ==========================  ==================================

Because ``z -> xi`` is a bijection that preserves the measure, the ANOVA
(Sobol) decomposition of a quantity of interest is identical in ``xi`` and in
``z``; consequently both bases give the same indices and the polynomial-chaos
expansion is exact in the sense of Xiu & Karniadakis (2002) and Sudret (2008).

Correlation
-----------
A correlation matrix ``R`` is applied on the latent Gaussian scale
(NORTA / Gaussian copula): ``z_corr = L z`` with ``L = chol(R)``, then each
marginal is inverted from ``u = Phi(z_corr)``.  For normal marginals ``R`` is
the Pearson correlation; for non-normal marginals ``R`` is the *latent*
correlation (Spearman's rho equals ``(6/pi) asin(R/2)``).

The Sobol indices reported for a correlated specification are therefore
*latent-coordinate* indices (ANOVA in the independent latent ``z``), which is
the only orthogonal decomposition available once inputs interact through a
copula; :class:`tensorphysics.uq.verdict` records this as a non-blocking note
rather than silently presenting them as classical marginal indices.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "SPEC_HASH_VERSION",
    "LATENT_EPS",
    "Marginal",
    "SampleSpec",
    "latent_from_unit",
]

#: Version tag folded into the specification hash so that a change of the
#: conventions above invalidates stored evidence instead of silently
#: re-interpreting it.
SPEC_HASH_VERSION = "tpuq-spec-v1"

#: Unit-hypercube coordinates are clipped to ``[eps, 1 - eps]`` before the
#: Gaussian inverse CDF, so that ``z`` stays finite for ``u = 0`` / ``u = 1``
#: (the scrambled Sobol sequence lives on ``[0, 1)`` and can hit ``0``).
LATENT_EPS = 1.0e-12

_KINDS = ("uniform", "normal", "lognormal")
_BASIS = {"uniform": "legendre", "normal": "hermite", "lognormal": "hermite"}


def latent_from_unit(unit: np.ndarray) -> np.ndarray:
    """``z = Phi^{-1}(clip(u))`` elementwise."""
    u = np.asarray(unit, dtype=float)
    return norm.ppf(np.clip(u, LATENT_EPS, 1.0 - LATENT_EPS))


def _unit_from_latent(z: np.ndarray) -> np.ndarray:
    """``u = Phi(z)`` elementwise."""
    return norm.cdf(np.asarray(z, dtype=float))


@dataclass(frozen=True)
class Marginal:
    """One uncertain input parameter.

    Parameters
    ----------
    name:
        Parameter identifier (used for reporting and hashing).
    kind:
        ``"uniform"``, ``"normal"`` or ``"lognormal"``.
    params:
        ``(low, high)`` for uniform, ``(mean, std)`` for normal and
        ``(mu, sigma)`` of the underlying normal for lognormal.
    """

    name: str
    kind: str
    params: tuple[float, float]

    # ---- construction helpers -------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("marginal name must be a non-empty string")
        if self.kind not in _KINDS:
            raise ValueError(
                f"marginal {self.name!r}: kind must be one of {_KINDS}, got {self.kind!r}"
            )
        if len(self.params) != 2:
            raise ValueError(f"marginal {self.name!r}: params must have length 2")
        a, b = (float(self.params[0]), float(self.params[1]))
        for value in (a, b):
            if not math.isfinite(value):
                raise ValueError(f"marginal {self.name!r}: non-finite parameter {value!r}")
        if self.kind == "uniform":
            if not a < b:
                raise ValueError(
                    f"marginal {self.name!r}: uniform requires low < high, got {a!r} >= {b!r}"
                )
        else:
            # for normal the second parameter is a scale (std) and must be > 0;
            # for lognormal it is the log-scale sigma, also > 0.  The first
            # parameter (mean / log-mean) may take any finite value.
            if not b > 0.0:
                raise ValueError(
                    f"marginal {self.name!r}: {self.kind} requires a positive scale, got {b!r}"
                )

    @classmethod
    def uniform(cls, name: str, low: float, high: float) -> "Marginal":
        return cls(name=name, kind="uniform", params=(float(low), float(high)))

    @classmethod
    def normal(cls, name: str, mean: float, std: float) -> "Marginal":
        return cls(name=name, kind="normal", params=(float(mean), float(std)))

    @classmethod
    def lognormal(cls, name: str, log_mean: float, log_sigma: float) -> "Marginal":
        """Lognormal with ``log(X) ~ Normal(log_mean, log_sigma)``."""
        return cls(name=name, kind="lognormal", params=(float(log_mean), float(log_sigma)))

    # ---- measure -------------------------------------------------------------
    @property
    def basis_family(self) -> str:
        return _BASIS[self.kind]

    @property
    def support(self) -> tuple[float, float]:
        """Physical support ``(low, high)`` (``inf`` for unbounded kinds)."""
        if self.kind == "uniform":
            return (float(self.params[0]), float(self.params[1]))
        if self.kind == "normal":
            return (-math.inf, math.inf)
        return (0.0, math.inf)

    def moments(self) -> tuple[float, float]:
        """``(mean, std)`` of the marginal in physical units."""
        a, b = (float(self.params[0]), float(self.params[1]))
        if self.kind == "uniform":
            return (0.5 * (a + b), (b - a) / math.sqrt(12.0))
        if self.kind == "normal":
            return (a, b)
        # lognormal: X = exp(a + b Z), Z ~ N(0, 1)
        mean = math.exp(a + 0.5 * b * b)
        return (mean, mean * math.sqrt(math.expm1(b * b)))

    def to_latent(self, unit: np.ndarray) -> np.ndarray:
        return latent_from_unit(unit)

    def from_latent(self, z: np.ndarray) -> np.ndarray:
        """Physical value ``x`` from the latent Gaussian ``z``."""
        z = np.asarray(z, dtype=float)
        a, b = (float(self.params[0]), float(self.params[1]))
        if self.kind == "uniform":
            u = _unit_from_latent(z)
            return a + (b - a) * u
        if self.kind == "normal":
            return a + b * z
        return np.exp(a + b * z)

    def basis_coordinate(self, z: np.ndarray) -> np.ndarray:
        """Measure-preserving coordinate used by the polynomial basis."""
        z = np.asarray(z, dtype=float)
        if self.kind == "uniform":
            return 2.0 * _unit_from_latent(z) - 1.0
        return z

    def from_unit(self, unit: np.ndarray) -> np.ndarray:
        return self.from_latent(self.to_latent(unit))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": list(self.params)}


@dataclass(frozen=True, eq=False)
class SampleSpec:
    """Joint uncertain-input specification (marginals + optional copula).

    ``correlation`` is a symmetric positive-definite matrix with unit diagonal
    on the *latent Gaussian* scale; ``None`` means statistically independent
    inputs.
    """

    marginals: tuple[Marginal, ...]
    correlation: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.marginals:
            raise ValueError("SampleSpec requires at least one marginal")
        names = [m.name for m in self.marginals]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate marginal names: {names}")
        for m in self.marginals:
            if not isinstance(m, Marginal):
                raise TypeError(f"marginals must be Marginal instances, got {type(m)!r}")
        if self.correlation is not None:
            R = np.asarray(self.correlation, dtype=float)
            d = len(self.marginals)
            if R.shape != (d, d):
                raise ValueError(
                    f"correlation must be {d}x{d} for {d} marginals, got {R.shape}"
                )
            if not np.all(np.isfinite(R)):
                raise ValueError("correlation matrix contains non-finite entries")
            if not np.allclose(R, R.T, atol=1.0e-12):
                raise ValueError("correlation matrix must be symmetric")
            if not np.allclose(np.diag(R), 1.0, atol=1.0e-12):
                raise ValueError("correlation matrix must have unit diagonal")
            if np.min(np.linalg.eigvalsh(R)) <= 0.0:
                raise ValueError("correlation matrix must be positive definite")
            object.__setattr__(self, "correlation", R)

    # ---- basic properties -----------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.marginals)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.marginals)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(m.basis_family for m in self.marginals)

    @property
    def correlated(self) -> bool:
        return self.correlation is not None

    def marginal(self, name: str) -> Marginal:
        for m in self.marginals:
            if m.name == name:
                return m
        raise KeyError(f"unknown marginal {name!r}")

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        means = np.array([m.moments()[0] for m in self.marginals])
        stds = np.array([m.moments()[1] for m in self.marginals])
        return means, stds

    def relative_scale(self) -> np.ndarray:
        """``std / |mean|`` per marginal (a natural 1-sigma relative scale)."""
        means, stds = self.moments()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = stds / np.abs(means)
        return ratio

    # ---- unit cube -> physical -------------------------------------------------
    def latent_from_unit(self, unit: np.ndarray) -> np.ndarray:
        """Standard-Gaussian latent coordinates ``z`` for unit-cube points."""
        u = np.asarray(unit, dtype=float)
        if u.ndim != 2 or u.shape[1] != self.dim:
            raise ValueError(
                f"unit points must be (n, {self.dim}), got {np.shape(u)}"
            )
        z = latent_from_unit(u)
        if self.correlation is not None:
            L = np.linalg.cholesky(self.correlation)
            z = z @ L.T
        return z

    def from_latent(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(physical points, basis coordinates)`` from latent ``z``."""
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"latent points must be (n, {self.dim}), got {z.shape}")
        points = np.column_stack(
            [self.marginals[k].from_latent(z[:, k]) for k in range(self.dim)]
        )
        coords = np.column_stack(
            [self.marginals[k].basis_coordinate(z[:, k]) for k in range(self.dim)]
        )
        return points, coords

    def transform(self, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(physical points, basis coordinates)`` from unit-hypercube points."""
        return self.from_latent(self.latent_from_unit(unit))

    def physical_to_latent(self, points: np.ndarray) -> np.ndarray:
        """Inverse map physical -> latent ``z`` (independent or copula case)."""
        x = np.asarray(points, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"points must be (n, {self.dim}), got {x.shape}")
        cols = []
        for k, m in enumerate(self.marginals):
            a, b = (float(m.params[0]), float(m.params[1]))
            if m.kind == "uniform":
                u = (x[:, k] - a) / (b - a)
                cols.append(latent_from_unit(u))
            elif m.kind == "normal":
                cols.append((x[:, k] - a) / b)
            else:
                cols.append((np.log(x[:, k]) - a) / b)
        z = np.column_stack(cols)
        if self.correlation is not None:
            L = np.linalg.cholesky(self.correlation)
            z = np.linalg.solve(L, z.T).T
        return z

    # ---- provenance -----------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": SPEC_HASH_VERSION,
            "marginals": [m.as_dict() for m in self.marginals],
        }
        if self.correlation is not None:
            out["correlation"] = np.round(self.correlation, 15).tolist()
        return out

    def fingerprint(self) -> str:
        """Stable sha256 of the specification (margin series + copula)."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_spec(
    marginals: Sequence[Marginal], correlation: np.ndarray | None = None
) -> SampleSpec:
    """Convenience wrapper mirroring the dataclass constructor."""
    return SampleSpec(marginals=tuple(marginals), correlation=correlation)