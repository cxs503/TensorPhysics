"""Rosenblatt probability-integral transform for the joint (H_s, T_z) sea state.

The metocean model used throughout the package is the classical two-block
description of a wave climate (DNV-RP-C205 / Haver-Nyhus type):

* **H_s marginal** — three-parameter Weibull

  ``F_H(h) = 1 - exp(-((h - gamma)/alpha)**k)``,   ``h > gamma``

* **T_z | H_s** — log-normal whose mean follows a power law in ``h``

  ``ln T_z | h ~ N(mu(h), sigma_ln**2)``,   ``mu(h) = ln(a1 + a2 * h**a3) - sigma_ln**2 / 2``

so that ``E[T_z | h] = a1 + a2 * h**a3``.

The Rosenblatt transform

``u1 = Phi^-1(F_H(h))``,   ``u2 = Phi^-1(F_{T|H}(t | h))``

maps the joint sea-state density onto the *independent* standard bivariate
normal.  This is the only ingredient IFORM needs: every environmental contour
is a circle ``u1**2 + u2**2 = beta**2`` in this space.

Because the conditional dispersion ``sigma_ln`` is constant and the H_s marginal
is invertible in closed form, the transform is exact and exactly invertible,
which is what makes the analytic anchors of
:mod:`~tensorphysics.design_sea_state.iform` strict to machine precision.

A second, deliberately trivial transform (:class:`BivariateNormalTransform`) is
provided for the fully analytic anchor case: a correlated bivariate *normal*
``(X, Y)`` whose Rosenblatt transform is the linear map
``u1 = z_x``, ``u2 = (z_y - rho z_x)/sqrt(1 - rho**2)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

__all__ = [
    "Weibull3",
    "LognormalConditional",
    "RosenblattTransform",
    "BivariateNormalTransform",
    "standard_normal_cdf",
    "standard_normal_ppf",
    "standard_normal_pdf",
]

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def standard_normal_pdf(x: float) -> float:
    """Standard normal density."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def standard_normal_cdf(x: float) -> float:
    """Standard normal CDF ``0.5 * erfc(-x/sqrt(2))`` (exact in the tails)."""
    return 0.5 * math.erfc(-x / _SQRT2)


def standard_normal_ppf(p: float) -> float:
    """Standard normal quantile, ~1e-15 relative accuracy (``NormalDist``)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p!r}")
    return NormalDist().inv_cdf(p)


@dataclass(frozen=True)
class Weibull3:
    """Three-parameter Weibull marginal (shape ``k``, scale ``alpha``, location ``gamma``)."""

    k: float
    alpha: float
    gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.k <= 0.0 or self.alpha <= 0.0:
            raise ValueError("Weibull shape and scale must be positive")

    def cdf(self, x: float) -> float:
        if x <= self.gamma:
            return 0.0
        return -math.expm1(-((x - self.gamma) / self.alpha) ** self.k)

    def sf(self, x: float) -> float:
        if x <= self.gamma:
            return 1.0
        return math.exp(-((x - self.gamma) / self.alpha) ** self.k)

    def pdf(self, x: float) -> float:
        if x <= self.gamma:
            return 0.0
        z = (x - self.gamma) / self.alpha
        return (self.k / self.alpha) * z ** (self.k - 1.0) * math.exp(-(z**self.k))

    def ppf(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must lie in (0, 1), got {p!r}")
        return self.gamma + self.alpha * (-math.log1p(-p)) ** (1.0 / self.k)

    def mode(self) -> float:
        """Mode (equals the location when ``k <= 1``, where the density decreases)."""
        if self.k <= 1.0:
            return self.gamma
        return self.gamma + self.alpha * ((self.k - 1.0) / self.k) ** (1.0 / self.k)

    def mean(self) -> float:
        return self.gamma + self.alpha * math.gamma(1.0 + 1.0 / self.k)


@dataclass(frozen=True)
class LognormalConditional:
    """``T_z | H_s`` log-normal with power-law conditional mean (Haver-Nyhus form)."""

    a1: float = 3.6
    a2: float = 1.0
    a3: float = 0.7
    sigma_ln: float = 0.08

    def __post_init__(self) -> None:
        if self.sigma_ln <= 0.0:
            raise ValueError("sigma_ln must be positive")

    def mean_tz(self, hs: float) -> float:
        """Conditional mean ``E[T_z | H_s = hs]`` (the physical power law)."""
        return self.a1 + self.a2 * max(hs, 0.0) ** self.a3

    def mu_ln(self, hs: float) -> float:
        """``E[ln T_z | H_s = hs]`` such that the conditional *mean* is :meth:`mean_tz`."""
        return math.log(self.mean_tz(hs)) - 0.5 * self.sigma_ln**2

    def cdf(self, tz: float, hs: float) -> float:
        if tz <= 0.0:
            return 0.0
        return standard_normal_cdf((math.log(tz) - self.mu_ln(hs)) / self.sigma_ln)

    def sf(self, tz: float, hs: float) -> float:
        if tz <= 0.0:
            return 1.0
        return standard_normal_cdf(-(math.log(tz) - self.mu_ln(hs)) / self.sigma_ln)

    def pdf(self, tz: float, hs: float) -> float:
        if tz <= 0.0:
            return 0.0
        z = (math.log(tz) - self.mu_ln(hs)) / self.sigma_ln
        return standard_normal_pdf(z) / (self.sigma_ln * tz)

    def ppf(self, p: float, hs: float) -> float:
        return math.exp(self.mu_ln(hs) + self.sigma_ln * standard_normal_ppf(p))

    def median(self, hs: float) -> float:
        return math.exp(self.mu_ln(hs))


@dataclass(frozen=True)
class RosenblattTransform:
    """Closed-form Rosenblatt transform of the two-block sea-state model."""

    hs_marginal: Weibull3
    tz_conditional: LognormalConditional

    # -- forward -----------------------------------------------------------
    def u1(self, hs: float) -> float:
        return standard_normal_ppf(self.hs_marginal.cdf(hs))

    def u2(self, hs: float, tz: float) -> float:
        return standard_normal_ppf(self.tz_conditional.cdf(tz, hs))

    def forward(self, hs: float, tz: float) -> tuple[float, float]:
        return self.u1(hs), self.u2(hs, tz)

    # -- inverse -----------------------------------------------------------
    def inverse(self, u1: float, u2: float) -> tuple[float, float]:
        hs = self.hs_marginal.ppf(standard_normal_cdf(u1))
        tz = self.tz_conditional.ppf(standard_normal_cdf(u2), hs)
        return hs, tz

    # -- densities ---------------------------------------------------------
    def joint_pdf(self, hs: float, tz: float) -> float:
        return self.hs_marginal.pdf(hs) * self.tz_conditional.pdf(tz, hs)

    def jacobian_det(self, hs: float, tz: float) -> float:
        """``|det d(u)/d(x)|`` evaluated analytically.

        ``u1`` depends on ``hs`` only and ``u2`` on both, hence
        ``det = (du1/dhs) * (du2/dtz)`` with ``du1/dhs = f_H(hs)/phi(u1)`` and
        ``du2/dtz = 1/(sigma_ln * tz)``.
        """
        u1 = self.u1(hs)
        return (self.hs_marginal.pdf(hs) / standard_normal_pdf(u1)) / (
            self.tz_conditional.sigma_ln * tz
        )

    def log_density_consistency(self, hs: float, tz: float) -> float:
        """``ln(phi(u1) phi(u2) |det J| / f(x))`` — zero if the transform is right."""
        u1, u2 = self.forward(hs, tz)
        mapped = standard_normal_pdf(u1) * standard_normal_pdf(u2) * self.jacobian_det(hs, tz)
        return math.log(mapped) - math.log(self.joint_pdf(hs, tz))

    def max_log_residual(self, samples: Sequence[tuple[float, float]]) -> float:
        """``max |ln(phi2(u) |det J| / f(x))|`` over samples."""
        return max(abs(self.log_density_consistency(hs, tz)) for hs, tz in samples)

    # -- sampling ----------------------------------------------------------
    def sample(self, n: int, seed: int = 20260101) -> list[tuple[float, float]]:
        """Draw ``n`` sea states through the *physical* branch (inverse-CDF sampling).

        Samples are generated in physical space (uniform -> Weibull ppf, then
        uniform -> conditional log-normal ppf) instead of mapping normals, so a
        subsequent ``u``-radius exceedance test genuinely cross-checks the
        transform rather than assuming it.
        """
        import random

        rng = random.Random(seed)
        out: list[tuple[float, float]] = []
        for _ in range(n):
            hs = self.hs_marginal.ppf(rng.random())
            tz = self.tz_conditional.ppf(rng.random(), hs)
            out.append((hs, tz))
        return out


@dataclass(frozen=True)
class BivariateNormalTransform:
    """Linear Rosenblatt transform of a correlated bivariate normal (analytic anchor)."""

    mu_x: float = 3.0
    sigma_x: float = 1.2
    mu_y: float = 8.0
    sigma_y: float = 1.5
    rho: float = 0.6

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly inside (-1, 1)")
        if self.sigma_x <= 0.0 or self.sigma_y <= 0.0:
            raise ValueError("standard deviations must be positive")

    @property
    def _root(self) -> float:
        return math.sqrt(1.0 - self.rho**2)

    def forward(self, x: float, y: float) -> tuple[float, float]:
        zx = (x - self.mu_x) / self.sigma_x
        zy = (y - self.mu_y) / self.sigma_y
        return zx, (zy - self.rho * zx) / self._root

    def inverse(self, u1: float, u2: float) -> tuple[float, float]:
        zx = u1
        zy = self.rho * u1 + self._root * u2
        return self.mu_x + self.sigma_x * zx, self.mu_y + self.sigma_y * zy

    def quadratic_form(self, x: float, y: float) -> float:
        """Independent evaluation of ``u1**2 + u2**2`` from physical coordinates."""
        zx = (x - self.mu_x) / self.sigma_x
        zy = (y - self.mu_y) / self.sigma_y
        return zx * zx + (zy - self.rho * zx) ** 2 / (1.0 - self.rho**2)

    def mahalanobis_form(self, x: float, y: float) -> float:
        """``z^T R^-1 z`` — the *chi-square* ellipse, deliberately NOT the IFORM contour."""
        zx = (x - self.mu_x) / self.sigma_x
        zy = (y - self.mu_y) / self.sigma_y
        return (zx * zx - 2.0 * self.rho * zx * zy + zy * zy) / (1.0 - self.rho**2)

    def sample(self, n: int, seed: int = 20260101) -> list[tuple[float, float]]:
        """Draw ``n`` bivariate-normal pairs (Box-Muller on a physical branch)."""
        import random

        rng = random.Random(seed)
        out: list[tuple[float, float]] = []
        for _ in range(n):
            u1 = standard_normal_ppf(rng.random())
            u2 = standard_normal_ppf(rng.random())
            out.append(self.inverse(u1, u2))
        return out