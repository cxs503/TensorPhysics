"""Winterstein Hermite moment-translation transform.

Maps a standard Gaussian variable u ~ N(0,1) to a non-Gaussian variable x
with prescribed skewness γ3 and kurtosis γ4 (excess kurtosis κ = γ4 - 3).

The monotone cubic Hermite polynomial:
    x = h3(u) = u + c3*(u^2 - 1) + c4*(u^3 - 3u)

where c3, c4 are determined by matching the third and fourth central moments.

For the "softening" model (positive skewness, moderate kurtosis):
    c3 = γ3 / (6 + 2*κ)  (approximate, from moment matching)
    c4 = (sqrt(1 + 2*κ) - 1) / 12  (approximate)

More precisely, we solve the system:
    E[h3(u)^3] = γ3
    E[h3(u)^4] - 3 = κ

using Newton iteration for robustness at high kurtosis.

Bounded variant: for ice loads, we clip the output to ±n_sigma*σ to enforce
physical boundedness (ice contact forces cannot be infinite).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

__all__ = [
    "WintersteinTransform",
    "winterstein_hermite_transform",
    "winterstein_inverse",
    "gaussianize_signal",
    "non_gaussianize_signal",
]


@dataclass(frozen=True)
class WintersteinTransform:
    """Winterstein Hermite moment-translation transform parameters.

    Attributes
    ----------
    gamma3 : float
        Skewness of the target non-Gaussian distribution.
    gamma4 : float
        Kurtosis of the target (NOT excess kurtosis; γ4 = 3 for Gaussian).
    c3 : float
        Quadratic Hermite coefficient.
    c4 : float
        Cubic Hermite coefficient.
    bounded : bool
        Whether the transform includes physical bounds.
    bound_sigma : float
        Number of standard deviations for the bound (if bounded).
    """
    gamma3: float
    gamma4: float
    c3: float
    c4: float
    bounded: bool = False
    bound_sigma: float = 4.0

    @property
    def excess_kurtosis(self) -> float:
        """Excess kurtosis κ = γ4 - 3."""
        return self.gamma4 - 3.0

    def forward(self, u: np.ndarray) -> np.ndarray:
        """Map Gaussian u → non-Gaussian x.

        Parameters
        ----------
        u : array_like
            Standard Gaussian samples.

        Returns
        -------
        x : ndarray
            Non-Gaussian samples with prescribed (γ3, γ4).
        """
        u = np.asarray(u, dtype=float)
        x = u + self.c3 * (u**2 - 1.0) + self.c4 * (u**3 - 3.0 * u)
        if self.bounded:
            x = np.clip(x, -self.bound_sigma, self.bound_sigma)
        return x

    def inverse(self, x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Map non-Gaussian x → Gaussian u (inverse transform).

        Uses Newton iteration since the cubic is monotone for moderate (γ3, γ4).

        Parameters
        ----------
        x : array_like
            Non-Gaussian samples.
        tol : float
            Convergence tolerance.

        Returns
        -------
        u : ndarray
            Equivalent Gaussian samples.
        """
        x = np.asarray(x, dtype=float)
        # Initial guess: u ≈ x (identity for small non-Gaussianity)
        u = x.copy()
        for _ in range(100):
            h = u + self.c3 * (u**2 - 1.0) + self.c4 * (u**3 - 3.0 * u)
            if self.bounded:
                h = np.clip(h, -self.bound_sigma, self.bound_sigma)
            dh = 1.0 + 2.0 * self.c3 * u + self.c4 * (3.0 * u**2 - 3.0)
            dh = np.where(np.abs(dh) < 1e-14, 1e-14, dh)
            du = (h - x) / dh
            u -= du
            if np.max(np.abs(du)) < tol:
                break
        return u


def _compute_hermite_moments(c3: float, c4: float, n_quad: int = 80):
    """Compute E[h^2], E[h^3], E[h^4] for h = u + c3*(u^2-1) + c4*(u^3-3u), u~N(0,1).

    Uses Gauss-Hermite quadrature for high accuracy.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    u = nodes * np.sqrt(2.0)
    w = weights / np.sqrt(np.pi)

    h = u + c3 * (u**2 - 1.0) + c4 * (u**3 - 3.0 * u)

    m1 = np.sum(w * h)
    m2 = np.sum(w * h**2)
    m3 = np.sum(w * h**3)
    m4 = np.sum(w * h**4)

    return m1, m2, m3, m4


def _solve_winterstein_coefficients(
    gamma3: float,
    gamma4: float,
) -> Tuple[float, float]:
    """Solve for Winterstein coefficients (c3, c4) by moment matching.

    Uses scipy.optimize.fsolve for robustness.
    """
    kappa = gamma4 - 3.0  # excess kurtosis

    # Gaussian case
    if abs(gamma3) < 1e-12 and abs(kappa) < 1e-12:
        return 0.0, 0.0

    def residuals(params):
        c3, c4 = params
        m1, m2, m3_raw, m4_raw = _compute_hermite_moments(c3, c4)
        var = max(m2, 1e-10)
        skew = m3_raw / var**1.5
        kurt = m4_raw / var**2
        return [skew - gamma3, kurt - gamma4]

    # Initial guess
    c3_init = gamma3 / 6.0 if abs(gamma3) > 1e-10 else 0.0
    c4_init = max(kappa / 36.0, 0.0) if abs(kappa) > 1e-10 else 0.0

    # Solve
    solution, info, ier, msg = fsolve(residuals, [c3_init, c4_init], full_output=True)

    if ier != 1:
        # fsolve didn't converge, try with different initial guess
        solution, info, ier, msg = fsolve(residuals, [0.0, 0.01], full_output=True)

    c3, c4 = solution
    return float(c3), float(c4)


def winterstein_hermite_transform(
    gamma3: float,
    gamma4: float,
    bounded: bool = False,
    bound_sigma: float = 4.0,
) -> WintersteinTransform:
    """Create a Winterstein transform for given skewness and kurtosis.

    Parameters
    ----------
    gamma3 : float
        Skewness of the target distribution.
    gamma4 : float
        Kurtosis (NOT excess; γ4=3 for Gaussian).
    bounded : bool
        If True, clip output to ±bound_sigma.
    bound_sigma : float
        Bound in standard deviations.

    Returns
    -------
    WintersteinTransform
        Transform object with forward/inverse methods.
    """
    if gamma4 < 1.0:
        raise ValueError(f"Kurtosis must be >= 1, got {gamma4}")

    c3, c4 = _solve_winterstein_coefficients(gamma3, gamma4)
    return WintersteinTransform(
        gamma3=gamma3, gamma4=gamma4,
        c3=c3, c4=c4,
        bounded=bounded, bound_sigma=bound_sigma,
    )


def winterstein_inverse(
    x: np.ndarray,
    gamma3: float,
    gamma4: float,
) -> np.ndarray:
    """Inverse Winterstein transform: non-Gaussian x → Gaussian u."""
    tfm = winterstein_hermite_transform(gamma3, gamma4)
    return tfm.inverse(x)


def gaussianize_signal(
    x: np.ndarray,
    gamma3: Optional[float] = None,
    gamma4: Optional[float] = None,
) -> np.ndarray:
    """Convert a non-Gaussian signal to equivalent Gaussian.

    Parameters
    ----------
    x : array_like
        Input signal (non-Gaussian).
    gamma3, gamma4 : float, optional
        If given, use these moments. Otherwise estimate from data.

    Returns
    -------
    u : ndarray
        Gaussianized signal.
    """
    x = np.asarray(x, dtype=float)
    if gamma3 is None:
        gamma3 = float(np.mean(((x - x.mean()) / x.std())**3))
    if gamma4 is None:
        gamma4 = float(np.mean(((x - x.mean()) / x.std())**4))
    tfm = winterstein_hermite_transform(gamma3, gamma4)
    # Normalize x to zero mean, unit variance
    x_norm = (x - x.mean()) / x.std()
    u = tfm.inverse(x_norm)
    return u


def non_gaussianize_signal(
    u: np.ndarray,
    gamma3: float,
    gamma4: float,
    bounded: bool = False,
    bound_sigma: float = 4.0,
) -> np.ndarray:
    """Convert a Gaussian signal to non-Gaussian with prescribed moments.

    Parameters
    ----------
    u : array_like
        Input Gaussian signal (should be approximately N(0,1)).
    gamma3 : float
        Target skewness.
    gamma4 : float
        Target kurtosis (γ4=3 for Gaussian).
    bounded : bool
        If True, clip output.
    bound_sigma : float
        Bound in standard deviations.

    Returns
    -------
    x : ndarray
        Non-Gaussian signal with prescribed (γ3, γ4).
    """
    u = np.asarray(u, dtype=float)
    tfm = winterstein_hermite_transform(gamma3, gamma4, bounded=bounded,
                                         bound_sigma=bound_sigma)
    return tfm.forward(u)
