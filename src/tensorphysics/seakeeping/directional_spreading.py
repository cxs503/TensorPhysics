"""Directional wave spreading model.

Implements the cos^2(s/2) directional spreading function for ocean
wave spectra, parameterized by the spreading parameter s.

Directional spectrum:
    S(omega, theta) = S(omega) * D(theta; s)

where D(theta; s) is the normalized spreading function:
    D(theta; s) = C(s) * |cos((theta - theta_0)/2)|^s

The normalization constant C(s) ensures:
    integral_{-pi}^{pi} D(theta; s) dtheta = 1

As s increases from 0 to infinity:
    s = 0   -> isotropic (uniform in all directions)
    s = 2   -> moderate spreading
    s = 36  -> strongly directional (narrow beam)

References
----------
- Mitsuyasu, H. et al. (1975). "Observations of the directional
  spectrum of ocean waves using the cloverleaf buoy."
- Goda, Y. (2000). "Random Seas and Design of Maritime Structures."
- ISO 15016:2015 — Guidelines for the assessment of speed and power
  performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import gamma as gamma_func

__all__ = [
    "cos2s_spreading",
    "directional_spectrum",
    "recover_spreading_parameter",
    "directional_moment",
    "DirectionalSpectrumResult",
]


def _spreading_normalization(s: float) -> float:
    """Compute normalization constant C(s) for cos^s/2 spreading.

    The integral of |cos(theta/2)|^s from -pi to pi is:
        2 * sqrt(pi) * Gamma((s+1)/2) / Gamma((s+2)/2)

    So C(s) = Gamma((s+2)/2) / (2 * sqrt(pi) * Gamma((s+1)/2))

    This ensures integral_{-pi}^{pi} C(s) * |cos(theta/2)|^s dtheta = 1.
    """
    if s <= 0:
        return 1.0 / (2.0 * np.pi)  # isotropic limit
    return float(
        gamma_func((s + 2) / 2.0) / (2.0 * np.sqrt(np.pi) * gamma_func((s + 1) / 2.0))
    )


def cos2s_spreading(
    theta: np.ndarray,
    s: float,
    theta_0: float = 0.0,
) -> np.ndarray:
    """Evaluate cos^2(s/2) directional spreading function.

    D(theta; s, theta_0) = C(s) * |cos((theta - theta_0)/2)|^s

    Parameters
    ----------
    theta : np.ndarray
        Direction angles [rad], typically in [-pi, pi].
    s : float
        Spreading parameter. s=0 -> isotropic, s=36 -> narrow beam.
    theta_0 : float
        Mean wave direction [rad]. Default 0.

    Returns
    -------
    np.ndarray
        Normalized spreading function D(theta) [1/rad].
    """
    theta = np.asarray(theta, dtype=float)
    dtheta = theta - theta_0
    # Wrap to [-pi, pi]
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    c_s = _spreading_normalization(s)
    return c_s * np.abs(np.cos(dtheta / 2.0)) ** s


def directional_spectrum(
    omega: np.ndarray,
    theta: np.ndarray,
    s_omega: np.ndarray,
    s: float,
    theta_0: float = 0.0,
) -> np.ndarray:
    """Compute 2D directional wave spectrum.

    S(omega, theta) = S(omega) * D(theta; s, theta_0)

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency array [rad/s], shape (n_omega,).
    theta : np.ndarray
        Direction angle array [rad], shape (n_theta,).
    s_omega : np.ndarray
        1D frequency spectrum S(omega) [m^2·s], shape (n_omega,).
    s : float
        Spreading parameter.
    theta_0 : float
        Mean wave direction [rad].

    Returns
    -------
    np.ndarray
        2D directional spectrum S(omega, theta) [m^2·s/rad],
        shape (n_omega, n_theta).
    """
    omega = np.asarray(omega, dtype=float)
    theta = np.asarray(theta, dtype=float)
    s_omega = np.asarray(s_omega, dtype=float)

    # D(theta) shape: (n_theta,)
    d_theta = cos2s_spreading(theta, s, theta_0)

    # Outer product: S(omega, theta) = S(omega) * D(theta)
    # Result shape: (n_omega, n_theta)
    return np.outer(s_omega, d_theta)


def directional_moment(
    omega: np.ndarray,
    theta: np.ndarray,
    s_omega_theta: np.ndarray,
    n_omega: int = 0,
    n_theta: int = 0,
) -> float:
    """Compute directional spectral moment.

    m_{n_omega, n_theta} = integral integral omega^{n_omega} * theta^{n_theta} * S(omega, theta) domega dtheta

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency array [rad/s].
    theta : np.ndarray
        Direction angle array [rad].
    s_omega_theta : np.ndarray
        2D directional spectrum [m^2·s/rad].
    n_omega : int
        Frequency moment order.
    n_theta : int
        Directional moment order.

    Returns
    -------
    float
        Directional spectral moment.
    """
    omega = np.asarray(omega, dtype=float)
    theta = np.asarray(theta, dtype=float)
    s_omega_theta = np.asarray(s_omega_theta, dtype=float)

    integrand = (omega[:, np.newaxis] ** n_omega) * (theta[np.newaxis, :] ** n_theta) * s_omega_theta

    # Double integration using trapezoidal rule
    integrated_theta = np.trapezoid(integrand, theta, axis=1)
    return float(np.trapezoid(integrated_theta, omega))


@dataclass
class DirectionalSpectrumResult:
    """Result of directional spectrum analysis.

    Attributes
    ----------
    omega : np.ndarray
        Angular frequency array [rad/s].
    theta : np.ndarray
        Direction angle array [rad].
    spectrum_2d : np.ndarray
        2D directional spectrum [m^2·s/rad].
    s_parameter : float
        Input spreading parameter.
    theta_0 : float
        Mean wave direction [rad].
    m0 : float
        Zeroth spectral moment [m^2].
    mean_direction : float
        Mean wave direction recovered from spectrum [rad].
    directional_spread : float
        Directional spread (standard deviation) [rad].
    """
    omega: np.ndarray
    theta: np.ndarray
    spectrum_2d: np.ndarray
    s_parameter: float
    theta_0: float
    m0: float
    mean_direction: float
    directional_spread: float


def recover_spreading_parameter(
    theta: np.ndarray,
    directional_distribution: np.ndarray,
) -> float:
    """Recover the spreading parameter s from a directional distribution.

    Uses the relationship between the first Fourier coefficient a1
    and the spreading parameter s:
        a1 = integral D(theta) * cos(theta) dtheta
        For cos^s/2 spreading: a1 = s / (s + 2)
        Therefore: s = 2 * a1 / (1 - a1)

    Parameters
    ----------
    theta : np.ndarray
        Direction angles [rad].
    directional_distribution : np.ndarray
        Normalized directional distribution D(theta) [1/rad].

    Returns
    -------
    float
        Recovered spreading parameter s.
    """
    theta = np.asarray(theta, dtype=float)
    d = np.asarray(directional_distribution, dtype=float)

    # First Fourier coefficients
    a1 = float(np.trapezoid(d * np.cos(theta), theta))
    b1 = float(np.trapezoid(d * np.sin(theta), theta))

    # Magnitude of the first harmonic
    a1_mag = np.sqrt(a1 ** 2 + b1 ** 2)

    # s = 2 * a1 / (1 - a1) for cos^s/2 model
    if a1_mag >= 1.0:
        return np.inf  # perfectly directional
    if a1_mag <= 0.0:
        return 0.0  # isotropic

    return float(2.0 * a1_mag / (1.0 - a1_mag))


def analyze_directional(
    omega: np.ndarray,
    s_omega: np.ndarray,
    s: float,
    theta_0: float = 0.0,
    n_theta: int = 180,
) -> DirectionalSpectrumResult:
    """Full analysis of a directional spectrum.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency array [rad/s].
    s_omega : np.ndarray
        1D frequency spectrum [m^2·s].
    s : float
        Spreading parameter.
    theta_0 : float
        Mean wave direction [rad].
    n_theta : int
        Number of directional bins.

    Returns
    -------
    DirectionalSpectrumResult
    """
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    spec_2d = directional_spectrum(omega, theta, s_omega, s, theta_0)

    # Zeroth moment (integrate over both omega and theta)
    m0 = float(np.trapezoid(
        np.trapezoid(spec_2d, theta, axis=1),
        omega
    ))

    # Recover mean direction from first Fourier coefficients
    # Integrate spectrum over omega to get directional distribution
    dir_dist = np.trapezoid(spec_2d, omega, axis=0)
    # Normalize
    dir_total = np.trapezoid(dir_dist, theta)
    if dir_total > 0:
        dir_dist = dir_dist / dir_total

    # Mean direction
    a1 = float(np.trapezoid(dir_dist * np.cos(theta), theta))
    b1 = float(np.trapezoid(dir_dist * np.sin(theta), theta))
    mean_dir = float(np.arctan2(b1, a1))

    # Directional spread (circular standard deviation)
    R = np.sqrt(a1 ** 2 + b1 ** 2)
    if R > 0 and R < 1:
        dir_spread = float(np.sqrt(-2.0 * np.log(R)))
    else:
        dir_spread = 0.0 if R >= 1 else np.pi

    return DirectionalSpectrumResult(
        omega=omega,
        theta=theta,
        spectrum_2d=spec_2d,
        s_parameter=s,
        theta_0=theta_0,
        m0=m0,
        mean_direction=mean_dir,
        directional_spread=dir_spread,
    )
