"""AR(1) and AR(2) process analytical PSD and sample generation.

Autoregressive processes have closed-form power spectral densities, making
them ideal test cases for validating spectral estimators.

AR(1) process:
    x[t] = φ * x[t-1] + ε[t],  ε ~ N(0, σ²)
    PSD(f) = σ²·dt / |1 - φ * exp(-i2πf·dt)|²

AR(2) process:
    x[t] = φ₁ * x[t-1] + φ₂ * x[t-2] + ε[t]
    PSD(f) = σ²·dt / |1 - φ₁ * exp(-i2πf·dt) - φ₂ * exp(-i4πf·dt)|²

The PSD is returned as a **one-sided** density in units²/Hz, directly
comparable to Welch/DPSS estimates (i.e. ∫₀^(fs/2) S(f) df = variance).

The AR(2) process can exhibit a resonance peak when the characteristic
polynomial has complex conjugate roots, making it a good test for
spectral estimators' ability to resolve narrow peaks.

References
----------
* Box, G.E.P., Jenkins, G.M., Reinsel, G.C. (2008) Time Series Analysis:
  Forecasting and Control. 4th ed., Wiley.
* Priestley, M.B. (1981) Spectral Analysis and Time Series. Academic Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def ar1_psd(
    freqs_hz: np.ndarray,
    phi: float,
    sigma2: float = 1.0,
    fs: float = 1.0,
) -> np.ndarray:
    """One-sided analytical PSD of an AR(1) process in units²/Hz.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency array [Hz].
    phi : float
        AR coefficient (|phi| < 1 for stationarity).
    sigma2 : float
        Innovation variance.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    np.ndarray
        One-sided PSD values [units²/Hz] at the given frequencies.
        DC and Nyquist are not doubled; all other bins are doubled
        so that ∫₀^(fs/2) S(f) df = σ²/(1-φ²).

    Notes
    -----
    The two-sided AR(1) PSD is:
        S_two(f) = σ²·dt / |1 - φ·exp(-i2πf·dt)|²
                 = σ²·dt / (1 + φ² - 2φ·cos(2πf/fs))

    The one-sided PSD doubles the positive-frequency content:
        S_one(f) = 2 · S_two(f)  for f > 0
        S_one(0) = S_two(0)
    """
    if abs(phi) >= 1.0:
        raise ValueError("AR(1) process is not stationary for |phi| >= 1")

    dt = 1.0 / fs
    f_norm = freqs_hz / fs  # normalized frequency
    # Two-sided PSD: σ²·dt / (1 + φ² - 2φ·cos(2πf))
    denom = 1.0 + phi**2 - 2.0 * phi * np.cos(2.0 * np.pi * f_norm)
    psd_two = sigma2 * dt / denom

    # Convert to one-sided
    psd_one = psd_two.copy()
    if len(psd_one) > 1:
        psd_one[1:] *= 2.0

    return psd_one


def ar2_psd(
    freqs_hz: np.ndarray,
    phi1: float,
    phi2: float,
    sigma2: float = 1.0,
    fs: float = 1.0,
) -> np.ndarray:
    """One-sided analytical PSD of an AR(2) process in units²/Hz.

    Parameters
    ----------
    freqs_hz : np.ndarray
        Frequency array [Hz].
    phi1 : float
        First AR coefficient.
    phi2 : float
        Second AR coefficient.
    sigma2 : float
        Innovation variance.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    np.ndarray
        One-sided PSD values [units²/Hz] at the given frequencies.

    Notes
    -----
    The two-sided AR(2) PSD is:
        S_two(f) = σ²·dt / |1 - φ₁·exp(-i2πf·dt) - φ₂·exp(-i4πf·dt)|²

    Stationarity requires:
        φ₁ + φ₂ < 1
        φ₂ - φ₁ < 1
        |φ₂| < 1

    When φ₁² + 4φ₂ < 0, the process has a resonant frequency:
        f_res = (fs / 2π) · arccos(-φ₁(1-φ₂) / (4φ₂))
    """
    # Check stationarity conditions
    if phi1 + phi2 >= 1.0 or phi2 - phi1 >= 1.0 or abs(phi2) >= 1.0:
        raise ValueError("AR(2) coefficients do not satisfy stationarity conditions")

    dt = 1.0 / fs
    f_norm = freqs_hz / fs
    omega = 2.0 * np.pi * f_norm

    # |1 - φ₁·exp(-iω) - φ₂·exp(-i2ω)|²
    real_part = 1.0 - phi1 * np.cos(omega) - phi2 * np.cos(2.0 * omega)
    imag_part = phi1 * np.sin(omega) + phi2 * np.sin(2.0 * omega)
    denom = real_part**2 + imag_part**2

    psd_two = sigma2 * dt / denom

    # Convert to one-sided
    psd_one = psd_two.copy()
    if len(psd_one) > 1:
        psd_one[1:] *= 2.0

    return psd_one


def ar2_resonant_frequency(
    phi1: float,
    phi2: float,
    fs: float = 1.0,
) -> Optional[float]:
    """Compute the resonant frequency of an AR(2) process.

    Parameters
    ----------
    phi1 : float
        First AR coefficient.
    phi2 : float
        Second AR coefficient.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    float or None
        Resonant frequency [Hz], or None if no resonance (real roots).
    """
    discriminant = phi1**2 + 4.0 * phi2
    if discriminant >= 0:
        return None  # Real roots, no resonance peak

    # Complex conjugate roots → resonance
    cos_omega = -phi1 * (1.0 - phi2) / (4.0 * phi2)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega = np.arccos(cos_omega)
    return float(omega * fs / (2.0 * np.pi))


def generate_ar1(
    n_samples: int,
    phi: float,
    sigma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sample path from an AR(1) process.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    phi : float
        AR coefficient (|phi| < 1).
    sigma : float
        Innovation standard deviation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Time series of length n_samples.
    """
    if abs(phi) >= 1.0:
        raise ValueError("AR(1) process is not stationary for |phi| >= 1")

    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n_samples)

    x = np.zeros(n_samples)
    # Burn-in: start from stationary distribution
    x[0] = eps[0] / np.sqrt(1 - phi**2)
    for i in range(1, n_samples):
        x[i] = phi * x[i - 1] + eps[i]

    return x


def generate_ar2(
    n_samples: int,
    phi1: float,
    phi2: float,
    sigma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sample path from an AR(2) process.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    phi1 : float
        First AR coefficient.
    phi2 : float
        Second AR coefficient.
    sigma : float
        Innovation standard deviation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Time series of length n_samples.
    """
    # Check stationarity
    if phi1 + phi2 >= 1.0 or phi2 - phi1 >= 1.0 or abs(phi2) >= 1.0:
        raise ValueError("AR(2) coefficients do not satisfy stationarity conditions")

    rng = np.random.default_rng(seed)

    # Use burn-in to reach stationary distribution
    burn_in = 500
    total = n_samples + burn_in
    eps = rng.normal(0, sigma, total)

    x = np.zeros(total)
    x[0] = eps[0]
    if total > 1:
        x[1] = phi1 * x[0] + eps[1]
    for i in range(2, total):
        x[i] = phi1 * x[i - 1] + phi2 * x[i - 2] + eps[i]

    return x[burn_in:]
