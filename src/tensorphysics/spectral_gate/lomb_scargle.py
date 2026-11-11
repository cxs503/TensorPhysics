"""Lomb–Scargle periodogram for non-uniformly sampled time series.

The Lomb–Scargle periodogram is the standard method for spectral analysis
of irregularly sampled data. It generalizes the classical periodogram to
non-uniform sampling by fitting sinusoids at each frequency via least squares.

This implementation uses scipy.signal.lombscargle with proper normalization
to produce a PSD estimate comparable to Welch and multi-taper methods.

References
----------
* Lomb, N.R. (1976) Least-squares frequency analysis of unequally spaced data.
  Astrophysics and Space Science, 39, 447-462.
* Scargle, J.D. (1982) Studies in astronomical time series analysis. II —
  Statistical aspects of spectral analysis of unevenly spaced data.
  ApJ, 263, 835-853.
* VanderPlas, J.T. (2018) Understanding the Lomb–Scargle Periodogram.
  ApJS, 236, 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import lombscargle as scipy_lombscargle


@dataclass(frozen=True)
class LombScargleResult:
    """Result of Lomb–Scargle PSD estimation.

    Attributes
    ----------
    freqs_hz : np.ndarray
        Frequency array [Hz] at which PSD was evaluated.
    psd : np.ndarray
        Power spectral density estimate [units²/Hz].
    n_points : int
        Number of data points used.
    """
    freqs_hz: np.ndarray
    psd: np.ndarray
    n_points: int

    @property
    def df_hz(self) -> float:
        """Mean frequency spacing [Hz]."""
        if len(self.freqs_hz) < 2:
            return 0.0
        return float(np.mean(np.diff(self.freqs_hz)))

    def band_energy(self, f_low: float, f_high: float) -> float:
        """Integrated energy in [f_low, f_high] band."""
        mask = (self.freqs_hz >= f_low) & (self.freqs_hz <= f_high)
        return float(np.trapezoid(self.psd[mask], self.freqs_hz[mask]))

    def total_energy(self) -> float:
        """Total integrated energy."""
        return float(np.trapezoid(self.psd, self.freqs_hz))

    def peak_frequency(self) -> float:
        """Frequency of maximum PSD."""
        idx = int(np.argmax(self.psd))
        return float(self.freqs_hz[idx])


def lomb_scargle_psd(
    time_s: np.ndarray,
    signal: np.ndarray,
    freqs_hz: Optional[np.ndarray] = None,
    n_freq: int = 1024,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> LombScargleResult:
    """Estimate PSD using Lomb–Scargle periodogram.

    Parameters
    ----------
    time_s : np.ndarray
        Time stamps [s] (may be non-uniform).
    signal : np.ndarray
        Signal values at the given time stamps.
    freqs_hz : np.ndarray, optional
        Frequencies at which to evaluate the periodogram.
        If None, a uniform grid from f_min to f_max is used.
    n_freq : int
        Number of frequency bins (used if freqs_hz is None).
    f_min : float, optional
        Minimum frequency [Hz]. Default: 1 / (max(time) - min(time)).
    f_max : float, optional
        Maximum frequency [Hz]. Default: 0.5 / median(dt) (Nyquist-like).

    Returns
    -------
    LombScargleResult
        Frequency array, PSD, and metadata.

    Notes
    -----
    The Lomb–Scargle periodogram from scipy returns unnormalized power.
    We normalize it to PSD units [units²/Hz] using the relation:

        PSD(f) = (2 / (N * fs_eff)) * P_LS(f)

    where fs_eff is an effective sampling rate and P_LS is the raw
    Lomb–Scargle power. For comparison with uniformly-sampled methods,
    we use fs_eff = N / T where T is the total observation time.
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(signal, dtype=float)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("time and signal must be 1-D")
    if len(t) != len(x):
        raise ValueError("time and signal must have the same length")
    if len(t) < 4:
        raise ValueError("too few data points for Lomb–Scargle")

    # Remove mean
    x = x - x.mean()

    # Determine frequency grid
    t_span = t.max() - t.min()
    if f_min is None:
        f_min = 1.0 / t_span if t_span > 0 else 0.01
    if f_max is None:
        # Estimate effective Nyquist from median sampling interval
        dt_sorted = np.sort(np.diff(t))
        median_dt = float(np.median(dt_sorted))
        f_max = 0.5 / median_dt if median_dt > 0 else 10.0

    if freqs_hz is None:
        freqs_hz = np.linspace(f_min, f_max, n_freq)

    # scipy lombscargle expects angular frequencies
    omega = 2.0 * np.pi * freqs_hz

    # Compute raw Lomb–Scargle periodogram
    raw_power = scipy_lombscargle(t, x, omega)

    # Normalize to PSD units
    # The raw LS power for a sinusoid of amplitude A is approximately
    # P_LS ≈ N * A² / 4 for large N
    # For PSD: S(f) = A² / (4 * df) where df is frequency resolution
    # We use: S(f) = (2 * T / N²) * P_LS(f)
    # This ensures that for white noise with variance σ²,
    # the expected PSD is σ² * T / N = σ² / fs_eff
    n = len(x)
    psd = (2.0 * t_span / (n * n)) * raw_power

    return LombScargleResult(
        freqs_hz=freqs_hz,
        psd=psd,
        n_points=n,
    )


def detect_missing_rate(
    time_uniform: np.ndarray,
    time_observed: np.ndarray,
    tol: float = 1e-10,
) -> float:
    """Compute the fraction of missing samples from a uniform grid.

    Parameters
    ----------
    time_uniform : np.ndarray
        The full uniform time grid.
    time_observed : np.ndarray
        The actually observed time stamps.
    tol : float
        Tolerance for matching time stamps.

    Returns
    -------
    float
        Fraction of missing samples (0 = complete, 1 = all missing).
    """
    n_total = len(time_uniform)
    n_observed = 0
    for t_obs in time_observed:
        if np.any(np.abs(time_uniform - t_obs) < tol):
            n_observed += 1
    return 1.0 - n_observed / n_total
