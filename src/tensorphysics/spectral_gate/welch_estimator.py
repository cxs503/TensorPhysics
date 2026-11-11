"""Welch's method: segmented, windowed, averaged periodogram PSD estimator.

Welch's method reduces variance of the raw periodogram by:
1. Splitting the signal into overlapping segments
2. Applying a window function to each segment
3. Computing the periodogram of each windowed segment
4. Averaging the periodograms

The result is a consistent estimator of the true PSD with reduced variance
at the cost of frequency resolution.

References
----------
* Welch, P.D. (1967) The use of fast Fourier transform for the estimation
  of power spectra: A method based on time averaging over short, modified
  periodograms. IEEE Trans. Audio Electroacoust., 15(2), 70-73.
* Stoica, P. & Moses, R.L. (2005) Spectral Analysis of Signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.signal import welch as scipy_welch


@dataclass(frozen=True)
class WelchResult:
    """Result of Welch PSD estimation.

    Attributes
    ----------
    freqs_hz : np.ndarray
        Frequency array [Hz].
    psd : np.ndarray
        One-sided power spectral density [units²/Hz].
    n_segments : int
        Number of segments averaged.
    nperseg : int
        Samples per segment.
    noverlap : int
        Samples of overlap between segments.
    window : str
        Window function name.
    """
    freqs_hz: np.ndarray
    psd: np.ndarray
    n_segments: int
    nperseg: int
    noverlap: int
    window: str

    @property
    def df_hz(self) -> float:
        """Frequency resolution [Hz]."""
        if len(self.freqs_hz) < 2:
            return 0.0
        return float(self.freqs_hz[1] - self.freqs_hz[0])

    def band_energy(self, f_low: float, f_high: float) -> float:
        """Integrated energy in [f_low, f_high] band."""
        mask = (self.freqs_hz >= f_low) & (self.freqs_hz <= f_high)
        return float(np.trapezoid(self.psd[mask], self.freqs_hz[mask]))

    def total_energy(self) -> float:
        """Total integrated energy (should equal signal variance for zero-mean)."""
        return float(np.trapezoid(self.psd, self.freqs_hz))


def welch_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    scaling: Literal["density", "spectrum"] = "density",
) -> WelchResult:
    """Estimate PSD using Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        Input time series (1-D).
    fs : float
        Sampling frequency [Hz].
    nperseg : int, optional
        Length of each segment. Default: min(256, len(signal)).
    noverlap : int, optional
        Number of points to overlap between segments. Default: nperseg // 2.
    window : str
        Window function name (passed to scipy.signal.get_window).
    scaling : str
        'density' for PSD [units²/Hz], 'spectrum' for power spectrum [units²].

    Returns
    -------
    WelchResult
        Frequency array, PSD, and metadata.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    if len(x) < 4:
        raise ValueError("signal too short for Welch estimation")

    n = len(x)
    if nperseg is None:
        nperseg = min(256, n)
    nperseg = min(nperseg, n)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = scipy_welch(
        x, fs=fs, window=window, nperseg=nperseg,
        noverlap=noverlap, scaling=scaling,
    )

    # Compute number of segments
    step = nperseg - noverlap
    n_segments = max(1, (n - noverlap) // step)

    return WelchResult(
        freqs_hz=freqs,
        psd=psd,
        n_segments=n_segments,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )
