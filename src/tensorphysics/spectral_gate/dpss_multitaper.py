"""DPSS multi-taper spectral estimation (Thomson's method).

Uses Discrete Prolate Spheroidal Sequences (Slepian sequences) as orthogonal
tapers to compute multiple independent spectral estimates, which are then
averaged. This reduces variance compared to a single-taper method while
controlling spectral leakage through the time-bandwidth product parameter.

The DPSS windows are optimally concentrated in both time and frequency domains,
making them ideal for spectral estimation.

References
----------
* Thomson, D.J. (1982) Spectrum estimation and harmonic analysis.
  Proc. IEEE, 70(9), 1055-1096.
* Percival, D.B. & Walden, A.T. (1993) Spectral Analysis for Physical
  Applications. Cambridge University Press, §7.
* Slepian, D. (1978) Prolate spheroidal wave functions, Fourier analysis,
  and uncertainty — V: The discrete case. Bell Syst. Tech. J., 57, 1371-1430.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal.windows import dpss


@dataclass(frozen=True)
class DPSSResult:
    """Result of DPSS multi-taper PSD estimation.

    Attributes
    ----------
    freqs_hz : np.ndarray
        Frequency array [Hz].
    psd : np.ndarray
        One-sided power spectral density [units²/Hz].
    n_tapers : int
        Number of DPSS tapers used.
    nw : float
        Time-bandwidth product.
    """
    freqs_hz: np.ndarray
    psd: np.ndarray
    n_tapers: int
    nw: float

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
        """Total integrated energy."""
        return float(np.trapezoid(self.psd, self.freqs_hz))


def dpss_multitaper_psd(
    signal: np.ndarray,
    fs: float,
    nw: float = 4.0,
    k_tapers: Optional[int] = None,
) -> DPSSResult:
    """Estimate PSD using DPSS multi-taper method.

    Parameters
    ----------
    signal : np.ndarray
        Input time series (1-D).
    fs : float
        Sampling frequency [Hz].
    nw : float
        Time-bandwidth product (typical: 2.5–4.0). Controls the trade-off
        between spectral resolution and variance reduction.
    k_tapers : int, optional
        Number of tapers to use. Default: int(2 * nw) - 1 (the number of
        well-concentrated Slepian sequences).

    Returns
    -------
    DPSSResult
        Frequency array, PSD, and metadata.

    Notes
    -----
    The number of useful tapers is approximately 2*nw - 1. Using more tapers
    includes sequences with poor spectral concentration, which can increase
    bias. The default k_tapers = 2*nw - 1 is the standard choice.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    n = len(x)
    if n < 16:
        raise ValueError("signal too short for multi-taper estimation")

    if k_tapers is None:
        k_tapers = max(1, int(2 * nw) - 1)
    k_tapers = min(k_tapers, n)

    # Generate DPSS tapers
    tapers = dpss(n, nw, k_tapers)

    # Compute eigenvalues (concentration factors) for weighting
    # scipy's dpss returns only the tapers, not eigenvalues
    # We use uniform weighting (standard practice) or can use eigenvalue weighting

    # Compute periodogram for each taper
    n_freq = n // 2 + 1
    psd_sum = np.zeros(n_freq)

    for k in range(k_tapers):
        tapered = x * tapers[k]
        # FFT
        spec = np.fft.rfft(tapered)
        psd_sum += np.abs(spec) ** 2

    # Average and normalize
    # Normalization: for a white noise process with variance σ²,
    # E[|X(f)|²] = N * σ² for the DFT
    # For PSD density: S(f) = (1/(fs * N)) * |X(f)|² per taper
    # After averaging K tapers: S(f) = (1/(fs * N * K)) * Σ|X_k(f)|²
    # But we need to account for taper energy: Σ|taper_k|² ≠ N
    # Correct normalization: S(f) = (1/(fs * K)) * Σ(|X_k(f)|² / Σ|taper_k|²)

    psd = np.zeros(n_freq)
    for k in range(k_tapers):
        taper_energy = float(np.sum(tapers[k] ** 2))
        tapered = x * tapers[k]
        spec = np.fft.rfft(tapered)
        psd += np.abs(spec) ** 2 / taper_energy

    psd /= (fs * k_tapers)

    # One-sided: double non-DC, non-Nyquist bins
    psd = psd.copy()
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    return DPSSResult(
        freqs_hz=freqs,
        psd=psd,
        n_tapers=k_tapers,
        nw=nw,
    )
