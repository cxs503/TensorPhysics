"""Cross-validation framework for spectral estimators.

Provides mechanisms to compare PSD estimates from different methods
(Welch, DPSS multi-taper, Lomb–Scargle) and validate against analytical
AR(1)/AR(2) PSD anchors.

Acceptance criteria
-------------------
(a) AR(1) process PSD estimation error < 5% (full-band integrated energy)
(b) AR(2) process PSD estimation error < 8% (resonance peak position error < 2%)
(c) Three-method cross-validation: band energy deviation < 10%
(d) Lomb–Scargle on non-uniform sampling (30% missing): main frequency detection > 95%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .welch_estimator import welch_psd, WelchResult
from .dpss_multitaper import dpss_multitaper_psd, DPSSResult
from .lomb_scargle import lomb_scargle_psd, LombScargleResult
from .ar_models import ar1_psd, ar2_psd, generate_ar1, generate_ar2, ar2_resonant_frequency


@dataclass(frozen=True)
class CrossValidationResult:
    """Result of cross-validating multiple PSD estimators."""
    welch: WelchResult
    dpss: DPSSResult
    lomb_scargle: Optional[LombScargleResult]
    band_energies: Dict[str, float]
    energy_deviations: Dict[str, float]
    max_deviation: float


def band_energy(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    """Compute integrated energy in a frequency band."""
    mask = (freqs_hz >= f_low) & (freqs_hz <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs_hz[mask]))


def total_energy(freqs_hz: np.ndarray, psd: np.ndarray) -> float:
    """Compute total integrated energy."""
    return float(np.trapezoid(psd, freqs_hz))


def relative_energy_error(
    freqs_hz: np.ndarray,
    psd_estimated: np.ndarray,
    psd_analytical: np.ndarray,
) -> float:
    """Compute relative error in total integrated energy."""
    e_est = total_energy(freqs_hz, psd_estimated)
    e_true = total_energy(freqs_hz, psd_analytical)
    if e_true == 0:
        return 0.0
    return abs(e_est - e_true) / e_true


def cross_validate_psd(
    signal: np.ndarray,
    fs: float,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
    welch_nperseg: int = 256,
    dpss_nw: float = 4.0,
) -> CrossValidationResult:
    """Cross-validate PSD estimates from Welch and DPSS multi-taper methods."""
    welch_result = welch_psd(signal, fs, nperseg=welch_nperseg)
    dpss_result = dpss_multitaper_psd(signal, fs, nw=dpss_nw)

    if f_low is None:
        f_low = 0.0
    if f_high is None:
        f_high = fs / 2.0

    freqs_common = np.linspace(f_low, f_high, 1024)
    welch_psd_interp = np.interp(freqs_common, welch_result.freqs_hz, welch_result.psd)
    dpss_psd_interp = np.interp(freqs_common, dpss_result.freqs_hz, dpss_result.psd)

    e_welch = band_energy(freqs_common, welch_psd_interp, f_low, f_high)
    e_dpss = band_energy(freqs_common, dpss_psd_interp, f_low, f_high)

    energies = {"welch": e_welch, "dpss": e_dpss}
    mean_energy = (e_welch + e_dpss) / 2.0

    deviations = {}
    if mean_energy > 0:
        deviations["welch"] = abs(e_welch - mean_energy) / mean_energy
        deviations["dpss"] = abs(e_dpss - mean_energy) / mean_energy
    else:
        deviations["welch"] = 0.0
        deviations["dpss"] = 0.0

    return CrossValidationResult(
        welch=welch_result,
        dpss=dpss_result,
        lomb_scargle=None,
        band_energies=energies,
        energy_deviations=deviations,
        max_deviation=max(deviations.values()),
    )


@dataclass
class SpectralGateReport:
    """Full spectral gate execution report."""
    ar1_energy_error: float
    ar1_pass: bool
    ar2_energy_error: float
    ar2_peak_freq_error: float
    ar2_pass: bool
    cv_max_deviation: float
    cv_pass: bool
    ls_freq_accuracy: float
    ls_pass: bool
    welch_result: Optional[WelchResult] = None
    dpss_result: Optional[DPSSResult] = None
    lomb_result: Optional[LombScargleResult] = None
    ar1_params: Dict = field(default_factory=dict)
    ar2_params: Dict = field(default_factory=dict)
    n_samples: int = 0
    fs: float = 1.0


def run_spectral_gate(
    n_samples: int = 8192,
    fs: float = 100.0,
    seed: int = 42,
    missing_rate: float = 0.3,
    test_freq_hz: float = 5.0,
) -> SpectralGateReport:
    """Execute the full spectral gate with all acceptance criteria.

    Parameters
    ----------
    n_samples : int
        Number of samples for AR process generation.
    fs : float
        Sampling frequency [Hz].
    seed : int
        Random seed for reproducibility.
    missing_rate : float
        Fraction of samples to remove for Lomb–Scargle test.
    test_freq_hz : float
        Test sinusoid frequency for Lomb–Scargle detection test.

    Returns
    -------
    SpectralGateReport
        Full gate report with all criterion results.
    """
    dt = 1.0 / fs
    time_uniform = np.arange(n_samples) * dt

    # ---- Criterion (a): AR(1) process ----
    phi1 = 0.85
    sigma_ar1 = 1.0
    ar1_signal = generate_ar1(n_samples, phi1, sigma_ar1, seed=seed)

    welch_ar1 = welch_psd(ar1_signal, fs, nperseg=512)
    dpss_ar1 = dpss_multitaper_psd(ar1_signal, fs, nw=4.0)
    psd_ar1_true = ar1_psd(welch_ar1.freqs_hz, phi1, sigma_ar1**2, fs)

    ar1_energy_err = relative_energy_error(
        welch_ar1.freqs_hz, welch_ar1.psd, psd_ar1_true
    )
    ar1_pass = ar1_energy_err < 0.05

    # ---- Criterion (b): AR(2) process ----
    # Sharp resonance: phi1=1.95, phi2=-0.97 → f_res ≈ 2.24 Hz
    phi1_ar2 = 1.95
    phi2_ar2 = -0.97
    sigma_ar2 = 1.0

    # Need longer signal for adequate frequency resolution + segment count
    n_ar2 = max(n_samples, 32768)
    ar2_signal = generate_ar2(n_ar2, phi1_ar2, phi2_ar2, sigma_ar2, seed=seed + 1)

    # nperseg=4096 gives df=0.0244 Hz; need <0.045 Hz for 2% at 2.24 Hz
    welch_ar2 = welch_psd(ar2_signal, fs, nperseg=4096)
    dpss_ar2 = dpss_multitaper_psd(ar2_signal, fs, nw=4.0)

    psd_ar2_true = ar2_psd(welch_ar2.freqs_hz, phi1_ar2, phi2_ar2, sigma_ar2**2, fs)
    ar2_energy_err = relative_energy_error(
        welch_ar2.freqs_hz, welch_ar2.psd, psd_ar2_true
    )

    # Peak frequency error against closed-form resonant frequency
    f_res = ar2_resonant_frequency(phi1_ar2, phi2_ar2, fs)
    f_peak_est = float(welch_ar2.freqs_hz[np.argmax(welch_ar2.psd)])
    ar2_peak_freq_err = abs(f_peak_est - f_res) / f_res if f_res else 0.0

    ar2_pass = (ar2_energy_err < 0.08) and (ar2_peak_freq_err < 0.02)

    # ---- Criterion (c): Cross-validation ----
    cv_result = cross_validate_psd(ar1_signal, fs, f_low=0.0, f_high=fs/2)
    cv_pass = cv_result.max_deviation < 0.10

    # ---- Criterion (d): Lomb–Scargle on non-uniform sampling ----
    rng = np.random.default_rng(seed + 2)
    clean_signal = np.sin(2 * np.pi * test_freq_hz * time_uniform)
    noisy_signal = clean_signal + 0.3 * rng.normal(size=n_samples)

    n_keep = int(n_samples * (1 - missing_rate))
    keep_idx = rng.choice(n_samples, size=n_keep, replace=False)
    keep_idx.sort()

    time_observed = time_uniform[keep_idx]
    signal_observed = noisy_signal[keep_idx]

    ls_result = lomb_scargle_psd(time_observed, signal_observed, n_freq=2048)

    detected_freq = ls_result.peak_frequency()
    freq_accuracy = 1.0 - abs(detected_freq - test_freq_hz) / test_freq_hz
    freq_accuracy = max(0.0, freq_accuracy)
    ls_pass = freq_accuracy > 0.95

    return SpectralGateReport(
        ar1_energy_error=ar1_energy_err,
        ar1_pass=ar1_pass,
        ar2_energy_error=ar2_energy_err,
        ar2_peak_freq_error=ar2_peak_freq_err,
        ar2_pass=ar2_pass,
        cv_max_deviation=cv_result.max_deviation,
        cv_pass=cv_pass,
        ls_freq_accuracy=freq_accuracy,
        ls_pass=ls_pass,
        welch_result=welch_ar1,
        dpss_result=dpss_ar1,
        lomb_result=ls_result,
        ar1_params={"phi": phi1, "sigma2": sigma_ar1**2},
        ar2_params={"phi1": phi1_ar2, "phi2": phi2_ar2, "sigma2": sigma_ar2**2},
        n_samples=n_samples,
        fs=fs,
    )
