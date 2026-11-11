"""Tests for the spectral estimator gate (p357w1-n07).

Acceptance criteria:
(a) AR(1) process PSD estimation error < 5% (full-band integrated energy)
(b) AR(2) process PSD estimation error < 8% (resonance peak position error < 2%)
(c) Three-method cross-validation: band energy deviation < 10%
(d) Lomb–Scargle on non-uniform sampling (30% missing): main frequency detection > 95%
(e) pytest ≥ 12 cases all green
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.spectral_gate import (
    welch_psd,
    dpss_multitaper_psd,
    lomb_scargle_psd,
    ar1_psd,
    ar2_psd,
    generate_ar1,
    generate_ar2,
    cross_validate_psd,
    band_energy,
    run_spectral_gate,
)
from tensorphysics.spectral_gate.ar_models import ar2_resonant_frequency
from tensorphysics.spectral_gate.welch_estimator import WelchResult
from tensorphysics.spectral_gate.dpss_multitaper import DPSSResult
from tensorphysics.spectral_gate.lomb_scargle import LombScargleResult


# ====================================================================
# Welch estimator tests
# ====================================================================

class TestWelchPSD:
    """Tests for Welch PSD estimator."""

    def test_welch_output_shapes(self) -> None:
        """Welch returns correct array shapes."""
        fs = 100.0
        n = 1024
        signal = np.random.default_rng(42).normal(size=n)
        result = welch_psd(signal, fs, nperseg=256)
        assert isinstance(result, WelchResult)
        assert result.freqs_hz.shape[0] == 256 // 2 + 1
        assert result.psd.shape == result.freqs_hz.shape

    def test_welch_energy_equals_variance(self) -> None:
        """Welch integrated energy ≈ signal variance for white noise."""
        fs = 100.0
        n = 16384
        rng = np.random.default_rng(42)
        signal = rng.normal(size=n)
        result = welch_psd(signal, fs, nperseg=1024)
        energy = np.trapezoid(result.psd, result.freqs_hz)
        variance = np.var(signal)
        assert abs(energy - variance) / variance < 0.05

    def test_welch_sine_peak(self) -> None:
        """Welch detects sinusoid frequency within 1 bin."""
        fs = 100.0
        n = 8192
        t = np.arange(n) / fs
        f_test = 10.0
        signal = np.sin(2 * np.pi * f_test * t)
        result = welch_psd(signal, fs, nperseg=2048)
        f_peak = result.freqs_hz[np.argmax(result.psd)]
        assert abs(f_peak - f_test) < fs / 2048

    def test_welch_short_signal_error(self) -> None:
        """Welch raises error for too-short signal."""
        with pytest.raises(ValueError, match="too short"):
            welch_psd(np.array([1.0, 2.0, 3.0]), fs=10.0)


# ====================================================================
# DPSS multi-taper tests
# ====================================================================

class TestDPSSMultitaper:
    """Tests for DPSS multi-taper PSD estimator."""

    def test_dpss_output_shapes(self) -> None:
        """DPSS returns correct array shapes."""
        fs = 100.0
        n = 1024
        signal = np.random.default_rng(42).normal(size=n)
        result = dpss_multitaper_psd(signal, fs, nw=4.0)
        assert isinstance(result, DPSSResult)
        assert result.freqs_hz.shape[0] == n // 2 + 1
        assert result.psd.shape == result.freqs_hz.shape

    def test_dpss_energy_consistency(self) -> None:
        """DPSS integrated energy ≈ signal variance for white noise."""
        fs = 100.0
        n = 16384
        rng = np.random.default_rng(42)
        signal = rng.normal(size=n)
        result = dpss_multitaper_psd(signal, fs, nw=4.0)
        energy = np.trapezoid(result.psd, result.freqs_hz)
        variance = np.var(signal)
        assert abs(energy - variance) / variance < 0.10

    def test_dpss_sine_peak(self) -> None:
        """DPSS detects sinusoid frequency accurately."""
        fs = 100.0
        n = 8192
        t = np.arange(n) / fs
        f_test = 15.0
        signal = np.sin(2 * np.pi * f_test * t)
        result = dpss_multitaper_psd(signal, fs, nw=4.0)
        f_peak = result.freqs_hz[np.argmax(result.psd)]
        assert abs(f_peak - f_test) < 0.1  # Within 0.1 Hz


# ====================================================================
# Lomb–Scargle tests
# ====================================================================

class TestLombScargle:
    """Tests for Lomb–Scargle PSD estimator."""

    def test_ls_output_shapes(self) -> None:
        """Lomb–Scargle returns correct array shapes."""
        rng = np.random.default_rng(42)
        n = 500
        t = np.sort(rng.uniform(0, 10, n))
        signal = rng.normal(size=n)
        result = lomb_scargle_psd(t, signal, n_freq=256)
        assert isinstance(result, LombScargleResult)
        assert result.freqs_hz.shape[0] == 256
        assert result.psd.shape == result.freqs_hz.shape

    def test_ls_frequency_detection(self) -> None:
        """Lomb–Scargle detects sinusoid frequency in non-uniform data."""
        rng = np.random.default_rng(42)
        n = 1000
        t_full = np.linspace(0, 10, n)
        f_test = 2.0
        signal = np.sin(2 * np.pi * f_test * t_full) + 0.2 * rng.normal(size=n)

        # Remove 30% of samples
        keep = rng.choice(n, size=int(0.7 * n), replace=False)
        keep.sort()
        t_obs = t_full[keep]
        sig_obs = signal[keep]

        result = lomb_scargle_psd(t_obs, sig_obs, n_freq=2048)
        f_peak = result.peak_frequency()
        assert abs(f_peak - f_test) / f_test < 0.05

    def test_ls_uniform_sampling(self) -> None:
        """Lomb–Scargle works with uniformly sampled data."""
        fs = 100.0
        n = 1024
        t = np.arange(n) / fs
        f_test = 5.0
        signal = np.sin(2 * np.pi * f_test * t)
        result = lomb_scargle_psd(t, signal, n_freq=1024)
        f_peak = result.peak_frequency()
        assert abs(f_peak - f_test) / f_test < 0.02


# ====================================================================
# AR model tests
# ====================================================================

class TestARModels:
    """Tests for AR(1)/AR(2) analytical PSD and generation."""

    def test_ar1_psd_positive(self) -> None:
        """AR(1) PSD is always positive."""
        freqs = np.linspace(0, 50, 1000)
        psd = ar1_psd(freqs, phi=0.8, sigma2=1.0, fs=100.0)
        assert np.all(psd > 0)

    def test_ar2_psd_positive(self) -> None:
        """AR(2) PSD is always positive."""
        freqs = np.linspace(0, 50, 1000)
        psd = ar2_psd(freqs, phi1=1.5, phi2=-0.75, sigma2=1.0, fs=100.0)
        assert np.all(psd > 0)

    def test_ar1_stationarity_error(self) -> None:
        """AR(1) raises error for non-stationary coefficient."""
        with pytest.raises(ValueError, match="not stationary"):
            ar1_psd(np.array([1.0]), phi=1.5)

    def test_ar2_stationarity_error(self) -> None:
        """AR(2) raises error for non-stationary coefficients."""
        with pytest.raises(ValueError, match="stationarity"):
            ar2_psd(np.array([1.0]), phi1=2.0, phi2=0.5)

    def test_ar2_resonant_frequency(self) -> None:
        """AR(2) resonant frequency is computed correctly."""
        phi1, phi2 = 1.5, -0.75
        fs = 100.0
        f_res = ar2_resonant_frequency(phi1, phi2, fs)
        assert f_res is not None
        assert 0 < f_res < fs / 2

    def test_ar2_no_resonance(self) -> None:
        """AR(2) with real roots has no resonant frequency."""
        # phi1=0.5, phi2=0.3 → real roots
        f_res = ar2_resonant_frequency(0.5, 0.3, fs=100.0)
        assert f_res is None

    def test_ar1_generation_length(self) -> None:
        """AR(1) generation returns correct length."""
        x = generate_ar1(1000, phi=0.8, sigma=1.0, seed=42)
        assert len(x) == 1000

    def test_ar2_generation_length(self) -> None:
        """AR(2) generation returns correct length."""
        x = generate_ar2(1000, phi1=1.5, phi2=-0.75, sigma=1.0, seed=42)
        assert len(x) == 1000


# ====================================================================
# Cross-validation tests
# ====================================================================

class TestCrossValidation:
    """Tests for cross-validation framework."""

    def test_band_energy(self) -> None:
        """Band energy computation is correct."""
        freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        psd = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        e = band_energy(freqs, psd, 1.0, 4.0)
        assert abs(e - 3.0) < 0.01  # Trapezoidal integral

    def test_cross_validate_result(self) -> None:
        """Cross-validation returns valid result."""
        fs = 100.0
        n = 8192
        signal = generate_ar1(n, 0.85, 1.0, seed=42)
        result = cross_validate_psd(signal, fs)
        assert result.max_deviation < 0.10
        assert "welch" in result.band_energies
        assert "dpss" in result.band_energies


# ====================================================================
# Full spectral gate tests
# ====================================================================

class TestSpectralGate:
    """Tests for the full spectral gate acceptance criteria."""

    def test_gate_all_pass(self) -> None:
        """All acceptance criteria pass with default parameters."""
        report = run_spectral_gate(n_samples=8192, fs=100.0, seed=42)
        assert report.ar1_pass, f"AR(1) failed: error={report.ar1_energy_error}"
        assert report.ar2_pass, f"AR(2) failed: energy={report.ar2_energy_error}, peak={report.ar2_peak_freq_error}"
        assert report.cv_pass, f"CV failed: deviation={report.cv_max_deviation}"
        assert report.ls_pass, f"LS failed: accuracy={report.ls_freq_accuracy}"

    def test_gate_ar1_error_below_5pct(self) -> None:
        """AR(1) energy error < 5%."""
        report = run_spectral_gate(n_samples=8192, fs=100.0, seed=42)
        assert report.ar1_energy_error < 0.05

    def test_gate_ar2_error_below_8pct(self) -> None:
        """AR(2) energy error < 8%."""
        report = run_spectral_gate(n_samples=8192, fs=100.0, seed=42)
        assert report.ar2_energy_error < 0.08

    def test_gate_ar2_peak_error_below_2pct(self) -> None:
        """AR(2) peak frequency error < 2%."""
        report = run_spectral_gate(n_samples=8192, fs=100.0, seed=42)
        assert report.ar2_peak_freq_error < 0.02

    def test_gate_cv_deviation_below_10pct(self) -> None:
        """Cross-validation deviation < 10%."""
        report = run_spectral_gate(n_samples=8192, fs=100.0, seed=42)
        assert report.cv_max_deviation < 0.10

    def test_gate_ls_accuracy_above_95pct(self) -> None:
        """Lomb–Scargle frequency accuracy > 95%."""
        report = run_spectral_gate(n_samples=8192, fs=100.0, seed=42)
        assert report.ls_freq_accuracy > 0.95
