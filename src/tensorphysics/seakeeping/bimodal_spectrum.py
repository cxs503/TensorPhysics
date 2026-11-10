"""Bimodal wave spectrum: Torsethaugen swell + wind sea.

Implements the Torsethaugen two-peak spectrum model for ocean wave
spectral analysis. The bimodal spectrum is constructed as the sum of
a swell component and a wind-sea component, each following the
Torsethaugen single-peak formulation.

Torsethaugen spectrum (single peak):
    S(omega) = A * gamma^exp(-0.5*((omega-omega_p)/sigma/omega_p)^2) * omega^(-5) * exp(-B*omega^(-4))

where:
    A, B are scaling constants derived from Hs and Tp
    gamma is the peak enhancement factor
    sigma controls the spectral width

Bimodal spectrum:
    S_bimodal(omega) = S_swell(omega) + S_wind(omega)

References
----------
- Torsethaugen, K. (1996). "A two-peak wave spectrum model for
  offshore engineering applications."
- DNV-RP-C205: Environmental conditions and environmental loads
- IEC 61400-3: Wind turbine design requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "TorsethaugenSpectrum",
    "BimodalSpectrum",
    "BimodalSpectrumResult",
    "spectral_moments",
    "significant_wave_height",
    "peak_period",
    "mean_zero_crossing_period",
    "mean_wave_period",
]

#: Gravitational acceleration [m/s^2]
G_ACC = 9.80665


@dataclass(frozen=True)
class TorsethaugenSpectrum:
    """Single-peak Torsethaugen wave spectrum.

    The spectrum is normalized so that the zeroth spectral moment
    m0 = Hs^2 / 16, ensuring Hs = 4*sqrt(m0) matches the input.

    Parameters
    ----------
    hs : float
        Significant wave height [m].
    tp : float
        Spectral peak period [s].
    gamma : float, optional
        Peak enhancement factor. Default 1.0 (JONSWAP-like).
    sigma_low : float, optional
        Spectral width parameter below peak. Default 0.07.
    sigma_high : float, optional
        Spectral width parameter above peak. Default 0.09.
    """
    hs: float
    tp: float
    gamma: float = 1.0
    sigma_low: float = 0.07
    sigma_high: float = 0.09

    @property
    def omega_p(self) -> float:
        """Peak angular frequency [rad/s]."""
        return 2.0 * np.pi / self.tp

    def _unnormalized_spectrum(self, omega: np.ndarray) -> np.ndarray:
        """Unnormalized Torsethaugen spectrum shape."""
        g = G_ACC
        omega_p = self.omega_p

        # Base Pierson-Moskowitz shape
        alpha = 0.0081  # Phillips constant
        b = 1.25 * omega_p ** 4
        s_base = alpha * g ** 2 * omega ** (-5) * np.exp(-b * omega ** (-4))

        # Peak enhancement
        sigma = np.where(omega <= omega_p, self.sigma_low, self.sigma_high)
        exponent = -0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2
        peak_factor = self.gamma ** np.exp(exponent)

        return s_base * peak_factor

    def _normalization_factor(self) -> float:
        """Compute normalization factor so that m0 = Hs^2/16."""
        # Integrate unnormalized spectrum over a wide frequency range
        omega = np.logspace(-2, 1, 2000)  # 0.01 to 10 rad/s
        s_unnorm = self._unnormalized_spectrum(omega)
        # m0 in omega domain: integral S(omega) domega
        m0_unnorm = float(np.trapezoid(s_unnorm, omega))
        # Target m0 = Hs^2 / 16
        m0_target = self.hs ** 2 / 16.0
        if m0_unnorm > 0:
            return m0_target / m0_unnorm
        return 1.0

    def evaluate(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the Torsethaugen spectrum at given angular frequencies.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array [rad/s].

        Returns
        -------
        np.ndarray
            Spectral density S(omega) [m^2·s].
        """
        omega = np.asarray(omega, dtype=float)
        norm = self._normalization_factor()
        return norm * self._unnormalized_spectrum(omega)

    def evaluate_hz(self, freq: np.ndarray) -> np.ndarray:
        """Evaluate spectrum at frequency in Hz.

        Parameters
        ----------
        freq : np.ndarray
            Frequency array [Hz].

        Returns
        -------
        np.ndarray
            Spectral density S(f) [m^2/Hz].
        """
        omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
        s_omega = self.evaluate(omega)
        return s_omega * 2.0 * np.pi


@dataclass(frozen=True)
class BimodalSpectrum:
    """Bimodal wave spectrum: swell + wind sea.

    The bimodal spectrum is the sum of two Torsethaugen spectra,
    one representing the swell component and one representing the
    wind-sea component.

    Parameters
    ----------
    swell_hs : float
        Significant wave height of swell component [m].
    swell_tp : float
        Peak period of swell component [s].
    wind_hs : float
        Significant wave height of wind-sea component [m].
    wind_tp : float
        Peak period of wind-sea component [s].
    swell_gamma : float, optional
        Peak enhancement factor for swell. Default 3.0.
    wind_gamma : float, optional
        Peak enhancement factor for wind sea. Default 1.0.
    """
    swell_hs: float
    swell_tp: float
    wind_hs: float
    wind_tp: float
    swell_gamma: float = 3.0
    wind_gamma: float = 1.0

    @property
    def swell(self) -> TorsethaugenSpectrum:
        return TorsethaugenSpectrum(
            hs=self.swell_hs, tp=self.swell_tp, gamma=self.swell_gamma
        )

    @property
    def wind(self) -> TorsethaugenSpectrum:
        return TorsethaugenSpectrum(
            hs=self.wind_hs, tp=self.wind_tp, gamma=self.wind_gamma
        )

    def evaluate(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate bimodal spectrum at angular frequencies.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array [rad/s].

        Returns
        -------
        np.ndarray
            S_bimodal(omega) = S_swell(omega) + S_wind(omega) [m^2·s].
        """
        return self.swell.evaluate(omega) + self.wind.evaluate(omega)

    def evaluate_hz(self, freq: np.ndarray) -> np.ndarray:
        """Evaluate bimodal spectrum at frequency in Hz."""
        return self.swell.evaluate_hz(freq) + self.wind.evaluate_hz(freq)


@dataclass
class BimodalSpectrumResult:
    """Result of bimodal spectrum analysis.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency vector [Hz].
    spectrum : np.ndarray
        Bimodal spectral density [m^2/Hz].
    swell_spectrum : np.ndarray
        Swell component spectral density [m^2/Hz].
    wind_spectrum : np.ndarray
        Wind-sea component spectral density [m^2/Hz].
    m0 : float
        Zeroth spectral moment [m^2].
    m1 : float
        First spectral moment [m^2·Hz].
    m2 : float
        Second spectral moment [m^2·Hz^2].
    m4 : float
        Fourth spectral moment [m^2·Hz^4].
    hs : float
        Significant wave height [m] = 4*sqrt(m0).
    tp : float
        Spectral peak period [s].
    tz : float
        Mean zero-crossing period [s] = sqrt(m0/m2).
    t1 : float
        Mean wave period [s] = m0/m1.
    """
    frequencies: np.ndarray
    spectrum: np.ndarray
    swell_spectrum: np.ndarray
    wind_spectrum: np.ndarray
    m0: float
    m1: float
    m2: float
    m4: float
    hs: float
    tp: float
    tz: float
    t1: float


def spectral_moments(
    frequencies: np.ndarray,
    psd: np.ndarray,
    orders: Tuple[int, ...] = (0, 1, 2, 4),
) -> dict:
    """Compute spectral moments of a PSD.

    m_n = integral(f^n * S(f) df)

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency vector [Hz].
    psd : np.ndarray
        Power spectral density [m^2/Hz].
    orders : tuple of int
        Moment orders to compute.

    Returns
    -------
    dict
        {order: moment_value}
    """
    result = {}
    for n in orders:
        result[n] = float(np.trapezoid(psd * frequencies ** n, frequencies))
    return result


def significant_wave_height(m0: float) -> float:
    """Compute significant wave height from zeroth spectral moment.

    Hs = 4 * sqrt(m0)

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment [m^2].

    Returns
    -------
    float
        Significant wave height [m].
    """
    return 4.0 * np.sqrt(max(m0, 0.0))


def peak_period(frequencies: np.ndarray, psd: np.ndarray) -> float:
    """Compute spectral peak period.

    Tp = 1 / f_peak where f_peak = argmax(S(f))

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency vector [Hz].
    psd : np.ndarray
        Power spectral density [m^2/Hz].

    Returns
    -------
    float
        Peak period [s].
    """
    idx = np.argmax(psd)
    return 1.0 / frequencies[idx]


def mean_zero_crossing_period(m0: float, m2: float) -> float:
    """Compute mean zero-crossing period.

    Tz = sqrt(m0 / m2)

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment [m^2].
    m2 : float
        Second spectral moment [m^2·Hz^2].

    Returns
    -------
    float
        Mean zero-crossing period [s].
    """
    if m2 <= 0 or m0 <= 0:
        return 0.0
    return float(np.sqrt(m0 / m2))


def mean_wave_period(m0: float, m1: float) -> float:
    """Compute mean wave period.

    T1 = m0 / m1

    Parameters
    ----------
    m0 : float
        Zeroth spectral moment [m^2].
    m1 : float
        First spectral moment [m^2·Hz].

    Returns
    -------
    float
        Mean wave period [s].
    """
    if m1 <= 0:
        return 0.0
    return m0 / m1


def analyze_bimodal(
    bimodal: BimodalSpectrum,
    f_min: float = 0.02,
    f_max: float = 0.5,
    n_points: int = 1000,
) -> BimodalSpectrumResult:
    """Full analysis of a bimodal spectrum.

    Parameters
    ----------
    bimodal : BimodalSpectrum
        Bimodal spectrum definition.
    f_min : float
        Minimum frequency [Hz]. Default 0.02.
    f_max : float
        Maximum frequency [Hz]. Default 0.5.
    n_points : int
        Number of frequency points. Default 1000.

    Returns
    -------
    BimodalSpectrumResult
    """
    freq = np.linspace(f_min, f_max, n_points)
    s_total = bimodal.evaluate_hz(freq)
    s_swell = bimodal.swell.evaluate_hz(freq)
    s_wind = bimodal.wind.evaluate_hz(freq)

    moments = spectral_moments(freq, s_total, orders=(0, 1, 2, 4))
    m0 = moments[0]
    m1 = moments[1]
    m2 = moments[2]
    m4 = moments[4]

    hs = significant_wave_height(m0)
    tp = peak_period(freq, s_total)
    tz = mean_zero_crossing_period(m0, m2)
    t1 = mean_wave_period(m0, m1)

    return BimodalSpectrumResult(
        frequencies=freq,
        spectrum=s_total,
        swell_spectrum=s_swell,
        wind_spectrum=s_wind,
        m0=m0,
        m1=m1,
        m2=m2,
        m4=m4,
        hs=hs,
        tp=tp,
        tz=tz,
        t1=t1,
    )
