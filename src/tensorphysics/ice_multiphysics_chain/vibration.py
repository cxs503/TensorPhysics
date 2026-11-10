"""Ice-induced vibration: 1/3-octave acceleration spectrum and ISO 19906 envelope.

Implements:
- 1/3-octave band filtering of acceleration PSD (IEC 61260 centre frequencies)
- ISO 19906 ice-action envelope for Arctic structures
- Energy conservation verification between full-band and band-summed spectra
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

__all__ = [
    "OneThirdOctaveBands",
    "ISO19906Envelope",
    "compute_one_third_octave_spectrum",
    "compute_iso19906_envelope",
    "acceleration_rms_by_band",
    "energy_conservation_error",
]

# IEC 61260 nominal 1/3-octave centre frequencies (Hz)
_IEC_13_BANDS = np.array([
    0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0,
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
    125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0,
])


@dataclass
class OneThirdOctaveBands:
    """1/3-octave band definitions per IEC 61260.

    For centre frequency f_c:
        f_lower = f_c / 2^(1/6)
        f_upper = f_c * 2^(1/6)
    """
    centre_frequencies: np.ndarray
    lower_edges: np.ndarray
    upper_edges: np.ndarray

    @classmethod
    def standard(cls, f_min: float = 0.5, f_max: float = 10000.0) -> "OneThirdOctaveBands":
        """Return standard IEC bands within [f_min, f_max]."""
        mask = (_IEC_13_BANDS >= f_min) & (_IEC_13_BANDS <= f_max)
        fc = _IEC_13_BANDS[mask]
        sixth_root_2 = 2.0 ** (1.0 / 6.0)
        return cls(
            centre_frequencies=fc,
            lower_edges=fc / sixth_root_2,
            upper_edges=fc * sixth_root_2,
        )


@dataclass
class ISO19906Envelope:
    """ISO 19906 ice-action design envelope for vibration.

    The envelope provides a conservative bound on ice-induced vibration
    acceleration levels as a function of frequency.

    Parameters
    ----------
    ice_thickness_m : float
        Level ice thickness (m).
    ice_speed_m_s : float
        Ice drift / vessel speed relative to ice (m/s).
    structure_width_m : float
        Nominal width of structure interacting with ice (m).
    ice_strength_mpa : float
        Uniaxial compressive strength of ice (MPa).
    """
    ice_thickness_m: float
    ice_speed_m_s: float
    structure_width_m: float
    ice_strength_mpa: float

    def acceleration_bound(self, frequencies: np.ndarray) -> np.ndarray:
        """Compute ISO 19906 acceleration envelope (m/s^2).

        The envelope follows a piecewise form:
        - Low freq: constant acceleration plateau
        - Mid freq: ~1/f decay (constant velocity)
        - High freq: ~1/f^2 decay (constant displacement)

        Reference: ISO 19906:2019 §A.8, ice-induced vibration guidance.
        """
        # Characteristic ice force scale
        F_char = self.ice_strength_mpa * 1e6 * self.ice_thickness_m * self.structure_width_m

        # Characteristic frequency from ice crushing
        f_crush = self.ice_speed_m_s / (2.0 * self.ice_thickness_m) if self.ice_thickness_m > 0 else 1.0

        # Reference acceleration at characteristic frequency
        # a_ref ~ F_char / (effective_mass) with empirical scaling
        effective_mass = 1e6 * self.structure_width_m  # kg, simplified
        a_ref = F_char / effective_mass  # m/s^2

        # Envelope: piecewise
        a_env = np.zeros_like(frequencies, dtype=float)
        for i, f in enumerate(frequencies):
            if f <= f_crush:
                # Low-freq plateau
                a_env[i] = a_ref
            elif f <= 5.0 * f_crush:
                # Mid-freq: constant velocity region (~1/f)
                a_env[i] = a_ref * (f_crush / f)
            else:
                # High-freq: constant displacement region (~1/f^2)
                a_env[i] = a_ref * (f_crush / f) ** 2

        return a_env


def compute_one_third_octave_spectrum(
    frequencies: np.ndarray,
    psd: np.ndarray,
    bands: Optional[OneThirdOctaveBands] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate PSD into 1/3-octave bands.

    For each band, integrate the PSD over the band edges:
        E_band = ∫_{f_lo}^{f_hi} S(f) df

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    psd : np.ndarray
        Power spectral density (e.g. (m/s^2)^2/Hz for acceleration).
    bands : OneThirdOctaveBands, optional
        Band definitions. Default = standard IEC bands 0.5-10000 Hz.

    Returns
    -------
    centre_freqs : np.ndarray
        Centre frequencies of bands with non-zero energy.
    band_energies : np.ndarray
        Integrated energy per band.
    """
    if bands is None:
        bands = OneThirdOctaveBands.standard()

    band_energies = np.zeros(len(bands.centre_frequencies))

    for i, (f_lo, f_hi) in enumerate(zip(bands.lower_edges, bands.upper_edges)):
        # Find indices within band
        mask = (frequencies >= f_lo) & (frequencies < f_hi)
        if np.any(mask):
            f_in = frequencies[mask]
            p_in = psd[mask]
            band_energies[i] = float(np.trapezoid(p_in, f_in))

    # Only return bands with energy
    active = band_energies > 0
    return bands.centre_frequencies[active], band_energies[active]


def compute_iso19906_envelope(
    frequencies: np.ndarray,
    ice_thickness_m: float = 1.0,
    ice_speed_m_s: float = 1.0,
    structure_width_m: float = 30.0,
    ice_strength_mpa: float = 3.0,
) -> ISO19906Envelope:
    """Create ISO 19906 envelope and compute acceleration bounds.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency vector (Hz).
    ice_thickness_m : float
        Level ice thickness (m).
    ice_speed_m_s : float
        Relative ice-vessel speed (m/s).
    structure_width_m : float
        Structure width interacting with ice (m).
    ice_strength_mpa : float
        Ice compressive strength (MPa).

    Returns
    -------
    ISO19906Envelope
    """
    env = ISO19906Envelope(
        ice_thickness_m=ice_thickness_m,
        ice_speed_m_s=ice_speed_m_s,
        structure_width_m=structure_width_m,
        ice_strength_mpa=ice_strength_mpa,
    )
    return env


def acceleration_rms_by_band(
    band_energies: np.ndarray,
) -> np.ndarray:
    """Convert band energies to RMS acceleration per band.

    a_rms_band = sqrt(E_band)

    Parameters
    ----------
    band_energies : np.ndarray
        Integrated energy per 1/3-octave band ((m/s^2)^2).

    Returns
    -------
    np.ndarray
        RMS acceleration per band (m/s^2).
    """
    return np.sqrt(band_energies)


def energy_conservation_error(
    full_band_energy: float,
    band_energies: np.ndarray,
) -> float:
    """Compute relative energy conservation error.

    error = |E_full - ΣE_band| / E_full

    Parameters
    ----------
    full_band_energy : float
        Total energy from full-band integration.
    band_energies : np.ndarray
        Sum of 1/3-octave band energies.

    Returns
    -------
    float
        Relative error (fraction). Should be < 0.01 (1%).
    """
    if full_band_energy <= 0:
        return 0.0
    summed = float(np.sum(band_energies))
    return abs(full_band_energy - summed) / full_band_energy
