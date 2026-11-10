"""Underwater acoustics: ice-breaking noise, Thorp propagation, marine mammal thresholds.

Implements:
- Thorp empirical absorption formula (dB/km vs frequency)
- Source level from ice-breaking energy
- Received level at range with spreading + absorption
- 1/3-octave SPL computation
- Marine mammal noise exposure criteria (Southall et al. 2007 / NMFS 2018)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

__all__ = [
    "ThorpAttenuation",
    "SourceLevel",
    "ReceivedLevel",
    "MarineMammalThresholds",
    "thorp_attenuation_db_per_km",
    "source_level_from_breaking_energy",
    "received_level_at_range",
    "one_third_octave_spl",
    "marine_mammal_exceedance",
]

# Reference pressure for underwater acoustics
P_REF = 1.0e-6  # 1 μPa


@dataclass
class ThorpAttenuation:
    """Thorp empirical absorption coefficient.

    Thorp (1967): α(f) in dB/km as a function of frequency f in kHz.

    α(f) = 0.11·f²/(1+f²) + 44·f²/(4100+f²) + 2.75e-4·f² + 0.003

    Valid for f = 0.1 to 100 kHz, seawater at typical conditions.
    """

    @staticmethod
    def alpha_db_per_km(f_khz: np.ndarray | float) -> np.ndarray | float:
        """Thorp absorption coefficient (dB/km).

        Parameters
        ----------
        f_khz : np.ndarray or float
            Frequency in kHz.

        Returns
        -------
        np.ndarray or float
            Absorption coefficient in dB/km.
        """
        f = np.asarray(f_khz, dtype=float)
        term1 = 0.11 * f**2 / (1.0 + f**2)       # Boric acid relaxation
        term2 = 44.0 * f**2 / (4100.0 + f**2)     # Magnesium sulfate relaxation
        term3 = 2.75e-4 * f**2                     # Pure water viscosity
        term4 = 0.003                               # Baseline
        return term1 + term2 + term3 + term4


def thorp_attenuation_db_per_km(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    """Convenience wrapper: Thorp absorption at given frequency in Hz.

    Parameters
    ----------
    frequency_hz : np.ndarray or float
        Frequency in Hz.

    Returns
    -------
    np.ndarray or float
        Absorption in dB/km.
    """
    return ThorpAttenuation.alpha_db_per_km(np.asarray(frequency_hz, dtype=float) / 1000.0)


@dataclass
class SourceLevel:
    """Acoustic source level from ice-breaking energy.

    Source level SL (dB re 1 μPa @ 1 m) is related to acoustic power W_ac:
        SL = 170.8 + 10·log10(W_ac)

    Ice-breaking energy is converted to acoustic power via a radiation
    efficiency factor η_ac (typically 1e-6 to 1e-4 for ice fracture).
    """
    breaking_energy_j: float
    radiation_efficiency: float = 1e-5
    duration_s: float = 1.0

    @property
    def acoustic_power_w(self) -> float:
        """Acoustic power (W)."""
        if self.breaking_energy_j <= 0:
            return 0.0
        return self.radiation_efficiency * self.breaking_energy_j / self.duration_s

    @property
    def sl_db(self) -> float:
        """Source level in dB re 1 μPa @ 1 m.

        SL = 170.8 + 10·log10(W_ac) for W_ac in watts.
        Reference: Urick, Principles of Underwater Sound.
        """
        if self.acoustic_power_w <= 0:
            return -np.inf
        return 170.8 + 10.0 * np.log10(self.acoustic_power_w)


def source_level_from_breaking_energy(
    breaking_energy_j: float,
    radiation_efficiency: float = 1e-5,
    duration_s: float = 1.0,
) -> SourceLevel:
    """Create SourceLevel from ice-breaking energy.

    Parameters
    ----------
    breaking_energy_j : float
        Ice-breaking energy (J).
    radiation_efficiency : float
        Fraction of mechanical energy radiated as sound.
    duration_s : float
        Duration of breaking event (s).

    Returns
    -------
    SourceLevel
    """
    return SourceLevel(
        breaking_energy_j=breaking_energy_j,
        radiation_efficiency=radiation_efficiency,
        duration_s=duration_s,
    )


@dataclass
class ReceivedLevel:
    """Received sound level at a given range.

    RL = SL - TL

    where TL = transmission loss = spreading loss + absorption loss.

    Spreading loss:
    - Spherical: 20·log10(r)
    - Cylindrical: 10·log10(r)
    - Practical (mixed): 20·log10(r) for r < r_transition,
      20·log10(r_t) + 10·log10(r/r_t) for r >= r_transition
    """
    source_level_db: float
    range_m: float
    spreading: str = "spherical"
    absorption_db_km: float = 0.0
    transition_range_m: float = 1000.0

    @property
    def spreading_loss_db(self) -> float:
        """Spreading loss in dB."""
        if self.range_m <= 0:
            return 0.0
        r_km = self.range_m / 1000.0
        if self.spreading == "spherical":
            return 20.0 * np.log10(self.range_m)
        elif self.spreading == "cylindrical":
            return 10.0 * np.log10(self.range_m)
        elif self.spreading == "practical":
            r_t = self.transition_range_m
            if self.range_m <= r_t:
                return 20.0 * np.log10(self.range_m)
            else:
                return 20.0 * np.log10(r_t) + 10.0 * np.log10(self.range_m / r_t)
        else:
            return 20.0 * np.log10(self.range_m)

    @property
    def absorption_loss_db(self) -> float:
        """Absorption loss in dB."""
        if self.range_m <= 0 or self.absorption_db_km <= 0:
            return 0.0
        return self.absorption_db_km * (self.range_m / 1000.0)

    @property
    def transmission_loss_db(self) -> float:
        """Total transmission loss (dB)."""
        return self.spreading_loss_db + self.absorption_loss_db

    @property
    def rl_db(self) -> float:
        """Received level in dB re 1 μPa."""
        if self.source_level_db == -np.inf:
            return -np.inf
        return self.source_level_db - self.transmission_loss_db


def received_level_at_range(
    source_level_db: float,
    range_m: float,
    frequency_hz: float,
    spreading: str = "spherical",
) -> ReceivedLevel:
    """Compute received level at a given range with Thorp absorption.

    Parameters
    ----------
    source_level_db : float
        Source level (dB re 1 μPa @ 1 m).
    range_m : float
        Range from source (m).
    frequency_hz : float
        Frequency for absorption calculation (Hz).
    spreading : str
        Spreading model: 'spherical', 'cylindrical', or 'practical'.

    Returns
    -------
    ReceivedLevel
    """
    alpha = thorp_attenuation_db_per_km(frequency_hz)
    return ReceivedLevel(
        source_level_db=source_level_db,
        range_m=range_m,
        spreading=spreading,
        absorption_db_km=float(alpha),
    )


def one_third_octave_spl(
    source_level_db: float,
    frequencies_hz: np.ndarray,
    range_m: float,
    breaking_energy_spectrum: Optional[np.ndarray] = None,
    radiation_efficiency: float = 1e-5,
    duration_s: float = 1.0,
    spreading: str = "spherical",
) -> Dict[str, np.ndarray]:
    """Compute 1/3-octave SPL spectrum at range.

    For each frequency band:
    1. Compute source level from band energy
    2. Compute Thorp absorption at band centre frequency
    3. Compute received level

    Parameters
    ----------
    source_level_db : float
        Broadband source level (dB re 1 μPa @ 1 m).
    frequencies_hz : np.ndarray
        Centre frequencies of 1/3-octave bands (Hz).
    range_m : float
        Range from source (m).
    breaking_energy_spectrum : np.ndarray, optional
        Energy per band (J). If None, assumes flat spectrum.
    radiation_efficiency : float
        Acoustic radiation efficiency.
    duration_s : float
        Event duration (s).
    spreading : str
        Spreading model.

    Returns
    -------
    dict
        {'frequencies_hz', 'spl_db', 'absorption_db_km', 'transmission_loss_db'}
    """
    n_bands = len(frequencies_hz)
    spl_db = np.full(n_bands, -np.inf)
    absorption = np.zeros(n_bands)
    tl = np.zeros(n_bands)

    if breaking_energy_spectrum is not None:
        total_energy = np.sum(breaking_energy_spectrum)
    else:
        total_energy = 1.0  # Normalised

    for i, f in enumerate(frequencies_hz):
        # Band energy fraction
        if breaking_energy_spectrum is not None and total_energy > 0:
            band_energy = breaking_energy_spectrum[i]
        else:
            band_energy = total_energy / n_bands

        if band_energy <= 0:
            continue

        # Source level for this band
        sl_band = source_level_db + 10.0 * np.log10(band_energy / total_energy) if total_energy > 0 else -np.inf

        # Thorp absorption
        alpha = thorp_attenuation_db_per_km(f)
        absorption[i] = float(alpha)

        # Received level
        rl = received_level_at_range(
            source_level_db=sl_band,
            range_m=range_m,
            frequency_hz=f,
            spreading=spreading,
        )
        spl_db[i] = rl.rl_db
        tl[i] = rl.transmission_loss_db

    return {
        "frequencies_hz": frequencies_hz,
        "spl_db": spl_db,
        "absorption_db_km": absorption,
        "transmission_loss_db": tl,
    }


@dataclass
class MarineMammalThresholds:
    """Marine mammal noise exposure criteria.

    Based on Southall et al. (2007) and NMFS (2018) Technical Guidance.

    Thresholds are given as dual criteria:
    - Peak SPL (dB re 1 μPa) — instantaneous injury threshold
    - Cumulative SEL (dB re 1 μPa^2·s) — TTS/PTS onset

    Functional hearing groups:
    - LF (low-frequency cetaceans): baleen whales
    - MF (mid-frequency cetaceans): dolphins, sperm whales
    - HF (high-frequency cetaceans): porpoises
    - PW (phocid seals)
    - OW (otariid seals)
    """
    # Peak SPL thresholds (dB re 1 μPa) — injury onset
    peak_spl_injury: Dict[str, float] = field(default_factory=lambda: {
        "LF": 219,
        "MF": 230,
        "HF": 202,
        "PW": 218,
        "OW": 232,
    })

    # Cumulative SEL thresholds (dB re 1 μPa^2·s) — PTS onset
    sel_pts: Dict[str, float] = field(default_factory=lambda: {
        "LF": 183,
        "MF": 185,
        "HF": 155,
        "PW": 185,
        "OW": 187,
    })

    # TTS onset SEL (dB re 1 μPa^2·s)
    sel_tts: Dict[str, float] = field(default_factory=lambda: {
        "LF": 171,
        "MF": 170,
        "HF": 145,
        "PW": 174,
        "OW": 175,
    })

    # Behavioural disturbance threshold (dB re 1 μPa)
    behavioural_spl: Dict[str, float] = field(default_factory=lambda: {
        "LF": 160,
        "MF": 160,
        "HF": 140,
        "PW": 160,
        "OW": 160,
    })


def marine_mammal_exceedance(
    spl_db: np.ndarray,
    frequencies_hz: np.ndarray,
    duration_s: float = 1.0,
    group: str = "MF",
    thresholds: Optional[MarineMammalThresholds] = None,
) -> Dict[str, bool | float]:
    """Check if SPL exceeds marine mammal thresholds.

    Parameters
    ----------
    spl_db : np.ndarray
        Received SPL per 1/3-octave band (dB re 1 μPa).
    frequencies_hz : np.ndarray
        Centre frequencies (Hz).
    duration_s : float
        Exposure duration (s).
    group : str
        Hearing group: 'LF', 'MF', 'HF', 'PW', 'OW'.
    thresholds : MarineMammalThresholds, optional
        Threshold values. Default = Southall/NMFS values.

    Returns
    -------
    dict
        {'peak_exceeds_injury', 'sel_exceeds_pts', 'sel_exceeds_tts',
         'behavioural_exceeds', 'max_spl_db', 'cumulative_sel_db'}
    """
    if thresholds is None:
        thresholds = MarineMammalThresholds()

    # Filter out -inf values
    valid = np.isfinite(spl_db)
    if not np.any(valid):
        return {
            "peak_exceeds_injury": False,
            "sel_exceeds_pts": False,
            "sel_exceeds_tts": False,
            "behavioural_exceeds": False,
            "max_spl_db": -np.inf,
            "cumulative_sel_db": -np.inf,
        }

    max_spl = float(np.max(spl_db[valid]))

    # Cumulative SEL: SEL = 10·log10(Σ 10^(SPL/10) · Δt)
    # For 1/3-octave bands, sum intensity across bands
    intensities = 10.0 ** (spl_db[valid] / 10.0)
    cumulative_intensity = float(np.sum(intensities))
    sel_db = 10.0 * np.log10(cumulative_intensity * duration_s) if cumulative_intensity > 0 else -np.inf

    return {
        "peak_exceeds_injury": max_spl > thresholds.peak_spl_injury.get(group, 230),
        "sel_exceeds_pts": sel_db > thresholds.sel_pts.get(group, 185),
        "sel_exceeds_tts": sel_db > thresholds.sel_tts.get(group, 170),
        "behavioural_exceeds": max_spl > thresholds.behavioural_spl.get(group, 160),
        "max_spl_db": max_spl,
        "cumulative_sel_db": sel_db,
    }
