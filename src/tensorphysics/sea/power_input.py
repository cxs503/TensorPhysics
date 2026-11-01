"""Ice load Power Spectral Density (PSD) and input power calculation.

Ice-induced loads on ship hulls are stochastic. The PSD characterizes
the frequency content of the ice force, which drives the SEA energy input.

Three PSD models are supported:
    1. White noise: constant PSD across frequency band
    2. 1/f (pink noise): PSD ∝ 1/f
    3. Measured PSD: user-provided frequency-PSD pairs (interpolated)

Input power to subsystem i:
    Pin_i(f) = PSD_F(f) * Re{Y_ii(f)}

where Y_ii is the point mobility (velocity/force) at the excitation point.

For a plate, the point mobility (averaged over position):
    |Y|² ≈ 1 / (8 * sqrt(D * rho * h))

References:
    - Kujala, P. (1994). Ice-induced loads on ships.
    - Riska, K. et al. (1997). Ice load modelling.
    - ISO 19906: Arctic offshore structures.
"""
from __future__ import annotations

import math
from typing import Literal

from .subsystems import PlateSubsystem


def ice_load_psd(
    frequency: float,
    psd_type: Literal["white", "one_over_f", "measured"] = "white",
    psd_level: float = 1.0,
    measured_freqs: list[float] | None = None,
    measured_psds: list[float] | None = None,
    reference_force: float = 1e6,
) -> float:
    """Ice load Power Spectral Density at a given frequency.

    Args:
        frequency: frequency (Hz).
        psd_type: type of PSD model.
        psd_level: reference PSD level (N²/Hz).
            - For white: constant PSD value.
            - For 1/f: PSD at 1 Hz reference.
            - For measured: scaling factor for measured data.
        measured_freqs: frequency points for measured PSD (Hz).
        measured_psds: PSD values at measured frequencies (N²/Hz).
        reference_force: reference ice force magnitude (N), used for scaling.

    Returns:
        PSD value at the given frequency (N²/Hz).
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    if psd_type == "white":
        return psd_level * (reference_force / 1e6) ** 2

    elif psd_type == "one_over_f":
        # PSD(f) = PSD_ref / f
        return psd_level * (reference_force / 1e6) ** 2 / frequency

    elif psd_type == "measured":
        if measured_freqs is None or measured_psds is None:
            raise ValueError("measured_freqs and measured_psds required for 'measured' PSD type")
        if len(measured_freqs) != len(measured_psds):
            raise ValueError("measured_freqs and measured_psds must have same length")
        if len(measured_freqs) < 2:
            raise ValueError("at least 2 measured points required for interpolation")

        # Linear interpolation in log-log space
        log_freqs = [math.log(f) for f in measured_freqs]
        log_psds = [math.log(p) for p in measured_psds]

        log_f = math.log(frequency)

        # Find bracketing points
        if log_f <= log_freqs[0]:
            # Extrapolate from first two points
            slope = (log_psds[1] - log_psds[0]) / (log_freqs[1] - log_freqs[0])
            log_psd = log_psds[0] + slope * (log_f - log_freqs[0])
        elif log_f >= log_freqs[-1]:
            # Extrapolate from last two points
            slope = (log_psds[-1] - log_psds[-2]) / (log_freqs[-1] - log_freqs[-2])
            log_psd = log_psds[-1] + slope * (log_f - log_freqs[-1])
        else:
            # Interpolate
            for k in range(len(log_freqs) - 1):
                if log_freqs[k] <= log_f <= log_freqs[k + 1]:
                    slope = (log_psds[k + 1] - log_psds[k]) / (log_freqs[k + 1] - log_freqs[k])
                    log_psd = log_psds[k] + slope * (log_f - log_freqs[k])
                    break

        return math.exp(log_psd) * (reference_force / 1e6) ** 2

    else:
        raise ValueError(f"Unknown psd_type: {psd_type}")


def point_mobility_plate(
    plate: PlateSubsystem,
    frequency: float,
) -> float:
    """Point mobility (mean-square velocity per unit force) for a plate.

    For a thin plate, the spatially averaged point mobility:
        |Y|² = 1 / (8 * sqrt(D * rho * h))

    This is frequency-independent in the asymptotic limit.

    Args:
        plate: PlateSubsystem definition.
        frequency: center frequency (Hz).

    Returns:
        Point mobility magnitude squared (m²/(N²·s²)).
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    D = plate.bending_stiffness
    rho_h = plate.density * plate.thickness

    return 1.0 / (8.0 * math.sqrt(D * rho_h))


def input_power_from_psd(
    plate: PlateSubsystem,
    frequency: float,
    psd_value: float,
) -> float:
    """Input power to a plate subsystem from ice load PSD.

    Pin(f) = PSD_F(f) * Re{Y(f)}

    For a plate, using the asymptotic point mobility:
    Pin(f) = PSD_F(f) / (8 * sqrt(D * rho * h))

    Args:
        plate: PlateSubsystem definition.
        frequency: center frequency (Hz).
        psd_value: force PSD at this frequency (N²/Hz).

    Returns:
        Input power (W/Hz).
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if psd_value < 0:
        raise ValueError("PSD must be non-negative")

    mobility = point_mobility_plate(plate, frequency)
    return psd_value * mobility
