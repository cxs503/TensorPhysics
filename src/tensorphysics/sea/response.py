"""Response quantities from SEA modal energies.

From the solved modal energies E_i(f), compute:
    1. Vibration velocity level (dB re 1e-9 m/s)
    2. Radiated sound power level (dB re 1e-12 W)
    3. Sound pressure level (dB re 20 μPa)

Reference values:
    - Velocity: v_ref = 1e-9 m/s (ISO standard for structure-borne vibration)
    - Sound power: W_ref = 1e-12 W
    - Sound pressure: p_ref = 20 μPa = 2e-5 Pa
"""
from __future__ import annotations

import math

from .subsystems import PlateSubsystem, CavitySubsystem


# Reference values
V_REF = 1e-9       # m/s
W_REF = 1e-12      # W
P_REF = 2e-5       # Pa


def vibration_velocity_level(
    modal_energy: float,
    mass: float,
    frequency: float,
    v_ref: float = V_REF,
) -> float:
    """Vibration velocity level in dB.

    From modal energy E and subsystem mass M:
        <v²> = omega² * E / M
        v_rms = sqrt(<v²>)
        Lv = 20 * log10(v_rms / v_ref)

    Args:
        modal_energy: modal energy (J).
        mass: subsystem mass (kg).
        frequency: center frequency (Hz).
        v_ref: reference velocity (m/s).

    Returns:
        Velocity level in dB re v_ref.
    """
    if modal_energy <= 0:
        return -float('inf')
    if mass <= 0:
        raise ValueError("mass must be positive")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    omega = 2.0 * math.pi * frequency
    v_squared = omega ** 2 * modal_energy / mass
    v_rms = math.sqrt(v_squared)

    return 20.0 * math.log10(v_rms / v_ref)


def sound_power_level(
    modal_energy: float,
    frequency: float,
    radiation_loss_factor: float,
    w_ref: float = W_REF,
) -> float:
    """Radiated sound power level in dB.

    W_rad = omega * eta_rad * E
    Lw = 10 * log10(W_rad / W_ref)

    Args:
        modal_energy: modal energy (J).
        frequency: center frequency (Hz).
        radiation_loss_factor: radiation loss factor.
        w_ref: reference sound power (W).

    Returns:
        Sound power level in dB re W_ref.
    """
    if modal_energy <= 0:
        return -float('inf')
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if radiation_loss_factor <= 0:
        return -float('inf')

    omega = 2.0 * math.pi * frequency
    w_rad = omega * radiation_loss_factor * modal_energy

    return 10.0 * math.log10(w_rad / w_ref)


def sound_pressure_level(
    modal_energy: float,
    volume: float,
    sound_speed: float = 343.0,
    density: float = 1.225,
    p_ref: float = P_REF,
) -> float:
    """Sound pressure level in an acoustic cavity.

    For a cavity with modal energy E_c:
        <p²> = rho * c² * E_c / V
        Lp = 10 * log10(<p²> / p_ref²)
           = 10 * log10(rho * c² * E_c / (V * p_ref²))

    Args:
        modal_energy: cavity modal energy (J).
        volume: cavity volume (m³).
        sound_speed: sound speed in fluid (m/s).
        density: fluid density (kg/m³).
        p_ref: reference pressure (Pa).

    Returns:
        Sound pressure level in dB re p_ref.
    """
    if modal_energy <= 0:
        return -float('inf')
    if volume <= 0:
        raise ValueError("volume must be positive")

    p_squared = density * sound_speed ** 2 * modal_energy / volume

    return 10.0 * math.log10(p_squared / p_ref ** 2)


def vibration_velocity_from_energy(
    modal_energy: float,
    mass: float,
    frequency: float,
) -> float:
    """RMS vibration velocity from modal energy.

    v_rms = omega * sqrt(E / M)

    Args:
        modal_energy: modal energy (J).
        mass: subsystem mass (kg).
        frequency: center frequency (Hz).

    Returns:
        RMS velocity (m/s).
    """
    if modal_energy <= 0:
        return 0.0
    if mass <= 0:
        raise ValueError("mass must be positive")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    omega = 2.0 * math.pi * frequency
    return omega * math.sqrt(modal_energy / mass)
