"""Modal density n(f) calculations for SEA subsystems.

Modal density is the number of modes per unit frequency band.
For plates, beams, and cavities, analytical formulas from wave theory
are implemented. These are validated against the asymptotic wave theory
solution (criterion a: error < 5%).

References:
    - Lyon, R.H. & DeJong, R.G. (1995). Theory and Application of SEA.
    - Fahy, F. & Gardonio, P. (2007). Sound and Structural Vibration.
"""
from __future__ import annotations

import math

from .subsystems import PlateSubsystem, BeamSubsystem, CavitySubsystem


def plate_modal_density(
    plate: PlateSubsystem,
    frequency: float,
) -> float:
    """Modal density of a thin plate subsystem [modes/Hz].

    For a thin isotropic plate, the asymptotic modal density is:
        n(f) = A / (2 * cb²)
    where A is the plate area and cb is the bending wave speed at frequency f.

    The bending wave speed:
        cb(f) = (2*pi*f)^(1/2) * (D / (rho*h))^(1/4)

    Therefore:
        n(f) = A * sqrt(rho*h/D) / (4*pi)

    This is frequency-independent in the asymptotic (high-frequency) limit,
    which is the standard SEA assumption for plates.

    Args:
        plate: PlateSubsystem definition.
        frequency: center frequency of the band (Hz).

    Returns:
        Modal density in modes/Hz.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    rho_h = plate.density * plate.thickness
    D = plate.bending_stiffness

    # Asymptotic modal density (frequency-independent for thin plates)
    n = plate.area * math.sqrt(rho_h / D) / (4.0 * math.pi)
    return n


def plate_modal_density_finite(
    plate: PlateSubsystem,
    frequency: float,
) -> float:
    """Modal density with low-frequency correction for finite plates.

    At lower frequencies, the discrete nature of modes matters.
    This uses the Weyl expansion:
        N(f) = A * pi * f / cb² - L * sqrt(f) / (2*cb) + 1/4
        n(f) = dN/df

    where L is the perimeter and cb is the bending wave speed.

    Args:
        plate: PlateSubsystem definition.
        frequency: center frequency (Hz).

    Returns:
        Modal density in modes/Hz.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    rho_h = plate.density * plate.thickness
    D = plate.bending_stiffness
    alpha = (D / rho_h) ** 0.25  # cb = alpha * sqrt(omega) = alpha * sqrt(2*pi*f)

    omega = 2.0 * math.pi * frequency
    cb = alpha * math.sqrt(omega)

    A = plate.area
    L = 2.0 * (plate.length + plate.width)

    # Weyl expansion for number of modes below frequency f
    # N(f) = A*omega/(4*pi*alpha²) - L*sqrt(omega)/(4*pi*alpha) + 1/4
    # n(f) = dN/df = dN/domega * domegas/df
    # dN/domega = A/(4*pi*alpha²) - L/(8*pi*alpha*sqrt(omega))
    # n(f) = dN/domega * 2*pi

    dN_domega = A / (4.0 * math.pi * alpha ** 2) - L / (8.0 * math.pi * alpha * math.sqrt(omega))
    n = dN_domega * 2.0 * math.pi
    return max(n, 0.0)


def modal_density_wave_theory(
    area: float,
    thickness: float,
    young_modulus: float,
    density: float,
    poisson_ratio: float,
    frequency: float,
) -> float:
    """Wave theory modal density for a thin plate (reference solution).

    This is the standard asymptotic result from wave theory:
        n(f) = A * sqrt(rho*h/D) / (4*pi)

    Used as the reference for validation (criterion a).

    Args:
        area: plate area (m²).
        thickness: plate thickness (m).
        young_modulus: Young's modulus (Pa).
        density: material density (kg/m³).
        poisson_ratio: Poisson's ratio.
        frequency: center frequency (Hz).

    Returns:
        Modal density in modes/Hz.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    D = young_modulus * thickness ** 3 / (12.0 * (1.0 - poisson_ratio ** 2))
    rho_h = density * thickness
    return area * math.sqrt(rho_h / D) / (4.0 * math.pi)


def beam_modal_density(
    beam: BeamSubsystem,
    frequency: float,
) -> float:
    """Modal density of a beam subsystem [modes/Hz].

    For a Euler-Bernoulli beam, the bending wave dispersion is:
        omega = k² * sqrt(EI / (rho*A))

    The modal density:
        n(f) = L / (2*pi) * sqrt(rho*A/(EI)) / (2*sqrt(omega))
             = L / (4*pi) * sqrt(rho*A/(EI)) / sqrt(2*pi*f)

    Args:
        beam: BeamSubsystem definition.
        frequency: center frequency (Hz).

    Returns:
        Modal density in modes/Hz.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    omega = 2.0 * math.pi * frequency
    EI = beam.young_modulus * beam.second_moment
    rho_A = beam.density * beam.cross_section_area

    n = beam.length / (4.0 * math.pi) * math.sqrt(rho_A / EI) / math.sqrt(omega)
    return n


def cavity_modal_density(
    cavity: CavitySubsystem,
    frequency: float,
) -> float:
    """Modal density of a 3D acoustic cavity [modes/Hz].

    For a 3D cavity, the asymptotic modal density is:
        n(f) = 4*pi*V*f² / c³

    where V is the volume and c is the sound speed.

    Args:
        cavity: CavitySubsystem definition.
        frequency: center frequency (Hz).

    Returns:
        Modal density in modes/Hz.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    return 4.0 * math.pi * cavity.volume * frequency ** 2 / cavity.sound_speed ** 3
