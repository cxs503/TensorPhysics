"""Kelvin wake wave system geometry.

This module implements the classical Kelvin wake pattern for ships moving
at constant speed in deep water. The Kelvin wake consists of:

1. **Transverse waves**: Wave crests perpendicular to ship track, with
   half-angle θ_t = arcsin(1/3) ≈ 19.47° from the track.

2. **Divergent waves**: Wave crests at angle to ship track, bounded by
   the cusp line at θ_d = arccos(1/3) ≈ 70.53° from track, or equivalently
   ±35.26° from the transverse direction.

The total wake half-angle is 19.47° + 19.47° = 38.94° (transverse) but
the visible wake envelope extends to ±35.26° from the transverse direction
(70.53° total opening angle).

References:
    - Kelvin, Lord (1887). "On ship waves". Proc. Inst. Mech. Engrs.
    - Lighthill, J. (1978). "Waves in Fluids". Cambridge University Press.
    - Rabaud, M. & Moisy, F. (2013). "Ship wakes: Kelvin or Mach angle?".
      Phys. Rev. Lett. 110, 214503.
"""

from __future__ import annotations

import math
from typing import Tuple


class KelvinWaveError(Exception):
    """Exception raised for Kelvin wave geometry errors."""
    pass


def transverse_wave_angle() -> float:
    """Return the transverse wave half-angle in degrees.
    
    The transverse waves are perpendicular to the ship track and propagate
    at the group velocity c_g = c/2 (where c is phase speed). The angle
    is determined by the condition that the wave pattern is stationary in
    the ship frame.
    
    Returns:
        float: Transverse wave half-angle in degrees (≈19.47°).
    
    Example:
        >>> theta_t = transverse_wave_angle()
        >>> abs(theta_t - 19.47) < 0.01
        True
    """
    # sin(θ_t) = 1/3 for deep water Kelvin wake
    theta_rad = math.asin(1.0 / 3.0)
    return math.degrees(theta_rad)


def divergent_wave_angle() -> float:
    """Return the divergent wave cusp angle in degrees.
    
    The divergent waves form the outer envelope of the Kelvin wake. The
    cusp line (where wave amplitude is maximum) occurs at angle θ_d from
    the ship track, where cos(θ_d) = 1/3.
    
    Returns:
        float: Divergent wave cusp angle from ship track in degrees (≈70.53°).
    
    Example:
        >>> theta_d = divergent_wave_angle()
        >>> abs(theta_d - 70.53) < 0.01
        True
    """
    # cos(θ_d) = 1/3 for deep water Kelvin wake
    theta_rad = math.acos(1.0 / 3.0)
    return math.degrees(theta_rad)


def wake_half_angle() -> float:
    """Return the total Kelvin wake half-angle in degrees.
    
    The visible wake extends from the transverse direction (90° from track)
    to the divergent cusp line. The half-angle from the ship track is the
    divergent wave angle.
    
    Returns:
        float: Wake half-angle from ship track in degrees (≈70.53°).
    """
    return divergent_wave_angle()


def wake_full_angle() -> float:
    """Return the total Kelvin wake opening angle in degrees.
    
    The full opening angle is twice the half-angle, representing the
    total angular width of the visible wake pattern.
    
    Returns:
        float: Wake full opening angle in degrees (≈141.06°).
    """
    return 2.0 * divergent_wave_angle()


def kelvin_wake_coordinates(
    ship_speed: float,
    distance_ahead: float = 0.0,
    n_points: int = 100
) -> Tuple[list, list]:
    """Compute Kelvin wake cusp line coordinates.
    
    The cusp line in the ship frame (origin at ship, x-axis along track,
    positive x = ahead of ship) is parameterized by the angle θ from the
    ship track direction.
    
    For the divergent wave cusp line:
        x(θ) = -r(θ) * cos(θ)
        y(θ) = r(θ) * sin(θ)
    
    where r(θ) is the distance from the ship to the wave crest at angle θ.
    
    The cusp line occurs at the angle where the wave amplitude is maximum,
    which is at θ_cusp = arccos(1/3) ≈ 70.53° from the ship track.
    
    Args:
        ship_speed: Ship speed in m/s.
        distance_ahead: Distance ahead of ship (negative = behind). Default 0.
        n_points: Number of points along cusp line. Default 100.
    
    Returns:
        Tuple of (x_coords, y_coords) lists for the cusp line.
    
    Raises:
        KelvinWaveError: If ship_speed <= 0.
    
    Example:
        >>> x, y = kelvin_wake_coordinates(5.0, n_points=10)
        >>> len(x) == 10
        True
    """
    if ship_speed <= 0:
        raise KelvinWaveError(f"Ship speed must be positive, got {ship_speed}")
    
    g = 9.81  # gravitational acceleration [m/s²]
    U = ship_speed
    
    # Characteristic length scale
    L = U**2 / g
    
    x_coords = []
    y_coords = []
    
    # Generate points along the wake
    # The wake extends from θ = 0 (transverse, directly behind) to θ = θ_cusp
    # We parameterize by angle from the transverse direction (0 to θ_max)
    theta_max = math.pi / 2 - math.acos(1.0 / 3.0)  # ≈ 19.47° from transverse
    
    for i in range(n_points):
        if n_points > 1:
            # Angle from transverse direction (0 to θ_max)
            alpha = (i / (n_points - 1)) * theta_max
        else:
            alpha = 0.0
        
        # Angle from ship track
        theta = math.pi / 2 - alpha
        
        # Distance from ship to wave crest along the cusp line
        # For the Kelvin wake cusp: r = L / (3 * sin²(θ))
        # At the cusp angle θ_cusp = arccos(1/3), sin(θ_cusp) = sqrt(2/3)
        # So r_cusp = L / (3 * 2/3) = L / 2
        
        # More generally, for the cusp line:
        # r(θ) = L / (3 * sin²(θ)) for θ > 0
        # At θ = π/2 (transverse), r = L
        
        if abs(math.sin(theta)) < 1e-10:
            # Avoid division by zero
            r = L
        else:
            r = L / (3.0 * math.sin(theta)**2)
        
        # Convert to Cartesian (ship at origin, x along track, positive = ahead)
        # Wake is behind the ship, so x should be negative
        x = -r * math.cos(theta) + distance_ahead
        y = r * math.sin(theta)
        
        x_coords.append(x)
        y_coords.append(y)
    
    return x_coords, y_coords


def wave_wavelength(ship_speed: float, wave_angle: float) -> float:
    """Compute wavelength of a wave component at given angle.
    
    For a stationary wave pattern in the ship frame, the wavelength depends
    on the angle from the ship track:
        λ(θ) = (2π U² / g) * cos²(θ)
    
    Args:
        ship_speed: Ship speed in m/s.
        wave_angle: Angle from ship track in degrees.
    
    Returns:
        float: Wavelength in meters.
    
    Raises:
        KelvinWaveError: If ship_speed <= 0.
    
    Example:
        >>> lam = wave_wavelength(5.0, 0.0)  # transverse wave
        >>> lam > 0
        True
    """
    if ship_speed <= 0:
        raise KelvinWaveError(f"Ship speed must be positive, got {ship_speed}")
    
    g = 9.81  # m/s²
    U = ship_speed
    theta_rad = math.radians(wave_angle)
    
    # λ = (2π U² / g) * cos²(θ)
    wavelength = (2.0 * math.pi * U**2 / g) * math.cos(theta_rad)**2
    
    return wavelength


def wave_period(ship_speed: float, wave_angle: float) -> float:
    """Compute wave period of a wave component at given angle.
    
    The wave period is related to wavelength by the deep water dispersion
    relation: T = sqrt(2π λ / g).
    
    Args:
        ship_speed: Ship speed in m/s.
        wave_angle: Angle from ship track in degrees.
    
    Returns:
        float: Wave period in seconds.
    
    Example:
        >>> T = wave_period(5.0, 0.0)
        >>> T > 0
        True
    """
    g = 9.81  # m/s²
    wavelength = wave_wavelength(ship_speed, wave_angle)
    
    # Deep water dispersion: T = sqrt(2π λ / g)
    period = math.sqrt(2.0 * math.pi * wavelength / g)
    
    return period


def froude_number(ship_speed: float, ship_length: float) -> float:
    """Compute Froude number based on ship length.
    
    The Froude number Fr = U / sqrt(g L) characterizes the wave-making
    resistance and wake pattern.
    
    Args:
        ship_speed: Ship speed in m/s.
        ship_length: Ship length (typically L_pp or L_wl) in meters.
    
    Returns:
        float: Froude number (dimensionless).
    
    Raises:
        KelvinWaveError: If ship_length <= 0.
    
    Example:
        >>> Fr = froude_number(5.0, 100.0)
        >>> 0 < Fr < 1
        True
    """
    if ship_length <= 0:
        raise KelvinWaveError(f"Ship length must be positive, got {ship_length}")
    
    g = 9.81  # m/s²
    Fr = ship_speed / math.sqrt(g * ship_length)
    
    return Fr


__all__ = [
    "KelvinWaveError",
    "transverse_wave_angle",
    "divergent_wave_angle",
    "wake_half_angle",
    "wake_full_angle",
    "kelvin_wake_coordinates",
    "wave_wavelength",
    "wave_period",
    "froude_number",
]
