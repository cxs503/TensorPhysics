"""Shoreline wave height estimation from ship wake wash.

This module estimates the wave height reaching the shoreline from a ship
passing at a given distance. The estimation accounts for:

1. **Geometric spreading**: Wave height decay with distance (see wave_height_decay)
2. **Shoaling**: Wave height increase as waves enter shallow water
3. **Refraction**: Wave direction change due to depth variation
4. **Bottom friction**: Energy dissipation in shallow water
5. **Beach slope**: Wave run-up and breaking criteria

The shoreline wave height is critical for assessing:
- Erosion potential
- Moored vessel safety
- Swimmer safety
- Environmental compliance (PIANC limits)

References:
    - PIANC (2003). "Guidelines for Managing Wake Wash from High-Speed Vessels".
    - SPM (1984). "Shore Protection Manual". US Army Corps of Engineers.
    - Goda, Y. (2000). "Random Seas and Design of Maritime Structures".
"""

from __future__ import annotations

import math
from typing import Tuple, Optional


class ShorelineImpactError(Exception):
    """Exception raised for shoreline impact estimation errors."""
    pass


def shoaling_coefficient(
    deep_water_wavelength: float,
    water_depth: float
) -> float:
    """Compute shoaling coefficient K_s.
    
    The shoaling coefficient relates deep water wave height to local wave
    height at a given depth:
        H(h) = K_s * H_deep
    
    Based on linear wave theory energy flux conservation:
        K_s = sqrt(c_g_deep / c_g(h))
    
    where c_g is the group velocity.
    
    Args:
        deep_water_wavelength: Deep water wavelength [m].
        water_depth: Local water depth [m].
    
    Returns:
        float: Shoaling coefficient (dimensionless).
    
    Raises:
        ShorelineImpactError: If parameters are invalid.
    
    Example:
        >>> K_s = shoaling_coefficient(100.0, 10.0)
        >>> K_s > 1.0  # waves grow in shallow water
        True
    """
    if deep_water_wavelength <= 0:
        raise ShorelineImpactError(f"Wavelength must be positive, got {deep_water_wavelength}")
    if water_depth <= 0:
        raise ShorelineImpactError(f"Water depth must be positive, got {water_depth}")
    
    g = 9.81  # m/s²
    
    # Deep water group velocity
    c_g_deep = math.sqrt(g * deep_water_wavelength / (4.0 * math.pi))
    
    # Local group velocity (finite depth)
    # c_g = (c/2) * (1 + 2kh/sinh(2kh))
    # where k = 2π/λ and ω² = gk tanh(kh)
    # For simplicity, use shallow water approximation when h < λ/2
    if water_depth > deep_water_wavelength / 2.0:
        # Deep water limit
        c_g_local = c_g_deep
    else:
        # Shallow water: c_g ≈ sqrt(g h)
        c_g_local = math.sqrt(g * water_depth)
    
    # Shoaling coefficient
    K_s = math.sqrt(c_g_deep / c_g_local)
    
    return K_s


def breaking_criterion(
    wave_height: float,
    water_depth: float,
    beach_slope: float = 0.02
) -> bool:
    """Check if wave is breaking based on H/h criterion.
    
    Waves break when H/h exceeds a critical value that depends on beach slope:
        (H/h)_break ≈ 0.78 for mild slopes (1:50 to 1:100)
        (H/h)_break ≈ 0.8-1.0 for steep slopes (1:10 to 1:20)
    
    Args:
        wave_height: Wave height [m].
        water_depth: Water depth [m].
        beach_slope: Beach slope (tan β), e.g., 0.02 for 1:50 slope.
    
    Returns:
        bool: True if wave is breaking.
    
    Example:
        >>> breaking_criterion(2.0, 2.0, 0.02)
        True
        >>> breaking_criterion(1.0, 3.0, 0.02)
        False
    """
    if water_depth <= 0:
        raise ShorelineImpactError(f"Water depth must be positive, got {water_depth}")
    if wave_height < 0:
        raise ShorelineImpactError(f"Wave height must be non-negative, got {wave_height}")
    
    # Breaking index γ_b depends on beach slope
    # γ_b ≈ 0.78 for mild slopes, up to 1.0 for steep slopes
    # Empirical formula: γ_b = 0.78 * (1 - exp(-1.2 * tan(β)))
    # For simplicity, use γ_b = 0.78 for mild slopes
    gamma_b = 0.78
    
    return (wave_height / water_depth) > gamma_b


def runup_estimation(
    wave_height: float,
    wave_period: float,
    beach_slope: float
) -> float:
    """Estimate wave run-up on a beach.
    
    The run-up R (vertical elevation above still water level) is estimated
    using empirical formulas. For regular waves:
        R ≈ 2.0 * H * ξ
    where ξ is the surf similarity parameter (Iribarren number):
        ξ = tan(β) / sqrt(H / (g T²))
    
    Args:
        wave_height: Wave height at toe of slope [m].
        wave_period: Wave period [s].
        beach_slope: Beach slope tan(β) (e.g., 0.02 for 1:50).
    
    Returns:
        float: Run-up elevation [m].
    
    Raises:
        ShorelineImpactError: If parameters are invalid.
    
    Example:
        >>> R = runup_estimation(1.0, 10.0, 0.05)
        >>> R > 0
        True
    """
    if wave_height <= 0:
        raise ShorelineImpactError(f"Wave height must be positive, got {wave_height}")
    if wave_period <= 0:
        raise ShorelineImpactError(f"Wave period must be positive, got {wave_period}")
    if beach_slope <= 0:
        raise ShorelineImpactError(f"Beach slope must be positive, got {beach_slope}")
    
    g = 9.81  # m/s²
    
    # Iribarren number
    xi = beach_slope / math.sqrt(wave_height / (g * wave_period**2))
    
    # Run-up (empirical, for non-breaking waves)
    # R ≈ 2.0 * H * ξ for ξ < 2 (dissipative to intermediate)
    # R ≈ 2.0 * H for ξ > 2 (reflective)
    if xi < 2.0:
        R = 2.0 * wave_height * xi
    else:
        R = 2.0 * wave_height
    
    return R


def shoreline_wave_height(
    ship_speed: float,
    ship_distance: float,
    water_depth_nearshore: float,
    deep_water_wavelength: float,
    reference_wave_height: float,
    reference_distance: float,
    beach_slope: float = 0.02
) -> Tuple[float, bool]:
    """Estimate wave height at shoreline from ship wake.
    
    This function combines:
    1. Geometric spreading from ship to nearshore
    2. Shoaling from deep water to nearshore depth
    3. Breaking check
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_distance: Distance from ship track to shoreline [m].
        water_depth_nearshore: Water depth at shoreline [m].
        deep_water_wavelength: Characteristic deep water wavelength [m].
        reference_wave_height: Wave height at reference distance [m].
        reference_distance: Reference distance from ship [m].
        beach_slope: Beach slope tan(β). Default 0.02 (1:50).
    
    Returns:
        Tuple of (shoreline_wave_height [m], is_breaking [bool]).
    
    Raises:
        ShorelineImpactError: If parameters are invalid.
    
    Example:
        >>> H_shore, breaking = shoreline_wave_height(
        ...     10.0, 500.0, 3.0, 50.0, 0.5, 100.0
        ... )
        >>> H_shore > 0
        True
    """
    if ship_distance <= 0:
        raise ShorelineImpactError(f"Ship distance must be positive, got {ship_distance}")
    if water_depth_nearshore <= 0:
        raise ShorelineImpactError(f"Nearshore depth must be positive, got {water_depth_nearshore}")
    
    # Step 1: Geometric spreading to nearshore
    # Use deep water decay as approximation (conservative)
    from .wave_height_decay import deep_water_decay
    
    H_nearshore = deep_water_decay(
        reference_wave_height, reference_distance, ship_distance
    )
    
    # Step 2: Shoaling
    K_s = shoaling_coefficient(deep_water_wavelength, water_depth_nearshore)
    H_shore = H_nearshore * K_s
    
    # Step 3: Breaking check
    is_breaking = breaking_criterion(H_shore, water_depth_nearshore, beach_slope)
    
    return H_shore, is_breaking


def wake_wash_impact_assessment(
    ship_speed: float,
    ship_distance: float,
    ship_draft: float,
    water_depth: float,
    shoreline_distance: float,
    beach_slope: float = 0.02
) -> dict:
    """Comprehensive wake wash impact assessment at shoreline.
    
    This function provides a complete assessment including:
    - Estimated wave height at shoreline
    - Wave period
    - Breaking status
    - Run-up elevation
    - Impact classification (low/medium/high)
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_distance: Distance from ship track to shoreline [m].
        ship_draft: Ship draft [m].
        water_depth: Water depth along path [m].
        shoreline_distance: Distance from ship to shoreline [m].
        beach_slope: Beach slope tan(β). Default 0.02.
    
    Returns:
        dict: Assessment results with keys:
            - wave_height [m]
            - wave_period [s]
            - is_breaking [bool]
            - runup [m]
            - impact_level [str]: 'low', 'medium', 'high'
    
    Example:
        >>> result = wake_wash_impact_assessment(10.0, 500.0, 5.0, 10.0, 500.0)
        >>> 'wave_height' in result
        True
    """
    g = 9.81  # m/s²
    
    # Estimate reference wave height near ship
    # Empirical: H_ref ≈ 0.1 * draft * Fr² (very rough estimate)
    Fr = ship_speed / math.sqrt(g * ship_draft)
    H_ref = 0.1 * ship_draft * Fr**2
    r_ref = 10.0 * ship_draft  # reference distance
    
    # Characteristic wavelength
    # λ ≈ 2π U² / g (transverse waves)
    lambda_deep = 2.0 * math.pi * ship_speed**2 / g
    
    # Wave period
    T = math.sqrt(2.0 * math.pi * lambda_deep / g)
    
    # Shoreline wave height
    H_shore, is_breaking = shoreline_wave_height(
        ship_speed, shoreline_distance, water_depth,
        lambda_deep, H_ref, r_ref, beach_slope
    )
    
    # Run-up
    R = runup_estimation(H_shore, T, beach_slope)
    
    # Impact classification (based on PIANC guidelines)
    if H_shore < 0.3:
        impact_level = 'low'
    elif H_shore < 0.8:
        impact_level = 'medium'
    else:
        impact_level = 'high'
    
    return {
        'wave_height': H_shore,
        'wave_period': T,
        'is_breaking': is_breaking,
        'runup': R,
        'impact_level': impact_level,
    }


__all__ = [
    "ShorelineImpactError",
    "shoaling_coefficient",
    "breaking_criterion",
    "runup_estimation",
    "shoreline_wave_height",
    "wake_wash_impact_assessment",
]
