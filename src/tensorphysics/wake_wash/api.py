"""Wake wash propagation API.

This module provides a unified API for wake wash analysis, combining:
- Kelvin wave geometry
- Wave height decay
- Shoreline impact assessment
- PIANC compliance checking

The API is designed for integration with the ship-performance-platform
and can be used for:
- Route planning (avoiding sensitive receptors)
- Speed optimization (minimizing wake wash impact)
- Environmental impact assessment
- Regulatory compliance verification
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

from .kelvin_wave import (
    transverse_wave_angle,
    divergent_wave_angle,
    wake_half_angle,
    wake_full_angle,
    kelvin_wake_coordinates,
    wave_wavelength,
    wave_period,
    froude_number,
)
from .wave_height_decay import (
    deep_water_decay,
    shallow_water_decay,
    wave_height_with_depth,
    decay_profile,
)
from .shoreline_impact import (
    shoreline_wave_height,
    wake_wash_impact_assessment,
    shoaling_coefficient,
    runup_estimation,
)
from .panc_compliance import (
    ReceptorType,
    ComplianceLevel,
    check_compliance,
    compliance_report,
    get_wave_height_limit,
    speed_limit_for_compliance,
)


class WakeWashError(Exception):
    """Exception raised for wake wash analysis errors."""
    pass


def analyze_wake_pattern(
    ship_speed: float,
    ship_length: float,
    n_points: int = 100
) -> dict:
    """Analyze the Kelvin wake pattern for a ship.
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_length: Ship length [m].
        n_points: Number of points for wake coordinates.
    
    Returns:
        dict: Wake pattern analysis with keys:
            - transverse_angle [deg]
            - divergent_angle [deg]
            - wake_full_angle [deg]
            - froude_number [-]
            - characteristic_wavelength [m]
            - characteristic_period [s]
            - wake_coordinates: (x, y) lists
    
    Example:
        >>> result = analyze_wake_pattern(10.0, 100.0)
        >>> 'transverse_angle' in result
        True
    """
    Fr = froude_number(ship_speed, ship_length)
    lambda_char = wave_wavelength(ship_speed, 0.0)  # transverse wave
    T_char = wave_period(ship_speed, 0.0)
    
    x_coords, y_coords = kelvin_wake_coordinates(ship_speed, n_points=n_points)
    
    return {
        'transverse_angle': transverse_wave_angle(),
        'divergent_angle': divergent_wave_angle(),
        'wake_full_angle': wake_full_angle(),
        'froude_number': Fr,
        'characteristic_wavelength': lambda_char,
        'characteristic_period': T_char,
        'wake_coordinates': (x_coords, y_coords),
    }


def predict_wave_height(
    ship_speed: float,
    ship_draft: float,
    distance: float,
    water_depth: Optional[float] = None
) -> float:
    """Predict wave height at a given distance from ship.
    
    Uses empirical relationship: H_ref ≈ 0.1 * draft * Fr² at reference
    distance r_ref = 10 * draft, then applies decay law.
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_draft: Ship draft [m].
        distance: Distance from ship [m].
        water_depth: Water depth [m]. If None, assumes deep water.
    
    Returns:
        float: Predicted wave height [m].
    
    Example:
        >>> H = predict_wave_height(10.0, 5.0, 500.0)
        >>> H > 0
        True
    """
    g = 9.81  # m/s²
    
    # Reference wave height near ship
    Fr = ship_speed / math.sqrt(g * ship_draft)
    H_ref = 0.1 * ship_draft * Fr**2
    r_ref = 10.0 * ship_draft
    
    # Characteristic wavelength
    lambda_char = 2.0 * math.pi * ship_speed**2 / g
    
    if water_depth is not None:
        H = wave_height_with_depth(H_ref, r_ref, distance, water_depth, lambda_char)
    else:
        H = deep_water_decay(H_ref, r_ref, distance)
    
    return H


def assess_shoreline_impact(
    ship_speed: float,
    ship_draft: float,
    ship_distance: float,
    water_depth: float,
    beach_slope: float = 0.02
) -> dict:
    """Assess wake wash impact at shoreline.
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_draft: Ship draft [m].
        ship_distance: Distance from ship track to shoreline [m].
        water_depth: Water depth [m].
        beach_slope: Beach slope tan(β). Default 0.02.
    
    Returns:
        dict: Impact assessment with keys:
            - wave_height [m]
            - wave_period [s]
            - is_breaking [bool]
            - runup [m]
            - impact_level [str]
    
    Example:
        >>> result = assess_shoreline_impact(10.0, 5.0, 500.0, 10.0)
        >>> 'wave_height' in result
        True
    """
    return wake_wash_impact_assessment(
        ship_speed, ship_distance, ship_draft, water_depth,
        ship_distance, beach_slope
    )


def check_pianc_compliance(
    ship_speed: float,
    ship_draft: float,
    receptor_distance: float,
    receptor_type: ReceptorType,
    water_depth: Optional[float] = None
) -> dict:
    """Check PIANC compliance for a receptor.
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_draft: Ship draft [m].
        receptor_distance: Distance from ship to receptor [m].
        receptor_type: Type of receptor.
        water_depth: Water depth [m]. If None, assumes deep water.
    
    Returns:
        dict: Compliance report with keys:
            - wave_height [m]
            - receptor [str]
            - limit [m]
            - utilization_ratio [-]
            - compliance_level [str]
            - risk_level [str]
            - recommendations [list]
    
    Example:
        >>> result = check_pianc_compliance(
        ...     10.0, 5.0, 500.0, ReceptorType.SWIMMERS
        ... )
        >>> 'compliance_level' in result
        True
    """
    H = predict_wave_height(ship_speed, ship_draft, receptor_distance, water_depth)
    
    return compliance_report(
        H, receptor_type,
        ship_speed=ship_speed,
        ship_distance=receptor_distance
    )


def optimize_speed_for_compliance(
    ship_draft: float,
    receptor_distance: float,
    receptor_type: ReceptorType,
    current_speed: float,
    water_depth: Optional[float] = None
) -> dict:
    """Optimize ship speed for PIANC compliance.
    
    Args:
        ship_draft: Ship draft [m].
        receptor_distance: Distance from ship to receptor [m].
        receptor_type: Type of receptor.
        current_speed: Current ship speed [m/s].
        water_depth: Water depth [m]. If None, assumes deep water.
    
    Returns:
        dict: Optimization results with keys:
            - current_speed [m/s]
            - current_wave_height [m]
            - current_compliance [str]
            - max_compliant_speed [m/s]
            - speed_reduction [m/s]
            - speed_reduction_pct [%]
    
    Example:
        >>> result = optimize_speed_for_compliance(
        ...     5.0, 500.0, ReceptorType.SWIMMERS, 15.0
        ... )
        >>> 'max_compliant_speed' in result
        True
    """
    g = 9.81  # m/s²
    
    # Current wave height
    H_current = predict_wave_height(current_speed, ship_draft, receptor_distance, water_depth)
    
    # Current compliance
    level_current, ratio_current = check_compliance(H_current, receptor_type)
    
    # Maximum compliant speed
    # H ∝ U², so U_max = U_current * sqrt(H_limit / H_current)
    limit = get_wave_height_limit(receptor_type)
    if H_current > 0:
        U_max = current_speed * math.sqrt(limit / H_current)
    else:
        U_max = current_speed * 2.0  # arbitrary large factor if H=0
    
    speed_reduction = current_speed - U_max
    speed_reduction_pct = (speed_reduction / current_speed) * 100.0 if current_speed > 0 else 0.0
    
    return {
        'current_speed': current_speed,
        'current_wave_height': H_current,
        'current_compliance': level_current.value,
        'max_compliant_speed': U_max,
        'speed_reduction': speed_reduction,
        'speed_reduction_pct': speed_reduction_pct,
    }


def multi_receptor_assessment(
    ship_speed: float,
    ship_draft: float,
    receptors: List[Tuple[float, ReceptorType]],
    water_depth: Optional[float] = None
) -> List[dict]:
    """Assess wake wash impact on multiple receptors.
    
    Args:
        ship_speed: Ship speed [m/s].
        ship_draft: Ship draft [m].
        receptors: List of (distance [m], receptor_type) tuples.
        water_depth: Water depth [m]. If None, assumes deep water.
    
    Returns:
        list: List of compliance reports for each receptor.
    
    Example:
        >>> receptors = [
        ...     (300.0, ReceptorType.SWIMMERS),
        ...     (500.0, ReceptorType.MOORED_LARGE_VESSEL),
        ... ]
        >>> results = multi_receptor_assessment(10.0, 5.0, receptors)
        >>> len(results) == 2
        True
    """
    results = []
    for distance, receptor_type in receptors:
        report = check_pianc_compliance(
            ship_speed, ship_draft, distance, receptor_type, water_depth
        )
        report['distance'] = distance
        results.append(report)
    
    return results


__all__ = [
    "WakeWashError",
    "analyze_wake_pattern",
    "predict_wave_height",
    "assess_shoreline_impact",
    "check_pianc_compliance",
    "optimize_speed_for_compliance",
    "multi_receptor_assessment",
]
