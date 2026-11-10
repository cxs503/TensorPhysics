"""PIANC (2003) environmental compliance assessment for wake wash.

This module implements the PIANC (Permanent International Association of
Navigation Congresses) guidelines for managing wake wash from high-speed
vessels. The guidelines provide limits on wave height to protect:

1. **Moored vessels**: H_max ≤ 0.3-0.5 m (depending on vessel type)
2. **Swimmers**: H_max ≤ 0.3 m
3. **Shoreline erosion**: H_max ≤ 0.3-0.5 m (depending on soil type)
4. **Marine structures**: H_max ≤ 0.5-1.0 m (depending on structure)

The module provides:
- Limit lookup by receptor type
- Compliance checking
- Risk classification
- Mitigation recommendations

References:
    - PIANC (2003). "Guidelines for Managing Wake Wash from High-Speed Vessels".
      Report of MarCom Working Group 38. Brussels, Belgium.
    - PIANC (2006). "Wake Wash from High-Speed Vessels: State of the Art".
    - PIANC (2011). "Harbour Approach Channels Design Guidelines".
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from enum import Enum


class ReceptorType(Enum):
    """Types of receptors affected by wake wash."""
    SWIMMERS = "swimmers"
    MOORED_SMALL_VESSEL = "moored_small_vessel"
    MOORED_LARGE_VESSEL = "moored_large_vessel"
    ERODIBLE_SHORELINE = "erodible_shoreline"
    STABLE_SHORELINE = "stable_shoreline"
    FLOATING_DOCK = "floating_dock"
    MARINE_STRUCTURE = "marine_structure"


class ComplianceLevel(Enum):
    """Compliance levels."""
    COMPLIANT = "compliant"
    MARGINAL = "marginal"
    NON_COMPLIANT = "non_compliant"


class PIANCComplianceError(Exception):
    """Exception raised for PIANC compliance errors."""
    pass


# PIANC (2003) wave height limits [m]
# These are maximum allowable significant wave heights
PIANC_LIMITS: Dict[ReceptorType, float] = {
    ReceptorType.SWIMMERS: 0.3,
    ReceptorType.MOORED_SMALL_VESSEL: 0.3,
    ReceptorType.MOORED_LARGE_VESSEL: 0.5,
    ReceptorType.ERODIBLE_SHORELINE: 0.3,
    ReceptorType.STABLE_SHORELINE: 0.5,
    ReceptorType.FLOATING_DOCK: 0.4,
    ReceptorType.MARINE_STRUCTURE: 1.0,
}

# Marginal compliance threshold (fraction of limit)
MARGINAL_THRESHOLD = 0.8


def get_wave_height_limit(receptor: ReceptorType) -> float:
    """Get the PIANC wave height limit for a receptor type.
    
    Args:
        receptor: Receptor type.
    
    Returns:
        float: Maximum allowable wave height [m].
    
    Example:
        >>> limit = get_wave_height_limit(ReceptorType.SWIMMERS)
        >>> abs(limit - 0.3) < 0.01
        True
    """
    return PIANC_LIMITS[receptor]


def check_compliance(
    wave_height: float,
    receptor: ReceptorType
) -> Tuple[ComplianceLevel, float]:
    """Check if wave height complies with PIANC limits.
    
    Args:
        wave_height: Actual wave height [m].
        receptor: Receptor type.
    
    Returns:
        Tuple of (compliance_level, utilization_ratio):
            - compliance_level: COMPLIANT, MARGINAL, or NON_COMPLIANT
            - utilization_ratio: H_actual / H_limit (1.0 = at limit)
    
    Raises:
        PIANCComplianceError: If wave_height is negative.
    
    Example:
        >>> level, ratio = check_compliance(0.25, ReceptorType.SWIMMERS)
        >>> level == ComplianceLevel.COMPLIANT
        True
        >>> abs(ratio - 0.25/0.3) < 0.01
        True
    """
    if wave_height < 0:
        raise PIANCComplianceError(f"Wave height must be non-negative, got {wave_height}")
    
    limit = get_wave_height_limit(receptor)
    ratio = wave_height / limit
    
    if ratio <= MARGINAL_THRESHOLD:
        level = ComplianceLevel.COMPLIANT
    elif ratio <= 1.0:
        level = ComplianceLevel.MARGINAL
    else:
        level = ComplianceLevel.NON_COMPLIANT
    
    return level, ratio


def assess_multiple_receptors(
    wave_height: float,
    receptors: list
) -> Dict[ReceptorType, Tuple[ComplianceLevel, float]]:
    """Assess compliance for multiple receptor types.
    
    Args:
        wave_height: Actual wave height [m].
        receptors: List of receptor types to assess.
    
    Returns:
        dict: Mapping from receptor to (compliance_level, utilization_ratio).
    
    Example:
        >>> receptors = [ReceptorType.SWIMMERS, ReceptorType.MOORED_LARGE_VESSEL]
        >>> results = assess_multiple_receptors(0.4, receptors)
        >>> len(results) == 2
        True
    """
    results = {}
    for receptor in receptors:
        level, ratio = check_compliance(wave_height, receptor)
        results[receptor] = (level, ratio)
    return results


def risk_classification(
    wave_height: float,
    receptor: ReceptorType
) -> str:
    """Classify risk level based on PIANC guidelines.
    
    Risk levels:
    - LOW: H < 0.5 * H_limit
    - MODERATE: 0.5 * H_limit ≤ H < H_limit
    - HIGH: H ≥ H_limit
    
    Args:
        wave_height: Actual wave height [m].
        receptor: Receptor type.
    
    Returns:
        str: Risk level ('low', 'moderate', 'high').
    
    Example:
        >>> risk = risk_classification(0.2, ReceptorType.SWIMMERS)
        >>> risk == 'low'
        True
    """
    limit = get_wave_height_limit(receptor)
    ratio = wave_height / limit
    
    if ratio < 0.5:
        return 'low'
    elif ratio < 1.0:
        return 'moderate'
    else:
        return 'high'


def mitigation_recommendations(
    wave_height: float,
    receptor: ReceptorType,
    ship_speed: Optional[float] = None,
    ship_distance: Optional[float] = None
) -> list:
    """Generate mitigation recommendations based on compliance assessment.
    
    Args:
        wave_height: Actual wave height [m].
        receptor: Receptor type.
        ship_speed: Ship speed [m/s] (optional, for speed reduction advice).
        ship_distance: Distance from receptor [m] (optional, for setback advice).
    
    Returns:
        list: List of recommendation strings.
    
    Example:
        >>> recs = mitigation_recommendations(0.5, ReceptorType.SWIMMERS)
        >>> len(recs) > 0
        True
    """
    level, ratio = check_compliance(wave_height, receptor)
    recommendations = []
    
    if level == ComplianceLevel.NON_COMPLIANT:
        recommendations.append(
            f"Wave height ({wave_height:.2f} m) exceeds PIANC limit "
            f"({get_wave_height_limit(receptor):.2f} m) for {receptor.value}."
        )
        
        if ship_speed is not None:
            # Recommend speed reduction
            # H ∝ U² approximately, so U_new = U * sqrt(H_limit / H_actual)
            limit = get_wave_height_limit(receptor)
            U_new = ship_speed * (limit / wave_height)**0.5
            recommendations.append(
                f"Reduce ship speed from {ship_speed:.1f} m/s to "
                f"{U_new:.1f} m/s to comply with limits."
            )
        
        if ship_distance is not None:
            # Recommend increased setback
            # H ∝ 1/√r, so r_new = r * (H_actual / H_limit)²
            limit = get_wave_height_limit(receptor)
            r_new = ship_distance * (wave_height / limit)**2
            recommendations.append(
                f"Increase ship-receptor distance from {ship_distance:.0f} m to "
                f"{r_new:.0f} m to comply with limits."
            )
        
        recommendations.append(
            "Consider alternative routing to avoid sensitive receptors."
        )
    
    elif level == ComplianceLevel.MARGINAL:
        recommendations.append(
            f"Wave height ({wave_height:.2f} m) is close to PIANC limit "
            f"({get_wave_height_limit(receptor):.2f} m) for {receptor.value}. "
            "Monitor conditions and consider precautionary measures."
        )
        
        if ship_speed is not None:
            recommendations.append(
                "Consider modest speed reduction (10-20%) to provide safety margin."
            )
    
    else:
        recommendations.append(
            f"Wave height ({wave_height:.2f} m) is within PIANC limits for "
            f"{receptor.value}. No mitigation required."
        )
    
    return recommendations


def compliance_report(
    wave_height: float,
    receptor: ReceptorType,
    ship_speed: Optional[float] = None,
    ship_distance: Optional[float] = None
) -> dict:
    """Generate comprehensive compliance report.
    
    Args:
        wave_height: Actual wave height [m].
        receptor: Receptor type.
        ship_speed: Ship speed [m/s] (optional).
        ship_distance: Distance from receptor [m] (optional).
    
    Returns:
        dict: Report with keys:
            - wave_height [m]
            - receptor [str]
            - limit [m]
            - utilization_ratio [-]
            - compliance_level [str]
            - risk_level [str]
            - recommendations [list]
    
    Example:
        >>> report = compliance_report(0.25, ReceptorType.SWIMMERS)
        >>> 'compliance_level' in report
        True
    """
    level, ratio = check_compliance(wave_height, receptor)
    risk = risk_classification(wave_height, receptor)
    recs = mitigation_recommendations(
        wave_height, receptor, ship_speed, ship_distance
    )
    
    return {
        'wave_height': wave_height,
        'receptor': receptor.value,
        'limit': get_wave_height_limit(receptor),
        'utilization_ratio': ratio,
        'compliance_level': level.value,
        'risk_level': risk,
        'recommendations': recs,
    }


def speed_limit_for_compliance(
    receptor: ReceptorType,
    reference_wave_height: float,
    reference_speed: float,
    reference_distance: float,
    target_distance: float
) -> float:
    """Calculate maximum ship speed for PIANC compliance.
    
    Assuming H ∝ U² and H ∝ 1/√r, calculate the maximum speed such that
    the wave height at the receptor does not exceed the PIANC limit.
    
    Args:
        receptor: Receptor type.
        reference_wave_height: Wave height at reference conditions [m].
        reference_speed: Reference ship speed [m/s].
        reference_distance: Reference distance from ship [m].
        target_distance: Distance to receptor [m].
    
    Returns:
        float: Maximum compliant ship speed [m/s].
    
    Example:
        >>> U_max = speed_limit_for_compliance(
        ...     ReceptorType.SWIMMERS, 0.5, 10.0, 100.0, 500.0
        ... )
        >>> U_max > 0
        True
    """
    limit = get_wave_height_limit(receptor)
    
    # H ∝ U² / √r
    # H_target = H_ref * (U_target/U_ref)² * √(r_ref/r_target)
    # Set H_target = limit and solve for U_target
    # U_target = U_ref * sqrt(limit / (H_ref * sqrt(r_ref/r_target)))
    
    distance_factor = (reference_distance / target_distance)**0.5
    U_max = reference_speed * (limit / (reference_wave_height * distance_factor))**0.5
    
    return U_max


__all__ = [
    "ReceptorType",
    "ComplianceLevel",
    "PIANCComplianceError",
    "PIANC_LIMITS",
    "MARGINAL_THRESHOLD",
    "get_wave_height_limit",
    "check_compliance",
    "assess_multiple_receptors",
    "risk_classification",
    "mitigation_recommendations",
    "compliance_report",
    "speed_limit_for_compliance",
]
