"""Hot-spot stress calculation via stress concentration factor (Kt).

Implements nominal stress to hot-spot stress conversion using
stress concentration factors from DNV guidelines and shell-beam coupling.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HotSpotResult:
    """Result of hot-spot stress calculation."""
    nominal_stress: float    # MPa
    kt_factor: float         # stress concentration factor
    hot_spot_stress: float   # MPa
    weld_type: str           # weld detail category
    thickness_effect: float  # thickness correction factor


def hotspot_stress(
    nominal_stress: float,
    kt_factor: float = 1.0,
    thickness: float = None,
    reference_thickness: float = 22.0,
    thickness_exponent: float = 0.25,
    weld_type: str = "fillet",
) -> HotSpotResult:
    """Calculate hot-spot stress from nominal stress.

    Parameters
    ----------
    nominal_stress : float
        Nominal stress range (MPa).
    kt_factor : float
        Stress concentration factor (geometric + weld toe).
    thickness : float, optional
        Plate thickness (mm). If None, no thickness correction.
    reference_thickness : float
        Reference thickness for DNV thickness effect (mm, default 22).
    thickness_exponent : float
        Thickness effect exponent (DNV: 0.25 for t > 22mm).
    weld_type : str
        Weld detail type: 'fillet', 'butt', 'cruciform'.

    Returns
    -------
    HotSpotResult
    """
    # Thickness correction per DNV-RP-C203
    if thickness is not None and thickness > reference_thickness:
        thickness_effect = (reference_thickness / thickness) ** thickness_exponent
    else:
        thickness_effect = 1.0

    hot_spot = nominal_stress * kt_factor * thickness_effect

    return HotSpotResult(
        nominal_stress=nominal_stress,
        kt_factor=kt_factor,
        hot_spot_stress=hot_spot,
        weld_type=weld_type,
        thickness_effect=thickness_effect,
    )


def kt_from_detail(detail_type: str) -> float:
    """Get typical Kt factor for common weld details.

    Parameters
    ----------
    detail_type : str
        Weld detail: 'fillet', 'butt', 'cruciform', 'attachment'.

    Returns
    -------
    float
        Typical stress concentration factor.
    """
    kt_values = {
        'fillet': 1.2,
        'butt': 1.0,
        'cruciform': 1.5,
        'attachment': 1.8,
        'stiffener_end': 1.4,
        'bracket_toe': 1.6,
    }
    return kt_values.get(detail_type.lower(), 1.2)


def shell_beam_coupling_stress(
    shell_stress: float,
    beam_stress: float,
    coupling_factor: float = 0.8,
) -> float:
    """Combine shell and beam stresses for hot-spot calculation.

    Parameters
    ----------
    shell_stress : float
        Stress from shell element analysis (MPa).
    beam_stress : float
        Stress from beam element analysis (MPa).
    coupling_factor : float
        Weighting factor for shell contribution (0-1).

    Returns
    -------
    float
        Combined hot-spot stress (MPa).
    """
    return coupling_factor * shell_stress + (1 - coupling_factor) * beam_stress
