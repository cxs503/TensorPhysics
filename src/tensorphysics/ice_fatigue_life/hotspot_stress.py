"""Hot-spot stress method for ice-induced fatigue assessment.

Implements the hot-spot stress approach for plate-beam connections (weld toes)
in ice zone structures:

1. Geometric stress concentration factor K_t (Peterson formula)
2. FEM sub-model mesh convergence check
3. Hot-spot stress = K_t × nominal stress
4. Thickness correction per DNV-RP-C203

References
----------
- DNV-RP-C203 (2021) Section 5: Hot-spot Stress Method
- Peterson, R.E. (1953): Stress Concentration Factors
- Niemi, E. (1995): Stress Determination for Fatigue Analysis (IIW)
- Fricke, W. (2012): IIW Guidelines for Weld Fatigue Assessment
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

__all__ = [
    "HotSpotStress",
    "PetersonKt",
    "FEMSubmodelConvergence",
    "plate_beam_kt_peterson",
    "hot_spot_stress_from_nominal",
]


@dataclass
class PetersonKt:
    """Peterson stress concentration factor for plate-beam connections.
    
    The Peterson formula gives the theoretical elastic stress concentration
    factor K_t for common weld geometries at plate-beam intersections.
    
    K_t = 1 + alpha * sqrt(a_prime / rho)
    
    where:
    - alpha: geometry factor (depends on joint type)
    - a_prime: Peterson's material constant (mm)
    - rho: weld toe radius (mm)
    
    Attributes
    ----------
    kt : float
        Stress concentration factor.
    alpha : float
        Geometry factor.
    a_prime_mm : float
        Peterson material constant (mm).
    rho_mm : float
        Weld toe radius (mm).
    """
    kt: float
    alpha: float
    a_prime_mm: float
    rho_mm: float


@dataclass
class HotSpotStress:
    """Hot-spot stress result at weld toe.
    
    Attributes
    ----------
    nominal_stress_mpa : float
        Nominal (far-field) stress range (MPa).
    kt_geometric : float
        Geometric stress concentration factor.
    kt_weld : float
        Weld profile contribution to K_t.
    kt_total : float
        Total stress concentration factor.
    thickness_factor : float
        Thickness correction factor.
    hot_spot_stress_mpa : float
        Hot-spot stress range at weld toe (MPa).
    plate_thickness_mm : float
        Plate thickness (mm).
    weld_toe_radius_mm : float
        Assumed weld toe radius (mm).
    """
    nominal_stress_mpa: float
    kt_geometric: float
    kt_weld: float
    kt_total: float
    thickness_factor: float
    hot_spot_stress_mpa: float
    plate_thickness_mm: float
    weld_toe_radius_mm: float


@dataclass
class FEMSubmodelConvergence:
    """FEM sub-model mesh convergence result for K_t.
    
    Attributes
    ----------
    mesh_sizes_mm : list
        Element sizes tested (mm).
    kt_values : list
        K_t values at each mesh size.
    converged_kt : float
        K_t at finest mesh.
    relative_change_pct : float
        Relative change between last two mesh sizes (%).
    is_converged : bool
        True if relative change ≤ 3%.
    """
    mesh_sizes_mm: List[float]
    kt_values: List[float]
    converged_kt: float
    relative_change_pct: float
    is_converged: bool


def peterson_material_constant(
    ultimate_strength_mpa: float = 450.0,
) -> float:
    """Peterson's material constant a' for steel.
    
    a' depends on the ultimate tensile strength of the material.
    For structural steels (S355, AH36, etc.):
    
    a' ≈ 25 / (Rm / 100)^1.6  (mm)
    
    where Rm is the ultimate tensile strength in MPa.
    
    Parameters
    ----------
    ultimate_strength_mpa : float
        Ultimate tensile strength Rm (MPa). Default 450 for AH36.
    
    Returns
    -------
    float
        Peterson material constant a' (mm).
    """
    return 25.0 / (ultimate_strength_mpa / 100.0) ** 1.6


def plate_beam_kt_peterson(
    plate_thickness_mm: float,
    beam_spacing_mm: float,
    weld_toe_radius_mm: float = 0.5,
    attachment_length_mm: Optional[float] = None,
    ultimate_strength_mpa: float = 450.0,
    joint_type: str = "stiffener",
) -> PetersonKt:
    """Stress concentration factor K_t for plate-beam connection.
    
    Uses Peterson's formula for the weld toe at a plate-stiffener intersection:
    
    K_t = 1 + alpha * sqrt(a' / rho)
    
    The geometry factor alpha depends on the joint type:
    - "stiffener": longitudinal stiffener to plate (alpha ≈ 0.5-1.0)
    - "transverse": transverse stiffener (alpha ≈ 0.8-1.2)
    - "bracket": bracket toe (alpha ≈ 1.0-1.5)
    - "gusset": gusset plate connection (alpha ≈ 0.6-1.0)
    
    Parameters
    ----------
    plate_thickness_mm : float
        Plate thickness t (mm).
    beam_spacing_mm : float
        Spacing between beams/stiffeners (mm).
    weld_toe_radius_mm : float
        Weld toe radius rho (mm). Typical: 0.3-1.0 mm.
    attachment_length_mm : float, optional
        Length of attachment (for bracket/gusset types).
    ultimate_strength_mpa : float
        Ultimate tensile strength (MPa). Default 450 for AH36.
    joint_type : str
        Joint type: 'stiffener', 'transverse', 'bracket', 'gusset'.
    
    Returns
    -------
    PetersonKt
        Stress concentration factor result.
    
    Notes
    -----
    For ice zone structures, the weld toe radius is typically small
    (0.3-0.5 mm) due to the harsh welding conditions. Default 0.5 mm
    is a conservative assumption per DNV-RP-C203.
    """
    # Material constant
    a_prime = peterson_material_constant(ultimate_strength_mpa)
    
    # Geometry factor based on joint type
    alpha_map = {
        "stiffener": 0.7,
        "transverse": 1.0,
        "bracket": 1.2,
        "gusset": 0.8,
    }
    
    if joint_type.lower() not in alpha_map:
        raise ValueError(
            f"Unknown joint type '{joint_type}'. "
            f"Available: {list(alpha_map.keys())}"
        )
    
    alpha = alpha_map[joint_type.lower()]
    
    # Aspect ratio correction for stiffener-to-plate
    aspect = beam_spacing_mm / plate_thickness_mm
    if joint_type.lower() == "stiffener":
        # Wider spacing → higher K_t (more plate bending)
        alpha *= min(1.0 + 0.1 * np.log10(aspect / 20.0), 1.3)
    
    # Peterson formula
    kt = 1.0 + alpha * np.sqrt(a_prime / weld_toe_radius_mm)
    
    return PetersonKt(
        kt=kt,
        alpha=alpha,
        a_prime_mm=a_prime,
        rho_mm=weld_toe_radius_mm,
    )


def hot_spot_stress_from_nominal(
    nominal_stress_mpa: float,
    plate_thickness_mm: float,
    beam_spacing_mm: float,
    weld_toe_radius_mm: float = 0.5,
    joint_type: str = "stiffener",
    ultimate_strength_mpa: float = 450.0,
    reference_thickness_mm: float = 25.0,
    thickness_exponent: float = 0.25,
) -> HotSpotStress:
    """Calculate hot-spot stress at weld toe from nominal stress.
    
    Combines:
    1. Geometric K_t (Peterson formula)
    2. Weld profile contribution (typically 1.0-1.2)
    3. Thickness correction per DNV-RP-C203
    
    sigma_hotspot = K_t_total × sigma_nominal × (t/t_ref)^k
    
    Parameters
    ----------
    nominal_stress_mpa : float
        Nominal stress range (MPa).
    plate_thickness_mm : float
        Plate thickness (mm).
    beam_spacing_mm : float
        Beam/stiffener spacing (mm).
    weld_toe_radius_mm : float
        Weld toe radius (mm). Default 0.5 mm.
    joint_type : str
        Joint type for K_t calculation.
    ultimate_strength_mpa : float
        Ultimate tensile strength (MPa).
    reference_thickness_mm : float
        Reference thickness for thickness correction (mm).
    thickness_exponent : float
        Thickness correction exponent k.
    
    Returns
    -------
    HotSpotStress
        Complete hot-spot stress result.
    """
    # Geometric K_t from Peterson
    peterson = plate_beam_kt_peterson(
        plate_thickness_mm=plate_thickness_mm,
        beam_spacing_mm=beam_spacing_mm,
        weld_toe_radius_mm=weld_toe_radius_mm,
        ultimate_strength_mpa=ultimate_strength_mpa,
        joint_type=joint_type,
    )
    
    # Weld profile contribution (as-welded condition)
    kt_weld = 1.1  # typical for as-welded fillet/butt weld
    
    # Total K_t
    kt_total = peterson.kt * kt_weld
    
    # Thickness correction
    if plate_thickness_mm > reference_thickness_mm:
        thick_factor = (plate_thickness_mm / reference_thickness_mm) ** thickness_exponent
    else:
        thick_factor = 1.0
    
    # Hot-spot stress
    hot_spot = nominal_stress_mpa * kt_total * thick_factor
    
    return HotSpotStress(
        nominal_stress_mpa=nominal_stress_mpa,
        kt_geometric=peterson.kt,
        kt_weld=kt_weld,
        kt_total=kt_total,
        thickness_factor=thick_factor,
        hot_spot_stress_mpa=hot_spot,
        plate_thickness_mm=plate_thickness_mm,
        weld_toe_radius_mm=weld_toe_radius_mm,
    )


def fem_submodel_convergence_study(
    plate_thickness_mm: float,
    beam_spacing_mm: float,
    weld_toe_radius_mm: float = 0.5,
    joint_type: str = "stiffener",
    mesh_sizes_mm: Optional[List[float]] = None,
) -> FEMSubmodelConvergence:
    """FEM sub-model mesh convergence study for K_t.
    
    Performs a simplified convergence study by evaluating K_t at
    different effective mesh sizes. In a real FEM analysis, this would
    involve actual FE sub-modeling; here we use the analytical Peterson
    formula with mesh-size-dependent notch rounding.
    
    The effective weld toe radius increases with mesh size:
        rho_eff = max(rho, h_mesh / 2)
    
    This simulates the mesh smoothing effect on stress concentration.
    
    Parameters
    ----------
    plate_thickness_mm : float
        Plate thickness (mm).
    beam_spacing_mm : float
        Beam spacing (mm).
    weld_toe_radius_mm : float
        Actual weld toe radius (mm).
    joint_type : str
        Joint type.
    mesh_sizes_mm : list, optional
        Element sizes to test. Default: [5.0, 2.5, 1.25, 0.625, 0.3125].
    
    Returns
    -------
    FEMSubmodelConvergence
        Convergence study result.
    """
    if mesh_sizes_mm is None:
        mesh_sizes_mm = [5.0, 2.5, 1.25, 0.625, 0.3125]
    
    kt_values = []
    
    for h in mesh_sizes_mm:
        # Effective notch radius increases with mesh size
        rho_eff = max(weld_toe_radius_mm, h / 2.0)
        
        peterson = plate_beam_kt_peterson(
            plate_thickness_mm=plate_thickness_mm,
            beam_spacing_mm=beam_spacing_mm,
            weld_toe_radius_mm=rho_eff,
            joint_type=joint_type,
        )
        kt_values.append(peterson.kt)
    
    converged_kt = kt_values[-1]
    
    # Relative change between last two mesh sizes
    if len(kt_values) >= 2 and kt_values[-2] > 0:
        rel_change = abs(kt_values[-1] - kt_values[-2]) / kt_values[-2] * 100.0
    else:
        rel_change = 0.0
    
    return FEMSubmodelConvergence(
        mesh_sizes_mm=mesh_sizes_mm,
        kt_values=kt_values,
        converged_kt=converged_kt,
        relative_change_pct=rel_change,
        is_converged=rel_change <= 3.0,
    )
