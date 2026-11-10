"""Ice sheet mechanics module for TensorPhysics.

Provides Winkler elastic foundation plate models for floating ice sheets:
analytical (Navier/Hetenyi-type) and finite-difference solvers, plus a
conforming Kirchhoff thin-plate finite-element solver (Bogner-Fox-Schmit)
used for level-ice bending-failure analysis under ship-bow indentation loads.

Physical domain: ice_mechanics
Campaigns: SIR-PC2 | F01 | Winkler elastic foundation plate
           IN11-H3 | level-ice bending failure, Kirchhoff plate FEM
           P54-W3-N02 | Pasternak (two-parameter) foundation, closed GAP-FEM-P0
           P366-W2-N01 | ice thickness / temperature / loading-rate design laws

Sub-modules
-----------
``ice_material``
    Rate- and temperature-dependent ice state: ``E(T, eps_dot)``,
    ``sigma_f(T, eps_dot)``, strain rate from the indentation rate, and the
    self-consistent ``(E, D, l, G_p)`` state of the sheet.
``design_laws``
    Peak bending stress / first-crack load of the sheet, the constant-pressure
    patch test of the BFS foundation element, and the thickness / temperature /
    loading-rate dependence study.
"""

from .kirchhoff_plate_fem import (
    BFSPlateSolver,
    KirchhoffWinklerPlate,
    PlateField,
    bending_stress,
    critical_point_load,
    fan_sector_area,
    fan_sector_mask,
    graded_axis,
    graded_axis_centered,
    kelvin_point_load,
    kelvin_point_load_moments,
    limiting_moment,
)
from .pasternak_foundation import (
    PasternakFoundation,
    bessel_k_reference,
    calibration_table,
    ice_foundation_calibration,
    pasternak_critical_patch_load,
    pasternak_critical_point_load,
    pasternak_patch_field,
    pasternak_patch_load,
    pasternak_point_load,
    pasternak_point_load_moments,
    pasternak_roots,
    shear_coupling,
)
from .ice_material import (
    GRAVITY,
    SEA_WATER_DENSITY,
    IceSheetMaterial,
    creep_rate_factor,
    ice_elastic_modulus,
    ice_flexural_strength,
    ice_shear_modulus,
    material_sweep,
    strain_rate_from_indentation_rate,
    temperature_stiffening,
)
from .design_laws import (
    dependence_study,
    first_crack_load,
    foundation_patch_test,
    loglog_slope,
    peak_bending_stress,
    sign_report,
    unit_moment_field,
)
from .winkler_plate import (
    WinklerPlateParams,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
    winkler_kw_parametric,
)

__all__ = [
    "WinklerPlateParams",
    "navier_center_deflection",
    "navier_deflection_field",
    "solve_winkler_plate_fd",
    "winkler_kw_parametric",
    "BFSPlateSolver",
    "KirchhoffWinklerPlate",
    "PlateField",
    "bending_stress",
    "critical_point_load",
    "fan_sector_area",
    "fan_sector_mask",
    "graded_axis",
    "graded_axis_centered",
    "kelvin_point_load",
    "kelvin_point_load_moments",
    "limiting_moment",
    "PasternakFoundation",
    "bessel_k_reference",
    "calibration_table",
    "ice_foundation_calibration",
    "pasternak_critical_patch_load",
    "pasternak_critical_point_load",
    "pasternak_patch_field",
    "pasternak_patch_load",
    "pasternak_point_load",
    "pasternak_point_load_moments",
    "pasternak_roots",
    "shear_coupling",
    "GRAVITY",
    "SEA_WATER_DENSITY",
    "IceSheetMaterial",
    "creep_rate_factor",
    "ice_elastic_modulus",
    "ice_flexural_strength",
    "ice_shear_modulus",
    "material_sweep",
    "strain_rate_from_indentation_rate",
    "temperature_stiffening",
    "dependence_study",
    "first_crack_load",
    "foundation_patch_test",
    "loglog_slope",
    "peak_bending_stress",
    "sign_report",
    "unit_moment_field",
]
