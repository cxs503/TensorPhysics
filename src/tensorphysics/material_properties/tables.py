"""Declared-once source tables for MPDB v1 (substance catalogue + tabulated values).

Everything in this module is a *declaration* with its source locator attached;
the numeric payloads are either table transcriptions (``TABLE_PRIMARY`` rows) or
inputs to the encoded correlations in :mod:`correlations`.  ``build.py`` turns
these declarations into validated JSONL rows.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Substance catalogue
# --------------------------------------------------------------------------- #
SUBSTANCES: Dict[str, dict] = {
    "water": {
        "class": "liquid",
        "label": "fresh water (ordinary water substance, 0.1 MPa)",
        "use_cases": [
            "model-scale resistance / towing-tank Reynolds number",
            "propeller open-water and cavitation scaling",
            "LBM/FVM working fluid at 15 degC",
        ],
    },
    "seawater": {
        "class": "liquid",
        "label": "standard seawater (S = 5..40 g/kg)",
        "use_cases": [
            "full-scale ship resistance and propeller inflow",
            "seakeeping added resistance / RAO in seawater",
            "condenser cooling water for ORC and waste-heat chains",
        ],
    },
    "air": {
        "class": "gas",
        "label": "dry air at 1 atm (0.5-2 bar sampled)",
        "use_cases": [
            "above-water aerodynamics / wind heeling moment",
            "FVM/DG convective-flux scaling (rho, mu, Pr)",
            "cabin ventilation and HVAC boundary conditions",
        ],
    },
    "ice_fresh": {
        "class": "ice",
        "label": "fresh-water (lake/river) ice Ih",
        "use_cases": ["lake/Baltic ice plate bending", "model-ice scaling checks"],
    },
    "ice_sea": {
        "class": "ice",
        "label": "first-year sea ice (S = 5 g/kg typical, T dependent)",
        "use_cases": [
            "ice resistance (Lindqvist / Riska) strength inputs",
            "ice plate FEM flexural capacity",
            "ice-structure friction and crushing",
        ],
    },
    "sand_quartz": {
        "class": "sediment",
        "label": "medium quartz sand (D50 = 0.3 mm)",
        "use_cases": ["grounding / seabed contact stiffness", "propeller-jet scour"],
    },
    "silt": {
        "class": "sediment",
        "label": "silt (D50 = 0.03 mm)",
        "use_cases": ["harbour-basin grounding", "manoeuvring in muddy water"],
    },
    "clay": {
        "class": "sediment",
        "label": "soft marine clay",
        "use_cases": ["anchor holding capacity", "grounding penetration"],
    },
    "gravel": {
        "class": "sediment",
        "label": "gravel / cobble bed (D50 = 20 mm)",
        "use_cases": ["grounding on coarse bed", "scour-protection design"],
    },
    "steel_S235": {"class": "metal", "label": "EN 10025-2 S235 structural steel",
                   "use_cases": ["hull scantling", "superstructure"]},
    "steel_S355": {"class": "metal", "label": "EN 10025-2 S355 structural steel",
                   "use_cases": ["hull girder", "high-stress frames"]},
    "steel_AH36": {"class": "metal", "label": "IACS AH36 hull structural steel",
                   "use_cases": ["ice belt", "shell plating"]},
    "steel_EH36": {"class": "metal", "label": "IACS EH36 hull structural steel (-40 degC toughness)",
                   "use_cases": ["polar-class shell plating"]},
    "alu_5083_O": {"class": "metal", "label": "AA5083-O marine aluminium (annealed)",
                   "use_cases": ["fast-craft plating"]},
    "alu_5083_H116": {"class": "metal", "label": "AA5083-H116 marine aluminium (strain hardened)",
                      "use_cases": ["fast-craft hull and superstructure"]},
    "alu_6082_T6": {"class": "metal", "label": "AA6082-T6 structural aluminium",
                    "use_cases": ["extruded stiffeners"]},
    "gfrp_laminate": {"class": "composite", "label": "marine GFRP single-skin laminate",
                      "use_cases": ["yacht / small-craft shell"]},
    "gfrp_sandwich": {"class": "composite", "label": "GFRP/balsa sandwich (equivalent single layer)",
                      "use_cases": ["cabin-noise panel", "deckhouse"]},
    "concrete_C30": {"class": "cementitious", "label": "concrete C30/37 (EN 1992)",
                     "use_cases": ["gravity-based caisson", "ice-class bow reinforcement"]},
}

# --------------------------------------------------------------------------- #
# Ice
# --------------------------------------------------------------------------- #
#: (property, value_SI, rel_uncertainty, locator, note, variant) -- Timco & Weeks (2010)
ICE_SEA_TABLE: List[Tuple[str, float, float, str, str, str]] = [
    ("bulk_density", 900.0, 0.05, "Table 2, p. 109 (first-year, 720-940)",
     "typical first-year bulk density", "first_year"),
    ("bulk_density", 910.0, 0.05, "Table 2, p. 109 (multi-year)",
     "typical multi-year bulk density", "multi_year"),
    ("youngs_modulus", 9.0e9, 0.40, "Table 3, p. 111 (initial modulus)",
     "initial / loading-rate modulus", "initial"),
    ("youngs_modulus", 3.0e9, 0.60, "Table 3, p. 111 (effective modulus)",
     "effective modulus for plate bending", "effective"),
    ("poisson_ratio", 0.33, 0.10, "Table 3, p. 111", "sea-ice Poisson ratio", None),
    ("tensile_strength", 1.0e6, 0.35, "Table 4, p. 113", "horizontal tensile strength", None),
    ("compressive_strength", 5.0e6, 0.40, "Table 4, p. 113 (horizontal)",
     "horizontal compressive strength", "horizontal"),
    ("compressive_strength", 2.5e6, 0.40, "Table 4, p. 113 (vertical)",
     "vertical compressive strength", "vertical"),
    ("shear_strength", 0.6e6, 0.35, "Table 4, p. 113", "shear strength", None),
    ("ice_friction_coefficient", 0.15, 0.40, "Table 6, p. 121", "ice-ice sliding friction", None),
    ("salinity", 5.0, 0.40, "Table 1, p. 108", "typical first-year bulk salinity", None),
]

#: Fresh-water ice Ih, Hobbs (1974)
ICE_FRESH_TABLE: List[Tuple[str, float, float, str, str]] = [
    ("youngs_modulus", 9.0e9, 0.30, "Table 6.1", "polycrystalline granular ice"),
    ("poisson_ratio", 0.33, 0.10, "Table 6.1", "ice Ih"),
    ("thermal_expansion", 5.05e-5, 0.10, "Table 2.4 (linear, near 0 degC)", "linear coefficient"),
    ("thermal_conductivity", 2.20, 0.06, "Table 2.6 (at 0 degC)", "polycrystalline ice Ih"),
    ("latent_heat_fusion", 333.55e3, 0.001, "Table 2.3", "at 0 degC"),
    ("specific_heat", 2100.0, 0.05, "Table 2.5 (at 0 degC)", "cp of ice Ih"),
]

#: Temperature grid for fresh-ice density [degC]
ICE_FRESH_DENSITY_T_C = [-40.0, -30.0, -20.0, -10.0, -5.0, -2.0, 0.0]

#: Sea-ice brine-volume / flexural-strength temperature grid [degC]
ICE_SEA_T_C = [-2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0, -12.0, -15.0, -18.0, -20.0]

# --------------------------------------------------------------------------- #
# Sediment (Soulsby 1997 Tables 2-3/2-4; DNV-RP-C212 Table 3-1)
# --------------------------------------------------------------------------- #
SEDIMENT_TABLE: Dict[str, List[Tuple[str, float, float, str, str]]] = {
    "sand_quartz": [
        ("grain_density", 2650.0, 0.01, "Soulsby 1997 Table 2-3", "quartz sand"),
        ("dry_bulk_density", 1600.0, 0.05, "Soulsby 1997 Table 2-4", "medium sand, loose-medium"),
        ("friction_angle", 32.0, 0.10, "DNV-RP-C212 Table 3-1", "medium dense sand"),
        ("cohesion", 0.0, 0.0, "DNV-RP-C212 Sec. 3 (drained sand)", "drained sand cohesion intercept"),
        ("angle_of_repose", 32.0, 0.10, "Soulsby 1997 Fig. 2.5", "dry sand"),
        ("median_grain_size", 3.0e-4, 0.30, "Soulsby 1997 Fig. 2.5 (medium sand)", "D50"),
    ],
    "silt": [
        ("grain_density", 2650.0, 0.01, "Soulsby 1997 Table 2-3", "quartz silt"),
        ("dry_bulk_density", 1450.0, 0.08, "Soulsby 1997 Table 2-4", "loose silt"),
        ("friction_angle", 28.0, 0.12, "DNV-RP-C212 Table 3-1", "silt"),
        ("cohesion", 2.0e3, 0.50, "DNV-RP-C212 Table 3-1", "slightly cohesive silt"),
        ("median_grain_size", 3.0e-5, 0.30, "Soulsby 1997 Fig. 2.5 (silt)", "D50"),
    ],
    "clay": [
        ("grain_density", 2700.0, 0.02, "Soulsby 1997 Table 2-3 (clay minerals)", "clay particle density"),
        ("dry_bulk_density", 1400.0, 0.10, "Soulsby 1997 Table 2-4 (soft clay)", "soft marine clay"),
        ("friction_angle", 24.0, 0.15, "DNV-RP-C212 Table 3-1 (undrained clay)", "equivalent friction angle"),
        ("cohesion", 20.0e3, 0.50, "DNV-RP-C212 Table 3-1 (soft clay su)", "undrained shear strength"),
        ("median_grain_size", 2.0e-6, 0.50, "Soulsby 1997 Fig. 2.5 (clay)", "D50"),
    ],
    "gravel": [
        ("grain_density", 2650.0, 0.01, "Soulsby 1997 Table 2-3", "quartz gravel"),
        ("dry_bulk_density", 1800.0, 0.05, "Soulsby 1997 Table 2-4", "dense gravel"),
        ("friction_angle", 38.0, 0.10, "DNV-RP-C212 Table 3-1", "dense gravel"),
        ("cohesion", 0.0, 0.0, "DNV-RP-C212 Sec. 3 (drained gravel)", "drained gravel"),
        ("angle_of_repose", 38.0, 0.10, "Soulsby 1997 Fig. 2.5", "gravel"),
        ("median_grain_size", 2.0e-2, 0.40, "Soulsby 1997 Fig. 2.5 (gravel)", "D50"),
    ],
}

# --------------------------------------------------------------------------- #
# Structural materials
# --------------------------------------------------------------------------- #
#: (property, value_SI, rel_uncertainty, source_id, locator, note)
METAL_TABLE: Dict[str, List[Tuple[str, float, float, str, str, str]]] = {
    "steel_S235": [
        ("density", 7850.0, 0.01, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "steel density"),
        ("youngs_modulus", 210.0e9, 0.01, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "E"),
        ("shear_modulus", 81.0e9, 0.01, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "G"),
        ("poisson_ratio", 0.30, 0.03, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "nu"),
        ("yield_strength", 235.0e6, 0.02, "SRC-EN10025", "EN 10025-2 Table 7 (t <= 16 mm)", "ReH"),
        ("ultimate_strength", 360.0e6, 0.02, "SRC-EN10025", "EN 10025-2 Table 7 (t < 3 mm)", "Rm"),
        ("thermal_expansion", 12.0e-6, 0.05, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "alpha"),
        ("damping_loss_factor", 0.01, 0.50, "SRC-SHIP-PLATE-MATERIALS",
         "asset field materials.steel.eta_struct", "welded structure total loss factor"),
    ],
    "steel_S355": [
        ("density", 7850.0, 0.01, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "steel density"),
        ("youngs_modulus", 210.0e9, 0.01, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "E"),
        ("shear_modulus", 81.0e9, 0.01, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "G"),
        ("poisson_ratio", 0.30, 0.03, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "nu"),
        ("yield_strength", 355.0e6, 0.02, "SRC-EN10025", "EN 10025-2 Table 7 (t <= 16 mm)", "ReH"),
        ("ultimate_strength", 470.0e6, 0.02, "SRC-EN10025", "EN 10025-2 Table 7 (t < 3 mm)", "Rm"),
        ("thermal_expansion", 12.0e-6, 0.05, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "alpha"),
        ("damping_loss_factor", 0.01, 0.50, "SRC-SHIP-PLATE-MATERIALS",
         "asset field materials.steel.eta_struct", "welded structure total loss factor"),
    ],
    "steel_AH36": [
        ("density", 7850.0, 0.01, "SRC-IACS-UR-REQ", "UR W11 Table 1", "hull structural steel"),
        ("youngs_modulus", 206.0e9, 0.01, "SRC-IACS-UR-REQ", "UR S6 Sec. 2 (E = 2.06e11 N/m^2)",
         "E used by the IACS ice-class scantling formulae"),
        ("poisson_ratio", 0.30, 0.03, "SRC-IACS-UR-REQ", "UR S6 Sec. 2", "nu"),
        ("yield_strength", 355.0e6, 0.02, "SRC-IACS-UR-REQ", "UR W11 Table 1 (AH36)", "ReH"),
        ("ultimate_strength", 490.0e6, 0.02, "SRC-IACS-UR-REQ", "UR W11 Table 1 (AH36)", "Rm"),
        ("thermal_expansion", 12.0e-6, 0.05, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "alpha"),
    ],
    "steel_EH36": [
        ("density", 7850.0, 0.01, "SRC-IACS-UR-REQ", "UR W11 Table 1", "hull structural steel"),
        ("youngs_modulus", 206.0e9, 0.01, "SRC-IACS-UR-REQ", "UR S6 Sec. 2", "E"),
        ("poisson_ratio", 0.30, 0.03, "SRC-IACS-UR-REQ", "UR S6 Sec. 2", "nu"),
        ("yield_strength", 355.0e6, 0.02, "SRC-IACS-UR-REQ", "UR W11 Table 1 (EH36)", "ReH"),
        ("ultimate_strength", 490.0e6, 0.02, "SRC-IACS-UR-REQ", "UR W11 Table 1 (EH36)", "Rm"),
        ("thermal_expansion", 12.0e-6, 0.05, "SRC-EN10025", "EN 1993-1-1 Table 3.1", "alpha"),
    ],
    "alu_5083_O": [
        ("density", 2660.0, 0.01, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "rho"),
        ("youngs_modulus", 70.0e9, 0.01, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "E"),
        ("poisson_ratio", 0.33, 0.03, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "nu"),
        ("yield_strength", 125.0e6, 0.05, "SRC-EN1999", "EN 1999-1-1 Table 3.2b (5083-O)", "f_o"),
        ("ultimate_strength", 275.0e6, 0.05, "SRC-EN1999", "EN 1999-1-1 Table 3.2b (5083-O)", "f_u"),
        ("haz_strength_ratio", 0.67, 0.10, "SRC-EN1999", "EN 1999-1-1 Table 3.2c (5083-O, MIG)", "rho_haz"),
    ],
    "alu_5083_H116": [
        ("density", 2660.0, 0.01, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "rho"),
        ("youngs_modulus", 70.0e9, 0.01, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "E"),
        ("poisson_ratio", 0.33, 0.03, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "nu"),
        ("yield_strength", 215.0e6, 0.05, "SRC-EN1999", "EN 1999-1-1 Table 3.2b (5083-H116)", "f_o"),
        ("ultimate_strength", 305.0e6, 0.05, "SRC-EN1999", "EN 1999-1-1 Table 3.2b (5083-H116)", "f_u"),
        ("haz_strength_ratio", 0.62, 0.10, "SRC-EN1999", "EN 1999-1-1 Table 3.2c (5083-H116, MIG)", "rho_haz"),
    ],
    "alu_6082_T6": [
        ("density", 2700.0, 0.01, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "rho"),
        ("youngs_modulus", 70.0e9, 0.01, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "E"),
        ("poisson_ratio", 0.33, 0.03, "SRC-EN1999", "EN 1999-1-1 Table 3.2", "nu"),
        ("yield_strength", 250.0e6, 0.05, "SRC-EN1999", "EN 1999-1-1 Table 3.2b (6082-T6)", "f_o"),
        ("ultimate_strength", 290.0e6, 0.05, "SRC-EN1999", "EN 1999-1-1 Table 3.2b (6082-T6)", "f_u"),
        ("haz_strength_ratio", 0.48, 0.10, "SRC-EN1999", "EN 1999-1-1 Table 3.2c (6082-T6, MIG)", "rho_haz"),
    ],
}

COMPOSITE_TABLE: Dict[str, List[Tuple[str, float, float, str, str, str]]] = {
    "gfrp_laminate": [
        ("density", 1600.0, 0.06, "SRC-LLOYDS-GFRP", "Pt 3 Ch 4 Sec 3 (single-skin laminate)", "rho"),
        ("youngs_modulus", 18.0e9, 0.20, "SRC-LLOYDS-GFRP", "Pt 3 Ch 4 Sec 3",
         "quasi-isotropic E, design band 15-20 GPa"),
        ("poisson_ratio", 0.30, 0.10, "SRC-LLOYDS-GFRP", "Pt 3 Ch 4 Sec 3", "nu"),
        ("tensile_strength", 125.0e6, 0.25, "SRC-LLOYDS-GFRP", "Pt 3 Ch 4 Sec 3", "design band 100-150 MPa"),
        ("damping_loss_factor", 0.02, 0.40, "SRC-SHIP-PLATE-MATERIALS",
         "engineering value for FRP panels", "eta_struct"),
    ],
    "gfrp_sandwich": [
        ("density", 600.0, 0.15, "SRC-SHIP-PLATE-MATERIALS", "asset field materials.sandwich.rho_kgm3",
         "equivalent homogeneous layer"),
        ("youngs_modulus", 3.0e9, 0.30, "SRC-SHIP-PLATE-MATERIALS", "asset field materials.sandwich.E_Pa",
         "equivalent bending stiffness (engineering, not measured)"),
        ("poisson_ratio", 0.30, 0.10, "SRC-SHIP-PLATE-MATERIALS", "asset field materials.sandwich.nu", "nu"),
        ("damping_loss_factor", 0.02, 0.40, "SRC-SHIP-PLATE-MATERIALS",
         "asset field materials.sandwich.eta_struct", "eta_struct"),
    ],
}

CONCRETE_TABLE: List[Tuple[str, float, float, str, str, str]] = [
    ("density", 2400.0, 0.02, "SRC-EN1992", "EN 1992-1-1 Table 3.1 (C30/37)", "rho"),
    ("youngs_modulus", 33.0e9, 0.05, "SRC-EN1992", "EN 1992-1-1 Table 3.1 (C30/37)", "E_cm"),
    ("poisson_ratio", 0.20, 0.10, "SRC-EN1992", "EN 1992-1-1 Sec. 3.1.3(3)", "nu"),
    ("compressive_strength", 30.0e6, 0.05, "SRC-EN1992", "EN 1992-1-1 Table 3.1 (f_ck)",
     "characteristic cylinder strength"),
    ("tensile_strength", 2.9e6, 0.15, "SRC-EN1992", "EN 1992-1-1 Table 3.1 (f_ctm for C30/37)",
     "mean axial tensile strength"),
    ("thermal_expansion", 10.0e-6, 0.10, "SRC-EN1992", "EN 1992-1-1 Sec. 3.1.3(3)", "alpha_t"),
]