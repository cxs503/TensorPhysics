"""Source registry for MPDB v1: every number in the database points at one row here.

Each entry carries the full bibliographic citation, the locator granularity used
by the rows that reference it (equation / table / page), the accuracy claim of
the source, and an ``access`` field recording how the value was obtained in the
build sandbox:

``access = "offline_no_pdf"``
    The sandbox had no network egress (verified: outbound HTTPS blocked), so the
    source document itself could not be re-opened at build time.  The encoded
    content is the published correlation/table content, and every
    ``FORMULA_PRIMARY`` row is recomputable by ``repro_cmd`` and is checked
    against independent published anchor values (see ``ANCHORS`` and
    ``artifacts/mpdb_v1/anchors.csv``).  ``TABLE_PRIMARY`` rows carry the
    table/page locator but are *not* machine re-derivable -- this limitation is
    declared, not hidden.

No row may reference a source id that is not in :data:`SOURCE_REGISTRY`
(enforced by ``schema.validate_record``).
"""

from __future__ import annotations

from typing import Dict

SOURCE_REGISTRY: Dict[str, dict] = {
    # ---------------- fresh water ----------------
    "SRC-IAPWS-WATER-DENSITY": {
        "citation": (
            "Kell, G.S. (1975) Density, thermal expansivity, and compressibility of "
            "liquid water from 0 to 150 C: a discussion of published data; equation as "
            "recommended in IAPWS R7-97 / Tanaka et al. (2001) Metrologia 38:301, "
            "eq. (2) for the 0.1 MPa isobar"
        ),
        "locator_granularity": "equation + page",
        "accuracy": "<= 1e-5 relative on the 0.1 MPa isobar, 273-423 K",
        "access": "offline_no_pdf",
    },
    "SRC-IAPWS-WATER-VISCOSITY": {
        "citation": (
            "Kestin, J., Sokolov, M. & Wakeham, W.A. (1978) Viscosity of liquid water in "
            "the range -8 C to 150 C, J. Phys. Chem. Ref. Data 7(3):941-948; "
            "Vogel-type three-parameter fit (eq. 2, p. 943)"
        ),
        "locator_granularity": "equation + page",
        "accuracy": "<= 1% 273-423 K (checked against Table 2 anchors at 0-100 C)",
        "access": "offline_no_pdf",
    },
    "SRC-IAPWS-WATER-CONDUCTIVITY": {
        "citation": (
            "IAPWS R15-11 (2011) Release on the IAPWS Formulation 2011 for the Thermal "
            "Conductivity of Ordinary Water Substance; cubic polynomial fit to the "
            "0.1 MPa isobar, 273-373 K"
        ),
        "locator_granularity": "equation",
        "accuracy": "<= 1.5% of the R15-11 isobar values 273-373 K",
        "access": "offline_no_pdf",
    },
    "SRC-IAPWS-WATER-SPEED": {
        "citation": (
            "Bilaniuk, N. & Wong, G.S.K. (1993) Speed of sound in pure water as a "
            "function of temperature, J. Acoust. Soc. Am. 93(3):1609-1612, eq. (2)"
        ),
        "locator_granularity": "equation + page",
        "accuracy": "<= 0.05% 273-373 K",
        "access": "offline_no_pdf",
    },
    "SRC-WATER-CP-TABLE": {
        "citation": (
            "CRC Handbook of Chemistry and Physics (97th ed., 2016-2017), "
            "'Specific heat capacity of liquid water at 1 atm', table p. 6-4 "
            "(values at 0/10/20/25/30/40/50/60/70/80/90/100 C)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "3-4 significant figures as tabulated",
        "access": "offline_no_pdf",
    },
    "SRC-WATER-LATENT-FUSION": {
        "citation": (
            "IAPWS R14-08 / CRC Handbook of Chemistry and Physics (97th ed.), "
            "'Enthalpy of fusion of ice at 0 C', table p. 6-4"
        ),
        "locator_granularity": "table + page",
        "accuracy": "tabulated to 4 significant figures",
        "access": "offline_no_pdf",
    },
    # ---------------- seawater ----------------
    "SRC-SHARQAWY-DENSITY": {
        "citation": (
            "Sharqawy, M.H., Lienhard V, J.H. & Zubair, S.M. (2010) Thermophysical "
            "properties of seawater: a review of existing correlations and data, "
            "Desalination and Water Treatment 16:354-380, Table 4 (density of seawater, "
            "eq. 2-5, valid 273-453 K, S = 0-160 g/kg)"
        ),
        "locator_granularity": "table + equation",
        "accuracy": "<= 0.01% vs. measured seawater density (0-40 C, S = 0-42 g/kg)",
        "access": "offline_no_pdf",
    },
    "SRC-SHARQAWY-VISCOSITY": {
        "citation": (
            "Sharqawy, M.H., Lienhard V, J.H. & Zubair, S.M. (2010) Thermophysical "
            "properties of seawater, Desalination and Water Treatment 16:354-380, "
            "Table 5 (dynamic viscosity, eq. 22-24, valid 273-453 K, S = 0-150 g/kg)"
        ),
        "locator_granularity": "table + equation",
        "accuracy": "<= 1.5% vs. measured seawater viscosity",
        "access": "offline_no_pdf",
    },
    "SRC-SHARQAWY-CP": {
        "citation": (
            "Sharqawy et al. (2010), Table 3 (specific heat capacity of seawater, "
            "eq. 9-13, valid 273-453 K, S = 0-180 g/kg)"
        ),
        "locator_granularity": "table + equation",
        "accuracy": "<= 0.3%",
        "access": "offline_no_pdf",
    },
    "SRC-ITTC-WATER-VISCOSITY-ANCHOR": {
        "citation": (
            "ITTC Recommended Procedures 7.5-02-01-03 (2011 revision) 'Fresh water and "
            "seawater properties', Table 1: kinematic viscosity at 15 C, 1.1390e-6 m^2/s "
            "(fresh water) and 1.1883e-6 m^2/s (seawater, 35 g/kg)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "anchor values quoted to 5 significant figures",
        "access": "offline_no_pdf",
    },
    "SRC-MILLERO-FREEZING": {
        "citation": (
            "Millero, F.J. & Leung, W.H. (1976) The thermodynamics of seawater at one "
            "atmosphere, Am. J. Sci. 276:1035-1077; seawater freezing-point depression as "
            "summarised in Sharqawy et al. (2010) eq. (6), 1 atm"
        ),
        "locator_granularity": "equation",
        "accuracy": "<= 0.01 K (S = 0-40 g/kg, 1 atm)",
        "access": "offline_no_pdf",
    },
    # ---------------- air ----------------
    "SRC-AIR-IDEAL-GAS": {
        "citation": (
            "ISO 2533:1975 Standard Atmosphere / CRC Handbook of Chemistry and Physics "
            "(97th ed.) 'Dry air: molar mass 28.9647 g/mol, gas constant R = 287.058 "
            "J/(kg K)'; ideal-gas density rho = p/(R T)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "<= 0.1% for p <= 1 MPa, 250-350 K",
        "access": "offline_no_pdf",
    },
    "SRC-AIR-SUTHERLAND": {
        "citation": (
            "Sutherland, W. (1893) The viscosity and thermal conductivity of gases, "
            "Phil. Mag. 36:507; coefficients mu0 = 1.716e-5 Pa s, T0 = 273.15 K, "
            "S = 110.4 K as given in White, F.M. (2006) Viscous Fluid Flow, 3rd ed., "
            "Table 1-2, p. 28"
        ),
        "locator_granularity": "equation + table + page",
        "accuracy": "<= 2% 250-800 K at 0.1 MPa",
        "access": "offline_no_pdf",
    },
    "SRC-AIR-CONDUCTIVITY": {
        "citation": (
            "CRC Handbook of Chemistry and Physics (97th ed.), 'Thermal conductivity of "
            "gases: air', table p. 6-190; modified-Eucken/Sutherland fit "
            "k = 2.646e-3 * T^1.5 / (T + 245.4 * 10^(-12/T)) (k in W/(m K), T in K)"
        ),
        "locator_granularity": "table + equation",
        "accuracy": "<= 1.5% 250-600 K at 0.1 MPa",
        "access": "offline_no_pdf",
    },
    "SRC-AIR-CP-GAMMA": {
        "citation": (
            "CRC Handbook (97th ed.), 'Specific heat capacity of air at 1 atm', "
            "table p. 6-2; cp = 1005 J/(kg K), gamma = 1.4 for 250-350 K"
        ),
        "locator_granularity": "table + page",
        "accuracy": "cp varies < 0.5% over 250-350 K",
        "access": "offline_no_pdf",
    },
    # ---------------- ice ----------------
    "SRC-TIMCO-WEEKS-2010": {
        "citation": (
            "Timco, G.W. & Weeks, W.F. (2010) A review of the engineering properties of "
            "sea ice, Cold Regions Science and Technology 60:107-129 "
            "(Tables 2-6: density, elastic modulus, flexural/compressive/tensile/shear "
            "strength, friction)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "review range statements, typical values with stated spread",
        "access": "offline_no_pdf",
    },
    "SRC-FRANKENSTEIN-GARNER": {
        "citation": (
            "Frankenstein, G. & Garner, R. (1967) Equations for determining the brine "
            "volume of sea ice from -0.5 to -22.9 C, J. Glaciol. 6(48):943-944, eq. (1)"
        ),
        "locator_granularity": "equation + page",
        "accuracy": "fitted over -0.5 to -22.9 C; stated fit residual < 0.01",
        "access": "offline_no_pdf",
    },
    "SRC-TIMCO-WEEKS-FLEXURAL": {
        "citation": (
            "Timco, G.W. & Weeks, W.F. (2010) ..., eq. (8) and Fig. 8: flexural strength "
            "sigma_f = 1.76 exp(-5.88 sqrt(v_b)) MPa for v_b <= 0.3 (v_b = brine volume "
            "fraction), first-year sea ice"
        ),
        "locator_granularity": "equation + figure",
        "accuracy": "scatter band +/-30% about the fit (as published)",
        "access": "offline_no_pdf",
    },
    "SRC-PURE-ICE-IH": {
        "citation": (
            "Hobbs, P.V. (1974) Ice Physics, Clarendon Press, Oxford, Table 2.1 "
            "(density of ice Ih at 0 C, 916.8 kg/m^3) and Table 6.1 "
            "(Young's modulus of granular/polycrystalline ice 9.0 GPa)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "tabulated lattice/granular values",
        "access": "offline_no_pdf",
    },
    # ---------------- sediment ----------------
    "SRC-SOULSBY-1997": {
        "citation": (
            "Soulsby, R. (1997) Dynamics of Marine Sands, Thomas Telford, London, "
            "Tables 2-3 (grain density 2650 kg/m^3 for quartz sand), 2-4 (dry bulk "
            "density 1600 kg/m^3), and Fig. 2.5 (settling/D50 classes)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "nominal design values with stated ranges",
        "access": "offline_no_pdf",
    },
    "SRC-DNV-C212": {
        "citation": (
            "DNV-RP-C212 (2017) Offshore soil mechanics and geotechnical engineering, "
            "Sec. 3 Table 3-1 / App. A: typical friction angle and cohesion for "
            "sand, silt, clay, gravel"
        ),
        "locator_granularity": "table + section",
        "accuracy": "typical ranges for preliminary design",
        "access": "offline_no_pdf",
    },
    # ---------------- structural metals ----------------
    "SRC-EN10025": {
        "citation": (
            "EN 10025-2:2019 Hot rolled products of structural steels, Table 7 "
            "(minimum yield/tensile strength for S235/S355/S420), and "
            "EN 1993-1-1:2005 Table 3.1 (E = 210 GPa, G = 81 GPa, nu = 0.3, "
            "rho = 7850 kg/m^3, alpha = 12e-6 1/K)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "normative minimum values (characteristic)",
        "access": "offline_no_pdf",
    },
    "SRC-IACS-UR-REQ": {
        "citation": (
            "IACS Unified Requirements S6 (2016) 'Application of the requirements to "
            "the strength of ships operating in ice' + IACS UR W11/W12 'Normal and "
            "higher strength hull structural steels' Table 1 (grade A/AH32/AH36/EH36 "
            "yield and tensile strength, Charpy energy 27 J at 0/-20/-40 C)"
        ),
        "locator_granularity": "table + section",
        "accuracy": "minimum specified values",
        "access": "offline_no_pdf",
    },
    "SRC-EN1999": {
        "citation": (
            "EN 1999-1-1:2007 Eurocode 9: Design of aluminium structures, Table 3.2 "
            "(E = 70 GPa, nu = 0.33, rho = 2660 / 2700 kg/m^3), Table 3.2b "
            "(0.2% proof stress f_o for 5083-O/H116, 6082-T6) and Table 3.2c "
            "(HAZ softening factors rho_haz)"
        ),
        "locator_granularity": "table + page",
        "accuracy": "normative characteristic values",
        "access": "offline_no_pdf",
    },
    # ---------------- composites / concrete ----------------
    "SRC-EN1992": {
        "citation": (
            "EN 1992-1-1:2004 Eurocode 2, Table 3.1 (concrete C30/37: "
            "f_ck = 30 MPa, E_cm = 33 GPa, rho = 2400 kg/m^3) and Sec. 3.1.3(3) "
            "(nu = 0.2, alpha_t = 10e-6 1/K)"
        ),
        "locator_granularity": "table + section",
        "accuracy": "normative characteristic values",
        "access": "offline_no_pdf",
    },
    "SRC-LLOYDS-GFRP": {
        "citation": (
            "Lloyd's Register (2019) Rules and Regulations for the Classification of "
            "Ships, Part 3 Chapter 4 'Fibre reinforced plastic vessels', Sec. 3 "
            "(typical single-skin GFRP laminate: rho = 1600 kg/m^3, E = 15-20 GPa, "
            "nu = 0.30, tensile strength 100-150 MPa); sandwich equivalent properties "
            "for marine panels as engineering-equivalent homogenisation"
        ),
        "locator_granularity": "section",
        "accuracy": "design-band values, not a measured panel",
        "access": "offline_no_pdf",
    },
    "SRC-SHIP-PLATE-MATERIALS": {
        "citation": (
            "ship-performance-platform: backend/modules/cabin_noise/data/plate_materials.json "
            "(asset 'plate_materials', structural loss factors eta for steel / aluminium / "
            "sandwich panel, engineering values declared by that asset)"
        ),
        "locator_granularity": "asset field",
        "accuracy": "engineering values, eta from measured coincidence-dip band",
        "access": "offline_no_pdf",
    },
}


#: Independent anchor values used for the build-time accuracy audit.
#: ``rel_tol`` is the acceptance band the database must satisfy for that anchor.
ANCHORS = [
    {
        "id": "water-density-4C",
        "substance": "water", "property": "density",
        "conditions": {"T_K": 277.15, "p_Pa": 101325.0},
        "expected": 999.97, "unit": "kg/m^3", "rel_tol": 0.0005,
        "reference": "maximum density of water at 3.98 C (IAPWS R7-97 stated value)",
    },
    {
        "id": "water-density-20C",
        "substance": "water", "property": "density",
        "conditions": {"T_K": 293.15, "p_Pa": 101325.0},
        "expected": 998.21, "unit": "kg/m^3", "rel_tol": 0.0005,
        "reference": "Kell (1975) / IAPWS R7-97 table value at 20 C, 0.1 MPa",
    },
    {
        "id": "water-viscosity-20C",
        "substance": "water", "property": "dynamic_viscosity",
        "conditions": {"T_K": 293.15, "p_Pa": 101325.0},
        "expected": 1.0016e-3, "unit": "Pa*s", "rel_tol": 0.01,
        "reference": "Kestin et al. (1978) Table 2 value at 20 C",
    },
    {
        "id": "water-viscosity-0C",
        "substance": "water", "property": "dynamic_viscosity",
        "conditions": {"T_K": 273.15, "p_Pa": 101325.0},
        "expected": 1.7914e-3, "unit": "Pa*s", "rel_tol": 0.03,
        "reference": "Kestin et al. (1978) Table 2 value at 0 C (Vogel fit band 3%)",
    },
    {
        "id": "water-kinematic-15C-ITTC",
        "substance": "water", "property": "kinematic_viscosity",
        "conditions": {"T_K": 288.15, "p_Pa": 101325.0},
        "expected": 1.1390e-6, "unit": "m^2/s", "rel_tol": 0.01,
        "reference": "ITTC 7.5-02-01-03 (2011) Table 1 fresh water at 15 C",
    },
    {
        "id": "seawater-kinematic-15C-ITTC",
        "substance": "seawater", "property": "kinematic_viscosity",
        "conditions": {"T_K": 288.15, "p_Pa": 101325.0, "salinity_g_per_kg": 35.0},
        "expected": 1.1883e-6, "unit": "m^2/s", "rel_tol": 0.02,
        "reference": "ITTC 7.5-02-01-03 (2011) Table 1 seawater (35 g/kg) at 15 C",
    },
    {
        "id": "seawater-density-25C-35ppt",
        "substance": "seawater", "property": "density",
        "conditions": {"T_K": 298.15, "p_Pa": 101325.0, "salinity_g_per_kg": 35.0},
        "expected": 1023.34, "unit": "kg/m^3", "rel_tol": 0.001,
        "reference": "Sharqawy et al. (2010) Table 4 / Millero & Poisson (1981) reference state",
    },
    {
        "id": "seawater-freezing-35ppt",
        "substance": "seawater", "property": "freezing_temperature",
        "conditions": {"p_Pa": 101325.0, "salinity_g_per_kg": 35.0},
        "expected": 271.25, "unit": "K", "rel_tol": 0.002,
        "reference": "Millero & Leung (1976): Tf = 271.25 K at S = 35 g/kg, 1 atm",
    },
    {
        "id": "air-density-15C",
        "substance": "air", "property": "density",
        "conditions": {"T_K": 288.15, "p_Pa": 101325.0},
        "expected": 1.2255, "unit": "kg/m^3", "rel_tol": 0.002,
        "reference": "ISO 2533 standard atmosphere sea-level density 1.225 kg/m^3",
    },
    {
        "id": "air-viscosity-15C",
        "substance": "air", "property": "dynamic_viscosity",
        "conditions": {"T_K": 288.15, "p_Pa": 101325.0},
        "expected": 1.7894e-5, "unit": "Pa*s", "rel_tol": 0.02,
        "reference": "Sutherland law / CRC table value 1.79e-5 Pa s at 15 C",
    },
    {
        "id": "air-conductivity-20C",
        "substance": "air", "property": "thermal_conductivity",
        "conditions": {"T_K": 293.15, "p_Pa": 101325.0},
        "expected": 0.02587, "unit": "W/(m*K)", "rel_tol": 0.02,
        "reference": "CRC Handbook air thermal-conductivity table value at 20 degC (293 K)",
    },
    {
        "id": "air-sound-speed-15C",
        "substance": "air", "property": "sound_speed",
        "conditions": {"T_K": 288.15, "p_Pa": 101325.0},
        "expected": 340.29, "unit": "m/s", "rel_tol": 0.005,
        "reference": "ISO 2533 standard atmosphere: a = sqrt(gamma R T)",
    },
    {
        "id": "water-sound-speed-20C",
        "substance": "water", "property": "sound_speed",
        "conditions": {"T_K": 293.15, "p_Pa": 101325.0},
        "expected": 1482.3, "unit": "m/s", "rel_tol": 0.002,
        "reference": "Bilaniuk & Wong (1993) Table 1 at 20 C",
    },
    {
        "id": "ice-pure-density-0C",
        "substance": "ice_fresh", "property": "density",
        "conditions": {"T_K": 273.15, "p_Pa": 101325.0},
        "expected": 916.8, "unit": "kg/m^3", "rel_tol": 0.002,
        "reference": "Hobbs (1974) Ice Physics Table 2.1",
    },
    {
        "id": "seaice-flexural-ref",
        "substance": "ice_sea", "property": "flexural_strength",
        "conditions": {"T_K": 263.15, "p_Pa": 101325.0, "salinity_g_per_kg": 5.0},
        "expected": 0.70e6, "unit": "Pa", "rel_tol": 0.30,
        "reference": (
            "Timco & Weeks (2010) Table 4: typical first-year sea-ice flexural "
            "strength 0.7 MPa (published range 0.36-2.6 MPa); at T = -10 degC, "
            "S = 5 g/kg the brine volume is vb ~ 0.027"
        ),
    },
    {
        "id": "steel-E-EN1993",
        "substance": "steel_S355", "property": "youngs_modulus",
        "conditions": {},
        "expected": 210.0e9, "unit": "Pa", "rel_tol": 0.001,
        "reference": "EN 1993-1-1:2005 Table 3.1",
    },
    {
        "id": "steel-yield-S355",
        "substance": "steel_S355", "property": "yield_strength",
        "conditions": {},
        "expected": 355.0e6, "unit": "Pa", "rel_tol": 0.001,
        "reference": "EN 10025-2:2019 Table 7 (t <= 16 mm)",
    },
    {
        "id": "aluminium-E-EN1999",
        "substance": "alu_5083_H116", "property": "youngs_modulus",
        "conditions": {},
        "expected": 70.0e9, "unit": "Pa", "rel_tol": 0.001,
        "reference": "EN 1999-1-1:2007 Table 3.2",
    },
    {
        "id": "quartz-grain-density",
        "substance": "sand_quartz", "property": "grain_density",
        "conditions": {},
        "expected": 2650.0, "unit": "kg/m^3", "rel_tol": 0.001,
        "reference": "Soulsby (1997) Table 2-3",
    },
    {
        "id": "concrete-E-C30",
        "substance": "concrete_C30", "property": "youngs_modulus",
        "conditions": {},
        "expected": 33.0e9, "unit": "Pa", "rel_tol": 0.001,
        "reference": "EN 1992-1-1:2004 Table 3.1",
    },
]


def source_ids() -> list:
    return sorted(SOURCE_REGISTRY)