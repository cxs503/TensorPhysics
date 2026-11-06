#!/usr/bin/env python3
"""Deterministically (re)build the ice-navigation reference corpus.

Every number in ``assets/*.json`` is *computed here* from the expression printed
in ``formula`` and the documented reference condition -- nothing is hand-copied.
Re-running this script must reproduce byte-identical files and therefore
identical sha256 anchors (asserted by ``tests/test_refs_corpus.py``).

Usage
-----
    python benchmarks/ice_nav_refs/build_corpus.py            # rewrite in place
    python benchmarks/ice_nav_refs/build_corpus.py --check    # verify only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_SRC))

from tensorphysics.ice_nav_refs import (  # noqa: E402
    SCHEMA,
    asset_from_dict,
    canonical_json,
    sha256_digest,
    value_digest,
)

HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE / "assets"
MANIFEST_PATH = HERE / "MANIFEST.json"

G = 9.81

#: Reference ship/ice condition shared by the resistance assets.  A 150 m
#: double-acting-type polar supply vessel in level first-year ice.
SHIP = {"L_m": 150.0, "B_m": 20.0, "bow_angle_deg": 30.0}
ICE = {
    "h_m": 1.0,
    "sigma_c_MPa": 5.0,
    "sigma_f_MPa": 0.5,
    "rho_kg_m3": 900.0,
    "mu_ice_steel": 0.15,
    "E_GPa": 5.0,
    "nu": 0.3,
    "rho_water_kg_m3": 1025.0,
}
BRASH = {"h_m": 0.6, "B_m": 20.0, "V_m_s": 2.0, "phi_deg": 30.0}


# ---------------------------------------------------------------------------
# closed forms (independent, hand-written; cross-checked against the repo
# kernels in tests/test_refs_corpus.py)
# ---------------------------------------------------------------------------
def lindqvist_coefficients(
    B: float = SHIP["B_m"],
    phi_deg: float = SHIP["bow_angle_deg"],
    mu: float = ICE["mu_ice_steel"],
    V: float = 2.0,
    h: float = ICE["h_m"],
) -> Dict[str, float]:
    """Lindqvist (1989) coefficients ``alpha, C_c, C_f, C_s, Fr_h``.

    ``alpha = 90 deg - phi/2`` is the ice/bow contact angle; ``C_c`` is the
    crushing-list coefficient, ``C_f`` the buoyancy/submergence coefficient
    (length units, it already contains the breadth) and
    ``C_s = 0.15 + 0.3 Fr_h`` the speed coefficient with ``Fr_h = V/sqrt(g h)``.
    """
    alpha = math.radians(90.0 - phi_deg / 2.0)
    phi = math.radians(phi_deg)
    C_c = 0.5 * (math.tan(alpha) + mu / math.cos(alpha))
    C_f = 0.5 * B * (math.sin(phi / 2.0) + mu * math.cos(phi / 2.0))
    Fr_h = V / math.sqrt(G * h)
    C_s = 0.15 + 0.3 * Fr_h
    return {"alpha_deg": math.degrees(alpha), "C_c": C_c, "C_f_m": C_f, "Fr_h": Fr_h, "C_s": C_s}


def lindqvist_components(
    B: float = SHIP["B_m"],
    phi_deg: float = SHIP["bow_angle_deg"],
    mu: float = ICE["mu_ice_steel"],
    V: float = 2.0,
    h: float = ICE["h_m"],
    sigma_c_Pa: float = ICE["sigma_c_MPa"] * 1e6,
    rho: float = ICE["rho_kg_m3"],
) -> Dict[str, float]:
    """Lindqvist three-component resistance as transcribed in this repository.

    ``R_c = C_c sigma_c h^2``, ``R_f = C_f rho g h``, ``R_s = C_s rho V^2 h B``.
    """
    co = lindqvist_coefficients(B=B, phi_deg=phi_deg, mu=mu, V=V, h=h)
    R_c = co["C_c"] * sigma_c_Pa * h**2
    R_f = co["C_f_m"] * rho * G * h
    R_s = co["C_s"] * rho * V**2 * h * B
    out = dict(co)
    out.update({"R_c_N": R_c, "R_f_N": R_f, "R_s_N": R_s, "R_total_N": R_c + R_f + R_s})
    out["R_total_kN"] = out["R_total_N"] / 1000.0
    return out


def keinonen_resistance(
    h: float = BRASH["h_m"],
    B: float = BRASH["B_m"],
    phi_deg: float = BRASH["phi_deg"],
    V: float = BRASH["V_m_s"],
    sigma_f_Pa: float = ICE["sigma_f_MPa"] * 1e6,
    C0: float = 1.8,
    C1: float = 0.25,
) -> Dict[str, float]:
    """Keinonen semi-empirical resistance ``R = C0 sf h^1.5 B^0.5 (1 + C1 Fr_h) f(phi)``.

    ``f(phi) = sin(phi/2) + 0.3 cos(phi/2)``.
    """
    phi = math.radians(phi_deg)
    f_phi = math.sin(phi / 2.0) + 0.3 * math.cos(phi / 2.0)
    Fr_h = V / math.sqrt(G * h)
    R = C0 * sigma_f_Pa * h**1.5 * B**0.5 * (1.0 + C1 * Fr_h) * f_phi
    return {"f_phi": f_phi, "Fr_h": Fr_h, "R_N": R, "R_kN": R / 1000.0}


def pressure_area(A_m2: float, p0_MPa: float, exponent: float) -> float:
    """Power-law ice pressure ``p = p0 A^-ex`` [MPa]."""
    return p0_MPa * (A_m2**-exponent)


def global_pressure(h_m: float, p_ref_MPa: float = 1.2, h_ref_m: float = 1.0, exponent: float = -0.5) -> float:
    """Whole-ship ice pressure vs thickness ``p_G = p_ref (h/h_ref)^-0.5`` [MPa]."""
    return p_ref_MPa * (h_m / h_ref_m) ** exponent


# ---------------------------------------------------------------------------
# asset definitions
# ---------------------------------------------------------------------------
LEVEL_REGIME: Dict[str, Any] = {
    "ice_regime": "level_first_year_ice",
    "ice_thickness_m": [0.3, 1.5],
    "bow_angle_deg": [15.0, 60.0],
    "speed_m_s": [0.5, 6.0],
    "ship_length_m": [50.0, 250.0],
    "friction_coefficient": [0.05, 0.20],
}

BRASH_REGIME: Dict[str, Any] = {
    "ice_regime": "brash_ice_channel",
    "ice_thickness_m": [0.3, 1.2],
    "bow_angle_deg": [20.0, 45.0],
    "speed_m_s": [0.5, 5.0],
    "ship_length_m": [50.0, 250.0],
}

CONTACT_REGIME: Dict[str, Any] = {
    "ice_regime": "level_first_year_ice_contact",
    "ice_thickness_m": [0.3, 2.0],
    "contact_area_m2": [0.25, 10.0],
    "polar_class": ["PC4", "PC7"],
}

PC6_REGIME: Dict[str, Any] = {
    "ice_regime": "level_first_year_ice_polar_class",
    "ice_thickness_m": [0.8, 1.5],
    "polar_class": ["PC6"],
    "panel_area_m2": [0.25, 8.0],
    "location": "ice_belt_shell_plating",
}

POLAR_CODE_REGIME: Dict[str, Any] = {
    "ice_regime": "any_polar_operation",
    "ice_thickness_m": [0.0, 3.0],
    "polar_ship_category": ["I-A"],
    "latitude_band_deg": [60.0, 90.0],
}


def build_assets() -> Dict[str, List[Dict[str, Any]]]:
    """Return ``{family: [asset, ...]}`` with all numbers computed."""
    lin = lindqvist_components()
    kei = keinonen_resistance()
    kei_co = keinonen_resistance(h=BRASH["h_m"], B=BRASH["B_m"], V=BRASH["V_m_s"])

    families: Dict[str, List[Dict[str, Any]]] = {}

    # ---------------- A. Lindqvist (1989) level-ice resistance -------------
    families["lindqvist_1989"] = [
        {
            "asset_id": "LIN_C_CRUSH_PHI30",
            "family": "lindqvist_1989",
            "source_citation": (
                "Lindqvist, G. (1989) 'A straightforward method for calculation of ice resistance "
                "of ships', POAC'89, Lulea, Vol. 2, pp. 722-735 -- crushing-list coefficient "
                "C_c = 0.5[tan(90-phi/2) + mu/cos(90-phi/2)]"
            ),
            "formula": "C_c = 0.5 * (tan(alpha) + mu/cos(alpha)), alpha = 90deg - phi/2",
            "values": {"C_c": lin["C_c"], "alpha_deg": lin["alpha_deg"]},
            "units": {"C_c": "1", "alpha_deg": "deg"},
            "tolerance": {
                "mode": "relative",
                "value": 1e-3,
                "note": "closed-form evaluation of the cited expression; tolerance covers "
                        "only rounding of the stored number",
            },
            "applicable_regime": dict(LEVEL_REGIME),
            "evidence_tier": "published_formula",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Expression re-derived symbolically and dimension-checked (dimensionless); the "
                "primary POAC'89 paper could not be opened in the build environment, so the "
                "coefficient form is taken from the repository transcription "
                "(ice_anchor/anchors.py::lindqvist_1989) and from the standard quotation of "
                "Lindqvist's method in the ice-resistance literature."
            ),
            "reference_condition": {"bow_angle_deg": SHIP["bow_angle_deg"], "mu": ICE["mu_ice_steel"]},
        },
        {
            "asset_id": "LIN_C_SPEED_FR",
            "family": "lindqvist_1989",
            "source_citation": (
                "Lindqvist, G. (1989) POAC'89, Vol. 2, pp. 722-735 -- speed coefficient "
                "C_s = 0.15 + 0.3*Fr_h with Froude number Fr_h = V/sqrt(g h)"
            ),
            "formula": "C_s = 0.15 + 0.3*(V/sqrt(g*h))",
            "values": {"Fr_h": lin["Fr_h"], "C_s": lin["C_s"]},
            "units": {"Fr_h": "1", "C_s": "1"},
            "tolerance": {
                "mode": "relative",
                "value": 1e-3,
                "note": "arithmetic evaluation at the documented reference condition",
            },
            "applicable_regime": dict(LEVEL_REGIME),
            "evidence_tier": "published_formula",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Dimensionless by construction (V/sqrt(g h)); coefficient pair (0.15, 0.3) taken "
                "from the repository transcription of Lindqvist (1989), primary paper not "
                "accessible for direct transcription."
            ),
            "reference_condition": {"V_m_s": 2.0, "h_m": ICE["h_m"]},
        },
        {
            "asset_id": "LIN_RESISTANCE_COMPONENTS",
            "family": "lindqvist_1989",
            "source_citation": (
                "Lindqvist, G. (1989) POAC'89, Vol. 2, pp. 722-735 (components R_c, R_f, R_s) as "
                "transcribed in TensorPhysics ice_anchor/anchors.py::lindqvist_1989"
            ),
            "formula": "R_c = C_c*s_c*h^2 ; R_f = C_f*rho*g*h ; R_s = C_s*rho*V^2*h*B",
            "values": {
                "C_c": lin["C_c"],
                "C_f_m": lin["C_f_m"],
                "C_s": lin["C_s"],
                "R_c_N": lin["R_c_N"],
                "R_f_N": lin["R_f_N"],
                "R_s_N": lin["R_s_N"],
                "R_total_kN": lin["R_total_kN"],
            },
            "units": {
                "C_c": "1",
                "C_f_m": "m",
                "C_s": "1",
                "R_c_N": "N",
                "R_f_N": "N",
                "R_s_N": "N",
                "R_total_kN": "kN",
            },
            "tolerance": {
                "mode": "relative",
                "value": 0.05,
                "note": "regression drift budget for the repository transcription; NOT an "
                        "accuracy claim against ice-tank data",
            },
            "applicable_regime": dict(LEVEL_REGIME),
            "evidence_tier": "repo_transcription",
            "role": "regression_anchor",
            "primary_source_verified": False,
            "verification_note": (
                "Frozen value of this repository's documented transcription; the crushing and "
                "buoyancy terms as transcribed carry no explicit breadth factor in R_c and a "
                "breadth-bearing coefficient C_f in R_f, i.e. the component set is not "
                "dimensionally homogeneous across breadth. Recorded as a regression anchor only: "
                "do not quote R_total_kN as validated absolute resistance. See "
                "docs/ICE_NAV_REFS_CORPUS.md, section 'Known defects caught by this corpus'."
            ),
            "known_defects": [
                "R_c omits the ship-breadth factor (force scales as sigma_c*h^2 only)",
                "R_f is C_f*rho*g*h with C_f containing B, so R_f is dimensionally a force per "
                "unit breadth times B/h-dependent factor -- breadth handling differs from R_s",
            ],
            "reference_condition": {
                "L_m": SHIP["L_m"],
                "B_m": SHIP["B_m"],
                "bow_angle_deg": SHIP["bow_angle_deg"],
                "h_m": ICE["h_m"],
                "V_m_s": 2.0,
                "sigma_c_MPa": ICE["sigma_c_MPa"],
                "rho_kg_m3": ICE["rho_kg_m3"],
                "mu": ICE["mu_ice_steel"],
            },
        },
        {
            "asset_id": "LIN_ICE_STEEL_FRICTION_BAND",
            "family": "lindqvist_1989",
            "source_citation": (
                "Lindqvist (1989) POAC'89 pp. 722-735 and ISO 19906 (2010) Annex A friction "
                "guidance -- ice/steel kinetic friction coefficient in the 0.05-0.20 band, "
                "0.15 used as the design mid-value"
            ),
            "values": {"mu_nominal": 0.15, "mu_lower": 0.05, "mu_upper": 0.20},
            "units": {"mu_nominal": "1", "mu_lower": "1", "mu_upper": "1"},
            "tolerance": {
                "mode": "band",
                "value": 0.15,
                "lower": 0.05,
                "upper": 0.20,
                "note": "band width of the published ice/steel friction scatter",
            },
            "applicable_regime": {
                **LEVEL_REGIME,
                "ice_thickness_m": [0.2, 2.0],
                "surface": "painted_steel",
            },
            "evidence_tier": "literature_band",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Band transcribed from the engineering consensus for ice-on-steel friction "
                "(0.05-0.20, temperature and sliding-speed dependent); the exact table of the "
                "cited standards was not accessible, only the band as quoted in the literature."
            ),
        },
    ]

    # ---------------- B. Keinonen brash-ice resistance coefficients --------
    families["keinonen_brash"] = [
        {
            "asset_id": "KEI_COEFFICIENTS",
            "family": "keinonen_brash",
            "source_citation": (
                "Keinonen, A. (1983) 'Icebreaking model testing -- a review of recent "
                "developments', POAC'83, Helsinki, pp. 429-444; coefficient family restated in "
                "Keinonen (1996) ice-resistance formulations for brash/fragmented ice"
            ),
            "formula": "R = C0*sigma_f*h^1.5*B^0.5*(1 + C1*Fr_h)*f(phi)",
            "values": {"C0": 1.8, "C1": 0.25},
            "units": {"C0": "1", "C1": "1"},
            "tolerance": {
                "mode": "relative",
                "value": 0.02,
                "note": "transcription tolerance for the published coefficient pair",
            },
            "applicable_regime": dict(BRASH_REGIME),
            "evidence_tier": "repo_transcription",
            "role": "regression_anchor",
            "primary_source_verified": False,
            "verification_note": (
                "Coefficient pair taken from the repository transcription of Keinonen (1983). "
                "The 1996 consolidated source named in the campaign brief was not accessible in "
                "the build environment, so the 1996 attribution of this coefficient set remains "
                "unverified; treat C0/C1 as a regression anchor until a primary source is read."
            ),
        },
        {
            "asset_id": "KEI_RESISTANCE_REF",
            "family": "keinonen_brash",
            "source_citation": (
                "Keinonen (1983) POAC'83 pp. 429-444 evaluation at a Baltic brash-ice reference "
                "condition (h = 0.6 m, B = 20 m, V = 2 m/s, bow angle 30 deg)"
            ),
            "formula": "R = C0*sigma_f*h^1.5*B^0.5*(1 + C1*Fr_h)*(sin(phi/2) + 0.3cos(phi/2))",
            "values": {"f_phi": kei["f_phi"], "Fr_h": kei["Fr_h"], "R_kN": kei["R_kN"]},
            "units": {"f_phi": "1", "Fr_h": "1", "R_kN": "kN"},
            "tolerance": {
                "mode": "relative",
                "value": 0.10,
                "note": "regression drift budget for the semi-empirical evaluation",
            },
            "applicable_regime": dict(BRASH_REGIME),
            "evidence_tier": "repo_transcription",
            "role": "regression_anchor",
            "primary_source_verified": False,
            "verification_note": (
                "Value evaluated here from the transcribed closed form, not measured values from "
                "a brash-ice model test; absolute accuracy against Keinonen's tank data is not "
                "established by this repository."
            ),
            "reference_condition": {
                "h_m": BRASH["h_m"],
                "B_m": BRASH["B_m"],
                "V_m_s": BRASH["V_m_s"],
                "bow_angle_deg": BRASH["phi_deg"],
                "sigma_f_MPa": ICE["sigma_f_MPa"],
            },
        },
        {
            "asset_id": "KEI_BRASH_COEFF_BAND",
            "family": "keinonen_brash",
            "source_citation": (
                "Keinonen (1983) POAC'83 pp. 429-444 semi-empirical coefficient scatter for "
                "fragmented/brash ice; recorded as the uncertainty band of C0 and C1"
            ),
            "values": {"C0_nominal": 1.8, "C0_lower": 1.5, "C0_upper": 2.2, "C1_lower": 0.15, "C1_upper": 0.35},
            "units": {"C0_nominal": "1", "C0_lower": "1", "C0_upper": "1", "C1_lower": "1", "C1_upper": "1"},
            "tolerance": {
                "mode": "relative",
                "value": 0.30,
                "note": "coefficient-band half-width; brash ice is the least repeatable ice regime",
            },
            "applicable_regime": {
                **BRASH_REGIME,
                "ice_thickness_m": [0.3, 1.0],
                "channel_type": "repeated_navigation_channel",
            },
            "evidence_tier": "literature_band",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Bands are the engineering uncertainty of the semi-empirical coefficient set for "
                "brash ice, not a digitised table: no single published value exists for the "
                "fragmented regime. Wide by construction; use only for plausibility windows."
            ),
        },
    ]

    # ---------------- C. Riska / Kujala ice pressure-area -----------------
    areas = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    p0_local = 3.0
    ex = 0.45
    families["riska_kujala_pressure_area"] = [
        {
            "asset_id": "RA_PRESSURE_AREA_EXPONENT",
            "family": "riska_kujala_pressure_area",
            "source_citation": (
                "ISO 19906 (2010) Annex A pressure-area relation p = p0*A^-ex with ex = 0.45 for "
                "first-year ice, as used by the Riska/Kujala pressure-area family and transcribed "
                "in TensorPhysics ice_plasticity/iso19906.py"
            ),
            "values": {"ex_first_year": ex, "ex_lower": 0.40, "ex_upper": 0.55},
            "units": {"ex_first_year": "1", "ex_lower": "1", "ex_upper": "1"},
            "tolerance": {
                "mode": "band",
                "value": 0.15,
                "lower": 0.40,
                "upper": 0.55,
                "note": "exponent band quoted for first-year ice local pressure",
            },
            "applicable_regime": dict(CONTACT_REGIME),
            "evidence_tier": "standards_clause",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Exponent 0.45 is the value used in this repository and in the ISO 19906-derived "
                "pressure-area practice; the ISO document itself is paywalled and was not opened, "
                "so the band 0.40-0.55 is the literature scatter, not a verbatim table."
            ),
        },
        {
            "asset_id": "RA_LOCAL_PRESSURE_TABLE",
            "family": "riska_kujala_pressure_area",
            "source_citation": (
                "Riska/Kujala pressure-area family p = p0*A^-0.45 (ISO 19906 Annex A form) with "
                "reference pressure p0 = 3.0 MPa at A = 1 m^2 for first-year ice"
            ),
            "formula": "p(A) = 3.0 MPa * A^-0.45",
            "values": {f"p_A{a:g}_MPa": pressure_area(a, p0_local, ex) for a in areas},
            "units": {f"p_A{a:g}_MPa": "MPa" for a in areas},
            "tolerance": {
                "mode": "relative",
                "value": 0.20,
                "note": "covers the p0 reference-pressure band (2.4-3.6 MPa) and the exponent band",
            },
            "applicable_regime": dict(CONTACT_REGIME),
            "evidence_tier": "derived_from_rule_form",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Power law is the cited rule form; the reference pressure p0 = 3.0 MPa is the "
                "nominal first-year local pressure used in this repository "
                "(ice_plasticity/iso19906.py::typical_ice_belt_params), i.e. a nominal engineering "
                "value with a 2.4-3.6 MPa band, not a measured datum."
            ),
            "reference_condition": {"p0_MPa": p0_local, "exponent": ex, "p0_band_MPa": [2.4, 3.6]},
        },
        {
            "asset_id": "RA_GLOBAL_PRESSURE_TABLE",
            "family": "riska_kujala_pressure_area",
            "source_citation": (
                "ISO 19906 / IACS Polar Class global ice pressure decreasing with contact area: "
                "p_G = p_ref*(h/h_ref)^-0.5 with p_ref = 1.2 MPa at h_ref = 1.0 m, as implemented "
                "in TensorPhysics hullgirder/ice_load.py::global_ice_pressure"
            ),
            "formula": "p_G(h) = 1.2 MPa * (h/1.0 m)^-0.5",
            "values": {f"pG_h{h:g}_MPa": global_pressure(h) for h in (0.5, 1.0, 1.5, 2.0, 3.0)},
            "units": {f"pG_h{h:g}_MPa": "MPa" for h in (0.5, 1.0, 1.5, 2.0, 3.0)},
            "tolerance": {
                "mode": "relative",
                "value": 0.15,
                "note": "rule-form pressure used as an order-of-magnitude global level",
            },
            "applicable_regime": {
                **CONTACT_REGIME,
                "ice_thickness_m": [0.3, 3.0],
                "contact_area_m2": [5.0, 200.0],
            },
            "evidence_tier": "derived_from_rule_form",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Anchored to the ISO 19906 / Polar Class global-pressure level as transcribed in "
                "this repository (p_ref = 1.2 MPa at 1 m); the standard clause was not read during "
                "this build, so both the 1.2 MPa level and the -0.5 thickness exponent are "
                "repo-transcribed values with rule-level (not experimental) authority."
            ),
            "reference_condition": {"p_ref_MPa": 1.2, "h_ref_m": 1.0, "exponent": -0.5},
        },
    ]

    # ---------------- D. IACS UR I PC6 local panel design pressure --------
    pc6_areas = (0.5, 1.0, 2.0, 4.0, 8.0)
    p0_pc6 = 2.5
    families["iacs_ur_i_pc6"] = [
        {
            "asset_id": "IACS_PC6_PANEL_PRESSURE_TABLE",
            "family": "iacs_ur_i_pc6",
            "source_citation": (
                "IACS Unified Requirement I2 (Polar Class hull structures, 2016) local ice "
                "pressure on shell plating p = p0*A^-0.45 applied at PC6 with nominal reference "
                "pressure p0 = 2.5 MPa at A = 1 m^2"
            ),
            "formula": "p(A) = 2.5 MPa * A^-0.45 over ice-belt panel areas 0.5-8 m^2",
            "values": {f"p_A{a:g}_MPa": pressure_area(a, p0_pc6, ex) for a in pc6_areas},
            "units": {f"p_A{a:g}_MPa": "MPa" for a in pc6_areas},
            "tolerance": {
                "mode": "relative",
                "value": 0.25,
                "note": "p0 for PC6 recorded as the band 2.0-3.0 MPa; 25% covers the band plus "
                        "the exponent scatter",
            },
            "applicable_regime": dict(PC6_REGIME),
            "evidence_tier": "derived_from_rule_form",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "The UR I2 pressure-area form is public and reproduced here; the PC6 reference "
                "pressure magnitude is given as a band (2.0-3.0 MPa) because the IACS document "
                "could not be downloaded in the build environment and the class-dependent "
                "reference pressure was therefore not transcribed verbatim. This asset must be "
                "re-verified against UR I2 before being used as a scantling-level datum."
            ),
            "reference_condition": {"p0_MPa": p0_pc6, "p0_band_MPa": [2.0, 3.0], "exponent": ex},
        },
        {
            "asset_id": "IACS_PC6_PATCH_GEOMETRY",
            "family": "iacs_ur_i_pc6",
            "source_citation": (
                "IACS UR I ice-load patch geometry as transcribed in TensorPhysics "
                "plate_buckling/loads.py and docs/papers/W3-N03/report.md: patch height = "
                "0.6*h_ice, patch aspect ratio 4:1"
            ),
            "values": {"patch_height_factor": 0.6, "patch_aspect_ratio": 4.0},
            "units": {"patch_height_factor": "1", "patch_aspect_ratio": "1"},
            "tolerance": {
                "mode": "relative",
                "value": 0.02,
                "note": "geometric ratio transcribed from the rule; drift budget only",
            },
            "applicable_regime": dict(PC6_REGIME),
            "evidence_tier": "repo_transcription",
            "role": "regression_anchor",
            "primary_source_verified": False,
            "verification_note": (
                "Patch geometry as already transcribed and used by the plate-buckling line of "
                "this repository; the UR I clause text was not re-read for this build."
            ),
        },
        {
            "asset_id": "IACS_PC6_DESIGN_ICE_THICKNESS",
            "family": "iacs_ur_i_pc6",
            "source_citation": (
                "IACS Polar Class design ice thickness for PC6 (UR I1 Polar Class descriptions): "
                "h = 1.2 m nominal, recorded as the band 1.0-1.4 m"
            ),
            "values": {"h_nominal_m": 1.2, "h_lower_m": 1.0, "h_upper_m": 1.4},
            "units": {"h_nominal_m": "m", "h_lower_m": "m", "h_upper_m": "m"},
            "tolerance": {
                "mode": "band",
                "value": 0.4,
                "lower": 1.0,
                "upper": 1.4,
                "note": "band of the design-thickness value quoted for PC6 in secondary sources",
            },
            "applicable_regime": dict(PC6_REGIME),
            "evidence_tier": "literature_band",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "PC6 design thickness is quoted as a band because the primary UR I1 table was "
                "not available; secondary sources give 1.0-1.4 m with 1.2 m as the usual nominal. "
                "Do not use as a verbatim class-rule value."
            ),
        },
    ]

    # ---------------- E. Polar Code I-A ------------------------------------
    families["polar_code_i_a"] = [
        {
            "asset_id": "PC_CATEGORY_TO_POLAR_CLASS",
            "family": "polar_code_i_a",
            "source_citation": (
                "IMO Polar Code (resolution MSC.385(94)), Part I-A, ship categories: A = at least "
                "medium first-year ice (PC1-PC5 or equivalent), B = at least thin first-year ice "
                "(PC6-PC7 or equivalent), C = open water or less severe ice conditions"
            ),
            "values": {
                "category_A_polar_classes": "PC1-PC5",
                "category_B_polar_classes": "PC6-PC7",
                "category_C_ice_strengthening": "none required or equivalent light ice class",
            },
            "units": {
                "category_A_polar_classes": "text",
                "category_B_polar_classes": "text",
                "category_C_ice_strengthening": "text",
            },
            "tolerance": {"mode": "exact", "value": 0.0, "note": "categorical requirement"},
            "applicable_regime": dict(POLAR_CODE_REGIME),
            "evidence_tier": "standards_clause",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "Category definitions and the category-to-Polar-Class mapping are the widely "
                "reproduced content of Polar Code Part I-A; the IMO resolution text itself could "
                "not be retrieved in the build environment, so the mapping is high-confidence but "
                "not primary-verified."
            ),
        },
        {
            "asset_id": "PC_I_A_PST_BAND",
            "family": "polar_code_i_a",
            "source_citation": (
                "IMO Polar Code Part I-A section 3 (polar service temperature, PST, derived from "
                "the lowest mean daily low temperature of the intended area and season): "
                "representative Arctic PST -30 degC within the -40..-20 degC band"
            ),
            "values": {"pst_nominal_C": -30.0, "pst_lower_C": -40.0, "pst_upper_C": -20.0},
            "units": {"pst_nominal_C": "degC", "pst_lower_C": "degC", "pst_upper_C": "degC"},
            "tolerance": {
                "mode": "band",
                "value": 20.0,
                "lower": -40.0,
                "upper": -20.0,
                "note": "PST is area/season specific; -30 degC is a representative Arctic value",
            },
            "applicable_regime": dict(POLAR_CODE_REGIME),
            "evidence_tier": "literature_band",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "The PST definition (LMDLT of intended area and season) is the Polar Code content; "
                "the -30 degC value is a representative engineering value for Arctic seasonal "
                "operation, not a clause verbatim. Use the band for equipment/material ratings."
            ),
        },
        {
            "asset_id": "PC_I_A_CATEGORY_THICKNESS",
            "family": "polar_code_i_a",
            "source_citation": (
                "IMO Polar Code Part I-A categories mapped onto WMO sea-ice thickness nomenclature: "
                "medium first-year ice 0.7-1.2 m, thin first-year ice 0.3-0.7 m, category C up to "
                "0.3 m"
            ),
            "values": {
                "A_min_m": 0.7,
                "A_max_m": 1.2,
                "B_min_m": 0.3,
                "B_max_m": 0.7,
                "C_max_m": 0.3,
            },
            "units": {k: "m" for k in ("A_min_m", "A_max_m", "B_min_m", "B_max_m", "C_max_m")},
            "tolerance": {
                "mode": "relative",
                "value": 0.15,
                "note": "nomenclature band edges; 15% covers the WMO definition rounding",
            },
            "applicable_regime": dict(POLAR_CODE_REGIME),
            "evidence_tier": "standards_clause",
            "role": "accuracy_reference",
            "primary_source_verified": False,
            "verification_note": (
                "WMO ice-thickness class boundaries are quoted from the sea-ice nomenclature that "
                "the Polar Code category descriptions refer to; the WMO document was not opened "
                "during this build, so the numbers carry standard-reported, not verbatim, status."
            ),
        },
    ]
    return families


def _prepare(families: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Add the value digest, validate every asset, and sort deterministically."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for family in sorted(families):
        prepared = []
        for payload in families[family]:
            record = dict(payload)
            record["schema"] = SCHEMA
            record["value_sha256"] = value_digest(
                record["asset_id"], record["values"], record["tolerance"]
            )
            # builder validates its own output through the public loader contract
            asset_from_dict(record)
            prepared.append(record)
        out[family] = sorted(prepared, key=lambda r: r["asset_id"])
    return out


def build(root: Path | str | None = None) -> Dict[str, Any]:
    """Write the asset files and MANIFEST.json; return the manifest."""
    root_path = Path(root) if root is not None else HERE
    asset_dir = root_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    families = _prepare(build_assets())
    files: Dict[str, Dict[str, Any]] = {}
    for family, records in families.items():
        rel = f"assets/{family}.json"
        document = {"schema": SCHEMA, "family": family, "assets": records}
        payload = canonical_json(document)
        (root_path / rel).write_bytes(payload)
        files[rel] = {
            "sha256": sha256_digest(payload),
            "bytes": len(payload),
            "family": family,
            "asset_ids": [r["asset_id"] for r in records],
        }

    aggregate = sha256_digest(
        "\n".join(f"{rel}:{files[rel]['sha256']}" for rel in sorted(files)).encode("utf-8")
    )
    manifest = {
        "schema": SCHEMA,
        "generated_by": "benchmarks/ice_nav_refs/build_corpus.py",
        "description": (
            "sha256-anchored reference-value corpus for the ice-navigation performance chain "
            "(level-ice resistance components, brash-ice coefficients, ice pressure-area, "
            "Polar Class local panel pressure, Polar Code I-A requirements)."
        ),
        "asset_count": sum(len(v) for v in families.values()),
        "family_count": len(families),
        "families": {fam: len(recs) for fam, recs in sorted(families.items())},
        "files": files,
        "aggregate_sha256": aggregate,
        "aggregate_rule": "sha256 over the sorted 'relative_path:sha256' lines of 'files'",
        "reference_condition_global": {"ship": SHIP, "ice": ICE},
    }
    (root_path / "MANIFEST.json").write_bytes(canonical_json(manifest))
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(HERE), help="corpus directory (default: script directory)")
    parser.add_argument("--check", action="store_true", help="do not write; verify the current files")
    args = parser.parse_args(argv)

    root_path = Path(args.root)
    if args.check:
        from tensorphysics.ice_nav_refs import load_corpus, verify_manifest

        report = verify_manifest(root_path)
        corpus = load_corpus(root_path)
        print(json.dumps({"manifest_ok": report["ok"], "assets": len(corpus),
                          "aggregate_sha256": report["aggregate_sha256"]}, indent=2))
        return 0 if report["ok"] else 1

    manifest = build(root_path)
    print(json.dumps({"asset_count": manifest["asset_count"],
                      "families": manifest["families"],
                      "aggregate_sha256": manifest["aggregate_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())