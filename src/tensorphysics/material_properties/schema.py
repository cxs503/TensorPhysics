"""Unified schema for the material/fluid property database (MPDB v1).

Record layout (one JSONL row per ``substance x conditions x property``)::

    substance -> class -> property -> value/unit -> conditions -> domain
              -> provenance (source_id, citation, table/page, method, evidence class)
              -> uncertainty

Design rules
------------
* **SI base units first.**  Every dimensional quantity is stored in the SI base
  unit listed in :data:`PROPERTY_UNITS`.  A small number of accepted non-SI
  access units (degrees Celsius, degrees of arc, g/kg salinity, ppt) are
  declared in :data:`NON_SI_ACCESS_UNITS` with their exact conversion factor;
  they never enter the record payload, only `conditions`/display helpers.
* **Provenance is mandatory.**  A row without ``provenance.source_id`` that
  resolves in the source registry, or without a page/table/section locator,
  is rejected by :func:`validate_record`.
* **Domain is explicit.**  ``domain`` gives the validity box of the *source*
  (not of the substance), so the query layer can refuse extrapolation instead
  of silently returning a number outside the correlation's range.

The schema is also exported as JSON Schema (``schema.to_json_schema()``) so the
data files can be validated by non-Python consumers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA_VERSION = "mpdb-1.0.0"

# --------------------------------------------------------------------------- #
# Property vocabulary
# --------------------------------------------------------------------------- #
# property -> (SI unit, physical dimension, description)
PROPERTY_UNITS: Dict[str, tuple] = {
    # transport / bulk fluid properties
    "density": ("kg/m^3", "M L^-3", "mass density"),
    "dynamic_viscosity": ("Pa*s", "M L^-1 T^-1", "shear (dynamic) viscosity"),
    "kinematic_viscosity": ("m^2/s", "L^2 T^-1", "kinematic viscosity nu = mu/rho"),
    "specific_heat": ("J/(kg*K)", "L^2 T^-2 Theta^-1", "specific heat capacity at constant pressure"),
    "thermal_conductivity": ("W/(m*K)", "M L T^-3 Theta^-1", "thermal conductivity"),
    "thermal_diffusivity": ("m^2/s", "L^2 T^-1", "alpha = k/(rho cp)"),
    "sound_speed": ("m/s", "L T^-1", "small-signal sound speed"),
    "prandtl_number": ("-", "1", "Prandtl number mu cp / k"),
    "latent_heat_fusion": ("J/kg", "L^2 T^-2", "latent heat of fusion"),
    "freezing_temperature": ("K", "Theta", "freezing/melting temperature"),
    "vapour_pressure": ("Pa", "M L^-1 T^-2", "saturation vapour pressure"),
    # solid mechanics
    "youngs_modulus": ("Pa", "M L^-1 T^-2", "elastic (Young's) modulus"),
    "poisson_ratio": ("-", "1", "Poisson's ratio"),
    "shear_modulus": ("Pa", "M L^-1 T^-2", "shear modulus"),
    "yield_strength": ("Pa", "M L^-1 T^-2", "0.2% proof / yield stress"),
    "ultimate_strength": ("Pa", "M L^-1 T^-2", "ultimate tensile strength"),
    "tensile_strength": ("Pa", "M L^-1 T^-2", "tensile strength"),
    "compressive_strength": ("Pa", "M L^-1 T^-2", "uniaxial compressive strength"),
    "flexural_strength": ("Pa", "M L^-1 T^-2", "flexural (bending) strength"),
    "shear_strength": ("Pa", "M L^-1 T^-2", "shear strength of ice / soil"),
    "fracture_toughness": ("Pa*m^0.5", "M L^-0.5 T^-2", "mode-I fracture toughness"),
    "thermal_expansion": ("1/K", "Theta^-1", "linear coefficient of thermal expansion"),
    "damping_loss_factor": ("-", "1", "structural loss factor eta"),
    "haz_strength_ratio": ("-", "1", "HAZ softened / parent yield ratio"),
    # granular / geotechnical
    "grain_density": ("kg/m^3", "M L^-3", "solid grain density"),
    "dry_bulk_density": ("kg/m^3", "M L^-3", "dry bulk density"),
    "submerged_unit_weight": ("N/m^3", "M L^-2 T^-2", "submerged (buoyant) unit weight"),
    "friction_angle": ("deg", "1", "internal friction angle (accepted non-SI access unit)"),
    "cohesion": ("Pa", "M L^-1 T^-2", "effective cohesion intercept"),
    "angle_of_repose": ("deg", "1", "angle of repose (accepted non-SI access unit)"),
    "median_grain_size": ("m", "L", "median grain diameter D50"),
    # sea-ice specific
    "bulk_density": ("kg/m^3", "M L^-3", "bulk (in-situ) density incl. brine/air"),
    "brine_volume": ("-", "1", "brine volume fraction v_b"),
    "salinity": ("g/kg", "1", "bulk salinity (accepted non-SI access unit, 1 g/kg = 1e-3)"),
    "ice_friction_coefficient": ("-", "1", "ice-ice / ice-structure friction coefficient"),
}

#: Accepted non-SI access units -> (SI equivalent, exact factor to SI)
NON_SI_ACCESS_UNITS: Dict[str, tuple] = {
    "degC": ("K", 1.0),          # offset unit, handled by T_K = T_C + 273.15
    "deg": ("rad", 0.017453292519943295),
    "g/kg": ("kg/kg", 1.0e-3),
    "ppt": ("kg/kg", 1.0e-3),
    "MPa": ("Pa", 1.0e6),
    "GPa": ("Pa", 1.0e9),
    "kN/m^3": ("N/m^3", 1.0e3),
    "cP": ("Pa*s", 1.0e-3),
    "cSt": ("m^2/s", 1.0e-6),
}

T0_K = 273.15

#: Recognised substance classes (>= 5 required by the asset spec)
SUBSTANCE_CLASSES: Dict[str, str] = {
    "liquid": "single-phase liquid (fresh water, seawater, ORC working fluid)",
    "gas": "single-phase gas (air, combustion/venting gas)",
    "ice": "fresh-water and sea ice (temperature/salinity dependent)",
    "sediment": "granular seabed material (sand/silt/clay/gravel)",
    "metal": "ship-structure metal alloy (steel, aluminium)",
    "composite": "FRP laminate / sandwich equivalent single layer",
    "cementitious": "concrete / grout (gravity-based structures)",
}

#: Condition axes accepted in ``conditions`` / ``domain`` (all SI)
CONDITION_AXES: Dict[str, tuple] = {
    "T_K": ("K", "absolute temperature"),
    "p_Pa": ("Pa", "absolute pressure"),
    "salinity_g_per_kg": ("g/kg", "bulk salinity (accepted non-SI access unit)"),
}

#: evidence classes used in ``provenance.evidence_class``
EVIDENCE_CLASSES = (
    "FORMULA_PRIMARY",   # value produced by an encoded published correlation
    "TABLE_PRIMARY",     # value transcribed from a cited table/figure
    "DERIVED",           # algebraic consequence of other rows (documented)
)

#: access modes recorded so that a reader can see whether the PDF was opened
ACCESS_MODES = ("offline_no_pdf", "pdf_opened", "api")


# --------------------------------------------------------------------------- #
# Record construction / validation
# --------------------------------------------------------------------------- #
def kelvin(t_celsius: float) -> float:
    """Convert a Celsius temperature to kelvin (schema-level helper)."""
    return float(t_celsius) + T0_K


def celsius(t_kelvin: float) -> float:
    return float(t_kelvin) - T0_K


def record_id(
    substance: str,
    prop: str,
    conditions: Mapping[str, Any],
    variant: Optional[str] = None,
) -> str:
    """Deterministic row id, e.g. ``seawater.density.T288.15K.S35``.

    ``variant`` disambiguates rows that share substance/property/conditions but
    describe different *states* of the same material (e.g. first-year vs
    multi-year sea ice, initial vs effective modulus).
    """
    parts = [substance, prop]
    if "T_K" in conditions and conditions["T_K"] is not None:
        parts.append(f"T{float(conditions['T_K']):.2f}K")
    if "p_Pa" in conditions and conditions["p_Pa"] is not None:
        parts.append(f"p{float(conditions['p_Pa']):.0f}Pa")
    if conditions.get("salinity_g_per_kg") is not None:
        parts.append(f"S{float(conditions['salinity_g_per_kg']):g}")
    if variant:
        parts.append(f"@{variant}")
    return ".".join(parts)


def make_record(
    *,
    substance: str,
    substance_class: str,
    prop: str,
    value: float,
    conditions: Mapping[str, Any],
    domain: Mapping[str, Sequence[float]],
    provenance: Mapping[str, Any],
    uncertainty_rel: Optional[float] = None,
    uncertainty_basis: Optional[str] = None,
    derived_from: Optional[str] = None,
    notes: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one validated MPDB record (raises ``ValueError`` on bad input)."""
    if prop not in PROPERTY_UNITS:
        raise ValueError(f"unknown property {prop!r}")
    unit = PROPERTY_UNITS[prop][0]
    cond = {ax: (None if ax not in conditions else conditions[ax]) for ax in CONDITION_AXES}
    rec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": record_id(substance, prop, cond, variant),
        "substance": substance,
        "substance_class": substance_class,
        "property": prop,
        "value": float(value),
        "unit": unit,
        "conditions": cond,
        "domain": {ax: list(v) for ax, v in domain.items()},
        "provenance": dict(provenance),
        "uncertainty": {
            "rel": None if uncertainty_rel is None else float(uncertainty_rel),
            "basis": uncertainty_basis,
        },
        "derived_from": derived_from,
        "notes": notes,
        "variant": variant,
    }
    errors = validate_record(rec)
    if errors:
        raise ValueError("invalid MPDB record: " + "; ".join(errors))
    return rec


def validate_record(rec: Mapping[str, Any], source_registry: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Return a list of schema violations (empty list = valid)."""
    errs: List[str] = []

    for key in (
        "schema_version",
        "id",
        "substance",
        "substance_class",
        "property",
        "value",
        "unit",
        "conditions",
        "domain",
        "provenance",
    ):
        if key not in rec:
            errs.append(f"missing key {key!r}")
    if errs:
        return errs

    if rec["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {rec['schema_version']!r} != {SCHEMA_VERSION!r}")

    prop = rec["property"]
    if prop not in PROPERTY_UNITS:
        errs.append(f"unknown property {prop!r}")
    elif rec["unit"] != PROPERTY_UNITS[prop][0]:
        errs.append(f"unit {rec['unit']!r} != SI base unit {PROPERTY_UNITS[prop][0]!r} for {prop}")

    if rec["substance_class"] not in SUBSTANCE_CLASSES:
        errs.append(f"unknown substance_class {rec['substance_class']!r}")

    if not isinstance(rec["value"], (int, float)) or isinstance(rec["value"], bool):
        errs.append("value must be a number")
    else:
        v = float(rec["value"])
        if v != v:  # NaN
            errs.append("value is NaN")
        elif v in (float("inf"), float("-inf")):
            errs.append("value is infinite")

    cond = rec["conditions"] or {}
    for ax in cond:
        if ax not in CONDITION_AXES:
            errs.append(f"unknown condition axis {ax!r}")
    if cond.get("T_K") is not None and not (0.0 < float(cond["T_K"]) < 3000.0):
        errs.append("T_K outside plausible range (0, 3000) K")
    if cond.get("p_Pa") is not None and float(cond["p_Pa"]) <= 0.0:
        errs.append("p_Pa must be positive")

    dom = rec["domain"] or {}
    if not dom:
        errs.append("domain must not be empty")
    for ax, box in dom.items():
        if ax not in CONDITION_AXES:
            errs.append(f"unknown domain axis {ax!r}")
        elif not (isinstance(box, (list, tuple)) and len(box) == 2 and float(box[0]) <= float(box[1])):
            errs.append(f"domain[{ax!r}] must be a [lo, hi] pair")

    variant = rec.get("variant")
    if variant is not None and not (isinstance(variant, str) and variant.strip()):
        errs.append("variant must be a non-empty string when present")

    prov = rec["provenance"] or {}
    for key in ("source_id", "citation", "locator", "method", "evidence_class", "access"):
        if not prov.get(key):
            errs.append(f"provenance.{key} is required (no provenance = no row)")
    if prov.get("evidence_class") and prov["evidence_class"] not in EVIDENCE_CLASSES:
        errs.append(f"bad evidence_class {prov['evidence_class']!r}")
    if prov.get("access") and prov["access"] not in ACCESS_MODES:
        errs.append(f"bad access mode {prov['access']!r}")
    if prov.get("evidence_class") == "FORMULA_PRIMARY" and not prov.get("formula_id"):
        errs.append("FORMULA_PRIMARY rows must name provenance.formula_id")
    if prov.get("evidence_class") == "DERIVED" and not rec.get("derived_from"):
        errs.append("DERIVED rows must set derived_from")
    if source_registry is not None and prov.get("source_id"):
        if prov["source_id"] not in source_registry:
            errs.append(f"provenance.source_id {prov['source_id']!r} not in source registry")

    return errs


def validate_records(
    records: Iterable[Mapping[str, Any]],
    source_registry: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Validate a whole table; ids must be unique."""
    errs: List[str] = []
    seen: Dict[str, int] = {}
    for i, rec in enumerate(records):
        for e in validate_record(rec, source_registry):
            errs.append(f"row[{i}] {rec.get('id', '?')}: {e}")
        rid = rec.get("id")
        if rid in seen:
            errs.append(f"duplicate id {rid!r} (rows {seen[rid]} and {i})")
        seen[rid] = i
    return errs


def to_json_schema() -> Dict[str, Any]:
    """JSON-Schema (draft-07, python-jsonschema compatible) for the row shape."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"https://tensorplatform.local/schemas/{SCHEMA_VERSION}.json",
        "title": "TensorPhysics material/fluid property record (MPDB v1)",
        "type": "object",
        "required": [
            "schema_version", "id", "substance", "substance_class", "property",
            "value", "unit", "conditions", "domain", "provenance", "uncertainty",
        ],
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "id": {"type": "string", "minLength": 3},
            "substance": {"type": "string", "minLength": 2},
            "substance_class": {"enum": sorted(SUBSTANCE_CLASSES)},
            "property": {"enum": sorted(PROPERTY_UNITS)},
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "conditions": {
                "type": "object",
                "properties": {
                    "T_K": {"type": ["number", "null"]},
                    "p_Pa": {"type": ["number", "null"]},
                    "salinity_g_per_kg": {"type": ["number", "null"]},
                },
                "additionalProperties": False,
            },
            "domain": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "array", "minItems": 2, "maxItems": 2,
                    "items": {"type": "number"},
                },
            },
            "provenance": {
                "type": "object",
                "required": ["source_id", "citation", "locator", "method", "evidence_class", "access"],
                "properties": {
                    "source_id": {"type": "string"},
                    "citation": {"type": "string"},
                    "locator": {"type": "string"},
                    "method": {"type": "string"},
                    "evidence_class": {"enum": list(EVIDENCE_CLASSES)},
                    "access": {"enum": list(ACCESS_MODES)},
                    "formula_id": {"type": ["string", "null"]},
                },
            },
            "variant": {"type": ["string", "null"]},
            "uncertainty": {
                "type": "object",
                "required": ["rel", "basis"],
                "properties": {
                    "rel": {"type": ["number", "null"]},
                    "basis": {"type": ["string", "null"]},
                },
            },
        },
    }