"""Deterministic builder for MPDB v1 (material/fluid property database).

One-shot reproducer::

    python scripts/build_material_properties_db.py              # rebuild in place
    python scripts/build_material_properties_db.py --check      # verify committed DB

Outputs (``src/tensorphysics/material_properties/data/``):

* ``mpdb_v1.jsonl``           one JSON object per (substance, conditions, property)
* ``mpdb_v1.substances.json`` substance catalogue
* ``mpdb_v1.anchors.json``    independent-anchor accuracy audit (expected / computed / rel err / pass)
* ``mpdb_v1.meta.json``       counts, class coverage, sha256 of the jsonl, repro_cmd
* ``mpdb_v1.schema.json``     JSON Schema of a row

No value is typed twice: fluid rows come from :mod:`correlations` calls, table
rows are declared once in :mod:`tables` with their locator.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import correlations as corr
from .schema import (
    PROPERTY_UNITS,
    SCHEMA_VERSION,
    SUBSTANCE_CLASSES,
    make_record,
    record_id,
    to_json_schema,
    validate_records,
)
from .tables import (
    COMPOSITE_TABLE,
    CONCRETE_TABLE,
    ICE_FRESH_DENSITY_T_C,
    ICE_FRESH_TABLE,
    ICE_SEA_T_C,
    ICE_SEA_TABLE,
    METAL_TABLE,
    SEDIMENT_TABLE,
    SUBSTANCES,
)
from .sources import ANCHORS, SOURCE_REGISTRY

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_NAME = "mpdb_v1"
REPRO_CMD = "python scripts/build_material_properties_db.py"
P_ATM = corr.P_ATM
T0 = corr.T0_K


def _rid(substance: str, prop: str, T: Optional[float] = None, p: Optional[float] = None,
         S: Optional[float] = None) -> str:
    return record_id(substance, prop, {"T_K": T, "p_Pa": p, "salinity_g_per_kg": S})


def _prov(source_id: str, locator: str, method: str, evidence_class: str,
          formula_id: Optional[str] = None) -> dict:
    src = SOURCE_REGISTRY[source_id]
    return {
        "source_id": source_id,
        "citation": src["citation"],
        "locator": locator,
        "method": method,
        "evidence_class": evidence_class,
        "access": src["access"],
        "formula_id": formula_id,
    }


# --------------------------------------------------------------------------- #
# Fluid builders
# --------------------------------------------------------------------------- #
def water_records() -> List[dict]:
    recs: List[dict] = []
    dom = {"T_K": [273.15, 373.15], "p_Pa": [101325.0, 101325.0]}
    for i in range(0, 201):                      # 0 .. 100 degC at 0.5 K
        T = T0 + 0.5 * i
        dens = corr.water_density(T)
        mu = corr.water_dynamic_viscosity(T)
        cond = {"T_K": T, "p_Pa": P_ATM}
        recs.append(make_record(
            substance="water", substance_class="liquid", prop="density", value=dens,
            conditions=cond, domain=dom,
            provenance=_prov("SRC-IAPWS-WATER-DENSITY", "eq. (2), p. 305 (IAPWS R7-97 isobar)",
                             "correlation", "FORMULA_PRIMARY", "kell1975_density"),
            uncertainty_rel=1e-5,
            uncertainty_basis="correlation accuracy <= 1e-5 on the 0.1 MPa isobar"))
        recs.append(make_record(
            substance="water", substance_class="liquid", prop="dynamic_viscosity", value=mu,
            conditions=cond, domain=dom,
            provenance=_prov("SRC-IAPWS-WATER-VISCOSITY", "eq. (2), p. 943 (Vogel fit)",
                             "correlation", "FORMULA_PRIMARY", "kestin1978_vogel"),
            uncertainty_rel=0.01 if i >= 20 else 0.03,
            uncertainty_basis="Vogel fit vs Kestin Table 2: <=1% for T>=10 degC, <=3% at 0 degC"))
        recs.append(make_record(
            substance="water", substance_class="liquid", prop="kinematic_viscosity", value=mu / dens,
            conditions=cond, domain=dom,
            provenance=_prov("SRC-IAPWS-WATER-VISCOSITY", "nu = mu/rho (see derived_from)",
                             "derived", "DERIVED"),
            uncertainty_rel=0.0101, uncertainty_basis="mu (1%) + rho (1e-5)",
            derived_from=f"{_rid('water', 'dynamic_viscosity', T, P_ATM)}/{_rid('water', 'density', T, P_ATM)}"))
        recs.append(make_record(
            substance="water", substance_class="liquid", prop="specific_heat",
            value=corr.water_specific_heat(T), conditions=cond, domain=dom,
            provenance=_prov("SRC-WATER-CP-TABLE", "table p. 6-4, linear interpolation in T",
                             "table", "TABLE_PRIMARY"),
            uncertainty_rel=0.002,
            uncertainty_basis="tabulated to 4-5 significant figures; linear interp adds <0.2%"))
        recs.append(make_record(
            substance="water", substance_class="liquid", prop="thermal_conductivity",
            value=corr.water_thermal_conductivity(T), conditions=cond, domain=dom,
            provenance=_prov("SRC-IAPWS-WATER-CONDUCTIVITY", "cubic fit to R15-11 isobar",
                             "correlation", "FORMULA_PRIMARY", "iaprws_r15_fit"),
            uncertainty_rel=0.015, uncertainty_basis="cubic fit vs R15-11 isobar <= 1.5%"))
        recs.append(make_record(
            substance="water", substance_class="liquid", prop="sound_speed",
            value=corr.water_sound_speed(T), conditions=cond, domain=dom,
            provenance=_prov("SRC-IAPWS-WATER-SPEED", "eq. (2), p. 1610",
                             "correlation", "FORMULA_PRIMARY", "bilaniuk_wong_1993"),
            uncertainty_rel=5e-4, uncertainty_basis="stated accuracy 0.05% for 0-100 degC"))
    recs.append(make_record(
        substance="water", substance_class="liquid", prop="latent_heat_fusion",
        value=corr.LATENT_HEAT_FUSION, conditions={"T_K": T0, "p_Pa": P_ATM}, domain=dom,
        provenance=_prov("SRC-WATER-LATENT-FUSION", "table p. 6-4", "table", "TABLE_PRIMARY"),
        uncertainty_rel=0.001, uncertainty_basis="tabulated 333.55 kJ/kg at 0 degC"))
    recs.append(make_record(
        substance="water", substance_class="liquid", prop="freezing_temperature", value=T0,
        conditions={"T_K": T0, "p_Pa": P_ATM}, domain=dom,
        provenance=_prov("SRC-IAPWS-WATER-DENSITY", "ITS-90 phase-equilibrium definition",
                         "definition", "DERIVED"),
        uncertainty_rel=1e-5, uncertainty_basis="definition of the 0 degC fixed point at 0.1 MPa",
        derived_from="ITS-90 temperature-scale definition"))
    return recs


def seawater_records() -> List[dict]:
    recs: List[dict] = []
    dom_dens = {"T_K": [273.15, 453.15], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 160.0]}
    dom_visc = {"T_K": [273.15, 453.15], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 150.0]}
    dom_cp = {"T_K": [273.15, 453.15], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 180.0]}
    for S in (5.0, 10.0, 20.0, 35.0, 40.0):
        for i in range(0, 81):                   # 0 .. 40 degC at 0.5 K
            T = T0 + 0.5 * i
            dens = corr.seawater_density(T, S)
            mu = corr.seawater_dynamic_viscosity(T, S)
            cond = {"T_K": T, "p_Pa": P_ATM, "salinity_g_per_kg": S}
            recs.append(make_record(
                substance="seawater", substance_class="liquid", prop="density", value=dens,
                conditions=cond, domain=dom_dens,
                provenance=_prov("SRC-SHARQAWY-DENSITY", "Table 4, eq. (2)-(5)", "correlation",
                                 "FORMULA_PRIMARY", "sharqawy2010_density"),
                uncertainty_rel=1e-4, uncertainty_basis="stated 0.01% for 0-40 degC, S <= 42 g/kg"))
            recs.append(make_record(
                substance="seawater", substance_class="liquid", prop="dynamic_viscosity", value=mu,
                conditions=cond, domain=dom_visc,
                provenance=_prov("SRC-SHARQAWY-VISCOSITY", "Table 5, eq. (22)-(24)", "correlation",
                                 "FORMULA_PRIMARY", "sharqawy2010_viscosity"),
                uncertainty_rel=0.015, uncertainty_basis="stated 1.5% vs measurements"))
            recs.append(make_record(
                substance="seawater", substance_class="liquid", prop="kinematic_viscosity",
                value=mu / dens, conditions=cond, domain=dom_visc,
                provenance=_prov("SRC-SHARQAWY-VISCOSITY", "nu = mu/rho (see derived_from)",
                                 "derived", "DERIVED"),
                uncertainty_rel=0.0151, uncertainty_basis="mu (1.5%) + rho (0.01%)",
                derived_from=(f"{_rid('seawater', 'dynamic_viscosity', T, P_ATM, S)}"
                              f"/{_rid('seawater', 'density', T, P_ATM, S)}")))
            recs.append(make_record(
                substance="seawater", substance_class="liquid", prop="specific_heat",
                value=corr.seawater_specific_heat(T, S), conditions=cond, domain=dom_cp,
                provenance=_prov("SRC-SHARQAWY-CP", "Table 3, eq. (9)-(13)", "correlation",
                                 "FORMULA_PRIMARY", "sharqawy2010_cp"),
                uncertainty_rel=0.003, uncertainty_basis="stated 0.3%"))
        tf = corr.seawater_freezing_temperature(S)
        recs.append(make_record(
            substance="seawater", substance_class="liquid", prop="freezing_temperature", value=tf,
            conditions={"T_K": None, "p_Pa": P_ATM, "salinity_g_per_kg": S},
            domain={"T_K": [268.0, 273.15], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 160.0]},
            provenance=_prov("SRC-MILLERO-FREEZING", "eq. (6) of Sharqawy et al. (2010)",
                             "correlation", "FORMULA_PRIMARY", "millero_leung_freeze"),
            uncertainty_rel=1e-4, uncertainty_basis="stated 0.01 K at 1 atm"))
    return recs


def air_records() -> List[dict]:
    recs: List[dict] = []
    dom = {"T_K": [233.15, 323.15], "p_Pa": [50000.0, 200000.0]}
    for i in range(0, 19):                       # 233.15 .. 323.15 K at 5 K
        T = 233.15 + 5.0 * i
        cond = {"T_K": T, "p_Pa": P_ATM}
        dens = corr.air_density(T, P_ATM)
        mu = corr.air_dynamic_viscosity(T)
        recs.append(make_record(
            substance="air", substance_class="gas", prop="density", value=dens, conditions=cond,
            domain=dom,
            provenance=_prov("SRC-AIR-IDEAL-GAS", "rho = p/(R T), R = 287.058 J/(kg K)",
                             "correlation", "FORMULA_PRIMARY", "air_ideal_gas"),
            uncertainty_rel=0.001, uncertainty_basis="real-gas deviation < 0.1% for p <= 1 MPa"))
        recs.append(make_record(
            substance="air", substance_class="gas", prop="dynamic_viscosity", value=mu, conditions=cond,
            domain=dom,
            provenance=_prov("SRC-AIR-SUTHERLAND", "Sutherland law, Table 1-2 coefficients",
                             "correlation", "FORMULA_PRIMARY", "sutherland_1893"),
            uncertainty_rel=0.02, uncertainty_basis="stated 2% for 250-800 K at 0.1 MPa"))
        recs.append(make_record(
            substance="air", substance_class="gas", prop="kinematic_viscosity", value=mu / dens,
            conditions=cond, domain=dom,
            provenance=_prov("SRC-AIR-SUTHERLAND", "nu = mu/rho (see derived_from)", "derived", "DERIVED"),
            uncertainty_rel=0.0201, uncertainty_basis="mu (2%) + rho (0.1%)",
            derived_from=f"{_rid('air', 'dynamic_viscosity', T, P_ATM)}/{_rid('air', 'density', T, P_ATM)}"))
        k = corr.air_thermal_conductivity(T)
        recs.append(make_record(
            substance="air", substance_class="gas", prop="thermal_conductivity", value=k, conditions=cond,
            domain={"T_K": [200.0, 1500.0], "p_Pa": [50000.0, 200000.0]},
            provenance=_prov("SRC-AIR-CONDUCTIVITY", "CRC table p. 6-190 + modified-Eucken fit",
                             "correlation", "FORMULA_PRIMARY", "air_crc_eucken"),
            uncertainty_rel=0.015, uncertainty_basis="stated 1.5% for 250-600 K"))
        recs.append(make_record(
            substance="air", substance_class="gas", prop="sound_speed",
            value=corr.air_sound_speed(T), conditions=cond, domain=dom,
            provenance=_prov("SRC-AIR-CP-GAMMA", "a = sqrt(gamma R T), gamma = 1.4",
                             "correlation", "FORMULA_PRIMARY", "air_sound_speed"),
            uncertainty_rel=5e-3, uncertainty_basis="gamma = 1.4 valid 250-350 K (< 0.5%)"))
        recs.append(make_record(
            substance="air", substance_class="gas", prop="prandtl_number",
            value=mu * corr.CP_AIR / k, conditions=cond,
            domain={"T_K": [233.15, 600.0], "p_Pa": [50000.0, 200000.0]},
            provenance=_prov("SRC-AIR-SUTHERLAND", "Pr = mu cp / k (see derived_from)", "derived", "DERIVED"),
            uncertainty_rel=0.03, uncertainty_basis="mu (2%) + k (1.5%) + cp (0.5%)",
            derived_from=(f"{_rid('air', 'dynamic_viscosity', T, P_ATM)}*{corr.CP_AIR}"
                          f"/{_rid('air', 'thermal_conductivity', T, P_ATM)}")))
    for p in (50000.0, 200000.0):                # pressure axis: ideal gas scales with p
        for T in (273.15, 288.15, 313.15):
            recs.append(make_record(
                substance="air", substance_class="gas", prop="density",
                value=corr.air_density(T, p), conditions={"T_K": T, "p_Pa": p}, domain=dom,
                provenance=_prov("SRC-AIR-IDEAL-GAS", "rho = p/(R T), p-axis sample",
                                 "correlation", "FORMULA_PRIMARY", "air_ideal_gas"),
                uncertainty_rel=0.002, uncertainty_basis="ideal gas at 0.5-2 bar, 0-40 degC"))
    recs.append(make_record(
        substance="air", substance_class="gas", prop="specific_heat", value=corr.CP_AIR,
        conditions={"T_K": 288.15, "p_Pa": P_ATM}, domain=dom,
        provenance=_prov("SRC-AIR-CP-GAMMA", "table p. 6-2", "table", "TABLE_PRIMARY"),
        uncertainty_rel=0.005, uncertainty_basis="cp varies < 0.5% over 250-350 K"))
    return recs


# --------------------------------------------------------------------------- #
# Ice / sediment / structural builders
# --------------------------------------------------------------------------- #
def ice_records() -> List[dict]:
    recs: List[dict] = []
    for t_c in ICE_FRESH_DENSITY_T_C:
        T = T0 + t_c
        recs.append(make_record(
            substance="ice_fresh", substance_class="ice", prop="density",
            value=corr.fresh_ice_density(T), conditions={"T_K": T, "p_Pa": P_ATM},
            domain={"T_K": [213.15, 273.15], "p_Pa": [P_ATM, P_ATM]},
            provenance=_prov("SRC-PURE-ICE-IH", "Table 2.1 + beta = 1.53e-4 1/K contraction",
                             "table+correlation", "FORMULA_PRIMARY", "hobbs_ice_density"),
            uncertainty_rel=0.002, uncertainty_basis="lattice value 916.8 kg/m^3 at 0 degC (0.2%)"))
    for prop, val, rel, loc, note in ICE_FRESH_TABLE:
        recs.append(make_record(
            substance="ice_fresh", substance_class="ice", prop=prop, value=val,
            conditions={"T_K": T0, "p_Pa": P_ATM},
            domain={"T_K": [213.15, 273.15], "p_Pa": [P_ATM, P_ATM]},
            provenance=_prov("SRC-PURE-ICE-IH", loc, "table", "TABLE_PRIMARY"),
            uncertainty_rel=rel, uncertainty_basis=note))
    for prop, val, rel, loc, note, variant in ICE_SEA_TABLE:
        recs.append(make_record(
            substance="ice_sea", substance_class="ice", prop=prop, value=val, variant=variant,
            conditions={"T_K": 263.15, "p_Pa": P_ATM, "salinity_g_per_kg": 5.0},
            domain={"T_K": [253.15, 271.15], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 20.0]},
            provenance=_prov("SRC-TIMCO-WEEKS-2010", loc, "table", "TABLE_PRIMARY"),
            uncertainty_rel=rel, uncertainty_basis=note))
    for t_c in ICE_SEA_T_C:
        T = T0 + t_c
        S = 5.0
        vb = corr.brine_volume_fraction(T, S)
        recs.append(make_record(
            substance="ice_sea", substance_class="ice", prop="brine_volume", value=vb,
            conditions={"T_K": T, "p_Pa": P_ATM, "salinity_g_per_kg": S},
            domain={"T_K": [250.25, 272.65], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 20.0]},
            provenance=_prov("SRC-FRANKENSTEIN-GARNER", "eq. (1), p. 943", "correlation",
                             "FORMULA_PRIMARY", "frankenstein_garner_1967"),
            uncertainty_rel=0.05, uncertainty_basis="fit residual < 0.01 absolute over -0.5..-22.9 degC"))
        recs.append(make_record(
            substance="ice_sea", substance_class="ice", prop="flexural_strength",
            value=corr.sea_ice_flexural_strength(vb),
            conditions={"T_K": T, "p_Pa": P_ATM, "salinity_g_per_kg": S},
            domain={"T_K": [250.25, 272.65], "p_Pa": [P_ATM, P_ATM], "salinity_g_per_kg": [0.0, 20.0]},
            provenance=_prov("SRC-TIMCO-WEEKS-FLEXURAL", "eq. (8) / Fig. 8", "correlation",
                             "FORMULA_PRIMARY", "timco_weeks_flexural"),
            uncertainty_rel=0.30, uncertainty_basis="published scatter band +/-30% about the fit"))
    return recs


def sediment_records() -> List[dict]:
    recs: List[dict] = []
    for substance, rows in SEDIMENT_TABLE.items():
        for prop, val, rel, loc, note in rows:
            recs.append(make_record(
                substance=substance, substance_class="sediment", prop=prop, value=val,
                conditions={}, domain={"p_Pa": [0.0, 1.0e6]},
                provenance=_prov("SRC-DNV-C212" if "DNV" in loc else "SRC-SOULSBY-1997",
                                 loc, "table", "TABLE_PRIMARY"),
                uncertainty_rel=rel, uncertainty_basis=note))
            if prop == "dry_bulk_density":
                # submerged unit weight gamma' = (rho_s - rho_w) g, rho_w = 1025 kg/m^3
                rho_s = [r[1] for r in rows if r[0] == "grain_density"][0]
                gamma = (rho_s - 1025.0) * 9.81
                recs.append(make_record(
                    substance=substance, substance_class="sediment", prop="submerged_unit_weight",
                    value=gamma, conditions={}, domain={"p_Pa": [0.0, 1.0e6]},
                    provenance=_prov("SRC-SOULSBY-1997",
                                     "gamma' = (rho_s - rho_w) g with rho_w = 1025 kg/m^3", "derived",
                                     "DERIVED"),
                    uncertainty_rel=0.05, uncertainty_basis="grain density (1%) + seawater density (0.01%)",
                    derived_from=f"{substance}.grain_density - 1025 kg/m^3; g = 9.81 m/s^2"))
    return recs


def structural_records() -> List[dict]:
    recs: List[dict] = []
    for group, cls in ((METAL_TABLE, "metal"), (COMPOSITE_TABLE, "composite")):
        for substance, rows in group.items():
            for prop, val, rel, src, loc, note in rows:
                recs.append(make_record(
                    substance=substance, substance_class=cls, prop=prop, value=val,
                    conditions={}, domain={"T_K": [233.15, 373.15]},
                    provenance=_prov(src, loc, "table", "TABLE_PRIMARY"),
                    uncertainty_rel=rel, uncertainty_basis=note))
    for prop, val, rel, src, loc, note in CONCRETE_TABLE:
        recs.append(make_record(
            substance="concrete_C30", substance_class="cementitious", prop=prop, value=val,
            conditions={}, domain={"T_K": [233.15, 373.15]},
            provenance=_prov(src, loc, "table", "TABLE_PRIMARY"),
            uncertainty_rel=rel, uncertainty_basis=note))
    return recs


def build_records() -> List[dict]:
    """Build every MPDB row (deterministic; no randomness, no clock dependence)."""
    recs: List[dict] = []
    recs += water_records()
    recs += seawater_records()
    recs += air_records()
    recs += ice_records()
    recs += sediment_records()
    recs += structural_records()
    recs.sort(key=lambda r: (r["substance"], r["property"],
                             -1.0 if r["conditions"].get("T_K") is None else r["conditions"]["T_K"],
                             -1.0 if r["conditions"].get("p_Pa") is None else r["conditions"]["p_Pa"],
                             -1.0 if r["conditions"].get("salinity_g_per_kg") is None
                             else r["conditions"]["salinity_g_per_kg"]))
    errors = validate_records(recs, SOURCE_REGISTRY)
    if errors:
        raise ValueError("MPDB build produced invalid rows:\n  " + "\n  ".join(errors[:20]))
    missing = {s for s in SUBSTANCES} - {r["substance"] for r in recs}
    if missing:
        raise ValueError(f"declared substances without rows: {sorted(missing)}")
    return recs


# --------------------------------------------------------------------------- #
# Anchor audit
# --------------------------------------------------------------------------- #
def anchor_audit(records: List[dict]) -> List[dict]:
    """Evaluate every independent anchor through the *public query surface*.

    Going through :class:`~tensorphysics.material_properties.query.MaterialDB`
    means the audit measures what a solver would actually get (including linear
    interpolation between sampled states), not the raw table.
    """
    from .query import MPDBError, MaterialDB

    db = MaterialDB(records)
    out: List[dict] = []
    for anc in ANCHORS:
        conds = dict(anc["conditions"])
        T = conds.pop("T_K", None)
        p = conds.pop("p_Pa", None)
        S = conds.pop("salinity_g_per_kg", None)
        base = {
            "id": anc["id"], "substance": anc["substance"], "property": anc["property"],
            "unit": anc["unit"], "expected": anc["expected"], "rel_tol": anc["rel_tol"],
            "reference": anc["reference"], "conditions": dict(anc["conditions"]),
        }
        try:
            pv = db.get(anc["substance"], anc["property"], T=T, p=p, salinity=S)
        except MPDBError as exc:
            out.append({**base, "computed": None, "rel_error": None, "passed": False,
                        "row_id": None, "note": f"{type(exc).__name__}: {exc}"})
            continue
        got = float(pv.value)
        rel = abs(got - anc["expected"]) / abs(anc["expected"]) if anc["expected"] else 0.0
        out.append({**base, "computed": got, "rel_error": rel,
                    "passed": bool(rel <= anc["rel_tol"]),
                    "row_id": pv.row_ids[0] if pv.row_ids else None,
                    "interpolated": dict(pv.interpolated)})
    return out


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out_dir: str = DATA_DIR, write: bool = True) -> Dict[str, Any]:
    recs = build_records()
    anchors = anchor_audit(recs)

    counts_by_class: Dict[str, int] = {}
    counts_by_substance: Dict[str, int] = {}
    counts_by_property: Dict[str, int] = {}
    for r in recs:
        counts_by_class[r["substance_class"]] = counts_by_class.get(r["substance_class"], 0) + 1
        counts_by_substance[r["substance"]] = counts_by_substance.get(r["substance"], 0) + 1
        counts_by_property[r["property"]] = counts_by_property.get(r["property"], 0) + 1

    meta = {
        "db": DB_NAME,
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repro_cmd": REPRO_CMD,
        "rows": len(recs),
        "rows_by_class": dict(sorted(counts_by_class.items())),
        "rows_by_substance": dict(sorted(counts_by_substance.items())),
        "rows_by_property": dict(sorted(counts_by_property.items())),
        "substances": len(SUBSTANCES),
        "substance_classes": len(counts_by_class),
        "sources": len(SOURCE_REGISTRY),
        "rows_with_provenance": sum(
            1 for r in recs if r["provenance"].get("source_id") in SOURCE_REGISTRY
            and r["provenance"].get("locator")),
        "rows_formula_primary": sum(1 for r in recs if r["provenance"]["evidence_class"] == "FORMULA_PRIMARY"),
        "rows_table_primary": sum(1 for r in recs if r["provenance"]["evidence_class"] == "TABLE_PRIMARY"),
        "rows_derived": sum(1 for r in recs if r["provenance"]["evidence_class"] == "DERIVED"),
        "anchors_total": len(anchors),
        "anchors_passed": sum(1 for a in anchors if a["passed"]),
        "anchor_max_rel_error": max((a["rel_error"] or 0.0) for a in anchors),
        "units_policy": "SI base units; degC / deg / g-per-kg are access units only",
        "access_policy": "offline build, no PDF re-opened (see sources.access)",
    }

    if write:
        os.makedirs(out_dir, exist_ok=True)
        jsonl = os.path.join(out_dir, f"{DB_NAME}.jsonl")
        with open(jsonl, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        meta["jsonl_sha256"] = _sha256(jsonl)
        meta["jsonl_bytes"] = os.path.getsize(jsonl)
        json.dump(SUBSTANCES, open(os.path.join(out_dir, f"{DB_NAME}.substances.json"), "w"),
                  indent=2, ensure_ascii=False, sort_keys=True)
        json.dump(anchors, open(os.path.join(out_dir, f"{DB_NAME}.anchors.json"), "w"),
                  indent=2, ensure_ascii=False)
        json.dump(to_json_schema(), open(os.path.join(out_dir, f"{DB_NAME}.schema.json"), "w"),
                  indent=2, ensure_ascii=False)
        json.dump(meta, open(os.path.join(out_dir, f"{DB_NAME}.meta.json"), "w"),
                  indent=2, ensure_ascii=False, sort_keys=True)
    return {"records": recs, "anchors": anchors, "meta": meta}


def check_committed(out_dir: str = DATA_DIR) -> Dict[str, Any]:
    """Rebuild in memory and compare with the committed artefacts."""
    fresh = build(out_dir=out_dir, write=False)
    problems: List[str] = []
    jsonl = os.path.join(out_dir, f"{DB_NAME}.jsonl")
    if not os.path.exists(jsonl):
        return {"ok": False, "problems": [f"missing {jsonl}"]}
    committed = [json.loads(line) for line in open(jsonl, encoding="utf-8")]
    if len(committed) != len(fresh["records"]):
        problems.append(f"row count {len(committed)} != rebuilt {len(fresh['records'])}")
    fresh_map = {r["id"]: r for r in fresh["records"]}
    for r in committed:
        f = fresh_map.get(r["id"])
        if f is None:
            problems.append(f"committed row {r['id']} not produced by rebuild")
            continue
        if abs(float(r["value"]) - float(f["value"])) > 1e-12 * max(1.0, abs(float(f["value"]))):
            problems.append(f"value drift on {r['id']}: {r['value']} vs {f['value']}")
        if r["provenance"] != f["provenance"]:
            problems.append(f"provenance drift on {r['id']}")
    meta_path = os.path.join(out_dir, f"{DB_NAME}.meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("jsonl_sha256") and meta["jsonl_sha256"] != _sha256(jsonl):
            problems.append("meta.jsonl_sha256 does not match the committed jsonl")
    return {"ok": not problems, "problems": problems,
            "anchors_passed": fresh["meta"]["anchors_passed"],
            "anchors_total": fresh["meta"]["anchors_total"]}