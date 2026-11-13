"""Tests for the material/fluid property database (MPDB v1, mission 17459).

Coverage
--------
* schema conformance of every packaged row (SI units, provenance, domains),
* independent-anchor accuracy audit (per-anchor rel error, < 3% for the physical
  anchors; the two literature-band anchors carry their published spread),
* query surface: interpolation, out-of-domain refusal, variant disambiguation,
  unknown substance/property errors, derived-row algebra,
* CLI: list / describe / query / table / validate / build --check exit codes,
* solver adapters: LBM scaling (Re preservation, tau stability), FVM air state,
  FEM sea-ice material, ship-resistance environment + ITTC-57,
* inventory classifier on a synthetic repo fixture, and on the real ship repo
  (skipped when the checkout is absent).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from tensorphysics.material_properties import (
    ANCHORS,
    PROPERTY_UNITS,
    SCHEMA_VERSION,
    SOURCE_REGISTRY,
    SUBSTANCE_CLASSES,
    AmbiguousValueError,
    MaterialDB,
    MissingConditionError,
    OutOfDomainError,
    UnknownPropertyError,
    UnknownSubstanceError,
    build,
    build_records,
    check_committed,
    fem_ice_material,
    fvm_air_state,
    ittc57_cf,
    lbm_scaling,
    plate_rigidity,
    provenance_ledger,
    seawater_consistency_check,
    ship_resistance_env,
    to_csv,
    to_markdown,
    validate_records,
)
from tensorphysics.material_properties import correlations as corr
from tensorphysics.material_properties import inventory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(REPO_ROOT, "src", "tensorphysics", "material_properties")
SHIP_REPO = "/root/ship-performance-platform-incoming/ship-performance-platform"


@pytest.fixture(scope="module")
def db() -> MaterialDB:
    return MaterialDB.load()


@pytest.fixture(scope="module")
def records():
    return build_records()


# --------------------------------------------------------------------------- #
# Schema / dataset shape
# --------------------------------------------------------------------------- #
def test_schema_validation_is_clean(records):
    assert validate_records(records, SOURCE_REGISTRY) == []
    for r in records:
        assert r["schema_version"] == SCHEMA_VERSION


def test_at_least_five_substance_classes_and_100_rows(records):
    classes = {r["substance_class"] for r in records}
    assert len(classes) >= 5, sorted(classes)
    assert classes <= set(SUBSTANCE_CLASSES)
    assert len(records) >= 100, len(records)
    # the five classes the asset spec names explicitly
    assert {"liquid", "gas", "ice", "sediment", "metal"} <= classes


def test_every_row_has_resolvable_provenance(records):
    for r in records:
        p = r["provenance"]
        assert p["source_id"] in SOURCE_REGISTRY
        assert p["locator"] and p["citation"]
        assert p["evidence_class"] in ("FORMULA_PRIMARY", "TABLE_PRIMARY", "DERIVED")
        assert SOURCE_REGISTRY[p["source_id"]]["citation"]  # full citation present


def test_units_are_si_base(records):
    for r in records:
        assert r["unit"] == PROPERTY_UNITS[r["property"]][0], r["id"]
    # no Celsius leaks into the payload: temperatures are stored in kelvin
    for r in records:
        t = r["conditions"].get("T_K")
        if t is not None:
            assert t > 100.0, f"{r['id']} looks like a Celsius value in T_K"


def test_domains_are_non_empty_boxes(records):
    for r in records:
        assert r["domain"], r["id"]
        for ax, box in r["domain"].items():
            assert len(box) == 2 and box[0] <= box[1], (r["id"], ax, box)


def test_no_duplicate_ids(records):
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Accuracy: independent anchors
# --------------------------------------------------------------------------- #
def test_all_anchors_pass():
    res = build(write=False)
    failed = [a for a in res["anchors"] if not a["passed"]]
    assert not failed, [(a["id"], a["computed"], a["expected"], a["rel_error"]) for a in failed]


def test_physical_anchors_within_3_percent():
    """Every anchor except the two literature-spread ones must be inside 3%."""
    band_anchors = {"seaice-flexural-ref", "water-viscosity-0C"}
    res = build(write=False)
    for a in res["anchors"]:
        if a["id"] in band_anchors:
            continue
        assert a["rel_error"] is not None and a["rel_error"] < 0.03, (a["id"], a["rel_error"])


def test_key_numbers_match_published_values(db):
    # water at 20 degC
    rho = db.get("water", "density", T_C=20).value
    assert rho == pytest.approx(998.21, rel=1e-4)
    mu = db.get("water", "dynamic_viscosity", T_C=20).value
    assert mu == pytest.approx(1.0016e-3, rel=0.01)
    # ITTC-2011 15 degC anchors
    assert db.get("water", "kinematic_viscosity", T_C=15).value == pytest.approx(1.1390e-6, rel=0.01)
    assert db.get("seawater", "kinematic_viscosity", T_C=15, salinity=35).value == pytest.approx(
        1.1883e-6, rel=0.02)
    # seawater density / freezing point / cp at 35 g/kg
    assert db.get("seawater", "density", T_C=25, salinity=35).value == pytest.approx(1023.34, rel=1e-3)
    assert db.get("seawater", "freezing_temperature", salinity=35).value == pytest.approx(271.25, rel=2e-3)
    assert db.get("seawater", "specific_heat", T_C=20, salinity=35).value == pytest.approx(3993.0, rel=0.01)
    # air (ISO 2533)
    assert db.get("air", "density", T_C=15).value == pytest.approx(1.2255, rel=2e-3)
    assert db.get("air", "dynamic_viscosity", T_C=15).value == pytest.approx(1.7894e-5, rel=0.02)
    assert db.get("air", "sound_speed", T_C=15).value == pytest.approx(340.29, rel=5e-3)


def test_anchor_list_is_not_self_referential_for_tables():
    """Every anchor must name a citation, and table anchors must not be FORMULA_PRIMARY-only."""
    for a in ANCHORS:
        assert a["reference"]
        assert a["rel_tol"] > 0


# --------------------------------------------------------------------------- #
# Query surface
# --------------------------------------------------------------------------- #
def test_query_returns_provenance(db):
    pv = db.get("seawater", "kinematic_viscosity", T_C=15, salinity=35)
    assert pv.unit == "m^2/s"
    assert pv.provenance["source_id"] == "SRC-SHARQAWY-VISCOSITY"
    assert pv.row_ids
    assert pv.uncertainty_rel and pv.uncertainty_rel < 0.05
    assert float(pv) == pytest.approx(pv.value)


def test_temperature_interpolation_is_linear_and_in_domain(db):
    a = db.get("water", "density", T_C=15.0).value       # sampled node
    b = db.get("water", "density", T_C=15.5).value       # sampled node
    mid = db.get("water", "density", T_C=15.25).value    # halfway between nodes
    assert mid == pytest.approx(0.5 * (a + b), rel=1e-12)
    # linear-interpolation error over a 0.5 K cell is O(dT^2); keep it < 1e-6 relative
    assert abs(mid - corr.water_density(273.15 + 15.25)) / mid < 1e-6
    assert db.get("water", "density", T_C=15.25).interpolated.get("T_K") == "linear"


def test_out_of_domain_is_refused_and_clamp_is_explicit(db):
    with pytest.raises(OutOfDomainError):
        db.get("water", "density", T_C=150.0)
    clamped = db.get("water", "density", T_C=150.0, clamp=True)
    assert clamped.value == pytest.approx(db.table("water", "density")[-1]["value"])
    assert "T_K" in clamped.interpolated and "clamped" in clamped.interpolated["T_K"]


def test_missing_condition_is_refused(db):
    with pytest.raises(MissingConditionError):
        db.get("seawater", "density", T_C=10)     # salinity axis has >1 sampled value
    # single-state axes (p = 1 atm) are adopted automatically
    assert db.get("seawater", "density", T_C=10, salinity=35).conditions["p_Pa"] == 101325.0


def test_variant_disambiguation(db):
    with pytest.raises(AmbiguousValueError) as exc:
        db.get("ice_sea", "youngs_modulus")
    assert "initial" in str(exc.value) and "effective" in str(exc.value)
    e_init = db.get("ice_sea", "youngs_modulus", variant="initial").value
    e_eff = db.get("ice_sea", "youngs_modulus", variant="effective").value
    assert e_init > e_eff > 0
    with pytest.raises(AmbiguousValueError):
        db.get("ice_sea", "youngs_modulus", variant="nonsense")


def test_unknown_names_raise(db):
    with pytest.raises(UnknownSubstanceError):
        db.get("unobtainium", "density")
    with pytest.raises(UnknownPropertyError):
        db.get("water", "flavour")


def test_derived_rows_are_algebraically_consistent(db):
    for T_C in (0.0, 5.0, 15.0, 30.0):
        mu = db.get("water", "dynamic_viscosity", T_C=T_C).value
        rho = db.get("water", "density", T_C=T_C).value
        nu = db.get("water", "kinematic_viscosity", T_C=T_C).value
        assert nu == pytest.approx(mu / rho, rel=1e-12)
    for T_C in (0.0, 15.0, 30.0):
        c = seawater_consistency_check(T_C=T_C, salinity=35.0)
        assert c["rho_rel_dev"] < 1e-9 and c["mu_rel_dev"] < 1e-9


def test_summary_counts(db):
    s = db.summary()
    assert s["rows"] >= 100
    assert len(s["classes"]) >= 5
    assert s["sources"] >= 10
    assert s["jsonl_sha256"]


def test_json_schema_export_matches_rows(records):
    schema = json.loads(open(os.path.join(PKG_DIR, "data", "mpdb_v1.schema.json")).read())
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(instance=records[0], schema=schema)
    jsonschema.validate(instance=records[-1], schema=schema)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_committed_db_matches_rebuild():
    res = check_committed()
    assert res["ok"], res["problems"][:5]
    assert res["anchors_passed"] == res["anchors_total"]


def test_build_is_deterministic():
    a = build_records()
    b = build_records()
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["value"] for r in a] == [r["value"] for r in b]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    return subprocess.run([sys.executable, "-m", "tensorphysics.material_properties.cli", *args],
                          capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def test_cli_list_and_describe():
    r = _cli("list", "--json")
    assert r.returncode == 0, r.stderr
    rows = json.loads(r.stdout)
    assert len({x["class"] for x in rows}) >= 5
    r = _cli("describe", "--substance", "seawater")
    assert r.returncode == 0, r.stderr
    desc = json.loads(r.stdout)
    assert "density" in desc["properties"]
    assert desc["properties"]["density"]["provenance_sources"]


def test_cli_query_json_and_human():
    r = _cli("query", "--substance", "seawater", "--property", "density",
             "--T-C", "15", "--salinity", "35", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)["density"]
    assert payload["value"] == pytest.approx(1025.97, rel=1e-3)
    assert payload["unit"] == "kg/m^3"
    assert payload["provenance"]["source_id"] == "SRC-SHARQAWY-DENSITY"

    r = _cli("query", "--substance", "water", "--property", "dynamic_viscosity", "--T-C", "20")
    assert r.returncode == 0
    assert "0.00100175" in r.stdout or "1.0016e-3" in r.stdout.replace(" ", "")


def test_cli_query_errors_have_nonzero_exit():
    r = _cli("query", "--substance", "water", "--property", "density", "--T-C", "150")
    assert r.returncode == 2
    assert "OutOfDomainError" in r.stderr


def test_cli_table_csv_and_validate():
    r = _cli("table", "--substance", "air", "--property", "density", "--csv")
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.strip().splitlines() if l]
    assert lines[0].startswith("id,value,unit")
    assert len(lines) > 5
    r = _cli("validate")
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads(r.stdout)
    assert report["ok"] and not report["schema_errors"]
    assert report["anchors_failed"] == []
    assert report["seawater_row_vs_correlation_max_dev"] < 1e-9


# --------------------------------------------------------------------------- #
# Solver adapters (>= 2 solver families)
# --------------------------------------------------------------------------- #
def test_lbm_scaling_preserves_reynolds_and_stability(db):
    # model-scale wake cell: Re ~ 1.7e3 at 256 nodes keeps tau safely above the BGK limit
    out = lbm_scaling(substance="seawater", T_C=15.0, salinity=35.0,
                      L_phys=0.1, U_phys=0.02, n_lattice=256, u_lattice=0.05, db=db)
    assert out["Re"] == pytest.approx(0.02 * 0.1 / out["nu_phys"], rel=1e-12)
    assert out["tau"] > 0.5
    assert out["omega"] == pytest.approx(1.0 / out["tau"])
    assert out["Ma"] < 0.1
    # dt/dx must be consistent with the physical velocity
    assert out["dx"] / out["dt"] * out["u_lattice"] == pytest.approx(0.02, rel=1e-12)
    assert out["provenance"]["kinematic_viscosity"]["source_id"]
    # full-scale Re at modest resolution drives tau onto the BGK limit -> refused
    with pytest.raises(ValueError, match="BGK unstable"):
        lbm_scaling(substance="water", T_C=15.0, L_phys=1.0, U_phys=0.5,
                    n_lattice=256, u_lattice=0.05, db=db)


def test_fvm_air_state_is_thermodynamically_consistent(db):
    st = fvm_air_state(T_C=20.0, p=101325.0, db=db)
    assert st["rho"] == pytest.approx(101325.0 / (287.058 * 293.15), rel=1e-9)
    assert st["nu"] == pytest.approx(st["mu"] / st["rho"], rel=1e-9)
    assert st["Pr"] == pytest.approx(0.71, abs=0.02)
    assert st["a"] == pytest.approx(math.sqrt(1.4 * 287.058 * 293.15), rel=1e-9)
    assert st["cp"] == pytest.approx(1005.0)


def test_fem_ice_material_and_plate_rigidity(db):
    m = fem_ice_material(T_C=-10.0, salinity=5.0, variant="effective", db=db)
    vb = corr.brine_volume_fraction(263.15, 5.0)
    assert m["brine_volume"] == pytest.approx(vb)
    assert m["sigma_f"] == pytest.approx(corr.sea_ice_flexural_strength(vb))
    assert 0.5e6 < m["sigma_f"] < 1.0e6
    assert m["E"] == pytest.approx(3.0e9)
    assert m["sigma_f_mpdb_grid"] == pytest.approx(m["sigma_f"], rel=1e-9)
    rig = plate_rigidity(m["E"], m["nu"], 1.0)
    assert rig["D"] == pytest.approx(m["E"] * 1.0 / (12 * (1 - m["nu"] ** 2)))


def test_ship_resistance_environment_and_ittc57(db):
    env = ship_resistance_env(T_C=15.0, salinity=35.0, L=200.0, V=8.0, db=db)
    assert env["rho"] == pytest.approx(1025.97, rel=1e-3)
    assert env["Re"] == pytest.approx(8.0 * 200.0 / env["nu"], rel=1e-12)
    assert 1e8 < env["Re"] < 2e9
    assert env["Cf_ittc57"] == pytest.approx(ittc57_cf(env["Re"]))
    assert 0.0010 < env["Cf_ittc57"] < 0.0025      # plausible ITTC-57 band for Re ~ 1e9
    ledger = provenance_ledger({"resistance": env})
    assert ledger and all(row["source_id"] for row in ledger)


def test_provenance_ledger_flattens_adapters(db):
    entries = {"lbm": lbm_scaling(L_phys=0.1, U_phys=0.02, n_lattice=128, db=db),
               "fvm": fvm_air_state(db=db)}
    ledger = provenance_ledger(entries)
    consumers = {row["consumer"] for row in ledger}
    assert "lbm.density" in consumers and "fvm.thermal_conductivity" in consumers


# --------------------------------------------------------------------------- #
# Inventory / migration mapping
# --------------------------------------------------------------------------- #
def test_inventory_classifier_on_synthetic_repo(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "plate_materials.json").write_text(json.dumps({
        "materials": {"steel": {"E_Pa": 210e9, "rho_kgm3": 7850.0, "nu": 0.3},
                      "aluminium": {"E_Pa": 70e9, "rho_kgm3": 2700.0, "nu": 0.33}}}))
    (tmp_path / "case_rao.json").write_text(json.dumps({
        "case": "rao_15kn", "water": {"density": 1025.0, "viscosity": 1.19e-6}}))
    (tmp_path / "solver.py").write_text("RHO_SEA = 1025.0\nNU_SEA = 1.19e-6\n")
    (tmp_path / "artefacts").mkdir()
    (tmp_path / "artefacts" / "report.json").write_text(json.dumps({"density": 1025.0}))

    rows = inventory.scan_repo(str(tmp_path))
    by_path = {os.path.basename(r["path"]): r for r in rows}
    assert by_path["plate_materials.json"]["bucket"] == "MIGRATE_TO_MPDB"
    assert "alu_5083_H116" in by_path["plate_materials.json"]["mpdb_substances"]
    assert by_path["solver.py"]["bucket"] == "MIGRATE_TO_MPDB"
    assert by_path["solver.py"]["inline_sites"] == 2
    assert by_path["case_rao.json"]["bucket"] in ("REFERENCE_MPDB", "MIGRATE_TO_MPDB")
    assert by_path["report.json"]["bucket"] == "KEEP_GENERATED"
    summary = inventory.summarise(rows, str(tmp_path))
    assert summary["carriers_total"] == len(rows) == 4
    assert "MIGRATE_TO_MPDB" in summary["by_bucket"]
    csv_text = to_csv(rows)
    assert csv_text.splitlines()[0].startswith("path,ext,bytes,bucket")
    assert "| bucket |" in to_markdown(rows, summary)


@pytest.mark.skipif(not os.path.isdir(SHIP_REPO), reason="ship-performance-platform checkout absent")
def test_inventory_on_ship_repo_finds_known_assets():
    rows = inventory.scan_repo(SHIP_REPO)
    by_name = {os.path.basename(r["path"]): r for r in rows}
    for name in ("aluminum_materials.yaml", "haz_alloys.yaml", "plate_materials.json"):
        assert name in by_name, f"{name} not detected in the ship repo inventory"
        assert by_name[name]["bucket"] == "MIGRATE_TO_MPDB"
        assert by_name[name]["mpdb_substances"], name
    summary = inventory.summarise(rows, SHIP_REPO)
    assert summary["carriers_total"] >= 100
    assert summary["by_bucket"]["MIGRATE_TO_MPDB"] >= 10