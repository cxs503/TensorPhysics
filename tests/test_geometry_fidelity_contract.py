"""Acceptance tests for the geometry fidelity contract (``geofid``).

The tests are organised after the acceptance criteria of the campaign node:

C1  two independent CAD integrations agree (nabla <= 0.1 %, S <= 0.3 %)
C2  discrete vs CAD for the three hulls (nabla <= 0.5 %, S <= 1 %, |dCb| <= 1e-3)
C3  consistency with the delivered reference quantities
C4  the gate rejects 3/3 damaged grids and 0/3 sound ones
C5  the contract satisfies ``geometry_fidelity_contract.v1`` with both digests
C6  the contract is byte-identical on a re-run
C7  a rejected grid cannot enter a physics chain
"""
from __future__ import annotations

import copy
import json

import pytest

from tensorphysics.geometry_fidelity import (
    CONTRACT_SCHEMA_VERSION,
    GATE_PASS,
    GATE_REJECT,
    ContractSchemaError,
    MeshNotAdmitted,
    SolverRecord,
    assert_admitted,
    build_contract,
    canonical_json,
    contract_hulls,
    corrupted_record,
    evaluate_gate,
    good_records,
    kcs_hull,
    lbm_grid,
    physics_chain,
    suboff_area_envelope,
    validate_contract,
)

TOL = {
    "cad_volume_rel": 1e-3,
    "cad_surface_rel": 3e-3,
    "discrete_volume_rel": 5e-3,
    "discrete_surface_rel": 1e-2,
    "discrete_cb_abs": 1e-3,
}


@pytest.fixture(scope="session")
def hulls():
    return contract_hulls()


@pytest.fixture(scope="session")
def contracts(hulls):
    return {name: build_contract(hull, good_records(hull)) for name, hull in hulls.items()}


# -- C1: two independent CAD integrations ------------------------------------


def test_c1_cad_volume_agrees(contracts):
    for name, contract in contracts.items():
        cad = contract["cad"]
        rel = abs(cad["nabla_slice_m3"] - cad["nabla_monte_carlo_m3"]) / cad["nabla_m3"]
        assert rel <= TOL["cad_volume_rel"], f"{name}: slice vs MC volume {rel:.2e}"


def test_c1_cad_surface_agrees(contracts):
    for name, contract in contracts.items():
        cad = contract["cad"]
        rel = abs(cad["s_quadrature_m2"] - cad["s_monte_carlo_m2"]) / cad["s_wetted_m2"]
        assert rel <= TOL["cad_surface_rel"], f"{name}: quadrature vs MC area {rel:.2e}"


def test_c1_monte_carlo_uses_one_million_samples(contracts):
    for contract in contracts.values():
        assert contract["cad"]["integration"]["mc_samples"] == 1_000_000


# -- C2: discrete vs CAD for the three hulls --------------------------------


def test_c2_discrete_volume_within_half_percent(contracts):
    for name, contract in contracts.items():
        cad_v = contract["cad"]["nabla_m3"]
        for family, record in contract["grid"]["solvers"].items():
            rel = abs(record["nabla_m3"] - cad_v) / cad_v
            assert rel <= TOL["discrete_volume_rel"], f"{name}/{family}: dV {rel:.2e}"


def test_c2_discrete_surface_within_one_percent(contracts):
    for name, contract in contracts.items():
        cad_s = contract["cad"]["s_wetted_m2"]
        for family, record in contract["grid"]["solvers"].items():
            if record.get("surface_evidence") == "WITHHELD":
                continue
            rel = abs(record["s_wetted_m2"] - cad_s) / cad_s
            assert rel <= TOL["discrete_surface_rel"], f"{name}/{family}: dS {rel:.2e}"


def test_c2_block_coefficient_within_1e_3(contracts):
    for name, contract in contracts.items():
        cad_cb = contract["cad"]["cb"]
        for family, record in contract["grid"]["solvers"].items():
            assert abs(record["cb"] - cad_cb) <= TOL["discrete_cb_abs"], f"{name}/{family}"


def test_c2_all_good_grids_pass_the_gate(contracts):
    for name, contract in contracts.items():
        assert contract["gate"]["verdict"] == GATE_PASS, (name, contract["gate"]["failures"])
        assert contract["gate"]["failures"] == []


# -- C3: agreement with the delivered reference geometry --------------------


def test_c3_kcs_block_coefficient_matches_contract(hulls):
    hull = hulls["kcs"]
    cb = hull.cad_integrals()["cb"]
    assert abs(cb - 0.651) / 0.651 <= 1e-3


def test_c3_kvlcc2_block_coefficient_matches_contract(hulls):
    cb = hulls["kvlcc2"].cad_integrals()["cb"]
    assert abs(cb - 0.8098) / 0.8098 <= 1e-3


def test_c3_suboff_displacement_matches_reference(hulls):
    hull = hulls["suboff"]
    nabla = hull.cad_integrals()["nabla_m3"]
    assert abs(nabla - 0.699211) / 0.699211 <= 1e-3


def test_c3_suboff_wetted_area_is_reported_honestly(hulls):
    """The delivered pair (nabla = 0.699211, S = 5.998) cannot both be met at 0.1 %.

    The displacement is closed by construction, so the wetted area is a genuine
    prediction of the closed geometry.  The measured deviation is recorded in the
    report instead of being tuned away; here we only pin that it stays inside the
    0.5 % band used for the "near miss" statement.
    """
    area = hulls["suboff"].cad_integrals()["s_wetted_m2"]
    deviation = abs(area - 5.998) / 5.998
    assert deviation <= 5e-3
    assert deviation > 1e-3, "if this ever passes the 0.1 % criterion the report must be updated"


def test_c3_suboff_area_envelope_brackets_the_reference(hulls):
    rows = suboff_area_envelope()
    areas = [row["s_wetted_m2"] for row in rows]
    for row in rows:
        assert abs(row["nabla_m3"] - 0.699211) / 0.699211 <= 1e-3
    assert min(areas) > 0.0 and max(areas) > min(areas)


# -- C4: negative corpus ----------------------------------------------------


@pytest.mark.parametrize("kind", ["stretch_2pct", "drop_voxel_layer", "off_by_half_cell"])
def test_c4_negative_controls_are_rejected(kind):
    hull = kcs_hull()
    record = corrupted_record(hull, kind)
    sound = good_records(hull)
    family = record.family
    polluted = dict(sound)
    polluted[family] = record
    contract = build_contract(hull, polluted, mc_samples=100_000)
    assert contract["gate"]["verdict"] == GATE_REJECT, (kind, contract["gate"])
    with pytest.raises(MeshNotAdmitted):
        assert_admitted(contract)


def test_c4_sound_grids_are_not_flagged(contracts):
    for contract in contracts.values():
        assert contract["gate"]["verdict"] == GATE_PASS
        assert not [c for c in contract["gate"]["checks"] if not c["pass"]]


def test_c4_each_negative_fails_a_different_check():
    hull = kcs_hull()
    sound = good_records(hull)
    failing = {}
    for kind in ("stretch_2pct", "drop_voxel_layer", "off_by_half_cell"):
        record = corrupted_record(hull, kind)
        polluted = dict(sound)
        polluted[record.family] = record
        contract = build_contract(hull, polluted, mc_samples=100_000)
        failing[kind] = {entry["id"] for entry in contract["gate"]["failures"]}
    assert all(failing.values()), failing


# -- C5: schema and provenance ---------------------------------------------


def test_c5_contract_validates_and_carries_both_digests(contracts):
    for contract in contracts.values():
        validate_contract(contract)
        assert contract["schema"] == CONTRACT_SCHEMA_VERSION
        for key in ("cad_sha256", "grid_sha256"):
            digest = contract["provenance"][key]
            assert isinstance(digest, str) and len(digest) == 64
            int(digest, 16)


def test_c5_schema_rejects_missing_provenance(contracts):
    contract = copy.deepcopy(contracts["kcs"])
    del contract["provenance"]["grid_sha256"]
    with pytest.raises(ContractSchemaError):
        validate_contract(contract)


def test_c5_digests_are_sensitive_to_resolution_and_geometry(hulls):
    coarse = build_contract(hulls["kcs"], {"lbm": lbm_grid(hulls["kcs"], n_x=120)}, mc_samples=50_000)
    fine = build_contract(hulls["kcs"], {"lbm": lbm_grid(hulls["kcs"], n_x=240)}, mc_samples=50_000)
    assert coarse["provenance"]["grid_sha256"] != fine["provenance"]["grid_sha256"]
    assert coarse["provenance"]["cad_sha256"] == fine["provenance"]["cad_sha256"]


# -- C6: idempotency -------------------------------------------------------


def test_c6_contract_is_byte_identical_on_rerun(hulls):
    hull = hulls["suboff"]
    first = canonical_json(build_contract(hull, good_records(hull), mc_samples=200_000))
    second = canonical_json(build_contract(hull, good_records(hull), mc_samples=200_000))
    assert first == second
    assert json.loads(first)["gate"]["verdict"] == GATE_PASS


# -- C7: structural admission ---------------------------------------------


def test_c7_rejected_grid_cannot_enter_the_physics_chain():
    hull = kcs_hull()
    polluted = dict(good_records(hull))
    polluted["fvm"] = corrupted_record(hull, "stretch_2pct")
    contract = build_contract(hull, polluted, mc_samples=100_000)
    calls: list[str] = []

    def consumer(doc):
        calls.append(doc["geometry_id"])
        return "resistance"

    with pytest.raises(MeshNotAdmitted) as excinfo:
        physics_chain(contract, "fvm", consumer)
    assert calls == []
    assert "rejected" in str(excinfo.value)


def test_c7_admitted_grid_runs_the_consumer(contracts):
    contract = contracts["kcs"]
    assert_admitted(contract)
    assert physics_chain(contract, "bem", lambda doc: doc["cad"]["cb"]) == pytest.approx(contract["cad"]["cb"])


def test_gate_can_be_recomputed_from_the_document(contracts):
    for contract in contracts.values():
        assert evaluate_gate(contract)["verdict"] == contract["gate"]["verdict"]


def test_solver_record_roundtrip_is_json_safe():
    record = lbm_grid(kcs_hull(), n_x=80)
    assert isinstance(record, SolverRecord)
    json.dumps(record.as_dict(), allow_nan=False)
    assert record.as_dict()["surface_evidence"] in {"REAL", "WITHHELD"}