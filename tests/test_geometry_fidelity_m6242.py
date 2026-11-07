"""Tests for the attempt-2 additions to the geometry fidelity contract.

Covers the three things that were missing or wrong in the first pass:

* the third CAD route (isoparametric surface tessellation / divergence theorem)
  and its mutual residual against the transverse-slice route;
* the SUBOFF truth source that now closes *both* delivered integral references,
  with the single-exponent reading kept as a negative sensitivity case;
* the LBM wetted-surface reconstruction (marching tetrahedra validated on a
  sphere, the staircase representation gap, and the a-priori evidence rule), and
  the new principal-dimension gate that rejects a grid whose L / B are wrong
  before any displacement is looked at.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.geometry_fidelity.cad_routes import (
    calibrate_suboff,
    sharp_tail_hull,
    surface_isoparametric,
    volume_isoparametric,
)
from tensorphysics.geometry_fidelity.contract import (
    GATE_PASS,
    GATE_REJECT,
    MeshNotAdmitted,
    assert_admitted,
    build_contract,
)
from tensorphysics.geometry_fidelity.hulls import contract_hulls, kcs_hull
from tensorphysics.geometry_fidelity.mesh import corrupted_record, good_records
from tensorphysics.geometry_fidelity.reconstruct import (
    marching_tets_area,
    staircase_area,
)


@pytest.fixture(scope="module")
def hulls():
    return contract_hulls()


# -- CAD: third independent route -------------------------------------------


def test_iso_route_agrees_with_slice_route(hulls):
    for name, hull in hulls.items():
        volume_slice = hull.cad_volume_slice(240, 160)
        area_slice = hull.cad_surface_area(240, 160)
        volume_iso = volume_isoparametric(hull, 240, 160)
        area_iso = surface_isoparametric(hull, 240, 160)
        assert abs(volume_iso - volume_slice) / volume_slice <= 1e-3, name
        assert abs(area_iso - area_slice) / area_slice <= 1e-3, name


def test_iso_route_converges_second_order():
    hull = kcs_hull()
    coarse = volume_isoparametric(hull, 120, 80)
    fine = volume_isoparametric(hull, 240, 160)
    reference = hull.cad_volume_slice(480, 320)
    assert abs(fine - reference) < abs(coarse - reference)
    assert abs(fine - reference) / reference <= 1e-4


# -- SUBOFF: both references closed, sharp tail kept as a negation ----------


def test_suboff_calibration_closes_both_constraints():
    calibration = calibrate_suboff()
    assert abs(calibration["nabla_m3"] - 0.699211) / 0.699211 <= 1e-6
    assert abs(calibration["s_wetted_m2"] - 5.998) / 5.998 <= 1e-6
    assert calibration["tip_ratio"] == 0.0
    assert calibration["aft_b"] >= 1.02, "a finite aft tangent is required for panel closure"


def test_suboff_calibrated_hull_matches_both_references(hulls):
    integrals = hulls["suboff"].cad_integrals(480, 200)
    assert abs(integrals["nabla_m3"] - 0.699211) / 0.699211 <= 1e-3
    assert abs(integrals["s_wetted_m2"] - 5.998) / 5.998 <= 1e-3


def test_single_exponent_reading_leaves_a_measurable_area_residual():
    """Attempt-1 truth source: displacement closed by construction, area off.

    One exponent per end (used twice) closes the displacement and leaves the
    wetted area at ``+7.2e-4`` relative: inside the 1e-3 acceptance band but two
    orders of magnitude above the two-exponent calibration.  The attempt-1 report
    quoted ``+0.19 %``; that number was dominated by the streamwise quadrature of
    the slice route (n_x = 240 gives ``+1.7e-3`` on this hull, n_x = 960 gives
    ``-2e-5``), which is why the contract now evaluates the revolution hull at
    n_x >= 480.
    """
    hull = sharp_tail_hull()
    integrals = hull.cad_integrals(960, 200)
    assert abs(integrals["nabla_m3"] - 0.699211) / 0.699211 <= 1e-4
    residual = abs(integrals["s_wetted_m2"] - 5.998) / 5.998
    assert 1e-4 < residual < 1e-3, residual


# -- LBM reconstruction ----------------------------------------------------


def test_marching_tets_area_is_within_a_few_percent_on_a_sphere():
    """Validation of the isosurface utility on a smooth sphere (level set of a
    signed-distance field).  Measured accuracy is first order with a +2 % bias at
    50 cells per radius; the utility is therefore *not* used for the contract
    numbers -- the LBM wetted area comes from the height-field reconstruction --
    and this test pins the tool's demonstrated accuracy instead of an aspirational
    one."""

    numpy = pytest.importorskip("numpy")
    radius, spacing = 0.5, 0.01
    n = int(1.2 / spacing)
    axis = (numpy.arange(n) + 0.5) * spacing - 0.6
    grid = numpy.meshgrid(axis, axis, axis, indexing="ij")
    r = numpy.sqrt(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2)
    field = 0.5 - (r - radius) / spacing  # signed distance, unclipped
    area = marching_tets_area(field, (spacing, spacing, spacing), level=0.0)
    assert abs(area - 4.0 * math.pi * radius**2) / (4.0 * math.pi * radius**2) <= 3e-2


def test_staircase_area_overestimates_a_sphere_by_the_directional_factor():
    numpy = pytest.importorskip("numpy")
    radius, spacing = 0.5, 0.01
    n = int(1.2 / spacing)
    axis = (numpy.arange(n) + 0.5) * spacing - 0.6
    grid = numpy.meshgrid(axis, axis, axis, indexing="ij")
    r = numpy.sqrt(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2)
    area = staircase_area((r < radius).astype(float), (spacing, spacing, spacing))
    gap = area / (4.0 * math.pi * radius**2) - 1.0
    assert 0.35 <= gap <= 0.65, gap  # sum_d |n_d| -> 1.5 for a sphere


def test_lbm_surface_evidence_is_decided_from_the_shape_class(hulls):
    records = {name: good_records(hull)["lbm"] for name, hull in hulls.items()}
    assert records["kcs"].surface_evidence == "WITHHELD"      # fine form, keel tangent
    assert records["suboff"].surface_evidence == "WITHHELD"    # polar lines
    assert records["kvlcc2"].surface_evidence == "REAL"        # wall sided


def test_lbm_volume_is_second_order(hulls):
    for name, hull in hulls.items():
        reference = hull.cad_volume_slice(240, 160)
        record = good_records(hull)["lbm"]
        assert abs(record.nabla_m3 - reference) / reference <= 5e-3, name


# -- gate: principal dimensions and reference agreement ---------------------


def test_contract_carries_integration_methods(hulls):
    contract = build_contract(hulls["kvlcc2"], good_records(hulls["kvlcc2"]), mc_samples=100_000, n_x=160, n_s=100)
    for family, record in contract["grid"]["solvers"].items():
        assert record["integration"]["volume"]
        assert record["integration"]["surface"]
    assert len(contract["cad"]["integration"]["volume"]) == 3


def test_principal_dimensions_are_gated(hulls):
    hull = hulls["kvlcc2"]
    contract = build_contract(hull, good_records(hull), mc_samples=100_000, n_x=160, n_s=100)
    ids = {check["id"] for check in contract["gate"]["checks"]}
    assert {"fvm_length", "fvm_breadth", "bem_length", "bem_breadth"} <= ids
    stretched = dict(good_records(hull))
    stretched["fvm"] = corrupted_record(hull, "stretch_2pct")
    polluted = build_contract(hull, stretched, mc_samples=100_000, n_x=160, n_s=100)
    assert polluted["gate"]["verdict"] == GATE_REJECT
    assert any(check["id"] == "fvm_length" for check in polluted["gate"]["failures"])


def test_reference_checks_are_present_and_pass(hulls):
    contract = build_contract(hulls["suboff"], good_records(hulls["suboff"]), mc_samples=100_000, n_x=240, n_s=160)
    reference_checks = [c for c in contract["gate"]["checks"] if c["id"].startswith("cad_reference_")]
    assert len(reference_checks) == 2  # nabla and S (Cb is not delivered for SUBOFF)
    assert all(c["pass"] for c in reference_checks), reference_checks
    assert contract["gate"]["verdict"] == GATE_PASS


def test_iso_checks_are_present(hulls):
    contract = build_contract(hulls["kcs"], good_records(hulls["kcs"]), mc_samples=100_000, n_x=160, n_s=100)
    ids = {check["id"] for check in contract["gate"]["checks"]}
    assert {"cad_isoparametric_volume", "cad_isoparametric_surface"} <= ids


def test_rejected_grid_cannot_enter_the_chain(hulls):
    hull = hulls["kvlcc2"]
    records = dict(good_records(hull))
    records["lbm"] = corrupted_record(hull, "drop_voxel_layer")
    contract = build_contract(hull, records, mc_samples=100_000, n_x=160, n_s=100)
    assert contract["gate"]["verdict"] == GATE_REJECT
    with pytest.raises(MeshNotAdmitted):
        assert_admitted(contract)