"""Conservation-gate tests: do the area/energy/mass gates actually discriminate?

The gates are only meaningful if they (a) pass on the real FEM→DEM handoff and
(b) reject a lossy transfer.  Both directions are asserted here, including the
"inside tolerance" cases that must *not* fire.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tensorphysics.failure_handoff import (
    AREA_TOLERANCE,
    ENERGY_TOLERANCE,
    HandoffGateError,
    area_conservation_gate,
    clump_volume_gate,
    energy_conservation_gate,
    fragments_from_packet,
    mass_conservation_gate,
    per_face_area_gate,
    polygon_area,
    run_handoff_gates,
)
from tensorphysics.failure_handoff.pipeline import gate_sensitivity


def test_tolerances_match_the_contract() -> None:
    assert AREA_TOLERANCE == pytest.approx(0.01)
    assert ENERGY_TOLERANCE == pytest.approx(0.02)


def test_all_gates_pass_on_fem_handoff(failure_handoff_run) -> None:
    report = run_handoff_gates(failure_handoff_run.packet, failure_handoff_run.fragment_set)
    assert report.passed, report.failures
    area = next(g for g in report.gates if g.name == "area")
    energy = next(g for g in report.gates if g.name == "energy")
    assert area.relative_error <= AREA_TOLERANCE
    assert energy.relative_error <= ENERGY_TOLERANCE
    assert area.producer_value == pytest.approx(area.consumer_value, rel=1e-12)
    assert energy.producer_value == pytest.approx(energy.consumer_value, rel=1e-12)


def test_area_gate_detects_lossy_triangulation(failure_handoff_run) -> None:
    """A consumer that keeps 3 of 4 vertices loses half the area → gate must fail."""
    fragments = failure_handoff_run.fragment_set.fragments
    halved = [replace(f, area_m2=f.area_m2 * 0.5) for f in fragments]
    mutated = replace(failure_handoff_run.fragment_set, fragments=halved)
    gate = area_conservation_gate(failure_handoff_run.packet, mutated)
    assert gate.relative_error == pytest.approx(0.5, rel=1e-9)
    assert not gate.passed


def test_per_face_gate_localises_the_worst_face(failure_handoff_run) -> None:
    fragments = failure_handoff_run.fragment_set.fragments
    if len(fragments) < 2:
        pytest.skip("needs at least two fragments")
    perturbed = list(fragments)
    perturbed[1] = replace(perturbed[1], area_m2=perturbed[1].area_m2 * 1.03)
    mutated = replace(failure_handoff_run.fragment_set, fragments=perturbed)
    gate = per_face_area_gate(failure_handoff_run.packet, mutated)
    assert not gate.passed
    assert gate.detail["worst_face"] == fragments[1].face_id


def test_energy_gate_detects_three_percent_loss(failure_handoff_run) -> None:
    fragments = failure_handoff_run.fragment_set.fragments
    low = [replace(f, fracture_energy_j=f.fracture_energy_j * 0.97) for f in fragments]
    mutated = replace(failure_handoff_run.fragment_set, fragments=low)
    gate = energy_conservation_gate(failure_handoff_run.packet, mutated)
    assert gate.relative_error == pytest.approx(0.03, rel=1e-9)
    assert not gate.passed


def test_energy_gate_accepts_inside_tolerance(failure_handoff_run) -> None:
    fragments = failure_handoff_run.fragment_set.fragments
    low = [replace(f, fracture_energy_j=f.fracture_energy_j * 0.985) for f in fragments]
    mutated = replace(failure_handoff_run.fragment_set, fragments=low)
    gate = energy_conservation_gate(failure_handoff_run.packet, mutated)
    assert gate.passed and gate.relative_error == pytest.approx(0.015, rel=1e-9)


def test_dropped_fragment_is_a_structural_error(failure_handoff_run) -> None:
    fragments = failure_handoff_run.fragment_set.fragments
    mutated = replace(failure_handoff_run.fragment_set, fragments=list(fragments[:-1]))
    with pytest.raises(HandoffGateError, match="fragments from"):
        area_conservation_gate(failure_handoff_run.packet, mutated)


def test_sensitivity_matrix_covers_both_verdicts(failure_handoff_run) -> None:
    cases = {c["case"]: c for c in gate_sensitivity(
        failure_handoff_run.packet, failure_handoff_run.fragment_set
    )}
    assert cases["control_unchanged"]["area_passed"] is True
    assert cases["control_unchanged"]["energy_passed"] is True
    assert cases["fragment_dropped"]["outcome"] == "structural_error"
    assert cases["faces_triangulated_lossily"]["area_passed"] is False
    assert cases["area_offset_0p5pct"]["area_passed"] is True
    assert cases["area_offset_1p5pct"]["area_passed"] is False
    assert cases["energy_offset_3pct"]["energy_passed"] is False
    assert cases["energy_offset_1p5pct"]["energy_passed"] is True


def test_mass_gate_is_exact(failure_handoff_run) -> None:
    gate = mass_conservation_gate(
        failure_handoff_run.packet, failure_handoff_run.fragment_set
    )
    assert gate.passed
    assert gate.relative_error <= 1e-12
    rho = failure_handoff_run.fragment_set.density_kg_m3
    expected = (
        rho
        * failure_handoff_run.fragment_set.area_total_m2
        * failure_handoff_run.fragment_set.layer_thickness_m
    )
    assert gate.producer_value == pytest.approx(expected, rel=1e-15)


def test_clump_volume_partition_is_exact(failure_handoff_run) -> None:
    clumped = fragments_from_packet(
        failure_handoff_run.packet,
        density_kg_m3=failure_handoff_run.fragment_set.density_kg_m3,
        layer_thickness_m=failure_handoff_run.fragment_set.layer_thickness_m,
        clump_count=4,
    )
    assert all(len(f.clump_spheres_m) == 4 for f in clumped.fragments)
    gate = clump_volume_gate(failure_handoff_run.packet, clumped)
    assert gate.passed and gate.detail["clump_requested"] is True
    assert clumped.clump_volume_total_m3 == pytest.approx(clumped.volume_total_m3, rel=1e-12)
    # masses are unchanged by the clump representation
    assert clumped.mass_total_kg == pytest.approx(
        failure_handoff_run.fragment_set.mass_total_kg, rel=1e-15
    )


def test_fragment_area_is_recomputed_from_vertices(failure_handoff_run) -> None:
    """The consumer's area comes from the coordinates, not from the declared field."""
    packet = failure_handoff_run.packet
    for face, fragment in zip(packet["faces"], failure_handoff_run.fragment_set.fragments):
        assert fragment.area_m2 == pytest.approx(polygon_area(face["vertices_m"]), rel=1e-15)
        assert fragment.area_m2 == pytest.approx(fragment.area_m2_triangulated, rel=1e-12)
        assert fragment.volume_m3 == pytest.approx(
            fragment.area_m2 * fragment.thickness_m, rel=1e-15
        )
        assert fragment.mass_kg == pytest.approx(
            failure_handoff_run.fragment_set.density_kg_m3 * fragment.volume_m3, rel=1e-15
        )
        assert fragment.fracture_energy_j == pytest.approx(
            face["energy_j"], rel=1e-15
        )


def test_fragment_geometry_matches_face_geometry(failure_handoff_run) -> None:
    """Each prism keeps the FEM polygon as its fracture face (by construction)."""
    for fragment in failure_handoff_run.fragment_set.fragments:
        normal = fragment.normal
        assert abs(float(np.linalg.norm(normal)) - 1.0) < 1e-12
        # the prism is built by extruding the face along -normal
        assert np.all(
            (fragment.fracture_polygon_m - fragment.fracture_polygon_m.mean(axis=0)) @ normal
            == pytest.approx(0.0, abs=1e-12)
        )
        assert fragment.centroid_m[0] == pytest.approx(
            fragment.fracture_polygon_m[:, 0].mean() - 0.5 * fragment.thickness_m * normal[0]
        )


def test_equivalent_radius_matches_volume(failure_handoff_run) -> None:
    for fragment in failure_handoff_run.fragment_set.fragments:
        volume = 4.0 / 3.0 * math.pi * fragment.equivalent_radius_m**3
        assert volume == pytest.approx(fragment.volume_m3, rel=1e-12)