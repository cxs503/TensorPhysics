"""Tests for FEM→DEM ice fragment data export interface (SIR-PC2 / F03).

Acceptance criteria (all must pass):
1. dem_loadable == True  (bool)
2. mass_conservation_error < 1e-12  (kg)
3. round_trip_energy_conservation < 1e-08  (ratio)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.ice_fragments import (
    SCHEMA_VERSION,
    MASS_CONSERVATION_TOL,
    ENERGY_CONSERVATION_TOL,
    DEFAULT_ICE_DENSITY,
    ICE_FRAGMENTS_JSON_SCHEMA,
    IceFragment,
    IceFragmentsPacket,
    MassConservationError,
    ice_fragments,
    volume_to_sphere_radius,
    face_volume,
    packet_to_json_dict,
    load_packet_from_json,
    save_packet,
    load_packet_file,
    validate_packet,
    make_test_failed_faces,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_failed_faces():
    """6-face test case on a 3×3 mesh."""
    faces, nodes = make_test_failed_faces(n_faces=6, side=1.0, thickness=0.01, G_c=100.0)
    return faces, nodes


@pytest.fixture
def large_failed_faces():
    """Larger test case: 20 faces."""
    faces, nodes = make_test_failed_faces(n_faces=20, side=2.0, thickness=0.02, G_c=200.0)
    return faces, nodes


@pytest.fixture
def single_face():
    """Single face for precision testing."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = [{
        "face_id": 0,
        "node_i": 1,
        "node_j": 2,
        "area": 0.01,   # 1.0 × 0.01
        "length": 1.0,
        "released_energy": 1.0,  # G_c=100 × 0.01
    }]
    return faces, nodes


# ---------------------------------------------------------------------------
# Test 1: Basic packet generation
# ---------------------------------------------------------------------------

class TestIceFragmentsBasic:
    """Basic functionality tests."""

    def test_packet_has_correct_schema_version(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        assert packet.schema_version == SCHEMA_VERSION

    def test_packet_has_fragments(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        assert len(packet.fragments) == len(faces)

    def test_fragment_ids_sequential(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        ids = [f.fragment_id for f in packet.fragments]
        assert ids == list(range(len(faces)))

    def test_fragment_positions_are_3d(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        for frag in packet.fragments:
            assert len(frag.position_m) == 3
            assert frag.position_m[2] == 0.0  # mid-plane

    def test_fragment_radii_positive(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        for frag in packet.fragments:
            assert frag.radius_m > 0.0

    def test_fragment_masses_positive(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        for frag in packet.fragments:
            assert frag.mass_kg > 0.0


# ---------------------------------------------------------------------------
# Test 2: Mass conservation gate (acceptance criterion #2)
# ---------------------------------------------------------------------------

class TestMassConservation:
    """Mass conservation gate tests."""

    def test_mass_conservation_exact(self, simple_failed_faces):
        """FEM mass == DEM mass (same formula → machine precision)."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        mass_error = abs(packet.fem_total_mass_kg - packet.dem_total_mass_kg)
        assert mass_error < 1e-12, f"mass_error = {mass_error}"

    def test_mass_conservation_single_face(self, single_face):
        faces, nodes = single_face
        packet = ice_fragments(faces, nodes)
        mass_error = abs(packet.fem_total_mass_kg - packet.dem_total_mass_kg)
        assert mass_error == 0.0  # identical formula → exact

    def test_mass_conservation_large_case(self, large_failed_faces):
        faces, nodes = large_failed_faces
        packet = ice_fragments(faces, nodes)
        mass_error = abs(packet.fem_total_mass_kg - packet.dem_total_mass_kg)
        assert mass_error < 1e-12

    def test_mass_conservation_error_below_threshold(self, simple_failed_faces):
        """Acceptance criterion: mass_conservation_error < 1e-12 kg."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)
        assert report["mass_conservation_error_kg"] < 1e-12

    def test_empty_faces_zero_mass(self):
        """No faces → zero mass, no error."""
        packet = ice_fragments([], np.zeros((4, 2)))
        assert packet.fem_total_mass_kg == 0.0
        assert packet.dem_total_mass_kg == 0.0


# ---------------------------------------------------------------------------
# Test 3: Round-trip energy conservation (acceptance criterion #3)
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Round-trip energy conservation tests."""

    def test_energy_conservation_simple(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        ratio = abs(packet.fem_total_energy_J - packet.dem_total_energy_J)
        if packet.dem_total_energy_J > 0:
            ratio /= packet.dem_total_energy_J
        assert ratio < 1e-12

    def test_round_trip_json_energy(self, simple_failed_faces):
        """Serialise → deserialise → energy preserved."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)

        # Round-trip through JSON
        json_dict = packet_to_json_dict(packet)
        json_str = json.dumps(json_dict)
        recovered = json.loads(json_str)
        recovered_packet = load_packet_from_json(recovered)

        original_energy = sum(f.fracture_energy_J for f in packet.fragments)
        recovered_energy = sum(f.fracture_energy_J for f in recovered_packet.fragments)

        if original_energy > 0:
            rel_error = abs(recovered_energy - original_energy) / original_energy
            assert rel_error < 1e-8, f"round-trip energy error: {rel_error}"

    def test_round_trip_energy_ratio_below_threshold(self, simple_failed_faces):
        """Acceptance criterion: round_trip_energy_conservation < 1e-08."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)
        assert report["round_trip_energy_conservation"] < 1e-8

    def test_round_trip_file_energy(self, simple_failed_faces):
        """Save to file → load from file → energy preserved."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_packet.json")
            save_packet(packet, path)
            loaded = load_packet_file(path)

        original_energy = sum(f.fracture_energy_J for f in packet.fragments)
        loaded_energy = sum(f.fracture_energy_J for f in loaded.fragments)

        if original_energy > 0:
            rel_error = abs(loaded_energy - original_energy) / original_energy
            assert rel_error < 1e-8


# ---------------------------------------------------------------------------
# Test 4: DEM-loadable (acceptance criterion #1)
# ---------------------------------------------------------------------------

class TestDEMLoadable:
    """DEM-loadable tests."""

    def test_dem_loadable_simple(self, simple_failed_faces):
        """Acceptance criterion: dem_loadable == True."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)
        assert report["dem_loadable"] is True

    def test_dem_loadable_single_face(self, single_face):
        faces, nodes = single_face
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)
        assert report["dem_loadable"] is True

    def test_dem_loadable_large_case(self, large_failed_faces):
        faces, nodes = large_failed_faces
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)
        assert report["dem_loadable"] is True

    def test_dem_loadable_empty(self):
        packet = ice_fragments([], np.zeros((4, 2)))
        report = validate_packet(packet)
        assert report["dem_loadable"] is True

    def test_json_round_trip_preserves_fragment_count(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)
        json_str = json.dumps(json_dict)
        recovered = json.loads(json_str)
        assert len(recovered["fragments"]) == len(packet.fragments)

    def test_json_round_trip_preserves_mass(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)
        json_str = json.dumps(json_dict)
        recovered = json.loads(json_str)
        recovered_packet = load_packet_from_json(recovered)

        original_mass = sum(f.mass_kg for f in packet.fragments)
        recovered_mass = sum(f.mass_kg for f in recovered_packet.fragments)
        assert abs(recovered_mass - original_mass) < 1e-12


# ---------------------------------------------------------------------------
# Test 5: JSON Schema compliance
# ---------------------------------------------------------------------------

class TestJSONSchema:
    """JSON schema tests."""

    def test_schema_has_required_fields(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)

        assert "schema_version" in json_dict
        assert "timestamp" in json_dict
        assert "packet_id" in json_dict
        assert "material" in json_dict
        assert "conservation" in json_dict
        assert "fragments" in json_dict

    def test_material_fields(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)

        mat = json_dict["material"]
        assert "density_kg_m3" in mat
        assert "fracture_energy_density_J_m2" in mat
        assert mat["density_kg_m3"] > 0

    def test_conservation_fields(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)

        cons = json_dict["conservation"]
        assert "fem_total_mass_kg" in cons
        assert "dem_total_mass_kg" in cons
        assert "mass_conservation_error_kg" in cons
        assert "fem_total_energy_J" in cons
        assert "dem_total_energy_J" in cons
        assert "round_trip_energy_ratio" in cons

    def test_fragment_fields(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)

        for frag in json_dict["fragments"]:
            assert "fragment_id" in frag
            assert "position_m" in frag
            assert "radius_m" in frag
            assert "mass_kg" in frag
            assert "velocity_m_s" in frag
            assert "source_face_id" in frag
            assert "source_volume_m3" in frag
            assert "fracture_energy_J" in frag
            assert len(frag["position_m"]) == 3
            assert len(frag["velocity_m_s"]) == 3

    def test_schema_version_matches(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        json_dict = packet_to_json_dict(packet)
        assert json_dict["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Test 6: Physics correctness
# ---------------------------------------------------------------------------

class TestPhysics:
    """Physics correctness tests."""

    def test_volume_to_sphere_radius(self):
        V = 1.0
        r = volume_to_sphere_radius(V)
        expected = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        assert abs(r - expected) < 1e-12

    def test_volume_to_sphere_radius_scaling(self):
        r1 = volume_to_sphere_radius(1.0)
        r2 = volume_to_sphere_radius(8.0)
        assert abs(r2 / r1 - 2.0) < 1e-12

    def test_face_volume_formula(self):
        assert face_volume(0.01, 1.0) == 0.01
        assert face_volume(0.005, 0.5) == 0.0025

    def test_particle_mass_equals_density_times_volume(self, single_face):
        faces, nodes = single_face
        density = 917.0
        packet = ice_fragments(faces, nodes, density=density)
        frag = packet.fragments[0]
        expected_vol = faces[0]["area"] * faces[0]["length"]
        expected_mass = density * expected_vol
        assert abs(frag.mass_kg - expected_mass) < 1e-12

    def test_particle_radius_from_volume(self, single_face):
        faces, nodes = single_face
        packet = ice_fragments(faces, nodes)
        frag = packet.fragments[0]
        expected_vol = faces[0]["area"] * faces[0]["length"]
        expected_radius = volume_to_sphere_radius(expected_vol)
        assert abs(frag.radius_m - expected_radius) < 1e-12

    def test_total_mass_equals_sum_of_fragments(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        sum_mass = sum(f.mass_kg for f in packet.fragments)
        assert abs(sum_mass - packet.dem_total_mass_kg) < 1e-12

    def test_total_energy_equals_sum_of_fragments(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        sum_energy = sum(f.fracture_energy_J for f in packet.fragments)
        assert abs(sum_energy - packet.dem_total_energy_J) < 1e-12


# ---------------------------------------------------------------------------
# Test 7: File I/O
# ---------------------------------------------------------------------------

class TestFileIO:
    """File I/O tests."""

    def test_save_and_load(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ice_fragments.json")
            save_packet(packet, path)
            loaded = load_packet_file(path)

        assert len(loaded.fragments) == len(packet.fragments)
        assert loaded.schema_version == packet.schema_version

    def test_saved_file_is_valid_json(self, simple_failed_faces):
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ice_fragments.json")
            save_packet(packet, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "fragments" in data
            assert "conservation" in data


# ---------------------------------------------------------------------------
# Test 8: Acceptance criteria integration
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    """Integration test for all three acceptance criteria."""

    def test_all_criteria_pass_simple(self, simple_failed_faces):
        """All three acceptance criteria must pass."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)

        # Criterion 1: dem_loadable == True
        assert report["dem_loadable"] is True

        # Criterion 2: mass_conservation_error < 1e-12 kg
        assert report["mass_conservation_error_kg"] < 1e-12

        # Criterion 3: round_trip_energy_conservation < 1e-08
        assert report["round_trip_energy_conservation"] < 1e-8

        # Overall status
        assert report["status"] == "PASS"

    def test_all_criteria_pass_large(self, large_failed_faces):
        faces, nodes = large_failed_faces
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)

        assert report["dem_loadable"] is True
        assert report["mass_conservation_error_kg"] < 1e-12
        assert report["round_trip_energy_conservation"] < 1e-8
        assert report["status"] == "PASS"

    def test_all_criteria_pass_single(self, single_face):
        faces, nodes = single_face
        packet = ice_fragments(faces, nodes)
        report = validate_packet(packet)

        assert report["dem_loadable"] is True
        assert report["mass_conservation_error_kg"] < 1e-12
        assert report["round_trip_energy_conservation"] < 1e-8
        assert report["status"] == "PASS"

    def test_full_pipeline_with_file_roundtrip(self, simple_failed_faces):
        """Full pipeline: generate → save → load → validate."""
        faces, nodes = simple_failed_faces
        packet = ice_fragments(faces, nodes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pipeline_test.json")
            save_packet(packet, path)
            loaded = load_packet_file(path)

        report = validate_packet(loaded)
        assert report["status"] == "PASS"
        assert report["dem_loadable"] is True
        assert report["mass_conservation_error_kg"] < 1e-12
        assert report["round_trip_energy_conservation"] < 1e-8
