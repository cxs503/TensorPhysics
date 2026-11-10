"""End-to-end tests: three-point-bend FEM → ``failure_faces.json`` → DEM fragments → free fall.

Criterion coverage: C1 (area ≤1 %), C2 (energy ≤2 %), C3 (schema), C4 (energy chain
≤5 %), C5 (export idempotency), plus the FEM field certification against beam theory
that makes the extracted traction field trustworthy in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.failure_handoff import (
    RigidDEM,
    packet_from_result,
    packet_to_json_text,
    validate_packet,
    write_packet,
)


# ---------------------------------------------------------------------------
# FEM field certification (the input of the whole chain)
# ---------------------------------------------------------------------------

def test_beam_deflection_matches_timoshenko(beam_theory_validation) -> None:
    error = beam_theory_validation["deflection"]["relative_error_vs_timoshenko"]
    assert error < 0.03, beam_theory_validation["deflection"]
    # Euler-Bernoulli differs only by the shear term at L/h = 20
    assert beam_theory_validation["deflection"]["shear_contribution_ratio"] < 0.01


def test_beam_bending_stress_matches_theory(beam_theory_validation) -> None:
    error = beam_theory_validation["bottom_fibre_stress"]["relative_error"]
    # Linear Hex8 elements under-predict the bending stress on coarse meshes
    # (parasitic shear); the honest bound for this mesh is the 15 % acceptance band.
    assert error < 0.15, beam_theory_validation["bottom_fibre_stress"]
    assert error < 0.05, "expected better than 5 % at this resolution (see REPORT §6)"


def test_fem_strain_energy_matches_external_work(beam_theory_validation) -> None:
    assert beam_theory_validation["strain_energy_relative_error"] < 1e-6


# ---------------------------------------------------------------------------
# Extraction and packet
# ---------------------------------------------------------------------------

def test_packet_is_schema_valid(failure_handoff_run) -> None:
    packet = failure_handoff_run.packet
    assert validate_packet(packet)["version"] == "3.0.0"
    assert packet["face_count"] == len(packet["faces"]) == len(failure_handoff_run.fragment_set.fragments)
    assert packet["source_solver"].startswith("TensorFEM")
    assert packet["timestamp"] == failure_handoff_run.settings["timestamp"]


def test_ligament_is_partially_fractured(failure_handoff_run) -> None:
    diagnostics = failure_handoff_run.result.diagnostics
    fraction = diagnostics["ligament_failed_fraction"]
    assert 0.0 < fraction < 1.0, diagnostics
    assert diagnostics["candidate_face_count"] > diagnostics["failed_face_count"]


def test_strength_criterion_separates_failed_and_surviving_faces(failure_handoff_run) -> None:
    sigma_t = failure_handoff_run.result.material.sigma_t_pa
    for face in failure_handoff_run.result.faces:
        if face.failed:
            assert face.sigma_n_pa >= sigma_t
        else:
            assert face.sigma_n_pa < sigma_t


def test_crack_front_sits_inside_the_ligament(failure_handoff_run) -> None:
    result = failure_handoff_run.result
    front = np.asarray(result.crack_front_m)
    assert front.shape[0] >= 2
    failed = [face for face in result.faces if face.failed]
    surviving = [face for face in result.faces if not face.failed]
    # the front is the boundary between failed and surviving material: its height
    # must lie strictly inside the ligament (notch root < front < top fibre)
    y_front = float(front[:, 1].mean())
    assert result.material is not None
    assert min(face.vertices_m[:, 1].max() for face in failed) - 1e-9 <= y_front
    assert y_front < max(face.vertices_m[:, 1].max() for face in surviving) + 1e-9
    assert result.diagnostics["crack_front_edge_count"] >= 2


def test_failed_faces_are_nearly_mode_I(failure_handoff_run) -> None:
    """The ligament root of a bending specimen opens in mode I; shear is secondary."""
    assert failure_handoff_run.result.diagnostics["mean_mode_mixity_failed"] < 0.1


def test_specific_energy_brackets_the_bk_law(failure_handoff_run) -> None:
    material = failure_handoff_run.result.material
    for face in failure_handoff_run.packet["faces"]:
        assert material.G_Ic_j_m2 - 1e-9 <= face["specific_energy_j_m2"] <= material.G_IIc_j_m2 + 1e-9
        assert face["energy_j"] == pytest.approx(
            face["specific_energy_j_m2"] * face["area_m2"], rel=1e-12
        )


# ---------------------------------------------------------------------------
# C1 / C2 / C5
# ---------------------------------------------------------------------------

def test_c1_area_conservation(failure_handoff_run) -> None:
    criteria = failure_handoff_run.criteria()
    assert criteria["C1_area_pass"]
    assert criteria["C1_area_relative_error"] <= 0.01


def test_c2_energy_conservation(failure_handoff_run) -> None:
    criteria = failure_handoff_run.criteria()
    assert criteria["C2_energy_pass"]
    assert criteria["C2_energy_relative_error"] <= 0.02
    # the inherited internal energy is the sealed packet energy, bit for bit
    assert failure_handoff_run.fragment_set.internal_energy_total_j == pytest.approx(
        failure_handoff_run.packet["energy_total_j"], rel=1e-15
    )


def test_c5_export_is_idempotent(failure_handoff_run, tmp_path: Path) -> None:
    packet = failure_handoff_run.packet
    first = write_packet(packet, tmp_path / "a" / "failure_faces.json")
    second = write_packet(packet_from_result(failure_handoff_run.result), tmp_path / "b" / "failure_faces.json")
    text_a = (tmp_path / "a" / "failure_faces.json").read_text()
    text_b = (tmp_path / "b" / "failure_faces.json").read_text()
    assert first == second == packet["sha256"]
    assert text_a == text_b
    assert failure_handoff_run.idempotency["equal"]
    # the digest really is the digest of the file the consumer reads
    assert json.loads(text_a)["sha256"] == first


def test_packet_text_is_canonical(failure_handoff_run) -> None:
    text = packet_to_json_text(failure_handoff_run.packet)
    reparsed = json.loads(text)
    assert packet_to_json_text(reparsed) == text
    assert text.endswith("\n") and "\t" not in text


# ---------------------------------------------------------------------------
# C4: energy chain
# ---------------------------------------------------------------------------

def test_c4_energy_chain_closes(failure_handoff_run) -> None:
    history = failure_handoff_run.elastic_history
    assert history.initial_total > 0.0
    assert history.max_relative_deviation() <= 0.05
    assert failure_handoff_run.criteria()["C4_pass"]


def test_elastic_run_dissipates_nothing(failure_handoff_run) -> None:
    history = failure_handoff_run.elastic_history
    assert history.dissipated[-1] == 0.0
    assert history.contact_events > 0, "the fragments must actually hit the ground"


def test_dissipative_run_closes_its_ledger(failure_handoff_run) -> None:
    history = failure_handoff_run.dissipative_history
    # non-conservative by construction (e = 0.5, mu = 0.3) ...
    assert history.max_relative_deviation() > 0.05
    assert history.dissipated[-1] > 0.0
    # ... but every joule is accounted for: E(t) + dissipated = E(0)
    assert history.closure_with_dissipation() < 1e-9
    assert failure_handoff_run.criteria()["dissipative_closure"] < 1e-9


def test_energy_chain_order_is_physical(failure_handoff_run) -> None:
    """Fracture energy released must be smaller than the elastic energy available."""
    fem_strain_energy = failure_handoff_run.fem["strain_energy_j"]
    released = failure_handoff_run.packet["energy_total_j"]
    assert released < fem_strain_energy
    assert released / fem_strain_energy < 1.0


def test_free_fall_converts_potential_to_kinetic(failure_handoff_run) -> None:
    history = failure_handoff_run.elastic_history
    potential = history.potential
    kinetic = history.kinetic
    assert potential[0] > potential[-1]
    assert max(kinetic) > 0.5 * (potential[0] - potential[-1])


def test_internal_energy_is_constant_during_free_fall(failure_handoff_run) -> None:
    history = failure_handoff_run.elastic_history
    internal = history.internal
    assert max(internal) - min(internal) <= 1e-15 * max(internal)


# ---------------------------------------------------------------------------
# Determinism at the pipeline level
# ---------------------------------------------------------------------------

def test_pipeline_output_is_reproducible(failure_handoff_run) -> None:
    again = packet_from_result(failure_handoff_run.result)
    assert again["sha256"] == failure_handoff_run.packet["sha256"]
    assert again["faces"] == failure_handoff_run.packet["faces"]


def test_dem_release_is_deterministic(failure_handoff_run) -> None:
    first = RigidDEM(failure_handoff_run.fragment_set, dt_s=1.0e-3)
    first.lift_to(0.3)
    history_a = first.run(0.05, sample_every=1)
    second = RigidDEM(failure_handoff_run.fragment_set, dt_s=1.0e-3)
    second.lift_to(0.3)
    history_b = second.run(0.05, sample_every=1)
    assert history_a.total == pytest.approx(history_b.total, rel=1e-15)