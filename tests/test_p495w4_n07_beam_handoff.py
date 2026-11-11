"""Node p495w4-n07: FEM→DEM ``failure_faces`` hand-off and conservation closure.

The tests exercise the producer (Euler–Bernoulli beam FEM driven to fracture),
the frozen interface (``failure_faces`` v3 packet + sealed release sidecar), the
consumer (element-sized fragment cloud + rigid-fragment DEM) and the five gates
of the campaign node:

* mass residual < 1 %
* momentum residual < 2 %
* energy residual < 5 %
* frozen schema validation of ``failure_faces.json``
* KS test of the fragment-size distribution against the FEM element size (p > 0.05)

A reduced spec (Δx = 100 mm, L = 2 m → 960 fragments) keeps the suite fast; the
full six-case sweep is exercised once in ``test_sweep_smoke``.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from tensorphysics.failure_handoff.beam_cases import (
    ENERGY_TOLERANCE,
    release_velocity_dofs,
    FragmentCloud,
    build_fragment_cloud,
    energy_gate,
    mass_gate,
    mesh_convergence,
    momentum_gate,
    negative_controls,
    run_case,
    schema_gate,
    size_consistency_gate,
    summary,
    verify_against_beam_theory,
    write_case_artifacts,
)
from tensorphysics.failure_handoff.beam_fem import (
    BeamFractureSpec,
    BeamMesh,
    SIX_CASES,
    analytic_midspan_deflection,
    assemble_beam,
    element_kinetic_energy,
    element_strain_energy,
    newmark_dynamic_load,
    nodal_lumped_masses,
    packet_from_fracture,
    release_analysis,
    release_sidecar,
    resolve_fracture,
    self_weight_vector,
    solve_static,
    tension_stress,
)
from tensorphysics.failure_handoff.schema import verify_seal

SMALL = BeamFractureSpec(length_m=2.0, element_length_m=0.10, case_id="small_qs")


@pytest.fixture(scope="module")
def small_run():
    return run_case(SMALL)


# ---------------------------------------------------------------------------
# Specification and derived quantities
# ---------------------------------------------------------------------------

class TestSpecification:
    def test_mesh_counts(self):
        assert SMALL.n_elements == 20
        assert (SMALL.n_cells_y, SMALL.n_cells_z) == (8, 6)
        assert SMALL.cells_per_element == 48
        assert SMALL.n_fragments == 20 * 48

    def test_section_properties(self):
        assert SMALL.area_m2 == pytest.approx(0.48)
        assert SMALL.inertia_m4 == pytest.approx(0.6 * 0.8**3 / 12)
        assert SMALL.mass_kg == pytest.approx(917.0 * 0.48 * 2.0)
        assert SMALL.flexural_rigidity_pa_m4 == pytest.approx(9.0e9 * SMALL.inertia_m4)

    def test_failure_moment_and_displacement(self):
        moment = 2.0 * SMALL.tensile_strength_pa * SMALL.inertia_m4 / SMALL.height_m
        assert SMALL.failure_moment_n_m == pytest.approx(moment)
        expected = moment * SMALL.length_m**2 / (12.0 * SMALL.flexural_rigidity_pa_m4)
        assert SMALL.static_failure_displacement_m == pytest.approx(expected)

    def test_rejects_odd_element_count(self):
        with pytest.raises(ValueError):
            BeamFractureSpec(length_m=2.0, element_length_m=0.07)

    def test_rejects_non_integer_cross_section_cells(self):
        with pytest.raises(ValueError):
            BeamFractureSpec(length_m=2.0, element_length_m=0.07, height_m=0.8, width_m=0.6)

    def test_rejects_unknown_rate_kind(self):
        with pytest.raises(ValueError):
            BeamFractureSpec(length_m=2.0, rate_kind="blast")

    def test_rejects_too_coarse_elements(self):
        with pytest.raises(ValueError):
            BeamFractureSpec(length_m=2.0, height_m=0.8, width_m=0.6, element_length_m=0.5)

    def test_six_cases_cover_spans_and_rates(self):
        assert len(SIX_CASES) == 6
        assert sorted({c.length_m for c in SIX_CASES}) == [2.0, 4.0, 6.0]
        assert {c.rate_kind for c in SIX_CASES} == {"quasi_static", "dynamic"}


# ---------------------------------------------------------------------------
# FEM assembly, static solution, stress recovery
# ---------------------------------------------------------------------------

class TestBeamFiniteElements:
    def test_matrices_are_symmetric_and_mass_positive_definite(self):
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        K, M = assemble_beam(mesh, SMALL)
        assert K.shape == (mesh.n_dof, mesh.n_dof)
        np.testing.assert_allclose(K, K.T, atol=1e-6 * np.abs(K).max())
        np.testing.assert_allclose(M, M.T, atol=1e-12 * np.abs(M).max())
        assert np.all(np.linalg.eigvalsh(M) > 0.0)

    def test_free_free_stiffness_has_two_rigid_body_modes(self):
        mesh = BeamMesh(1.0, 10)
        K, M = assemble_beam(mesh, SMALL)
        eig = np.sort(np.abs(np.linalg.eigvals(np.linalg.solve(M, K))))
        assert eig[0] < 1e-8 * eig[2]
        assert eig[1] < 1e-8 * eig[2]
        assert math.sqrt(eig[2]) == pytest.approx(
            22.3733 * math.sqrt(SMALL.flexural_rigidity_pa_m4 / (SMALL.mass_per_length_kg_m * 1.0**4)),
            rel=1e-3,
        )

    def test_self_weight_load_is_balanced(self):
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        f = self_weight_vector(mesh, SMALL)
        assert f[0::2].sum() == pytest.approx(SMALL.weight_per_length_n_m * SMALL.length_m)

    def test_no_self_weight_option(self):
        spec = replace(SMALL, include_self_weight=False)
        mesh = BeamMesh(spec.length_m, spec.n_elements)
        assert np.allclose(self_weight_vector(mesh, spec), 0.0)

    def test_lumped_masses_sum_to_bar_mass(self):
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        assert nodal_lumped_masses(mesh, SMALL).sum() == pytest.approx(SMALL.mass_kg)

    def test_static_deflection_matches_closed_form(self):
        state, load = solve_static(SMALL, SMALL.static_failure_displacement_m)
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        fem = state.u[mesh.dof(mesh.n_nodes // 2, 0)]
        analytic = analytic_midspan_deflection(SMALL, load) + (
            5.0 * SMALL.weight_per_length_n_m * SMALL.length_m**4 / (384.0 * SMALL.flexural_rigidity_pa_m4)
        )
        assert abs(fem - analytic) / analytic < 1e-6

    def test_static_stress_matches_closed_form_within_3pct_on_the_production_mesh(self):
        production = BeamFractureSpec(length_m=2.0, element_length_m=0.05)
        verification = verify_against_beam_theory(production)
        assert verification["stress_relative_error"] < 0.03
        assert verification["frequency_relative_error"] < 1e-3
        assert verification["pass_3pct"] is True

    def test_stress_error_converges_with_mesh_refinement(self):
        rows = mesh_convergence(SMALL, element_lengths=(0.2, 0.1, 0.05))
        errors = [row["stress_relative_error"] for row in rows]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.03
        for row in rows:
            assert row["frequency_relative_error"] < 1e-3

    def test_tension_stress_sign_is_sagging_tension(self):
        state, _load = solve_static(SMALL, SMALL.static_failure_displacement_m)
        stress = tension_stress(BeamMesh(SMALL.length_m, SMALL.n_elements), SMALL, state.u)
        assert stress.max() > 0.0
        assert abs(stress).max() == pytest.approx(stress.max())

    def test_element_strain_energy_sums_to_state_energy(self):
        state, _load = solve_static(SMALL, SMALL.static_failure_displacement_m)
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        per_element = element_strain_energy(mesh, SMALL, state.u)
        assert per_element.sum() == pytest.approx(state.strain_energy_j)
        assert np.all(per_element >= 0.0)

    def test_element_kinetic_energy_is_positive_for_a_uniform_velocity(self):
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        v = np.zeros(mesh.n_dof)
        v[0::2] = 1.0
        per_element = element_kinetic_energy(mesh, SMALL, v)
        assert per_element.sum() == pytest.approx(0.5 * SMALL.mass_kg)


# ---------------------------------------------------------------------------
# Fracture detection
# ---------------------------------------------------------------------------

class TestFractureDetection:
    def test_static_fracture_reaches_the_tensile_strength(self):
        event = resolve_fracture(SMALL)
        assert event.mode == "quasi_static"
        assert event.state.max_tension_pa == pytest.approx(SMALL.tensile_strength_pa, rel=1e-6)
        assert 0.0 < event.imposed_displacement_m < 1.0

    def test_static_fracture_load_is_close_to_the_closed_form_load(self):
        # coarse mesh: the extreme-fibre stress is 4.9 % low, so the fracture load
        # is biased high by the same amount (documented mesh effect)
        event = resolve_fracture(SMALL)
        closed_form = SMALL.static_failure_load_n
        assert abs(event.load_n - closed_form) / closed_form < 0.10

    def test_static_fracture_load_on_the_production_mesh(self):
        production = BeamFractureSpec(length_m=2.0, element_length_m=0.05)
        event = resolve_fracture(production)
        closed_form = production.static_failure_load_n
        assert abs(event.load_n - closed_form) / closed_form < 0.05

    def test_static_fracture_has_no_kinetic_energy(self):
        event = resolve_fracture(SMALL)
        assert event.state.kinetic_energy_j == pytest.approx(0.0)

    def test_dynamic_fracture_carries_inertia(self):
        spec = replace(SMALL, rate_kind="dynamic", rise_time_s=1.0e-3, case_id="small_dyn")
        event = resolve_fracture(spec)
        assert event.mode == "dynamic"
        assert event.state.max_tension_pa == pytest.approx(spec.tensile_strength_pa, rel=1e-6)
        assert event.state.kinetic_energy_j > 0.0
        assert event.dynamic_amplification > 0.0

    def test_dynamic_amplification_grows_with_span(self):
        daes = []
        for length in (2.0, 6.0):
            spec = BeamFractureSpec(
                length_m=length, rate_kind="dynamic", rise_time_s=2.0e-3
            )
            daes.append(resolve_fracture(spec).dynamic_amplification)
        assert daes[1] > daes[0] > 0.5

    def test_force_ramp_returns_history_free_record(self):
        spec = replace(SMALL, rate_kind="dynamic", rise_time_s=2.0e-3)
        out = newmark_dynamic_load(spec)
        assert out["steps"] > 50
        assert out["dt_s"] > 0.0
        assert out["force_rate_n_per_s"] > 0.0


# ---------------------------------------------------------------------------
# Release dynamics
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release_conserves_mechanical_energy(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        assert release.mechanical_closure < 1e-3

    def test_release_kinetic_energy_exceeds_strain_energy(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        assert release.kinetic_energy_j > release.strain_energy_j

    def test_release_arrays_have_one_entry_per_element(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        assert release.element_velocity_m_s.size == SMALL.n_elements
        assert release.element_displacement_m.size == SMALL.n_elements
        assert release.element_strain_energy_j.size == SMALL.n_elements
        assert release.element_kinetic_energy_j.size == SMALL.n_elements

    def test_subgrid_kinetic_energy_is_small_and_positive(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        subgrid = release.element_subgrid_kinetic_j
        assert np.all(subgrid >= 0.0)
        assert subgrid.sum() < 0.5 * release.kinetic_energy_j

    def test_element_kinetic_plus_subgrid_equals_release_kinetic(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        lumped = 0.5 * SMALL.mass_per_length_kg_m * SMALL.element_length_m * np.sum(
            release.element_velocity_m_s**2
        )
        total = lumped + release.element_subgrid_kinetic_j.sum()
        assert total == pytest.approx(release.kinetic_energy_j, rel=1e-6)

    def test_velocity_reconstruction_round_trip(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        v = release_velocity_dofs(SMALL, release)
        mesh = BeamMesh(SMALL.length_m, SMALL.n_elements)
        assert v.size == mesh.n_dof
        assert np.all(np.isfinite(v))
        # nodal reconstruction re-averaged per element reproduces the element field
        nodal = v[0::2]
        for element in range(mesh.n_elements):
            approx = 0.5 * (nodal[element] + nodal[element + 1])
            assert approx == pytest.approx(release.element_velocity_m_s[element], abs=0.1 * (1.0 + abs(release.element_velocity_m_s[element])))


# ---------------------------------------------------------------------------
# Interface: packet + sidecar
# ---------------------------------------------------------------------------

class TestInterface:
    def test_packet_validates_against_the_frozen_schema(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        packet = packet_from_fracture(SMALL, event, release)
        validated = __import__(
            "tensorphysics.failure_handoff.schema", fromlist=["validate_packet"]
        ).validate_packet(packet)
        assert validated["version"] == "3.0.0"
        assert packet["face_count"] == SMALL.n_cells_y * SMALL.n_cells_z

    def test_packet_area_and_energy_totals(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        packet = packet_from_fracture(SMALL, event, release)
        assert packet["area_total_m2"] == pytest.approx(SMALL.area_m2)
        assert packet["energy_total_j"] == pytest.approx(SMALL.surface_energy_j, rel=1e-12)
        assert packet["fracture_layer_thickness_m"] == pytest.approx(SMALL.element_length_m)

    def test_packet_is_sealed_and_idempotent(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        first = packet_from_fracture(SMALL, event, release)
        second = packet_from_fracture(SMALL, event, release)
        assert first["sha256"] == second["sha256"]
        assert verify_seal(first)

    def test_packet_tamper_breaks_the_seal(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        packet = packet_from_fracture(SMALL, event, release)
        tampered = json.loads(json.dumps(packet))
        tampered["faces"][0]["area_m2"] *= 1.01
        assert not verify_seal(tampered)

    def test_sidecar_references_the_packet_digest(self):
        event = resolve_fracture(SMALL)
        release = release_analysis(SMALL, event)
        packet = packet_from_fracture(SMALL, event, release)
        sidecar = release_sidecar(SMALL, event, release, packet)
        assert sidecar["packet_sha256"] == packet["sha256"]
        assert verify_seal(sidecar)
        assert len(sidecar["element_velocity_m_s"]) == SMALL.n_elements


# ---------------------------------------------------------------------------
# Consumer: fragment cloud + gates
# ---------------------------------------------------------------------------

class TestConsumer:
    def test_cloud_counts_and_mass(self, small_run):
        cloud = small_run.cloud
        assert cloud.count == SMALL.n_fragments
        assert cloud.mass_total_kg == pytest.approx(SMALL.mass_kg, rel=1e-12)
        assert cloud.internal_energy_j > cloud.fracture_energy_j

    def test_surface_sink_preserves_momentum(self, small_run):
        sink = small_run.cloud.surface_sink
        assert 0.0 <= sink["surface_draw_ratio"] < 1.0
        assert 0.0 < sink["dispersive_scale_factor"] <= 1.0
        assert sink["mass_weighted_mean_after_sink_m_s"] == pytest.approx(
            sink["centre_of_mass_velocity_m_s"], abs=1e-12
        )

    def test_mass_gate_passes(self, small_run):
        assert small_run.gate("mass").passed
        assert small_run.gate("mass").relative_error < 0.01

    def test_momentum_gate_passes(self, small_run):
        gate = small_run.gate("momentum")
        assert gate.passed
        assert gate.relative_error < 0.02
        detail = gate.detail
        np.testing.assert_allclose(
            detail["dem_momentum_kg_m_s"], detail["fem_momentum_kg_m_s"], rtol=1e-6
        )

    def test_energy_gate_passes(self, small_run):
        gate = small_run.gate("energy")
        assert gate.passed
        assert gate.relative_error < ENERGY_TOLERANCE
        assert gate.detail["fe_release_closure"] < 1e-3

    def test_size_gate_passes(self, small_run):
        gate = small_run.gate("fragment_size_consistency")
        assert gate.passed
        assert gate.detail["ks_p_value"] > 0.05

    def test_schema_gate_passes(self, small_run):
        assert small_run.gate("failure_faces_schema").passed

    def test_case_passes_all_gates(self, small_run):
        assert small_run.passed
        criteria = small_run.criteria()
        assert criteria["all_passed"]
        assert criteria["schema_valid"]
        assert criteria["ks_passed"]

    def test_dem_integrator_conserves_energy(self, small_run):
        assert small_run.dem["energy_closure_with_dissipation"] < 1e-6
        assert small_run.dem["horizontal_momentum_residual"] < 1e-9
        assert small_run.dem["mass_total_kg"] == pytest.approx(SMALL.mass_kg, rel=1e-12)

    def test_dem_has_contact_events(self, small_run):
        assert small_run.dem["contact_events"] > 0
        assert small_run.dem["dissipated_j"] > 0.0


class TestNegativeControls:
    def test_every_control_turns_its_gate_red(self, small_run):
        controls = negative_controls(small_run)
        assert len(controls) >= 4
        for control in controls:
            assert control["gate_passed"] is False, control
            assert control["ok"] is True, control

    def test_dropped_fragments_break_the_mass_gate(self, small_run):
        # 1 % of 960 fragments = 9.6, so 10 lost fragments must trip the gate
        drop = 10
        reduced = FragmentCloud(
            fragments=small_run.cloud.fragments[drop:], spec=SMALL
        )
        assert not mass_gate(SMALL, reduced, small_run.packet).passed

    def test_single_lost_fragment_is_below_the_mass_resolution(self, small_run):
        """Honest resolution statement: one fragment is 0.1 % of the bar."""
        reduced = FragmentCloud(fragments=small_run.cloud.fragments[1:], spec=SMALL)
        gate = mass_gate(SMALL, reduced, small_run.packet)
        assert gate.relative_error < 0.01  # undetectable — documented limitation
        assert abs(gate.relative_error - 1.0 / small_run.cloud.count) < 1e-3

    def test_inflated_velocity_breaks_momentum_and_energy(self, small_run):
        inflated = FragmentCloud(
            fragments=[
                replace(f, velocity_m_s=1.2 * f.velocity_m_s)
                for f in small_run.cloud.fragments
            ],
            spec=SMALL,
        )
        assert not momentum_gate(SMALL, small_run.event, inflated, small_run.release).passed
        assert not energy_gate(
            SMALL, small_run.event, small_run.release, inflated, small_run.packet
        ).passed

    def test_coarse_fragments_break_the_ks_test(self, small_run):
        gate = size_consistency_gate(SMALL, small_run.cloud, scale=2.0)
        assert gate.detail["ks_p_value"] <= 0.05
        assert not gate.passed

    def test_broken_packet_fails_schema_gate(self, small_run):
        broken = json.loads(json.dumps(small_run.packet))
        broken["faces"] = broken["faces"][:-1]
        assert not schema_gate(broken).passed


# ---------------------------------------------------------------------------
# End-to-end, artifacts and reproducibility
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_summary_reports_all_criteria(self, small_run):
        spec_dyn = replace(SMALL, rate_kind="dynamic", rise_time_s=2.0e-3, case_id="small_dyn")
        runs = [small_run, run_case(spec_dyn)]
        doc = summary(runs)
        assert doc["case_count"] == 2
        assert doc["mass_all_pass"] and doc["momentum_all_pass"] and doc["energy_all_pass"]
        assert doc["schema_all_pass"] and doc["ks_all_pass"]
        assert doc["max_energy_residual"] < 0.05

    def test_artifacts_round_trip(self, small_run, tmp_path):
        paths = write_case_artifacts(small_run, tmp_path)
        assert set(paths) == {
            "failure_faces.json",
            "release_state.json",
            "conservation.json",
            "negative_controls.json",
        }
        on_disk = json.loads((tmp_path / "failure_faces.json").read_text())
        assert verify_seal(on_disk)
        from tensorphysics.failure_handoff.schema import validate_packet

        validate_packet(on_disk)
        ledger = json.loads((tmp_path / "conservation.json").read_text())
        assert ledger["criteria"]["all_passed"] is True

    def test_packet_digest_is_reproducible(self, small_run):
        again = run_case(SMALL)
        assert again.packet["sha256"] == small_run.packet["sha256"]
        assert again.release_doc["sha256"] == small_run.release_doc["sha256"]

    def test_cloud_is_rebuilt_from_the_serialised_interface(self, small_run):
        """The DEM side must work from the files alone, not from live objects."""
        rebuilt = build_fragment_cloud(
            SMALL, small_run.release, small_run.packet, small_run.release_doc
        )
        assert rebuilt.count == small_run.cloud.count
        assert rebuilt.mass_total_kg == pytest.approx(small_run.cloud.mass_total_kg)
        assert rebuilt.surface_sink["surface_energy_j"] == pytest.approx(
            small_run.packet["energy_total_j"]
        )


class TestSweepSmoke:
    def test_full_six_case_sweep_passes(self):
        runs = [run_case(spec) for spec in SIX_CASES]
        doc = summary(runs)
        assert doc["case_count"] == 6
        assert doc["all_passed"] is True
        assert doc["max_mass_residual"] < 0.01
        assert doc["max_momentum_residual"] < 0.02
        assert doc["max_energy_residual"] < 0.05
        assert doc["min_ks_p_value"] > 0.05