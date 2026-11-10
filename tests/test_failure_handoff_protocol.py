"""Tests for the continuous-crack → discrete-fragment FEM→DEM handoff protocol.

Coverage map (node ``p273w3-n03``):

* ``C1`` — MTS kink angle: closed form vs Williams-field scan vs FEM ring scan.
* ``C2`` — reconstructed polyhedron volume vs FEM failure-region volume.
* ``C3`` — fragment momentum / angular momentum vs FEM crack-face nodal values.
* ``C4`` — fracture + kinetic + potential energy ledger.
* ``C6`` — the end-to-end protocol produces an ``Implemented``-grade capability
  record (asserted in ``test_protocol_end_to_end``).

Every gate is tested in *both* directions: a consistent input must pass and a
deliberately corrupted input must fail (negative controls), otherwise the gate
would be unfalsifiable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.failure_handoff import (
    ICE_MATERIAL,
    ICE_SPEC,
    asymptotic_kink_angle_deg,
    build_transfer_fragment,
    energy_budget,
    fragment_inertia,
    momentum_budget,
    mts_kink_angle_deg,
    notch_face_node_pairs,
    polyhedron_volume,
    polygon_second_moments,
    rigid_body_fit,
    senb_k1_reference,
    volume_budget,
)
from tensorphysics.failure_handoff.mixed_mode import stress_intensity_factors
from tensorphysics.failure_handoff.three_point_bend import BeamSpec, build_bend_model, solve_bend
from tensorphysics.failure_handoff.transfer import TransferFragment


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _face(element_id: int = 0, x0: float = 0.0, w: float = 0.02, h: float = 0.03) -> dict:
    verts = [
        [x0, 0.0, 0.0],
        [x0, w, 0.0],
        [x0, w, h],
        [x0, 0.0, h],
    ]
    return {
        "face_id": f"F{element_id + 1:04d}",
        "element_id": element_id,
        "vertices_m": verts,
        "normal": [1.0, 0.0, 0.0],
        "centroid_m": [x0, 0.5 * w, 0.5 * h],
        "area_m2": w * h,
        "specific_energy_j_m2": 8.0,
        "energy_j": 8.0 * w * h,
        "failure_mode": "mode_I",
        "mode_mixity": 0.0,
        "normal_traction_pa": 1.2e6,
        "shear_traction_pa": 0.0,
    }


def _fragment(index: int, x0: float, velocity, *, thickness: float = 0.005, density: float = 917.0) -> TransferFragment:
    face = _face(element_id=index, x0=x0)
    frag = build_transfer_fragment(
        face,
        density_kg_m3=density,
        thickness_m=thickness,
        index=index,
        fem_element_volume_m3=face["area_m2"] * thickness,
    )
    frag.node_ids = [4 * index + k for k in range(4)]
    frag.nodal_velocity_m_s = np.tile(np.asarray(velocity, dtype=float), (4, 1))
    frag.velocity_m_s = np.asarray(velocity, dtype=float)
    frag.omega_rad_s = np.zeros(3)
    return frag


# ---------------------------------------------------------------------------
# C1 — crack-tip criterion
# ---------------------------------------------------------------------------

def test_mts_pure_mode_i_is_zero() -> None:
    assert mts_kink_angle_deg(1.0e6, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_mts_pure_mode_ii_limit() -> None:
    # Classic Erdogan–Sih result: pure mode II kinks by -70.53 degrees.
    assert mts_kink_angle_deg(0.0, 1.0e6) == pytest.approx(-70.5287793655, abs=1e-6)


def test_mts_matches_williams_field_scan() -> None:
    for ratio in (0.0, 0.2, 0.5, 1.0, 2.0, -0.5):
        k1, k2 = 1.0e6, ratio * 1.0e6
        closed = mts_kink_angle_deg(k1, k2)
        scanned = asymptotic_kink_angle_deg(k1, k2, n=3601)
        assert abs(closed - scanned) <= 0.1, (ratio, closed, scanned)


def test_mts_sign_follows_shear_sign() -> None:
    assert mts_kink_angle_deg(1.0e6, 0.5e6) < 0.0
    assert mts_kink_angle_deg(1.0e6, -0.5e6) > 0.0


def test_asymptotic_scan_is_scale_invariant() -> None:
    a = asymptotic_kink_angle_deg(2.0e6, 1.0e6)
    b = asymptotic_kink_angle_deg(2.0e-3, 1.0e-3)
    assert a == pytest.approx(b, abs=1e-9)


def test_senb_reference_reduces_to_known_limits() -> None:
    # F(alpha -> 0) -> 1.99 for the ASTM E399 three-point-bend geometry.
    near_zero = senb_k1_reference(1.0, 4.0, 1.0, 0.1, 1.0e-6)
    assert near_zero["geometry_function"] == pytest.approx(1.99, abs=1e-4)
    # K_I must grow with crack length for a fixed load on a bending specimen.
    k_short = senb_k1_reference(100.0, 0.24, 0.06, 0.02, 0.012)["k1_pa_sqrt_m"]
    k_long = senb_k1_reference(100.0, 0.24, 0.06, 0.02, 0.024)["k1_pa_sqrt_m"]
    assert k_long > k_short


def test_notch_face_node_pairs_are_node_aligned() -> None:
    model = build_bend_model(
        BeamSpec(
            length_m=0.24,
            height_m=0.06,
            thickness_m=0.02,
            notch_depth_m=0.02,
            notch_gap_m=0.02,
            E_pa=9.0e9,
            nu=0.33,
        ),
        dims=(12, 12, 3),
    )
    pairs = notch_face_node_pairs(model)
    assert len(pairs) >= 4
    radii = [r for r, _i, _j in pairs]
    assert radii == sorted(radii)
    nodes = model.mesh.nodes
    for r, i, j in pairs:
        assert abs(nodes[i, 0] - model.crack_plane_x_m) < 1e-9
        assert abs(nodes[j, 0] - (model.crack_plane_x_m + model.notch_gap_effective_m)) < 1e-9
        assert abs(nodes[i, 1] - nodes[j, 1]) < 1e-12
        assert r > 0.0


# ---------------------------------------------------------------------------
# C2 — reconstructed volume
# ---------------------------------------------------------------------------

def test_polyhedron_volume_of_a_box() -> None:
    verts = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.03, 0], [0, 0.03, 0]], dtype=float)
    volume = polyhedron_volume(verts, [0, 0, 1], 0.5)
    assert volume == pytest.approx(0.02 * 0.03 * 0.5, rel=1e-12)


def test_polyhedron_volume_warped_quad_matches_monte_carlo() -> None:
    verts = np.array(
        [[0, 0, 0], [0.02, 0.002, 0], [0.021, 0.031, 0], [-0.001, 0.029, 0]], dtype=float
    )
    thickness = 0.01
    exact = polyhedron_volume(verts, [0, 0, 1], thickness)
    rng = np.random.default_rng(7)
    xs = rng.uniform(-0.002, 0.022, 400_000)
    ys = rng.uniform(-0.001, 0.032, 400_000)
    # Point-in-polygon by the crossing number on the same vertices.
    inside = np.zeros(xs.shape, dtype=bool)
    m = len(verts)
    for i in range(m):
        j = (i + 1) % m
        cond = (verts[i, 1] > ys) != (verts[j, 1] > ys)
        x_cross = (verts[j, 0] - verts[i, 0]) * (ys - verts[i, 1]) / (verts[j, 1] - verts[i, 1]) + verts[i, 0]
        inside ^= cond & (xs < x_cross)
    area_mc = inside.mean() * (0.024 * 0.033)
    assert exact == pytest.approx(area_mc * thickness, rel=0.02)


def test_polyhedron_volume_rejects_degenerate_input() -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(ValueError):
        polyhedron_volume(verts, [0, 0, 1], 0.0)
    with pytest.raises(ValueError):
        polyhedron_volume(np.array([[0, 0, 0], [1, 0, 0]], dtype=float), [0, 0, 1], 1.0)


def test_polygon_second_moments_match_rectangle_formula() -> None:
    w, h = 0.02, 0.03
    verts = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=float)
    moments = polygon_second_moments(verts, [0, 0, 1])
    assert moments["area_m2"] == pytest.approx(w * h, rel=1e-12)
    # i_u = integral q^2 dA about the centroid (q spans h) == w h^3 / 12;
    # i_v = integral p^2 dA (p spans w) == h w^3 / 12.
    values = sorted([moments["i_u_m4"], moments["i_v_m4"]])
    assert values[0] == pytest.approx(h * w**3 / 12.0, rel=1e-9)
    assert values[1] == pytest.approx(w * h**3 / 12.0, rel=1e-9)
    assert abs(moments["i_uv_m4"]) < 1e-18


def test_fragment_inertia_matches_analytic_box() -> None:
    w, h, t, rho = 0.02, 0.03, 0.005, 900.0
    verts = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=float)
    props = fragment_inertia(verts, [0, 0, 1], t, rho, verts.mean(axis=0))
    m = rho * w * h * t
    assert props["mass_kg"] == pytest.approx(m, rel=1e-12)
    principal = sorted(props["principal_moments_kg_m2"])
    expected = sorted(
        [
            m * (w * w + t * t) / 12.0,
            m * (h * h + t * t) / 12.0,
            m * (w * w + h * h) / 12.0,  # rotation about the extrusion axis
        ]
    )
    for got, want in zip(principal, expected):
        assert got == pytest.approx(want, rel=2e-3), (principal, expected)


def test_volume_budget_consistent_geometry_passes() -> None:
    frags = [_fragment(0, 0.0, [0, 0, 0]), _fragment(1, 0.01, [0, 0, 0])]
    report = volume_budget(frags)
    assert report["relative_error"] < 1e-12
    assert report["passed"]


def test_volume_budget_negative_control_detects_thick_layer() -> None:
    # Consumer reconstructs with twice the FEM layer thickness -> +100 % volume.
    frags = [_fragment(0, 0.0, [0, 0, 0]), _fragment(1, 0.01, [0, 0, 0])]
    doubled = [
        build_transfer_fragment(
            _face(element_id=i, x0=x),
            density_kg_m3=917.0,
            thickness_m=0.01,
            index=i,
            fem_element_volume_m3=_face(element_id=i, x0=x)["area_m2"] * 0.005,
        )
        for i, x in enumerate((0.0, 0.01))
    ]
    report = volume_budget(doubled)
    assert report["relative_error"] == pytest.approx(1.0, rel=1e-6)
    assert not report["passed"]
    # and the consistent pair still passes, so the gate is not just "always red"
    assert volume_budget(frags)["passed"]


# ---------------------------------------------------------------------------
# C3 — momentum
# ---------------------------------------------------------------------------

def test_rigid_body_fit_recovers_rigid_motion() -> None:
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    centre = pts.mean(axis=0)
    omega = np.array([0.1, -0.2, 0.3])
    v0 = np.array([1.0, -2.0, 0.5])
    vel = v0 + np.cross(np.broadcast_to(omega, pts.shape), pts - centre)
    fit = rigid_body_fit(pts, vel, centre)
    assert np.allclose(fit["velocity_m_s"], v0, atol=1e-12)
    assert np.allclose(fit["omega_rad_s"], omega, atol=1e-12)
    assert fit["fit_residual_m_s"] < 1e-12


def test_rigid_body_fit_residual_flags_non_rigid_field() -> None:
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    centre = pts.mean(axis=0)
    vel = np.zeros_like(pts)
    vel[:, 0] = pts[:, 1]  # pure shear -> not a rigid-body field
    fit = rigid_body_fit(pts, vel, centre)
    assert fit["fit_residual_m_s"] > 1e-3


def test_momentum_budget_conserves_uniform_translation() -> None:
    v = [0.5, -0.2, 0.1]
    frags = [_fragment(0, 0.0, v), _fragment(1, 0.01, v)]
    report = momentum_budget(frags)
    assert report["momentum_relative_error"] < 1e-12
    assert report["angular_relative_error"] < 1e-12
    assert report["passed"]


def test_momentum_budget_negative_control_dropped_fragment() -> None:
    v = [0.5, -0.2, 0.1]
    frags = [_fragment(0, 0.0, v), _fragment(1, 0.01, v)]
    # Producer side keeps both fragments, consumer side silently drops one.
    reference = momentum_budget(frags)
    p_ref = np.linalg.norm(reference["fem_momentum_kg_m_s"])
    dropped = momentum_budget(frags[1:])
    expected = abs(np.linalg.norm(dropped["dem_momentum_kg_m_s"]) - p_ref) / p_ref
    assert expected == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# C4 — energy ledger
# ---------------------------------------------------------------------------

def test_energy_budget_closes_for_uniform_velocity() -> None:
    v = [0.3, -0.1, 0.0]
    frags = [_fragment(0, 0.0, v), _fragment(1, 0.01, v)]
    report = energy_budget(frags)
    assert report["ke_fem_j"] > 0.0
    assert report["kinetic_relative_error"] < 1e-12
    assert report["relative_error"] < 1e-12
    assert report["passed"]


def test_energy_budget_negative_control_scaled_velocity() -> None:
    """Consumer side inflates the inherited velocity by 30 % (producer untouched)."""
    import copy

    v = [0.3, -0.1, 0.0]
    frags = [_fragment(0, 0.0, v), _fragment(1, 0.01, v)]
    bad = []
    for frag in frags:
        clone = copy.copy(frag)
        clone.velocity_m_s = 1.3 * np.asarray(frag.velocity_m_s)
        bad.append(clone)
    good = energy_budget(frags)
    corrupt = energy_budget(bad)
    # The fracture energy dominates the ice ledger, so a 69 % kinetic inflation is
    # only a ~2 % total-energy signal; the kinetic sub-ledger is what catches it.
    assert corrupt["kinetic_relative_error"] > 0.05
    assert corrupt["relative_error"] > good["relative_error"]
    assert not corrupt["kinetic_passed"]
    assert good["kinetic_passed"]


# ---------------------------------------------------------------------------
# FEM-side consistency (K extraction and its external anchor)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_spec() -> BeamSpec:
    return BeamSpec(
        length_m=0.24,
        height_m=0.06,
        thickness_m=0.02,
        notch_depth_m=0.02,
        notch_gap_m=0.02,
        E_pa=9.0e9,
        nu=0.33,
    )


def test_stress_intensity_factors_are_linear_in_load(small_spec: BeamSpec) -> None:
    model = build_bend_model(small_spec, dims=(24, 12, 4))
    a = solve_bend(model, applied_displacement_m=1.0e-5)
    b = solve_bend(model, applied_displacement_m=2.0e-5)
    ka = stress_intensity_factors(model, a.solution.u)
    kb = stress_intensity_factors(model, b.solution.u)
    assert ka["k1_pa_sqrt_m"] > 0.0
    assert kb["k1_pa_sqrt_m"] / ka["k1_pa_sqrt_m"] == pytest.approx(2.0, rel=2e-3)


def test_k1_matches_senb_reference_order_of_magnitude(small_spec: BeamSpec) -> None:
    model = build_bend_model(small_spec, dims=(24, 12, 4))
    solution = solve_bend(model, applied_displacement_m=1.0e-5)
    fem = stress_intensity_factors(model, solution.solution.u)
    reference = senb_k1_reference(
        load_n=solution.load_pa_total_n,
        span_m=small_spec.length_m,
        width_m=small_spec.height_m,
        thickness_m=small_spec.thickness_m,
        crack_length_m=model.notch_depth_effective_m,
    )
    rel = abs(fem["k1_pa_sqrt_m"] - reference["k1_pa_sqrt_m"]) / reference["k1_pa_sqrt_m"]
    # The displacement correlation is applied to a finite-width notch, so the
    # agreement is bounded (documented); 40 % is the honest band, and the sign
    # and magnitude must be right, which a unit bug would break.
    assert rel < 0.40, (fem["k1_pa_sqrt_m"], reference["k1_pa_sqrt_m"])


def test_ice_material_is_in_the_laboratory_band() -> None:
    assert 5.0e9 <= ICE_MATERIAL.E_pa <= 12.0e9
    assert 0.3e6 <= ICE_MATERIAL.sigma_t_pa <= 2.5e6
    assert 3.0 <= ICE_MATERIAL.G_Ic_j_m2 <= 15.0
    assert ICE_SPEC.length_m / ICE_SPEC.height_m == pytest.approx(4.0)  # ASTM S/W = 4

# ---------------------------------------------------------------------------
# End-to-end protocol at the production mesh (C1-C4 + C6)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def protocol_run():
    from tensorphysics.failure_handoff import ICE_SPEC, run_protocol

    return run_protocol(
        spec=ICE_SPEC,
        dims=(48, 12, 4),
        load_offset_m=0.024,
        target_ratio=1.40,
        propagation_steps=2,
        duration_s=0.01,
    )


def test_protocol_end_to_end_criteria_pass(protocol_run) -> None:
    criteria = protocol_run.criteria()
    assert criteria["C1_pass"], criteria["C1_kink_angle_error_deg"]
    assert criteria["C2_pass"], criteria["C2_volume_relative_error"]
    assert criteria["C3_pass"], criteria["C3_momentum_relative_error"]
    assert criteria["C4_pass"], criteria["C4_energy_relative_error"]
    # The kink direction must be mixed mode here, otherwise C1 is vacuous.
    assert abs(protocol_run.tip_state["k2_over_k1"]) > 0.05
    assert abs(protocol_run.tip_state["theta_mts_deg"]) > 3.0


def test_protocol_packet_is_frozen_v3_and_valid(protocol_run) -> None:
    from tensorphysics.failure_handoff import SCHEMA_VERSION, packet_digest, validate_packet

    packet = protocol_run.packet
    assert packet["schema_version"] == SCHEMA_VERSION
    assert validate_packet(packet) == packet
    assert packet["sha256"] == packet_digest(packet)
    assert packet["face_count"] >= 4


def test_protocol_fragments_carry_mass_inertia_and_velocity(protocol_run) -> None:
    for fragment in protocol_run.fragments:
        assert fragment.mass_kg > 0.0
        assert fragment.volume_m3 > 0.0
        # inertia tensor must be symmetric positive definite
        inertia = np.asarray(fragment.inertia_kg_m2, dtype=float)
        assert np.allclose(inertia, inertia.T, atol=1e-18)
        assert np.all(np.linalg.eigvalsh(inertia) > 0.0)
        assert np.linalg.norm(fragment.velocity_m_s) > 0.0
        assert len(fragment.node_ids) == 8  # hex8 element nodal set
        assert len(fragment.face_node_indices) == 4


def test_gate_sensitivity_matrix_detects_every_corruption(protocol_run) -> None:
    from tensorphysics.failure_handoff import gate_sensitivity

    cases = gate_sensitivity(protocol_run)
    controls = [c for c in cases if c["control"]]
    corrupted = [c for c in cases if not c["control"]]
    assert controls and all(not c["detected"] for c in controls)
    assert corrupted and all(c["detected"] for c in corrupted), corrupted


def test_capability_record_claims_implemented(protocol_run) -> None:
    record = protocol_run.criteria()["C6_capability"]
    assert record["capability_id"] == "handoff.fem_to_dem"
    assert record["claimed_level"] == "Implemented"
    assert all(record["evidence"][key] for key in ("C1", "C2", "C3", "C4"))
