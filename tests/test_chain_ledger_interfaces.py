"""Interface faces A/B/C, the stabilised partitioned coupling and the time ledger."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.chain_ledger.body import BodyDynamics, BoxBody, RigidPose
from tensorphysics.chain_ledger.interfaces import (
    duhamel_response,
    face_a_convergence,
    face_a_force,
    face_b_closed_flux,
    face_b_power_reciprocity,
    face_b_volume_rate,
    face_c_generalised_force,
    rock4_modal,
)
from tensorphysics.chain_ledger.readiness import evaluate_chain, write_readiness
from tensorphysics.chain_ledger.slamming import WagnerSlamming
from tensorphysics.chain_ledger.stabilization import (
    InterfaceProblem,
    divergence_detection_rate,
    solve_interface,
    spectral_radius,
)
from tensorphysics.chain_ledger.time_ledger import detection, footprint, reconstruct, signal_error
from tensorphysics.chain_ledger.waves import StokesWave


@pytest.fixture()
def wave() -> StokesWave:
    return StokesWave(height=1.0, period=8.0, depth=10.0)


@pytest.fixture()
def body() -> BoxBody:
    return BoxBody(half_beam=5.0, draught=0.15, width=1.0, mass=1537.5, depth=0.6)


# --------------------------------------------------------------------- face A
def test_face_a_force_is_converged_and_second_order(wave, body):
    conv = face_a_convergence(wave, body, t=0.25 * wave.period, n_edges=(2, 4, 8, 16), order=2,
                              reference=(48, 6))
    assert conv["rel_err"] < 0.02
    assert conv["order"] >= 1.0
    assert conv["reference"]["self_consistency"] < 0.02
    # the boundary-node hold must be the *less* accurate of the two routes
    assert conv["rel_err"] >= conv["rel_err_gauss"]


def test_face_a_force_equals_the_hydrostatic_buoyancy_at_rest(body):
    """With no wave (still water) the integral must return rho g nabla."""
    panels = body.panels(RigidPose(), n_edge=16)
    F, _ = face_a_force(lambda pts: -body.rho * body.g * pts[:, 2], panels, np.zeros(3), order=3)
    # upward buoyancy: F = -int p n dA with n pointing out of the solid
    assert F[2] == pytest.approx(body.rho * body.g * body.volume, rel=1e-10)


# --------------------------------------------------------------------- face B
def test_face_b_volume_rate_matches_the_exact_reference(body):
    out = face_b_volume_rate(body, heave=0.02, heave_rate=0.30, n_edges=(4, 8, 16))
    assert out["rel_err"] < 1e-10
    assert out["rows"][-1]["dVdt_m3_s"] == pytest.approx(
        2 * body.half_beam * body.width * 0.30, rel=1e-12
    )


def test_face_b_closed_flux_and_negative_control(body):
    pose = RigidPose(r=np.array([0.0, 0.0, 0.02]), theta=np.array([0.0, 0.03, 0.0]))
    ok = face_b_closed_flux(body, pose, np.array([0.1, 0.0, 0.3]), np.array([0.0, 0.2, 0.0]))
    broken = face_b_closed_flux(
        body, pose, np.array([0.1, 0.0, 0.3]), np.array([0.0, 0.2, 0.0]), rotate_normals=False
    )
    assert ok["rel_flux"] < 1e-10
    assert broken["rel_flux"] > 1e-3          # the defect must be visible


def test_face_b_power_reciprocity():
    from tensorphysics.chain_ledger.body import box_panels

    body = BoxBody(half_beam=5.0, draught=0.15, width=1.0, mass=1537.5, depth=0.6)
    panels = box_panels(body, RigidPose(), n_edge=8)
    f = -panels.normal * (1000.0 * panels.area)[:, None]
    u = np.ones_like(panels.centroid)
    good = face_b_power_reciprocity(panels, f, u, u * 1.0001)
    bad = face_b_power_reciprocity(panels, f, u, 2.0 * u)
    assert good["rel_err"] < 1e-3
    assert bad["rel_err"] == pytest.approx(1.0, rel=1e-9)


# --------------------------------------------------------------------- face C
def test_face_c_generalised_force_routes_agree():
    x = np.linspace(0.0, 10.0, 161)
    shapes = np.vstack([np.sin(np.pi * x / 10.0), np.sin(2 * np.pi * x / 10.0)])

    def field(xx, xi):
        xx = np.asarray(xx, dtype=float).reshape(-1, 1)
        xi = np.asarray(xi, dtype=float).reshape(1, -1)
        return 1000.0 * np.exp(-((xx - 8.0) / 1.5) ** 2) * np.sqrt(
            np.clip(1.0 - (xi / 0.4) ** 2, 0.0, None)
        )

    out = face_c_generalised_force(field, x, 0.4, shapes, n_xi=161, n_x_field=801)
    assert out["max_rel_err"] < 0.03
    # the load sits at x = 8, where the first mode is positive and the second
    # has already changed sign - the projection must reproduce that
    assert out["Q_field_N"][0] > 0.0
    assert out["Q_field_N"][1] < 0.0


def test_duhamel_matches_the_direct_modal_integration():
    omega = 2.0 * np.pi * np.array([1.0, 3.0])
    zeta = np.array([0.02, 0.02])
    t = np.linspace(0.0, 4.0, 4001)
    Q = np.vstack([np.sin(2 * np.pi * 0.7 * t), np.exp(-((t - 0.3) / 0.05) ** 2)])
    q1 = duhamel_response(t, Q, omega, zeta)
    q2 = rock4_modal(t, Q, omega, zeta)
    err = np.max(np.abs(q1 - q2)) / np.max(np.abs(q2))
    assert err < 1e-3


def test_wagner_field_caps_at_the_contact_width():
    model = WagnerSlamming(rho=1025.0, deadrise_deg=10.0, max_half_width=0.45)
    t = np.linspace(0.0, 0.2, 201)
    v = np.full(t.size, 1.0)
    c = model.contact_half_width(t, v)
    assert c[0] == 0.0
    assert np.max(c) == pytest.approx(model.max_half_width)
    assert np.all(np.diff(c) >= 0.0)
    # rate check against the Wagner width rate
    early = (c[10] - c[5]) / (t[10] - t[5])
    assert early == pytest.approx(model.width_rate * 1.0, rel=1e-6)


# ------------------------------------------------------------- stabilisation (N)
def _chain_problem(ratio: float = 1.0) -> InterfaceProblem:
    return InterfaceProblem(
        structural_mass=np.diag([1537.5, 1.67e4]),
        added_mass=np.diag([6338.0, 5.28e4]) * ratio,
        external_force=np.array([5000.0, 1.0e5]),
    )


def test_spectral_radius_marks_the_light_body_unstable():
    assert spectral_radius(_chain_problem()) > 1.0
    assert spectral_radius(_chain_problem(ratio=0.01)) < 1.0


def test_naive_iteration_diverges_and_is_detected():
    res = solve_interface(_chain_problem(), scheme="none", max_iterations=60)
    assert res.divergent and not res.converged
    assert res.residual_history[-1] > 10 * res.residual_history[0]


@pytest.mark.parametrize("scheme", ["aitken", "iqn-ils"])
def test_stabilisation_converges_the_light_body(scheme):
    res = solve_interface(_chain_problem(), scheme=scheme, max_iterations=60, tol=1e-6)
    assert res.converged
    assert res.final_residual < 1e-6
    assert res.iterations <= 20


def test_divergence_detection_rate_is_complete():
    problems = [_chain_problem(ratio=r) for r in (0.25, 0.5, 1.0, 2.0)]
    out = divergence_detection_rate(problems, scheme="none", max_iterations=40)
    assert out["detection_rate"] == 1.0


# ------------------------------------------------------------------ time ledger
def test_reconstruct_orders_have_the_expected_accuracy():
    period, omega = 8.0, 2 * np.pi / 8.0
    t_src = np.arange(0.0, period + 1e-9, period / 64.0)
    y = np.cos(omega * t_src)
    t = np.arange(0.0, period, 0.01)
    ref = np.cos(omega * t)
    e0 = signal_error(t_src, y, t, ref, order=0, offset=0.0)["max_rel_err"]
    e1 = signal_error(t_src, y, t, ref, order=1, offset=0.0)["max_rel_err"]
    e2 = signal_error(t_src, y, t, ref, order=2, offset=0.0)["max_rel_err"]
    assert e0 > 10 * e1 > 10 * e2


def test_half_step_offset_is_detected_by_the_power_ledger():
    period, omega = 8.0, 2 * np.pi / 8.0
    t_src = np.arange(0.0, period + 1e-9, period / 8.0)
    y = np.cos(omega * t_src)
    t = np.arange(0.0, period, 0.005)
    vel = np.sin(omega * t)
    ref = np.cos(omega * t)

    def work(order: int, offset: float) -> float:
        rec = reconstruct(t_src, y, t, order=order, offset=offset)
        return float(abs(np.trapezoid(rec * vel, t)))

    rows = []
    for order in (1, 2):
        for frac in (0.0, 0.5):
            rows.append({"samples_per_period": 8, "order": order, "offset_fraction": frac,
                         "work": work(order, frac * period / 8.0)})
    det = detection(rows, "work", 0.10)
    assert det["all_detected"]


def test_footprint_table_is_complete():
    table = footprint(lambda dt, order, off: {"metric": abs(off) + dt}, period=8.0,
                      densities=(8, 16), orders=(0, 1), offset_fractions=(0.0, 0.5))
    assert len(table["rows"]) == 8
    assert set(table["rows"][0]) >= {"samples_per_period", "dt_h_s", "order", "offset_fraction", "metric"}


# --------------------------------------------------------------------- readiness
def _products(**overrides) -> dict:
    base = {
        "A": {"force_rel_err": 0.001, "observed_order": 2.0, "reference_self_consistency": 1e-9},
        "B": {"closed_flux_rel": 1e-12, "volume_rel_err": 1e-9, "normal_velocity_rel_err": 1e-4,
              "power_rel_err": 1e-5},
        "C": {"modal_force_rel_err": 0.001, "duhamel_rel_err": 1e-5},
        "E": {"wave_ledger_rel_err": 0.001, "body_ledger_rel_err": 1e-5},
        "T": {"zero_offset_rel_err": 0.001, "offset_residual": 2.4, "zero_order_force_err": 0.36,
              "offset_detected": 1.0, "zero_order_detected": 1.0},
        "N": {"divergence_detection_rate": 1.0, "stabilised_converged_rate": 1.0},
    }
    base.update(overrides)
    return base


def test_readiness_passes_when_all_faces_speak():
    verdict = evaluate_chain(_products())
    assert verdict["overall_status"] == "PASS"
    assert verdict["missing_products"] == []


@pytest.mark.parametrize("face", ["A", "B", "C", "E", "T", "N"])
def test_missing_face_withholds_never_passes(face):
    verdict = evaluate_chain(_products(**{face: None}))
    assert verdict["overall_status"] == "WITHHELD"
    assert face in verdict["missing_products"]
    assert verdict["faces"][face]["verdict"] == "WITHHELD"


def test_measured_but_failing_face_is_an_honest_fail():
    verdict = evaluate_chain(_products(A={"force_rel_err": 0.05, "observed_order": 0.5,
                                         "reference_self_consistency": 0.03}))
    assert verdict["overall_status"] == "FAIL"
    assert any(k.startswith("A:") for k in verdict["blocking_criteria"])


def test_missing_key_is_not_silently_ignored():
    verdict = evaluate_chain(_products(C={"modal_force_rel_err": 0.001}))
    assert verdict["overall_status"] == "FAIL"
    assert any(k == "C:duhamel_rel_err" for k in verdict["blocking_criteria"])


def test_readiness_written_as_stable_json(tmp_path):
    verdict = evaluate_chain(_products())
    path = write_readiness(verdict, tmp_path / "chain_readiness.json")
    text = (tmp_path / "chain_readiness.json").read_text(encoding="utf-8")
    import json
    again = json.loads(text)
    assert again["schema"] == "chain_readiness.v1"
    assert again["gate_id"] == "P35-N10-CHAIN"
    assert path.endswith("chain_readiness.json")
    # deterministic content: the same verdict serialises byte-identically
    write_readiness(verdict, tmp_path / "again.json")
    assert (tmp_path / "again.json").read_text(encoding="utf-8") == text