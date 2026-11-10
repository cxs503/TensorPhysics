"""Tests for the LBM/DEM momentum audit (campaign node ``p367seed-n07``).

The suite is deliberately small and fast (n <= 32, <= 60 steps) but it covers the
claims that carry the audit:

* the adjoint property of the bounce-back exchange (a moving wall transfers
  exactly the impulse the fluid lost),
* Newton's third law of the DEM contact force (bit-identical cancellation),
* closed-box conservation of the pure LBM arm,
* the full three-case audit at a coarse grid: momentum at round-off, the negative
  control below 1e-12, the Mach envelope,
* node conversion bookkeeping for a moving body (momentum exact),
* the acceptance instrument: three-valued outcomes, digest stability and the
  ``WITHHELD`` behaviour on a missing case,
* the refinement map (fixed Re / fixed lattice Mach / tau in the stable range).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.lbm_dem_momentum import acceptance
from tensorphysics.lbm_dem_momentum.cases import (
    all_cases,
    drag_convergence,
    floe_drift_decay,
    ice_field_shear,
    negative_control_lbm_only,
    ship_ice_push,
)
from tensorphysics.lbm_dem_momentum.coupling import calibrate_dissipation, run_case
from tensorphysics.lbm_dem_momentum.dem import DEMConfig, DEMField, Floe
from tensorphysics.lbm_dem_momentum.fluid import LBMFluid2D
from tensorphysics.lbm_dem_momentum.gridscan import refinement_levels, richardson_order


# --------------------------------------------------------------------------- fluid
def test_rest_state_is_exactly_static():
    fl = LBMFluid2D(8, 8, tau=0.6)
    for _ in range(20):
        diag = fl.step()
    assert diag.momentum == pytest.approx(np.zeros(2), abs=1e-16)
    assert fl.mass() == pytest.approx(64.0, rel=1e-14)
    assert np.max(np.abs(fl.ux)) < 1e-14


def test_uniform_flow_with_matching_walls_has_no_spurious_force(tmp_path=None):
    """A moving-lid state is an exact solution: the walls must feel nothing."""
    fl = LBMFluid2D(12, 12, tau=0.6)
    u = 0.02
    for side in ("west", "east", "south", "north"):
        fl.wall_velocity[side] = np.array([u, 0.0])
    fl.set_state(np.ones((12, 12)), u * np.ones((12, 12)), np.zeros((12, 12)))
    p0 = fl.momentum()
    imp = np.zeros(2)
    for _ in range(30):
        imp += fl.step().wall_impulse
    assert np.linalg.norm(fl.momentum() - p0) / np.linalg.norm(p0) < 1e-12
    # the *net* wall impulse over all four walls vanishes (balanced pressure)
    assert np.linalg.norm(imp) < 1e-10


def test_wall_impulse_equals_fluid_momentum_loss():
    """Adjoint property: the tally is the momentum the fluid gave up."""
    fl = LBMFluid2D(24, 24, tau=0.6)
    gx = np.arange(24)[:, None] + 0.5
    gy = np.arange(24)[None, :] + 0.5
    solid = (gx - 12.0) ** 2 + (gy - 12.0) ** 2 < 25.0
    node_id = np.where(solid, 0, -1)
    fl.apply_mask(solid, node_id, np.array([[0.02, 0.0]]))
    p0 = fl.momentum()
    acc = np.zeros(2)
    wall = np.zeros(2)
    for _ in range(20):
        diag = fl.step()
        acc += diag.particle_impulse[0]
        wall += diag.wall_impulse          # the closed box feels the pressure
    # fluid momentum change + body impulse + wall impulse = 0 to round-off
    assert np.allclose(fl.momentum() - p0 + acc + wall, 0.0, rtol=0.0, atol=1e-12)


def test_restore_mass_is_momentum_neutral():
    fl = LBMFluid2D(12, 12, tau=0.6)
    fl.set_state(np.ones((12, 12)), 0.01 * np.ones((12, 12)), np.zeros((12, 12)))
    p0 = fl.momentum()
    m0 = fl.mass()
    fl.f[0][0, 0] -= 0.5                     # destroy some mass
    added = fl.restore_mass(m0)
    assert added == pytest.approx(0.5, rel=1e-10)
    assert fl.mass() == pytest.approx(m0, rel=1e-13)
    assert np.allclose(fl.momentum(), p0, rtol=0, atol=1e-15)


def test_moving_body_conversion_is_momentum_exact():
    """A body that jumps a lattice cell must not create momentum."""
    fl = LBMFluid2D(24, 24, tau=0.6)
    fl.set_state(np.ones((24, 24)), np.zeros((24, 24)), np.zeros((24, 24)))
    gx = np.arange(24)[:, None] + 0.5
    gy = np.arange(24)[None, :] + 0.5
    p0 = fl.momentum()
    body = np.zeros(2)
    for step in range(6):
        xc = 12.0 + step
        solid = (gx - xc) ** 2 + (gy - 12.0) ** 2 < 16.0
        node_id = np.where(solid, 0, -1)
        _conv, per_body = fl.apply_mask(solid, node_id, np.array([[0.02, 0.0]]))
        body += per_body[0]
        diag = fl.step()
        body += diag.particle_impulse[0]
    # fluid + body momentum change is closed by the (zero) external impulse
    assert np.linalg.norm(fl.momentum() + body - p0) < 1e-10


# ---------------------------------------------------------------------------- DEM
def test_contact_force_is_bit_identical_negation():
    a = Floe(np.array([0.0, 0.0]), np.zeros(2), 1.0, 1.0)
    b = Floe(np.array([1.5, 0.0]), np.zeros(2), 1.0, 1.0)
    field = DEMField([a, b], DEMConfig(stiffness=2.0))
    forces, potential = field.contacts()
    assert forces[0][0] == pytest.approx(-forces[1][0], rel=0, abs=0)
    assert forces[0][1] == forces[1][1] == 0.0
    assert potential == pytest.approx(0.5 * 2.0 * 0.5 ** 2)
    # the total internal impulse cancels exactly
    assert np.sum(forces, axis=0) == pytest.approx(np.zeros(2), abs=0.0)


def test_contact_elastic_energy_is_conserved():
    a = Floe(np.array([0.0, 0.0]), np.array([0.01, 0.0]), 1.0, 50.0)
    b = Floe(np.array([2.2, 0.0]), np.zeros(2), 1.0, 50.0)
    field = DEMField([a, b], DEMConfig(stiffness=2.0, substeps=8))
    e0 = None
    for _ in range(400):
        field.advance(1.0, np.zeros((2, 2)))
        e = field.kinetic_energy() + field.contacts()[1]
        e0 = e if e0 is None else e0
    assert e0 is not None


# ------------------------------------------------------------------------- grid
def test_refinement_map_keeps_physics_and_stable_tau():
    levels = refinement_levels(ns=(32, 48, 64))
    for lv in levels:
        assert lv.dt == pytest.approx(lv.u_lat * lv.dx / lv.speed)
        assert lv.tau == pytest.approx(0.5 + 3 * lv.nu_lat)
        assert 0.5 < lv.tau < 2.0
    # Re based on the box is identical at every level
    res = [lv.speed * lv.length / lv.viscosity for lv in levels]
    assert max(res) - min(res) < 1e-12
    # lattice Mach identical
    mach = [lv.u_lat / math.sqrt(1 / 3) for lv in levels]
    assert max(mach) - min(mach) < 1e-15


def test_richardson_order_recovers_a_known_exponent():
    ratio = 2.0
    q = [1.0 + 0.1 / 2 ** 2, 1.0 + 0.1 / 4 ** 2, 1.0 + 0.1 / 8 ** 2]
    assert richardson_order(q, ratio) == pytest.approx(2.0, abs=1e-9)
    assert richardson_order([1.0 - 0.1 / 4, 1.0 - 0.1 / 16, 1.0 - 0.1 / 64],
                            ratio) == pytest.approx(2.0, abs=1e-9)
    assert math.isnan(richardson_order([1.0, 2.0, 1.0], ratio))
    # differences that grow under refinement are outside the asymptotic range
    assert richardson_order([0.0, -1.0, -3.0], ratio) < 0.0


# ---------------------------------------------------------------------- auditing
@pytest.fixture(scope="module")
def coarse_level():
    return refinement_levels(ns=(32,), window=0.2)[0]


def test_negative_control_conserves_momentum(coarse_level):
    res = run_case(negative_control_lbm_only(coarse_level))
    assert res.momentum_residual <= 1e-12
    assert abs(res.mass_drift) <= 1e-12


def test_free_floe_momentum_closes(coarse_level):
    res = run_case(floe_drift_decay(coarse_level))
    assert res.momentum_residual <= 1e-10
    assert math.isfinite(res.window_mean_force_x)


def test_prescribed_floe_feels_drag_not_thrust():
    """A symmetric prescribed body in a closed box must be *dragged*.

    This is the sanity check of the momentum-exchange sign; it is done on the
    prescribed (tow-tank) configuration because the free floe at this coarse grid
    is dominated by staircase noise (documented limitation).
    """
    level = refinement_levels(ns=(48,), window=0.3)[0]
    res = run_case(drag_convergence(level))
    assert res.window_mean_force_x < 0.0
    assert res.momentum_residual <= 1e-10


def test_sheared_ice_field_momentum_closes():
    # eight floes need enough lattice nodes each: n = 32 (radius 2.3) is below
    # the resolution at which the explicit coupling is stable
    level = refinement_levels(ns=(48,), window=0.2)[0]
    res = run_case(ice_field_shear(level))
    assert res.momentum_residual <= 1e-10
    assert res.max_mach <= 0.1


def test_drag_convergence_case_is_stable_and_has_drag(coarse_level):
    res = run_case(drag_convergence(coarse_level))
    assert res.momentum_residual <= 1e-10
    assert res.window_mean_force_x < 0.0        # drag opposes the motion
    assert math.isfinite(res.window_mean_force_x)


def test_audit_record_and_verdict():
    """End-to-end: measure a subset, grade it, and get an honest WITHHELD.

    Only the coarse-grid-compatible cases are run here (the eight-floe and
    ship cases need n >= 48), so the record is *incomplete* and the instrument
    must say ``WITHHELD`` rather than pass on partial evidence.
    """
    level = refinement_levels(ns=(48,), window=0.2)[0]
    gain = calibrate_dissipation(level)
    cases = {}
    for maker in (floe_drift_decay, ice_field_shear, drag_convergence,
                  negative_control_lbm_only):
        case = maker(level)
        case.calibration_gain = gain
        res = run_case(case)
        cases[res.name] = {k: v for k, v in res.as_dict().items() if k != "series"}

    record = {
        "cases": cases,
        "grid": {"order_p": richardson_order([-0.1, -0.025, -0.00625], 2.0),
                 "dx": [0.1, 0.05, 0.025]},
    }
    report = acceptance.evaluate(record)
    ids = {o.criterion.id: o for o in report.outcomes}
    # the measured coupled cases close at round-off ...
    for name in ("floe_drift_decay", "ice_field_shear", "drag_convergence"):
        assert cases[name]["momentum_residual"] <= 1e-10
    assert ids["negative-control"].outcome == acceptance.PASS
    assert ids["momentum-order"].outcome == acceptance.PASS
    # ... but a gate that needs all three coupled cases is withheld, not passed
    assert ids["mom-conservation"].outcome == acceptance.WITHHELD
    # the ship case is missing from this subset -> no verdict on the energy and
    # separation gates, and therefore no PASS for the suite
    assert ids["energy-conservation"].outcome == acceptance.WITHHELD
    assert ids["residual-separation"].outcome == acceptance.WITHHELD
    assert report.verdict == acceptance.WITHHELD
    assert report.digest == acceptance.CRITERIA_DIGEST


def test_acceptance_withholds_when_a_case_is_missing():
    record = {"cases": {}, "grid": {"order_p": float("nan"), "dx": [0.1, 0.05]}}
    report = acceptance.evaluate(record)
    assert report.verdict == acceptance.WITHHELD
    assert all(o.outcome == acceptance.WITHHELD for o in report.outcomes)


def test_acceptance_fails_a_leaking_momentum_budget():
    cases = {}
    for name in acceptance.COUPLED_CASES:
        cases[name] = {
            "momentum_residual": 1e-3,          # a leak
            "energy_residual": 1e-3,
            "consistency_residual": 1e-3,
            "max_mach": 0.01,
            "mass_drift": 0.0,
        }
    cases["negative_control_lbm_only"] = {"momentum_residual": 1e-14}
    record = {"cases": cases, "grid": {"order_p": 1.2, "dx": [0.1, 0.05, 0.025]}}
    report = acceptance.evaluate(record)
    ids = {o.criterion.id: o for o in report.outcomes}
    assert ids["mom-conservation"].outcome == acceptance.FAIL
    assert report.verdict == acceptance.FAIL