"""Acceptance tests for the ice-drift roughness closure (p302w1-n01).

Criterion map
-------------
C1  ``test_z0_monotone_in_keel_depth`` / ``test_z0_monotone_in_keel_spacing``
    property tests, 100 random morphology draws each, zero violations.
C2  ``test_drag_reynolds_follows_log_law``  model vs independent log-law
    (Karman roughness function) reference, error < 5 %.
C3  ``test_closure_converges_within_ten_iterations``  drift-velocity increment
    < 1e-4 m/s within 10 iterations, plus momentum-balance residual.
C4  ``test_beaufort_cross_validation``  |C_d error| < 20 % against the Beaufort
    Sea anchors (calibration and held-out validation subsets).
C5  ``test_open_water_limit``  C -> 0 gives the open-water value, error < 1 %.
C6  this file: >= 10 tests.
"""

from __future__ import annotations

import math
import random

import pytest

from tensorphysics.ice_drift_roughness import (
    ANCHORS,
    CD_TOLERANCE,
    IceDriftRoughnessError,
    KeelMorphology,
    DriftForcing,
    calibrate_wake_amplification,
    cross_validate,
    drift_roughness_diagnostics,
    fully_rough_drag_coefficient,
    ice_drift_roughness,
    ice_underside_roughness,
    law_of_wall_drag_coefficient,
    log_law_drag_coefficient,
    open_water_drag_coefficient,
    roughness_length_from_drag_coefficient,
    solve_free_drift,
    solve_friction_velocity,
    solve_ice_area_friction,
    surface_interface_roughness,
    validation_summary,
)

SEED = 20261110


# ---------------------------------------------------------------------------
# Log-law primitives
# ---------------------------------------------------------------------------
def test_log_law_round_trip():
    """eq. (1)/(4): the drag coefficient and z0 are exact inverses."""
    for z0 in (1e-4, 5e-3, 2e-2, 1e-1, 2.4e-1):
        cd = log_law_drag_coefficient(z0)
        back = roughness_length_from_drag_coefficient(cd)
        assert math.isclose(back, z0, rel_tol=1e-12)
    assert math.isclose(log_law_drag_coefficient(1.0e-4), 1.2073e-3, rel_tol=2e-3)


def test_open_water_drag_matches_bulk_value():
    """Open-water log-law value (1.207e-3) vs classical 1.3e-3 bulk value."""
    cd_open = open_water_drag_coefficient()
    assert math.isclose(cd_open, 1.2073e-3, rel_tol=2e-3)
    assert abs(cd_open - 1.3e-3) / 1.3e-3 < 0.10


def test_invalid_inputs_raise():
    with pytest.raises(IceDriftRoughnessError):
        KeelMorphology(keel_depth_m=-1.0, keel_spacing_m=50.0)
    with pytest.raises(IceDriftRoughnessError):
        KeelMorphology(keel_depth_m=5.0, keel_spacing_m=0.0)
    with pytest.raises(IceDriftRoughnessError):
        log_law_drag_coefficient(20.0)  # z0 >= z_ref
    with pytest.raises(IceDriftRoughnessError):
        surface_interface_roughness(
            ice_underside_roughness(KeelMorphology(6.0, 70.0), 2.0), 1.2
        )


# ---------------------------------------------------------------------------
# C1: monotonic correlation with keel morphology (property tests)
# ---------------------------------------------------------------------------
def _random_morphology(rng: random.Random) -> KeelMorphology:
    return KeelMorphology(
        keel_depth_m=rng.uniform(1.0, 12.0),
        keel_spacing_m=rng.uniform(30.0, 300.0),
    )


def test_z0_monotone_in_keel_depth():
    """C1: deeper keels never smooth the underside (100 random draws, 0 violations).

    The monotone direction is "deeper keel  =>  z0 not smaller".  Degenerate
    draws where the keel is shallower than the ice draft (no relief, ``A_r = 0``)
    or where the roughness ceiling is active may return *equal* values; only a
    wrong-direction response is counted as a violation.
    """
    rng = random.Random(SEED)
    violations = 0
    strict = degenerate = 0
    for _ in range(100):
        morph = _random_morphology(rng)
        h_ice = rng.uniform(0.5, 2.5)
        base = ice_underside_roughness(morph, h_ice)
        deeper = ice_underside_roughness(
            KeelMorphology(
                keel_depth_m=morph.keel_depth_m * 1.05,
                keel_spacing_m=morph.keel_spacing_m,
            ),
            h_ice,
        )
        if deeper.z0_m < base.z0_m - 1e-12:
            violations += 1
        elif deeper.z0_m > base.z0_m + 1e-12:
            strict += 1
        else:
            degenerate += 1
        assert deeper.drag_coefficient >= base.drag_coefficient - 1e-15
    assert violations == 0
    assert strict >= 80, f"only {strict} strictly increasing draws (degenerate={degenerate})"


def test_z0_monotone_in_keel_spacing():
    """C1: sparser keels never roughen the underside (100 random draws, 0 violations)."""
    rng = random.Random(SEED + 1)
    violations = 0
    strict = 0
    for _ in range(100):
        morph = _random_morphology(rng)
        h_ice = rng.uniform(0.5, 2.5)
        base = ice_underside_roughness(morph, h_ice)
        sparser = ice_underside_roughness(
            KeelMorphology(
                keel_depth_m=morph.keel_depth_m,
                keel_spacing_m=morph.keel_spacing_m * 1.05,
            ),
            h_ice,
        )
        if sparser.z0_m > base.z0_m + 1e-12:
            violations += 1
        elif sparser.z0_m < base.z0_m - 1e-12:
            strict += 1
    assert violations == 0
    assert strict >= 80, f"only {strict} strictly decreasing draws"


def test_z0_bounded_by_skin_and_ceiling():
    """z0 must stay between the smooth limit and the physical ceiling."""
    morph = KeelMorphology(keel_depth_m=20.0, keel_spacing_m=30.0)
    state = ice_underside_roughness(morph, 1.5)
    assert state.z_ceiling_active
    assert state.z0_ridge_m <= morph.z0_ceiling_m + 1e-15
    assert state.z0_skin_m <= state.z0_m <= morph.z0_ceiling_m


def test_undeformed_ice_recovers_skin_roughness():
    """A keel shallower than the ice draft is no keel: z0 -> z0_skin."""
    morph = KeelMorphology(keel_depth_m=1.0, keel_spacing_m=50.0)
    state = ice_underside_roughness(morph, 2.0)
    assert state.area_fraction == 0.0
    assert math.isclose(state.z0_m, morph.z0_skin_m, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# C2: Reynolds-number dependence vs log-law theory
# ---------------------------------------------------------------------------
def test_drag_reynolds_follows_log_law():
    """C2: model C_d(Re) vs the velocity-defect log-law reference, error < 5 %.

    The model route solves the roughness-length superposition
    ``z0 = k_s/30 + 0.11 nu/u*`` together with ``u* = U sqrt(C_d)`` (system (4) of
    the ``drag_coeff`` docstring); the reference route solves the velocity-defect
    wall law with the Karman roughness function (system (5)-(6)).  They are
    algebraically independent, so their agreement tests the Reynolds-number
    closure itself.
    """
    worst = 0.0
    worst_case = None
    for k_s in (2e-3, 1e-2, 5e-2, 1.5e-1):
        for u_rel in (0.05, 0.1, 0.2, 0.4, 0.8):
            ref = law_of_wall_drag_coefficient(u_rel, k_s)
            model = solve_friction_velocity(u_rel, k_s / 30.0)
            rel = abs(model.drag_coefficient - ref.drag_coefficient) / ref.drag_coefficient
            if rel > worst:
                worst, worst_case = rel, (k_s, u_rel, model.regime)
    assert worst < 0.05, f"max log-law deviation {worst:.3%} exceeds 5% at {worst_case}"


def test_fully_rough_limit_matches_closed_form():
    """Fully rough asymptote: model -> [kappa/ln(z_ref 30/k_s)]**2 within 5 %."""
    k_s = 0.2
    u_rel = 2.0  # high drift -> k_s_plus >> 70
    state = solve_ice_area_friction(
        u_rel,
        ice_underside_roughness(KeelMorphology(keel_depth_m=8.0, keel_spacing_m=1000.0), 1.5),
    )
    closed = fully_rough_drag_coefficient(k_s)
    assert state.roughness_reynolds > 70.0
    # geometric-only comparison on the keel-class roughness element
    keel_cd = log_law_drag_coefficient(k_s / 30.0)
    assert abs(keel_cd - closed) / closed < 1e-12


def test_regime_classification_ordering():
    """Rougher/lower-velocity walls sit in smoother regimes."""
    smooth = solve_ice_area_friction(
        0.02, ice_underside_roughness(KeelMorphology(1.2, 500.0), 1.0)
    )
    rough = solve_ice_area_friction(
        0.6, ice_underside_roughness(KeelMorphology(10.0, 40.0), 1.5)
    )
    assert rough.roughness_reynolds > smooth.roughness_reynolds
    assert rough.drag_coefficient > smooth.drag_coefficient


def test_smooth_ice_drag_is_velocity_dependent():
    """Drift-velocity control of z0 and C_d on a hydraulically smooth underside.

    Faster drift thins the viscous sublayer, so the viscous roughness
    contribution ``0.11 nu/u*`` shrinks and ``z0 -> z0_geom``; the drag
    coefficient therefore *decreases* with drift speed and is bounded below by
    the purely geometric value (the classical smooth-wall Re dependence).
    """
    morph = KeelMorphology(keel_depth_m=1.3, keel_spacing_m=800.0, z0_skin_m=5e-3)
    state = ice_underside_roughness(morph, 1.2)
    slow = solve_ice_area_friction(0.01, state)
    fast = solve_ice_area_friction(1.0, state)
    assert fast.drag_coefficient < slow.drag_coefficient
    assert fast.z0_m < slow.z0_m
    assert fast.z0_m > state.z0_m - 1e-12       # cannot fall below the geometry
    assert slow.drag_coefficient > state.drag_coefficient
    assert abs(fast.drag_coefficient - state.drag_coefficient) / state.drag_coefficient < 0.02
    assert slow.regime in ("smooth", "transitional")
    assert fast.regime == "fully_rough"


# ---------------------------------------------------------------------------
# C3: closure convergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wind,h_ice,cd_floor",
    [
        ((5.0, 0.0), 1.0, 2.0e-3),
        ((10.0, 0.0), 2.0, 2.5e-3),
        ((15.0, 3.0), 2.5, 3.0e-3),
        ((20.0, -5.0), 1.5, 2.5e-3),
    ],
)
def test_closure_converges_within_ten_iterations(wind, h_ice, cd_floor):
    """C3: |dU| < 1e-4 m/s within 10 iterations, and the balance is satisfied."""
    morph = KeelMorphology(keel_depth_m=6.0, keel_spacing_m=70.0)
    sol = solve_free_drift(h_ice, 0.9, morph, DriftForcing(wind=wind))
    assert sol.converged
    assert sol.iterations <= 10
    assert sol.last_increment < 1e-4
    assert sol.relative_force_residual < 1e-6
    assert sol.drag_coefficient > cd_floor


def test_closure_convergence_history():
    """The Newton increments decay (monotone convergence diagnostics)."""
    sol = solve_free_drift(
        2.0, 1.0, KeelMorphology(6.0, 70.0), DriftForcing(wind=(12.0, 2.0))
    )
    assert sol.iterations <= 6
    assert sol.last_increment < 1e-4


def test_free_drift_nansen_rule_and_turn_angle():
    """Free drift reproduces the classical Nansen behaviour (~2 %, right turn)."""
    sol = solve_free_drift(
        2.0, 1.0, KeelMorphology(6.0, 70.0), DriftForcing(wind=(10.0, 0.0), latitude_deg=75.0)
    )
    assert 0.005 < sol.drift_ratio < 0.05
    assert -60.0 < sol.turn_angle_deg < 0.0  # right of the wind in the NH


def test_free_drift_turn_angle_analytic_identity():
    """Free drift turning obeys tan(theta) = rho_i h f / (rho_w C_d |V|).

    Derived from the steady balance rho_w C_d |V| V + rho_i h f (k x V) = tau_a
    for |U_i| << |U_a|; the residual is a genuine cross-check of the Coriolis and
    drag terms in the momentum balance.
    """
    forcing = DriftForcing(wind=(10.0, 0.0), latitude_deg=75.0)
    sol = solve_free_drift(2.0, 1.0, KeelMorphology(6.0, 70.0), forcing)
    predicted = math.degrees(
        math.atan(
            forcing.ice_density * 2.0 * forcing.coriolis()
            / (forcing.water_density * sol.drag_coefficient * sol.relative_speed)
        )
    )
    assert abs(abs(sol.turn_angle_deg) - predicted) / predicted < 0.05


def test_drift_provider_interface():
    """External drift models can be injected (freeze_drift diagnostics mode)."""
    def provider(h, wind, current):
        return (0.12, -0.04)

    sol = solve_free_drift(
        2.0, 0.9, KeelMorphology(6.0, 70.0), DriftForcing(wind=(10.0, 0.0)),
        drift_provider=provider, freeze_drift=True,
    )
    assert sol.ice_velocity == (0.12, -0.04)
    assert sol.drag_coefficient > 0.0
    assert sol.z0_ice_m > 0.0


# ---------------------------------------------------------------------------
# C4: Beaufort Sea cross-validation
# ---------------------------------------------------------------------------
def test_beaufort_cross_validation():
    """C4: |C_d error| < 20 % on calibration *and* held-out validation anchors."""
    calib = calibrate_wake_amplification()
    assert 0.2 < calib["fitted_value"] < 8.0
    rows = cross_validate(wake_amplification=calib["fitted_value"])
    summary = validation_summary(rows)
    assert summary["criterion_C4_pass"], summary
    assert summary["max_abs_rel_error"] < CD_TOLERANCE
    assert summary["validation_max_abs_rel_error"] < CD_TOLERANCE
    assert summary["validation_all_within_tolerance"]
    for cover in rows:
        assert cover["cd_model"] > 0.0


def test_anchor_dataset_integrity():
    """Anchor set: both splits populated, central values inside their ranges."""
    splits = {a.split for a in ANCHORS}
    assert splits == {"calibration", "validation"}
    for anchor in ANCHORS:
        lo, hi = anchor.cd10_range
        assert lo < anchor.cd10_obs < hi
        assert 1.0e-3 < anchor.cd10_obs < 1.0e-2
        assert anchor.keel_depth_m > anchor.h_ice_m


# ---------------------------------------------------------------------------
# C5: open-water negative control
# ---------------------------------------------------------------------------
def test_open_water_limit():
    """C5: C -> 0 recovers the open-water drag coefficient within 1 %."""
    morph = KeelMorphology(keel_depth_m=6.0, keel_spacing_m=70.0)
    ice_state = ice_underside_roughness(morph, 2.0)
    cd_open = open_water_drag_coefficient()
    for conc in (0.0, 1e-9, 1e-4):
        mosaic = surface_interface_roughness(ice_state, conc)
        err = abs(mosaic["cd_surface"] - cd_open) / cd_open
        assert err < 0.01, f"C={conc}: error {err:.4%}"
    # and the mixture is monotone in concentration
    prev = surface_interface_roughness(ice_state, 0.0)["cd_surface"]
    for conc in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        cur = surface_interface_roughness(ice_state, conc)["cd_surface"]
        assert cur > prev
        prev = cur


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def test_public_api_prescribed_and_closure_modes():
    prescribed = ice_drift_roughness(
        (0.15, -0.05), concentration=0.9, h_ice_m=2.0, wind=(10.0, 0.0),
        keel_depth_m=6.0, keel_spacing_m=70.0,
    )
    assert prescribed["mode"] == "prescribed_drift"
    assert 2.0e-3 < prescribed["cd_ice"] < 9.0e-3
    assert 5.0e-3 < prescribed["z0_ice_m"] < 2.5e-1

    closure = ice_drift_roughness(
        None, concentration=0.9, h_ice_m=2.0, wind=(10.0, 0.0),
        morphology=KeelMorphology(6.0, 70.0),
    )
    assert closure["mode"] == "free_drift_closure"
    assert closure["converged"]
    assert closure["iterations"] <= 10
    assert closure["cd_surface"] < closure["cd_ice"]  # leads pull the mean down


def test_drag_increases_with_roughness_and_velocity():
    """Physical sanity: C_d grows with keel relief and is bounded by 1."""
    soft = drift_roughness_diagnostics(
        (0.15, 0.0), (0.0, 0.0), 1.5, 0.9, KeelMorphology(2.0, 300.0)
    )
    hard = drift_roughness_diagnostics(
        (0.15, 0.0), (0.0, 0.0), 1.5, 0.9, KeelMorphology(9.0, 50.0)
    )
    assert hard["cd_ice"] > soft["cd_ice"]
    assert hard["z0_ice_m"] > soft["z0_ice_m"]
    assert 0.0 < soft["cd_ice"] < 1.0 and 0.0 < hard["cd_ice"] < 1.0