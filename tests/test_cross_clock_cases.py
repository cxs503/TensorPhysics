"""Case tests: FEM-LBM FSI and DEM-LBM particle flow under clock mismatch.

Runtimes are deliberately small (a fraction of a structure/acoustic period); the
full studies are produced by ``tensorphysics.cross_clock.acceptance``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.cross_clock import (
    AdaptiveClock,
    ContractError,
    DemParticle,
    FemStructure,
    LbmFluid1D,
    acoustic_structure_frequency,
    attach_reference,
    default_dem_params,
    default_fem_params,
    dem_lbm_ode_reference,
    drift_report,
    lattice_sound_speed,
    momentum_ledger,
    run_dem_lbm,
    run_fem_lbm,
)


# --------------------------------------------------------------------------- #
# instruments
# --------------------------------------------------------------------------- #
def test_lattice_sound_speed_and_mass_momentum_booking():
    f = LbmFluid1D(cells=16, tau=0.8)
    m0 = f.total_mass()
    for _ in range(100):
        f.step(0.0)
    assert f.total_mass() == pytest.approx(m0, rel=0, abs=1e-12)
    assert abs(f.total_momentum()) < 1e-15
    assert lattice_sound_speed() == pytest.approx(1.0 / math.sqrt(3.0))


def test_lbm_acoustic_period_matches_lattice_sound_speed():
    """A closed column rings at ``2L/c_s`` in lattice steps (dispersion check)."""
    n = 64
    f = LbmFluid1D(cells=n, tau=0.55)
    x = (np.arange(n) + 0.5) / n
    f.set_density_profile(1.0 + 1.0e-4 * np.cos(np.pi * x))
    probe = []
    for _ in range(700):
        f.step(0.0)
        probe.append(f.density()[n // 2])
    probe = np.asarray(probe) - 1.0
    zc = np.where(np.diff(np.sign(probe)) != 0)[0]
    period = 2.0 * float(np.mean(np.diff(zc)))
    expected = 2.0 * n / lattice_sound_speed()
    assert abs(period - expected) / expected < 0.02


def test_lbm_body_force_moves_only_the_momentum_mode():
    g = LbmFluid1D(cells=8, tau=0.8, periodic=True)
    g.add_body_force(1.0e-3)
    for _ in range(50):
        g.step()
    assert g.total_momentum() == pytest.approx(8.0 * 1.0e-3, rel=1e-10)
    assert g.total_mass() == pytest.approx(8.0, rel=0, abs=1e-12)
    assert np.allclose(g.velocity(), g.velocity()[0], atol=1e-14)


def test_fem_structure_exact_step_matches_analytic_oscillator():
    m, k = 0.5, 1.0e6
    st = FemStructure(mass=m, stiffness=k, x=1.0e-6)
    omega = math.sqrt(k / m)
    period = 2.0 * math.pi / omega
    h = period / 400.0
    for _ in range(100):
        st.advance(h, 0.0)
    exact = 1.0e-6 * math.cos(omega * 100 * h)
    assert st.x == pytest.approx(exact, rel=1e-9, abs=1e-18)
    assert st.energy() == pytest.approx(0.5 * k * 1.0e-12, rel=1e-9)


def test_dem_particle_relaxes_exactly_to_the_terminal_velocity():
    p = DemParticle(mass=1.0, drag=2.0, gravity=0.0, v=0.0)
    u = 0.25
    tau = p.relaxation_time
    for _ in range(40):
        p.advance(tau / 10.0, u)
    assert p.v == pytest.approx(u * (1.0 - math.exp(-4.0)), rel=1e-9)


def test_acoustic_structure_frequency_reduces_to_the_lumped_limit():
    an = acoustic_structure_frequency(0.5, 1.0e6, 1.0e-4, 1000.0, 1500.0, 0.1)
    lumped = math.sqrt(
        (an["acoustic_stiffness"] + 1.0e6) / (0.5 + an["added_mass"])
    )
    assert an["omega_exact"] == pytest.approx(lumped, rel=1e-4)
    assert an["omega_exact"] < an["first_acoustic_mode"]
    assert an["acoustic_stiffness"] == pytest.approx(1000.0 * 1500.0 ** 2 * 1.0e-4 / 0.1)
    assert an["added_mass"] == pytest.approx(1000.0 * 1.0e-4 * 0.1 / 3.0)


# --------------------------------------------------------------------------- #
# FEM-LBM FSI case
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fem_reference():
    # two periods: the zero-crossing estimator then averages several half periods,
    # which is what makes the frequency comparison a coupling measurement rather
    # than a single-crossing artefact
    return run_fem_lbm(1, params={"periods": 2.0})


def test_fem_lbm_frequency_matches_the_analytic_eigenfrequency(fem_reference):
    """The coupled piston-acoustic mode is reproduced to well below 3 %."""
    assert fem_reference.frequency_error < 0.01
    assert fem_reference.frequency == pytest.approx(fem_reference.analytic["f_exact"], rel=1e-3)


@pytest.mark.parametrize("ratio", [5, 10])
def test_fem_lbm_exchange_is_bookkeeping_exact_and_physics_bounded(fem_reference, ratio):
    case = run_fem_lbm(ratio, params={"periods": 2.0})
    attach_reference(case, fem_reference)
    rep = drift_report(case.windows, scale=case.exchange_scale)
    # the ledger invariant: what the fluid produced is what the structure applied
    assert rep["momentum"]["ledger_drift_max"] < 1e-12
    # the physics: the same window integral against the fine-clock reference
    assert rep["momentum"]["reference_drift_max"] < 1e-3
    assert case.derived["stagger_max"] < 1e-2
    # the declared bound must be finite and above the realised drift
    assert case.derived["bound_max"] > rep["momentum"]["reference_drift_max"]


def test_fem_lbm_frequency_error_grows_with_the_clock_ratio(fem_reference):
    coarse = run_fem_lbm(20, params={"periods": 2.0})
    assert coarse.frequency_error > fem_reference.frequency_error
    assert coarse.frequency_error < 0.03


def test_fem_lbm_windows_tile_and_bounds_are_reported(fem_reference):
    from tensorphysics.cross_clock import IntervalSample, check_tiling

    assert check_tiling([IntervalSample(w.t0, w.t1, {"mass": 0.0}) for w in fem_reference.windows]) == []
    assert all(w.ratio == 1 for w in fem_reference.windows)
    assert all(w.bound is not None and np.isfinite(w.bound) for w in fem_reference.windows)
    assert fem_reference.derived["coupling_step"] == pytest.approx(fem_reference.derived["dt"])


# --------------------------------------------------------------------------- #
# DEM-LBM particle case
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dem_case():
    p = default_dem_params() | {"periods": 1.0}
    period = 2.0 * math.pi / p["omega"]
    ref = dem_lbm_ode_reference(1, p, period)
    case = run_dem_lbm(10, params=p, reference=ref)
    return case, ref, p


def test_dem_lbm_tracks_the_closed_form_two_dof_reference(dem_case):
    case, _, _ = dem_case
    assert case["derived"]["velocity_error_max"] < 0.05
    assert case["derived"]["fluid_velocity_error_max"] < 0.05


def test_dem_lbm_total_momentum_balance(dem_case):
    case, _, _ = dem_case
    led = momentum_ledger(case)
    assert led["relative_residual"] < 0.05
    assert led["particle_mass_conserved"]["fluid_mass_final"] == pytest.approx(
        led["particle_mass_conserved"]["fluid_mass_initial"], rel=1e-12
    )


def test_dem_lbm_clock_convergence(dem_case):
    case, ref, p = dem_case
    fine = run_dem_lbm(1, params=p, reference=ref)
    assert fine["derived"]["velocity_error_max"] < case["derived"]["velocity_error_max"]
    assert fine["derived"]["bound_max"] < case["derived"]["bound_max"]


def test_dem_lbm_ledger_keeps_the_lag_defect_visible(dem_case):
    """Unlike the impulse-exact FEM pair, the DEM pair *applies* the predicted impulse.

    The produced/applied difference is therefore the staggering defect; it must be
    reported (not hidden) and must shrink when the clock is refined.
    """
    case, ref, p = dem_case
    coarse = drift_report(case["windows"], scale=p["mass"] * case["exchange_scale"])
    fine = run_dem_lbm(2, params=p, reference=ref)
    fine_rep = drift_report(fine["windows"], scale=p["mass"] * fine["exchange_scale"])
    assert coarse["momentum"]["ledger_drift_max"] > fine_rep["momentum"]["ledger_drift_max"]
    assert fine_rep["momentum"]["ledger_drift_max"] < 1e-2


def test_dem_lbm_rejects_invalid_ratio():
    with pytest.raises(ContractError):
        run_dem_lbm(0)


def test_adaptive_clock_drives_the_dem_pair_to_tolerance():
    """End-to-end: measured bounds from real runs feed the controller."""
    p = default_dem_params() | {"periods": 1.0}
    measurements = {}

    def bound_of(ratio: int) -> float:
        ratio = max(1, int(ratio))
        if ratio not in measurements:
            measurements[ratio] = run_dem_lbm(ratio, params=p)["derived"]["bound_max"]
        return measurements[ratio]

    ctrl = AdaptiveClock(ratio=100, tolerance=1.0e-2, order=1)
    res = ctrl.settle(bound_of, max_iter=8)
    assert res["converged"] is True
    assert res["final_ratio"] < 100
    assert bound_of(res["final_ratio"]) <= 1.0e-2


def test_fem_defaults_are_the_physical_parameter_set():
    p = default_fem_params()
    an = acoustic_structure_frequency(p["mass"], p["stiffness"], p["area"], p["rho"], p["c"], p["length"])
    assert 100.0 < an["f_exact"] < 1000.0
    assert p["length"] / p["cells"] < an["first_acoustic_mode"] and p["amplitude"] < 1e-5

def test_lbm_static_fluid_exchanges_no_impulse():
    """Gauge guard: a wall in a resting fluid must exchange exactly nothing.

    The population-level wall impulse contains the static pressure ``rho0/3`` per
    step; if that gauge is not subtracted, the coupled structure sees a constant
    ~2e5 N push and never oscillates (the defect that cost the first day of this
    node).  This test pins the gauge.
    """
    for with_wall in (False, True):
        f = LbmFluid1D(cells=16, tau=0.8, periodic=with_wall)
        for _ in range(50):
            f.step(0.0)
        d = f.drain()
        assert abs(d["impulse_left"]) < 1e-14
        assert abs(d["impulse_right"]) < 1e-14
        assert abs(d["mass_left"]) < 1e-14


def test_lbm_moving_wall_impulse_is_antisymmetric_in_the_wall_velocity():
    """Reversing the wall velocity reverses the exchanged impulse and mass.

    The antisymmetry is *first-order*, not exact: the moving-wall term is weighted
    by the local wall density ``rho(0)``, which is itself compressed/expanded by
    the wall motion.  Measured relative asymmetry 3.2e-4 at ``u = 1e-4`` and
    3.2e-5 at ``u = 1e-5``, i.e. proportional to ``u`` -- the tolerance below is a
    physical statement about the scheme, not a round-off statement.
    """
    imp, mass = [], []
    for u in (1e-4, -1e-4):
        f = LbmFluid1D(cells=16, tau=0.8)
        for _ in range(20):
            f.step(u)
        d = f.drain()
        imp.append(d["impulse_left"])
        mass.append(d["mass_left"])
    # antisymmetric only up to the O(u^2) compressibility of the moving wall term,
    # so the tolerance is a physical statement, not a round-off statement
    assert imp[0] == pytest.approx(-imp[1], rel=1e-3, abs=1e-18)
    assert mass[0] == pytest.approx(-mass[1], rel=1e-3, abs=1e-18)
    assert abs(mass[0]) > 0.0
