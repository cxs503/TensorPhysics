"""Contract tests for the implementation-level cross-clock library binding.

These tests are the executable form of the frozen contract
(``tensorphysics.cross_clock.library_binding``): sync points, units, clock
round-trip, impulse bookkeeping against the real TensorLBM / TensorDEM code, and
the anti-drift pins.  NumPy/torch only; no cluster, no fixtures beyond the two
libraries.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tensorphysics.cross_clock.contract import ContractError, SolverClock
from tensorphysics.cross_clock.library_binding import (
    FROZEN_PINS,
    CoupledClockRun,
    LibraryDrift,
    SyncSchedule,
    TensorDEMLeg,
    TensorLBMLeg,
    UnitMap,
    pin_library,
    verify_pins,
)

DX = 1.0e-4
DT_LBM = 1.0e-3
DT_DEM = 5.0e-5
TAU = 0.8
RHO0 = 1000.0
U0 = 1.0e-3


@pytest.fixture(scope="module")
def units():
    return UnitMap(dx=DX, dt=DT_LBM, rho0=RHO0)


@pytest.fixture(scope="module")
def schedule():
    return SyncSchedule(
        window=DT_LBM,
        clocks={"lbm": SolverClock("lbm", DT_LBM), "dem": SolverClock("dem", DT_DEM)},
    )


@pytest.fixture(scope="module")
def dem_config():
    from tensordem.coupling_kernel import DEMKernelConfig

    return DEMKernelConfig(
        radius=2.1e-3,
        density=1000.0,
        dt=DT_DEM,
        gravity=(0.0, 0.0, 0.0),
        fluid_density=RHO0,
        fluid_viscosity=1.0e-3,
        fluid_velocity=(U0, 0.0, 0.0),
        drag_model="stokes",
        box_lo=(-0.05, -0.05, -0.05),
        box_hi=(0.05, 0.05, 0.05),
        wall_axes=(False, False, False),
    )


# --------------------------------------------------------------- anti-drift
def test_frozen_pins_match_installed_libraries():
    """No version/signature/source drift against the frozen contract."""
    assert verify_pins() == []


def test_pin_records_the_api_surface_actually_used():
    for key, pin in FROZEN_PINS.items():
        assert pin.api, f"{key}: a pin without an API surface pins nothing"
        assert not pin.source_sha256.startswith("__"), f"{key}: placeholder digest left in the pin"
        for name in pin.api:
            assert name.startswith(key), f"{key}: {name} is outside the pinned package"


def test_pin_drift_is_reported_not_swallowed():
    """A tampered pin must produce a named, actionable drift message."""
    good = FROZEN_PINS["tensorlbm"]
    tampered = dataclasses.replace(good, version="0.0.0-drifted")
    messages = verify_pins({"tensorlbm": tampered})
    assert any("version" in m and "0.0.0-drifted" in m for m in messages)

    moved_api = dict(good.api)
    moved_api["tensorlbm.stream"] = "(f) -> f"  # a calling convention that moved
    messages = verify_pins({"tensorlbm": dataclasses.replace(good, api=moved_api)})
    assert any("tensorlbm.stream" in m for m in messages)


def test_missing_library_raises_library_drift():
    with pytest.raises(LibraryDrift):
        pin_library("no_such_package_xyz", "no-such-distribution-xyz", ())


# -------------------------------------------------------------------- units
def test_unit_roundtrip_is_exact(units):
    assert units.roundtrip_error() < 1e-15
    for v in (0.0, 1.0, -3.5e-2, 1e3):
        assert units.momentum_si(units.momentum_lat(v)) == pytest.approx(v, rel=1e-15)


def test_unit_exponents_are_consistent(units):
    """force = momentum / dt and impulse = momentum must share one scale."""
    assert units.force_scale == pytest.approx(units.momentum_scale / units.dt, rel=1e-15)
    # velocity scale and momentum scale must agree through the cell mass
    cell_mass = units.rho0 * units.dx**3
    assert units.momentum_scale == pytest.approx(cell_mass / units.dt, rel=1e-15)
    assert units.velocity_si(1.0) == pytest.approx(units.lattice_speed(), rel=1e-15)


def test_viscosity_matches_the_bgk_definition(units):
    # nu = (tau - 1/2)/3 * dx^2/dt  (the tau-1/3 variant is a factor 1.55 off)
    nu = units.viscosity_si(TAU)
    assert nu == pytest.approx(1.0e-6, rel=1e-12)
    nu_wrong = (TAU - 1.0 / 3.0) / 3.0 * DX**2 / DT_LBM
    assert abs(nu_wrong / nu - 1.0) > 0.5


def test_unit_map_rejects_degenerate_scales():
    with pytest.raises(ContractError):
        UnitMap(dx=0.0, dt=1.0)
    with pytest.raises(ContractError):
        UnitMap(dx=1e-4, dt=-1.0)


# ---------------------------------------------------------------- sync/clock
def test_schedule_requires_integer_ratios(schedule):
    assert schedule.ratios() == {"lbm": 1, "dem": 20}
    bad = SyncSchedule(window=7.5e-5, clocks=schedule.clocks)  # 1.5 x dt_dem
    with pytest.raises(ContractError):
        bad.validate()


def test_sync_points_coincide_for_every_window(schedule):
    n = 64
    sched = schedule.edges(n)
    assert sched.size == n + 1
    for name in schedule.clocks:
        leg = schedule.leg_edges(name, n)
        r = schedule.ratio(name)
        assert leg.size == n * r + 1
        assert np.allclose(leg[::r], sched, rtol=0.0, atol=1e-15)
    assert schedule.coincidence_error(n) < 1e-15


def test_window_time_is_integer_indexed_not_accumulated(schedule):
    """t_k must be exactly t0 + k*h, so no time error can accumulate."""
    worst = 0.0
    for k in (0, 1, 7, 199, 5000):
        worst = max(worst, abs(schedule.time_at(k) - (0.0 + k * schedule.window)))
    assert worst == 0.0


# -------------------------------------------------------------- fluid leg
def test_lbm_leg_momentum_kick_is_exact(units, schedule):
    fluid = TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=8, ny=4, tau=TAU,
                         initial_velocity_si=(0.0, 0.0))
    impulse = np.array([4.0e-10, -1.0e-10])
    before = fluid.momentum_si()
    applied = fluid.advance(1, impulse_si=impulse)
    after = fluid.momentum_si()
    assert np.allclose(applied, impulse, rtol=0, atol=1e-30)
    rel = np.abs((after - before) - impulse).max() / np.abs(impulse).max()
    # the residual floor is set by tensorlbm's float32 stencil constants, not by
    # the kick algebra; measure it instead of assuming machine epsilon
    assert rel < 1e-6


def test_lbm_leg_preserves_momentum_without_exchange(units, schedule):
    fluid = TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=8, ny=4, tau=TAU,
                         initial_velocity_si=(U0, 0.0))
    before = fluid.momentum_si()
    fluid.advance(20)
    after = fluid.momentum_si()
    assert np.abs(after - before).max() / np.abs(before).max() < 1e-12
    assert fluid.steps_taken == 20
    assert fluid.time == pytest.approx(20 * DT_LBM, rel=0, abs=1e-18)


def test_lbm_leg_rejects_bad_inputs(units, schedule):
    with pytest.raises(ContractError):
        TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=1, ny=4)
    with pytest.raises(ContractError):
        TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=4, ny=4, tau=0.4)
    fluid = TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=4, ny=4)
    with pytest.raises(ContractError):
        fluid.advance(0)


# ------------------------------------------------------------ particle leg
def test_dem_leg_rebuild_is_state_neutral(dem_config, schedule):
    """The frozen DEMKernelConfig forces a rebuild per window; it must not move state."""
    leg = TensorDEMLeg("dem", schedule.clocks["dem"], dem_config, positions=[[0.0, 0.0, 0.0]])
    assert leg.rebuild_roundtrip_error() == 0.0
    leg.advance(100)
    t_before = leg.time
    assert leg.rebuild_roundtrip_error() == 0.0
    assert leg.time == t_before


def test_dem_leg_matches_stokes_relaxation(dem_config, schedule):
    leg = TensorDEMLeg("dem", schedule.clocks["dem"], dem_config, positions=[[0.0, 0.0, 0.0]])
    tau_p = leg.mass / dem_config.stokes_coefficient
    leg.advance(4000)  # 0.2 s = 0.2 tau_p
    v = leg.velocities()[0][0]
    analytic = U0 * (1.0 - math.exp(-leg.time / tau_p))
    assert abs(v - analytic) / analytic < 1e-3
    assert leg.steps_taken == 4000


# ------------------------------------------------------------- coupled run
def test_coupled_run_closes_the_ledger_on_heterogeneous_clocks(units, dem_config, schedule):
    fluid = TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=16, ny=8, tau=TAU,
                         initial_velocity_si=(U0, 0.0))
    particles = TensorDEMLeg("dem", schedule.clocks["dem"], dem_config, positions=[[0.0, 0.0, 0.0]])
    run = CoupledClockRun(fluid, particles, schedule)
    out = run.run(20)

    assert out["clock_ratios"] == {"lbm": 1, "dem": 20}
    rt = out["clock_roundtrip"]
    assert rt["fluid_steps"] == rt["fluid_steps_expected"] == 20
    assert rt["particle_steps"] == rt["particle_steps_expected"] == 400
    assert abs(rt["fluid_time_s"] - rt["schedule_time_s"]) < 1e-15
    assert abs(rt["particle_time_s"] - rt["schedule_time_s"]) < 1e-15
    assert rt["sync_coincidence_error_s"] < 1e-15

    # bookkeeping invariant: produced == applied
    assert out["ledger"]["momentum"]["ledger_drift_max"] < 1e-15
    assert out["book_vs_produced_rel"] == 0.0
    # independent measurement of the system momentum closes too
    assert out["measured_system_momentum_residual_rel"] < 1e-9
    # the staggering defect is present and positive (a zero here would mean the
    # exchange is not actually staggered and the test would prove nothing)
    assert out["ledger"]["momentum"]["stagger_residual_max"] > 0.0
    assert out["tiling_violations"] == []


def test_coupled_run_is_reproducible(units, dem_config, schedule):
    def once():
        fluid = TensorLBMLeg("lbm", schedule.clocks["lbm"], units, nx=8, ny=4, tau=TAU,
                             initial_velocity_si=(U0, 0.0))
        particles = TensorDEMLeg("dem", schedule.clocks["dem"], dem_config,
                                 positions=[[0.0, 0.0, 0.0]])
        return CoupledClockRun(fluid, particles, schedule).run(5, keep_series=False)

    a, b = once(), once()
    assert a["produced_impulse_si"] == b["produced_impulse_si"]
    assert a["measured_fluid_momentum_delta_si"] == b["measured_fluid_momentum_delta_si"]