"""Gates for the multi-solver coupling energy/momentum contract (P40W3C-N09).

The tests are deliberately small instances of the *same* scenarios the
acceptance harness runs, so a red test is always a red report number.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.coupling_energy import (EnergyLedger, MomentumLedger,
                                           CouplingLedgerError, Q4Mesh,
                                           TemporalScheme, ThermalConfig,
                                           ThermalLBM2D, ContactLaw, DEM2D,
                                           gather_nodes_to_columns,
                                           interval_exchange,
                                           quadrature_error,
                                           reconstruction_order_scan,
                                           run_mechanical_coupling,
                                           run_thermal_coupling,
                                           run_three_field,
                                           scatter_columns_to_nodes,
                                           solver_verification)
from tensorphysics.coupling_energy.exchange import ExchangeSample


# ---------------------------------------------------------------------------
# ledger semantics
# ---------------------------------------------------------------------------
def test_energy_ledger_books_transfers_without_loss():
    led = EnergyLedger(("a", "b"))
    led.set("a", 100.0)
    led.set("b", 0.0)
    led.sample(0.0)
    led.exchange("a", "b", 30.0)
    led.book_external(5.0)          # heat that entered the system from outside
    led.book_dissipation(2.0)       # explicitly modelled sink
    s = led.sample(1.0)
    assert s.sides["a"] == pytest.approx(70.0)
    assert s.sides["b"] == pytest.approx(30.0)
    assert led.total() == pytest.approx(100.0 - 5.0 + 2.0)
    assert s.interface == pytest.approx(30.0)


def test_energy_ledger_rejects_nonsense():
    led = EnergyLedger(("a", "b"))
    with pytest.raises(CouplingLedgerError):
        led.exchange("a", "a", 1.0)
    with pytest.raises(CouplingLedgerError):
        led.exchange("a", "ghost", 1.0)
    with pytest.raises(CouplingLedgerError):
        led.book_dissipation(-1.0)
    with pytest.raises(CouplingLedgerError):
        led.residual()               # no samples yet


def test_momentum_ledger_requires_seeded_initial_state():
    led = MomentumLedger(("solid", "particles"), scale=1.0)
    led.set("particles", -2.0, 0.0)
    led.record()
    led.exchange("particles", "solid", -2.0, 0.0)
    led.record()
    assert led.residual() == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# interface operator
# ---------------------------------------------------------------------------
def test_scatter_gather_are_adjoint_and_conservative():
    rng = np.random.default_rng(7)
    q = rng.normal(size=8)
    t = rng.normal(size=9)
    nodes = scatter_columns_to_nodes(q)
    assert nodes.sum() == pytest.approx(q.sum())
    assert np.dot(nodes, t) == pytest.approx(np.dot(q, gather_nodes_to_columns(t)))


# ---------------------------------------------------------------------------
# lattice: this is the regression test for the D2Q9 opposite-table defect
# ---------------------------------------------------------------------------
def test_lattice_is_conservative_and_injection_is_measurable():
    lbm = ThermalLBM2D(12, 12, tau=1.0, T_init=1.0)
    start = lbm.total()
    for _ in range(40):
        lbm.step(None)
    assert abs(lbm.total() - start) / start < 1e-13     # adiabatic walls

    lbm2 = ThermalLBM2D(12, 12, tau=1.0, T_init=1.0)
    wall = np.full(12, 0.25)
    injected, before = 0.0, lbm2.total()
    for _ in range(40):
        injected += float(lbm2.step(wall).sum())
    assert injected == pytest.approx(lbm2.total() - before, rel=1e-12)


def test_reconstruction_order_on_an_analytic_signal():
    signal = lambda t: math.exp(-0.7 * t) * (1.0 + 0.4 * math.cos(3.0 * t))
    scan = reconstruction_order_scan(signal, 0.05, 4.0,
                                     schemes=(TemporalScheme.ZOH,
                                              TemporalScheme.LINEAR))
    assert scan["zoh"].order == pytest.approx(1.0, abs=0.25)
    assert scan["linear"].order > 1.8
    assert scan["linear"].order > scan["zoh"].order


def test_interval_exchange_uses_the_target_window():
    # a linear history is integrated exactly by the linear rule, whatever the
    # distance to the last sample -- this is what makes it second order
    prev = ExchangeSample(0.0, 2.0)
    cur = ExchangeSample(1.0, 1.0)          # linear history q(t) = 2 - t
    # int_1^2 (2-t) dt = 0.5, int_3^4 (2-t) dt = -1.5
    assert interval_exchange(TemporalScheme.LINEAR, prev, cur, 1.0,
                             t_start=1.0) == pytest.approx(0.5)
    assert interval_exchange(TemporalScheme.LINEAR, prev, cur, 1.0,
                             t_start=3.0) == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# (a) thermal coupling
# ---------------------------------------------------------------------------
def test_thermal_coupling_closes_energy_and_flux():
    cfg = ThermalConfig(grid=12, solid_rows=3, n_fine_steps=48)
    res = run_thermal_coupling(2, TemporalScheme.ZOH, config=cfg, n_fine_steps=48)
    assert res["residual_total"] < 5e-3
    assert res["residual_max"] < 5e-3
    assert res["flux_consistency"] < 0.15
    # measured wall flux equals the analytic cavity flux k_f A dT / (dx/2)
    k_f = cfg.alpha_fluid * cfg.rho_c_fluid
    q_ref = k_f * cfg.length * cfg.delta_T / (cfg.dx / 2.0)
    # order-of-magnitude check: the first interval is a transient, so 25 %
    # is the honest tolerance of the cavity-flux estimate
    assert abs(res["history"][0]["q_fluid"]) == pytest.approx(q_ref, rel=0.25)
    # the plate really warms up: the exchange is not bookkeeping only
    assert res["history"][-1]["e_solid"] > res["history"][0]["e_solid"]


def test_thermal_independent_control_is_red():
    cfg = ThermalConfig(grid=12, solid_rows=3, n_fine_steps=48)
    shared = run_thermal_coupling(2, TemporalScheme.ZOH, config=cfg, n_fine_steps=48)
    control = run_thermal_coupling(2, TemporalScheme.INDEPENDENT, config=cfg,
                                  n_fine_steps=48)
    assert control["exchange_mismatch"] > 5.0 * shared["exchange_mismatch"]


# ---------------------------------------------------------------------------
# (b) mechanical coupling
# ---------------------------------------------------------------------------
def test_mechanical_newton_third_law_and_momentum():
    res = run_mechanical_coupling(1, TemporalScheme.ZOH, n_fine_steps=240)
    assert res["force_antisymmetry"] < 1e-12
    assert res["momentum_residual"] < 1e-9
    assert res["peak_contact_force"] > 0.0            # a contact really happened


def test_mechanical_loose_coupling_leaks_momentum():
    res = run_mechanical_coupling(1, TemporalScheme.INDEPENDENT, n_fine_steps=240)
    assert res["momentum_residual"] > 1e-6
    assert res["force_antisymmetry"] > 1e-6


# ---------------------------------------------------------------------------
# (c) three-field chain
# ---------------------------------------------------------------------------
def test_three_field_chain_closes():
    res = run_three_field(2, TemporalScheme.ZOH)
    assert res["residual_total"] < 2e-2
    assert res["momentum_mismatch"] < 1e-12
    assert len(res["history"]) >= 4


# ---------------------------------------------------------------------------
# kernels vs analytic references
# ---------------------------------------------------------------------------
def test_kernels_match_analytic_references():
    ver = solver_verification(grids=(16, 32), steps=80)
    assert ver["lattice_diffusion"]["order"]["order"] > 1.5
    for row in ver["plate_conduction"]:
        assert row["energy_identity_error"] < 1e-10
        assert row["flux_error"] < 0.1
    dem = ver["dem_contact"]
    assert abs(abs(dem["restitution"]) - 1.0) < 0.05
    assert abs(dem["contact_time_numeric"] - dem["contact_time_reference"]) \
        / dem["contact_time_reference"] < 0.05


def test_dem_contact_force_is_repulsive_and_newton_consistent():
    mesh = Q4Mesh(2, 2, 1.0, 1.0)
    wall_x = np.full(3, 1.0)
    wall_v = np.zeros(3)
    dem = DEM2D(np.array([[1.0005, 0.5]]), np.array([[-1.0, 0.0]]),
                np.array([0.001]), 1.0, law=ContactLaw(stiffness=1e5))
    f_p, f_nodes, ov, _ = dem.contact_forces(wall_x, wall_v, np.full(3, 0.5),
                                            np.array([1]))
    assert ov[0] > 0.0
    assert f_p[0, 0] > 0.0                       # pushed away from the wall
    assert f_nodes[:, 0].sum() == pytest.approx(-f_p[:, 0].sum())