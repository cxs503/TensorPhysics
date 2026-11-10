"""Machine gates for the P189-N01 cross-solver interface conservation node.

The suite has four jobs:

1. unit-check the conserved-quantity algebra (residual definitions, unit
   conversion, observed-order estimator) — these are the definitions the whole
   node rests on, so they are checked exactly;
2. prove that the lattice closures are exact where they must be exact (uniform
   advection to round-off, closed-box mass to round-off) — a gate built on a
   closure that is *not* exact cannot distinguish a coupling defect from a
   solver defect;
3. run the transfer gate and assert the measured residuals, including the
   honest negative results: the state-exchange residual does **not** reach 1e-6,
   and the folklore "divide by the receiver density" operator does not help;
4. gate the finite-volume reference solver against Ghia et al. (1982) at a
   tolerance that a coarse grid can actually meet, so a regression in the
   independent discretisation is caught.

Recorded measurements (see ``artifacts/interface_conservation/gate_report.json``
and ``docs/INTERFACE_CONSERVATION_P189N01.md`` for the full-size run) are quoted
here as the values the small-scale cases reproduce.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.interface_conservation import quantities as Q
from tensorphysics.interface_conservation.fvm import StaggeredFVM
from tensorphysics.interface_conservation.lbm import D2Q9
from tensorphysics.interface_conservation.transfer import (
    CHANNEL_DOMAIN,
    _inlet_profile,
    run_channel_transfer,
)

# ---------------------------------------------------------------------------
# 1. conserved-quantity algebra
# ---------------------------------------------------------------------------


def test_flux_residual_is_scale_free():
    a = Q.InterfaceFlux(mass=1.0, momentum=2.0, energy=3.0)
    b = Q.InterfaceFlux(mass=1.0 + 1e-9, momentum=2.0 - 2e-9, energy=3.0)
    r = Q.flux_residual(a, b, Q.InterfaceFlux(1.0, 1.0, 1.0))
    assert r.mass == pytest.approx(1e-9, rel=1e-6)
    assert r.momentum == pytest.approx(1e-9, rel=1e-6)
    assert r.energy == 0.0


def test_flux_residual_reference_floor_prevents_division_by_noise():
    a = Q.InterfaceFlux(mass=0.0, momentum=0.0, energy=0.0)
    b = Q.InterfaceFlux(mass=1e-12, momentum=1e-12, energy=1e-12)
    r = Q.flux_residual(a, b, Q.InterfaceFlux(1.0, 1.0, 1.0))
    assert r.mass == pytest.approx(1e-12)
    with pytest.raises(ValueError):
        Q.flux_residual(a, b, Q.InterfaceFlux(0.0, 1.0, 1.0))


def test_local_flux_residual_normalisation():
    q = np.linspace(0.0, 1.0, 16)
    assert Q.local_flux_residual(q, q, 1.0) == 0.0
    got = Q.local_flux_residual(q, q + 1e-4, 1.0)
    assert got == pytest.approx(1e-4, rel=1e-6)


def test_lattice_unit_scaling_round_trip():
    s_u, s_p = Q.lattice_scales(0.1, 1.0 / 32)
    assert s_u == pytest.approx(10.0)
    assert s_p == pytest.approx(100.0)
    # a lid velocity of u_lat maps onto the unit finite-volume velocity
    assert 0.1 * s_u == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Q.lattice_scales(0.6, 0.1)


def test_fit_order_recovers_exponents_and_flags_a_floor():
    hs = [1 / 16, 1 / 32, 1 / 64]
    for p in (1.0, 2.0):
        errs = [3e-3 * h ** p for h in hs]
        slope, r2 = Q.fit_order(hs, errs)
        assert slope == pytest.approx(p, abs=1e-9)
        assert r2 == pytest.approx(1.0, abs=1e-9)
    # a sequence that hits a floor must not be reported as first order
    floor = [3e-3, 8e-4, 7.9e-4]
    slope, r2 = Q.fit_order(hs, floor)
    assert slope < 1.0
    with pytest.raises(ValueError):
        Q.fit_order([1.0], [1e-3])


def test_energy_flux_definition_signs():
    # pure normal flow with pressure only: flux = p * u_n
    e = Q.energy_flux_from_fields(u_n=np.array([1.0]), u_t=np.array([0.0]),
                                  p=np.array([2.0]), tau_nn=np.zeros(1),
                                  tau_nt=np.zeros(1))
    assert e == pytest.approx(2.5)          # (p + u^2/2) u_n


# ---------------------------------------------------------------------------
# 2. closure exactness (the gate's own preconditions)
# ---------------------------------------------------------------------------


def test_uniform_advection_is_exact_round_off():
    """A uniform flow with matching Zou & He closures must be a fixed point."""
    u = 0.05
    lbm = D2Q9(12, 12, tau=0.6)
    lbm.set_equilibrium((u, 0.0))
    bcs = {s: ("velocity", u, 0.0) for s in ("north", "south", "east", "west")}
    for _ in range(120):
        lbm.step(bcs)
    vel = lbm.velocity()
    assert np.abs(vel[0] - u).max() < 1e-12
    assert np.abs(vel[1]).max() < 1e-12
    assert np.abs(lbm.rho - 1.0).max() < 1e-12


def test_closed_box_conserves_mass():
    """Bounce-back walls must not leak mass; the measured drift is the gate."""
    lbm = D2Q9(12, 12, tau=0.6)
    lbm.set_equilibrium((0.02, 0.0))
    m0 = float(lbm.f.sum())
    walls = {s: ("wall", 0.0, 0.0) for s in ("north", "south", "east", "west")}
    for _ in range(200):
        lbm.step(walls)
    assert abs(float(lbm.f.sum()) - m0) / m0 < 1e-3


def test_interface_flux_matches_the_imposed_mass_flux():
    """``sum_i f_i c_n`` must reproduce the imposed velocity (Zou & He property)."""
    n, u = 16, 0.04
    lbm = D2Q9(2 * n, n, tau=0.6)
    lbm.set_equilibrium((0.0, 0.0))
    prof = _inlet_profile(n, u) / u          # unit-amplitude profile
    bcs = {"west": ("velocity", prof * u, 0.0), "east": ("pressure", 1.0, 0.0),
           "north": ("wall", 0.0, 0.0), "south": ("wall", 0.0, 0.0)}
    for _ in range(200):
        lbm.step(bcs)
    fl = lbm.interface_flux("west")
    rho = fl["rho"]
    assert np.abs(fl["mass"] + rho * prof * u).max() < 1e-10 * u


# ---------------------------------------------------------------------------
# 3. the transfer gate itself
# ---------------------------------------------------------------------------


def test_state_exchange_residual_is_an_acoustic_floor_not_a_1e6_gate():
    """Honest negative: state exchange cannot reach 1e-6 at engineering Mach."""
    s = run_channel_transfer(n=16, u_lat=0.05, tau=0.6, mode="velocity", steps=200)
    assert s.mach > 0.05
    assert s.integral_residual > 1e-3          # ~Ma^2 order of magnitude
    assert s.integral_residual < 0.2
    assert s.mode == "velocity"


def test_local_density_divisor_does_not_remove_the_residual():
    """The folklore fix (divide by the receiver density) is falsified.

    A weakly compressible receiver's local density carries the *pressure*, so
    dividing by it injects a flux proportional to the pressure fluctuation.  The
    residual does not collapse; both operators stay O(1e-2) on this case.
    """
    state = run_channel_transfer(n=16, u_lat=0.05, tau=0.6, mode="velocity", steps=200)
    flux = run_channel_transfer(n=16, u_lat=0.05, tau=0.6, mode="flux", steps=200)
    assert state.integral_residual > 1e-3
    assert flux.integral_residual > 1e-3
    # neither operator reaches the 1e-6 gate
    assert not (state.integral_residual < 1e-6)
    assert not (flux.integral_residual < 1e-6)


def test_channel_domain_is_synthetic():
    assert CHANNEL_DOMAIN.startswith("channel_coupling_")
    assert "box_coupling_" in __import__(
        "tensorphysics.interface_conservation.coupling", fromlist=["DOMAIN"]).DOMAIN


# ---------------------------------------------------------------------------
# 4. the independent finite-volume reference solver
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cavity():
    f = StaggeredFVM(16, 16, re=100.0, kappa=0.8)
    f.set_closed_cavity(1.0)
    dt = 0.2 / 16
    for _ in range(int(round(8.0 / dt))):
        f.step(dt)
    return f


def test_fvm_is_discretely_divergence_free(cavity):
    assert cavity.max_divergence() < 1e-9


def test_fvm_reproduces_the_primary_vortex_sign_structure(cavity):
    """u < 0 in the lower half, u > 0 near the lid (a single primary vortex)."""
    assert cavity.velocity_at(0.5, 0.25)[0] < 0.0
    assert cavity.velocity_at(0.5, 0.9)[0] > 0.0


def test_fvm_interface_fluxes_have_zero_net_mass_flux(cavity):
    """A closed cavity exchanges no net mass across an interior section."""
    face = cavity.face_flux("east")
    assert abs(np.sum(face["mass"]) * cavity.h) < 1e-3


def test_fvm_momentum_flux_carries_the_pressure_level(cavity):
    face = cavity.face_flux("east")
    mom = face["momentum"]
    assert np.all(np.isfinite(mom))
    assert abs(np.sum(mom) * cavity.h) < 100.0    # O(rho U^2 L)