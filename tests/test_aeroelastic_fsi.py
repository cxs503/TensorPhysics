"""Tests for the aeroelastic FSI module (campaign node P21-N04).

The suite is deliberately fast (< 20 s) and covers:

* fluid model: exact steady lift slope and aerodynamic centre (spectral UVLM);
* state-space (Wagner/Jones) unsteady aerodynamics vs exact Theodorsen forces;
* interface gates C1 (momentum/work conjuacy) and C2 (energy closure);
* coupled FSI: static-divergence onset vs the analytic boundary;
* added-mass instability: explicit staggered coupling diverges at large
  added-mass ratio while the implicit sub-iterated coupling stays bounded
  (campaign gates C5 / C6, including the negative control).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.aeroelastic import (Case, GlauertVortexLattice,
                                       RationalAero, run_coupled,
                                       theodorsen_c, time_domain_run)
from tensorphysics.aeroelastic.flutter import envelope_growth
from tensorphysics.aeroelastic.interface import (generalised_force, panel_grid,
                                                 work_conjugacy_residual)
from tensorphysics.aeroelastic.theodorsen import (aero_force_harmonic,
                                                  divergence_speed,
                                                  flutter_boundary_tracked)
from tensorphysics.aeroelastic.wagner import c_rational

RHO = 1.225
B = 0.5


def test_steady_lift_slope_and_aerodynamic_centre():
    """Exact thin-aerofoil limits: L = 2 pi rho U^2 b alpha, AC at c/4."""
    U, alpha = 10.0, 0.02
    aero = GlauertVortexLattice(semi_chord=B, n_terms=20, rho=RHO,
                                elastic_axis=-0.5)
    aero.reset(U)
    st = aero.solve(0.0, alpha, 0.0, 1e-3, steady=True)
    theory = 2.0 * np.pi * RHO * U ** 2 * B * alpha
    assert abs(st.lift - theory) / theory < 1e-9
    # zero moment about c/4; the residual is the quadrature error of the
    # leading-edge 1/sqrt pressure singularity on the theta grid
    assert abs(st.moment) / theory < 5e-3

    aero_mid = GlauertVortexLattice(semi_chord=B, n_terms=20, rho=RHO,
                                    elastic_axis=0.0)
    aero_mid.reset(U)
    st_mid = aero_mid.solve(0.0, alpha, 0.0, 1e-3, steady=True)
    # lift acts at the quarter chord -> moment about mid-chord is L*b/2
    assert abs(st_mid.moment / (st.lift * B / 2.0) - 1.0) < 5e-3


def test_theodorsen_limits_and_fit():
    assert abs(theodorsen_c(1e-4) - 1.0) < 1e-3
    assert abs(abs(theodorsen_c(50.0)) - 0.5) < 5e-2
    ks = np.geomspace(0.005, 20.0, 200)
    err = np.abs(c_rational(ks) - theodorsen_c(ks)) / np.abs(theodorsen_c(ks))
    assert err.max() < 5e-3


@pytest.mark.parametrize("k", [0.05, 0.2, 0.5])
def test_state_space_aero_matches_theodorsen(k):
    """Time-domain (Wagner/Jones) force vs exact Theodorsen, prescribed motion."""
    U, a = 10.0, -0.5
    om = k * U / B
    dt = 2.0 * np.pi / om / 300
    h0 = al0 = 1e-4
    aero = RationalAero(semi_chord=B, rho=RHO, elastic_axis=a)
    aero.reset(U)
    st = aero.solve(h0 * om, 0.0, al0 * om, dt, steady=True)
    aero.commit(st, dt)
    F, ts = [], []
    for n in range(1, 9 * 300 + 1):
        t = n * dt
        s = aero.solve(h0 * om * np.cos(om * t), al0 * np.sin(om * t),
                       al0 * om * np.cos(om * t), dt)
        aero.commit(s, dt)
        add = aero.added_mass_force(-h0 * om * om * np.sin(om * t),
                                    -al0 * om * om * np.sin(om * t))
        F.append([s.lift + add[0], s.moment + add[1]])
        ts.append(t)
    F = np.array(F)
    ts = np.array(ts)
    m = ts >= 0.5 * ts[-1]
    w = ts[m]

    def amp(y):
        # least-squares harmonic fit (exact for a pure sinusoid, independent of
        # how many periods the window contains)
        A = np.column_stack([np.ones(w.size), np.cos(om * w), np.sin(om * w)])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        return complex(coef[1], -coef[2])

    ref = aero_force_harmonic(k, U, B, a, RHO, -1e-4j, -1e-4j, 1.0)
    for got, want in ((amp(F[:, 0]), ref[0]), (amp(F[:, 1]), ref[1])):
        assert abs(got - want) / abs(want) < 0.01


def test_interface_work_conjugacy_is_machine_precise():
    """C1: the flux the fluid sends is the flux the structure receives."""
    x_mid, dx = panel_grid(B, 200, -0.5)
    rng = np.random.default_rng(3)
    dp = rng.normal(size=x_mid.size) * 1e2
    for h_dot, alpha_dot in ((1e-3, 2e-3), (-4.0, 7.0), (0.0, 0.0)):
        res = work_conjugacy_residual(x_mid, dp, dx, -0.25, h_dot, alpha_dot)
        assert res < 1e-12
    lift, moment = generalised_force(x_mid, dp, dx, -0.25)
    # projecting the same field through the operator twice changes nothing
    assert abs(generalised_force(x_mid, dp, dx, -0.25)[0] - lift) == 0.0
    assert abs(generalised_force(x_mid, dp, dx, -0.25)[1] - moment) == 0.0


def _growth(case, U, dt, t_end, scheme, **kw):
    run = time_domain_run(case, U, dt, t_end, scheme=scheme, **kw)
    return run


def test_divergence_onset_matches_analytic_boundary():
    """Coupled FSI reproduces U_D = sqrt(k_alpha / (2 pi rho b^2 (a+1/2)))."""
    case = Case(semi_chord=B, mass=20.0, x_alpha=0.1, r_alpha2=0.25,
                elastic_axis=0.0, omega_h=32.0, omega_alpha=80.0, rho=RHO)
    sec = case.section()
    Ud = divergence_speed(sec, RHO)
    pb = flutter_boundary_tracked(sec, RHO)
    assert pb["converged"] and pb["kind"] == "divergence"
    assert abs(pb["U_crit"] - Ud) / Ud < 5e-3
    T = 2.0 * np.pi / (0.5 * (case.omega_h + case.omega_alpha))
    g_below = _growth(case, 0.85 * Ud, T / 240, 12 * T, "tight").growth
    g_above = _growth(case, 1.15 * Ud, T / 240, 12 * T, "tight").growth
    assert g_below < 0.0 < g_above


def test_energy_closure_below_one_percent():
    """C2: fluid work closes against the change of structural energy."""
    case = Case(semi_chord=B, mass=20.0, x_alpha=0.1, r_alpha2=0.25,
                elastic_axis=0.0, omega_h=32.0, omega_alpha=80.0, rho=RHO)
    T = 2.0 * np.pi / (0.5 * (case.omega_h + case.omega_alpha))
    Ud = divergence_speed(case.section(), RHO)
    for scheme in ("tight", "loose"):
        run = time_domain_run(case, 0.9 * Ud, T / 240, 10 * T, scheme=scheme)
        assert run.energy_residual() < 1e-2


def test_added_mass_instability_negative_control():
    """C5/C6: explicit coupling diverges at large added-mass ratio, implicit not.

    Negative control: if the implicit (tight) scheme also diverged, the
    mechanism would not be demonstrated and the test fails.
    """
    light = Case(semi_chord=B, mass=0.4, x_alpha=0.1, r_alpha2=0.25,
                 elastic_axis=0.0, omega_h=32.0, omega_alpha=80.0, rho=RHO)
    sec = light.section()
    assert sec.added_mass_ratio(RHO) > 1.0     # fluid inertia > structure
    T = 2.0 * np.pi / (0.5 * (light.omega_h + light.omega_alpha))
    dt = T / 480
    loose = _growth(light, 10.0, dt, 8 * T, "loose")
    tight = _growth(light, 10.0, dt, 8 * T, "tight")
    assert loose.diverged and not tight.diverged
    assert tight.converged
    assert tight.iterations.max() <= 8


def test_tight_coupling_subiteration_converges_and_is_accurate():
    """Implicit scheme converges in a few Newton steps and stays bounded."""
    case = Case(semi_chord=B, mass=20.0, x_alpha=0.1, r_alpha2=0.25,
                elastic_axis=0.0, omega_h=32.0, omega_alpha=80.0, rho=RHO)
    T = 2.0 * np.pi / (0.5 * (case.omega_h + case.omega_alpha))
    run = time_domain_run(case, 100.0, T / 240, 6 * T, scheme="tight")
    assert run.converged
    assert run.iterations.max() <= 6
    assert abs(run.q[:, 1]).max() < 1e-2


def test_growth_rate_sign_convention():
    t = np.linspace(0.0, 2.0, 2001)
    assert envelope_growth(t, 1e-4 * np.exp(3.0 * t)) == pytest.approx(3.0,
                                                                     rel=0.05)
    assert envelope_growth(t, 1e-4 * np.exp(-2.0 * t)) == pytest.approx(-2.0,
                                                                      rel=0.05)