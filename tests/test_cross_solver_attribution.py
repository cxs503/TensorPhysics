"""Acceptance tests for the cross-solver attribution node (P78-N08).

The tests are deliberately split into cheap unit checks (analytic solutions,
solver orders, decomposition algebra) and two bounded numerical legs (SPH
Poiseuille, LBM/SPH celerity at a modest resolution) so the whole file runs in
under a minute.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.cross_solver_attribution import attribution as A
from tensorphysics.cross_solver_attribution import cases as C
from tensorphysics.cross_solver_attribution import orders as O
from tensorphysics.cross_solver_attribution.dg import (SolverConfig, dg_solve_scalar,
                                                        dg_solve_swe)
from tensorphysics.cross_solver_attribution.fvm import (fvm_solve_scalar,
                                                         fvm_solve_swe)
from tensorphysics.cross_solver_attribution.lbm import lbm_linear_wave, lbm_poiseuille
from tensorphysics.cross_solver_attribution.sph import sph_poiseuille, sph_shallow_wave


# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------
def test_advection_sine_is_exact_solution():
    case = C.advection_sine(t_end=0.5)
    x = np.linspace(0.05, 0.95, 7)
    dt, dx = 1e-5, 1e-5
    u_t = (case.exact(x, dt) - case.exact(x, -dt)) / (2 * dt)
    u_x = (case.exact(x + dx, 0.0) - case.exact(x - dx, 0.0)) / (2 * dx)
    assert np.allclose(u_t + case.a * u_x, 0.0, atol=1e-4)


def test_diffusion_front_is_exact_heat_solution():
    case = C.diffusion_front()
    x = np.linspace(0.4, 1.6, 9)
    h = 1e-5
    u_xx = (case.exact(x + h, 0.1) - 2 * case.exact(x, 0.1) + case.exact(x - h, 0.1)) / h ** 2
    dt = 1e-6
    u_t = (case.exact(x, 0.1 + dt) - case.exact(x, 0.1)) / dt
    assert np.allclose(u_t, case.nu * u_xx, atol=2e-3)


def test_cell_averages_converge_to_exact_mean():
    case = C.advection_sine(t_end=0.5)
    n = 64
    edges = np.linspace(case.x0, case.x1, n + 1)
    fine = np.linspace(edges[0], edges[1], 200001)
    mean0 = float(np.trapezoid(case.exact(fine, case.t_end), fine) / (edges[1] - edges[0]))
    assert abs(case.exact_averages(n)[0] - mean0) < 1e-9


def test_ritter_invariants():
    case = C.ritter_dry_bed(h0=0.5, t_end=0.3)
    c0 = math.sqrt(case.g * case.h0)
    # depth at the dam section is 4 h0 / 9 for every h0
    h_dam, _ = case.exact(np.array([case.dam_x]), case.t_end)
    assert abs(float(h_dam[0]) - 4.0 * case.h0 / 9.0) < 1e-9
    # water released past the dam section is exactly (8/27) h0 c0 t
    xs = np.linspace(case.dam_x, case.dam_x + 2.0 * c0 * case.t_end, 40001)
    h, u = case.exact(xs, case.t_end)
    released = float(np.trapezoid(h, xs))
    assert abs(released - (8.0 / 27.0) * case.h0 * c0 * case.t_end) \
        / ((8.0 / 27.0) * case.h0 * c0 * case.t_end) < 1e-4
    # front speed is exactly 2 c0
    front = case.dam_x + 2.0 * c0 * case.t_end
    h_front, u_front = case.exact(np.array([front - 1e-9]), case.t_end)
    assert abs(float(h_front[0])) < 1e-6
    assert abs(float(u_front[0]) - 2.0 * c0) < 1e-4
    # Riemann invariant across the rarefaction fan (wet points only: the dry
    # state at the front has h = 0 and carries no invariant)
    wet = h > 1e-6
    inv = u[wet] + 2.0 * np.sqrt(case.g * h[wet])
    assert np.allclose(inv, 2.0 * c0, atol=1e-6)


def test_lake_at_rest_is_steady_solution():
    case = C.lake_at_rest(bump=True)
    x = np.linspace(0.0, 2.0, 21)
    h, u = case.exact(x, 0.0)
    assert np.allclose(u, 0.0)
    assert np.allclose(h + case.bed(x), case.h0, atol=1e-14)


# ---------------------------------------------------------------------------
# order of accuracy (C1 ingredients)
# ---------------------------------------------------------------------------
def test_dg_second_order_on_advection():
    case = C.advection_sine(t_end=0.5)
    cfg = SolverConfig(cfl=0.2, boundary="periodic")
    study = O.order_study(case, dg_solve_scalar, resolutions=(20, 40, 80),
                          cfg=cfg, degree=1)
    assert O.in_band(study["observed_order"], 2.0, 0.15), study


def test_dg_third_order_on_diffusion():
    case = C.diffusion_front()
    cfg = SolverConfig(cfl=0.4, boundary="analytic")
    study = O.order_study(case, dg_solve_scalar, resolutions=(20, 40, 80),
                          cfg=cfg, degree=2)
    assert O.in_band(study["observed_order"], 3.0, 0.15), study


def test_fvm_orders_on_advection():
    case = C.advection_sine(t_end=0.5)
    cfg = SolverConfig(cfl=0.2, boundary="periodic")
    first = O.order_study(case, fvm_solve_scalar, resolutions=(20, 40, 80),
                          cfg=cfg, order=1)
    second = O.order_study(case, fvm_solve_scalar, resolutions=(20, 40, 80),
                           cfg=cfg, order=2)
    assert O.in_band(first["observed_order"], 1.0, 0.15), first
    assert O.in_band(second["observed_order"], 2.0, 0.15), second


def test_lake_at_rest_preserved_by_both_schemes():
    case = C.lake_at_rest(bump=True)
    cfg = SolverConfig(cfl=0.3, boundary="periodic")
    dg = dg_solve_swe(case, 40, cfg, degree=1)
    fvm = fvm_solve_swe(case, 40, cfg, order=2)
    assert dg.finite and fvm.finite
    assert np.max(np.abs(dg.cell_averages[1])) < 1e-8
    assert np.max(np.abs(fvm.cell_averages[1])) < 1e-12


# ---------------------------------------------------------------------------
# decomposition algebra
# ---------------------------------------------------------------------------
def test_disagreement_obeys_triangle_inequality():
    case = C.advection_sine(t_end=0.5)
    cfg = SolverConfig(cfl=0.2, boundary="periodic", limiter=False)
    a = dg_solve_scalar(case, 40, cfg, degree=1)
    b = fvm_solve_scalar(case, 40, cfg, order=1)
    ref = case.exact_averages(40)
    d = A.disagreement(a, b, case)
    e_a = C.relative_l2(a.cell_averages, ref)
    e_b = C.relative_l2(b.cell_averages, ref)
    assert d <= e_a + e_b + 1e-12
    assert d > 0.0


def test_shapley_efficiency_and_anova_residual():
    # a synthetic three-factor design with a known interaction
    values = {
        frozenset(): 0.0,
        frozenset({"T"}): 0.10,
        frozenset({"B"}): 0.02,
        frozenset({"L"}): 0.01,
        frozenset({"T", "B"}): 0.125,
        frozenset({"T", "L"}): 0.115,
        frozenset({"B", "L"}): 0.035,
        frozenset({"T", "B", "L"}): 0.14,
    }
    phi = A.shapley(values)
    total = values[frozenset(A.FACTORS)] - values[frozenset()]
    assert abs(sum(phi.values()) - total) < 1e-12
    main = A.anova_main_effects(values)
    assert abs(sum(main.values()) - total) <= total


def test_factorial_ledger_closes_and_orders_terms():
    adv = C.advection_sine(t_end=0.5)
    leg = A.LegSpec(key="mini", case=adv, kind="scalar", cfl=0.2, n=16,
                    boundary_on="analytic", order_pairs={0: (1, 2), 1: (1, 1)})
    led = A.factorial_ledger(leg)
    assert led["all_finite"]
    assert led["closure_rel_residual"] < 1e-12
    assert led["dominant_term"] == "T"
    assert led["shares"]["T"] > 0.5


def test_negative_control_kills_the_l_term():
    """With the L factor reduced to sub-cycling only, the L term must vanish."""
    adv = C.advection_sine(t_end=0.5)
    leg = A.LegSpec(key="mini", case=adv, kind="scalar", cfl=0.2, n=16,
                    boundary_on="analytic", order_pairs={0: (1, 2), 1: (1, 1)})
    on = A.factorial_ledger(leg)
    ctl = A.factorial_ledger(leg, control=True)
    assert abs(on["shapley"]["L"]) > 5.0 * max(abs(ctl["shapley"]["L"]), 1e-12)


# ---------------------------------------------------------------------------
# SPH and LBM legs (bounded resolution)
# ---------------------------------------------------------------------------
def test_sph_poiseuille_peak_matches_analytic():
    res = sph_poiseuille(n_fluid=320)
    assert res["peak_rel_error"] < 0.03, res["peak_rel_error"]
    # the profile must not be a plug flow: peak / mean -> 1.5 for a parabola
    assert res["peak_over_mean_numeric"] > 1.4


def test_lbm_poiseuille_peak_matches_analytic():
    res = lbm_poiseuille(nx=2, ny=21, force=8e-5, max_steps=40000, tol=1e-8)
    assert res["peak_rel_error"] < 0.02, res["peak_rel_error"]


def test_wave_celerity_legs():
    lbm = lbm_linear_wave(cells=100)
    sph = sph_shallow_wave(n_particles=100)
    assert lbm["celerity_rel_error"] < 0.01, lbm["celerity_rel_error"]
    assert sph["celerity_rel_error"] < 0.05, sph["celerity_rel_error"]