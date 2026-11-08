"""Verification tests for the geometric multigrid Poisson solver.

Each test maps onto one of the acceptance criteria C1--C8 of the campaign node
(see :mod:`tensorphysics.multigrid_poisson.verification`).  The heavy
end-to-end record is produced by ``run_verification()``; here the same physics
is asserted on small grids so that ``pytest`` stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.multigrid_poisson import benchmarks as bm
from tensorphysics.multigrid_poisson.core import (
    MultigridSolver,
    analytic_solution,
    convergence_history,
    discrete_operator,
    poisson_rhs,
    prolong_bilinear,
    red_black_gs_sweeps,
    residual,
    restrict_full_weighting,
    jacobi_sweeps,
)
from tensorphysics.multigrid_poisson.tla import (
    fine_symbol,
    periodic_lfa_max,
    periodic_two_grid_spectral_radius,
    smoothing_factor,
    tla_characteristic_coefficients,
    tla_closed_form_radius,
    tla_spectral_radius,
    tla_symbol_matrix,
    tla_two_grid_factor,
)
from tensorphysics.multigrid_poisson.verification import run_verification, sparse_direct_solve

PRIMARY = dict(nu1=1, nu2=1, smoother="jacobi", omega=0.8)


# ---------------------------------------------------------------------------
# transfers / discrete operator
# ---------------------------------------------------------------------------
def test_restriction_is_partition_of_unity():
    """Full weighting preserves constants (its 16 weights sum to one)."""
    n = 16
    const = np.ones((n + 1, n + 1))
    r = restrict_full_weighting(const)
    assert np.abs(r[1:-1, 1:-1] - 1.0).max() < 1e-15
    # stencil check: a unit spike at an even fine point contributes its 4/16
    # centre weight to exactly one coarse point, and the restriction conserves
    # the integral (sum of coarse values = 1/4 sum of fine values)
    spike = np.zeros((n + 1, n + 1))
    spike[8, 8] = 1.0
    rs = restrict_full_weighting(spike)
    assert rs[4, 4] == pytest.approx(4.0 / 16.0)
    assert np.abs(rs.sum() - 0.25) < 1e-15
    # a spike at an odd (edge-neighbour) point splits 2/16 + 2/16 between the
    # two adjacent coarse points
    spike[9, 8] = 1.0
    spike[8, 8] = 0.0
    rs = restrict_full_weighting(spike)
    assert rs[4, 4] == pytest.approx(2.0 / 16.0)
    assert rs[5, 4] == pytest.approx(2.0 / 16.0)


def test_prolongation_reproduces_linear_field_exactly():
    """Bilinear prolongation is exact for functions linear in x and y."""
    n = 8
    xc = np.linspace(0.0, 1.0, n // 2 + 1)
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    uc = 1.0 + 2.0 * X - 3.0 * Y
    uf = prolong_bilinear(uc)
    xf = np.linspace(0.0, 1.0, n + 1)
    XF, YF = np.meshgrid(xf, xf, indexing="ij")
    assert np.abs(uf - (1.0 + 2.0 * XF - 3.0 * YF)).max() < 1e-14


def test_fine_symbol_matches_explicit_periodic_operator():
    """The 5-point symbol ``4-2cos wx-2cos wy`` reproduces the assembled
    periodic Laplacian eigenvalues (validates the discretisation convention used
    by the two-level analysis)."""
    from tensorphysics.multigrid_poisson.tla import _periodic_laplacian

    n = 8
    h = 1.0 / n
    A = _periodic_laplacian(n, h)
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for kx, ky in ((1, 2), (3, 3), (2, 5)):
        wx, wy = 2 * np.pi * kx / n, 2 * np.pi * ky / n
        v = np.exp(1j * (wx * X + wy * Y)).reshape(-1)
        lam = (A @ v) / v
        assert np.abs(lam - fine_symbol((wx, wy)) / (h * h)).max() < 1e-10


def test_poisson_rhs_is_negative_laplacian_of_analytic_solution():
    """Truncation check: ``A u* -> f = 2 pi^2 sin(pi x) sin(pi y)`` at O(h^2)."""
    errs = []
    for n in (32, 64):
        h = 1.0 / n
        u = analytic_solution(n)
        r = residual(u, poisson_rhs(n), h)
        errs.append(np.abs(r).max())
    order = np.log(errs[0] / errs[1]) / np.log(2.0)
    assert 1.9 < order < 2.1


# ---------------------------------------------------------------------------
# C4: numerical vs analytical spectral radius
# ---------------------------------------------------------------------------
def test_closed_form_quartic_matches_eigenvalues():
    """C4a: roots of the closed-form quartic == eigenvalues of the 4x4 symbol."""
    worst = 0.0
    for tx in np.linspace(0.05, 1.5, 7):
        for ty in np.linspace(0.05, 1.5, 7):
            worst = max(
                worst,
                abs(tla_spectral_radius((tx, ty), 1, 1, 0.8)
                    - tla_closed_form_radius((tx, ty), 1, 1, 0.8)),
            )
    assert worst < 1e-6, worst


def test_lfa_symbol_matches_explicit_periodic_operator():
    """C4b: LFA prediction == spectral radius of the assembled periodic operator."""
    for nu1, nu2, om in ((1, 1, 0.8), (2, 2, 2.0 / 3.0)):
        op = periodic_two_grid_spectral_radius(16, nu1, nu2, om)
        lfa = periodic_lfa_max(16, nu1, nu2, om)
        assert abs(op["rho"] - lfa["rho_lfa"]) < 1e-8, (nu1, nu2, om)


def test_closed_form_characteristic_polynomial():
    """The closed-form quartic coefficients equal ``det(lambda I - M)`` built
    from the eigenvalues of the 4x4 symbol matrix."""
    th = (0.4, 0.9)
    M = tla_symbol_matrix(th, 1, 1, 0.8)
    c_cf = tla_characteristic_coefficients(th, 1, 1, 0.8)  # descending, monic
    c_num = np.poly(np.linalg.eigvals(M))                 # descending, monic
    assert np.abs(c_cf - c_num).max() < 1e-12


# ---------------------------------------------------------------------------
# C1 / C2 / C5: cycle factors vs two-level analysis
# ---------------------------------------------------------------------------
def test_vcycle_factor_matches_tla():
    """C1: V-cycle convergence factor vs the closed-form TLA factor (<=5%)."""
    rho_tla = tla_two_grid_factor(1, 1, 0.8)["rho_tla"]
    solver = MultigridSolver(n_fine=64, cycle="V", **PRIMARY)
    hist = convergence_history(solver, 64, cycles=30)
    rel = abs(hist["rho_asymptotic"] - rho_tla) / rho_tla
    assert rel <= 0.05, (hist["rho_asymptotic"], rho_tla, rel)


def test_wcycle_not_worse_than_vcycle_and_matches_tla():
    """C2: ``rho_W <= rho_V`` and W vs TLA within 5%."""
    rho_tla = tla_two_grid_factor(1, 1, 0.8)["rho_tla"]
    rv = convergence_history(MultigridSolver(n_fine=64, cycle="V", **PRIMARY), 64, 30)["rho_asymptotic"]
    rw = convergence_history(MultigridSolver(n_fine=64, cycle="W", **PRIMARY), 64, 30)["rho_asymptotic"]
    assert rw <= rv * (1 + 1e-6), (rw, rv)
    assert abs(rw - rho_tla) / rho_tla <= 0.05, (rw, rho_tla)


def test_two_level_operator_matches_tla_on_dirichlet_grid():
    """Two-grid operator assembled on the Dirichlet grid vs LFA (<=5%)."""
    rho_tla = tla_two_grid_factor(1, 1, 0.8)["rho_tla"]
    tg = bm.two_grid_operator_spectral_radius(16, 1, 1, "jacobi", 0.8)
    assert abs(tg["rho"] - rho_tla) / rho_tla <= 0.05, (tg["rho"], rho_tla)


def test_grid_independence_of_vcycle():
    """C5: rho_V changes by <=5% between N=64 and N=128."""
    r64 = convergence_history(MultigridSolver(n_fine=64, cycle="V", **PRIMARY), 64, 30)["rho_asymptotic"]
    r128 = convergence_history(MultigridSolver(n_fine=128, cycle="V", **PRIMARY), 128, 30)["rho_asymptotic"]
    assert abs(r64 - r128) / (0.5 * (r64 + r128)) <= 0.05, (r64, r128)


def test_smoothing_factor_values():
    """Textbook smoothing factors of damped Jacobi for the 2-D 5-point stencil."""
    assert smoothing_factor(2.0 / 3.0) == pytest.approx(1.0 / 3.0, abs=1e-2)
    assert smoothing_factor(0.8) == pytest.approx(0.6, abs=1e-2)


# ---------------------------------------------------------------------------
# C6 / C7: negative controls
# ---------------------------------------------------------------------------
def test_jacobi_spectral_radius_matches_cos_pi_h():
    """C6a: exact Jacobi iteration-matrix spectrum == cos(pi h)."""
    res = bm.jacobi_iteration_matrix_spectral_radius(16)
    assert res["rel_error"] <= 1e-6, res


def test_jacobi_does_not_reach_tolerance():
    """C6b: Jacobi stagnates - 200 sweeps leave almost the whole residual."""
    out = bm.jacobi_solver(poisson_rhs(128), omega=1.0, max_iter=200)
    assert not out["converged"]
    assert out["residual_history"][-1] / out["residual_history"][0] > 0.9


def test_overdamped_jacobi_diverges():
    """C6c: ``omega > 1`` makes the high-frequency symbol exceed 1 -> divergence."""
    n = 32
    h = 1.0 / n
    rng = np.random.default_rng(0)
    u = np.zeros((n + 1, n + 1))
    u[1:-1, 1:-1] = rng.standard_normal((n - 1, n - 1))
    f0 = np.zeros_like(u)
    u0 = np.abs(u).max()
    for _ in range(50):
        u = jacobi_sweeps(u, f0, h, omega=1.1, sweeps=1)
    assert np.abs(u).max() / u0 > 1.0


def test_low_frequency_stagnation_vs_multigrid():
    """C7: the slowest mode stagnates under Jacobi, is damped by multigrid."""
    slow = bm.lowest_mode_decay(n=128, iterations=100, mg_cycles=10, nu1=1, nu2=1, omega=0.8)
    assert slow["jacobi_error_after"] >= 0.95
    assert slow["mg_rate"] <= 0.30
    # the measured Jacobi decay of the pure mode equals the analytic rate
    assert slow["jacobi_rate"] == pytest.approx(slow["jacobi_rate_analytic_slow_mode"], rel=1e-9)


# ---------------------------------------------------------------------------
# C3 / C8: work and accuracy
# ---------------------------------------------------------------------------
def test_multigrid_solves_faster_than_jacobi_at_same_tolerance():
    """C3 (small-grid form): cycle count is O(1) whereas Jacobi needs O(N^2)."""
    n = 128
    f = poisson_rhs(n)
    mg = MultigridSolver(n_fine=n, **PRIMARY)
    res = mg.solve(f, tol=1e-10, max_cycles=100)
    assert res["converged"] and res["cycles"] <= 30
    jac = bm.jacobi_solver(f, omega=1.0, max_iter=300)
    assert not jac["converged"]  # Jacobi is nowhere near the same tolerance
    # work accounting: multigrid cost in fine-grid sweeps << Jacobi iteration count
    units = bm.work_units_per_cycle(1, 1, mg.n_levels)["smoothing_fine_sweeps_per_cycle"]
    needed = np.log(1e-10) / np.log(np.cos(np.pi / n))
    assert units * res["cycles"] * 10 < needed


def test_solution_matches_independent_direct_solve():
    """C8: multigrid solution vs an independent sparse direct solve (<=1e-6)."""
    n = 64
    f = poisson_rhs(n)
    mg = MultigridSolver(n_fine=n, **PRIMARY)
    res = mg.solve(f, tol=1e-11, max_cycles=100)
    u_ref = sparse_direct_solve(f)  # SciPy sparse LU, CG fallback if absent
    assert np.abs(res["u"] - u_ref).max() <= 1e-6


def test_discretisation_second_order():
    """The recovered accuracy is the discretisation error, with order ~2."""
    study = bm.discretisation_error_study(sizes=(32, 64, 128))
    orders = [r["observed_order"] for r in study["rows"] if r["observed_order"]]
    assert all(abs(o - 2.0) < 0.05 for o in orders), orders


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------
def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        MultigridSolver(n_fine=63)
    with pytest.raises(ValueError):
        MultigridSolver(n_fine=64, smoother="sor")
    with pytest.raises(ValueError):
        MultigridSolver(n_fine=64, cycle="F")
    with pytest.raises(ValueError):
        MultigridSolver(n_fine=64, n_coarse=5)
    with pytest.raises(ValueError):
        MultigridSolver(n_fine=2, n_coarse=4)


def test_levels_and_cycle_count():
    s = MultigridSolver(n_fine=64)
    assert s.levels == [64, 32, 16, 8, 4]
    assert s.n_levels == 5
    out = s.solve(poisson_rhs(64), tol=1e-10, max_cycles=40)
    assert s.cycles_applied == out["cycles"] > 0
    assert out["converged"]


def test_red_black_gs_smoother_converges_and_is_monotone():
    """Red-black GS as a *solver* converges at the slowest-mode rate
    ``~1 - pi^2 h^2`` -- it is only useful as a smoother."""
    n = 16
    h = 1.0 / n
    u = np.zeros((n + 1, n + 1))
    f = poisson_rhs(n)
    hist = [np.abs(residual(u, f, h)).max()]
    for _ in range(60):
        u = red_black_gs_sweeps(u, f, h, sweeps=1)
        hist.append(np.abs(residual(u, f, h)).max())
    # monotone decrease after the initial transient
    assert all(b <= a * (1 + 1e-12) for a, b in zip(hist[4:], hist[5:])), hist[:8]
    assert hist[-1] < 0.5 * hist[0]
    rate = (hist[-1] / hist[4]) ** (1.0 / 56.0)
    assert rate == pytest.approx(1.0 - np.pi**2 * h * h, rel=0.02), rate


def test_gs_multigrid_vcycle_solves():
    s = MultigridSolver(n_fine=64, nu1=1, nu2=1, smoother="gs", cycle="V")
    res = s.solve(poisson_rhs(64), tol=1e-11, max_cycles=40)
    assert res["converged"]
    assert res["cycles"] <= 20


@pytest.mark.parametrize("n_coarse", [4, 8])
def test_coarse_level_independence(n_coarse):
    s = MultigridSolver(n_fine=64, n_coarse=n_coarse, **PRIMARY)
    res = s.solve(poisson_rhs(64), tol=1e-10, max_cycles=60)
    assert res["converged"]
    assert s.levels[-1] == n_coarse


def test_verification_record_fast_mode():
    """End-to-end: the machine-checkable criteria all pass in ``fast`` mode."""
    rec = run_verification(n_fine=64, fast=True)
    failed = [c["criterion"] for c in rec["checks"] if not c["passed"]]
    assert not failed, [(c["criterion"], c["detail"]) for c in rec["checks"] if not c["passed"]]
    assert len(rec["checks"]) >= 8