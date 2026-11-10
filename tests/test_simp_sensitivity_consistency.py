# -*- coding: utf-8 -*-
"""SIMP adjoint vs finite-difference sensitivity consistency tests.

The adjoint sensitivity uses the stationarity of the compliance functional
(``dC/drho = -u^T dK/drho u``); the finite-difference reference re-solves the
FE problem with one design variable perturbed.  Agreement is therefore a real
cross-check: an error in the adjoint expression, in the SIMP slope or in the
filter chain rule shows up as a mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.topology_optimization.simp import (
    DensityFilter,
    adjoint_sensitivity,
    build_cantilever,
    complex_step_sensitivity,
    fd_sensitivity,
    sensitivity_error_stats,
    step_convergence_study,
)


def _grey_design(n_elems: int, seed: int = 0):
    """Well-conditioned grey design (no deep voids) for differentiating."""
    rng = np.random.default_rng(seed)
    return 0.35 + 0.3 * rng.random(n_elems)


@pytest.fixture(scope="module")
def problem():
    return build_cantilever(12, 6, 0.4)


def test_adjoint_matches_central_fd_unfiltered(problem):
    rho = _grey_design(problem.n_elems)
    C, dc_t, dc = adjoint_sensitivity(problem.mesh, rho, problem.F, problem.fixed_dofs)
    assert C is not None and C > 0
    fd = fd_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs,
        delta=1e-6, scheme="central", workers=1,
    )
    stats = sensitivity_error_stats(dc, fd)
    # thresholds follow the campaign acceptance level (max 1e-4 / mean 1e-5);
    # the residual is the central-difference roundoff floor ~ eps_eff*C/Delta.
    assert stats["max_abs"] < 1e-4
    assert stats["mean_abs"] < 1e-5
    assert stats["max_rel"] < 1e-4
    # unfiltered: physical == design sensitivity
    assert np.allclose(dc, dc_t, rtol=0, atol=0)


def test_adjoint_matches_central_fd_with_filter(problem):
    rho = _grey_design(problem.n_elems, seed=1)
    dens = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 1.5)
    _, dc_t, dc = adjoint_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=dens
    )
    fd = fd_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs,
        delta=1e-6, scheme="central", dens_filter=dens, workers=1,
    )
    stats = sensitivity_error_stats(dc, fd)
    assert stats["max_abs"] < 1e-4
    assert stats["mean_abs"] < 1e-5
    # the filter chain rule actually changed the field
    assert np.max(np.abs(dc - dc_t)) > 1e-3


def test_complex_step_matches_adjoint(problem):
    rho = _grey_design(problem.n_elems, seed=2)
    dens = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 1.5)
    _, _, dc = adjoint_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=dens
    )
    cs = complex_step_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=dens, workers=1
    )
    stats = sensitivity_error_stats(dc, cs)
    assert stats["max_abs"] < 1e-8


def test_sensitivities_are_non_positive(problem):
    """Adding material can never increase the compliance of a SIMP design."""
    rho = _grey_design(problem.n_elems, seed=4)
    dens = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 1.5)
    _, _, dc = adjoint_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=dens
    )
    assert dc.max() <= 1e-12


def test_fd_error_decreases_with_step(problem):
    rho = _grey_design(problem.n_elems, seed=5)
    dens = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 1.5)
    _, _, dc = adjoint_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=dens
    )
    idx = np.arange(0, problem.n_elems, 6)  # 12 sample elements
    errs = []
    for delta in (1e-3, 1e-4, 1e-5):
        fd = fd_sensitivity(
            problem.mesh, rho, problem.F, problem.fixed_dofs,
            delta=delta, scheme="forward", dens_filter=dens,
            elements=idx, workers=1,
        )
        errs.append(sensitivity_error_stats(dc[idx], fd)["mean_abs"])
    assert errs[0] > errs[1] > errs[2]


def test_negative_control_zero_radius_filter(problem):
    """r -> 0: the filter degenerates to the identity for density *and* sensitivity."""
    rho = _grey_design(problem.n_elems, seed=6)
    _, _, dc_raw = adjoint_sensitivity(problem.mesh, rho, problem.F, problem.fixed_dofs)
    tiny = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 0.0)
    _, _, dc_tiny = adjoint_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=tiny
    )
    assert np.max(np.abs(dc_tiny - dc_raw)) < 1e-12
    assert np.max(np.abs(tiny.apply(rho) - rho)) < 1e-15


def test_negative_control_delta_to_zero(problem):
    """Delta rho -> 0: complex-step derivative converges onto the analytic value."""
    rho = _grey_design(problem.n_elems, seed=7)
    dens = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 1.5)
    _, _, dc = adjoint_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs, dens_filter=dens
    )
    idx = np.arange(0, problem.n_elems, 8)
    coarse = complex_step_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs,
        delta=1e-3, dens_filter=dens, elements=idx, workers=1,
    )
    fine = complex_step_sensitivity(
        problem.mesh, rho, problem.F, problem.fixed_dofs,
        delta=1e-8, dens_filter=dens, elements=idx, workers=1,
    )
    e_coarse = sensitivity_error_stats(dc[idx], coarse)["mean_abs"]
    e_fine = sensitivity_error_stats(dc[idx], fine)["mean_abs"]
    assert e_fine < e_coarse
    assert e_fine < 1e-9


def test_error_stats_flags_mismatch(problem):
    rho = _grey_design(problem.n_elems, seed=8)
    _, _, dc = adjoint_sensitivity(problem.mesh, rho, problem.F, problem.fixed_dofs)
    good = sensitivity_error_stats(dc, dc.copy())
    bad = sensitivity_error_stats(dc, dc * 1.01)
    assert good["max_abs"] == 0.0 and good["mean_abs"] == 0.0
    assert bad["max_abs"] > 1e-3
    assert bad["worst_index"] >= 0
    with pytest.raises(ValueError):
        sensitivity_error_stats(dc, dc[:-1])


def test_step_convergence_study_reports_rows(problem):
    rho = _grey_design(problem.n_elems, seed=9)
    dens = DensityFilter(problem.mesh.nelx, problem.mesh.nely, 1.5)
    idx = np.arange(0, problem.n_elems, 10)
    rows = step_convergence_study(
        problem.mesh, rho, problem.F, problem.fixed_dofs,
        deltas=(1e-4, 1e-5, 1e-6), scheme="complex", dens_filter=dens,
        elements=idx, workers=1,
    )
    assert len(rows) == 3
    assert all(r["mean_abs"] >= 0.0 for r in rows)
    assert rows[0]["mean_abs"] > rows[-1]["mean_abs"]
    assert all("monotone" in r for r in rows)


def test_adjoint_sensitivity_singular_design(problem):
    mesh = problem.mesh
    _, _, dc = adjoint_sensitivity(mesh, np.full(mesh.n_elems, 0.5), problem.F, [])
    assert dc is None