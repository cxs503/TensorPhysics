# -*- coding: utf-8 -*-
"""SIMP density filter / FE kernel acceptance tests.

Covers the linear density filter (normalisation, radius -> 0 limit,
checkerboard suppression), the checkerboard diagnostic itself, the plane-stress
Q4 kernel (symmetry, rank, rigid-body modes, constant-strain patch) and the
benchmark problems / OC update.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.topology_optimization.simp import (
    DensityFilter,
    Q4Mesh,
    assemble_stiffness,
    build_cantilever,
    build_mbb_beam,
    checkerboard_index,
    compliance_and_displacements,
    run_simp,
)
from tensorphysics.topology_optimization.simp.fem2d import q4_unit_stiffness
from tensorphysics.topology_optimization.simp.problems import (
    evaluate_binary_design,
    oc_update,
)


# --------------------------------------------------------------------- kernel
def test_q4_element_stiffness_symmetric_and_rank_five():
    k = q4_unit_stiffness(1.0, 1.0)
    assert k.shape == (8, 8)
    assert np.allclose(k, k.T, atol=1e-14)
    # 8x8 plane-stress element: 3 rigid-body modes -> rank 5
    assert np.linalg.matrix_rank(k, tol=1e-8) == 5
    # rows sum to zero (self-equilibrated)
    assert np.allclose(k.sum(axis=1), 0.0, atol=1e-12)


def test_rigid_body_translation_has_zero_energy():
    mesh = Q4Mesh(4, 3)
    K = assemble_stiffness(mesh, 1.0)
    for comp in (0, 1):
        u = np.zeros(mesh.n_dof)
        u[comp::2] = 1.0
        assert abs(u @ (K @ u)) < 1e-10


def test_constant_strain_patch_is_exact():
    """u_x = eps*x, u_y = -nu*eps*y is reproduced exactly by bilinear Q4."""
    nu, eps, E = 0.3, 1e-3, 1.0
    mesh = Q4Mesh(5, 4)
    x, y = mesh.coords
    u = np.zeros(mesh.n_dof)
    u[0::2] = eps * x
    u[1::2] = -nu * eps * y
    K = assemble_stiffness(mesh, E)
    energy = 0.5 * u @ (K @ u)
    area_total = mesh.n_elems * mesh.hx * mesh.hy
    # uniaxial stress sigma_xx = E*eps on a unit-thickness domain
    expected = 0.5 * E * eps ** 2 * area_total
    assert abs(energy - expected) / expected < 1e-12


def test_singular_system_returns_none():
    """Unconstrained structure -> singular reduced system, reported as None."""
    mesh = Q4Mesh(3, 2)
    F = np.zeros(mesh.n_dof)
    F[1] = -1.0
    C, u = compliance_and_displacements(mesh, np.full(mesh.n_elems, 0.5), F, [])
    assert C is None and u is None


# --------------------------------------------------------------------- filter
def test_zero_radius_filter_is_identity():
    dens = DensityFilter(12, 6, 0.0)
    assert dens.is_identity()
    rng = np.random.default_rng(3)
    rho = rng.random(72)
    assert np.allclose(dens.apply(rho), rho)


def test_filter_rows_are_normalised_and_preserve_constants():
    dens = DensityFilter(10, 5, 1.5)
    P = dens.matrix
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12)
    assert np.allclose(P @ np.full(50, 0.37), 0.37, atol=1e-12)
    # all weights non-negative -> the filtered field stays inside the data range
    rng = np.random.default_rng(7)
    rho = rng.random(50)
    rho_f = dens.apply(rho)
    assert rho_f.min() >= rho.min() - 1e-12
    assert rho_f.max() <= rho.max() + 1e-12


def test_filter_kernel_is_symmetric_hat():
    dens = DensityFilter(6, 6, 1.5)
    H = dens._H.toarray()
    assert np.allclose(H, H.T)
    # distance-1 neighbour weight = r - 1, diagonal = r, distance > r -> 0
    assert H[0, 0] == pytest.approx(1.5)
    assert H[0, 1] == pytest.approx(0.5)
    assert H[0, 2] == pytest.approx(0.0)


def test_checkerboard_index_limits():
    nely, nelx = 8, 10
    ii, jj = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    perfect = ((ii + jj) % 2).astype(float)
    assert checkerboard_index(perfect) == pytest.approx(1.0)
    assert checkerboard_index(np.full((nely, nelx), 0.42)) == pytest.approx(0.0)
    ramp = (ii + 0.0 * jj) / (nely - 1)
    assert checkerboard_index(ramp) < 1e-12


def test_checkerboard_index_rejects_degenerate_input():
    with pytest.raises(ValueError):
        checkerboard_index(np.zeros(10))
    with pytest.raises(ValueError):
        checkerboard_index(np.zeros((2, 2)))


def test_filter_suppresses_checkerboard_index():
    nelx, nely = 40, 20
    ii, jj = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    field = ((ii + jj) % 2).astype(float)
    raw = checkerboard_index(field)
    filtered = checkerboard_index(DensityFilter(nelx, nely, 1.5).apply(field).reshape(nely, nelx))
    assert raw > 0.9
    assert filtered < 0.05
    assert filtered < raw / 10.0


# ------------------------------------------------------------------ problems
def test_mbb_beam_boundary_conditions():
    prob = build_mbb_beam(20, 10, 0.5)
    mesh = prob.mesh
    # all left-edge x-DOFs + bottom-right corner y-DOF are supported
    assert mesh.node_dofs(0, 5)[0] in prob.fixed_dofs
    assert mesh.node_dofs(20, 0)[1] in prob.fixed_dofs
    # single unit downward load at the top-left corner
    assert prob.F[mesh.node_dofs(0, 0)[1]] == -1.0
    assert np.count_nonzero(prob.F) == 1
    C, u = compliance_and_displacements(
        mesh, np.full(mesh.n_elems, 0.5), prob.F, prob.fixed_dofs
    )
    assert C is not None and C > 0.0
    assert np.all(np.isfinite(u))


def test_cantilever_boundary_conditions():
    prob = build_cantilever(20, 10, 0.4)
    mesh = prob.mesh
    # clamped left edge: both DOFs of every left-edge node
    for iy in range(11):
        ux, uy = mesh.node_dofs(0, iy)
        assert ux in prob.fixed_dofs and uy in prob.fixed_dofs
    # tip load at mid-height of the right edge
    assert prob.F[mesh.node_dofs(20, 5)[1]] == -1.0
    assert np.count_nonzero(prob.F) == 1


def test_oc_update_satisfies_volume_fraction():
    n = 100
    rho = np.full(n, 0.5)
    dc = -np.ones(n)
    dv = np.ones(n) / n
    new = oc_update(rho, dc, dv, 0.4)
    assert np.mean(new) == pytest.approx(0.4, abs=1e-3)
    assert new.min() >= 0.0 and new.max() <= 1.0
    # move limit is respected
    assert np.max(np.abs(new - rho)) <= 0.2 + 1e-12


def test_binary_design_projection():
    prob = build_cantilever(10, 5, 0.4)
    rho = np.where(np.arange(prob.n_elems) % 2 == 0, 1.0, 0.0)
    C, solid = evaluate_binary_design(prob, rho)
    assert set(np.unique(solid)) <= {0.0, 1.0}
    assert C is not None and C > 0.0


def test_simp_run_filtered_suppresses_checkerboard():
    """Coarse MBB: the filter removes alternating density modes."""
    unf = run_simp(build_mbb_beam(20, 10, 0.5), radius=None, iters=30, tol=0.0)
    fil = run_simp(build_mbb_beam(20, 10, 0.5), radius=1.5, iters=30, tol=0.0)
    assert np.isfinite(unf["compliance"]) and np.isfinite(fil["compliance"])
    assert fil["checkerboard_index"] < 0.35 * unf["checkerboard_index"]
    assert fil["checkerboard_index"] < 0.1
    assert unf["compliance"] < unf["history"][0]["compliance"]  # optimisation works
    assert fil["compliance"] < fil["history"][0]["compliance"]


def test_compliance_ordering_under_filter():
    """C_filter >= C_no_filter: the filter restricts the design space.

    This is the variational statement that falsifies the campaign's literal
    "C_no_filter / C_filter >= 5" criterion (the measured ratio is <= 1).
    """
    unf = run_simp(build_mbb_beam(20, 10, 0.5), radius=None, iters=40, tol=0.0)
    fil = run_simp(build_mbb_beam(20, 10, 0.5), radius=1.5, iters=40, tol=0.0)
    assert fil["compliance"] >= unf["compliance"]
    assert unf["compliance"] / fil["compliance"] < 1.0


def test_zero_radius_run_equals_unfiltered_run():
    """Negative control: a radius-0 filter is exactly the unfiltered problem."""
    a = run_simp(build_cantilever(10, 5, 0.4), radius=None, iters=12, tol=0.0)
    b = run_simp(build_cantilever(10, 5, 0.4), radius=0.0, iters=12, tol=0.0)
    assert np.allclose(a["rho"], b["rho"], atol=1e-14)
    assert a["compliance"] == pytest.approx(b["compliance"], rel=1e-12)