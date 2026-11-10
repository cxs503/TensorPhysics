# -*- coding: utf-8 -*-
"""Regularisation-chain tests: PDE (Helmholtz) filter, Heaviside projection,
composed chain adjoints, and the mesh-independence diagnostics.

The chain ``design density -> filter -> projection`` must satisfy three
properties that the whole SIMP acceptance surface rests on:

* the filter preserves a constant field (no spurious mass source/sink) and its
  transpose is the exact adjoint of its forward operator;
* the projection derivative is the analytic derivative of the closed form
  (checked against finite differences of the projection itself);
* the *composed* adjoint sensitivity ``dC/drho`` equals central finite
  differences of the composed compliance -- this is the only test that can
  catch a wrong chain-rule order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.topology_optimization.simp.density_filter import DensityFilter
from tensorphysics.topology_optimization.simp.fem2d import Q4Mesh
from tensorphysics.topology_optimization.simp.filter_acceptance import (
    BENCHMARKS,
    _level_h,
    convergence_order,
    jaccard_similarity,
    manufactured_compliance,
    radius_for_level,
    refine_ratio,
    sample_to_grid,
)
from tensorphysics.topology_optimization.simp.filter_chain import (
    FilterChain,
    HeavisideProjection,
    IdentityFilter,
    PdeFilter,
    adjoint_sensitivity_chain,
    check_radius_above_mesh,
    fd_sensitivity_chain,
    make_filter_chain,
    pde_radius_from_hat_radius,
    run_simp_chain,
)
from tensorphysics.topology_optimization.simp.problems import build_cantilever


# --------------------------------------------------------------------- PDE
def test_pde_filter_preserves_constant_field():
    f = PdeFilter(24, 12, pde_radius_from_hat_radius(1.5))
    assert np.allclose(f.apply(np.ones(f.n_elems)), 1.0, atol=1e-12)


def test_pde_filter_transpose_is_exact_adjoint():
    rng = np.random.default_rng(3)
    f = PdeFilter(20, 10, pde_radius_from_hat_radius(1.5))
    rho = rng.random(f.n_elems)
    g = rng.random(f.n_elems)
    lhs = float(g @ f.apply(rho))
    rhs = float(f.apply_transpose(g) @ rho)
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)


def test_pde_filter_mass_is_approximately_conserved():
    f = PdeFilter(30, 15, pde_radius_from_hat_radius(2.0))
    rho = np.zeros(f.n_elems)
    rho[f.n_elems // 2] = 1.0
    w = f.apply(rho)
    assert w.min() >= 0.0
    assert abs(w.sum() - 1.0) < 0.05


def test_pde_effective_radius_matches_hat_filter():
    """The ``r_min/(2*sqrt(3))`` choice must land near the hat filter's width."""
    nelx, nely = 40, 20
    pde = PdeFilter(nelx, nely, pde_radius_from_hat_radius(1.5))
    hat = DensityFilter(nelx, nely, 1.5)
    centre = (nely // 2) * nelx + nelx // 2
    imp = np.zeros(nelx * nely)
    imp[centre] = 1.0
    w = hat.apply(imp)
    ys, xs = np.meshgrid(np.arange(nely), np.arange(nelx), indexing="ij")
    ey, ex = divmod(centre, nelx)
    d2 = (xs.ravel() - ex) ** 2 + (ys.ravel() - ey) ** 2
    r_hat = float(np.sqrt((w * d2).sum() / w.sum()))
    r_pde = pde.effective_radius(centre)
    assert 0.9 < r_pde / r_hat < 1.6  # discrete kernels agree in width class


def test_pde_filter_validates_input_size():
    f = PdeFilter(8, 4, 0.4)
    with pytest.raises(ValueError):
        f.apply(np.ones(f.n_elems + 1))
    with pytest.raises(ValueError):
        f.apply_transpose(np.ones(f.n_elems + 1))
    with pytest.raises(ValueError):
        PdeFilter(8, 4, 0.0)


# --------------------------------------------------------------- projection
def test_heaviside_endpoints_and_monotonicity():
    proj = HeavisideProjection(beta=8.0, eta=0.5)
    vals = proj.apply(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert vals[0] == pytest.approx(0.0, abs=1e-12)
    assert vals[-1] == pytest.approx(1.0, abs=1e-12)
    assert vals[2] == pytest.approx(0.5, abs=1e-12)
    assert np.all(np.diff(vals) > 0.0)


def test_heaviside_derivative_matches_finite_difference():
    proj = HeavisideProjection(beta=12.0, eta=0.5)
    rho = np.linspace(0.01, 0.99, 41)
    h = 1e-7
    fd = (proj.apply(rho + h) - proj.apply(rho - h)) / (2.0 * h)
    assert np.allclose(proj.derivative(rho), fd, rtol=1e-6, atol=1e-8)


def test_heaviside_sharpens_with_beta():
    """Larger beta pushes the field further from eta (toward black/white)."""
    rho = np.array([0.4, 0.45, 0.55, 0.6])
    soft = HeavisideProjection(beta=2.0).apply(rho)
    sharp = HeavisideProjection(beta=16.0).apply(rho)
    assert np.all(np.abs(sharp - 0.5) >= np.abs(soft - 0.5) - 1e-12)
    assert np.all((sharp >= 0.0) & (sharp <= 1.0))
    # the limit is a step function: every value ends up on a side
    hard = HeavisideProjection(beta=200.0).apply(rho)
    assert np.all((hard < 0.01) | (hard > 0.99))


def test_heaviside_rejects_bad_parameters():
    with pytest.raises(ValueError):
        HeavisideProjection(beta=0.0)
    with pytest.raises(ValueError):
        HeavisideProjection(beta=1.0, eta=1.5)


# -------------------------------------------------------------------- chain
@pytest.mark.parametrize("kind", ["none", "density", "pde", "density+heaviside",
                                  "pde+heaviside"])
def test_chain_adjoint_matches_finite_difference(kind):
    prob = build_cantilever(14, 7, 0.5)
    mesh = prob.mesh
    rng = np.random.default_rng(7)
    rho = np.clip(0.5 + 0.2 * (rng.random(mesh.n_elems) - 0.5), 0.05, 0.95)
    chain = make_filter_chain(kind, mesh.nelx, mesh.nely, radius=1.5, beta=8.0,
                              hx=mesh.hx, hy=mesh.hy)
    C, _, dc = adjoint_sensitivity_chain(
        mesh, rho, prob.F, prob.fixed_dofs, chain)
    assert C is not None and np.isfinite(C)
    fd = fd_sensitivity_chain(mesh, rho, prob.F, prob.fixed_dofs, chain,
                              delta=1e-6, workers=1)
    assert np.linalg.norm(dc - fd) / np.linalg.norm(dc) < 1e-4


def test_chain_transpose_is_the_jacobian_transpose():
    """Adjoint identity on the linearised operator: <g, J d> == <J^T g, d>.

    For the non-linear chain (Heaviside) the identity only holds for the
    Jacobian at the recorded state, which is exactly how the optimiser uses it.
    """
    chain = make_filter_chain("pde+heaviside", 16, 8, radius=1.5, beta=8.0)
    rng = np.random.default_rng(11)
    rho = np.clip(rng.random(chain.n_elems), 0.05, 0.95)
    g = rng.random(chain.n_elems)
    d = rng.standard_normal(chain.n_elems)
    chain.apply(rho)  # records the state the Jacobian is evaluated at
    jt = chain.apply_transpose(g)
    h = 1e-7
    jd = (chain.apply(rho + h * d) - chain.apply(rho - h * d)) / (2.0 * h)
    lhs = float(g @ jd)
    rhs = float(jt @ d)
    assert abs(lhs - rhs) <= 1e-5 * max(abs(lhs), 1.0)


def test_chain_volume_gradient_sums_to_one():
    for kind in ("none", "density", "pde", "pde+heaviside"):
        chain = make_filter_chain(kind, 18, 9, radius=1.5, beta=8.0)
        dv = chain.volume_gradient()
        assert dv.shape == (chain.n_elems,)
        assert dv.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(dv >= 0.0)


def test_identity_filter_is_neutral():
    f = IdentityFilter(12)
    x = np.linspace(0.0, 1.0, 12)
    assert np.allclose(f.apply(x), x)
    assert np.allclose(f.apply_transpose(x), x)


def test_unknown_chain_kind_raises():
    with pytest.raises(ValueError):
        make_filter_chain("navier-stokes", 4, 4)


def test_chain_with_projection_has_no_matrix():
    chain = make_filter_chain("density+heaviside", 6, 3, radius=1.5, beta=8.0)
    with pytest.raises(ValueError):
        _ = chain.matrix
    linear = make_filter_chain("density", 6, 3, radius=1.5)
    assert linear.matrix.shape == (18, 18)


def test_radius_above_mesh_guard():
    with pytest.raises(ValueError):
        check_radius_above_mesh(0.5, 1.0)
    with pytest.raises(ValueError):
        check_radius_above_mesh(1.0, 1.0)
    check_radius_above_mesh(1.5, 1.0)  # must not raise


# ------------------------------------------------------------- optimisation
def test_filtered_design_suppresses_checkerboard():
    prob = build_cantilever(30, 15, 0.4)
    rng = np.random.default_rng(5)
    rho0 = 0.4 + 0.1 * (rng.random(prob.n_elems) - 0.5)
    unf = run_simp_chain(prob, kind="none", iters=30, tol=1e-3, rho0=rho0)
    fil = run_simp_chain(prob, kind="pde+heaviside", radius=1.5, beta=8.0,
                         iters=30, tol=1e-3, rho0=rho0)
    assert fil.checkerboard_index < 0.5 * unf.checkerboard_index
    assert 0.0 <= fil.checkerboard_index <= 1.0
    assert fil.history and fil.iterations == len(fil.history)


def test_run_simp_chain_result_round_trip():
    prob = build_cantilever(20, 10, 0.5)
    res = run_simp_chain(prob, kind="density", radius=1.5, iters=12, tol=1e-3)
    d = res.as_dict()
    for key in ("problem", "nelx", "nely", "aspect_ratio", "volfrac",
                "kind", "compliance", "checkerboard_index", "grey_fraction",
                "volume_fraction", "iterations", "converged"):
        assert key in d
    assert d["aspect_ratio"] == pytest.approx(2.0)
    assert res.rho.size == prob.n_elems
    assert res.rho_physical.size == prob.n_elems


# ------------------------------------------------------- mesh diagnostics
def test_sample_to_grid_identity_and_volume():
    rng = np.random.default_rng(2)
    rho = rng.random(40 * 20)
    same = sample_to_grid(rho, 40, 20, 40, 20)
    assert np.allclose(same, rho.reshape(20, 40))
    up = sample_to_grid(rho, 40, 20, 80, 40)
    assert up.shape == (40, 80)
    assert up.mean() == pytest.approx(rho.mean(), abs=2e-3)


def test_jaccard_limits():
    a = np.zeros(100)
    a[:40] = 1.0
    assert jaccard_similarity(a, a) == pytest.approx(1.0)
    b = np.zeros(100)
    b[60:] = 1.0
    assert jaccard_similarity(a, b) == pytest.approx(0.0)
    c = np.zeros(100)
    c[20:60] = 1.0  # |A n C| = 20, |A u C| = 60
    assert jaccard_similarity(a, c) == pytest.approx(20 / 60)


def test_jaccard_is_symmetric_and_bounded():
    rng = np.random.default_rng(4)
    a = (rng.random(500) > 0.5).astype(float)
    b = (rng.random(500) > 0.5).astype(float)
    j1, j2 = jaccard_similarity(a, b), jaccard_similarity(b, a)
    assert j1 == pytest.approx(j2)
    assert 0.0 <= j1 <= 1.0


def test_convergence_order_recovers_second_order():
    exact = 100.0
    a = 3.0
    cs = [exact + a * h ** 2 for h in (0.1, 0.05, 0.025)]
    out = convergence_order(*cs, refine=2.0)
    assert out["order"] == pytest.approx(2.0, abs=1e-9)
    assert out["monotone"] is True


def test_convergence_order_flags_non_monotone():
    out = convergence_order(101.0, 99.0, 100.5)
    assert out["monotone"] is False

# ------------------------------------------------- manufactured FE anchor
def test_manufactured_probe_converges_second_order():
    """The Q4 kernel must show its 2nd-order rate on a smooth manufactured problem."""
    nelx = (24, 36, 48)
    cs = [manufactured_compliance(n, n // 2) for n in nelx]
    out = convergence_order(*cs, hs=tuple(nelx[0] / n for n in nelx))
    assert 1.8 < out["order"] < 2.2
    assert out["monotone"] is True
    assert cs[-1] > 0.0


def test_benchmark_geometry_is_uniformly_refined_and_lh_exact():
    for name, cfg in BENCHMARKS.items():
        assert refine_ratio(cfg) > 1.0
        for _, nx, ny in cfg["levels"]:
            assert nx / ny == pytest.approx(cfg["aspect_ratio"])
        # the physical filter radius is the same at every level (protocol B)
        radii = [radius_for_level(cfg, nx) for _, nx, _ in cfg["levels"]]
        assert radii[0] == pytest.approx(1.5)
        assert all(r > 1.0 for r in radii)
        assert _level_h(cfg)[0] == pytest.approx(1.0)
