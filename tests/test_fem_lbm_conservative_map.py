"""Tests for the FEM -> LBM conservative scalar remap (campaign node p399w2-n06).

Every test targets one of the two properties the node promises (global conservation,
no new extrema) or one of the stages that make them provable (geometry gate, source
integrals, patch test, bounded redistribution, LBM bookkeeping).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.fem_lbm_conservative_map.acceptance import run_acceptance
from tensorphysics.fem_lbm_conservative_map.lattice import CartesianLattice, build_overlap
from tensorphysics.fem_lbm_conservative_map.mesh import (TriMesh, make_perturbed_tri_mesh,
                                                        manufactured_linear, manufactured_sine)
from tensorphysics.fem_lbm_conservative_map.remap import (conservation_report,
                                                          conservative_repair,
                                                          exact_cell_average,
                                                          l2_error_vs_exact, map_scalar_field,
                                                          monotonicity_report, _barth_jespersen)
from tensorphysics.fem_lbm_conservative_map.thermal_lbm import (_stream5, _stream9, _CS2,
                                                                run_thermal_lbm)

MODES = ("average", "linear", "bounded")


# ---------------------------------------------------------------------------
# geometry gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [8, 16, 24, 32])
def test_mesh_geometry_gate(n):
    mesh = make_perturbed_tri_mesh(n, jitter=0.25, seed=7 + n)
    q = mesh.validate(expected_area=1.0)
    assert q["ok"], q
    assert q["n_inverted"] == 0
    assert q["area_error_rel"] < 1e-12


def test_mesh_detects_inverted_element():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tri = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 1]])  # last element is clockwise
    mesh = TriMesh(nodes=nodes, tri=tri)
    q = mesh.validate()
    assert q["n_inverted"] == 1
    assert not q["ok"]


def test_mesh_is_unstructured():
    """The mesh must not be lattice-aligned, otherwise the test case is trivial."""
    mesh = make_perturbed_tri_mesh(16, jitter=0.25, seed=3)
    dy = np.abs(mesh.nodes[:, 1] - np.round(mesh.nodes[:, 1] * 16) / 16).max()
    assert dy > 1e-3


# ---------------------------------------------------------------------------
# FEM source solver
# ---------------------------------------------------------------------------
def test_fem_linear_field_is_exact():
    mesh = make_perturbed_tri_mesh(12, jitter=0.2, seed=5)
    sol = manufactured_linear(mesh, t_bottom=1.0, t_top=0.0)
    assert np.max(np.abs(sol["phi"] - sol["exact_nodal"])) < 1e-12
    assert sol["solve_residual_norm"] < 1e-10


def test_fem_manufactured_solution_converges():
    errs = []
    for n in (8, 16, 32):
        mesh = make_perturbed_tri_mesh(n, jitter=0.2, seed=11 + n)
        sol = manufactured_sine(mesh)
        errs.append(np.sqrt(np.mean((sol["phi"] - sol["exact_nodal"]) ** 2)))
    order = np.polyfit(np.log([1 / 8, 1 / 16, 1 / 32]), np.log(errs), 1)[0]
    assert order > 1.5, f"P1 heat solver convergence order {order:.2f}"
    assert errs[-1] < 0.02


# ---------------------------------------------------------------------------
# overlap table and operators
# ---------------------------------------------------------------------------
def _pair(n_mesh=24, n_lat=48, seed=3, domain=(0.0, 0.0, 1.0, 1.0)):
    mesh = make_perturbed_tri_mesh(n_mesh, n_mesh, jitter=0.25, seed=seed, domain=domain)
    lattice = CartesianLattice(nx=n_lat, ny=n_lat,
                               length_x=domain[2] - domain[0], length_y=domain[3] - domain[1])
    return mesh, lattice, build_overlap(mesh, lattice)


def test_overlap_partition_of_domain():
    mesh, lattice, table = _pair()
    cov = table.coverage
    assert abs(cov.sum() - mesh.areas().sum()) < 1e-12
    assert np.allclose(table.coverage_fraction(), 1.0, atol=1e-12)


def test_source_cell_integral_is_exact_for_p1():
    mesh = make_perturbed_tri_mesh(10, jitter=0.2, seed=2)
    a, b, c = 0.3, 1.1, -0.7
    phi = a + b * mesh.nodes[:, 0] + c * mesh.nodes[:, 1]
    exact = a + b * mesh.centroids()[:, 0] + c * mesh.centroids()[:, 1]
    assert np.max(np.abs(mesh.cell_integrals(phi) - exact * mesh.areas())) < 1e-12


def test_patch_test_linear_field_all_modes():
    mesh, lattice, table = _pair(n_mesh=16, n_lat=32)
    a, b, c = 0.7, -1.3, 2.1
    phi = a + b * mesh.nodes[:, 0] + c * mesh.nodes[:, 1]
    ref = exact_cell_average(lattice, lambda xy: a + b * xy[:, 0] + c * xy[:, 1])
    for mode in MODES:
        res = map_scalar_field(mesh, phi, table, mode=mode)
        assert l2_error_vs_exact(lattice, res.phi, ref)["linf_abs"] < 1e-12, mode


def test_global_conservation_of_bounded_operator():
    mesh, lattice, table = _pair()
    sol = manufactured_sine(mesh)
    for mode in ("average", "bounded"):
        res = map_scalar_field(mesh, sol["phi"], table, mode=mode)
        cr = conservation_report(res)
        assert cr["rel_defect"] < 1e-12, (mode, cr)


def test_unlimited_reconstruction_breaks_conservation_on_coarse_target():
    """Negative control: without limiting + redistribution neither property survives."""
    mesh = make_perturbed_tri_mesh(48, jitter=0.25, seed=11)
    lattice = CartesianLattice(nx=24, ny=24)
    table = build_overlap(mesh, lattice)
    fn = lambda xy: np.exp(-((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2) / 0.03 ** 2)  # noqa: E731
    phi = fn(mesh.nodes)
    bad = map_scalar_field(mesh, phi, table, mode="linear")
    good = map_scalar_field(mesh, phi, table, mode="bounded")
    assert conservation_report(bad)["rel_defect"] > 1e-2
    assert conservation_report(good)["rel_defect"] < 1e-12
    assert monotonicity_report(mesh, phi, bad)["n_cells_below_nodal_min"] > 0
    mr = monotonicity_report(mesh, phi, good)
    assert mr["n_cells_above_nodal_max"] == 0 and mr["n_cells_below_nodal_min"] == 0


@pytest.mark.parametrize("name,fn", [
    ("hot_spot", lambda xy: np.exp(-((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2) / 0.02 ** 2)),
    ("front", lambda xy: 0.5 * (1.0 + np.tanh((xy[:, 1] - 0.5) / 0.01))),
])
def test_no_new_extrema_on_steep_fields(name, fn):
    mesh = make_perturbed_tri_mesh(32, jitter=0.25, seed=13)
    lattice = CartesianLattice(nx=16, ny=16)
    table = build_overlap(mesh, lattice)
    phi = fn(mesh.nodes)
    res = map_scalar_field(mesh, phi, table, mode="bounded")
    mr = monotonicity_report(mesh, phi, res)
    assert mr["n_cells_above_nodal_max"] == 0, name
    assert mr["n_cells_below_nodal_min"] == 0, name
    assert mr["target_min"] >= phi.min() - 1e-12
    assert mr["target_max"] <= phi.max() + 1e-12


# ---------------------------------------------------------------------------
# limiter / repair building blocks
# ---------------------------------------------------------------------------
def test_barth_jespersen_factor_bounds_values():
    rng = np.random.default_rng(0)
    phi_avg = rng.random(200)
    d = 0.3 * rng.standard_normal(200)
    phi_lin = phi_avg + d
    lo = phi_avg - 0.05
    hi = phi_avg + 0.05
    psi, limited = _barth_jespersen(phi_avg, phi_lin, lo, hi)
    assert np.all(psi >= 0.0) and np.all(psi <= 1.0)
    val = phi_avg + psi * d
    assert np.all(val >= lo - 1e-15) and np.all(val <= hi + 1e-15)
    assert limited.sum() > 0


def test_conservative_repair_restores_integral_without_leaving_bounds():
    rng = np.random.default_rng(1)
    area = np.full(64, 1.0)
    phi = rng.random(64)
    lo = np.zeros(64)
    hi = np.ones(64)
    target = float(phi @ area) + 3.0  # ask for extra mass
    out, info = conservative_repair(phi, area, target, lo, hi)
    assert abs(out @ area - target) < 1e-10
    assert np.all(out >= lo - 1e-14) and np.all(out <= hi + 1e-14)
    assert info["feasible"]


def test_conservative_repair_reports_infeasible_case():
    area = np.full(4, 1.0)
    phi = np.array([1.0, 1.0, 1.0, 1.0])
    out, info = conservative_repair(phi, area, 100.0, np.zeros(4), np.ones(4))
    assert not info["feasible"]
    assert abs(out @ area - 4.0) < 1e-9  # saturated at the bound, no overshoot


# ---------------------------------------------------------------------------
# thermal LBM bookkeeping
# ---------------------------------------------------------------------------
def test_streaming_conserves_mass_exactly():
    rng = np.random.default_rng(2)
    f = rng.random((7, 9, 9))
    out, _ = _stream9(f)
    assert abs(out.sum() - f.sum()) < 1e-12
    g = rng.random((7, 9, 5))
    out5, _, _ = _stream5(g, None, None)
    assert abs(out5.sum() - g.sum()) < 1e-12


def test_dirichlet_wall_budget_closes():
    rng = np.random.default_rng(3)
    g = rng.random((7, 9, 5))
    out, q_bottom, q_top = _stream5(g, 1.0, 0.0)
    assert abs(out.sum() - g.sum() - (q_bottom + q_top)) < 1e-12


def test_conduction_profile_and_nusselt():
    ny = nx = 17
    y = np.linspace(0.0, 1.0, ny)
    T0 = np.repeat((1.0 - y)[:, None], nx, axis=1)
    res = run_thermal_lbm(T0, steps=150, Ra=0.0, hot_wall=T0[0], cold_wall=T0[-1], nu=0.1)
    profile = res.T.mean(axis=1)
    y_nodes = y[:]  # nodes sit on the wall temperatures passed in
    assert np.max(np.abs(profile - (1.0 - y_nodes))) < 1e-12
    assert abs(np.nanmean(res.nusselt_history[-20:]) - 1.0) < 1e-8


def test_insulated_box_conserves_energy():
    ny = nx = 17
    y = np.linspace(0.0, 1.0, ny)
    T0 = np.repeat((1.0 - y)[:, None], nx, axis=1)
    res = run_thermal_lbm(T0, steps=120, Ra=0.0, hot_wall=None, cold_wall=None)
    drift = abs(res.energy_history[-1] - res.energy_history[0]) / res.energy_history[0]
    assert drift < 1e-12
    assert res.budget_mismatch < 1e-9
    assert res.wall_budget[-1] == 0.0


# ---------------------------------------------------------------------------
# end-to-end acceptance
# ---------------------------------------------------------------------------
def test_acceptance_quick_all_pass():
    payload = run_acceptance(quick=True)
    failed = [c["id"] for c in payload["criteria"] if not c["pass"]]
    assert not failed, f"failed criteria: {failed}"
    ids = {c["id"] for c in payload["criteria"]}
    for required in ("C4_remap_conservation_average", "C6_monotonicity_bounded",
                     "C15_coupled_exchange_integral", "C17_coupled_energy_balance"):
        assert required in ids