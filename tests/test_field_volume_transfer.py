"""Tests for the conservative *volumetric* field transfer (LBM -> FEM, DEM -> FVM).

Organisation:

* AC1 -- LBM (Eulerian cell field) -> FEM (Lagrange nodes) stress transfer:
  uniforms, linear and quadratic fields, tensor components, coverage, scheme contrast.
* AC2 -- DEM (Lagrangian particles) -> FVM (Eulerian cells) volume fraction:
  uniform, gradient, cluster, out-of-domain accounting, exact-overlap kernel.
* AC3/AC4 -- conservation residuals of the two primitives against the pre-registered
  1e-8 threshold.
* AC5 -- grid convergence with the Roache GCI helper (three refinements).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.field_exchange.volume_transfer import (
    CartGrid,
    Q1Mesh,
    audit_nodal_field,
    audit_volume_fraction,
    cube_volume,
    deposition_weights,
    map_cell_field_to_nodes,
    map_particles_to_volume_fraction,
    roache_gci,
    sphere_volume,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def square_grid(n: int, length: float = 1.0) -> tuple[CartGrid, np.ndarray]:
    h = length / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    return grid, grid.centers()


def square_mesh(n_elem: int, length: float = 1.0) -> Q1Mesh:
    axis = np.linspace(0.0, length, n_elem + 1)
    return Q1Mesh.structured([axis, axis])


def interior_nodes(mesh: Q1Mesh, margin: float) -> np.ndarray:
    x = mesh.nodes
    return ((x[:, 0] > margin) & (x[:, 0] < 1.0 - margin)
            & (x[:, 1] > margin) & (x[:, 1] < 1.0 - margin))


def nodal_error(nmap, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Weighted relative L2 error of a nodal field, normalised by ``|truth|_inf``."""
    if mask is None:
        mask = np.ones(nmap.mesh.n_nodes, dtype=bool)
    w = nmap.nodal_volumes
    diff = nmap.values[mask, 0] - truth[mask]
    return float(np.sqrt(np.sum(w[mask] * diff ** 2) / np.sum(w[mask]))
                 / np.abs(truth[mask]).max())


# ---------------------------------------------------------------------------
# AC1: LBM (cell field) -> FEM (nodes) stress transfer
# ---------------------------------------------------------------------------
def test_ac1_uniform_tensor_field_reproduced_exactly():
    """A uniform stress tensor must be handed over unchanged (round-off level)."""
    grid, _centres = square_grid(12)
    mesh = square_mesh(6)
    sigma = np.zeros((grid.n_cells, 3, 3))
    sigma[:, 0, 0] = 120.0
    sigma[:, 1, 1] = 80.0
    sigma[:, 2, 2] = 40.0
    sigma[:, 0, 1] = sigma[:, 1, 0] = 15.0
    nmap = map_cell_field_to_nodes(sigma, grid, mesh)
    assert nmap.mask.all()
    assert np.abs(nmap.values - sigma[0]).max() == pytest.approx(0.0, abs=1e-12)
    assert nmap.coverage == pytest.approx(1.0, rel=1e-12)


def test_ac1_linear_field_second_order_and_moments_exact():
    """Linear field: both moments exact, nodal error decreasing at second order."""
    errs = []
    for n in (16, 32, 64):
        grid, centres = square_grid(n)
        mesh = square_mesh(n // 2)
        field = centres[:, :1].copy()          # sigma_xx = x
        nmap = map_cell_field_to_nodes(field, grid, mesh)
        assert nmap.zeroth_moment_residual <= 1e-12
        assert nmap.first_moment_residual <= 1e-12
        mask = interior_nodes(mesh, 2.0 / (n // 2))
        errs.append(nodal_error(nmap, mesh.nodes[:, 0].copy(), mask))
    assert errs[0] > errs[1] > errs[2]
    assert np.log2(errs[0] / errs[1]) > 1.5


def test_ac1_quadratic_field_converges_second_order():
    """Quadratic field: converged nodal values, monotone decreasing error."""
    errs = []
    for n in (16, 32, 64):
        grid, centres = square_grid(n)
        mesh = square_mesh(n // 2)
        field = (centres[:, 0] * centres[:, 1])[:, None]
        nmap = map_cell_field_to_nodes(field, grid, mesh)
        mask = interior_nodes(mesh, 2.0 / (n // 2))
        errs.append(nodal_error(nmap, mesh.nodes[:, 0] * mesh.nodes[:, 1], mask))
    assert errs[0] > errs[1] > errs[2]
    assert 1.5 < np.log2(errs[1] / errs[2]) < 2.5


def test_ac1_tensor_components_map_independently():
    """A tensor field maps exactly like the component-wise stack of scalar fields."""
    grid, centres = square_grid(10)
    mesh = square_mesh(5)
    rng = np.random.default_rng(0)
    sigma = rng.normal(size=(grid.n_cells, 3, 3))
    sigma = 0.5 * (sigma + np.swapaxes(sigma, 1, 2))   # symmetric stress
    tensor_map = map_cell_field_to_nodes(sigma, grid, mesh)
    for i in range(3):
        for j in range(3):
            single = map_cell_field_to_nodes(sigma[:, i, j], grid, mesh)
            assert np.allclose(tensor_map.values[:, i, j], single.values, atol=1e-12)


def test_ac1_nodal_volumes_are_the_element_intgrals_of_the_basis():
    """``V_i = int N_i dV``: nodal volumes must sum to the covered volume."""
    grid, _centres = square_grid(8)
    mesh = square_mesh(4)
    field = np.ones((grid.n_cells, 1))
    nmap = map_cell_field_to_nodes(field, grid, mesh)
    assert nmap.nodal_volumes.sum() == pytest.approx(grid.volume, rel=1e-12)
    # corner node of a uniform 4x4 Q1 mesh carries a quarter element = 1/64 of the area
    corner = int(np.argmin(np.linalg.norm(mesh.nodes, axis=1)))
    assert nmap.nodal_volumes[corner] == pytest.approx(1.0 / 64.0, rel=1e-12)


def test_ac1_partial_mesh_coverage_is_reported_not_hidden():
    """A target mesh covering half the LBM grid must report coverage = 0.5."""
    grid, _centres = square_grid(10)
    mesh = Q1Mesh.structured([np.linspace(0.0, 0.5, 6), np.linspace(0.0, 1.0, 6)])
    nmap = map_cell_field_to_nodes(np.ones((grid.n_cells, 1)), grid, mesh)
    assert nmap.coverage == pytest.approx(0.5, rel=1e-12)
    assert nmap.nodal_volumes.sum() == pytest.approx(0.5 * grid.volume, rel=1e-12)
    assert nmap.zeroth_moment_residual <= 1e-12
    assert nmap.first_moment_residual <= 1e-12


def test_ac1_lumped_scheme_is_conservative_but_less_accurate():
    """The cheap lumped scheme stays conservative; the L2 projection is more accurate."""
    grid, centres = square_grid(16)
    mesh = square_mesh(7)                      # deliberately non-matching mesh
    field = (centres[:, 0] * centres[:, 1])[:, None]
    truth = mesh.nodes[:, 0] * mesh.nodes[:, 1]
    mask = interior_nodes(mesh, 0.29)
    lumped = map_cell_field_to_nodes(field, grid, mesh, scheme="lumped")
    l2 = map_cell_field_to_nodes(field, grid, mesh, scheme="l2")
    assert lumped.zeroth_moment_residual <= 1e-12
    assert lumped.first_moment_residual <= 1e-12
    assert nodal_error(lumped, truth, mask) > nodal_error(l2, truth, mask)


def test_ac1_input_validation():
    grid, _ = square_grid(4)
    mesh = square_mesh(2)
    with pytest.raises(ValueError):
        map_cell_field_to_nodes(np.ones((7, 1)), grid, mesh)
    with pytest.raises(ValueError):
        map_cell_field_to_nodes(np.ones((grid.n_cells, 1)), grid, mesh, scheme="nope")
    with pytest.raises(ValueError):
        Q1Mesh(np.zeros((3, 4)), [[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        CartGrid((0.0, 0.0), (1.0, -1.0), (2, 2))


# ---------------------------------------------------------------------------
# AC2: DEM (particles) -> FVM (cells) volume fraction
# ---------------------------------------------------------------------------
def test_ac2_uniform_particle_distribution_is_exact():
    """Uniform solid fraction: every cell carries exactly the analytic fraction."""
    n, k = 8, 4
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    lp = h / k
    axis = (np.arange(n * k) + 0.5) * lp
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    r = 0.3 * lp
    vf = map_particles_to_volume_fraction(grid, pts, radii=np.full(len(pts), r),
                                          scheme="cic")
    assert vf.mass_residual <= 1e-12
    assert np.abs(vf.alpha - np.pi * r ** 2 / (lp * lp)).max() == pytest.approx(0.0,
                                                                              abs=1e-12)


def test_ac2_gradient_distribution_moments_exact_for_interior_particles():
    """A linear solid-fraction field is reproduced exactly on interior cells."""
    n, k = 16, 4
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    centres = grid.centers()
    lp = h / k
    rel = (np.arange(k) + 0.5) * lp - 0.5 * h
    P = np.stack(np.meshgrid(rel, rel, indexing="ij"), axis=-1).reshape(-1, 2)
    pts = (centres[:, None, :] + P[None, :, :]).reshape(-1, 2)
    alpha_ref = lambda x: 0.10 + 0.20 * x          # noqa: E731  linear in x
    vol = lp * lp * alpha_ref(pts[:, 0])
    vf = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="cic")
    audit = audit_volume_fraction(vf, pts, volumes=vol)
    assert audit["mass_residual"] <= 1e-12
    interior = ((centres[:, 0] > 2 * h) & (centres[:, 0] < 1 - 2 * h)
                & (centres[:, 1] > 2 * h) & (centres[:, 1] < 1 - 2 * h))
    ref = alpha_ref(centres[interior, 0])
    assert np.abs(vf.alpha.ravel()[interior] - ref).max() <= 1e-12


def test_ac2_cluster_distribution_conserves_mass_and_centre():
    """A dense cluster keeps its full volume and its centre of solid volume."""
    n = 16
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    rng = np.random.default_rng(4)
    centre = np.array([0.5, 0.5])
    pts = centre + rng.normal(scale=0.08, size=(300, 2))
    pts = pts[(pts > 0.2).all(axis=1) & (pts < 0.8).all(axis=1)]
    vol = rng.uniform(0.5, 1.5, size=len(pts))
    vf = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="cic")
    audit = audit_volume_fraction(vf, pts, volumes=vol)
    assert audit["mass_residual"] <= 1e-12
    assert audit["first_moment_residual"] <= 1e-12
    # no volume may leak outside the cluster's bounding box plus one hat width
    lo = pts.min(axis=0) - 2 * h
    hi = pts.max(axis=0) + 2 * h
    cs = grid.centers()
    outside = ((cs[:, 0] < lo[0]) | (cs[:, 0] > hi[0])
               | (cs[:, 1] < lo[1]) | (cs[:, 1] > hi[1]))
    assert vf.cell_volume[outside].max() <= 1e-15


def test_ac2_particles_outside_the_domain_are_accounted_for():
    """Out-of-domain particles must be reported, not silently dropped."""
    n = 8
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    pts = np.array([[0.3, 0.3], [0.7, 0.7], [1.2, 0.5], [-0.1, 0.4]])
    vol = np.array([1.0, 1.0, 2.0, 3.0])
    vf = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="cic")
    assert vf.deposited_volume == pytest.approx(2.0, rel=1e-12)
    assert vf.outside_volume == pytest.approx(5.0, rel=1e-12)
    assert vf.mass_residual == pytest.approx(5.0 / 7.0, rel=1e-12)
    audit = audit_volume_fraction(vf, pts, volumes=vol)
    assert audit["outside_fraction"] == pytest.approx(5.0 / 7.0, rel=1e-12)


def test_ac2_exact_overlap_kernel_is_geometrically_exact():
    """``cube`` scheme: a particle inside one cell lands entirely in that cell."""
    n = 4
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    r = 0.1 * h
    pts = np.array([[0.875 * h, 0.875 * h]])
    vf = map_particles_to_volume_fraction(grid, pts, radii=np.array([r]), scheme="cube")
    assert vf.mass_residual <= 1e-14
    assert vf.alpha[0, 0] == pytest.approx(cube_volume(r, 2) / (h * h), rel=1e-12)
    assert vf.alpha.sum() - vf.alpha[0, 0] == pytest.approx(0.0, abs=1e-15)
    # a particle straddling a cell face is split by exact overlap
    pts = np.array([[2 * h, 0.875 * h]])
    vf = map_particles_to_volume_fraction(grid, pts, radii=np.array([r]), scheme="cube")
    assert vf.alpha[1, 0] == pytest.approx(cube_volume(r, 2) / (h * h) / 2.0, rel=1e-12)
    assert vf.alpha[2, 0] == pytest.approx(cube_volume(r, 2) / (h * h) / 2.0, rel=1e-12)
    assert vf.mass_residual <= 1e-12


def test_ac2_cube_scheme_rejects_particles_larger_than_a_cell():
    n = 4
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    with pytest.raises(ValueError):
        map_particles_to_volume_fraction(grid, [[0.5, 0.5]], radii=[2.0 * h],
                                         scheme="cube")


def test_ac2_nearest_grid_point_loses_first_moment_negative_control():
    """NGP lumping is mass conservative but first-moment defective (negative control)."""
    n = 16
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    rng = np.random.default_rng(9)
    pts = rng.uniform(h, 1 - h, size=(200, 2))
    vol = rng.uniform(0.5, 1.5, size=200)
    cic = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="cic")
    ngp = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="ngp")
    cic_audit = audit_volume_fraction(cic, pts, volumes=vol)
    ngp_audit = audit_volume_fraction(ngp, pts, volumes=vol)
    assert cic_audit["mass_residual"] <= 1e-12 and ngp_audit["mass_residual"] <= 1e-12
    assert cic_audit["first_moment_residual"] <= 1e-12
    assert ngp_audit["first_moment_residual"] > 1e-3


def test_ac2_deposition_partition_of_unity_and_bounds():
    grid, _ = square_grid(6)
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 1.0, size=(50, 2))
    dep = deposition_weights(grid, pts, scheme="cic")
    assert np.allclose(dep.weights.sum(axis=1), 1.0, atol=1e-12)
    assert dep.inside.all()
    assert sphere_volume(0.1, 3) == pytest.approx(4.0 / 3.0 * np.pi * 1e-3, rel=1e-12)


# ---------------------------------------------------------------------------
# AC3 / AC4: conservation residuals against the pre-registered 1e-8 threshold
# ---------------------------------------------------------------------------
def test_ac3_uniform_stress_contract_residual_below_1e8_threshold():
    grid, _ = square_grid(16)
    mesh = square_mesh(8)
    sigma = np.zeros((grid.n_cells, 3, 3))
    sigma[:, 0, 1] = sigma[:, 1, 0] = 37.5      # uniform shear traction
    nmap = map_cell_field_to_nodes(sigma, grid, mesh)
    audit = audit_nodal_field(nmap)
    assert audit["zeroth_moment_residual"] <= 1e-8
    assert audit["first_moment_residual"] <= 1e-8
    # the measured values are reported by the acceptance payload; keep the contract tight
    assert audit["zeroth_moment_residual"] <= 1e-13
    assert audit["first_moment_residual"] <= 1e-13
    assert audit["coverage"] == pytest.approx(1.0, rel=1e-12)


def test_ac4_uniform_particles_volume_fraction_residual_below_1e8_threshold():
    n, k = 16, 4
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0), (h, h), (n, n))
    axis = (np.arange(n * k) + 0.5) * (h / k)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    r = 0.25 * h / k
    vf = map_particles_to_volume_fraction(grid, pts, radii=np.full(len(pts), r),
                                          scheme="cic")
    audit = audit_volume_fraction(vf, pts, radii=np.full(len(pts), r))
    assert audit["mass_residual"] <= 1e-8
    assert audit["mass_residual"] <= 1e-13
    assert audit["first_moment_residual"] <= 1e-13
    # total volume fraction equals particle volume / domain volume by construction
    expected = len(pts) * np.pi * r ** 2 / grid.volume
    assert vf.alpha.mean() == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# AC5: grid convergence with the Roache GCI helper
# ---------------------------------------------------------------------------
def _stress_mapping_errors(levels=(16, 32, 64)) -> list[float]:
    errs = []
    for n in levels:
        grid, centres = square_grid(n)
        mesh = square_mesh(n // 2)
        field = (1.0 + 0.3 * np.sin(2 * np.pi * centres[:, 0])
                 * np.cos(2 * np.pi * centres[:, 1]))[:, None]
        nmap = map_cell_field_to_nodes(field, grid, mesh)
        mask = interior_nodes(mesh, 2.0 / (n // 2))
        truth = 1.0 + 0.3 * np.sin(2 * np.pi * mesh.nodes[:, 0]) * np.cos(
            2 * np.pi * mesh.nodes[:, 1])
        errs.append(nodal_error(nmap, truth, mask))
    return errs


def _volume_fraction_errors(levels=(8, 16, 32)) -> list[float]:
    errs = []
    for n in levels:
        k = 4
        h = 1.0 / n
        grid = CartGrid((0.0, 0.0), (h, h), (n, n))
        centres = grid.centers()
        lp = h / k
        rel = (np.arange(k) + 0.5) * lp + 0.3 * lp - 0.5 * h
        P = np.stack(np.meshgrid(rel, rel, indexing="ij"), axis=-1).reshape(-1, 2)
        pts = (centres[:, None, :] + P[None, :, :]).reshape(-1, 2)
        f = lambda x, y: 0.20 + 0.10 * np.sin(2 * np.pi * x) * np.cos(  # noqa: E731
            2 * np.pi * y)
        vol = lp * lp * f(pts[:, 0], pts[:, 1])
        vf = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="cic")
        mask = ((centres[:, 0] > 1.5 / n) & (centres[:, 0] < 1 - 1.5 / n)
                & (centres[:, 1] > 1.5 / n) & (centres[:, 1] < 1 - 1.5 / n))
        ref = f(centres[mask, 0], centres[mask, 1])
        errs.append(float(np.sqrt(np.sum((vf.alpha.ravel()[mask] - ref) ** 2)
                                  / np.sum(ref ** 2))))
    return errs


def test_ac5_stress_mapping_converges_monotonically_second_order():
    errs = _stress_mapping_errors()
    gci = roache_gci(errs, ratio=2.0)
    assert gci["monotone"]
    assert 1.5 < gci["observed_order"] < 2.5
    assert gci["gci_fine"] < 1e-3


def test_ac5_volume_fraction_mapping_converges_monotonically_second_order():
    errs = _volume_fraction_errors()
    gci = roache_gci(errs, ratio=2.0)
    assert gci["monotone"]
    assert 1.5 < gci["observed_order"] < 2.5
    assert gci["gci_fine"] < 1e-2


def test_roache_gci_validation_and_negative_slope():
    clean = roache_gci([4.0e-2, 1.0e-2, 2.5e-3])
    assert clean["observed_order"] == pytest.approx(2.0, abs=1e-9)
    assert clean["monotone"]
    with pytest.raises(ValueError):
        roache_gci([1.0, 0.1])
    with pytest.raises(ValueError):
        roache_gci([1.0, -0.1, 0.01])

# ---------------------------------------------------------------------------
# three-dimensional smoke tests (the operators are dimension generic)
# ---------------------------------------------------------------------------
def test_3d_volume_fraction_and_hex_projection_stay_conservative():
    n = 6
    h = 1.0 / n
    grid = CartGrid((0.0, 0.0, 0.0), (h, h, h), (n, n, n))
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.1, 0.9, size=(60, 3))
    vol = rng.uniform(0.1, 0.3, size=60)
    cic = map_particles_to_volume_fraction(grid, pts, volumes=vol, scheme="cic")
    audit = audit_volume_fraction(cic, pts, volumes=vol)
    assert audit["mass_residual"] <= 1e-12
    assert audit["first_moment_residual"] <= 1e-12
    cube = map_particles_to_volume_fraction(grid, pts, radii=np.full(60, 0.5 * h),
                                            scheme="cube")
    assert cube.mass_residual <= 1e-12

    mesh = Q1Mesh.structured([np.linspace(0.0, 1.0, 4)] * 3)
    grid3 = CartGrid((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (10, 10, 10))
    centres = grid3.centers()
    field = (1.0 + 0.2 * np.sin(2 * np.pi * centres[:, 0]))[:, None]
    nmap = map_cell_field_to_nodes(field, grid3, mesh)
    assert nmap.coverage == pytest.approx(1.0, rel=1e-12)
    assert nmap.zeroth_moment_residual <= 1e-12
    assert nmap.first_moment_residual <= 1e-12
    assert nmap.nodal_volumes.sum() == pytest.approx(grid3.volume, rel=1e-12)
