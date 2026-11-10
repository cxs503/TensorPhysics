"""Tests for the conservative cross-solver field exchange (LBM/DEM -> FEM).

The tests are organised as: geometry/generic properties first, then the two
exchange operators (with a negative control), then each FEM target against its
own closed-form anchor, then the end-to-end acceptance payload structure.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.field_exchange import (
    LITERATURE_PLATE_COEF,
    LatticeScale,
    PlaneStressQ4,
    PlateBFS,
    PlateStripBending,
    QuadGrid,
    audit_contact_map,
    audit_panel_map,
    case_contact,
    clip_rect,
    hertz_force,
    map_contacts,
    map_panels,
    map_panels_to_basis,
    navier_center_deflection,
    navier_coefficients,
    nearest_node_lumping,
    polygon_area,
    run_channel,
    strip_sine_pressure_center,
    strip_uniform_pressure_center,
)


# ---------------------------------------------------------------------------
# generic properties
# ---------------------------------------------------------------------------
def test_clip_rect_area_intersection():
    rect = (0.0, 1.0, 0.0, 1.0)
    poly = [(-0.5, 0.2), (0.5, 0.2), (0.5, 0.8), (-0.5, 0.8)]
    out = clip_rect(poly, rect)
    assert polygon_area(out) == pytest.approx(0.5 * 0.6, rel=1e-12)


def test_bilinear_partition_of_unity_and_linear_reproduction():
    g = QuadGrid(0.0, 2.0, 0.0, 1.0, 4, 2)
    xy = g.coords()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.uniform(0, 2), rng.uniform(0, 1)
        ci, cj = g.locate(x, y)
        w = g.bilinear(ci, cj, x, y)
        nodes = g.cell_nodes(ci, cj)
        assert w.sum() == pytest.approx(1.0, abs=1e-14)
        assert float(w @ xy[nodes, 0]) == pytest.approx(x, abs=1e-13)
        assert float(w @ xy[nodes, 1]) == pytest.approx(y, abs=1e-13)


# ---------------------------------------------------------------------------
# operator 1: panels -> nodes
# ---------------------------------------------------------------------------
def _panel_problem(nx=5, ny=3, seed=1):
    grid = QuadGrid(0.0, 1.0, 0.0, 0.6, 7, 5)
    xe = np.linspace(0.0, 1.0, nx + 1)
    ye = np.linspace(0.0, 0.6, ny + 1)
    rng = np.random.default_rng(seed)
    tr = rng.normal(size=(nx, ny, 3))
    return grid, xe, ye, tr


def test_panel_map_conserves_force_and_moment():
    grid, xe, ye, tr = _panel_problem()
    f = map_panels(grid, xe, ye, tr)
    audit = audit_panel_map(grid, f, xe, ye, tr)
    assert audit["force_residual_rel"] < 1e-12
    assert audit["moment_residual_rel"] < 1e-12


def test_panel_map_uniform_traction_equals_consistent_load():
    grid = QuadGrid(0.0, 1.0, 0.0, 1.0, 4, 4)
    xe = np.array([0.0, 0.5, 1.0])
    ye = np.array([0.0, 1.0])
    tr = np.zeros((2, 1, 3))
    tr[:, 0, 2] = -3.0
    f = map_panels(grid, xe, ye, tr)
    # a fully interior node receives value * hx * hy (4 elements x area/4 each)
    expect = tr[0, 0, 2] * 0.25 * 0.25
    assert f[grid.node_index(1, 1), 2] == pytest.approx(expect, rel=1e-12)
    assert f[:, 2].sum() == pytest.approx(-3.0 * 1.0 * 1.0, rel=1e-12)


def test_panel_map_rejects_incomplete_panel_coverage():
    grid = QuadGrid(0.0, 1.0, 0.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        map_panels(grid, np.array([0.0, 0.5]), np.array([0.0, 1.0]), np.zeros((1, 1, 3)))


def test_nearest_node_lumping_is_detected_by_the_moment_audit():
    grid = QuadGrid(0.0, 1.0, 0.0, 0.2, 6, 2)
    pts = np.array([[0.11, 0.05], [0.62, 0.13]])
    frc = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
    f_naive = nearest_node_lumping(grid, pts, frc)
    audit = audit_contact_map(grid, f_naive, pts, frc)
    assert audit["force_residual_rel"] < 1e-14          # force is conserved
    assert audit["moment_residual_rel"] > 1e-2          # but the moment is not


# ---------------------------------------------------------------------------
# operator 2: contacts -> nodes
# ---------------------------------------------------------------------------
def test_contact_map_conserves_force_and_moment_for_pure_forces():
    """On a planar interface the partition of unity conserves the moment exactly.

    Proof: ``sum_i N_i (r_i x F) = (sum_i N_i r_i) x F = r x F`` because the
    bilinear weights reproduce the coordinates.
    """
    grid = QuadGrid(0.0, 1.0, 0.0, 0.2, 8, 2)
    pts = np.array([[0.113, 0.041], [0.617, 0.133]])
    frc = np.array([[0.2, 0.0, -1.0], [-0.1, 0.05, -2.0]])
    f, _ = map_contacts(grid, pts, frc)
    audit = audit_contact_map(grid, f, pts, frc)
    assert audit["force_residual_rel"] < 1e-14
    assert audit["moment_residual_rel"] < 1e-14


def test_contact_couple_needs_moment_completion():
    """A contact couple cannot be carried by forces alone; completion restores it."""
    grid = QuadGrid(0.0, 1.0, 0.0, 0.2, 8, 2)
    pts = np.array([[0.113, 0.041], [0.617, 0.133]])
    frc = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
    moms = np.zeros_like(frc)
    moms[1, 1] = 5.0e-3                      # rocking couple on contact 2

    raw, _ = map_contacts(grid, pts, frc, moments=moms, complete_moment=False)
    audit_raw = audit_contact_map(grid, raw, pts, frc, moments=moms)
    lost = np.linalg.norm(np.array(audit_raw["moment_target"]) - np.array(audit_raw["moment_source"]))
    assert lost / np.linalg.norm(moms.sum(axis=0)) == pytest.approx(1.0, abs=1e-6)

    fixed, _ = map_contacts(grid, pts, frc, moments=moms, complete_moment=True)
    audit = audit_contact_map(grid, fixed, pts, frc, moments=moms)
    assert audit["force_residual_rel"] < 1e-12
    assert audit["moment_residual_rel"] < 1e-12


def test_contact_map_rejects_points_outside_the_grid():
    grid = QuadGrid(0.0, 1.0, 0.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        map_contacts(grid, np.array([[1.5, 0.5]]), np.array([[0.0, 0.0, -1.0]]))


# ---------------------------------------------------------------------------
# FEM targets against their own closed-form anchors
# ---------------------------------------------------------------------------
def test_plane_stress_bar_solution():
    """nu=0 membrane with uniform axial traction: u(L) = tau L^2 / (2 E t)."""
    L, W, t, E, tau = 0.1, 0.02, 1e-3, 2.0e9, 3.0
    memb = PlaneStressQ4(L, W, t, E, 0.0, 40, 4)
    grid = QuadGrid(0.0, L, 0.0, W, 40, 4)
    hx, hy = L / 40, W / 4
    f = np.zeros((grid.n_nodes, 3))
    for e in range(40):
        for j in range(4):
            for (i0, j0) in ((e, j), (e + 1, j), (e + 1, j + 1), (e, j + 1)):
                f[grid.node_index(i0, j0), 0] += tau * hx * hy / 4.0
    u = memb.solve(f, fixed="left")
    exact = tau * L ** 2 / (2.0 * E * t)
    assert memb.tip_mean_ux(u) == pytest.approx(exact, rel=1e-10)


def test_plate_strip_uniform_and_sine_pressure():
    L, D, n = 0.1, 1.5, 80
    strip = PlateStripBending(L, D, n, "simple-simple")
    q = 25.0
    u = strip.solve(strip.consistent_load(lambda x: q))
    assert strip.mid_deflection(u) == pytest.approx(
        strip_uniform_pressure_center(q, L, D), rel=2e-3)
    p0 = 40.0
    u2 = strip.solve(strip.consistent_load(lambda x: p0 * np.sin(np.pi * x / L)))
    assert strip.mid_deflection(u2) == pytest.approx(
        strip_sine_pressure_center(p0, L, D), rel=2e-3)


def test_bfs_plate_curvature_patch_test():
    """A constant-curvature field must give exactly its analytic energy."""
    plate = PlateBFS(1.0, 1.0, 0.01, 1.0e9, 0.0, 1, 1)
    K = plate.element_stiffness()
    u = np.zeros(16)
    for c, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        u[4 * c:4 * c + 4] = [x * x / 2.0, x, 0.0, 0.0]
    assert 0.5 * u @ K @ u == pytest.approx(0.5 * plate.D, rel=1e-12)


def test_bfs_plate_uniform_load_matches_tabulated_coefficient():
    a, t, E, nu = 0.1, 0.002, 2.1e9, 0.3
    plate = PlateBFS(a, a, t, E, nu, 16, 16)
    q = 1.0
    u = plate.solve(plate.consistent_load(lambda x, y: q))
    w = plate.center_deflection(u)
    assert w == pytest.approx(LITERATURE_PLATE_COEF * q * a ** 4 / plate.D, rel=1e-2)
    # and the energy audit closes
    assert plate.strain_energy(u) == pytest.approx(
        plate.external_work(plate.consistent_load(lambda x, y: q), u), rel=1e-10)


def test_navier_single_mode_matches_closed_form():
    """p = p0 sin(pi x/a) sin(pi y/b) has the one-term exact Navier solution."""
    a = b = 0.1
    D = 1.5
    p0 = 10.0
    n = 40
    xe = np.linspace(0.0, a, n + 1)
    ye = np.linspace(0.0, b, n + 1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    P = p0 * np.sin(np.pi * xc / a)[:, None] * np.sin(np.pi * yc / b)[None, :]
    coef = navier_coefficients(a, b, xe, ye, P, 199, 199)
    w = navier_center_deflection(a, b, D, coef)
    exact = p0 / (D * np.pi ** 4 * (1 / a ** 2 + 1 / b ** 2) ** 2)
    # the panels approximate the smooth sine by 40 piecewise constants, so the
    # series recovers the single-mode solution to the panel discretisation error
    assert w == pytest.approx(exact, rel=2e-3)


def test_plate_mapping_consistent_load_beats_nodal_lumping():
    """Consistent (generalised) mapping reproduces the Navier solution to < 1%."""
    a = b = 0.05
    t, E, nu = 5e-4, 2.1e9, 0.3
    n = 16
    plate = PlateBFS(a, b, t, E, nu, n, n)
    grid = QuadGrid(0.0, a, 0.0, b, n, n)
    nx = 8
    xe = np.linspace(0.0, a, nx + 1)
    ye = np.array([0.0, b])
    p = -np.linspace(10.0, 2.0, nx)             # linear wall pressure, downward
    traction = np.zeros((nx, 1, 3))
    traction[:, 0, 2] = p

    hx, hy = a / n, b / n
    dxx = map_panels_to_basis(
        grid, xe, ye, p.reshape(nx, 1, 1),
        lambda ci, cj, x, y: plate._basis((x - ci * hx) / hx, (y - cj * hy) / hy, hx, hy)[0], 4)
    w_fem = plate.center_deflection(plate.solve(dxx))

    coef = navier_coefficients(a, b, xe, ye, p[:, None], 199, 199)
    w_ref = navier_center_deflection(a, b, plate.D, coef)
    assert abs(w_fem - w_ref) / abs(w_ref) < 1e-2

    f_nodal = map_panels(grid, xe, ye, traction)
    audit = audit_panel_map(grid, f_nodal, xe, ye, traction)
    assert audit["force_residual_rel"] < 1e-12
    assert audit["moment_residual_rel"] < 1e-12


# ---------------------------------------------------------------------------
# LBM source and end-to-end case
# ---------------------------------------------------------------------------
def test_lbm_couette_reproduces_linear_profile_and_wall_shear():
    scale = LatticeScale(rho_phys=998.0, nu_phys=1.004e-6, h_phys=1e-3, ny=16, tau_relax=0.62)
    u_lat = scale.u_lattice(0.05)
    res = run_channel(nx=4, ny=16, lid_velocity=u_lat, tau_relax=0.62,
                      max_steps=60000, tol=1e-7, check_every=500)
    assert res.profile_error_linear(u_lat) < 1e-2
    assert abs(res.tau_w_bottom - res.mu * u_lat / 16) / (res.mu * u_lat / 16) < 3e-2


def test_hertz_force_round_trip():
    P = hertz_force(2.1e9, 0.3, 70e9, 0.22, 5e-3, 2e-6)
    assert P > 0
    assert hertz_force(2.1e9, 0.3, 70e9, 0.22, 5e-3, 2 * 2e-6) == pytest.approx(
        P * 2 ** 1.5, rel=1e-12)


def test_contact_case_end_to_end_conserves_and_matches_closed_form():
    out = case_contact(n_elem=60)
    assert out["exchange"]["conserved"]["force_residual_rel"] < 1e-12
    assert out["exchange"]["conserved"]["moment_residual_rel"] < 1e-12
    assert out["fem"]["rel_error_max"] < 1e-2
    assert out["fem"]["energy_residual_rel"] < 1e-8
    assert out["exchange"]["moment_carrying"]["completed_couple_loss_rel"] < 1e-12
    assert out["exchange"]["moment_carrying"]["raw_couple_loss_rel"] > 1e-2
    assert out["fem"]["interface_work_defect_rel"] < 1e-2