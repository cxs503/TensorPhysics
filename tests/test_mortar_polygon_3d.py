"""Acceptance tests for the 2D/3D mortar operators (IN50-N08, criteria C2-C6).

The 1D contract is tested in ``tests/test_interface_operators.py``; this module
covers the polygon-clipping and surface-patch lifts:

* clipping correctness (two independent algorithms + analytic overlaps),
* constant/linear patch tests and observed order on 2D and 3D partitions,
* local conservation measure correction (including the positivity-clip repair),
* surface measure accuracy of the 3D patch mapping.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.interface_operators.core import conservation_residual
from tensorphysics.interface_operators.mortar_nd import (
    FIELD_2D,
    FIELD_3D,
    MortarProjection2D,
    SurfaceMortarProjection,
    analytic_surface_area,
    bulged_surface,
    faceted_surface_area,
    local_conservation_fix,
    overlap_measure_closure,
)
from tensorphysics.interface_operators.polygons import (
    axisymmetric_mesh,
    cartesian_mesh,
    clip_convex,
    convex_hull,
    convex_intersection,
    fan_mesh,
    polygon_area,
    triangulated_mesh,
)


@pytest.fixture(scope="module")
def surface():
    return bulged_surface()


@pytest.fixture(scope="module")
def mesh2d():
    return cartesian_mesh(16, 16), triangulated_mesh(12, 12)


@pytest.fixture(scope="module")
def mesh3d(surface):
    r, dr = surface
    return axisymmetric_mesh(16, 12, r, dr), axisymmetric_mesh(12, 9, r, dr)


# -- clipping --------------------------------------------------------------


def test_two_independent_clipping_algorithms_agree():
    rng = np.random.default_rng(20261106)
    worst = 0.0
    checked = 0
    for _ in range(120):
        a = convex_hull(rng.uniform(0.0, 1.0, (8, 2)))
        b = convex_hull(rng.uniform(0.0, 1.0, (9, 2)))
        if a.shape[0] < 3 or b.shape[0] < 3:
            continue
        area_sh = polygon_area(clip_convex(a, b))
        area_hull = polygon_area(convex_intersection(a, b))
        if max(area_sh, area_hull) > 1e-6:
            checked += 1
            worst = max(worst, abs(area_sh - area_hull) / max(area_sh, area_hull))
    assert checked > 100
    assert worst < 1e-12


def test_clipping_matches_analytic_rectangle_overlap():
    quad = np.array([[0.2, 0.2], [0.7, 0.2], [0.7, 0.6], [0.2, 0.6]])
    win = np.array([[0.5, 0.1], [1.0, 0.1], [1.0, 0.4], [0.5, 0.4]])
    exact = (0.7 - 0.5) * (0.4 - 0.2)          # analytic intersection
    assert abs(polygon_area(clip_convex(quad, win)) - exact) / exact < 1e-14


def test_partition_closure_for_three_partition_families():
    cart, tri, fan = cartesian_mesh(8, 8), triangulated_mesh(8, 8), fan_mesh(10, 4)
    for src, tgt in ((cart, tri), (tri, fan), (cart, fan)):
        assert overlap_measure_closure(src, tgt) < 1e-12
        assert abs(float(np.sum(src.measures)) - 1.0) < 1e-12
        assert abs(float(np.sum(tgt.measures)) - 1.0) < 1e-12


# -- C2: constant field ----------------------------------------------------


def test_c2_constant_field_conserved_2d(mesh2d):
    src, tgt = mesh2d
    op = MortarProjection2D()
    field = FIELD_2D[0]
    u = src.cell_averages(field.func)
    v = op.transfer(src, u, tgt)
    assert np.max(np.abs(v - 3.7)) < 1e-12
    assert op.conservation_residual(src, u, tgt, v) < 1e-14


def test_c2_constant_field_conserved_3d(mesh3d):
    src, tgt = mesh3d
    op = SurfaceMortarProjection()
    field = FIELD_3D[0]
    u = src.cell_averages(field.func)
    v = op.transfer(src, u, tgt)
    assert np.max(np.abs(v - 2.5)) < 1e-11
    assert op.conservation_residual(src, u, tgt, v) < 1e-14


# -- C3: consistency and observed order ------------------------------------


def test_linear_field_reproduced_2d(mesh2d):
    src, tgt = mesh2d
    op = MortarProjection2D()
    field = FIELD_2D[1]
    v = op.transfer(src, src.cell_averages(field.func), tgt)
    assert op.l2_error(tgt, v, field) < 1e-12


def test_linear_field_observed_order_2d():
    op = MortarProjection2D()
    pairs = [(cartesian_mesh(n, n), triangulated_mesh(n, n)) for n in (8, 16, 32)]
    est = op.observed_order(FIELD_2D[3], pairs)
    assert est.order is not None and est.order >= 1.9


def test_first_order_operator_measures_first_order():
    op = MortarProjection2D(order=1)
    pairs = [(cartesian_mesh(n, n), triangulated_mesh(n, n)) for n in (8, 16, 32)]
    est = op.observed_order(FIELD_2D[3], pairs)
    assert est.order is not None and 0.8 < est.order < 1.3


def test_linear_field_observed_order_3d(surface):
    r, dr = surface
    op = SurfaceMortarProjection()

    def pair(n, m):
        kw = dict(theta0=0.3, theta_span=2.6)
        return (axisymmetric_mesh(n, n, r, dr, **kw), axisymmetric_mesh(m, m, r, dr, **kw))

    est = op.observed_order(FIELD_3D[1], [pair(6, 4), pair(12, 8), pair(24, 16)])
    assert est.order is None or est.order >= 1.9
    assert est.errors[-1] < 1e-3


def test_smooth_field_observed_order_3d(surface):
    r, dr = surface
    op = SurfaceMortarProjection()

    def pair(n, m):
        kw = dict(theta0=0.3, theta_span=2.6)
        return (axisymmetric_mesh(n, n, r, dr, **kw), axisymmetric_mesh(m, m, r, dr, **kw))

    est = op.observed_order(FIELD_3D[2], [pair(6, 4), pair(12, 8), pair(24, 16)])
    assert est.order is not None and est.order >= 1.9


# -- C4: local conservation measure correction -----------------------------


def test_c4_positivity_clip_is_detected_then_repaired_2d(mesh2d):
    src, tgt = mesh2d
    op = MortarProjection2D()
    field = FIELD_2D[3]
    u = src.cell_averages(field.func)
    a_t = op.effective_target_measure(src, tgt)
    clipped = np.maximum(op.transfer(src, u, tgt), 0.0)
    broken = conservation_residual(u, src.measures, clipped, a_t)
    assert broken > 1e-2, "the negative control must lose mass"
    fixed = local_conservation_fix(clipped, op.overlap_budget(src, u, tgt), a_t)
    assert conservation_residual(u, src.measures, fixed, a_t) < 1e-12
    # the repair lands exactly on the conservative (area-weighted) transfer; for a
    # sign-changing field a target cell mean may legitimately be negative
    np.testing.assert_allclose(fixed, op.corrected_transfer(src, u, tgt), atol=1e-14)


def test_c4_positivity_clip_is_detected_then_repaired_3d(mesh3d):
    src, tgt = mesh3d
    op = SurfaceMortarProjection()
    field = FIELD_3D[3]
    u = src.cell_averages(field.func)
    a_t = op.effective_target_measure(src, tgt)
    clipped = np.maximum(op.transfer(src, u, tgt), 0.0)
    broken = conservation_residual(u, src.measures, clipped, a_t)
    assert broken > 1e-2
    fixed = local_conservation_fix(clipped, op.overlap_budget(src, u, tgt), a_t)
    assert conservation_residual(u, src.measures, fixed, a_t) < 1e-12


def test_measure_correction_is_a_roundoff_level_repair(mesh2d, mesh3d):
    op2 = MortarProjection2D()
    op3 = SurfaceMortarProjection()
    assert op2.measure_correction(*mesh2d) < 1e-12
    assert op3.measure_correction(*mesh3d) < 1e-6


def test_point_sampling_operator_is_non_conservative(mesh2d):
    """Donor-cell point sampling loses mass on a steep, non-symmetric field.

    On a *symmetric* field the sampling errors cancel by symmetry (a harmonic
    field gives ~1e-16), so the negative control has to use the boundary layer:
    that asymmetry is exactly what a coupled run would see at a wall.
    """
    src, tgt = mesh2d
    op = MortarProjection2D(order=1, broken="centroid")
    field = FIELD_2D[4]
    u = src.cell_averages(field.func)
    assert op.conservation_residual(src, u, tgt) > 1e-2


def test_bounding_box_overlap_is_non_conservative():
    src, tgt = cartesian_mesh(10, 10), fan_mesh(12, 5)
    op = MortarProjection2D(broken="bbox")
    W, _M, _a_s, _a_t = op._assemble(src, tgt)
    error = abs(float(np.sum(W)) - float(np.sum(src.measures))) / float(np.sum(src.measures))
    assert error > 1e-3


# -- C5/C6: geometry accuracy of the mapping -------------------------------


def test_c6_surface_patch_measure_matches_analytic_area(surface):
    r, dr = surface
    analytic = analytic_surface_area(r, dr)
    mesh = axisymmetric_mesh(16, 12, r, dr)
    assert abs(float(np.sum(mesh.areas)) - analytic) / analytic < 1e-10


def test_c6_faceted_area_error_within_gate(surface):
    r, dr = surface
    analytic = analytic_surface_area(r, dr)
    mesh = axisymmetric_mesh(32, 24, r, dr)
    error = abs(faceted_surface_area(mesh) - analytic) / analytic
    assert error < 5e-3


def test_faceted_area_error_is_second_order(surface):
    r, dr = surface
    analytic = analytic_surface_area(r, dr)
    errors = []
    for n in (8, 16, 32):
        mesh = axisymmetric_mesh(n, int(0.75 * n), r, dr)
        errors.append(abs(faceted_surface_area(mesh) - analytic) / analytic)
    slopes = [np.log(errors[k] / errors[k + 1]) / np.log(2.0) for k in range(len(errors) - 1)]
    assert min(slopes) > 1.5


def test_surface_patch_centroids_lie_on_the_surface(surface):
    r, dr = surface
    mesh = axisymmetric_mesh(8, 6, r, dr)
    centers = mesh.centroids
    radial = np.asarray(r(centers[:, 1]))
    # the chart centroid is not the surface centroid, but the Jacobian is
    # evaluated there: the radius it reports must bracket the patch radii
    patch_radii = []
    for poly in mesh.patches:
        patch_radii.append(np.asarray(r(poly[:, 1])).min())
    assert np.all(radial >= min(patch_radii) - 1e-12)