"""Tests for grid-to-grid conservative remapping (GAP-CROSS-MAP).

Tests cover:
- Geometry: mesh validation, overlap coverage
- Conservation: global integral preservation < 1% (hard gate)
- Monotonicity: no new extrema with bounded mode
- Symmetry: A->B->A roundtrip integral preservation
- Convergence: observed order of accuracy
- Vector (momentum) remapping
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.grid_to_grid_remap.overlap import (
    TriMesh,
    build_overlap_table,
    clip_triangle_by_triangle,
    polygon_area,
)
from tensorphysics.grid_to_grid_remap.remap import (
    RemapResult,
    conservation_report,
    l2_error,
    monotonicity_report,
    remap_cell_average,
    remap_p1_nodal,
    remap_vector,
    symmetry_roundtrip,
)
from tensorphysics.grid_to_grid_remap.acceptance import (
    case_conservation,
    case_convergence,
    case_symmetry,
    make_perturbed_tri_mesh,
    make_uniform_tri_mesh,
    run_acceptance,
)


# ---------------------------------------------------------------------------
# geometry tests
# ---------------------------------------------------------------------------
def test_uniform_mesh_area():
    mesh = make_uniform_tri_mesh(10, 10)
    q = mesh.validate(expected_area=1.0)
    assert q["ok"]
    assert abs(q["total_area"] - 1.0) < 1e-12


def test_perturbed_mesh_area():
    mesh = make_perturbed_tri_mesh(10, jitter=0.25, seed=7)
    q = mesh.validate(expected_area=1.0)
    assert q["ok"]
    assert abs(q["total_area"] - 1.0) < 1e-10


def test_triangle_triangle_clip_overlap():
    """Two overlapping triangles should produce a polygon with positive area."""
    tri_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    tri_b = np.array([[0.3, 0.3], [1.3, 0.3], [0.8, 1.3]])
    poly = clip_triangle_by_triangle(tri_a, tri_b)
    assert len(poly) >= 3
    assert polygon_area(poly) > 0.0


def test_triangle_triangle_clip_no_overlap():
    """Two disjoint triangles should produce an empty polygon."""
    tri_a = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]])
    tri_b = np.array([[1.0, 1.0], [1.5, 1.0], [1.25, 1.5]])
    poly = clip_triangle_by_triangle(tri_a, tri_b)
    assert len(poly) < 3 or polygon_area(poly) < 1e-14


def test_overlap_coverage():
    """Overlap table should cover the full domain when meshes cover the same area."""
    src = make_perturbed_tri_mesh(16, jitter=0.25, seed=7)
    tgt = make_perturbed_tri_mesh(24, jitter=0.20, seed=107)
    table = build_overlap_table(src, tgt)
    assert abs(table.total_overlap - 1.0) < 1e-10
    assert table.coverage_fraction().min() > 0.99


# ---------------------------------------------------------------------------
# conservation tests
# ---------------------------------------------------------------------------
def test_conservation_smooth_field():
    src = make_perturbed_tri_mesh(24, jitter=0.25, seed=7)
    tgt = make_perturbed_tri_mesh(36, jitter=0.20, seed=107)
    table = build_overlap_table(src, tgt)
    phi = np.sin(np.pi * src.centroids()[:, 0]) * np.sin(np.pi * src.centroids()[:, 1])
    res = remap_cell_average(table, phi)
    cr = conservation_report(res)
    assert cr["rel_defect"] < 1e-12


def test_conservation_hot_spot():
    src = make_perturbed_tri_mesh(32, jitter=0.25, seed=11)
    tgt = make_perturbed_tri_mesh(48, jitter=0.20, seed=111)
    table = build_overlap_table(src, tgt)
    cen = src.centroids()
    phi = np.exp(-((cen[:, 0] - 0.5) ** 2 + (cen[:, 1] - 0.5) ** 2) / 0.03 ** 2)
    res = remap_cell_average(table, phi)
    cr = conservation_report(res)
    assert cr["rel_defect"] < 1e-12


def test_conservation_thermal_front():
    src = make_perturbed_tri_mesh(32, jitter=0.25, seed=13)
    tgt = make_perturbed_tri_mesh(48, jitter=0.20, seed=113)
    table = build_overlap_table(src, tgt)
    cen = src.centroids()
    phi = 0.5 * (1.0 + np.tanh((cen[:, 1] - 0.5) / 0.01))
    res = remap_cell_average(table, phi)
    cr = conservation_report(res)
    assert cr["rel_defect"] < 1e-12


def test_bounded_mode_conservation():
    src = make_perturbed_tri_mesh(24, jitter=0.25, seed=7)
    tgt = make_perturbed_tri_mesh(36, jitter=0.20, seed=107)
    table = build_overlap_table(src, tgt)
    phi = np.sin(np.pi * src.centroids()[:, 0]) * np.sin(np.pi * src.centroids()[:, 1])
    res = remap_p1_nodal(table, phi, mode="bounded")
    cr = conservation_report(res)
    assert cr["rel_defect"] < 1e-10


# ---------------------------------------------------------------------------
# monotonicity tests
# ---------------------------------------------------------------------------
def test_no_new_extrema_bounded():
    src = make_perturbed_tri_mesh(32, jitter=0.25, seed=13)
    tgt = make_perturbed_tri_mesh(16, jitter=0.20, seed=113)  # coarser target
    table = build_overlap_table(src, tgt)
    cen = src.centroids()
    phi = np.exp(-((cen[:, 0] - 0.5) ** 2 + (cen[:, 1] - 0.5) ** 2) / 0.02 ** 2)
    res = remap_p1_nodal(table, phi, mode="bounded")
    mr = monotonicity_report(table, phi, res)
    assert mr["n_cells_above_max"] == 0
    assert mr["n_cells_below_min"] == 0


# ---------------------------------------------------------------------------
# symmetry tests
# ---------------------------------------------------------------------------
def test_symmetry_roundtrip_integral():
    mesh_a = make_perturbed_tri_mesh(20, jitter=0.25, seed=11)
    mesh_b = make_perturbed_tri_mesh(30, jitter=0.20, seed=211)
    table_ab = build_overlap_table(mesh_a, mesh_b)
    table_ba = build_overlap_table(mesh_b, mesh_a)
    phi_a = np.sin(2.0 * np.pi * mesh_a.centroids()[:, 0]) * np.cos(2.0 * np.pi * mesh_a.centroids()[:, 1])
    rt = symmetry_roundtrip(table_ab, table_ba, phi_a)
    assert rt["integral_rel_error"] < 1e-10


# ---------------------------------------------------------------------------
# convergence tests
# ---------------------------------------------------------------------------
def test_convergence_order():
    conv = case_convergence(levels=(8, 16, 24, 32), n_tgt=64)
    # Average mode should be ~1st order
    avg_order = conv["observed_order"]["avg_l2_abs"]
    assert avg_order is not None and avg_order > 0.5, f"avg order={avg_order}"
    # Bounded mode should be between 1st and 2nd order
    bnd_order = conv["observed_order"]["bnd_l2_abs"]
    assert bnd_order is not None and bnd_order > 0.8, f"bnd order={bnd_order}"


# ---------------------------------------------------------------------------
# vector (momentum) tests
# ---------------------------------------------------------------------------
def test_vector_remap_conservation():
    src = make_perturbed_tri_mesh(20, jitter=0.25, seed=7)
    tgt = make_perturbed_tri_mesh(30, jitter=0.20, seed=107)
    table = build_overlap_table(src, tgt)
    # 2D momentum field
    cen = src.centroids()
    vec = np.column_stack([
        np.sin(np.pi * cen[:, 0]),
        np.cos(np.pi * cen[:, 1]),
    ])
    res = remap_vector(table, vec)
    cr = conservation_report(res)
    assert cr["rel_defect"] < 1e-12


# ---------------------------------------------------------------------------
# acceptance suite
# ---------------------------------------------------------------------------
def test_acceptance_all_pass():
    payload = run_acceptance(quick=False)
    failed = [c["id"] for c in payload["criteria"] if not c["pass"]]
    assert not failed, f"failed criteria: {failed}"
    assert payload["all_pass"]
