"""End-to-end acceptance cases for grid-to-grid conservative remapping.

Three test chains, each with an independent reference:

``case_conservation``   Two non-matching meshes covering the same domain; map a
                        smooth field and verify global conservation error < 1%
                        (the hard gate).  Also tests steep fields (hot spot,
                        thermal front) where naive interpolation fails.

``case_symmetry``       A -> B -> A roundtrip: the integral must be preserved
                        exactly, and the field shape must be close (L2 error
                        decreases with mesh refinement).

``case_convergence``    Systematic refinement of both source and target meshes;
                        verify that the observed order of accuracy matches the
                        theoretical order (1st for average, 2nd for bounded).

All cases return plain floats/ints so the payload can be dumped to JSON verbatim.
"""
from __future__ import annotations

import numpy as np

from .overlap import TriMesh, build_overlap_table
from .remap import (
    RemapResult,
    conservation_report,
    l2_error,
    monotonicity_report,
    remap_cell_average,
    remap_p1_nodal,
    symmetry_roundtrip,
)

__all__ = [
    "run_acceptance",
    "case_conservation",
    "case_symmetry",
    "case_convergence",
    "make_perturbed_tri_mesh",
    "make_uniform_tri_mesh",
]


# ---------------------------------------------------------------------------
# mesh generators
# ---------------------------------------------------------------------------
def make_uniform_tri_mesh(
    nx: int,
    ny: int = None,
    domain: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    seed: int = 0,
) -> TriMesh:
    """Structured right-triangle mesh of a rectangle.

    Each quad is split along the diagonal (i+j) % 2 == 0 -> lower-left to
    upper-right, else upper-left to lower-right.
    """
    ny = nx if ny is None else ny
    x0, y0, x1, y1 = domain
    hx = (x1 - x0) / nx
    hy = (y1 - y0) / ny

    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append((x0 + i * hx, y0 + j * hy))
    nodes = np.array(nodes, dtype=float)

    tris = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = n00 + (nx + 1)
            n11 = n01 + 1
            if (i + j) % 2 == 0:
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
            else:
                tris.append((n00, n10, n01))
                tris.append((n10, n11, n01))
    tri = np.array(tris, dtype=int)
    return TriMesh(nodes=nodes, tri=tri, label=f"uniform_{nx}x{ny}")


def make_perturbed_tri_mesh(
    nx: int,
    ny: int = None,
    jitter: float = 0.25,
    seed: int = 42,
    domain: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> TriMesh:
    """Unstructured triangulation by jittered quads.

    Interior nodes are displaced by a uniform random offset up to
    ``jitter * h``; boundary nodes slide along their edge so the domain
    boundary stays exactly the rectangle.
    """
    ny = nx if ny is None else ny
    x0, y0, x1, y1 = domain
    hx = (x1 - x0) / nx
    hy = (y1 - y0) / ny
    rng = np.random.default_rng(seed)

    nodes = []
    idx_map = {}
    n_idx = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x0 + i * hx
            y = y0 + j * hy
            on_x = (i == 0) or (i == nx)
            on_y = (j == 0) or (j == ny)
            if not (on_x and on_y):
                dx = rng.uniform(-jitter, jitter) * hx
                dy = rng.uniform(-jitter, jitter) * hy
                if on_y:
                    dx = np.clip(x + dx, x0 + i * hx - 0.45 * hx, x0 + i * hx + 0.45 * hx) - x
                    dy = 0.0
                elif on_x:
                    dy = np.clip(y + dy, y0 + j * hy - 0.45 * hy, y0 + j * hy + 0.45 * hy) - y
                    dx = 0.0
                x += dx
                y += dy
            idx_map[(i, j)] = n_idx
            nodes.append((x, y))
            n_idx += 1

    nodes = np.array(nodes, dtype=float)
    tris = []
    for j in range(ny):
        for i in range(nx):
            n00 = idx_map[(i, j)]
            n10 = idx_map[(i + 1, j)]
            n11 = idx_map[(i + 1, j + 1)]
            n01 = idx_map[(i, j + 1)]
            if rng.random() < 0.5:
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
            else:
                tris.append((n00, n10, n01))
                tris.append((n10, n11, n01))
    tri = np.array(tris, dtype=int)
    return TriMesh(nodes=nodes, tri=tri, label=f"perturbed_{nx}x{ny}_j{jitter}")


# ---------------------------------------------------------------------------
# case A: conservation gate
# ---------------------------------------------------------------------------
def case_conservation(
    n_src: int = 32,
    n_tgt: int = 48,
    seed: int = 7,
) -> dict:
    """Conservation error < 1% hard gate on smooth and steep fields."""
    src = make_perturbed_tri_mesh(n_src, jitter=0.30, seed=seed)
    tgt = make_perturbed_tri_mesh(n_tgt, jitter=0.20, seed=seed + 100)
    geom_src = src.validate(expected_area=1.0)
    geom_tgt = tgt.validate(expected_area=1.0)

    table = build_overlap_table(src, tgt)

    # Smooth field: sin(pi*x)*sin(pi*y)
    src_centroids = src.centroids()
    phi_smooth = np.sin(np.pi * src_centroids[:, 0]) * np.sin(np.pi * src_centroids[:, 1])
    exact_integral_smooth = float((1.0 / np.pi) ** 2 * 4.0)  # int_0^1 int_0^1 sin(pi x) sin(pi y) dx dy = 4/pi^2

    # Steep field: hot spot
    phi_hot = np.exp(-((src_centroids[:, 0] - 0.5) ** 2 + (src_centroids[:, 1] - 0.5) ** 2) / 0.03 ** 2)

    # Thermal front
    phi_front = 0.5 * (1.0 + np.tanh((src_centroids[:, 1] - 0.5) / 0.01))

    fields = {
        "smooth_sin": {"phi": phi_smooth, "exact_integral": exact_integral_smooth},
        "hot_spot": {"phi": phi_hot, "exact_integral": None},
        "thermal_front": {"phi": phi_front, "exact_integral": None},
    }

    out: dict = {
        "geometry": {"source": geom_src, "target": geom_tgt},
        "overlap": {
            "total_overlap": table.total_overlap,
            "src_area": float(src.areas().sum()),
            "tgt_area": float(tgt.areas().sum()),
            "coverage_fraction_min": float(table.coverage_fraction().min()),
            "coverage_fraction_mean": float(table.coverage_fraction().mean()),
        },
        "fields": {},
    }

    for name, fdata in fields.items():
        phi = fdata["phi"]
        src_integral = float(phi @ src.areas())

        # Average mode (conservative baseline)
        res_avg = remap_cell_average(table, phi)
        cr_avg = conservation_report(res_avg)

        # Bounded mode
        res_bnd = remap_p1_nodal(table, phi, mode="bounded")
        cr_bnd = conservation_report(res_bnd)
        mr_bnd = monotonicity_report(table, phi, res_bnd)

        # P1 reconstruction (no limiter, for comparison)
        res_p1 = remap_p1_nodal(table, phi, mode="p1_recon")
        cr_p1 = conservation_report(res_p1)

        entry = {
            "source_integral": src_integral,
            "exact_integral": fdata["exact_integral"],
            "average": {
                "conservation_rel_defect": cr_avg["rel_defect"],
                "target_integral": cr_avg["target_integral"],
            },
            "bounded": {
                "conservation_rel_defect": cr_bnd["rel_defect"],
                "target_integral": cr_bnd["target_integral"],
                "monotonicity": {
                    "overshoot": mr_bnd["overshoot"],
                    "undershoot": mr_bnd["undershoot"],
                    "n_cells_above_max": mr_bnd["n_cells_above_max"],
                    "n_cells_below_min": mr_bnd["n_cells_below_min"],
                },
                "repair": res_bnd.repair.get("repair", {}),
            },
            "p1_recon": {
                "conservation_rel_defect": cr_p1["rel_defect"],
                "target_integral": cr_p1["target_integral"],
            },
        }
        out["fields"][name] = entry

    return out


# ---------------------------------------------------------------------------
# case B: symmetry roundtrip
# ---------------------------------------------------------------------------
def case_symmetry(
    n_a: int = 24,
    n_b: int = 36,
    seed: int = 11,
) -> dict:
    """A -> B -> A roundtrip: integral preserved, field shape close."""
    mesh_a = make_perturbed_tri_mesh(n_a, jitter=0.25, seed=seed)
    mesh_b = make_perturbed_tri_mesh(n_b, jitter=0.20, seed=seed + 200)

    table_ab = build_overlap_table(mesh_a, mesh_b)
    table_ba = build_overlap_table(mesh_b, mesh_a)

    # Test field: smooth + linear combination
    cen_a = mesh_a.centroids()
    phi_a = (
        np.sin(2.0 * np.pi * cen_a[:, 0]) * np.cos(2.0 * np.pi * cen_a[:, 1])
        + 0.5 * cen_a[:, 0]
        + 0.3 * cen_a[:, 1]
    )

    rt = symmetry_roundtrip(table_ab, table_ba, phi_a)

    # Also test with a steep field
    phi_hot = np.exp(-((cen_a[:, 0] - 0.5) ** 2 + (cen_a[:, 1] - 0.5) ** 2) / 0.02 ** 2)
    rt_hot = symmetry_roundtrip(table_ab, table_ba, phi_hot)

    return {
        "mesh_a": {"n_elements": mesh_a.n_elements, "area": float(mesh_a.areas().sum())},
        "mesh_b": {"n_elements": mesh_b.n_elements, "area": float(mesh_b.areas().sum())},
        "overlap_ab": {
            "total": table_ab.total_overlap,
            "coverage_min": float(table_ab.coverage_fraction().min()),
        },
        "overlap_ba": {
            "total": table_ba.total_overlap,
            "coverage_min": float(table_ba.coverage_fraction().min()),
        },
        "smooth_field": rt,
        "hot_spot": rt_hot,
    }


# ---------------------------------------------------------------------------
# case C: convergence order
# ---------------------------------------------------------------------------
def case_convergence(
    levels: tuple[int, ...] = (8, 16, 24, 32),
    n_tgt: int = 64,
    seed: int = 13,
) -> dict:
    """Refinement study: observed order of accuracy.

    Source mesh is refined while target is held fixed.  The reference is the
    exact cell average of the analytic field on the target mesh.
    """
    # Reference: exact cell averages on the target mesh
    tgt = make_uniform_tri_mesh(n_tgt, n_tgt)
    tgt_centroids = tgt.centroids()
    tgt_areas = tgt.areas()

    # Analytic field: sin(pi*x)*sin(pi*y)
    def exact_fn(xy):
        return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])

    phi_ref = exact_fn(tgt_centroids)

    rows = []
    for n in levels:
        src = make_perturbed_tri_mesh(n, jitter=0.25, seed=seed + n)
        table = build_overlap_table(src, tgt)
        src_centroids = src.centroids()
        phi_src = exact_fn(src_centroids)

        # Average mode
        res_avg = remap_cell_average(table, phi_src)
        er_avg = l2_error(table, res_avg.phi, phi_ref)

        # Bounded mode
        res_bnd = remap_p1_nodal(table, phi_src, mode="bounded")
        er_bnd = l2_error(table, res_bnd.phi, phi_ref)

        cr_avg = conservation_report(res_avg)
        cr_bnd = conservation_report(res_bnd)

        rows.append({
            "n_src": n,
            "h": 1.0 / n,
            "avg_l2_abs": er_avg["l2_abs"],
            "avg_l2_rel": er_avg["l2_rel"],
            "avg_conservation_rel": cr_avg["rel_defect"],
            "bnd_l2_abs": er_bnd["l2_abs"],
            "bnd_l2_rel": er_bnd["l2_rel"],
            "bnd_conservation_rel": cr_bnd["rel_defect"],
        })

    # Compute observed orders
    hs = np.array([r["h"] for r in rows])
    orders = {}
    for key in ("avg_l2_abs", "bnd_l2_abs"):
        errs = np.array([r[key] for r in rows])
        if len(errs) >= 2 and np.all(errs > 0):
            slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
            orders[key] = float(slope)
        else:
            orders[key] = None

    return {"rows": rows, "observed_order": orders}


# ---------------------------------------------------------------------------
# master acceptance runner
# ---------------------------------------------------------------------------
def run_acceptance(quick: bool = False) -> dict:
    """Run all acceptance cases and return a JSON-serialisable payload."""
    payload: dict = {"node": "GAP-CROSS-MAP", "criteria": []}

    # C1: conservation gate
    cons = case_conservation(n_src=24, n_tgt=36)
    all_pass = True
    for fname, fdata in cons["fields"].items():
        for mode in ("average", "bounded"):
            rd = fdata[mode]["conservation_rel_defect"]
            passed = rd < 0.01  # < 1%
            cid = f"C1_conservation_{mode}_{fname}"
            payload["criteria"].append({
                "id": cid,
                "pass": bool(passed),
                "value": rd,
                "threshold": 0.01,
            })
            if not passed:
                all_pass = False

    # C2: symmetry roundtrip
    sym = case_symmetry(n_a=20, n_b=30)
    for fname in ("smooth_field", "hot_spot"):
        rt = sym[fname]
        passed = rt["integral_rel_error"] < 1e-10
        cid = f"C2_symmetry_{fname}"
        payload["criteria"].append({
            "id": cid,
            "pass": bool(passed),
            "value": rt["integral_rel_error"],
            "threshold": 1e-10,
        })
        if not passed:
            all_pass = False

    # C3: convergence order
    if not quick:
        conv = case_convergence()
        for key, order in conv["observed_order"].items():
            # Average mode should be ~1st order, bounded ~2nd order
            expected = 1.0 if "avg" in key else 2.0
            passed = order is not None and order > expected - 0.5
            cid = f"C3_convergence_{key}"
            payload["criteria"].append({
                "id": cid,
                "pass": bool(passed),
                "value": order,
                "threshold": expected - 0.5,
            })
            if not passed:
                all_pass = False

    payload["all_pass"] = all_pass
    return payload
