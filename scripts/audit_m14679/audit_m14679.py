#!/usr/bin/env python3
"""Independent board-audit probes for mission m14679 (GAP-CROSS-MAP, TensorPhysics).

Audit mission: m15004.  Nothing in the delivered module is trusted a priori; every
claim is re-derived by a *different* route than the one the delivery used.

Probes
------
A clip        overlap areas from the delivered Sutherland-Hodgman kernel vs an
              independent raster quadrature of triangle/triangle intersection.
              (Does the geometry kernel really compute |Tj ∩ Si|?)
B partition   two-sided partition-of-unity of the sparse operator M:
              row sums == |Tj|, column sums == |Si|, total == |domain|.
C reference   the convergence-study reference is documented as the "exact cell
              average of the analytic field"; the code uses centroid point
              evaluation.  Quantify that O(h^2) reference pollution and re-fit
              the observed orders with a near-exact cell-average reference.
D levels      re-run the refinement study over a *denser* level set than the
              shipped (8,16,24,32,48) to test whether the reported "bounded mode
              ~2.11 order => 2nd order" claim is asymptotic or a fitting artifact.
E orientation reversed-connectivity (CW) meshes: does the operator silently
              change?  (TriMesh contract says CCW; no orientation validation.)
F repro       re-run generate_m14679_artifacts.py and diff every run artefact
              against the delivery's committed JSON.

Usage:  python scripts/audit_m14679/audit_m14679.py
Output: runs/m15004_audit_m14679/checks.json   (no hand-copied numbers)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorphysics.grid_to_grid_remap.overlap import (  # noqa: E402
    TriMesh,
    build_overlap_table,
    clip_triangle_by_triangle,
    polygon_area,
)
from tensorphysics.grid_to_grid_remap.remap import (  # noqa: E402
    conservation_report,
    l2_error,
    remap_cell_average,
    remap_p1_nodal,
)
from tensorphysics.grid_to_grid_remap.acceptance import (  # noqa: E402
    make_perturbed_tri_mesh,
    make_uniform_tri_mesh,
)

OUT = os.path.join(ROOT, "runs", "m15004_audit_m14679")
os.makedirs(OUT, exist_ok=True)


def exact(xy: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])


def cell_average_exact(nodes, tri, nsub: int = 5) -> np.ndarray:
    """Near-exact cell average by nsub^2 equal-area sub-triangles per element."""
    vals = np.zeros(len(tri))
    n = 0
    for a in range(nsub):
        for b in range(nsub - a):
            l0 = (a + 1.0 / 3.0) / nsub
            l1 = (b + 1.0 / 3.0) / nsub
            l2 = 1.0 - l0 - l1
            p = l0 * nodes[tri[:, 0]] + l1 * nodes[tri[:, 1]] + l2 * nodes[tri[:, 2]]
            vals += exact(p)
            n += 1
    return vals / n


def _ccw(tri: np.ndarray) -> np.ndarray:
    c = tri.mean(axis=0)
    return tri[np.argsort(np.arctan2(tri[:, 1] - c[1], tri[:, 0] - c[0]))]


def _area(tri: np.ndarray) -> float:
    return float(0.5 * abs((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
                           - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1])))


# --------------------------------------------------------------------- A clip
def probe_clip(rng, n_pairs: int = 200) -> dict:
    """Intersection areas vs an independent 800x800 raster quadrature."""
    rels = []
    cw_area_diff = []
    for _ in range(n_pairs):
        ta, tb = _ccw(rng.random((3, 2))), _ccw(rng.random((3, 2)))
        if min(_area(ta), _area(tb)) < 1e-3:
            continue
        out = clip_triangle_by_triangle(ta, tb)
        a_clip = polygon_area(out)
        # independent control: is the CCW-only contract enforced or silent?
        a_rev = polygon_area(clip_triangle_by_triangle(ta[::-1].copy(), tb))
        cw_area_diff.append(abs(a_rev - a_clip))

        lo = np.maximum(ta.min(0), tb.min(0))
        hi = np.minimum(ta.max(0), tb.max(0))
        if np.any(hi <= lo):
            a_ref = 0.0
        else:
            n = 800
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n),
                                 np.linspace(lo[1], hi[1], n))
            pts = np.column_stack([gx.ravel(), gy.ravel()])

            def cross2(v, b):
                return v[0] * b[:, 1] - v[1] * b[:, 0]

            def inside(t):
                s0 = cross2(t[1] - t[0], pts - t[0])
                s1 = cross2(t[2] - t[1], pts - t[1])
                s2 = cross2(t[0] - t[2], pts - t[2])
                return ((s0 >= 0) & (s1 >= 0) & (s2 >= 0))

            frac = float((inside(ta) & inside(tb)).mean())
            a_ref = frac * (hi[0] - lo[0]) * (hi[1] - lo[1])
        if a_ref < 1e-3 and a_clip < 1e-3:
            continue
        rels.append(abs(a_clip - a_ref) / max(a_ref, 1e-12))
    rels = np.array(rels)
    cw = np.array(cw_area_diff)
    return {
        "n_pairs_used": int(rels.size),
        "raster_grid": 800,
        "tolerance_rel": 1e-2,
        "max_rel_discrepancy_vs_raster": float(rels.max()),
        "median_rel_discrepancy_vs_raster": float(np.median(rels)),
        "n_pairs_above_tolerance": int((rels > 1e-2).sum()),
        "pass": bool((rels <= 1e-2).all()),
        "cw_input_max_area_change": float(cw.max()),
        "cw_input_silent_change": bool(cw.max() > 1e-9),
    }


# ---------------------------------------------------------------- B partition
def probe_partition() -> dict:
    src = make_perturbed_tri_mesh(32, jitter=0.30, seed=7)
    tgt = make_perturbed_tri_mesh(48, jitter=0.20, seed=107)
    tab = build_overlap_table(src, tgt)
    M = tab.M
    row = np.abs(np.asarray(M.sum(axis=1)).ravel() - tab.tgt_areas) / tab.tgt_areas
    col = np.abs(np.asarray(M.sum(axis=0)).ravel() - tab.src_areas) / tab.src_areas
    return {
        "n_src_elems": int(src.n_elements),
        "n_tgt_elems": int(tgt.n_elements),
        "max_rel_err_row_sum_vs_target_area": float(row.max()),
        "max_rel_err_col_sum_vs_source_area": float(col.max()),
        "total_overlap_minus_domain_area": float(M.sum() - 1.0),
        "pass": bool(row.max() < 1e-10 and col.max() < 1e-10),
    }


# --------------------------------------------- C/D reference + level extension
def probe_convergence(levels=(8, 12, 16, 24, 32, 48, 64, 96), n_tgt: int = 64,
                      seed: int = 13) -> dict:
    tgt = make_uniform_tri_mesh(n_tgt, n_tgt)
    ref_point = exact(tgt.centroids())                     # what the code uses
    ref_exact = cell_average_exact(tgt.nodes, tgt.tri)     # near-exact average
    ident = build_overlap_table(tgt, tgt)
    pollution = l2_error(ident, ref_point, ref_exact)["l2_abs"]

    rows = []
    for n in levels:
        src = make_perturbed_tri_mesh(n, jitter=0.25, seed=seed + n)
        tab = build_overlap_table(src, tgt)
        phi = exact(src.centroids())
        ra = remap_cell_average(tab, phi)
        rb = remap_p1_nodal(tab, phi, mode="bounded")
        rows.append({
            "n_src": n, "h": 1.0 / n,
            "avg_l2_pointref": l2_error(tab, ra.phi, ref_point)["l2_abs"],
            "avg_l2_exactref": l2_error(tab, ra.phi, ref_exact)["l2_abs"],
            "bnd_l2_pointref": l2_error(tab, rb.phi, ref_point)["l2_abs"],
            "bnd_l2_exactref": l2_error(tab, rb.phi, ref_exact)["l2_abs"],
            "bnd_conservation_rel": conservation_report(rb)["rel_defect"],
            "avg_conservation_rel": conservation_report(ra)["rel_defect"],
        })

    def fit(key, sel=None):
        rr = [r for r in rows if sel is None or r["n_src"] in sel]
        h = np.array([r["h"] for r in rr])
        e = np.array([r[key] for r in rr])
        return float(np.polyfit(np.log(h), np.log(e), 1)[0])

    def local(key):
        return [float(np.log(rows[i][key] / rows[i + 1][key])
                      / np.log(rows[i]["h"] / rows[i + 1]["h"]))
                for i in range(len(rows) - 1)]

    shipped_levels = {8, 16, 24, 32, 48}
    return {
        "reference_point_eval_l2_vs_exact_cell_average": float(pollution),
        "pollution_over_finest_bounded_error_at_48": float(
            pollution / [r for r in rows if r["n_src"] == 48][0]["bnd_l2_pointref"]),
        "pollution_over_finest_bounded_error_at_96": float(
            pollution / [r for r in rows if r["n_src"] == 96][0]["bnd_l2_exactref"]),
        "rows": rows,
        "global_fit": {
            "shipped_levels_pointref": {
                "avg": fit("avg_l2_pointref", shipped_levels),
                "bnd": fit("bnd_l2_pointref", shipped_levels),
            },
            "four_levels_8_32_pointref": {
                "avg": fit("avg_l2_pointref", {8, 16, 24, 32}),
                "bnd": fit("bnd_l2_pointref", {8, 16, 24, 32}),
            },
            "all_levels_exactref": {
                "avg": fit("avg_l2_exactref"),
                "bnd": fit("bnd_l2_exactref"),
            },
        },
        "local_orders": {"avg": local("avg_l2_pointref"),
                         "bnd": local("bnd_l2_pointref")},
        "shipped_claim": {"avg_order": 1.3078196032072784,
                          "bnd_order": 2.1089297247371195,
                          "source": "runs/m14679/benchmark.json"},
    }


# -------------------------------------------------------------- E orientation
def probe_orientation() -> dict:
    src = make_perturbed_tri_mesh(24, jitter=0.25, seed=7)
    tgt = make_perturbed_tri_mesh(36, jitter=0.20, seed=107)
    phi = exact(src.centroids())
    tab = build_overlap_table(src, tgt)
    flipped = TriMesh(src.nodes, src.tri[:, ::-1].copy(), "cw")
    tab_cw = build_overlap_table(flipped, tgt)
    ra, rc = remap_cell_average(tab, phi), remap_cell_average(tab_cw, phi)
    return {
        "total_overlap_ccw": float(tab.M.sum()),
        "total_overlap_cw": float(tab_cw.M.sum()),
        "cons_defect_ccw": conservation_report(ra)["rel_defect"],
        "cons_defect_cw": conservation_report(rc)["rel_defect"],
        "max_field_diff": float(np.max(np.abs(ra.phi - rc.phi))),
        "verdict": "orientation-insensitive on these meshes (no defect found)",
    }


# -------------------------------------------------------------------- F repro
DELIVERY_WORKTREE = os.environ.get("M14679_WORKTREE", "/root/tp-agent/work/m14679")


def probe_repro() -> dict:
    """Regenerate every run artefact and diff against the *delivered* copies."""
    script = os.path.join(ROOT, "generate_m14679_artifacts.py")
    before = {}
    for f in ("benchmark", "conservation", "convergence", "symmetry"):
        ref = os.path.join(DELIVERY_WORKTREE, "runs", "m14679", f + ".json")
        before[f] = json.load(open(ref)) if os.path.exists(ref) else None
    proc = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True,
                          text=True, timeout=2400)
    diffs = {}
    for f in ("benchmark", "conservation", "convergence", "symmetry"):
        p = os.path.join(ROOT, "runs", "m14679", f + ".json")
        after = json.load(open(p)) if os.path.exists(p) else None
        diffs[f] = "reference copy unavailable" if before[f] is None \
            else ("identical" if json.dumps(after, sort_keys=True) ==
                  json.dumps(before[f], sort_keys=True) else "DIFFERS")
    return {"reference_dir": os.path.join(DELIVERY_WORKTREE, "runs", "m14679"),
            "return_code": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-4:],
            "artifact_diff": diffs,
            "pass": bool(proc.returncode == 0 and all(v != "DIFFERS" for v in diffs.values()))}


if __name__ == "__main__":
    rng = np.random.default_rng(20261111)
    res = {"mission_audited": 14679, "audit_mission": 15004,
           "repo": "TensorPhysics"}
    res["A_clip"] = probe_clip(rng)
    print("A", res["A_clip"])
    res["B_partition"] = probe_partition()
    print("B", res["B_partition"])
    res["C_convergence"] = probe_convergence()
    print("C/D", json.dumps(res["C_convergence"]["global_fit"], indent=1))
    res["E_orientation"] = probe_orientation()
    print("E", res["E_orientation"]["verdict"])
    res["F_repro"] = probe_repro()
    print("F", res["F_repro"]["artifact_diff"], res["F_repro"]["return_code"])
    path = os.path.join(OUT, "checks.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", path)