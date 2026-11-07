"""Acceptance harness for IN50-N08: geometry gate + 2D/3D mortar operators.

Every number in the returned payload is machine computed; nothing is typed in by
hand.  Criteria (mission metadata):

====  ==================================================================
C1    CAD vs grid Cb/L/B/nabla/S deviations < 0.5 % on a five-hull corpus
C2    constant-field mortar conservation residual < 1e-14 (2D and 3D)
C3    linear-field observed order p >= 1.9 (2D and 3D)
C4    total-flux conservation residual / sum|flux| < 1e-12 after the fix
C5    2D polygon-clipping area error < 0.1 %
C6    3D patch-mapping area error < 0.5 %
C7    >= 12 pytest cases green
====  ==================================================================

Run ``python -c "from tensorphysics.interface_operators.acceptance_nd import run_acceptance; run_acceptance(verbose=True)"``
or the repository script ``scripts/run_in50_n08_acceptance.py`` (writes the JSON
artifact and the figures).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any

import numpy as np

from ..geometry_fidelity.gate import (C1_TOLERANCES, GeometryGate,
                                      corrupted_discretisation, discretise,
                                      gate_corpus)
from .core import conservation_residual
from .mortar_nd import (FIELD_2D, FIELD_3D, MortarProjection2D,
                        SurfaceMortarProjection, analytic_surface_area,
                        bulged_surface, faceted_surface_area,
                        local_conservation_fix, overlap_measure_closure)
from .polygons import (cartesian_mesh, clip_convex, convex_intersection,
                       fan_mesh, polygon_area, triangulated_mesh)

__all__ = ["TOL", "run_acceptance", "count_pytest_cases"]

TOL = {
    "C1_rel": 5e-3,
    "C2_residual": 1e-14,
    "C3_order": 1.9,
    "C4_residual": 1e-12,
    "C5_area_rel": 1e-3,
    "C6_area_rel": 5e-3,
    "C7_min_tests": 12,
}

SEED = 20261106


# ---------------------------------------------------------------------------
# geometry gate (C1)
# ---------------------------------------------------------------------------


def _c1() -> dict[str, Any]:
    gate = GeometryGate()
    corpus = gate.evaluate_corpus(resolutions=((120, 48), (96, 40)))
    controls = gate.negative_controls()
    # honest resolution study on the hardest (fullest) hull
    hull = gate_corpus()["bulk_carrier"]
    convergence = []
    for n_x, n_s in ((60, 24), (96, 40), (120, 48), (180, 72)):
        report = gate.evaluate(hull, discretise(hull, n_x, n_s))
        convergence.append({
            "n_x": n_x, "n_s": n_s,
            "deviations": dict(report.deviations),
            "verdict": report.verdict,
        })
    return {
        "criterion": "C1",
        "description": "CAD vs discretisation: L, B, Cb, nabla, S of five hulls",
        "tolerances": dict(C1_TOLERANCES),
        "corpus": corpus,
        "negative_controls": controls,
        "bulk_carrier_resolution_study": convergence,
        "worst_deviation": corpus["worst_deviation"],
        "max_deviation": float(max(corpus["worst_deviation"].values())),
        "pass": bool(corpus["verdict"] == "GATE_PASS"
                     and controls["n_rejected"] == controls["n_controls"]),
    }


# ---------------------------------------------------------------------------
# mortar operators (C2..C6)
# ---------------------------------------------------------------------------


def mesh_pair_2d(n: int = 16, m: int = 12):
    """Non-matching 2D partition pair: Cartesian source, triangulated target."""
    return cartesian_mesh(n, n), triangulated_mesh(m, m)


def _c2() -> dict[str, Any]:
    op2 = MortarProjection2D()
    src2, tgt2 = mesh_pair_2d()
    f2 = FIELD_2D[0]
    u2 = src2.cell_averages(f2.func)
    v2 = op2.transfer(src2, u2, tgt2)
    res2 = op2.conservation_residual(src2, u2, tgt2, v2)

    op3 = SurfaceMortarProjection()
    r, dr = bulged_surface()
    from .polygons import axisymmetric_mesh
    src3 = axisymmetric_mesh(16, 12, r, dr)
    tgt3 = axisymmetric_mesh(12, 9, r, dr)
    f3 = FIELD_3D[0]
    u3 = src3.cell_averages(f3.func)
    v3 = op3.transfer(src3, u3, tgt3)
    res3 = op3.conservation_residual(src3, u3, tgt3, v3)
    return {
        "criterion": "C2",
        "description": "constant-field mortar conservation residual (2D, 3D)",
        "residual_2d": float(res2),
        "residual_3d": float(res3),
        "max_residual": float(max(res2, res3)),
        "limit": TOL["C2_residual"],
        "measure_correction_2d": float(op2.measure_correction(src2, tgt2)),
        "measure_correction_3d": float(op3.measure_correction(src3, tgt3)),
        "pass": bool(max(res2, res3) < TOL["C2_residual"]),
    }


def _c3() -> dict[str, Any]:
    from .polygons import axisymmetric_mesh

    op2 = MortarProjection2D()
    pairs_2d = [(cartesian_mesh(n, n), triangulated_mesh(n, n)) for n in (8, 16, 32)]
    est_2d = op2.observed_order(FIELD_2D[3], pairs_2d)          # harmonic
    lin_2d = op2.observed_order(FIELD_2D[1], pairs_2d)          # linear: exact

    op3 = SurfaceMortarProjection()
    r, dr = bulged_surface()

    def pair(n: int, m: int):
        kw = dict(theta0=0.3, theta_span=2.6)
        return (axisymmetric_mesh(n, n, r, dr, **kw), axisymmetric_mesh(m, m, r, dr, **kw))

    pairs_3d = [pair(6, 4), pair(12, 8), pair(24, 16)]
    est_3d = op3.observed_order(FIELD_3D[2], pairs_3d)          # smooth
    lin_3d = op3.observed_order(FIELD_3D[1], pairs_3d)          # linear in chart

    def passes(est):
        return bool(est.exact or (est.order is not None and est.order >= TOL["C3_order"]))

    orders = {
        "2d_harmonic": est_2d.order,
        "2d_linear": "exact" if lin_2d.exact else lin_2d.order,
        "3d_smooth": est_3d.order,
        "3d_linear": "exact" if lin_3d.exact else lin_3d.order,
    }
    numeric = [v for v in (est_2d.order, est_3d.order) if v is not None]
    return {
        "criterion": "C3",
        "description": "linear/smooth field observed order (2D, 3D)",
        "orders": orders,
        "errors_2d": list(est_2d.errors),
        "errors_3d": list(est_3d.errors),
        "linear_exact_2d": bool(lin_2d.exact),
        "linear_exact_3d": bool(lin_3d.exact),
        "linear_error_3d": float(lin_3d.errors[-1]),
        "order_2d": est_2d.order,
        "order_3d": est_3d.order,
        "min_order": float(min(numeric)) if numeric else None,
        "limit": TOL["C3_order"],
        "pass": bool(passes(est_2d) and passes(est_3d) and lin_3d.errors[-1] < 5e-3),
    }


def _c4() -> dict[str, Any]:
    from .polygons import axisymmetric_mesh

    # 2D: sign-changing field, an upstream positivity clip destroys mass, then
    # the local measure correction restores it (operator-agnostic repair)
    src2, tgt2 = mesh_pair_2d(16, 12)
    f2 = FIELD_2D[3]
    u2 = src2.cell_averages(f2.func)
    op2 = MortarProjection2D()
    v2_clipped = np.maximum(op2.transfer(src2, u2, tgt2), 0.0)
    a_t2 = op2.effective_target_measure(src2, tgt2)
    res2 = conservation_residual(u2, src2.measures, v2_clipped, a_t2)
    budget2 = op2.overlap_budget(src2, u2, tgt2)
    fixed2 = local_conservation_fix(v2_clipped, budget2, a_t2)
    res2_fixed = conservation_residual(u2, src2.measures, fixed2, a_t2)

    # 3D: same story on the curved surface patches
    r, dr = bulged_surface()
    src3 = axisymmetric_mesh(16, 12, r, dr)
    tgt3 = axisymmetric_mesh(12, 9, r, dr)
    f3 = FIELD_3D[3]
    u3 = src3.cell_averages(f3.func)
    op3 = SurfaceMortarProjection()
    v3_clipped = np.maximum(op3.transfer(src3, u3, tgt3), 0.0)
    a_t3 = op3.effective_target_measure(src3, tgt3)
    res3 = conservation_residual(u3, src3.measures, v3_clipped, a_t3)
    budget3 = op3.overlap_budget(src3, u3, tgt3)
    fixed3 = local_conservation_fix(v3_clipped, budget3, a_t3)
    res3_fixed = conservation_residual(u3, src3.measures, fixed3, a_t3)

    # operator-level reference: the same field with the sound operator
    sound2 = MortarProjection2D()
    res2_sound = sound2.conservation_residual(src2, u2, tgt2)
    sound3 = SurfaceMortarProjection()
    res3_sound = sound3.conservation_residual(src3, u3, tgt3)
    return {
        "criterion": "C4",
        "description": "total-flux conservation after the local measure correction",
        "broken_2d": float(res2),
        "fixed_2d": float(res2_fixed),
        "sound_2d": float(res2_sound),
        "broken_3d": float(res3),
        "fixed_3d": float(res3_fixed),
        "sound_3d": float(res3_sound),
        "max_fixed": float(max(res2_fixed, res3_fixed)),
        "limit": TOL["C4_residual"],
        "pass": bool(max(res2_fixed, res3_fixed) < TOL["C4_residual"] and res2 > 1e-4 and res3 > 1e-4),
    }


def _analytic_overlap_error() -> float:
    """Clip error against the analytic rectangle--rectangle overlap area."""
    worst = 0.0
    for nx, ny in ((4, 4), (5, 7), (6, 9)):
        xs = np.linspace(0.0, 1.0, nx + 1)
        ys = np.linspace(0.0, 1.0, ny + 1)
        xs_t = np.linspace(0.0, 1.0, 8 + 1)
        ys_t = np.linspace(0.0, 1.0, 5 + 1)
        for i in range(nx):
            for j in range(ny):
                quad = np.array([[xs[i], ys[j]], [xs[i + 1], ys[j]],
                                 [xs[i + 1], ys[j + 1]], [xs[i], ys[j + 1]]])
                for a in range(8):
                    for b in range(5):
                        win = np.array([[xs_t[a], ys_t[b]], [xs_t[a + 1], ys_t[b]],
                                        [xs_t[a + 1], ys_t[b + 1]], [xs_t[a], ys_t[b + 1]]])
                        ox = max(0.0, min(quad[1, 0], win[1, 0]) - max(quad[0, 0], win[0, 0]))
                        oy = max(0.0, min(quad[2, 1], win[2, 1]) - max(quad[0, 1], win[0, 1]))
                        exact = ox * oy
                        got = polygon_area(clip_convex(quad, win))
                        if exact > 1e-12:
                            worst = max(worst, abs(got - exact) / exact)
    return float(worst)


def _c5() -> dict[str, Any]:
    # (a) two independent clipping algorithms on random convex pairs
    rng = np.random.default_rng(SEED)
    from .polygons import convex_hull
    worst_algo = 0.0
    n_pairs = 0
    for _ in range(200):
        a = convex_hull(rng.uniform(0.0, 1.0, (8, 2)))
        b = convex_hull(rng.uniform(0.0, 1.0, (9, 2)))
        if a.shape[0] < 3 or b.shape[0] < 3:
            continue
        aa = polygon_area(clip_convex(a, b))
        bb = polygon_area(convex_intersection(a, b))
        if max(aa, bb) > 1e-6:
            n_pairs += 1
            worst_algo = max(worst_algo, abs(aa - bb) / max(aa, bb))

    # (b) partition closure for the three partition families
    closures = {}
    cart, tri, fan = cartesian_mesh(8, 8), triangulated_mesh(8, 8), fan_mesh(10, 4)
    for label, s, t in (("cart-tri", cart, tri), ("tri-fan", tri, fan), ("cart-fan", cart, fan)):
        closures[label] = float(overlap_measure_closure(s, t))

    # (c) analytic rectangle-rectangle overlaps
    analytic = _analytic_overlap_error()

    # negative control: bounding-box "clipping" (what a lazy overlap does)
    op_bbox = MortarProjection2D(broken="bbox")
    src_b, tgt_b = cartesian_mesh(12, 12), fan_mesh(12, 5)
    W, _M, _a_s, a_t = op_bbox._assemble(src_b, tgt_b)
    bbox_closure = abs(float(np.sum(W)) - float(np.sum(src_b.measures))) / float(np.sum(src_b.measures))

    worst = max([worst_algo, analytic, *closures.values()])
    return {
        "criterion": "C5",
        "description": "2D polygon clipping area error",
        "algo_agreement_worst": float(worst_algo),
        "analytic_overlap_worst": float(analytic),
        "partition_closure": closures,
        "worst": float(worst),
        "n_random_pairs": n_pairs,
        "negative_control_bbox_closure_error": float(bbox_closure),
        "limit": TOL["C5_area_rel"],
        "pass": bool(worst < TOL["C5_area_rel"] and bbox_closure > 1e-3),
    }


def _c6() -> dict[str, Any]:
    from .polygons import axisymmetric_mesh
    r, dr = bulged_surface()
    analytic = analytic_surface_area(r, dr)
    rows = []
    for n_theta, n_z in ((8, 6), (16, 12), (24, 18), (32, 24)):
        mesh = axisymmetric_mesh(n_theta, n_z, r, dr)
        faceted = faceted_surface_area(mesh)
        chart = float(np.sum(mesh.areas))
        rows.append({
            "n_theta": n_theta, "n_z": n_z,
            "analytic_m2": analytic,
            "faceted_m2": faceted,
            "chart_m2": chart,
            "faceted_rel_error": abs(faceted - analytic) / analytic,
            "chart_rel_error": abs(chart - analytic) / analytic,
        })
    production = rows[-1]
    return {
        "criterion": "C6",
        "description": "3D patch mapping area error on the bulged surface",
        "analytic_m2": analytic,
        "rows": rows,
        "production_resolution": {"n_theta": production["n_theta"], "n_z": production["n_z"]},
        "faceted_rel_error": production["faceted_rel_error"],
        "chart_rel_error": production["chart_rel_error"],
        "limit": TOL["C6_area_rel"],
        "pass": bool(production["faceted_rel_error"] < TOL["C6_area_rel"]),
    }


def count_pytest_cases(paths: list[str] | None = None) -> int | None:
    """Number of collected pytest cases in the two new test modules."""
    paths = paths or ["tests/test_geometry_gate_corpus.py", "tests/test_mortar_polygon_3d.py"]
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
                              capture_output=True, text=True, timeout=300)
    except Exception:                                        # pragma: no cover
        return None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line and line.split()[0].isdigit() and "test" in line:
            try:
                return int(line.split()[0])
            except ValueError:
                continue
    return sum(1 for ln in proc.stdout.splitlines() if "::" in ln) or None


def _c7(pytest_cases: int | None = None) -> dict[str, Any]:
    n = count_pytest_cases() if pytest_cases is None else pytest_cases
    return {
        "criterion": "C7",
        "description": "pytest coverage of the new contracts (>= 12 cases)",
        "collected_tests": n,
        "limit": TOL["C7_min_tests"],
        "pass": bool(n is not None and n >= TOL["C7_min_tests"]),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_acceptance(verbose: bool = False, pytest_cases: int | None = None) -> dict[str, Any]:
    """Run every IN50-N08 criterion and return a JSON-ready payload."""
    results: dict[str, Any] = {}
    for key, func in (("C1", _c1), ("C2", _c2), ("C3", _c3), ("C4", _c4),
                      ("C5", _c5), ("C6", _c6)):
        results[key] = func()
        if verbose:
            print(f"[{key}] pass={results[key]['pass']}  "
                  f"{ {k: v for k, v in results[key].items() if k.endswith(('residual', 'worst', 'order', 'deviation'))} }")
    results["C7"] = _c7(pytest_cases)
    if verbose:
        print(f"[C7] pass={results['C7']['pass']} collected={results['C7']['collected_tests']}")
    n_pass = sum(1 for v in results.values() if v["pass"])
    payload = {
        "campaign_node": "IN50-N08",
        "module": "tensorphysics.interface_operators (+ tensorphysics.geometry_fidelity.gate)",
        "tolerances": TOL,
        "criteria": results,
        "n_pass": n_pass,
        "n_total": len(results),
        "verdict": "PASS" if n_pass == len(results) else "FAIL",
    }
    if verbose:
        print(f"verdict {payload['verdict']} ({n_pass}/{len(results)})")
    return payload


def main() -> int:                                          # pragma: no cover - CLI
    payload = run_acceptance(verbose=True)
    print(json.dumps({"verdict": payload["verdict"], "n_pass": payload["n_pass"]}))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())