"""Machine-checkable tests for adaptive h-refinement + ZZ error estimation.

Coverage map (campaign node p284w3-n03-fem-adaptive-refinement):

mesh layer
    * structured/mapped mesh integrity (area, orientation, quality)
    * conforming red-green refinement (no hanging nodes, area preserving)
    * Rivara longest-edge refinement (angle bound, conformity, area)
    * guarded Delaunay flipping (area/quality monotonicity)

recovery & estimator layer
    * SPR reproducibility of a linear stress field (exactness property)
    * CST patch test (linear displacement reproduced exactly)
    * ZZ indicators vs the true error on the Lamé cylinder (effectivity)
    * Dörfler marking minimality / coverage

physics anchors
    * Lamé: L2 convergence order and adaptive accuracy < 1%
    * Cook: adaptive beats uniform at equal DOF (measured ratio recorded)
    * crack tip: K_I within 5% of the analytic finite-width value
    * crack tip: adaptive grading concentrates nodes at the tip
    * energy-norm consistency with the discrete strain energy
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.adaptive_fem import (
    LameFields,
    Material2D,
    PlaneProblem,
    annulus_mesh,
    boundary_edges,
    center_crack_KI,
    cook_mesh,
    cook_problem,
    cook_tip_displacement,
    crack_mesh,
    crack_problem,
    displacement_l2_error,
    dorfler_mark,
    edge_records,
    flip_to_quality,
    ki_from_cod,
    ki_from_jintegral,
    ki_from_stress,
    lame_problem,
    mesh_summary,
    min_angles,
    cst_b_matrices,
    element_stress,
    refine_longest_edge,
    refine_red_green,
    signed_areas,
    zz_estimate,
)
from tensorphysics.adaptive_fem.estimator import ReferenceStressField, actual_energy_error


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def has_hanging_nodes(nodes: np.ndarray, tris: np.ndarray, tol: float = 1e-12) -> bool:
    """True if a vertex lies strictly inside an element edge (non-conforming)."""
    be = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    p0 = nodes[be[:, 0]]
    p1 = nodes[be[:, 1]]
    # cheap filter: only vertices inside an edge's bounding box
    lo = np.minimum(p0, p1)
    hi = np.maximum(p0, p1)
    span = np.linalg.norm(p1 - p0, axis=1)
    for k in np.argsort(span)[: min(len(span), 1500)]:
        a, b = p0[k], p1[k]
        d = b - a
        L2 = float(d @ d)
        if L2 <= 0.0:
            continue
        rel = nodes - a
        t = (rel @ d) / L2
        inside = (t > tol) & (t < 1.0 - tol) & (np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / math.sqrt(L2) < 1e-10)
        # exclude the two edge endpoints themselves
        inside[be[k, 0]] = False
        inside[be[k, 1]] = False
        if np.any(inside):
            return True
    return False


def conforming(nodes: np.ndarray, tris: np.ndarray) -> bool:
    _, counts, _ = edge_records(tris)
    return bool(np.all(counts <= 2)) and not has_hanging_nodes(nodes, tris)


# ---------------------------------------------------------------------------
# mesh layer
# ---------------------------------------------------------------------------
def test_cook_mesh_geometry_and_quality():
    m = cook_mesh(8)
    s = mesh_summary(m.nodes, m.tris)
    assert s["ccw"] is True
    # trapezoid: length 48, heights 60 (clamped) and 44 (loaded)
    assert s["area"] == pytest.approx(48.0 * (60.0 + 44.0) / 2.0, rel=1e-12)
    assert s["n_nonmanifold_edges"] == 0
    assert s["min_angle_deg"] > 30.0
    assert m.n_nodes == 81 and m.n_tris == 128


def test_annulus_mesh_area_matches_analytic():
    a, b = 0.1, 0.25
    m = annulus_mesh(8, 32, a, b)
    s = mesh_summary(m.nodes, m.tris)
    # polygonal (chordal) approximation of the quarter annulus: O((dtheta)^2)
    assert s["area"] == pytest.approx(0.25 * math.pi * (b * b - a * a), rel=2e-3)
    assert s["ccw"] is True


def test_red_green_refinement_is_conforming_and_area_preserving():
    m = cook_mesh(6)
    area0 = mesh_summary(m.nodes, m.tris)["area"]
    marked = np.arange(0, m.n_tris, 3)
    r = refine_red_green(m, marked)
    assert mesh_summary(r.nodes, r.tris)["area"] == pytest.approx(area0, rel=1e-12)
    assert np.all(signed_areas(r.nodes, r.tris) > 0.0)
    assert conforming(r.nodes, r.tris)
    assert r.n_tris > m.n_tris


def test_red_green_twice_keeps_conformity():
    m = cook_mesh(5)
    r1 = refine_red_green(m, [0, 1, 2, 7])
    r2 = refine_red_green(r1, [0, 3, 5, 9, 11])
    assert conforming(r2.nodes, r2.tris)
    assert mesh_summary(r2.nodes, r2.tris)["n_nonmanifold_edges"] == 0


def test_longest_edge_refinement_angle_bound_and_conformity():
    m = cook_mesh(8)
    area0 = mesh_summary(m.nodes, m.tris)["area"]
    min_angle0 = min_angles(m.nodes, m.tris).min()
    cur = m
    for _ in range(5):
        cur = refine_longest_edge(cur, np.arange(0, cur.n_tris, 4))
        assert conforming(cur.nodes, cur.tris)
    s = mesh_summary(cur.nodes, cur.tris)
    assert s["area"] == pytest.approx(area0, rel=1e-10)
    # Rivara bound: angles stay above half of the initial minimum angle
    assert s["min_angle_deg"] > 0.5 * min_angle0
    assert s["min_angle_deg"] > 20.0
    assert s["ccw"] is True


def test_delaunay_flip_preserves_area_and_never_worsens_quality():
    m = cook_mesh(6)
    r = refine_red_green(m, [0, 2, 5, 8, 11])
    before = mesh_summary(r.nodes, r.tris)
    f = flip_to_quality(r)
    after = mesh_summary(f.nodes, f.tris)
    assert after["area"] == pytest.approx(before["area"], rel=1e-12)
    assert after["min_angle_deg"] >= before["min_angle_deg"] - 1e-9
    assert conforming(f.nodes, f.tris)


# ---------------------------------------------------------------------------
# recovery / estimator layer
# ---------------------------------------------------------------------------
def test_cst_patch_test_reproduces_linear_displacement():
    """A linear displacement field must be reproduced exactly by T3 elements."""
    mat = Material2D(E=1000.0, nu=0.3, plane="stress")
    A = np.array([[1.0e-3, 2.0e-4], [-1.0e-4, 5.0e-4]])
    t = np.array([1.0e-3, -2.0e-3])

    def exact(p):
        return p @ A.T + t

    mesh = cook_mesh(6)
    B, _area = cst_b_matrices(mesh.nodes, mesh.tris)
    u_ex = exact(mesh.nodes).reshape(-1)
    stress = element_stress(B, u_ex, mesh.tris, mat)
    strain_exact = np.array([A[0, 0], A[1, 1], A[0, 1] + A[1, 0]])
    stress_exact = mat.D @ strain_exact
    assert np.allclose(stress, stress_exact[None, :], rtol=1e-10, atol=1e-10)


def test_spr_recovery_is_exact_for_linear_stress_field():
    """SPR must reproduce a stress field that is linear in (x, y)."""
    mesh = cook_mesh(8)
    cent = mesh.centroids()
    grad = np.array([[0.3, -0.2], [0.1, 0.4], [0.25, 0.05]])
    offset = np.array([1.0, -0.5, 0.2])
    elem_stress = offset[None, :] + cent @ grad.T
    from tensorphysics.adaptive_fem import spr_recover_nodal_stress

    nodal = spr_recover_nodal_stress(mesh.nodes, mesh.tris, elem_stress)
    exact = offset[None, :] + mesh.nodes @ grad.T
    # exactness holds for patches with >= 3 non-collinear sampling points; corner
    # nodes own a rank-deficient patch and fall back to the area-weighted mean
    # (documented SPR safeguard), so they are excluded from the exactness claim
    n_inc = np.bincount(mesh.tris.reshape(-1), minlength=mesh.n_nodes)
    sel = n_inc >= 4
    assert sel.sum() > 20
    assert np.allclose(nodal[sel], exact[sel], atol=1e-8)


def test_zz_effectivity_on_lame_cylinder():
    a, b, p, E, nu = 0.1, 0.25, 100e6, 200e9, 0.3
    prob = lame_problem(a, b, p, E, nu)
    fields = LameFields(a, b, p, prob.material)
    mesh = annulus_mesh(8, 16, a, b)
    sol = prob.solve(mesh)
    est = zz_estimate(sol)
    actual = actual_energy_error(sol, fields.stress(mesh.centroids()))
    eff = est.eta_total / actual
    assert 0.8 <= eff <= 1.2, f"effectivity {eff} outside [0.8, 1.2]"


def test_zz_estimate_decreases_under_refinement():
    a, b, p = 0.1, 0.25, 100e6
    prob = lame_problem(a, b, p)
    etas = []
    for nr, nt in ((4, 8), (8, 16), (16, 32)):
        etas.append(zz_estimate(prob.solve(annulus_mesh(nr, nt, a, b))).eta_total)
    assert etas[0] > etas[1] > etas[2]
    # second-order-ish reduction of the energy norm for smooth data
    assert etas[0] / etas[1] > 2.0


def test_dorfler_marking_covers_half_the_error_with_a_minimal_set():
    rng = np.random.default_rng(7)
    eta = rng.random(500)
    marked = dorfler_mark(eta, 0.5)
    total = float(np.sum(eta**2))
    assert np.sum(eta[marked] ** 2) >= 0.5 * total - 1e-12
    # minimality: dropping the smallest marked element must break the coverage
    smallest = marked[np.argmin(eta[marked])]
    rest = np.setdiff1d(marked, [smallest])
    assert np.sum(eta[rest] ** 2) < 0.5 * total


def test_dorfler_marking_edge_cases():
    assert dorfler_mark(np.zeros(10), 0.5).size == 0
    with pytest.raises(ValueError):
        dorfler_mark(np.ones(3), 0.0)
    marked = dorfler_mark(np.array([1.0, 0.0, 0.0]), 1.0)
    assert list(marked) == [0]


def test_energy_norm_consistent_with_strain_energy():
    prob = cook_problem()
    mesh = cook_mesh(8)
    sol = prob.solve(mesh)
    # 0.5 * int sigma^T D^-1 sigma dA must equal 0.5 u^T f for the FE solution
    uf = float(sol.u @ prob.external_forces(mesh))
    assert sol.energy_norm_sq() == pytest.approx(uf, rel=1e-10)


# ---------------------------------------------------------------------------
# Lamé anchor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lame_case():
    a, b, p = 0.1, 0.25, 100e6
    prob = lame_problem(a, b, p)
    fields = LameFields(a, b, p, prob.material)
    return prob, fields, a, b


def test_lame_uniform_l2_convergence_order(lame_case):
    prob, fields, a, b = lame_case
    errs = []
    for nr, nt in ((4, 8), (8, 16), (16, 32)):
        mesh = annulus_mesh(nr, nt, a, b)
        sol = prob.solve(mesh)
        errs.append(displacement_l2_error(mesh, sol.u, fields.displacement)["rel"])
    rate = math.log2(errs[0] / errs[2]) / 2.0
    assert 1.6 <= rate <= 2.6, f"observed L2 rate {rate}"


def test_lame_adaptive_l2_error_below_one_percent(lame_case):
    prob, fields, a, b = lame_case
    mesh = annulus_mesh(4, 8, a, b)
    for _ in range(8):
        sol = prob.solve(mesh)
        mesh = refine_longest_edge(mesh, dorfler_mark(zz_estimate(sol).eta_elem, 0.5))
    sol = prob.solve(mesh)
    err = displacement_l2_error(mesh, sol.u, fields.displacement)
    assert err["rel"] < 0.01, err
    assert min_angles(mesh.nodes, mesh.tris).min() > 20.0


def test_lame_inner_radius_displacement_matches_closed_form(lame_case):
    prob, fields, a, b = lame_case
    mesh = annulus_mesh(16, 32, a, b)
    sol = prob.solve(mesh)
    r = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    ur = np.sum(sol.u.reshape(-1, 2) * mesh.nodes / r[:, None], axis=1)
    sel = np.abs(r - a) < 1e-12
    exact = float(fields.u_r(np.array([a]))[0])
    # CST on a 16x32 polar mesh: ~0.4% (second-order convergence)
    assert np.mean(ur[sel]) == pytest.approx(exact, rel=8e-3)


# ---------------------------------------------------------------------------
# Cook anchor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cook_case():
    prob = cook_problem()
    # independent reference: finer start mesh, 9 adaptive levels
    ref_mesh = cook_mesh(16)
    for _ in range(9):
        sol = prob.solve(ref_mesh)
        ref_mesh = refine_longest_edge(ref_mesh, dorfler_mark(zz_estimate(sol).eta_elem, 0.5))
    ref_sol = prob.solve(ref_mesh)
    ref = ReferenceStressField(ref_sol.mesh.nodes, ref_sol.mesh.tris, ref_sol.stress)
    tip = cook_tip_displacement(ref_sol.mesh, ref_sol.u)
    return prob, ref, tip


def test_cook_tip_displacement_is_converged(cook_case):
    prob, ref, tip_ref = cook_case
    # converged value under this convention (E=1, nu=1/3, t=1, total shear 6.25)
    assert 34.0 < tip_ref < 39.0
    # a coarse uniform mesh must be stiffer (CST is displacement-underestimating)
    coarse = cook_tip_displacement(cook_mesh(8), prob.solve(cook_mesh(8)).u)
    assert coarse < tip_ref


def test_cook_adaptive_beats_uniform_at_equal_dof(cook_case):
    prob, ref, _ = cook_case

    def rel_error(mesh):
        sol = prob.solve(mesh)
        rs = ref.at(mesh.centroids())
        act = actual_energy_error(sol, rs)
        nrm = math.sqrt(
            float(
                np.sum(
                    np.einsum("mi,ij,mj->m", rs, prob.material.compliance, rs)
                    * sol.area
                    * prob.material.thickness
                )
            )
        )
        return act / nrm

    uniform = []
    cur = cook_mesh(8)
    for _ in range(3):
        uniform.append((2 * cur.n_nodes, rel_error(cur)))
        cur = refine_red_green(cur, range(cur.n_tris))

    cur = cook_mesh(8)
    adaptive = []
    for _ in range(6):
        adaptive.append((2 * cur.n_nodes, rel_error(cur)))
        sol = prob.solve(cur)
        cur = refine_longest_edge(cur, dorfler_mark(zz_estimate(sol).eta_elem, 0.5))

    # compare at the largest adaptive DOF that is still inside the uniform range
    udof = np.log(np.array([d for d, _ in uniform], float))
    uerr = np.log(np.array([e for _, e in uniform], float))
    d, e = [x for x in adaptive if x[0] <= uniform[-1][0]][-1]
    uniform_at_d = float(np.exp(np.interp(np.log(d), udof, uerr)))
    ratio = e / uniform_at_d
    # regression guard on the measured (honest, modest) adaptive advantage
    assert ratio < 0.95, f"adaptive/uniform ratio at DOF {d} = {ratio}"
    assert 0.5 < ratio < 0.95


def test_cook_estimator_effectivity(cook_case):
    prob, ref, _ = cook_case
    for n in (8, 16):
        mesh = cook_mesh(n)
        sol = prob.solve(mesh)
        est = zz_estimate(sol)
        actual = actual_energy_error(sol, ref.at(mesh.centroids()))
        assert 0.75 <= est.eta_total / actual <= 1.3


# ---------------------------------------------------------------------------
# crack-tip anchor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crack_case():
    a, W, H, sigma = 1.0, 10.0, 10.0, 1.0
    prob = crack_problem(a=a, W=W, H=H, sigma=sigma)
    K_ref = center_crack_KI(a, W, sigma)
    mesh = crack_mesh(a, W, H)
    hist = []
    for _ in range(12):
        sol = prob.solve(mesh)
        K, _info = ki_from_jintegral(mesh, sol.u, sol.stress, a, prob.material)
        hist.append((2 * mesh.n_nodes, K, abs(K - K_ref) / K_ref))
        est = zz_estimate(sol)
        mesh = refine_longest_edge(mesh, dorfler_mark(est.eta_elem, 0.5))
    sol = prob.solve(mesh)
    K_final, info = ki_from_jintegral(mesh, sol.u, sol.stress, a, prob.material)
    return prob, K_ref, hist, (2 * mesh.n_nodes, K_final, mesh, sol, info)


def test_crack_reference_sif_value():
    assert center_crack_KI(1.0, 10.0, 1.0) == pytest.approx(
        1.0 * math.sqrt(math.pi) * math.sqrt(1.0 / math.cos(0.5 * math.pi * 0.1)), rel=1e-12
    )


def test_crack_KI_within_five_percent(crack_case):
    prob, K_ref, hist, final = crack_case
    dof, K_final, mesh, sol, info = final
    err = abs(K_final - K_ref) / K_ref
    assert err < 0.05, f"K_I error {err:.4f} (K_J={K_final:.4f}, K_ref={K_ref:.4f})"
    assert min_angles(mesh.nodes, mesh.tris).min() > 20.0
    assert info["G_total"] > 0.0


def test_crack_KI_error_monotone_trend(crack_case):
    prob, K_ref, hist, final = crack_case
    errs = [e for _, _, e in hist]
    assert errs[-1] < errs[0] / 2.0
    first_ok = next(i for i, e in enumerate(errs) if e < 0.05)
    assert hist[first_ok][0] < 700, "5% accuracy should be reached below 700 free DOF"


def test_crack_KI_cross_check_from_cod(crack_case):
    prob, K_ref, hist, final = crack_case
    _, _, mesh, sol, _ = final
    K_cod, info = ki_from_cod(mesh, sol.u, 1.0, prob.material, n_fit=6)
    err = abs(K_cod - K_ref) / K_ref
    # a pointwise near-tip fit is far more sensitive to mesh grading than the
    # domain J-integral; 20% is the honest cross-check tolerance
    assert err < 0.20, f"COD extrapolation error {err:.3f}"
    assert info["n_fit"] >= 3


def test_crack_adaptive_grading_concentrates_nodes_at_tip(crack_case):
    prob, K_ref, hist, final = crack_case
    _, _, mesh, _, _ = final
    r = np.hypot(mesh.nodes[:, 0] - 1.0, mesh.nodes[:, 1])
    n_tip_adaptive = int(np.sum(r < 0.05))
    uni = crack_mesh(1.0, 10.0, 10.0, n_crack=2, n_right=int(math.sqrt(mesh.n_tris / 2)), n_y=int(math.sqrt(mesh.n_tris / 2)))
    r_u = np.hypot(uni.nodes[:, 0] - 1.0, uni.nodes[:, 1])
    n_tip_uniform = int(np.sum(r_u < 0.05))
    assert n_tip_adaptive > 5 * max(n_tip_uniform, 1)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def test_adaptive_driver_records_and_monotone_estimate():
    from tensorphysics.adaptive_fem import adaptive_refinement

    prob = lame_problem(0.1, 0.25, 100e6)
    run = adaptive_refinement(prob, annulus_mesh(4, 8, 0.1, 0.25), n_steps=4)
    assert len(run.steps) == 5
    etas = [s.eta_total for s in run.steps]
    assert all(b <= a for a, b in zip(etas, etas[1:]))
    assert run.final_solution is not None
    assert run.final.n_free_dof > run.steps[0].n_free_dof
    row = run.table()[0]
    for key in ("n_dof", "eta_total", "min_angle_deg", "n_marked"):
        assert key in row


def test_uniform_refinement_doubles_dof_fourfold():
    from tensorphysics.adaptive_fem import doubling_refinement

    prob = lame_problem(0.1, 0.25, 100e6)
    run = doubling_refinement(prob, annulus_mesh(3, 6, 0.1, 0.25), n_steps=2)
    assert run.steps[1].n_tris == pytest.approx(4 * run.steps[0].n_tris)
    assert run.steps[2].n_tris == pytest.approx(16 * run.steps[0].n_tris)