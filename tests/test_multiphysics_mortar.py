"""Tests for the conservative multiphysics mortar (campaign node P162W1-N06).

Every criterion C1..C7 of the node is exercised here with the *declared* tolerance
from :data:`tensorphysics.multiphysics_mortar.gate.GATE_TOLERANCES`, plus the
structural properties the operator relies on (partition of unity, biorthogonality,
equivalence of the two transfer routes) and two implementation cross-checks
(fast clipper vs canonical clipper, hand-off solve vs LAPACK).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.interface_operators.polygons import (clip_convex, polygon_area,
                                                        polygon_integral)
from tensorphysics.multiphysics_mortar import (ConservationGate, ConservationGateError,
                                               GATE_TOLERANCES, MaterialState,
                                               conservation_residuals)
from tensorphysics.multiphysics_mortar.acceptance import (N_COUPLED_STEPS,
                                                          _edge_nodes,
                                                          _identity_defect,
                                                          _refinement_study,
                                                          _timing_study,
                                                          run_case,
                                                          state_from_conserved,
                                                          transport_conserved)
from tensorphysics.multiphysics_mortar.assembly import (_bary_rows, _clip_tri_fast,
                                                        _piece_matrices, spd_inverse,
                                                        spd_solve)
from tensorphysics.multiphysics_mortar.cases import (flow_ice_3d_case, fsi_2d_case,
                                                     matching_2d_case,
                                                     matching_line_case,
                                                     refined_flow_ice_case,
                                                     scalar_transport_1d_case)
from tensorphysics.multiphysics_mortar.spaces import (LineTraceSpace, TriTraceSpace,
                                                      cosine_nodes,
                                                      structured_triangles)


# ---------------------------------------------------------------------------
# structural properties
# ---------------------------------------------------------------------------


def test_line_space_partition_of_unity_and_measure():
    nodes = cosine_nodes(7, 0.0, 2.0)
    space = LineTraceSpace(nodes)
    assert space.n == 8 and space.n_cells == 7
    assert space.partition_of_unity_error() < 1e-15
    # sum of lumped measures == interface length (independent of the basis)
    assert pytest.approx(space.measures().sum(), rel=1e-14) == 2.0


def test_tri_space_partition_of_unity_and_measure():
    tris = structured_triangles(4, 3, (0.0, 0.0), (2.0, 1.0))
    space = TriTraceSpace(tris)
    assert space.partition_of_unity_error() < 1e-12
    # the lumped measures must add up to the patch area 2 x 1
    assert pytest.approx(space.measures().sum(), rel=1e-13) == 2.0
    # ... and to the per-triangle areas computed by the shared kernel
    assert pytest.approx(float(np.sum(space.areas())), rel=1e-13) == 2.0


def test_dual_basis_is_biorthogonal_but_not_a_partition_of_unity():
    """The conservation proof must not rely on the dual basis summing to one."""
    transfer = fsi_2d_case().transfer()
    assert transfer.biorthogonality_defect() < 1e-13
    # L2 dual basis is biorthogonal, *not* a partition of unity (o(1) defect)
    assert transfer.dual_partition_of_unity_defect() > 1.0


def test_dual_and_matrix_transfer_routes_agree():
    case = flow_ice_3d_case()
    transfer = case.transfer()
    payload = case.master_state.momentum
    direct = transfer.apply(payload)
    dual = transfer.apply_dual(payload)
    scale = float(np.max(np.abs(direct)))
    assert float(np.max(np.abs(direct - dual))) / scale < 1e-12


def test_fast_clipper_matches_canonical_clipper():
    """The optimised clipper is an optimisation with a reference, not a rewrite."""
    case = flow_ice_3d_case()
    src, tgt = case.master_space, case.slave_space
    rng = np.random.default_rng(7)
    lo_s, hi_s = src.triangles.min(axis=1), src.triangles.max(axis=1)
    lo_t, hi_t = tgt.triangles.min(axis=1), tgt.triangles.max(axis=1)
    mask = np.all(np.minimum(hi_s[:, None], hi_t[None]) - np.maximum(lo_s[:, None], lo_t[None]) > 1e-12,
                  axis=2)
    overlapping = np.argwhere(mask)
    sample = overlapping[rng.choice(overlapping.shape[0], size=40, replace=False)]
    checked = 0
    for i, j in sample:
        i, j = int(i), int(j)
        fast = _clip_tri_fast(tgt.triangles[j], src.triangles[i])
        canon = clip_convex(tgt.triangles[j], src.triangles[i])
        if len(fast) < 3 or canon.shape[0] < 3:
            assert len(fast) < 3 and canon.shape[0] < 3
            continue
        poly = np.array(fast, dtype=float)
        assert polygon_area(poly) == pytest.approx(polygon_area(canon), abs=1e-14)

        def f(q, i=i, j=j):
            ls = _bary_rows(src.triangles[i], q)
            lt = _bary_rows(tgt.triangles[j], q)
            return (lt[:, :, None] * ls[:, None, :]).reshape(q.shape[0], 9)

        ref = np.asarray(polygon_integral(canon, f, weight=tgt.weights)).reshape(3, 3)
        got = _piece_matrices([poly], src.triangles[i][None],
                              tgt.triangles[j][None], tgt.weights)[0]
        assert float(np.max(np.abs(ref - got))) < 1e-14
        checked += 1
    assert checked > 10


def test_spd_solve_matches_lapack():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((24, 24))
    m = a @ a.T + 24.0 * np.eye(24)
    b = rng.standard_normal((24, 3))
    scale = float(np.max(np.abs(np.linalg.solve(m, b))))
    assert float(np.max(np.abs(spd_solve(m, b) - np.linalg.solve(m, b)))) / scale < 1e-14
    eye = np.eye(24)
    assert float(np.max(np.abs(m @ spd_inverse(m) - eye))) < 1e-12


# ---------------------------------------------------------------------------
# C1 / C2 / C3: residuals on the three physical cases
# ---------------------------------------------------------------------------


def test_c1_1d_scalar_transport_mass_residual():
    payload = run_case(scalar_transport_1d_case())
    residual = payload["residuals_master_to_slave"]["mass"]
    assert residual <= GATE_TOLERANCES["C1_1d_mass"]
    # the reverse direction must be conservative too
    assert payload["residuals_slave_to_master"]["mass"] <= GATE_TOLERANCES["C1_1d_mass"]


def test_c2_2d_fsi_residuals():
    payload = run_case(fsi_2d_case())
    assert payload["residuals_master_to_slave"]["momentum"] <= GATE_TOLERANCES["C2_2d_momentum"]
    assert payload["residuals_master_to_slave"]["energy"] <= GATE_TOLERANCES["C2_2d_energy"]


def test_c3_3d_flow_ice_residuals():
    payload = run_case(flow_ice_3d_case())
    assert payload["residuals_master_to_slave"]["mass"] <= GATE_TOLERANCES["C3_3d_mass"]
    assert payload["residuals_master_to_slave"]["momentum"] <= GATE_TOLERANCES["C3_3d_momentum"]


def test_state_round_trip_is_exact():
    """transport_conserved must invert (rho, rho u, rho E) -> (rho, u, T) exactly."""
    case = fsi_2d_case()
    transfer = case.transfer(np.eye(1, dtype=int)[0]) if False else case.transfer()
    state = case.master_state
    q = state.quantities()
    back = state_from_conserved(q["mass"][:, 0], q["momentum"], q["energy"][:, 0],
                                state.specific_heat)
    assert float(np.max(np.abs(back.density - state.density))) < 1e-12
    assert float(np.max(np.abs(back.velocity - state.velocity))) < 1e-12
    assert float(np.max(np.abs(back.temperature - state.temperature))) < 1e-9
    assert transfer.assembly.src is case.master_space


# ---------------------------------------------------------------------------
# C4: matching meshes -> identity
# ---------------------------------------------------------------------------


def test_c4_matching_meshes_are_the_identity():
    one_d = _identity_defect(matching_line_case())
    two_d = _identity_defect(matching_2d_case())
    assert one_d["max_over_quantities"] <= GATE_TOLERANCES["C4_identity_1d"]
    assert two_d["max_over_quantities"] <= GATE_TOLERANCES["C4_identity_2d"]


# ---------------------------------------------------------------------------
# C5: refinement monotonicity
# ---------------------------------------------------------------------------


def test_c5_refinement_is_monotone():
    study = _refinement_study()
    mass = [row["residuals"]["mass"] for row in study["levels"]]
    assert study["monotone_mass"] is True
    assert mass[-1] < 1e-3 * mass[0]
    assert study["observed_order_mass"] is not None
    assert study["observed_order_mass"] > 3.0


# ---------------------------------------------------------------------------
# C6: versus the nearest-neighbour baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [scalar_transport_1d_case, fsi_2d_case,
                                     flow_ice_3d_case])
def test_c6_mortar_beats_nearest_neighbor(factory):
    payload = run_case(factory())
    for name, ratio in payload["mortar_over_nearest_neighbor"].items():
        assert ratio >= 1e3, (name, ratio)


# ---------------------------------------------------------------------------
# C7: setup cost amortised over a coupled run
# ---------------------------------------------------------------------------


def test_c7_setup_is_amortised():
    timing = _timing_study(flow_ice_3d_case(), steps=N_COUPLED_STEPS)
    assert timing["setup_fraction"] <= 0.10
    # per-step solver work must dominate: the loop really solves something
    assert timing["per_step_seconds"] > 1e-3


# ---------------------------------------------------------------------------
# gate behaviour
# ---------------------------------------------------------------------------


def test_gate_accepts_clean_operator_and_rejects_corruption():
    gate = ConservationGate()
    clean = {"C1_1d_mass": 1e-16, "C3_3d_mass": 1e-15}
    assert gate.evaluate(clean).passed is True

    case = scalar_transport_1d_case()
    transfer = case.transfer()
    corrupted = transfer.matrix() * (1.0 + 1e-6)  # 1 ppm mass injection
    slave = corrupted @ case.master_state.density
    res = conservation_residuals(case.master_space, case.master_state,
                                case.slave_space,
                                state_from_conserved(
                                    slave,
                                    corrupted @ case.master_state.momentum,
                                    corrupted @ case.master_state.energy,
                                    case.master_state.specific_heat))
    with pytest.raises(ConservationGateError):
        gate.require({"C1_1d_mass": res["mass"].residual})
    assert res["mass"].residual > GATE_TOLERANCES["C1_1d_mass"]


def test_gate_is_fail_closed():
    with pytest.raises(KeyError):
        ConservationGate().evaluate({"C9_undeclared": 1e-20})