"""Tests for the cross-solver interface-operator contract (campaign IN19-T2).

Every campaign gate gets one test that reads the *same* acceptance payload that
the report quotes, plus one negative-control test per operator (criterion C6,
these are the tests that must be "red" for the broken variants).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorphysics.interface_operators import (ANALYTIC_FIELDS,
                                              ConservativeScatter,
                                              LBMDEMMomentumExchange,
                                              MortarProjection,
                                              OperatorReport, OrderEstimate,
                                              SymmetryFlip, conservation_residual,
                                              exact_values, fvm_lbm_chain,
                                              graded_mesh, lbm_dem_chain,
                                              negative_control_suite,
                                              operator_suite, run_acceptance,
                                              uniform_mesh)
from tensorphysics.interface_operators.acceptance import (P_MIN, RESOLUTIONS,
                                                          TOL_C1, TOL_C3,
                                                          TOL_C4, TOL_C5,
                                                          hierarchy,
                                                          particle_lattice)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_acceptance()


# ---------------------------------------------------------------------------
# contract shape
# ---------------------------------------------------------------------------

def test_contract_exposes_three_quantities(payload):
    """The contract must expose conservation_residual / observed_order / symmetry_flip."""
    for name, rows in payload["operator_table"].items():
        for field, row in rows.items():
            assert {"conservation_residual", "l2_error", "symmetry_flip"} <= set(row)
            assert set(row["symmetry_flip"]) >= {"reflection_defect",
                                                 "roundtrip_defect", "value"}
    for name, entry in payload["criteria"]["C2"]["values"].items():
        assert {"linear_order_fit", "linear_errors", "quadratic_order_fit"} <= set(entry)
    assert isinstance(json.dumps(payload, default=str), str)


def test_operator_report_dataclasses_roundtrip():
    rep = OperatorReport("op", "linear", 1e-16, 1e-16,
                         OrderEstimate("linear", (40, 80), (1e-16, 1e-16), None, True),
                         SymmetryFlip(1e-16, 1e-2))
    d = rep.to_dict()
    assert d["observed_order"]["exact_to_roundoff"] is True
    assert d["symmetry_flip"]["value"] == pytest.approx(1e-16)


def test_mesh_cell_average_is_exact_for_polynomials():
    mesh = uniform_mesh(0.0, 1.0, 17)
    quad = ANALYTIC_FIELDS[2]
    # exact cell mean of a parabola: value at midpoint + u''h^2/24
    xc, h = mesh.centers, mesh.widths
    ref = quad.func(xc) + (2.0 * 2.0) * h ** 2 / 24.0
    assert np.max(np.abs(mesh.cell_averages(quad.func) - ref)) < 1e-14


# ---------------------------------------------------------------------------
# C1 / C3: conservation
# ---------------------------------------------------------------------------

def test_c1_constant_field_conservation(payload):
    vals = payload["criteria"]["C1"]["values"]
    for name, v in vals.items():
        assert v < TOL_C1, f"{name} constant-field conservation residual {v:.3e}"
    assert payload["criteria"]["C1"]["passed"] is True


def test_c3_total_flux_conservation_all_fields(payload):
    c3 = payload["criteria"]["C3"]
    assert c3["worst_value"] < TOL_C3, c3["worst_value"]
    assert c3["passed"] is True
    # every operator/field combination, not just the worst
    for name, rows in payload["operator_table"].items():
        for field, row in rows.items():
            assert row["conservation_residual"] < TOL_C3, (name, field)


def test_conservation_residual_is_scale_free():
    assert conservation_residual(np.array([2.0]), np.array([1.0]),
                                 np.array([2.0]), np.array([1.0])) == 0.0
    assert conservation_residual(np.array([1.0]), np.array([1.0]),
                                 np.array([1.5]), np.array([1.0])) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# C2: observed order
# ---------------------------------------------------------------------------

def test_c2_observed_order_remap_operators(payload):
    c2 = payload["criteria"]["C2"]
    for name in ("mortar", "scatter"):
        entry = c2["values"][name]
        assert entry["applicable"] is True
        # linear data are reproduced to round-off (order -> inf)
        assert entry["linear_exact"] is True
        assert max(entry["linear_errors"]) < 1e-13
        # the tangible slope comes from the quadratic field
        assert entry["quadratic_order_fit"] >= P_MIN, entry
        assert entry["passed"] is True
    assert c2["passed"] is True


def test_c2_me_is_documented_exception(payload):
    me = payload["criteria"]["C2"]["values"]["me"]
    assert me["applicable"] is False
    assert me["passed"] is None
    assert me["linear_order_fit"] is not None     # honestly reported
    assert "scope_note" in me


def test_observed_order_matches_analytic_convergence():
    """A first-order operator must be measured as ~1, a second-order one as ~2."""
    src_target = [hierarchy(n) for n in RESOLUTIONS]
    coarse = MortarProjection(order=1)
    fine = MortarProjection(order=2)
    quad = ANALYTIC_FIELDS[2]
    p1 = coarse.observed_order(quad, src_target)
    p2 = fine.observed_order(quad, src_target)
    assert 0.8 <= p1.order <= 1.2
    assert p2.order >= P_MIN


# ---------------------------------------------------------------------------
# C4: momentum exchange write-back
# ---------------------------------------------------------------------------

def test_c4_me_writeback_momentum(payload):
    c4 = payload["criteria"]["C4"]
    assert max(c4["values"].values()) < TOL_C4
    assert c4["passed"] is True


def test_me_kernel_normalisations():
    nodes = uniform_mesh(0.2, 0.8, 60)
    op = LBMDEMMomentumExchange(epsilon=1.2)
    particles = op.dem_pointset(nodes, particle_lattice(60, 0.2, 0.8))
    lam = op._gather_weights(nodes, particles)
    # per-node normalisation: every node hands out 100 % of its momentum
    assert np.max(np.abs(lam.sum(axis=1) - 1.0)) < 1e-14
    # the exchange volumes sum back to the lattice measure (no measure lost)
    assert abs(float(np.sum(particles.measures)) - nodes.length) < 1e-12 * nodes.length


# ---------------------------------------------------------------------------
# C5: chains
# ---------------------------------------------------------------------------

def test_c5_chains_field_difference(payload):
    c5 = payload["criteria"]["C5"]
    assert c5["per_chain"]["FVM<->LBM"] < TOL_C5
    assert c5["per_chain"]["LBM<->DEM"] < TOL_C5
    assert c5["passed"] is True


def test_chains_are_conservative_per_field():
    for fld in ANALYTIC_FIELDS:
        fl = fvm_lbm_chain(fld, MortarProjection(order=2))
        assert fl["conservation_residual"] < TOL_C3
        ld = lbm_dem_chain(fld, LBMDEMMomentumExchange())
        assert ld["gather_residual"] < TOL_C3
        assert ld["momentum_residual"] < TOL_C4


def test_clustered_dem_packing_is_a_documented_limitation(payload):
    """Strongly clustered packings exceed the 2 % target for steep fields.

    The reason is physical (kernel width vs local particle spacing), so the case
    is *reported* rather than hidden: the gate is only claimed for the resolved
    packing.
    """
    sens = {row["field"]: row["field_diff"]
            for row in payload["chain_sensitivity_clustered_dem"]}
    assert sens["linear"] < TOL_C5
    assert sens["boundary_layer"] > TOL_C5     # honest limitation, not hidden
    assert max(sens.values()) == pytest.approx(sens["boundary_layer"])


# ---------------------------------------------------------------------------
# symmetry_flip
# ---------------------------------------------------------------------------

def test_symmetry_flip_reflection_equivariance(payload):
    for name, rows in payload["operator_table"].items():
        for field, row in rows.items():
            flip = row["symmetry_flip"]
            assert flip["reflection_defect"] < 1e-12, (name, field, flip)
            assert flip["value"] == flip["reflection_defect"]


def test_asymmetric_operator_has_nonzero_symmetry_flip():
    """A one-sided "operator" must be detected as label-dependent."""

    class UpwindShift(MortarProjection):
        name = "broken-upwind-shift"

        def transfer(self, src, src_values, tgt):
            u = np.asarray(src_values, dtype=float)
            shifted = np.empty_like(u)
            shifted[:-1] = u[:-1]
            shifted[-1] = 2.0 * u[-1] - u[-2]      # one-sided extrapolation
            return super().transfer(src, shifted, tgt)

    src, tgt = hierarchy(80)
    u = exact_values(src, ANALYTIC_FIELDS[3])
    op = UpwindShift(order=2)
    flip = op.symmetry_flip(src, u, tgt)
    assert flip.reflection > 1e-6
    assert MortarProjection(order=2).symmetry_flip(src, u, tgt).reflection < 1e-12


# ---------------------------------------------------------------------------
# C6: negative controls (must be red)
# ---------------------------------------------------------------------------

def test_c6_negative_controls_are_red(payload):
    c6 = payload["criteria"]["C6"]
    assert c6["passed"] is True
    for name, detail in c6["values"].items():
        assert detail["red"] is True, (name, detail)
        assert detail["trips"], name


def test_c6_mortar_midpoint_flux_breaks_conservation():
    """Point-sampled flux on a non-matching mesh silently loses mass."""
    bad = MortarProjection(order=2, broken="midpoint")
    good = MortarProjection(order=2)
    src, tgt = hierarchy(160)
    u = exact_values(src, ANALYTIC_FIELDS[4])
    assert bad.conservation_residual(src, u, tgt) > 1e-9
    assert good.conservation_residual(src, u, tgt) < TOL_C3


def test_c6_scatter_without_conservative_fix_loses_mass():
    src, tgt = hierarchy(160)
    u = exact_values(src, ANALYTIC_FIELDS[4])
    bad = ConservativeScatter(broken="no_fix")
    good = ConservativeScatter()
    assert bad.conservation_residual(src, u, tgt) > 1e-6
    assert good.conservation_residual(src, u, tgt) < TOL_C3


def test_c6_me_wrong_spread_normalisation_breaks_momentum_budget():
    nodes = uniform_mesh(0.2, 0.8, 60)
    pos = particle_lattice(60, 0.2, 0.8)
    m = exact_values(nodes, ANALYTIC_FIELDS[4])

    def budget(op):
        particles = op.dem_pointset(nodes, pos)
        v_p = op.gather(nodes, m, particles)
        reac = op.spread(particles, v_p, nodes)
        imp = v_p * particles.measures
        return abs(float(np.sum(reac * nodes.widths) + np.sum(imp))) / \
            float(np.sum(np.abs(imp)))

    assert budget(LBMDEMMomentumExchange()) < TOL_C4
    assert budget(LBMDEMMomentumExchange(broken="per_node_norm")) > 1e-4


def test_negative_control_suite_is_distinct_from_real_suite():
    real = operator_suite()
    bad = negative_control_suite()
    assert set(real) == set(bad)
    for name in real:
        assert "NEGATIVE-CONTROL" not in real[name].name
        assert "NEGATIVE-CONTROL" in bad[name].name


# ---------------------------------------------------------------------------
# acceptance payload integrity
# ---------------------------------------------------------------------------

def test_payload_is_json_serialisable_and_has_all_gates(payload):
    text = json.dumps(payload, default=str)
    assert json.loads(text)["criteria"].keys() == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert payload["digest"]
    assert payload["all_passed"] is True
    for key in ("C1", "C3", "C4", "C5"):
        block = payload["criteria"][key]
        value = block.get("worst_value")
        if value is None:
            value = max(block["values"].values())
        assert math.isfinite(value), key