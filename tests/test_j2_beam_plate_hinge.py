"""Tests for the J2 elastoplastic beam/plate verification surface.

Covers :mod:`tensorphysics.plasticity.beam_hinge`,
:mod:`tensorphysics.plasticity.plate_limit` and the acceptance harness
:mod:`tensorphysics.plasticity.verification_hinge`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.plasticity import verification_hinge as vh
from tensorphysics.plasticity.beam_hinge import (
    FiberSection,
    FiberState,
    J2UniaxialMaterial,
    elastic_deflection_point_load,
    interior_hinge_centroid,
    moment_curvature,
    propped_cantilever_limit,
    rectangle_ideal_plastic_moment,
    section_energy_audit,
    section_response,
    solve_beam,
    unload_from,
    _section_response_batch,
)
from tensorphysics.plasticity.plate_limit import (
    circular_plate_mechanism,
    plastic_moment_per_width,
    rectangular_plate_mechanism,
    square_plate_limit,
    strip_plate_mechanism,
)

E = 210.0e3
SY = 235.0


@pytest.fixture(scope="module")
def sections() -> dict[str, FiberSection]:
    return {
        "rectangle": FiberSection.rectangle(100.0, 200.0),
        "ibeam": FiberSection.ibeam(300.0, 150.0, 15.0, 10.0),
        "circle": FiberSection.circle(100.0),
    }


@pytest.mark.parametrize("key", ["rectangle", "ibeam", "circle"])
def test_fiber_section_matches_closed_form(sections, key):
    section = sections[key]
    fiber = section.fiber_properties()
    analytic = section.analytic()
    for name in ("A", "I", "W_el", "Z"):
        assert fiber[name] == pytest.approx(analytic[name], rel=1e-4)
    assert analytic["shape_factor"] == pytest.approx(analytic["Z"] / analytic["W_el"])


def test_shape_factors_are_classical():
    assert FiberSection.rectangle(1.0, 2.0).analytic()["shape_factor"] == pytest.approx(
        1.5
    )
    assert FiberSection.circle(1.0).analytic()["shape_factor"] == pytest.approx(
        16.0 / (3.0 * math.pi)
    )
    ibeam = FiberSection.ibeam(300.0, 150.0, 15.0, 10.0).analytic()
    assert 1.10 < ibeam["shape_factor"] < 1.20


def test_batch_return_mapping_matches_scalar(sections):
    section = sections["ibeam"]
    material = J2UniaxialMaterial(E=E, sigma_y0=SY, hardening_ratio=1e-3)
    y, _ = section.fibers()
    kappa_y = material.sigma_y0 / (material.E * section.half_height)
    kappas = np.linspace(0.0, 6.0 * kappa_y, 9)
    state = FiberState.virgin(y.size)
    scalar = []
    for k in kappas:
        resp = section_response(section, material, float(k), state, n_sub=4)
        state = resp.state
        scalar.append(resp.moment)
    batch, *_ = _section_response_batch(
        section,
        material,
        kappas,
        np.zeros_like(kappas),
        np.zeros((kappas.size, y.size)),
        np.zeros(kappas.size),
        n_sub=4,
    )
    np.testing.assert_allclose(batch, scalar, rtol=1e-5)


def test_rectangle_closed_form_moment():
    """The fiber model must reproduce the analytic ideal-plastic rectangle."""
    section = FiberSection.rectangle(100.0, 200.0)
    material = J2UniaxialMaterial(E=E, sigma_y0=SY, hardening_ratio=0.0)
    kappa_y = material.sigma_y0 / (material.E * section.half_height)
    curve = moment_curvature(section, material, kappa_max=6.0 * kappa_y, n_steps=60)
    for kappa, moment in zip(curve.kappa[1::5], curve.moment[1::5]):
        ref = rectangle_ideal_plastic_moment(kappa, 100.0, 200.0, E, SY)
        assert moment == pytest.approx(ref, rel=2e-3)


def test_moment_curvature_criteria():
    out = vh.verify_moment_curvature()
    assert out["criteria"]
    for crit in out["criteria"]:
        assert crit["passed"], crit


def test_springback_criteria():
    out = vh.verify_springback()
    for crit in out["criteria"]:
        assert crit["passed"], crit
    # unloading is elastic: no reverse yielding below 6 kappa_y
    assert not any(row["reverse_yielded"] for row in out["rows"])


def test_springback_reduces_with_peak_curvature(sections):
    material = J2UniaxialMaterial(E=E, sigma_y0=SY, hardening_ratio=0.0)
    section = sections["rectangle"]
    kappa_y = material.sigma_y0 / (material.E * section.half_height)
    curve = moment_curvature(section, material, kappa_max=6.0 * kappa_y, n_steps=120)
    ratios = []
    for factor in (2.0, 4.0, 6.0):
        idx = int(np.argmin(np.abs(curve.kappa - factor * kappa_y)))
        sb = unload_from(curve, section, material, peak_index=idx)
        ratios.append(sb.ratio_residual)
    # the plastic part of the curvature grows with the peak curvature, so the
    # residual-to-peak ratio increases monotonically towards 1
    assert ratios[0] < ratios[1] < ratios[2]
    assert 0.2 < ratios[0] < 0.6
    # asymptotic ratio is 1 - shape_factor * kappa_y/kappa_peak -> 0.75 at 6 kappa_y
    assert 0.6 < ratios[2] < 0.9


def test_energy_criteria():
    out = vh.verify_energy(fast=True)
    for crit in out["criteria"]:
        assert crit["passed"], crit


def test_plate_limit_classical_anchors():
    M_p = plastic_moment_per_width(SY, 10.0)
    b = 1000.0
    assert square_plate_limit(b, M_p) == pytest.approx(24.0 * M_p / b**2, rel=1e-9)
    circ = circular_plate_mechanism(b, M_p)
    assert circ.pressure == pytest.approx(6.0 * M_p / b**2, rel=1e-4)
    strip = strip_plate_mechanism(4.0 * b, b, M_p)
    assert strip.pressure == pytest.approx(8.0 * M_p / b**2, rel=1e-9)
    rect = rectangular_plate_mechanism(2.0 * b, b, M_p)
    # upper bound must exceed the (longer) strip limit and shrink towards it
    assert strip.pressure < rect.pressure < square_plate_limit(b, M_p)
    assert rect.pressure == pytest.approx(15.0 * M_p / b**2, rel=1e-9)


def test_limit_analysis_is_statically_consistent():
    """The limit load and hinge position must satisfy statics."""
    M_p = 1.0
    length = 1.0
    lim = propped_cantilever_limit(M_p, length)
    q = lim["q_collapse"]
    t = lim["hinge_position_ratio"]
    # reaction at the fixed end from statics
    R_A = 0.5 * q * length + M_p / length
    assert lim["support_reaction_ratio"] == pytest.approx(R_A / (q * length))
    # moment at the sagging hinge equals M_p
    x1 = t * length
    M_x1 = -M_p + R_A * x1 - 0.5 * q * x1**2
    assert M_x1 == pytest.approx(M_p, rel=1e-9)
    # shear vanishes there
    assert R_A - q * x1 == pytest.approx(0.0, abs=1e-9)
    assert t == pytest.approx(0.5857864376269049, rel=1e-12)
    assert lim["load_factor"] == pytest.approx(11.65685424949238, rel=1e-12)


def test_beam_fe_elastic_deflection(sections):
    """Displacement-based FE must reproduce the elastic beam solution."""
    section = sections["ibeam"]
    material = J2UniaxialMaterial(E=E, sigma_y0=SY, hardening_ratio=1e-3)
    EI = material.E * section.fiber_properties()["I"]
    length = 6000.0
    q = 1.0
    res = solve_beam(
        section, material, length, 40, "simply_supported", uniform_load=q, n_steps=1
    )
    assert np.abs(res.deflection).max() == pytest.approx(
        5.0 * q * length**4 / (384.0 * EI), rel=1e-6
    )
    # simply supported with an eccentric point load
    P = 1.0e4
    a = 0.3 * length
    res2 = solve_beam(
        section, material, length, 50, "simply_supported", nodal_loads={15: P}, n_steps=1
    )
    node = int(round(a / (length / 50)))
    ref = elastic_deflection_point_load(
        P, a, length - a, length, material.E, section.fiber_properties()["I"]
    )
    assert abs(res2.deflection[node]) == pytest.approx(ref, rel=2e-3)


def test_propped_cantilever_hinge_location():
    out = vh.verify_hinge_location(fast=True)
    names = {c["name"]: c for c in out["criteria"]}
    assert names["b_hinge_propped_cantilever"]["passed"]
    assert names["b_hinge_simply_supported"]["passed"]
    pc = out["cases"]["propped_cantilever_udl"]
    assert pc["converged"]
    # the fixed-end hinge forms first, the sagging hinge appears near collapse
    assert len(pc["all_peaks_ratio"]) >= 2


def test_interior_hinge_centroid_recovers_location():
    x = np.linspace(0.0, 1.0, 60)
    kappa_p = np.exp(-((x - 0.4) ** 2) / (2 * 0.01**2))
    est = interior_hinge_centroid(x, kappa_p, 1.0)
    assert est == pytest.approx(0.4, abs=5e-3)
    # a boundary peak must be ignored when an interior hinge exists
    kappa_p2 = kappa_p.copy()
    kappa_p2[0] = 10.0
    assert interior_hinge_centroid(x, kappa_p2, 1.0) == pytest.approx(0.4, abs=5e-3)


def test_negative_controls_fail():
    nc = vh.negative_controls()
    assert len(nc["criteria"]) >= 2
    for crit in nc["criteria"]:
        assert crit["expected"] == "fail"
        assert not crit["passed"], crit
        assert crit["control_ok"]


def test_full_acceptance_report():
    report = vh.run_all_criteria(fast=True)
    assert report["failed"] == []
    assert report["negative_controls_ok"]
    assert report["all_passed"]
    # every requested criterion family is present
    names = [c["name"] for c in report["criteria"]]
    for prefix in ("a_", "b_", "c_", "d_", "p_"):
        assert any(n.startswith(prefix) for n in names)
    assert len(report["momcurv"]) == 3
    assert len(report["springback"]) == 9