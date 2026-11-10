"""Bimetallic-strip thermo-mechanical verification (Timoshenko 1925 anchor)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.thermo_mechanical import (
    arc_end_deflection,
    bimetal_beam_fe,
    bimetal_q8_fe,
    exact_composite_solution,
    timoshenko_curvature,
    timoshenko_end_deflection,
)

H1 = 1.0e-3
H2 = 1.0e-3
E1 = 200.0e9
E2 = 200.0e9
ALPHA2 = 6.0e-6
DT = 100.0
LENGTH = 0.10
RATIOS = (2.0, 3.0, 5.0)


@pytest.mark.parametrize("ratio", RATIOS)
def test_timoshenko_curvature_matches_layered_section(ratio):
    """Closed-form 1925 formula == independent modular-ratio section solution."""
    a1 = ratio * ALPHA2
    kappa_closed = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
    kappa_section = exact_composite_solution(H1, H2, E1, E2, a1, ALPHA2, DT)["kappa"]
    assert kappa_closed > 0.0
    assert abs(kappa_section - kappa_closed) / kappa_closed < 1.0e-9


@pytest.mark.parametrize("ratio", RATIOS)
def test_beam_fe_curvature_three_alpha_ratios(ratio):
    """Composite frame element reproduces the analytic curvature (<=3%)."""
    a1 = ratio * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
    fe = bimetal_beam_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH, n_elems=8)
    assert abs(fe["kappa_mean"] - kappa_ref) / kappa_ref <= 0.03
    assert np.std(fe["kappa"]) / abs(fe["kappa_mean"]) < 1.0e-9  # uniform curvature


@pytest.mark.parametrize("ratio", RATIOS)
def test_beam_fe_tip_deflection_three_alpha_ratios(ratio):
    """Free-end deflection vs kappa*L^2/2 within 3% (sign: towards low-alpha side)."""
    a1 = ratio * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
    fe = bimetal_beam_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH, n_elems=8)
    ref = timoshenko_end_deflection(kappa_ref, LENGTH)
    assert abs(abs(fe["tip_deflection"]) - ref) / ref <= 0.03
    assert fe["tip_deflection"] < 0.0        # high-alpha layer on top -> bends down
    assert fe["kappa_mean"] > 0.0


def test_end_deflection_small_rotation_vs_elastica():
    """kappa*L^2/2 vs the exact arc: <0.2% for the acceptance case."""
    kappa = timoshenko_curvature(H1, H2, E1, E2, 2 * ALPHA2, ALPHA2, DT)
    small = timoshenko_end_deflection(kappa, LENGTH)
    arc = arc_end_deflection(kappa, LENGTH)
    assert abs(small - arc) / arc < 2.0e-3


def test_unequal_thickness_and_modulus_against_section():
    """Timoshenko formula also matches the section solution for m=2, n=2."""
    h1, h2 = 1.5e-3, 0.75e-3
    a1, a2 = 18.0e-6, 4.0e-6
    kappa_closed = timoshenko_curvature(h1, h2, E1, E2, a1, a2, DT)
    kappa_section = exact_composite_solution(h1, h2, E1, E2, a1, a2, DT)["kappa"]
    assert abs(kappa_section - kappa_closed) / kappa_closed < 1.0e-9


def test_q8_2d_curvature_within_3pct():
    """2-D continuum (Q8) curvature vs Timoshenko analytical anchor (<=3%)."""
    a1 = 2 * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
    fe2d = bimetal_q8_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH,
                         nx=25, ny1=2, ny2=2)
    assert abs(fe2d["kappa"] - kappa_ref) / kappa_ref <= 0.03


def test_q8_2d_tip_deflection_within_3pct():
    """2-D continuum (Q8) tip deflection vs analytical anchor (<=3%)."""
    a1 = 3 * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E1, E2, a1, ALPHA2, DT)
    fe2d = bimetal_q8_fe(H1, H2, E1, E2, a1, ALPHA2, DT, length=LENGTH,
                         nx=25, ny1=2, ny2=2)
    ref = timoshenko_end_deflection(kappa_ref, LENGTH)
    assert abs(abs(fe2d["tip_deflection"]) - ref) / ref <= 0.03