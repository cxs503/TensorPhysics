"""Degenerate limits of the thermo-mechanical chain (pure-elastic recovery)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.thermo_mechanical import (
    bimetal_beam_fe,
    bimetal_q8_fe,
    bimetal_section,
    hole_fe_sweep,
    timoshenko_curvature,
)

H = 1.0e-3
E = 200.0e9
LENGTH = 0.10


def test_zero_delta_T_recovers_pure_elastic_cantilever():
    """dT -> 0 with a tip load: FE must reproduce delta = P L^3 / (3 EI) (<=1%)."""
    P = 1.0
    sec = bimetal_section(H, H, E, E, 12e-6, 6e-6, width=1.0, dT=0.0)
    fe = bimetal_beam_fe(H, H, E, E, 12e-6, 6e-6, 0.0, length=LENGTH,
                         n_elems=16, tip_load=P)
    ref = P * LENGTH ** 3 / (3.0 * sec.EI)
    assert abs(fe["tip_deflection"] - ref) / ref <= 0.01
    # no thermal load -> curvature is purely mechanical: kappa = -M/EI (sign convention)
    kappa_root = -P * (LENGTH - 0.5 * LENGTH / 16) / sec.EI   # element-average moment
    assert abs(fe["kappa"][0] - kappa_root) / abs(kappa_root) < 0.01


def test_symmetric_alpha_gives_zero_curvature():
    """Equal expansion coefficients -> no thermal curvature (both layers equal)."""
    kappa = timoshenko_curvature(H, H, E, E, 8e-6, 8e-6, 100.0)
    assert abs(kappa) < 1e-18
    fe = bimetal_beam_fe(H, H, E, E, 8e-6, 8e-6, 100.0, length=LENGTH, n_elems=8)
    assert abs(fe["kappa_mean"]) < 1e-12
    assert abs(fe["tip_deflection"]) < 1e-12


def test_zero_alpha_gives_zero_stress():
    """Uniform heating with alpha = 0 produces no stress in the 2-D continuum."""
    res = bimetal_q8_fe(H, H, E, E, 0.0, 0.0, 100.0, length=LENGTH,
                        nx=10, ny1=1, ny2=1)
    scale = E * 12e-6 * 100.0
    assert np.max(np.abs(res["sigma"])) / scale < 1e-9
    assert abs(res["kappa"]) * LENGTH < 1e-9


def test_zero_temperature_gradient_gives_zero_hole_stress():
    """Annulus with Ta = Tb: no thermal load, K_t = 0 and sigma -> 0."""
    fe = hole_fe_sweep(0.01, 0.2, 50.0, 50.0, E, 12e-6, nu=0.3, n_elems=64)
    assert abs(fe["K_t_fe"]) < 1e-9
    assert np.max(np.abs(fe["sigma_theta_fe"])) < 1e3     # << E*alpha*dT = 2.4e8


def test_equal_alpha_gives_zero_curvature_in_2d():
    """dT uniform and alpha1 = alpha2: no strain gradient (curvature) in 2-D."""
    res = bimetal_q8_fe(H, H, E, E, 12e-6, 12e-6, 100.0, length=LENGTH,
                        nx=20, ny1=2, ny2=2)
    assert abs(res["kappa"]) * LENGTH < 1e-6
    assert abs(res["tip_deflection"]) < 1e-12