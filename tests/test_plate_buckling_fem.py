"""Verification tests for the plate-buckling finite element kernels.

These tests cover the *linear* mechanics: geometry gates, bending and membrane
element accuracy, the elastoplastic return mapping, element rank sufficiency,
and the linearised buckling eigenvalue for four boundary-condition
combinations against an independent Rayleigh-Ritz reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.plate_buckling import (
    BilinearMaterial,
    PlateGeometry,
    PlateModel,
    element_kernels,
    navier_simply_supported_deflection,
    plane_stress_return_map,
)
from tensorphysics.plate_buckling.benchmarks import (
    euler_buckling_stress,
    ritz_buckling_stress,
)


# --------------------------------------------------------------------------
# geometry gates
# --------------------------------------------------------------------------
def test_geometry_validation_accepts_design_envelope():
    geom = PlateGeometry(a=3.0, b=1.0, t=0.0195, sigma_y=313e6, nx=12, ny=4)
    checks = geom.validate()
    assert checks["a_over_b"] == pytest.approx(3.0)
    assert 1.0 <= checks["beta"] <= 3.0
    assert checks["element_aspect"] <= 3.0


@pytest.mark.parametrize("kwargs", [
    {"a": 1.0, "b": 1.0, "t": 0.02},      # aspect ratio 1 -> outside [2, 6]
    {"a": 3.0, "b": 1.0, "t": 0.005},     # beta >> 3
    {"a": 3.0, "b": 1.0, "t": 0.06},      # beta << 1
])
def test_geometry_validation_rejects_out_of_range(kwargs):
    geom = PlateGeometry(sigma_y=313e6, nx=12, ny=4, **kwargs)
    with pytest.raises(Exception):
        geom.validate()


def test_mesh_element_aspect_gate():
    geom = PlateGeometry(a=4.0, b=1.0, t=0.02, sigma_y=313e6, nx=8, ny=8)
    with pytest.raises(Exception):
        geom.validate()   # elements are 0.5 x 0.125 -> aspect 4 > 3


# --------------------------------------------------------------------------
# element / element-rank checks
# --------------------------------------------------------------------------
def test_bending_element_rank_and_rigid_modes():
    """The bicubic Hermite rectangle must have exactly 3 rigid-body modes."""
    t = 0.01
    ex = ey = 0.1
    kern = element_kernels(ex, ey)
    mat = BilinearMaterial(E=2.06e11, nu=0.3)
    Db = mat.elastic_matrix * t**3 / 12.0
    Kb = np.zeros((16, 16))
    for g in range(kern["Bm"].shape[0]):
        B = kern["Bb"][g]
        Kb += kern["weight"][g] * (B.T @ Db @ B)
    eig = np.linalg.eigvalsh(Kb)
    # exactly three zero-energy (rigid body) modes; the twist dofs additionally
    # carry near-zero energy (|lambda| ~ 1e-8 of the bending scale), which is
    # why the assembled tangent is stabilised on the w_xy diagonal
    assert np.sum(np.abs(eig) < 1e-12 * eig[-1]) == 3
    soft = np.sum(np.abs(eig) < 1e-6 * eig[-1])
    assert soft >= 3
    # a linear field carries no curvature energy
    d = np.zeros(16)
    for c, (x, y) in enumerate([(0, 0), (ex, 0), (ex, ey), (0, ey)]):
        d[4 * c + 0] = x + 2.0 * y          # w
        d[4 * c + 1] = 2.0                  # theta_x = w,y
        d[4 * c + 2] = -1.0                 # theta_y = -w,x
    assert abs(0.5 * d @ Kb @ d) < 1e-12 * abs(eig[-1])


def test_bending_stiffness_matches_beam_theory():
    """Cylindrical bending of a strip must reproduce 12EI/L^3 exactly."""
    ex, ey, t = 0.1, 0.05, 0.01
    kern = element_kernels(ex, ey)
    mat = BilinearMaterial(E=2.06e11, nu=0.0)
    Db = mat.elastic_matrix * t**3 / 12.0
    Kb = np.zeros((16, 16))
    for g in range(kern["Bm"].shape[0]):
        B = kern["Bb"][g]
        Kb += kern["weight"][g] * (B.T @ Db @ B)
    red = np.zeros((8, 16))
    for k in range(4):
        red[k, k] = 1.0
        red[k, 12 + k] = 1.0
        red[4 + k, 4 + k] = 1.0
        red[4 + k, 8 + k] = 1.0
    K8 = red @ Kb @ red.T
    dofs = [0, 2, 4, 6]
    S = np.diag([1.0, -1.0, 1.0, -1.0])
    Kb_beam = S @ K8[np.ix_(dofs, dofs)] @ S
    EI = Db[0, 0] * ey
    L = ex
    k11 = 12.0 * EI / L**3
    assert Kb_beam[0, 0] == pytest.approx(k11, rel=1e-10)
    assert Kb_beam[1, 1] == pytest.approx(4.0 * EI / L, rel=1e-10)


# --------------------------------------------------------------------------
# linear bending and membrane patch tests
# --------------------------------------------------------------------------
def test_simply_supported_deflection_matches_navier():
    geom = PlateGeometry(a=1.0, b=1.0, t=0.01, sigma_y=9e9, nx=16, ny=16)
    model = PlateModel(geom, "SSSS", validate=False)
    q = 1.0e4
    u = model.solve_linear(model.pressure_load(q))
    w_max = u[6 * np.arange(geom.n_nodes) + 2].max()
    ref = navier_simply_supported_deflection(geom, q)
    assert abs(w_max / ref - 1.0) < 0.002


def test_clamped_plate_deflection_coefficient():
    geom = PlateGeometry(a=1.0, b=1.0, t=0.01, sigma_y=9e9, nx=16, ny=16)
    model = PlateModel(geom, "CCCC", validate=False)
    u = model.solve_linear(model.pressure_load(1.0e4))
    coeff = u[6 * np.arange(geom.n_nodes) + 2].max() * geom.flexural_rigidity / 1e4
    assert coeff == pytest.approx(0.00126, rel=0.02)


def test_membrane_patch_uniform_compression():
    geom = PlateGeometry(a=3.0, b=1.0, t=0.02, sigma_y=313e6, nx=12, ny=4)
    model = PlateModel(geom, "SSSS")
    sigma = 1.0e7
    u = model.solve_linear(model.compression_load(sigma))
    N = model.membrane_resultants(u)
    assert N[:, :, 0].mean() == pytest.approx(-sigma * geom.t, rel=1e-8)
    assert abs(N[:, :, 1]).max() < 1e-3 * abs(N[:, :, 0]).max()


# --------------------------------------------------------------------------
# plasticity
# --------------------------------------------------------------------------
def test_return_map_elastic_below_yield():
    mat = BilinearMaterial(E=2.06e11, nu=0.3, sigma_y=313e6, E_t=0.0)
    eps = np.array([[1.0e-4, -3.0e-5, 0.0]])
    sig, C, eps_p, alpha = plane_stress_return_map(
        eps, np.zeros((1, 3)), np.zeros(1), mat)
    assert alpha[0] == 0.0
    assert np.allclose(sig, eps @ mat.elastic_matrix.T)


def test_return_map_consistency_and_hardening():
    mat = BilinearMaterial(E=2.06e11, nu=0.3, sigma_y=313e6, E_t=2.06e9)
    eps = np.array([[6.0e-3, 0.0, 0.0]])
    sig, C, eps_p, alpha = plane_stress_return_map(
        eps, np.zeros((1, 3)), np.zeros(1), mat)
    mean = (sig[0, 0] + sig[0, 1]) / 3.0
    q = np.sqrt(1.5 * ((sig[0, 0] - mean) ** 2 + (sig[0, 1] - mean) ** 2
                       + mean**2 + 2 * sig[0, 2] ** 2))
    assert alpha[0] > 0.0
    assert q == pytest.approx(mat.sigma_y + mat.plastic_modulus * alpha[0], rel=1e-5)
    assert np.allclose(C[0], C[0].T)


# --------------------------------------------------------------------------
# linear buckling: four boundary-condition combinations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bc", ["SSSS", "CCCC", "SCSC", "CSCS"])
def test_linear_buckling_against_ritz_reference(bc):
    geom = PlateGeometry(a=4.0, b=1.0, t=0.02, sigma_y=313e6, nx=24, ny=6)
    model = PlateModel(geom, bc)
    fem = model.linear_buckling(1.0)
    k_fem = fem.sigma_cr * geom.t * geom.b**2 / (np.pi**2 * geom.flexural_rigidity)
    k_ritz = ritz_buckling_stress(geom, bc, n_terms=8)["k"]
    assert abs(k_fem / k_ritz - 1.0) < 0.02


def test_euler_k_factor_simply_supported():
    geom = PlateGeometry(a=4.0, b=1.0, t=0.02, sigma_y=313e6, nx=24, ny=6)
    model = PlateModel(geom, "SSSS")
    fem = model.linear_buckling(1.0)
    ref = euler_buckling_stress(geom, "SSSS")
    assert abs(fem.sigma_cr / ref["sigma_cr"] - 1.0) < 0.01
    assert ref["k"] == pytest.approx(4.0)


def test_slenderness_controls_critical_stress():
    """sigma_cr must scale as (t/b)^2, i.e. as 1/beta^2."""
    ratios = []
    for beta, t in ((1.5, 0.03898 / 1.5), (3.0, 0.03898 / 3.0)):
        geom = PlateGeometry(a=4.0, b=1.0, t=t, sigma_y=313e6, nx=16, ny=4)
        model = PlateModel(geom, "SSSS")
        ratios.append(model.linear_buckling(1.0).sigma_cr * geom.beta**2)
    assert ratios[0] == pytest.approx(ratios[1], rel=2e-3)