"""Mesh-convergence and thermal-BC verification of the thermo-mechanical chain."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.thermo_mechanical import (
    bimetal_q8_fe,
    heat1d_steady,
    hole_fe_sweep,
    q4_steady_conduction,
    timoshenko_curvature,
)

H1 = H2 = 1.0e-3
E = 200.0e9
ALPHA2 = 6.0e-6
DT = 100.0
LENGTH = 0.10


def test_bimetal_curvature_second_order_convergence():
    """Q8 bimetal curvature error converges at O(h^2) under 2x refinement."""
    alpha1 = 2 * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E, E, alpha1, ALPHA2, DT)
    errs = []
    for nx, ny in ((10, 1), (20, 2), (40, 4)):
        res = bimetal_q8_fe(H1, H2, E, E, alpha1, ALPHA2, DT, length=LENGTH,
                            nx=nx, ny1=ny, ny2=ny)
        errs.append(abs(res["kappa"] - kappa_ref) / kappa_ref)
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(r > 1.6 for r in rates), (errs, rates)
    assert errs[-1] < 5.0e-3


def test_hole_kt_second_order_convergence():
    """Ring-element K_t error converges at ~O(h^2)."""
    a, b, Ta, Tb = 0.01, 0.2, 100.0, 0.0
    errs = []
    for ne in (64, 128, 256):
        fe = hole_fe_sweep(a, b, Ta, Tb, E, 12e-6, nu=0.3, n_elems=ne)
        errs.append(fe["rel_error_Kt"])
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(r > 1.5 for r in rates), (errs, rates)
    assert errs[-1] < 0.01


def test_bilinear_q4_is_locking_limited():
    """Documented limitation: Q4 in-plane bending converges below second order."""
    from tensorphysics.thermo_mechanical import bimetal_q4_fe

    alpha1 = 2 * ALPHA2
    kappa_ref = timoshenko_curvature(H1, H2, E, E, alpha1, ALPHA2, DT)
    errs = []
    for nx, ny in ((25, 2), (50, 4)):
        res = bimetal_q4_fe(H1, H2, E, E, alpha1, ALPHA2, DT, length=LENGTH,
                            nx=nx, ny1=ny, ny2=ny)
        errs.append(abs(res["kappa"] - kappa_ref) / kappa_ref)
    rate = np.log2(errs[0] / errs[1])
    assert rate < 1.5            # honest measurement: Q4 locks, Q8 does not


# ---------------------------------------------------------------- thermal BCs
def test_steady_plane_wall_dirichlet():
    """Plane wall: linear temperature field and q = k dT/L exactly."""
    x = np.linspace(0.0, 0.1, 21)
    sol = heat1d_steady(x, k=50.0, bc_left={"type": "dirichlet", "value": 100.0},
                        bc_right={"type": "dirichlet", "value": 0.0})
    assert np.allclose(sol["T"], 100.0 * (1.0 - x / 0.1), atol=1e-10)
    assert np.allclose(sol["flux"], 50.0 * 100.0 / 0.1, rtol=1e-10)


def test_convection_robin_boundary():
    """Convection + conduction: interface temperature from the series resistance."""
    x = np.linspace(0.0, 0.1, 41)
    sol = heat1d_steady(x, k=50.0, bc_left={"type": "dirichlet", "value": 100.0},
                        bc_right={"type": "convection", "h": 20.0, "T_inf": 20.0})
    R_tot = 1.0 / 20.0 + 0.1 / 50.0               # 1/h + L/k  [m^2 K/W]
    q = (100.0 - 20.0) / R_tot                    # W/m^2
    assert sol["flux"][0] == pytest.approx(q, rel=1e-8)
    assert sol["T"][-1] == pytest.approx(20.0 + q / 20.0, rel=1e-10)


def test_insulated_neumann_boundary():
    """Zero prescribed flux at one end: adiabatic wall, zero gradient."""
    x = np.linspace(0.0, 0.1, 11)
    sol = heat1d_steady(x, k=50.0, bc_left={"type": "dirichlet", "value": 100.0},
                        bc_right={"type": "flux", "value": 0.0})
    assert np.allclose(sol["T"], 100.0, atol=1e-9)


def test_2d_q4_manufactured_solution_second_order():
    """Q4 conduction vs the harmonic field T = exp(x) cos(y): O(h^2) convergence."""
    def solve(n):
        xs = np.linspace(0.0, 1.0, n + 1)
        nodes = np.array([[x, y] for y in xs for x in xs])
        elems = []
        for j in range(n):
            for i in range(n):
                a = j * (n + 1) + i
                elems.append([a, a + 1, a + 1 + (n + 1), a + (n + 1)])
        exact = np.exp(nodes[:, 0]) * np.cos(nodes[:, 1])
        border = (np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1)
                  | np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1))
        fixed = {i: exact[i] for i in np.where(border)[0]}
        sol = q4_steady_conduction(nodes, np.asarray(elems), k=1.0,
                                   dirichlet=fixed, source=0.0)
        return np.max(np.abs(sol["T"] - exact))

    e1, e2 = solve(4), solve(8)
    assert e2 < e1
    assert np.log2(e1 / e2) > 1.5
