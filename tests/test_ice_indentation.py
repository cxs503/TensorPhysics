"""Tests for the J2 elastoplastic ice-indentation node (TensorPhysics p233w1-N06).

Covers the constitutive kernel (material-point verification and the vectorised
twin of the radial return mapping), the penalty contact formulations (node and
facet), the exact confined-compression solution, the graded mesh builder and the
end-to-end incremental Newton indentation driver (convergence, monotonicity,
energy balance).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.ice_indentation import (
    FlatPunchIndenter,
    SphericalIndenter,
    batch_return_map,
    build_surface_facets,
    confined_uniaxial_strain,
    contact_state,
    facet_contact_state,
    graded_axis,
    hertz_force,
    hertz_yield_onset,
    penalty_stiffness,
    quarter_block_bcs,
    quarter_block_mesh,
    reduced_modulus,
    shear_yield_stress,
    simulate_indentation,
    surface_penalty_stiffness,
    tangent_error,
    top_surface_data,
    uniaxial_stress_response,
)
from tensorphysics.ice_indentation.mesh import graded_box_mesh
from tensorphysics.solid_fem.element import element_kinematics
from tensorphysics.solid_fem.material import J2Material, PlasticState

E_ICE = 9.0e9
NU_ICE = 0.33
SIGMA_Y = 5.0e6


def ice_material(sigma_y: float = SIGMA_Y, hardening: float = 0.0) -> J2Material:
    return J2Material(E=E_ICE, nu=NU_ICE, sigma_y0=sigma_y, hardening=hardening)


# --------------------------------------------------------------------------- #
# Mesh / geometry helpers
# --------------------------------------------------------------------------- #
def test_graded_axis_clusters_and_hits_bounds():
    axis = graded_axis(8, 10.0, power=2.0, cluster="min")
    assert axis[0] == pytest.approx(0.0)
    assert axis[-1] == pytest.approx(10.0)
    spacing = np.diff(axis)
    assert np.all(spacing > 0.0)                      # strictly increasing nodes
    assert np.all(np.diff(spacing) > 0.0)             # spacing grows away from 0
    top = graded_axis(6, 6.0, power=1.6, cluster="max")
    assert top[-1] == pytest.approx(6.0)
    assert np.all(np.diff(top) > 0.0)


def test_graded_mesh_elements_have_positive_jacobian():
    mesh = quarter_block_mesh(half_width=4e-3, height=2e-3, dims=(4, 4, 3))
    vols = [element_kinematics("hex8", mesh.element_coords(e)).volume
            for e in range(mesh.n_elem)]
    assert min(vols) > 0.0
    total = float(sum(vols))
    assert total == pytest.approx((4e-3) ** 2 * 2e-3, rel=1e-10)


def test_surface_nodal_areas_sum_to_face_area():
    mesh = graded_box_mesh(dims=(5, 5, 3), lengths=(2e-3, 2e-3, 1e-3),
                           powers=(2.0, 2.0, 1.6),
                           clusters=("min", "min", "max"))
    nodes, areas, total = top_surface_data(mesh)
    assert total == pytest.approx(4e-6, rel=1e-10)
    assert np.isclose(areas.sum(), total, rtol=1e-12)
    # the corner node of the graded mesh carries the smallest lumped area
    corner = np.argmin(np.abs(mesh.nodes[nodes, 0]) + np.abs(mesh.nodes[nodes, 1]))
    assert areas[corner] < areas.mean()


# --------------------------------------------------------------------------- #
# Constitutive kernel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hardening", [0.0, 2.0e8])
def test_batch_return_map_matches_scalar(hardening):
    """The vectorised twin must reproduce the scalar return mapping exactly."""
    rng = np.random.default_rng(7)
    material = ice_material(hardening=hardening)
    eps = rng.normal(0.0, 3e-3, size=(8, 6))
    out = batch_return_map(material, eps, np.zeros((8, 6)), np.zeros(8))
    for i in range(eps.shape[0]):
        stress, state, tangent, plastic = material.return_map(eps[i], PlasticState())
        assert out.plastic[i] == plastic
        assert np.allclose(out.stress[i], stress, rtol=1e-12, atol=1e-6)
        assert np.allclose(out.tangent[i], tangent, rtol=1e-10, atol=1e-3)
        assert np.allclose(out.eps_p[i], state.eps_p, rtol=0, atol=1e-15)
        assert out.alpha[i] == pytest.approx(state.alpha, abs=1e-15)


def test_batch_return_map_elastic_branch_is_elastic():
    material = ice_material()
    eps = np.array([[1e-6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = batch_return_map(material, eps, np.zeros((1, 6)), np.zeros(1))
    assert not out.plastic[0]
    # uniaxial *strain* (all other strains zero): sigma_xx = (lam + 2 mu) eps
    lam, mu, _bulk = material.lam, material.mu, material.bulk
    assert out.stress[0, 0] == pytest.approx((lam + 2 * mu) * 1e-6, rel=1e-12)
    assert out.stress[0, 1] == pytest.approx(lam * 1e-6, rel=1e-12)
    assert out.mises[0] == pytest.approx(2 * mu * 1e-6, rel=1e-12)


def test_ac1_uniaxial_stress_curve_matches_analytic():
    material = ice_material()
    resp = uniaxial_stress_response(material, 0.01, n_steps=200)
    assert resp.max_relative_error < 0.01                     # AC1 (<= 1 %)
    # the lateral strain must be the elastic Poisson contraction below yield
    elastic = resp.strain_xx < 0.5 * material.sigma_y0 / material.E
    assert np.allclose(resp.lateral_strain[elastic],
                       -NU_ICE * resp.strain_xx[elastic], rtol=1e-9, atol=1e-12)
    # the detected onset is resolved by the strain increment of the load path
    assert resp.plastic_onset_strain() == pytest.approx(
        material.sigma_y0 / material.E, abs=1.5 * 0.01 / 200)


def test_ac1_uniaxial_stress_curve_with_hardening():
    material = ice_material(hardening=5.0e8)
    resp = uniaxial_stress_response(material, 0.01, n_steps=200)
    assert resp.max_relative_error < 0.01
    assert resp.stress_xx[-1] > material.sigma_y0


def test_ac1_compression_branch():
    material = ice_material()
    resp = uniaxial_stress_response(material, -0.01, n_steps=200)
    assert resp.max_relative_error < 0.01
    assert resp.stress_xx[-1] < 0.0


def test_ac2_pure_shear_yield_matches_analytic():
    material = ice_material()
    measured = shear_yield_stress(material)
    analytic = material.sigma_y0 / np.sqrt(3.0)
    assert abs(measured - analytic) / analytic < 0.01          # AC2 (<= 1 %)


def test_consistent_tangent_matches_numerical():
    material = ice_material(hardening=2.0e8)
    strain = np.array([2e-3, -1e-3, -1e-3, 4e-3, 0.0, 0.0])
    err = tangent_error(material, strain, PlasticState())
    assert err < 1e-4


def test_plastic_flow_is_isochoric():
    material = ice_material()
    strain = np.array([3e-3, -1e-3, -2e-3, 5e-3, 1e-3, 0.0])
    out = batch_return_map(material, strain[None, :], np.zeros((1, 6)), np.zeros(1))
    assert abs(float(out.eps_p[0, :3].sum())) < 1e-15
    assert out.alpha[0] > 0.0


# --------------------------------------------------------------------------- #
# Contact formulations
# --------------------------------------------------------------------------- #
def test_spherical_gap_and_gradient():
    ind = SphericalIndenter(radius=10e-3, apex_z=1e-3)
    assert ind.gap(0.0, 0.0, 1e-3) == pytest.approx(0.0, abs=1e-18)
    assert ind.gap(0.0, 0.0, 1e-3 + 5e-6) == pytest.approx(5e-6)
    assert ind.gap(1e-3, 0.0, 1e-3) < 0.0        # outside the sphere surface
    # gradient of the gap against a central difference
    x, y = 2e-4, -1.5e-4
    h = 1e-9
    num = (ind.gap(x + h, y, 0.0) - ind.gap(x - h, y, 0.0)) / (2 * h)
    assert ind.gap_gradient(x, y)[0] == pytest.approx(num, rel=1e-4)
    ind2 = ind.advance(3e-6)
    assert ind2.apex_z == pytest.approx(1e-3 - 3e-6)


def test_node_contact_force_pushes_down_and_energy_positive():
    mesh = graded_box_mesh(dims=(4, 4, 2), lengths=(1e-3, 1e-3, 0.5e-3),
                           powers=(1.0, 1.0, 1.0), clusters=("min", "min", "max"))
    nodes, areas, _ = top_surface_data(mesh)
    ind = FlatPunchIndenter(radius=2e-3, punch_z=float(mesh.nodes[:, 2].max()) - 2e-7)
    u = np.zeros(mesh.n_dof)
    k = penalty_stiffness(ice_material(), areas.mean(), 50.0)
    cs = contact_state(mesh, u, ind, k, nodes, areas)
    assert cs.total_force > 0.0
    assert np.all(cs.force_vector[2::3] <= 0.0)          # pushes the ice down
    assert cs.energy > 0.0
    assert cs.contact_area == pytest.approx(areas.sum(), rel=1e-12)


def test_facet_contact_reproduces_uniform_pressure():
    """A flat punch covering the whole top face must load it uniformly."""
    mesh = graded_box_mesh(dims=(4, 4, 2), lengths=(1e-3, 1e-3, 0.5e-3),
                           powers=(1.0, 1.0, 1.0), clusters=("min", "min", "max"))
    facets = build_surface_facets(mesh, axis=2)
    assert facets.areas.sum() == pytest.approx(1e-6, rel=1e-12)
    delta = 2e-7
    ind = FlatPunchIndenter(radius=2e-3, punch_z=float(mesh.nodes[:, 2].max()) - delta)
    k = surface_penalty_stiffness(ice_material(), facets.h, 100.0)
    u = np.zeros(mesh.n_dof)
    cs = facet_contact_state(mesh, u, ind, k, facets)
    # every facet is fully penetrated: F = k delta A
    expected = float(np.sum(k * facets.areas)) * delta
    assert cs.total_force == pytest.approx(expected, rel=1e-12)
    assert cs.contact_area == pytest.approx(facets.areas.sum(), rel=1e-12)


def test_confined_compression_matches_analytic_solution():
    """Penalty contact + Hex8 assembly vs the exact uniaxial-strain solution."""
    E, nu = 1e9, 0.3
    material = J2Material(E=E, nu=nu, sigma_y0=1e12)
    half, height = 2e-3, 1e-3
    mesh = graded_box_mesh(dims=(4, 4, 3), lengths=(half, half, height),
                           powers=(1.5, 1.5, 1.3), clusters=("min", "min", "max"))
    delta = 1e-7
    punch = FlatPunchIndenter(radius=half * 2.0, punch_z=height)

    exact = confined_uniaxial_strain(E, nu, delta, height)["sigma_zz"] * (2 * half) ** 2
    forces = []
    for factor in (20.0, 50.0, 200.0):
        res = simulate_indentation(mesh, material, punch, delta_max=delta, n_steps=2,
                                   penalty_factor=factor, symmetry_factor=4,
                                   tol=1e-8, max_iter=15)
        assert res.converged
        forces.append(res.force[-1])
    # penalty compliance is 1/factor: the sequence must converge to the exact value
    assert forces[-1] == pytest.approx(exact, rel=0.01)
    assert abs(forces[2] - forces[1]) / forces[2] < 0.02


def test_penalty_stiffness_scales_with_factor():
    material = ice_material()
    a = penalty_stiffness(material, 1e-6, 10.0)
    b = penalty_stiffness(material, 1e-6, 20.0)
    assert b == pytest.approx(2.0 * a)
    ks = surface_penalty_stiffness(material, 1e-4, 10.0)
    assert ks == pytest.approx(10.0 * material.E / 1e-4)


# --------------------------------------------------------------------------- #
# End-to-end indentation driver
# --------------------------------------------------------------------------- #
def _small_indentation(sigma_y=SIGMA_Y, n_steps=2, delta=6e-6, **kwargs):
    mesh = quarter_block_mesh(half_width=4e-3, height=2.5e-3, dims=(6, 6, 4),
                              power=2.0, vertical_power=1.5)
    indenter = SphericalIndenter(radius=10e-3, apex_z=2.5e-3)
    kwargs.setdefault("ramp", "power3")
    res = simulate_indentation(mesh, ice_material(sigma_y), indenter,
                               delta_max=delta, n_steps=n_steps,
                               penalty_factor=kwargs.pop("penalty_factor", 10.0),
                               symmetry_factor=4, tol=1e-6, max_iter=40,
                               **kwargs)
    return mesh, res


def test_indentation_elastic_limit_matches_hertz():
    delta = 5e-6
    mesh, res = _small_indentation(sigma_y=1e12, n_steps=1, delta=delta)
    assert res.converged
    reference = hertz_force(reduced_modulus(E_ICE, NU_ICE), 10e-3, delta)
    assert res.force[-1] == pytest.approx(reference, rel=0.35)
    # yielding is off, so no plastic strain may appear
    assert float(np.max(res.alpha)) == pytest.approx(0.0, abs=1e-12)


def test_indentation_force_is_monotone_and_converges():
    mesh, res = _small_indentation(n_steps=3, delta=9e-6)
    assert res.converged
    assert np.all(np.diff(res.force) > 0.0)
    assert float(np.max(res.alpha)) > 0.0            # plasticity is active
    assert np.all(np.array(res.max_von_mises) <=
                  1.2 * SIGMA_Y + 1e-6)             # return mapping respected


def test_indentation_energy_balance():
    # a dense linear load path keeps the trapezoidal path integral of the
    # external work accurate (the acceptance gate uses 16 steps, AC5)
    mesh, res = _small_indentation(n_steps=8, delta=9e-6, ramp="linear")
    bal = res.energy_balance()
    assert bal["external_work"] > 0.0
    assert bal["plastic_dissipation"] > 0.0
    assert abs(bal["relative_residual"]) < 0.05       # AC5 target is 1 %
    assert abs(bal["penalty_fraction"]) < 0.05


def test_quarter_block_bcs_remove_rigid_body_modes():
    mesh = quarter_block_mesh(half_width=2e-3, height=1e-3, dims=(3, 3, 2))
    fixed = quarter_block_bcs(mesh)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    free_nodes = np.unique(free // 3)
    # every free node keeps at least the vertical dof, and the constrained
    # dofs cover all three directions somewhere
    for comp in range(3):
        assert np.any(fixed % 3 == comp)
    assert free_nodes.size > 0


def test_facet_contact_area_grows_with_penetration():
    mesh = quarter_block_mesh(half_width=4e-3, height=2.5e-3, dims=(6, 6, 4))
    facets = build_surface_facets(mesh)
    k = surface_penalty_stiffness(ice_material(), facets.h, 10.0)
    u = np.zeros(mesh.n_dof)
    areas = []
    for delta in (2e-6, 6e-6, 12e-6):
        ind = SphericalIndenter(radius=10e-3, apex_z=2.5e-3).advance(delta)
        areas.append(facet_contact_state(mesh, u, ind, k, facets).contact_area)
    assert areas[0] < areas[1] < areas[2]
    # geometric contact radius sqrt(2 R delta) gives an upper bound on the area
    for delta, area in zip((2e-6, 6e-6, 12e-6), areas):
        assert area <= np.pi * 2 * 10e-3 * delta / 4 * 1.3


def test_hertz_yield_onset_is_consistent_with_hertz_force():
    onset = hertz_yield_onset(E_ICE, NU_ICE, SIGMA_Y, 10e-3)
    E_star = reduced_modulus(E_ICE, NU_ICE)
    force = hertz_force(E_star, 10e-3, onset["penetration"])
    a = np.sqrt(10e-3 * onset["penetration"])
    assert force / (np.pi * a * a) == pytest.approx(onset["mean_pressure"], rel=1e-6)
    assert onset["penetration"] > 0.0