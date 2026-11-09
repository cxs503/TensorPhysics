"""Verification tests for the 3-D isoparametric solid element kernel.

Coverage map
------------
* shape functions: Kronecker delta, partition of unity, analytic derivatives
  vs central differences, Gauss rule weights;
* element level: volume, B-matrix rank, Flanagan-Belytschko hourglass modes
  (rank deficiency of the 1-point element, stabilised rank), hourglass gamma
  orthogonality to the linear field;
* constant-strain patch test (Hex8/Hex20, undistorted and distorted, full and
  B-bar) -- machine-precision exactness;
* volumetric locking: ``nu = 0.4999`` cylinder, full integration locks while
  B-bar keeps the volumetric strain error well below 1 %;
* hourglass control: energy fraction below 5 % on a smooth field, rank
  deficiency without stabilisation;
* J2 plasticity: consistent tangent vs central differences, uniaxial tension
  vs the analytic bilinear curve, plane-strain cylinder limit pressure vs
  ``2 sigma_y / sqrt(3) ln(b/a)``;
* mesh helpers: Hex20 promotion, consistent pressure loads.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.solid_fem import (
    GAUSS_RULES,
    HEX20_NATURAL,
    HEX8_NATURAL,
    J2Material,
    PlasticState,
    annulus_mesh,
    box_mesh,
    element_kinematics,
    element_volume,
    gauss_stresses_plastic,
    hex8_shape,
    hex20_shape,
    hourglass_energy_ratio,
    hourglass_gamma,
    hourglass_rank,
    isotropic_elasticity,
    lame_limit_pressure,
    lame_radial_displacement,
    lame_stresses,
    numerical_tangent,
    patch_test,
    pressure_load,
    promote_hex8_to_hex20,
    solve_elastic,
    solve_plastic,
    thick_cylinder,
    uniaxial_plasticity,
)
from tensorphysics.solid_fem.element import element_stiffness, hourglass_stiffness
from tensorphysics.solid_fem.verification import cantilever_tip_deflection, cylinder_limit_pressure

E_STEEL = 210.0e9
NU_STEEL = 0.3


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------


def test_hex8_shape_kronecker_delta():
    for a, node in enumerate(HEX8_NATURAL):
        N, _ = hex8_shape(*node)
        assert N[a] == pytest.approx(1.0, abs=1e-15)
        assert np.sum(np.delete(N, a)) == pytest.approx(0.0, abs=1e-15)


def test_hex20_shape_kronecker_delta_and_partition():
    N, _ = hex20_shape(0.31, -0.47, 0.62)
    assert N.sum() == pytest.approx(1.0, abs=1e-14)
    for a, node in enumerate(HEX20_NATURAL):
        Na, _ = hex20_shape(*node)
        assert Na[a] == pytest.approx(1.0, abs=1e-14), f"node {a}"
        assert np.sum(np.delete(Na, a)) == pytest.approx(0.0, abs=1e-13), f"node {a}"


@pytest.mark.parametrize("shape_fn,natural", [(hex8_shape, HEX8_NATURAL), (hex20_shape, HEX20_NATURAL)])
def test_analytic_derivatives_match_central_differences(shape_fn, natural):
    xi = np.array([0.23, -0.31, 0.44])
    _N, dN = shape_fn(*xi)
    h = 1e-6
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        Np, _ = shape_fn(*(xi + d))
        Nm, _ = shape_fn(*(xi - d))
        assert np.max(np.abs((Np - Nm) / (2 * h) - dN[i])) < 1e-8


def test_gauss_rules_integrate_parent_volume():
    for name, (_pts, wts) in GAUSS_RULES.items():
        assert wts.sum() == pytest.approx(8.0, abs=1e-14), name


# ---------------------------------------------------------------------------
# Element level: volume, rank, hourglass vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("element", ["hex8", "hex20"])
def test_element_volume_of_unit_cube(element):
    mesh = box_mesh(dims=(1, 1, 1), lengths=(2.0, 3.0, 0.5), element=element)
    vol = element_volume(element, mesh.element_coords(0))
    assert vol == pytest.approx(3.0, rel=1e-13)


def test_b_matrix_rank_is_six():
    mesh = box_mesh(dims=(1, 1, 1), distort=0.2, seed=7)
    kin = element_kinematics("hex8", mesh.element_coords(0), "2x2x2")
    for g in range(kin.integration_points):
        assert np.linalg.matrix_rank(kin.B[g]) == 6


def test_one_point_hex8_is_rank_deficient_without_hourglass_control():
    rank = hourglass_rank(alpha_hg=0.0)
    # 24 dofs, rank 6 -> 18 zero-energy modes = 6 rigid + 12 spurious
    assert rank["zero_modes"] == 18


def test_hourglass_control_removes_all_spurious_modes():
    rank = hourglass_rank(alpha_hg=1.0)
    assert rank["zero_modes"] == 6  # rigid body modes only


def test_hourglass_gamma_orthogonal_to_linear_field():
    mesh = box_mesh(dims=(1, 1, 1), lengths=(1.0, 2.0, 3.0), distort=0.2, seed=5)
    coords = mesh.element_coords(0)
    gamma = hourglass_gamma(coords)
    for a in range(4):
        assert gamma[a].sum() == pytest.approx(0.0, abs=1e-12)
        for i in range(3):
            assert gamma[a] @ coords[:, i] == pytest.approx(0.0, abs=1e-12)


def test_hourglass_stiffness_scales_with_shear_modulus():
    coords = box_mesh(dims=(1, 1, 1)).element_coords(0)
    k1 = hourglass_stiffness(coords, 1.0, 1.0e9)
    k2 = hourglass_stiffness(coords, 2.0, 1.0e9)
    assert np.allclose(k2, 2.0 * k1)


# ---------------------------------------------------------------------------
# Patch tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("element,dims", [("hex8", (2, 2, 2)), ("hex20", (3, 3, 3))])
@pytest.mark.parametrize("distort", [0.0, 0.25])
def test_constant_strain_patch_test(element, dims, distort):
    res = patch_test(element=element, dims=dims, distort=distort, seed=11)
    assert res["displacement_error"] < 1e-10
    assert res["stress_error"] < 1e-10


@pytest.mark.parametrize(
    "element,dims,formulation",
    [("hex8", (2, 2, 2), "bbar"), ("hex20", (2, 2, 2), "bbar_stab")],
)
def test_patch_test_survives_reduced_volumetric_formulation(element, dims, formulation):
    res = patch_test(element=element, dims=dims, distort=0.2, seed=3,
                     formulation=formulation)
    assert res["displacement_error"] < 1e-10
    assert res["stress_error"] < 1e-10


def test_plain_bbar_is_rejected_for_hex20():
    """Mean dilatation leaves three spurious modes for the 20-node brick."""
    m = box_mesh(dims=(1, 1, 1), element="hex20")
    with pytest.raises(ValueError):
        element_stiffness("hex20", m.element_coords(0),
                          isotropic_elasticity(E_STEEL, NU_STEEL), "3x3x3", "bbar")


@pytest.mark.parametrize("formulation", ["sri", "bbar_stab"])
def test_hex20_volumetric_formulations_are_stable(formulation):
    m = box_mesh(dims=(1, 1, 1), lengths=(1.25, 1.0, 1.0), element="hex20")
    k = element_stiffness("hex20", m.element_coords(0),
                          isotropic_elasticity(E_STEEL, NU_STEEL), "3x3x3", formulation)
    w = np.linalg.eigvalsh(0.5 * (k + k.T))
    assert int(np.sum(np.abs(w) < 1e-8 * np.abs(w).max())) == 6


def test_patch_test_rejects_nonlinear_interpolation():
    """A patch with a *quadratic* field must NOT be reproduced exactly.

    Guards against a patch test that would pass trivially (i.e. confirms the
    test actually measures interpolation error).
    """
    mesh = box_mesh(dims=(2, 2, 2), lengths=(1.0, 1.0, 1.0))
    damping = np.array([[2.0e-3, 0.0, 0.0], [0.0, -1.5e-3, 0.0], [0.0, 0.0, 0.0]])
    u_exact = np.column_stack(
        [(mesh.nodes**2) @ damping[i] for i in range(3)]
    )
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    tol = 1e-10
    boundary = np.nonzero(
        np.any(
            np.isclose(mesh.nodes, lo, atol=tol) | np.isclose(mesh.nodes, hi, atol=tol),
            axis=1,
        )
    )[0]
    C = isotropic_elasticity(E_STEEL, NU_STEEL)
    fixed = mesh.dofs_of_nodes(boundary)
    sol = solve_elastic(mesh, C, fixed_dofs=fixed, fixed_values=u_exact[boundary].ravel())
    err = np.max(np.abs(sol.u - u_exact.ravel())) / np.max(np.abs(u_exact))
    assert err > 1e-6  # trilinear elements cannot represent a quadratic field


# ---------------------------------------------------------------------------
# Volumetric locking / B-bar
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cylinder_nu_lock():
    return {
        form: thick_cylinder(nu=0.4999, n_radial=4, n_circ=8, element="hex8", formulation=form)
        for form in ("full", "bbar")
    }


def test_locking_with_full_integration_at_nu_0_4999(cylinder_nu_lock):
    """Reference case: full integration locks badly (this is the defect)."""
    fe = cylinder_nu_lock["full"].gauss_field_errors()
    assert fe["eps_v_rel_error"] > 0.5
    ur = cylinder_nu_lock["full"].inner_displacement_error()
    assert ur["rel_error"] > 0.1


def test_bbar_volumetric_strain_error_below_one_percent(cylinder_nu_lock):
    fe = cylinder_nu_lock["bbar"].gauss_field_errors()
    assert fe["eps_v_rel_error"] < 0.01
    assert fe["eps_v_pointwise_max_rel_error"] < 0.01


def test_bbar_keeps_nearly_incompressible_displacement_accurate(cylinder_nu_lock):
    ur = cylinder_nu_lock["bbar"].inner_displacement_error()
    assert ur["rel_error"] < 0.02


def test_selective_reduced_integration_removes_locking():
    sri = thick_cylinder(nu=0.4999, n_radial=4, n_circ=8, element="hex8", formulation="sri")
    ur = sri.inner_displacement_error()
    assert ur["rel_error"] < 0.02


# ---------------------------------------------------------------------------
# Lame thick-walled cylinder (elastic anchor)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cylinder_hex20():
    return thick_cylinder(nu=NU_STEEL, n_radial=4, n_circ=8, element="hex20",
                          formulation="bbar_stab")


def test_lame_stress_anchor_hex20(cylinder_hex20):
    fe = cylinder_hex20.gauss_field_errors()
    assert fe["sigma_theta_max_rel_error"] < 0.02
    assert fe["sigma_r_max_rel_error"] < 0.02


def test_hex20_bbar_stab_removes_locking_at_nu_0_4999():
    setup = thick_cylinder(nu=0.4999, n_radial=4, n_circ=8, element="hex20",
                           formulation="bbar_stab")
    fe = setup.gauss_field_errors()
    assert fe["eps_v_rel_error"] < 0.01
    assert fe["sigma_theta_max_rel_error"] < 0.02


def test_lame_radial_displacement_anchor_hex20(cylinder_hex20):
    ur = cylinder_hex20.inner_displacement_error()
    assert ur["rel_error"] < 0.01


def test_lame_analytic_satisfies_equilibrium_and_boundary_conditions():
    a, b, p = 1.0, 2.0, 50.0e6
    r = np.linspace(a, b, 101)
    sr, st = lame_stresses(r, a, b, p)
    assert sr[0] == pytest.approx(-p, rel=1e-12)
    assert sr[-1] == pytest.approx(0.0, abs=1e-3)
    # d(sigma_r)/dr + (sigma_r - sigma_theta)/r = 0 (checked on interior points;
    # np.gradient is only first order accurate at the end points)
    dsr = np.gradient(sr, r)
    interior = slice(2, -2)
    resid = np.abs(dsr + (sr - st) / r)[interior]
    assert np.max(resid) / p < 1e-3


def test_lame_radial_displacement_is_incompressible_limit_consistent():
    a, b, p, E, nu = 1.0, 2.0, 50e6, E_STEEL, 0.49999
    r = np.linspace(a, b, 50)
    u = lame_radial_displacement(r, a, b, p, E, nu)
    C = (1.0 + nu) * p * a**2 / (E * (b**2 - a**2))
    eps_r = C * ((1.0 - 2.0 * nu) - b**2 / r**2)
    eps_t = u / r
    assert np.allclose(np.gradient(u, r)[2:-2], eps_r[2:-2], rtol=1e-3)
    # volumetric strain of the plane-strain Lame solution is 2 C (1 - 2 nu)
    assert np.max(np.abs(eps_r + eps_t)) == pytest.approx(2.0 * C * (1.0 - 2.0 * nu), rel=1e-10)
    assert np.max(np.abs(eps_r + eps_t)) / np.max(np.abs(eps_t)) < 1e-4


def test_hex8_lame_improves_with_refinement():
    coarse = thick_cylinder(nu=NU_STEEL, n_radial=2, n_circ=8, element="hex8", formulation="bbar")
    fine = thick_cylinder(nu=NU_STEEL, n_radial=6, n_circ=12, element="hex8", formulation="bbar")
    e_coarse = coarse.gauss_field_errors()["sigma_theta_max_rel_error"]
    e_fine = fine.gauss_field_errors()["sigma_theta_max_rel_error"]
    assert e_fine < e_coarse
    assert e_fine < 0.05


# ---------------------------------------------------------------------------
# Hourglass control
# ---------------------------------------------------------------------------


def test_hourglass_energy_fraction_below_five_percent():
    """Smooth (well-resolved) field: stabilisation energy must stay small."""
    setup = thick_cylinder(
        nu=NU_STEEL, n_radial=4, n_circ=8, element="hex8", formulation="full",
        rule="1pt", hourglass=0.1,
    )
    C = isotropic_elasticity(E_STEEL, NU_STEEL)
    e_hg, e_tot = hourglass_energy_ratio(
        setup.mesh, setup.u, C, hourglass=0.1, shear_modulus=E_STEEL / (2 * (1 + NU_STEEL)),
        rule="1pt",
    )
    assert e_hg > 0.0
    assert e_hg / e_tot < 0.05
    # stabilized solution stays physical
    assert setup.inner_displacement_error()["rel_error"] < 0.02


def test_unstabilized_one_point_element_blows_up_in_bending():
    """Rank-deficient 1-point meshes activate spurious modes under bending."""
    ref = cantilever_tip_deflection(n_length=8, n_thick=1, n_width=1, element="hex20")
    bad = cantilever_tip_deflection(
        n_length=8, n_thick=1, n_width=1, rule="1pt", hourglass=0.0
    )
    assert abs(bad["uy_tip_fe"]) > 100.0 * abs(ref["uy_tip_timoshenko"])


def test_cantilever_hex20_tip_deflection_close_to_timoshenko():
    res = cantilever_tip_deflection(n_length=8, n_thick=2, n_width=1, element="hex20")
    assert res["rel_error"] < 0.05


# ---------------------------------------------------------------------------
# J2 plasticity
# ---------------------------------------------------------------------------


def test_consistent_tangent_matches_numerical_differentiation():
    mat = J2Material(E=E_STEEL, nu=NU_STEEL, sigma_y0=250e6, hardening=20e9)
    for eps in (
        np.array([1.5e-3, -4e-4, -6e-4, 3e-4, -2e-4, 1e-4]),
        np.array([2.0e-3, -7e-4, -7e-4, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 4.0e-3, 0.0, 0.0]),
    ):
        state = PlasticState()
        _sig, _st, C_ep, plastic = mat.return_map(eps, state)
        assert plastic
        C_num = numerical_tangent(mat, eps, state)
        assert np.max(np.abs(C_ep - C_num)) / np.max(np.abs(C_num)) < 1e-6
        assert np.allclose(C_ep, C_ep.T, atol=1e-6 * np.max(np.abs(C_ep)))


def test_return_map_hits_the_yield_surface():
    mat = J2Material(E=E_STEEL, nu=NU_STEEL, sigma_y0=250e6, hardening=20e9)
    eps = np.array([2.0e-3, -7e-4, -7e-4, 0.0, 0.0, 0.0])
    sig, state, _C, plastic = mat.return_map(eps, PlasticState())
    assert plastic
    s = sig - np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) * (sig[:3].sum() / 3.0)
    q = np.sqrt(1.5 * (s @ s + s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
    assert q == pytest.approx(mat.yield_stress(state.alpha), rel=1e-10)
    assert state.eps_p[:3].sum() == pytest.approx(0.0, abs=1e-12)  # isochoric


def test_elastic_return_map_is_exact_for_small_strain():
    mat = J2Material(E=E_STEEL, nu=NU_STEEL, sigma_y0=250e6, hardening=20e9)
    eps = np.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig, state, C, plastic = mat.return_map(eps, PlasticState())
    assert not plastic
    assert np.allclose(sig, mat.elastic_matrix @ eps)
    assert state.alpha == 0.0


@pytest.mark.parametrize("dims", [(1, 1, 1), (2, 2, 2)])
def test_uniaxial_tension_matches_analytic_curve(dims):
    res = uniaxial_plasticity(n_steps=20, max_strain=0.003, dims=dims)
    assert res["converged"]
    assert res["alpha_max"] > 0.0
    assert res["rel_error"] < 0.01


def test_plane_strain_cylinder_limit_pressure():
    res = cylinder_limit_pressure(n_radial=4, n_circ=8, n_steps=30)
    assert res["p_limit_fe"] is not None
    assert res["rel_error"] < 0.05
    assert res["p_limit_analytic"] == pytest.approx(
        lame_limit_pressure(1.0, 2.0, 250e6), rel=1e-12
    )


def test_plastic_cylinder_relieves_hoop_stress_beyond_yield():
    """Plasticity must cap the hoop stress near the yield surface."""
    a, b, p = 1.0, 2.0, 160.0e6  # above first yield (~108 MPa), below p_L (~200 MPa)
    mesh = annulus_mesh(a=a, b=b, length=1.0, n_radial=4, n_circ=8, n_axial=1, element="hex8",
                        theta_span=np.pi / 2)
    tol = 1e-9 * b
    theta = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
    fixed = np.unique(
        np.concatenate(
            [
                mesh.dofs_of_nodes(np.arange(mesh.n_nodes), component=2),
                mesh.dofs_of_nodes(np.nonzero(np.abs(theta) < tol)[0], component=1),
                mesh.dofs_of_nodes(np.nonzero(np.abs(theta - np.pi / 2) < tol)[0], component=0),
            ]
        )
    )
    loads = pressure_load(mesh, p, lambda c, n, corners: bool(
        np.max(np.hypot(corners[:, 0], corners[:, 1])) <= a + 1e-9 * b))
    mat = J2Material(E=E_STEEL, nu=NU_STEEL, sigma_y0=250e6, hardening=0.0)
    sol = solve_plastic(mesh, mat, fixed, loads, n_steps=20, formulation="bbar",
                        tol=1e-8, max_iter=60)
    assert sol.converged
    gs = gauss_stresses_plastic(mesh, sol.u, mat, sol.states, formulation="bbar")
    sig = np.vstack([s for _pts, s in gs])
    s_dev = sig[:, :3] - sig[:, :3].mean(axis=1, keepdims=True)
    q = np.sqrt(1.5 * (s_dev**2).sum(axis=1)
                + 3.0 * (sig[:, 3:] ** 2).sum(axis=1))
    assert q.max() <= 250e6 * (1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def test_hex20_promotion_shares_edge_nodes():
    m8 = box_mesh(dims=(2, 2, 2))
    m20 = promote_hex8_to_hex20(m8)
    assert m20.element == "hex20"
    # 27 corners + 54 edges of a 2x2x2 grid = 81 nodes
    assert m20.n_nodes == 81
    assert len(np.unique(m20.connectivity)) == 81


def test_arc_midpoints_land_on_the_cylinder_wall():
    m = annulus_mesh(a=1.0, b=2.0, n_radial=2, n_circ=4, n_axial=1, element="hex20")
    r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
    assert np.all(r >= 1.0 - 1e-12)
    assert np.all(r <= 2.0 + 1e-12)


def test_pressure_load_resultant_on_quarter_arc():
    a, b = 1.0, 2.0
    m = annulus_mesh(a=a, b=b, n_radial=2, n_circ=8, n_axial=1, theta_span=np.pi / 2)
    p = 7.0e6
    f = pressure_load(m, p, lambda c, n, corners: bool(
        np.max(np.hypot(corners[:, 0], corners[:, 1])) <= a + 1e-9 * b))
    resultant = f.reshape(-1, 3).sum(axis=0)
    # int_0^{pi/2} p a (cos, sin) dtheta = p a (1, 1)
    assert resultant[0] == pytest.approx(p * a, rel=1e-10)
    assert resultant[1] == pytest.approx(p * a, rel=1e-10)
    assert resultant[2] == pytest.approx(0.0, abs=1e-6)


def test_singular_jacobian_is_rejected():
    mesh = box_mesh(dims=(2, 2, 2))
    coords = mesh.element_coords(0).copy()
    coords[:, 2] = 0.0  # flatten the element -> zero Jacobian everywhere
    with pytest.raises(ValueError):
        element_kinematics("hex8", coords, "2x2x2")


def test_element_stiffness_is_symmetric():
    coords = box_mesh(dims=(1, 1, 1), distort=0.15, seed=2).element_coords(0)
    k = element_stiffness("hex8", coords, isotropic_elasticity(E_STEEL, NU_STEEL), "2x2x2")
    assert np.allclose(k, k.T)
    w = np.linalg.eigvalsh(k)
    assert np.sum(np.abs(w) < 1e-6 * np.abs(w).max()) == 6