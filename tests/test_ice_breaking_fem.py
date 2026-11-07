"""Verification and negative tests for the Kirchhoff/Winkler BFS plate FEM.

Coverage map (campaign node IN11-H3):

* closed-form agreement - point load on an infinite plate vs the Hetenyi/Kelvin
  solution (deflection *and* radial moment), uniform load on a simply supported
  plate vs the Navier series, Gaussian contact patch vs the semi-analytic Hankel
  integral;
* element behaviour - exact load resultant of the consistent load vector,
  moment capacity formula, first-crack radius detector (including the no-crack
  branch);
* negative controls - non-physical thickness / modulus / foundation modulus /
  Poisson ratio / mesh must be rejected with ``ValueError``;
* physics sanity - the edge-loaded critical force and flake length grow with ice
  thickness.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorphysics.ice.static_plate import hankel_gauss_deflection
from tensorphysics.ice_sheet.kirchhoff_plate_fem import (
    BFSPlateSolver,
    KirchhoffWinklerPlate,
    bending_stress,
    critical_point_load,
    fan_sector_area,
    fan_sector_mask,
    graded_axis,
    kelvin_point_load,
    kelvin_point_load_moments,
    limiting_moment,
)
from tensorphysics.ice_sheet.winkler_plate import (
    WinklerPlateParams,
    navier_center_deflection,
)

E_ICE = 5.0e9
NU = 0.3
K_W = 1025.0 * 9.81
ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_breaking_fem.json"


def _plate(h=1.0, lx=1.0, ly=1.0, **kw):
    return KirchhoffWinklerPlate(h=h, E=E_ICE, nu=NU, k_w=K_W, lx=lx, ly=ly, **kw)


# ---------------------------------------------------------------------------
# closed-form references
# ---------------------------------------------------------------------------
def test_point_load_centre_deflection_matches_kelvin_closed_form():
    """FEM centre deflection of a point load vs Hetenyi's closed form (< 3 %)."""
    plate0 = _plate()
    D, l = plate0.D, plate0.char_length
    P = 1.0e6
    L = 6.0 * l
    solver = BFSPlateSolver(_plate(lx=L, ly=L), 30, 30)
    solver.add_point_load(0.25 * P, 0.0, 0.0)  # quarter model
    field = solver.solve({"left": "sym", "bottom": "sym", "right": "free", "top": "free"})
    exact = kelvin_point_load(0.0, P, D, K_W)
    assert exact > 0.0
    assert abs(field.w[0, 0] - exact) / exact < 0.03
    assert field.w[0, 0] == pytest.approx(exact, rel=5e-3)


def test_point_load_radial_profile_matches_kelvin_closed_form():
    """Full radial deflection profile, L2 relative error < 1 %."""
    plate0 = _plate()
    D, l = plate0.D, plate0.char_length
    P = 1.0e6
    L = 6.0 * l
    solver = BFSPlateSolver(_plate(lx=L, ly=L), 30, 30)
    solver.add_point_load(0.25 * P, 0.0, 0.0)
    field = solver.solve({"left": "sym", "bottom": "sym", "right": "free", "top": "free"})
    r = np.linspace(0.05, 3.0, 40) * l
    w_ref = kelvin_point_load(r, P, D, K_W)
    w_fem = field.sample_ray((0.0, 0.0), 0.0, r=r)["w"]
    l2 = np.linalg.norm(w_fem - w_ref) / np.linalg.norm(w_ref)
    assert l2 < 0.01


def test_radial_moment_matches_kelvin_ode_solution():
    """Moment field (second derivatives) agrees with the Kelvin ODE solution."""
    plate0 = _plate()
    D, l = plate0.D, plate0.char_length
    P = 1.0e6
    L = 6.0 * l
    solver = BFSPlateSolver(_plate(lx=L, ly=L), 60, 60)
    solver.add_point_load(0.25 * P, 0.0, 0.0)
    field = solver.solve({"left": "sym", "bottom": "sym", "right": "free", "top": "free"})
    r = np.array([0.5, 1.0, 1.5, 2.0]) * l
    fem = field.sample_ray((0.0, 0.0), 0.0, r=r)["M_r"]
    kel = kelvin_point_load_moments(r, P, D, K_W, NU)["M_r"]
    assert np.all(kel != 0.0)
    assert np.max(np.abs(fem / kel - 1.0)) < 0.03
    # the hogging (negative) branch exists, which is what drives the ring crack
    assert kel.min() < 0.0


def test_uniform_load_simply_supported_matches_navier():
    """Simply supported plate under uniform load vs the Navier double series."""
    plate0 = _plate()
    l = plate0.char_length
    a = b = 6.0 * l
    q = 1.0e4
    params = WinklerPlateParams(a=a, b=b, h=1.0, E=E_ICE, nu=NU, k_w=K_W, q=q)
    w_ref = navier_center_deflection(params, n_terms=61)
    solver = BFSPlateSolver(_plate(lx=a, ly=b), 8, 8)
    solver.add_pressure(lambda x, y: np.full(np.shape(x), q, dtype=float))
    field = solver.solve({e: "simply" for e in ("left", "right", "bottom", "top")})
    n = field.w.shape[0] // 2
    assert abs(field.w[n, n] - w_ref) / w_ref < 1e-3
    # the consistent load vector integrates the pressure exactly
    assert solver.nodal_load_sum() == pytest.approx(q * a * b, rel=1e-12)


def test_gaussian_patch_matches_hankel_integral():
    """Distributed-load path vs an independent semi-analytic Hankel solution."""
    plate0 = _plate()
    D, l = plate0.D, plate0.char_length
    P, sigma = 1.0e6, 0.1 * l
    L = 6.0 * l

    def q(x, y):
        return P / (2.0 * math.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    solver = BFSPlateSolver(
        _plate(lx=L, ly=L), 24, 24, xs=graded_axis(L, 24, 0.30), ys=graded_axis(L, 24, 0.30)
    )
    solver.add_pressure(q, inside=lambda x, y: np.hypot(x, y) <= 4.0 * sigma, gauss_n=4)
    field = solver.solve({"left": "sym", "bottom": "sym", "right": "free", "top": "free"})
    r = np.array([0.0, 1.0, 2.0]) * sigma
    w_fem = field.sample_ray((0.0, 0.0), 0.0, r=r)["w"]
    w_ref = hankel_gauss_deflection(r, P, sigma, D)
    assert np.max(np.abs(w_fem - w_ref) / np.abs(w_ref)) < 0.03


# ---------------------------------------------------------------------------
# element / load / failure-mechanics behaviour
# ---------------------------------------------------------------------------
def test_fan_load_resultant_is_exact():
    """The fan-patch load resultant equals the requested force exactly."""
    P = 4.0e5
    solver = BFSPlateSolver(_plate(lx=40.0, ly=40.0), 40, 40)
    solver.add_fan_load(P, 0.0, 0.0, 1.0, 60.0, gauss_n=6)
    assert solver.nodal_load_sum() == pytest.approx(P, rel=1e-12)
    assert solver.applied_load == pytest.approx(P, rel=1e-12)
    # interleaved nodal moment components carry no net force
    assert solver.nodal_load_sum() == pytest.approx(solver._f[0::4].sum(), rel=1e-15)


def test_moment_capacity_and_stress_are_consistent():
    """M_f = sigma_f h^2/6 and sigma = 6M/h^2 are inverse to each other."""
    h, sigma_f = 1.0, 0.7e6
    M_f = limiting_moment(sigma_f, h)
    assert M_f == pytest.approx(sigma_f * h * h / 6.0, rel=1e-15)
    assert bending_stress(M_f, h) == pytest.approx(sigma_f, rel=1e-12)
    plate = _plate(h=h)
    assert plate.moment_capacity(sigma_f) == pytest.approx(M_f, rel=1e-15)


def test_crack_radius_detector_and_no_crack_branch():
    """The ring-crack radius is found when it exists and reported as nan otherwise."""
    plate0 = _plate()
    D, l = plate0.D, plate0.char_length
    M_f = limiting_moment(0.7e6, 1.0)
    kel = critical_point_load(M_f, D, K_W, NU)
    solver = BFSPlateSolver(_plate(lx=6.0 * l, ly=6.0 * l), 60, 60)
    solver.add_point_load(0.25 * 1.05 * kel["P_crit_N"], 0.0, 0.0)
    field = solver.solve({"left": "sym", "bottom": "sym", "right": "free", "top": "free"})
    # the hogging peak sits where the Kelvin solution puts it
    _m, r_peak_fem = field.peak_hogging_moment(origin=(0.0, 0.0))
    assert r_peak_fem == pytest.approx(kel["r_peak_m"], rel=0.15)
    # just above the critical load the cracked band brackets that radius
    r_in = field.crack_radius(M_f, origin=(0.0, 0.0), outermost=False)
    r_out = field.crack_radius(M_f, origin=(0.0, 0.0), outermost=True)
    assert r_in < r_peak_fem < r_out
    assert r_out / r_peak_fem < 1.5
    # a capacity far above the applied moment cannot crack anything
    assert math.isnan(field.crack_radius(10.0 * M_f, origin=(0.0, 0.0)))


def test_graded_axis_is_monotone_and_spanning():
    """Mesh grading helper produces a strictly increasing axis that ends at L."""
    xs = graded_axis(100.0, 40, 0.05)
    assert xs.size == 41
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(100.0, rel=1e-12)
    assert np.all(np.diff(xs) > 0.0)
    assert np.diff(xs)[0] == pytest.approx(0.05, rel=1e-9)


def test_fan_sector_geometry():
    """Sector area and mask agree with the analytic half-disc limit."""
    assert fan_sector_area(2.0, 90.0) == pytest.approx(0.5 * math.pi * 4.0, rel=1e-12)
    mask = fan_sector_mask(0.0, 0.0, 1.0, 90.0, direction_deg=0.0)
    assert bool(mask(np.array([0.5]), np.array([0.0]))[0])
    assert not bool(mask(np.array([-0.5]), np.array([0.0]))[0])
    assert not bool(mask(np.array([2.0]), np.array([0.0]))[0])


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k_w", [0.0, -1.0, -1.0e4])
def test_negative_or_zero_foundation_modulus_rejected(k_w):
    """A non-positive Winkler modulus is physically meaningless -> ValueError."""
    with pytest.raises(ValueError, match="k_w"):
        KirchhoffWinklerPlate(h=1.0, E=E_ICE, nu=NU, k_w=k_w, lx=10.0, ly=10.0)


@pytest.mark.parametrize("h", [0.0, -0.5, float("nan")])
def test_negative_or_zero_thickness_rejected(h):
    """A non-positive (or NaN) plate thickness -> ValueError."""
    with pytest.raises(ValueError, match="h"):
        KirchhoffWinklerPlate(h=h, E=E_ICE, nu=NU, k_w=K_W, lx=10.0, ly=10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(E=0.0),
        dict(E=-5.0e9),
        dict(nu=0.5),
        dict(nu=-1.0),
        dict(lx=0.0),
    ],
)
def test_inadmissible_plate_parameters_rejected(kwargs):
    """Zero/negative modulus, out-of-range Poisson ratio, zero size -> ValueError."""
    base = dict(h=1.0, E=E_ICE, nu=NU, k_w=K_W, lx=10.0, ly=10.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        KirchhoffWinklerPlate(**base)


@pytest.mark.parametrize("nx,ny", [(0, 4), (4, 0), (-2, 4)])
def test_invalid_mesh_rejected(nx, ny):
    """A mesh with fewer than one element is rejected."""
    with pytest.raises(ValueError, match="nx and ny"):
        BFSPlateSolver(_plate(lx=10.0, ly=10.0), nx, ny)


def test_zero_strength_rejected():
    """Zero flexural strength cannot define a moment capacity -> ValueError."""
    with pytest.raises(ValueError, match="sigma_f"):
        limiting_moment(0.0, 1.0)
    with pytest.raises(ValueError, match="h"):
        limiting_moment(0.7e6, 0.0)


# ---------------------------------------------------------------------------
# physics sanity: edge-loaded bow patch, break length grows with thickness
# ---------------------------------------------------------------------------
def _edge_case(h, n=36, h0=0.6):
    plate0 = _plate(h=h)
    l = plate0.char_length
    M_f = limiting_moment(0.7e6, h)
    Lx, Ly = 10.0 * l, 8.0 * l
    solver = BFSPlateSolver(
        _plate(h=h, lx=Lx, ly=Ly), n, n, xs=graded_axis(Lx, n, h0), ys=graded_axis(Ly, n, h0)
    )
    solver.add_fan_load(1.0, 0.0, 0.0, 1.0, 60.0, gauss_n=6, symmetry_factor=2.0)
    field = solver.solve({"left": "free", "right": "free", "bottom": "sym", "top": "free"})
    hog = field.extreme_hogging()
    per_unit = hog["M_2"] / solver.applied_load
    return M_f / abs(per_unit), hog["r"], l


def test_edge_critical_load_and_flake_length_grow_with_thickness():
    """Physically expected monotone h-dependence of the breaking parameters."""
    results = {h: _edge_case(h) for h in (0.8, 1.2)}
    P1, r1, l1 = results[0.8]
    P2, r2, l2 = results[1.2]
    assert P2 > P1 and r2 > r1
    # capacity scaling: P_crit ~ M_f / arm with M_f ~ h^2 and arm ~ l ~ h^{3/4}
    # P_crit = M_f / c with c dimensionless (M has units of moment per unit width):
    # M_f ~ h^2 and c is h-independent, so P_crit ~ h^2 -> ratio (1.2/0.8)^2 = 2.25
    assert 2.0 < P2 / P1 < 2.5
    assert 1.15 < r2 / r1 < 1.5
    # the crack sits at ~1.1-1.3 characteristic lengths from the contact
    assert 1.0 < r1 / l1 < 1.4
    assert 1.0 < r2 / l2 < 1.4


def test_artifact_schema_if_present():
    """The committed artifact carries the acceptance numbers and the break curve."""
    if not ARTIFACT.exists():
        pytest.skip("artifact not generated in this checkout")
    data = json.loads(ARTIFACT.read_text())
    assert data["mission"]["campaign_node"] == "IN11-H3"
    assert [c["h_m"] for c in data["breaking"]["curve"]] == [0.6, 0.8, 1.0, 1.2]
    for c in data["breaking"]["curve"]:
        assert c["edge"]["P_crit_N"] > 0.0
        assert c["edge"]["r_crack_m"] > 0.0
        assert c["edge"]["r_crack_over_l"] > 0.5
    assert data["verification"]["point_load"]["criterion_met"] is True
    assert data["acceptance"]["negative_controls_all_rejected"]["met"] is True

def test_symmetry_factor_books_the_physical_force():
    """A patch cut in half by a symmetry plane still reports its full force."""
    solver = BFSPlateSolver(_plate(lx=40.0, ly=20.0), 40, 20)
    solver.add_fan_load(8.0e5, 20.0, 0.0, 1.0, 60.0, gauss_n=6, symmetry_factor=2.0)
    # half of the patch lies outside the half-domain
    assert solver.nodal_load_sum() == pytest.approx(4.0e5, rel=1e-12)
    assert solver.applied_load == pytest.approx(8.0e5, rel=1e-12)
    with pytest.raises(ValueError, match="symmetry_factor"):
        BFSPlateSolver(_plate(lx=10.0, ly=10.0), 4, 4).add_fan_load(
            1.0, 0.0, 0.0, 1.0, 60.0, symmetry_factor=0.0
        )
