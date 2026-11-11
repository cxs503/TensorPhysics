"""Accuracy and integrity tests for the geometric-nonlinear module (m8870 R1).

Every test here compares a module result against something the module does not
use to produce it:

* the exact elastica integrals are compared with an independent numerical
  solution of the strong form (two-point boundary value problem, unknown load),
* the elastica strain energy is compared with the work of deformation
  ``int P d(delta)`` (Clausius) and with the generalised-force identity
  ``dU/d(delta) = P``,
* the von Karman plate energy balance compares the dead-load work with the
  strain energy of the converged finite element state,
* the plate closed-form load-deflection law is compared with the finite element
  solution,
* the reported convergence order of the plate FEM is computed, not assumed.

Background / numbers: docs/m8870_r1_audit.md (artifacts/m8870_r1/).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.integrate import solve_bvp

from tensorphysics.geometric_nonlinear.elastica import ElasticaExact
from tensorphysics.geometric_nonlinear.von_karman_plate import (
    VonKarmanPlateExact,
    VonKarmanPlateFEM,
    load_path_work,
    plate_cubic_coefficient,
)
from tensorphysics.geometric_nonlinear.verification import (
    camp_berger_reference,
    convergence_order_plate,
    full_verification_plate,
    verify_energy_conservation_elastica,
    verify_energy_conservation_plate,
)


# ----------------------------------------------------------------------------
# independent references
# ----------------------------------------------------------------------------

def _reference_integrals_quad(theta_B: float, L: float = 1.0, EI: float = 1.0):
    """(P, delta) from adaptive quadrature with a singularity-free variable.

    theta = theta_B (1 - xi^2),  xi in [0, 1]
    =>  sin(theta_B) - sin(theta) ~ xi^2 cos(theta_B)  (no singularity left)
    """
    lim = 1.0

    def integrand(kind):
        def f(xi):
            theta = theta_B * (1.0 - xi**2)
            diff = 2.0 * np.cos(0.5 * (theta_B + theta)) * np.sin(0.5 * (theta_B - theta))
            diff = max(diff, 1e-300)
            trig = 1.0 if kind == 0 else (np.cos(theta) if kind == 1 else np.sin(theta))
            # |dtheta/dxi| = 2 theta_B xi for theta = theta_B (1 - xi^2)
            return 2.0 * theta_B * xi * trig / np.sqrt(diff)

        return integrate.quad(f, 0.0, lim, limit=400, epsabs=1e-14, epsrel=1e-13)[0]

    I_L, I_y = integrand(0), integrand(2)
    P = EI * I_L**2 / (2.0 * L**2)
    delta = L * I_y / I_L
    return P, delta


def _reference_elastica_bvp(theta_B: float, L: float = 1.0, EI: float = 1.0):
    """(P, delta) from a direct solve of the strong form, independent of integrals.

    EI theta'' + P cos(theta) = 0,  theta(0) = 0, theta'(L) = 0, theta(L) = theta_B
    with P as an unknown parameter and the tip deflection
    delta = int_0^L sin(theta) ds.
    """
    n = 201
    s = np.linspace(0.0, L, n)

    def ode(sx, y, p):
        return np.vstack((y[1], -p[0] / EI * np.cos(y[0])))

    def bc(ya, yb, p):
        return np.array([ya[0], yb[1], yb[0] - theta_B])

    # initial guess: linear beam  theta = theta_B (2 s/L - (s/L)^2) * 0.9
    u = theta_B * (2.0 * s / L - (s / L) ** 2) * 0.9
    y0 = np.vstack((u, np.gradient(u, s)))
    p0 = np.array([3.0 * EI * theta_B / L**2])

    sol = solve_bvp(ode, bc, s, y0, p=p0, tol=1e-10, max_nodes=20000)
    if sol.status != 0:
        raise RuntimeError(f"BVP did not converge (status {sol.status}: {sol.message})")

    P = float(sol.p[0])
    delta = float(integrate.trapezoid(np.sin(sol.y[0]), sol.x))
    return P, delta


# ----------------------------------------------------------------------------
# elastica: exact solution
# ----------------------------------------------------------------------------

class TestElasticaQuadratureAccuracy:
    """The elliptic-type integrals must be converged, not merely consistent."""

    @pytest.mark.parametrize("theta_B", [0.1, 0.5, 1.0, 1.4])
    def test_against_adaptive_quadrature(self, theta_B):
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(theta_B)
        P_ref, delta_ref = _reference_integrals_quad(theta_B)

        assert abs(sol.P - P_ref) / P_ref < 1e-8, (
            f"P={sol.P:.10f} vs reference {P_ref:.10f} at theta_B={theta_B}"
        )
        assert abs(sol.y_tip - delta_ref) / delta_ref < 1e-8

    @pytest.mark.parametrize("theta_B", [0.3, 1.1])
    def test_against_independent_bvp(self, theta_B):
        """Direct BVP solve of the strong form (no elliptic integrals at all)."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(theta_B)
        P_bvp, delta_bvp = _reference_elastica_bvp(theta_B)

        assert abs(sol.P - P_bvp) / P_bvp < 1e-4, (
            f"P={sol.P:.8f} vs BVP {P_bvp:.8f} at theta_B={theta_B}"
        )
        assert abs(sol.y_tip - delta_bvp) / delta_bvp < 1e-3

    def test_linear_limit(self):
        exact = ElasticaExact(L=1.0, EI=1.0)
        result = exact.verify_linear_limit()
        assert result["error_pct"] < 0.01

    def test_free_end_angle_beyond_half_pi_rejected(self):
        """theta_B > pi/2 has no monotone first integral: must raise, not silently truncate."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        with pytest.raises(ValueError):
            exact.compute_from_theta_B(2.5)
        with pytest.raises(ValueError):
            exact.strain_energy(2.5)


class TestElasticaEnergyIntegrity:
    """Energy criteria that can actually fail."""

    def test_work_of_deformation_equals_strain_energy(self):
        res = verify_energy_conservation_elastica()
        assert np.all(res["error_pct"] < 1e-2), (
            f"|W - U|/U too large: {res['error_pct']}"
        )

    def test_P_delta_is_not_the_strain_energy(self):
        """The old criterion U = P*delta is wrong (it yields 50-70% 'errors')."""
        res = verify_energy_conservation_elastica()
        P_delta = res["P"] * res["delta"]
        ratio = res["U_strain"] / P_delta
        assert np.all(ratio < 0.55), (
            "U/(P delta) should be well below 1/2 at large deflection, got "
            f"{ratio}"
        )
        assert np.all(ratio > 0.0)

    def test_generalized_force_identity(self):
        """dU/d(delta) = P (the load is the conjugate force of the tip deflection)."""
        res = verify_energy_conservation_elastica()
        err = res["generalized_force_error_pct"]
        assert np.all(np.isfinite(err))
        assert np.all(err < 1e-3), f"dU/ddelta deviates from P: {err}%"


# ----------------------------------------------------------------------------
# von Karman plate: energy balance and closed-form law
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def balance():
    """Energy balance at w0/h = 1 (module-scoped: one FE run shared by 3 tests)."""
    return verify_energy_conservation_plate(
        w0_h_values=np.array([1.0]), n_elem=12, n_steps=8
    )


class TestPlateEnergyIntegrity:
    """Genuine energy balance on the converged finite element state."""

    def test_balance_is_small(self, balance):
        assert np.all(balance["converged"])
        assert np.all(balance["error_pct"] < 2.0), (
            f"|W_ext - U_FE|/U_FE = {balance['error_pct']}%"
        )

    def test_balance_is_not_identically_zero(self, balance):
        """A reconstructed identity (the previous implementation) would give exactly 0."""
        assert np.all(balance["error_pct"] > 0.0)

    def test_energy_split_is_positive(self, balance):
        assert np.all(balance["u_bending"] > 0.0)
        assert np.all(balance["u_membrane"] > 0.0)
        assert np.all(
            balance["u_total"]
            == pytest.approx(balance["u_bending"] + balance["u_membrane"], rel=1e-12)
        )

    def test_energy_gradient_equals_internal_force(self):
        """dU/du must reproduce the assembled internal force (times 2*pi)."""
        a, h, E, nu = 1.0, 0.01, 210e9, 0.3
        fem = VonKarmanPlateFEM(a=a, h=h, E=E, nu=nu, n_elem=10)
        ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)
        _, _, P_vals = ref.compute_load_deflection_curve(np.array([0.5]))
        res = fem.solve(float(P_vals[0]), n_steps=5, max_iter=60, tol=1e-10)
        assert res["converged"]

        u = res["u"]
        f_int = fem.assemble_internal_force(u)
        U0 = fem.strain_energy(u)
        eps = 1e-6 * max(1.0, np.abs(u).max())

        checked = 0
        for dof in range(u.size):
            if abs(f_int[dof]) < 1e-3 * np.abs(f_int).max():
                continue  # FD noise dominates for (near-)zero force components
            up = u.copy()
            up[dof] += eps
            um = u.copy()
            um[dof] -= eps
            dU = (fem.strain_energy(up) - fem.strain_energy(um)) / (2 * eps)
            expected = 2 * np.pi * f_int[dof]
            assert abs(dU - expected) <= 1e-4 * abs(expected) + 1e-9 * max(abs(U0), 1.0), (
                f"dof {dof}: dU/du={dU:.6f} vs 2*pi*f_int={expected:.6f}"
            )
            checked += 1
        assert checked >= 3, "not enough significant force components to test"


class TestPlateClosedFormLaw:
    def test_coefficient_single_source(self):
        for nu in (0.25, 0.3, 0.35):
            assert camp_berger_reference(np.array([1.0]), nu)["coefficient_c"] == pytest.approx(
                plate_cubic_coefficient(nu), abs=0.0
            )

    def test_coefficient_range_guard(self):
        with pytest.raises(ValueError):
            plate_cubic_coefficient(0.05)

    def test_law_verified_against_fem(self):
        """Closed-form law vs FE: deflection error must stay below 5% (aim <3%)."""
        res = full_verification_plate(
            w0_h_check=np.array([0.5, 2.0]), n_elem=12, n_steps=6
        )
        assert np.all(res["fem_converged"])
        assert np.all(res["P_error_pct"] < 5.0), res["P_error_pct"]
        assert np.all(res["P_error_pct"] < 3.0), res["P_error_pct"]
        assert res["pass_linear_limit"]
        assert res["pass_stiffening"]

    def test_convergence_order_is_computed(self):
        """The reported order must come from meshes, not from theory."""
        res = convergence_order_plate(n_elem_list=[8, 16, 32], n_steps=4)
        assert np.all(res["converged"])
        assert np.isfinite(res["convergence_order"])
        assert 1.5 < res["convergence_order"] < 3.5, res["convergence_order"]


class TestLoadPathWork:
    def test_work_integral_of_linear_path(self):
        """Sanity: for P = k w0 the dead-load work is k w0^2 / 2."""
        hist = [{"P": 10.0 * f, "w0": f} for f in np.linspace(0.1, 1.0, 10)]
        assert load_path_work(hist) == pytest.approx(0.5 * 10.0 * 1.0**2, rel=2e-3)

class TestVonMisesTrussLimitLoad:
    """D6 regression: limit load must be an interior maximum of the snap-through branch."""

    @pytest.mark.parametrize("rise_span", [0.1, 0.2, 0.3])
    def test_limit_load_matches_closed_form(self, rise_span):
        from tensorphysics.geometric_nonlinear.shallow_arch import (
            ShallowArchModel,
            VonMisesTruss,
        )

        L, EA = 10.0, 1e9
        H = rise_span * L
        res = VonMisesTruss(L=L, H=H, EA=EA).find_limit_points(n_pts=4000)

        assert res["upper_is_interior"], "limit load sits on the edge of the search window"
        assert res["q_limit_upper"] > 0.0, "downward load must be reported positive"

        closed_form = ShallowArchModel(L=L, H=H, EI=1e6, EA=EA, n_elem=10).analytical_limit_load()
        assert closed_form > 0.0
        assert abs(res["q_limit_upper"] - closed_form) / closed_form < 1e-3, (
            f"grid {res['q_limit_upper']:.6e} vs closed form {closed_form:.6e}"
        )

    def test_downward_load_sign(self):
        from tensorphysics.geometric_nonlinear.shallow_arch import VonMisesTruss

        truss = VonMisesTruss(L=10.0, H=1.0, EA=1e9)
        # apex pushed down by half the rise -> compressed bars, positive load
        q, _ = truss.equilibrium_path(np.array([0.3 * truss.H, 0.5 * truss.H]))
        assert q[0] > 0.0
        # apex pulled up -> bars in tension, load changes sign
        q_up, _ = truss.equilibrium_path(np.array([-0.5 * truss.H, -0.3 * truss.H]))
        assert q_up[0] < 0.0
        # a single-sample path must not raise (previous IndexError)
        q1, st1 = truss.equilibrium_path(np.array([0.5 * truss.H]))
        assert q1[0] > 0.0 and st1[0] == "unknown"
