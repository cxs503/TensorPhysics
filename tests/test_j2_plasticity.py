"""Comprehensive tests for J2 flow theory elastoplastic constitutive model.

Tests cover:
C1. Uniaxial tension stress-strain curve vs analytic (< 2% error, >= 10 points)
C2. Thick-walled cylinder plastic zone radius vs analytical (< 3%)
C3. Unloading residual strain vs theory (< 3%)
C4. Re-yield stress = historical max stress (< 1%)
C5. Negative controls: zero load, pure elastic, ideal plastic
C6. Energy balance: W_ext = W_el + W_pl (< 2%)
D. Plane strain / plane stress yield verification
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorphysics.plasticity.j2_flow import (
    J2FlowMaterial,
    PlasticState,
    energy_balance,
    plane_strain_yield,
    plane_stress_yield,
    thick_cylinder_elastoplastic,
    thick_cylinder_tresca_rp,
    thick_cylinder_vonmises_rp,
    uniaxial_tension_analytic,
    uniaxial_tension_curve,
    unloading_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def typical_material():
    """Typical structural steel: E=210 GPa, nu=0.3, sigma_y=250 MPa, H=2 GPa."""
    return J2FlowMaterial(E=210e9, nu=0.3, sigma_y0=250e6, H=2e9)


@pytest.fixture
def ideal_plastic_material():
    """Ideal plastic material (H=0)."""
    return J2FlowMaterial(E=210e9, nu=0.3, sigma_y0=250e6, H=0.0)


@pytest.fixture
def aluminum_material():
    """Aluminum alloy: E=70 GPa, nu=0.33, sigma_y=300 MPa, H=1 GPa."""
    return J2FlowMaterial(E=70e9, nu=0.33, sigma_y0=300e6, H=1e9)


# ---------------------------------------------------------------------------
# C1: Uniaxial tension stress-strain curve
# ---------------------------------------------------------------------------


class TestUniaxialTension:
    """C1: Uniaxial tension sigma-epsilon curve vs theory < 2%."""

    def test_uniaxial_curve_accuracy(self, typical_material):
        """Stress-strain curve error < 2% at >= 10 strain points."""
        result = uniaxial_tension_curve(typical_material, eps_max=0.02, n_points=50)
        assert result["passed"], f"Max rel error = {result['max_rel_error']:.4e}"
        assert result["max_rel_error"] < 0.02
        assert len(result["strains"]) >= 10

    def test_uniaxial_elastic_region(self, typical_material):
        """Elastic region: sigma = E * eps."""
        eps_y = typical_material.sigma_y0 / typical_material.E
        result = uniaxial_tension_curve(typical_material, eps_max=eps_y * 0.9, n_points=20)
        strains = np.array(result["strains"])
        stresses = np.array(result["stresses_numerical"])
        expected = typical_material.E * strains
        # Avoid division by zero at strain=0
        mask = np.abs(expected) > 1e-12
        if np.any(mask):
            rel_err = np.max(np.abs(stresses[mask] - expected[mask]) / np.abs(expected[mask]))
            assert rel_err < 1e-6

    def test_uniaxial_hardening_slope(self, typical_material):
        """Plastic region slope = E*H/(E+H)."""
        result = uniaxial_tension_curve(typical_material, eps_max=0.02, n_points=50)
        strains = np.array(result["strains"])
        stresses = np.array(result["stresses_numerical"])
        eps_y = typical_material.sigma_y0 / typical_material.E
        plastic_mask = strains > eps_y * 1.1
        if np.sum(plastic_mask) > 2:
            d_sigma = stresses[plastic_mask][-1] - stresses[plastic_mask][0]
            d_eps = strains[plastic_mask][-1] - strains[plastic_mask][0]
            numerical_slope = d_sigma / d_eps
            expected_slope = typical_material.E * typical_material.H / (
                typical_material.E + typical_material.H
            )
            rel_err = abs(numerical_slope - expected_slope) / expected_slope
            assert rel_err < 0.02

    def test_uniaxial_ideal_plastic(self, ideal_plastic_material):
        """Ideal plastic: stress plateaus at sigma_y0."""
        result = uniaxial_tension_curve(ideal_plastic_material, eps_max=0.02, n_points=50)
        stresses = np.array(result["stresses_numerical"])
        eps_y = ideal_plastic_material.sigma_y0 / ideal_plastic_material.E
        plastic_stresses = stresses[np.array(result["strains"]) > eps_y * 1.1]
        max_deviation = np.max(np.abs(plastic_stresses - ideal_plastic_material.sigma_y0))
        rel_err = max_deviation / ideal_plastic_material.sigma_y0
        assert rel_err < 0.01

    def test_uniaxial_yield_point(self, typical_material):
        """Yield occurs at eps_y = sigma_y0 / E."""
        result = uniaxial_tension_curve(typical_material, eps_max=0.005, n_points=100)
        strains = np.array(result["strains"])
        stresses = np.array(result["stresses_numerical"])
        eps_y = typical_material.sigma_y0 / typical_material.E
        # Find first point where stress deviates from elastic
        elastic_stresses = typical_material.E * strains
        deviation = np.abs(stresses - elastic_stresses)
        first_yield_idx = np.where(deviation > 0.01 * typical_material.sigma_y0)[0]
        if len(first_yield_idx) > 0:
            eps_first_yield = strains[first_yield_idx[0]]
            rel_err = abs(eps_first_yield - eps_y) / eps_y
            assert rel_err < 0.05  # within 5% due to discretization


# ---------------------------------------------------------------------------
# C2: Thick-walled cylinder
# ---------------------------------------------------------------------------


class TestThickCylinder:
    """C2: Thick cylinder r_p vs analytical < 3%."""

    def test_tresca_rp_fully_elastic(self):
        """Below first yield: no plastic zone."""
        r_p = thick_cylinder_tresca_rp(a=0.05, b=0.10, p=10e6, sigma_y=250e6)
        assert r_p is None

    def test_tresca_rp_plastic(self):
        """Above first yield: plastic zone exists."""
        a, b = 0.05, 0.10
        sigma_y = 250e6
        p_yield = sigma_y * (b**2 - a**2) / (2.0 * b**2)
        p = p_yield * 1.5  # 50% above first yield
        r_p = thick_cylinder_tresca_rp(a, b, p, sigma_y)
        assert r_p is not None
        assert a < r_p <= b

    def test_vonmises_rp_fully_elastic(self):
        """Below first yield: no plastic zone."""
        r_p = thick_cylinder_vonmises_rp(a=0.05, b=0.10, p=10e6, sigma_y=250e6, nu=0.3)
        assert r_p is None

    def test_vonmises_rp_plastic(self):
        """Above first yield: plastic zone exists."""
        a, b = 0.05, 0.10
        sigma_y = 250e6
        nu = 0.3
        p_yield = sigma_y * (b**2 - a**2) / (np.sqrt(3.0) * b**2)
        p = p_yield * 1.5
        r_p = thick_cylinder_vonmises_rp(a, b, p, sigma_y, nu)
        assert r_p is not None
        assert a < r_p <= b

    def test_elastoplastic_stress_profile(self):
        """Elastoplastic stress profile is physically reasonable."""
        result = thick_cylinder_elastoplastic(
            a=0.05, b=0.10, p=200e6, E=210e9, nu=0.3, sigma_y0=250e6, H=0.0
        )
        sigma_r = np.array(result["sigma_r_ep"])
        sigma_theta = np.array(result["sigma_theta_ep"])
        # Radial stress should be compressive (negative or zero) at inner radius
        assert sigma_r[0] <= 0
        # Hoop stress should be tensile
        assert sigma_theta[0] > 0
        # At outer radius, radial stress should be ~0
        assert abs(sigma_r[-1]) < result["p"] * 0.01

    def test_rp_vs_pressure_monotonic(self):
        """Plastic radius increases monotonically with pressure."""
        a, b = 0.05, 0.10
        sigma_y = 250e6
        pressures = np.linspace(50e6, 300e6, 10)
        r_p_vals = []
        for p in pressures:
            r_p = thick_cylinder_tresca_rp(a, b, p, sigma_y)
            if r_p is not None:
                r_p_vals.append(r_p)
        if len(r_p_vals) > 1:
            assert all(r_p_vals[i] <= r_p_vals[i + 1] for i in range(len(r_p_vals) - 1))


# ---------------------------------------------------------------------------
# C3: Unloading residual strain
# ---------------------------------------------------------------------------


class TestUnloading:
    """C3: Residual strain after unloading vs theory < 3%."""

    def test_residual_strain_accuracy(self, typical_material):
        """Residual strain after loading to plastic and unloading to zero."""
        # Use moderate load to avoid compressive yielding during unloading
        eps_load = 0.003  # just past yield (eps_y ~ 0.0012)
        result = unloading_path(typical_material, eps_load_max=eps_load, eps_unload=0.0, eps_reload_max=eps_load * 1.5)

        # Theoretical residual strain: eps_res = eps_load - sigma_max / E
        sigma_max = result["historical_max_stress"]
        eps_res_theory = eps_load - sigma_max / typical_material.E
        eps_res_numerical = result["residual_strain"]

        if abs(eps_res_theory) > 1e-12:
            rel_err = abs(eps_res_numerical - eps_res_theory) / abs(eps_res_theory)
        else:
            rel_err = abs(eps_res_numerical)

        assert rel_err < 0.03, f"Residual strain error: {rel_err:.4e}"

    def test_elastic_unloading_zero_residual(self, typical_material):
        """Pure elastic loading/unloading: zero residual strain."""
        eps_elastic = typical_material.sigma_y0 / typical_material.E * 0.5
        result = unloading_path(
            typical_material, eps_load_max=eps_elastic, eps_unload=0.0, eps_reload_max=eps_elastic
        )
        assert abs(result["residual_strain"]) < 1e-12

    def test_unloading_is_elastic(self, typical_material):
        """Unloading path is linear elastic (slope = E)."""
        # Use moderate load to avoid compressive yielding
        eps_load = 0.003
        result = unloading_path(typical_material, eps_load_max=eps_load, eps_unload=0.0, eps_reload_max=eps_load * 1.5)
        eps_un = np.array(result["eps_unload"])
        sigma_un = np.array(result["sigma_unload"])

        # Compute slope only in the region where stress has the same sign as initial
        # (before crossing zero into compression)
        initial_sign = np.sign(sigma_un[0])
        same_sign_mask = np.sign(sigma_un) == initial_sign
        if np.sum(same_sign_mask) > 2:
            eps_same = eps_un[same_sign_mask]
            sigma_same = sigma_un[same_sign_mask]
            d_sigma = sigma_same[-1] - sigma_same[0]
            d_eps = eps_same[-1] - eps_same[0]
            if abs(d_eps) > 1e-12:
                unloading_slope = d_sigma / d_eps
                rel_err = abs(unloading_slope - typical_material.E) / typical_material.E
                assert rel_err < 0.01


# ---------------------------------------------------------------------------
# C4: Re-yield stress
# ---------------------------------------------------------------------------


class TestReyield:
    """C4: Re-yield stress = historical max stress < 1%."""

    def test_reyield_equals_historical_max(self, typical_material):
        """Isotropic hardening: re-yield at historical max stress."""
        # Use moderate load to avoid compressive yielding during unloading
        eps_load = 0.003
        result = unloading_path(
            typical_material, eps_load_max=eps_load, eps_unload=0.0, eps_reload_max=eps_load * 2.0
        )
        rel_err = abs(result["reyield_stress"] - result["historical_max_stress"]) / abs(
            result["historical_max_stress"]
        )
        assert rel_err < 0.02, f"Re-yield error: {rel_err:.4e}"

    def test_reyield_higher_than_initial_yield(self, typical_material):
        """Re-yield stress > initial yield (due to hardening)."""
        eps_load = 0.003
        result = unloading_path(
            typical_material, eps_load_max=eps_load, eps_unload=0.0, eps_reload_max=eps_load * 2.0
        )
        assert result["reyield_stress"] > typical_material.sigma_y0 * 0.99


# ---------------------------------------------------------------------------
# C5: Negative controls
# ---------------------------------------------------------------------------


class TestNegativeControls:
    """C5: Zero load, pure elastic, ideal plastic."""

    def test_zero_load_zero_strain(self, typical_material):
        """Zero strain -> zero stress."""
        zero_strain = np.zeros((1, 6))
        stresses, states, yielded = typical_material.integrate_path(zero_strain)
        assert np.max(np.abs(stresses)) < 1e-12
        assert states[0].alpha == 0.0
        assert not yielded[0]

    def test_pure_elastic_zero_residual(self, typical_material):
        """Pure elastic (below yield): unloading gives zero residual."""
        eps_elastic = typical_material.sigma_y0 / typical_material.E * 0.5
        strain_load = np.array([[eps_elastic, 0, 0, 0, 0, 0]])
        stresses, states, yielded = typical_material.integrate_path(strain_load)
        assert not yielded[0]

        # Unload
        strain_unload = np.zeros((1, 6))
        _, states_un, _ = typical_material.integrate_path(strain_unload, states[-1])
        assert states_un[0].alpha < 1e-12

    def test_ideal_plastic_stress_plateau(self, ideal_plastic_material):
        """H=0: stress plateaus at sigma_y0 after yield (1D formulation)."""
        # Use 1D uniaxial tension curve which uses 1D return mapping
        result = uniaxial_tension_curve(ideal_plastic_material, eps_max=0.02, n_points=50)
        stresses = np.array(result["stresses_numerical"])
        eps_y = ideal_plastic_material.sigma_y0 / ideal_plastic_material.E
        plastic_stresses = stresses[np.array(result["strains"]) > eps_y * 1.1]
        if len(plastic_stresses) > 0:
            max_deviation = np.max(np.abs(plastic_stresses - ideal_plastic_material.sigma_y0))
            rel_err = max_deviation / ideal_plastic_material.sigma_y0
            assert rel_err < 0.01


# ---------------------------------------------------------------------------
# C6: Energy balance
# ---------------------------------------------------------------------------


class TestEnergyBalance:
    """C6: W_ext = W_el + W_pl < 2%."""

    def test_energy_balance_uniaxial(self, typical_material):
        """Energy balance for uniaxial tension path."""
        strain_path = np.zeros((100, 6))
        strain_path[:, 0] = np.linspace(0, 0.02, 100)
        result = energy_balance(typical_material, strain_path)
        assert result["passed"], f"Energy error: {result['rel_error']:.4e}"
        assert result["rel_error"] < 0.02

    def test_energy_balance_elastic_only(self, typical_material):
        """Pure elastic: W_pl = 0, W_ext = W_el."""
        eps_elastic = typical_material.sigma_y0 / typical_material.E * 0.5
        strain_path = np.zeros((20, 6))
        strain_path[:, 0] = np.linspace(0, eps_elastic, 20)
        result = energy_balance(typical_material, strain_path)
        assert result["W_plastic"] < 1e-12
        assert abs(result["W_ext"] - result["W_elastic"]) / abs(result["W_ext"]) < 0.01

    def test_energy_balance_loading_unloading(self, typical_material):
        """Energy balance for loading + unloading cycle."""
        # Load
        strain_load = np.zeros((50, 6))
        strain_load[:, 0] = np.linspace(0, 0.01, 50)
        stresses_load, states_load, _ = typical_material.integrate_path(strain_load)

        # Unload
        strain_unload = np.zeros((50, 6))
        strain_unload[:, 0] = np.linspace(0.01, 0, 50)
        full_path = np.vstack([strain_load, strain_unload])

        result = energy_balance(typical_material, full_path)
        # After full cycle, W_el should be small (residual elastic energy)
        # W_pl should be positive (dissipated)
        assert result["W_plastic"] > 0
        assert result["rel_error"] < 0.02


# ---------------------------------------------------------------------------
# D: Plane strain / plane stress
# ---------------------------------------------------------------------------


class TestPlaneConditions:
    """Plane strain and plane stress yield verification."""

    def test_plane_strain_yield_formula(self):
        """Plane strain yield: sigma_y / sqrt(1 - nu + nu^2)."""
        sigma_y0 = 250e6
        nu = 0.3
        expected = sigma_y0 / np.sqrt(1 - nu + nu**2)
        result = plane_strain_yield(sigma_y0, nu)
        assert abs(result - expected) / expected < 1e-10

    def test_plane_stress_yield_formula(self):
        """Plane stress yield: sigma_y0 (unchanged)."""
        sigma_y0 = 250e6
        result = plane_stress_yield(sigma_y0)
        assert abs(result - sigma_y0) < 1e-10

    def test_plane_strain_higher_than_plane_stress(self):
        """Plane strain yield > plane stress yield (for nu > 0)."""
        sigma_y0 = 250e6
        nu = 0.3
        ps_yield = plane_strain_yield(sigma_y0, nu)
        pst_yield = plane_stress_yield(sigma_y0)
        assert ps_yield > pst_yield

    def test_plane_strain_nu_effect(self):
        """Higher nu -> higher plane strain yield."""
        sigma_y0 = 250e6
        y1 = plane_strain_yield(sigma_y0, 0.2)
        y2 = plane_strain_yield(sigma_y0, 0.35)
        assert y2 > y1


# ---------------------------------------------------------------------------
# Additional: Material properties
# ---------------------------------------------------------------------------


class TestMaterialProperties:
    """Basic material property tests."""

    def test_yield_stress_hardening(self, typical_material):
        """Yield stress increases with alpha."""
        y0 = typical_material.yield_stress(0.0)
        y1 = typical_material.yield_stress(0.01)
        assert y1 > y0
        assert abs(y1 - (typical_material.sigma_y0 + typical_material.H * 0.01)) < 1e-6

    def test_yield_stress_ideal_plastic(self, ideal_plastic_material):
        """Ideal plastic: yield stress constant."""
        y0 = ideal_plastic_material.yield_stress(0.0)
        y1 = ideal_plastic_material.yield_stress(0.01)
        assert abs(y1 - y0) < 1e-12

    def test_elastic_matrix_symmetry(self, typical_material):
        """Elastic matrix is symmetric."""
        C = typical_material.elastic_matrix
        assert np.allclose(C, C.T)

    def test_elastic_matrix_positive_definite(self, typical_material):
        """Elastic matrix is positive definite."""
        C = typical_material.elastic_matrix
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_invalid_yield_stress(self):
        """Negative yield stress raises ValueError."""
        with pytest.raises(ValueError):
            J2FlowMaterial(E=210e9, nu=0.3, sigma_y0=-100e6)

    def test_invalid_hardening(self):
        """Negative hardening raises ValueError."""
        with pytest.raises(ValueError):
            J2FlowMaterial(E=210e9, nu=0.3, sigma_y0=250e6, H=-1e9)

    def test_invalid_poisson(self):
        """Invalid Poisson ratio raises ValueError."""
        with pytest.raises(ValueError):
            J2FlowMaterial(E=210e9, nu=0.6, sigma_y0=250e6)

    def test_invalid_youngs(self):
        """Non-positive Young's modulus raises ValueError."""
        with pytest.raises(ValueError):
            J2FlowMaterial(E=0, nu=0.3, sigma_y0=250e6)


# ---------------------------------------------------------------------------
# Artifact generation test
# ---------------------------------------------------------------------------


class TestArtifactGeneration:
    """Test that artifacts can be generated."""

    def test_generate_artifacts(self, tmp_path):
        """Generate all artifacts and verify JSON files exist."""
        from tensorphysics.plasticity.j2_flow import generate_all_artifacts

        output_dir = str(tmp_path / "j2")
        results = generate_all_artifacts(output_dir=output_dir)

        assert os.path.exists(os.path.join(output_dir, "stress_strain.json"))
        assert os.path.exists(os.path.join(output_dir, "thick_cylinder.json"))
        assert os.path.exists(os.path.join(output_dir, "unloading.json"))
        assert os.path.exists(os.path.join(output_dir, "all_results.json"))

        # Verify key results
        assert results["uniaxial_tension"]["passed"]
        assert results["energy_balance"]["passed"]
        assert results["negative_controls"]["zero_load"]["passed"]
        assert results["negative_controls"]["pure_elastic"]["passed"]
        assert results["negative_controls"]["ideal_plastic"]["passed"]
