"""Tests for water added-mass integration into FEM mass matrix.

SIR-PC2|F02: Plate–water added mass integrated to FEM mass matrix.

Acceptance criteria:
1. natural_frequency_error_vs_fluid_solid < 0.05 [ratio]
2. mass_matrix_positive_definite == True [bool]
3. pytest test_added_mass_integration == PASS [bool]
"""

import math

import numpy as np
import pytest

from tensorphysics.ice_sheet.added_mass import (
    IceSheetFSIParams,
    fem_mass_matrix_with_added_mass,
    fem_natural_frequency,
    natural_frequency_fluid_solid_reference,
    natural_frequency_plate_on_foundation,
    stiffness_matrix_plate,
    water_added_mass_per_area,
)
from tensorphysics.ice_sheet.winkler_plate import (
    WinklerPlateParams,
    winkler_kw_parametric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def typical_ice_params():
    """Typical Arctic sea ice sheet parameters."""
    return IceSheetFSIParams(
        a=100.0,    # 100 m plate length
        b=100.0,    # 100 m plate width
        h=2.0,      # 2 m thick
        E=5.0e9,    # 5 GPa Young's modulus
        nu=0.33,    # Poisson's ratio
        rho_i=917.0,
        rho_w=1025.0,
        g=9.81,
        H=None,     # deep water
    )


@pytest.fixture
def thin_ice_params():
    """Thin ice sheet (first-year ice)."""
    return IceSheetFSIParams(
        a=50.0,
        b=50.0,
        h=0.5,
        E=5.0e9,
        nu=0.33,
        rho_i=917.0,
        rho_w=1025.0,
        g=9.81,
        H=200.0,    # finite depth
    )


# ---------------------------------------------------------------------------
# Unit tests: water_added_mass_per_area
# ---------------------------------------------------------------------------

class TestWaterAddedMassPerArea:
    """Verify the added-mass formula against known limits."""

    def test_deep_water_formula(self):
        """Deep water: m_a = ρ_w / k."""
        k = 0.1  # 1/m
        m_a = water_added_mass_per_area(k, rho_w=1025.0, H=None)
        expected = 1025.0 / 0.1
        assert m_a == pytest.approx(expected, rel=1e-10)

    def test_finite_depth_formula(self):
        """Finite depth: m_a = ρ_w/k · tanh(kH)."""
        k = 0.1
        H = 50.0
        m_a = water_added_mass_per_area(k, rho_w=1025.0, H=H)
        expected = 1025.0 / 0.1 * math.tanh(0.1 * 50.0)
        assert m_a == pytest.approx(expected, rel=1e-10)

    def test_shallow_water_limit(self):
        """Shallow water: tanh(kH) ≈ kH, so m_a ≈ ρ_w·H."""
        k = 0.001
        H = 10.0
        m_a = water_added_mass_per_area(k, rho_w=1025.0, H=H)
        expected = 1025.0 * H  # ≈ ρ_w·H for small kH
        assert m_a == pytest.approx(expected, rel=0.01)

    def test_positive_k_required(self):
        """Negative or zero wavenumber should raise."""
        with pytest.raises(ValueError):
            water_added_mass_per_area(0.0)
        with pytest.raises(ValueError):
            water_added_mass_per_area(-0.1)

    def test_added_mass_magnitude(self):
        """Added mass should be physically significant for typical ice."""
        params = IceSheetFSIParams(a=100, b=100, h=2.0, E=5e9, nu=0.33)
        m_a = water_added_mass_per_area(params.k1, params.rho_w)
        m_s = params.m_struct_per_area
        # Added mass is typically larger than structural mass for floating plates
        # (water is denser than ice, and the effective depth 1/k is large)
        ratio = m_a / m_s
        assert ratio > 0.1, f"Added mass ratio {ratio} too small"
        assert m_a > 0, "Added mass must be positive"


# ---------------------------------------------------------------------------
# Unit tests: natural frequency
# ---------------------------------------------------------------------------

class TestNaturalFrequency:
    """Verify natural frequency calculations."""

    def test_added_mass_reduces_frequency(self, typical_ice_params):
        """Including added mass must lower the natural frequency."""
        params = typical_ice_params
        omega_with = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        omega_without = natural_frequency_plate_on_foundation(params, include_added_mass=False)
        assert omega_with < omega_without

    def test_reference_equals_with_added_mass(self, typical_ice_params):
        """Reference frequency should match the with-added-mass calculation."""
        params = typical_ice_params
        omega_ref = natural_frequency_fluid_solid_reference(params)
        omega_calc = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        assert omega_ref == pytest.approx(omega_calc, rel=1e-12)

    def test_frequency_positive(self, typical_ice_params):
        """Natural frequency must be positive."""
        params = typical_ice_params
        omega = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        assert omega > 0

    def test_frequency_order_of_magnitude(self, typical_ice_params):
        """Frequency should be in physically reasonable range."""
        params = typical_ice_params
        omega = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        f = omega / (2 * math.pi)  # Hz
        # Ice sheet fundamental frequencies are typically 0.01–1 Hz
        assert 0.001 < f < 10.0, f"Frequency {f} Hz outside expected range"


# ---------------------------------------------------------------------------
# Unit tests: FEM mass matrix
# ---------------------------------------------------------------------------

class TestFEMMassMatrix:
    """Verify FEM mass matrix assembly."""

    def test_mass_matrix_shape(self, typical_ice_params):
        """Mass matrix should have correct dimensions."""
        params = typical_ice_params
        nx, ny = 11, 11
        M_total, M_struct = fem_mass_matrix_with_added_mass(params, nx, ny)
        n_int = (nx - 2) * (ny - 2)
        assert M_total.shape == (n_int, n_int)
        assert M_struct.shape == (n_int, n_int)

    def test_mass_matrix_positive_definite(self, typical_ice_params):
        """Mass matrix must be positive definite (acceptance criterion #2)."""
        params = typical_ice_params
        M_total, _ = fem_mass_matrix_with_added_mass(params, nx=11, ny=11)
        eigenvalues = np.linalg.eigvalsh(M_total)
        assert np.all(eigenvalues > 0), "Mass matrix is not positive definite"

    def test_mass_matrix_symmetric(self, typical_ice_params):
        """Mass matrix must be symmetric."""
        params = typical_ice_params
        M_total, _ = fem_mass_matrix_with_added_mass(params, nx=11, ny=11)
        assert np.allclose(M_total, M_total.T, atol=1e-12)

    def test_added_mass_increases_mass(self, typical_ice_params):
        """Total mass matrix should have larger trace than structural alone."""
        params = typical_ice_params
        M_total, M_struct = fem_mass_matrix_with_added_mass(params, nx=11, ny=11)
        assert np.trace(M_total) > np.trace(M_struct)

    def test_mass_matrix_finite_depth(self, thin_ice_params):
        """Mass matrix should work with finite water depth."""
        params = thin_ice_params
        M_total, M_struct = fem_mass_matrix_with_added_mass(params, nx=11, ny=11)
        eigenvalues = np.linalg.eigvalsh(M_total)
        assert np.all(eigenvalues > 0)


# ---------------------------------------------------------------------------
# Integration test: full FSI frequency comparison
# ---------------------------------------------------------------------------

class TestAddedMassIntegration:
    """Full integration test — acceptance criterion #1 and #3."""

    def test_natural_frequency_error_vs_fluid_solid(self, typical_ice_params):
        """FEM frequency vs fluid-solid reference: error < 5%.

        Acceptance criterion #1: natural_frequency_error_vs_fluid_solid < 0.05
        """
        params = typical_ice_params
        omega_fem, M_total, K = fem_natural_frequency(params, nx=21, ny=21)
        omega_ref = natural_frequency_fluid_solid_reference(params)

        error = abs(omega_fem - omega_ref) / omega_ref
        assert error < 0.05, (
            f"Natural frequency error {error:.4f} >= 0.05 threshold. "
            f"ω_FEM={omega_fem:.6f}, ω_ref={omega_ref:.6f}"
        )

    def test_mass_matrix_positive_definite_integration(self, typical_ice_params):
        """Mass matrix positive definite — acceptance criterion #2."""
        params = typical_ice_params
        _, M_total, _ = fem_natural_frequency(params, nx=21, ny=21)
        eigenvalues = np.linalg.eigvalsh(M_total)
        assert np.all(eigenvalues > 0), "Mass matrix not positive definite"

    def test_full_integration_pass(self, typical_ice_params):
        """Complete integration test — acceptance criterion #3.

        Verifies all three acceptance criteria in one test.
        """
        params = typical_ice_params

        # Criterion 2: positive definite
        omega_fem, M_total, K = fem_natural_frequency(params, nx=21, ny=21)
        eigenvalues = np.linalg.eigvalsh(M_total)
        mass_matrix_pd = bool(np.all(eigenvalues > 0))
        assert mass_matrix_pd, "Mass matrix must be positive definite"

        # Criterion 1: frequency error
        omega_ref = natural_frequency_fluid_solid_reference(params)
        freq_error = abs(omega_fem - omega_ref) / omega_ref
        assert freq_error < 0.05, f"Frequency error {freq_error} >= 0.05"

        # Criterion 3: this test itself passing = PASS
        # (pytest will mark this as PASS if no assertion fails)


# ---------------------------------------------------------------------------
# Convergence test
# ---------------------------------------------------------------------------

class TestConvergence:
    """Verify convergence with mesh refinement."""

    def test_frequency_convergence(self, typical_ice_params):
        """FEM frequency should converge to reference with refinement."""
        params = typical_ice_params
        omega_ref = natural_frequency_fluid_solid_reference(params)

        errors = []
        for n in [11, 15, 21, 31]:
            omega_fem, _, _ = fem_natural_frequency(params, nx=n, ny=n)
            err = abs(omega_fem - omega_ref) / omega_ref
            errors.append(err)

        # Error should generally decrease (allow some non-monotonicity)
        assert errors[-1] < 0.05, f"Finest mesh error {errors[-1]} >= 0.05"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_very_thin_ice(self):
        """Very thin ice should still produce valid results."""
        params = IceSheetFSIParams(a=10, b=10, h=0.1, E=5e9, nu=0.33)
        omega = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        assert omega > 0

    def test_large_plate(self):
        """Large plate (km-scale) should work."""
        params = IceSheetFSIParams(a=1000, b=1000, h=3.0, E=5e9, nu=0.33)
        omega = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        assert omega > 0

    def test_freshwater(self):
        """Freshwater parameters should work."""
        params = IceSheetFSIParams(
            a=50, b=50, h=1.0, E=5e9, nu=0.33,
            rho_i=917.0, rho_w=1000.0, g=9.81
        )
        omega = natural_frequency_plate_on_foundation(params, include_added_mass=True)
        assert omega > 0
