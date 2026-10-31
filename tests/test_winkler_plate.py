"""Tests for Winkler elastic foundation plate model (ice_sheet module).

Acceptance criteria (SIR-PC2|F01):
  - center_deflection_relative_error < 0.05  (ratio)
  - kw0_free_plate_error < 1e-10             (m)
  - pytest test_winkler_plate == PASS        (bool)
"""

import math

import numpy as np
import pytest

from tensorphysics.ice_sheet.winkler_plate import (
    WinklerPlateParams,
    navier_center_deflection,
    navier_deflection_field,
    solve_winkler_plate_fd,
    winkler_kw_parametric,
)


# ---------------------------------------------------------------------------
# 1. k_w parameterization
# ---------------------------------------------------------------------------

class TestKwParametric:
    """Verify Winkler modulus from buoyancy: k_w = ρ_w · g."""

    def test_default_seawater(self):
        kw = winkler_kw_parametric()
        expected = 1025.0 * 9.81
        assert abs(kw - expected) < 1e-10, f"k_w = {kw}, expected {expected}"

    def test_fresh_water(self):
        kw = winkler_kw_parametric(rho_w=1000.0, g=9.81)
        expected = 1000.0 * 9.81
        assert abs(kw - expected) < 1e-10

    def test_custom(self):
        kw = winkler_kw_parametric(rho_w=900.0, g=1.62)  # Moon ocean
        expected = 900.0 * 1.62
        assert abs(kw - expected) < 1e-10


# ---------------------------------------------------------------------------
# 2. Free plate (k_w = 0, q = 0) → zero deflection
# ---------------------------------------------------------------------------

class TestFreePlateZeroDeflection:
    """When k_w = 0 and q = 0 the plate must have zero deflection."""

    def test_kw0_q0_fd_error_below_threshold(self):
        params = WinklerPlateParams(
            a=100.0, b=100.0, h=2.0,
            E=5.0e9, nu=0.33,
            k_w=0.0, q=0.0,
        )
        _, _, w = solve_winkler_plate_fd(params, nx=21, ny=21)
        max_err = float(np.max(np.abs(w)))
        assert max_err < 1e-10, (
            f"kw0_free_plate_error = {max_err:.2e} m, threshold 1e-10 m"
        )

    def test_kw0_q0_navier_zero(self):
        params = WinklerPlateParams(
            a=100.0, b=100.0, h=2.0,
            E=5.0e9, nu=0.33,
            k_w=0.0, q=0.0,
        )
        wc = navier_center_deflection(params, n_terms=5)
        assert abs(wc) < 1e-15, f"Navier center deflection = {wc}"


# ---------------------------------------------------------------------------
# 3. Center deflection: FD vs Navier analytical
# ---------------------------------------------------------------------------

class TestCenterDeflectionAccuracy:
    """Center deflection relative error between FD and Navier < 5%."""

    @pytest.fixture()
    def ice_params(self):
        """Typical floating ice sheet parameters."""
        return WinklerPlateParams(
            a=100.0, b=100.0, h=2.0,
            E=5.0e9, nu=0.33,
            k_w=winkler_kw_parametric(),
            q=1000.0,
        )

    def test_center_deflection_relative_error(self, ice_params):
        w_nav = navier_center_deflection(ice_params, n_terms=51)
        nx = ny = 51
        _, _, w_fd_field = solve_winkler_plate_fd(ice_params, nx=nx, ny=ny)
        w_fd = float(w_fd_field[nx // 2, ny // 2])
        rel_err = abs(w_fd - w_nav) / abs(w_nav)
        assert rel_err < 0.05, (
            f"center_deflection_relative_error = {rel_err:.4f} (> 0.05); "
            f"w_nav = {w_nav:.6e} m, w_fd = {w_fd:.6e} m"
        )

    def test_navier_convergence(self, ice_params):
        """Navier series should converge as n_terms increases."""
        w5 = navier_center_deflection(ice_params, n_terms=5)
        w51 = navier_center_deflection(ice_params, n_terms=51)
        # More terms should give a different (more accurate) answer
        # but the change should be small
        assert abs(w5 - w51) / abs(w51) < 0.01

    def test_fd_boundary_zero(self, ice_params):
        """FD solution must satisfy simply-supported BCs (w=0 on boundary)."""
        _, _, w = solve_winkler_plate_fd(ice_params, nx=21, ny=21)
        assert np.max(np.abs(w[0, :])) < 1e-12
        assert np.max(np.abs(w[-1, :])) < 1e-12
        assert np.max(np.abs(w[:, 0])) < 1e-12
        assert np.max(np.abs(w[:, -1])) < 1e-12


# ---------------------------------------------------------------------------
# 4. Physical consistency checks
# ---------------------------------------------------------------------------

class TestPhysicalConsistency:
    """Sanity checks on physical behaviour."""

    def test_stiffer_foundation_less_deflection(self):
        """Increasing k_w should decrease center deflection."""
        base = WinklerPlateParams(
            a=100.0, b=100.0, h=2.0,
            E=5.0e9, nu=0.33,
            k_w=winkler_kw_parametric(), q=1000.0,
        )
        stiff = WinklerPlateParams(
            a=100.0, b=100.0, h=2.0,
            E=5.0e9, nu=0.33,
            k_w=3.0 * winkler_kw_parametric(), q=1000.0,
        )
        w_base = navier_center_deflection(base, n_terms=31)
        w_stiff = navier_center_deflection(stiff, n_terms=31)
        assert w_stiff < w_base, "Stiffer foundation must reduce deflection"

    def test_thicker_plate_less_deflection(self):
        """Increasing h (hence D) should decrease center deflection."""
        thin = WinklerPlateParams(
            a=100.0, b=100.0, h=1.0,
            E=5.0e9, nu=0.33,
            k_w=winkler_kw_parametric(), q=1000.0,
        )
        thick = WinklerPlateParams(
            a=100.0, b=100.0, h=3.0,
            E=5.0e9, nu=0.33,
            k_w=winkler_kw_parametric(), q=1000.0,
        )
        w_thin = navier_center_deflection(thin, n_terms=31)
        w_thick = navier_center_deflection(thick, n_terms=31)
        assert w_thick < w_thin, "Thicker plate must reduce deflection"

    def test_char_length_positive(self):
        params = WinklerPlateParams(
            a=100.0, b=100.0, h=2.0,
            E=5.0e9, nu=0.33,
            k_w=winkler_kw_parametric(), q=1000.0,
        )
        l = params.char_length
        assert l > 0
        # For typical ice: l ≈ 20-30 m
        assert 10.0 < l < 50.0, f"char_length = {l:.1f} m outside expected range"


# ---------------------------------------------------------------------------
# 5. Flexural rigidity
# ---------------------------------------------------------------------------

class TestFlexuralRigidity:
    """Verify D = Eh³/[12(1-ν²)]."""

    def test_known_value(self):
        params = WinklerPlateParams(
            a=1.0, b=1.0, h=1.0,
            E=12.0e9, nu=0.0,
            k_w=1.0, q=1.0,
        )
        # D = 12e9 * 1 / (12 * 1) = 1e9
        assert abs(params.D - 1.0e9) < 1e-3

    def test_poisson_effect(self):
        """D increases with ν (denominator 1-ν² decreases)."""
        p0 = WinklerPlateParams(1, 1, 1, 12e9, 0.0, 1, 1)
        p3 = WinklerPlateParams(1, 1, 1, 12e9, 0.3, 1, 1)
        assert p3.D > p0.D
