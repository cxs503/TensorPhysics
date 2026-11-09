"""Tests for detail fatigue: weld toe geometry → Kt → hot-spot extrapolation → S-N life.

Tests cover:
1. Kt analytical formulas (Nii, Maddox, fillet)
2. Hot-spot stress extrapolation (linear & quadratic)
3. FEM stress field surrogates
4. Mesh convergence analysis
5. Full detail fatigue assessment
6. Acceptance criteria validation
"""

import pytest
import numpy as np

from tensorphysics.fatigue.detail_fatigue import (
    WeldToeGeometry,
    kt_nii_t_joint,
    kt_maddox_butt_joint,
    kt_fillet_joint_approx,
    linear_extrapolation,
    quadratic_extrapolation,
    fem_stress_field_t_joint,
    fem_stress_field_butt_joint,
    fem_stress_field_fillet_joint,
    mesh_convergence_analysis,
    detail_fatigue_assessment,
)


# ============================================================================
# Kt analytical formula tests
# ============================================================================

class TestKtNiiTJoint:
    """Test Nii formula for T-joint Kt."""

    def test_basic_t_joint(self):
        """Basic T-joint Kt calculation."""
        t = 12.0
        r = 1.0
        a = 60.0
        kt = kt_nii_t_joint(t, r, a)
        assert kt > 1.0, "Kt must be > 1 for stress concentration"
        # Typical range for T-joint: 1.5 - 4.0
        assert 1.5 < kt < 5.0, f"Kt={kt} outside expected range"

    def test_radius_effect(self):
        """Kt decreases with increasing weld toe radius."""
        t = 12.0
        a = 60.0
        kt_small = kt_nii_t_joint(t, 0.5, a)
        kt_large = kt_nii_t_joint(t, 2.0, a)
        assert kt_small > kt_large, "Kt should decrease with larger radius"

    def test_attachment_length_effect(self):
        """Kt increases with attachment length."""
        t = 12.0
        r = 1.0
        kt_short = kt_nii_t_joint(t, r, 30.0)
        kt_long = kt_nii_t_joint(t, r, 90.0)
        assert kt_long > kt_short, "Kt should increase with longer attachment"

    def test_angle_effect(self):
        """Kt varies with weld angle."""
        t = 12.0
        r = 1.0
        a = 60.0
        kt_45 = kt_nii_t_joint(t, r, a, 45.0)
        kt_60 = kt_nii_t_joint(t, r, a, 60.0)
        # At 45°, f_theta = 1.0; at 60°, f_theta > 1.0
        assert kt_60 >= kt_45, "Kt should increase with deviation from 45°"

    def test_known_value(self):
        """Test against a known reference value."""
        # For t=12, r=1, a=60, theta=45:
        # a/t = 5, t/r = 12
        # Kt = 1 + 0.55 * 5^0.65 * 12^0.35 * 1.0
        expected = 1.0 + 0.55 * (5.0 ** 0.65) * (12.0 ** 0.35)
        kt = kt_nii_t_joint(12.0, 1.0, 60.0, 45.0)
        assert abs(kt - expected) < 0.001, f"Kt={kt} != expected={expected}"


class TestKtMaddoxButtJoint:
    """Test Maddox formula for butt joint Kt."""

    def test_basic_butt_joint(self):
        """Basic butt joint Kt calculation."""
        t = 12.0
        r = 1.0
        h = 3.0
        kt = kt_maddox_butt_joint(t, r, h)
        assert kt > 1.0, "Kt must be > 1 for stress concentration"
        # Typical range for butt joint: 1.2 - 4.0 (depends on h/r ratio)
        assert 1.2 < kt < 5.0, f"Kt={kt} outside expected range"

    def test_radius_effect(self):
        """Kt decreases with increasing weld toe radius."""
        t = 12.0
        h = 3.0
        kt_small = kt_maddox_butt_joint(t, 0.5, h)
        kt_large = kt_maddox_butt_joint(t, 2.0, h)
        assert kt_small > kt_large, "Kt should decrease with larger radius"

    def test_reinforcement_effect(self):
        """Kt increases with reinforcement height."""
        t = 12.0
        r = 1.0
        kt_low = kt_maddox_butt_joint(t, r, 1.0)
        kt_high = kt_maddox_butt_joint(t, r, 5.0)
        assert kt_high > kt_low, "Kt should increase with more reinforcement"

    def test_known_value(self):
        """Test against a known reference value."""
        # For t=12, r=1, h=3:
        # Kt = 1 + 2*sqrt(3/(2*1)) * (1 + 0.5*3/12)
        #    = 1 + 2*sqrt(1.5) * 1.125
        #    = 1 + 2*1.2247 * 1.125
        #    = 1 + 2.7557
        #    = 3.7557
        expected = 1.0 + 2.0 * np.sqrt(3.0 / 2.0) * (1.0 + 0.5 * 3.0 / 12.0)
        kt = kt_maddox_butt_joint(12.0, 1.0, 3.0)
        assert abs(kt - expected) < 0.001, f"Kt={kt} != expected={expected}"


class TestKtFilletJoint:
    """Test approximate Kt for fillet joints."""

    def test_basic_fillet_joint(self):
        """Basic fillet joint Kt calculation."""
        t = 12.0
        r = 1.0
        z = 8.0
        kt = kt_fillet_joint_approx(t, r, z)
        assert kt > 1.0, "Kt must be > 1 for stress concentration"

    def test_radius_effect(self):
        """Kt decreases with increasing weld toe radius."""
        t = 12.0
        z = 8.0
        kt_small = kt_fillet_joint_approx(t, 0.5, z)
        kt_large = kt_fillet_joint_approx(t, 2.0, z)
        assert kt_small > kt_large, "Kt should decrease with larger radius"


# ============================================================================
# Hot-spot stress extrapolation tests
# ============================================================================

class TestLinearExtrapolation:
    """Test linear hot-spot stress extrapolation."""

    def test_basic_linear(self):
        """Basic linear extrapolation."""
        # If stress decays linearly from 150 at 0.4t to 120 at 0.9t
        # σ_hs = 1.8 * 150 - 0.8 * 120 = 270 - 96 = 174
        result = linear_extrapolation(150.0, 120.0, 12.0)
        expected = 1.8 * 150.0 - 0.8 * 120.0
        assert abs(result.hot_spot_stress - expected) < 0.001

    def test_uniform_stress(self):
        """Uniform stress field should give same hot-spot stress."""
        result = linear_extrapolation(100.0, 100.0, 12.0)
        assert abs(result.hot_spot_stress - 100.0) < 0.001

    def test_method_label(self):
        """Method should be 'linear'."""
        result = linear_extrapolation(150.0, 120.0, 12.0)
        assert result.method == 'linear'


class TestQuadraticExtrapolation:
    """Test quadratic hot-spot stress extrapolation."""

    def test_basic_quadratic(self):
        """Basic quadratic extrapolation."""
        result = quadratic_extrapolation(150.0, 120.0, 100.0, 12.0)
        assert result.hot_spot_stress > 150.0, "Hot-spot should be higher than 0.4t stress"
        assert result.method == 'quadratic'

    def test_uniform_stress(self):
        """Uniform stress field should give same hot-spot stress."""
        result = quadratic_extrapolation(100.0, 100.0, 100.0, 12.0)
        assert abs(result.hot_spot_stress - 100.0) < 0.001

    def test_lagrange_coefficients(self):
        """Verify Lagrange interpolation coefficients."""
        # At x=0:
        # L1(0) = (0-0.9)(0-1.4) / ((0.4-0.9)(0.4-1.4)) = 1.26 / 0.5 = 2.52
        # L2(0) = (0-0.4)(0-1.4) / ((0.9-0.4)(0.9-1.4)) = 0.56 / (-0.25) = -2.24
        # L3(0) = (0-0.4)(0-0.9) / ((1.4-0.4)(1.4-0.9)) = 0.36 / 0.5 = 0.72
        # Check: L1 + L2 + L3 = 2.52 - 2.24 + 0.72 = 1.0 ✓
        result = quadratic_extrapolation(150.0, 120.0, 100.0, 12.0)
        expected = 150.0 * 2.52 + 120.0 * (-2.24) + 100.0 * 0.72
        assert abs(result.hot_spot_stress - expected) < 0.01


# ============================================================================
# FEM stress field surrogate tests
# ============================================================================

class TestFEMStressFieldTJoint:
    """Test FEM stress field surrogate for T-joint."""

    def test_stress_at_weld_toe(self):
        """Stress at weld toe should be Kt * nominal."""
        t = 12.0
        r = 1.0
        a = 60.0
        nominal = 100.0
        kt = kt_nii_t_joint(t, r, a)

        x_near = np.array([r * 0.01])
        stress = fem_stress_field_t_joint(x_near, nominal, t, r, a)

        # Should be close to Kt * nominal
        expected = nominal * kt
        error = abs(float(stress[0]) - expected) / expected * 100
        assert error < 5.0, f"Stress at weld toe error: {error:.2f}%"

    def test_stress_decay(self):
        """Stress should decay with distance from weld toe."""
        t = 12.0
        r = 1.0
        a = 60.0
        nominal = 100.0

        x_near = np.array([0.1])
        x_far = np.array([10.0])

        stress_near = fem_stress_field_t_joint(x_near, nominal, t, r, a)
        stress_far = fem_stress_field_t_joint(x_far, nominal, t, r, a)

        assert stress_near[0] > stress_far[0], "Stress should decay with distance"

    def test_far_field_stress(self):
        """Far field stress should approach nominal."""
        t = 12.0
        r = 1.0
        a = 60.0
        nominal = 100.0

        x_far = np.array([100.0])
        stress = fem_stress_field_t_joint(x_far, nominal, t, r, a)

        # Should be close to nominal
        error = abs(float(stress[0]) - nominal) / nominal * 100
        assert error < 1.0, f"Far field stress error: {error:.2f}%"


class TestFEMStressFieldButtJoint:
    """Test FEM stress field surrogate for butt joint."""

    def test_stress_at_weld_toe(self):
        """Stress at weld toe should be Kt * nominal."""
        t = 12.0
        r = 1.0
        h = 3.0
        nominal = 100.0
        kt = kt_maddox_butt_joint(t, r, h)

        x_near = np.array([r * 0.01])
        stress = fem_stress_field_butt_joint(x_near, nominal, t, r, h)

        expected = nominal * kt
        error = abs(float(stress[0]) - expected) / expected * 100
        assert error < 5.0, f"Stress at weld toe error: {error:.2f}%"


# ============================================================================
# Mesh convergence tests
# ============================================================================

class TestMeshConvergence:
    """Test mesh convergence analysis."""

    def test_convergence_t_joint(self):
        """T-joint should show mesh convergence."""
        t = 12.0
        r = 1.0
        a = 60.0
        nominal = 100.0

        result = mesh_convergence_analysis(
            t, r, nominal,
            attachment_length=a,
            joint_type='t_joint',
            tolerance=0.02,
        )

        assert len(result.element_sizes) >= 2
        assert len(result.hot_spot_stresses) == len(result.element_sizes)
        assert result.reference_stress > 0

    def test_convergence_butt_joint(self):
        """Butt joint should show mesh convergence."""
        t = 12.0
        r = 1.0
        h = 3.0
        nominal = 100.0

        result = mesh_convergence_analysis(
            t, r, nominal,
            reinforcement_height=h,
            joint_type='butt_joint',
            tolerance=0.02,
        )

        assert result.reference_stress > 0

    def test_custom_element_sizes(self):
        """Custom element sizes should be used."""
        t = 12.0
        r = 1.0
        a = 60.0
        nominal = 100.0
        custom_sizes = [6.0, 3.0, 1.5, 0.75]

        result = mesh_convergence_analysis(
            t, r, nominal,
            attachment_length=a,
            joint_type='t_joint',
            element_sizes=custom_sizes,
        )

        assert result.element_sizes == custom_sizes


# ============================================================================
# Full detail fatigue assessment tests
# ============================================================================

class TestDetailFatigueAssessment:
    """Test complete detail fatigue assessment."""

    def test_t_joint_assessment(self):
        """Full assessment for T-joint."""
        geometry = WeldToeGeometry(
            joint_type='t_joint',
            plate_thickness=12.0,
            weld_toe_radius=1.0,
            attachment_length=60.0,
        )

        result = detail_fatigue_assessment(
            geometry=geometry,
            nominal_stress=100.0,
            sn_curve_class='D',
            environment='seawater_cp',
        )

        assert result.kt_analytical > 1.0
        assert result.kt_from_fem > 1.0
        assert result.hot_spot_stress > 100.0
        assert result.n_to_failure > 0
        assert result.n_analytical > 0

    def test_butt_joint_assessment(self):
        """Full assessment for butt joint."""
        geometry = WeldToeGeometry(
            joint_type='butt_joint',
            plate_thickness=12.0,
            weld_toe_radius=1.0,
            reinforcement_height=3.0,
        )

        result = detail_fatigue_assessment(
            geometry=geometry,
            nominal_stress=100.0,
            sn_curve_class='D',
            environment='seawater_cp',
        )

        assert result.kt_analytical > 1.0
        assert result.n_to_failure > 0

    def test_fillet_joint_assessment(self):
        """Full assessment for fillet joint."""
        geometry = WeldToeGeometry(
            joint_type='fillet_joint',
            plate_thickness=12.0,
            weld_toe_radius=1.0,
            weld_leg_length=8.0,
        )

        result = detail_fatigue_assessment(
            geometry=geometry,
            nominal_stress=100.0,
            sn_curve_class='D',
            environment='seawater_cp',
        )

        assert result.kt_analytical > 1.0
        assert result.n_to_failure > 0


# ============================================================================
# Acceptance criteria tests
# ============================================================================

class TestAcceptanceCriteria:
    """Test that acceptance criteria are met."""

    def test_t_joint_kt_error_within_3pct(self):
        """Criterion (a): T-joint Kt error ≤ 3% vs Nii formula."""
        t = 12.0
        a = 60.0
        nominal = 100.0

        for r in [0.5, 1.0, 1.5, 2.0]:
            kt_analytical = kt_nii_t_joint(t, r, a)

            x_near = np.array([r * 0.01])
            stress_near = fem_stress_field_t_joint(x_near, nominal, t, r, a)
            kt_fem = float(stress_near[0] / nominal)

            error = abs(kt_fem - kt_analytical) / kt_analytical * 100
            assert error <= 3.0, f"T-joint r={r}mm: Kt error {error:.2f}% > 3%"

    def test_butt_joint_kt_error_within_3pct(self):
        """Criterion (b): Butt joint Kt error ≤ 3% vs Maddox formula."""
        t = 12.0
        h = 3.0
        nominal = 100.0

        for r in [0.5, 1.0, 1.5, 2.0]:
            kt_analytical = kt_maddox_butt_joint(t, r, h)

            x_near = np.array([r * 0.01])
            stress_near = fem_stress_field_butt_joint(x_near, nominal, t, r, h)
            kt_fem = float(stress_near[0] / nominal)

            error = abs(kt_fem - kt_analytical) / kt_analytical * 100
            assert error <= 3.0, f"Butt joint r={r}mm: Kt error {error:.2f}% > 3%"

    def test_mesh_convergence_2x_within_2pct(self):
        """Criterion (c): Hot-spot stress mesh convergence 2x refinement ≤ 2%."""
        t = 12.0
        r = 1.0
        a = 60.0
        nominal = 100.0

        result = mesh_convergence_analysis(
            t, r, nominal,
            attachment_length=a,
            joint_type='t_joint',
            tolerance=0.02,
        )

        # Check last two meshes (2x refinement)
        if len(result.hot_spot_stresses) >= 2:
            change = abs(result.hot_spot_stresses[-1] - result.hot_spot_stresses[-2])
            change_pct = change / result.reference_stress * 100
            assert change_pct <= 2.0, f"2x refinement change {change_pct:.2f}% > 2%"

    def test_fatigue_life_error_within_15pct(self):
        """Criterion (d): Fatigue life N error ≤ 15% vs S-N analytical."""
        geometry = WeldToeGeometry(
            joint_type='t_joint',
            plate_thickness=12.0,
            weld_toe_radius=1.0,
            attachment_length=60.0,
        )

        result = detail_fatigue_assessment(
            geometry=geometry,
            nominal_stress=100.0,
            sn_curve_class='D',
            environment='seawater_cp',
        )

        assert result.n_error_pct <= 15.0, f"Fatigue life error {result.n_error_pct:.2f}% > 15%"


# ============================================================================
# WeldToeGeometry validation tests
# ============================================================================

class TestWeldToeGeometry:
    """Test WeldToeGeometry dataclass."""

    def test_valid_t_joint(self):
        """Valid T-joint geometry."""
        geo = WeldToeGeometry(
            joint_type='t_joint',
            plate_thickness=12.0,
            weld_toe_radius=1.0,
            attachment_length=60.0,
        )
        warnings = geo.validate()
        assert len(warnings) == 0

    def test_missing_attachment_length(self):
        """T-joint without attachment length should warn."""
        geo = WeldToeGeometry(
            joint_type='t_joint',
            plate_thickness=12.0,
            weld_toe_radius=1.0,
        )
        warnings = geo.validate()
        assert any('attachment_length' in w for w in warnings)

    def test_invalid_radius(self):
        """Zero radius should warn."""
        geo = WeldToeGeometry(
            joint_type='t_joint',
            plate_thickness=12.0,
            weld_toe_radius=0.0,
            attachment_length=60.0,
        )
        warnings = geo.validate()
        assert any('weld_toe_radius' in w for w in warnings)

    def test_radius_exceeds_thickness(self):
        """Radius > thickness should warn."""
        geo = WeldToeGeometry(
            joint_type='t_joint',
            plate_thickness=12.0,
            weld_toe_radius=15.0,
            attachment_length=60.0,
        )
        warnings = geo.validate()
        assert any('exceed' in w for w in warnings)
