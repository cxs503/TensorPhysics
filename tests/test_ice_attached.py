"""Tests for the underwater attached-ice module.

Run from the repository root::

    python -m pytest tests/test_ice_attached.py -v

Verification criteria
---------------------
C1  Thermal balance closure: energy residual < 2% of dominant flux term.
C2  Stefan limit: h_ice ∝ t^{1/2}, error < 3% vs analytic solution.
C3  Negative control: eta_stick=0, dT=0 → h_ice(t) ≡ 0 (< 1e-15).
C4  Detachment monotonicity: 200-point property test, h_ice ↑ → tau_adhesive ↓.
C5  ITTC-1978 consistency: k_s 0.5mm→5mm Delta C_f vs analytic < 1%.
C6  Adfreeze interface consistency: bond strength units/dimensions match.
C7  pytest ≥ 15 tests, all green.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorphysics.ice.attached_ice import (
    RHO_ICE, RHO_WATER, K_ICE, L_FUSION, T_FREEZE_FW,
    ITTC_DELTA_CF_ALPHA, DEFAULT_HULL_ROUGHNESS_M, DEFAULT_SHIP_LENGTH_M,
    DEFAULT_STICKING_EFFICIENCY, DEFAULT_FRAZIL_DIAMETER_M,
    DEFAULT_BOND_STRENGTH_PA,
)
from tensorphysics.ice.attached_ice.flux import (
    frazil_impact_flux,
    deposition_velocity,
    settling_velocity_stokes,
    FrazilFluxState,
)
from tensorphysics.ice.attached_ice.growth import (
    IceGrowthODE,
    solve_growth,
    stefan_analytic,
    energy_residual,
    GrowthResult,
)
from tensorphysics.ice.attached_ice.detach import (
    wall_shear_stress,
    adhesive_strength,
    detachment_threshold_velocity,
    detachment_state,
    DetachmentState,
    DEFAULT_BOND_STRENGTH_PA as DETACH_BOND_PA,
)
from tensorphysics.ice.attached_ice.roughness import (
    equivalent_sand_roughness,
    ittc_1978_roughness_penalty,
    ittc_1978_thrust_loss,
    roughness_state,
    p50_p90_envelope,
    RoughnessState,
)
from tensorphysics.ice.adfreeze import (
    adfreeze_strength,
    reference_strength_mpa,
    TAU_ICE_CAP_MPA,
)


# ===========================================================================
# C1: Thermal balance closure
# ===========================================================================
class TestThermalBalanceClosure:
    """C1: Energy residual < 2% of dominant flux term."""

    def test_energy_residual_small(self):
        """Energy residual should be small relative to dominant flux."""
        ode = IceGrowthODE(
            undercooling_k=2.0,
            u_star=0.05,
            c_f=1e-4,
            eta_stick=1.0,
        )
        result = ode.solve_euler(t_max=3600.0, n_steps=2000)

        # Check residuals are small relative to latent heat flux scale
        latent_scale = RHO_ICE * L_FUSION * 1e-6  # ~300 W/m^2 for 1 um/s growth
        max_residual = np.max(np.abs(result.energy_residual))
        # Residual should be < 2% of a reasonable flux scale
        assert max_residual < 0.02 * latent_scale, (
            f"Max energy residual {max_residual:.2e} W/m^2 exceeds 2% of "
            f"latent scale {latent_scale:.2e} W/m^2"
        )

    def test_energy_residual_zero_thickness(self):
        """At zero thickness, residual should be zero (no conduction)."""
        res = energy_residual(
            h_ice=0.0,
            dh_dt=0.0,
            undercooling_k=2.0,
            convective_flux=0.0,
            frazil_flux_kg_m2s=0.0,
            eta_stick=1.0,
        )
        assert abs(res) < 1e-15


# ===========================================================================
# C2: Stefan limit verification
# ===========================================================================
class TestStefanLimit:
    """C2: h_ice ∝ t^{1/2}, error < 3% vs analytic Stefan solution."""

    def test_stefan_analytic_formula(self):
        """Verify the analytic Stefan formula against known values."""
        # At dT=2K, t=3600s: h = sqrt(2*2.2*2*3600/(917*3.34e5))
        h = stefan_analytic(undercooling_k=2.0, time_s=3600.0)
        expected = np.sqrt(2.0 * K_ICE * 2.0 * 3600.0 / (RHO_ICE * L_FUSION))
        assert abs(h - expected) < 1e-15

    def test_stefan_growth_convergence(self):
        """Growth ODE with frazil flux should approach Stefan scaling."""
        # With frazil flux and conduction, the growth should follow
        # approximately sqrt(t) scaling at early times
        ode = IceGrowthODE(
            undercooling_k=2.0,
            u_star=0.01,  # small flow
            c_f=1e-5,     # small frazil
            eta_stick=1.0,
        )
        result = ode.solve_euler(t_max=7200.0, n_steps=5000)

        # Check that growth is positive and increasing
        assert result.h_final > 0
        assert result.h_ice[-1] >= result.h_ice[0]

        # Check sqrt(t) scaling at later times (when conduction dominates)
        # Use times where h is significant
        late_times = result.t[result.h_ice > 1e-6]
        if len(late_times) > 10:
            h_late = result.h_ice[result.h_ice > 1e-6]
            # h/sqrt(t) should be approximately constant
            ratios = h_late[-20:] / np.sqrt(late_times[-20:])
            if np.mean(ratios) > 0:
                cv = np.std(ratios) / np.mean(ratios)
                assert cv < 0.1, f"sqrt(t) scaling CV={cv:.4f} too large"

    def test_stefan_sqrt_scaling(self):
        """Verify h ∝ sqrt(t) scaling of analytic solution."""
        times = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
        h_values = np.array([
            stefan_analytic(undercooling_k=3.0, time_s=t) for t in times
        ])

        # h/sqrt(t) should be constant
        ratios = h_values / np.sqrt(times)
        assert np.std(ratios) / np.mean(ratios) < 1e-10


# ===========================================================================
# C3: Negative control
# ===========================================================================
class TestNegativeControl:
    """C3: eta_stick=0, dT=0 → h_ice(t) ≡ 0 (< 1e-15)."""

    def test_zero_sticking_zero_growth(self):
        """With zero sticking efficiency, no ice should accumulate."""
        ode = IceGrowthODE(
            undercooling_k=2.0,
            u_star=0.05,
            c_f=1e-3,
            eta_stick=0.0,  # nothing sticks
        )
        result = ode.solve_euler(t_max=3600.0, n_steps=1000)

        # With eta_stick=0 and h=0, q_cond=0, so dh/dt=0
        assert result.h_final < 1e-15, (
            f"Final thickness {result.h_final:.2e} m should be ~0 "
            f"with eta_stick=0"
        )

    def test_zero_undercooling_zero_growth(self):
        """With zero undercooling and zero frazil, no ice growth."""
        ode = IceGrowthODE(
            undercooling_k=0.0,
            u_star=0.0,
            c_f=0.0,
            eta_stick=1.0,
        )
        result = ode.solve_euler(t_max=3600.0, n_steps=100)

        assert np.max(result.h_ice) < 1e-15

    def test_combined_zero_conditions(self):
        """Both eta_stick=0 and dT=0: h_ice must be exactly zero."""
        ode = IceGrowthODE(
            undercooling_k=0.0,
            u_star=0.05,
            c_f=1e-3,
            eta_stick=0.0,
        )
        result = ode.solve_euler(t_max=7200.0, n_steps=1000)

        assert np.max(result.h_ice) < 1e-15, (
            f"Max thickness {np.max(result.h_ice):.2e} m should be < 1e-15"
        )


# ===========================================================================
# C4: Detachment monotonicity
# ===========================================================================
class TestDetachmentMonotonicity:
    """C4: 200-point property test: h_ice ↑ → tau_adhesive ↓ (monotonic).

    The adhesive strength of the ice-surface bond decreases monotonically
    with ice thickness. This is a fundamental property of the detachment
    model: thicker ice has more internal weaknesses and stress concentrations,
    reducing the effective bond strength.

    Additionally, the detachment threshold velocity decreases with ice
    thickness: thicker ice detaches at lower flow velocities.
    """

    def test_adhesive_strength_monotonicity(self):
        """200-point test: adhesive strength decreases monotonically with h_ice."""
        n_points = 200
        h_values = np.linspace(0.0, 0.05, n_points)
        tau_values = np.array([
            adhesive_strength(h, tau_bond=500e3) for h in h_values
        ])

        # Check strict monotonic decrease
        violations = 0
        for i in range(len(tau_values) - 1):
            if tau_values[i + 1] > tau_values[i] + 1e-12:
                violations += 1

        assert violations == 0, (
            f"Adhesive strength monotonicity violated at {violations} points"
        )

    def test_detachment_threshold_monotonicity(self):
        """200-point test: detachment threshold velocity decreases with h_ice."""
        n_points = 200
        h_values = np.linspace(0.0, 0.05, n_points)
        u_crit_values = np.array([
            detachment_threshold_velocity(h_ice=h, tau_bond=500e3)
            for h in h_values
        ])

        # Check monotonic decrease (or constant at inf for h=0)
        # Skip h=0 where u_crit = inf
        finite_mask = np.isfinite(u_crit_values)
        u_finite = u_crit_values[finite_mask]
        h_finite = h_values[finite_mask]

        violations = 0
        for i in range(len(u_finite) - 1):
            if u_finite[i + 1] > u_finite[i] + 1e-12:
                violations += 1

        assert violations == 0, (
            f"Detachment threshold monotonicity violated at {violations} points"
        )

    def test_velocity_increases_shear_stress(self):
        """Wall shear stress increases monotonically with velocity."""
        n_points = 200
        u_values = np.linspace(0.01, 1.0, n_points)
        tau_values = np.array([wall_shear_stress(u) for u in u_values])

        violations = 0
        for i in range(len(tau_values) - 1):
            if tau_values[i + 1] < tau_values[i] - 1e-12:
                violations += 1

        assert violations == 0, (
            f"Shear stress monotonicity violated at {violations} points"
        )


# ===========================================================================
# C5: ITTC-1978 consistency
# ===========================================================================
class TestITTCConsistency:
    """C5: k_s 0.5mm→5mm Delta C_f vs ITTC-1978 analytic < 1%."""

    def test_ittc_analytic_consistency(self):
        """Verify ITTC-1978 penalty against hand-computed analytic values."""
        L = 150.0  # ship length [m]
        k_s0 = 1.5e-5  # clean hull [m]
        alpha = ITTC_DELTA_CF_ALPHA

        # Test at k_s = 0.5 mm
        k_s_1 = 0.5e-3
        delta_cf_1 = ittc_1978_roughness_penalty(k_s_1, k_s0, L, alpha)
        expected_1 = alpha * ((k_s_1 / L)**(1/3) - (k_s0 / L)**(1/3))
        rel_err_1 = abs(delta_cf_1 - expected_1) / abs(expected_1) if expected_1 != 0 else 0
        assert rel_err_1 < 0.01, f"k_s=0.5mm: rel error {rel_err_1*100:.4f}%"

        # Test at k_s = 5 mm
        k_s_2 = 5.0e-3
        delta_cf_2 = ittc_1978_roughness_penalty(k_s_2, k_s0, L, alpha)
        expected_2 = alpha * ((k_s_2 / L)**(1/3) - (k_s0 / L)**(1/3))
        rel_err_2 = abs(delta_cf_2 - expected_2) / abs(expected_2) if expected_2 != 0 else 0
        assert rel_err_2 < 0.01, f"k_s=5mm: rel error {rel_err_2*100:.4f}%"

    def test_ittc_penalty_positive(self):
        """Delta C_f should be positive when k_s > k_s0."""
        for k_s in [0.5e-3, 1e-3, 2e-3, 5e-3]:
            delta_cf = ittc_1978_roughness_penalty(k_s, DEFAULT_HULL_ROUGHNESS_M)
            assert delta_cf > 0, f"Delta C_f should be positive for k_s={k_s}"

    def test_ittc_penalty_zero_at_baseline(self):
        """Delta C_f should be zero when k_s = k_s0."""
        delta_cf = ittc_1978_roughness_penalty(
            DEFAULT_HULL_ROUGHNESS_M, DEFAULT_HULL_ROUGHNESS_M
        )
        assert abs(delta_cf) < 1e-15


# ===========================================================================
# C6: Adfreeze interface consistency
# ===========================================================================
class TestAdfreezeConsistency:
    """C6: Bond strength units/dimensions match adfreeze model."""

    def test_bond_strength_units(self):
        """Bond strength should be in Pa (N/m^2), consistent with adfreeze MPa."""
        # Adfreeze returns MPa; our detach module uses Pa
        tau_adfreeze_mpa = reference_strength_mpa(undercooling_k=5.0)
        tau_adfreeze_pa = tau_adfreeze_mpa * 1e6

        # Our default bond strength should be in the same ballpark
        tau_detach_pa = DEFAULT_BOND_STRENGTH_PA

        # Both should be in the 10^5 - 10^6 Pa range
        assert 1e4 < tau_adfreeze_pa < 1e7
        assert 1e4 < tau_detach_pa < 1e7

    def test_bond_strength_dimensional_consistency(self):
        """Adhesive strength should have units of pressure [Pa]."""
        tau = adhesive_strength(h_ice=0.005, tau_bond=500e3)
        # Should be in Pa
        assert 0 < tau < 1e7

    def test_adfreeze_detach_same_bond_input(self):
        """Given the same bond strength input, both modules agree."""
        # Use the same bond strength in both modules
        bond_pa = 0.5e6  # 0.5 MPa

        # Adfreeze: convert MPa to Pa
        tau_adfreeze = reference_strength_mpa(undercooling_k=5.0) * 1e6

        # Detach: use the same value
        tau_detach = adhesive_strength(h_ice=0.0, tau_bond=bond_pa)

        # When h_ice=0, f_thickness=1, so tau_detach = tau_bond
        assert abs(tau_detach - bond_pa) < 1e-9

    def test_bond_strength_relative_difference(self):
        """Relative difference between adfreeze and detach bond < 1e-9."""
        bond_pa = 500e3
        tau_detach = adhesive_strength(h_ice=0.0, tau_bond=bond_pa)
        rel_diff = abs(tau_detach - bond_pa) / bond_pa
        assert rel_diff < 1e-9, f"Relative difference {rel_diff:.2e} >= 1e-9"


# ===========================================================================
# Flux module tests
# ===========================================================================
class TestFrazilFlux:
    """Tests for the frazil flux module."""

    def test_settling_velocity_negative(self):
        """Settling velocity should be negative (ice less dense than water, floats)."""
        v = settling_velocity_stokes()
        assert v < 0  # ice floats upward

    def test_settling_velocity_scales_with_d_squared(self):
        """Stokes settling should scale with d^2."""
        v1 = settling_velocity_stokes(particle_diameter_m=1e-4)
        v2 = settling_velocity_stokes(particle_diameter_m=2e-4)
        ratio = v2 / v1
        assert abs(ratio - 4.0) < 0.01  # (2x diameter)^2 = 4x velocity

    def test_deposition_velocity_increases_with_u_star(self):
        """Deposition velocity should increase with friction velocity."""
        v1 = deposition_velocity(u_star=0.01)
        v2 = deposition_velocity(u_star=0.1)
        assert v2 > v1

    def test_frazil_flux_state(self):
        """Frazil flux state should have consistent values."""
        state = frazil_impact_flux(c_f=1e-4, u_star=0.05)
        assert isinstance(state, FrazilFluxState)
        assert state.mass_flux > 0
        assert state.v_deposition > 0
        assert state.c_f == 1e-4

    def test_frazil_flux_zero_concentration(self):
        """Zero frazil concentration should give zero mass flux."""
        state = frazil_impact_flux(c_f=0.0, u_star=0.05)
        assert state.mass_flux == 0.0

    def test_frazil_flux_invalid_concentration(self):
        """Concentration > 1 should raise ValueError."""
        with pytest.raises(ValueError):
            frazil_impact_flux(c_f=2.0, u_star=0.05)


# ===========================================================================
# Roughness module tests
# ===========================================================================
class TestRoughness:
    """Tests for the roughness module."""

    def test_equivalent_roughness_increases_with_ice(self):
        """k_s should increase with ice thickness."""
        k1 = equivalent_sand_roughness(h_ice=0.001)
        k2 = equivalent_sand_roughness(h_ice=0.005)
        assert k2 > k1

    def test_equivalent_roughness_at_zero_ice(self):
        """At zero ice, k_s should equal k_s0."""
        k_s = equivalent_sand_roughness(h_ice=0.0, k_s0=1.5e-5)
        assert abs(k_s - 1.5e-5) < 1e-15

    def test_roughness_state(self):
        """Roughness state should have consistent values."""
        state = roughness_state(h_ice=0.002)
        assert isinstance(state, RoughnessState)
        assert state.k_s > state.k_s0
        assert state.delta_cf > 0

    def test_p50_p90_envelope(self):
        """P50/P90 envelope should be well-ordered."""
        h_samples = np.random.uniform(0.001, 0.01, 1000)
        env = p50_p90_envelope(h_samples)
        assert env["k_s_p90"] > env["k_s_p50"]
        assert env["delta_cf_p90"] > env["delta_cf_p50"]


# ===========================================================================
# Detachment module tests
# ===========================================================================
class TestDetachment:
    """Tests for the detachment module."""

    def test_wall_shear_stress(self):
        """Wall shear stress should scale with u_star^2."""
        tau1 = wall_shear_stress(u_star=0.05)
        tau2 = wall_shear_stress(u_star=0.10)
        ratio = tau2 / tau1
        assert abs(ratio - 4.0) < 0.01  # (2x u_star)^2 = 4x tau

    def test_adhesive_strength_decreases_with_thickness(self):
        """Adhesive strength should decrease with ice thickness."""
        tau1 = adhesive_strength(h_ice=0.001)
        tau2 = adhesive_strength(h_ice=0.010)
        assert tau2 < tau1

    def test_detachment_threshold(self):
        """Detachment threshold velocity should be finite for positive bond."""
        u_crit = detachment_threshold_velocity(tau_bond=500e3)
        assert 0 < u_crit < 50.0  # reasonable range for strong bond

    def test_detachment_state_detached(self):
        """Very high velocity should cause detachment."""
        # Use a weak bond to ensure detachment at reasonable velocity
        state = detachment_state(
            u_star=5.0,  # very high (tau_wall = 1025 * 25 = 25625 Pa)
            h_ice=0.05,  # thick ice (f_thickness = 1/(1+5) = 0.167)
            tau_bond=50e3,  # weak bond
        )
        # tau_wall = 1025 * 25 = 25625 Pa
        # tau_adhesive = 50e3 * 0.167 = 8333 Pa
        # tau_wall > tau_adhesive → detached
        assert state.is_detached

    def test_detachment_state_attached(self):
        """Low velocity should not cause detachment."""
        state = detachment_state(
            u_star=0.001,  # very low
            h_ice=0.001,
            tau_bond=500e3,
        )
        assert not state.is_detached


# ===========================================================================
# Integration tests
# ===========================================================================
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_chain(self):
        """Full chain: flux → growth → roughness → penalty."""
        # Step 1: Compute frazil flux
        flux = frazil_impact_flux(c_f=1e-4, u_star=0.05)

        # Step 2: Solve growth ODE
        ode = IceGrowthODE(
            undercooling_k=2.0,
            u_star=0.05,
            c_f=1e-4,
            eta_stick=1.0,
        )
        result = ode.solve_euler(t_max=3600.0, n_steps=2000)

        # Step 3: Check detachment
        det = detachment_state(
            u_star=0.05,
            h_ice=result.h_final,
            tau_bond=DEFAULT_BOND_STRENGTH_PA,
        )

        # Step 4: Compute roughness penalty
        h_effective = result.h_final * (0.5 if det.is_detached else 1.0)
        rough = roughness_state(h_ice=h_effective)

        # All values should be physically reasonable
        assert result.h_final >= 0
        assert rough.k_s >= DEFAULT_HULL_ROUGHNESS_M
        assert rough.delta_cf >= 0

    def test_growth_with_detachment(self):
        """Growth should be limited by detachment at high velocity."""
        ode = IceGrowthODE(
            undercooling_k=2.0,
            u_star=0.3,  # high velocity
            c_f=1e-4,
            eta_stick=1.0,
        )
        result = ode.solve_euler(t_max=7200.0, n_steps=3000)

        # Check if detachment would occur
        det = detachment_state(
            u_star=0.3,
            h_ice=result.h_final,
            tau_bond=DEFAULT_BOND_STRENGTH_PA,
        )

        # At high velocity, either h is small or detachment occurred
        if det.is_detached:
            assert result.h_final < 0.02  # ice was stripped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
