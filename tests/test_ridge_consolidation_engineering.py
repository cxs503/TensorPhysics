"""Pytest suite for the simplified engineering ice-ridge consolidation model.

Tests the acceptance criteria of campaign node P46W3-N05:
  (a) C(t) = 1 - exp(-t/τ), τ(T) via Arrhenius
  (b) σ_c = σ_0 · C^m, m ∈ [1.5, 2.0]
  (c) R_ice = R_0 · (1 + k·C), k ≥ 0.5
  (d) Heat conduction residual ≤ 1%
  (e) Negative control: T → T_melt → C → 0 → σ_c → 0
"""

from __future__ import annotations

import math
import pytest

from tensorphysics.ice_mechanics.ridge_consolidation_engineering import (
    EngineeringConsolidationParams,
    arrhenius_time_constant,
    consolidation_degree_exponential,
    heat_conduction_residual_1d,
    resistance_linear_coupling,
    run_engineering_verification,
    strength_power_law,
    R_GAS,
    SIGMA_ICE_REF,
    T_MELT_SEA,
    R_ICE_REF,
)

DAY = 86_400.0


# ===========================================================================
# (a) Consolidation degree: exponential law with Arrhenius τ
# ===========================================================================

class TestArrheniusTimeConstant:
    """Criterion (a): Arrhenius time constant τ(T)."""

    def test_tau_decreases_with_colder_temperature(self):
        """Colder temperature → smaller τ → faster consolidation."""
        tau_m10 = arrhenius_time_constant(-10.0)
        tau_m20 = arrhenius_time_constant(-20.0)
        tau_m30 = arrhenius_time_constant(-30.0)
        assert tau_m30 < tau_m20 < tau_m10

    def test_tau_at_reference_temperature(self):
        """τ(T_ref) = τ_0 by construction."""
        tau_0 = 1.0e6
        tau = arrhenius_time_constant(-10.0, tau_0=tau_0, T_ref=-10.0)
        assert abs(tau - tau_0) / tau_0 < 1e-10

    def test_tau_infinite_at_melting_point(self):
        """τ → ∞ as T → T_melt (no freezing)."""
        tau = arrhenius_time_constant(T_MELT_SEA)
        assert tau == math.inf

    def test_tau_infinite_above_melting(self):
        """τ = ∞ for T ≥ T_melt."""
        tau = arrhenius_time_constant(0.0)
        assert tau == math.inf

    def test_arrhenius_activation_energy_effect(self):
        """Higher E_a → stronger temperature dependence."""
        tau_low_Ea = arrhenius_time_constant(-20.0, E_a=30e3)
        tau_high_Ea = arrhenius_time_constant(-20.0, E_a=70e3)
        # Higher E_a means more sensitivity to temperature
        # At T < T_ref, higher E_a → smaller τ (faster consolidation)
        assert tau_high_Ea < tau_low_Ea


class TestConsolidationDegreeExponential:
    """Criterion (a): C(t) = 1 - exp(-t/τ)."""

    def test_consolidation_starts_at_zero(self):
        """C(0) = 0."""
        C = consolidation_degree_exponential(0.0, -20.0)
        assert C == 0.0

    def test_consolidation_approaches_one(self):
        """C(t→∞) → 1."""
        C = consolidation_degree_exponential(1e10, -20.0)
        assert abs(C - 1.0) < 1e-10

    def test_consolidation_monotonic_in_time(self):
        """C(t) is monotonically increasing."""
        times = [0, 1, 5, 10, 20, 30, 60]
        C_values = [consolidation_degree_exponential(t * DAY, -20.0) for t in times]
        for i in range(len(C_values) - 1):
            assert C_values[i] < C_values[i + 1]

    def test_consolidation_zero_at_melting(self):
        """C = 0 when T ≥ T_melt (negative control)."""
        C = consolidation_degree_exponential(30 * DAY, T_MELT_SEA)
        assert C == 0.0

    def test_consolidation_zero_above_melting(self):
        """C = 0 for T > T_melt."""
        C = consolidation_degree_exponential(30 * DAY, 0.0)
        assert C == 0.0

    def test_consolidation_negative_time_raises(self):
        """Negative time raises ValueError."""
        with pytest.raises(ValueError):
            consolidation_degree_exponential(-1.0, -20.0)

    def test_colder_temperature_faster_consolidation(self):
        """At same time, colder T → higher C."""
        C_m10 = consolidation_degree_exponential(30 * DAY, -10.0)
        C_m20 = consolidation_degree_exponential(30 * DAY, -20.0)
        C_m30 = consolidation_degree_exponential(30 * DAY, -30.0)
        assert C_m30 > C_m20 > C_m10

    def test_exponential_form_verified(self):
        """Verify C(t) = 1 - exp(-t/τ) exactly."""
        T = -20.0
        t = 15 * DAY
        tau = arrhenius_time_constant(T)
        C = consolidation_degree_exponential(t, T)
        C_expected = 1.0 - math.exp(-t / tau)
        assert abs(C - C_expected) < 1e-14


# ===========================================================================
# (b) Strength-consolidation: power-law relation
# ===========================================================================

class TestStrengthPowerLaw:
    """Criterion (b): σ_c = σ_0 · C^m, m ∈ [1.5, 2.0]."""

    def test_strength_zero_at_zero_consolidation(self):
        """σ_c(0) = 0."""
        sigma = strength_power_law(0.0)
        assert sigma == 0.0

    def test_strength_full_at_full_consolidation(self):
        """σ_c(1) = σ_0."""
        sigma = strength_power_law(1.0)
        assert abs(sigma - SIGMA_ICE_REF) / SIGMA_ICE_REF < 1e-14

    def test_strength_monotonic_in_consolidation(self):
        """σ_c is monotonically increasing in C."""
        C_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        sigmas = [strength_power_law(C) for C in C_values]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] < sigmas[i + 1]

    def test_strength_exponent_range(self):
        """m ∈ [1.5, 2.0] produces physically reasonable strength."""
        for m in [1.5, 1.75, 2.0]:
            sigma_half = strength_power_law(0.5, m=m)
            # At C=0.5, σ_c/σ_0 = 0.5^m
            expected = 0.5 ** m
            assert abs(sigma_half / SIGMA_ICE_REF - expected) < 1e-14

    def test_higher_exponent_lower_strength_at_partial(self):
        """Higher m → lower strength at C < 1 (super-linear growth)."""
        sigma_m15 = strength_power_law(0.5, m=1.5)
        sigma_m175 = strength_power_law(0.5, m=1.75)
        sigma_m20 = strength_power_law(0.5, m=2.0)
        assert sigma_m15 > sigma_m175 > sigma_m20

    def test_invalid_consolidation_raises(self):
        """C outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            strength_power_law(-0.1)
        with pytest.raises(ValueError):
            strength_power_law(1.1)

    def test_invalid_exponent_raises(self):
        """m < 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            strength_power_law(0.5, m=0.5)


# ===========================================================================
# (c) Resistance-consolidation: linear coupling
# ===========================================================================

class TestResistanceLinearCoupling:
    """Criterion (c): R_ice = R_0 · (1 + k·C), k ≥ 0.5."""

    def test_resistance_baseline_at_zero_consolidation(self):
        """R_ice(0) = R_0."""
        R = resistance_linear_coupling(0.0)
        assert abs(R - R_ICE_REF) / R_ICE_REF < 1e-14

    def test_resistance_increases_with_consolidation(self):
        """R_ice is monotonically increasing in C."""
        C_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        R_values = [resistance_linear_coupling(C) for C in C_values]
        for i in range(len(R_values) - 1):
            assert R_values[i] < R_values[i + 1]

    def test_resistance_at_full_consolidation(self):
        """R_ice(1) = R_0 · (1 + k)."""
        k = 0.7
        R = resistance_linear_coupling(1.0, k=k)
        expected = R_ICE_REF * (1.0 + k)
        assert abs(R - expected) / expected < 1e-14

    def test_resistance_ratio_ge_1p5_at_full(self):
        """At C=1 with k≥0.5, R_ice/R_0 ≥ 1.5."""
        for k in [0.5, 0.7, 1.0]:
            R = resistance_linear_coupling(1.0, k=k)
            ratio = R / R_ICE_REF
            assert ratio >= 1.5 - 1e-14

    def test_k_minimum_constraint(self):
        """k < 0.5 raises ValueError."""
        with pytest.raises(ValueError):
            resistance_linear_coupling(0.5, k=0.3)

    def test_invalid_consolidation_raises(self):
        """C outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            resistance_linear_coupling(-0.1)
        with pytest.raises(ValueError):
            resistance_linear_coupling(1.1)

    def test_linear_form_verified(self):
        """Verify R = R_0 · (1 + k·C) exactly."""
        C = 0.6
        k = 0.8
        R = resistance_linear_coupling(C, k=k)
        expected = R_ICE_REF * (1.0 + k * C)
        assert abs(R - expected) < 1e-10


# ===========================================================================
# (d) Heat conduction residual
# ===========================================================================

class TestHeatConductionResidual:
    """Criterion (d): Heat conduction residual ≤ 1%."""

    def test_steady_state_residual_near_zero(self):
        """Steady-state linear profile has near-zero residual."""
        res = heat_conduction_residual_1d(-20.0, -5.0, 10.0, 2.0, n_points=100)
        assert res < 0.01  # ≤ 1%

    def test_residual_machine_precision_for_linear(self):
        """For linear profile, central differences are exact → residual ~ machine precision."""
        res_coarse = heat_conduction_residual_1d(-20.0, -5.0, 10.0, 2.0, n_points=10)
        res_fine = heat_conduction_residual_1d(-20.0, -5.0, 10.0, 2.0, n_points=1000)
        # For a linear profile, central differences are exact
        # Both should be at or near machine precision
        assert res_fine < 1e-8
        assert res_coarse < 1e-8

    def test_zero_temperature_difference(self):
        """ΔT = 0 → zero residual."""
        res = heat_conduction_residual_1d(-10.0, -10.0, 5.0, 2.0)
        assert res == 0.0

    def test_invalid_domain_length_raises(self):
        """L ≤ 0 raises ValueError."""
        with pytest.raises(ValueError):
            heat_conduction_residual_1d(-20.0, -5.0, 0.0, 2.0)
        with pytest.raises(ValueError):
            heat_conduction_residual_1d(-20.0, -5.0, -1.0, 2.0)

    def test_too_few_points_raises(self):
        """n_points < 3 raises ValueError."""
        with pytest.raises(ValueError):
            heat_conduction_residual_1d(-20.0, -5.0, 10.0, 2.0, n_points=2)


# ===========================================================================
# (e) Negative control
# ===========================================================================

class TestNegativeControl:
    """Criterion (e): T → T_melt → C → 0 → σ_c → 0."""

    def test_chain_T_melt_to_zero_strength(self):
        """Full chain: T → T_melt → C → 0 → σ_c → 0."""
        # At melting point, no consolidation
        C = consolidation_degree_exponential(30 * DAY, T_MELT_SEA)
        assert C == 0.0
        # Zero consolidation → zero strength
        sigma = strength_power_law(C)
        assert sigma == 0.0

    def test_chain_above_melting(self):
        """Above melting: T > T_melt → C = 0 → σ_c = 0."""
        C = consolidation_degree_exponential(30 * DAY, 0.0)
        assert C == 0.0
        sigma = strength_power_law(C)
        assert sigma == 0.0


# ===========================================================================
# Full verification suite
# ===========================================================================

class TestEngineeringVerificationSuite:
    """Run the full engineering verification suite."""

    def test_all_criteria_pass(self):
        """All acceptance criteria pass."""
        results = run_engineering_verification()
        assert results["overall_passed"] is True

    def test_criterion_a_passes(self):
        """Criterion (a): exponential consolidation passes."""
        results = run_engineering_verification()
        assert results["criterion_a_exponential_consolidation"]["passed"] is True

    def test_criterion_b_passes(self):
        """Criterion (b): power-law strength passes."""
        results = run_engineering_verification()
        assert results["criterion_b_power_law_strength"]["passed"] is True

    def test_criterion_c_passes(self):
        """Criterion (c): linear resistance coupling passes."""
        results = run_engineering_verification()
        assert results["criterion_c_linear_resistance"]["passed"] is True

    def test_criterion_d_passes(self):
        """Criterion (d): heat conduction residual ≤ 1%."""
        results = run_engineering_verification()
        assert results["criterion_d_heat_conduction"]["passed"] is True
        assert results["criterion_d_heat_conduction"]["residual"] <= 0.01

    def test_criterion_e_passes(self):
        """Criterion (e): negative control passes."""
        results = run_engineering_verification()
        assert results["criterion_e_negative_control"]["passed"] is True
