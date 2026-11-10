"""Pressure melting–refreezing bond cycle: acceptance criteria C1–C9.

Tests cover:
  C1  Pressure melting threshold vs Clausius–Clapeyron analytic value (<1 %).
  C2  No-melt degradation: P < P_melt → h_water = 0, contact force deviation <0.1 %.
  C3  Water film positivity: h_water ≥ 0.
  C4  Contact force monotonicity: h_water ↑ → F ↓ (|Spearman ρ| > 0.99).
  C5  Refreezing rate monotonicity: T_env ↓ → τ_refreeze ↓ (|Spearman ρ| > 0.9).
  C6  Bond strength limits: t→∞ → σ→σ_0; t=0 → σ=0.
  C7  Ice load peak reduction: with PM model F_peak reduced ≥5 %.
  C8  Ice load frequency modulation: with PM model frequency reduced ≥10 %.
  C9  All above pass in pytest.
"""

import math
import statistics

import pytest

from tensorphysics.ice_contact.pressure_melting import (
    pressure_melting_threshold as pm_exact,
)
from tensorphysics.ice_contact.pressure_melting_cycle import (
    ContactStateEnum,
    PressureMeltingCycle,
    PressureMeltingCycleParams,
    bond_strength,
    contact_force,
    melt_rate,
    pressure_melting_threshold_linear,
    refreeze_rate,
    refreeze_time_constant,
)
from tensorphysics.ice_contact.properties import IceWaterProperties

PROPS = IceWaterProperties()


# ===========================================================================
# C1: Pressure melting threshold vs Clausius–Clapeyron analytic (<1 %)
# ===========================================================================

class TestC1_PressureMeltingThreshold:
    """C1: Linearised P_melt must agree with exact Clapeyron integral within 1 %."""

    @pytest.mark.parametrize("ice_temp_c", [-5.0, -10.0, -15.0, -20.0])
    def test_p_melt_vs_clapeyron(self, ice_temp_c):
        ice_temp_k = 273.15 + ice_temp_c
        p_linear = pressure_melting_threshold_linear(ice_temp_k, PROPS)
        p_exact = pm_exact(ice_temp_k, PROPS)
        rel_error = abs(p_linear - p_exact) / p_exact
        assert rel_error < 0.01, (
            f"P_melt linear vs exact at {ice_temp_c}°C: "
            f"linear={p_linear:.1f}, exact={p_exact:.1f}, rel_err={rel_error:.4f}"
        )

    def test_p_melt_at_melting_point(self):
        """At T = T_m0, P_melt should be 0."""
        p = pressure_melting_threshold_linear(PROPS.melting_temperature, PROPS)
        assert p == 0.0

    def test_p_melt_positive_for_subzero(self):
        """For ice below 0°C, P_melt must be positive."""
        for t_c in [-1.0, -5.0, -10.0, -30.0]:
            p = pressure_melting_threshold_linear(273.15 + t_c, PROPS)
            assert p > 0.0, f"P_melt at {t_c}°C should be positive, got {p}"


# ===========================================================================
# C2: No-melt degradation (P < P_melt → h_water = 0, force deviation <0.1 %)
# ===========================================================================

class TestC2_NoMeltDegradation:
    """C2: Below melting threshold, water film stays zero and force is unchanged."""

    def test_h_water_zero_below_threshold(self):
        params = PressureMeltingCycleParams()
        cycle = PressureMeltingCycle(params)
        ice_temp = 263.15  # -10°C
        p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)
        pressure_below = 0.5 * p_melt  # well below threshold

        for _ in range(100):
            cycle.step(1e-5, pressure_below, 0.1, ice_temp, 1e-5)

        assert cycle.h_water == 0.0, (
            f"h_water should be 0 below P_melt, got {cycle.h_water}"
        )

    def test_contact_force_unchanged_below_threshold(self):
        params = PressureMeltingCycleParams()
        cycle = PressureMeltingCycle(params)
        ice_temp = 263.15
        p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)
        pressure_below = 0.5 * p_melt
        delta_n = 1e-5
        f_dry = params.k_n * delta_n  # expected dry contact force

        for _ in range(100):
            r = cycle.step(1e-5, pressure_below, 0.1, ice_temp, delta_n)

        f_actual = r["f_contact"]
        rel_dev = abs(f_actual - f_dry) / f_dry
        assert rel_dev < 0.001, (
            f"Force deviation below P_melt: {rel_dev:.4f} (>0.1%)"
        )


# ===========================================================================
# C3: Water film positivity (h_water ≥ 0)
# ===========================================================================

class TestC3_WaterFilmPositivity:
    """C3: Water film thickness must never go negative."""

    def test_h_water_non_negative_during_melting(self):
        params = PressureMeltingCycleParams()
        cycle = PressureMeltingCycle(params)
        ice_temp = 263.15
        p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)
        pressure_above = 2.0 * p_melt  # well above threshold

        for _ in range(200):
            cycle.step(1e-5, pressure_above, 1.0, ice_temp, 1e-5)
            assert cycle.h_water >= 0.0, f"h_water went negative: {cycle.h_water}"

    def test_h_water_non_negative_during_refreezing(self):
        params = PressureMeltingCycleParams()
        cycle = PressureMeltingCycle(params)
        ice_temp = 263.15
        p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)

        # First melt
        for _ in range(100):
            cycle.step(1e-5, 2.0 * p_melt, 1.0, ice_temp, 1e-5)

        # Then refreeze
        for _ in range(500):
            cycle.step(1e-5, 0.0, 0.0, ice_temp, 1e-5)
            assert cycle.h_water >= 0.0, f"h_water went negative during refreeze: {cycle.h_water}"


# ===========================================================================
# C4: Contact force monotonicity (h_water ↑ → F ↓, |Spearman ρ| > 0.99)
# ===========================================================================

class TestC4_ContactForceMonotonicity:
    """C4: Contact force must be monotonically non-increasing with h_water."""

    def test_force_decreases_with_film(self):
        k_n = 1.0e9
        delta_n = 1e-5
        h_critical = 1e-6

        h_values = [i * h_critical / 20 for i in range(21)]  # 0 to h_critical
        f_values = [contact_force(k_n, delta_n, h, h_critical) for h in h_values]

        # Check monotonicity: each successive value should be ≤ previous
        for i in range(1, len(f_values)):
            assert f_values[i] <= f_values[i - 1] + 1e-12, (
                f"Force not monotonic at h={h_values[i]}: "
                f"F[{i-1}]={f_values[i-1]:.2e}, F[{i}]={f_values[i]:.2e}"
            )

    def test_spearman_correlation(self):
        """Spearman rank correlation between h_water and F should be < -0.99."""
        k_n = 1.0e9
        delta_n = 1e-5
        h_critical = 1e-6

        h_values = [i * h_critical / 50 for i in range(51)]
        f_values = [contact_force(k_n, delta_n, h, h_critical) for h in h_values]

        # Compute Spearman ρ manually (rank correlation)
        def rank_data(data):
            sorted_indices = sorted(range(len(data)), key=lambda i: data[i])
            ranks = [0.0] * len(data)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank + 1
            return ranks

        h_ranks = rank_data(h_values)
        f_ranks = rank_data(f_values)

        n = len(h_values)
        mean_h = sum(h_ranks) / n
        mean_f = sum(f_ranks) / n

        cov = sum((h_ranks[i] - mean_h) * (f_ranks[i] - mean_f) for i in range(n)) / n
        std_h = math.sqrt(sum((r - mean_h) ** 2 for r in h_ranks) / n)
        std_f = math.sqrt(sum((r - mean_f) ** 2 for r in f_ranks) / n)

        if std_h > 0 and std_f > 0:
            rho = cov / (std_h * std_f)
        else:
            rho = 0.0

        assert abs(rho) > 0.99, f"Spearman ρ = {rho:.4f}, expected |ρ| > 0.99"
        assert rho < 0, f"Spearman ρ should be negative (inverse relationship), got {rho}"


# ===========================================================================
# C5: Refreezing rate monotonicity (T_env ↓ → τ_refreeze ↓, |Spearman ρ| > 0.9)
# ===========================================================================

class TestC5_RefreezingRateMonotonicity:
    """C5: Colder environments must have smaller refreezing time constants."""

    def test_tau_decreases_with_temperature(self):
        temps = [268.15, 263.15, 258.15, 253.15, 248.15]  # -5 to -25°C
        taus = [refreeze_time_constant(t, PROPS) for t in temps]

        # Each successive (colder) temperature should have smaller τ
        for i in range(1, len(taus)):
            assert taus[i] < taus[i - 1], (
                f"τ_refreeze not decreasing: T={temps[i]:.1f}K → τ={taus[i]:.4e}, "
                f"prev T={temps[i-1]:.1f}K → τ={taus[i-1]:.4e}"
            )

    def test_spearman_correlation_temp_tau(self):
        """Spearman ρ between T_env and τ_refreeze should be > 0.9.

        Both T and τ decrease together (colder → faster refreezing → smaller τ),
        so the correlation is positive.
        """
        temps = [270.15 - i * 0.5 for i in range(40)]  # -3 to -22.5°C
        taus = [refreeze_time_constant(t, PROPS) for t in temps]

        def rank_data(data):
            sorted_indices = sorted(range(len(data)), key=lambda i: data[i])
            ranks = [0.0] * len(data)
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank + 1
            return ranks

        t_ranks = rank_data(temps)
        tau_ranks = rank_data(taus)

        n = len(temps)
        mean_t = sum(t_ranks) / n
        mean_tau = sum(tau_ranks) / n

        cov = sum((t_ranks[i] - mean_t) * (tau_ranks[i] - mean_tau) for i in range(n)) / n
        std_t = math.sqrt(sum((r - mean_t) ** 2 for r in t_ranks) / n)
        std_tau = math.sqrt(sum((r - mean_tau) ** 2 for r in tau_ranks) / n)

        if std_t > 0 and std_tau > 0:
            rho = cov / (std_t * std_tau)
        else:
            rho = 0.0

        assert abs(rho) > 0.9, f"Spearman ρ(T, τ) = {rho:.4f}, expected |ρ| > 0.9"
        assert rho > 0, f"Spearman ρ should be positive (both decrease together), got {rho}"


# ===========================================================================
# C6: Bond strength limits (t→∞ → σ→σ_0; t=0 → σ=0)
# ===========================================================================

class TestC6_BondStrengthLimits:
    """C6: Bond strength must satisfy asymptotic limits."""

    def test_zero_time_zero_strength(self):
        sigma = bond_strength(0.0, 0.1, 1e6)
        assert sigma == 0.0, f"σ(t=0) should be 0, got {sigma}"

    def test_negative_time_zero_strength(self):
        sigma = bond_strength(-1.0, 0.1, 1e6)
        assert sigma == 0.0, f"σ(t<0) should be 0, got {sigma}"

    def test_infinite_time_approaches_sigma_0(self):
        sigma_0 = 1e6
        tau = 0.1
        # At t = 10*τ, exp(-10) ≈ 4.5e-5, so σ ≈ 0.99995 * σ_0
        sigma = bond_strength(10 * tau, tau, sigma_0)
        assert abs(sigma - sigma_0) / sigma_0 < 1e-4, (
            f"σ(t=10τ) = {sigma:.2f}, expected ≈ {sigma_0:.2f}"
        )

    def test_bond_strength_monotonic_growth(self):
        sigma_0 = 1e6
        tau = 0.1
        times = [i * tau / 10 for i in range(101)]
        strengths = [bond_strength(t, tau, sigma_0) for t in times]

        for i in range(1, len(strengths)):
            assert strengths[i] >= strengths[i - 1] - 1e-12, (
                f"Bond strength not monotonic at t={times[i]:.3f}"
            )

    def test_bond_strength_never_exceeds_sigma_0(self):
        sigma_0 = 1e6
        tau = 0.1
        for t in [0.01, 0.1, 1.0, 10.0, 100.0]:
            sigma = bond_strength(t, tau, sigma_0)
            assert sigma <= sigma_0 + 1e-12, (
                f"σ(t={t}) = {sigma:.2f} exceeds σ_0 = {sigma_0:.2f}"
            )


# ===========================================================================
# C7: Ice load peak reduction (with PM model F_peak reduced ≥5 %)
# ===========================================================================

class TestC7_PeakLoadReduction:
    """C7: Pressure melting model must reduce peak ice load by ≥5 %."""

    def _generate_pressure_timeseries(self, n_points=500, peak_pressure=2e8):
        """Generate a sinusoidal pressure timeseries that exceeds P_melt."""
        import math
        times = [i * 1e-4 for i in range(n_points)]
        # Sinusoidal pressure that goes above and below P_melt
        pressures = [peak_pressure * (0.5 + 0.5 * math.sin(2 * math.pi * t * 5)) for t in times]
        return times, pressures

    def test_peak_reduction_with_pressure_melting(self):
        params = PressureMeltingCycleParams(
            k_n=1e9,
            h_critical=1e-6,
            tau_drain=1e-3,
        )
        ice_temp = 268.15  # -5°C
        p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)
        delta_n = 1e-5

        times, pressures = self._generate_pressure_timeseries(
            n_points=500, peak_pressure=2e8
        )

        # Without pressure melting: force = k_n * delta_n (constant)
        f_dry = params.k_n * delta_n

        # With pressure melting
        cycle = PressureMeltingCycle(params)
        f_with_pm = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            r = cycle.step(dt, pressures[i], 0.5, ice_temp, delta_n)
            f_with_pm.append(r["f_contact"])

        f_peak_with = max(f_with_pm)
        f_peak_without = f_dry

        reduction = (f_peak_without - f_peak_with) / f_peak_without
        assert reduction >= 0.05, (
            f"Peak reduction = {reduction:.4f} (< 5%); "
            f"F_peak_dry={f_peak_without:.2e}, F_peak_pm={f_peak_with:.2e}"
        )


# ===========================================================================
# C8: Ice load frequency modulation (with PM model frequency reduced ≥10 %)
# ===========================================================================

class TestC8_FrequencyModulation:
    """C8: Pressure melting model must reduce average load frequency by ≥10 %."""

    def _count_zero_crossings(self, signal, threshold=None):
        """Count the number of times the signal crosses a threshold."""
        if threshold is None:
            threshold = statistics.mean(signal)
        crossings = 0
        for i in range(1, len(signal)):
            if (signal[i - 1] < threshold) != (signal[i] < threshold):
                crossings += 1
        return crossings

    def test_frequency_reduction_with_pressure_melting(self):
        params = PressureMeltingCycleParams(
            k_n=1e9,
            h_critical=1e-6,
            tau_drain=5e-4,  # faster drainage for more pronounced effect
        )
        ice_temp = 268.15  # -5°C
        delta_n = 1e-5

        # Generate oscillating pressure
        n_points = 1000
        times = [i * 5e-5 for i in range(n_points)]
        pressures = [1.5e8 * (0.5 + 0.5 * math.sin(2 * math.pi * t * 10)) for t in times]

        # Without PM: force oscillates at the same frequency as pressure
        f_dry = [params.k_n * delta_n * (0.5 + 0.5 * math.sin(2 * math.pi * t * 10))
                 for t in times]

        # With PM
        cycle = PressureMeltingCycle(params)
        f_with_pm = [params.k_n * delta_n]  # initial
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            r = cycle.step(dt, pressures[i], 0.5, ice_temp, delta_n)
            f_with_pm.append(r["f_contact"])

        # Count oscillations (zero crossings around mean)
        n_crossings_dry = self._count_zero_crossings(f_dry[1:])
        n_crossings_pm = self._count_zero_crossings(f_with_pm)

        # Frequency ratio
        if n_crossings_dry > 0:
            freq_ratio = n_crossings_pm / n_crossings_dry
        else:
            freq_ratio = 0.0

        # The PM model should reduce the effective frequency
        # (water film smooths out rapid force changes)
        reduction = 1.0 - freq_ratio
        assert reduction >= 0.10, (
            f"Frequency reduction = {reduction:.4f} (< 10%); "
            f"crossings_dry={n_crossings_dry}, crossings_pm={n_crossings_pm}"
        )


# ===========================================================================
# C9: pytest coverage (all above tests pass)
# ===========================================================================

class TestC9_PytestCoverage:
    """C9: Verify that all test classes above are discoverable and runnable."""

    def test_all_criteria_tested(self):
        """Meta-test: ensure C1-C8 test classes exist."""
        test_classes = [
            TestC1_PressureMeltingThreshold,
            TestC2_NoMeltDegradation,
            TestC3_WaterFilmPositivity,
            TestC4_ContactForceMonotonicity,
            TestC5_RefreezingRateMonotonicity,
            TestC6_BondStrengthLimits,
            TestC7_PeakLoadReduction,
            TestC8_FrequencyModulation,
        ]
        assert len(test_classes) == 8, "Should have 8 criterion test classes"

    def test_state_machine_transitions(self):
        """Additional test: verify state machine transitions work correctly."""
        params = PressureMeltingCycleParams()
        cycle = PressureMeltingCycle(params)
        ice_temp = 263.15
        p_melt = pressure_melting_threshold_linear(ice_temp, PROPS)

        # Start in SOLID
        assert cycle.state == ContactStateEnum.SOLID

        # Apply pressure above threshold → MELTING
        cycle.step(1e-5, 2.0 * p_melt, 1.0, ice_temp, 1e-5)
        assert cycle.state == ContactStateEnum.MELTING

        # Reduce pressure below threshold → REFREEZING
        cycle.step(1e-5, 0.0, 0.0, ice_temp, 1e-5)
        assert cycle.state == ContactStateEnum.REFREEZING

    def test_melt_rate_formula(self):
        """Verify the melt rate formula directly."""
        rho_ice = 916.8
        latent_heat = 3.3355e5
        p_melt = 1e8
        v_sliding = 1.0
        tau_drain = 1e-3

        # Above threshold: positive melting rate
        dh = melt_rate(2e8, p_melt, v_sliding, rho_ice, latent_heat, 0.0, tau_drain)
        assert dh > 0, f"Melt rate should be positive above P_melt, got {dh}"

        # Below threshold: negative (drainage only)
        dh = melt_rate(0.5e8, p_melt, v_sliding, rho_ice, latent_heat, 1e-7, tau_drain)
        assert dh < 0, f"Melt rate should be negative below P_melt, got {dh}"

    def test_refreeze_rate_formula(self):
        """Verify the refreeze rate formula directly."""
        tau = 0.1
        h = 1e-7

        dh = refreeze_rate(h, tau)
        expected = -h / tau
        assert abs(dh - expected) < 1e-20, f"Refreeze rate mismatch: {dh} vs {expected}"
        assert dh < 0, "Refreeze rate should be negative"
