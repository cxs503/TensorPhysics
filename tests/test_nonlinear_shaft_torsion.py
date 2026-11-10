"""
Comprehensive tests for nonlinear shaft torsional vibration analysis.

Acceptance criteria
-------------------
(a) δ=0 degenerates to linear solution: error < 0.1%
(b) Backlash nonlinearity signature consistent with theory (3rd harmonic enhancement)
(c) Random spectrum PSD error < 5%
(d) Two-segment shaft natural frequencies vs analytical: error < 2%
(e) Monte Carlo convergence standard error < 5%
(f) Zero load fatigue damage D = 0
(g) pytest ≥ 25 items
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorphysics.structural_dynamics.nonlinear_shaft_torsion import (
    BacklashSpring,
    backlash_torque_vector,
    backlash_stiffness_vector,
    RandomIceLoadSpectrum,
    ShaftSegment,
    MultiSegmentShaft,
    NonlinearNewmarkSolver,
    MonteCarloAnalyzer,
    MonteCarloResult,
    MinerFatigueDamage,
    NonlinearShaftTorsionAnalyzer,
    NonlinearShaftTorsionResult,
    sdof_backlash_degenerate_test,
    two_segment_analytical_comparison,
    backlash_subharmonic_check,
    backlash_nonlinearity_check,
    zero_load_fatigue_test,
)


# ─────────────────────────────────────────────────────────────
# (a) Backlash degenerate to linear (δ=0)
# ─────────────────────────────────────────────────────────────

class TestBacklashDegenerate:
    """Criterion (a): δ=0 degenerates to linear solution, error < 0.1%."""

    def test_backlash_zero_is_transparent(self):
        """Backlash spring with δ=0 should be transparent."""
        spring = BacklashSpring(k=100.0, delta=0.0)
        assert spring.is_linear()
        thetas = np.linspace(-0.1, 0.1, 21)
        for theta in thetas:
            assert spring.torque(theta) == 0.0
            assert spring.stiffness(theta) == 0.0

    def test_backlash_torque_positive(self):
        spring = BacklashSpring(k=100.0, delta=0.01)
        assert spring.torque(0.02) == pytest.approx(1.0, abs=1e-10)
        assert spring.torque(0.05) == pytest.approx(4.0, abs=1e-10)

    def test_backlash_torque_negative(self):
        spring = BacklashSpring(k=100.0, delta=0.01)
        assert spring.torque(-0.02) == pytest.approx(-1.0, abs=1e-10)

    def test_backlash_torque_deadzone(self):
        spring = BacklashSpring(k=100.0, delta=0.01)
        assert spring.torque(0.0) == 0.0
        assert spring.torque(0.005) == 0.0

    def test_backlash_stiffness(self):
        spring = BacklashSpring(k=100.0, delta=0.01)
        assert spring.stiffness(0.02) == 100.0
        assert spring.stiffness(0.0) == 0.0

    def test_backlash_vectorized(self):
        theta = np.array([-0.02, -0.005, 0.0, 0.005, 0.02])
        T = backlash_torque_vector(theta, k=100.0, delta=0.01)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(T, expected)

    def test_degenerate_error_less_than_01_percent(self):
        """Full SDOF test: δ=0 error < 0.1%."""
        result = sdof_backlash_degenerate_test(k=100.0, delta=0.0, F0=10.0, omega=1.0, n_cycles=50, pp_cycles=40)
        assert result["linear_error"] < 0.001, f"Degenerate error = {result['linear_error']:.2e}"
        assert result["all_converged"]


# ─────────────────────────────────────────────────────────────
# (b) Backlash nonlinearity signature
# ─────────────────────────────────────────────────────────────

class TestBacklashNonlinearity:
    """Criterion (b): backlash nonlinearity signature consistent with theory."""

    def test_nonlinearity_signature_present(self):
        """Nonlinear system should show enhanced 3rd harmonic."""
        result = backlash_nonlinearity_check(k=1000.0, delta=0.1, F0=500.0, n_cycles=200, pp_cycles=80)
        assert result["has_nonlinearity"], f"3rd harmonic ratio = {result['nonlinear_3rd_ratio']:.4f}"
        assert result["enhancement_factor"] > 10, f"Enhancement = {result['enhancement_factor']:.1f}x"
        assert result["all_converged"]

    def test_linear_no_nonlinearity_signature(self):
        """Linear system should not show significant 3rd harmonic."""
        result = backlash_nonlinearity_check(k=1000.0, delta=0.0, F0=500.0, n_cycles=200, pp_cycles=80)
        assert result["nonlinear_3rd_ratio"] < 0.001

    def test_subharmonic_check_runs(self):
        """Subharmonic check should run without error."""
        result = backlash_subharmonic_check(k=1000.0, delta=0.01, F0=50.0, omega_drive=2.0*np.sqrt(1000.0), n_cycles=100, pp_cycles=80)
        assert "subharmonic_ratio" in result
        assert "all_converged" in result


# ─────────────────────────────────────────────────────────────
# (c) Random spectrum PSD error
# ─────────────────────────────────────────────────────────────

class TestRandomSpectrumPSD:
    """Criterion (c): random spectrum PSD error < 5%."""

    def test_psd_error_less_than_5_percent(self):
        """PSD estimation error should be < 5%."""
        spectrum = RandomIceLoadSpectrum(T_mean=50000.0, n_blades=5, RPM=120.0, n_harmonics=3, seed=42, broadband_level=0.05)
        t = np.linspace(0, 100.0, 100000)
        error = spectrum.psd_error(t, n_segments=64)
        assert error < 0.05, f"PSD error = {error:.4f}"

    def test_bpf_frequency(self):
        spectrum = RandomIceLoadSpectrum(n_blades=5, RPM=120.0)
        assert spectrum.BPF == 10.0

    def test_time_history_shape(self):
        spectrum = RandomIceLoadSpectrum(seed=42)
        t = np.linspace(0, 1.0, 1000)
        T = spectrum.generate_time_history(t)
        assert T.shape == t.shape

    def test_time_history_mean(self):
        spectrum = RandomIceLoadSpectrum(T_mean=50000.0, seed=42)
        t = np.linspace(0, 10.0, 10000)
        T = spectrum.generate_time_history(t)
        assert abs(np.mean(T) - 50000.0) < 5000.0

    def test_psd_has_bpf_peaks(self):
        spectrum = RandomIceLoadSpectrum(n_blades=5, RPM=120.0, n_harmonics=3)
        freqs = np.linspace(0, 50, 1000)
        psd = spectrum.psd(freqs)
        bpf = spectrum.BPF
        for k in range(1, 4):
            idx = np.argmin(np.abs(freqs - k * bpf))
            assert psd[idx] > 0


# ─────────────────────────────────────────────────────────────
# (d) Two-segment shaft natural frequencies
# ─────────────────────────────────────────────────────────────

class TestTwoSegmentShaft:
    """Criterion (d): two-segment shaft frequencies vs analytical < 2%."""

    def test_two_segment_error_less_than_2_percent(self):
        result = two_segment_analytical_comparison(L1=6.0, D1=0.5, L2=6.0, D2=0.4, G=79.3e9, rho=7850.0)
        assert result["max_error"] < 0.02, f"Two-segment max error = {result['max_error']:.4f}"
        assert result["n_modes_matched"] >= 2

    def test_shaft_segment_properties(self):
        seg = ShaftSegment("test", L=1.0, D=0.5, d=0.0, G=79.3e9, rho=7850.0)
        J_expected = np.pi / 32.0 * 0.5**4
        assert abs(seg.J - J_expected) < 1e-10

    def test_shaft_segment_hollow(self):
        seg = ShaftSegment("hollow", L=1.0, D=0.5, d=0.3, G=79.3e9, rho=7850.0)
        J_expected = np.pi / 32.0 * (0.5**4 - 0.3**4)
        assert abs(seg.J - J_expected) < 1e-10

    def test_multi_segment_total_length(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=3.0, D=0.5), ShaftSegment("s2", L=4.0, D=0.4)])
        assert shaft.total_length == 7.0

    def test_multi_segment_n_nodes(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=3.0, D=0.5), ShaftSegment("s2", L=4.0, D=0.4)])
        assert shaft.n_segments == 2
        assert shaft.n_nodes == 3

    def test_multi_segment_matrices_shape(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=3.0, D=0.5), ShaftSegment("s2", L=4.0, D=0.4)])
        M, K = shaft.build_matrices()
        assert M.shape == (3, 3)
        assert K.shape == (3, 3)

    def test_multi_segment_matrices_symmetric(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=3.0, D=0.5), ShaftSegment("s2", L=4.0, D=0.4)])
        M, K = shaft.build_matrices()
        np.testing.assert_array_almost_equal(K, K.T)

    def test_transfer_matrix_identity_at_zero_freq(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=3.0, D=0.5)])
        T = shaft.transfer_matrix(0.0)
        np.testing.assert_array_almost_equal(T, np.eye(2), decimal=6)


# ─────────────────────────────────────────────────────────────
# (e) Monte Carlo convergence
# ─────────────────────────────────────────────────────────────

class TestMonteCarloConvergence:
    """Criterion (e): Monte Carlo standard error < 5%."""

    def test_mc_standard_error_less_than_5_percent(self):
        m = 1.0; k = 100.0; c = 0.02 * 2 * np.pi * np.sqrt(k / m)
        M = np.array([[m]]); C = np.array([[c]]); K = np.array([[k]])
        def solver_factory(seed=42): return NonlinearNewmarkSolver(M, C, K)
        t = np.linspace(0, 1.0, 300)
        mc = MonteCarloAnalyzer(solver_factory=solver_factory, spectrum_params={"T_mean": 50000.0, "n_blades": 5, "RPM": 120.0, "n_harmonics": 3, "broadband_level": 0.05}, t=t, n_samples=30)
        result = mc.run(dof_index=0)
        assert result.standard_error < 0.05, f"MC SE = {result.standard_error:.4f}"

    def test_mc_result_shape(self):
        m = 1.0; k = 100.0; c = 0.0
        M = np.array([[m]]); C = np.array([[c]]); K = np.array([[k]])
        def solver_factory(seed=42): return NonlinearNewmarkSolver(M, C, K)
        t = np.linspace(0, 1.0, 200)
        mc = MonteCarloAnalyzer(solver_factory=solver_factory, spectrum_params={"T_mean": 1000.0, "n_blades": 4, "RPM": 100.0}, t=t, n_samples=20)
        result = mc.run(dof_index=0)
        assert result.mean_response.shape == (len(t),)

    def test_mc_to_dict(self):
        result = MonteCarloResult(n_samples=50, mean_response=np.zeros(10), std_response=np.ones(10), max_response=np.ones(10), min_response=-np.ones(10), percentile_95=np.ones(10), percentile_5=-np.ones(10), standard_error=0.02)
        d = result.to_dict()
        assert d["n_samples"] == 50


# ─────────────────────────────────────────────────────────────
# (f) Zero load fatigue damage
# ─────────────────────────────────────────────────────────────

class TestZeroLoadFatigue:
    """Criterion (f): zero load gives D = 0."""

    def test_zero_load_damage_is_zero(self):
        assert zero_load_fatigue_test(dt=0.001, duration=1.0) == 0.0

    def test_miner_zero_stress(self):
        miner = MinerFatigueDamage(A=1e30, m=3.0, S_endurance=100e6)
        assert miner.compute_damage(np.zeros(1000), dt=0.001) == 0.0

    def test_miner_below_endurance(self):
        miner = MinerFatigueDamage(A=1e30, m=3.0, S_endurance=100e6)
        assert miner.compute_damage(np.full(1000, 50e6), dt=0.001) == 0.0

    def test_miner_cycles_to_failure(self):
        miner = MinerFatigueDamage(A=1e30, m=3.0, S_endurance=100e6)
        N = miner.cycles_to_failure(200e6)
        assert abs(N - 1e30 * (200e6)**(-3.0)) / (1e30 * (200e6)**(-3.0)) < 1e-10

    def test_miner_infinite_life(self):
        miner = MinerFatigueDamage(A=1e30, m=3.0, S_endurance=100e6)
        assert miner.cycles_to_failure(50e6) == np.inf


# ─────────────────────────────────────────────────────────────
# Nonlinear solver tests
# ─────────────────────────────────────────────────────────────

class TestNonlinearSolver:
    def test_solver_convergence(self):
        m = 1.0; k = 100.0; c = 0.0
        M = np.array([[m]]); C = np.array([[c]]); K = np.array([[k]])
        solver = NonlinearNewmarkSolver(M, C, K, backlash_springs={0: BacklashSpring(k, 0.01)})
        n_steps = 100; dt = 0.01; t = np.arange(n_steps + 1) * dt
        F_ext = np.zeros((n_steps + 1, 1)); F_ext[:, 0] = 10.0 * np.sin(2 * np.pi * 1.0 * t)
        result = solver.solve(t, F_ext)
        assert np.all(result["converged"])

    def test_solver_energy_bounded(self):
        m = 1.0; k = 100.0; c = 0.02 * 2 * np.pi * np.sqrt(k / m)
        M = np.array([[m]]); C = np.array([[c]]); K = np.array([[k]])
        solver = NonlinearNewmarkSolver(M, C, K, backlash_springs={0: BacklashSpring(k, 0.005)})
        n_steps = 500; dt = 0.005; t = np.arange(n_steps + 1) * dt
        F_ext = np.zeros((n_steps + 1, 1)); F_ext[:, 0] = 10.0 * np.sin(2 * np.pi * 1.5 * t)
        result = solver.solve(t, F_ext, x0=np.array([0.1]), v0=np.array([0.0]))
        energy = 0.5 * m * result["v"][:, 0]**2 + 0.5 * k * result["x"][:, 0]**2
        assert energy[0] > 0 and np.max(energy) < 100 * energy[0]

    def test_solver_2dof(self):
        M = np.array([[2.0, 0], [0, 1.0]])
        K = np.array([[100.0, -50.0], [-50.0, 50.0]])
        C = 0.02 * np.diag([2.0, 1.0]) * 2 * np.pi * np.sqrt(50.0)
        solver = NonlinearNewmarkSolver(M, C, K, backlash_springs={0: BacklashSpring(100.0, 0.01)})
        n_steps = 200; dt = 0.01; t = np.arange(n_steps + 1) * dt
        F_ext = np.zeros((n_steps + 1, 2)); F_ext[:, 0] = 10.0 * np.sin(2 * np.pi * 1.0 * t)
        result = solver.solve(t, F_ext)
        assert np.all(result["converged"])


# ─────────────────────────────────────────────────────────────
# High-level analyzer tests
# ─────────────────────────────────────────────────────────────

class TestNonlinearAnalyzer:
    def test_analyzer_linear_frequencies(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=6.0, D=0.5), ShaftSegment("s2", L=6.0, D=0.4)])
        analyzer = NonlinearShaftTorsionAnalyzer(shaft)
        freqs = analyzer.linear_frequencies(n_modes=4)
        # Filter out near-zero (rigid body) modes
        freqs_filtered = freqs[freqs > 0.01]
        assert len(freqs_filtered) >= 2
        assert np.all(freqs_filtered > 0)

    def test_analyzer_transient(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=6.0, D=0.5)])
        analyzer = NonlinearShaftTorsionAnalyzer(shaft)
        t = np.linspace(0, 1.0, 200)
        result = analyzer.transient_analysis(t, lambda t: 10000.0 * np.sin(2 * np.pi * 5.0 * t), dof_index=0)
        assert "x" in result and "v" in result

    def test_analyzer_monte_carlo(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=6.0, D=0.5)])
        analyzer = NonlinearShaftTorsionAnalyzer(shaft)
        t = np.linspace(0, 1.0, 200)
        mc_result = analyzer.monte_carlo_analysis(t, {"T_mean": 10000.0, "n_blades": 5, "RPM": 120.0, "n_harmonics": 3, "seed": 42}, n_samples=20, dof_index=0)
        assert mc_result.n_samples == 20

    def test_analyzer_fatigue(self):
        shaft = MultiSegmentShaft(segments=[ShaftSegment("s1", L=6.0, D=0.5)])
        analyzer = NonlinearShaftTorsionAnalyzer(shaft)
        t = np.linspace(0, 1.0, 200)
        D = analyzer.fatigue_analysis(150e6 * np.sin(2 * np.pi * 10.0 * t), t[1] - t[0])
        assert D >= 0


# ─────────────────────────────────────────────────────────────
# Acceptance criteria summary
# ─────────────────────────────────────────────────────────────

class TestAcceptanceCriteria:
    def test_all_criteria(self):
        results = {}

        # (a) δ=0 degenerate linear error < 0.1%
        deg_result = sdof_backlash_degenerate_test(k=100.0, delta=0.0, F0=10.0, omega=1.0, n_cycles=50, pp_cycles=40)
        results["degenerate_linear_error"] = deg_result["linear_error"]
        assert deg_result["linear_error"] < 0.001

        # (b) Nonlinearity signature
        nl_result = backlash_nonlinearity_check(k=1000.0, delta=0.1, F0=500.0, n_cycles=200, pp_cycles=80)
        results["nonlinearity_3rd_ratio"] = nl_result["nonlinear_3rd_ratio"]
        results["nonlinearity_enhancement"] = nl_result["enhancement_factor"]
        assert nl_result["has_nonlinearity"], f"3rd harmonic ratio = {nl_result['nonlinear_3rd_ratio']:.4f}"

        # (c) PSD error < 5%
        spectrum = RandomIceLoadSpectrum(T_mean=50000.0, n_blades=5, RPM=120.0, n_harmonics=3, seed=42, broadband_level=0.05)
        t_psd = np.linspace(0, 100.0, 100000)
        psd_error = spectrum.psd_error(t_psd, n_segments=64)
        results["psd_error"] = psd_error
        assert psd_error < 0.05

        # (d) Two-segment shaft error < 2%
        seg_result = two_segment_analytical_comparison(L1=6.0, D1=0.5, L2=6.0, D2=0.4, G=79.3e9, rho=7850.0)
        results["two_segment_max_error"] = seg_result["max_error"]
        assert seg_result["max_error"] < 0.02

        # (e) Monte Carlo standard error < 5%
        m = 1.0; k = 100.0; c = 0.02 * 2 * np.pi * np.sqrt(k / m)
        M_mc = np.array([[m]]); C_mc = np.array([[c]]); K_mc = np.array([[k]])
        def solver_factory(seed=42): return NonlinearNewmarkSolver(M_mc, C_mc, K_mc)
        t_mc = np.linspace(0, 1.0, 300)
        mc = MonteCarloAnalyzer(solver_factory=solver_factory, spectrum_params={"T_mean": 50000.0, "n_blades": 5, "RPM": 120.0}, t=t_mc, n_samples=20)
        mc_result = mc.run(dof_index=0)
        results["mc_standard_error"] = mc_result.standard_error
        assert mc_result.standard_error < 0.05

        # (f) Zero load damage = 0
        damage = zero_load_fatigue_test(dt=0.001, duration=1.0)
        results["zero_load_damage"] = damage
        assert damage == 0.0

        results["all_criteria_pass"] = True
        results["pytest_test_nonlinear_shaft_torsion"] = True

        import json
        results_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "nonlinear_shaft_torsion_acceptance.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n=== Acceptance Criteria Summary ===")
        print(f"(a) Degenerate linear error: {deg_result['linear_error']:.2e} < 0.1% ✓")
        print(f"(b) Nonlinearity 3rd ratio: {nl_result['nonlinear_3rd_ratio']:.4f}, enhancement: {nl_result['enhancement_factor']:.1f}x ✓")
        print(f"(c) PSD error: {psd_error:.4f} < 5% ✓")
        print(f"(d) Two-segment max error: {seg_result['max_error']:.4f} < 2% ✓")
        print(f"(e) MC standard error: {mc_result.standard_error:.4f} < 5% ✓")
        print(f"(f) Zero load damage: {damage} = 0 ✓")
        print(f"All criteria PASSED")
