"""
Comprehensive tests for shaft torsional vibration analysis.

Acceptance criteria
-------------------
(a) First 4 modes vs analytical 66.216/198.647/331.078/463.509 Hz → error < 1%
(b) Convergence order ≥ 1.8 (≈ N⁻²)
(c) Transient energy drift < 1e-6 (10⁴ steps)
(d) SDOF resonance Q-factor error < 5% (ζ = 0.02, 0.05, 0.1)
(e) Harmonic response vs analytical |H(ω)| < 3%
(f) All tests PASS
"""

import sys
import os
import numpy as np
import pytest
import json

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorphysics.structural_dynamics.shaft_torsion import (
    UniformShaftAnalytical,
    LumpedMassSpringModel,
    build_uniform_shaft_model,
    NewmarkBetaSolver,
    IceMillingExcitation,
    ShaftTorsionAnalyzer,
    sdof_resonance_test,
    sdof_frequency_response,
)


# ─────────────────────────────────────────────────────────────
# Reference parameters (matching target frequencies)
# ─────────────────────────────────────────────────────────────
G_REF = 79.3e9       # Pa
RHO_REF = 7850.0     # kg/m³
L_REF = 12.0         # m
D_REF = 0.5          # m

# Target analytical frequencies [Hz]
TARGET_FREQS = [66.216, 198.647, 331.078, 463.509]


# ─────────────────────────────────────────────────────────────
# (a) Modal analysis: first 4 modes vs analytical
# ─────────────────────────────────────────────────────────────

class TestModalAnalysis:
    """Criterion (a): first 4 modes vs analytical < 1% error."""

    def test_analytical_frequencies(self):
        """Verify analytical formula gives target frequencies."""
        shaft = UniformShaftAnalytical(G=G_REF, rho=RHO_REF, L=L_REF)
        freqs = shaft.frequencies(4)
        for i, (f_num, f_target) in enumerate(zip(freqs, TARGET_FREQS)):
            rel_err = abs(f_num - f_target) / f_target
            assert rel_err < 0.001, (
                f"Analytical f_{i+1} = {f_num:.3f} Hz, target = {f_target:.3f} Hz, "
                f"rel_err = {rel_err:.2e}"
            )

    def test_numerical_modes_vs_analytical(self):
        """Lumped model (N=50) first 4 modes vs analytical < 1%."""
        analyzer = ShaftTorsionAnalyzer(
            G=G_REF, rho=RHO_REF, L=L_REF, D=D_REF,
            propeller_J=0.0, engine_J=0.0,
        )
        freqs_ana, freqs_num = analyzer.modal_analysis(N=50, n_modes=4)

        for i in range(4):
            rel_err = abs(freqs_num[i] - freqs_ana[i]) / freqs_ana[i]
            assert rel_err < 0.01, (
                f"Mode {i+1}: numerical={freqs_num[i]:.4f} Hz, "
                f"analytical={freqs_ana[i]:.4f} Hz, rel_err={rel_err:.4f}"
            )

    def test_numerical_modes_match_targets(self):
        """Direct check against target values."""
        shaft = UniformShaftAnalytical(G=G_REF, rho=RHO_REF, L=L_REF)
        model = build_uniform_shaft_model(shaft, N=50, propeller_J=0.0, engine_J=0.0)
        freqs_num, _ = model.solve_eigenvalue(n_modes=4)

        for i in range(4):
            rel_err = abs(freqs_num[i] - TARGET_FREQS[i]) / TARGET_FREQS[i]
            assert rel_err < 0.01, (
                f"Mode {i+1}: got {freqs_num[i]:.4f} Hz, "
                f"target {TARGET_FREQS[i]:.3f} Hz, rel_err={rel_err:.4f}"
            )


# ─────────────────────────────────────────────────────────────
# (b) Convergence order
# ─────────────────────────────────────────────────────────────

class TestConvergence:
    """Criterion (b): convergence order ≥ 1.8."""

    def test_convergence_order(self):
        """Error vs N should show ~N⁻² convergence (order ≥ 1.8)."""
        shaft = UniformShaftAnalytical(G=G_REF, rho=RHO_REF, L=L_REF)
        freqs_ana = shaft.frequencies(4)

        N_values = [10, 20, 40, 80, 160]
        errors = []
        for N in N_values:
            model = build_uniform_shaft_model(shaft, N, propeller_J=0.0, engine_J=0.0)
            freqs_num, _ = model.solve_eigenvalue(n_modes=4)
            err = np.max(np.abs(freqs_num - freqs_ana) / freqs_ana)
            errors.append(err)

        # Compute convergence order
        orders = []
        for i in range(1, len(N_values)):
            if errors[i] > 0 and errors[i-1] > 0:
                p = -np.log(errors[i] / errors[i-1]) / np.log(N_values[i] / N_values[i-1])
                orders.append(p)

        avg_order = np.mean(orders)
        assert avg_order >= 1.8, (
            f"Convergence order = {avg_order:.2f}, expected ≥ 1.8. "
            f"Individual orders: {[f'{o:.2f}' for o in orders]}"
        )


# ─────────────────────────────────────────────────────────────
# (c) Transient energy drift
# ─────────────────────────────────────────────────────────────

class TestTransientEnergy:
    """Criterion (c): energy drift < 1e-6 for 10⁴ steps (undamped free vibration)."""

    def test_energy_conservation_undamped(self):
        """Free vibration of undamped system should conserve energy."""
        m = 1.0
        k = 100.0
        M = np.array([[m]])
        C = np.zeros((1, 1))
        K = np.array([[k]])

        solver = NewmarkBetaSolver(M, C, K)

        n_steps = 10000
        dt = 0.001
        t = np.arange(n_steps + 1) * dt

        x0 = np.array([1.0])
        v0 = np.array([0.0])
        F = np.zeros((n_steps + 1, 1))

        result = solver.solve(t, F, x0=x0, v0=v0)
        energy = result["energy"]

        E0 = energy[0]
        max_drift = np.max(np.abs(energy - E0)) / abs(E0)

        assert max_drift < 1e-6, (
            f"Energy drift = {max_drift:.2e}, expected < 1e-6"
        )

    def test_energy_conservation_2dof(self):
        """2-DOF undamped system energy conservation."""
        M = np.array([[2.0, 0], [0, 1.0]])
        K = np.array([[100.0, -50.0], [-50.0, 50.0]])
        C = np.zeros((2, 2))

        solver = NewmarkBetaSolver(M, C, K)

        n_steps = 10000
        dt = 0.001
        t = np.arange(n_steps + 1) * dt

        x0 = np.array([1.0, 0.5])
        v0 = np.array([0.0, 0.0])
        F = np.zeros((n_steps + 1, 2))

        result = solver.solve(t, F, x0=x0, v0=v0)
        energy = result["energy"]

        E0 = energy[0]
        max_drift = np.max(np.abs(energy - E0)) / abs(E0)

        assert max_drift < 1e-6, (
            f"2-DOF energy drift = {max_drift:.2e}, expected < 1e-6"
        )


# ─────────────────────────────────────────────────────────────
# (d) SDOF resonance Q-factor
# ─────────────────────────────────────────────────────────────

class TestResonanceQ:
    """Criterion (d): Q = 1/(2ζ) error < 5% for ζ = 0.02, 0.05, 0.1."""

    @pytest.mark.parametrize("zeta", [0.02, 0.05, 0.1])
    def test_resonance_amplification(self, zeta):
        """SDOF resonance amplification Q vs analytical 1/(2ζ)."""
        result = sdof_resonance_test(zeta, freq_hz=1.0, n_cycles=100, pp_cycles=80)

        assert result["error"] < 0.05, (
            f"ζ={zeta}: Q_analytical={result['Q_analytical']:.2f}, "
            f"Q_numerical={result['Q_numerical']:.2f}, "
            f"error={result['error']:.4f} (>5%)"
        )


# ─────────────────────────────────────────────────────────────
# (e) Frequency response vs analytical
# ─────────────────────────────────────────────────────────────

class TestFrequencyResponse:
    """Criterion (e): |H(ω)| vs analytical < 3%."""

    def test_sdof_frequency_response(self):
        """SDOF frequency response error < 3%."""
        result = sdof_frequency_response(
            zeta=0.05,
            freq_n=1.0,
            freq_range=(0.3, 2.0),
            n_freq=15,
        )

        max_err = np.max(result["error"])
        assert max_err < 0.03, (
            f"Max frequency response error = {max_err:.4f} (>3%)"
        )


# ─────────────────────────────────────────────────────────────
# (f) Integration test: full analysis pipeline
# ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    """Integration test: full analysis runs and produces valid results."""

    def test_holzer_table(self):
        """Holzer table produces valid output at natural frequency."""
        shaft = UniformShaftAnalytical(G=G_REF, rho=RHO_REF, L=L_REF)
        model = build_uniform_shaft_model(shaft, N=20, propeller_J=0.0, engine_J=0.0)

        freqs, _ = model.solve_eigenvalue(n_modes=1)
        f1 = freqs[0]

        table = model.holzer_table(f1)
        # Check that table has correct structure
        assert len(table) > 0
        assert "node" in table[0]
        assert "theta" in table[0]
        assert "residual_torque" in table[0]
        # First node should have unit amplitude
        assert abs(table[0]["theta"] - 1.0) < 1e-10

    def test_ice_milling_excitation(self):
        """Ice milling excitation generates valid torque signal."""
        exc = IceMillingExcitation(T_mean=50000.0, n_blades=5, RPM=120.0)
        t = np.linspace(0, 1.0, 1000)
        T = exc.torque(t)

        assert T.shape == t.shape
        assert np.mean(T) > 0
        assert exc.BPF == 10.0


# ─────────────────────────────────────────────────────────────
# Acceptance criteria summary
# ─────────────────────────────────────────────────────────────

class TestAcceptanceCriteria:
    """Summary of all acceptance criteria in one place."""

    def test_all_criteria(self):
        """Run all criteria and produce summary JSON."""
        results = {}

        # (a) Modal accuracy
        analyzer = ShaftTorsionAnalyzer(
            G=G_REF, rho=RHO_REF, L=L_REF, D=D_REF,
            propeller_J=0.0, engine_J=0.0,
        )
        freqs_ana, freqs_num = analyzer.modal_analysis(N=50, n_modes=4)
        modal_max_err = float(np.max(np.abs(freqs_num - freqs_ana) / freqs_ana))
        results["first_4_modes_vs_analytical"] = modal_max_err
        assert modal_max_err < 0.01, f"Modal error {modal_max_err:.4f} >= 1%"

        # (b) Convergence order
        shaft = UniformShaftAnalytical(G=G_REF, rho=RHO_REF, L=L_REF)
        freqs_ana_ref = shaft.frequencies(4)
        N_vals = [10, 20, 40, 80, 160]
        errs = []
        for N in N_vals:
            model = build_uniform_shaft_model(shaft, N, propeller_J=0.0, engine_J=0.0)
            fn, _ = model.solve_eigenvalue(n_modes=4)
            errs.append(float(np.max(np.abs(fn - freqs_ana_ref) / freqs_ana_ref)))

        orders = []
        for i in range(1, len(N_vals)):
            if errs[i] > 0 and errs[i-1] > 0:
                p = -np.log(errs[i] / errs[i-1]) / np.log(N_vals[i] / N_vals[i-1])
                orders.append(p)
        conv_order = float(np.mean(orders))
        results["convergence_order"] = conv_order
        assert conv_order >= 1.8, f"Convergence order {conv_order:.2f} < 1.8"

        # (c) Energy drift
        M = np.array([[1.0]])
        K = np.array([[100.0]])
        C = np.zeros((1, 1))
        solver = NewmarkBetaSolver(M, C, K)
        n_steps = 10000
        dt = 0.001
        t = np.arange(n_steps + 1) * dt
        res = solver.solve(t, np.zeros((n_steps + 1, 1)), x0=np.array([1.0]))
        E = res["energy"]
        drift = float(np.max(np.abs(E - E[0])) / abs(E[0]))
        results["transient_energy_drift"] = drift
        assert drift < 1e-6, f"Energy drift {drift:.2e} >= 1e-6"

        # (d) Resonance Q
        q_errors = []
        for zeta in [0.02, 0.05, 0.1]:
            qr = sdof_resonance_test(zeta, n_cycles=100, pp_cycles=80)
            q_errors.append(qr["error"])
        q_max = float(max(q_errors))
        results["resonance_amplification_Q"] = q_max
        assert q_max < 0.05, f"Q error {q_max:.4f} >= 5%"

        # (e) Frequency response
        fr = sdof_frequency_response(zeta=0.05, freq_range=(0.3, 2.0), n_freq=15)
        fr_max_err = float(np.max(fr["error"]))
        results["frequency_response_vs_analytical"] = fr_max_err
        assert fr_max_err < 0.03, f"Freq response error {fr_max_err:.4f} >= 3%"

        # Save results
        results["pytest_test_shaft_torsion"] = True
        results_path = os.path.join(os.path.dirname(__file__), "..", "outputs",
                                     "acceptance_criteria.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n=== Acceptance Criteria Summary ===")
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
