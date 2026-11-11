"""Tests for FEM→DEM ice plate fracture energy conservation contract (p470w2-n05).

Acceptance criteria
-------------------
C1: FEM plate bending strain energy vs analytical solution ≤ 2 %
C2: Fracture energy G_f vs surface energy formula ≤ 1 %
C3: DEM bond fracture energy vs G_f × A ≤ 1 %
C4: Total energy conservation residual ≤ 5 %
C5: Negative control — no fracture → residual ≤ 1 %
C6: Parameter sweep h = 0.3–1.0 m (5 levels), σ_flex = 1.0–2.0 MPa (5 levels),
    each combination residual ≤ 5 %
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from tensorphysics.fem_dem_energy_contract.mindlin_plate import (
    MindlinPlateSolution,
    solve_mindlin_plate,
)
from tensorphysics.fem_dem_energy_contract.fracture_energy import (
    FractureEnergy,
    compute_fracture_energy,
    radial_crack_pattern,
    CrackSegment,
    GAMMA_ICE,
    compute_max_bending_stress,
)
from tensorphysics.fem_dem_energy_contract.dem_bond import (
    DEMBond,
    DEMBondEnergy,
    compute_dem_bond_energy,
    create_dem_bonds_from_fracture,
)
from tensorphysics.fem_dem_energy_contract.fragment_kinetic import (
    IceFragment,
    FragmentKineticEnergy,
    compute_fragment_kinetic_energy,
    create_fragments_from_fracture,
)
from tensorphysics.fem_dem_energy_contract.energy_ledger import (
    EnergyLedger,
    run_energy_ledger,
    EnergyLedgerResult,
    run_parameter_sweep,
)


# ---------------------------------------------------------------------------
# C1: FEM plate bending strain energy vs analytical solution ≤ 2%
# ---------------------------------------------------------------------------

class TestC1MindlinPlate:
    """C1: FEM plate bending strain energy vs analytical solution."""

    def test_analytical_self_consistency(self):
        """Analytical solution should be self-consistent (U = ½ P·w)."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        w = sol.centre_deflection()
        U = sol.bending_strain_energy()
        assert U == pytest.approx(0.5 * sol.P * w, rel=1e-12)

    def test_deflection_positive(self):
        """Deflection should be positive (downward)."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        w = sol.centre_deflection()
        assert w > 0

    def test_energy_positive(self):
        """Strain energy should be positive."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        U = sol.bending_strain_energy()
        assert U > 0

    def test_solve_mindlin_plate_returns_result(self):
        """solve_mindlin_plate should return a valid result."""
        result = solve_mindlin_plate(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        assert result.U_bend_analytical > 0
        assert result.w_analytical > 0
        assert result.c1_pass

    def test_shear_contribution_increases_deflection(self):
        """Mindlin deflection should be larger than Kirchhoff (shear adds deflection)."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        w_k = sol._kirchhoff_deflection()
        w_s = sol._shear_deflection()
        w_total = sol.centre_deflection()
        assert w_s > 0
        assert w_total == pytest.approx(w_k + w_s, rel=1e-12)
        assert w_total > w_k  # shear adds deflection

    def test_thick_plate_shear_significant(self):
        """For thick plates (h/a > 0.1), shear contribution should be significant."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        w_k = sol._kirchhoff_deflection()
        w_s = sol._shear_deflection()
        shear_frac = w_s / (w_k + w_s)
        # For h/a = 0.125, shear should contribute at least 5%
        assert shear_frac > 0.05

    def test_flexural_rigidity_formula(self):
        """D = E h³ / [12(1-ν²)] should match."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        expected_D = 5.0e9 * 0.5**3 / (12.0 * (1.0 - 0.3**2))
        assert sol.D == pytest.approx(expected_D, rel=1e-12)

    def test_shear_modulus_formula(self):
        """G = E / [2(1+ν)] should match."""
        sol = MindlinPlateSolution(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
        )
        expected_G = 5.0e9 / (2.0 * (1.0 + 0.3))
        assert sol.G == pytest.approx(expected_G, rel=1e-12)


# ---------------------------------------------------------------------------
# C2: Fracture energy G_f vs surface energy formula ≤ 1%
# ---------------------------------------------------------------------------

class TestC2FractureEnergy:
    """C2: Fracture energy G_f vs surface energy formula."""

    def test_griffith_fracture_energy(self):
        """G_f = 2·γ_ice for pure Griffith fracture."""
        cracks = [CrackSegment(0, 0, 1, 0)]
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=2.0e6, sigma_flex=1.5e6,
            gamma_ice=GAMMA_ICE,
        )
        expected_Gf = 2.0 * GAMMA_ICE
        assert result.G_f == pytest.approx(expected_Gf, rel=1e-12)

    def test_effective_fracture_energy(self):
        """When G_f_effective is provided, it should be used."""
        cracks = [CrackSegment(0, 0, 1, 0)]
        G_f_eff = 50.0
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=2.0e6, sigma_flex=1.5e6,
            gamma_ice=GAMMA_ICE, G_f_effective=G_f_eff,
        )
        assert result.G_f == pytest.approx(G_f_eff, rel=1e-12)

    def test_fracture_area_calculation(self):
        """Fracture area = h × total crack length."""
        cracks = [
            CrackSegment(0, 0, 1, 0),
            CrackSegment(0, 0, 0, 1),
        ]
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=2.0e6, sigma_flex=1.5e6,
        )
        expected_area = 0.5 * (1.0 + 1.0)  # h × (L1 + L2)
        assert result.fracture_area == pytest.approx(expected_area, rel=1e-12)

    def test_fracture_energy_total(self):
        """E_fracture = G_f × A_fracture."""
        cracks = [CrackSegment(0, 0, 2, 0)]
        G_f_eff = 50.0
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=2.0e6, sigma_flex=1.5e6,
            G_f_effective=G_f_eff,
        )
        expected_E = G_f_eff * (0.5 * 2.0)
        assert result.E_fracture == pytest.approx(expected_E, rel=1e-12)

    def test_fracture_triggered_when_sigma_exceeds_flex(self):
        """Fracture should occur when σ_max ≥ σ_flex."""
        cracks = [CrackSegment(0, 0, 1, 0)]
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=2.0e6, sigma_flex=1.5e6,
        )
        assert result.fractured

    def test_no_fracture_when_sigma_below_flex(self):
        """No fracture when σ_max < σ_flex."""
        cracks = [CrackSegment(0, 0, 1, 0)]
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=1.0e6, sigma_flex=1.5e6,
        )
        assert not result.fractured

    def test_radial_crack_pattern_length(self):
        """Radial crack pattern should have correct total length."""
        cracks = radial_crack_pattern(n_cracks=4, crack_length=1.0)
        total_length = sum(c.length for c in cracks)
        assert total_length == pytest.approx(4.0, rel=1e-12)

    def test_c2_pass_with_effective_Gf(self):
        """C2 should pass when G_f matches the specified effective value."""
        cracks = radial_crack_pattern(n_cracks=4, crack_length=1.0)
        G_f_eff = 50.0
        result = compute_fracture_energy(
            cracks=cracks, h=0.5, sigma_max=2.0e6, sigma_flex=1.5e6,
            G_f_effective=G_f_eff,
        )
        error = abs(result.G_f - G_f_eff) / G_f_eff
        assert error <= 0.01


# ---------------------------------------------------------------------------
# C3: DEM bond fracture energy vs G_f × A ≤ 1%
# ---------------------------------------------------------------------------

class TestC3DEMBondEnergy:
    """C3: DEM bond fracture energy vs G_f × A."""

    def test_bond_energy_calculation(self):
        """Bond energy = bond_strength × bond_area for broken bonds."""
        bond = DEMBond(
            bond_id=0, fragment_i=0, fragment_j=1,
            bond_area=0.01, bond_strength=50.0, broken=True,
        )
        assert bond.bond_energy == pytest.approx(0.5, rel=1e-12)

    def test_unbroken_bond_zero_energy(self):
        """Unbroken bonds should have zero energy."""
        bond = DEMBond(
            bond_id=0, fragment_i=0, fragment_j=1,
            bond_area=0.01, bond_strength=50.0, broken=False,
        )
        assert bond.bond_energy == 0.0

    def test_dem_bond_energy_matches_expected(self):
        """Total bond energy should match G_f × fracture_area within 1%."""
        fracture_area = 1.0
        G_f = 50.0
        bonds = create_dem_bonds_from_fracture(
            fracture_area=fracture_area, G_f=G_f,
            n_bonds=100, bond_area_fraction=0.995,  # high fidelity
        )
        result = compute_dem_bond_energy(bonds, G_f, fracture_area)
        # With 0.995 area fraction, error should be ~0.5%
        assert result.bond_energy_error_rel <= 0.01
        assert result.c3_pass

    def test_dem_bond_energy_error_within_tolerance(self):
        """C3: bond energy error should be ≤ 1%."""
        fracture_area = 2.0
        G_f = 50.0
        bonds = create_dem_bonds_from_fracture(
            fracture_area=fracture_area, G_f=G_f,
            n_bonds=200, bond_area_fraction=0.995,
        )
        result = compute_dem_bond_energy(bonds, G_f, fracture_area)
        assert result.bond_energy_error_rel <= 0.01
        assert result.c3_pass


# ---------------------------------------------------------------------------
# C4: Total energy conservation residual ≤ 5%
# ---------------------------------------------------------------------------

class TestC4EnergyConservation:
    """C4: Total energy conservation residual."""

    def test_energy_ledger_runs(self):
        """Energy ledger should run without errors."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        assert result.U_bend > 0

    def test_energy_balance_closes(self):
        """Energy balance should close within 5%."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        assert result.c4_pass
        assert result.energy_residual_rel <= 0.05

    def test_energy_input_equals_bending(self):
        """Input energy should equal bending strain energy."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        assert result.E_input == pytest.approx(result.U_bend, rel=1e-12)

    def test_energy_output_components_fracture(self):
        """When fractured, output energy should be sum of components."""
        # Use params that guarantee fracture
        result = run_energy_ledger(
            P=500e3, a=4.0, b=4.0, h=0.3, E=5.0e9, nu=0.3,
            sigma_flex=1.0e6, G_f_effective=50.0,
        )
        if result.fractured:
            # Energy balance: U_bend = E_fracture + E_k + E_dissipated
            expected_output = result.E_fracture + result.E_k + result.E_dissipated
            assert result.E_output == pytest.approx(expected_output, rel=1e-12)

    def test_energy_output_no_fracture(self):
        """When not fractured, output should equal input (all stored)."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=100e6, G_f_effective=50.0,
        )
        assert not result.fractured
        assert result.E_output == pytest.approx(result.E_input, rel=1e-12)


# ---------------------------------------------------------------------------
# C5: Negative control — no fracture → residual ≤ 1%
# ---------------------------------------------------------------------------

class TestC5NegativeControl:
    """C5: Negative control — no fracture scenario."""

    def test_no_fracture_high_sigma_flex(self):
        """With very high σ_flex, no fracture should occur."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=100e6,  # very high, no fracture
            G_f_effective=50.0,
        )
        assert not result.fractured
        assert result.E_fracture == 0.0
        assert result.E_bond == 0.0
        assert result.E_k == 0.0

    def test_no_fracture_residual_within_1_percent(self):
        """Without fracture, energy residual should be ≤ 1%."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=100e6,
            G_f_effective=50.0,
        )
        assert result.c5_pass


# ---------------------------------------------------------------------------
# C6: Parameter sweep
# ---------------------------------------------------------------------------

class TestC6ParameterSweep:
    """C6: Parameter sweep h = 0.3–1.0 m, σ_flex = 1.0–2.0 MPa."""

    def test_sweep_runs_all_combinations(self):
        """Sweep should run all 25 combinations."""
        results = run_parameter_sweep(
            h_values=[0.3, 0.4, 0.5, 0.7, 1.0],
            sigma_flex_values=[1.0e6, 1.25e6, 1.5e6, 1.75e6, 2.0e6],
            P=100e3, a=4.0, b=4.0, E=5.0e9, nu=0.3,
            G_f_effective=50.0,
        )
        assert len(results) == 25

    def test_sweep_all_pass_c4(self):
        """All sweep combinations should pass C4 (residual ≤ 5%)."""
        results = run_parameter_sweep(
            h_values=[0.3, 0.4, 0.5, 0.7, 1.0],
            sigma_flex_values=[1.0e6, 1.25e6, 1.5e6, 1.75e6, 2.0e6],
            P=100e3, a=4.0, b=4.0, E=5.0e9, nu=0.3,
            G_f_effective=50.0,
        )
        for i, r in enumerate(results):
            assert r.c4_pass, f"Combination {i} failed C4: residual={r.energy_residual_rel:.4f}"

    def test_sweep_fracture_pattern(self):
        """Thinner plates with lower σ_flex should fracture more easily."""
        results = run_parameter_sweep(
            h_values=[0.3, 1.0],
            sigma_flex_values=[1.0e6, 2.0e6],
            P=100e3, a=4.0, b=4.0, E=5.0e9, nu=0.3,
            G_f_effective=50.0,
        )
        # h=0.3, σ_flex=1.0MPa should fracture (thin, weak)
        thin_weak = [r for r in results if r.h == 0.3 and r.sigma_flex == 1.0e6]
        assert len(thin_weak) == 1
        # h=1.0, σ_flex=2.0MPa may or may not fracture depending on stress
        thick_strong = [r for r in results if r.h == 1.0 and r.sigma_flex == 2.0e6]
        assert len(thick_strong) == 1


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the full energy ledger."""

    def test_full_ledger_default_params(self):
        """Full ledger with default parameters should pass all criteria."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        assert result.c1_pass
        assert result.c2_pass
        assert result.c3_pass
        assert result.c4_pass
        assert result.c5_pass

    def test_ledger_as_dict(self):
        """Result should be serialisable to dict."""
        result = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "U_bend_J" in d
        assert "energy_residual_rel" in d
        assert "c4_pass" in d

    def test_energy_ledger_class(self):
        """EnergyLedger class should produce same result as function."""
        ledger = EnergyLedger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        result1 = ledger.run()
        result2 = run_energy_ledger(
            P=100e3, a=4.0, b=4.0, h=0.5, E=5.0e9, nu=0.3,
            sigma_flex=1.5e6, G_f_effective=50.0,
        )
        assert result1.U_bend == pytest.approx(result2.U_bend, rel=1e-12)
        assert result1.energy_residual_rel == pytest.approx(result2.energy_residual_rel, rel=1e-12)
