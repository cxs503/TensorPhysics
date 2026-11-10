"""
Tests for Ice-Steel Cohesive Zone Model

Validates:
1. Bilinear traction-separation law (energy conservation, irreversibility)
2. Frederking 1984 Mode I calibration (σ_max within 15% of 1.0 MPa)
3. Timco 1986 Mode II calibration (τ_max within 15% of 0.35 MPa)
4. B-K mixed mode criterion (G_c(ψ) accuracy)
5. Zero-thickness interface element (FEM integration)
6. TensorDEM adfreeze bond mapping (cross-library contract)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tensorphysics.ice_czm import (
    BilinearTractionSeparation,
    TractionSeparationState,
    IceSteelCalibration,
    Frederking1984,
    Timco1986,
    calibrate_ice_steel_czm,
    BenzeggaghKenaneCriterion,
    MixedModeCZM,
    ZeroThicknessInterfaceElement,
    InterfaceElementState,
    AdfreezeBondMapper,
    AdfreezeBondState,
)
from tensorphysics.ice_czm.dem_mapping import AdfreezeBondNetwork


# ============================================================================
# Test 1: Bilinear Traction-Separation Law
# ============================================================================

class TestBilinearTractionSeparation:
    """Test bilinear traction-separation law."""
    
    def test_initialization(self):
        """Test law initialization with valid parameters."""
        law = BilinearTractionSeparation(
            K=1e10,
            sigma_max=1e6,
            G_c=100.0,
        )
        assert law.K == 1e10
        assert law.sigma_max == 1e6
        assert law.G_c == 100.0
        assert law.delta_0 == 1e6 / 1e10  # 1e-4 m
        assert law.delta_f == 2.0 * 100.0 / 1e6  # 2e-4 m
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            BilinearTractionSeparation(K=-1e10, sigma_max=1e6, G_c=100.0)
        with pytest.raises(ValueError):
            BilinearTractionSeparation(K=1e10, sigma_max=-1e6, G_c=100.0)
        with pytest.raises(ValueError):
            BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=-100.0)
    
    def test_elastic_regime(self):
        """Test elastic (undamaged) response."""
        law = BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=100.0)
        
        # Small separation: elastic
        state = law.compute_traction(delta=0.5e-4)
        assert state.damage == 0.0
        assert state.traction == 1e10 * 0.5e-4  # K * delta
        assert not state.failed
    
    def test_damage_onset(self):
        """Test damage onset at delta_0."""
        law = BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=100.0)
        
        # At delta_0: damage just starts
        state = law.compute_traction(delta=law.delta_0)
        assert state.damage == 0.0  # Just at onset
        assert_allclose(state.traction, law.sigma_max, rtol=1e-10)
    
    def test_softening_regime(self):
        """Test softening (damaged) response."""
        law = BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=100.0)
        
        # Mid-softening
        delta_mid = 0.5 * (law.delta_0 + law.delta_f)
        state = law.compute_traction(delta=delta_mid)
        assert 0.0 < state.damage < 1.0
        assert state.traction > 0.0
        assert not state.failed
    
    def test_complete_failure(self):
        """Test complete failure at delta_f."""
        law = BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=100.0)
        
        # At delta_f: fully failed
        state = law.compute_traction(delta=law.delta_f)
        assert state.damage >= 1.0 - 1e-10
        assert state.traction == 0.0
        assert state.failed
    
    def test_energy_conservation(self):
        """Test that area under curve equals G_c."""
        law = BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=100.0)
        
        # Integrate traction over separation path
        n_points = 1000
        delta_path = np.linspace(0, law.delta_f, n_points)
        states = law.compute_path(delta_path)
        
        # Trapezoidal integration
        energy = 0.0
        for i in range(1, len(states)):
            d_delta = states[i].delta - states[i-1].delta
            avg_traction = 0.5 * (states[i].traction + states[i-1].traction)
            energy += avg_traction * d_delta
        
        # Should equal G_c (within numerical tolerance)
        assert_allclose(energy, law.G_c, rtol=1e-3)
    
    def test_irreversibility(self):
        """Test that damage cannot heal (irreversibility)."""
        law = BilinearTractionSeparation(K=1e10, sigma_max=1e6, G_c=100.0)
        
        # Load to damage
        delta_max = 0.5 * (law.delta_0 + law.delta_f)
        state1 = law.compute_traction(delta=delta_max)
        damage1 = state1.damage
        
        # Unload
        state2 = law.compute_traction(
            delta=0.5 * delta_max,
            delta_max_prev=delta_max,
            damage_prev=damage1,
        )
        
        # Damage should not decrease
        assert state2.damage >= damage1 - 1e-10


# ============================================================================
# Test 2: Frederking 1984 Mode I Calibration
# ============================================================================

class TestFrederking1984:
    """Test Frederking 1984 Mode I calibration."""
    
    def test_mean_strength(self):
        """Test mean tensile strength is 1.0 MPa."""
        sigma_max = Frederking1984.get_strength(confidence=0.5)
        # Should be close to 1.0 MPa (within 15%)
        assert_allclose(sigma_max, 1.0e6, rtol=0.15)
    
    def test_strength_range(self):
        """Test strength range [0.5, 1.5] MPa."""
        sigma_min = Frederking1984.get_strength(confidence=0.0)
        sigma_max = Frederking1984.get_strength(confidence=1.0)
        assert sigma_min >= 0.5e6 * 0.85  # Allow some tolerance
        assert sigma_max <= 1.5e6 * 1.15
    
    def test_fracture_energy(self):
        """Test Mode I fracture energy."""
        G_Ic = Frederking1984.get_fracture_energy(confidence=0.5)
        assert G_Ic > 0
        assert 50.0 < G_Ic < 200.0  # Reasonable range
    
    def test_calibration(self):
        """Test complete Mode I calibration."""
        params = Frederking1984.calibrate()
        assert "sigma_max" in params
        assert "G_Ic" in params
        assert "K_n" in params
        assert params["sigma_max"] > 0
        assert params["G_Ic"] > 0
        assert params["K_n"] > 0


# ============================================================================
# Test 3: Timco 1986 Mode II Calibration
# ============================================================================

class TestTimco1986:
    """Test Timco 1986 Mode II calibration."""
    
    def test_mean_strength(self):
        """Test mean shear strength is ~0.35 MPa (ice-steel)."""
        tau_max = Timco1986.get_strength(confidence=0.5)
        # Ice-ice is 0.5 MPa, ice-steel is 0.7 * 0.5 = 0.35 MPa
        # Should be within 15% of 0.35 MPa
        assert_allclose(tau_max, 0.35e6, rtol=0.15)
    
    def test_strength_range(self):
        """Test shear strength range."""
        tau_min = Timco1986.get_strength(confidence=0.0)
        tau_max = Timco1986.get_strength(confidence=1.0)
        assert tau_min > 0
        assert tau_max > tau_min
    
    def test_fracture_energy(self):
        """Test Mode II fracture energy."""
        G_IIc = Timco1986.get_fracture_energy(confidence=0.5)
        assert G_IIc > 0
        assert 100.0 < G_IIc < 600.0  # Reasonable range
    
    def test_calibration(self):
        """Test complete Mode II calibration."""
        params = Timco1986.calibrate()
        assert "tau_max" in params
        assert "G_IIc" in params
        assert "K_s" in params
        assert params["tau_max"] > 0
        assert params["G_IIc"] > 0
        assert params["K_s"] > 0


# ============================================================================
# Test 4: Complete Ice-Steel Calibration
# ============================================================================

class TestIceSteelCalibration:
    """Test complete ice-steel CZM calibration."""
    
    def test_calibrate_ice_steel(self):
        """Test complete calibration."""
        cal = calibrate_ice_steel_czm(temperature=-10.0)
        
        assert isinstance(cal, IceSteelCalibration)
        assert cal.sigma_max > 0
        assert cal.tau_max > 0
        assert cal.G_Ic > 0
        assert cal.G_IIc > 0
        assert cal.K_n > 0
        assert cal.K_s > 0
    
    def test_mode_I_law(self):
        """Test Mode I law from calibration."""
        cal = calibrate_ice_steel_czm()
        law_I = cal.to_mode_I_law()
        
        assert isinstance(law_I, BilinearTractionSeparation)
        assert law_I.sigma_max == cal.sigma_max
        assert law_I.G_c == cal.G_Ic
    
    def test_mode_II_law(self):
        """Test Mode II law from calibration."""
        cal = calibrate_ice_steel_czm()
        law_II = cal.to_mode_II_law()
        
        assert isinstance(law_II, BilinearTractionSeparation)
        assert law_II.sigma_max == cal.tau_max
        assert law_II.G_c == cal.G_IIc
    
    def test_frederking_accuracy(self):
        """Test Frederking 1984 calibration accuracy (≤15% error)."""
        cal = calibrate_ice_steel_czm()
        
        # Reference value: 1.0 MPa
        sigma_ref = 1.0e6
        error = abs(cal.sigma_max - sigma_ref) / sigma_ref
        
        assert error <= 0.15, f"Frederking calibration error {error:.2%} > 15%"
    
    def test_timco_accuracy(self):
        """Test Timco 1986 calibration accuracy (≤15% error)."""
        cal = calibrate_ice_steel_czm()
        
        # Reference value: 0.35 MPa (ice-steel)
        tau_ref = 0.35e6
        error = abs(cal.tau_max - tau_ref) / tau_ref
        
        assert error <= 0.15, f"Timco calibration error {error:.2%} > 15%"


# ============================================================================
# Test 5: B-K Mixed Mode Criterion
# ============================================================================

class TestBenzeggaghKenane:
    """Test B-K mixed mode criterion."""
    
    def test_initialization(self):
        """Test B-K criterion initialization."""
        bk = BenzeggaghKenaneCriterion(G_Ic=100.0, G_IIc=300.0, eta=1.45)
        assert bk.G_Ic == 100.0
        assert bk.G_IIc == 300.0
        assert bk.eta == 1.45
    
    def test_pure_mode_I(self):
        """Test pure Mode I (ψ=0)."""
        bk = BenzeggaghKenaneCriterion(G_Ic=100.0, G_IIc=300.0, eta=1.45)
        G_c = bk.effective_toughness(psi=0.0)
        assert_allclose(G_c, 100.0, rtol=1e-10)
    
    def test_pure_mode_II(self):
        """Test pure Mode II (ψ=1)."""
        bk = BenzeggaghKenaneCriterion(G_Ic=100.0, G_IIc=300.0, eta=1.45)
        G_c = bk.effective_toughness(psi=1.0)
        assert_allclose(G_c, 300.0, rtol=1e-10)
    
    def test_mixed_mode(self):
        """Test mixed mode (0 < ψ < 1)."""
        bk = BenzeggaghKenaneCriterion(G_Ic=100.0, G_IIc=300.0, eta=1.45)
        
        psi = 0.5
        G_c = bk.effective_toughness(psi)
        
        # Should be between G_Ic and G_IIc
        assert 100.0 < G_c < 300.0
        
        # Check formula: G_c = G_Ic + (G_IIc - G_Ic) * ψ^η
        G_c_expected = 100.0 + (300.0 - 100.0) * 0.5**1.45
        assert_allclose(G_c, G_c_expected, rtol=1e-10)
    
    def test_critical_energy_release_rate(self):
        """Test critical energy release rate check."""
        bk = BenzeggaghKenaneCriterion(G_Ic=100.0, G_IIc=300.0, eta=1.45)
        
        # Sub-critical
        G_c, G_total, is_critical = bk.critical_energy_release_rate(
            G_I=50.0, G_II=50.0
        )
        assert not is_critical
        assert G_total == 100.0
        
        # Critical
        G_c, G_total, is_critical = bk.critical_energy_release_rate(
            G_I=100.0, G_II=100.0
        )
        assert is_critical


# ============================================================================
# Test 6: Mixed Mode CZM
# ============================================================================

class TestMixedModeCZM:
    """Test mixed-mode cohesive zone model."""
    
    def test_initialization(self):
        """Test mixed-mode CZM initialization."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n,
            K_s=cal.K_s,
            sigma_max=cal.sigma_max,
            tau_max=cal.tau_max,
            G_Ic=cal.G_Ic,
            G_IIc=cal.G_IIc,
        )
        assert czm.sigma_max == cal.sigma_max
        assert czm.tau_max == cal.tau_max
    
    def test_pure_mode_I(self):
        """Test pure Mode I loading."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        
        # Pure opening
        result = czm.compute_traction(delta_n=1e-4, delta_s=0.0)
        assert result["sigma_n"] > 0
        assert abs(result["tau_s"]) < 1e-10
    
    def test_pure_mode_II(self):
        """Test pure Mode II loading."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        
        # Pure shear
        result = czm.compute_traction(delta_n=0.0, delta_s=1e-4)
        assert abs(result["sigma_n"]) < 1e-10
        assert result["tau_s"] > 0
    
    def test_mixed_mode(self):
        """Test mixed mode loading."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        
        # Mixed opening and shear
        result = czm.compute_traction(delta_n=1e-4, delta_s=1e-4)
        assert result["sigma_n"] > 0
        assert result["tau_s"] > 0
        assert 0.0 < result["psi"] < 1.0


# ============================================================================
# Test 7: Zero-Thickness Interface Element
# ============================================================================

class TestZeroThicknessInterfaceElement:
    """Test zero-thickness interface element."""
    
    def test_initialization(self):
        """Test element initialization."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        elem = ZeroThicknessInterfaceElement(czm, element_id=0)
        assert elem.element_id == 0
    
    def test_update(self):
        """Test element update."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        elem = ZeroThicknessInterfaceElement(czm)
        
        state = elem.update(delta_n=1e-4, delta_s=0.5e-4)
        assert isinstance(state, InterfaceElementState)
        assert state.delta_n == 1e-4
        assert state.delta_s == 0.5e-4
    
    def test_tangent_stiffness(self):
        """Test tangent stiffness matrix."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        elem = ZeroThicknessInterfaceElement(czm)
        
        # Undamaged
        elem.update(delta_n=0.5e-4, delta_s=0.0)
        K = elem.get_tangent_stiffness()
        assert K.shape == (3, 3)
        assert K[0, 0] > 0  # Normal stiffness
        assert K[1, 1] > 0  # Shear stiffness
    
    def test_reset(self):
        """Test element reset."""
        cal = calibrate_ice_steel_czm()
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        elem = ZeroThicknessInterfaceElement(czm)
        
        elem.update(delta_n=1e-4, delta_s=0.5e-4)
        elem.reset()
        
        state = elem.get_state()
        assert state.damage == 0.0
        assert state.delta_n == 0.0


# ============================================================================
# Test 8: TensorDEM Adfreeze Bond Mapping
# ============================================================================

class TestAdfreezeBondMapper:
    """Test TensorDEM adfreeze bond mapping."""
    
    def test_initialization(self):
        """Test mapper initialization."""
        cal = calibrate_ice_steel_czm()
        mapper = AdfreezeBondMapper(
            calibration=cal,
            bond_area=1e-6,
        )
        assert mapper.bond_area == 1e-6
    
    def test_from_particle_size(self):
        """Test mapper from particle size."""
        mapper = AdfreezeBondMapper.from_particle_size(
            particle_radius=1e-3,  # 1 mm
        )
        assert mapper.bond_area > 0
        assert mapper.characteristic_length == 1e-3
    
    def test_bond_strengths(self):
        """Test bond strength calculation."""
        mapper = AdfreezeBondMapper.from_particle_size(particle_radius=1e-3)
        strengths = mapper.get_bond_strengths()
        
        assert "F_n_max" in strengths
        assert "F_s_max" in strengths
        assert "G_I_bond" in strengths
        assert "G_II_bond" in strengths
        assert strengths["F_n_max"] > 0
        assert strengths["F_s_max"] > 0
    
    def test_dem_parameters(self):
        """Test DEM parameter export."""
        mapper = AdfreezeBondMapper.from_particle_size(particle_radius=1e-3)
        params = mapper.get_dem_parameters()
        
        assert "K_n_bond" in params
        assert "K_s_bond" in params
        assert "eta_bk" in params
        assert params["K_n_bond"] > 0
        assert params["K_s_bond"] > 0
    
    def test_bond_force_computation(self):
        """Test bond force computation."""
        mapper = AdfreezeBondMapper.from_particle_size(particle_radius=1e-3)
        
        result = mapper.compute_bond_force(
            delta_n=1e-4,
            delta_s=0.5e-4,
        )
        
        assert "F_n" in result
        assert "F_s" in result
        assert "damage" in result
        assert "active" in result


class TestAdfreezeBondNetwork:
    """Test adfreeze bond network."""
    
    def test_create_bond(self):
        """Test bond creation."""
        mapper = AdfreezeBondMapper.from_particle_size(particle_radius=1e-3)
        network = AdfreezeBondNetwork(mapper)
        
        bond = network.create_bond()
        assert isinstance(bond, AdfreezeBondState)
        assert bond.active
        assert bond.bond_id == 0
    
    def test_update_bond(self):
        """Test bond update."""
        mapper = AdfreezeBondMapper.from_particle_size(particle_radius=1e-3)
        network = AdfreezeBondNetwork(mapper)
        
        bond = network.create_bond()
        updated = network.update_bond(
            bond_id=bond.bond_id,
            delta_n=1e-4,
            delta_s=0.5e-4,
            dt=0.01,
        )
        
        assert updated.delta_n == 1e-4
        assert updated.age == 0.01
    
    def test_total_force(self):
        """Test total force calculation."""
        mapper = AdfreezeBondMapper.from_particle_size(particle_radius=1e-3)
        network = AdfreezeBondNetwork(mapper)
        
        # Create multiple bonds
        for _ in range(5):
            network.create_bond()
        
        # Update all
        for bond_id in network.bonds:
            network.update_bond(bond_id, delta_n=1e-4, delta_s=0.0, dt=0.01)
        
        F_n, F_s = network.total_force()
        assert F_n > 0  # Tension
        assert abs(F_s) < 1e-10  # No shear


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_calibration_to_czm(self):
        """Test calibration → CZM workflow."""
        # Calibrate
        cal = calibrate_ice_steel_czm(temperature=-10.0)
        
        # Create CZM
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        
        # Simulate loading
        result = czm.compute_traction(delta_n=1e-4, delta_s=0.5e-4)
        assert result["sigma_n"] > 0
        assert result["tau_s"] > 0
    
    def test_czm_to_dem(self):
        """Test CZM → DEM mapping workflow."""
        # Calibrate
        cal = calibrate_ice_steel_czm()
        
        # Map to DEM
        mapper = AdfreezeBondMapper(calibration=cal, bond_area=1e-6)
        params = mapper.get_dem_parameters()
        
        # Verify parameters
        assert params["F_n_max"] > 0
        assert params["K_n_bond"] > 0
    
    def test_full_simulation(self):
        """Test full simulation workflow."""
        # 1. Calibrate
        cal = calibrate_ice_steel_czm(temperature=-10.0)
        
        # 2. Create CZM
        czm = MixedModeCZM(
            K_n=cal.K_n, K_s=cal.K_s,
            sigma_max=cal.sigma_max, tau_max=cal.tau_max,
            G_Ic=cal.G_Ic, G_IIc=cal.G_IIc,
        )
        
        # 3. Create interface element
        elem = ZeroThicknessInterfaceElement(czm, element_id=0)
        
        # 4. Simulate loading path
        delta_n_path = np.linspace(0, 2e-4, 100)
        delta_s_path = np.zeros_like(delta_n_path)
        
        for dn, ds in zip(delta_n_path, delta_s_path):
            state = elem.update(dn, ds)
        
        # 5. Check final state
        final_state = elem.get_state()
        assert final_state.delta_n == 2e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
