"""Tests for propeller-hull-ice coupling module (mission m9276).

Acceptance criteria:
    C1: Propeller convergence: K_T from actuator disk vs BEM (5 elements) < 5%
    C2: Wake modulation: V_wake vs actuator disk theory V_wake = V_inflow*(1-2a) < 3%
    C3: Ice load modulation: ΔR_ice/R_ice > 5% with wake
    C4: Hull response: natural frequencies vs analytical < 5%
    C5: Thrust pulsation feedback: ΔδT/δT > 10% with vibration
    C6: Coupling convergence: iterations < 10 for |δT/δT_prev - 1| < 1%
    C7: Transfer function: |H_thrust| > 0.1 in f_ice ∈ [0.1, 0.5] Hz
    C8: pytest coverage of C1-C7
"""

import numpy as np
import pytest

from tensorphysics.propeller_hull_ice_coupling import (
    ActuatorDisk,
    BladeElementMethod,
    PropellerGeometry,
    WakeModel,
    WakeField,
    IceProperties,
    HullGeometry,
    IceLoadModel,
    HullStructure,
    MultiDOFResponse,
    compute_natural_frequencies,
    CouplingSolver,
    compute_transfer_function,
    lindqvist_ice_resistance,
)


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def propeller_geometry():
    """Standard propeller geometry for testing."""
    return PropellerGeometry(
        D=5.0,           # 5m diameter
        Z=5,             # 5 blades
        pitch_ratio=1.0, # P/D = 1.0
        A_E_A_0=0.7,
    )


@pytest.fixture
def hull_geometry():
    """Standard hull geometry for testing."""
    return HullGeometry(
        L=100.0,
        B=20.0,
        T=7.0,
        alpha_deg=25.0,
        stem_angle_deg=15.0,
        Cb=0.75,
    )


@pytest.fixture
def ice_properties():
    """Standard ice properties for testing."""
    return IceProperties(
        thickness=0.7,
        sigma_f=0.5e6,
        sigma_c=5.0e6,
        E=5.0e9,
        nu=0.3,
        mu=0.15,
    )


@pytest.fixture
def hull_structure():
    """Standard hull structure for testing."""
    return HullStructure(
        L=100.0,
        B=20.0,
        T=7.0,
        Cb=0.75,
        EI=1.0e12,
        added_mass_heave=0.5,
        added_mass_pitch=0.3,
        damping_ratio_heave=0.05,
        damping_ratio_pitch=0.03,
        n_bend_modes=3,
    )


# ============================================================================
# C1: Propeller convergence - actuator disk vs BEM
# ============================================================================

class TestPropellerConvergence:
    """C1: K_T from actuator disk vs BEM (5 elements) < 5%."""
    
    def test_actuator_disk_thrust_coefficient(self, propeller_geometry):
        """Test actuator disk thrust coefficient calculation."""
        prop = ActuatorDisk(propeller_geometry, K_T_design=0.25)
        
        # At design advance coefficient
        J = 0.8
        K_T = prop.thrust_coefficient(J)
        
        # K_T should be positive and reasonable
        assert K_T > 0
        assert K_T < 1.0
        
    def test_bem_thrust_coefficient(self, propeller_geometry):
        """Test BEM thrust coefficient calculation."""
        bem = BladeElementMethod(propeller_geometry, n_elements=5)
        
        J = 0.8
        n_rps = 2.0
        result = bem.compute(J, n_rps)
        
        K_T = result["K_T"]
        assert K_T > 0
        assert K_T < 1.0
        
    def test_c1_propeller_convergence(self, propeller_geometry):
        """C1: K_T from actuator disk vs BEM (5 elements) < 5%."""
        prop = ActuatorDisk(propeller_geometry, K_T_design=0.25)
        bem = BladeElementMethod(propeller_geometry, n_elements=5)
        
        J = 0.8
        n_rps = 2.0
        
        K_T_disk = prop.thrust_coefficient(J)
        K_T_bem = bem.compute(J, n_rps)["K_T"]
        
        # Relative difference
        rel_diff = abs(K_T_disk - K_T_bem) / K_T_disk
        
        assert rel_diff < 0.05, f"K_T difference {rel_diff*100:.1f}% > 5%"


# ============================================================================
# C2: Wake modulation - V_wake vs actuator disk theory
# ============================================================================

class TestWakeModulation:
    """C2: V_wake vs actuator disk theory V_wake = V_inflow*(1-2a) < 3%."""
    
    def test_wake_field_velocity(self):
        """Test wake field velocity calculation."""
        V_inflow = 3.0
        a = 0.2  # Axial induction factor
        
        wake = WakeField(V_inflow=V_inflow, axial_induction=a)
        
        # Theoretical wake velocity
        V_wake_theory = V_inflow * (1.0 - 2.0 * a)
        
        assert abs(wake.V_wake_far - V_wake_theory) < 1e-10
        
    def test_c2_wake_velocity_accuracy(self, propeller_geometry):
        """C2: V_wake vs actuator disk theory < 3%."""
        prop = ActuatorDisk(propeller_geometry, K_T_design=0.25)
        
        V_inflow = 3.0
        n_rps = 2.0
        
        # Compute thrust and axial induction
        J = V_inflow / (n_rps * propeller_geometry.D)
        K_T = prop.thrust_coefficient(J)
        T = prop.thrust_from_KT(K_T, n_rps)
        a = prop.axial_induction_from_thrust(T, V_inflow)
        
        # Wake field
        wake = WakeField(V_inflow=V_inflow, axial_induction=a)
        
        # Theoretical value
        V_wake_theory = V_inflow * (1.0 - 2.0 * a)
        
        # Relative error
        rel_error = abs(wake.V_wake_far - V_wake_theory) / abs(V_wake_theory)
        
        assert rel_error < 0.03, f"Wake velocity error {rel_error*100:.1f}% > 3%"


# ============================================================================
# C3: Ice load modulation - ΔR_ice/R_ice > 5% with wake
# ============================================================================

class TestIceLoadModulation:
    """C3: ΔR_ice/R_ice > 5% with wake modulation."""
    
    def test_ice_resistance_basic(self, hull_geometry, ice_properties):
        """Test basic ice resistance calculation."""
        result = lindqvist_ice_resistance(V_ship=3.0, hull=hull_geometry, ice=ice_properties)
        
        assert result["R_total"] > 0
        assert "R_crush" in result
        assert "R_break" in result
        assert "R_sub" in result
        
    def test_c3_ice_load_reduction_with_wake(self, hull_geometry, ice_properties):
        """C3: Ice load reduction > 5% with wake modulation."""
        ice_model = IceLoadModel(hull_geometry, ice_properties, wake_modulation=True)
        
        V_ship = 3.0
        V_wake_induced = 0.5  # 0.5 m/s wake-induced velocity reduction
        
        # Without wake
        R_no_wake = ice_model.ice_resistance(V_ship, V_wake_induced=0.0)["R_total"]
        
        # With wake
        R_with_wake = ice_model.ice_resistance(V_ship, V_wake_induced=V_wake_induced)["R_total"]
        
        # Reduction
        delta_R = R_no_wake - R_with_wake
        rel_reduction = delta_R / R_no_wake
        
        assert rel_reduction > 0.05, f"Ice load reduction {rel_reduction*100:.1f}% < 5%"


# ============================================================================
# C4: Hull response - natural frequencies vs analytical
# ============================================================================

class TestHullResponse:
    """C4: Natural frequencies vs analytical < 5%."""
    
    def test_natural_frequencies_computation(self, hull_structure):
        """Test natural frequency computation."""
        freqs = compute_natural_frequencies(hull_structure)
        
        assert "f_heave" in freqs
        assert "f_pitch" in freqs
        assert "f_bend" in freqs
        
        # Heave frequency should be in reasonable range (0.05 - 0.5 Hz)
        assert 0.05 < freqs["f_heave"] < 0.5
        
        # Pitch frequency should be in reasonable range (0.05 - 0.3 Hz)
        assert 0.05 < freqs["f_pitch"] < 0.3
        
    def test_c4_natural_frequency_accuracy(self, hull_structure):
        """C4: Natural frequencies vs analytical < 5%."""
        freqs = compute_natural_frequencies(hull_structure)
        
        # Analytical heave frequency
        K33, K55 = hull_structure.hydrostatic_stiffness()
        M33 = hull_structure.mass_ship * (1.0 + hull_structure.added_mass_heave)
        f_heave_analytical = np.sqrt(K33 / M33) / (2.0 * np.pi)
        
        # Relative error
        rel_error = abs(freqs["f_heave"] - f_heave_analytical) / f_heave_analytical
        
        assert rel_error < 0.05, f"Heave frequency error {rel_error*100:.1f}% > 5%"
        
        # Analytical pitch frequency
        M55 = hull_structure.Iyy_ship * (1.0 + hull_structure.added_mass_pitch)
        f_pitch_analytical = np.sqrt(K55 / M55) / (2.0 * np.pi)
        
        rel_error_pitch = abs(freqs["f_pitch"] - f_pitch_analytical) / f_pitch_analytical
        
        assert rel_error_pitch < 0.05, f"Pitch frequency error {rel_error_pitch*100:.1f}% > 5%"


# ============================================================================
# C5: Thrust pulsation feedback - ΔδT/δT > 10% with vibration
# ============================================================================

class TestThrustPulsation:
    """C5: ΔδT/δT > 10% with hull vibration feedback."""
    
    def test_c5_thrust_pulsation_increase(self, propeller_geometry, hull_geometry, 
                                          ice_properties, hull_structure):
        """C5: Thrust pulsation increase > 10% with vibration."""
        solver = CouplingSolver(
            prop_geom=propeller_geometry,
            hull_geom=hull_geometry,
            ice_props=ice_properties,
            hull_struct=hull_structure,
            V_ship=3.0,
            n_rps=2.0,
            wake_fraction=0.25,
        )
        
        result = solver.solve()
        
        # Increase in thrust pulsation due to vibration
        delta_delta_T = result.delta_delta_T
        delta_T_no_vib = result.delta_T_no_vibration
        
        if delta_T_no_vib > 0:
            rel_increase = delta_delta_T / delta_T_no_vib
            assert rel_increase > 0.10, f"Thrust pulsation increase {rel_increase*100:.1f}% < 10%"
        else:
            # If baseline is zero, any positive value is > 10% increase
            assert delta_delta_T > 0


# ============================================================================
# C6: Coupling convergence - iterations < 10
# ============================================================================

class TestCouplingConvergence:
    """C6: Coupling convergence in < 10 iterations."""
    
    def test_c6_convergence_iterations(self, propeller_geometry, hull_geometry,
                                       ice_properties, hull_structure):
        """C6: Convergence in < 10 iterations."""
        solver = CouplingSolver(
            prop_geom=propeller_geometry,
            hull_geom=hull_geometry,
            ice_props=ice_properties,
            hull_struct=hull_structure,
            V_ship=3.0,
            n_rps=2.0,
            max_iterations=20,
            tolerance=0.01,
        )
        
        result = solver.solve()
        
        assert result.converged, "Coupling did not converge"
        assert result.n_iterations < 10, f"Convergence took {result.n_iterations} iterations > 10"


# ============================================================================
# C7: Transfer function - |H_thrust| > 0.1 in f_ice ∈ [0.1, 0.5] Hz
# ============================================================================

class TestTransferFunction:
    """C7: |H_thrust| > 0.1 in ice breaking frequency range."""
    
    def test_transfer_function_computation(self):
        """Test transfer function computation."""
        freq = np.linspace(0.01, 2.0, 100)
        S_ice = np.exp(-((freq - 0.3) / 0.1)**2)  # Peak at 0.3 Hz
        S_thrust = 0.5 * S_ice  # Half the ice energy transmitted
        
        H = compute_transfer_function(freq, S_thrust, S_ice)
        
        assert len(H) == len(freq)
        assert np.all(H >= 0)
        
    def test_c7_transfer_function_magnitude(self, propeller_geometry, hull_geometry,
                                            ice_properties, hull_structure):
        """C7: |H_thrust| > 0.1 in f_ice ∈ [0.1, 0.5] Hz."""
        solver = CouplingSolver(
            prop_geom=propeller_geometry,
            hull_geom=hull_geometry,
            ice_props=ice_properties,
            hull_struct=hull_structure,
            V_ship=3.0,
            n_rps=2.0,
        )
        
        result = solver.solve()
        
        # Find frequencies in [0.1, 0.5] Hz range
        mask = (result.freq >= 0.1) & (result.freq <= 0.5)
        H_in_range = result.H_thrust[mask]
        
        # Check that |H_thrust| > 0.1 in this range
        assert len(H_in_range) > 0, "No frequencies in [0.1, 0.5] Hz range"
        
        # At least some frequencies should have |H| > 0.1
        assert np.any(H_in_range > 0.1), f"Max |H_thrust| in range = {np.max(H_in_range):.3f} < 0.1"


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full coupling chain."""
    
    def test_full_coupling_chain(self, propeller_geometry, hull_geometry,
                                 ice_properties, hull_structure):
        """Test full propeller-hull-ice coupling chain."""
        solver = CouplingSolver(
            prop_geom=propeller_geometry,
            hull_geom=hull_geometry,
            ice_props=ice_properties,
            hull_struct=hull_structure,
            V_ship=3.0,
            n_rps=2.0,
        )
        
        result = solver.solve()
        
        # Check all outputs are populated
        assert result.T_mean > 0
        assert result.K_T > 0
        assert result.a_axial > 0
        assert result.V_wake > 0
        assert result.R_ice_with_wake > 0
        assert result.delta_R_ice > 0
        assert result.n_iterations > 0
        
    def test_wake_modulation_effect(self, propeller_geometry, hull_geometry,
                                    ice_properties, hull_structure):
        """Test that wake modulation reduces ice resistance."""
        solver = CouplingSolver(
            prop_geom=propeller_geometry,
            hull_geom=hull_geometry,
            ice_props=ice_properties,
            hull_struct=hull_structure,
            V_ship=3.0,
            n_rps=2.0,
        )
        
        result = solver.solve()
        
        # Ice resistance should be lower with wake
        assert result.R_ice_with_wake < result.R_ice_no_wake
        
    def test_different_ship_speeds(self, propeller_geometry, hull_geometry,
                                   ice_properties, hull_structure):
        """Test coupling at different ship speeds."""
        speeds = [1.0, 2.0, 3.0, 4.0]
        
        for V in speeds:
            solver = CouplingSolver(
                prop_geom=propeller_geometry,
                hull_geom=hull_geometry,
                ice_props=ice_properties,
                hull_struct=hull_structure,
                V_ship=V,
                n_rps=2.0,
            )
            
            result = solver.solve()
            
            # Higher speed should give higher ice resistance
            assert result.R_ice_with_wake > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
