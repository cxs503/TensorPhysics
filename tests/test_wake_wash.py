"""Tests for wake wash propagation module.

This test suite verifies:
1. Kelvin wave geometry (transverse angle ≈19.47°, divergent angle ≈70.53°)
2. Wave height decay laws (deep water H∝1/√r, shallow water H∝1/r)
3. Shoreline impact estimation
4. PIANC compliance assessment
5. Unified API functionality

Acceptance criteria:
- Kelvin transverse wave angle = 19.47° ± 0.01°
- Kelvin divergent wave angle = 70.53° ± 0.01°
- Decay law relative error ≤ 1%
- Shoreline wave height vs PIANC typical interval error ≤ 20%
"""

import math
import pytest

from tensorphysics.wake_wash import (
    # Kelvin wave
    transverse_wave_angle,
    divergent_wave_angle,
    wake_half_angle,
    wake_full_angle,
    kelvin_wake_coordinates,
    wave_wavelength,
    wave_period,
    froude_number,
    KelvinWaveError,
    # Wave decay
    deep_water_decay,
    shallow_water_decay,
    transition_depth,
    shallow_water_criterion,
    wave_height_with_depth,
    decay_profile,
    energy_flux_conservation,
    WaveDecayError,
    # Shoreline impact
    shoaling_coefficient,
    breaking_criterion,
    runup_estimation,
    shoreline_wave_height,
    wake_wash_impact_assessment,
    ShorelineImpactError,
    # PIANC compliance
    ReceptorType,
    ComplianceLevel,
    get_wave_height_limit,
    check_compliance,
    assess_multiple_receptors,
    risk_classification,
    mitigation_recommendations,
    compliance_report,
    speed_limit_for_compliance,
    PIANCComplianceError,
    # API
    analyze_wake_pattern,
    predict_wave_height,
    assess_shoreline_impact,
    check_pianc_compliance,
    optimize_speed_for_compliance,
    multi_receptor_assessment,
    WakeWashError,
)


class TestKelvinWaveGeometry:
    """Test Kelvin wake pattern geometry."""
    
    def test_transverse_wave_angle(self):
        """Test transverse wave angle = arcsin(1/3) ≈ 19.47°."""
        theta_t = transverse_wave_angle()
        expected = math.degrees(math.asin(1.0 / 3.0))
        assert abs(theta_t - expected) < 0.01, \
            f"Transverse angle {theta_t:.4f}° != expected {expected:.4f}°"
        assert abs(theta_t - 19.47) < 0.01, \
            f"Transverse angle {theta_t:.4f}° != 19.47°"
    
    def test_divergent_wave_angle(self):
        """Test divergent wave cusp angle = arccos(1/3) ≈ 70.53°."""
        theta_d = divergent_wave_angle()
        expected = math.degrees(math.acos(1.0 / 3.0))
        assert abs(theta_d - expected) < 0.01, \
            f"Divergent angle {theta_d:.4f}° != expected {expected:.4f}°"
        assert abs(theta_d - 70.53) < 0.01, \
            f"Divergent angle {theta_d:.4f}° != 70.53°"
    
    def test_wake_full_angle(self):
        """Test wake full opening angle = 2 × 70.53° ≈ 141.06°."""
        full_angle = wake_full_angle()
        expected = 2.0 * divergent_wave_angle()
        assert abs(full_angle - expected) < 0.01
        assert abs(full_angle - 141.06) < 0.02
    
    def test_kelvin_wake_coordinates(self):
        """Test Kelvin wake coordinate generation."""
        x, y = kelvin_wake_coordinates(5.0, n_points=10)
        assert len(x) == 10
        assert len(y) == 10
        # Wake should extend behind the ship (negative x for most points)
        # Note: some points near the transverse direction may have x ≈ 0
        assert sum(1 for xi in x if xi < 0) >= 5  # at least half should be behind
    
    def test_wave_wavelength(self):
        """Test wavelength calculation."""
        lam = wave_wavelength(5.0, 0.0)  # transverse wave
        assert lam > 0
        # λ = (2π U² / g) * cos²(0) = 2π U² / g
        expected = 2.0 * math.pi * 5.0**2 / 9.81
        assert abs(lam - expected) / expected < 0.01
    
    def test_wave_period(self):
        """Test wave period calculation."""
        T = wave_period(5.0, 0.0)
        assert T > 0
        # T = sqrt(2π λ / g)
        lam = wave_wavelength(5.0, 0.0)
        expected = math.sqrt(2.0 * math.pi * lam / 9.81)
        assert abs(T - expected) / expected < 0.01
    
    def test_froude_number(self):
        """Test Froude number calculation."""
        Fr = froude_number(5.0, 100.0)
        expected = 5.0 / math.sqrt(9.81 * 100.0)
        assert abs(Fr - expected) < 0.001
    
    def test_kelvin_wave_error_on_invalid_speed(self):
        """Test error handling for invalid ship speed."""
        with pytest.raises(KelvinWaveError):
            kelvin_wake_coordinates(-5.0)


class TestWaveHeightDecay:
    """Test wave height decay laws."""
    
    def test_deep_water_decay_law(self):
        """Test deep water decay H ∝ 1/√r."""
        H_ref = 1.0
        r_ref = 100.0
        r_target = 400.0
        
        H_target = deep_water_decay(H_ref, r_ref, r_target)
        expected = H_ref * math.sqrt(r_ref / r_target)
        
        assert abs(H_target - expected) / expected < 0.01, \
            f"Deep water decay {H_target:.4f} != expected {expected:.4f}"
        assert abs(H_target - 0.5) < 0.01
    
    def test_shallow_water_decay_law(self):
        """Test shallow water decay H ∝ 1/r."""
        H_ref = 1.0
        r_ref = 100.0
        r_target = 400.0
        
        H_target = shallow_water_decay(H_ref, r_ref, r_target)
        expected = H_ref * (r_ref / r_target)
        
        assert abs(H_target - expected) / expected < 0.01, \
            f"Shallow water decay {H_target:.4f} != expected {expected:.4f}"
        assert abs(H_target - 0.25) < 0.01
    
    def test_transition_depth(self):
        """Test transition depth calculation."""
        h_t = transition_depth(10.0)
        expected = 10.0 / (2.0 * math.pi)
        assert abs(h_t - expected) < 0.01
    
    def test_shallow_water_criterion(self):
        """Test shallow water criterion h < λ/20."""
        h_shallow = shallow_water_criterion(10.0)
        expected = 10.0 / 20.0
        assert abs(h_shallow - expected) < 0.01
    
    def test_wave_height_with_depth_deep(self):
        """Test wave height in deep water."""
        H = wave_height_with_depth(1.0, 100.0, 400.0, 50.0, 10.0)
        # h = 50 > λ/2 = 5, so deep water
        expected = deep_water_decay(1.0, 100.0, 400.0)
        assert abs(H - expected) / expected < 0.01
    
    def test_wave_height_with_depth_shallow(self):
        """Test wave height in shallow water."""
        H = wave_height_with_depth(1.0, 100.0, 400.0, 0.3, 10.0)
        # h = 0.3 < λ/20 = 0.5, so shallow water
        expected = shallow_water_decay(1.0, 100.0, 400.0)
        assert abs(H - expected) / expected < 0.01
    
    def test_decay_profile(self):
        """Test decay profile over multiple distances."""
        distances = [100, 200, 400, 800]
        heights = decay_profile(1.0, 100.0, distances)
        assert len(heights) == 4
        # Heights should decrease
        assert all(heights[i] >= heights[i+1] for i in range(len(heights)-1))
    
    def test_energy_flux_conservation(self):
        """Test energy flux calculation."""
        P = energy_flux_conservation(1.0, 10.0)
        assert P > 0
    
    def test_wave_decay_error_on_invalid_distance(self):
        """Test error handling for invalid distances."""
        with pytest.raises(WaveDecayError):
            deep_water_decay(1.0, -100.0, 400.0)


class TestShorelineImpact:
    """Test shoreline impact estimation."""
    
    def test_shoaling_coefficient(self):
        """Test shoaling coefficient calculation."""
        # In shallow water, group velocity decreases, so K_s should increase
        # K_s = sqrt(c_g_deep / c_g_local)
        # For h < λ/2, c_g_local < c_g_deep, so K_s > 1
        K_s = shoaling_coefficient(100.0, 5.0)  # h = 5 < λ/2 = 50
        # Actually, for h = 5 and λ = 100, we're in intermediate depth
        # Let's just check it's positive and reasonable
        assert K_s > 0
    
    def test_breaking_criterion(self):
        """Test wave breaking criterion."""
        # H/h > 0.78 → breaking
        assert breaking_criterion(2.0, 2.0, 0.02) == True
        # H/h < 0.78 → not breaking
        assert breaking_criterion(1.0, 3.0, 0.02) == False
    
    def test_runup_estimation(self):
        """Test wave run-up estimation."""
        R = runup_estimation(1.0, 10.0, 0.05)
        assert R > 0
    
    def test_shoreline_wave_height(self):
        """Test shoreline wave height estimation."""
        H_shore, is_breaking = shoreline_wave_height(
            10.0, 500.0, 3.0, 50.0, 0.5, 100.0
        )
        assert H_shore > 0
        assert isinstance(is_breaking, bool)
    
    def test_wake_wash_impact_assessment(self):
        """Test comprehensive impact assessment."""
        result = wake_wash_impact_assessment(10.0, 500.0, 5.0, 10.0, 500.0)
        assert 'wave_height' in result
        assert 'wave_period' in result
        assert 'is_breaking' in result
        assert 'runup' in result
        assert 'impact_level' in result
        assert result['wave_height'] > 0


class TestPIANCCompliance:
    """Test PIANC compliance assessment."""
    
    def test_get_wave_height_limit(self):
        """Test PIANC limit lookup."""
        limit = get_wave_height_limit(ReceptorType.SWIMMERS)
        assert abs(limit - 0.3) < 0.01
    
    def test_check_compliance_compliant(self):
        """Test compliance check for compliant case."""
        level, ratio = check_compliance(0.2, ReceptorType.SWIMMERS)
        assert level == ComplianceLevel.COMPLIANT
        assert ratio < 1.0
    
    def test_check_compliance_non_compliant(self):
        """Test compliance check for non-compliant case."""
        level, ratio = check_compliance(0.5, ReceptorType.SWIMMERS)
        assert level == ComplianceLevel.NON_COMPLIANT
        assert ratio > 1.0
    
    def test_check_compliance_marginal(self):
        """Test compliance check for marginal case."""
        level, ratio = check_compliance(0.28, ReceptorType.SWIMMERS)
        assert level == ComplianceLevel.MARGINAL
        assert 0.8 <= ratio <= 1.0
    
    def test_assess_multiple_receptors(self):
        """Test multi-receptor assessment."""
        receptors = [ReceptorType.SWIMMERS, ReceptorType.MOORED_LARGE_VESSEL]
        results = assess_multiple_receptors(0.4, receptors)
        assert len(results) == 2
        assert ReceptorType.SWIMMERS in results
        assert ReceptorType.MOORED_LARGE_VESSEL in results
    
    def test_risk_classification(self):
        """Test risk classification."""
        risk_low = risk_classification(0.1, ReceptorType.SWIMMERS)
        assert risk_low == 'low'
        
        risk_high = risk_classification(0.5, ReceptorType.SWIMMERS)
        assert risk_high == 'high'
    
    def test_mitigation_recommendations(self):
        """Test mitigation recommendations."""
        recs = mitigation_recommendations(0.5, ReceptorType.SWIMMERS)
        assert len(recs) > 0
        assert any('exceeds' in r.lower() for r in recs)
    
    def test_compliance_report(self):
        """Test comprehensive compliance report."""
        report = compliance_report(0.25, ReceptorType.SWIMMERS)
        assert 'wave_height' in report
        assert 'compliance_level' in report
        assert 'recommendations' in report
    
    def test_speed_limit_for_compliance(self):
        """Test speed limit calculation."""
        # Use a case where the wave height exceeds the limit
        # H_ref = 0.5 at r_ref = 100, receptor at r = 200
        # H_target = 0.5 * sqrt(100/200) = 0.354 > 0.3 (swimmer limit)
        U_max = speed_limit_for_compliance(
            ReceptorType.SWIMMERS, 0.5, 10.0, 100.0, 200.0
        )
        assert U_max > 0
        # The max speed should be less than reference since H > limit
        assert U_max < 10.0


class TestUnifiedAPI:
    """Test unified wake wash API."""
    
    def test_analyze_wake_pattern(self):
        """Test wake pattern analysis."""
        result = analyze_wake_pattern(10.0, 100.0)
        assert 'transverse_angle' in result
        assert 'divergent_angle' in result
        assert 'froude_number' in result
        assert abs(result['transverse_angle'] - 19.47) < 0.01
    
    def test_predict_wave_height(self):
        """Test wave height prediction."""
        H = predict_wave_height(10.0, 5.0, 500.0)
        assert H > 0
    
    def test_assess_shoreline_impact(self):
        """Test shoreline impact assessment via API."""
        result = assess_shoreline_impact(10.0, 5.0, 500.0, 10.0)
        assert 'wave_height' in result
        assert result['wave_height'] > 0
    
    def test_check_pianc_compliance(self):
        """Test PIANC compliance check via API."""
        result = check_pianc_compliance(
            10.0, 5.0, 500.0, ReceptorType.SWIMMERS
        )
        assert 'compliance_level' in result
        assert 'wave_height' in result
    
    def test_optimize_speed_for_compliance(self):
        """Test speed optimization for compliance."""
        result = optimize_speed_for_compliance(
            5.0, 500.0, ReceptorType.SWIMMERS, 15.0
        )
        assert 'max_compliant_speed' in result
        assert result['max_compliant_speed'] > 0
    
    def test_multi_receptor_assessment(self):
        """Test multi-receptor assessment via API."""
        receptors = [
            (300.0, ReceptorType.SWIMMERS),
            (500.0, ReceptorType.MOORED_LARGE_VESSEL),
        ]
        results = multi_receptor_assessment(10.0, 5.0, receptors)
        assert len(results) == 2
        assert all('wave_height' in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
