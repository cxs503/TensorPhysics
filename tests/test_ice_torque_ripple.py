"""
Pytest test cases for propeller-ice torque ripple analysis.

Tests the PropellerIceInteraction model and validation against reference values.
"""

import pytest
import numpy as np
from src.tensorphysics.propulsion.ice_torque import (
    PropellerIceInteraction,
    validate_torque_ripple_analysis,
    get_reference_values,
    run_complete_analysis,
)


class TestPropellerIceInteraction:
    """Test suite for propeller-ice interaction model."""
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters."""
        model = PropellerIceInteraction(
            propeller_diameter=4.0,
            num_blades=4,
            pitch_ratio=0.8,
            rotational_speed=2.0,
        )
        
        assert model.D == 4.0
        assert model.Z == 4
        assert model.P_D == 0.8
        assert model.n == 2.0
        assert model.blade_passing_freq == 8.0
    
    def test_open_water_torque(self):
        """Test open-water torque computation."""
        model = PropellerIceInteraction()
        Q_ow = model.compute_open_water_torque()
        
        # Should be positive and in reasonable range
        assert Q_ow > 0
        assert Q_ow > 50000  # At least 50 kNm for this size
        assert Q_ow < 500000  # Less than 500 kNm
    
    def test_ice_field_generation(self):
        """Test ice field generation."""
        model = PropellerIceInteraction()
        ice_field = model.generate_ice_field(duration=5.0, dt=0.001)
        
        # Should have columns: time, size, angle
        assert ice_field.shape[1] == 3
        
        # Times should be within duration
        assert np.all(ice_field[:, 0] >= 0)
        assert np.all(ice_field[:, 0] < 5.0)
        
        # Sizes should be positive
        assert np.all(ice_field[:, 1] > 0)
        
        # Angles should be in [0, 2π]
        assert np.all(ice_field[:, 2] >= 0)
        assert np.all(ice_field[:, 2] <= 2 * np.pi)
    
    def test_torque_time_series_simulation(self):
        """Test torque time series simulation."""
        model = PropellerIceInteraction()
        t, Q = model.simulate_torque_time_series(
            duration=2.0,
            dt=0.001,
            seed=42
        )
        
        # Check arrays have same length
        assert len(t) == len(Q)
        
        # Check time array
        assert t[0] == 0.0
        assert t[-1] < 2.0
        assert np.all(np.diff(t) > 0)  # Monotonically increasing
        
        # Check torque is positive
        assert np.all(Q > 0)
        
        # Check torque has fluctuations
        Q_std = np.std(Q)
        assert Q_std > 0
    
    def test_torque_statistics(self):
        """Test torque statistics computation."""
        model = PropellerIceInteraction()
        t, Q = model.simulate_torque_time_series(
            duration=5.0,
            dt=0.001,
            seed=42
        )
        
        stats = model.compute_torque_statistics(t, Q)
        
        # Check all required keys exist
        required_keys = [
            'mean_torque', 'rms_fluctuation', 'rms_ratio',
            'peak_fluctuation', 'peak_factor', 'std_deviation',
            'skewness', 'kurtosis', 'max_torque', 'min_torque'
        ]
        for key in required_keys:
            assert key in stats
        
        # Check physical consistency
        assert stats['mean_torque'] > 0
        assert stats['rms_fluctuation'] > 0
        assert stats['rms_ratio'] > 0
        assert stats['peak_factor'] > 0
        assert stats['max_torque'] >= stats['mean_torque']
        assert stats['min_torque'] <= stats['mean_torque']
    
    def test_torque_spectrum(self):
        """Test torque spectrum computation."""
        model = PropellerIceInteraction()
        t, Q = model.simulate_torque_time_series(
            duration=5.0,
            dt=0.001,
            seed=42
        )
        
        freqs, psd = model.compute_torque_spectrum(t, Q)
        
        # Check arrays have same length
        assert len(freqs) == len(psd)
        
        # Check frequencies are positive
        assert np.all(freqs >= 0)
        
        # Check PSD is non-negative
        assert np.all(psd >= 0)
    
    def test_dominant_frequency(self):
        """Test dominant frequency detection."""
        model = PropellerIceInteraction()
        t, Q = model.simulate_torque_time_series(
            duration=10.0,
            dt=0.001,
            seed=42
        )
        
        freqs, psd = model.compute_torque_spectrum(t, Q)
        dominant_freq = model.find_dominant_frequency(freqs, psd)
        
        # Should be close to blade-passing frequency (8 Hz)
        assert dominant_freq > 0
        assert abs(dominant_freq - model.blade_passing_freq) < 2.0


class TestValidation:
    """Test suite for validation against reference values."""
    
    def test_reference_values(self):
        """Test that reference values are reasonable."""
        ref_stats, ref_freq = get_reference_values()
        
        assert ref_stats['mean_torque'] > 0
        assert ref_stats['rms_ratio'] > 0
        assert ref_stats['rms_ratio'] < 1.0  # RMS should be less than mean
        assert ref_stats['peak_factor'] > 1.0  # Peak factor > 1
        assert ref_freq > 0
    
    def test_validation_function(self):
        """Test validation function."""
        computed_stats = {
            'rms_ratio': 0.28,
            'peak_factor': 3.5,
        }
        computed_freq = 8.0
        
        ref_stats = {
            'rms_ratio': 0.28,
            'peak_factor': 3.5,
        }
        ref_freq = 8.0
        
        validation = validate_torque_ripple_analysis(
            computed_stats, computed_freq, ref_stats, ref_freq
        )
        
        # Perfect match should pass all criteria
        assert validation['torque_RMS_error'] == 0.0
        assert validation['peak_factor_error'] == 0.0
        assert validation['dominant_frequency_match'] == 0.0
        assert validation['all_pass'] is True
    
    def test_validation_with_errors(self):
        """Test validation with some errors."""
        computed_stats = {
            'rms_ratio': 0.30,  # 7% error
            'peak_factor': 3.8,  # 8.6% error
        }
        computed_freq = 8.5  # 6.25% error
        
        ref_stats = {
            'rms_ratio': 0.28,
            'peak_factor': 3.5,
        }
        ref_freq = 8.0
        
        validation = validate_torque_ripple_analysis(
            computed_stats, computed_freq, ref_stats, ref_freq
        )
        
        # Should pass all criteria (errors < thresholds)
        assert validation['torque_RMS_error'] < 0.15
        assert validation['peak_factor_error'] < 0.2
        assert validation['dominant_frequency_match'] < 0.1
        assert validation['all_pass'] is True


class TestCompleteAnalysis:
    """Test suite for complete analysis workflow."""
    
    def test_complete_analysis_passes_criteria(self):
        """Test that complete analysis passes all validation criteria."""
        results = run_complete_analysis(seed=42, duration=10.0, dt=0.001)
        
        # Check structure
        assert 'model_parameters' in results
        assert 'computed_statistics' in results
        assert 'computed_dominant_frequency_hz' in results
        assert 'reference_statistics' in results
        assert 'validation' in results
        assert 'time_series' in results
        assert 'spectrum' in results
        
        # Check validation passes
        assert results['validation']['all_pass'] == True
        assert results['validation']['torque_RMS_error'] < 0.15
        assert results['validation']['peak_factor_error'] < 0.2
        assert results['validation']['dominant_frequency_match'] < 0.1
    
    def test_complete_analysis_reproducibility(self):
        """Test that analysis is reproducible with same seed."""
        results1 = run_complete_analysis(seed=123, duration=5.0, dt=0.001)
        results2 = run_complete_analysis(seed=123, duration=5.0, dt=0.001)
        
        # Should be identical
        assert results1['computed_statistics']['rms_ratio'] == \
               results2['computed_statistics']['rms_ratio']
        assert results1['computed_dominant_frequency_hz'] == \
               results2['computed_dominant_frequency_hz']
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        results1 = run_complete_analysis(seed=42, duration=5.0, dt=0.001)
        results2 = run_complete_analysis(seed=99, duration=5.0, dt=0.001)
        
        # RMS ratios should be different (but both should pass validation)
        assert results1['computed_statistics']['rms_ratio'] != \
               results2['computed_statistics']['rms_ratio']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
