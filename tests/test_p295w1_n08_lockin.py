"""Tests for P295W1-N08: ice-induced frequency lock-in parameter space mapping.

Validates:
1. Ice breaking frequency model physics
2. Lock-in criterion and bandwidth
3. Parameter space sweep correctness
4. Kujala benchmark acceptance criteria
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Add scenario to path
SCENARIO = WORKTREE / "scenarios" / "p295w1_n08"
if str(SCENARIO) not in sys.path:
    sys.path.insert(0, str(SCENARIO))

from scenarios.p295w1_n08.ice_lockin_parameter_space import (
    IceBreakingFrequencyModel,
    LockInParameterSpace,
    KujalaBenchmark,
    run_parameter_sweep,
    generate_benchmark_data,
)


class TestIceBreakingFrequencyModel:
    """Tests for ice breaking frequency model."""
    
    def test_buoyancy_length_scaling(self):
        """Buoyancy length scales as h^(3/4)."""
        model = IceBreakingFrequencyModel()
        L1 = model.buoyancy_length(0.5)
        L2 = model.buoyancy_length(1.0)
        # L_b ~ h^(3/4), so L2/L1 = (1.0/0.5)^(3/4) = 2^(3/4) ≈ 1.682
        ratio = L2 / L1
        expected_ratio = 2.0**0.75
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)
    
    def test_ice_breaking_frequency_positive(self):
        """Ice breaking frequency must be positive for positive V and h."""
        model = IceBreakingFrequencyModel()
        for V in [0.5, 1.0, 2.0]:
            for h in [0.3, 0.8, 1.5]:
                f = model.ice_breaking_frequency(V, h, "flexural")
                assert f > 0
                f = model.ice_breaking_frequency(V, h, "crushing")
                assert f > 0
    
    def test_ice_breaking_frequency_increases_with_speed(self):
        """f_ice increases linearly with V."""
        model = IceBreakingFrequencyModel()
        h = 1.0
        f1 = model.ice_breaking_frequency(1.0, h)
        f2 = model.ice_breaking_frequency(2.0, h)
        assert f2 == pytest.approx(2 * f1, rel=1e-10)
    
    def test_lock_in_speed_range_contains_natural_frequency(self):
        """Lock-in speed range should produce f_ice near f_n."""
        model = IceBreakingFrequencyModel(f_n=1.2)
        h = 1.0
        V_min, V_max = model.lock_in_speed_range(h, delta=0.05)
        
        # At V_min, f_ice should be ~0.95 * f_n
        f_min = model.ice_breaking_frequency(V_min, h)
        assert f_min == pytest.approx(0.95 * model.f_n, rel=1e-6)
        
        # At V_max, f_ice should be ~1.05 * f_n
        f_max = model.ice_breaking_frequency(V_max, h)
        assert f_max == pytest.approx(1.05 * model.f_n, rel=1e-6)
    
    def test_rejects_invalid_inputs(self):
        """Model should reject non-positive inputs."""
        model = IceBreakingFrequencyModel()
        with pytest.raises(ValueError):
            model.ice_breaking_frequency(0.0, 1.0)
        with pytest.raises(ValueError):
            model.ice_breaking_frequency(1.0, 0.0)
        with pytest.raises(ValueError):
            model.ice_breaking_frequency(-1.0, 1.0)


class TestLockInParameterSpace:
    """Tests for parameter space sweep."""
    
    def test_sweep_produces_correct_shapes(self):
        """Sweep should produce arrays of correct dimensions."""
        model = IceBreakingFrequencyModel()
        sweep = LockInParameterSpace(model=model, n_V=10, n_h=8)
        results = sweep.sweep()
        
        assert len(results["V_arr"]) == 10
        assert len(results["h_arr"]) == 8
        assert len(results["lock_in_map"]) == 8
        assert len(results["lock_in_map"][0]) == 10
        assert len(results["f_ice_map"]) == 8
        assert len(results["amplification_map"]) == 8
    
    def test_lock_in_fraction_in_range(self):
        """Lock-in fraction should be between 0 and 1."""
        model = IceBreakingFrequencyModel()
        sweep = LockInParameterSpace(model=model, n_V=20, n_h=15)
        results = sweep.sweep()
        
        assert 0.0 <= results["lock_in_fraction"] <= 1.0
        assert results["lock_in_points"] <= results["total_points"]
    
    def test_bandwidth_data_consistency(self):
        """Bandwidth data should be consistent with lock-in map."""
        model = IceBreakingFrequencyModel()
        sweep = LockInParameterSpace(model=model, n_V=30, n_h=10)
        results = sweep.sweep()
        
        for entry in results["bandwidth_data"]:
            assert entry["V_min_m_s"] <= entry["V_max_m_s"]
            assert entry["bandwidth_m_s"] >= 0
            assert entry["bandwidth_m_s"] == pytest.approx(
                entry["V_max_m_s"] - entry["V_min_m_s"], rel=1e-10
            )


class TestKujalaBenchmark:
    """Tests for Kujala benchmark comparison."""
    
    def test_benchmark_produces_required_fields(self):
        """Benchmark should produce all required comparison fields."""
        model = IceBreakingFrequencyModel(f_n=1.2)
        benchmark = KujalaBenchmark(model=model)
        results = benchmark.compare(mode="flexural")
        
        required_fields = [
            "observed_V_range_m_s",
            "predicted_V_range_m_s",
            "V_min_relative_error",
            "V_max_relative_error",
            "frequency_relative_error",
            "lock_in_coverage",
        ]
        for field in required_fields:
            assert field in results
    
    def test_frequency_error_acceptable(self):
        """Frequency error should be ≤15% (acceptance criterion)."""
        model = IceBreakingFrequencyModel(f_n=1.2)
        benchmark = KujalaBenchmark(model=model)
        results = benchmark.compare(mode="flexural")
        
        assert results["frequency_relative_error"] <= 0.15, (
            f"Frequency error {results['frequency_relative_error']:.2%} exceeds 15% threshold"
        )
    
    def test_coverage_acceptable(self):
        """Lock-in coverage should be ≥80% (acceptance criterion)."""
        model = IceBreakingFrequencyModel(f_n=1.2)
        benchmark = KujalaBenchmark(model=model)
        results = benchmark.compare(mode="flexural")
        
        assert results["lock_in_coverage"] >= 0.80, (
            f"Coverage {results['lock_in_coverage']:.2%} below 80% threshold"
        )


class TestRunParameterSweep:
    """Integration tests for the full sweep."""
    
    def test_run_parameter_sweep_default(self):
        """Default sweep should complete and produce valid results."""
        results = run_parameter_sweep()
        
        assert "metadata" in results
        assert results["metadata"]["f_n_hz"] == 1.2
        assert results["lock_in_fraction"] >= 0.0
        assert len(results["bandwidth_data"]) > 0
    
    def test_run_parameter_sweep_custom_range(self):
        """Custom range sweep should respect bounds."""
        results = run_parameter_sweep(
            V_range=(1.0, 2.0),
            h_range=(0.5, 1.0),
            f_n=1.5,
        )
        
        assert results["metadata"]["V_range_m_s"] == [1.0, 2.0]
        assert results["metadata"]["h_range_m"] == [0.5, 1.0]
        assert results["metadata"]["f_n_hz"] == 1.5


class TestGenerateBenchmarkData:
    """Integration tests for benchmark data generation."""
    
    def test_generate_benchmark_data(self):
        """Benchmark data generation should produce both modes."""
        results = generate_benchmark_data()
        
        assert "flexural" in results
        assert "crushing" in results
        assert "reference" in results
        assert "note" in results


class TestAcceptanceCriteria:
    """Final acceptance criteria tests."""
    
    def test_all_acceptance_criteria(self):
        """All acceptance criteria must be satisfied."""
        # Run sweep
        sweep = run_parameter_sweep(
            V_range=(0.5, 3.0),
            h_range=(0.3, 1.5),
            f_n=1.2,
            zeta=0.02,
        )
        
        # Run benchmark
        benchmark = generate_benchmark_data()
        flex = benchmark["flexural"]
        
        # Criterion 1: Main frequency error ≤ 15%
        freq_error = flex["frequency_relative_error"]
        assert freq_error <= 0.15, f"Frequency error {freq_error:.2%} > 15%"
        
        # Criterion 2: Lock-in interval coverage ≥ 80%
        coverage = flex["lock_in_coverage"]
        assert coverage >= 0.80, f"Coverage {coverage:.2%} < 80%"
        
        # Criterion 3: Lock-in fraction should be non-trivial
        assert sweep["lock_in_fraction"] > 0.01, "Lock-in fraction too small"
        
        # Criterion 4: Bandwidth data should exist
        assert len(sweep["bandwidth_data"]) > 0, "No bandwidth data"
