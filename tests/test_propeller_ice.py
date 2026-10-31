"""Pytest tests for propeller–ice interaction model (m722).

Machine acceptance criteria:
  1. ice_milling_force_error_vs_experiment < 0.2 [ratio]
  2. torque_fluctuation_spectrum_match < 0.25 [ratio]
  3. pytest_test_propeller_ice == PASS [bool]
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from tensorphysics.propeller_ice import (
    PropellerGeometry,
    IceProperties,
    OperatingCondition,
    ice_milling_force,
    torque_spectrum,
    validate_against_experiment,
    run_propeller_ice_simulation,
    _milling_coefficient,
)


# ---------------------------------------------------------------------------
# Configuration validation tests
# ---------------------------------------------------------------------------

class TestPropellerGeometry:
    def test_default_construction(self):
        p = PropellerGeometry()
        assert p.D == 4.0
        assert p.n_blades == 4
        assert p.R == 2.0
        assert p.pitch == 3.2

    def test_invalid_diameter(self):
        with pytest.raises(ValueError, match="D must be positive"):
            PropellerGeometry(D=-1.0)

    def test_invalid_blades(self):
        with pytest.raises(ValueError, match="n_blades"):
            PropellerGeometry(n_blades=0)

    def test_mean_chord(self):
        p = PropellerGeometry(D=6.0, chord_fraction=0.3)
        assert abs(p.mean_chord - 0.3 * 3.0) < 1e-10


class TestIceProperties:
    def test_default_construction(self):
        ice = IceProperties()
        assert ice.sigma_c == 5.0e6
        assert ice.rho_ice == 917.0

    def test_invalid_strength(self):
        with pytest.raises(ValueError):
            IceProperties(sigma_c=-1.0)


class TestOperatingCondition:
    def test_omega(self):
        op = OperatingCondition(rpm=120.0)
        assert abs(op.omega - 2 * math.pi * 2.0) < 1e-10

    def test_invalid_rpm(self):
        with pytest.raises(ValueError):
            OperatingCondition(rpm=0.0)

    def test_invalid_concentration(self):
        with pytest.raises(ValueError):
            OperatingCondition(ice_concentration=1.5)


# ---------------------------------------------------------------------------
# Milling force model tests
# ---------------------------------------------------------------------------

class TestIceMillingForce:
    """Test the ice milling force computation."""

    def test_force_positive(self):
        """Milling force must be positive."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        result = ice_milling_force(prop, ice, op, n_samples=500, seed=42)
        assert result['mean_force'] > 0
        assert result['max_force'] > 0

    def test_force_increases_with_ice_strength(self):
        """Stronger ice → higher milling force."""
        prop = PropellerGeometry()
        op = OperatingCondition()
        ice_weak = IceProperties(sigma_c=2.0e6)
        ice_strong = IceProperties(sigma_c=10.0e6)
        r_weak = ice_milling_force(prop, ice_weak, op, n_samples=500, seed=42)
        r_strong = ice_milling_force(prop, ice_strong, op, n_samples=500, seed=42)
        assert r_strong['mean_force'] > r_weak['mean_force']

    def test_force_increases_with_concentration(self):
        """Higher ice concentration → higher force."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op_sparse = OperatingCondition(ice_concentration=0.1)
        op_dense = OperatingCondition(ice_concentration=0.8)
        r_sparse = ice_milling_force(prop, ice, op_sparse, n_samples=500, seed=42)
        r_dense = ice_milling_force(prop, ice, op_dense, n_samples=500, seed=42)
        assert r_dense['mean_force'] > r_sparse['mean_force']

    def test_force_increases_with_rpm(self):
        """Higher RPM → higher relative velocity → higher force."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op_slow = OperatingCondition(rpm=60.0)
        op_fast = OperatingCondition(rpm=180.0)
        r_slow = ice_milling_force(prop, ice, op_slow, n_samples=500, seed=42)
        r_fast = ice_milling_force(prop, ice, op_fast, n_samples=500, seed=42)
        assert r_fast['mean_force'] > r_slow['mean_force']

    def test_time_series_length(self):
        """Time series must have correct length."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        result = ice_milling_force(prop, ice, op, n_samples=1000, seed=42)
        assert len(result['force_time_series']) > 0
        assert len(result['torque_time_series']) == len(result['force_time_series'])

    def test_reproducibility(self):
        """Same seed → same result."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        r1 = ice_milling_force(prop, ice, op, n_samples=500, seed=123)
        r2 = ice_milling_force(prop, ice, op, n_samples=500, seed=123)
        assert abs(r1['mean_force'] - r2['mean_force']) < 1e-10

    def test_torque_positive(self):
        """Torque must be positive (resisting torque)."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        result = ice_milling_force(prop, ice, op, n_samples=500, seed=42)
        assert result['mean_torque'] > 0


# ---------------------------------------------------------------------------
# Torque spectrum tests
# ---------------------------------------------------------------------------

class TestTorqueSpectrum:
    """Test torque fluctuation spectrum computation."""

    def test_spectrum_computed(self):
        """Spectrum must be computed without error."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        result = ice_milling_force(prop, ice, op, n_samples=2000, seed=42)
        spec = torque_spectrum(
            result['torque_time_series'],
            result['dt'],
            op.rpm,
            prop.n_blades,
        )
        assert len(spec['freq']) > 0
        assert len(spec['psd']) > 0
        assert spec['bpf'] > 0

    def test_bpf_correct(self):
        """BPF must equal n_blades × RPM/60."""
        prop = PropellerGeometry(n_blades=5)
        op = OperatingCondition(rpm=100.0)
        ice = IceProperties()
        result = ice_milling_force(prop, ice, op, n_samples=2000, seed=42)
        spec = torque_spectrum(
            result['torque_time_series'],
            result['dt'],
            op.rpm,
            prop.n_blades,
        )
        expected_bpf = 5 * 100.0 / 60.0
        assert abs(spec['bpf'] - expected_bpf) < 1e-10

    def test_peak_at_bpf(self):
        """PSD should have elevated power near BPF."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        result = ice_milling_force(prop, ice, op, n_samples=5000, seed=42)
        spec = torque_spectrum(
            result['torque_time_series'],
            result['dt'],
            op.rpm,
            prop.n_blades,
        )
        # Peak ratio should be > 1 (energy concentrated at BPF)
        assert spec['peak_ratio_bpf'] > 1.0

    def test_fluctuation_coefficient_positive(self):
        """Cv must be positive."""
        prop = PropellerGeometry()
        ice = IceProperties()
        op = OperatingCondition()
        result = ice_milling_force(prop, ice, op, n_samples=2000, seed=42)
        spec = torque_spectrum(
            result['torque_time_series'],
            result['dt'],
            op.rpm,
            prop.n_blades,
        )
        assert spec['torque_fluctuation_coeff'] > 0


# ---------------------------------------------------------------------------
# Validation against experiment (machine acceptance criteria)
# ---------------------------------------------------------------------------

class TestValidationAgainstExperiment:
    """Machine acceptance criteria tests.

    Reference experimental data from Aker Arctic model tests
    (representative values for a 4-blade CPP propeller in broken ice):
      - Mean ice milling force: ~150 kN (full scale)
      - Torque std: ~25 kN·m
      - BPF peak ratio: ~8.0
    """

    # Experimental reference values (Aker Arctic / KVLY model tests, scaled)
    EXP_MEAN_FORCE = 150_000.0   # N
    EXP_TORQUE_STD = 25_000.0    # N·m
    EXP_BPF_PEAK_RATIO = 8.0

    def test_force_error_below_threshold(self):
        """Criterion 1: ice_milling_force_error_vs_experiment < 0.2."""
        prop = PropellerGeometry()
        ice = IceProperties(sigma_c=5.0e6)
        op = OperatingCondition(rpm=120.0, Va=3.0, ice_concentration=0.3)
        val = validate_against_experiment(
            prop, ice, op,
            experimental_mean_force=self.EXP_MEAN_FORCE,
            experimental_torque_std=self.EXP_TORQUE_STD,
            experimental_bpf_peak_ratio=self.EXP_BPF_PEAK_RATIO,
            seed=42,
        )
        assert val['force_error_ratio'] < 0.2, (
            f"Force error {val['force_error_ratio']:.3f} >= 0.2 "
            f"(model={val['F_model_N']:.0f} N, exp={val['F_exp_N']:.0f} N)"
        )

    def test_torque_spectrum_match_below_threshold(self):
        """Criterion 2: torque_fluctuation_spectrum_match < 0.25."""
        prop = PropellerGeometry()
        ice = IceProperties(sigma_c=5.0e6)
        op = OperatingCondition(rpm=120.0, Va=3.0, ice_concentration=0.3)
        val = validate_against_experiment(
            prop, ice, op,
            experimental_mean_force=self.EXP_MEAN_FORCE,
            experimental_torque_std=self.EXP_TORQUE_STD,
            experimental_bpf_peak_ratio=self.EXP_BPF_PEAK_RATIO,
            seed=42,
        )
        # Combined spectrum metric: average of torque_std and BPF peak errors
        spectrum_error = 0.5 * (val['torque_std_error_ratio'] + val['bpf_peak_error_ratio'])
        assert spectrum_error < 0.25, (
            f"Spectrum match error {spectrum_error:.3f} >= 0.25 "
            f"(torque_err={val['torque_std_error_ratio']:.3f}, "
            f"bpf_err={val['bpf_peak_error_ratio']:.3f})"
        )

    def test_all_pass_flags(self):
        """All validation pass flags should be True."""
        prop = PropellerGeometry()
        ice = IceProperties(sigma_c=5.0e6)
        op = OperatingCondition(rpm=120.0, Va=3.0, ice_concentration=0.3)
        val = validate_against_experiment(
            prop, ice, op,
            experimental_mean_force=self.EXP_MEAN_FORCE,
            experimental_torque_std=self.EXP_TORQUE_STD,
            experimental_bpf_peak_ratio=self.EXP_BPF_PEAK_RATIO,
            seed=42,
        )
        assert val['pass_force'] is True
        assert val['pass_torque'] is True
        assert val['pass_spectrum'] is True


# ---------------------------------------------------------------------------
# Full simulation pipeline test
# ---------------------------------------------------------------------------

class TestFullSimulation:
    def test_run_simulation(self):
        """Full pipeline must complete and return valid structure."""
        result = run_propeller_ice_simulation(seed=42)
        assert 'propeller' in result
        assert 'ice' in result
        assert 'operating' in result
        assert 'force_statistics' in result
        assert 'torque_statistics' in result
        assert 'spectrum' in result
        assert result['force_statistics']['mean_force_N'] > 0
        assert result['torque_statistics']['mean_torque_Nm'] > 0
        assert result['spectrum']['bpf_Hz'] > 0


# ---------------------------------------------------------------------------
# Milling coefficient tests
# ---------------------------------------------------------------------------

class TestMillingCoefficient:
    def test_range(self):
        """K_m must be in physical range [0.1, 2.0]."""
        for skew in [0, 15, 25, 35, 45]:
            prop = PropellerGeometry(skew_angle_deg=skew)
            K_m = _milling_coefficient(prop)
            assert 0.1 <= K_m <= 2.0

    def test_skew_reduces_coefficient(self):
        """Higher skew → lower K_m (more gradual engagement)."""
        prop_low = PropellerGeometry(skew_angle_deg=10.0)
        prop_high = PropellerGeometry(skew_angle_deg=40.0)
        assert _milling_coefficient(prop_high) < _milling_coefficient(prop_low)
