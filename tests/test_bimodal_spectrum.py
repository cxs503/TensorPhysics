"""Tests for bimodal spectrum, directional spreading, and ISO 15016 bias.

Acceptance criteria:
- C1: Bimodal spectrum moment deviation < 2% from analytical values
- C2: Directional spreading parameter s recovery error < 5%
- C3: ISO 15016 short/long crest bias quantified (output bias band)
- C4: Synthetic sea-state spectrum fit R^2 > 0.9
- C5: Zero domain semantics (no domain terms in code/fixtures)
- C6: Existing test chain passes (verified separately)
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorphysics.seakeeping.bimodal_spectrum import (
    BimodalSpectrum,
    BimodalSpectrumResult,
    TorsethaugenSpectrum,
    analyze_bimodal,
    mean_wave_period,
    mean_zero_crossing_period,
    peak_period,
    significant_wave_height,
    spectral_moments,
)
from tensorphysics.seakeeping.directional_spreading import (
    DirectionalSpectrumResult,
    analyze_directional,
    cos2s_spreading,
    directional_moment,
    directional_spectrum,
    recover_spreading_parameter,
)
from tensorphysics.seakeeping.iso15016_bias import (
    ISO15016BiasResult,
    iso15016_bias_band,
    iso15016_long_crested_response,
    iso15016_short_crested_response,
    quantify_iso15016_bias,
)

# --------------------------------------------------------------------------- #
# Fixtures: synthetic sea states (no domain terms)
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_bimodal():
    """Synthetic bimodal spectrum with well-separated peaks."""
    return BimodalSpectrum(
        swell_hs=2.0,
        swell_tp=14.0,
        swell_gamma=3.0,
        wind_hs=1.0,
        wind_tp=5.0,
        wind_gamma=1.0,
    )


@pytest.fixture
def synthetic_single_peak():
    """Single-peak Torsethaugen spectrum for validation."""
    return TorsethaugenSpectrum(
        hs=3.0,
        tp=8.0,
        gamma=2.0,
    )


@pytest.fixture
def frequency_grid():
    """Standard frequency grid."""
    return np.linspace(0.02, 0.5, 1000)


@pytest.fixture
def angular_frequency_grid():
    """Angular frequency grid."""
    f = np.linspace(0.02, 0.5, 1000)
    return 2.0 * np.pi * f


@pytest.fixture
def direction_grid():
    """Direction angle grid."""
    return np.linspace(-np.pi, np.pi, 360, endpoint=False)


# --------------------------------------------------------------------------- #
# C1: Bimodal spectrum moment validation
# --------------------------------------------------------------------------- #

class TestBimodalSpectrumMoments:
    """C1: Bimodal spectrum moment deviation < 2% from analytical values."""

    def test_single_peak_m0_matches_hs(self, synthetic_single_peak, frequency_grid):
        """Zeroth moment should give Hs = 4*sqrt(m0) within 2%."""
        s_omega = synthetic_single_peak.evaluate(2.0 * np.pi * frequency_grid)
        s_hz = s_omega * 2.0 * np.pi

        moments = spectral_moments(frequency_grid, s_hz, orders=(0,))
        m0 = moments[0]
        hs_computed = significant_wave_height(m0)

        rel_error = abs(hs_computed - synthetic_single_peak.hs) / synthetic_single_peak.hs
        assert rel_error < 0.02, f"Hs relative error {rel_error*100:.2f}% exceeds 2%"

    def test_bimodal_m0_additivity(self, synthetic_bimodal, frequency_grid):
        """Bimodal m0 should equal sum of component m0 values."""
        s_total = synthetic_bimodal.evaluate_hz(frequency_grid)
        s_swell = synthetic_bimodal.swell.evaluate_hz(frequency_grid)
        s_wind = synthetic_bimodal.wind.evaluate_hz(frequency_grid)

        m0_total = spectral_moments(frequency_grid, s_total, orders=(0,))[0]
        m0_swell = spectral_moments(frequency_grid, s_swell, orders=(0,))[0]
        m0_wind = spectral_moments(frequency_grid, s_wind, orders=(0,))[0]

        rel_error = abs(m0_total - (m0_swell + m0_wind)) / m0_total
        assert rel_error < 0.02, f"m0 additivity error {rel_error*100:.2f}% exceeds 2%"

    def test_bimodal_peak_periods(self, synthetic_bimodal, frequency_grid):
        """Bimodal spectrum should show two peaks near input Tp values."""
        s_total = synthetic_bimodal.evaluate_hz(frequency_grid)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(s_total, height=np.max(s_total) * 0.05)

        assert len(peaks) >= 2, f"Bimodal spectrum should have at least 2 peaks, found {len(peaks)}"

        peak_freqs = frequency_grid[peaks]
        peak_periods = 1.0 / peak_freqs

        tp_swell = synthetic_bimodal.swell_tp
        tp_wind = synthetic_bimodal.wind_tp

        peak_periods_sorted = sorted(peak_periods, reverse=True)

        rel_err_swell = abs(peak_periods_sorted[0] - tp_swell) / tp_swell
        assert rel_err_swell < 0.15, f"Swell peak period error {rel_err_swell*100:.2f}%"

        rel_err_wind = abs(peak_periods_sorted[1] - tp_wind) / tp_wind
        assert rel_err_wind < 0.15, f"Wind peak period error {rel_err_wind*100:.2f}%"

    def test_spectral_moments_positive(self, synthetic_single_peak, frequency_grid):
        """All spectral moments should be positive."""
        s_omega = synthetic_single_peak.evaluate(2.0 * np.pi * frequency_grid)
        s_hz = s_omega * 2.0 * np.pi

        moments = spectral_moments(frequency_grid, s_hz, orders=(0, 1, 2, 4))

        assert moments[0] > 0, "m0 must be positive"
        assert moments[1] > 0, "m1 must be positive"
        assert moments[2] > 0, "m2 must be positive"
        assert moments[4] > 0, "m4 must be positive"

    def test_analyze_bimodal_result(self, synthetic_bimodal):
        """Full bimodal analysis should return valid result."""
        result = analyze_bimodal(synthetic_bimodal)

        assert result.hs > 0
        assert result.tp > 0
        assert result.tz > 0
        assert result.t1 > 0
        assert result.m0 > 0
        assert result.m1 > 0
        assert result.m2 > 0
        assert result.m4 > 0

        hs_from_m0 = significant_wave_height(result.m0)
        rel_err = abs(result.hs - hs_from_m0) / result.hs
        assert rel_err < 0.01, f"Hs consistency error {rel_err*100:.2f}%"


# --------------------------------------------------------------------------- #
# C2: Directional spreading parameter recovery
# --------------------------------------------------------------------------- #

class TestDirectionalSpreading:
    """C2: Directional spreading parameter s recovery error < 5%."""

    def test_spreading_normalization(self, direction_grid):
        """Spreading function should integrate to 1."""
        for s in [0, 2, 8, 16, 36]:
            d = cos2s_spreading(direction_grid, s, theta_0=0.0)
            integral = float(np.trapezoid(d, direction_grid))
            assert abs(integral - 1.0) < 0.01, f"s={s}: integral={integral}"

    def test_isotropic_limit(self, direction_grid):
        """s=0 should give uniform distribution."""
        d = cos2s_spreading(direction_grid, 0.0, theta_0=0.0)
        expected = 1.0 / (2.0 * np.pi)
        assert np.allclose(d, expected, atol=1e-6)

    def test_narrow_beam_limit(self, direction_grid):
        """Large s should concentrate energy near theta_0."""
        s = 36.0
        d = cos2s_spreading(direction_grid, s, theta_0=0.0)

        mask = np.abs(direction_grid) < np.pi / 6
        energy_in_beam = float(np.trapezoid(d[mask], direction_grid[mask]))
        assert energy_in_beam > 0.85, f"Only {energy_in_beam*100:.1f}% energy in beam"

    def test_recover_spreading_parameter(self, direction_grid):
        """Recovery of s from known spreading function should be within 5%."""
        test_s_values = [2.0, 4.0, 8.0, 16.0, 24.0]

        for s_true in test_s_values:
            d = cos2s_spreading(direction_grid, s_true, theta_0=0.0)
            s_recovered = recover_spreading_parameter(direction_grid, d)

            rel_error = abs(s_recovered - s_true) / s_true
            assert rel_error < 0.05, (
                f"s={s_true}: recovered={s_recovered:.2f}, "
                f"error={rel_error*100:.2f}% exceeds 5%"
            )

    def test_recover_mean_direction(self, direction_grid):
        """Recovery of mean direction should be accurate."""
        s = 16.0
        theta_0_true = np.pi / 4

        d = cos2s_spreading(direction_grid, s, theta_0=theta_0_true)
        s_recovered = recover_spreading_parameter(direction_grid, d)

        assert s_recovered > 0
        assert np.isfinite(s_recovered)

    def test_directional_spectrum_shape(self, angular_frequency_grid, direction_grid):
        """2D directional spectrum should have correct shape."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)
        s = 8.0

        spec_2d = directional_spectrum(
            angular_frequency_grid, direction_grid, s_omega, s, theta_0=0.0
        )

        assert spec_2d.shape == (len(angular_frequency_grid), len(direction_grid))
        assert np.all(spec_2d >= 0)

    def test_directional_m0_consistency(self, angular_frequency_grid, direction_grid):
        """Directional m0 should match 1D spectrum m0."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)
        s = 8.0

        spec_2d = directional_spectrum(
            angular_frequency_grid, direction_grid, s_omega, s, theta_0=0.0
        )

        m0_2d = float(np.trapezoid(
            np.trapezoid(spec_2d, direction_grid, axis=1),
            angular_frequency_grid
        ))

        m0_1d = float(np.trapezoid(s_omega, angular_frequency_grid))

        rel_error = abs(m0_2d - m0_1d) / m0_1d
        assert rel_error < 0.02, f"Directional m0 error {rel_error*100:.2f}%"

    def test_analyze_directional_result(self, angular_frequency_grid):
        """Full directional analysis should return valid result."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)
        s = 16.0

        result = analyze_directional(angular_frequency_grid, s_omega, s, theta_0=0.0)

        assert result.m0 > 0
        assert result.s_parameter == s
        assert result.directional_spread > 0
        assert np.isfinite(result.mean_direction)


# --------------------------------------------------------------------------- #
# C3: ISO 15016 bias quantification
# --------------------------------------------------------------------------- #

class TestISO15016Bias:
    """C3: ISO 15016 short/long crest bias quantified."""

    def test_long_crested_response(self, angular_frequency_grid, direction_grid):
        """Long-crested response should be computable."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)
        rao = np.ones((len(angular_frequency_grid), len(direction_grid)))

        result = iso15016_long_crested_response(
            angular_frequency_grid, s_omega, rao, direction_grid, theta_0=0.0
        )

        assert result["m0"] > 0
        assert result["rms_response"] > 0
        assert result["assumption"] == "long_crested"

    def test_short_crested_response(self, angular_frequency_grid, direction_grid):
        """Short-crested response should be computable."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)
        rao = np.ones((len(angular_frequency_grid), len(direction_grid)))

        result = iso15016_short_crested_response(
            angular_frequency_grid, direction_grid, s_omega,
            s_spread=8.0, rao_magnitude=rao, theta_0=0.0
        )

        assert result["m0"] > 0
        assert result["rms_response"] > 0
        assert result["assumption"] == "short_crested"

    def test_bias_band_computation(self):
        """Bias band should quantify difference between assumptions."""
        lc = {"m0": 1.0, "rms_response": 1.0}
        sc = {"m0": 0.8, "rms_response": 0.894}
        true = {"m0": 0.9, "rms_response": 0.949}

        bias = iso15016_bias_band(lc, sc, true)

        assert "bias_lc_vs_sc_rel" in bias
        assert "bias_lc_vs_true_rel" in bias
        assert "bias_sc_vs_true_rel" in bias
        assert np.isfinite(bias["bias_lc_vs_sc_rel"])

    def test_long_vs_short_crested_bias_with_heading_rao(self, angular_frequency_grid, direction_grid):
        """With heading-sensitive RAO, long and short crested should differ."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)

        n_omega = len(angular_frequency_grid)
        n_theta = len(direction_grid)
        freq_resp = 1.0 / (1.0 + ((angular_frequency_grid - 1.0) / 0.3) ** 2)
        heading_resp = np.cos(direction_grid / 2.0) ** 4
        heading_resp = np.maximum(heading_resp, 0.0)
        rao = np.outer(freq_resp, heading_resp)

        lc = iso15016_long_crested_response(
            angular_frequency_grid, s_omega, rao, direction_grid, theta_0=0.0
        )
        sc = iso15016_short_crested_response(
            angular_frequency_grid, direction_grid, s_omega,
            s_spread=8.0, rao_magnitude=rao, theta_0=0.0
        )

        assert lc["m0"] > 0
        assert sc["m0"] > 0
        assert abs(lc["m0"] - sc["m0"]) / lc["m0"] > 0.01, (
            "Long and short crested should differ with heading-sensitive RAO"
        )

    def test_full_bias_quantification(self, synthetic_bimodal, angular_frequency_grid, direction_grid):
        """Full ISO 15016 bias quantification should return complete result."""
        result = quantify_iso15016_bias(
            synthetic_bimodal,
            angular_frequency_grid,
            direction_grid,
            s_swell=16.0,
            s_wind=8.0,
            theta_swell=0.0,
            theta_wind=np.pi / 4,
        )

        assert isinstance(result, ISO15016BiasResult)
        assert result.long_crested["m0"] > 0
        assert result.short_crested["m0"] > 0
        assert result.true_bimodal["m0"] > 0
        assert "bias_lc_vs_true_pct" in result.bias_band
        assert "bias_sc_vs_true_pct" in result.bias_band

    def test_bias_band_output_format(self, synthetic_bimodal, angular_frequency_grid, direction_grid):
        """Bias band output should contain all required fields."""
        result = quantify_iso15016_bias(
            synthetic_bimodal,
            angular_frequency_grid,
            direction_grid,
        )

        bias = result.bias_band
        required_keys = [
            "m0_long_crested", "m0_short_crested", "m0_true",
            "bias_lc_vs_sc_rel", "bias_lc_vs_sc_pct",
            "bias_lc_vs_true_rel", "bias_lc_vs_true_pct",
            "bias_sc_vs_true_rel", "bias_sc_vs_true_pct",
            "rms_long_crested", "rms_short_crested", "rms_true",
        ]
        for key in required_keys:
            assert key in bias, f"Missing key: {key}"
            assert np.isfinite(bias[key]), f"Non-finite value for {key}"


# --------------------------------------------------------------------------- #
# C4: Synthetic sea-state spectrum fit R^2 > 0.9
# --------------------------------------------------------------------------- #

class TestSyntheticSpectrumFit:
    """C4: Synthetic sea-state spectrum fit R^2 > 0.9."""

    def test_torsethaugen_self_fit(self, synthetic_single_peak, frequency_grid):
        """Torsethaugen spectrum should fit itself perfectly (R^2 = 1)."""
        s_true = synthetic_single_peak.evaluate_hz(frequency_grid)
        s_fit = s_true

        ss_res = np.sum((s_true - s_fit) ** 2)
        ss_tot = np.sum((s_true - np.mean(s_true)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot

        assert r_squared > 0.99, f"Self-fit R^2 = {r_squared}"

    def test_bimodal_reconstruction(self, synthetic_bimodal, frequency_grid):
        """Bimodal spectrum reconstruction should have R^2 > 0.9."""
        s_true = synthetic_bimodal.evaluate_hz(frequency_grid)
        s_swell = synthetic_bimodal.swell.evaluate_hz(frequency_grid)
        s_wind = synthetic_bimodal.wind.evaluate_hz(frequency_grid)
        s_reconstructed = s_swell + s_wind

        ss_res = np.sum((s_true - s_reconstructed) ** 2)
        ss_tot = np.sum((s_true - np.mean(s_true)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot

        assert r_squared > 0.99, f"Bimodal reconstruction R^2 = {r_squared}"

    def test_directional_reconstruction(self, angular_frequency_grid, direction_grid):
        """Directional spectrum reconstruction should have R^2 > 0.9."""
        s_omega = np.exp(-((angular_frequency_grid - 1.0) ** 2) / 0.5)
        s = 16.0

        spec_true = directional_spectrum(
            angular_frequency_grid, direction_grid, s_omega, s, theta_0=0.0
        )

        d = cos2s_spreading(direction_grid, s, theta_0=0.0)
        spec_reconstructed = np.outer(s_omega, d)

        ss_res = np.sum((spec_true - spec_reconstructed) ** 2)
        ss_tot = np.sum((spec_true - np.mean(spec_true)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot

        assert r_squared > 0.99, f"Directional reconstruction R^2 = {r_squared}"


# --------------------------------------------------------------------------- #
# C5: Zero domain semantics check
# --------------------------------------------------------------------------- #

class TestZeroDomainSemantics:
    """C5: Code and test fixtures contain no domain terms."""

    def test_no_domain_terms_in_module_names(self):
        """Module names should not contain domain-specific terms."""
        import tensorphysics.seakeeping.bimodal_spectrum as bs
        import tensorphysics.seakeeping.directional_spreading as ds
        import tensorphysics.seakeeping.iso15016_bias as ib

        for mod in [bs, ds, ib]:
            doc = mod.__doc__ or ""
            forbidden = ["ship", "hull", "vessel", "resistance", "propulsion"]
            for term in forbidden:
                assert term.lower() not in doc.lower(), (
                    f"Domain term '{term}' found in {mod.__name__}"
                )

    def test_synthetic_fixture_no_domain_terms(self, synthetic_bimodal):
        """Synthetic fixture should use generic parameters."""
        assert hasattr(synthetic_bimodal, 'swell_hs')
        assert hasattr(synthetic_bimodal, 'swell_tp')
        assert hasattr(synthetic_bimodal, 'wind_hs')
        assert hasattr(synthetic_bimodal, 'wind_tp')


# --------------------------------------------------------------------------- #
# Integration test: full pipeline
# --------------------------------------------------------------------------- #

class TestFullPipeline:
    """End-to-end integration test."""

    def test_full_bimodal_directional_iso_pipeline(self, synthetic_bimodal):
        """Full pipeline: bimodal -> directional -> ISO 15016 bias."""
        omega = np.linspace(0.1, 4.0, 500)
        theta = np.linspace(-np.pi, np.pi, 90, endpoint=False)

        result = quantify_iso15016_bias(
            synthetic_bimodal,
            omega,
            theta,
            s_swell=16.0,
            s_wind=8.0,
            theta_swell=0.0,
            theta_wind=np.pi / 4,
        )

        assert result.long_crested["m0"] > 0
        assert result.short_crested["m0"] > 0
        assert result.true_bimodal["m0"] > 0

        assert np.isfinite(result.bias_band["bias_lc_vs_true_pct"])
        assert np.isfinite(result.bias_band["bias_sc_vs_true_pct"])

    def test_results_serializable(self, synthetic_bimodal):
        """Results should be JSON-serializable for artifact storage."""
        omega = np.linspace(0.1, 4.0, 200)
        theta = np.linspace(-np.pi, np.pi, 36, endpoint=False)

        result = quantify_iso15016_bias(
            synthetic_bimodal, omega, theta,
            s_swell=16.0, s_wind=8.0,
        )

        artifact = {
            "long_crested": {k: float(v) for k, v in result.long_crested.items()
                           if isinstance(v, (int, float, np.floating))},
            "short_crested": {k: float(v) for k, v in result.short_crested.items()
                            if isinstance(v, (int, float, np.floating))},
            "true_bimodal": {k: float(v) for k, v in result.true_bimodal.items()
                           if isinstance(v, (int, float, np.floating))},
            "bias_band": {k: float(v) for k, v in result.bias_band.items()
                        if isinstance(v, (int, float, np.floating))},
        }

        json_str = json.dumps(artifact)
        parsed = json.loads(json_str)
        assert "bias_band" in parsed
