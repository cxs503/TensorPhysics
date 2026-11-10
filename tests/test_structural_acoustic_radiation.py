"""Verification tests for structural acoustic radiation chain.

Physical anchors (all analytic, no tuned parameters):

(a) Pulsating sphere radiated power vs exact solution at f=100/500/1000 Hz,
    tolerance ≤ 3%.
(b) Piston radiation efficiency σ(ka) vs exact Bessel solution for ka=0.1–10,
    tolerance ≤ 5%.
(c) Far-field pressure p(r) ∝ 1/r decay, tolerance < 2%.
(d) Negative control: f→0 → W_rad < 0.01·ρ·c·S·v_0².
(e) Energy conservation: W = 4πr²·p_rms²/(ρc) for pulsating sphere.
(f) Provenance: sha256 hash of parameters.

References:
- Morse & Ingard, "Theoretical Acoustics" (1968).
- Junger & Feit, "Sound, Structures, and Their Interaction" (1986).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.domains.structural_acoustic_radiation.radiation import (
    C_WATER,
    P_REF,
    RHO_WATER,
    CircularPiston,
    PulsatingSphere,
    RadiationChain,
    dipole_radiation_efficiency,
    far_field_pressure,
    piston_radiation_efficiency,
    pulsating_sphere_radiation_efficiency,
    radiated_power,
    semi_empirical_radiation_efficiency,
    sound_pressure_level,
)
from tensorphysics.domains.structural_acoustic_radiation.vibration import (
    HarmonicVibrator,
    chain_provenance,
    vibration_radiation_chain,
)


# ===========================================================================
# (a) Pulsating sphere radiated power vs analytical solution
# ===========================================================================
class TestPulsatingSpherePower:
    """Criterion (a): W_rad vs analytical solution at f=100/500/1000 Hz, ≤3%."""

    # Physical parameters
    RADIUS = 0.5       # sphere radius [m]
    V_AMP = 0.01       # velocity amplitude [m/s]
    RHO = RHO_WATER
    C = C_WATER

    @pytest.mark.parametrize("freq", [100.0, 500.0, 1000.0])
    def test_radiated_power_vs_analytical(self, freq):
        """W_rad must match the exact analytical formula within 3%."""
        sphere = PulsatingSphere(
            radius=self.RADIUS,
            frequency=freq,
            velocity_amplitude=self.V_AMP,
            rho=self.RHO,
            c=self.C,
        )

        # Analytical: W = 4πa² ρc <v²> σ
        ka = sphere.ka
        sigma_exact = ka**2 / (1.0 + ka**2)
        S = 4.0 * np.pi * self.RADIUS**2
        v_rms_sq = self.V_AMP**2 / 2.0
        W_analytical = sigma_exact * self.RHO * self.C * S * v_rms_sq

        W_computed = sphere.radiated_power()

        rel_err = abs(W_computed - W_analytical) / W_analytical
        assert rel_err < 0.03, (
            f"f={freq}Hz: W_computed={W_computed:.6e}, "
            f"W_analytical={W_analytical:.6e}, rel_err={rel_err:.4%}"
        )

    @pytest.mark.parametrize("freq", [100.0, 500.0, 1000.0])
    def test_radiation_efficiency(self, freq):
        """σ(ka) must match (ka)²/(1+(ka)²) exactly."""
        sphere = PulsatingSphere(
            radius=self.RADIUS, frequency=freq, velocity_amplitude=self.V_AMP
        )
        ka = sphere.ka
        sigma_expected = ka**2 / (1.0 + ka**2)
        sigma_computed = sphere.radiation_efficiency()
        assert abs(sigma_computed - sigma_expected) < 1e-12

    def test_low_frequency_limit(self):
        """For ka << 1: σ ≈ (ka)²."""
        sphere = PulsatingSphere(
            radius=0.01, frequency=1.0, velocity_amplitude=0.001
        )
        ka = sphere.ka
        assert ka < 0.1, "Test setup error: ka should be small"
        sigma = sphere.radiation_efficiency()
        sigma_approx = ka**2
        rel_err = abs(sigma - sigma_approx) / sigma_approx
        assert rel_err < 0.01, f"Low-freq limit: σ={sigma:.6e}, (ka)²={sigma_approx:.6e}"

    def test_high_frequency_limit(self):
        """For ka >> 1: σ → 1."""
        sphere = PulsatingSphere(
            radius=1.0, frequency=10000.0, velocity_amplitude=0.01
        )
        ka = sphere.ka
        assert ka > 10, "Test setup error: ka should be large"
        sigma = sphere.radiation_efficiency()
        assert abs(sigma - 1.0) < 0.01, f"High-freq limit: σ={sigma:.6f}, expected ~1"


# ===========================================================================
# (b) Piston radiation efficiency σ(ka) vs analytical Bessel solution
# ===========================================================================
class TestPistonRadiationEfficiency:
    """Criterion (b): σ(ka) vs Bessel solution for ka=0.1–10, ≤5%."""

    @pytest.mark.parametrize("ka", [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    def test_piston_efficiency_vs_bessel(self, ka):
        """Piston σ(ka) must match 1 - 2J₁(2ka)/(2ka) within 5%."""
        from scipy.special import j1

        sigma_computed = piston_radiation_efficiency(ka)
        sigma_exact = 1.0 - 2.0 * j1(2.0 * ka) / (2.0 * ka)

        # For small ka, use absolute tolerance since values are small
        if sigma_exact < 0.01:
            assert abs(sigma_computed - sigma_exact) < 0.005
        else:
            rel_err = abs(sigma_computed - sigma_exact) / sigma_exact
            assert rel_err < 0.05, (
                f"ka={ka}: σ_computed={sigma_computed:.6e}, "
                f"σ_exact={sigma_exact:.6e}, rel_err={rel_err:.4%}"
            )

    def test_piston_efficiency_array(self):
        """Vectorized computation must match scalar."""
        ka_array = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        sigma_array = piston_radiation_efficiency(ka_array)
        sigma_scalar = np.array([piston_radiation_efficiency(k) for k in ka_array])
        np.testing.assert_allclose(sigma_array, sigma_scalar, rtol=1e-12)

    def test_piston_low_frequency_limit(self):
        """For ka << 1: σ ≈ (ka)²/2."""
        ka = 0.01
        sigma = piston_radiation_efficiency(ka)
        sigma_approx = ka**2 / 2.0
        rel_err = abs(sigma - sigma_approx) / sigma_approx
        assert rel_err < 0.01, f"Low-freq: σ={sigma:.6e}, (ka)²/2={sigma_approx:.6e}"

    def test_piston_high_frequency_limit(self):
        """For ka >> 1: σ → 1."""
        ka = 50.0
        sigma = piston_radiation_efficiency(ka)
        assert abs(sigma - 1.0) < 0.05, f"High-freq: σ={sigma:.6f}, expected ~1"

    def test_piston_bounded(self):
        """σ(ka) should be non-negative and approach 1 for large ka."""
        ka_values = np.linspace(0.01, 50.0, 500)
        sigma_values = piston_radiation_efficiency(ka_values)
        assert np.all(sigma_values >= 0.0)
        # Note: The exact formula σ = 1 - 2J₁(2ka)/(2ka) can slightly exceed 1
        # due to Bessel function oscillations. This is a known feature of the
        # exact solution. For large ka, σ → 1 with small oscillations.
        assert np.all(sigma_values <= 1.2)  # physical bound with tolerance
        # Check that σ → 1 for large ka
        assert abs(sigma_values[-1] - 1.0) < 0.05


# ===========================================================================
# (c) Far-field pressure p(r) ∝ 1/r decay
# ===========================================================================
class TestFarFieldDecay:
    """Criterion (c): p(r) ∝ 1/r, tolerance < 2%."""

    def test_pulsating_sphere_1_over_r(self):
        """Far-field pressure must decay as 1/r for pulsating sphere."""
        sphere = PulsatingSphere(
            radius=0.1, frequency=500.0, velocity_amplitude=0.01
        )
        r1 = 10.0   # far field
        r2 = 100.0  # 10× farther

        p1 = sphere.far_field_pressure_amplitude(r1)
        p2 = sphere.far_field_pressure_amplitude(r2)

        # p ∝ 1/r → p1/p2 = r2/r1
        ratio_expected = r2 / r1
        ratio_computed = p1 / p2
        rel_err = abs(ratio_computed - ratio_expected) / ratio_expected
        assert rel_err < 0.02, (
            f"1/r decay: p1/p2={ratio_computed:.6f}, "
            f"r2/r1={ratio_expected:.1f}, rel_err={rel_err:.4%}"
        )

    def test_general_chain_1_over_r(self):
        """General chain: p_rms(r) ∝ 1/r."""
        chain = RadiationChain(
            frequency=500.0,
            velocity_amplitude=0.01,
            surface_area=1.0,
            sigma_rad=0.5,
        )
        r1 = 5.0
        r2 = 50.0

        result1 = chain.compute_far_field_pressure(r1)
        result2 = chain.compute_far_field_pressure(r2)

        p1 = result1["p_rms"]
        p2 = result2["p_rms"]

        ratio_expected = r2 / r1
        ratio_computed = p1 / p2
        rel_err = abs(ratio_computed - ratio_expected) / ratio_expected
        assert rel_err < 0.02, (
            f"1/r decay: p1/p2={ratio_computed:.6f}, "
            f"r2/r1={ratio_expected:.1f}, rel_err={rel_err:.4%}"
        )

    def test_spl_decrease_6dB_per_doubling(self):
        """SPL should decrease by ~6 dB when distance doubles (point source)."""
        chain = RadiationChain(
            frequency=1000.0,
            velocity_amplitude=0.01,
            surface_area=0.5,
            sigma_rad=0.8,
        )
        r1 = 10.0
        r2 = 20.0

        spl1 = chain.compute_far_field_pressure(r1)["SPL_dB"]
        spl2 = chain.compute_far_field_pressure(r2)["SPL_dB"]

        delta_spl = spl1 - spl2
        expected = 20.0 * np.log10(2.0)  # ≈ 6.02 dB
        assert abs(delta_spl - expected) < 0.01, (
            f"SPL decrease: {delta_spl:.4f} dB, expected {expected:.4f} dB"
        )


# ===========================================================================
# (d) Negative control: f→0 → W_rad → 0
# ===========================================================================
class TestNegativeControl:
    """Criterion (d): f→0 → W_rad < 0.01·ρ·c·S·v_0²."""

    def test_pulsating_sphere_low_freq(self):
        """At very low frequency, radiated power should be negligible."""
        # Very low frequency → ka << 1 → σ ≈ (ka)² → very small
        sphere = PulsatingSphere(
            radius=0.1, frequency=0.1, velocity_amplitude=0.01
        )
        W = sphere.radiated_power()
        S = sphere.surface_area
        v0_sq = sphere.velocity_amplitude**2
        threshold = 0.01 * sphere.rho * sphere.c * S * v0_sq
        assert W < threshold, (
            f"Low-freq control: W={W:.6e}, threshold={threshold:.6e}"
        )

    def test_piston_low_freq(self):
        """At very low frequency, piston radiated power should be negligible."""
        piston = CircularPiston(
            radius=0.1, frequency=0.1, velocity_amplitude=0.01
        )
        W = piston.radiated_power()
        S = piston.surface_area
        v0_sq = piston.velocity_amplitude**2
        threshold = 0.01 * piston.rho * piston.c * S * v0_sq
        assert W < threshold, (
            f"Low-freq control: W={W:.6e}, threshold={threshold:.6e}"
        )

    def test_radiation_efficiency_approaches_zero(self):
        """σ → 0 as ka → 0 for both sphere and piston."""
        ka_small = 0.001
        sigma_sphere = pulsating_sphere_radiation_efficiency(ka_small)
        sigma_piston = piston_radiation_efficiency(ka_small)
        assert sigma_sphere < 1e-5
        assert sigma_piston < 1e-5


# ===========================================================================
# (e) Energy conservation
# ===========================================================================
class TestEnergyConservation:
    """Energy conservation: W = 4πr²·p_rms²/(ρc) for pulsating sphere."""

    @pytest.mark.parametrize("freq", [50.0, 200.0, 1000.0])
    @pytest.mark.parametrize("r", [5.0, 50.0, 500.0])
    def test_power_intensity_conservation(self, freq, r):
        """W_rad must equal 4πr²·I for the exact pulsating sphere solution."""
        sphere = PulsatingSphere(
            radius=0.1, frequency=freq, velocity_amplitude=0.01
        )
        assert sphere.verify_power_consistency(r, tol=1e-8), (
            f"Energy conservation failed: f={freq}Hz, r={r}m"
        )

    def test_general_chain_power_intensity(self):
        """General chain: W = 4πr²·p_rms²/(ρc)."""
        chain = RadiationChain(
            frequency=500.0,
            velocity_amplitude=0.01,
            surface_area=1.0,
            sigma_rad=0.5,
        )
        r = 100.0
        result = chain.compute_far_field_pressure(r)
        W = result["W_rad"]
        p_rms = result["p_rms"]
        intensity = p_rms**2 / (chain.rho * chain.c)
        W_from_I = 4.0 * np.pi * r**2 * intensity
        rel_err = abs(W - W_from_I) / W
        assert rel_err < 1e-10, f"Power-intensity mismatch: {rel_err:.2e}"


# ===========================================================================
# (f) Provenance
# ===========================================================================
class TestProvenance:
    """Provenance: sha256 hash of parameters for reproducibility."""

    def test_provenance_hash_deterministic(self):
        """Same parameters → same hash."""
        chain1 = RadiationChain(
            frequency=500.0, velocity_amplitude=0.01,
            surface_area=1.0, sigma_rad=0.5,
        )
        chain2 = RadiationChain(
            frequency=500.0, velocity_amplitude=0.01,
            surface_area=1.0, sigma_rad=0.5,
        )
        d1 = chain1.to_provenance_dict()
        d2 = chain2.to_provenance_dict()
        assert d1["provenance"]["sha256"] == d2["provenance"]["sha256"]

    def test_provenance_hash_changes_with_params(self):
        """Different parameters → different hash."""
        chain1 = RadiationChain(
            frequency=500.0, velocity_amplitude=0.01,
            surface_area=1.0, sigma_rad=0.5,
        )
        chain2 = RadiationChain(
            frequency=1000.0, velocity_amplitude=0.01,
            surface_area=1.0, sigma_rad=0.5,
        )
        d1 = chain1.to_provenance_dict()
        d2 = chain2.to_provenance_dict()
        assert d1["provenance"]["sha256"] != d2["provenance"]["sha256"]

    def test_provenance_json_serializable(self):
        """Provenance dict must be JSON-serializable."""
        chain = RadiationChain(
            frequency=500.0, velocity_amplitude=0.01,
            surface_area=1.0, sigma_rad=0.5,
        )
        d = chain.to_provenance_dict()
        json_str = json.dumps(d)
        d_back = json.loads(json_str)
        assert d_back["provenance"]["sha256"] == d["provenance"]["sha256"]

    def test_provenance_hash_manual_verification(self):
        """Verify hash matches manual computation."""
        chain = RadiationChain(
            frequency=500.0, velocity_amplitude=0.01,
            surface_area=1.0, sigma_rad=0.5,
        )
        d = chain.to_provenance_dict()
        params = d["provenance"]["parameters"]
        param_str = json.dumps(params, sort_keys=True)
        expected_hash = hashlib.sha256(param_str.encode()).hexdigest()
        assert d["provenance"]["sha256"] == expected_hash


# ===========================================================================
# Additional physics tests
# ===========================================================================
class TestSoundPressureLevel:
    """SPL computation tests."""

    def test_spl_reference(self):
        """SPL at p_ref should be 0 dB."""
        spl = sound_pressure_level(P_REF)
        assert abs(spl) < 1e-10

    def test_spl_positive(self):
        """SPL for p > p_ref should be positive."""
        spl = sound_pressure_level(1.0)  # 1 Pa >> 1 μPa
        assert spl > 0

    def test_spl_value(self):
        """1 Pa → SPL = 20 log10(1/1e-6) = 120 dB."""
        spl = sound_pressure_level(1.0)
        expected = 20.0 * np.log10(1.0 / 1e-6)
        assert abs(spl - expected) < 1e-10

    def test_spl_negative_raises(self):
        """Negative pressure should raise ValueError."""
        with pytest.raises(ValueError):
            sound_pressure_level(-1.0)

    def test_spl_zero_raises(self):
        """Zero pressure should raise ValueError."""
        with pytest.raises(ValueError):
            sound_pressure_level(0.0)


class TestRadiatedPowerFunction:
    """Tests for the standalone radiated_power function."""

    def test_zero_efficiency(self):
        """σ=0 → W=0."""
        W = radiated_power(0.0, 1025.0, 1500.0, 1.0, 0.01)
        assert W == 0.0

    def test_unit_efficiency(self):
        """σ=1 → W = ρcS<v²>."""
        rho, c, S, v_rms = 1025.0, 1500.0, 1.0, 0.01
        W = radiated_power(1.0, rho, c, S, v_rms)
        expected = rho * c * S * v_rms**2
        assert abs(W - expected) < 1e-12

    def test_proportionality(self):
        """W ∝ σ, W ∝ S, W ∝ v_rms²."""
        base = radiated_power(0.5, 1025.0, 1500.0, 1.0, 0.01)
        double_sigma = radiated_power(1.0, 1025.0, 1500.0, 1.0, 0.01)
        assert abs(double_sigma / base - 2.0) < 1e-12


class TestFarFieldFunction:
    """Tests for the standalone far_field_pressure function."""

    def test_inverse_r(self):
        """p_rms ∝ 1/r."""
        W = 1.0
        rho, c = 1025.0, 1500.0
        p1 = far_field_pressure(W, rho, c, 10.0)
        p2 = far_field_pressure(W, rho, c, 100.0)
        assert abs(p1 / p2 - 10.0) < 1e-12

    def test_positive(self):
        """Pressure must be positive for positive power."""
        p = far_field_pressure(1.0, 1025.0, 1500.0, 10.0)
        assert p > 0


class TestCircularPiston:
    """Tests for CircularPiston class."""

    def test_piston_radiated_power(self):
        """Piston W = σ ρc S <v²>."""
        piston = CircularPiston(
            radius=0.2, frequency=500.0, velocity_amplitude=0.01
        )
        W = piston.radiated_power()
        sigma = piston.radiation_efficiency()
        S = piston.surface_area
        v_rms_sq = piston.mean_square_velocity
        expected = sigma * piston.rho * piston.c * S * v_rms_sq
        assert abs(W - expected) < 1e-12

    def test_piston_properties(self):
        """Check computed properties."""
        piston = CircularPiston(
            radius=0.2, frequency=500.0, velocity_amplitude=0.01
        )
        assert abs(piston.omega - 2 * np.pi * 500.0) < 1e-10
        assert abs(piston.k - 2 * np.pi * 500.0 / 1500.0) < 1e-10
        assert abs(piston.ka - 2 * np.pi * 500.0 * 0.2 / 1500.0) < 1e-10
        assert abs(piston.surface_area - np.pi * 0.2**2) < 1e-12


class TestRadiationChain:
    """Tests for RadiationChain orchestrator."""

    def test_chain_compute(self):
        """Full chain computation."""
        chain = RadiationChain(
            frequency=500.0,
            velocity_amplitude=0.01,
            surface_area=1.0,
            sigma_rad=0.5,
        )
        result = chain.compute_chain([1.0, 10.0, 100.0])
        assert result["W_rad"] > 0
        assert len(result["p_rms"]) == 3
        assert len(result["SPL_dB"]) == 3
        # SPL should decrease with distance
        assert result["SPL_dB"][0] > result["SPL_dB"][1] > result["SPL_dB"][2]

    def test_chain_v_rms(self):
        """v_rms = v_0 / √2."""
        chain = RadiationChain(
            frequency=500.0, velocity_amplitude=0.02,
            surface_area=1.0, sigma_rad=0.5,
        )
        assert abs(chain.v_rms - 0.02 / np.sqrt(2.0)) < 1e-12

# ===========================================================================
# (g) Baffled-piston far field: network/power consistency and the exact
#     Rayleigh-integral on-axis pressure anchor
# ===========================================================================
class TestPistonFarField:
    """Power/peak bookkeeping of CircularPiston must agree with Rayleigh integral."""

    @pytest.mark.parametrize("ka", [0.05, 0.1, 0.3])
    def test_power_based_amplitude_matches_exact_on_axis_low_ka(self, ka):
        """For ka << 1 the piston is omni-directional: √2·p_rms == p_onaxis.

        At ka = 0.3 the directivity narrowing already removes ~0.75% of the
        on-axis level, hence the 1% tolerance.
        """
        c = 1500.0
        radius = 0.1
        freq = ka * c / (2.0 * np.pi * radius)
        piston = CircularPiston(radius=radius, frequency=freq, velocity_amplitude=0.01)
        r = 20.0
        ratio = piston.far_field_pressure_amplitude(r) / piston.on_axis_pressure_amplitude(r)
        assert abs(ratio - 1.0) < 1e-2, f"ka={ka}: ratio={ratio}"

    def test_rms_amplitude_relation(self):
        """p_amplitude = √2 p_rms and rms obeys W = 2π r² p_rms²/(ρc)."""
        piston = CircularPiston(radius=0.1, frequency=500.0, velocity_amplitude=0.01)
        r = 10.0
        p_rms = piston.far_field_pressure_rms(r)
        p_amp = piston.far_field_pressure_amplitude(r)
        assert abs(p_amp / p_rms - np.sqrt(2.0)) < 1e-12
        W = piston.radiated_power()
        W_back = 2.0 * np.pi * r**2 * p_rms**2 / (piston.rho * piston.c)
        assert abs(W_back - W) / W < 1e-12

    @pytest.mark.parametrize("ka", [1.0, 2.0, 5.0])
    def test_on_axis_far_field_vs_rayleigh_exact(self, ka):
        """Far-field on-axis pressure vs exact Rayleigh solution (≤5%)."""
        c = 1500.0
        radius = 0.1
        freq = ka * c / (2.0 * np.pi * radius)
        piston = CircularPiston(radius=radius, frequency=freq, velocity_amplitude=0.01)
        # far-field condition r >> a²/λ, use r = 40 a²/λ
        wavelength = c / freq
        r = 40.0 * radius**2 / wavelength
        exact = piston.on_axis_pressure_exact(r)
        approx = piston.on_axis_pressure_amplitude(r)
        rel = abs(approx - exact) / exact
        assert rel < 0.05, f"ka={ka}: rel={rel}"

    def test_directivity_axis_and_broadside(self):
        """D(0)=1 and |D| ≤ 1 for all angles."""
        piston = CircularPiston(radius=0.2, frequency=2000.0, velocity_amplitude=0.01)
        assert abs(piston.directivity(0.0) - 1.0) < 1e-12
        thetas = np.linspace(0.0, np.pi / 2, 200)
        assert np.all(np.abs([piston.directivity(t) for t in thetas]) <= 1.0 + 1e-12)


# ===========================================================================
# (h) Semi-empirical sigma_rad(ka) model ladder
# ===========================================================================
class TestSemiEmpiricalEfficiency:
    """Monopole (ka)², dipole (ka)⁴ and piston models + dispatcher."""

    def test_dispatch_consistency(self):
        ka = np.array([0.1, 1.0, 10.0])
        np.testing.assert_allclose(
            semi_empirical_radiation_efficiency(ka, "monopole"),
            pulsating_sphere_radiation_efficiency(ka),
        )
        np.testing.assert_allclose(
            semi_empirical_radiation_efficiency(ka, "dipole"),
            dipole_radiation_efficiency(ka),
        )
        np.testing.assert_allclose(
            semi_empirical_radiation_efficiency(ka, "piston"),
            piston_radiation_efficiency(ka),
        )

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            semi_empirical_radiation_efficiency(0.5, "quadrupole")

    def test_dipole_low_frequency_quartic(self):
        """Dipole efficiency scales as (ka)^4: doubling ka → ×16."""
        sigma1 = dipole_radiation_efficiency(0.05)
        sigma2 = dipole_radiation_efficiency(0.10)
        assert abs(sigma2 / sigma1 - 16.0) < 0.5

    def test_dipole_high_frequency_limit(self):
        assert abs(dipole_radiation_efficiency(50.0) - 1.0) < 1e-3

    def test_dipole_negative_control_low_frequency(self):
        """f→0: dipole efficiency is far below the 1% power threshold."""
        ka = 0.1
        assert dipole_radiation_efficiency(ka) < 1e-3


# ===========================================================================
# (i) Structural vibration stage: SDOF force → velocity
# ===========================================================================
class TestHarmonicVibrator:
    """SDOF harmonic response against closed-form formulas."""

    def make(self):
        return HarmonicVibrator(
            mass=10.0, stiffness=10.0 * (2 * np.pi * 50.0) ** 2,
            damping_ratio=0.02, force_amplitude=100.0,
        )

    def test_natural_frequency(self):
        v = self.make()
        assert abs(v.natural_frequency - 50.0) < 1e-9
        assert abs(v.omega_n - 2 * np.pi * 50.0) < 1e-9

    def test_resonance_velocity_closed_form(self):
        """v0(ω_n) = F0/c_d exactly."""
        v = self.make()
        assert abs(v.velocity_amplitude(v.omega_n) - v.resonance_velocity_amplitude()) < 1e-12
        expected = 100.0 / (2 * 0.02 * v.omega_n * 10.0)
        assert abs(v.resonance_velocity_amplitude() - expected) < 1e-12

    def test_quasi_static_limit(self):
        """ω → 0: v0 ≈ F0 ω/k, |x| ≈ F0/k."""
        v = self.make()
        omega = 1e-3 * v.omega_n
        assert abs(v.velocity_amplitude(omega) / (v.force_amplitude * omega / v.stiffness) - 1) < 1e-4
        assert abs(v.displacement_amplitude(omega) / v.static_displacement - 1) < 1e-4

    def test_impedance_consistency(self):
        v = self.make()
        omega = v.omega_n
        Z = v.mechanical_impedance(omega)
        assert abs(Z - v.damping_coefficient) < 1e-9
        assert abs(v.velocity_amplitude(omega) - v.force_amplitude / abs(Z)) < 1e-12

    def test_acceleration_and_disp_relations(self):
        v = self.make()
        omega = 2 * np.pi * 70.0
        v0 = v.velocity_amplitude(omega)
        assert abs(v.displacement_amplitude(omega) - v0 / omega) < 1e-12
        assert abs(v.acceleration_amplitude(omega) - v0 * omega) < 1e-12
        assert abs(v.mean_square_velocity(omega) - v0**2 / 2) < 1e-12

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            HarmonicVibrator(mass=0.0, stiffness=1.0, damping_ratio=0.1, force_amplitude=1.0)
        with pytest.raises(ValueError):
            HarmonicVibrator(mass=1.0, stiffness=1.0, damping_ratio=0.0, force_amplitude=1.0)


# ===========================================================================
# (j) End-to-end chain: force → <v²> → W_rad → p(r) → SPL
# ===========================================================================
class TestVibrationRadiationChain:
    """Full prediction chain verified against hand-computable closed forms."""

    def chain(self, **kwargs):
        vib = HarmonicVibrator(
            mass=5.0, stiffness=5.0 * (2 * np.pi * 100.0) ** 2,
            damping_ratio=0.05, force_amplitude=50.0,
        )
        defaults = dict(surface_area=0.5, characteristic_length=0.15,
                        model="monopole", distances=[1.0, 10.0, 100.0])
        defaults.update(kwargs)
        return vib, vibration_radiation_chain(vib, 100.0, **defaults)

    def test_chain_against_manual_formula(self):
        vib, res = self.chain()
        omega = 2 * np.pi * 100.0
        v0 = vib.velocity_amplitude(omega)   # driven at resonance f_n = 100 Hz
        assert abs(v0 - vib.resonance_velocity_amplitude()) < 1e-12
        sigma = res["sigma_rad"]
        W_expected = sigma * res["rho"] * res["c"] * 0.5 * v0**2 / 2
        assert abs(res["W_rad"] - W_expected) / W_expected < 1e-12
        p_expected = np.sqrt(res["rho"] * res["c"] * W_expected / (4 * np.pi * np.asarray([1.0, 10.0, 100.0])**2))
        np.testing.assert_allclose(res["p_rms"], p_expected, rtol=1e-12)

    def test_chain_spl_matches_pressure(self):
        _, res = self.chain()
        expected_spl = 20 * np.log10(res["p_rms"] / P_REF)
        np.testing.assert_allclose(res["SPL_dB"], expected_spl, atol=1e-10)

    def test_chain_inverse_square_power_and_1_over_r(self):
        _, res = self.chain(distances=[1.0, 2.0, 20.0, 200.0])
        p = res["p_rms"]
        assert abs(p[0] / p[1] - 2.0) < 1e-12
        assert abs(p[2] / p[3] - 10.0) < 1e-12

    def test_chain_spl_decay_6db_per_doubling(self):
        _, res = self.chain(distances=[1.0, 2.0, 4.0])
        drops = np.diff(res["SPL_dB"])
        np.testing.assert_allclose(drops, -20 * np.log10(2.0), rtol=1e-12)

    def test_piston_model_uses_half_space(self):
        _, mono = self.chain(model="monopole")
        _, pist = self.chain(model="piston")
        assert mono["solid_angle_sr"] == pytest.approx(4 * np.pi)
        assert pist["solid_angle_sr"] == pytest.approx(2 * np.pi)
        ratio = (pist["p_rms"] / mono["p_rms"]) ** 2
        sigma_ratio = pist["sigma_rad"] / mono["sigma_rad"]
        # p_rms² ∝ W/Ω, with Ω halved for the baffled piston
        assert np.allclose(ratio, 2.0 * sigma_ratio, rtol=1e-12)

    def test_chain_provenance_deterministic_and_sensitive(self):
        _, res1 = self.chain()
        _, res2 = self.chain()
        _, res3 = self.chain(model="dipole")
        assert chain_provenance(res1)["sha256"] == chain_provenance(res2)["sha256"]
        assert chain_provenance(res1)["sha256"] != chain_provenance(res3)["sha256"]
        assert json.loads(json.dumps({k: v for k, v in chain_provenance(res1).items()
                                      if k != "payload_keys"}))["sha256"]

    def test_chain_invalid_inputs(self):
        vib = HarmonicVibrator(mass=1.0, stiffness=1.0, damping_ratio=0.1, force_amplitude=1.0)
        with pytest.raises(ValueError):
            vibration_radiation_chain(vib, 100.0, surface_area=1.0, characteristic_length=0.1,
                                      distances=[0.0])
        with pytest.raises(ValueError):
            vibration_radiation_chain(vib, 0.0, surface_area=1.0, characteristic_length=0.1)
