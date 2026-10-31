"""Tests for the MIZ attenuation model (W5-N05)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.ice_wave import miz


def _alpha(n_c: float, h_i: float, loss: float = 0.02) -> float:
    props = miz.effective_medium_properties(n_c, h_i, loss_factor=loss)
    k = miz.kelvin_voigt_wavenumber(10.0, 1000.0, props["D_real"], props["D_imag"])
    return miz.attenuation_from_complex_wavenumber(k)


def test_exponential_profile_fit_recovers_alpha():
    x = np.linspace(0.0, 3.0e4, 301)
    for alpha in (1e-6, 2.5e-5, 4e-4):
        profile = miz.exponential_decay_profile(x, 1.7, alpha)
        fit = miz.fit_attenuation_coefficient(x, profile)
        assert fit["alpha"] == pytest.approx(alpha, rel=1e-10)
        assert fit["amplitude0"] == pytest.approx(1.7, rel=1e-10)
        assert fit["r2"] > 0.999999


@pytest.mark.parametrize("n_c,h_i", [(0.3, 1.0), (0.5, 1.0), (0.7, 2.0), (0.9, 2.0)])
def test_spatial_and_temporal_attenuation_agree(n_c, h_i):
    """Acceptance criterion (d): two independent numerical routes to alpha."""
    props = miz.effective_medium_properties(n_c, h_i, loss_factor=0.02)
    k = miz.kelvin_voigt_wavenumber(10.0, 1000.0, props["D_real"], props["D_imag"])
    alpha_spatial = miz.attenuation_from_complex_wavenumber(k)
    temporal = miz.temporal_attenuation(10.0, 1000.0, props["D_real"], props["D_imag"])
    rel = abs(temporal["alpha"] - alpha_spatial) / alpha_spatial
    assert rel < 0.15
    assert rel < 1e-3  # in practice the weak-damping limit is far tighter


def test_alpha_grows_with_concentration_and_thickness():
    base = _alpha(0.2, 1.0)
    mid = _alpha(0.5, 1.0)
    high = _alpha(0.9, 1.0)
    assert base < mid < high
    assert _alpha(0.5, 0.5) < _alpha(0.5, 1.0) < _alpha(0.5, 2.0)


def test_alpha_lies_in_literature_order_of_magnitude():
    band = miz.LITERATURE_ALPHA_BAND
    alpha = _alpha(0.5, 1.0)
    assert band["alpha_min"] <= alpha <= band["alpha_max"]
    # The literature leg is explicitly flagged as WITHHELD provenance.
    assert band["provenance"] == "WITHHELD"


def test_lossless_medium_has_no_attenuation():
    props = miz.effective_medium_properties(0.6, 1.0, loss_factor=0.0)
    k = miz.kelvin_voigt_wavenumber(10.0, 1000.0, props["D_real"], props["D_imag"])
    assert miz.attenuation_from_complex_wavenumber(k) < 1e-12


def test_sweep_exponents_are_positive():
    sweep = miz.sweep_attenuation([0.3, 0.6, 0.9], [0.5, 1.0, 2.0])
    assert sweep["alpha_grid"].shape == (3, 3)
    assert sweep["alpha_vs_concentration_exponent"] > 0.0
    assert sweep["alpha_vs_thickness_exponent"] > 0.0
    assert np.all(sweep["alpha_grid"] > 0.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: miz.effective_medium_properties(0.0, 1.0),
        lambda: miz.effective_medium_properties(1.5, 1.0),
        lambda: miz.effective_medium_properties(0.5, 0.0),
        lambda: miz.effective_medium_properties(0.5, 1.0, loss_factor=-0.1),
        lambda: miz.kelvin_voigt_wavenumber(0.0, 1000.0, 1e8, 1e6),
        lambda: miz.exponential_decay_profile([0.0, 1.0], 1.0, -1.0),
        lambda: miz.fit_attenuation_coefficient([0.0, 1.0], [1.0, 0.5]),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(miz.MIZError):
        call()