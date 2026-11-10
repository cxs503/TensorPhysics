"""ISO 15016 directional assumption bias quantification.

Quantifies the bias introduced by ISO 15016's short-crested and
long-crested wave assumptions when applied to bimodal sea states.

ISO 15016 assumptions:
- Long-crested: waves are infinitely long in the transverse direction
  (no directional spreading, all energy in one direction)
- Short-crested: waves have finite crest length with a prescribed
  directional spreading model

For bimodal sea states (swell + wind sea), the directional distribution
is more complex than the ISO 15016 assumptions capture, leading to
biases in predicted structural responses.

This module:
1. Computes response under long-crested assumption (all energy at theta_0)
2. Computes response under short-crested assumption (cos^2(s/2) spreading)
3. Computes response under true bimodal directional distribution
4. Quantifies the bias bands between these assumptions

References
----------
- ISO 15016:2015 — Guidelines for the assessment of speed and power
  performance by analysis of speed trial data.
- ITTC Recommended Procedures (2017): Seakeeping tests.
- Faltinsen, O.M. (1990). "Sea Loads on Offshore Structures."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .bimodal_spectrum import BimodalSpectrum, spectral_moments
from .directional_spreading import cos2s_spreading, directional_spectrum

__all__ = [
    "iso15016_short_crested_response",
    "iso15016_long_crested_response",
    "iso15016_bias_band",
    "quantify_iso15016_bias",
    "ISO15016BiasResult",
]


def _find_theta_index(theta: np.ndarray, theta_0: float) -> int:
    """Find the index in theta array closest to theta_0."""
    # Wrap theta_0 to [-pi, pi]
    theta_0_wrapped = (theta_0 + np.pi) % (2.0 * np.pi) - np.pi
    # Find closest index
    return int(np.argmin(np.abs(theta - theta_0_wrapped)))


def _response_rao(
    omega: np.ndarray,
    theta: np.ndarray,
    rao_magnitude: np.ndarray,
) -> np.ndarray:
    """Apply RAO to directional spectrum to get response spectrum."""
    return rao_magnitude ** 2


def _generic_rao(
    omega: np.ndarray,
    theta: np.ndarray,
    theta_0: float = 0.0,
    peak_omega: float = 1.0,
    peak_response: float = 1.0,
    bandwidth: float = 0.3,
    heading_sensitivity: float = 2.0,
) -> np.ndarray:
    """Generic response RAO model."""
    omega = np.asarray(omega, dtype=float)
    theta = np.asarray(theta, dtype=float)

    freq_resp = peak_response / (1.0 + ((omega - peak_omega) / (bandwidth * peak_omega)) ** 2)

    dtheta = theta - theta_0
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    heading_resp = np.cos(dtheta / 2.0) ** heading_sensitivity
    heading_resp = np.maximum(heading_resp, 0.0)

    return np.outer(freq_resp, heading_resp)


def iso15016_long_crested_response(
    omega: np.ndarray,
    s_omega: np.ndarray,
    rao_magnitude: np.ndarray,
    theta: np.ndarray,
    theta_0: float = 0.0,
) -> dict:
    """Compute response under ISO 15016 long-crested assumption.

    Long-crested: all wave energy is assumed to come from a single
    direction (theta_0). The response spectrum is:
        S_resp(omega) = |RAO(omega, theta_0)|^2 * S(omega)
    """
    omega = np.asarray(omega, dtype=float)
    s_omega = np.asarray(s_omega, dtype=float)
    theta = np.asarray(theta, dtype=float)

    theta_idx = _find_theta_index(theta, theta_0)
    rao_at_theta0 = rao_magnitude[:, theta_idx]

    s_resp = rao_at_theta0 ** 2 * s_omega

    moments = spectral_moments(omega / (2.0 * np.pi), s_resp / (2.0 * np.pi), orders=(0, 1, 2, 4))

    return {
        "assumption": "long_crested",
        "response_spectrum_1d": s_resp,
        "m0": moments[0],
        "m1": moments[1],
        "m2": moments[2],
        "m4": moments[4],
        "rms_response": float(np.sqrt(max(moments[0], 0.0))),
        "mean_period": moments[0] / moments[1] if moments[1] > 0 else 0.0,
    }


def iso15016_short_crested_response(
    omega: np.ndarray,
    theta: np.ndarray,
    s_omega: np.ndarray,
    s_spread: float,
    rao_magnitude: np.ndarray,
    theta_0: float = 0.0,
) -> dict:
    """Compute response under ISO 15016 short-crested assumption."""
    omega = np.asarray(omega, dtype=float)
    theta = np.asarray(theta, dtype=float)
    s_omega = np.asarray(s_omega, dtype=float)

    d_theta = cos2s_spreading(theta, s_spread, theta_0)
    s_2d = np.outer(s_omega, d_theta)

    rao_sq = rao_magnitude ** 2
    s_resp_2d = rao_sq * s_2d

    s_resp_1d = np.trapezoid(s_resp_2d, theta, axis=1)

    freq_hz = omega / (2.0 * np.pi)
    s_resp_hz = s_resp_1d / (2.0 * np.pi)

    moments = spectral_moments(freq_hz, s_resp_hz, orders=(0, 1, 2, 4))

    return {
        "assumption": "short_crested",
        "s_parameter": s_spread,
        "response_spectrum_1d": s_resp_1d,
        "m0": moments[0],
        "m1": moments[1],
        "m2": moments[2],
        "m4": moments[4],
        "rms_response": float(np.sqrt(max(moments[0], 0.0))),
        "mean_period": moments[0] / moments[1] if moments[1] > 0 else 0.0,
    }


def iso15016_bias_band(
    long_crested: dict,
    short_crested: dict,
    true_response: Optional[dict] = None,
) -> dict:
    """Compute ISO 15016 bias band between assumptions."""
    m0_lc = long_crested["m0"]
    m0_sc = short_crested["m0"]

    if m0_sc > 0:
        bias_lc_vs_sc = (m0_lc - m0_sc) / m0_sc
    else:
        bias_lc_vs_sc = 0.0

    result = {
        "m0_long_crested": m0_lc,
        "m0_short_crested": m0_sc,
        "bias_lc_vs_sc_rel": bias_lc_vs_sc,
        "bias_lc_vs_sc_pct": bias_lc_vs_sc * 100.0,
        "rms_long_crested": long_crested["rms_response"],
        "rms_short_crested": short_crested["rms_response"],
    }

    if true_response is not None:
        m0_true = true_response["m0"]
        if m0_true > 0:
            bias_lc_vs_true = (m0_lc - m0_true) / m0_true
            bias_sc_vs_true = (m0_sc - m0_true) / m0_true
        else:
            bias_lc_vs_true = 0.0
            bias_sc_vs_true = 0.0

        result["m0_true"] = m0_true
        result["bias_lc_vs_true_rel"] = bias_lc_vs_true
        result["bias_lc_vs_true_pct"] = bias_lc_vs_true * 100.0
        result["bias_sc_vs_true_rel"] = bias_sc_vs_true
        result["bias_sc_vs_true_pct"] = bias_sc_vs_true * 100.0
        result["rms_true"] = true_response["rms_response"]

    return result


@dataclass
class ISO15016BiasResult:
    """Result of ISO 15016 bias analysis."""
    long_crested: dict
    short_crested: dict
    true_bimodal: dict
    bias_band: dict


def quantify_iso15016_bias(
    bimodal: BimodalSpectrum,
    omega: np.ndarray,
    theta: np.ndarray,
    s_swell: float = 16.0,
    s_wind: float = 8.0,
    theta_swell: float = 0.0,
    theta_wind: float = np.pi / 4,
    rao_magnitude: Optional[np.ndarray] = None,
    peak_omega: float = 1.0,
    peak_response: float = 1.0,
) -> ISO15016BiasResult:
    """Full ISO 15016 bias quantification for bimodal sea state."""
    omega = np.asarray(omega, dtype=float)
    theta = np.asarray(theta, dtype=float)

    s_omega = bimodal.evaluate(omega)

    if rao_magnitude is None:
        rao_magnitude = _generic_rao(
            omega, theta,
            theta_0=0.0,
            peak_omega=peak_omega,
            peak_response=peak_response,
        )

    # Long-crested response
    lc = iso15016_long_crested_response(omega, s_omega, rao_magnitude, theta, theta_0=0.0)

    # Short-crested response
    s_effective = (s_swell + s_wind) / 2.0
    sc = iso15016_short_crested_response(
        omega, theta, s_omega, s_effective, rao_magnitude, theta_0=0.0
    )

    # True bimodal directional response
    d_swell = cos2s_spreading(theta, s_swell, theta_swell)
    s_swell_1d = bimodal.swell.evaluate(omega)
    s_swell_2d = np.outer(s_swell_1d, d_swell)

    d_wind = cos2s_spreading(theta, s_wind, theta_wind)
    s_wind_1d = bimodal.wind.evaluate(omega)
    s_wind_2d = np.outer(s_wind_1d, d_wind)

    s_2d_true = s_swell_2d + s_wind_2d

    rao_sq = rao_magnitude ** 2
    s_resp_2d = rao_sq * s_2d_true

    s_resp_1d = np.trapezoid(s_resp_2d, theta, axis=1)

    freq_hz = omega / (2.0 * np.pi)
    s_resp_hz = s_resp_1d / (2.0 * np.pi)
    moments = spectral_moments(freq_hz, s_resp_hz, orders=(0, 1, 2, 4))

    true_resp = {
        "assumption": "true_bimodal_directional",
        "response_spectrum_1d": s_resp_1d,
        "m0": moments[0],
        "m1": moments[1],
        "m2": moments[2],
        "m4": moments[4],
        "rms_response": float(np.sqrt(max(moments[0], 0.0))),
        "mean_period": moments[0] / moments[1] if moments[1] > 0 else 0.0,
    }

    bias = iso15016_bias_band(lc, sc, true_resp)

    return ISO15016BiasResult(
        long_crested=lc,
        short_crested=sc,
        true_bimodal=true_resp,
        bias_band=bias,
    )
