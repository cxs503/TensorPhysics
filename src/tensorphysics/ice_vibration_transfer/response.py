"""Hull-girder acceleration response to the ice torque excitation couple.

The hull response is obtained by modal superposition on the *wet* modal basis of
:mod:`.hull_modal` (mass-normalised wet modes, generalised mass 1 kg):

.. math::

    H_u(f) = -\\omega^2 \\sum_n \\frac{\\phi_n(x_f)\\,[\\phi_n(x_e)-\\phi_n(x_r)]}
                                   {\\omega_n^2-\\omega^2+2 i \\zeta_n \\omega_n \\omega}

so that ``a(f) = H_u(f) F(f)`` has units m/s^2 per N for a unit force couple of
arm ``x_e - x_r``.  The couple is self-equilibrated (equal and opposite forces at
the two shaft-line bearing stations), which keeps the free-free rigid-body modes
out of the response and is the physical reason why a torque fluctuation excites
bending of the hull girder at all.

Everything up to here is linear and random-vibration based: the excitation is
described by its one-sided PSD ``S_F(f)``, the response PSD follows as
``S_a(f) = |H_u(f)|^2 S_F(f)``, and the RMS values in the ice band are obtained by
integrating that PSD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .hull_modal import HullModalBasis

__all__ = [
    "ResponseSpectrum",
    "acceleration_psd",
    "acceleration_transfer",
    "band_rms",
    "point_force_acceleration",
]


def acceleration_transfer(
    basis: HullModalBasis,
    freqs: np.ndarray,
    x_foundation: float | None = None,
    x_excitation: float | None = None,
    x_reaction: float | None = None,
) -> np.ndarray:
    """Complex acceleration transfer ``a(f)/F`` [m/s^2 per N] at the foundation.

    A unit force couple (``+1`` N at ``x_excitation``, ``-1`` N at ``x_reaction``)
    is applied along the hull girder; the returned value is the vertical
    acceleration at ``x_foundation`` for unit couple amplitude.
    """
    f = np.asarray(freqs, dtype=float)
    if np.any(f < 0.0):
        raise ValueError("frequencies must be non-negative")
    xf = basis.x_foundation_m if x_foundation is None else float(x_foundation)
    xe = basis.x_excitation_m if x_excitation is None else float(x_excitation)
    xr = basis.x_reaction_m if x_reaction is None else float(x_reaction)

    phi_f = basis.shape_at(xf)
    arm = basis.shape_at(xe) - basis.shape_at(xr)
    wn = 2.0 * np.pi * basis.wet_frequencies_hz
    zeta = basis.zeta
    omega = 2.0 * np.pi * f
    h = np.zeros(f.size, dtype=complex)
    for n in range(basis.n_modes):
        denom = wn[n] ** 2 - omega**2 + 2j * zeta[n] * wn[n] * omega
        h += phi_f[n] * arm[n] / denom
    return -(omega**2) * h


def band_rms(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """RMS of a one-sided PSD inside ``[f_lo, f_hi]``."""
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(psd, dtype=float)
    m = (f >= float(f_lo)) & (f <= float(f_hi))
    if np.count_nonzero(m) < 2:
        return 0.0
    return float(np.sqrt(max(np.trapezoid(s[m], f[m]), 0.0)))


@dataclass
class ResponseSpectrum:
    """Hull acceleration transfer and response PSD of one load case."""

    freqs_hz: np.ndarray
    transfer_m_s2_per_n: np.ndarray
    s_force_n2_per_hz: np.ndarray
    s_acceleration_m2_s4_per_hz: np.ndarray
    rms_m_s2: float
    peak_frequency_hz: float
    band_rms: dict[str, float]
    stations: dict[str, float]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rms_m_s2": float(self.rms_m_s2),
            "peak_frequency_hz": float(self.peak_frequency_hz),
            "band_rms_m_s2": {k: float(v) for k, v in self.band_rms.items()},
            "stations_m": dict(self.stations),
            "acceleration_psd_peak_m2_s4_per_hz": float(np.max(self.s_acceleration_m2_s4_per_hz)),
            "transfer_peak_m_s2_per_n": float(np.max(np.abs(self.transfer_m_s2_per_n))),
        }


def acceleration_psd(
    basis: HullModalBasis,
    freqs: np.ndarray,
    s_force_n2_per_hz: np.ndarray,
    bands: Sequence[tuple[str, float, float]] = (
        ("0.2-2Hz", 0.2, 2.0),
        ("2-5Hz", 2.0, 5.0),
        ("5-10Hz", 5.0, 10.0),
        ("10-20Hz", 10.0, 20.0),
        ("20-50Hz", 20.0, 50.0),
        ("50-100Hz", 50.0, 100.0),
    ),
) -> ResponseSpectrum:
    """Hull acceleration PSD at the foundation station for a force PSD."""
    f = np.asarray(freqs, dtype=float)
    s_f = np.asarray(s_force_n2_per_hz, dtype=float)
    if f.shape != s_f.shape:
        raise ValueError("freqs and force PSD must have the same shape")
    h = acceleration_transfer(basis, f)
    s_a = np.abs(h) ** 2 * s_f
    rms = float(np.sqrt(max(np.trapezoid(s_a, f), 0.0)))
    peak = float(f[int(np.argmax(s_a))]) if s_a.size else 0.0
    return ResponseSpectrum(
        freqs_hz=f,
        transfer_m_s2_per_n=h,
        s_force_n2_per_hz=s_f,
        s_acceleration_m2_s4_per_hz=s_a,
        rms_m_s2=rms,
        peak_frequency_hz=peak,
        band_rms={name: band_rms(f, s_a, lo, hi) for name, lo, hi in bands},
        stations={
            "foundation_m": basis.x_foundation_m,
            "excitation_m": basis.x_excitation_m,
            "reaction_m": basis.x_reaction_m,
        },
    )

def point_force_acceleration(
    freqs: np.ndarray,
    modal_mass_kg: float,
    f_n_hz: float,
    zeta: float,
) -> np.ndarray:
    """Complex acceleration of a local foundation mass under a point force.

    ``a(f)/F = -omega^2 / (m (omega_n^2 - omega^2 + 2 i zeta omega_n omega))``

    This is the *local* load path of the same ice torque fluctuation: the
    shaft-line reaction (thrust/torque bearing, engine mounts) shakes the
    engine-room foundation mass directly, without having to excite a global
    hull-girder mode.  Two limits make the model interpretable: at
    ``f >> f_n`` the response tends to the rigid-mass limit ``F/m`` and below
    ``f_n`` it is stiffness controlled (``-omega^2 F / (m omega_n^2)``).
    """
    if modal_mass_kg <= 0.0:
        raise ValueError("foundation modal mass must be positive")
    if f_n_hz <= 0.0 or not 0.0 <= zeta < 1.0:
        raise ValueError("invalid local foundation mode")
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    wn = 2.0 * np.pi * float(f_n_hz)
    denom = wn**2 - omega**2 + 2j * float(zeta) * wn * omega
    return -(omega**2) / (float(modal_mass_kg) * denom)
