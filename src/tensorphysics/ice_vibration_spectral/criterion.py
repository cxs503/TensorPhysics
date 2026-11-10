"""Resonance-risk criterion for ice-induced vibration.

The criterion compares the dominant ice-excitation frequency ``f_ice`` with the
structural natural frequency ``f_n`` of a mode.  Two *equivalent* formulations
are implemented so that the classification can be verified against itself:

1. **Frequency-band formulation** - the half-power (3 dB) band of the SDOF
   amplification curve,

       f_{+-} = f_n sqrt(1 - 2 zeta^2 +- 2 zeta sqrt(1 - zeta^2)),

   solved from ``(1-u)^2 + 4 zeta^2 u = 8 zeta^2 (1-zeta^2)`` with
   ``u = (f/f_n)^2``.  For small damping the half-bandwidth tends to
   ``zeta f_n``, i.e. the familiar ``|f_ice - f_n| < zeta f_n`` rule.

2. **Amplification formulation** - the dynamic amplification at the excitation
   frequency relative to the resonance peak,

       A_r = |H(f_ice)| / max|H|,      max|H| = 1/(2 zeta k sqrt(1-zeta^2)).

   ``A_r >= 1/sqrt(2)`` is exactly the half-power condition and therefore
   equivalent to (1); it is evaluated through a numerically independent path
   (the complex FRF instead of the quadratic solution) and is the consistency
   anchor of the acceptance criterion.

Tiers
-----
``lock_in``
    ``f_ice`` inside the half-power band.  The ice-crushing process can lock
    onto the mode: the response is amplified by more than 3 dB and, for
    velocity-weakening ice, the interaction becomes self-excited.
``near_resonance``
    ``f_ice`` inside three half-power bandwidths: intermittent/partial
    amplification, no sustained lock-in under a stable drift velocity.
``clear``
    outside; the response follows the quasi-static ice load.

References
----------
* Karna, T. (1992) A procedure for dynamic ice-structure interaction analysis,
  VTT Publications.
* Maeaettaenen, M. (1978) On conditions for the rise of self-excited
  ice-induced motional forces, IAHR Ice Symposium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .structural import SDOFSystem

__all__ = [
    "NEAR_RESONANCE_BAND_FACTOR",
    "HALF_POWER_AMPLITUDE_RATIO",
    "ResonanceVerdict",
    "half_power_edges_hz",
    "amplification_ratio",
    "resonance_risk",
    "classify_frequency_grid",
    "classification_consistency",
]

#: Number of half-power bandwidths that still counts as "near resonance".
NEAR_RESONANCE_BAND_FACTOR = 3.0
#: Amplitude ratio that defines the half-power points.
HALF_POWER_AMPLITUDE_RATIO = 1.0 / np.sqrt(2.0)


def half_power_edges_hz(f_n_hz: float, damping_ratio: float) -> tuple[float, float]:
    """Exact half-power band edges [Hz] of the SDOF amplification curve."""
    if f_n_hz <= 0.0:
        raise ValueError("natural frequency must be positive")
    if not 0.0 < damping_ratio < 1.0:
        raise ValueError("damping ratio must lie in (0, 1)")
    z = damping_ratio
    base = 1.0 - 2.0 * z * z
    rad = 2.0 * z * np.sqrt(1.0 - z * z)
    lo = float(np.sqrt(max(base - rad, 0.0)) * f_n_hz)
    hi = float(np.sqrt(base + rad) * f_n_hz)
    return lo, hi


def amplification_ratio(f_ice_hz: float, f_n_hz: float, damping_ratio: float) -> float:
    """``|H(f_ice)| / max|H|`` for a unit-mass-free SDOF (independent of k).

    The stiffness cancels in the ratio, so this is purely a function of the
    frequency ratio and the damping ratio.
    """
    x = float(f_ice_hz) / float(f_n_hz)
    z = float(damping_ratio)
    denom = np.hypot(1.0 - x * x, 2.0 * z * x)
    peak = 2.0 * z * np.sqrt(1.0 - z * z)  # |H(x_peak)| * k, normalised
    return float(peak / denom)


@dataclass(frozen=True)
class ResonanceVerdict:
    """Result of the resonance-risk criterion for one (mode, ice) pair."""

    f_ice_hz: float
    f_n_hz: float
    damping_ratio: float
    relative_detuning: float
    half_power_edges_hz: tuple[float, float]
    amplification_ratio: float
    tier: str
    is_resonance_risk: bool
    margin_to_band_edge_hz: float

    @property
    def outside_half_power_band(self) -> bool:
        lo, hi = self.half_power_edges_hz
        return not (lo <= self.f_ice_hz <= hi)

    def payload(self) -> dict[str, Any]:
        lo, hi = self.half_power_edges_hz
        return {
            "f_ice_hz": self.f_ice_hz,
            "f_n_hz": self.f_n_hz,
            "damping_ratio": self.damping_ratio,
            "relative_detuning": self.relative_detuning,
            "half_power_edges_hz": [lo, hi],
            "half_power_bandwidth_hz": hi - lo,
            "amplification_ratio": self.amplification_ratio,
            "tier": self.tier,
            "is_resonance_risk": self.is_resonance_risk,
            "margin_to_band_edge_hz": self.margin_to_band_edge_hz,
        }


def resonance_risk(f_ice_hz: float, f_n_hz: float, damping_ratio: float) -> ResonanceVerdict:
    """Classify an ice-excitation / natural-frequency pair.

    ``lock_in`` for ``f_ice`` inside the half-power band, ``near_resonance``
    within ``NEAR_RESONANCE_BAND_FACTOR`` half-power bandwidths, otherwise
    ``clear``.
    """
    lo, hi = half_power_edges_hz(f_n_hz, damping_ratio)
    half_bw = 0.5 * (hi - lo)
    detuning = abs(float(f_ice_hz) - float(f_n_hz)) / float(f_n_hz)
    amp = amplification_ratio(f_ice_hz, f_n_hz, damping_ratio)

    if lo <= f_ice_hz <= hi:
        tier = "lock_in"
    elif abs(f_ice_hz - f_n_hz) <= NEAR_RESONANCE_BAND_FACTOR * half_bw:
        tier = "near_resonance"
    else:
        tier = "clear"

    if f_ice_hz < lo:
        margin = lo - f_ice_hz
    elif f_ice_hz > hi:
        margin = f_ice_hz - hi
    else:
        margin = -min(f_ice_hz - lo, hi - f_ice_hz)

    return ResonanceVerdict(
        f_ice_hz=float(f_ice_hz),
        f_n_hz=float(f_n_hz),
        damping_ratio=float(damping_ratio),
        relative_detuning=detuning,
        half_power_edges_hz=(lo, hi),
        amplification_ratio=amp,
        tier=tier,
        is_resonance_risk=(tier == "lock_in"),
        margin_to_band_edge_hz=float(margin),
    )


def classify_frequency_grid(
    detuning_ratios: np.ndarray, damping_ratio: float, f_n_hz: float = 1.0
) -> list[ResonanceVerdict]:
    """Vectorised helper: verdicts for a grid of ``f_ice/f_n`` ratios."""
    return [
        resonance_risk(float(r) * f_n_hz, f_n_hz, damping_ratio)
        for r in np.asarray(detuning_ratios, dtype=float)
    ]

def classification_consistency(
    detuning_ratios: tuple[float, ...] = (0.70, 0.85, 0.94, 0.96, 1.00, 1.04, 1.06, 1.15, 1.30),
    damping_ratios: tuple[float, ...] = (0.01, 0.02, 0.05),
    f_n_hz: float = 1.10,
) -> dict[str, Any]:
    """Cross-check the band criterion against the amplification predicate.

    For every ``(f_ice/f_n, zeta)`` point the classification produced by the
    band/quadratic formulation (:func:`resonance_risk`) is compared with the
    predicate ``|H(f_ice)| >= |H|max/sqrt(2)`` evaluated from the complex FRF
    (:func:`amplification_ratio`), i.e. two numerically independent
    implementations of the same half-power condition.  The classical
    small-damping approximation ``|f_ice - f_n| < zeta f_n`` is reported as a
    third column so the size of its approximation error is visible.
    """
    points: list[dict[str, Any]] = []
    n_agree = 0
    n_approx_agree = 0
    for zeta in damping_ratios:
        for ratio in detuning_ratios:
            f_ice = float(ratio) * float(f_n_hz)
            verdict = resonance_risk(f_ice, f_n_hz, zeta)
            amp = amplification_ratio(f_ice, f_n_hz, zeta)
            band_pred = bool(verdict.is_resonance_risk)
            amp_pred = bool(amp >= HALF_POWER_AMPLITUDE_RATIO)
            approx_pred = bool(abs(f_ice - f_n_hz) / f_n_hz < zeta)
            agree = band_pred == amp_pred
            n_agree += int(agree)
            n_approx_agree += int(approx_pred == band_pred)
            points.append(
                {
                    "frequency_ratio": float(ratio),
                    "damping_ratio": zeta,
                    "f_ice_hz": f_ice,
                    "relative_detuning": verdict.relative_detuning,
                    "tier": verdict.tier,
                    "band_criterion_risk": band_pred,
                    "amplification_ratio": amp,
                    "amplification_criterion_risk": amp_pred,
                    "small_damping_approximation_risk": approx_pred,
                    "agree": bool(agree),
                }
            )
    return {
        "n_points": len(points),
        "n_agree": n_agree,
        "agreement": n_agree / len(points),
        "n_agree_small_damping_approximation": n_approx_agree,
        "agreement_small_damping_approximation": n_approx_agree / len(points),
        "points": points,
    }
