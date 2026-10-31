"""Frequency lock-in criteria and response amplification.

Physics
-------
For harmonic ice crushing at ``f_ice`` acting on the lowest hull-girder mode
(``f_n``, damping ratio ``zeta``) the steady-state amplitude is

    X = (F0 / k) * D(r, zeta),      r = f_ice / f_n
    D = 1 / sqrt((1 - r^2)^2 + (2 zeta r)^2)

``D`` is the dynamic magnification factor: ``D -> 1`` quasi-statically, and at
exact resonance ``D = 1/(2 zeta)``.

*Lock-in* is declared when the excitation sits inside a narrow band around the
natural frequency, ``|f_ice - f_n| / f_n < 5 %``.  Inside that band the
response is governed by the damping: ``D = 3`` at 5 % detuning requires
``zeta < 0.151`` — practically always satisfied by a hull girder
(``zeta`` ~ 1-3 %), so a 5 % lock-in band implies an amplification well above 3.

Large ice forces also push the response out of the linear regime, where the
restoring characteristic stiffens (contact/geometric: ``k x + k3 x^3``, the
Duffing oscillator).  Harmonic balance on ``m x'' + c x' + k x + k3 x^3 = F0 cos(W t)``
with ``x ~ a cos(W t)`` gives

    (k + (3/4) k3 a^2 - m W^2)^2 a^2 + (c W)^2 a^2 = F0^2,

a cubic in ``A = a^2``.  Where three real roots exist the response curve folds
back on itself and the jump phenomenon appears — the amplitude-dependent
hysteresis observed in measured ice-induced vibration records.  This module
therefore reports both the linear magnification factor and the Duffing
harmonic-balance branches, and :mod:`tensorphysics.ice_vibration.analysis`
confirms the time-domain response against them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "magnification_factor",
    "amplification_band_half_width",
    "detuning_ratio",
    "lock_in_band_hz",
    "is_locked_in",
    "damping_limit_for_amplification",
    "duffing_harmonic_balance_amplitudes",
    "duffing_backbone_amplitude",
    "steady_state_amplitude",
    "harmonic_projection_amplitude",
    "LockInVerdict",
    "classify_lock_in",
]

#: Default relative half-width of the lock-in band (5 %).
LOCK_IN_RELATIVE_TOLERANCE = 0.05

#: Amplification factor that qualifies as "lock-in response amplification".
LOCK_IN_AMPLIFICATION_THRESHOLD = 3.0


def detuning_ratio(excitation_hz: float, natural_hz: float) -> float:
    """``(f_ice - f_n) / f_n`` (signed relative detuning)."""
    if natural_hz <= 0.0:
        raise ValueError("natural_hz must be positive")
    if excitation_hz <= 0.0:
        raise ValueError("excitation_hz must be positive")
    return (excitation_hz - natural_hz) / natural_hz


def magnification_factor(frequency_ratio: float, damping_ratio: float) -> float:
    """Dynamic magnification factor ``D = 1/sqrt((1-r^2)^2 + (2 zeta r)^2)``."""
    r = float(frequency_ratio)
    zeta = float(damping_ratio)
    if r <= 0.0:
        raise ValueError("frequency_ratio must be positive")
    if not 0.0 <= zeta < 1.0:
        raise ValueError("damping_ratio must satisfy 0 <= zeta < 1")
    return 1.0 / math.sqrt((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2)


def lock_in_band_hz(natural_hz: float, relative_tolerance: float = LOCK_IN_RELATIVE_TOLERANCE) -> Tuple[float, float]:
    """Frequency band ``[f_n (1-tol), f_n (1+tol)]`` declared as lock-in."""
    if natural_hz <= 0.0:
        raise ValueError("natural_hz must be positive")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in (0, 1)")
    return natural_hz * (1.0 - relative_tolerance), natural_hz * (1.0 + relative_tolerance)


def is_locked_in(
    excitation_hz: float,
    natural_hz: float,
    relative_tolerance: float = LOCK_IN_RELATIVE_TOLERANCE,
) -> bool:
    """True when ``|f_ice - f_n| / f_n < relative_tolerance``."""
    return abs(detuning_ratio(excitation_hz, natural_hz)) < relative_tolerance


def damping_limit_for_amplification(
    frequency_ratio: float,
    target_amplification: float = LOCK_IN_AMPLIFICATION_THRESHOLD,
) -> float:
    """Damping ratio above which ``D(r, zeta)`` falls below ``target``.

    Inverts the magnification factor: at ``r = 1 + tol`` with ``tol = 0.05`` the
    result is ``zeta_max ~ 0.151``.  Reported as the *damping condition* for the
    lock-in amplification criterion.
    """
    r = float(frequency_ratio)
    target = float(target_amplification)
    if r <= 0.0 or target <= 0.0:
        raise ValueError("frequency_ratio and target_amplification must be positive")
    inside = 1.0 / target**2 - (1.0 - r**2) ** 2
    if inside <= 0.0:
        return 0.0
    return math.sqrt(inside) / (2.0 * r)


def amplification_band_half_width(
    damping_ratio: float,
    target_amplification: float = LOCK_IN_AMPLIFICATION_THRESHOLD,
) -> float:
    """Relative frequency offset at which ``D(r, zeta)`` falls to ``target``.

    Solves ``(1 - r^2)^2 + (2 zeta r)^2 = 1/target^2`` for ``r > 1``.  At
    ``zeta = 0.02`` and ``target = 3`` this gives ``r - 1 = 0.153``: the whole
    5 % lock-in band sits well inside the amplification-3 bandwidth, which is
    why any hull-girder detuning below 5 % is amplified by more than 3 as long
    as ``zeta < 15 %``.
    """
    zeta = float(damping_ratio)
    target = float(target_amplification)
    if not 0.0 <= zeta < 1.0:
        raise ValueError("damping_ratio must satisfy 0 <= zeta < 1")
    if target <= 0.0:
        raise ValueError("target_amplification must be positive")
    b = 4.0 * zeta**2 - 2.0
    c = 1.0 - 1.0 / target**2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return 0.0
    u = (-b + math.sqrt(disc)) / 2.0
    if u <= 1.0:
        return 0.0
    return math.sqrt(u) - 1.0


# ---------------------------------------------------------------------------
# Duffing (cubic nonlinear restoring force) via harmonic balance
# ---------------------------------------------------------------------------


def duffing_harmonic_balance_amplitudes(
    excitation_hz: float,
    mass_kg: float,
    damping_n_s_m: float,
    stiffness_n_m: float,
    force_amplitude_n: float,
    cubic_stiffness_n_m3: float = 0.0,
    harmonic: int = 1,
) -> List[float]:
    """Steady-state harmonic-balance amplitudes of the Duffing oscillator.

    Solves ``(k + (3/4) k3 a^2 - m W^2)^2 a^2 + (c W)^2 a^2 = F0^2`` for
    ``A = a^2`` (a cubic in ``A``) and returns all real non-negative
    amplitudes, sorted ascending.  A single returned value means the response
    is single-valued at that frequency; three values mean the frequency lies in
    the fold region (the middle root is statically unstable).
    """
    if mass_kg <= 0.0 or stiffness_n_m <= 0.0:
        raise ValueError("mass and stiffness must be positive")
    if damping_n_s_m < 0.0:
        raise ValueError("damping must be non-negative")
    if force_amplitude_n <= 0.0:
        raise ValueError("force amplitude must be positive")
    if harmonic != 1:
        raise ValueError("only the fundamental harmonic is supported")
    w = 2.0 * math.pi * float(excitation_hz)
    d = stiffness_n_m - mass_kg * w**2
    k3r = 0.75 * cubic_stiffness_n_m3
    c2w2 = (damping_n_s_m * w) ** 2
    # k3r^2 A^3 + 2 d k3r A^2 + (d^2 + c^2 w^2) A - F0^2 = 0
    coeffs = [k3r**2, 2.0 * d * k3r, d**2 + c2w2, -(force_amplitude_n**2)]
    if abs(k3r) < 1e-30:
        coeffs = coeffs[1:]
    roots = np.roots(np.asarray(coeffs, dtype=float))
    amplitudes: List[float] = []
    for root in roots:
        if abs(root.imag) > 1e-9 * max(1.0, abs(root.real)):
            continue
        a2 = float(root.real)
        if a2 < 0.0:
            continue
        amplitudes.append(math.sqrt(a2))
    amplitudes.sort()
    if not amplitudes:
        raise ValueError("harmonic balance found no real solution (check input magnitudes)")
    return amplitudes


def duffing_backbone_amplitude(
    amplitude_m: float,
    mass_kg: float,
    stiffness_n_m: float,
    cubic_stiffness_n_m3: float,
) -> float:
    """Free-vibration frequency of the Duffing oscillator at amplitude ``a``.

    From the harmonic-balance backbone ``W^2 = (k + (3/4) k3 a^2) / m``; a
    hardening (``k3 > 0``) oscillator therefore *detunes upward* with amplitude,
    which is what produces the jump frequencies bounding the lock-in region.
    """
    if amplitude_m < 0.0:
        raise ValueError("amplitude must be non-negative")
    if mass_kg <= 0.0:
        raise ValueError("mass must be positive")
    return math.sqrt(max(stiffness_n_m + 0.75 * cubic_stiffness_n_m3 * amplitude_m**2, 0.0) / mass_kg) / (
        2.0 * math.pi
    )


# ---------------------------------------------------------------------------
# Post-processing of time-domain results
# ---------------------------------------------------------------------------


def steady_state_amplitude(
    time_s: np.ndarray,
    displacement: np.ndarray,
    discard_fraction: float = 0.6,
) -> Dict[str, float]:
    """Peak |x| on the tail of a record (after discarding the start-up transient)."""
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(displacement, dtype=float)
    if t.shape != x.shape or t.size < 4:
        raise ValueError("time and displacement must be equal-length arrays")
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in [0, 1)")
    start = int(discard_fraction * x.size)
    tail = x[start:]
    return {
        "peak_amplitude_m": float(np.max(np.abs(tail))),
        "rms_amplitude_m": float(np.sqrt(np.mean(tail**2))),
        "tail_start_s": float(t[start]),
        "tail_end_s": float(t[-1]),
    }


def harmonic_projection_amplitude(
    time_s: np.ndarray,
    signal: np.ndarray,
    frequency_hz: float,
) -> Dict[str, float]:
    """Least-squares fundamental amplitude of a (possibly still decaying) record.

    Fits ``x(t) ~ c + a sin(wt) + b cos(wt)`` on the supplied window and returns
    ``sqrt(a^2 + b^2)``.  Compared with peak picking this suppresses the bias
    from a residual start-up transient, which is what limits the accuracy of a
    steady-state amplification measurement near resonance.

    For a Duffing response it also isolates the fundamental, matching the
    single-harmonic assumption of the harmonic-balance reference.
    """
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(signal, dtype=float)
    if t.shape != x.shape or t.size < 8:
        raise ValueError("time and signal must be equal-length arrays with at least 8 samples")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    w = 2.0 * math.pi * frequency_hz
    design = np.column_stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)])
    coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    amplitude = float(math.hypot(coeffs[1], coeffs[2]))
    residual = x - design @ coeffs
    return {
        "amplitude": amplitude,
        "offset": float(coeffs[0]),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "signal_rms": float(np.sqrt(np.mean((x - np.mean(x)) ** 2))),
        "fundamental_fraction": float(
            amplitude / (math.sqrt(2.0) * np.sqrt(np.mean((x - np.mean(x)) ** 2)))
            if np.mean((x - np.mean(x)) ** 2) > 0
            else 0.0
        ),
    }


@dataclass(frozen=True)
class LockInVerdict:
    """Classification of one operating point against the lock-in criteria."""

    excitation_hz: float
    natural_hz: float
    relative_tolerance: float
    detuning: float
    in_band: bool
    amplification: float
    amplification_threshold: float
    state: str
    notes: str = ""

    @property
    def amplified(self) -> bool:
        return self.amplification >= self.amplification_threshold

    @property
    def lock_in_confirmed(self) -> bool:
        return bool(self.in_band and self.amplified)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "excitation_hz": self.excitation_hz,
            "natural_hz": self.natural_hz,
            "relative_tolerance": self.relative_tolerance,
            "detuning_relative": self.detuning,
            "in_lock_in_band": self.in_band,
            "amplification": self.amplification,
            "amplification_threshold": self.amplification_threshold,
            "lock_in_confirmed": self.lock_in_confirmed,
            "state": self.state,
            "notes": self.notes,
        }


def classify_lock_in(
    excitation_hz: float,
    natural_hz: float,
    amplification: float,
    relative_tolerance: float = LOCK_IN_RELATIVE_TOLERANCE,
    amplification_threshold: float = LOCK_IN_AMPLIFICATION_THRESHOLD,
) -> LockInVerdict:
    """Apply the two-part lock-in criterion (5 % band AND amplification).

    States: ``LOCKED`` (in band and amplified), ``IN_BAND_WEAK`` (in band but
    not amplified), ``OUT_OF_BAND_AMPLIFIED`` (amplified by resonance proximity
    but outside the band) and ``FREE`` (neither).
    """
    det = detuning_ratio(excitation_hz, natural_hz)
    in_band = abs(det) < relative_tolerance
    amplified = amplification >= amplification_threshold
    if in_band and amplified:
        state = "LOCKED"
        notes = "excitation inside the 5 % band and response amplified above threshold"
    elif in_band and not amplified:
        state = "IN_BAND_WEAK"
        notes = "inside the band but damping keeps the amplification below threshold"
    elif (not in_band) and amplified:
        state = "OUT_OF_BAND_AMPLIFIED"
        notes = "amplified although outside the nominal band (broadband/transient energy)"
    else:
        state = "FREE"
        notes = "outside the band and not amplified"
    return LockInVerdict(
        excitation_hz=float(excitation_hz),
        natural_hz=float(natural_hz),
        relative_tolerance=float(relative_tolerance),
        detuning=float(det),
        in_band=bool(in_band),
        amplification=float(amplification),
        amplification_threshold=float(amplification_threshold),
        state=state,
        notes=notes,
    )