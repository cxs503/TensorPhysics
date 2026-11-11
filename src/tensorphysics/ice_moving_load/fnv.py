"""FNV (frequency-normalised value) habitability criterion for ice-induced vibration.

Definition
----------
The mission defines

    FNV = a_max / (g * f_dom)

with ``a_max`` the peak acceleration and ``f_dom`` the dominant response
frequency.  FNV is therefore a *narrowband* severity index: it is only
dimensionally and dynamically coherent if the acceleration and the frequency
belong to the same spectral component.  A broad-band peak acceleration of a
broken-ice impact history is a superposition of several hull-girder modes at
frequencies that have nothing to do with ``f_dom``, so this module evaluates
both and uses the narrowband one for the class decision:

``a_band_m_s2``
    amplitude of the harmonic at the dominant response frequency, obtained by a
    tapered least-squares fit of ``A cos + B sin`` over the steady-state window
    (``amplitude = sqrt(A^2 + B^2)``).  This is the physically consistent
    ``a_max`` of the FNV definition and drives :attr:`FnvResult.fnv`.
``a_raw_m_s2``
    ``max|a(t)|`` of the same window, i.e. the literal reading.  It is reported
    as :attr:`FnvResult.fnv_raw` for transparency; it mixes modes and is
    therefore *not* used for the class decision (documented deviation, see the
    node report).

Well above the resonance the dominant peak of the acceleration spectrum is a
higher hull-girder mode (the sharp force drops ring every mode), which is
exactly the behaviour FNV is meant to discount: dividing by the higher
frequency keeps high-frequency ringing from inflating the habitability index.

Ice-class tiers (mission specification)
---------------------------------------
``habitable``    FNV < 0.5
``tolerable``    0.5 <= FNV < 1.0
``unacceptable`` FNV >= 1.0

Units: SI (m/s^2, Hz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from .ice_field import dominant_frequency_hz

__all__ = [
    "FNV_HABITABLE_MAX",
    "FNV_TOLERABLE_MAX",
    "STANDARD_GRAVITY",
    "FnvResult",
    "band_amplitude",
    "classify_fnv",
    "fnv_value",
    "evaluate_speed",
]

STANDARD_GRAVITY = 9.80665

#: Upper bound of the *habitable* tier.
FNV_HABITABLE_MAX = 0.5
#: Upper bound of the *tolerable* tier; above it the response is unacceptable.
FNV_TOLERABLE_MAX = 1.0


def fnv_value(acceleration_m_s2: float, dominant_frequency_hz: float, g: float = STANDARD_GRAVITY) -> float:
    """``FNV = a / (g f)`` with a positive-frequency guard."""
    if dominant_frequency_hz <= 0.0:
        raise ValueError("dominant frequency must be positive")
    if acceleration_m_s2 < 0.0:
        raise ValueError("acceleration magnitude must be non-negative")
    return acceleration_m_s2 / (g * dominant_frequency_hz)


def classify_fnv(fnv: float) -> str:
    """Ice-class habitability tier of an FNV value."""
    if fnv < FNV_HABITABLE_MAX:
        return "habitable"
    if fnv < FNV_TOLERABLE_MAX:
        return "tolerable"
    return "unacceptable"


def band_amplitude(
    time_s: np.ndarray, signal: np.ndarray, frequency_hz: float, detrend: bool = True
) -> float:
    """Amplitude of the harmonic at ``frequency_hz`` (tapered least-squares fit).

    Solves ``y ~ A cos(2 pi f t) + B sin(2 pi f t) + C`` over the window and
    returns ``sqrt(A^2 + B^2)``.  A Hann taper is applied so that a
    non-integer number of periods in the window does not bias the amplitude.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size != y.size or t.size < 8:
        raise ValueError("time and signal must be equally sized with >= 8 samples")
    if frequency_hz <= 0.0:
        raise ValueError("frequency must be positive")
    w = np.hanning(t.size)
    Om = 2.0 * math.pi * frequency_hz
    cols = [np.cos(Om * t), np.sin(Om * t)]
    if detrend:
        cols.append(np.ones_like(t))
    A = np.column_stack(cols)
    W = w[:, None]
    coef, *_ = np.linalg.lstsq(A * W, y * w, rcond=None)
    return float(math.hypot(coef[0], coef[1]))


@dataclass(frozen=True)
class FnvResult:
    """FNV evaluation of one operating speed."""

    speed_m_s: float
    speed_ratio: float
    f_ice_hz: float
    f_dominant_hz: float
    a_band_m_s2: float
    a_raw_m_s2: float
    fnv: float
    verdict: str
    fnv_raw: float = 0.0
    peak_displacement_m: float = 0.0
    peak_acceleration_m_s2: float = 0.0
    dynamic_amplification: float = 0.0
    energy_residual_relative: float = 0.0
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    @property
    def f_dominant_over_f_ice(self) -> float:
        return self.f_dominant_hz / self.f_ice_hz if self.f_ice_hz > 0 else float("nan")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = {
            "class": "FnvResult",
            "speed_m_s": self.speed_m_s,
            "speed_ratio_V_over_Vcr": self.speed_ratio,
            "f_ice_hz": self.f_ice_hz,
            "f_dominant_hz": self.f_dominant_hz,
            "a_band_m_s2": self.a_band_m_s2,
            "a_raw_m_s2": self.a_raw_m_s2,
            "fnv": self.fnv,
            "fnv_raw": self.fnv_raw,
            "verdict": self.verdict,
            "peak_displacement_m": self.peak_displacement_m,
            "dynamic_amplification": self.dynamic_amplification,
            "energy_residual_relative": self.energy_residual_relative,
        }
        if self.metadata:
            payload.update(self.metadata)
        return payload


def evaluate_speed(
    response,
    speed_m_s: float,
    speed_ratio: float,
    f_ice_hz: float,
    steady_window_s: Tuple[float, float] | None = None,
    static_displacement_m: float | None = None,
    min_hz: float = 0.05,
) -> FnvResult:
    """Reduce a :class:`~.moving_load.MovingLoadResponse` to an FNV verdict.

    ``steady_window_s`` selects the part of the history used for the spectral
    evaluation (default: the last 60 % of the record, which drops the initial
    transient while keeping enough periods for a sharp spectrum).
    """
    t = response.time_s
    acc = response.acceleration_m_s2
    u = response.displacement_m
    if steady_window_s is None:
        t0 = 0.4 * float(t[-1])
        t1 = float(t[-1])
    else:
        t0, t1 = steady_window_s
    mask = (t >= t0) & (t <= t1)
    if np.count_nonzero(mask) < 64:
        raise ValueError("steady window too short")
    tw, aw = t[mask], acc[mask]
    f_dom = dominant_frequency_hz(tw, aw, min_hz=min_hz)
    a_band = band_amplitude(tw, aw, f_dom)
    a_raw = float(np.max(np.abs(aw)))
    fnv = fnv_value(a_band, f_dom)
    daf = 0.0
    if static_displacement_m and static_displacement_m > 0.0:
        daf = float(np.max(np.abs(u[mask]))) / static_displacement_m
    return FnvResult(
        speed_m_s=float(speed_m_s),
        speed_ratio=float(speed_ratio),
        f_ice_hz=float(f_ice_hz),
        f_dominant_hz=float(f_dom),
        a_band_m_s2=float(a_band),
        a_raw_m_s2=float(a_raw),
        fnv=float(fnv),
        verdict=classify_fnv(fnv),
        fnv_raw=fnv_value(a_raw, f_dom),
        peak_displacement_m=float(np.max(np.abs(u))),
        peak_acceleration_m_s2=float(np.max(np.abs(acc))),
        dynamic_amplification=float(daf),
        energy_residual_relative=float(getattr(response, "energy_residual_relative", float("nan"))),
        metadata={"steady_window_s": [t0, t1], "n_steady_samples": int(np.count_nonzero(mask))},
    )