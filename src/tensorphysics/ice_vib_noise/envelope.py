"""ISO 19906-informed vibration envelope and the two-domain noise envelope.

**Provenance statement (read before using the numbers).**
ISO 19906 (Arctic offshore structures) requires that ice-induced vibration of
the structure be assessed and refers the habitability/comfort assessment to the
ISO 6897 / ISO 10137 practice.  The numerical table values of that clause are
*not* reproduced here: we could not verify the digits against a controlled copy
of the standard inside this task, so they are declared **WITHHELD** rather than
invented.  What is implemented is the *documented curve shape* used by those
standards for whole-body vibration comfort:

* a constant-acceleration plateau over the frequency band in which the human
  body is most sensitive and in which ice-induced vibration of a hull girder
  lives (``f_lo`` to ``f_hi``, default 0.5-8 Hz);
* above ``f_hi`` a constant-velocity branch (``~1/f``);
* below ``f_lo`` a constant-displacement branch (``~1/f^2`` ... implemented as
  ``~f`` rise in acceleration, i.e. constant displacement).

The plateau value is an **explicit input parameter** with a sensitivity band
(default ````{0.02, 0.05, 0.1}`` m/s^2 RMS, the order of magnitude of the ISO
2631/10137 "not uncomfortable" to "fairly uncomfortable" boundaries), so the
verdict can be reported as a function of the unverifiable digit instead of
being decided by it.

For the acoustic domain there is *no* SPL limit in ISO 19906 at all
(declared NOT_SUPPORTED).  The chain therefore builds the SPL envelope that is
**equivalent** to the vibration envelope through the same radiation transfer
function: the SPL a hull would radiate if it vibrated exactly at the ISO limit.
This makes the vibration and noise verdicts translatable into each other, and
the coherence of that translation is itself a machine-checkable criterion (see
``chain.py``, criterion C4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "VibrationEnvelope",
    "iso19906_informed_envelope",
    "band_margins",
]


@dataclass(frozen=True)
class VibrationEnvelope:
    """Piecewise 1/3-octave RMS acceleration envelope [m/s^2]."""

    plateau_rms_m_s2: float
    f_lo_hz: float = 0.5
    f_hi_hz: float = 8.0
    provenance: str = (
        "ISO 19906 requires ice-induced vibration assessment; numeric comfort digits WITHHELD. "
        "Shape from ISO 6897/ISO 10137 practice; plateau is a declared parameter."
    )

    def limit(self, frequencies_hz: Sequence[float] | np.ndarray) -> np.ndarray:
        f = np.asarray(frequencies_hz, dtype=float)
        if np.any(f <= 0.0):
            raise ValueError("frequencies must be positive")
        out = np.full(f.shape, self.plateau_rms_m_s2, dtype=float)
        below = f < self.f_lo_hz
        above = f > self.f_hi_hz
        # constant displacement below the plateau: a ~ f
        out[below] = self.plateau_rms_m_s2 * (f[below] / self.f_lo_hz)
        # constant velocity above the plateau: a ~ 1/f
        out[above] = self.plateau_rms_m_s2 * (self.f_hi_hz / f[above])
        return out

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "plateau_rms_m_s2": self.plateau_rms_m_s2,
            "f_lo_hz": self.f_lo_hz,
            "f_hi_hz": self.f_hi_hz,
            "provenance": self.provenance,
            "standard_status": "ISO 19906 numeric values WITHHELD (unverifiable in-task); "
            "ISO 19906 contains no SPL limit (NOT_SUPPORTED)",
        }


def iso19906_informed_envelope(
    plateau_rms_m_s2: float = 0.05,
    f_lo_hz: float = 0.5,
    f_hi_hz: float = 8.0,
) -> VibrationEnvelope:
    """Construct the ISO 19906-informed envelope (see module docstring)."""
    if plateau_rms_m_s2 <= 0.0:
        raise ValueError("plateau must be positive")
    if not 0.0 < f_lo_hz < f_hi_hz:
        raise ValueError("require 0 < f_lo < f_hi")
    return VibrationEnvelope(plateau_rms_m_s2, f_lo_hz, f_hi_hz)


def band_margins(
    centre_hz: Sequence[float],
    computed_rms: Sequence[float],
    limit_rms: Sequence[float],
) -> Dict[str, Any]:
    """Per-band ratio ``computed / limit`` plus the governing (worst) band."""
    fc = np.asarray(centre_hz, dtype=float)
    a = np.asarray(computed_rms, dtype=float)
    lim = np.asarray(limit_rms, dtype=float)
    if not (fc.shape == a.shape == lim.shape):
        raise ValueError("band arrays must match")
    if np.any(lim <= 0.0):
        raise ValueError("envelope limits must be positive")
    ratio = a / lim
    worst = int(np.argmax(ratio)) if ratio.size else -1
    return {
        "ratio": ratio.tolist(),
        "max_ratio": float(ratio.max()) if ratio.size else 0.0,
        "governing_band_hz": float(fc[worst]) if worst >= 0 else None,
        "worst_margin_db": float(20.0 * math.log10(max(ratio.max(), 1e-30))) if ratio.size else 0.0,
        "n_bands_over_limit": int(np.sum(ratio > 1.0)),
        "pass": bool(ratio.size > 0 and ratio.max() <= 1.0),
    }