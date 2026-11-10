"""Underwater propagation, source levels and assessment criteria.

Propagation model (declared, standard engineering acoustics):

    RL(f, r) = SL(f) - 20 log10(r / 1 m) - alpha(f) * r / 1000        (1)

with spherical spreading (appropriate for the near/mid field of a hull source
in deep water), the Thorp absorption ``alpha(f)`` in dB/km, and ``SL`` the
source level in dB re 1 uPa m.  The 1/r decay itself is *not* re-imposed here:
it is already contained in the Rayleigh far-field of the radiator (the
pressure is proportional to 1/r), so Eq. (1) is used as an independent
consistency check of the propagation stage (the chain verifies
``RL_computed - (SL - 20log10 r - alpha r) ~ 0``).

Assessment criteria:

* **ISO 19906 ice-induced vibration** — the standard requires that
  ice-induced vibration be assessed; the numerical habitability limits are
  inherited from ISO 6897 / ISO 10137 comfort practice.  The exact ISO 19906
  table values are *not* embedded here (provenance WITHHELD, see
  :mod:`tensorphysics.ice_vib_noise.envelope`); what is implemented is the
  documented curve *shape* with a declared plateau, plus a sensitivity band so
  that the verdict cannot hinge on an unverifiable digit.
* **Marine mammal noise exposure** — the widely used NMFS (2018) technical
  guidance thresholds for underwater continuous noise (120 dB re 1 uPa RMS for
  PTS onset in cetaceans; 185 dB re 1 uPa^2 s SEL / 218 dB peak for impulsive
  sound) are implemented as range-to-threshold queries.  They are *criteria*,
  not model outputs, and are labelled as such in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "P_REF_UPASCAL",
    "thorp_absorption_db_per_km",
    "spreading_loss_db",
    "transmission_loss_db",
    "source_level_db",
    "received_level_db",
    "range_to_threshold_m",
    "NMFS_CONTINUOUS_PTS_DB",
    "NMFS_IMPULSIVE_SEL_DB",
    "NMFS_IMPULSIVE_PEAK_DB",
]

#: Reference pressure, 1 uPa, expressed in Pa.
P_REF_UPASCAL = 1.0e-6

#: NMFS (2018) technical guidance thresholds (dB re 1 uPa, or dB re 1 uPa^2 s).
NMFS_CONTINUOUS_PTS_DB = 120.0
NMFS_IMPULSIVE_SEL_DB = 185.0
NMFS_IMPULSIVE_PEAK_DB = 218.0


def thorp_absorption_db_per_km(frequency_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Thorp (1967) absorption coefficient in dB/km for ``f`` in Hz.

    ``alpha = 0.11 f2/(1+f2) + 44 f2/(4100+f2) + 2.75e-4 f2 + 0.003`` with
    ``f2`` in kHz^2.  Valid roughly 0.1-100 kHz in seawater.
    """
    f_khz = np.asarray(frequency_hz, dtype=float) / 1000.0
    f2 = f_khz ** 2
    alpha = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003
    return float(alpha) if np.ndim(frequency_hz) == 0 else alpha


def spreading_loss_db(range_m: float, geometry: str = "spherical") -> float:
    """Spreading loss referred to 1 m (0 dB at 1 m)."""
    if range_m <= 0.0:
        raise ValueError("range must be positive")
    if geometry == "spherical":
        return 20.0 * math.log10(range_m)
    if geometry == "cylindrical":
        return 10.0 * math.log10(range_m)
    raise ValueError(f"unknown spreading geometry {geometry!r}")


def transmission_loss_db(
    frequency_hz: Union[float, np.ndarray],
    range_m: float,
    geometry: str = "spherical",
) -> Union[float, np.ndarray]:
    """Total transmission loss ``TL = spreading + absorption`` [dB]."""
    if range_m <= 0.0:
        raise ValueError("range must be positive")
    spread = spreading_loss_db(range_m, geometry)
    absorb = thorp_absorption_db_per_km(frequency_hz) * range_m / 1000.0
    return spread + absorb


def source_level_db(pressure_amplitude_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Source level re 1 uPa m from the far-field pressure amplitude at 1 m.

    The radiator's ``p(r)`` already carries ``1/r``; the source level is the
    amplitude referred to 1 m, i.e. ``20 log10(p(r) * r / p_ref)`` is applied by
    the caller through :func:`band_source_level_db`, this helper only converts
    an amplitude at 1 m.
    """
    p = np.asarray(pressure_amplitude_pa, dtype=float)
    out = 20.0 * np.log10(np.maximum(p, 1e-30) / P_REF_UPASCAL)
    return float(out) if np.ndim(pressure_amplitude_pa) == 0 else out


def received_level_db(
    source_level: Union[float, np.ndarray],
    frequency_hz: Union[float, np.ndarray],
    range_m: float,
    geometry: str = "spherical",
) -> Union[float, np.ndarray]:
    """``RL = SL - TL``, Eq. (1)."""
    return np.asarray(source_level) - transmission_loss_db(frequency_hz, range_m, geometry)


def range_to_threshold_m(
    source_level_db_value: float,
    frequency_hz: float,
    threshold_db: float,
    geometry: str = "spherical",
    r_max_m: float = 1.0e6,
) -> Optional[float]:
    """Range at which ``RL`` falls to ``threshold_db`` (bisection, absorption kept).

    Returns ``None`` when the threshold is not reached even at ``r_max_m``
    (i.e. the source is below the criterion at 1 m).
    """
    if threshold_db >= source_level_db_value:
        return None
    lo, hi = 1.0, 1.0
    if received_level_db(source_level_db_value, frequency_hz, lo, geometry) <= threshold_db:
        return 0.0
    while received_level_db(source_level_db_value, frequency_hz, hi, geometry) > threshold_db:
        hi *= 2.0
        if hi > r_max_m:
            return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if received_level_db(source_level_db_value, frequency_hz, mid, geometry) > threshold_db:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)