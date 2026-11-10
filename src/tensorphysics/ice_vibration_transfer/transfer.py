"""Equipment foundation transfer function ``H(f)`` (hull deck -> equipment base).

The equipment package sits on resilient mounts whose dynamic behaviour is the
classical base-excited single-degree-of-freedom transmissibility

.. math::

    H(f) = \\frac{1 + 2 i \\zeta r}{(1 - r^2) + 2 i \\zeta r},
    \\qquad r = f / f_n

which is the ratio of the *base* acceleration (top of the mount = the equipment
base plate) to the *support* acceleration (hull deck at the foundation station).
Its modulus is

.. math::

    |H(f)| = \\sqrt{\\frac{1 + (2\\zeta r)^2}{(1-r^2)^2 + (2\\zeta r)^2}}

with the two textbook limits that make the node's second criterion meaningful:

* ``|H| -> 1`` as ``f -> 0`` (the mount is rigid at low frequency), so an
  isolation criterion can only be posed *above* the isolation frequency;
* ``|H| < 1`` for every ``f > \\sqrt{2} f_n`` (attenuation band), and
  ``|H| < 1`` holds everywhere once ``\\zeta \\ge 1/\\sqrt{2}`` (no amplification
  at all: a heavily damped mount is not an isolator but it never amplifies).

The node therefore evaluates ``|H| < 1`` over the *hull acceleration variance
band* (the band that carries the ice-induced energy, computed from the response
PSD, not chosen by hand) and reports the amplification band ``f < \\sqrt{2} f_n``
explicitly instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["MountIsolation", "transmissibility_envelope"]


@dataclass
class MountIsolation:
    """Resilient mount of the equipment package (base-excited SDOF)."""

    f_n_hz: float
    zeta: float
    equipment_mass_kg: float | None = None
    label: str = "equipment mount"

    def __post_init__(self) -> None:
        if not np.isfinite(self.f_n_hz) or self.f_n_hz <= 0.0:
            raise ValueError("mount natural frequency must be positive and finite")
        if not 0.0 < self.zeta < 1.0:
            raise ValueError("mount damping ratio must satisfy 0 < zeta < 1")
        if self.equipment_mass_kg is not None and self.equipment_mass_kg <= 0.0:
            raise ValueError("equipment mass must be positive")

    # ------------------------------------------------------------------ core
    @property
    def isolation_frequency_hz(self) -> float:
        """``sqrt(2) f_n``: upper edge of the amplification band."""
        return float(np.sqrt(2.0) * self.f_n_hz)

    @property
    def static_deflection_m(self) -> float:
        """``g / (2 pi f_n)^2``: mount static deflection under the equipment weight."""
        return float(9.81 / (2.0 * np.pi * self.f_n_hz) ** 2)

    @property
    def peak_amplification(self) -> float:
        """Maximum of ``|H(f)|`` over frequency (exact, not the ``1/(2 zeta)`` rule of thumb).

        ``|H|^2 = (1 + a u) / ((1-u)^2 + a u)`` with ``u = r^2`` and ``a = 4 zeta^2``
        is stationary at ``u* = (sqrt(1 + 2a) - 1) / a``; the value there is returned.
        (The frequently quoted ``1/(2 zeta sqrt(1-zeta^2))`` applies to the
        *relative* transmissibility, not to this base-excited form.)
        """
        a = 4.0 * self.zeta**2
        u = (np.sqrt(1.0 + 2.0 * a) - 1.0) / a
        value = (1.0 + a * u) / ((1.0 - u) ** 2 + a * u)
        return float(np.sqrt(value))

    def ratio(self, freqs: np.ndarray) -> np.ndarray:
        return np.asarray(freqs, dtype=float) / self.f_n_hz

    def H(self, freqs: np.ndarray) -> np.ndarray:
        """Complex transmissibility ``a_base / a_support``."""
        r = self.ratio(freqs)
        num = 1.0 + 2j * self.zeta * r
        den = (1.0 - r**2) + 2j * self.zeta * r
        return num / den

    def magnitude(self, freqs: np.ndarray) -> np.ndarray:
        return np.abs(self.H(freqs))

    def magnitude_db(self, freqs: np.ndarray) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(self.magnitude(freqs), 1e-300))

    def phase_deg(self, freqs: np.ndarray) -> np.ndarray:
        return np.degrees(np.angle(self.H(freqs)))

    def bode(self, freqs: np.ndarray) -> dict[str, Any]:
        """Bode data (frequency sweep, dB magnitude, phase, |H|=1 crossing)."""
        f = np.asarray(freqs, dtype=float)
        mag = self.magnitude(f)
        return {
            "freqs_hz": f,
            "magnitude": mag,
            "magnitude_db": 20.0 * np.log10(np.maximum(mag, 1e-300)),
            "phase_deg": self.phase_deg(f),
            "f_n_hz": float(self.f_n_hz),
            "zeta": float(self.zeta),
            "isolation_frequency_hz": self.isolation_frequency_hz,
            "peak_amplification": self.peak_amplification,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "f_n_hz": float(self.f_n_hz),
            "zeta": float(self.zeta),
            "equipment_mass_kg": None if self.equipment_mass_kg is None else float(self.equipment_mass_kg),
            "label": self.label,
            "isolation_frequency_hz": self.isolation_frequency_hz,
            "peak_amplification": self.peak_amplification,
            "static_deflection_m": self.static_deflection_m,
            "attenuation_above_hz": self.isolation_frequency_hz,
            "model": "base-excited SDOF transmissibility (1+2izr)/((1-r^2)+2izr)",
        }


def transmissibility_envelope(
    freqs: np.ndarray,
    mount: MountIsolation,
    band_hz: tuple[float, float],
    checkpoints: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate ``H(f)`` on the ice band and at named check-point frequencies."""
    f = np.asarray(freqs, dtype=float)
    mag = mount.magnitude(f)
    lo, hi = float(band_hz[0]), float(band_hz[1])
    m_band = (f >= lo) & (f <= hi)
    band_max = float(np.max(mag[m_band])) if np.any(m_band) else float("nan")
    amp = f < mount.isolation_frequency_hz
    points = {}
    for name, fi in (checkpoints or {}).items():
        val = float(mount.magnitude(np.array([float(fi)]))[0])
        points[name] = {"frequency_hz": float(fi), "magnitude": val, "magnitude_db": 20.0 * np.log10(max(val, 1e-300))}
    return {
        "band_hz": [lo, hi],
        "band_max_magnitude": band_max,
        "band_max_magnitude_db": 20.0 * np.log10(max(band_max, 1e-300)),
        "band_all_below_unity": bool(np.all(mag[m_band] < 1.0)),
        "amplification_band_hz": [0.0, mount.isolation_frequency_hz],
        "amplification_band_max": float(np.max(mag[amp])) if np.any(amp) else 1.0,
        "checkpoints": points,
    }