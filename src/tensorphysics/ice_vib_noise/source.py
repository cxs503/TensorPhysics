"""Ice load time history and the crushing-energy budget of the coupled chain.

The excitation of the hull is the ice crushing load of a ship advancing through
level ice.  Two things are needed by the acoustic chain and both are produced
here *from the same series*:

* the total load ``F_ice(t)`` and its spatial distribution along the hull
  ``q(x, t) = F_ice(t) w(x)`` with ``int w dx = 1`` over the bow contact
  length, which gives the generalised force ``Q_i(t) = int q phi_i dx``;
* the **crushing energy**, i.e. the mechanical energy the ice cover absorbs
  while the ship advances: ``E_crush = int F_ice(t) U dt`` where ``U`` is the
  ship speed relative to the ice (crushing work is done by the drift, not by
  the hull vibration).  ``E_crush`` is the denominator of the acoustic
  conversion efficiency reported by the chain.

The intermittent character of crushing (flaking events at ``f_ice = U / s``
with scatter in arrival and amplitude) is taken from
:func:`tensorphysics.ice_vibration.ice_load.quasi_periodic_pulse_train`, so the
spectral content of the load is a modelled quantity, not a hand-tuned spectrum.
A zero-amplitude series is provided as the negative control of the chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..ice_vibration.ice_load import IceLoadSeries, quasi_periodic_pulse_train, harmonic_load

__all__ = [
    "IceExcitationCase",
    "build_ice_excitation",
    "bow_load_distribution",
    "crushing_energy_j",
]


@dataclass(frozen=True)
class IceExcitationCase:
    """Parameters defining the ice load and its spatial distribution."""

    ice_thickness_m: float = 0.8
    contact_pressure_mpa: float = 2.0
    contact_width_m: float = 3.0
    drift_speed_m_s: float = 1.0
    flaking_length_m: float = 0.5
    duration_s: float = 60.0
    dt_s: float = 0.005
    bow_contact_length_m: float = 6.0
    arrival_jitter: float = 0.08
    amplitude_cv: float = 0.12
    force_scale: float = 1.0
    seed: int = 20261110

    def __post_init__(self) -> None:
        for name in (
            "ice_thickness_m",
            "contact_pressure_mpa",
            "contact_width_m",
            "drift_speed_m_s",
            "flaking_length_m",
            "duration_s",
            "dt_s",
            "bow_contact_length_m",
        ):
            val = getattr(self, name)
            if not math.isfinite(float(val)) or float(val) <= 0.0:
                raise ValueError(f"IceExcitationCase.{name} must be positive, got {val!r}")
        if not 0.0 <= self.arrival_jitter < 1.0:
            raise ValueError("arrival_jitter must be in [0, 1)")
        if self.amplitude_cv < 0.0:
            raise ValueError("amplitude_cv must be non-negative")
        if not math.isfinite(float(self.force_scale)) or float(self.force_scale) <= 0.0:
            raise ValueError("force_scale must be positive")

    @property
    def peak_force_n(self) -> float:
        """Nominal peak crushing force ``p_c h w`` (local contact pressure model)."""
        return self.contact_pressure_mpa * 1.0e6 * self.ice_thickness_m * self.contact_width_m * self.force_scale

    @property
    def crushing_frequency_hz(self) -> float:
        """Mean flaking frequency ``f_ice = U / s``."""
        return self.drift_speed_m_s / self.flaking_length_m

    def to_json_dict(self) -> Dict[str, Any]:
        d = {
            "ice_thickness_m": self.ice_thickness_m,
            "contact_pressure_mpa": self.contact_pressure_mpa,
            "contact_width_m": self.contact_width_m,
            "drift_speed_m_s": self.drift_speed_m_s,
            "flaking_length_m": self.flaking_length_m,
            "duration_s": self.duration_s,
            "dt_s": self.dt_s,
            "bow_contact_length_m": self.bow_contact_length_m,
            "arrival_jitter": self.arrival_jitter,
            "amplitude_cv": self.amplitude_cv,
            "force_scale": self.force_scale,
            "seed": self.seed,
            "peak_force_n": self.peak_force_n,
            "crushing_frequency_hz": self.crushing_frequency_hz,
        }
        return d


def build_ice_excitation(
    case: IceExcitationCase,
    mode: str = "crushing",
) -> IceLoadSeries:
    """Build the ice load series.

    ``mode`` is one of ``"crushing"`` (intermittent flaking, the physical case),
    ``"harmonic"`` (single-frequency control used to compare against the
    closed-form amplifier) and ``"zero"`` (negative control: ``F = 0``).
    """
    if mode == "crushing":
        return quasi_periodic_pulse_train(
            pulse_frequency_hz=case.crushing_frequency_hz,
            amplitude_n=case.peak_force_n,
            duration_s=case.duration_s,
            dt_s=case.dt_s,
            pulse_duration_s=0.55 / case.crushing_frequency_hz,
            shape="sawtooth",
            arrival_jitter=case.arrival_jitter,
            amplitude_cv=case.amplitude_cv,
            seed=case.seed,
        )
    if mode == "harmonic":
        return harmonic_load(
            amplitude_n=case.peak_force_n,
            frequency_hz=case.crushing_frequency_hz,
            duration_s=case.duration_s,
            dt_s=case.dt_s,
        )
    if mode == "zero":
        n = int(round(case.duration_s / case.dt_s)) + 1
        t = np.arange(n) * case.dt_s
        return IceLoadSeries(
            time_s=t,
            force_n=np.zeros(n),
            dt_s=case.dt_s,
            label="zero-load-negative-control",
            metadata={"model": "zero", "peak_force_n": 0.0},
        )
    raise ValueError(f"unknown excitation mode {mode!r}")


def bow_load_distribution(
    x_m: Sequence[float] | np.ndarray,
    hull_length_m: float,
    contact_length_m: float,
) -> np.ndarray:
    """Normalised bow contact distribution ``w(x)`` with ``int w dx = 1``.

    A half-sine over the forward ``contact_length_m`` metres (zero aft of the
    contact patch).  ``int q dx = F_ice(t)`` by construction, so the generalised
    force is a pure kinematic projection of the total load.
    """
    if contact_length_m <= 0.0 or contact_length_m > hull_length_m:
        raise ValueError("contact length must be in (0, hull length]")
    x = np.asarray(x_m, dtype=float)
    if np.any(x < 0.0) or np.any(x > hull_length_m):
        raise ValueError("x outside the hull length")
    x0 = hull_length_m - contact_length_m
    w = np.zeros_like(x)
    m = x >= x0
    w[m] = np.sin(math.pi * (x[m] - x0) / contact_length_m)
    norm = float(np.trapezoid(w, x))
    if norm <= 0.0:
        raise ValueError("contact patch not resolved by the x grid; increase resolution")
    return w / norm


def crushing_energy_j(force_n: np.ndarray, time_s: np.ndarray, drift_speed_m_s: float) -> float:
    """Crushing energy ``int F U dt`` absorbed by the ice cover [J]."""
    f = np.asarray(force_n, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if f.shape != t.shape:
        raise ValueError("force and time arrays must match")
    if drift_speed_m_s < 0.0:
        raise ValueError("drift speed must be non-negative")
    return float(drift_speed_m_s * np.trapezoid(f, t))